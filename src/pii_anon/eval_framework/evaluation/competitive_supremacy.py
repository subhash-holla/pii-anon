"""CompetitiveSupremacyGate — the SOTA-Dominance Objective (SDO) (S4-CS-01).

This gate IS the program's optimization function + the definition of "enhancement
run complete." It consumes the benchmark JSON (read-only) plus an optional
bayes-bt joint posterior, checks the hard guarantees G1..G7, computes the SDO
objective ``J = P(rank(pii-anon) = 1)``, and emits exactly ONE verdict
``{CLAIM_GRADE_SOTA | PROVISIONAL_SOTA | NOT_YET}`` together with the single
**binding constraint** — the next thing to fix — so the program always knows its
target (the SDO philosophy, mirrored from :mod:`convergence`).

Layering / boundary (story §7)
------------------------------
Lives in ``eval_framework/evaluation`` and imports ONLY
:mod:`pii_anon.eval_framework.rating` (the J-meter) + reads JSON dicts. It does
NOT import ``swarm`` / ``moe`` / ``fusion`` / ``policy`` (import-boundary test).
``evaluation/competitor_compare.py`` is untouched (RISK-6) — the Tier-R/Tier-C
metadata lives in :mod:`competitor_tiers`.

The hard guarantees (story §3)
------------------------------
Each guarantee is a pure ``(benchmark, posterior?) -> GuaranteeResult`` with a
THREE-VALUED ``passed``: ``True`` / ``False`` / ``None`` (PENDING). PENDING never
blocks ``PROVISIONAL`` but ALWAYS blocks ``CLAIM_GRADE`` — never collapse ``None``
to ``False`` with ``all(...)``.

* **G1** Recall-floor by construction — entities(ensemble) ⊇ entities(shared) ∧
  per-language recall ε ≤ 0.005. PENDING when the per-language artifact is absent
  (never fabricated). **computable now.**
* **G2** Pseudonymization-integrity / reversibility — PENDING ← S4-01 scorers.
* **G3** Recall dominance — pii-anon recall ladder ≥ max(competitor recall).
  **computable now.**
* **G4** Calibration / selective-risk — per-class ECE ≤ NFR-017 bars ∧ AURC
  monotone ∧ ≥3 abstention points ∧ calibrated-confidence-coverage = 1.0
  (NFR-020). Computed by ``_g4_calibration_selective_risk`` from the per-class
  calibration fields the S4-03 ``SelectiveRiskReporter`` emits; PENDING when the
  artifact lacks those fields (the current smoke artifact does — never
  fabricated). **computable now (S4-03).**
* **G5** Audit + orchestration latency / interception — PENDING ← S5/S6.
* **G6** Non-inferiority on raw detection — core F2 ≥ best **Tier-R** F2 − ε_F ∧
  entity coverage ≥ 0.80. A Tier-C raw-F1 (e.g. OpenAI ≈0.96) exceeding pii-anon
  does NOT fail G6 — the explicit honesty carve-out, always recorded. **now.**
* **G7** Certified run — ``canonical_claim_run == True`` ∧ full provenance stamp.
  **provenance computable now.**

Completion predicate (story §5)
-------------------------------
* ``CLAIM_GRADE_SOTA`` ⟺ canonical ∧ (every in-scope Gk ``True``) ∧ J ≥ 0.95 ∧
  (Tier-R ∪ Tier-C all RUN-or-WAIVED).
* ``PROVISIONAL_SOTA`` ⟺ same but blocked only by unrun tiers (unrun Tier-C
  and/or unrun Tier-R — the full Tier-R ∪ Tier-C set must be RUN-or-WAIVED to
  reach CLAIM_GRADE).
* ``NOT_YET`` ⟺ otherwise — report the binding constraint (priority:
  canonical-run → failed G (lowest k) → J gap → unrun Tier-C → unrun Tier-R).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import NDArray

from pii_anon.eval_framework.rating.bradley_terry import BradleyTerryMLEEngine
from pii_anon.eval_framework.rating.significance import rank_one_distribution

from .competitor_tiers import (
    TIER_R_NAMES,
    TierEntry,
    apply_run_status,
    default_registry,
    run_status_from_benchmark,
    unrun_tier_c,
    unrun_tier_r,
    waive,
)

__all__ = [
    "J_BAR",
    "EPS_F2",
    "ENTITY_COVERAGE_MIN",
    "EPS_RECALL_PER_LANG",
    "Verdict",
    "GuaranteeResult",
    "SupremacyVerdict",
    "f_beta",
    "recall_floor_breachers",
]

# -- SDO threshold literals: these ARE the completion contract (§3/§5). -------
J_BAR: float = 0.95
EPS_F2: float = 0.01
ENTITY_COVERAGE_MIN: float = 0.80
EPS_RECALL_PER_LANG: float = 0.005

# G4 (calibration / selective-risk, S4-03) thresholds — the NFR-017/020/021 bars.
# A per-class ECE bar is read per class from the artifact when present (a class
# may be high-resource [0.05] or long-tail [0.08]); these are the DEFAULT bars the
# gate falls back to when the artifact does not stamp a per-class threshold.
G4_ECE_BAR_HIGH_RESOURCE: float = 0.05  # NFR-017 high-resource ECE bar
G4_ECE_BAR_LONG_TAIL: float = 0.08  # NFR-017 long-tail ECE bar
G4_MIN_ABSTENTION_POINTS: int = 3  # NFR-021 ≥3 operating points
G4_COVERAGE_REQUIRED: float = 1.0  # NFR-020 calibrated-confidence-coverage (the MUST)

# The system-under-test names (the "pii-anon recall ladder"); the moat axes are
# measured on the best of these, raw non-inferiority on the core engine.
_CORE_SYSTEM = "pii-anon"
_LADDER_SYSTEMS: frozenset[str] = frozenset({"pii-anon", "pii-anon-swarm"})

# The provenance fields a certified (G7) run must stamp (story §2a / §3).
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "git_sha",
    "dataset_sha256",
    "matrix_sha256",
    "timestamp_utc",
)

# Guarantee order is load-bearing: the binding-constraint reporter names the
# LOWEST-k failing guarantee, so G1 < G2 < ... < G7.
_GUARANTEE_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")

# -- RecallFloorVerdictGuard (story §3 G7): "a floor-breaching system can never
# top-rank." The guard is RECALL-SPECIFIC: it keys ONLY on recall-floor signals,
# never on the conflated ensemble `floor_pass` (which mixes in latency/throughput
# floors). A non-qualifying qualification_status, a failing recall/f1 profile
# floor-check, or (when the per-language ε artifact is present) a language ε >
# EPS_RECALL_PER_LANG constitutes a recall-floor breach. Latency/throughput floor
# failures are a DIFFERENT axis (the speed-floor) and never a recall breach.
_QUALIFYING_STATUSES: frozenset[str] = frozenset({"core", "qualified"})
_RECALL_FLOOR_METRICS: frozenset[str] = frozenset({"recall", "f1", "f2"})

# Pending successors (named in the gate output as tracked work).
_PENDING_SUCCESSORS: dict[str, str] = {
    "G2": "G2←S4-01 (anon/pseudo reversibility scorers)",
    "G4": "G4←S4-03 (calibration / selective-risk reporter)",
    "G5": "G5←S5/S6 (latency budgets + 4-channel interception)",
}

# The explicit OpenAI raw-F1 honesty carve-out, ALWAYS surfaced (§3/§7).
_CARVE_OUT_NOTE = (
    "Honesty carve-out: G6 measures non-inferiority on raw detection F2 against "
    "the runnable Tier-R competitors ONLY. A cited Tier-C raw F1 (e.g. OpenAI "
    "≈0.96) exceeding pii-anon does NOT fail G6 — the SDO claims dominance on the "
    "moat axes (reversibility, recall, calibration, audit) and non-inferiority, "
    "not raw-F1 supremacy over every cloud API."
)

# MLE-bootstrap J-fallback knobs (deterministic; fixed seed → reproducible J).
_J_BOOTSTRAP_SEED = 20240601
_J_BOOTSTRAP_B = 400
_J_PAIR_GAMES = 100  # comparisons synthesised per system pair (non-separable)
_J_PAIR_GAMMA = 10.0  # logistic steepness of the composite→outcome map


class Verdict(Enum):
    """The single SDO verdict (§5)."""

    CLAIM_GRADE_SOTA = "CLAIM_GRADE_SOTA"
    PROVISIONAL_SOTA = "PROVISIONAL_SOTA"
    NOT_YET = "NOT_YET"


def f_beta(precision: float, recall: float, *, beta: float) -> float:
    """The F-beta score ``(1+β²)·P·R / (β²·P + R)``; 0.0 when P=R=0.

    ``beta=2`` (the SDO G6 measure) weights recall 4× precision:
    ``5·P·R / (4P + R)``.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


@dataclass(frozen=True)
class GuaranteeResult:
    """One guarantee Gk's three-valued outcome.

    Attributes
    ----------
    axis:
        Guarantee id (``"G1"`` … ``"G7"``).
    passed:
        ``True`` (pass) / ``False`` (fail) / ``None`` (PENDING — successor not
        yet landed; never collapsed to ``False``).
    observed:
        The measured quantity (``float('nan')`` when not numerically defined,
        e.g. a PENDING or purely-structural guarantee).
    bar:
        The threshold the observed value is compared against (NaN when N/A).
    binding_detail:
        Human-readable description of the outcome / the next thing to fix.
    """

    axis: str
    passed: bool | None
    observed: float
    bar: float
    binding_detail: str


@dataclass(frozen=True)
class SupremacyVerdict:
    """The SDO gate verdict over one benchmark (+ optional posterior).

    Mirrors :class:`convergence.ConvergenceReport`: a frozen dataclass built by a
    :meth:`from_artifacts` classmethod, carrying the single ``binding_constraint``
    (the next thing to fix) and the full honesty surface.

    Attributes
    ----------
    verdict:
        The single :class:`Verdict`.
    binding_constraint:
        The single most important failing item (priority: canonical-run → failed
        G (lowest k) → J gap → unrun Tier-C → unrun Tier-R). ``""`` IFF
        ``CLAIM_GRADE_SOTA``.
    j_value:
        The SDO objective ``P(rank(pii-anon)=1)`` (``None`` when unavailable).
    j_source:
        ``"bayes"`` (real posterior) / ``"mle-bootstrap"`` (in-tree fallback) /
        ``"unavailable"`` (no posterior and no competitor pairs).
    guarantees:
        Per-axis :class:`GuaranteeResult` for G1..G7 (three-valued).
    canonical_claim_run:
        The ``run_metadata.canonical_claim_run`` flag (the #1 gate).
    unrun_tier_c:
        The Tier-C competitor names still blocking ``CLAIM_GRADE``.
    axes_pending:
        Human-readable names of the PENDING successor guarantees.
    j_rank1_system:
        The system the SDO J-meter CROWNS as rank-1 (argmax of the rank-1
        distribution over the recall-floor-COMPLIANT systems). A recall-floor
        breacher can never appear here (the RecallFloorVerdictGuard). ``None``
        only when J is unavailable.
    recall_floor_breachers:
        The systems flagged as recall-floor breachers (the guard's input set) —
        a visible honesty field; populated whenever any system breaches.
    unrun_tier_r:
        The Tier-R competitor names still UNRUN (also CLAIM_GRADE blockers per
        §5 — e.g. ``gliner2`` today); a visible honesty field distinct from
        ``unrun_tier_c``.
    carve_out_note:
        The OpenAI raw-F1 honesty carve-out (always populated).
    """

    verdict: Verdict
    binding_constraint: str
    j_value: float | None
    j_source: str
    guarantees: tuple[GuaranteeResult, ...]
    canonical_claim_run: bool
    unrun_tier_c: frozenset[str]
    axes_pending: tuple[str, ...]
    j_rank1_system: str | None = None
    recall_floor_breachers: frozenset[str] = frozenset()
    unrun_tier_r: frozenset[str] = frozenset()
    carve_out_note: str = _CARVE_OUT_NOTE
    tier_registry: dict[str, TierEntry] = field(default_factory=default_registry)

    def guarantee(self, axis: str) -> GuaranteeResult:
        """Return the :class:`GuaranteeResult` for ``axis`` (``"G3"`` …).

        Raises :class:`KeyError` for an unknown axis (no silent default).
        """
        for g in self.guarantees:
            if g.axis == axis:
                return g
        raise KeyError(f"no guarantee {axis!r}; have {[g.axis for g in self.guarantees]}")

    @classmethod
    def from_artifacts(
        cls,
        benchmark: dict[str, Any],
        *,
        theta_samples: NDArray[np.floating[Any]] | None = None,
        posterior_names: list[str] | None = None,
        pending_overrides: dict[str, bool | None] | None = None,
        tier_c_waivers: dict[str, str] | None = None,
        unrun_tier_r_waivers: dict[str, str] | None = None,
    ) -> SupremacyVerdict:
        """Compute the SDO verdict from a benchmark dict (+ optional posterior).

        Parameters
        ----------
        benchmark:
            The parsed ``benchmark-results.json`` (read-only; never mutated).
        theta_samples / posterior_names:
            An optional bayes-bt joint posterior ``(n_draws, n_systems)`` and its
            column names. When present, J is read off it (``j_source='bayes'``);
            otherwise J falls back to the in-tree MLE-bootstrap
            (``j_source='mle-bootstrap'``), or ``'unavailable'`` if there are no
            competitor pairs to fit. Either way the rank-1 argmax is restricted to
            the recall-floor-COMPLIANT systems (the RecallFloorVerdictGuard).
        pending_overrides:
            Three-valued overrides for the successor guarantees (``"G2"``,
            ``"G4"``, ``"G5"``) — a successor story supplies its computed
            ``True``/``False``; the default leaves them PENDING (``None``).
        tier_c_waivers:
            ``{tier_c_name: reason}`` waivers (reason mandatory). Waived Tier-C
            entries stop blocking ``CLAIM_GRADE``.
        unrun_tier_r_waivers:
            ``{tier_r_name: reason}`` waivers for an UNRUN Tier-R adapter (today:
            ``gliner2``). Per §5 the WHOLE Tier-R ∪ Tier-C set must be RUN-or-
            WAIVED before ``CLAIM_GRADE``; an unrun-unwaived Tier-R member blocks
            a claim just as an unrun Tier-C API does. Reason mandatory.
        """
        # Container-shape hardened (the S7-02 close-2 crash class): `benchmark` MUST
        # be a dict, and `run_metadata` MUST be a dict — but a malformed artifact can
        # ship `benchmark` as a list/scalar (`benchmark.get` → AttributeError) or
        # `run_metadata` as a JSON array (`{} or []` kept the truthy list, then
        # `.get` → AttributeError, crashing the shipped CLI). Normalise BOTH
        # fail-CLOSED: a non-dict benchmark / run_metadata is treated as ABSENT (an
        # empty dict) → canonical_claim_run=False → a clean NOT_YET, never a crash.
        if not isinstance(benchmark, dict):
            benchmark = {}
        run_metadata = benchmark.get("run_metadata")
        if not isinstance(run_metadata, dict):
            run_metadata = {}
        canonical = bool(run_metadata.get("canonical_claim_run", False))

        systems = _systems_by_name(benchmark)
        overrides = pending_overrides or {}

        # RecallFloorVerdictGuard input: the recall-floor-breaching systems (§3 G7).
        breachers = recall_floor_breachers(benchmark, systems)

        # G2 is now COMPUTED from the benchmark's distinct anon/pseudo family
        # fields (S4-01); an explicit pending_overrides['G2'] still wins (the
        # successor-override seam is preserved for callers that drive G2 directly).
        g2 = (
            _pending_guarantee("G2", overrides["G2"])
            if "G2" in overrides
            else _g2_pseudonymization_integrity(systems)
        )
        # G4 is now COMPUTED from the benchmark's per-class calibration / selective-
        # risk fields (S4-03); an explicit pending_overrides['G4'] still wins (the
        # successor-override seam is preserved for callers that drive G4 directly).
        g4 = (
            _pending_guarantee("G4", overrides["G4"])
            if "G4" in overrides
            else _g4_calibration_selective_risk(systems)
        )
        guarantees = (
            _g1_recall_floor(benchmark, systems),
            g2,
            _g3_recall_dominance(systems),
            g4,
            _pending_guarantee("G5", overrides.get("G5")),
            _g6_raw_noninferiority(systems),
            _g7_certified_run(run_metadata, systems, breachers),
        )

        # The Tier registry, with run-status derived from the benchmark + waivers.
        # Tier-R and Tier-C waivers share the same `waive` (reason-mandatory) path.
        # `run_status_from_benchmark` lives in the OFF-LIMITS (byte-identical)
        # competitor_tiers module and reads `systems` (iterating + `.get`) +
        # `available_competitors` (in a `set(...)`) UNGUARDED — a malformed-container
        # artifact would crash it (the S7-02 close-2 crash class) exactly as it crashed
        # `_systems_by_name`. Since that module cannot be edited, it is fed a
        # SHAPE-NORMALISED view: the already-validated `systems` map (dict-only
        # elements) re-listed, and `available_competitors` coerced to a list (a
        # non-list → empty). The gate shields the off-limits reader at the call site.
        registry = apply_run_status(
            default_registry(),
            run_status_from_benchmark(_tier_status_input(benchmark, systems)),
        )
        for name, reason in {**(tier_c_waivers or {}), **(unrun_tier_r_waivers or {})}.items():
            registry = waive(registry, name, reason)
        blocking_tier_c = unrun_tier_c(registry)
        blocking_tier_r = unrun_tier_r(registry)

        # J — the SDO objective (bayes if posterior supplied, else MLE-bootstrap);
        # the rank-1 argmax excludes recall-floor breachers (a breacher can never
        # be crowned), and `crowned` is the floor-compliant rank-1 system.
        j_value, j_source, crowned = _compute_j(
            benchmark, systems, theta_samples, posterior_names, breachers
        )

        verdict, binding = _decide(
            canonical=canonical,
            guarantees=guarantees,
            j_value=j_value,
            blocking_tier_c=blocking_tier_c,
            blocking_tier_r=blocking_tier_r,
        )

        axes_pending = tuple(
            _PENDING_SUCCESSORS[g.axis]
            for g in guarantees
            if g.passed is None and g.axis in _PENDING_SUCCESSORS
        )

        return cls(
            verdict=verdict,
            binding_constraint=binding,
            j_value=j_value,
            j_source=j_source,
            guarantees=guarantees,
            canonical_claim_run=canonical,
            unrun_tier_c=blocking_tier_c,
            axes_pending=axes_pending,
            j_rank1_system=crowned,
            recall_floor_breachers=breachers,
            unrun_tier_r=blocking_tier_r,
            tier_registry=registry,
        )


# ---------------------------------------------------------------------------
# Benchmark accessors
# ---------------------------------------------------------------------------


def _systems_by_name(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the benchmark's top-level ``systems`` list by ``system`` name.

    Container-shape hardened (the S7-02 close-2 crash class): the artifact-sourced
    ``systems`` field must be a LIST of system DICTs, but a malformed artifact can
    ship it as a JSON object (``{...}`` — iterating yields string keys → ``str.get``
    → ``AttributeError``), a scalar (not iterable), or a list carrying a bare-string /
    ``None`` / scalar element (``element.get`` → ``AttributeError``). Each of those
    crashed the shipped ``pii-anon supremacy`` CLI with a raw traceback (a fail-loud
    denial-of-verdict). This normalises fail-CLOSED: a non-list ``systems`` is treated
    as EMPTY (no systems), and a non-dict element is IGNORED (skipped) — so the
    returned map is ALWAYS ``dict[str, dict]`` and every downstream consumer
    (``_g1``..``_g7`` / ``_compute_j`` / ``_top_composite_system`` / ``_competitor_names``)
    can read ``.get`` / ``.items`` / iterate without a type check of its own.
    """
    out: dict[str, dict[str, Any]] = {}
    raw = benchmark.get("systems", [])
    if not isinstance(raw, list):
        return out  # a non-list `systems` (dict / scalar) → no systems (fail-closed)
    for system in raw:
        if not isinstance(system, dict):
            continue  # a bare-string / None / scalar element is ignored (never `.get`)
        name = system.get("system")
        if isinstance(name, str):
            out[name] = system
    return out


def _tier_status_input(
    benchmark: dict[str, Any], systems: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """A SHAPE-NORMALISED benchmark view for the OFF-LIMITS ``run_status_from_benchmark``.

    ``competitor_tiers.run_status_from_benchmark`` (byte-identical, un-editable —
    RISK-6) reads exactly two artifact-sourced containers UNGUARDED: ``systems``
    (iterated + ``.get`` per element) and ``available_competitors`` (consumed by
    ``set(...)``). A malformed-container artifact (``systems`` as a dict/scalar, a
    bare-string element, or ``available_competitors`` as a non-list) would crash it
    (the S7-02 close-2 crash class). The gate cannot edit that module, so it shields
    it at the call site: pass the ALREADY-VALIDATED ``systems`` map (dict-only
    elements, from :func:`_systems_by_name`) re-listed, and ``available_competitors``
    coerced to a list of its str (hashable) elements — a non-list → empty; non-str /
    unhashable elements (dict / nested list) are DROPPED so the off-limits ``set(...)``
    never sees an unhashable element. Pure: builds a new dict, never mutates the input.
    """
    available = benchmark.get("available_competitors")
    available_list = available if isinstance(available, list) else []
    return {
        "systems": list(systems.values()),
        # Only str (hashable) names survive: an UNHASHABLE element (dict / nested list)
        # would crash the off-limits ``set(available_competitors)`` (the S7-02 final-close
        # unhashable-element class); a non-str can never match a known competitor name, so
        # dropping it is honest + behaviour-preserving.
        "available_competitors": [c for c in available_list if isinstance(c, str)],
    }


def _competitor_names(systems: dict[str, dict[str, Any]]) -> list[str]:
    """The benchmarked competitor systems (everything outside the pii-anon ladder)."""
    return sorted(name for name in systems if name not in _LADDER_SYSTEMS)


def _recall_map(system: dict[str, Any]) -> dict[str, Any]:
    """A system's ``per_entity_recall`` as a dict — ``{}`` when absent OR malformed.

    Container-shape hardened (the S7-02 close-2 crash class): a malformed artifact can
    ship ``per_entity_recall`` as a JSON array — and the old ``(... or {}).items()`` /
    ``.values()`` then raised ``AttributeError: 'list' object has no attribute 'items'``
    (in G1) / ``...'values'`` (in G6). A non-dict ``per_entity_recall`` is treated as
    EMPTY (no per-entity recall), fail-CLOSED, so the callers can read ``.items`` /
    ``.values`` / ``len`` unconditionally.
    """
    per = system.get("per_entity_recall")
    return per if isinstance(per, dict) else {}


# ---------------------------------------------------------------------------
# RecallFloorVerdictGuard — the recall-floor-breach predicate (story §3 G7)
# ---------------------------------------------------------------------------


def _run_breaches_recall_floor(benchmark: dict[str, Any]) -> bool:
    """Does the RUN as a whole breach the recall floor? (recall-specific).

    Uses only RECALL signals, never the conflated ensemble ``floor_pass``:

    * a per-language ε artifact (``per_language_recall_delta``), if present, with
      any language ε > :data:`EPS_RECALL_PER_LANG` is a breach. An ABSENT artifact
      is PENDING-not-fabricated — it never manufactures a breach on its own;
    * any per-profile ``floor_checks`` entry whose ``metric`` is recall-like
      (recall / f1 / f2) and whose ``passed`` is ``False`` is a breach. A LATENCY
      or throughput floor-check failure is explicitly NOT a recall breach.
    """
    per_lang = benchmark.get("per_language_recall_delta")
    if isinstance(per_lang, dict) and per_lang:
        # Each ε is a DELTA (can be negative) — validated by `_is_finite_number`, NOT
        # `_finite_unit_score`. A non-finite ε (NaN / ±inf) or a Python int wider than
        # a C double (10**400) is NOT a real measurement: it must be treated as a
        # breach (the run cannot be certified floor-compliant from a corrupt ε), never
        # slip ``abs() > bound`` (silently False against NaN) and never raise
        # OverflowError in ``float()`` (the S7-02 close MAJOR-3 — the shipped
        # ``pii-anon supremacy`` CLI crashed). Fail CLOSED.
        worst = 0.0
        for v in per_lang.values():
            if not _is_finite_number(v):
                return True  # corrupt ε ⇒ uncertifiable ⇒ breach (never a crash)
            worst = max(worst, abs(float(v)))
        if worst > EPS_RECALL_PER_LANG:
            return True

    # `profile_results` / `floor_checks` must be LISTS — but a malformed artifact can
    # ship either as a scalar (the S7-02 close-2 crash class): `42 or []` keeps the
    # truthy `42`, then `for profile in 42` → `TypeError: 'int' object is not iterable`.
    # A non-list is treated as EMPTY (no profiles / no floor-checks) — fail-CLOSED, no
    # crash. (A dict is technically iterable but yields keys, caught by the per-element
    # `isinstance(..., dict)` skips below; the list guard handles the scalar.)
    profile_results = benchmark.get("profile_results")
    if not isinstance(profile_results, list):
        return False
    for profile in profile_results:
        if not isinstance(profile, dict):
            continue
        floor_checks = profile.get("floor_checks")
        if not isinstance(floor_checks, list):
            continue
        for check in floor_checks:
            if not isinstance(check, dict):
                continue
            metric = check.get("metric")
            # ``metric`` must be a str: an UNHASHABLE metric (dict / list) would crash
            # ``metric in _RECALL_FLOOR_METRICS`` (the S7-02 final-close unhashable class);
            # a non-str metric can never name a recall-floor metric, so it is no breach.
            if (
                isinstance(metric, str)
                and metric in _RECALL_FLOOR_METRICS
                and check.get("passed") is False
            ):
                return True
    return False


def recall_floor_breachers(
    benchmark: dict[str, Any], systems: dict[str, dict[str, Any]]
) -> frozenset[str]:
    """Return the systems that BREACH the recall floor (the guard's input set).

    The RecallFloorVerdictGuard (story §3 G7) enforces that a recall-floor
    breacher can never top-rank. A system breaches when EITHER:

    * its per-system ``qualification_status`` is non-qualifying — i.e. NOT in
      :data:`_QUALIFYING_STATUSES` (a disqualified / floor-breach / failed
      status). A missing status is treated as non-qualifying (fail loud, never
      silently pass); OR
    * the RUN breaches the recall floor (:func:`_run_breaches_recall_floor` —
      a per-language ε regression or a failing recall/f1 profile floor-check). A
      run-level recall breach taints EVERY system in the run (no clean top-rank
      crowning is possible while the run's recall floor is breached).

    Pure: reads the benchmark, never mutates it. The conflated ensemble
    ``floor_pass`` (which also folds in latency/throughput floors) is deliberately
    NOT consulted — the guard is recall-specific (see module note).
    """
    run_breach = _run_breaches_recall_floor(benchmark)
    breachers: set[str] = set()
    for name, system in systems.items():
        status = system.get("qualification_status")
        status_ok = isinstance(status, str) and status in _QUALIFYING_STATUSES
        if run_breach or not status_ok:
            breachers.add(name)
    return frozenset(breachers)


# ---------------------------------------------------------------------------
# Guarantees (pure functions over the benchmark; three-valued)
# ---------------------------------------------------------------------------


def _g1_recall_floor(
    benchmark: dict[str, Any], systems: dict[str, dict[str, Any]]
) -> GuaranteeResult:
    """G1 — entities(ensemble) ⊇ entities(shared) ∧ per-language ε ≤ 0.005.

    The structural superset is computable from per-entity recall; the
    per-language ε comes from a ``per_language_recall_delta`` artifact and is
    PENDING (``None``) when that artifact is absent — NEVER fabricated.
    """
    core = systems.get(_CORE_SYSTEM)
    if core is None:
        return GuaranteeResult(
            "G1", None, float("nan"), EPS_RECALL_PER_LANG,
            "G1 PENDING: pii-anon core absent from benchmark systems",
        )

    # Structural: the pii-anon ladder must detect every entity any competitor
    # detects (the recall-floor-by-construction invariant).
    ensemble_entities: set[str] = set()
    for ladder in _LADDER_SYSTEMS:
        sys = systems.get(ladder)
        if sys:
            ensemble_entities |= {
                e for e, r in _recall_map(sys).items()
                if isinstance(r, (int, float))
            }
    shared_entities: set[str] = set()
    for name in _competitor_names(systems):
        per = _recall_map(systems[name])
        shared_entities |= {
            e for e, r in per.items()
            if isinstance(r, (int, float)) and r > 0.0
        }
    missing = shared_entities - ensemble_entities
    superset_ok = not missing

    # Per-language ε (PENDING if the artifact is absent OR MALFORMED). A non-dict
    # per_language_recall_delta (a list / scalar — the S7-02 close-2 crash class) is
    # absent-EQUIVALENT: it cannot certify the per-language ε bound, so it is PENDING,
    # NEVER silently read as present-with-no-items (which fabricated a "worst ε=0" PASS
    # on malformed input). PENDING never fabricates and never crashes.
    per_lang = benchmark.get("per_language_recall_delta")
    if not isinstance(per_lang, dict):
        return GuaranteeResult(
            "G1", None, float("nan"), EPS_RECALL_PER_LANG,
            "G1 PENDING: per_language_recall_delta artifact absent or malformed "
            "(structural superset "
            + ("holds" if superset_ok else f"BREACHED on {sorted(missing)}")
            + "); per-language ε never fabricated",
        )
    # Each ε is a DELTA (can be negative) — validated by `_is_finite_number`, NOT
    # `_finite_unit_score`. A non-finite ε (NaN / ±inf) or a Python int wider than a
    # C double (10**400) is rejected and fails CLOSED: G1 FAIL (the run's recall floor
    # cannot be certified from a corrupt ε), never an OverflowError crash in ``float()``
    # (the S7-02 close MAJOR-3) and never a silent PASS (NaN would otherwise slip
    # ``abs() ≤ bound``). The worst corrupt language is named.
    per_lang_items = list(per_lang.items())
    corrupt_langs = [lang for lang, v in per_lang_items if not _is_finite_number(v)]
    worst_lang_eps = max(
        (abs(float(v)) for _, v in per_lang_items if _is_finite_number(v)),
        default=0.0,
    )
    eps_ok = not corrupt_langs and worst_lang_eps <= EPS_RECALL_PER_LANG

    passed = superset_ok and eps_ok
    if passed:
        detail = (
            f"G1 PASS: ensemble entities ⊇ shared; worst per-language "
            f"ε={worst_lang_eps:.4g} ≤ {EPS_RECALL_PER_LANG}"
        )
    elif not superset_ok:
        detail = f"G1 FAIL: ensemble misses competitor-detected entities {sorted(missing)}"
    elif corrupt_langs:
        detail = (
            f"G1 FAIL: per-language recall ε on {sorted(corrupt_langs)} is non-finite / "
            f"out-of-float-range — a corrupt measurement that cannot certify the recall "
            f"floor (never slips ≤ {EPS_RECALL_PER_LANG}, never crashes)"
        )
    else:
        detail = (
            f"G1 FAIL: per-language recall ε={worst_lang_eps:.4g} > "
            f"{EPS_RECALL_PER_LANG}"
        )
    return GuaranteeResult("G1", passed, worst_lang_eps, EPS_RECALL_PER_LANG, detail)


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Is ``value`` a real, finite number (an ``int``/``float`` that is not a bool)?

    A :data:`~typing.TypeGuard` so callers narrow ``value`` to ``int | float`` after
    the check. Used where a finiteness check (not a unit-interval bound) is what's
    wanted — e.g. a per-class ECE must be finite (a ``NaN`` must never slip
    ``ece > bar``, which is silently ``False`` against ``NaN``) but its magnitude is
    bounded by the bar comparison itself, not by ``[0, 1]``. Excludes ``bool``
    (``isinstance(True, int)`` is ``True`` in Python, but a flag is not a
    measurement) so ``True`` / ``False`` never count as numbers.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    # math.isfinite raises OverflowError on a Python int wider than a C double
    # (e.g. 10**400 — "int too large to convert to float"). A control-path
    # validator must REJECT such an out-of-float-range field (fail CLOSED), never
    # crash on adversarial input (vector #11 — a fail-loud denial-of-verdict).
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _finite_unit_score(value: object) -> float | None:
    """Return ``value`` as a float IFF it is a real, finite, unit-interval score.

    A *usable* score for the SDO moat axes (G2 pseudonymization-integrity, G4
    per-class ECE, the NFR-014 unauthorized-reversal rate) is, by construction,
    a real number in ``[0, 1]`` (e.g. ``deid_families.integrity_score`` is bounded
    ``[0, 1]``; a calibration error / coverage fraction is a probability). This
    guard returns the value only when it is ALL of:

    * a genuine ``int``/``float`` and **not** a ``bool`` — ``isinstance(True, int)``
      is ``True`` in Python, so a flag would otherwise coerce to ``1.0`` and
      fabricate a perfect score / a vacuous dominance win;
    * finite — ``NaN`` / ``±inf`` are not measurements (``+inf`` must never
      "dominate", a ``NaN`` must never slip a ``x > bar`` comparison whose result
      against ``NaN`` is silently ``False``);
    * within ``[0, 1]`` — a value outside the unit interval is a corrupt artifact,
      never a stronger score.

    Returns ``None`` otherwise, so a caller can treat the field as ABSENT
    (PENDING / fall back to a sanctioned bar) rather than trust a fabricated value.
    Built on :func:`_is_finite_number` (the shared type-and-finiteness predicate);
    this adds only the unit-interval bound.
    """
    if not _is_finite_number(value):
        return None
    v = float(value)
    if v < 0.0 or v > 1.0:
        return None
    return v


def _g4_class_bar(artifact_threshold: object) -> float:
    """The authoritative per-class ECE bar, TIGHTENED — never loosened — by the
    artifact-supplied threshold.

    The gate owns the sanctioned bars (NFR-017): high-resource
    :data:`G4_ECE_BAR_HIGH_RESOURCE` (0.05) and long-tail
    :data:`G4_ECE_BAR_LONG_TAIL` (0.08). The LOOSEST bar the gate will EVER permit
    is the long-tail 0.08 — that is the hard sanctioned ceiling. An artifact-
    supplied per-class threshold may only make the bar STRICTER:
    ``bar = min(artifact_threshold, G4_ECE_BAR_LONG_TAIL)`` — so a legitimate
    long-tail 0.08 (or a tighter 0.03) is honored, but a bar-loosening 0.99 is
    clamped down to 0.08 and cannot mask a high ECE. A non-finite / out-of-range
    artifact threshold (``NaN`` / ``+inf`` / negative / ``> 1``) is REJECTED and the
    conservative high-resource 0.05 default stands. Absent ⇒ the high-resource 0.05
    default (a class is long-tail only when the artifact explicitly requests the
    looser bar). The gate's bar is authoritative, not artifact-overridable upward.
    """
    candidate = _finite_unit_score(artifact_threshold)
    if candidate is None:
        return G4_ECE_BAR_HIGH_RESOURCE
    return min(candidate, G4_ECE_BAR_LONG_TAIL)


def _g2_pseudonymization_integrity(
    systems: dict[str, dict[str, Any]],
) -> GuaranteeResult:
    """G2 — pseudonymization-integrity strict dominance (S4-01; three-valued).

    The SDO moat axis for reversibility (AX-004 / FR-006 / FR-009 / NFR-014). The
    benchmark must carry, per system, BOTH distinct de-id family fields written by
    the S4-01 scorers (kept structurally separate — never one merged number):

    * ``pseudonymization_integrity_score`` — the reversible-under-key family score;
    * ``anonymization_score`` — the irreversible family score (present-and-distinct
      is required so the two-family contract is actually satisfiable from the
      artifact — a half-populated artifact cannot fabricate the missing half).

    Three-valued outcome:

    * **PASS** iff pii-anon's ``pseudonymization_integrity_score`` STRICTLY
      dominates every *real* competitor's, the pii-anon
      ``unauthorized_reversal_rate`` is 0 (NFR-014), and BOTH family fields are
      present-and-valid (finite, in ``[0, 1]``, non-boolean) on the claimant;
    * **FAIL** iff a real comparator exists but pii-anon does not strictly dominate
      (binding detail names the gap), OR the unauthorized-reversal rate is > 0;
    * **PENDING** (``None``) iff there is no real claimant score — both family
      fields absent, a boolean masquerading as one (which would coerce to ``1.0``),
      or a non-finite / out-of-``[0, 1]`` value (a corrupt artifact, never a
      stronger score) — OR no competitor carries a real
      ``pseudonymization_integrity_score`` (no comparator ⇒ dominance is unprovable
      and must NEVER be fabricated against a phantom ``0``; this includes the
      zero-competitor case). The current smoke artifact lacks the fields; the S7
      canonical run emits them.

    A *real* score is validated by :func:`_finite_unit_score` (finite, in
    ``[0, 1]``, non-boolean): a bool is a flag, a non-finite float is not a
    measurement, and an out-of-range value is corrupt — none may seed a dominance
    claim or stand in as a comparator bar.

    Pure: reads the benchmark systems, never mutates them. Determinism: a pure
    function of the (numeric) fields (AX-002).
    """
    core = systems.get(_CORE_SYSTEM)
    if core is None:
        return GuaranteeResult(
            "G2", None, float("nan"), float("nan"),
            "G2 PENDING: pii-anon core absent from benchmark systems "
            f"({_PENDING_SUCCESSORS['G2']})",
        )

    # BOTH distinct family fields must be present as REAL (finite, [0,1],
    # non-boolean) scores (two-family contract). A missing field — a boolean
    # masquerading as one (isinstance(True, int) is True, which would coerce to
    # 1.0) — or a non-finite / out-of-range value (a corrupt artifact:
    # deid_families.integrity_score is bounded [0,1] by construction) ⇒ PENDING
    # (never fabricated, never a vacuous +inf/1.5 "win").
    core_pi = _finite_unit_score(core.get("pseudonymization_integrity_score"))
    core_anon = _finite_unit_score(core.get("anonymization_score"))
    if core_pi is None or core_anon is None:
        return GuaranteeResult(
            "G2", None, float("nan"), float("nan"),
            "G2 PENDING: benchmark lacks a valid (finite, in [0,1]) value for the "
            "distinct anon/pseudo family fields (pseudonymization_integrity_score + "
            f"anonymization_score); {_PENDING_SUCCESSORS['G2']} — emitted by the S7 "
            "canonical run",
        )

    # The NFR-014 unauthorized-reversal rate is a fraction in [0,1]; a non-finite /
    # out-of-range value is a corrupt artifact, not a real measurement ⇒ PENDING
    # (we cannot certify "exactly 0" from a corrupt rate, and must not fabricate it).
    unauthorized_raw = core.get("unauthorized_reversal_rate", 0.0)
    unauthorized = _finite_unit_score(unauthorized_raw)
    if unauthorized is None:
        return GuaranteeResult(
            "G2", None, float("nan"), float("nan"),
            "G2 PENDING: pii-anon unauthorized_reversal_rate is not a valid "
            "(finite, in [0,1]) rate — cannot certify NFR-014 from a corrupt value; "
            f"{_PENDING_SUCCESSORS['G2']} — emitted by the S7 canonical run",
        )

    # Competitors' pseudonymization-integrity — ONLY those carrying a REAL, finite,
    # in-range, non-boolean score count as comparators (a bool / non-finite /
    # out-of-range value is not a real measurement, so it is EXCLUDED rather than
    # coerced or treated as a phantom bar).
    competitor_pis = [
        cp
        for name in _competitor_names(systems)
        if (cp := _finite_unit_score(systems[name].get("pseudonymization_integrity_score")))
        is not None
    ]
    # No real comparator ⇒ strict dominance is UNPROVABLE. Return PENDING rather
    # than fabricate a win against a phantom 0.0 baseline — symmetric with the
    # core-side guard above. This covers BOTH "competitors present but none carry a
    # valid field" and the zero-competitor case; never fabricated.
    if not competitor_pis:
        return GuaranteeResult(
            "G2", None, float("nan"), float("nan"),
            "G2 PENDING: no competitor carries a valid pseudonymization_integrity_"
            "score — strict dominance is unprovable without a comparator (never "
            f"fabricated against a phantom 0); {_PENDING_SUCCESSORS['G2']} — emitted "
            "by the S7 canonical run",
        )

    best_competitor_pi = max(competitor_pis)
    dominates = core_pi > best_competitor_pi
    no_unauthorized = unauthorized == 0.0

    passed = dominates and no_unauthorized
    if passed:
        detail = (
            f"G2 PASS: pii-anon pseudonymization-integrity {core_pi:.4g} strictly "
            f"dominates best competitor {best_competitor_pi:.4g}; "
            "unauthorized-reversal=0 (NFR-014); both anon/pseudo family fields "
            "present-and-distinct (AX-004)"
        )
    elif not no_unauthorized:
        detail = (
            f"G2 FAIL: pii-anon unauthorized-reversal rate {unauthorized:.4g} > 0 "
            "(NFR-014 requires exactly 0 — a reversal without the authorized key "
            "is a breach)"
        )
    else:
        detail = (
            f"G2 FAIL: pii-anon pseudonymization-integrity {core_pi:.4g} does not "
            f"strictly dominate best competitor {best_competitor_pi:.4g}"
        )
    return GuaranteeResult("G2", passed, core_pi, best_competitor_pi, detail)


def _g3_recall_dominance(systems: dict[str, dict[str, Any]]) -> GuaranteeResult:
    """G3 — the recall-optimised pii-anon system's recall ≥ max(competitor recall).

    Per story §3 the recall-dominance guarantee is measured on the
    recall-optimised member of the pii-anon ladder (``pii-anon-swarm`` when
    present, else ``pii-anon``) — the system the SDO claims wins the recall moat
    axis (de-risk: pii-anon-swarm 0.818 ≥ gliner 0.658 today).
    """
    competitors = _competitor_names(systems)
    if not competitors:
        return GuaranteeResult(
            "G3", None, float("nan"), float("nan"),
            "G3 PENDING: no competitor systems in benchmark",
        )
    swarm = systems.get("pii-anon-swarm") or systems.get(_CORE_SYSTEM)
    if swarm is None:
        return GuaranteeResult(
            "G3", None, float("nan"), float("nan"),
            "G3 PENDING: no pii-anon ladder system in benchmark",
        )
    # Recall is a fraction in [0, 1] — validated by `_finite_unit_score` so a corrupt
    # recall (a bool, a NaN / ±inf, or a Python int wider than a C double like 10**400)
    # is REJECTED rather than crashing ``float()`` (the S7-02 close also-scan) or
    # fabricating a vacuous dominance (``+inf >= best`` would be a phantom win). An
    # invalid CLAIMANT recall ⇒ PENDING (no usable measurement, never a fabricated
    # PASS); an invalid COMPETITOR recall is EXCLUDED from the max bar.
    swarm_recall = _finite_unit_score(swarm.get("recall"))
    if swarm_recall is None:
        return GuaranteeResult(
            "G3", None, float("nan"), float("nan"),
            "G3 PENDING: pii-anon ladder recall is not a valid (finite, in [0,1]) "
            "measurement — dominance is unprovable from a corrupt value (never "
            "fabricated, never crashed)",
        )
    competitor_recalls = [
        cr
        for c in competitors
        if (cr := _finite_unit_score(systems[c].get("recall"))) is not None
    ]
    if not competitor_recalls:
        return GuaranteeResult(
            "G3", None, float("nan"), float("nan"),
            "G3 PENDING: no competitor carries a valid (finite, in [0,1]) recall — "
            "dominance is unprovable without a real comparator (never fabricated)",
        )
    best_competitor = max(competitor_recalls)
    passed = swarm_recall >= best_competitor
    detail = (
        f"G3 {'PASS' if passed else 'FAIL'}: pii-anon-swarm recall "
        f"{swarm_recall:.4g} {'≥' if passed else '<'} best competitor "
        f"{best_competitor:.4g}"
    )
    return GuaranteeResult("G3", passed, swarm_recall, best_competitor, detail)


def _g4_calibration_selective_risk(
    systems: dict[str, dict[str, Any]],
) -> GuaranteeResult:
    """G4 — per-class calibration / selective-risk (S4-03; three-valued).

    The SDO moat axis for calibrated abstention (AX-005 / FR-005 / NFR-017..021).
    The benchmark must carry, on the pii-anon claimant, the calibration block the
    :class:`~pii_anon.eval_framework.metrics.selective_risk.SelectiveRiskReporter`
    emits (the S7 canonical run stamps it):

    * ``per_class_ece`` — ``{entity_type: ece}`` (post-temperature-scaling); each
      ECE must be FINITE **and NON-NEGATIVE** — a non-finite ECE (``NaN`` / ``±inf``)
      OR a negative ECE (an Expected Calibration Error is ``≥ 0`` by construction) is
      a corrupt present-but-unusable measurement and a breach (a ``NaN`` must never
      slip the ``ece > bar`` comparison, whose result against ``NaN`` is silently
      ``False``; a negative ECE like ``-1.0`` is ``< bar`` and would otherwise count
      as "within bar" and forge a moat-axis PASS — the S7-02 close-3 fabrication);
    * ``per_class_ece_threshold`` (optional) — ``{entity_type: bar}``; the gate's
      OWN sanctioned bars (:data:`G4_ECE_BAR_HIGH_RESOURCE` 0.05 high-resource /
      :data:`G4_ECE_BAR_LONG_TAIL` 0.08 long-tail) are AUTHORITATIVE. An artifact-
      supplied per-class threshold may only TIGHTEN the bar, never loosen it:
      ``bar = min(artifact_threshold, sanctioned_bar)``; a non-finite / out-of-
      range artifact threshold is REJECTED (fall back to the sanctioned bar) so a
      benchmark can never mask a high ECE with a loosened (e.g. ``0.99`` / ``+inf``)
      threshold;
    * ``risk_coverage_curve`` — ``[{"coverage", "risk"}, …]`` ascending in
      coverage (the monotone-non-increasing selective-risk curve, NFR-019);
    * ``abstention_operating_points`` — ``≥3`` rows (NFR-021);
    * ``calibrated_confidence_coverage`` — the NFR-020 audit; must be EXACTLY
      ``1.0`` — finite AND in ``[0, 1]`` AND ``== 1.0`` (a value ``> 1.0`` / ``+inf``
      is corrupt, never a pass; the lone MUST is exact 100% coverage).

    Three-valued outcome:

    * **PASS** iff EVERY per-class ECE is finite AND non-negative AND ≤ its (clamped)
      bar AND the risk-coverage curve is monotone non-increasing (risk never drops as
      coverage grows) AND there are ≥ :data:`G4_MIN_ABSTENTION_POINTS` points AND
      ``calibrated_confidence_coverage`` is finite, in ``[0, 1]``, and ``== 1.0``
      (NFR-020 — 0 bare-logit);
    * **FAIL** iff the fields are present but a check is breached (the binding
      detail names the breach — the worst / non-finite class, the coverage gap, the
      non-monotone curve, or the missing operating points);
    * **PENDING** (``None``) iff the benchmark lacks the calibration fields (the
      current smoke artifact lacks them — NO fabrication; the S7 canonical run
      emits them).

    The gate's bar is authoritative, NOT artifact-overridable upward, and no
    artifact-supplied value (a non-finite ECE, a coverage ``> 1.0``, a loosened
    threshold) can fabricate a PASS — the inputs are VALIDATED before they gate.

    Pure: reads the benchmark systems, never mutates them. Determinism (AX-002):
    a pure function of the (numeric) fields. The reporting MATH lives in
    ``metrics/selective_risk.py``; this gate only READS the emitted summary.
    """
    core = systems.get(_CORE_SYSTEM)
    if core is None:
        return GuaranteeResult(
            "G4", None, float("nan"), float("nan"),
            "G4 PENDING: pii-anon core absent from benchmark systems "
            f"({_PENDING_SUCCESSORS['G4']})",
        )

    per_class_ece = core.get("per_class_ece")
    coverage = core.get("calibrated_confidence_coverage")
    # The calibration block must be present (per-class ECE + the NFR-020 coverage
    # are the load-bearing fields); a missing field ⇒ PENDING (never fabricated).
    if not isinstance(per_class_ece, dict) or not per_class_ece or not isinstance(
        coverage, (int, float)
    ):
        return GuaranteeResult(
            "G4", None, float("nan"), float("nan"),
            "G4 PENDING: benchmark lacks the per-class calibration / selective-risk "
            "fields (per_class_ece + calibrated_confidence_coverage + "
            "risk_coverage_curve + abstention_operating_points); "
            f"{_PENDING_SUCCESSORS['G4']} — emitted by the S7 canonical run",
        )

    thresholds = core.get("per_class_ece_threshold")
    if not isinstance(thresholds, dict):
        thresholds = {}

    # 1. Per-class ECE ≤ its (clamped) bar (worst class drives the FAIL detail).
    #    Each ECE must be FINITE and NON-NEGATIVE — a non-finite ECE (NaN / ±inf) OR a
    #    negative ECE (ECE ≥ 0 by construction) is a corrupt present-but-unusable
    #    measurement and a breach. NaN would otherwise slip `ece > bar` (silently
    #    False against NaN); a negative ECE (-1.0) is `< bar` and would otherwise
    #    count as "within bar" and forge a PASS ("worst -1.0") — the S7-02 close-3
    #    fabrication. The per-class bar is the gate's sanctioned default (high-resource
    #    0.05), TIGHTENED — never loosened — by an artifact-supplied threshold via
    #    `_g4_class_bar` (so a benchmark cannot mask a high ECE with a 0.99 / +inf
    #    threshold).
    worst_detail = ""
    worst_slack = -math.inf  # (ece - bar); a non-finite ECE is the worst breach
    ece_ok = True
    finite_eces: list[float] = []
    for et in sorted(per_class_ece):
        raw = per_class_ece[et]
        bar = _g4_class_bar(thresholds.get(et))
        if not _is_finite_number(raw):
            ece_ok = False
            if worst_slack < math.inf:
                worst_slack = math.inf
                worst_detail = (
                    f"per-class ECE on {et!r} is non-finite ({raw!r}) — a corrupt "
                    f"measurement, not ≤ bar {bar:.4g} (NFR-017)"
                )
            continue
        ece = float(raw)
        # A NEGATIVE ECE is non-physical: an Expected Calibration Error is ≥ 0 by
        # construction. A sub-zero value (e.g. -1.0) is a corrupt present-but-
        # unusable measurement (symmetric with rejecting a coverage outside [0,1])
        # — a BREACH, NOT "within bar" (the S7-02 close-3 fabrication: -1.0 < bar
        # slipped 'every ECE ≤ bar' and forged a moat-axis PASS, "worst -1.0").
        # It is EXCLUDED from `finite_eces` so garbage never stands in as the worst
        # observed ECE, and ranked by its (negative) `ece - bar` slack so a co-located
        # GENUINE over-bar class (positive slack) stays the headline breach while a
        # lone negative is the reported breach.
        if ece < 0.0:
            ece_ok = False
            slack = ece - bar
            if slack > worst_slack:
                worst_slack = slack
                worst_detail = (
                    f"per-class ECE on {et!r} is negative ({ece:.4g}) — non-physical "
                    f"(ECE ≥ 0 by construction), a corrupt measurement, not ≤ bar "
                    f"{bar:.4g} (NFR-017)"
                )
            continue
        finite_eces.append(ece)
        if ece > bar:
            ece_ok = False
            slack = ece - bar
            if slack > worst_slack:
                worst_slack = slack
                worst_detail = (
                    f"per-class ECE breach on {et!r} (ECE={ece:.4g} > bar "
                    f"{bar:.4g}; NFR-017)"
                )
    max_ece = max(finite_eces, default=0.0)

    # 2. Monotone-non-increasing risk-coverage curve (NFR-019): ordered by
    #    ascending coverage, risk must be non-decreasing (selective risk does not
    #    increase as you abstain MORE / does not drop as coverage grows).
    curve = core.get("risk_coverage_curve") or []
    monotone_ok = _risk_coverage_is_monotone(curve)

    # 3. ≥3 abstention operating points (NFR-021). The artifact-sourced field must be
    #    a LIST — but a malformed artifact can ship it as a scalar (42 / 3.14 / True).
    #    The count is computed ONCE, safely (0 for a non-list), and reused in BOTH the
    #    PASS- and FAIL-detail branches below: the old FAIL-detail `else` called
    #    `len(points)` UNCONDITIONALLY → `TypeError: object of type 'int' has no len()`
    #    when the other three G4 checks passed and `points` was a non-list (the S7-02
    #    close-2 crash-1). A non-list now reads 0 points ⇒ G4 FAIL (it cannot supply
    #    ≥3 operating points), never a crash.
    points = core.get("abstention_operating_points")
    n_points = len(points) if isinstance(points, list) else 0
    points_ok = n_points >= G4_MIN_ABSTENTION_POINTS

    # 4. Calibrated-confidence-coverage == 1.0 (NFR-020 — the lone MUST). The
    #    coverage is a fraction: it must be FINITE and in [0, 1] (a value > 1.0 /
    #    +inf is corrupt, never a pass) AND exactly 1.0 (100% calibrated-confidence
    #    coverage; a tiny eps absorbs float round-trip noise but never admits >1.0).
    #    Routed through `_finite_unit_score` (the shared moat-axis validator) so a
    #    BOOL coverage — `isinstance(True, int)` is True in Python and `float(True)`
    #    is 1.0 — is REJECTED (a flag is not a measurement; it would otherwise
    #    fabricate a perfect-coverage PASS, the S7-02 close MAJOR-1), exactly as the
    #    G2 family fields reject a bool. A rejected (None) coverage is a corrupt /
    #    out-of-range present value ⇒ a FAIL (not a pass), reported with its raw repr.
    coverage_score = _finite_unit_score(coverage)
    coverage_in_range = coverage_score is not None
    coverage_ok = (
        coverage_score is not None
        and coverage_score >= G4_COVERAGE_REQUIRED - 1e-9
    )

    passed = ece_ok and monotone_ok and points_ok and coverage_ok
    if passed:
        detail = (
            f"G4 PASS: every per-class ECE ≤ its bar (worst {max_ece:.4g}); "
            f"risk-coverage monotone non-increasing; {n_points} abstention "
            f"operating points (≥{G4_MIN_ABSTENTION_POINTS}); "
            f"calibrated-confidence-coverage={coverage_score:.4g} (NFR-020 / AX-005)"
        )
    elif not coverage_ok:
        if not coverage_in_range:
            detail = (
                f"G4 FAIL: calibrated-confidence-coverage={coverage!r} is not a "
                f"valid fraction (finite, in [0,1], non-bool) — a bool / value >1.0 / "
                f"non-finite is a corrupt artifact, never a pass (NFR-020 requires "
                f"EXACTLY 1.0)"
            )
        else:
            detail = (
                f"G4 FAIL: calibrated-confidence-coverage={coverage_score:.4g} < "
                f"{G4_COVERAGE_REQUIRED} (NFR-020 — every finding MUST carry a "
                "calibrated confidence + provenance; a bare-logit finding is a "
                "breach)"
            )
    elif not ece_ok:
        detail = f"G4 FAIL: {worst_detail}"
    elif not monotone_ok:
        detail = (
            "G4 FAIL: risk-coverage curve is NOT monotone non-increasing — "
            "selective risk increases as coverage drops (NFR-019 violated)"
        )
    else:
        detail = (
            f"G4 FAIL: only {n_points} abstention operating point(s) "
            f"(< {G4_MIN_ABSTENTION_POINTS} required; NFR-021)"
        )
    return GuaranteeResult("G4", passed, max_ece, G4_ECE_BAR_HIGH_RESOURCE, detail)


def _risk_coverage_is_monotone(curve: Any) -> bool:
    """True iff the risk-coverage ``curve`` is monotone non-increasing in
    abstention — i.e. ordered by ascending coverage, ``risk`` is non-decreasing.

    The curve is a list of ``{"coverage", "risk"}`` dicts. An empty / malformed
    curve is treated as NOT monotone (fail loud — a missing curve cannot certify
    the NFR-019 property). Sorted by coverage before the check so emission order
    does not matter.
    """
    if not isinstance(curve, list) or not curve:
        return False
    try:
        pts = sorted(
            ((float(p["coverage"]), float(p["risk"])) for p in curve),
            key=lambda cr: cr[0],
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        # OverflowError: a coverage/risk that is a Python int wider than a C double
        # (10**400) — a malformed curve cannot certify NFR-019; fail loud, never crash.
        return False
    risks = [r for _, r in pts]
    return all(hi >= lo - 1e-9 for lo, hi in zip(risks, risks[1:]))


def _finite_f2(system: dict[str, Any]) -> float | None:
    """The F2 of a system from its ``precision`` + ``recall``, or ``None`` if either
    is not a real (finite, in ``[0, 1]``, non-boolean) measurement.

    Precision and recall are fractions in ``[0, 1]``; routing each through
    :func:`_finite_unit_score` rejects a corrupt value (a bool, a ``NaN`` / ``±inf``,
    or a Python int wider than a C double like ``10**400``) BEFORE it reaches
    :func:`f_beta` — so G6 never crashes on ``float()`` (the S7-02 close also-scan) and
    a corrupt ``+inf`` recall never produces a vacuous F2 that silently passes the
    non-inferiority comparison (a ``NaN`` F2 makes ``core_f2 >= bar`` silently False).
    """
    precision = _finite_unit_score(system.get("precision"))
    recall = _finite_unit_score(system.get("recall"))
    if precision is None or recall is None:
        return None
    return f_beta(precision, recall, beta=2.0)


def _g6_raw_noninferiority(systems: dict[str, dict[str, Any]]) -> GuaranteeResult:
    """G6 — core F2 ≥ best **Tier-R** competitor F2 − ε_F ∧ coverage ≥ 0.80.

    Only the runnable Tier-R competitors count (the OpenAI carve-out): a Tier-C
    system in the benchmark is ignored here.
    """
    core = systems.get(_CORE_SYSTEM)
    if core is None:
        return GuaranteeResult(
            "G6", None, float("nan"), float("nan"),
            "G6 PENDING: pii-anon core absent from benchmark systems",
        )
    # The core F2 is computed only from VALIDATED precision/recall (finite, in [0,1],
    # non-boolean); a corrupt core precision/recall ⇒ PENDING (no usable claimant F2,
    # never a fabricated non-inferiority PASS, never an OverflowError crash — the
    # S7-02 close also-scan).
    core_f2 = _finite_f2(core)
    if core_f2 is None:
        return GuaranteeResult(
            "G6", None, float("nan"), float("nan"),
            "G6 PENDING: pii-anon core precision/recall is not a valid (finite, in "
            "[0,1]) measurement — raw non-inferiority is unprovable from a corrupt "
            "value (never fabricated, never crashed). " + _CARVE_OUT_NOTE,
        )

    # Only Tier-R competitors carrying VALIDATED precision/recall seed the F2 bar; a
    # corrupt Tier-R precision/recall is EXCLUDED rather than crashing float() or
    # standing in as a phantom +inf bar.
    tier_r_f2 = [
        f2
        for name in systems
        if name in TIER_R_NAMES and (f2 := _finite_f2(systems[name])) is not None
    ]
    best_tier_r = max(tier_r_f2, default=0.0)
    f2_ok = core_f2 >= best_tier_r - EPS_F2

    # Entity coverage: fraction of entity types the core actually detects. A non-dict
    # per_entity_recall (the S7-02 close-2 crash class) reads as EMPTY via `_recall_map`
    # ⇒ 0 detected / 0 total ⇒ coverage 0.0 (fail-closed), never an AttributeError on
    # `.values()` / `len`.
    per = _recall_map(core)
    detected = sum(1 for r in per.values() if isinstance(r, (int, float)) and r > 0.0)
    total = len(per)
    coverage = (detected / total) if total else 0.0
    coverage_ok = coverage >= ENTITY_COVERAGE_MIN

    passed = f2_ok and coverage_ok
    detail = (
        f"G6 {'PASS' if passed else 'FAIL'}: core F2={core_f2:.4g} vs best Tier-R "
        f"F2={best_tier_r:.4g} (ε_F={EPS_F2}); entity coverage={coverage:.3g} "
        f"(≥{ENTITY_COVERAGE_MIN}). " + _CARVE_OUT_NOTE
    )
    return GuaranteeResult("G6", passed, core_f2, best_tier_r - EPS_F2, detail)


def _g7_certified_run(
    run_metadata: dict[str, Any],
    systems: dict[str, dict[str, Any]],
    breachers: frozenset[str],
) -> GuaranteeResult:
    """G7 — canonical_claim_run True ∧ full provenance ∧ RecallFloorVerdictGuard.

    The RecallFloorVerdictGuard sub-condition (story §3 G7): a recall-floor
    breaching system can never top-rank, so G7 FAILS if either

    * the highest-composite system in the benchmark is a recall-floor breacher
      (a breacher would otherwise be crowned rank-1); or
    * the pii-anon claimant itself breaches the recall floor.

    Guard failure is reported with a ``floor``-bearing ``binding_detail``. The
    guard is evaluated ALONGSIDE the canonical/provenance checks but is reported
    only when canonical + provenance already hold, so the headline binding
    constraint on a non-canonical run stays ``canonical_claim_run=False`` (the
    canonical gate is binding-priority #1; the guard never displaces it).
    """
    canonical = bool(run_metadata.get("canonical_claim_run", False))
    missing_prov = [f for f in _PROVENANCE_FIELDS if not run_metadata.get(f)]

    top_system = _top_composite_system(systems)
    core_breaches = _CORE_SYSTEM in breachers
    top_breaches = top_system is not None and top_system in breachers
    guard_ok = not (core_breaches or top_breaches)

    passed = canonical and not missing_prov and guard_ok
    if not canonical:
        detail = (
            "G7 FAIL: canonical_claim_run=False — the run is a provisional smoke "
            "(claim-grade emission requires a certified canonical run)"
        )
    elif missing_prov:
        detail = f"G7 FAIL: canonical run missing provenance stamp(s) {missing_prov}"
    elif not guard_ok:
        culprit = _CORE_SYSTEM if core_breaches else top_system
        detail = (
            f"G7 FAIL: RecallFloorVerdictGuard — {culprit!r} breaches the recall "
            f"floor yet holds/contests the top composite; a recall-floor-breaching "
            f"system can never top-rank (a floor-compliant claimant must be crowned)"
        )
    else:
        detail = (
            "G7 PASS: canonical_claim_run=True with full provenance stamp; "
            "RecallFloorVerdictGuard satisfied (top-ranked claimant is "
            "recall-floor-compliant)"
        )
    return GuaranteeResult("G7", passed, float(canonical), 1.0, detail)


def _top_composite_system(systems: dict[str, dict[str, Any]]) -> str | None:
    """Name of the system with the highest ``composite_score`` (``None`` if none).

    Ties broken by name for determinism. Only systems carrying a REAL composite
    (finite, in ``[0, 1]``, non-boolean — :func:`_finite_unit_score`) are considered.
    A non-finite / out-of-range composite is EXCLUDED rather than crowned or crashed
    on (the S7-02 close MAJOR-2): ``composite_score=+inf`` would fabricate a phantom
    rank-1 over a genuinely-superior finite system; ``NaN`` would crash the old
    ``s == best`` argmax (``NaN == best`` is always False ⇒ empty list ⇒ ``[0]``
    IndexError); a Python int wider than a C double (``10**400``) would raise
    OverflowError in ``float()`` — all three now fail CLOSED (the corrupt system is
    simply not eligible for the top-composite crown).
    """
    scored = [
        (name, score)
        for name, s in systems.items()
        if (score := _finite_unit_score(s.get("composite_score"))) is not None
    ]
    if not scored:
        return None
    best = max(s for _, s in scored)
    return sorted(name for name, s in scored if s == best)[0]


def _pending_guarantee(axis: str, override: bool | None) -> GuaranteeResult:
    """A successor guarantee (G2/G4/G5): three-valued, PENDING by default.

    A successor story may inject its computed ``True``/``False`` via
    ``pending_overrides``; absent that, the guarantee is PENDING (``None``) and
    surfaced as a tracked successor — never auto-passed.
    """
    successor = _PENDING_SUCCESSORS.get(axis, axis)
    if override is None:
        detail = f"{axis} PENDING: {successor}"
    else:
        detail = f"{axis} {'PASS' if override else 'FAIL'} (supplied by successor)"
    return GuaranteeResult(axis, override, float("nan"), float("nan"), detail)


# ---------------------------------------------------------------------------
# J — the SDO objective (bayes posterior, else MLE-bootstrap fallback)
# ---------------------------------------------------------------------------


def _compute_j(
    benchmark: dict[str, Any],
    systems: dict[str, dict[str, Any]],
    theta_samples: NDArray[np.floating[Any]] | None,
    posterior_names: list[str] | None,
    breachers: frozenset[str],
) -> tuple[float | None, str, str | None]:
    """Return ``(J, j_source, crowned_system)`` for pii-anon's rank-1 probability.

    RecallFloorVerdictGuard coupling (story §3 G7): a recall-floor breaching
    system can never be J-argmax. The rank-1 distribution is therefore computed
    over the floor-COMPLIANT columns only — a breacher's rank-1 mass is dropped
    (it can never be crowned), and if pii-anon ITSELF breaches, its J is forced to
    ``0.0`` (it cannot be the crowned rank-1 system). ``crowned_system`` is the
    argmax over the compliant set (``None`` only when J is unavailable).

    Prefers the supplied bayes posterior; otherwise the in-tree MLE-BT paired
    bootstrap. Returns ``(None, "unavailable", None)`` only when neither is
    possible (e.g. a single-system benchmark with no competitor pairs and no
    posterior).
    """
    if theta_samples is not None and posterior_names is not None:
        j_value, crowned = _guarded_rank1(theta_samples, posterior_names, breachers)
        return j_value, "bayes", crowned

    draws, names = _mle_bootstrap_draws(systems)
    if draws is None or _CORE_SYSTEM not in (names or []):
        return None, "unavailable", None
    j_value, crowned = _guarded_rank1(draws, names, breachers)
    return j_value, "mle-bootstrap", crowned


def _guarded_rank1(
    theta_samples: NDArray[np.floating[Any]],
    names: list[str],
    breachers: frozenset[str],
) -> tuple[float, str | None]:
    """Guarded rank-1: J(pii-anon) + crowned system over the COMPLIANT columns.

    Drops every recall-floor-breaching column before the argmax so a breacher can
    never be crowned (the RecallFloorVerdictGuard). If every system breaches (no
    compliant column survives), J is ``0.0`` and no system is crowned. If pii-anon
    is a breacher it is among the dropped columns, so its J is ``0.0``.
    """
    compliant = [n for n in names if n not in breachers]
    if not compliant:
        return 0.0, None
    cols = [names.index(n) for n in compliant]
    sub = theta_samples[:, cols]
    dist = rank_one_distribution(sub, compliant)
    # Crown the highest rank-1 mass; tie-break by LOWEST name for determinism,
    # consistent with _top_composite_system (negate the prob so min picks the max).
    crowned = min(compliant, key=lambda n: (-dist[n], n))
    j_core = dist.get(_CORE_SYSTEM, 0.0)
    return float(j_core), crowned


def _mle_bootstrap_draws(
    systems: dict[str, dict[str, Any]],
) -> tuple[NDArray[np.float64] | None, list[str]]:
    """Build MLE-BT bootstrap θ draws from composite-derived comparison records.

    The benchmark stores per-system composites, not raw paired outcomes, so we
    synthesise a NON-separable comparison multiset: for every unordered system
    pair we emit ``_J_PAIR_GAMES`` comparisons split by the logistic of the
    composite gap (``σ(γ·Δ)``). This is the same composite→outcome shape the
    ``bradley-terry-mle`` port path uses, but emitting BOTH directions keeps the
    design non-separable so the MM fit converges and the bootstrap draws are
    finite. The draws are stacked ``(b, n_systems)`` and fed to
    :func:`_guarded_rank1` (which restricts the rank-1 argmax to the
    recall-floor-compliant columns).

    Returns ``(None, [])`` when there are fewer than two systems (no pairs).
    """
    # Only systems carrying a REAL composite (finite, in [0, 1], non-boolean) seed
    # the bootstrap (the S7-02 close MAJOR-2): a +inf composite would saturate the
    # logistic gap into a fabricated phantom rank-1, a NaN would poison the MM fit,
    # and a 10**400 would raise OverflowError in float() — the corrupt system is
    # EXCLUDED (it can never be crowned by the J-meter), consistent with
    # `_top_composite_system`.
    valid_composites: dict[str, float] = {
        name: score
        for name, s in systems.items()
        if (score := _finite_unit_score(s.get("composite_score"))) is not None
    }
    names = sorted(valid_composites)
    if len(names) < 2:
        return None, []

    composites = valid_composites
    records: list[tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            gap = composites[a] - composites[b]
            s_ab = 1.0 / (1.0 + np.exp(-_clip(_J_PAIR_GAMMA * gap)))
            wins_a = int(round(s_ab * _J_PAIR_GAMES))
            wins_b = _J_PAIR_GAMES - wins_a
            records.extend([(a, b)] * wins_a)
            records.extend([(b, a)] * wins_b)

    engine = BradleyTerryMLEEngine()
    draws, draw_names = engine.paired_bootstrap_draws(
        records, _J_BOOTSTRAP_B, seed=_J_BOOTSTRAP_SEED
    )
    return draws, draw_names


def _clip(x: float) -> float:
    """Clamp a logistic argument to a numerically safe range (matches elo/BT)."""
    return max(-20.0, min(20.0, x))


# ---------------------------------------------------------------------------
# Verdict machine + binding-constraint reporter (priority-ordered)
# ---------------------------------------------------------------------------


def _decide(
    *,
    canonical: bool,
    guarantees: tuple[GuaranteeResult, ...],
    j_value: float | None,
    blocking_tier_c: frozenset[str],
    blocking_tier_r: frozenset[str],
) -> tuple[Verdict, str]:
    """The §5 verdict state machine + the single binding constraint.

    Binding priority (the program's next-thing-to-fix): canonical-run →
    failed G (lowest k) → J gap → unrun tiers (Tier-C, then unrun Tier-R).
    ``binding_constraint`` is ``""`` IFF the verdict is ``CLAIM_GRADE_SOTA``.

    §5 completion predicate: ``CLAIM_GRADE`` ⟺ canonical ∧ (every in-scope Gk
    ``True``) ∧ J ≥ bar ∧ (Tier-R ∪ Tier-C ALL RUN-or-WAIVED). So an unrun-
    unwaived Tier-R adapter (today ``gliner2``) blocks ``CLAIM_GRADE`` exactly as
    an unrun Tier-C API does → at most ``PROVISIONAL``.

    Three-valued discipline: ``CLAIM_GRADE`` requires EVERY guarantee ``True``;
    a PENDING (``None``) guarantee never counts toward a ``PROVISIONAL`` blocker
    but always blocks ``CLAIM_GRADE``.
    """
    by_axis = {g.axis: g for g in guarantees}

    # -- 1. canonical-run is the #1 gate. -----------------------------------
    if not canonical:
        return Verdict.NOT_YET, "canonical_claim_run=False (G7 certified-run gate)"

    # -- 2. any FAILED guarantee (lowest k) — a hard NOT_YET. ----------------
    for axis in _GUARANTEE_ORDER:
        g = by_axis.get(axis)
        if g is not None and g.passed is False:
            return Verdict.NOT_YET, g.binding_detail

    # -- 3. J gap (J must be computable and ≥ bar for any claim). ------------
    if j_value is None:
        # No failed G, but J cannot be computed → cannot be claim-grade; report
        # it as the binding constraint (ahead of unrun tiers).
        return Verdict.NOT_YET, "J unavailable (no posterior and no competitor pairs to fit)"
    if j_value < J_BAR:
        return (
            Verdict.NOT_YET,
            f"J={j_value:.4g} < J_BAR={J_BAR} (rank-1 probability below the bar)",
        )

    # -- 4. PENDING guarantees block CLAIM_GRADE (but not PROVISIONAL). ------
    pending = [
        by_axis[axis]
        for axis in _GUARANTEE_ORDER
        if by_axis.get(axis) is not None and by_axis[axis].passed is None
    ]

    # -- 5. unrun Tier-C OR unrun Tier-R blocks CLAIM_GRADE (§5: Tier-R ∪ Tier-C).
    tier_blocked = bool(blocking_tier_c) or bool(blocking_tier_r)

    if not pending and not tier_blocked:
        # Everything in scope is True, J ≥ bar, Tiers satisfied → CLAIM_GRADE.
        return Verdict.CLAIM_GRADE_SOTA, ""

    # Otherwise PROVISIONAL — report the binding constraint by priority:
    # a PENDING guarantee outranks the unrun tiers (it is closer to "real work");
    # within the tiers, unrun Tier-C (the cited cloud-API honesty boundary) leads,
    # and any unrun Tier-R (gliner2) is surfaced alongside it.
    if pending:
        binding = "pending guarantee(s): " + ", ".join(g.binding_detail for g in pending)
    else:
        parts: list[str] = []
        if blocking_tier_c:
            parts.append("unrun Tier-C: " + ", ".join(sorted(blocking_tier_c)))
        if blocking_tier_r:
            parts.append("unrun Tier-R: " + ", ".join(sorted(blocking_tier_r)))
        binding = "CLAIM_GRADE blocked — " + "; ".join(parts)
    return Verdict.PROVISIONAL_SOTA, binding
