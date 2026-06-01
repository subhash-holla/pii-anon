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
* **G4** Calibration / selective-risk — PENDING ← S4-03 reporter.
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
* ``PROVISIONAL_SOTA`` ⟺ same but blocked ONLY by unrun Tier-C.
* ``NOT_YET`` ⟺ otherwise — report the binding constraint (priority:
  canonical-run → failed G (lowest k) → J gap → unrun Tier-C).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pii_anon.eval_framework.rating.bradley_terry import BradleyTerryMLEEngine
from pii_anon.eval_framework.rating.significance import rank_one_probability

from .competitor_tiers import (
    TIER_R_NAMES,
    TierEntry,
    apply_run_status,
    default_registry,
    run_status_from_benchmark,
    unrun_tier_c,
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
]

# -- SDO threshold literals: these ARE the completion contract (§3/§5). -------
J_BAR: float = 0.95
EPS_F2: float = 0.01
ENTITY_COVERAGE_MIN: float = 0.80
EPS_RECALL_PER_LANG: float = 0.005

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
        G (lowest k) → J gap → unrun Tier-C). ``""`` IFF ``CLAIM_GRADE_SOTA``.
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
            competitor pairs to fit.
        pending_overrides:
            Three-valued overrides for the successor guarantees (``"G2"``,
            ``"G4"``, ``"G5"``) — a successor story supplies its computed
            ``True``/``False``; the default leaves them PENDING (``None``).
        tier_c_waivers:
            ``{tier_c_name: reason}`` waivers (reason mandatory). Waived Tier-C
            entries stop blocking ``CLAIM_GRADE``.
        """
        run_metadata = benchmark.get("run_metadata", {}) or {}
        canonical = bool(run_metadata.get("canonical_claim_run", False))

        systems = _systems_by_name(benchmark)
        overrides = pending_overrides or {}

        guarantees = (
            _g1_recall_floor(benchmark, systems),
            _pending_guarantee("G2", overrides.get("G2")),
            _g3_recall_dominance(systems),
            _pending_guarantee("G4", overrides.get("G4")),
            _pending_guarantee("G5", overrides.get("G5")),
            _g6_raw_noninferiority(systems),
            _g7_certified_run(run_metadata),
        )

        # The Tier registry, with run-status derived from the benchmark + waivers.
        registry = apply_run_status(
            default_registry(), run_status_from_benchmark(benchmark)
        )
        for name, reason in (tier_c_waivers or {}).items():
            registry = waive(registry, name, reason)
        blocking_tier_c = unrun_tier_c(registry)

        # J — the SDO objective (bayes if posterior supplied, else MLE-bootstrap).
        j_value, j_source = _compute_j(
            benchmark, systems, theta_samples, posterior_names
        )

        verdict, binding = _decide(
            canonical=canonical,
            guarantees=guarantees,
            j_value=j_value,
            blocking_tier_c=blocking_tier_c,
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
            tier_registry=registry,
        )


# ---------------------------------------------------------------------------
# Benchmark accessors
# ---------------------------------------------------------------------------


def _systems_by_name(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the benchmark's top-level ``systems`` list by ``system`` name."""
    out: dict[str, dict[str, Any]] = {}
    for system in benchmark.get("systems", []) or []:
        name = system.get("system")
        if isinstance(name, str):
            out[name] = system
    return out


def _competitor_names(systems: dict[str, dict[str, Any]]) -> list[str]:
    """The benchmarked competitor systems (everything outside the pii-anon ladder)."""
    return sorted(name for name in systems if name not in _LADDER_SYSTEMS)


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
                e for e, r in (sys.get("per_entity_recall") or {}).items()
                if isinstance(r, (int, float))
            }
    shared_entities: set[str] = set()
    for name in _competitor_names(systems):
        per = systems[name].get("per_entity_recall") or {}
        shared_entities |= {
            e for e, r in per.items()
            if isinstance(r, (int, float)) and r > 0.0
        }
    missing = shared_entities - ensemble_entities
    superset_ok = not missing

    # Per-language ε (PENDING if the artifact is absent).
    per_lang = benchmark.get("per_language_recall_delta")
    if per_lang is None:
        return GuaranteeResult(
            "G1", None, float("nan"), EPS_RECALL_PER_LANG,
            "G1 PENDING: per_language_recall_delta artifact absent "
            "(structural superset "
            + ("holds" if superset_ok else f"BREACHED on {sorted(missing)}")
            + "); per-language ε never fabricated",
        )
    worst_lang_eps = max((abs(float(v)) for v in per_lang.values()), default=0.0)
    eps_ok = worst_lang_eps <= EPS_RECALL_PER_LANG

    passed = superset_ok and eps_ok
    if passed:
        detail = (
            f"G1 PASS: ensemble entities ⊇ shared; worst per-language "
            f"ε={worst_lang_eps:.4g} ≤ {EPS_RECALL_PER_LANG}"
        )
    elif not superset_ok:
        detail = f"G1 FAIL: ensemble misses competitor-detected entities {sorted(missing)}"
    else:
        detail = (
            f"G1 FAIL: per-language recall ε={worst_lang_eps:.4g} > "
            f"{EPS_RECALL_PER_LANG}"
        )
    return GuaranteeResult("G1", passed, worst_lang_eps, EPS_RECALL_PER_LANG, detail)


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
    swarm_recall = float(swarm["recall"])
    best_competitor = max(float(systems[c]["recall"]) for c in competitors)
    passed = swarm_recall >= best_competitor
    detail = (
        f"G3 {'PASS' if passed else 'FAIL'}: pii-anon-swarm recall "
        f"{swarm_recall:.4g} {'≥' if passed else '<'} best competitor "
        f"{best_competitor:.4g}"
    )
    return GuaranteeResult("G3", passed, swarm_recall, best_competitor, detail)


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
    core_f2 = f_beta(float(core["precision"]), float(core["recall"]), beta=2.0)

    tier_r_f2 = [
        f_beta(float(systems[name]["precision"]), float(systems[name]["recall"]), beta=2.0)
        for name in systems
        if name in TIER_R_NAMES
    ]
    best_tier_r = max(tier_r_f2, default=0.0)
    f2_ok = core_f2 >= best_tier_r - EPS_F2

    # Entity coverage: fraction of entity types the core actually detects.
    per = core.get("per_entity_recall") or {}
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


def _g7_certified_run(run_metadata: dict[str, Any]) -> GuaranteeResult:
    """G7 — canonical_claim_run True ∧ full provenance stamp present."""
    canonical = bool(run_metadata.get("canonical_claim_run", False))
    missing_prov = [f for f in _PROVENANCE_FIELDS if not run_metadata.get(f)]
    passed = canonical and not missing_prov
    if not canonical:
        detail = (
            "G7 FAIL: canonical_claim_run=False — the run is a provisional smoke "
            "(claim-grade emission requires a certified canonical run)"
        )
    elif missing_prov:
        detail = f"G7 FAIL: canonical run missing provenance stamp(s) {missing_prov}"
    else:
        detail = "G7 PASS: canonical_claim_run=True with full provenance stamp"
    return GuaranteeResult("G7", passed, float(canonical), 1.0, detail)


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
) -> tuple[float | None, str]:
    """Return ``(J, j_source)`` for pii-anon's rank-1 probability.

    Prefers the supplied bayes posterior; otherwise falls back to the in-tree
    MLE-BT paired bootstrap over composite-derived comparison records. Returns
    ``(None, "unavailable")`` only when neither is possible (e.g. a
    single-system benchmark with no competitor pairs and no posterior).
    """
    if theta_samples is not None and posterior_names is not None:
        return (
            float(rank_one_probability(theta_samples, posterior_names, _CORE_SYSTEM)),
            "bayes",
        )

    draws, names = _mle_bootstrap_draws(systems)
    if draws is None or _CORE_SYSTEM not in (names or []):
        return None, "unavailable"
    return float(rank_one_probability(draws, names, _CORE_SYSTEM)), "mle-bootstrap"


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
    finite. The draws are stacked ``(b, n_systems)`` for
    :func:`significance.rank_one_probability`.

    Returns ``(None, [])`` when there are fewer than two systems (no pairs).
    """
    names = sorted(
        name for name, s in systems.items()
        if isinstance(s.get("composite_score"), (int, float))
    )
    if len(names) < 2:
        return None, []

    composites = {name: float(systems[name]["composite_score"]) for name in names}
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
) -> tuple[Verdict, str]:
    """The §5 verdict state machine + the single binding constraint.

    Binding priority (the program's next-thing-to-fix): canonical-run →
    failed G (lowest k) → J gap → unrun Tier-C. ``binding_constraint`` is ``""``
    IFF the verdict is ``CLAIM_GRADE_SOTA``.

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
        # it as the binding constraint (ahead of unrun Tier-C).
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

    # -- 5. unrun Tier-C blocks CLAIM_GRADE. ---------------------------------
    tier_c_blocked = bool(blocking_tier_c)

    if not pending and not tier_c_blocked:
        # Everything in scope is True, J ≥ bar, Tiers satisfied → CLAIM_GRADE.
        return Verdict.CLAIM_GRADE_SOTA, ""

    # Otherwise PROVISIONAL — report the binding constraint by priority:
    # a PENDING guarantee outranks unrun Tier-C (it is closer to "real work").
    if pending:
        binding = "pending guarantee(s): " + ", ".join(g.binding_detail for g in pending)
    else:
        binding = "unrun Tier-C (CLAIM_GRADE blocked): " + ", ".join(sorted(blocking_tier_c))
    return Verdict.PROVISIONAL_SOTA, binding
