"""Tiered, honesty-gated competitor registry for the SDO gate (S4-CS-01).

PURE STDLIB. This module imports **nothing** beyond the standard library (frozen
dataclasses + enums), so it loads in any environment and is import-isolated from
the detection / orchestration layers (S3-05 / story §7 boundary). It carries the
Tier-R / Tier-C competitor metadata the :class:`CompetitiveSupremacyGate`
consumes as its honesty boundary.

Why this registry is SEPARATE from ``evaluation/competitor_compare.py`` (RISK-6)
-------------------------------------------------------------------------------
``competitor_compare.py``'s ``_COMPETITOR_META.keys()`` IS the *run / expected*
competitor set — it drives ``detector_factories_keys`` (:2147) and the
``expected`` set (:3029, :3348). Adding ``gliner2`` or the Tier-C cloud APIs
there would pull *un-runnable* competitors into the run/expected set → a real
behaviour change and failed runs. So the Tier-R / Tier-C metadata lives ENTIRELY
here, and ``competitor_compare.py`` stays byte-identical. Run status is DERIVED
from the benchmark JSON's own ``available_competitors`` + per-system ``available``
flag (:func:`run_status_from_benchmark`) — never fabricated.

Tiers
-----
* **Tier-R (run now):** ``presidio``, ``scrubadub``, ``gliner``, ``gliner2``.
  The SDO PROVISIONAL verdict is computed vs Tier-R.
* **Tier-C (cited, must-run-before-claim):** ``openai-privacy-filter``,
  ``azure-ai-language``, ``aws-comprehend``. Real APIs/keys ⇒ Pass-2; UNRUN by
  default. Until each is RUN or WAIVED-with-a-documented-reason, ``CLAIM_GRADE``
  is blocked — the unrun set is the gate's visible honesty boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

__all__ = [
    "CompetitorTier",
    "RunStatus",
    "TierEntry",
    "TIER_R_NAMES",
    "TIER_C_NAMES",
    "default_registry",
    "waive",
    "apply_run_status",
    "unrun_tier_c",
    "unrun_tier_r",
    "run_status_from_benchmark",
]


class CompetitorTier(Enum):
    """Which evidence tier a competitor belongs to.

    ``R`` = run-now (local, runnable in-tree); ``C`` = cited cloud API
    (must-run-before-claim, Pass-2).
    """

    R = "R"
    C = "C"


class RunStatus(Enum):
    """Whether a competitor has actually been benchmarked.

    ``RUN`` — present in the benchmark with real measurements; ``UNRUN`` — not
    yet benchmarked (the default for Tier-C and for any adapter not in the run);
    ``WAIVED`` — explicitly excluded with a documented reason recorded in the
    gate (the only other way Tier-C stops blocking ``CLAIM_GRADE``).
    """

    RUN = "RUN"
    UNRUN = "UNRUN"
    WAIVED = "WAIVED"


@dataclass(frozen=True)
class TierEntry:
    """One competitor's tier metadata (frozen — values are swapped, not mutated).

    Attributes
    ----------
    name:
        Canonical competitor identifier (matches the benchmark ``system`` field
        for Tier-R systems that have been run).
    tier:
        :class:`CompetitorTier` R or C.
    source:
        Package import path (Tier-R) or API/service identifier (Tier-C).
    citation:
        A non-empty citation URL/reference so the gate can cite the competitor it
        is being measured against (audit requirement).
    run_status:
        :class:`RunStatus` — ``UNRUN`` by default for Tier-C and gliner2.
    waiver_reason:
        The documented reason recorded when an entry is ``WAIVED`` (``None``
        otherwise).
    """

    name: str
    tier: CompetitorTier
    source: str
    citation: str
    run_status: RunStatus = RunStatus.UNRUN
    waiver_reason: str | None = None


# -- Tier membership (the registry contract). --------------------------------
TIER_R_NAMES: frozenset[str] = frozenset(
    {"presidio", "scrubadub", "gliner", "gliner2"}
)
TIER_C_NAMES: frozenset[str] = frozenset(
    {"openai-privacy-filter", "azure-ai-language", "aws-comprehend"}
)

# Frozen metadata template. ``default_registry`` returns a fresh dict over these
# immutable entries, so callers can never mutate the canonical table.
_REGISTRY_TEMPLATE: tuple[TierEntry, ...] = (
    # Tier-R — runnable. The three native competitors are also driven by
    # competitor_compare._COMPETITOR_META; gliner2 is the new adapter (UNRUN).
    TierEntry(
        name="presidio",
        tier=CompetitorTier.R,
        source="presidio_analyzer",
        citation="https://github.com/microsoft/presidio",
    ),
    TierEntry(
        name="scrubadub",
        tier=CompetitorTier.R,
        source="scrubadub",
        citation="https://github.com/LeapBeyond/scrubadub",
    ),
    TierEntry(
        name="gliner",
        tier=CompetitorTier.R,
        source="gliner",
        citation="https://github.com/urchade/GLiNER",
    ),
    TierEntry(
        name="gliner2",
        tier=CompetitorTier.R,
        source="gliner2",
        citation="https://github.com/fastino-ai/GLiNER2",
    ),
    # Tier-C — cited cloud APIs; real keys ⇒ Pass-2; UNRUN by default.
    TierEntry(
        name="openai-privacy-filter",
        tier=CompetitorTier.C,
        source="api:openai",
        citation="https://platform.openai.com/docs/guides/moderation",
    ),
    TierEntry(
        name="azure-ai-language",
        tier=CompetitorTier.C,
        source="api:azure-ai-language",
        citation="https://learn.microsoft.com/azure/ai-services/language-service/personally-identifiable-information/overview",
    ),
    TierEntry(
        name="aws-comprehend",
        tier=CompetitorTier.C,
        source="api:aws-comprehend",
        citation="https://docs.aws.amazon.com/comprehend/latest/dg/how-pii.html",
    ),
)


def default_registry() -> dict[str, TierEntry]:
    """Return a FRESH registry mapping (name → frozen :class:`TierEntry`).

    Every call builds a new dict over the immutable template entries, so mutating
    the returned mapping (or any operation here) never affects the canonical
    table or another caller's copy.
    """
    return {entry.name: entry for entry in _REGISTRY_TEMPLATE}


def waive(
    registry: dict[str, TierEntry], name: str, reason: str
) -> dict[str, TierEntry]:
    """Return a NEW registry with ``name`` set to ``WAIVED`` and its reason recorded.

    A waiver is the only way (besides an actual run) for a Tier-C competitor to
    stop blocking ``CLAIM_GRADE`` — so the ``reason`` is MANDATORY: a blank or
    whitespace-only reason raises :class:`ValueError`. An unknown ``name`` raises
    :class:`KeyError` (a typo in a waiver must fail loud, never silently no-op).
    Pure: the input mapping is not mutated.
    """
    if name not in registry:
        raise KeyError(
            f"{name!r} is not in the competitor registry; cannot waive an "
            f"unknown competitor"
        )
    if not reason or not reason.strip():
        raise ValueError(
            f"a waiver for {name!r} must carry a non-empty documented reason "
            f"(it is recorded in the gate output for audit)"
        )
    out = dict(registry)
    out[name] = replace(
        registry[name], run_status=RunStatus.WAIVED, waiver_reason=reason
    )
    return out


def apply_run_status(
    registry: dict[str, TierEntry], statuses: dict[str, RunStatus]
) -> dict[str, TierEntry]:
    """Return a NEW registry with ``run_status`` overridden per ``statuses``.

    PURE — neither the input ``registry`` nor the module-level default is
    mutated. An entry not named in ``statuses`` keeps its prior status. A status
    for an unknown competitor raises :class:`KeyError` (fail loud).
    """
    unknown = set(statuses) - set(registry)
    if unknown:
        raise KeyError(
            f"run-status given for unknown competitor(s) {sorted(unknown)!r}"
        )
    out = dict(registry)
    for name, status in statuses.items():
        out[name] = replace(registry[name], run_status=status)
    return out


def unrun_tier_c(registry: dict[str, TierEntry]) -> frozenset[str]:
    """Return the set of Tier-C names still ``UNRUN`` (the CLAIM_GRADE blockers).

    A Tier-C entry that is ``RUN`` or ``WAIVED`` drops out — only genuinely
    unrun Tier-C competitors block a claim-grade verdict (the honesty boundary).
    """
    return frozenset(
        name
        for name, entry in registry.items()
        if entry.tier is CompetitorTier.C and entry.run_status is RunStatus.UNRUN
    )


def unrun_tier_r(registry: dict[str, TierEntry]) -> frozenset[str]:
    """Return the set of Tier-R names still ``UNRUN`` (also CLAIM_GRADE blockers).

    The completion predicate (§5) requires the WHOLE Tier-R ∪ Tier-C set to be
    RUN-or-WAIVED before ``CLAIM_GRADE`` — so an unrun-unwaived Tier-R adapter
    (today: ``gliner2``, which is not in the benchmark run) blocks a claim just as
    an unrun Tier-C API does. A ``RUN`` or ``WAIVED`` Tier-R entry drops out.
    """
    return frozenset(
        name
        for name, entry in registry.items()
        if entry.tier is CompetitorTier.R and entry.run_status is RunStatus.UNRUN
    )


def run_status_from_benchmark(benchmark: dict[str, Any]) -> dict[str, RunStatus]:
    """Derive ``RUN`` statuses from a benchmark JSON — NEVER fabricated (RISK-6).

    A competitor is ``RUN`` iff it appears in the benchmark's
    ``available_competitors`` AND its per-system record carries ``available ==
    True``. Systems absent from the run (gliner2, all Tier-C today) simply do not
    appear in the returned mapping, so :func:`apply_run_status` leaves them at
    their ``UNRUN`` default. Only names that exist in the default registry are
    reported (the benchmark may include pii-anon / pii-anon-swarm, which are not
    competitors).
    """
    known = TIER_R_NAMES | TIER_C_NAMES
    available_list = set(benchmark.get("available_competitors", []) or [])
    available_flag: dict[str, bool] = {}
    for system in benchmark.get("systems", []) or []:
        name = system.get("system")
        if isinstance(name, str):
            available_flag[name] = bool(system.get("available", False))

    statuses: dict[str, RunStatus] = {}
    for name in available_list:
        if name in known and available_flag.get(name, False):
            statuses[name] = RunStatus.RUN
    return statuses
