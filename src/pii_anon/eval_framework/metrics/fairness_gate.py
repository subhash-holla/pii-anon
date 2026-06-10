"""Powered worst-group multilingual fairness gate (S7-03, DC-15).

Implements **FR-039** (per-language fairness gap bounded + gated) with the
**NFR-025** semantics: the worst-group recall gap across **POWERED** language
groups must stay within the threshold (default 0.10), where "powered" is
defined by the **NFR-004** statistical-power floor (gold positives per slice;
the long-tail tier of 200 is the default — test-scale floors are injectable).

The gate is a pure, deterministic primitive (AX-002):

* Per-language recall uses the SAME greedy span-alignment primitive
  (:func:`~pii_anon.eval_framework.metrics.span_metrics._aligned_prf`,
  STRICT mode by default) that the S1-03 per-language recall-floor gate and
  :mod:`~pii_anon.eval_framework.metrics.fairness_metrics` use — ONE recall
  definition program-wide (consistency-pinned by the story's A10 contract
  test).
* **Fail-closed:** zero OR one powered group can never PASS — the verdict is
  ``INSUFFICIENT_POWER`` (a gap needs at least two powered groups, and an
  unpowered cohort is not evidence of fairness). Corrupt parameters are
  refused with domain-named :class:`ValueError` (the S5-02/03 fail-loud
  discipline).
* Unpowered groups are reported observationally (``unpowered_groups`` +
  their recalls in ``per_language_recall``) — visible, never gate-driving,
  never silently dropped.

# SWITCH-POINT(CANONICAL): emitting this verdict from
# ``evaluation/canonical_run.py`` would make it a control-path-artifact
# field and MANDATE the adversarial SDO close — explicitly deferred Pass-2.
# SWITCH-POINT(DATA): the corpus-scale fairness number over the 60-language
# eval set at the full NFR-004 power tiers is eval-data Pass-2; the gate
# itself is real now.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .base import LabeledSpan, MatchMode
from .span_metrics import _aligned_prf

__all__ = [
    "FairnessGateReport",
    "LanguageGroupSlice",
    "evaluate_language_fairness",
]


@dataclass(frozen=True)
class LanguageGroupSlice:
    """One language group's labelled evaluation slice.

    ``gold`` are the annotated PII spans for the group; ``predicted`` are
    the system's spans on the same records. Recall is computed by greedy
    STRICT alignment of ``predicted`` against ``gold``.
    """

    language: str
    gold: Sequence[LabeledSpan]
    predicted: Sequence[LabeledSpan]


@dataclass(frozen=True)
class FairnessGateReport:
    """The gate's verdict + full per-group evidence (FR-039/NFR-025)."""

    verdict: str  # "PASS" | "FAIL" | "INSUFFICIENT_POWER"
    worst_group_recall_gap: float | None
    gap_threshold: float
    power_floor: int
    per_language_recall: dict[str, float] = field(default_factory=dict)
    powered_groups: list[str] = field(default_factory=list)
    unpowered_groups: list[str] = field(default_factory=list)
    violating_groups: list[str] = field(default_factory=list)
    n_powered: int = 0


def evaluate_language_fairness(
    slices: Sequence[LanguageGroupSlice],
    *,
    gap_threshold: float = 0.10,
    power_floor: int = 200,
    match_mode: MatchMode = MatchMode.STRICT,
) -> FairnessGateReport:
    """Gate the worst-group per-language recall gap over POWERED groups.

    Parameters
    ----------
    slices:
        One :class:`LanguageGroupSlice` per language (duplicates refused).
    gap_threshold:
        NFR-025 bound on ``max(recall) - min(recall)`` over powered groups;
        the PASS comparison is inclusive (``gap <= threshold`` passes).
    power_floor:
        NFR-004 power filter: a group is POWERED iff it carries at least
        this many gold positives (default = the 200 long-tail tier).
    match_mode:
        Span-alignment mode for recall (STRICT by default — the program-wide
        recall definition).

    Returns
    -------
    FairnessGateReport
        Fail-closed verdict: ``PASS`` / ``FAIL`` only when >= 2 powered
        groups exist; otherwise ``INSUFFICIENT_POWER`` (never PASS without
        evidence).
    """
    if not slices:
        raise ValueError(
            "evaluate_language_fairness requires at least one language slice"
        )
    if not 0.0 <= gap_threshold <= 1.0:
        raise ValueError(
            f"gap_threshold must be within [0, 1]; got {gap_threshold!r}"
        )
    if power_floor < 1:
        raise ValueError(
            f"power_floor must be a positive gold-positive count; "
            f"got {power_floor!r}"
        )
    seen: set[str] = set()
    for group in slices:
        if group.language in seen:
            raise ValueError(
                f"duplicate language slice {group.language!r}; pass one "
                "slice per language"
            )
        seen.add(group.language)

    per_language_recall: dict[str, float] = {}
    powered: list[str] = []
    unpowered: list[str] = []
    for group in slices:
        _, recall, *_ = _aligned_prf(
            list(group.predicted), list(group.gold), match_mode
        )
        per_language_recall[group.language] = recall
        if len(group.gold) >= power_floor:
            powered.append(group.language)
        else:
            unpowered.append(group.language)
    powered.sort()
    unpowered.sort()

    if len(powered) < 2:
        # Fail-closed: a gap needs >= 2 powered groups; an unpowered cohort
        # is never evidence of fairness.
        return FairnessGateReport(
            verdict="INSUFFICIENT_POWER",
            worst_group_recall_gap=None,
            gap_threshold=gap_threshold,
            power_floor=power_floor,
            per_language_recall=per_language_recall,
            powered_groups=powered,
            unpowered_groups=unpowered,
            violating_groups=[],
            n_powered=len(powered),
        )

    powered_recalls = [per_language_recall[lang] for lang in powered]
    best = max(powered_recalls)
    gap = best - min(powered_recalls)
    violating = sorted(
        lang
        for lang in powered
        if (best - per_language_recall[lang]) > gap_threshold
    )
    return FairnessGateReport(
        verdict="PASS" if gap <= gap_threshold else "FAIL",
        worst_group_recall_gap=gap,
        gap_threshold=gap_threshold,
        power_floor=power_floor,
        per_language_recall=per_language_recall,
        powered_groups=powered,
        unpowered_groups=unpowered,
        violating_groups=violating,
        n_powered=len(powered),
    )
