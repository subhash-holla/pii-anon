"""Utility / fairness / compliance assessors (spec §6.4).

The user requests one ``utility_fairness_compliance`` dimension; the runner expands it
into three honestly-separated axes, each with its own claim strength:

* **utility** — trigram-Jaccard preservation of the NON-PII skeleton (original minus the
  referenced PII spans) vs the transformed text (reuses ``SemanticPreservationMetric``).
  Higher is better. Requires a ``transform``; ``MEASURED`` in labeled mode if powered.
* **fairness** — the worst-group per-group detection-recall gap over powered strata
  (reuses the certified ``fairness_gate``). Lower is better. Requires group/language
  metadata + ≥2 powered groups + gold recall; ``ADVISORY`` (a representative probe, no CI).
* **compliance** — a CAPABILITY/coverage axis: which standard-required PII types a
  system's output vocabulary covers (reuses ``ComplianceValidator``). ``ADVISORY``,
  vocabulary-dependent, never a head-to-head winner (the §6.4 / G2 phantom-0 rule).

Each self-marks ``NOT_ASSESSABLE`` (never a phantom 0) when its input is absent.
"""

from __future__ import annotations

from dataclasses import replace

from ..adjudication import Substrate
from ..claim_strength import (
    DimensionResult,
    MeasurementMode,
    PowerThresholds,
    advisory,
    not_assessable,
    resolve,
)
from ..stats import mean_bootstrap_ci

UTILITY = "utility"
FAIRNESS = "fairness"
COMPLIANCE = "compliance"

# fail-closed: if fewer than this fraction of records yield a scoreable (non-degenerate)
# transform output, the whole dimension is NOT_ASSESSABLE (mirrors the leakage assessor).
_MIN_SCOREABLE_FRACTION = 0.5


def _non_pii_skeleton(text: str, spans: tuple[tuple[str, int, int], ...]) -> str:
    """The original text minus its referenced PII spans (what SemanticPreservationMetric compares
    the transform output against). Stripping highest-start-first keeps the surviving offsets valid."""
    out = text
    for span in sorted(spans, key=lambda sp: sp[1], reverse=True):
        out = out[: span[1]] + out[span[2] :]
    return out

_UTILITY_CAVEAT = (
    "utility = trigram-Jaccard preservation of the NON-PII skeleton (the original text minus "
    "the referenced PII spans) vs the transformed text; higher is better"
)
_FAIRNESS_CAVEAT = (
    "fairness = the max-minus-min per-group detection-RECALL gap over powered strata; lower is "
    "better. A representative probe over the dataset's own groups, not a certified fairness audit"
)
_COMPLIANCE_CAVEAT = (
    "coverage depends on the system's entity-TYPE vocabulary matching each standard's canonical "
    "type names; a system using different type names will under-report coverage. This is a "
    "capability axis, not a data-dependent accuracy and not a head-to-head winner"
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def assess_utility(
    substrate: Substrate,
    *,
    thresholds: PowerThresholds,
    seed: int,
    n_resamples: int = 500,
) -> dict[str, DimensionResult]:
    from pii_anon.eval_framework.metrics.base import LabeledSpan
    from pii_anon.eval_framework.metrics.utility_metrics import SemanticPreservationMetric

    metric = SemanticPreservationMetric()
    results: dict[str, DimensionResult] = {}
    total_records = len(substrate.records)
    for sys in substrate.systems:
        values: list[float] = []
        unscoreable = 0
        for rec in substrate.records:
            out = rec.transform_by_system.get(sys)
            if out is None:
                unscoreable += 1
                continue
            stripped = out.strip()
            input_len = len(rec.original_text.strip())
            # A degenerate / content-DESTROYING output cannot be fairly scored for utility: an
            # empty output trivially "preserves" nothing, yet the preservation metric awards a
            # perfect 1.0 on falsy input — a phantom-perfect that would OUT-RANK a genuinely
            # preserving transform (the unsafe direction, overstating the pipeline). Mirror the
            # certified leakage degeneracy guard: such output is UNSCOREABLE, never 1.0 or 0.0.
            if len(stripped) < 3 or (input_len >= 10 and len(stripped) < 0.3 * input_len):
                unscoreable += 1
                continue
            # A wholly-PII record has an empty/near-empty NON-PII skeleton, on which the
            # preservation metric returns a phantom-perfect 1.0 REGARDLESS of the output (so a
            # zero-anonymization echo would score 1.0). There is no non-PII content to preserve ⇒
            # the record is UNSCOREABLE for utility (sibling of the empty-output guard above).
            if len(_non_pii_skeleton(rec.original_text, rec.reference).strip()) < 3:
                unscoreable += 1
                continue
            labels = [LabeledSpan(t, s, e, rec.record_id) for (t, s, e) in rec.reference]
            res = metric.compute(
                [], labels, context={"original_text": rec.original_text, "anonymized_text": out}
            )
            values.append(float(res.value))
        scoreable = len(values)
        if scoreable == 0:
            results[sys] = not_assessable(
                UTILITY, sys,
                "no scoreable transform output (no transform, or all outputs degenerate/"
                "content-destroying — an empty output is unscoreable, never a phantom-perfect 1.0)",
            )
            continue
        if scoreable < _MIN_SCOREABLE_FRACTION * total_records:
            results[sys] = not_assessable(
                UTILITY, sys,
                f"only {scoreable}/{total_records} records had scoreable (non-degenerate) "
                "transform output (fail-closed)",
            )
            continue
        point, ci = mean_bootstrap_ci(values, seed=seed, n_resamples=n_resamples)
        res2 = resolve(
            dimension=UTILITY, system=sys, mode=substrate.mode, value=point,
            support=scoreable, n_clusters=scoreable, confidence_interval=ci, significant=None,
            thresholds=thresholds, caveats=(_UTILITY_CAVEAT,),
            silver_reasons=("non-PII skeleton built from silver spans (advisory)",)
            if substrate.mode is MeasurementMode.SILVER else (),
        )
        results[sys] = replace(res2, metadata={
            **res2.metadata, "direction": "higher_is_better",
            "scoreable_records": scoreable, "unscoreable_records": unscoreable,
            "metric": "semantic_preservation_trigram_jaccard",
        })
    return results


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------
def assess_fairness(
    substrate: Substrate,
    *,
    min_support: int,
    gap_threshold: float = 0.10,
) -> dict[str, DimensionResult]:
    from pii_anon.eval_framework.metrics.base import LabeledSpan, MatchMode
    from pii_anon.eval_framework.metrics.fairness_gate import (
        LanguageGroupSlice,
        evaluate_language_fairness,
    )

    results: dict[str, DimensionResult] = {}
    if substrate.mode is MeasurementMode.SILVER:
        for sys in substrate.systems:
            results[sys] = not_assessable(
                FAIRNESS, sys,
                "fairness uses gold detection recall; the silver reference is detector-derived "
                "(circular) — not assessable",
            )
        return results

    for sys in substrate.systems:
        groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        total_pred = 0
        for rec in substrate.records:
            key = rec.group or rec.language
            if not key:
                continue
            gold = [LabeledSpan(t, s, e, rec.record_id) for (t, s, e) in rec.reference]
            pred = [LabeledSpan(t, s, e, rec.record_id) for (t, s, e) in rec.pred_by_system.get(sys, ())]
            total_pred += len(pred)
            g, p = groups.setdefault(key, ([], []))
            g.extend(gold)
            p.extend(pred)
        if len(groups) < 2:
            results[sys] = not_assessable(
                FAIRNESS, sys,
                "fewer than 2 records carry group/language metadata; fairness strata not assessable",
            )
            continue
        if total_pred == 0:
            results[sys] = not_assessable(
                FAIRNESS, sys,
                "system predicted no spans across the corpus; detection-fairness not assessable",
            )
            continue
        slices = [
            LanguageGroupSlice(language=key, gold=gold, predicted=pred)
            for key, (gold, pred) in groups.items()
        ]
        report = evaluate_language_fairness(
            slices, gap_threshold=gap_threshold, power_floor=min_support, match_mode=MatchMode.STRICT
        )
        if report.verdict == "INSUFFICIENT_POWER" or report.worst_group_recall_gap is None:
            results[sys] = not_assessable(
                FAIRNESS, sys,
                f"fewer than 2 groups reach the power floor of {min_support} gold positives "
                "(fail-closed: no fairness verdict without evidence)",
            )
            continue
        powered_recalls = [report.per_language_recall.get(g, 0.0) for g in report.powered_groups]
        if not any(r > 0.0 for r in powered_recalls):
            # A 0 gap because the system matches NO gold in any powered group is "equally
            # UNDETECTED everywhere", not "fair" — a phantom-0 PASS that would advertise a
            # do-nothing detector as perfectly fair. Fail closed.
            results[sys] = not_assessable(
                FAIRNESS, sys,
                "system matches no gold in any powered group; detection-fairness is vacuous "
                "(a 0 gap here is 'equally undetected', not 'fair')",
            )
            continue
        results[sys] = advisory(
            FAIRNESS, sys, report.worst_group_recall_gap,
            reasons=(
                f"worst-group per-group recall gap over {report.n_powered} powered group(s); "
                f"verdict {report.verdict} at gap threshold {report.gap_threshold}",
            ),
            caveats=(_FAIRNESS_CAVEAT,),
            support=report.n_powered,
            metadata={
                "verdict": report.verdict,
                "per_group_recall": dict(report.per_language_recall),
                "powered_groups": list(report.powered_groups),
                "unpowered_groups": list(report.unpowered_groups),
                "violating_groups": list(report.violating_groups),
                "gap_threshold": report.gap_threshold,
                "direction": "lower_is_better",
            },
        )
    return results


# ---------------------------------------------------------------------------
# Compliance (capability/coverage axis — never a head-to-head winner)
# ---------------------------------------------------------------------------
def assess_compliance(substrate: Substrate) -> dict[str, DimensionResult]:
    from pii_anon.eval_framework.standards.compliance import ComplianceValidator

    validator = ComplianceValidator()
    results: dict[str, DimensionResult] = {}
    for sys in substrate.systems:
        detected: set[str] = set()
        for rec in substrate.records:
            for (t, _s, _e) in rec.pred_by_system.get(sys, ()):
                if isinstance(t, str) and t.strip():
                    detected.add(t)
        if not detected:
            results[sys] = not_assessable(
                COMPLIANCE, sys,
                "system predicted no entity types; standard-coverage not assessable",
            )
            continue
        multi = validator.validate_all(sorted(detected))
        per_standard = {r.standard: float(r.coverage_ratio) for r in multi.reports}
        results[sys] = advisory(
            COMPLIANCE, sys, float(multi.overall_coverage_ratio),
            reasons=(
                "compliance is a CAPABILITY/coverage axis (which standard-required PII types the "
                "system's output vocabulary covers), not a data-dependent accuracy or a winner",
            ),
            caveats=(_COMPLIANCE_CAVEAT,),
            support=len(detected),
            metadata={
                "overall_coverage_ratio": float(multi.overall_coverage_ratio),
                "per_standard_coverage": per_standard,
                "detected_type_count": len(detected),
                "direction": "higher_is_better",
            },
        )
    return results
