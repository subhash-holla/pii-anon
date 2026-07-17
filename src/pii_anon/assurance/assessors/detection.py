"""Detection assessor — per-entity recall/precision/F1 vs the reference.

The one dimension that genuinely supports a publishable head-to-head claim (in
labeled mode, subject to the power gate). Cluster bootstrap for the CI; paired
bootstrap vs the baseline (pii-anon) for significance; per-entity-type breakdown.
"""

from __future__ import annotations

from collections import defaultdict

from ..adjudication import Substrate
from ..claim_strength import (
    DimensionResult,
    MeasurementMode,
    PowerThresholds,
    demote_to_advisory,
    resolve,
)
from ..stats import (
    Counts,
    PairedResult,
    cluster_bootstrap_ci,
    paired_bootstrap,
    pooled_f1,
    pooled_fbeta,
    pooled_precision,
    pooled_recall,
)
from ..spans import canonical_type
from .base import match_counts, per_type_counts

DIMENSION = "detection"

# Boundary-F1 minus strict-F1 above this gap signals a type-vocabulary mismatch with gold
# (correct spans, non-reconcilable type labels) rather than genuine detection quality.
_TYPE_MISMATCH_GAP = 0.2
# Cap on per-entity-type breakdown rows (top-N by support) — bounds artifact size.
_MAX_BREAKDOWN_TYPES = 200


def assess_detection(
    substrate: Substrate,
    *,
    baseline_system: str,
    thresholds: PowerThresholds,
    seed: int,
    alpha: float = 0.05,
    n_resamples: int = 500,
) -> tuple[dict[str, DimensionResult], dict[str, PairedResult]]:
    """Return (per-system DimensionResult, per-system paired comparison vs baseline)."""
    per_record: dict[str, list[Counts]] = {s: [] for s in substrate.systems}
    per_record_boundary: dict[str, list[Counts]] = {s: [] for s in substrate.systems}
    per_type: dict[str, dict[str, Counts]] = {s: defaultdict(Counts) for s in substrate.systems}
    for rec in substrate.records:
        for sys in substrate.systems:
            pred = rec.pred_by_system.get(sys, ())
            per_record[sys].append(match_counts(pred, rec.reference))
            per_record_boundary[sys].append(match_counts(pred, rec.reference, type_sensitive=False))
            for t, c in per_type_counts(pred, rec.reference).items():
                per_type[sys][t] = per_type[sys][t] + c

    comparisons: dict[str, PairedResult] = {}
    for sys in substrate.systems:
        if sys != baseline_system:
            comparisons[sys] = paired_bootstrap(
                per_record[sys], per_record[baseline_system], pooled_f1,
                metric_name="f1", seed=seed, n_resamples=n_resamples, alpha=alpha,
            )

    results: dict[str, DimensionResult] = {}
    for sys in substrate.systems:
        counts = per_record[sys]
        pooled = sum(counts, Counts())
        f1, support = pooled_f1(counts), pooled.tp + pooled.fn
        point, ci = cluster_bootstrap_ci(counts, pooled_f1, seed=seed, n_resamples=n_resamples)
        significant: bool | None = None
        p_value = None
        effect = None
        if sys in comparisons:
            significant = comparisons[sys].p_value < alpha
            p_value = comparisons[sys].p_value
            effect = comparisons[sys].effect_size
        # canonical_type buckets any non-allowlisted (possibly PII-bearing) type label so a
        # raw type string never reaches the serialized breakdown; cap the row count (top-N by
        # support) so a high-type-cardinality pipeline can't blow the artifact to multi-MB.
        ranked = sorted(per_type[sys].items(), key=lambda kv: (-(kv[1].tp + kv[1].fn), kv[0]))
        breakdown = {
            canonical_type(t): dict(zip(("precision", "recall", "f1"), _prf1(c)), **{"support": c.tp + c.fn})
            for t, c in ranked[:_MAX_BREAKDOWN_TYPES]
        }
        res = resolve(
            dimension=DIMENSION, system=sys, mode=substrate.mode, value=f1,
            support=support, n_clusters=len(counts), confidence_interval=ci, significant=significant,
            thresholds=thresholds,
            silver_reasons=("silver reference: advisory only, no publishable winner",)
            if substrate.mode is MeasurementMode.SILVER else (),
        )
        res.metadata.update({
            "precision": pooled_precision(counts),
            "recall": pooled_recall(counts),
            "f2": pooled_fbeta(counts, 2.0),
            "match_mode": "strict",
            "per_entity_type": breakdown,
            "per_entity_type_total": len(per_type[sys]),  # >len(breakdown) => capped top-N by support
            "reference_kind": substrate.reference_kind,
        })
        if p_value is not None:
            res.metadata["p_value_vs_baseline"] = p_value
            res.metadata["effect_size_vs_baseline"] = effect
        # Type-schema mismatch safeguard: if a system detects correct SPANS (high boundary-
        # only F1) but few of its TYPE labels match the gold vocabulary (low strict F1), the
        # strict F1 is a labelling artifact, not detection quality — demote to ADVISORY so it
        # is never a publishable MEASURED win (cascades to the head-to-head via weaker-of-both).
        boundary_f1 = pooled_f1(per_record_boundary[sys])
        res.metadata["boundary_f1"] = boundary_f1
        if boundary_f1 - f1 > _TYPE_MISMATCH_GAP:
            res = demote_to_advisory(
                res,
                "type-schema mismatch: detects correct spans under different type labels than "
                "the gold vocabulary — strict type-F1 is a labelling artifact, not a publishable score",
            )
            res.metadata["type_schema_mismatch"] = True
        results[sys] = res
    return results, comparisons


def _prf1(c: Counts) -> tuple[float, float, float]:
    from ..stats import prf1_from_counts
    return prf1_from_counts(c.tp, c.fp, c.fn)
