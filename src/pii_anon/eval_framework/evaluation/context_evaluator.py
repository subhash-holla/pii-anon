"""Document-level and long-context evaluation.

Evidence basis:
- RegLER (2021): regularization for long named entity recognition
- PII-Bench (2025): query-aware contextual PII evaluation
- pii-anon IdentityLedger: session-scoped entity clustering
"""

from __future__ import annotations

from typing import Any

from ..metrics.base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    safe_div,
)
from ..metrics.span_metrics import _aligned_prf


class DocumentContextEvaluator:
    """Evaluates PII detection quality across document-level context.

    Covers mention clustering (coreference), entity consistency across
    segments, and context-length-stratified performance.
    """

    def evaluate_mention_consistency(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        entity_clusters: dict[str, list[int]] | None = None,
        document_text: str = "",
    ) -> EvalMetricResult:
        """Check that mentions of the same entity receive consistent labels.

        Parameters
        ----------
        entity_clusters:
            Mapping from cluster ID to list of label indices belonging to
            the same real-world entity.
        """
        if not entity_clusters or not document_text:
            return EvalMetricResult(
                name="mention_consistency", value=1.0,
                level=EvaluationLevel.MENTION, match_mode=MatchMode.STRICT,
            )

        consistent_clusters = 0
        total_clusters = 0
        for _cluster_id, label_indices in entity_clusters.items():
            if len(label_indices) < 2:
                continue
            total_clusters += 1
            # Find predicted types for each label in the cluster
            pred_types_for_cluster: set[str] = set()
            for li in label_indices:
                if li >= len(labels):
                    continue
                label = labels[li]
                # Find matching prediction
                for pred in predictions:
                    if (
                        pred.start == label.start
                        and pred.end == label.end
                        and pred.entity_type == label.entity_type
                    ):
                        pred_types_for_cluster.add(pred.entity_type)
                        break
                else:
                    pred_types_for_cluster.add("__MISSED__")

            if len(pred_types_for_cluster) <= 1:
                consistent_clusters += 1

        score = safe_div(consistent_clusters, total_clusters) if total_clusters > 0 else 1.0
        return EvalMetricResult(
            name="mention_consistency", value=round(score, 6),
            level=EvaluationLevel.MENTION, match_mode=MatchMode.STRICT,
            metadata={
                "consistent_clusters": consistent_clusters,
                "total_clusters": total_clusters,
            },
        )

    def evaluate_cross_segment_recall(
        self,
        segment_predictions: list[list[LabeledSpan]],
        labels: list[LabeledSpan],
        *,
        match_mode: MatchMode = MatchMode.STRICT,
    ) -> EvalMetricResult:
        """Evaluate recall across document segments after boundary reconciliation.

        Flattens segment predictions and measures how many ground-truth
        labels are recovered.
        """
        all_predictions: list[LabeledSpan] = []
        for segment in segment_predictions:
            all_predictions.extend(segment)

        p, r, f1, tp, fp, fn = _aligned_prf(all_predictions, labels, match_mode)
        return EvalMetricResult(
            name="cross_segment_recall", value=round(r, 6),
            level=EvaluationLevel.DOCUMENT, match_mode=match_mode,
            precision=round(p, 6), recall=round(r, 6), f1=round(f1, 6),
            support=len(labels),
            metadata={"segments": len(segment_predictions), "total_predictions": len(all_predictions)},
        )

    def evaluate_by_context_length(
        self,
        records: list[dict[str, Any]],
        *,
        match_mode: MatchMode = MatchMode.STRICT,
    ) -> dict[str, EvalMetricResult]:
        """Evaluate performance stratified by context-length tier.

        Parameters
        ----------
        records:
            List of dicts with keys ``predictions`` (list[LabeledSpan]),
            ``labels`` (list[LabeledSpan]), and ``context_length_tier`` (str).
        """
        by_tier: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        for rec in records:
            tier = str(rec.get("context_length_tier", "unknown"))
            preds = rec.get("predictions", [])
            lbls = rec.get("labels", [])
            if tier not in by_tier:
                by_tier[tier] = ([], [])
            by_tier[tier][0].extend(preds)
            by_tier[tier][1].extend(lbls)

        results: dict[str, EvalMetricResult] = {}
        for tier, (preds, lbls) in sorted(by_tier.items()):
            p, r, f1, *_ = _aligned_prf(preds, lbls, match_mode)
            results[tier] = EvalMetricResult(
                name=f"context_length_{tier}", value=round(f1, 6),
                level=EvaluationLevel.DOCUMENT, match_mode=match_mode,
                precision=round(p, 6), recall=round(r, 6), f1=round(f1, 6),
                support=len(lbls),
            )
        return results

    def evaluate_boundary_reconciliation(
        self,
        pre_reconciliation: list[LabeledSpan],
        post_reconciliation: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        match_mode: MatchMode = MatchMode.STRICT,
    ) -> dict[str, EvalMetricResult]:
        """Compare F1 before and after boundary reconciliation."""
        _, _, f1_pre, *_ = _aligned_prf(pre_reconciliation, labels, match_mode)
        _, _, f1_post, *_ = _aligned_prf(post_reconciliation, labels, match_mode)
        return {
            "pre_reconciliation": EvalMetricResult(
                name="pre_reconciliation_f1", value=round(f1_pre, 6),
                level=EvaluationLevel.DOCUMENT, match_mode=match_mode,
            ),
            "post_reconciliation": EvalMetricResult(
                name="post_reconciliation_f1", value=round(f1_post, 6),
                level=EvaluationLevel.DOCUMENT, match_mode=match_mode,
            ),
            "improvement": EvalMetricResult(
                name="reconciliation_improvement", value=round(f1_post - f1_pre, 6),
                level=EvaluationLevel.DOCUMENT, match_mode=match_mode,
            ),
        }
