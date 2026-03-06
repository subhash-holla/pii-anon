"""Token-level, entity-level, and document-level span detection metrics.

Evidence basis:
- SemEval 2013 Task 9.1: strict / exact / partial / type matching
- nervaluate (Batista, 2018): entity-level NER evaluation
- seqeval (Nakayama, 2018): token-level BIO-tag F1
"""

from __future__ import annotations

from typing import Any

from .base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    MultiLevelMetric,
    compute_f1,
    compute_iou,
    safe_div,
)


# ---------------------------------------------------------------------------
# Matching predicates
# ---------------------------------------------------------------------------

def _strict_match(pred: LabeledSpan, label: LabeledSpan) -> bool:
    """Exact boundaries AND exact entity type."""
    return (
        pred.start == label.start
        and pred.end == label.end
        and pred.entity_type == label.entity_type
    )


def _exact_match(pred: LabeledSpan, label: LabeledSpan) -> bool:
    """Exact boundaries, type ignored."""
    return pred.start == label.start and pred.end == label.end


def _partial_match(
    pred: LabeledSpan, label: LabeledSpan, *, threshold: float = 0.5,
) -> bool:
    """IoU overlap >= *threshold*."""
    return compute_iou(pred.start, pred.end, label.start, label.end) >= threshold


def _type_match(pred: LabeledSpan, label: LabeledSpan) -> bool:
    """Entity type match, boundaries ignored (any overlap suffices)."""
    has_overlap = max(pred.start, label.start) < min(pred.end, label.end)
    return has_overlap and pred.entity_type == label.entity_type


def _dispatch_match(
    pred: LabeledSpan,
    label: LabeledSpan,
    mode: MatchMode,
    *,
    partial_threshold: float = 0.5,
) -> bool:
    if mode == MatchMode.STRICT:
        return _strict_match(pred, label)
    if mode == MatchMode.EXACT:
        return _exact_match(pred, label)
    if mode == MatchMode.PARTIAL:
        return _partial_match(pred, label, threshold=partial_threshold)
    if mode == MatchMode.TYPE:
        return _type_match(pred, label)
    return False


# ---------------------------------------------------------------------------
# Core P / R / F1 calculation with greedy alignment
# ---------------------------------------------------------------------------

def _aligned_prf(
    predictions: list[LabeledSpan],
    labels: list[LabeledSpan],
    mode: MatchMode,
    *,
    partial_threshold: float = 0.5,
) -> tuple[float, float, float, int, int, int]:
    """Greedy alignment giving (precision, recall, f1, tp, fp, fn).

    Each label is matched at most once (greedy, left-to-right by label order).
    """
    matched_labels: set[int] = set()
    matched_preds: set[int] = set()

    for li, label in enumerate(labels):
        for pi, pred in enumerate(predictions):
            if pi in matched_preds:
                continue
            if _dispatch_match(pred, label, mode, partial_threshold=partial_threshold):
                matched_labels.add(li)
                matched_preds.add(pi)
                break

    tp = len(matched_labels)
    fp = len(predictions) - len(matched_preds)
    fn = len(labels) - len(matched_labels)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1, tp, fp, fn


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

class StrictMatchMetric(MultiLevelMetric):
    """Exact boundary + exact entity type (SemEval'13 *strict*)."""

    name = "strict_match"

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        return [MatchMode.STRICT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        p, r, f1, tp, fp, fn = _aligned_prf(predictions, labels, MatchMode.STRICT)
        return EvalMetricResult(
            name=self.name, value=f1, level=level, match_mode=MatchMode.STRICT,
            precision=p, recall=r, f1=f1, support=len(labels),
        )


class ExactMatchMetric(MultiLevelMetric):
    """Exact boundary, type ignored (SemEval'13 *exact*)."""

    name = "exact_match"

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        return [MatchMode.EXACT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.EXACT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        p, r, f1, *_ = _aligned_prf(predictions, labels, MatchMode.EXACT)
        return EvalMetricResult(
            name=self.name, value=f1, level=level, match_mode=MatchMode.EXACT,
            precision=p, recall=r, f1=f1, support=len(labels),
        )


class PartialMatchMetric(MultiLevelMetric):
    """IoU overlap above threshold (SemEval'13 *partial*; default IoU >= 0.5)."""

    name = "partial_match"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        return [MatchMode.PARTIAL]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.PARTIAL,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        p, r, f1, *_ = _aligned_prf(
            predictions, labels, MatchMode.PARTIAL, partial_threshold=self.threshold,
        )
        return EvalMetricResult(
            name=self.name, value=f1, level=level, match_mode=MatchMode.PARTIAL,
            precision=p, recall=r, f1=f1, support=len(labels),
            metadata={"iou_threshold": self.threshold},
        )


class TypeMatchMetric(MultiLevelMetric):
    """Type match only, boundary ignored (SemEval'13 *type*)."""

    name = "type_match"

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        return [MatchMode.TYPE]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.TYPE,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        p, r, f1, *_ = _aligned_prf(predictions, labels, MatchMode.TYPE)
        return EvalMetricResult(
            name=self.name, value=f1, level=level, match_mode=MatchMode.TYPE,
            precision=p, recall=r, f1=f1, support=len(labels),
        )


class EntityLevelF1Metric(MultiLevelMetric):
    """Configurable entity-level F1 supporting all four SemEval'13 modes."""

    name = "entity_level_f1"

    def __init__(self, partial_threshold: float = 0.5) -> None:
        self.partial_threshold = partial_threshold

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        return list(MatchMode)

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        p, r, f1, tp, fp, fn = _aligned_prf(
            predictions, labels, match_mode, partial_threshold=self.partial_threshold,
        )
        per_entity = self._per_entity(predictions, labels, match_mode)
        return EvalMetricResult(
            name=self.name, value=f1, level=level, match_mode=match_mode,
            precision=p, recall=r, f1=f1, support=len(labels),
            per_entity_breakdown=per_entity,
        )

    def _per_entity(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        mode: MatchMode,
    ) -> dict[str, dict[str, float]]:
        types = sorted({s.entity_type for s in labels} | {s.entity_type for s in predictions})
        out: dict[str, dict[str, float]] = {}
        for et in types:
            pf = [s for s in predictions if s.entity_type == et]
            lf = [s for s in labels if s.entity_type == et]
            p, r, f1, *_ = _aligned_prf(pf, lf, mode, partial_threshold=self.partial_threshold)
            out[et] = {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6), "support": len(lf)}
        return out


class TokenLevelF1Metric(MultiLevelMetric):
    """Character-level (token) F1 akin to seqeval micro-averaged BIO-tag F1.

    Each character position within a labeled span is treated as a positive.
    """

    name = "token_level_f1"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.TOKEN]

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        return [MatchMode.STRICT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.TOKEN,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        pred_chars: set[tuple[str, str, int]] = set()
        for span in predictions:
            for i in range(span.start, span.end):
                pred_chars.add((span.record_id, span.entity_type, i))

        label_chars: set[tuple[str, str, int]] = set()
        for span in labels:
            for i in range(span.start, span.end):
                label_chars.add((span.record_id, span.entity_type, i))

        tp = len(pred_chars & label_chars)
        fp = len(pred_chars - label_chars)
        fn = len(label_chars - pred_chars)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = compute_f1(precision, recall)
        return EvalMetricResult(
            name=self.name, value=f1, level=EvaluationLevel.TOKEN,
            match_mode=MatchMode.STRICT, precision=precision, recall=recall,
            f1=f1, support=len(label_chars),
        )


class DocumentLevelConsistencyMetric(MultiLevelMetric):
    """Measures whether the same entity string gets the same label across a document.

    Groups predictions by their text content (requires ``context["document_text"]``)
    and checks that entities with identical surface forms receive the same type.
    """

    name = "document_consistency"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.DOCUMENT,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        doc_text = (context or {}).get("document_text", "")
        if not doc_text or not predictions:
            return EvalMetricResult(
                name=self.name, value=1.0, level=level, match_mode=match_mode,
                metadata={"consistent_groups": 0, "total_groups": 0},
            )

        surface_to_types: dict[str, set[str]] = {}
        for span in predictions:
            surface = doc_text[span.start: span.end].lower().strip()
            if surface:
                surface_to_types.setdefault(surface, set()).add(span.entity_type)

        total_groups = len(surface_to_types)
        consistent = sum(1 for types in surface_to_types.values() if len(types) == 1)
        score = safe_div(consistent, total_groups)
        return EvalMetricResult(
            name=self.name, value=round(score, 6), level=level, match_mode=match_mode,
            metadata={"consistent_groups": consistent, "total_groups": total_groups},
        )
