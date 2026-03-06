"""Base classes for multi-level, multi-mode evaluation metrics.

Evidence basis:
- SemEval 2013 Task 9.1 (Segura-Bedmar et al., 2013): strict/exact/partial/type
  matching modes for entity-level evaluation
- nervaluate (Batista, 2018): entity-level NER evaluation with four matching modes
- seqeval (Nakayama, 2018): token-level sequence labelling metrics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvaluationLevel(str, Enum):
    """Granularity at which evaluation is performed.

    TOKEN     – character / sub-word level (akin to BIO-tag accuracy)
    ENTITY    – full entity span matching (standard NER metric)
    DOCUMENT  – whole-document consistency checks
    MENTION   – co-reference / entity-mention clustering
    """

    TOKEN = "token"
    ENTITY = "entity"
    DOCUMENT = "document"
    MENTION = "mention"


class MatchMode(str, Enum):
    """Entity matching strategies per SemEval'13 / nervaluate.

    STRICT  – boundaries AND entity type must match exactly
    EXACT   – boundaries must match exactly, type ignored
    PARTIAL – overlap above a configurable IoU threshold
    TYPE    – entity type must match, boundary ignored
    """

    STRICT = "strict"
    EXACT = "exact"
    PARTIAL = "partial"
    TYPE = "type"


@dataclass
class EvalMetricResult:
    """Outcome of a single metric computation."""

    name: str
    value: float
    level: EvaluationLevel = EvaluationLevel.ENTITY
    match_mode: MatchMode = MatchMode.STRICT
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    per_entity_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Span representation used throughout the metrics package
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabeledSpan:
    """A single annotated or predicted PII span."""

    entity_type: str
    start: int
    end: int
    record_id: str = ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float) -> float:
    """Return *numerator / denominator*, or ``0.0`` when *denominator* is zero."""
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def compute_f1(precision: float, recall: float) -> float:
    """Harmonic mean of *precision* and *recall*."""
    return safe_div(2.0 * precision * recall, precision + recall)


def compute_iou(pred_start: int, pred_end: int, label_start: int, label_end: int) -> float:
    """Intersection-over-Union for two integer spans."""
    inter_start = max(pred_start, label_start)
    inter_end = min(pred_end, label_end)
    intersection = max(0, inter_end - inter_start)
    union = (pred_end - pred_start) + (label_end - label_start) - intersection
    return safe_div(intersection, union)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MultiLevelMetric(ABC):
    """Abstract base for evaluation metrics that support multiple levels and match modes.

    Sub-classes implement :meth:`compute` and optionally override
    :attr:`supported_levels` / :attr:`supported_match_modes` to declare
    their capabilities.
    """

    name: str = "base_metric"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        """Evaluation levels this metric can operate at."""
        return [EvaluationLevel.ENTITY]

    @property
    def supported_match_modes(self) -> list[MatchMode]:
        """Match modes this metric supports."""
        return [MatchMode.STRICT]

    @abstractmethod
    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        """Run the metric and return a result.

        Parameters
        ----------
        predictions:
            Detected / predicted spans.
        labels:
            Ground-truth spans.
        level:
            Granularity (TOKEN, ENTITY, DOCUMENT, MENTION).
        match_mode:
            Boundary matching strategy (STRICT, EXACT, PARTIAL, TYPE).
        context:
            Optional extra information (document text, metadata, etc.).
        """
        ...

    def compute_per_entity_type(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
    ) -> dict[str, EvalMetricResult]:
        """Compute the metric independently for each entity type."""
        entity_types = sorted({span.entity_type for span in labels})
        results: dict[str, EvalMetricResult] = {}
        for et in entity_types:
            pred_filtered = [s for s in predictions if s.entity_type == et]
            label_filtered = [s for s in labels if s.entity_type == et]
            results[et] = self.compute(
                pred_filtered, label_filtered, level=level, match_mode=match_mode,
            )
        return results
