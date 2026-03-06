from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class MetricResult:
    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricPlugin:
    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        raise NotImplementedError


class SpanFBetaMetric(MetricPlugin):
    def __init__(self, beta: float = 2.0) -> None:
        self.beta = beta

    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        pred_set = {tuple(sorted(item.items())) for item in predictions}
        label_set = {tuple(sorted(item.items())) for item in labels}
        tp = len(pred_set & label_set)
        fp = len(pred_set - label_set)
        fn = len(label_set - pred_set)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        b2 = self.beta ** 2
        denom = (b2 * precision) + recall
        fbeta = ((1 + b2) * precision * recall / denom) if denom else 0.0
        return MetricResult("span_fbeta", fbeta, {"precision": precision, "recall": recall, "beta": self.beta})


class LeakageAtTMetric(MetricPlugin):
    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        horizon = context.get("horizon_seconds", 10)
        leaked = 0
        for label in labels:
            token = str(label.get("token", ""))
            if token and token in str(predictions):
                leaked += 1
        rate = leaked / max(1, len(labels))
        return MetricResult("leakage_at_t", rate, {"horizon_seconds": horizon, "leaked": leaked, "total": len(labels)})


class BoundaryLossMetric(MetricPlugin):
    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        missed_boundary = 0
        boundary_total = 0
        predicted_keys = {p.get("id") for p in predictions}
        for label in labels:
            if label.get("boundary_case"):
                boundary_total += 1
                if label.get("id") not in predicted_keys:
                    missed_boundary += 1
        value = missed_boundary / boundary_total if boundary_total else 0.0
        return MetricResult("boundary_loss", value, {"missed": missed_boundary, "total": boundary_total})


class TokenStabilityMetric(MetricPlugin):
    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        groups: dict[str, set[str]] = {}
        for p in predictions:
            entity_key = str(p.get("entity_key", ""))
            token = str(p.get("token", ""))
            if not entity_key or not token:
                continue
            groups.setdefault(entity_key, set()).add(token)
        stable = sum(1 for tokens in groups.values() if len(tokens) == 1)
        value = stable / max(1, len(groups))
        return MetricResult("token_stability", value, {"stable_entities": stable, "entities": len(groups)})


class LLMLeakageMetric(MetricPlugin):
    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        outputs = [str(item.get("output", "")) for item in predictions]
        leaked = 0
        for label in labels:
            token = str(label.get("token", ""))
            if token and any(token in out for out in outputs):
                leaked += 1
        value = leaked / max(1, len(labels))
        return MetricResult("llm_leakage", value, {"leaked": leaked, "total": len(labels)})


class FairnessGapMetric(MetricPlugin):
    def compute(self, predictions: list[dict[str, Any]], labels: list[dict[str, Any]], context: dict[str, Any]) -> MetricResult:
        recalls: dict[str, list[float]] = {}
        pred_ids = {p.get("id") for p in predictions}
        for label in labels:
            group = str(label.get("group", "unknown"))
            hit = 1.0 if label.get("id") in pred_ids else 0.0
            recalls.setdefault(group, []).append(hit)
        by_group = {g: mean(vals) if vals else 0.0 for g, vals in recalls.items()}
        gap = (max(by_group.values()) - min(by_group.values())) if by_group else 0.0
        return MetricResult("fairness_gap", gap, {"by_group": by_group})
