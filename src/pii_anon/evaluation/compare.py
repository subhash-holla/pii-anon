from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pii_anon.benchmarks import load_benchmark_dataset
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.metrics import SpanFBetaMetric
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan, StrategyComparisonResult


@dataclass
class StrategyEvaluationReport:
    results: list[StrategyComparisonResult]
    winner: str | None


class StrategyEvaluator:
    def __init__(self, orchestrator: PIIOrchestrator) -> None:
        self.orchestrator = orchestrator

    def _evaluate_strategy(
        self,
        strategy: str,
        records: list[Any],
    ) -> StrategyComparisonResult:
        metric = SpanFBetaMetric(beta=1.0)
        labels_total: list[dict[str, Any]] = []
        predictions_total: list[dict[str, Any]] = []
        confidences: list[float] = []

        profile = ProcessingProfileSpec(
            profile_id=f"eval-{strategy}",
            mode=strategy,
            entity_tracking_enabled=False,
            audit_enabled=False,
        )
        segmentation = SegmentationPlan(enabled=False)
        for sample in records:
            result = self.orchestrator.run(
                {"text": sample.text},
                profile=profile,
                segmentation=segmentation,
                scope="evaluation",
                token_version=1,
            )
            findings = result["ensemble_findings"]
            for item in findings:
                predictions_total.append(
                    {
                        "entity_type": item["entity_type"],
                        "start": item["span"]["start"],
                        "end": item["span"]["end"],
                    }
                )
                confidences.append(float(item["confidence"]))
            for label in sample.labels:
                labels_total.append(
                    {
                        "entity_type": label.get("entity_type"),
                        "start": label.get("start"),
                        "end": label.get("end"),
                    }
                )

        metric_result = metric.compute(predictions_total, labels_total, context={})
        return StrategyComparisonResult(
            strategy=strategy,
            span_fbeta=metric_result.value,
            findings_count=len(predictions_total),
            avg_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        )

    def compare_strategies(
        self,
        strategies: list[str],
        *,
        dataset: str = "pii_anon_benchmark_v1",
        max_workers: int | None = None,
    ) -> StrategyEvaluationReport:
        records = load_benchmark_dataset(dataset)
        workers = max_workers or min(8, (os.cpu_count() or 1))

        if len(strategies) <= 1:
            outputs = [self._evaluate_strategy(s, records) for s in strategies]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._evaluate_strategy, s, records) for s in strategies]
                outputs = [f.result() for f in futures]

        winner = None
        if outputs:
            winner = sorted(outputs, key=lambda item: item.span_fbeta, reverse=True)[0].strategy
        return StrategyEvaluationReport(results=outputs, winner=winner)
