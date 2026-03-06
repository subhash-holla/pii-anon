from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pii_anon.benchmarks import load_benchmark_dataset
from pii_anon.eval_framework import EvaluationFramework, LabeledSpan
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@dataclass
class PipelineEvaluationReport:
    dataset: str
    samples: int
    transform_mode: Literal["pseudonymize", "anonymize"]
    precision: float
    recall: float
    f1: float
    privacy_score: float
    fairness_score: float
    avg_findings_per_record: float
    avg_link_audit_per_record: float


def evaluate_pipeline(
    orchestrator: PIIOrchestrator,
    *,
    dataset: str = "pii_anon_benchmark_v1",
    mode: str = "weighted_consensus",
    transform_mode: Literal["pseudonymize", "anonymize"] = "pseudonymize",
    language: str | None = None,
    max_samples: int | None = None,
) -> PipelineEvaluationReport:
    records = load_benchmark_dataset(dataset)
    if language:
        records = [item for item in records if item.language == language]
    if max_samples is not None:
        records = records[: max(0, max_samples)]
    if not records:
        raise ValueError("no records available for pipeline evaluation")

    predictions: list[LabeledSpan] = []
    labels: list[LabeledSpan] = []
    findings_count = 0
    link_audit_count = 0

    for sample in records:
        result = orchestrator.run(
            {"text": sample.text},
            profile=ProcessingProfileSpec(
                profile_id=f"pipeline-eval-{mode}",
                mode=mode,
                language=sample.language,
                transform_mode=transform_mode,
                entity_tracking_enabled=True,
            ),
            segmentation=SegmentationPlan(enabled=False),
            scope="pipeline-eval",
            token_version=1,
        )
        finding_rows = result.get("ensemble_findings", [])
        findings_count += len(finding_rows)
        link_audit_count += len(result.get("link_audit", []))

        for finding in finding_rows:
            span = finding.get("span", {})
            start = span.get("start")
            end = span.get("end")
            if start is None or end is None:
                continue
            predictions.append(
                LabeledSpan(
                    entity_type=str(finding.get("entity_type", "UNKNOWN")),
                    start=int(start),
                    end=int(end),
                    record_id=sample.record_id,
                )
            )
        for label in sample.labels:
            labels.append(
                LabeledSpan(
                    entity_type=str(label.get("entity_type", "UNKNOWN")),
                    start=int(label.get("start", 0)),
                    end=int(label.get("end", 0)),
                    record_id=sample.record_id,
                )
            )

    fw = EvaluationFramework()
    report = fw.evaluate(predictions, labels, language=language or "mixed")

    return PipelineEvaluationReport(
        dataset=dataset,
        samples=len(records),
        transform_mode=transform_mode,
        precision=report.precision,
        recall=report.recall,
        f1=report.f1,
        privacy_score=report.privacy_score,
        fairness_score=report.fairness_score,
        avg_findings_per_record=(findings_count / len(records)) if records else 0.0,
        avg_link_audit_per_record=(link_audit_count / len(records)) if records else 0.0,
    )
