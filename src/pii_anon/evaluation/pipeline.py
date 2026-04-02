from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pii_anon.benchmarks import load_benchmark_dataset
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
    dataset: str = "pii_anon_benchmark",
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

    total_tp = 0
    total_fp = 0
    total_fn = 0
    findings_count = 0
    link_audit_count = 0

    profile = ProcessingProfileSpec(
        profile_id=f"pipeline-eval-{mode}",
        mode=mode,
        language=language or "en",
        transform_mode=transform_mode,
        entity_tracking_enabled=True,
        audit_enabled=False,
    )

    for sample in records:
        result = orchestrator.run(
            {"text": sample.text},
            profile=ProcessingProfileSpec(
                profile_id=profile.profile_id,
                mode=profile.mode,
                language=sample.language,
                transform_mode=profile.transform_mode,
                entity_tracking_enabled=profile.entity_tracking_enabled,
                audit_enabled=False,
            ),
            segmentation=SegmentationPlan(enabled=False),
            scope="pipeline-eval",
            token_version=1,
        )
        finding_rows = result.get("ensemble_findings", [])
        findings_count += len(finding_rows)
        link_audit_count += len(result.get("link_audit", []))

        # Streaming TP/FP/FN accumulation per record
        pred_spans: set[tuple[str, int, int]] = set()
        for finding in finding_rows:
            span = finding.get("span", {})
            start = span.get("start")
            end = span.get("end")
            if start is None or end is None:
                continue
            pred_spans.add((str(finding.get("entity_type", "UNKNOWN")), int(start), int(end)))

        label_spans: set[tuple[str, int, int]] = set()
        for label in sample.labels:
            label_spans.add(
                (
                    str(label.get("entity_type", "UNKNOWN")),
                    int(label.get("start", 0)),
                    int(label.get("end", 0)),
                )
            )

        total_tp += len(pred_spans & label_spans)
        total_fp += len(pred_spans - label_spans)
        total_fn += len(label_spans - pred_spans)

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return PipelineEvaluationReport(
        dataset=dataset,
        samples=len(records),
        transform_mode=transform_mode,
        precision=precision,
        recall=recall,
        f1=f1,
        privacy_score=recall,
        fairness_score=1.0,
        avg_findings_per_record=(findings_count / len(records)) if records else 0.0,
        avg_link_audit_per_record=(link_audit_count / len(records)) if records else 0.0,
    )
