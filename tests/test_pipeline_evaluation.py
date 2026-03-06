from __future__ import annotations

from pii_anon.evaluation.pipeline import evaluate_pipeline
from pii_anon.orchestrator import PIIOrchestrator


def test_evaluate_pipeline_returns_metrics() -> None:
    orchestrator = PIIOrchestrator(token_key="k")
    report = evaluate_pipeline(
        orchestrator,
        dataset="pii_anon_benchmark_v1",
        transform_mode="pseudonymize",
        max_samples=5,
    )
    assert report.samples == 5
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1 <= 1.0


def test_evaluate_pipeline_supports_anonymize_mode() -> None:
    orchestrator = PIIOrchestrator(token_key="k")
    report = evaluate_pipeline(
        orchestrator,
        dataset="pii_anon_benchmark_v1",
        transform_mode="anonymize",
        max_samples=3,
    )
    assert report.transform_mode == "anonymize"
    assert report.samples == 3
