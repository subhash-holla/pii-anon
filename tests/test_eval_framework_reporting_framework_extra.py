from __future__ import annotations

import json

from pii_anon.eval_framework.datasets.schema import EvalBenchmarkRecord
from pii_anon.eval_framework.evaluation.framework import (
    BatchEvaluationReport,
    EvaluationFramework,
    EvaluationFrameworkConfig,
    EvaluationReport,
)
from pii_anon.eval_framework.evaluation.reporting import ReportGenerator, _report_to_dict
from pii_anon.eval_framework.metrics.base import EvaluationLevel, LabeledSpan, MatchMode


def _span(entity_type: str, start: int, end: int, record_id: str = "r1") -> LabeledSpan:
    return LabeledSpan(entity_type=entity_type, start=start, end=end, record_id=record_id)


def _record(
    record_id: str,
    *,
    language: str,
    difficulty: str,
    data_type: str,
    context_length_tier: str,
) -> EvalBenchmarkRecord:
    return EvalBenchmarkRecord(
        record_id=record_id,
        text="John john@example.com",
        labels=[
            {"entity_type": "PERSON_NAME", "start": 0, "end": 4},
            {"entity_type": "EMAIL_ADDRESS", "start": 5, "end": 21},
        ],
        language=language,
        source_type="synthetic",
        source_id="synthetic://test",
        license="CC0-1.0",
        difficulty_level=difficulty,
        data_type=data_type,  # type: ignore[arg-type]
        context_length_tier=context_length_tier,  # type: ignore[arg-type]
    )


def test_evaluate_batch_populates_aggregate_dimensions() -> None:
    framework = EvaluationFramework()
    records = [
        _record("en-1", language="en", difficulty="easy", data_type="unstructured_text", context_length_tier="short"),
        _record("es-1", language="es", difficulty="hard", data_type="structured", context_length_tier="long"),
    ]

    report = framework.evaluate_batch(records)
    assert report.records_evaluated == 2
    assert report.micro_averaged["f1"] == 1.0
    assert set(report.per_language.keys()) == {"en", "es"}
    assert set(report.per_difficulty.keys()) == {"easy", "hard"}
    assert set(report.per_data_type.keys()) == {"structured", "unstructured_text"}
    assert set(report.per_context_length.keys()) == {"long", "short"}
    assert report.confidence_interval is not None


def test_evaluate_batch_honors_predict_fn_and_ci_toggle() -> None:
    config = EvaluationFrameworkConfig(
        include_confidence_intervals=False,
        include_fairness_metrics=False,
    )
    framework = EvaluationFramework(config)
    records = [
        _record("en-1", language="en", difficulty="easy", data_type="unstructured_text", context_length_tier="short"),
    ]

    report = framework.evaluate_batch(records, predict_fn=lambda _record: [])
    assert report.confidence_interval is None
    assert report.fairness_score == 0.0
    assert report.micro_averaged["recall"] == 0.0


def test_evaluate_with_token_and_document_levels() -> None:
    config = EvaluationFrameworkConfig(
        evaluation_levels=[EvaluationLevel.ENTITY, EvaluationLevel.TOKEN, EvaluationLevel.DOCUMENT],
        match_modes=[MatchMode.STRICT],
    )
    framework = EvaluationFramework(config)
    predictions = [_span("PERSON_NAME", 0, 4)]
    labels = [_span("PERSON_NAME", 0, 4)]

    report = framework.evaluate(
        predictions,
        labels,
        context={"text": "John", "anonymized_text": "[PERSON]", "original_text": "John"},
    )
    assert "token" in report.metrics_by_level
    assert "document" in report.metrics_by_level


def test_evaluate_by_language_discovers_languages_from_record_map() -> None:
    framework = EvaluationFramework()
    predictions = [_span("PERSON_NAME", 0, 4, "en-1"), _span("PERSON_NAME", 0, 4, "fr-1")]
    labels = list(predictions)

    results = framework.evaluate_by_language(
        predictions,
        labels,
        record_languages={"en-1": "en", "fr-1": "fr"},
    )
    assert set(results.keys()) == {"en", "fr"}
    assert results["en"].f1 == 1.0
    assert results["fr"].f1 == 1.0


def _single_report() -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="single-1",
        timestamp="2026-02-16T00:00:00Z",
        config=EvaluationFrameworkConfig(),
        per_entity_breakdown={"PERSON_NAME": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}},
        language="en",
        precision=1.0,
        recall=1.0,
        f1=1.0,
        privacy_score=0.9,
        utility_score=0.8,
        fairness_score=0.95,
        records_evaluated=1,
    )


def _batch_report() -> BatchEvaluationReport:
    return BatchEvaluationReport(
        evaluation_id="batch-1",
        timestamp="2026-02-16T00:00:00Z",
        records_evaluated=10,
        micro_averaged={"precision": 0.8, "recall": 0.7, "f1": 0.75},
        macro_averaged={"precision": 0.79, "recall": 0.69, "f1": 0.74},
        weighted_averaged={"precision": 0.81, "recall": 0.71, "f1": 0.76},
        per_entity_type={"PERSON_NAME": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 5}},
        per_language={"en": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        per_difficulty={"easy": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        per_data_type={"structured": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        per_context_length={"short": {"precision": 0.9, "recall": 0.8, "f1": 0.85}},
        confidence_interval=(0.72, 0.79),
        privacy_score=0.82,
        fairness_score=0.91,
    )


def test_report_generator_batch_json_and_markdown_and_csv() -> None:
    report = _batch_report()

    payload = json.loads(ReportGenerator.to_json(report))
    assert payload["confidence_interval"] == [0.72, 0.79]
    assert payload["records_evaluated"] == 10

    markdown = ReportGenerator.to_markdown(report)
    assert "Aggregated Metrics" in markdown
    assert "Per-Language Performance" in markdown
    assert "Per-Difficulty Performance" in markdown
    assert "95% Confidence Interval" in markdown

    csv_output = ReportGenerator.to_csv(report)
    assert "entity_type,precision,recall,f1,support" in csv_output
    assert "PERSON_NAME,0.900000,0.800000,0.850000,5" in csv_output


def test_report_generator_single_and_comparison_paths() -> None:
    first = _single_report()
    second = EvaluationReport(
        evaluation_id="single-2",
        timestamp="2026-02-16T00:00:00Z",
        config=EvaluationFrameworkConfig(),
        language="es",
        precision=0.8,
        recall=0.7,
        f1=0.75,
        privacy_score=0.8,
        utility_score=0.7,
        fairness_score=0.85,
        records_evaluated=1,
    )

    markdown = ReportGenerator.to_markdown(first)
    assert "Language" in markdown
    assert "Precision" in markdown
    assert "Recall" in markdown

    comparison = ReportGenerator.render_comparison([first, second], comparison_type="system")
    assert "| single-1 | en |" in comparison
    assert "| single-2 | es |" in comparison

    single_dict = _report_to_dict(first)
    batch_dict = _report_to_dict(_batch_report())
    assert single_dict["utility_score"] == 0.8
    assert batch_dict["privacy_score"] == 0.82
