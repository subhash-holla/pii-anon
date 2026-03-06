from __future__ import annotations

import json

from pii_anon.eval_framework.datasets import generator as dataset_generator
from pii_anon.eval_framework.evaluation.context_evaluator import DocumentContextEvaluator
from pii_anon.eval_framework.metrics.base import LabeledSpan


def _span(entity_type: str, start: int, end: int, record_id: str = "r1") -> LabeledSpan:
    return LabeledSpan(entity_type=entity_type, start=start, end=end, record_id=record_id)


def test_generate_dataset_is_deterministic_and_sorted() -> None:
    first = dataset_generator.generate_dataset(seed=123, target_records=120)
    second = dataset_generator.generate_dataset(seed=123, target_records=120)
    assert first == second
    assert len(first) >= len(dataset_generator._LANGUAGE_WEIGHTS) * 50

    record_ids = [row["id"] for row in first]
    assert record_ids == sorted(record_ids)


def test_fill_template_creates_valid_label_offsets() -> None:
    rng = dataset_generator._seeded_rng(99)
    template = dataset_generator._TEMPLATES[0]
    row = dataset_generator._fill_template(template, rng, "en", 7)

    assert row["id"] == "EVAL-000007"
    assert row["text"]
    assert row["labels"]
    assert "gdpr" in row["regulatory_domain"]

    text = row["text"]
    for label in row["labels"]:
        assert 0 <= label["start"] < label["end"] <= len(text)


def test_write_dataset_writes_jsonl_records(tmp_path) -> None:
    out = tmp_path / "eval_framework_v1.jsonl"
    count = dataset_generator.write_dataset(out, seed=7, target_records=140)
    assert count > 0
    assert out.exists()

    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == count

    first = json.loads(lines[0])
    assert {"id", "text", "labels", "language"}.issubset(first.keys())


def test_mention_consistency_defaults_to_perfect_without_context() -> None:
    evaluator = DocumentContextEvaluator()
    result = evaluator.evaluate_mention_consistency(
        predictions=[_span("PERSON_NAME", 0, 10)],
        labels=[_span("PERSON_NAME", 0, 10)],
    )
    assert result.value == 1.0
    assert result.name == "mention_consistency"


def test_mention_consistency_handles_missed_mentions() -> None:
    evaluator = DocumentContextEvaluator()
    labels = [_span("PERSON_NAME", 0, 4), _span("PERSON_NAME", 10, 14)]
    predictions = [_span("PERSON_NAME", 0, 4)]
    clusters = {"person-1": [0, 1, 999], "singleton": [0]}

    result = evaluator.evaluate_mention_consistency(
        predictions=predictions,
        labels=labels,
        entity_clusters=clusters,
        document_text="John met John",
    )
    assert result.value == 0.0
    assert result.metadata["total_clusters"] == 1
    assert result.metadata["consistent_clusters"] == 0


def test_cross_segment_recall_aggregates_segments() -> None:
    evaluator = DocumentContextEvaluator()
    labels = [_span("PERSON_NAME", 0, 4), _span("EMAIL_ADDRESS", 9, 21)]
    segments = [[_span("PERSON_NAME", 0, 4)], []]

    result = evaluator.evaluate_cross_segment_recall(segments, labels)
    assert result.recall == 0.5
    assert result.metadata["segments"] == 2
    assert result.metadata["total_predictions"] == 1


def test_evaluate_by_context_length_groups_by_tier() -> None:
    evaluator = DocumentContextEvaluator()
    records = [
        {
            "context_length_tier": "short",
            "predictions": [_span("PERSON_NAME", 0, 4)],
            "labels": [_span("PERSON_NAME", 0, 4)],
        },
        {
            "context_length_tier": "long",
            "predictions": [],
            "labels": [_span("EMAIL_ADDRESS", 10, 20)],
        },
    ]
    results = evaluator.evaluate_by_context_length(records)
    assert set(results.keys()) == {"long", "short"}
    assert results["short"].f1 == 1.0
    assert results["long"].f1 == 0.0


def test_boundary_reconciliation_reports_improvement() -> None:
    evaluator = DocumentContextEvaluator()
    labels = [_span("PERSON_NAME", 0, 4)]

    results = evaluator.evaluate_boundary_reconciliation(
        pre_reconciliation=[],
        post_reconciliation=[_span("PERSON_NAME", 0, 4)],
        labels=labels,
    )
    assert results["pre_reconciliation"].value == 0.0
    assert results["post_reconciliation"].value == 1.0
    assert results["improvement"].value == 1.0
