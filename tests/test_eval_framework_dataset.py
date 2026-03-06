"""Tests for pii_anon.eval_framework.datasets module.

Validates:
- Dataset schema (EvalBenchmarkRecord)
- Dataset loading and filtering
- Dataset summary statistics
- Generated dataset integrity (50K records)
"""

from __future__ import annotations

import os

import pytest

from pii_anon.eval_framework.datasets.schema import (
    EvalBenchmarkRecord,
    load_eval_dataset,
    resolve_eval_dataset_path as resolve_eval_path,
    resolve_eval_dataset_path,
    summarize_eval_dataset,
)
from pii_anon.eval_framework.datasets import schema as eval_schema_module


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestEvalBenchmarkRecord:
    """EvalBenchmarkRecord dataclass validation."""

    def test_minimal_record(self) -> None:
        rec = EvalBenchmarkRecord(
            record_id="test_001",
            text="John Smith lives at 123 Main St.",
            labels=[
                {"entity_type": "PERSON_NAME", "start": 0, "end": 10},
                {"entity_type": "ADDRESS", "start": 20, "end": 32},
            ],
            language="en",
            source_type="synthetic",
            source_id="test",
            license="Apache-2.0",
        )
        assert rec.record_id == "test_001"
        assert len(rec.labels) == 2
        assert rec.language == "en"

    def test_extended_fields_have_defaults(self) -> None:
        rec = EvalBenchmarkRecord(
            record_id="test_002",
            text="Hello",
            labels=[],
            language="en",
            source_type="synthetic",
            source_id="test",
            license="Apache-2.0",
        )
        assert rec.data_type == "unstructured_text"
        assert rec.context_length_tier == "medium"
        assert rec.token_count == 0
        assert rec.regulatory_domain == []
        assert rec.adversarial_type is None
        assert rec.script == "Latin"
        assert rec.entity_types_present == []


# ---------------------------------------------------------------------------
# Dataset loading (uses the default dataset name, resolves via _DATA_DIR)
# ---------------------------------------------------------------------------

# The load_eval_dataset function takes a dataset *name* (not path),
# and resolves it relative to its internal _DATA_DIR.
_DATASET_NAME = "eval_framework_v1"


class TestDatasetLoading:
    """Load the generated 50K-record dataset."""

    @pytest.fixture(autouse=True)
    def _check_dataset_exists(self) -> None:
        dataset_file = resolve_eval_dataset_path(_DATASET_NAME)
        if dataset_file is None or not os.path.exists(dataset_file):
            pytest.skip("Dataset file not found (expected at eval_framework_v1.jsonl)")

    def test_load_full_dataset(self) -> None:
        records = load_eval_dataset(_DATASET_NAME)
        assert len(records) >= 50_000

    def test_load_filter_by_language(self) -> None:
        records = load_eval_dataset(_DATASET_NAME, language="en")
        assert len(records) > 0
        assert all(r.language == "en" for r in records)

    def test_load_filter_by_difficulty(self) -> None:
        records = load_eval_dataset(_DATASET_NAME, difficulty="hard")
        assert len(records) > 0
        assert all(r.difficulty_level == "hard" for r in records)

    def test_all_records_have_required_fields(self) -> None:
        records = load_eval_dataset(_DATASET_NAME)
        for rec in records[:100]:  # spot-check first 100
            assert rec.record_id != ""
            assert rec.text != ""
            assert rec.language != ""
            assert isinstance(rec.labels, list)


class TestDatasetSummary:
    """Dataset summary statistics."""

    @pytest.fixture(autouse=True)
    def _check_dataset_exists(self) -> None:
        dataset_file = resolve_eval_dataset_path(_DATASET_NAME)
        if dataset_file is None or not os.path.exists(dataset_file):
            pytest.skip("Dataset file not found")

    def test_summarize_returns_distributions(self) -> None:
        summary = summarize_eval_dataset(_DATASET_NAME)
        assert "total_records" in summary
        assert "by_language" in summary
        assert "entity_types" in summary
        assert summary["total_records"] >= 50_000

    def test_multilingual_coverage(self) -> None:
        summary = summarize_eval_dataset(_DATASET_NAME)
        languages = summary.get("by_language", {})
        assert len(languages) >= 40, (
            f"Expected 40+ languages in dataset, found {len(languages)}"
        )


def test_eval_package_only_resolution_requires_package(monkeypatch: pytest.MonkeyPatch) -> None:
    path = resolve_eval_path("eval_framework_v1", source="package-only")
    if path is None:
        pytest.skip("package-only eval dataset not available in this environment")

    monkeypatch.setattr(
        eval_schema_module.resources,
        "files",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing package")),
    )
    missing = resolve_eval_path("eval_framework_v1", source="package-only")
    assert missing is None
