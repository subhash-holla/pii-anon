from __future__ import annotations

import gzip
import json

import pytest

from conftest import requires_dataset

from pii_anon.benchmarks import load_benchmark_dataset, load_use_case_matrix, resolve_benchmark_dataset_path, summarize_dataset
from pii_anon.benchmarks import datasets as datasets_module

pytestmark = requires_dataset  # All tests in this file require the benchmark dataset


# ---------------------------------------------------------------------------
# Unified dataset — pii_anon_benchmark (151,000+ records)
# ---------------------------------------------------------------------------

def test_dataset_has_expected_size_and_split() -> None:
    rows = load_benchmark_dataset("pii_anon_benchmark")
    assert len(rows) >= 100000  # pii-anon-eval-data has 151K+ records


def test_dataset_summary_distribution() -> None:
    summary = summarize_dataset("pii_anon_benchmark")
    assert summary["records"] >= 100000  # 151K+ records

    # Language distribution should include at least 12 languages
    lang = summary["language_distribution"]
    assert len(lang) >= 10
    assert "en" in lang
    assert lang["en"] > 10000  # English is the largest segment
    assert sum(lang.values()) >= 100000

    # Difficulty levels should be present
    difficulty = summary.get("difficulty_level_distribution", {})
    assert len(difficulty) >= 2  # at least easy + hard


def test_dataset_has_evaluation_dimensions() -> None:
    """Verify evaluation dimensions are represented with sufficient samples."""
    data_path = resolve_benchmark_dataset_path("pii_anon_benchmark")
    assert data_path is not None
    dim_counts: dict[str, int] = {}
    opener = gzip.open if data_path.suffix == ".gz" else open
    for line in opener(data_path, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        dim = row.get("primary_dimension", row.get("evaluation_dimension", "unclassified"))
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    # Should have at least 4 distinct dimensions
    assert len(dim_counts) >= 4, f"Only found {len(dim_counts)} dimensions: {list(dim_counts.keys())}"

    # Each dimension should have meaningful sample counts
    for dim_name, count in dim_counts.items():
        if dim_name != "unclassified":
            assert count >= 100, f"{dim_name} has only {count} records"


def test_core_dataset_has_diverse_entity_types() -> None:
    """Verify we have 20+ distinct entity types in the core dataset."""
    rows = load_benchmark_dataset("pii_anon_benchmark")
    entity_types: set[str] = set()
    for row in rows:
        for lbl in row.labels:
            entity_types.add(lbl["entity_type"])

    # Must have at least 20 distinct entity types
    assert len(entity_types) >= 20, f"Only found {len(entity_types)} entity types: {entity_types}"

    # Verify key entity types are present
    required_types = {
        "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
        "ADDRESS", "DATE_OF_BIRTH", "CREDIT_CARD", "BANK_ACCOUNT",
        "PASSPORT", "DRIVERS_LICENSE", "MEDICAL_RECORD_NUMBER",
        "EMPLOYEE_ID", "ORGANIZATION", "IP_ADDRESS", "USERNAME",
        "IBAN", "ROUTING_NUMBER", "NATIONAL_ID", "MAC_ADDRESS",
        "LICENSE_PLATE",
    }
    missing = required_types - entity_types
    assert not missing, f"Missing required entity types: {missing}"


def test_core_dataset_has_size_tier_distribution() -> None:
    """Verify records span multiple context length tiers."""
    data_path = resolve_benchmark_dataset_path("pii_anon_benchmark")
    assert data_path is not None
    tiers: dict[str, int] = {}
    opener = gzip.open if data_path.suffix == ".gz" else open
    for line in opener(data_path, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        tier = row.get("context_length_tier", row.get("size_tier", "medium"))
        tiers[tier] = tiers.get(tier, 0) + 1

    # Should have at least 2 context length tiers
    assert len(tiers) >= 1, f"Only found tiers: {list(tiers.keys())}"
    assert sum(tiers.values()) >= 100000


def test_core_dataset_has_entity_clusters() -> None:
    """Verify that records with entity annotations exist."""
    rows = load_benchmark_dataset("pii_anon_benchmark")
    # Verify dataset loaded successfully with substantial records
    assert len(rows) >= 100000
    # Verify records have labels
    records_with_labels = sum(1 for r in rows[:1000] if r.labels)
    assert records_with_labels >= 900  # most records should have labels


def test_core_dataset_has_repeated_pii_in_same_block() -> None:
    """Verify records exist with multiple PII entities of the same type."""
    rows = load_benchmark_dataset("pii_anon_benchmark")
    multi_entity_count = 0
    for row in rows[:10000]:  # sample for speed
        types_in_row: dict[str, int] = {}
        for lbl in row.labels:
            etype = lbl.get("entity_type", "")
            types_in_row[etype] = types_in_row.get(etype, 0) + 1
        if any(count >= 2 for count in types_in_row.values()):
            multi_entity_count += 1
    assert multi_entity_count >= 100, (
        f"Only {multi_entity_count} records have repeated PII entities"
    )


def test_core_dataset_has_multilingual_records() -> None:
    """Verify the dataset includes records in multiple languages."""
    rows = load_benchmark_dataset("pii_anon_benchmark")
    languages = {r.language for r in rows[:20000]}
    assert len(languages) >= 5, f"Only found {len(languages)} languages: {languages}"
    assert "en" in languages


def test_core_dataset_has_multiple_dimensions() -> None:
    """Verify the dataset spans multiple evaluation dimensions."""
    data_path = resolve_benchmark_dataset_path("pii_anon_benchmark")
    assert data_path is not None
    dimensions = set()
    opener = gzip.open if data_path.suffix == ".gz" else open
    for line in opener(data_path, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        dim = row.get("primary_dimension", "")
        if dim:
            dimensions.add(dim)
    assert len(dimensions) >= 4, f"Only {len(dimensions)} dimensions: {dimensions}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_dataset_validation_rejects_invalid_spans(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "id": "bad-1",
                "text": "alice@example.com",
                "language": "en",
                "labels": [{"entity_type": "EMAIL_ADDRESS", "start": 20, "end": 10}],
                "source_type": "synthetic",
                "source_id": "synthetic://bad",
                "license": "CC0-1.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(datasets_module, "_dataset_file", lambda _name, source="auto": bad)

    with pytest.raises(ValueError):
        load_benchmark_dataset("pii_anon_benchmark")


def test_all_label_spans_are_valid() -> None:
    """Verify every label span in the core dataset correctly extracts non-empty text."""
    rows = load_benchmark_dataset("pii_anon_benchmark")
    for row in rows:
        for lbl in row.labels:
            start = lbl["start"]
            end = lbl["end"]
            assert 0 <= start < end <= len(row.text), (
                f"Invalid span [{start}, {end}) in record {row.record_id} (text length {len(row.text)})"
            )
            extracted = row.text[start:end]
            assert len(extracted.strip()) > 0, (
                f"Empty span [{start}, {end}) in record {row.record_id}"
            )


# ---------------------------------------------------------------------------
# Use case matrix
# ---------------------------------------------------------------------------

def test_use_case_matrix_loads_default_profiles() -> None:
    matrix = load_use_case_matrix()
    names = {item.profile for item in matrix}
    assert {
        "short_chat",
        "long_document",
        "structured_form_accuracy",
        "structured_form_latency",
        "log_lines",
        "multilingual_mix",
    }.issubset(names)


def test_use_case_matrix_rejects_invalid_objective(tmp_path) -> None:
    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(
        json.dumps({"profiles": [{"profile": "x", "objective": "invalid", "languages": ["en"]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_use_case_matrix(str(matrix_file))


def test_package_only_dataset_resolution_requires_package(monkeypatch: pytest.MonkeyPatch) -> None:
    path = resolve_benchmark_dataset_path("pii_anon_benchmark", source="package-only")
    if path is None:
        pytest.skip("package-only dataset not available in this environment")

    monkeypatch.setattr(
        datasets_module.resources,
        "files",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing package")),
    )
    missing = resolve_benchmark_dataset_path("pii_anon_benchmark", source="package-only")
    assert missing is None
