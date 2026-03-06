from __future__ import annotations

import gzip
import json

import pytest

from pii_anon.benchmarks import load_benchmark_dataset, load_use_case_matrix, resolve_benchmark_dataset_path, summarize_dataset
from pii_anon.benchmarks import datasets as datasets_module


# ---------------------------------------------------------------------------
# Unified dataset — pii_anon_benchmark_v1 (v1.0.0, 50 000 records)
# ---------------------------------------------------------------------------

def test_dataset_has_expected_size_and_split() -> None:
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    assert len(rows) == 50000

    synthetic = load_benchmark_dataset("pii_anon_benchmark_v1", split="synthetic")
    curated = load_benchmark_dataset("pii_anon_benchmark_v1", split="curated_public")
    assert len(synthetic) == 39293
    assert len(curated) == 10707


def test_dataset_summary_distribution() -> None:
    summary = summarize_dataset("pii_anon_benchmark_v1")
    assert summary["records"] == 50000

    # Source split
    assert summary["source_distribution"]["synthetic"] == 39293
    assert summary["source_distribution"]["curated_public"] == 10707

    # Language distribution (12 languages, merged from core + tracking + dimension)
    lang = summary["language_distribution"]
    assert lang["en"] == 28000
    assert lang["es"] == 5499
    assert lang["fr"] == 4029
    assert lang["de"] == 2804
    assert lang["it"] == 1390
    assert lang["pt"] == 1040
    assert lang["nl"] == 525
    assert lang["ja"] == 1168
    assert lang["ar"] == 1400
    assert lang["hi"] == 1400
    assert lang["zh"] == 1400
    assert lang["ko"] == 1345
    assert sum(lang.values()) == 50000

    # Scenarios (core + tracking)
    assert "baseline" in summary["scenario_distribution"]
    assert "context_loss" in summary["scenario_distribution"]
    assert "continuity_tracking" in summary["scenario_distribution"]
    assert "continuity_ambiguous" in summary["scenario_distribution"]

    datatype = summary["datatype_group_distribution"]
    assert datatype["general_pii"] > 0
    assert datatype["passport_intl"] > 0
    assert datatype["tax_id_intl"] > 0
    assert datatype["crypto_wallet"] > 0
    assert datatype["api_key_like_negative"] > 0
    assert datatype["device_id"] > 0
    assert datatype["national_health_id"] > 0

    difficulty = summary["difficulty_level_distribution"]
    assert difficulty["easy"] > 0
    assert difficulty["moderate"] > 0
    assert difficulty["challenging"] > 0
    assert difficulty["hard"] > 0


def test_dataset_has_evaluation_dimensions() -> None:
    """Verify all 7 evaluation dimensions are represented with sufficient samples."""
    data_path = resolve_benchmark_dataset_path("pii_anon_benchmark_v1")
    assert data_path is not None
    dim_counts: dict[str, int] = {}
    opener = gzip.open if data_path.suffix == ".gz" else open
    for line in opener(data_path, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        dim = row.get("evaluation_dimension", "unclassified")
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    required_dims = {
        "entity_consistency", "multilingual", "context_preservation",
        "pii_type_coverage", "edge_cases", "format_variations",
        "temporal_consistency",
    }
    missing = required_dims - set(dim_counts.keys())
    assert not missing, f"Missing evaluation dimensions: {missing}"

    # Minimum sample requirements per dimension (sized for statistical robustness)
    assert dim_counts["entity_consistency"] >= 2000, f"entity_consistency: {dim_counts.get('entity_consistency', 0)}"
    assert dim_counts["multilingual"] >= 5000, f"multilingual: {dim_counts.get('multilingual', 0)}"
    assert dim_counts["context_preservation"] >= 2000, f"context_preservation: {dim_counts.get('context_preservation', 0)}"
    assert dim_counts["pii_type_coverage"] >= 2500, f"pii_type_coverage: {dim_counts.get('pii_type_coverage', 0)}"
    assert dim_counts["edge_cases"] >= 2000, f"edge_cases: {dim_counts.get('edge_cases', 0)}"
    assert dim_counts["format_variations"] >= 1000, f"format_variations: {dim_counts.get('format_variations', 0)}"
    assert dim_counts["temporal_consistency"] >= 1500, f"temporal_consistency: {dim_counts.get('temporal_consistency', 0)}"


def test_core_dataset_has_diverse_entity_types() -> None:
    """Verify we have 20+ distinct entity types in the core dataset."""
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
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
    """Verify records span all four size tiers."""
    # Read the raw JSONL to get size_tier (not part of BenchmarkRecord dataclass)
    data_path = resolve_benchmark_dataset_path("pii_anon_benchmark_v1")
    assert data_path is not None
    size_tiers: dict[str, int] = {}
    opener = gzip.open if data_path.suffix == ".gz" else open
    for line in opener(data_path, "rt", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        tier = row.get("size_tier", "medium")
        size_tiers[tier] = size_tiers.get(tier, 0) + 1

    assert "small" in size_tiers
    assert "medium" in size_tiers
    assert "large" in size_tiers
    assert "very_large" in size_tiers
    # Each tier should have a meaningful number of records
    assert size_tiers["small"] >= 1000
    assert size_tiers["medium"] >= 3000
    assert size_tiers["large"] >= 1000
    assert size_tiers["very_large"] >= 200


def test_core_dataset_has_entity_clusters() -> None:
    """Verify that records with entity clusters exist for coreference testing."""
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    clustered = [r for r in rows if r.entity_cluster_id != "none"]
    assert len(clustered) >= 500, f"Only {len(clustered)} records have entity clusters"


def test_core_dataset_has_repeated_pii_in_same_block() -> None:
    """Verify records exist with multiple mentions of the same entity (same cluster)."""
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    multi_mention_count = 0
    for row in rows:
        clusters_in_row: dict[str, int] = {}
        for lbl in row.labels:
            cid = lbl.get("entity_cluster_id", "none")
            if cid != "none":
                clusters_in_row[cid] = clusters_in_row.get(cid, 0) + 1
        if any(count >= 2 for count in clusters_in_row.values()):
            multi_mention_count += 1
    assert multi_mention_count >= 200, (
        f"Only {multi_mention_count} records have repeated PII for same entity"
    )


def test_core_dataset_has_name_variant_mentions() -> None:
    """Verify we have records with diverse name mention variants
    (full_name, formal, first_name, first_last_initial, etc.)."""
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    variant_types: set[str] = set()
    for row in rows:
        for lbl in row.labels:
            v = lbl.get("mention_variant", "none")
            if v != "none":
                variant_types.add(v)

    expected_variants = {"full_name", "formal", "first_name"}
    missing = expected_variants - variant_types
    assert not missing, f"Missing expected mention variants: {missing}"


def test_core_dataset_context_loss_scenarios() -> None:
    """Verify context-loss scenario records exist."""
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    ctx_loss = [r for r in rows if r.scenario_id == "context_loss"]
    assert len(ctx_loss) >= 100, f"Only {len(ctx_loss)} context-loss records"

    context_groups = {r.context_group for r in ctx_loss}
    # Should have various context-loss sub-categories
    assert len(context_groups) >= 2, f"Only {len(context_groups)} context groups in context_loss"


# ---------------------------------------------------------------------------
# Tracking subset — continuity records within pii_anon_benchmark_v1 (2,800 records)
# ---------------------------------------------------------------------------

def test_long_context_tracking_dataset_loads_with_expected_schema() -> None:
    all_rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    rows = [r for r in all_rows if r.scenario_id.startswith("continuity_")]
    assert len(rows) == 2800
    first = rows[0]
    assert first.scenario_id in {"continuity_tracking", "continuity_ambiguous"}
    assert first.context_group == "tracking"
    assert first.entity_cluster_id.startswith("person-")


def test_tracking_dataset_has_canonical_and_ambiguous_splits() -> None:
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    canonical = [r for r in rows if r.scenario_id == "continuity_tracking"]
    ambiguous = [r for r in rows if r.scenario_id == "continuity_ambiguous"]
    assert len(canonical) == 1750
    assert len(ambiguous) == 1050


def test_tracking_dataset_has_rich_alias_sets() -> None:
    """Verify tracking records have multiple mention variants per entity cluster."""
    all_rows = load_benchmark_dataset("pii_anon_benchmark_v1")
    rows = [r for r in all_rows if r.scenario_id.startswith("continuity_")]
    for row in rows[:50]:  # Check first 50 continuity records
        variants = {lbl.get("mention_variant", "none") for lbl in row.labels if lbl.get("mention_variant", "none") != "none"}
        if row.scenario_id == "continuity_tracking":
            # Should have at least 3 different variant types
            assert len(variants) >= 3, (
                f"Record {row.record_id} has only {len(variants)} variants: {variants}"
            )


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
        load_benchmark_dataset("pii_anon_benchmark_v1")


def test_all_label_spans_are_valid() -> None:
    """Verify every label span in the core dataset correctly extracts non-empty text."""
    rows = load_benchmark_dataset("pii_anon_benchmark_v1")
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
    path = resolve_benchmark_dataset_path("pii_anon_benchmark_v1", source="package-only")
    if path is None:
        pytest.skip("package-only dataset not available in this environment")

    monkeypatch.setattr(
        datasets_module.resources,
        "files",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing package")),
    )
    missing = resolve_benchmark_dataset_path("pii_anon_benchmark_v1", source="package-only")
    assert missing is None
