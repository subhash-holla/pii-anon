"""Tests for the dataset statistics module.

Tests cover statistical validation, dataset adequacy assessment, and
comprehensive statistics computation for PII Anonymization Evaluation Framework.
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.datasets.statistics import (
    calculate_sample_size_per_stratum,
    power_analysis_for_comparison,
    validate_dataset_coverage,
    compute_dataset_statistics,
)
from pii_anon.eval_framework.datasets.schema import EvalBenchmarkRecord


class TestCalculateSampleSizePerStratum:
    """Test Cochran's formula for sample size calculation."""

    def test_default_parameters(self):
        """Test with default parameters."""
        n = calculate_sample_size_per_stratum(20000)
        assert n > 0
        assert isinstance(n, int)

    def test_confidence_level_90(self):
        """Test with 90% confidence level."""
        n = calculate_sample_size_per_stratum(20000, confidence_level=0.90)
        assert n > 0
        assert isinstance(n, int)

    def test_confidence_level_95(self):
        """Test with 95% confidence level (default)."""
        n = calculate_sample_size_per_stratum(20000, confidence_level=0.95)
        assert n > 0
        assert isinstance(n, int)

    def test_confidence_level_99(self):
        """Test with 99% confidence level."""
        n = calculate_sample_size_per_stratum(20000, confidence_level=0.99)
        assert n > 0
        assert isinstance(n, int)

    def test_higher_confidence_requires_more_samples(self):
        """Higher confidence level should require more samples."""
        n_90 = calculate_sample_size_per_stratum(20000, confidence_level=0.90)
        n_95 = calculate_sample_size_per_stratum(20000, confidence_level=0.95)
        n_99 = calculate_sample_size_per_stratum(20000, confidence_level=0.99)

        assert n_90 < n_95 < n_99

    def test_larger_margin_of_error_requires_fewer_samples(self):
        """Larger margin of error should require fewer samples."""
        n_1 = calculate_sample_size_per_stratum(20000, margin_of_error=0.01)
        n_5 = calculate_sample_size_per_stratum(20000, margin_of_error=0.05)
        n_10 = calculate_sample_size_per_stratum(20000, margin_of_error=0.10)

        assert n_1 > n_5 > n_10

    def test_small_population(self):
        """Test with small population size."""
        n = calculate_sample_size_per_stratum(100, confidence_level=0.95)
        assert 0 < n <= 100  # Can't sample more than population

    def test_very_large_population(self):
        """Test with very large population."""
        n = calculate_sample_size_per_stratum(1_000_000, confidence_level=0.95)
        assert n > 0

    def test_zero_population_returns_positive(self):
        """Test with zero population returns positive minimum."""
        n = calculate_sample_size_per_stratum(0, confidence_level=0.95)
        assert n >= 1

    def test_example_from_docstring(self):
        """Test example from docstring."""
        n = calculate_sample_size_per_stratum(20000, confidence_level=0.95, margin_of_error=0.05)
        # Expected: 377 samples (from docstring)
        assert 350 <= n <= 400  # Approximate range

    def test_returns_integer(self):
        """Test that result is always integer."""
        for pop in [100, 1000, 10000]:
            n = calculate_sample_size_per_stratum(pop)
            assert isinstance(n, int)

    def test_custom_variance(self):
        """Test with custom variance parameter."""
        n1 = calculate_sample_size_per_stratum(20000, variance=0.25)
        n2 = calculate_sample_size_per_stratum(20000, variance=0.1)

        assert n1 > 0 and n2 > 0
        assert n1 > n2  # Higher variance requires more samples

    def test_finite_population_correction(self):
        """Test that finite population correction is applied."""
        # For small populations, FPC should reduce sample size
        n_small = calculate_sample_size_per_stratum(100)
        n_large = calculate_sample_size_per_stratum(100000)

        # Both should be positive
        assert n_small > 0 and n_large > 0
        # Small population should require fewer samples due to FPC
        assert n_small < n_large


class TestPowerAnalysisForComparison:
    """Test power analysis for hypothesis testing."""

    def test_default_parameters(self):
        """Test with default parameters."""
        n = power_analysis_for_comparison()
        assert n > 0
        assert isinstance(n, int)

    def test_small_effect_size(self):
        """Test with small effect size (d=0.2)."""
        n = power_analysis_for_comparison(effect_size=0.2)
        assert n > 0

    def test_medium_effect_size(self):
        """Test with medium effect size (d=0.5)."""
        n = power_analysis_for_comparison(effect_size=0.5)
        assert n > 0

    def test_large_effect_size(self):
        """Test with large effect size (d=0.8)."""
        n = power_analysis_for_comparison(effect_size=0.8)
        assert n > 0

    def test_smaller_effect_size_requires_more_samples(self):
        """Smaller effect size should require more samples."""
        n_small = power_analysis_for_comparison(effect_size=0.2)
        n_medium = power_analysis_for_comparison(effect_size=0.5)
        n_large = power_analysis_for_comparison(effect_size=0.8)

        assert n_small > n_medium > n_large

    def test_higher_significance_level_requires_more_samples(self):
        """Higher significance level should require more samples."""
        n_01 = power_analysis_for_comparison(alpha=0.01)
        n_05 = power_analysis_for_comparison(alpha=0.05)

        assert n_01 > n_05

    def test_lower_power_requires_fewer_samples(self):
        """Lower statistical power should require fewer samples."""
        n_80 = power_analysis_for_comparison(power=0.80)
        n_90 = power_analysis_for_comparison(power=0.90)

        assert n_80 < n_90

    def test_example_small_effect(self):
        """Test example from docstring for small effect."""
        n = power_analysis_for_comparison(effect_size=0.2, alpha=0.05, power=0.80)
        # Computed value should be around 197
        assert 180 <= n <= 220

    def test_example_medium_effect(self):
        """Test example from docstring for medium effect."""
        n = power_analysis_for_comparison(effect_size=0.5, alpha=0.05, power=0.90)
        # Computed value
        assert 30 <= n <= 50

    def test_negative_effect_size_raises(self):
        """Test that negative effect size raises ValueError."""
        with pytest.raises(ValueError, match="effect_size must be positive"):
            power_analysis_for_comparison(effect_size=-0.5)

    def test_zero_effect_size_raises(self):
        """Test that zero effect size raises ValueError."""
        with pytest.raises(ValueError, match="effect_size must be positive"):
            power_analysis_for_comparison(effect_size=0.0)

    def test_returns_integer(self):
        """Test that result is always integer."""
        n = power_analysis_for_comparison(effect_size=0.5, alpha=0.05, power=0.80)
        assert isinstance(n, int)

    def test_supported_alpha_levels(self):
        """Test supported alpha levels."""
        for alpha in [0.01, 0.05, 0.001]:
            n = power_analysis_for_comparison(alpha=alpha)
            assert n > 0

    def test_supported_power_levels(self):
        """Test supported power levels."""
        for power in [0.80, 0.85, 0.90, 0.95]:
            n = power_analysis_for_comparison(power=power)
            assert n > 0

    def test_unsupported_alpha_defaults_gracefully(self):
        """Test that unsupported alpha levels are handled."""
        n = power_analysis_for_comparison(alpha=0.07)  # Not in map
        assert n > 0


class TestValidateDatasetCoverage:
    """Test dataset coverage validation."""

    def test_empty_dataset(self):
        """Test validation of empty dataset."""
        report = validate_dataset_coverage([])

        assert report["total_records"] == 0
        assert report["dimension_adequacy"] is False
        assert report["recommendation"] == "FAIL"
        assert len(report["coverage_gaps"]) == 0

    def test_small_dataset_fails(self):
        """Test that small dataset (<50K) fails."""
        records = [
            EvalBenchmarkRecord(
                record_id=f"test_{i}",
                text="Test",
                labels=[],
                language="en",
                dimension_tags=["entity_tracking"],
            )
            for i in range(100)
        ]

        report = validate_dataset_coverage(records)
        assert report["recommendation"] == "FAIL"

    def test_report_structure(self):
        """Test that report has required structure."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            language="en",
        )

        report = validate_dataset_coverage([record])

        required_keys = {
            "total_records",
            "by_dimension",
            "by_language",
            "by_entity_type",
            "coverage_gaps",
            "dimension_adequacy",
            "statistical_power",
            "recommendation",
            "details",
        }
        assert set(report.keys()) == required_keys

    def test_total_records_count(self):
        """Test that total_records is correct."""
        records = [
            EvalBenchmarkRecord(record_id=f"test_{i}", text="Test", labels=[])
            for i in range(10)
        ]

        report = validate_dataset_coverage(records)
        assert report["total_records"] == 10

    def test_by_dimension_counting(self):
        """Test dimension counting."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              dimension_tags=["entity_tracking"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              dimension_tags=["entity_tracking"]),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              dimension_tags=["multilingual"]),
        ]

        report = validate_dataset_coverage(records)
        assert report["by_dimension"]["entity_tracking"] == 2
        assert report["by_dimension"]["multilingual"] == 1

    def test_by_language_counting(self):
        """Test language counting."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[], language="es"),
        ]

        report = validate_dataset_coverage(records)
        assert report["by_language"]["en"] == 2
        assert report["by_language"]["es"] == 1

    def test_by_entity_type_counting(self):
        """Test entity type counting."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              entity_types_present=["PERSON"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              entity_types_present=["PERSON", "EMAIL"]),
        ]

        report = validate_dataset_coverage(records)
        assert report["by_entity_type"]["PERSON"] == 2
        assert report["by_entity_type"]["EMAIL"] == 1

    def test_sparse_language_detected_as_gap(self):
        """Test that sparse languages are detected as gaps."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], language="xx"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[], language="xx"),
        ]

        report = validate_dataset_coverage(records)
        sparse_gap = [g for g in report["coverage_gaps"] if g.get("type") == "language"]
        # Only 2 records of language "xx", below threshold of 5
        assert len(sparse_gap) > 0

    def test_sparse_entity_type_detected_as_gap(self):
        """Test that sparse entity types are detected as gaps."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              entity_types_present=["PERSON"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              entity_types_present=["RARE_TYPE"]),
        ]

        report = validate_dataset_coverage(records)
        sparse_gap = [g for g in report["coverage_gaps"] if g.get("type") == "entity_type"]
        # "RARE_TYPE" has only 1 record, below threshold of 10
        assert len(sparse_gap) > 0

    def test_statistical_power_computed(self):
        """Test that statistical power is computed."""
        records = [
            EvalBenchmarkRecord(record_id=f"test_{i}", text="Test", labels=[],
                              dimension_tags=["entity_tracking"])
            for i in range(100)
        ]

        report = validate_dataset_coverage(records)
        assert report["statistical_power"] >= 0
        assert isinstance(report["statistical_power"], float)

    def test_details_string_populated(self):
        """Test that details string is populated."""
        records = [
            EvalBenchmarkRecord(record_id="test_1", text="Test", labels=[])
        ]

        report = validate_dataset_coverage(records)
        assert len(report["details"]) > 0
        assert "records" in report["details"].lower()


class TestComputeDatasetStatistics:
    """Test comprehensive dataset statistics computation."""

    def test_empty_dataset(self):
        """Test statistics for empty dataset."""
        stats = compute_dataset_statistics([])

        assert stats["count"] == 0
        assert len(stats["dimensions"]) == 0
        assert len(stats["languages"]) == 0
        assert stats["adversarial_stats"]["clean"] == 0

    def test_single_record_statistics(self):
        """Test statistics for single record."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test text with five tokens",
            labels=[{"entity_id": "e1", "start": 0, "end": 4, "entity_cluster_id": "c1"}],
            language="en",
            dimension_tags=["entity_tracking"],
            entity_types_present=["PERSON"],
            token_count=5,
            difficulty_level="easy",
            data_type="unstructured_text",
            context_length_tier="short",
        )

        stats = compute_dataset_statistics([record])

        assert stats["count"] == 1
        assert stats["dimensions"]["entity_tracking"]["count"] == 1
        assert stats["languages"]["en"]["count"] == 1
        assert stats["entity_types"]["PERSON"] == 1

    def test_statistics_structure(self):
        """Test that statistics has required structure."""
        record = EvalBenchmarkRecord(record_id="1", text="Test", labels=[])
        stats = compute_dataset_statistics([record])

        required_keys = {
            "count",
            "dimensions",
            "languages",
            "data_types",
            "context_lengths",
            "entity_types",
            "difficulty_levels",
            "adversarial_stats",
            "reidentification_risk",
            "edge_cases",
            "quasi_identifiers",
            "temporal_records",
            "token_statistics",
            "entity_statistics",
        }
        assert set(stats.keys()) == required_keys

    def test_dimension_statistics(self):
        """Test dimension statistics computation."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              dimension_tags=["entity_tracking"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              dimension_tags=["entity_tracking", "multilingual"]),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["dimensions"]["entity_tracking"]["count"] == 2
        assert stats["dimensions"]["multilingual"]["count"] == 1
        assert stats["dimensions"]["entity_tracking"]["percentage"] == 100.0
        assert stats["dimensions"]["multilingual"]["percentage"] == 50.0

    def test_language_statistics(self):
        """Test language statistics computation."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[], language="es"),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["languages"]["en"]["count"] == 2
        assert stats["languages"]["en"]["percentage"] == 66.7
        assert stats["languages"]["es"]["count"] == 1
        assert stats["languages"]["es"]["percentage"] == 33.3

    def test_entity_type_statistics(self):
        """Test entity type statistics computation."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              entity_types_present=["PERSON"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              entity_types_present=["PERSON", "EMAIL"]),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["entity_types"]["PERSON"] == 2
        assert stats["entity_types"]["EMAIL"] == 1

    def test_data_type_statistics(self):
        """Test data type statistics computation."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              data_type="unstructured_text"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              data_type="structured"),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["data_types"]["unstructured_text"] == 1
        assert stats["data_types"]["structured"] == 1

    def test_adversarial_statistics(self):
        """Test adversarial statistics computation."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              adversarial_type=None),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              adversarial_type="typos"),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["adversarial_stats"]["clean"] == 1
        assert stats["adversarial_stats"]["adversarial"] == 1
        assert stats["adversarial_stats"]["percentage_adversarial"] == 50.0

    def test_token_statistics_with_valid_tokens(self):
        """Test token statistics computation."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], token_count=10),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], token_count=20),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[], token_count=30),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["token_statistics"]["total"] == 60
        assert stats["token_statistics"]["count"] == 3
        assert stats["token_statistics"]["mean"] == 20.0
        assert stats["token_statistics"]["median"] == 20.0
        assert stats["token_statistics"]["min"] == 10
        assert stats["token_statistics"]["max"] == 30

    def test_token_statistics_empty(self):
        """Test token statistics with no valid tokens."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], token_count=0),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], token_count=0),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["token_statistics"]["total"] == 0
        assert stats["token_statistics"]["count"] == 0

    def test_entity_statistics(self):
        """Test entity statistics computation."""
        records = [
            EvalBenchmarkRecord(
                record_id="1",
                text="Test",
                labels=[
                    {"entity_id": "e1", "entity_cluster_id": "c1"},
                    {"entity_id": "e2", "entity_cluster_id": "c1"},
                ],
            ),
            EvalBenchmarkRecord(
                record_id="2",
                text="Test",
                labels=[
                    {"entity_id": "e3", "entity_cluster_id": "c2"},
                ],
            ),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["entity_statistics"]["total_entities"] == 3
        assert stats["entity_statistics"]["unique_clusters"] == 2
        assert stats["entity_statistics"]["max_cluster_size"] == 2

    def test_quasi_identifier_statistics(self):
        """Test quasi-identifier statistics (top 10)."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              quasi_identifiers_present=["name", "email"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              quasi_identifiers_present=["name", "email"]),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              quasi_identifiers_present=["phone"]),
        ]

        stats = compute_dataset_statistics(records)

        assert len(stats["quasi_identifiers"]) > 0
        # Should have quasi-identifiers listed
        qis = [q["type"] for q in stats["quasi_identifiers"]]
        assert any("email" in qi or "name" in qi for qi in qis)

    def test_temporal_record_counting(self):
        """Test temporal record counting."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              is_time_series=True),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              is_time_series=False),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              is_time_series=True),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["temporal_records"] == 2

    def test_difficulty_level_statistics(self):
        """Test difficulty level statistics."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              difficulty_level="easy"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              difficulty_level="hard"),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["difficulty_levels"]["easy"] == 1
        assert stats["difficulty_levels"]["hard"] == 1

    def test_context_length_statistics(self):
        """Test context length tier statistics."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              context_length_tier="short"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              context_length_tier="long"),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["context_lengths"]["short"] == 1
        assert stats["context_lengths"]["long"] == 1

    def test_reidentification_risk_statistics(self):
        """Test re-identification risk statistics."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              reidentification_risk_tier="low"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              reidentification_risk_tier="high"),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["reidentification_risk"]["low"] == 1
        assert stats["reidentification_risk"]["high"] == 1

    def test_edge_cases_statistics(self):
        """Test edge case statistics."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              edge_case_types=["abbreviation", "ambiguous"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              edge_case_types=["abbreviation"]),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["edge_cases"]["abbreviation"] == 2
        assert stats["edge_cases"]["ambiguous"] == 1

    def test_percentage_calculations(self):
        """Test that percentages are calculated correctly."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              dimension_tags=["entity_tracking"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              dimension_tags=["entity_tracking"]),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              dimension_tags=["multilingual"]),
        ]

        stats = compute_dataset_statistics(records)

        assert stats["dimensions"]["entity_tracking"]["percentage"] == 66.7
        assert stats["dimensions"]["multilingual"]["percentage"] == 33.3
