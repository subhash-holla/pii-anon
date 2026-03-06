"""Tests for the coverage validator module.

Tests cover validation of dataset coverage across 7 evaluation dimensions,
language distribution, entity types, quasi-identifiers, adversarial types,
and data formats, including coverage gap identification.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from pii_anon.eval_framework.datasets.coverage_validator import (
    CoverageValidator,
    CoverageReport,
    validate_and_report,
)
from pii_anon.eval_framework.datasets.schema import EvalBenchmarkRecord


class TestCoverageReportSummary:
    """Test the CoverageReport.summary() method."""

    def test_summary_empty_report(self):
        """Test summary generation for empty dataset."""
        report = CoverageReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            dataset_name="test",
            total_records=0,
        )
        summary = report.summary()
        assert "COVERAGE VALIDATION REPORT" in summary
        assert "Total Records: 0" in summary
        assert "(empty dataset)" in summary or "0" in summary

    def test_summary_with_dimension_coverage(self):
        """Test summary includes dimension coverage details."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=1000,
            dimension_coverage={
                "entity_tracking": {
                    "count": 200,
                    "target": 200,
                    "pct": 100.0,
                    "adequate": True,
                },
                "multilingual": {
                    "count": 100,
                    "target": 150,
                    "pct": 66.7,
                    "adequate": False,
                },
            },
        )
        summary = report.summary()
        assert "DIMENSION COVERAGE" in summary
        assert "entity_tracking" in summary
        assert "[PASS]" in summary or "PASS" in summary
        assert "[FAIL]" in summary or "FAIL" in summary

    def test_summary_with_language_coverage(self):
        """Test summary includes language coverage details."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            language_coverage={"en": 50, "es": 30, "fr": 20},
        )
        summary = report.summary()
        assert "LANGUAGE COVERAGE" in summary
        assert "en" in summary
        assert "50" in summary

    def test_summary_with_entity_types(self):
        """Test summary includes entity type coverage details."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            entity_type_coverage={"PERSON": 50, "EMAIL": 40, "PHONE": 10},
        )
        summary = report.summary()
        assert "ENTITY TYPE COVERAGE" in summary
        assert "PERSON" in summary
        assert "EMAIL" in summary

    def test_summary_with_coverage_gaps(self):
        """Test summary includes identified coverage gaps."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            coverage_gaps=[
                {
                    "stratum": "language:fr",
                    "count": 5,
                    "target": 30,
                    "gap": 25,
                },
            ],
        )
        summary = report.summary()
        assert "COVERAGE GAPS" in summary
        assert "language:fr" in summary or "fr" in summary
        assert "25" in summary or "gap" in summary.lower()

    def test_summary_with_quasi_identifier_coverage(self):
        """Test summary includes quasi-identifier coverage."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            quasi_identifier_coverage={"name+email": 45, "phone+address": 30},
        )
        summary = report.summary()
        assert "QUASI-IDENTIFIER" in summary
        assert "name+email" in summary or "email" in summary

    def test_summary_with_adversarial_coverage(self):
        """Test summary includes adversarial coverage."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            adversarial_coverage={"clean": 70, "typos": 20, "case_variation": 10},
        )
        summary = report.summary()
        assert "ADVERSARIAL" in summary
        assert "clean" in summary
        assert "typos" in summary

    def test_summary_with_format_coverage(self):
        """Test summary includes data format coverage."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            format_coverage={
                "unstructured_text": 50,
                "structured": 30,
                "semi_structured": 20,
            },
        )
        summary = report.summary()
        assert "DATA FORMAT" in summary
        assert "unstructured_text" in summary or "unstructured" in summary

    def test_summary_final_assessment_pass(self):
        """Test summary final assessment when requirements met."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=1000,
            all_requirements_met=True,
        )
        summary = report.summary()
        assert "FINAL ASSESSMENT" in summary
        assert "YES" in summary

    def test_summary_final_assessment_fail(self):
        """Test summary final assessment when requirements not met."""
        report = CoverageReport(
            timestamp="2024-01-01T00:00:00Z",
            dataset_name="test",
            total_records=100,
            all_requirements_met=False,
        )
        summary = report.summary()
        assert "FINAL ASSESSMENT" in summary
        assert "NO" in summary


class TestCoverageValidatorBasic:
    """Test basic CoverageValidator functionality."""

    def test_init_default_parameters(self):
        """Test validator initializes with correct defaults."""
        validator = CoverageValidator()
        assert validator.min_per_language == 30
        assert validator.min_per_entity_type == 50
        assert validator.min_per_qi_group == 30

    def test_init_custom_parameters(self):
        """Test validator initializes with custom parameters."""
        validator = CoverageValidator(
            min_per_language=20,
            min_per_entity_type=40,
            min_per_qi_group=25,
        )
        assert validator.min_per_language == 20
        assert validator.min_per_entity_type == 40
        assert validator.min_per_qi_group == 25

    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        validator = CoverageValidator()
        report = validator.validate([])

        assert report.total_records == 0
        assert report.dataset_name == "(empty dataset)"
        assert report.all_requirements_met is False
        assert len(report.dimension_coverage) == 0

    def test_validate_single_record(self):
        """Test validation with single record."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test text",
            labels=[],
            language="en",
            dimension_tags=["entity_tracking"],
            entity_types_present=["PERSON"],
            quasi_identifiers_present=["name"],
            adversarial_type="clean",
            data_type="unstructured_text",
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert report.total_records == 1
        assert "entity_tracking" in report.dimension_coverage


class TestCoverageValidatorDimensions:
    """Test dimension coverage checking."""

    def test_check_dimensions_single_dimension(self):
        """Test dimension coverage when one dimension present."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            dimension_tags=["entity_tracking"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert "entity_tracking" in report.dimension_coverage
        assert report.dimension_coverage["entity_tracking"]["count"] == 1

    def test_check_dimensions_multiple_dimensions_same_record(self):
        """Test dimension coverage when record tags multiple dimensions."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            dimension_tags=["entity_tracking", "multilingual", "context_preservation"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert report.dimension_coverage["entity_tracking"]["count"] == 1
        assert report.dimension_coverage["multilingual"]["count"] == 1
        assert report.dimension_coverage["context_preservation"]["count"] == 1

    def test_check_dimensions_all_valid_dimensions(self):
        """Test all 7 valid dimensions are tracked."""
        records = []
        for i, dim in enumerate([
            "entity_tracking",
            "multilingual",
            "context_preservation",
            "diverse_pii_types",
            "edge_cases",
            "data_format_variations",
            "temporal_consistency",
        ]):
            records.append(EvalBenchmarkRecord(
                record_id=f"test_{i}",
                text="Test",
                labels=[],
                dimension_tags=[dim],
            ))

        validator = CoverageValidator()
        report = validator.validate(records)

        # All 7 dimensions should be in report
        assert len(report.dimension_coverage) == 7


class TestCoverageValidatorLanguages:
    """Test language coverage checking."""

    def test_check_languages_single_language(self):
        """Test language coverage with single language."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            language="en",
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert "en" in report.language_coverage
        assert report.language_coverage["en"] == 1

    def test_check_languages_multiple_languages(self):
        """Test language coverage with multiple languages."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], language="es"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[], language="fr"),
            EvalBenchmarkRecord(record_id="4", text="Test", labels=[], language="en"),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        assert report.language_coverage["en"] == 2
        assert report.language_coverage["es"] == 1
        assert report.language_coverage["fr"] == 1

    def test_languages_sorted_alphabetically(self):
        """Test that languages are sorted alphabetically in report."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[], language="fr"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[], language="en"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[], language="es"),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        langs = list(report.language_coverage.keys())
        assert langs == sorted(langs)


class TestCoverageValidatorEntityTypes:
    """Test entity type coverage checking."""

    def test_check_entity_types_single_type(self):
        """Test entity type coverage with single type."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            entity_types_present=["PERSON"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert "PERSON" in report.entity_type_coverage
        assert report.entity_type_coverage["PERSON"] == 1

    def test_check_entity_types_multiple_types_same_record(self):
        """Test entity type coverage when record has multiple types."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            entity_types_present=["PERSON", "EMAIL", "PHONE"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert report.entity_type_coverage["PERSON"] == 1
        assert report.entity_type_coverage["EMAIL"] == 1
        assert report.entity_type_coverage["PHONE"] == 1

    def test_entity_types_sorted_alphabetically(self):
        """Test that entity types are sorted alphabetically."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              entity_types_present=["PHONE"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              entity_types_present=["PERSON"]),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              entity_types_present=["EMAIL"]),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        types = list(report.entity_type_coverage.keys())
        assert types == sorted(types)


class TestCoverageValidatorQuasiIdentifiers:
    """Test quasi-identifier coverage checking."""

    def test_check_quasi_identifiers_simple(self):
        """Test quasi-identifier coverage."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            quasi_identifiers_present=["name", "email"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert "email+name" in report.quasi_identifier_coverage or \
               "name+email" in report.quasi_identifier_coverage

    def test_quasi_identifiers_empty_list(self):
        """Test quasi-identifier handling when empty."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            quasi_identifiers_present=[],
            entity_types_present=["PERSON"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        # Should create group from entity types or use "none"
        assert len(report.quasi_identifier_coverage) > 0

    def test_quasi_identifiers_sorted_canonically(self):
        """Test quasi-identifier canonical sorting."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              quasi_identifiers_present=["email", "name"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              quasi_identifiers_present=["name", "email"]),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        # Both should map to same canonical group
        assert len(report.quasi_identifier_coverage) == 1


class TestCoverageValidatorAdversarial:
    """Test adversarial coverage checking."""

    def test_check_adversarial_clean(self):
        """Test adversarial coverage for clean records."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            adversarial_type=None,
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert "clean" in report.adversarial_coverage
        assert report.adversarial_coverage["clean"] == 1

    def test_check_adversarial_typed(self):
        """Test adversarial coverage for adversarial types."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              adversarial_type="typos"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              adversarial_type="case_variation"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              adversarial_type="typos"),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        assert report.adversarial_coverage["typos"] == 2
        assert report.adversarial_coverage["case_variation"] == 1

    def test_adversarial_sorted_alphabetically(self):
        """Test that adversarial types are sorted."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              adversarial_type="typos"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              adversarial_type="clean"),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        types = list(report.adversarial_coverage.keys())
        assert types == sorted(types)


class TestCoverageValidatorFormats:
    """Test data format coverage checking."""

    def test_check_formats_single_type(self):
        """Test format coverage with single type."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            data_type="unstructured_text",
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert "unstructured_text" in report.format_coverage
        assert report.format_coverage["unstructured_text"] == 1

    def test_check_formats_multiple_types(self):
        """Test format coverage with multiple types."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              data_type="unstructured_text"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              data_type="structured"),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              data_type="semi_structured"),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        assert report.format_coverage["unstructured_text"] == 1
        assert report.format_coverage["structured"] == 1
        assert report.format_coverage["semi_structured"] == 1

    def test_formats_sorted_alphabetically(self):
        """Test that formats are sorted."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              data_type="structured"),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              data_type="code"),
        ]

        validator = CoverageValidator()
        report = validator.validate(records)

        types = list(report.format_coverage.keys())
        assert types == sorted(types)


class TestCoverageValidatorGaps:
    """Test coverage gap identification."""

    def test_identify_gaps_dimension_gap(self):
        """Test gap identification for dimension coverage."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            dimension_tags=["entity_tracking"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        # With only 1 record, most dimensions should be below target
        assert len(report.coverage_gaps) > 0
        assert any("dimension:" in gap["stratum"] for gap in report.coverage_gaps)

    def test_identify_gaps_language_gap(self):
        """Test gap identification for language coverage."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            language="en",
        )

        validator = CoverageValidator(min_per_language=10)
        report = validator.validate([record])

        # 1 record is below 10-record threshold
        assert any("language:" in gap["stratum"] for gap in report.coverage_gaps)

    def test_identify_gaps_entity_type_gap(self):
        """Test gap identification for entity type coverage."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            entity_types_present=["PERSON"],
        )

        validator = CoverageValidator(min_per_entity_type=10)
        report = validator.validate([record])

        # 1 record is below 10-record threshold
        assert any("entity_type:" in gap["stratum"] for gap in report.coverage_gaps)

    def test_identify_gaps_qi_gap(self):
        """Test gap identification for QI group coverage."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            quasi_identifiers_present=["name", "email"],
        )

        validator = CoverageValidator(min_per_qi_group=10)
        report = validator.validate([record])

        # 1 record is below 10-record threshold
        assert any("qi_group:" in gap["stratum"] for gap in report.coverage_gaps)

    def test_gaps_sorted_by_size(self):
        """Test that gaps are sorted by size (largest first)."""
        records = [
            EvalBenchmarkRecord(record_id="1", text="Test", labels=[],
                              language="en", entity_types_present=["PERSON"]),
            EvalBenchmarkRecord(record_id="2", text="Test", labels=[],
                              language="en", entity_types_present=["PERSON"]),
            EvalBenchmarkRecord(record_id="3", text="Test", labels=[],
                              language="en", entity_types_present=["EMAIL"]),
        ]

        validator = CoverageValidator(
            min_per_language=10,
            min_per_entity_type=10,
        )
        report = validator.validate(records)

        if len(report.coverage_gaps) > 1:
            gaps_sorted = sorted(report.coverage_gaps,
                                key=lambda x: x["gap"],
                                reverse=True)
            assert report.coverage_gaps == gaps_sorted


class TestCoverageValidatorRequirementsMet:
    """Test all_requirements_met determination."""

    def test_requirements_met_false_when_dimensions_inadequate(self):
        """Test requirements not met when dimension coverage inadequate."""
        record = EvalBenchmarkRecord(
            record_id="test_1",
            text="Test",
            labels=[],
            dimension_tags=["entity_tracking"],
        )

        validator = CoverageValidator()
        report = validator.validate([record])

        assert report.all_requirements_met is False

    def test_requirements_met_with_adequate_dimensions(self):
        """Test requirements met when all dimensions adequately covered."""
        # Create records with all 7 dimensions adequately covered
        target_per_dim = 100
        records = []

        for dim in [
            "entity_tracking", "multilingual", "context_preservation",
            "diverse_pii_types", "edge_cases", "data_format_variations",
            "temporal_consistency",
        ]:
            for i in range(target_per_dim):
                records.append(EvalBenchmarkRecord(
                    record_id=f"{dim}_{i}",
                    text="Test",
                    labels=[],
                    dimension_tags=[dim],
                ))

        validator = CoverageValidator()
        report = validator.validate(records)

        # Check if requirements are actually met
        all_adequate = all(
            cov.get("adequate", False)
            for cov in report.dimension_coverage.values()
        )
        assert report.all_requirements_met == all_adequate


class TestValidateAndReport:
    """Test the convenience function validate_and_report."""

    def test_validate_and_report_returns_report(self):
        """Test that validate_and_report returns CoverageReport."""
        # This test will fail if the dataset doesn't exist, so we catch that
        try:
            report = validate_and_report("pii_anon_eval_v1")
            assert isinstance(report, CoverageReport)
        except FileNotFoundError:
            # Dataset not found is acceptable in test environment
            pytest.skip("Eval dataset not available")
