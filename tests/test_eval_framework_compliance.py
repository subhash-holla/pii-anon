"""Tests for pii_anon.eval_framework.standards.compliance module.

Validates:
- NIST SP 800-122, GDPR, ISO 27701, HIPAA, CCPA compliance checking
- Coverage gap analysis and remediation planning
- Multi-standard validation
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.standards.compliance import (
    ComplianceValidator,
    CoverageGap,
    MultiStandardComplianceReport,
    validate_all_standards,
    validate_compliance,
)
from pii_anon.eval_framework.taxonomy import PII_TAXONOMY


class TestComplianceValidator:
    """ComplianceValidator core tests."""

    def setup_method(self) -> None:
        self.validator = ComplianceValidator()
        self.all_types = PII_TAXONOMY.all_types()

    # ── Full coverage ─────────────────────────────────────────────────

    def test_full_nist_coverage(self) -> None:
        report = self.validator.validate(self.all_types, standard="nist")
        assert report.compliant is True
        assert report.coverage_ratio == 1.0
        assert len(report.missing_types) == 0
        assert len(report.gaps) == 0

    def test_full_gdpr_coverage(self) -> None:
        report = self.validator.validate(self.all_types, standard="gdpr")
        assert report.compliant is True
        assert report.coverage_ratio == 1.0

    def test_full_iso27701_coverage(self) -> None:
        report = self.validator.validate(self.all_types, standard="iso27701")
        assert report.compliant is True

    def test_full_hipaa_coverage(self) -> None:
        report = self.validator.validate(self.all_types, standard="hipaa")
        assert report.compliant is True

    def test_full_ccpa_coverage(self) -> None:
        report = self.validator.validate(self.all_types, standard="ccpa")
        assert report.compliant is True

    # ── Partial coverage ──────────────────────────────────────────────

    def test_partial_nist_coverage(self) -> None:
        """Only provide 3 entity types — should be non-compliant."""
        report = self.validator.validate(
            ["PERSON_NAME", "EMAIL_ADDRESS", "US_SSN"],
            standard="nist",
        )
        assert report.compliant is False
        assert 0.0 < report.coverage_ratio < 1.0
        assert len(report.missing_types) > 0
        assert len(report.gaps) > 0

    def test_empty_entity_types(self) -> None:
        report = self.validator.validate([], standard="nist")
        assert report.compliant is False
        assert report.coverage_ratio == 0.0

    # ── Gap analysis ──────────────────────────────────────────────────

    def test_gaps_have_remediation_hints(self) -> None:
        report = self.validator.validate(["PERSON_NAME"], standard="gdpr")
        assert len(report.gaps) > 0
        for gap in report.gaps:
            assert isinstance(gap, CoverageGap)
            assert gap.remediation_hint != ""
            assert gap.risk_level != ""
            assert gap.category != ""

    def test_risk_summary_populated(self) -> None:
        report = self.validator.validate(["PERSON_NAME"], standard="nist")
        assert len(report.risk_summary) > 0
        total_missing = sum(report.risk_summary.values())
        assert total_missing == len(report.missing_types)

    # ── Standard aliases ──────────────────────────────────────────────

    def test_standard_alias_nist(self) -> None:
        r1 = self.validator.validate(self.all_types, standard="nist")
        r2 = self.validator.validate(self.all_types, standard="nist_sp_800_122")
        assert r1.standard == r2.standard

    def test_unknown_standard_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown standard"):
            self.validator.validate(self.all_types, standard="made_up_standard")

    # ── Report serialisation ──────────────────────────────────────────

    def test_to_dict(self) -> None:
        report = self.validator.validate(["PERSON_NAME"], standard="nist")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "standard" in d
        assert "coverage_ratio" in d
        assert "gaps" in d
        assert isinstance(d["gaps"], list)
        assert isinstance(d["required_types"], list)


class TestMultiStandardValidation:
    """Multi-standard compliance validation."""

    def setup_method(self) -> None:
        self.validator = ComplianceValidator()

    def test_validate_all_with_full_coverage(self) -> None:
        multi = self.validator.validate_all(PII_TAXONOMY.all_types())
        assert isinstance(multi, MultiStandardComplianceReport)
        assert multi.overall_coverage_ratio == 1.0
        assert len(multi.fully_compliant_standards) == 5
        assert len(multi.non_compliant_standards) == 0

    def test_validate_all_with_partial_coverage(self) -> None:
        multi = self.validator.validate_all(["PERSON_NAME", "EMAIL_ADDRESS"])
        assert multi.overall_coverage_ratio < 1.0
        assert len(multi.non_compliant_standards) > 0
        assert len(multi.all_missing_types) > 0

    def test_validate_specific_standards(self) -> None:
        multi = self.validator.validate_all(
            PII_TAXONOMY.all_types(),
            standards=["nist", "gdpr"],
        )
        assert len(multi.reports) == 2

    def test_multi_standard_to_dict(self) -> None:
        multi = self.validator.validate_all(["PERSON_NAME"])
        d = multi.to_dict()
        assert "overall_coverage_ratio" in d
        assert "reports" in d
        assert len(d["reports"]) == 5  # all 5 standards


class TestCategoryGapAnalysis:
    """Per-category coverage analysis."""

    def test_category_gap_analysis(self) -> None:
        validator = ComplianceValidator()
        analysis = validator.category_gap_analysis(["PERSON_NAME"], standard="nist")
        assert isinstance(analysis, dict)
        assert "personal_identity" in analysis
        assert analysis["personal_identity"]["covered"] >= 1


class TestRemediationPlan:
    """Risk-prioritised remediation planning."""

    def test_remediation_plan_ordered_by_risk(self) -> None:
        validator = ComplianceValidator()
        plan = validator.remediation_plan(["PERSON_NAME"], standard="nist")
        assert len(plan) > 0
        # Should be ordered: priority 1 (critical) first
        priorities = [int(item["priority"]) for item in plan]
        assert priorities == sorted(priorities)

    def test_remediation_plan_has_hints(self) -> None:
        validator = ComplianceValidator()
        plan = validator.remediation_plan(["PERSON_NAME"], standard="gdpr")
        for item in plan:
            assert "hint" in item
            assert item["hint"] != ""


class TestConvenienceFunctions:
    """Module-level convenience functions."""

    def test_validate_compliance(self) -> None:
        report = validate_compliance(PII_TAXONOMY.all_types(), standard="nist")
        assert report.compliant is True

    def test_validate_all_standards(self) -> None:
        multi = validate_all_standards(PII_TAXONOMY.all_types())
        assert multi.overall_coverage_ratio == 1.0
