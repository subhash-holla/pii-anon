"""Integration tests for the pii-anon evaluation framework.

End-to-end tests that validate:
- Full import chain from pii_anon top-level
- Framework instantiation and evaluation pipeline
- Version 1.0.0
- Backward compatibility with existing API
"""

from __future__ import annotations



class TestTopLevelImports:
    """Verify the new evaluation framework is accessible from pii_anon."""

    def test_version_is_1_3_0(self) -> None:
        import pii_anon
        assert pii_anon.__version__ == "1.0.0"

    def test_eval_framework_imports(self) -> None:
        from pii_anon import (
            PII_TAXONOMY,
            SUPPORTED_LANGUAGES,
            EntityTypeRegistry,
        )
        assert PII_TAXONOMY.all_types()
        assert SUPPORTED_LANGUAGES.count() >= 52
        assert len(EntityTypeRegistry()) == 48

    def test_backward_compat_existing_exports(self) -> None:
        """Existing v1.x exports must still work."""
        from pii_anon import (
            PIIOrchestrator,
            AsyncPIIOrchestrator,
        )
        assert PIIOrchestrator is not None
        assert AsyncPIIOrchestrator is not None


class TestEndToEndEvaluation:
    """Full pipeline: create spans → evaluate → report."""

    def test_single_record_evaluation(self) -> None:
        from pii_anon.eval_framework import (
            EvaluationFramework,
            LabeledSpan,
        )

        fw = EvaluationFramework()
        predictions = [
            LabeledSpan("PERSON_NAME", 0, 10, "r1"),
            LabeledSpan("EMAIL_ADDRESS", 15, 35, "r1"),
        ]
        labels = [
            LabeledSpan("PERSON_NAME", 0, 10, "r1"),
            LabeledSpan("EMAIL_ADDRESS", 15, 35, "r1"),
            LabeledSpan("PHONE_NUMBER", 40, 55, "r1"),
        ]
        report = fw.evaluate(predictions, labels, language="en")
        # 2 out of 3 labels matched → recall < 1.0
        assert report.precision == 1.0
        assert report.recall < 1.0
        assert 0.0 < report.f1 < 1.0

    def test_compliance_roundtrip(self) -> None:
        from pii_anon.eval_framework import (
            ComplianceValidator,
            PII_TAXONOMY,
        )

        validator = ComplianceValidator()
        report = validator.validate(PII_TAXONOMY.all_types(), standard="gdpr")
        assert report.compliant is True
        d = report.to_dict()
        assert d["compliant"] is True
        assert d["coverage_ratio"] == 1.0

    def test_multi_standard_compliance(self) -> None:
        from pii_anon.eval_framework import (
            validate_all_standards,
            PII_TAXONOMY,
        )

        multi = validate_all_standards(PII_TAXONOMY.all_types())
        assert multi.overall_coverage_ratio == 1.0
        assert len(multi.fully_compliant_standards) == 5

    def test_research_evidence_chain(self) -> None:
        from pii_anon.eval_framework import (
            all_references,
            get_references_for,
            EVIDENCE_REGISTRY,
        )

        assert len(all_references()) >= 18
        # Every listed feature has backing evidence
        for feature in EVIDENCE_REGISTRY:
            refs = get_references_for(feature)
            assert len(refs) > 0, f"No evidence for {feature}"
