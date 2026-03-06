"""Tests for pii_anon.eval_framework.research.references module.

Validates:
- All 18 research references are present and well-formed
- Evidence registry maps features to supporting research
- Every framework component has at least one evidence citation
"""

from __future__ import annotations


from pii_anon.eval_framework.research.references import (
    EVIDENCE_REGISTRY,
    ResearchReference,
    all_references,
    get_references_for,
)


class TestResearchReferences:
    """Research reference registry validation."""

    def test_at_least_18_references(self) -> None:
        refs = all_references()
        assert len(refs) >= 18

    def test_all_references_have_required_fields(self) -> None:
        for ref in all_references():
            assert isinstance(ref, ResearchReference)
            assert ref.key != ""
            assert ref.title != ""
            assert ref.authors != ""
            assert ref.year >= 1952  # Bradley-Terry (1952) is the earliest reference
            assert ref.venue != ""
            assert ref.doi_or_url != ""
            assert ref.relevance != ""

    def test_key_references_present(self) -> None:
        refs = {r.key for r in all_references()}
        expected_keys = {
            "semeval13", "nervaluate", "seqeval", "openner10", "tab2022",
            "ratbench2025", "nist800122", "gdpr_art4", "iso27701",
            "i2b2_2014", "kanonymity", "ldiversity", "tcloseness",
        }
        for key in expected_keys:
            assert key in refs, f"Missing reference: {key}"

    def test_references_sorted_by_key(self) -> None:
        refs = all_references()
        keys = [r.key for r in refs]
        assert keys == sorted(keys)


class TestEvidenceRegistry:
    """Evidence registry: maps features to supporting research."""

    def test_registry_has_entries(self) -> None:
        assert len(EVIDENCE_REGISTRY) > 0

    def test_all_features_have_evidence(self) -> None:
        expected_features = {
            "span_metrics", "token_level_metrics", "privacy_metrics",
            "utility_metrics", "fairness_metrics", "taxonomy",
            "language_support", "context_evaluation", "dataset",
            "reidentification_risk", "standards_compliance",
        }
        for feature in expected_features:
            refs = get_references_for(feature)
            assert len(refs) > 0, f"No evidence for feature: {feature}"

    def test_get_references_for_unknown(self) -> None:
        refs = get_references_for("nonexistent_feature")
        assert refs == []

    def test_all_registry_values_are_reference_lists(self) -> None:
        for feature, refs in EVIDENCE_REGISTRY.items():
            assert isinstance(refs, list), f"{feature} has non-list value"
            for ref in refs:
                assert isinstance(ref, ResearchReference), (
                    f"{feature} contains non-ResearchReference: {type(ref)}"
                )
