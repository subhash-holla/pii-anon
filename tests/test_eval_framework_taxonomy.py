"""Tests for pii_anon.eval_framework.taxonomy module.

Validates:
- 44 entity types across 7 categories
- Regulatory reference mappings (NIST, GDPR, ISO 27701)
- EntityTypeRegistry queries and extension
"""

from __future__ import annotations


from pii_anon.eval_framework.taxonomy import (
    EntityCategory,
    EntityTypeProfile,
    EntityTypeRegistry,
    PII_TAXONOMY,
    RiskLevel,
    RegulatoryReference,
)


class TestPIITaxonomy:
    """PII_TAXONOMY namespace validation."""

    def test_total_entity_types_is_44(self) -> None:
        """NIST+GDPR+ISO union + quasi-ID types yields exactly 48 canonical types."""
        assert len(PII_TAXONOMY.all_types()) == 48

    def test_all_types_are_uppercase_strings(self) -> None:
        for t in PII_TAXONOMY.all_types():
            assert isinstance(t, str)
            assert t == t.upper()

    def test_all_types_are_unique(self) -> None:
        types = PII_TAXONOMY.all_types()
        assert len(types) == len(set(types))

    def test_known_types_present(self) -> None:
        types = set(PII_TAXONOMY.all_types())
        for expected in [
            "PERSON_NAME", "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD_NUMBER",
            "IBAN", "PASSPORT_NUMBER", "MEDICAL_RECORD_NUMBER", "IP_ADDRESS",
            "API_KEY", "BIOMETRIC_ID", "ETHNIC_ORIGIN", "POLITICAL_OPINION",
        ]:
            assert expected in types, f"{expected} missing from taxonomy"

    def test_types_for_category(self) -> None:
        personal = PII_TAXONOMY.types_for_category(EntityCategory.PERSONAL_IDENTITY)
        assert len(personal) == 10
        assert "PERSON_NAME" in personal

        financial = PII_TAXONOMY.types_for_category(EntityCategory.FINANCIAL)
        assert len(financial) == 7

        medical = PII_TAXONOMY.types_for_category(EntityCategory.MEDICAL)
        assert len(medical) == 5


class TestEntityCategory:
    """EntityCategory enum coverage."""

    def test_seven_categories(self) -> None:
        assert len(EntityCategory) == 7

    def test_category_values(self) -> None:
        expected = {
            "personal_identity", "financial", "government_id",
            "medical", "digital_technical", "employment",
            "behavioral_contextual",
        }
        assert {c.value for c in EntityCategory} == expected


class TestRiskLevel:
    """RiskLevel enum coverage."""

    def test_four_levels(self) -> None:
        assert len(RiskLevel) == 4

    def test_ordering_concept(self) -> None:
        """Levels go LOW -> MODERATE -> HIGH -> CRITICAL."""
        levels = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert [level.value for level in levels] == ["low", "moderate", "high", "critical"]


class TestEntityTypeRegistry:
    """EntityTypeRegistry query and extension tests."""

    def setup_method(self) -> None:
        self.registry = EntityTypeRegistry()

    def test_registry_has_44_entries(self) -> None:
        assert len(self.registry) == 48

    def test_get_known_type(self) -> None:
        profile = self.registry.get("PERSON_NAME")
        assert profile is not None
        assert profile.entity_type == "PERSON_NAME"
        assert profile.category == EntityCategory.PERSONAL_IDENTITY
        assert profile.risk_level == RiskLevel.HIGH

    def test_get_unknown_type_returns_none(self) -> None:
        assert self.registry.get("NONEXISTENT") is None

    def test_contains_operator(self) -> None:
        assert "EMAIL_ADDRESS" in self.registry
        assert "FAKE_TYPE" not in self.registry

    def test_all_profiles_sorted(self) -> None:
        profiles = self.registry.all_profiles()
        assert len(profiles) == 48
        names = [p.entity_type for p in profiles]
        assert names == sorted(names)

    def test_by_category(self) -> None:
        financial = self.registry.by_category(EntityCategory.FINANCIAL)
        assert len(financial) == 7
        assert all(p.category == EntityCategory.FINANCIAL for p in financial)

    def test_by_risk_level(self) -> None:
        critical = self.registry.by_risk_level(RiskLevel.CRITICAL)
        assert len(critical) >= 5  # US_SSN, PASSPORT, NATIONAL_ID, etc.
        assert all(p.risk_level == RiskLevel.CRITICAL for p in critical)

    def test_by_standard_nist(self) -> None:
        nist_types = self.registry.by_standard("NIST")
        assert len(nist_types) > 0
        for p in nist_types:
            assert any("nist" in ref.standard.lower() for ref in p.regulatory_refs)

    def test_by_standard_gdpr(self) -> None:
        gdpr_types = self.registry.by_standard("GDPR")
        assert len(gdpr_types) > 0

    def test_register_custom_type(self) -> None:
        custom = EntityTypeProfile(
            entity_type="CUSTOM_PII",
            category=EntityCategory.PERSONAL_IDENTITY,
            risk_level=RiskLevel.LOW,
            description="A custom PII type for testing.",
        )
        self.registry.register(custom)
        assert len(self.registry) == 49
        assert "CUSTOM_PII" in self.registry
        retrieved = self.registry.get("CUSTOM_PII")
        assert retrieved is not None
        assert retrieved.description == "A custom PII type for testing."


class TestRegulatoryReferences:
    """Verify every entity type has at least one regulatory reference."""

    def test_all_types_have_regulatory_refs(self) -> None:
        registry = EntityTypeRegistry()
        for profile in registry.all_profiles():
            assert len(profile.regulatory_refs) > 0, (
                f"{profile.entity_type} has no regulatory references"
            )

    def test_regulatory_ref_fields(self) -> None:
        registry = EntityTypeRegistry()
        profile = registry.get("US_SSN")
        assert profile is not None
        ref = profile.regulatory_refs[0]
        assert isinstance(ref, RegulatoryReference)
        assert ref.standard != ""
        assert ref.section != ""
        assert ref.classification != ""
