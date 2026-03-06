"""Tests for the pattern registry system (PatternSpec and PATTERN_REGISTRY).

Covers: registry integrity, PatternSpec fields, validator references,
context types, confidence ranges, immutability, and registry size.
"""

from __future__ import annotations

from pii_anon.engines.regex.patterns import PATTERN_REGISTRY, PatternSpec
from pii_anon.engines.regex.confidence import CONTEXT_WORDS
from pii_anon.engines.regex import validators


class TestPatternRegistryStructure:
    """Test the PatternSpec dataclass and registry integrity."""

    def test_registry_is_tuple(self) -> None:
        """PATTERN_REGISTRY must be a tuple (immutable)."""
        assert isinstance(PATTERN_REGISTRY, tuple)

    def test_registry_has_minimum_patterns(self) -> None:
        """Registry should contain at least 50 patterns."""
        assert len(PATTERN_REGISTRY) >= 50

    def test_registry_contains_pattern_specs(self) -> None:
        """All registry entries must be PatternSpec instances."""
        for spec in PATTERN_REGISTRY:
            assert isinstance(spec, PatternSpec)

    def test_pattern_spec_is_frozen(self) -> None:
        """PatternSpec dataclass must be frozen (immutable)."""
        spec = PATTERN_REGISTRY[0]
        try:
            spec.base_confidence = 0.5
            assert False, "PatternSpec should be immutable"
        except (AttributeError, TypeError):
            pass  # Expected


class TestPatternSpecFields:
    """Test that all PatternSpec instances have valid required fields."""

    def test_all_have_entity_type(self) -> None:
        """Every PatternSpec must have a non-empty entity_type."""
        for spec in PATTERN_REGISTRY:
            assert spec.entity_type
            assert isinstance(spec.entity_type, str)
            assert len(spec.entity_type) > 0

    def test_all_have_pattern(self) -> None:
        """Every PatternSpec must have a compiled regex pattern."""
        for spec in PATTERN_REGISTRY:
            assert spec.pattern is not None
            assert hasattr(spec.pattern, "finditer")  # Regex pattern object

    def test_all_have_base_confidence(self) -> None:
        """Every PatternSpec must have a base_confidence."""
        for spec in PATTERN_REGISTRY:
            assert spec.base_confidence is not None

    def test_base_confidence_in_valid_range(self) -> None:
        """base_confidence must be between 0 and 1."""
        for spec in PATTERN_REGISTRY:
            assert 0 <= spec.base_confidence <= 1, (
                f"Invalid confidence {spec.base_confidence} "
                f"for {spec.entity_type}"
            )

    def test_group_is_non_negative_int(self) -> None:
        """The group field must be a non-negative integer."""
        for spec in PATTERN_REGISTRY:
            assert isinstance(spec.group, int)
            assert spec.group >= 0

    def test_explanation_is_string(self) -> None:
        """The explanation field must be a string (may be empty)."""
        for spec in PATTERN_REGISTRY:
            assert isinstance(spec.explanation, str)


class TestValidatorReferences:
    """Test that all validator names reference valid validators."""

    def test_all_validator_names_are_valid(self) -> None:
        """All validator names in specs must be in the _VALIDATORS dispatch table."""
        from pii_anon.engines.regex_adapter import _VALIDATORS

        for spec in PATTERN_REGISTRY:
            if spec.validator is None:
                continue
            assert spec.validator in _VALIDATORS, (
                f"Validator '{spec.validator}' for {spec.entity_type} "
                f"not found in _VALIDATORS dispatch table"
            )

    def test_npi_validator_exists(self) -> None:
        """NPI validator must be defined."""
        assert hasattr(validators, "is_valid_npi")

    def test_dea_validator_exists(self) -> None:
        """DEA validator must be defined."""
        assert hasattr(validators, "is_valid_dea_number")


class TestContextTypes:
    """Test that all context_type values are valid."""

    def test_all_context_types_valid(self) -> None:
        """All context_type values must be in CONTEXT_WORDS or None."""
        valid_types = set(CONTEXT_WORDS.keys())
        for spec in PATTERN_REGISTRY:
            if spec.context_type is not None:
                assert spec.context_type in valid_types, (
                    f"Unknown context_type '{spec.context_type}' "
                    f"for {spec.entity_type}"
                )


class TestEntityTypeGrouping:
    """Test entity type uniqueness and intentional grouping."""

    def test_multiple_patterns_allowed_for_type(self) -> None:
        """Some entity types (like PERSON_NAME) have multiple patterns."""
        entity_counts = {}
        for spec in PATTERN_REGISTRY:
            entity_counts[spec.entity_type] = entity_counts.get(spec.entity_type, 0) + 1

        # PERSON_NAME should have multiple patterns (title, full name, alias, etc.)
        assert entity_counts.get("PERSON_NAME", 0) >= 4
        # US_SSN should have multiple patterns (dash, space, nodash)
        assert entity_counts.get("US_SSN", 0) >= 3

    def test_us_ssn_multiple_patterns(self) -> None:
        """US_SSN must have at least 3 patterns (dash, space, nodash)."""
        ssn_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "US_SSN"]
        assert len(ssn_specs) >= 3

    def test_phone_number_multilingual(self) -> None:
        """PHONE_NUMBER patterns should exist for multiple languages."""
        phone_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "PHONE_NUMBER"]
        langs = {s.language for s in phone_specs if s.language}
        # Should have at least en, es, fr
        assert "en" in langs
        assert "es" in langs
        assert "fr" in langs

    def test_person_name_multilingual(self) -> None:
        """PERSON_NAME patterns should exist for multiple languages."""
        person_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "PERSON_NAME"]
        # At least some should be language-specific
        langs = {s.language for s in person_specs if s.language}
        assert len(langs) > 0


class TestNewEntityTypes:
    """Test that new entity types are present in the registry."""

    def test_ipv6_in_registry(self) -> None:
        """IPv6 pattern must be in registry."""
        ipv6_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "IP_ADDRESS"]
        assert any(s.pre_filter == ":" for s in ipv6_specs), "IPv6 pattern not found"

    def test_url_with_pii_in_registry(self) -> None:
        """URL_WITH_PII pattern must be in registry."""
        url_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "URL_WITH_PII"]
        assert len(url_specs) >= 1

    def test_age_in_registry(self) -> None:
        """AGE pattern must be in registry."""
        age_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "AGE"]
        assert len(age_specs) >= 1

    def test_date_time_in_registry(self) -> None:
        """DATE_TIME pattern must be in registry."""
        date_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "DATE_TIME"]
        assert len(date_specs) >= 1

    def test_medical_license_in_registry(self) -> None:
        """MEDICAL_LICENSE patterns (NPI, DEA) must be in registry."""
        medical_specs = [
            s for s in PATTERN_REGISTRY if s.entity_type == "MEDICAL_LICENSE"
        ]
        assert len(medical_specs) >= 2  # At least NPI and DEA


class TestValidConfidenceScores:
    """Test confidence score validity and consistency."""

    def test_valid_confidence_in_range(self) -> None:
        """valid_confidence must be 0-1 if specified."""
        for spec in PATTERN_REGISTRY:
            if spec.valid_confidence is not None:
                assert 0 <= spec.valid_confidence <= 1

    def test_invalid_confidence_in_range(self) -> None:
        """invalid_confidence must be 0-1 if specified."""
        for spec in PATTERN_REGISTRY:
            if spec.invalid_confidence is not None:
                assert 0 <= spec.invalid_confidence <= 1

    def test_valid_confidence_higher_than_base(self) -> None:
        """When specified, valid_confidence should typically be >= base_confidence."""
        for spec in PATTERN_REGISTRY:
            if spec.valid_confidence is not None:
                # Valid should be at least as high as base (usually higher)
                assert spec.valid_confidence >= spec.base_confidence * 0.9


class TestPreFilterOptimization:
    """Test pre_filter fields for performance optimization."""

    def test_pre_filter_values(self) -> None:
        """pre_filter should be a single character or None."""
        valid_filters = {None, "@", ":", ".", "http", "eyJ", "0x"}
        for spec in PATTERN_REGISTRY:
            assert spec.pre_filter in valid_filters, (
                f"Invalid pre_filter '{spec.pre_filter}' for {spec.entity_type}"
            )

    def test_email_has_at_prefilter(self) -> None:
        """EMAIL_ADDRESS pattern should have '@' pre-filter."""
        email_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "EMAIL_ADDRESS"]
        assert any(s.pre_filter == "@" for s in email_specs)

    def test_ipv6_has_colon_prefilter(self) -> None:
        """IPv6 pattern should have ':' pre-filter."""
        ipv6_specs = [
            s for s in PATTERN_REGISTRY
            if s.entity_type == "IP_ADDRESS" and s.pre_filter == ":"
        ]
        assert len(ipv6_specs) >= 1

    def test_url_has_http_prefilter(self) -> None:
        """URL_WITH_PII should have 'http' pre-filter."""
        url_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "URL_WITH_PII"]
        assert any(s.pre_filter == "http" for s in url_specs)


class TestDenyCheckFlag:
    """Test deny_check field for patterns requiring deny-list filtering."""

    def test_person_name_has_deny_check(self) -> None:
        """PERSON_NAME patterns should have deny_check=True."""
        person_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "PERSON_NAME"]
        assert any(s.deny_check for s in person_specs)

    def test_organization_has_deny_check(self) -> None:
        """ORGANIZATION pattern should have deny_check=True."""
        org_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "ORGANIZATION"]
        assert any(s.deny_check for s in org_specs)

    def test_location_has_deny_check(self) -> None:
        """LOCATION pattern should have deny_check=True."""
        loc_specs = [s for s in PATTERN_REGISTRY if s.entity_type == "LOCATION"]
        assert any(s.deny_check for s in loc_specs)
