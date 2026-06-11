"""Tests for the pattern registry system (PatternSpec and PATTERN_REGISTRY).

Covers: registry integrity, PatternSpec fields, validator references,
context types, confidence ranges, immutability, and registry size.
"""

from __future__ import annotations

from pii_anon.engines.regex.patterns import PATTERN_REGISTRY, PatternSpec
from pii_anon.engines.regex.confidence import CONTEXT_WORDS
from pii_anon.engines.regex import validators
from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.types import Payload


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

    def test_npi_number_in_registry(self) -> None:
        """NPI_NUMBER and DEA_NUMBER patterns must be in registry."""
        npi_specs = [
            s for s in PATTERN_REGISTRY if s.entity_type == "NPI_NUMBER"
        ]
        assert len(npi_specs) >= 1  # NPI emits NPI_NUMBER
        dea_specs = [
            s for s in PATTERN_REGISTRY if s.entity_type == "DEA_NUMBER"
        ]
        assert len(dea_specs) >= 1  # DEA now has its own label


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


class TestCreditCardFragmentBehavior:
    """Behavioral tests for the credit card fragment pattern (context-free path)."""

    def test_card_ending_in_emits_exactly_one_credit_card_at_flat_088(self) -> None:
        """'card ending in 1234' emits exactly one CREDIT_CARD finding at 0.88.

        The fragment PatternSpec sets context_type=None, so it never enters
        adjust_confidence — the base_confidence of 0.88 is the final score
        regardless of surrounding context.  There must be no context-boost or
        context-penalty applied (the CREDIT_CARD_CONTEXT set in confidence.py
        intentionally excludes CREDIT_CARD to avoid penalising Luhn-valid full
        cards lacking context; the fragment spec's context_type=None means it
        never reaches that branch at all).
        """
        adapter = RegexEngineAdapter()
        payload = Payload(text="card ending in 1234")
        findings = adapter.detect(payload, {})
        assert len(findings) == 1, (
            f"Expected exactly 1 finding, got {len(findings)}: {findings}"
        )
        finding = findings[0]
        assert finding.entity_type == "CREDIT_CARD", (
            f"Expected entity_type CREDIT_CARD, got {finding.entity_type!r}"
        )
        assert finding.confidence == 0.88, (
            f"Expected confidence 0.88 (flat, no context adjustment), "
            f"got {finding.confidence}"
        )


class TestDriversLicenseCorpusForms:
    """DATA v2 corpus DL surface forms (23,461 spans) — formats the value
    sub-pattern missed entirely (baseline R=0.06)."""

    def _detect(self, text: str):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == "DRIVERS_LICENSE"
        ]

    def test_two_letter_hyphen_six_digit_form(self) -> None:
        # Corpus-dominant: "Driver License Number: LC-908984"
        found = self._detect(
            "Record for Stefan Weber, Driver License Number: QR-427697, contact x@y.de"
        )
        assert len(found) == 1, f"Expected 1 DRIVERS_LICENSE finding, got {len(found)}: {found}"
        # The captured span should cover the value "QR-427697" (not the keyword)
        text = "Record for Stefan Weber, Driver License Number: QR-427697, contact x@y.de"
        assert found[0].span_start is not None
        assert found[0].span_end is not None
        assert text[found[0].span_start:found[0].span_end] == "QR-427697", (
            f"Span text was {text[found[0].span_start:found[0].span_end]!r}, "
            f"expected 'QR-427697'"
        )

    def test_existing_forms_still_match(self) -> None:
        assert len(self._detect("driver's license: D1234567")) == 1
        # DL-prefix standalone: only _DRIVERS_LICENSE_DL fires (CTX keyword absent)
        assert len(self._detect("DL-A12345-678")) == 1

    def test_bare_value_without_context_does_not_match(self) -> None:
        # The XX-NNNNNN shape is generic (order codes); context stays required.
        assert self._detect("Tracking ref QR-427697 shipped today") == []

    def test_spdx_software_license_not_detected(self) -> None:
        # Software/SPDX identifiers after a bare 'license' keyword must not match.
        assert self._detect("license # MIT-12345 applies to this package") == []
        assert self._detect("License Number: GPL-30000 (open source)") == []

    def test_bare_license_keyword_still_matches_classic_forms(self) -> None:
        assert len(self._detect("license number: A12345678")) == 1

    def test_driver_phrase_single_emission(self) -> None:
        # Both keyword paths overlap on this phrase; exactly one finding must emerge.
        # The negative lookbehind in _DRIVERS_LICENSE_BARE_KW prevents it from
        # firing inside the driver-specific phrase already handled by _DRIVERS_LICENSE_CTX.
        assert len(self._detect("driver's license number: D1234567")) == 1
