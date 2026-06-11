"""Tests for the pattern registry system (PatternSpec and PATTERN_REGISTRY).

Covers: registry integrity, PatternSpec fields, validator references,
context types, confidence ranges, immutability, and registry size.
"""

from __future__ import annotations

import pytest

from pii_anon.engines.regex.patterns import PATTERN_REGISTRY, PatternSpec, _FIELD_LABEL_NOUNS
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


class TestLicensePlateParenthetical:
    """Corpus miss: parenthetical qualifier between keyword and separator
    (e.g. "License Plate (rental): JBL-4117")."""

    def _detect(self, text: str):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == "LICENSE_PLATE"
        ]

    def test_parenthetical_qualifier_form(self) -> None:
        # Corpus miss: "License Plate (rental): JBL-4117"
        text = "License Plate (rental): JBL-4117 DOB: 1982-01-01"
        found = self._detect(text)
        assert len(found) == 1, (
            f"Expected 1 LICENSE_PLATE finding, got {len(found)}: {found}"
        )
        assert found[0].span_start is not None
        assert found[0].span_end is not None
        assert text[found[0].span_start : found[0].span_end] == "JBL-4117", (
            f"Span text was {text[found[0].span_start:found[0].span_end]!r}, "
            f"expected 'JBL-4117'"
        )

    def test_plain_form_still_matches(self) -> None:
        # Both _LICENSE_PLATE and _LICENSE_PLATE_US may fire on the same span;
        # at least one finding is the contract here (plain forms must not regress).
        assert len(self._detect("License Plate: MOO-8530")) >= 1

    def test_parenthetical_too_long_not_bridged(self) -> None:
        # A >24-char parenthetical is not a qualifier; don't bridge it.
        assert (
            self._detect(
                "License Plate (this is a long unrelated aside about rentals): JBL-4117"
            )
            == []
        )

    def test_parenthetical_length_boundary_exact(self) -> None:
        # Exact boundary of the {1,24} cap: a 24-char body bridges, 25 does not.
        assert len(self._detect(f"License Plate ({'x' * 24}): JBL-4117")) == 1
        assert self._detect(f"License Plate ({'x' * 25}): JBL-4117") == []


class TestPersonNameFieldLabelExclusion:
    """Field labels ('Medication Name', 'Docket Number') are the largest
    PERSON_NAME FP factory on the DATA corpus (6,670 FPs at N=10,000)."""

    FIELD_LABEL_PHRASES = [
        # "Swift Bic Code" still suppressed because Bic/Code are in the guard.
        "Swift Bic Code", "Bic Code", "Medication Name", "Docket Number",
        "Notary License", "National Id Number", "Health Insurance",
        "Insurance Policy Number", "Bank Routing Number",
    ]
    REAL_NAMES = ["Kenneth Anderson", "Andrea Romano", "Sara Conti", "Heike Becker"]

    def _names(self, text: str):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == "PERSON_NAME"
        ]

    def test_field_labels_not_detected_as_names(self) -> None:
        for phrase in self.FIELD_LABEL_PHRASES:
            assert self._names(f"Record for X, {phrase}: ABC-123.") == [], phrase

    def test_real_names_still_detected(self) -> None:
        for name in self.REAL_NAMES:
            assert len(self._names(f"Record for {name}, contact x@y.com")) >= 1, name


class TestFieldLabelNounsMechanismGuard:
    """Mechanism test: every noun in _FIELD_LABEL_NOUNS must suppress the
    full-name pattern when it appears as the 2nd or 3rd capitalized token.

    Parametrized over the constant itself — any future element addition or
    removal is automatically covered.
    """

    def _names(self, text: str) -> list:
        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == "PERSON_NAME"
        ]

    @pytest.mark.parametrize("noun", _FIELD_LABEL_NOUNS)
    def test_two_token_prefix_suppressed(self, noun: str) -> None:
        """'Foo <noun> xyz' must not produce a PERSON_NAME."""
        assert self._names(f"Foo {noun} xyz") == [], (
            f"Expected no PERSON_NAME for 'Foo {noun} xyz'"
        )

    @pytest.mark.parametrize("noun", _FIELD_LABEL_NOUNS)
    def test_three_token_prefix_suppressed(self, noun: str) -> None:
        """'Foo Bar <noun> xyz' must not produce a PERSON_NAME."""
        assert self._names(f"Foo Bar {noun} xyz") == [], (
            f"Expected no PERSON_NAME for 'Foo Bar {noun} xyz'"
        )

    def test_positive_control_non_vocab_trigram(self) -> None:
        """A genuine 3-word name not ending in a vocab noun still matches."""
        # "Foo Barland xyz" — Barland is not a field-label noun.
        found = self._names("Foo Barland attended the meeting")
        assert len(found) >= 1, (
            "Expected at least 1 PERSON_NAME for non-vocab trigram 'Foo Barland'"
        )


class TestPersonNameRescueRegression:
    """Rescue-regression pins: names with medical/title context must survive.

    These cases exercise the boundary between the field-label guard (which
    suppresses bare 2-token caps that look like form labels) and the
    keyword-context patterns (which actively promote genuine person names).
    """

    def _names(self, text: str) -> list:
        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == "PERSON_NAME"
        ]

    def test_patient_context_rescues_name(self) -> None:
        """'patient Anna Swift was discharged' → ≥1 PERSON_NAME."""
        found = self._names("patient Anna Swift was discharged")
        assert len(found) >= 1, (
            "Expected ≥1 PERSON_NAME from 'patient Anna Swift was discharged' "
            f"— got {found}"
        )

    def test_title_prefix_rescues_name(self) -> None:
        """'Ms. Taylor Swift' → ≥1 PERSON_NAME (title-prefix pattern fires)."""
        found = self._names("Ms. Taylor Swift")
        assert len(found) >= 1, (
            "Expected ≥1 PERSON_NAME from 'Ms. Taylor Swift' "
            f"— got {found}"
        )

    def test_bare_name_detected_after_swift_removal(self) -> None:
        """Bare 'Anna Swift attended the meeting' → ≥1 PERSON_NAME.

        BEHAVIOR NOTE (2026-06-11): 'Swift' was removed from _FIELD_LABEL_NOUNS
        (measured 0 FP-delta at n=2000; Bic/Code cover SWIFT-BIC field labels).
        As a result, 'Anna Swift' is now correctly detected as a PERSON_NAME by
        the full-name pattern.  If this test starts failing it means the guard
        logic changed in a way that re-suppresses bare 'Swift'-surname names —
        review the _FIELD_LABEL_NOUNS tuple and the measured FP impact.
        """
        found = self._names("Anna Swift attended the meeting")
        assert len(found) >= 1, (
            "Expected ≥1 PERSON_NAME for 'Anna Swift attended the meeting' "
            f"after Swift removal from the guard — got {found}"
        )


class TestNationalIdValueShape:
    """SP1 Task 8: NATIONAL_ID must require a digit, accept hyphenated values,
    and not capture the literal qualifier word 'Number' as the value span.

    FP class (observed n=2000): span='Number' from 'National Id Number: NID-…'
    FN class A (dominant): Tax Id values like '37-3808704' (digit-hyphen-digit).
    FN class B: NID-prefixed mixed-alnum values like 'NID-275ZI1979'.
    FN class C: mixed-case Tax Id values like 'gG-674ZT4G', '5T-243002S'.
    """

    def _ids(self, text: str):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == "NATIONAL_ID"
        ]

    def _span_texts(self, text: str) -> list[str]:
        return [
            text[f.span_start : f.span_end]
            for f in self._ids(text)
            if f.span_start is not None and f.span_end is not None
        ]

    def test_literal_word_number_is_not_a_national_id(self) -> None:
        # FP class: the captured span was literally "Number".
        # Both _NATIONAL_ID (context) and _NATIONAL_ID_NID (standalone) may
        # fire on the same NID span — ≥1 finding is the contract; dedup upstream.
        text = "Record for Stefan Müller, National Id Number: NID-114836089."
        spans = self._span_texts(text)
        assert "Number" not in spans, (
            f"Span 'Number' must not be captured as NATIONAL_ID; got {spans!r}"
        )
        assert len(self._ids(text)) >= 1, (
            f"Expected ≥1 NATIONAL_ID (the real value), got {self._ids(text)}"
        )

    def test_hyphenated_nid_value_matches(self) -> None:
        # NID-prefixed hyphenated value with context keyword.
        # Both _NATIONAL_ID (context) and _NATIONAL_ID_NID (standalone) may
        # fire on the same span — ≥1 finding is the contract; dedup is upstream.
        found = self._ids("National Id: NID-114836089")
        assert len(found) >= 1, f"Expected ≥1 NATIONAL_ID, got {found}"

    def test_tax_id_hyphenated_numeric_value_matches(self) -> None:
        # FN class A: 'Tax Id: 37-3808704' — digit-hyphen-digits form.
        found = self._ids("Record for John Smith, Tax Id: 37-3808704, contact john@x.com")
        assert len(found) == 1, (
            f"Expected 1 NATIONAL_ID for Tax Id hyphenated-numeric value, got {found}"
        )
        text = "Record for John Smith, Tax Id: 37-3808704, contact john@x.com"
        spans = self._span_texts(text)
        assert "37-3808704" in spans, (
            f"Expected span '37-3808704', got {spans!r}"
        )

    def test_tax_id_mixed_case_alphanum_hyphen_matches(self) -> None:
        # FN class C: mixed-case alphanumeric with hyphen, e.g. '5T-243002S'.
        found = self._ids("Record for Emily Ramirez, Tax Id: 5T-243002S, contact emily@x.com")
        assert len(found) == 1, (
            f"Expected 1 NATIONAL_ID for Tax Id mixed-case value, got {found}"
        )

    def test_pure_alpha_value_not_captured(self) -> None:
        # Negative: a value with no digits must not be captured.
        spans = self._span_texts("National Id Number: Pending")
        assert all(
            any(c.isdigit() for c in s) for s in spans
        ), f"All captured spans must contain at least one digit; got {spans!r}"

    def test_number_qualifier_consumed_nid_value_captured(self) -> None:
        # 'Number' qualifier is consumed; the actual NID- value is the span.
        text = "National Id Number: NID-275ZI1979"
        spans = self._span_texts(text)
        assert "Number" not in spans, f"'Number' must not be a span; got {spans!r}"
        assert any("NID" in s or s[0].isdigit() for s in spans), (
            f"Expected the real ID value in spans; got {spans!r}"
        )
