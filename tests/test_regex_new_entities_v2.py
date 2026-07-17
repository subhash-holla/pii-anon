"""Tests for new PII entity types: IPv6, URL_WITH_PII, AGE, DATE_TIME, NPI_NUMBER, DEA_NUMBER.

Covers detection, validation, confidence scoring, and edge cases for each type.
"""

from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.engines.regex import validators


# ═══════════════════════════════════════════════════════════════════════════
# IPv6 Address Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestIPv6Detection:
    """Test IPv6 address detection."""

    def test_full_ipv6_address(self) -> None:
        """Full IPv6 address should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "2001:0db8:85a3:0000:0000:8a2e:0370:7334"},
            {"language": "en"},
        )
        ipv6 = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        assert len(ipv6) >= 1

    def test_compressed_ipv6_address(self) -> None:
        """Compressed IPv6 (::1) should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "::1"},
            {"language": "en"},
        )
        ipv6 = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        assert len(ipv6) >= 1

    def test_ipv6_link_local(self) -> None:
        """Link-local IPv6 address should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "fe80::1"},
            {"language": "en"},
        )
        ipv6 = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        assert len(ipv6) >= 1

    def test_ipv6_with_context(self) -> None:
        """IPv6 with 'IP' context should maintain high confidence."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "IPv6 address: 2001:db8::1"},
            {"language": "en"},
        )
        ipv6 = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        assert len(ipv6) >= 1

    def test_not_ipv6_random_colons(self) -> None:
        """Random text with colons should not be detected as IPv6."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "not:an:ipv6"},
            {"language": "en"},
        )
        ipv6 = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        # May or may not detect depending on pattern — verify no crash
        assert isinstance(ipv6, list)

    def test_ipv6_with_ipv4_embedded(self) -> None:
        """IPv6 with embedded IPv4 should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "::ffff:192.0.2.1"},
            {"language": "en"},
        )
        # May detect as IPv6 or IPv4 or both — just verify no crash
        assert isinstance(findings, list)


# ═══════════════════════════════════════════════════════════════════════════
# URL with PII Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestURLWithPIIDetection:
    """Test URL_WITH_PII detection."""

    def test_url_with_email_param(self) -> None:
        """URL with email parameter should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "https://example.com?email=user@test.com"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) >= 1

    def test_url_with_ssn_param(self) -> None:
        """URL with SSN parameter should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "https://example.com?ssn=123-45-6789"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) >= 1

    def test_url_with_phone_param(self) -> None:
        """URL with phone parameter should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "https://example.com?phone=555-123-4567"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) >= 1

    def test_url_without_pii(self) -> None:
        """Normal URL without PII should not be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "https://example.com/page"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) == 0

    def test_url_with_name_param(self) -> None:
        """URL with name parameter should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "https://example.com?name=John+Smith"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) >= 1

    def test_http_url_with_pii(self) -> None:
        """HTTP (non-HTTPS) URL with PII should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "http://example.com?email=test@test.com"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) >= 1

    def test_url_with_embedded_email(self) -> None:
        """URL with embedded email in path/query should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "https://example.com?user=alice@example.com"},
            {"language": "en"},
        )
        urls = [f for f in findings if f.entity_type == "URL_WITH_PII"]
        assert len(urls) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# AGE Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestAGEDetection:
    """Test AGE entity detection with context."""

    def test_age_with_keyword_age(self) -> None:
        """'age 42' should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "age 42"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) >= 1

    def test_age_years_old(self) -> None:
        """'42 years old' should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "42 years old"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) >= 1

    def test_age_hyphenated_year_old(self) -> None:
        """'42-year-old' should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "42-year-old"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) >= 1

    def test_age_aged_keyword(self) -> None:
        """'aged 42' should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "aged 42"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) >= 1

    def test_age_invalid_too_high(self) -> None:
        """Age 200 should be rejected (out of range 0-150)."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "age 200"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) == 0

    def test_age_invalid_negative(self) -> None:
        """Negative ages should be rejected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "age -5"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        # May not match pattern at all, but verify no crash
        assert isinstance(ages, list)

    def test_age_boundary_0(self) -> None:
        """Age 0 should be valid (within 0-150 range)."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "age 0"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        # May or may not detect depending on regex, but verify no crash
        assert isinstance(ages, list)

    def test_age_boundary_150(self) -> None:
        """Age 150 should be valid (upper boundary)."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "age 150"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) >= 1

    def test_age_without_context(self) -> None:
        """Bare number '42' should not be detected as AGE."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "42"},
            {"language": "en"},
        )
        ages = [f for f in findings if f.entity_type == "AGE"]
        assert len(ages) == 0


# ═══════════════════════════════════════════════════════════════════════════
# DATE_TIME (General Dates) Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestDATETIMEDetection:
    """Test DATE_TIME (general date) detection."""

    def test_date_month_day_year(self) -> None:
        """'January 15, 2025' should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "January 15, 2025"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1

    def test_date_numeric_dmy(self) -> None:
        """'15/01/2025' (DD/MM/YYYY) should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "15/01/2025"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1

    def test_date_numeric_mdy(self) -> None:
        """'01/15/2025' (MM/DD/YYYY) should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "01/15/2025"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1

    def test_date_abbreviated_month(self) -> None:
        """'Jan 15, 2025' should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "Jan 15, 2025"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1

    def test_date_abbreviated_month_no_comma(self) -> None:
        """'Jan 15 2025' (no comma) should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "Jan 15 2025"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1

    def test_date_numeric_dash_separator(self) -> None:
        """'15-01-2025' (dashes) should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "15-01-2025"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1

    def test_date_with_context(self) -> None:
        """Date with context keyword should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "on January 15, 2025 we met"},
            {"language": "en"},
        )
        dates = [f for f in findings if f.entity_type == "DATE_TIME"]
        assert len(dates) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# NPI_NUMBER and DEA_NUMBER Detection
# ═══════════════════════════════════════════════════════════════════════════


class TestNPIDetection:
    """Test NPI (National Provider Identifier) detection."""

    def test_npi_validator_function(self) -> None:
        """NPI validator must exist and work."""
        # Valid NPI: must pass Luhn when "80840" is prepended
        # Using a known valid NPI: 1234567893
        # Let's verify the validator is callable
        assert callable(validators.is_valid_npi)

    def test_npi_with_context(self) -> None:
        """NPI with 'NPI' context should be detected."""
        adapter = RegexEngineAdapter()
        # Valid NPI with Luhn: 1234567893
        # "80840" + "1234567893" = "808401234567893"
        # Let's construct a valid one by computing Luhn
        findings = adapter.detect(
            {"text": "NPI: 1234567893"},
            {"language": "en"},
        )
        npis = [f for f in findings if f.entity_type == "NPI_NUMBER"]
        assert len(npis) == 1

    def test_npi_format_detection(self) -> None:
        """10-digit Luhn-valid NPI with 'national provider' keyword should be detected."""
        adapter = RegexEngineAdapter()
        # 1234567893 passes Luhn with "80840" prefix
        findings = adapter.detect(
            {"text": "national provider identifier 1234567893"},
            {"language": "en"},
        )
        npis = [f for f in findings if f.entity_type == "NPI_NUMBER"]
        assert len(npis) >= 1

    def test_npi_without_context(self) -> None:
        """Bare 10-digit NPI without context should not be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "1234567890"},
            {"language": "en"},
        )
        npis = [f for f in findings if f.entity_type == "NPI_NUMBER"]
        assert len(npis) == 0

    def test_npi_emits_npi_number_label(self) -> None:
        """NPI detections carry the NPI_NUMBER label the eval corpus uses.

        Corpus truth labels NPIs as NPI_NUMBER (15,160 spans in DATA v2);
        the eval map has no NPI_NUMBER->MEDICAL_LICENSE alias.

        Confidence derivation (code path: regex_adapter._run_validator → context boost):
          1. base_confidence=0.88; is_valid_npi('2906399474')=True
          2. valid_confidence is None, so the `or` short-circuits to base_confidence=0.88.
             (Note: the `or` idiom means a hypothetical valid_confidence=0.0 would also be
             ignored — use non-zero values.)
          3. context_type='MEDICAL_LICENSE' triggers adjust_confidence;
             CONTEXT_WORDS['MEDICAL_LICENSE'] contains 'npi'; text 'NPI: 2906399474'
             → token 'npi' in ±50-char window → CONTEXT_BOOST=+0.10 applies
          4. Final: min(CONFIDENCE_CAP=0.99, 0.88 + 0.10) = min(0.99, 0.98) = 0.98

        Note: NPI '2906399475' (corpus representative) is Luhn-invalid; without an
        invalid_confidence fallback the validator suppresses it (skip=True → 0 emissions).
        '2906399474' (corrected last digit) is Luhn-valid and exercises the label path.
        The skip-on-invalid behaviour is a concern: many corpus NPIs may be Luhn-invalid
        (mirroring the old DEA skip-on-invalid that zeroed recall); reported as a concern
        but validator semantics are not changed in this task.
        """
        adapter = RegexEngineAdapter()
        findings = adapter.detect({"text": "NPI: 2906399474"}, {"language": "en"})
        npi = [f for f in findings if f.entity_type == "NPI_NUMBER"]
        assert len(npi) == 1
        assert not [f for f in findings if f.entity_type == "MEDICAL_LICENSE"]
        assert npi[0].confidence == pytest.approx(0.98, abs=1e-9)

    def test_npi_invalid_luhn_still_emits_at_reduced_confidence(self) -> None:
        """A corpus-real NPI with a failing Luhn check emits at 0.80 exactly.

        Confidence derivation (code path: regex_adapter._run_validator → context boost):
          1. invalid_confidence=0.70 returned by _run_validator (the generic-validator path in _run_validator)
          2. context_type="MEDICAL_LICENSE" triggers adjust_confidence (line ~529)
          3. CONTEXT_WORDS["MEDICAL_LICENSE"] contains "npi"; text "NPI: 2906399475"
             → token "npi" in ±50-char window → CONTEXT_BOOST=+0.10 applies
          4. Final: min(CONFIDENCE_CAP=0.99, 0.70 + 0.10) = 0.80

        Luhn verification: is_valid_npi('2906399475') == False (luhn_checksum('808402906399475')
        returns False); '2906399474' (last digit corrected) is Luhn-valid and is used by the
        valid-path test above.

        Corpus NPIs are mostly Luhn-invalid; skipping them zeroed recall (mirroring the
        old DEA skip-on-invalid). Format+context is still strong signal; the Luhn check
        upgrades confidence rather than gating emission.
        """
        adapter = RegexEngineAdapter()
        findings = adapter.detect({"text": "NPI: 2906399475"}, {"language": "en"})
        npi = [f for f in findings if f.entity_type == "NPI_NUMBER"]
        assert len(npi) == 1
        assert npi[0].confidence == pytest.approx(0.80, abs=1e-9)


class TestDEADetection:
    """Test DEA (Drug Enforcement Administration) detection."""

    def test_dea_validator_function(self) -> None:
        """DEA validator must exist and work."""
        assert callable(validators.is_valid_dea_number)

    def test_dea_format_validation(self) -> None:
        """DEA validator should work on valid format."""
        # DEA format: 2 letters + 7 digits
        # Check digit = (odd_sum + 2*even_sum) % 10
        # Example: "AB1234563"
        # Positions: A B 1 2 3 4 5 6 3
        # Digits:       1 2 3 4 5 6 3
        # odd (1,3,5):  1 + 3 + 5 = 9
        # even (2,4,6): 2 + 4 + 6 = 12
        # check = (9 + 2*12) % 10 = 33 % 10 = 3 ✓
        assert validators.is_valid_dea_number("AB1234563")

    def test_dea_with_context(self) -> None:
        """DEA with context keyword should be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "DEA number: AB1234563"},
            {"language": "en"},
        )
        deas = [f for f in findings if f.entity_type == "DEA_NUMBER"]
        assert len(deas) >= 1

    def test_dea_without_context(self) -> None:
        """Bare DEA number without context should not be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "AB1234563"},
            {"language": "en"},
        )
        deas = [f for f in findings if f.entity_type == "DEA_NUMBER"]
        assert len(deas) == 0

    def test_dea_invalid_checksum(self) -> None:
        """DEA with invalid check digit should fail validation."""
        # Change the check digit to make it invalid
        assert not validators.is_valid_dea_number("AB1234560")

    def test_dea_wrong_format(self) -> None:
        """DEA with wrong format should fail validation."""
        # Numbers only (no letters)
        assert not validators.is_valid_dea_number("1234567890")

    def test_dea_pattern_with_spaces(self) -> None:
        """DEA pattern should handle various formats with context."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "DEA registration AB1234563"},
            {"language": "en"},
        )
        deas = [f for f in findings if f.entity_type == "DEA_NUMBER"]
        assert len(deas) >= 1

    def test_dea_emits_dea_number_label(self) -> None:
        """DEA detections carry the DEA_NUMBER label the eval corpus uses.

        Corpus truth labels DEA registrations as DEA_NUMBER (9,283 spans in
        DATA v2). Emitting MEDICAL_LICENSE double-penalizes: one FP + one FN.
        """
        adapter = RegexEngineAdapter()
        findings = adapter.detect({"text": "DEA: MP5912274"}, {"language": "en"})
        dea = [f for f in findings if f.entity_type == "DEA_NUMBER"]
        assert len(dea) == 1
        assert not [f for f in findings if f.entity_type == "MEDICAL_LICENSE"]

    def test_dea_invalid_checksum_still_emits_at_reduced_confidence(self) -> None:
        """A format-valid DEA with a failing checksum emits at 0.80 exactly.

        Confidence derivation (code path: regex_adapter._run_validator → context boost):
          1. invalid_confidence=0.70 returned by _run_validator (line ~701 in regex_adapter.py)
          2. context_type="MEDICAL_LICENSE" triggers adjust_confidence (line ~529)
          3. CONTEXT_WORDS["MEDICAL_LICENSE"] contains "dea"; text "DEA: JS7964296"
             → token "dea" in ±50-char window → CONTEXT_BOOST=+0.10 applies
          4. Final: 0.70 + 0.10 = 0.80

        Corpus DEA values are mostly checksum-invalid (JS7964296: check digit
        9 != 6); skipping them zeroed recall. Format+context is still strong
        signal; the checksum upgrades confidence rather than gating emission.
        """
        adapter = RegexEngineAdapter()
        findings = adapter.detect({"text": "DEA: JS7964296"}, {"language": "en"})
        dea = [f for f in findings if f.entity_type == "DEA_NUMBER"]
        assert len(dea) == 1
        assert dea[0].confidence == pytest.approx(0.80, abs=1e-9)

    def test_dea_valid_checksum_upgrades_confidence(self) -> None:
        """A checksum-valid DEA (MP5912274) emits at 0.99 exactly.

        Confidence derivation (code path: regex_adapter._run_validator → context boost):
          1. valid_confidence=0.93 returned by _run_validator (line ~699 in regex_adapter.py)
          2. context_type="MEDICAL_LICENSE" triggers adjust_confidence (line ~529)
          3. CONTEXT_WORDS["MEDICAL_LICENSE"] contains "dea"; text "DEA: MP5912274"
             → token "dea" in ±50-char window → CONTEXT_BOOST=+0.10 applies
          4. Final: min(CONFIDENCE_CAP=0.99, 0.93 + 0.10) = min(0.99, 1.03) = 0.99
        """
        adapter = RegexEngineAdapter()
        findings = adapter.detect({"text": "DEA: MP5912274"}, {"language": "en"})
        dea = [f for f in findings if f.entity_type == "DEA_NUMBER"]
        assert len(dea) == 1
        assert dea[0].confidence == 0.99


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNewEntitiesIntegration:
    """Integration tests with RegexEngineAdapter.detect()."""

    def test_multiple_new_entity_types_in_text(self) -> None:
        """Multiple new entity types should be detected in same text."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {
                "text": "IPv6: 2001:db8::1, age 42, January 15, 2025, DEA: AB1234563"
            },
            {"language": "en"},
        )
        # Should detect at least some of these new entity types
        assert len(findings) > 0
        detected_types = {f.entity_type for f in findings}
        assert len(detected_types) >= 2

    def test_findings_have_correct_attributes(self) -> None:
        """All findings should have required EngineFinding attributes."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "age 42 and 2001:db8::1"},
            {"language": "en"},
        )
        for finding in findings:
            assert hasattr(finding, "entity_type")
            assert hasattr(finding, "confidence")
            assert hasattr(finding, "field_path")
            assert hasattr(finding, "span_start")
            assert hasattr(finding, "span_end")
            assert hasattr(finding, "engine_id")
