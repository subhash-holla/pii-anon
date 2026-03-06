"""Tests for new entity type patterns (Enhancement 2).

Covers all 10 new entity types: CRYPTO_WALLET, GPS_COORDINATES, SWIFT_BIC,
VIN, ZIP_CODE, CANADIAN_SIN, UK_NI_NUMBER, JWT_TOKEN, API_KEY, AADHAAR.
Each type has positive and negative test cases.
"""

from __future__ import annotations

from pii_anon.engines.regex_adapter import RegexEngineAdapter


def _detect(text: str, **ctx: object) -> list:
    adapter = RegexEngineAdapter()
    context = {"language": "en", **ctx}
    return adapter.detect({"text": text}, context)


def _find(text: str, entity_type: str) -> list:
    return [f for f in _detect(text) if f.entity_type == entity_type]


# ---------------------------------------------------------------------------
# CRYPTO_WALLET
# ---------------------------------------------------------------------------


class TestCryptoWallet:
    def test_bitcoin_legacy(self) -> None:
        findings = _find("Send to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "CRYPTO_WALLET")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.90

    def test_bitcoin_bech32(self) -> None:
        findings = _find(
            "Address: bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
            "CRYPTO_WALLET",
        )
        assert len(findings) >= 1

    def test_ethereum(self) -> None:
        findings = _find(
            "ETH wallet 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18",
            "CRYPTO_WALLET",
        )
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.95

    def test_short_string_not_matched(self) -> None:
        findings = _find("Value is 0x1234", "CRYPTO_WALLET")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# GPS_COORDINATES
# ---------------------------------------------------------------------------


class TestGPSCoordinates:
    def test_valid_pair(self) -> None:
        findings = _find("Located at 37.7749, -122.4194 in SF.", "GPS_COORDINATES")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.85

    def test_negative_lat_lon(self) -> None:
        findings = _find("Position: -33.8688, 151.2093", "GPS_COORDINATES")
        assert len(findings) >= 1

    def test_boundary_values(self) -> None:
        findings = _find("Edge: 90.0, 180.0", "GPS_COORDINATES")
        assert len(findings) >= 1

    def test_out_of_range_lat(self) -> None:
        # 91 is outside latitude range — regex should not match or validate should fail
        findings = _find("Invalid: 91.0, 45.0", "GPS_COORDINATES")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# SWIFT_BIC
# ---------------------------------------------------------------------------


class TestSwiftBic:
    def test_with_context(self) -> None:
        findings = _find("Wire transfer BIC: DEUTDEFF500", "SWIFT_BIC")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.80

    def test_without_context_not_detected(self) -> None:
        """SWIFT_BIC requires bank/wire/transfer context — no context → no match."""
        findings = _find("Code DEUTDEFF500 appears in text.", "SWIFT_BIC")
        assert len(findings) == 0

    def test_8_char_code(self) -> None:
        findings = _find("Bank BIC is BOFAUS3N for wire.", "SWIFT_BIC")
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# VIN
# ---------------------------------------------------------------------------


class TestVIN:
    def test_valid_vin_with_check_digit(self) -> None:
        # 11111111111111111 has a valid check digit (position 9 = '1')
        findings = _find("VIN: 11111111111111111", "VIN")
        assert len(findings) >= 1

    def test_vin_format_only(self) -> None:
        # A random 17-char string (no I/O/Q) should match at format level
        findings = _find("VIN: WVWZZZ3CZWE123456", "VIN")
        assert len(findings) >= 1

    def test_too_short(self) -> None:
        findings = _find("VIN: ABC123", "VIN")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# ZIP_CODE
# ---------------------------------------------------------------------------


class TestZipCode:
    def test_5_digit_with_context(self) -> None:
        findings = _find("Zip code: 90210", "ZIP_CODE")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.88

    def test_5_plus_4_with_context(self) -> None:
        findings = _find("Postal code: 90210-1234", "ZIP_CODE")
        assert len(findings) >= 1

    def test_without_context_not_detected(self) -> None:
        """ZIP_CODE requires context keyword — bare number should not match."""
        findings = _find("The number is 90210 in the list.", "ZIP_CODE")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# CANADIAN_SIN
# ---------------------------------------------------------------------------


class TestCanadianSIN:
    def test_with_context_and_luhn(self) -> None:
        # 046 454 286 is Luhn-valid
        findings = _find("SIN: 046 454 286", "CANADIAN_SIN")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.90

    def test_with_context_no_luhn(self) -> None:
        findings = _find("SIN number: 123-456-780", "CANADIAN_SIN")
        assert len(findings) >= 1
        # Should still detect but at lower confidence
        assert findings[0].confidence <= 0.80

    def test_without_context_not_detected(self) -> None:
        findings = _find("The code is 123456789 here.", "CANADIAN_SIN")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# UK_NI_NUMBER
# ---------------------------------------------------------------------------


class TestUKNINumber:
    def test_valid_nino(self) -> None:
        findings = _find("National Insurance: AB 12 34 56 C", "UK_NI_NUMBER")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.85

    def test_compact_nino(self) -> None:
        findings = _find("NINO: AB123456C", "UK_NI_NUMBER")
        assert len(findings) >= 1

    def test_invalid_suffix_letter(self) -> None:
        # Suffix must be A-D; 'E' is invalid
        findings = _find("NINO: AB123456E", "UK_NI_NUMBER")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# JWT_TOKEN
# ---------------------------------------------------------------------------


class TestJWTToken:
    def test_valid_jwt_structure(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        findings = _find(f"Token: {jwt}", "JWT_TOKEN")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.93

    def test_not_jwt_prefix(self) -> None:
        findings = _find("Token: abc.def.ghi", "JWT_TOKEN")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# API_KEY
# ---------------------------------------------------------------------------


class TestAPIKey:
    def test_api_key_with_context(self) -> None:
        findings = _find(
            "api_key: sk_fake_abcdefghij1234567890abcdefghij12",
            "API_KEY",
        )
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.89

    def test_bearer_token(self) -> None:
        findings = _find(
            "bearer: ghp_1234567890abcdefGHIJ1234567890abcd",
            "API_KEY",
        )
        assert len(findings) >= 1

    def test_without_context_not_detected(self) -> None:
        findings = _find("Value sk_fake_abcdefghij1234567890abcdefghij12", "API_KEY")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# AADHAAR
# ---------------------------------------------------------------------------


class TestAadhaar:
    def test_with_context(self) -> None:
        findings = _find("Aadhaar number: 2345 6789 0123", "AADHAAR")
        assert len(findings) >= 1

    def test_compact_with_context(self) -> None:
        findings = _find("UID: 234567890123", "AADHAAR")
        assert len(findings) >= 1

    def test_without_context_not_detected(self) -> None:
        findings = _find("The number is 234567890123 in records.", "AADHAAR")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# DATE_ISO (Enhancement 5)
# ---------------------------------------------------------------------------


class TestDateISO:
    def test_valid_iso_date(self) -> None:
        findings = _find("Date: 2025-01-15 was recorded.", "DATE_ISO")
        assert len(findings) >= 1
        assert findings[0].confidence >= 0.80

    def test_invalid_month(self) -> None:
        findings = _find("Date: 2025-13-01", "DATE_ISO")
        assert len(findings) == 0

    def test_invalid_day(self) -> None:
        findings = _find("Date: 2025-01-32", "DATE_ISO")
        assert len(findings) == 0

    def test_valid_boundary_date(self) -> None:
        findings = _find("Date: 2025-12-31", "DATE_ISO")
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# Improved regex patterns for better benchmark F1
# ---------------------------------------------------------------------------


class TestAddressFullFormat:
    """ADDRESS regex should capture full mailing addresses including city, state, zip."""

    def test_street_only(self) -> None:
        findings = _find("Lives at 1234 Elm Street in town", "ADDRESS")
        assert len(findings) >= 1

    def test_full_address_with_city_state_zip(self) -> None:
        findings = _find("Address: 6959 Dogwood Drive, Madison, MI 36659", "ADDRESS")
        assert len(findings) >= 1
        # The full match should include city, state, and zip.
        matched = findings[0]
        actual = "Address: 6959 Dogwood Drive, Madison, MI 36659"[
            matched.span_start : matched.span_end
        ]
        assert "36659" in actual

    def test_full_address_alternate_suffix(self) -> None:
        findings = _find("Home: 2033 Hickory Boulevard, Salem, FL 11933", "ADDRESS")
        assert len(findings) >= 1
        matched = findings[0]
        actual = "Home: 2033 Hickory Boulevard, Salem, FL 11933"[
            matched.span_start : matched.span_end
        ]
        assert "11933" in actual


class TestDriversLicenseDLPrefix:
    """DRIVERS_LICENSE regex should match DL-prefixed IDs like DL-C19362-62."""

    def test_dl_prefix_format(self) -> None:
        findings = _find("DL: DL-C19362-62.", "DRIVERS_LICENSE")
        assert len(findings) >= 1

    def test_dl_prefix_in_context(self) -> None:
        findings = _find("Bar License: DL-G20640-40", "DRIVERS_LICENSE")
        assert len(findings) >= 1

    def test_classic_format_still_works(self) -> None:
        findings = _find("driver's license: A1234567890", "DRIVERS_LICENSE")
        assert len(findings) >= 1


class TestPassportMixedAlphanumeric:
    """PASSPORT regex should match mixed alphanumeric IDs like P7H104167."""

    def test_mixed_format(self) -> None:
        findings = _find("Passport: P7H104167", "PASSPORT")
        assert len(findings) >= 1

    def test_classic_letter_digit_format(self) -> None:
        findings = _find("Passport: AB1234567", "PASSPORT")
        assert len(findings) >= 1


class TestNationalIdPrefixed:
    """NATIONAL_ID regex should match TAX- and NID- prefixed IDs."""

    def test_tax_prefix(self) -> None:
        findings = _find("International tax id: TAX-700015019.", "NATIONAL_ID")
        assert len(findings) >= 1

    def test_nid_prefix(self) -> None:
        findings = _find("National ID: NID-900074165", "NATIONAL_ID")
        assert len(findings) >= 1

    def test_classic_format(self) -> None:
        findings = _find("National ID: ABC12345678", "NATIONAL_ID")
        assert len(findings) >= 1


class TestUsernameContextExtended:
    """USERNAME regex should match usernames in log and config contexts."""

    def test_at_handle(self) -> None:
        findings = _find("Follow @janedoe123 on Twitter", "USERNAME")
        assert len(findings) >= 1

    def test_log_context(self) -> None:
        findings = _find("[INFO] User adammorris582 logged in", "USERNAME")
        assert len(findings) >= 1

    def test_config_context(self) -> None:
        findings = _find('config = {"db_user": "brianmiller876"}', "USERNAME")
        assert len(findings) >= 1


class TestEmployeeIdHyphenated:
    """EMPLOYEE_ID regex should match EMP- prefixed IDs."""

    def test_emp_prefix(self) -> None:
        findings = _find("Employee ID: EMP-20165", "EMPLOYEE_ID")
        assert len(findings) >= 1

    def test_emp_in_sentence(self) -> None:
        findings = _find("My employee ID is EMP-18310 in case you need it", "EMPLOYEE_ID")
        assert len(findings) >= 1
