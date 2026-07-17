"""sp7 panel (robustness lens) — dotted NANP phones rejected by the validator.

The version-number FP rule (``v?\\d+\\.\\d+\\.\\d+``) matched ANY dot-separated
phone, so real numbers like ``765.340.8856`` failed validation — missed PII on
the masking path. A 3-3-4 dotted string is never a version number, date, or
ZIP+4, so that shape is exempted from the FP rule. ADDITIVE (validator accepts
more => more masking) => leak-safe.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.validators import is_valid_phone_number
from pii_anon.engines.regex_adapter import RegexEngineAdapter

_PROD = RegexEngineAdapter(enabled=True)


class TestDottedPhoneValidator:
    @pytest.mark.parametrize("phone", ["765.340.8856", "800.555.1212", "415.555.0198"])
    def test_dotted_nanp_phone_valid(self, phone: str) -> None:
        assert is_valid_phone_number(phone), f"real dotted phone rejected: {phone}"

    @pytest.mark.parametrize(
        "not_phone",
        [
            "v1.2.3",        # version
            "1.2.3",         # version
            "12/25/2024",    # date
            "12345-6789",    # ZIP+4
            "5555555555",    # repeated digits
            "911.555.1234",  # N11 area code stays invalid
        ],
    )
    def test_non_phones_still_rejected(self, not_phone: str) -> None:
        assert not is_valid_phone_number(not_phone), f"non-phone accepted: {not_phone}"

    def test_dotted_phone_detected_and_masked(self) -> None:
        text = "Call the office at 765.340.8856 tomorrow."
        spans = [
            text[f.span_start:f.span_end]
            for f in _PROD.detect({"text": text}, {"language": "en"})
            if str(f.entity_type) == "PHONE_NUMBER"
        ]
        assert any("765.340.8856" in s for s in spans), spans
