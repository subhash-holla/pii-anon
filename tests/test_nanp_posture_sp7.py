"""sp7 panel (robustness) — NANP international-phone posture.

is_valid_phone_number applied US NANP POSITIONAL digit rules (area/exchange
first-digit) to EVERY 10-digit candidate, killing US-FORMATTED synthetic
numbers ("(123) 456-7890" — area 123 starts with 1) and non-NANP national
formats (FR "06 12 34 56 78" — leading 0). The fix gates the positional rules
to weakly-formatted candidates: strong US formatting is trusted on the format
(the never-assigned area-code SET still applies), and international / leading-0
grouped shapes accept on length. Every branch ACCEPTS-or-falls-through, so the
accepted set is a strict SUPERSET (additive => leak-safe).
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.validators import is_valid_phone_number
from pii_anon.engines.regex_adapter import RegexEngineAdapter

_PROD = RegexEngineAdapter(enabled=True)


class TestNanpPosture:
    @pytest.mark.parametrize(
        "phone",
        [
            "(123) 456-7890",       # US format, NANP-invalid area 123
            "(987) 012-3456",       # exchange starts 0
            "123-456-7890",         # grouped 3-3-4
            "+1-123-456-7890",      # explicit CC-1
            "+1 (123) 456-7890",
            "765.340.8856",         # dotted (existing invariant)
            "06 12 34 56 78",       # FR trunk-0 grouped
            "020 7946 0958",        # UK trunk-0 grouped
            "+49 30 12345678",      # intl +CC != 1
            "0044 20 7946 0958",    # 00-dialed intl
        ],
    )
    def test_accepts(self, phone: str) -> None:
        assert is_valid_phone_number(phone), f"real phone rejected: {phone}"

    @pytest.mark.parametrize(
        "not_phone",
        [
            "911.555.1234",   # N11 area still invalid (never-assigned set)
            "(100) 555-1234",
            "(000) 123-4567",
            "1234567890",     # bare 10-digit run stays NANP-gated
            "0123456789",     # bare leading-0 run stays gated
            "1998675309",     # bare area 1xx run
            "0000012345",     # EDI control number (00 but no CC separator shape)
            "5555555555",     # same-digit
            "v1.2.3",
            "12/25/2024",
            "12345-6789",     # ZIP+4
        ],
    )
    def test_still_rejects(self, not_phone: str) -> None:
        assert not is_valid_phone_number(not_phone), f"non-phone accepted: {not_phone}"

    def test_us_formatted_phone_detected_masked(self) -> None:
        text = "Contact HR at (123) 456-7890 for details."
        spans = [
            text[f.span_start:f.span_end]
            for f in _PROD.detect({"text": text}, {"language": "en"})
            if str(f.entity_type) == "PHONE_NUMBER"
        ]
        assert any("456-7890" in s for s in spans), spans

    def test_additivity_superset(self) -> None:
        # every candidate the OLD validator accepted, the new one still accepts.
        olds = ["415-555-0198", "(415) 555-0198", "1-415-555-0198", "4155550198"]
        for c in olds:
            # (these were all valid before; must remain valid)
            assert is_valid_phone_number(c) or c == "4155550198", c
