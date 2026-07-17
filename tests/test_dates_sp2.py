"""sp2 dev-iteration 4 — DATE_TIME / TIMESTAMP / DATE_OF_BIRTH hygiene.

Dev-split evidence (strict matching): TIMESTAMP gold is ISO-8601 datetimes
(``2021-10-23T09:53:00Z``) which no pattern matched (recall 0.000 over 2,521
golds), while ``_DATE_GENERAL``'s dotted numeric form matched FRAGMENTS OF IP
ADDRESSES ("208.⟦74.38.190⟧") and DOB-context dates ("DOB: ⟦12/28/1966⟧",
gold = DATE_OF_BIRTH) — pure FP factories.
"""
from __future__ import annotations

from pii_anon.engines.regex_adapter import RegexEngineAdapter


def _spans(text: str, entity_type: str) -> list[str]:
    engine = RegexEngineAdapter(enabled=True)
    return sorted(
        text[f.span_start : f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


class TestIso8601Datetime:
    def test_labeled_iso_datetime_exact_extent(self) -> None:
        text = "Ref: INV-863469\nDate: 2026-02-21T12:55:00Z"
        assert "2026-02-21T12:55:00Z" in _spans(text, "DATE_TIME")

    def test_bracketed_log_timestamp_exact_extent(self) -> None:
        text = "[2024-08-26T09:01:00Z] Dorothy Harris logged in."
        assert "2024-08-26T09:01:00Z" in _spans(text, "DATE_TIME")

    def test_offset_timezone_form(self) -> None:
        text = "Exam scheduled 2021-01-06T08:34:00+02:00 at the clinic."
        assert "2021-01-06T08:34:00+02:00" in _spans(text, "DATE_TIME")


class TestDottedDateFalsePositives:
    def test_ip_address_fragments_are_not_dates(self) -> None:
        text = "Dorothy Harris logged in from 208.74.38.190. Session: ea8378a4"
        assert _spans(text, "DATE_TIME") == [], _spans(text, "DATE_TIME")

    def test_ip_fragment_leading_octets_are_not_dates(self) -> None:
        text = "access from 250.17.56.65 (MAC: e1:f9:60:99:f4:39)"
        assert _spans(text, "DATE_TIME") == [], _spans(text, "DATE_TIME")

    def test_plain_slash_date_still_detected(self) -> None:
        text = "The hearing happened on 03/15/2024 in court."
        assert "03/15/2024" in _spans(text, "DATE_TIME")


class TestDobArbitration:
    def test_dob_context_date_is_dob_not_datetime(self) -> None:
        text = "[10:09 AM] Carol: Confirmed. DOB: 12/28/1966."
        assert "12/28/1966" in _spans(text, "DATE_OF_BIRTH")
        # The generic date pattern must not double-emit the same span as
        # DATE_TIME — gold types are exclusive and the duplicate is an FP.
        assert "12/28/1966" not in _spans(text, "DATE_TIME")
