"""sp7 Phase-A A3 — natural-language / locale date grammar + DOB-cue promotion
(sp6 mining candidate #5).

The largest single external-recall class: prose dates the ASCII "Month d, yyyy"
+ numeric grammar misses. TAB/ECHR court prose is dominated by day-first and
legal ordinal forms — "27 May 1994", "1st day of January, 2023", "May 1994" —
which are 79-82% of ALL TAB DATETIME gold.

Discipline:
  * ADDITIVE — new DATE_TIME emits; DATE_TIME is benchmark-ignored on HOME
    scoring (home-neutral) but masked in production (leak-safe over-mask) and
    scored on externals (the recall win).
  * DOB promotion is a leak-safe RELABEL — a DATE_TIME/DATE_ISO within ~30
    chars after a birth cue ("born", "DOB", "date of birth") becomes
    DATE_OF_BIRTH. Cue-gated for precision; the span stays masked either way.
  * boundary hygiene — the span is the date only, never a leading determiner
    ("no later than March 20, 2023" -> "March 20, 2023").
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENG = RegexEngineAdapter(enabled=True)


def _find(text: str, etype: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in _ENG.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == etype and f.span_start is not None
    ]


def _types(text: str) -> set[str]:
    return {str(f.entity_type) for f in _ENG.detect({"text": text}, {"language": "en"})}


class TestProseDateGrammar:
    @pytest.mark.parametrize(
        "text,value",
        [
            ("The hearing was held on 27 May 1994 in chambers.", "27 May 1994"),
            ("Signed the 1st day of January, 2023 by both parties.", "1st day of January, 2023"),
            ("Payments began in May 1994 and continued.", "May 1994"),
            ("Filed on 3rd of April 1980 with the registry.", "3rd of April 1980"),
            ("Delivered on 15 January 2021 by courier.", "15 January 2021"),
            ("The order dated 7 Sept 2019 remains in force.", "7 Sept 2019"),
        ],
    )
    def test_prose_date_detected_as_datetime(self, text: str, value: str) -> None:
        got = _find(text, "DATE_TIME")
        assert any(value == g for g in got), f"{value!r} not in DATE_TIME spans {got!r}"

    def test_boundary_hygiene_no_leading_determiner(self) -> None:
        got = _find("Submit no later than March 20, 2023 please.", "DATE_TIME")
        assert "March 20, 2023" in got
        assert not any(g.startswith(("no ", "later", "than")) for g in got)

    def test_month_word_not_overmatched(self) -> None:
        # "May"/"March" as non-date words must not manufacture a date span.
        assert _find("The Mayor spoke at the March for rights.", "DATE_TIME") == []


class TestDobPromotion:
    def test_prose_dob_promoted_by_cue(self) -> None:
        # a prose date within ~30 chars after a birth cue becomes DATE_OF_BIRTH.
        got = _find("The patient was born on 27 May 1994 at County General.", "DATE_OF_BIRTH")
        assert "27 May 1994" in got, f"prose DOB not promoted: {got!r}"

    def test_iso_dob_promoted_by_cue(self) -> None:
        got = _find("Employee record — born 1994-05-27, active.", "DATE_OF_BIRTH")
        assert any("1994-05-27" in g for g in got)

    def test_uncued_prose_date_stays_datetime(self) -> None:
        # no birth cue nearby -> stays the general date type (no false DOB).
        assert _find("The contract was signed on 27 May 1994 downtown.", "DATE_OF_BIRTH") == []
        assert "27 May 1994" in _find("The contract was signed on 27 May 1994 downtown.", "DATE_TIME")

    def test_reborn_does_not_promote(self) -> None:
        # word-bounded cue: "reborn" must not trigger DOB promotion.
        assert _find("The brand was reborn in May 1994 with new owners.", "DATE_OF_BIRTH") == []


class TestLeakSafetyAdditive:
    def test_datetime_additions_are_masked_in_production(self) -> None:
        # additive: the prose date is emitted (and thus masked) — over-masking a
        # date is the safe direction.
        assert "DATE_TIME" in _types("Meeting on 27 May 1994.") or \
               "DATE_OF_BIRTH" in _types("Meeting on 27 May 1994.")
