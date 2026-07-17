"""Roadmap lever #8 arm 1 — court docket / application-number grammar.

TAB/ECHR-style case identifiers shaped ``no. 39272/98`` (application number
slash 2-digit year). The 2026-07-16 investigation measured ~328 of 357 TAB
CODE false-negatives are this shape — the single largest TAB CODE recall
class (~+0.04 TAB relaxed F2).

Grounding — REAL surface forms mined from the TAB DEV split ONLY
(external_eval/data/tab/echr_dev.json; the test split was never opened).
434 of 484 unique gold CODE spans in dev are this shape; application-part
digit census: 3 digits x6, 4 digits x23, 5 digits x405; year part always
exactly 2 digits. Cited dev examples:

  1. [001-83927]  "The case originated in an application (no. 40593/04)
     against the Republic of Turkey" — the dominant single-application
     form (117 of 434 dev dockets carry this exact lead-in).
  2. [001-86867]  "originated in an application (no. 287/03) against the
     Republic of Turkey" — 3-digit application part.
  3. [001-148610] "The case originated in two applications (nos. 31706/10
     and 33088/10) against ..." — plural "nos." + "and"-joined pair.
  4. [001-59864]  "raising the same complaints (applications nos.
     26480/95, 28291/95, 29280/95, ... 39428/96 and 43362/96)" — long
     comma-separated list with an "and"-joined final member.
  5. [001-67243]  "originated in an application (no. 515/02) against the
     United Kingdom" — 3-digit application part, bare "(no. ...)" cue.
  6. [001-79436]  "The case originated in an application (no. 63354/00)
     against the Republic of Turkey".

Discipline:
  * ADDITIVE — new COURT_CASE_NUMBER emits only; the existing _COURT_CASE /
    _DOCKET value alternations are hyphen-based and cannot match slash
    forms, so nothing is shadowed or narrowed. COURT_CASE_NUMBER is
    census-ignored on HOME scoring (home-neutral) but masked in production
    (leak-safe over-mask) and maps to CODE on TAB (the recall win).
  * Cue-gated — a bare "NNNNN/YY" never fires without a preceding
    "no." / "nos." / "application(s) no" cue (fractions, month/year dates,
    phone fragments stay clean); list members are gated on a preceding
    docket-shaped number + separator.
  * The application part requires >=3 digits (matches the full dev census)
    so date-like "12/98" cannot fire even when cued.
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


class TestSingleApplicationNumber:
    """Dev form 1/2/5/6: 'application (no. NNNNN/YY)'."""

    @pytest.mark.parametrize(
        "text,value",
        [
            # dev [001-83927] — the dominant lead-in (117x in dev)
            (
                "The case originated in an application (no. 40593/04) "
                "against the Republic of Turkey.",
                "40593/04",
            ),
            # dev [001-79436]
            (
                "The case originated in an application (no. 63354/00) "
                "against the Republic of Turkey.",
                "63354/00",
            ),
            # dev [001-86867] — 3-digit application part
            (
                "The case originated in an application (no. 287/03) "
                "against the Republic of Turkey.",
                "287/03",
            ),
            # dev [001-67243] — 3-digit application part
            (
                "The case originated in an application (no. 515/02) "
                "against the United Kingdom.",
                "515/02",
            ),
            # bare cue without the "application" word
            ("The judgment in the case (no. 39272/98) was final.", "39272/98"),
            # dotless "no" is accepted when the application word gates it
            ("She lodged application no 21773/02 with the Court.", "21773/02"),
            # capitalised cue
            ("Application No. 56745/00 was declared admissible.", "56745/00"),
        ],
    )
    def test_single_application_number_detected(self, text: str, value: str) -> None:
        got = _find(text, "COURT_CASE_NUMBER")
        assert value in got, f"{value!r} not in COURT_CASE_NUMBER spans {got!r}"

    def test_span_is_number_only(self) -> None:
        # boundary hygiene: the span is the identifier, never the cue.
        got = _find(
            "The case originated in an application (no. 40593/04) against Turkey.",
            "COURT_CASE_NUMBER",
        )
        assert "40593/04" in got
        assert not any("no." in g or "(" in g for g in got)


class TestApplicationNumberLists:
    """Dev forms 3/4: 'applications (nos. A, B, ... Y and Z)'."""

    def test_two_application_and_pair(self) -> None:
        # dev [001-148610]
        text = (
            "The case originated in two applications (nos. 31706/10 and "
            "33088/10) against the Republic of Turkey."
        )
        got = _find(text, "COURT_CASE_NUMBER")
        assert "31706/10" in got
        assert "33088/10" in got

    def test_long_comma_list_all_members(self) -> None:
        # dev [001-59864] (truncated to 6 members; same separators)
        text = (
            "raising the same complaints (applications nos. 26480/95, "
            "28291/95, 29280/95, 33645/96, 39428/96 and 43362/96) "
            "(Rule 43 § 2)."
        )
        got = _find(text, "COURT_CASE_NUMBER")
        for member in (
            "26480/95",
            "28291/95",
            "29280/95",
            "33645/96",
            "39428/96",
            "43362/96",
        ):
            assert member in got, f"list member {member!r} missing from {got!r}"

    def test_no_paren_plural_list(self) -> None:
        # dev [001-59864] variant: "applications nos." without the paren
        text = "applications nos. 29911/96, 29912/96 and 29913/96 were joined."
        got = _find(text, "COURT_CASE_NUMBER")
        assert {"29911/96", "29912/96", "29913/96"} <= set(got)


class TestNegativeCases:
    """Fractions, dates, phones, ISO dates, statute refs must NOT fire."""

    @pytest.mark.parametrize(
        "text",
        [
            # ordinary fraction — no cue
            "Add 3/4 of a cup of flour and stir.",
            # fraction even WITH a nearby 'no' word
            "There is no 3/4 majority requirement.",
            # month/year date alone — no cue
            "The invoice period 12/98 was disputed.",
            # month/year date WITH the cue — app part < 3 digits
            "See circular no. 12/98 of the ministry.",
            # date sequence — 2-digit members cannot chain
            "The statements covered 11/22, 33/44 and 55/66 splits.",
            # ISO date
            "The decision of 2005-10-28 was upheld.",
            # phone-like string
            "Call the registry on 0171 555 9898 for details.",
            # statute / article refs (dev non-docket CODE shapes we do NOT chase)
            "compensation under section 215(1) of the Land Act",
            "lodged with the Court under Article 34 of the Convention",
            # Turkish file-number shape (4-digit-year style; out of scope)
            "Civil Court of First Instance (file no. 2005/8415).",
        ],
    )
    def test_no_false_positive(self, text: str) -> None:
        assert _find(text, "COURT_CASE_NUMBER") == [], (
            f"unexpected COURT_CASE_NUMBER in {text!r}"
        )

    def test_list_continuation_requires_docket_head(self) -> None:
        # a comma-joined number pair with NO docket-shaped predecessor and no
        # cue must not fire ("33/44" head is 2-digit, can't seed a chain).
        assert _find("ratios 33/44, 39272/98 were tabulated.", "COURT_CASE_NUMBER") == []


class TestAdditiveNoShadowing:
    """The existing hyphen-based COURT_CASE / DOCKET grammars keep firing."""

    def test_us_federal_case_form_still_detected(self) -> None:
        got = _find("Case No. 1:21-cv-01234 was dismissed.", "COURT_CASE_NUMBER")
        assert "1:21-cv-01234" in got

    def test_state_year_form_still_detected(self) -> None:
        got = _find("filed as 2024-CIV-00123 in state court.", "COURT_CASE_NUMBER")
        assert "2024-CIV-00123" in got

    def test_docket_keyword_form_still_detected(self) -> None:
        got = _find("Docket No. 2024-CV-00123 remains open.", "DOCKET_NUMBER")
        assert "2024-CV-00123" in got
