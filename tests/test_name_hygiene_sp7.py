"""sp7 Phase-A #6 — name grammar & span hygiene, SAFE subset (mining candidate #6).

Additive Unicode/honorific/initial PERSON patterns + eval-only span hygiene.
EXCLUDED by mandate (needs eval-data-owner sign-off): document-level mention
propagation, bare ALL-CAPS two-token names. Honorific patterns capture the name
SANS title (home convention — no home masking change). ALL-CAPS is gated to a
required middle-initial anchor so it cannot re-open the A1 Title-Case FP flood.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_PROD = RegexEngineAdapter(enabled=True)
_SCORE = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)


def _persons(engine, text: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == "PERSON_NAME" and f.span_start is not None
    ]


class TestTitleFullNameUnicode:
    @pytest.mark.parametrize(
        "text,name",
        [
            ("Represented by Mr Osman Çağlayan at the hearing.", "Osman Çağlayan"),
            ("Signed by Mr Bahattin Sarısoy yesterday.", "Bahattin Sarısoy"),
            ("Ms Véronique Olivier chaired the panel.", "Véronique Olivier"),
            ("Dr. Ivo Treves-Torlonia reviewed it.", "Ivo Treves-Torlonia"),
        ],
    )
    def test_title_full_name_sans_honorific(self, text: str, name: str) -> None:
        got = _persons(_PROD, text)
        assert any(name == g for g in got), f"{name!r} not in {got!r}"


class TestTitleInitials:
    @pytest.mark.parametrize(
        "text,name",
        [
            ("The witness Mr S. Esmer testified.", "S. Esmer"),
            ("Filed by Mr C.A. Whomersley today.", "C.A. Whomersley"),
            ("Ms J. Chrzanowska objected.", "J. Chrzanowska"),
        ],
    )
    def test_title_plus_initials_surname(self, text: str, name: str) -> None:
        got = _persons(_PROD, text)
        assert any(name == g for g in got), f"{name!r} not in {got!r}"


class TestUntitledMidInitial:
    @pytest.mark.parametrize(
        "text,name",
        [
            ("Reviewed by Timothée C. Rocher last week.", "Timothée C. Rocher"),
            ("Contact Kelly C. Moran for details.", "Kelly C. Moran"),
            ("Approved by SARAH D. JIMENEZ on file.", "SARAH D. JIMENEZ"),
        ],
    )
    def test_first_midinitial_last(self, text: str, name: str) -> None:
        got = _persons(_PROD, text)
        assert any(name == g for g in got), f"{name!r} not in {got!r}"

    def test_section_reference_not_a_name(self) -> None:
        # the section-word guard: "Section A. Overview" is not a person.
        assert not any("Overview" in p for p in _persons(_PROD, "See Section A. Overview below."))

    def test_bare_allcaps_two_token_not_a_name(self) -> None:
        # ALL-CAPS is gated to a mid-initial anchor; bare "JOHN SNOW" must NOT
        # fire (would re-open the A1 header FP flood).
        assert "PROJECT OVERVIEW" not in _persons(_PROD, "PROJECT OVERVIEW section follows.")


class TestDiacriticFullName:
    @pytest.mark.parametrize(
        "text,name",
        [
            ("The account holder Fabián Montalbán called.", "Fabián Montalbán"),
            ("Message from François Gilbert received.", "François Gilbert"),
        ],
    )
    def test_diacritic_full_name_detected(self, text: str, name: str) -> None:
        got = _persons(_PROD, text)
        assert any(name == g for g in got), f"{name!r} not in {got!r}"


class TestEvalOnlyHygiene:
    def test_salutation_trimmed_on_scoring_path(self) -> None:
        # "Ciao Nalda" -> the greeting is trimmed on the scoring path...
        text = "Ciao Nalda, come stai?"
        score = _persons(_SCORE, text)
        assert any(p == "Nalda" for p in score) or all("Ciao" not in p for p in score)

    def test_multilingual_article_dropped_on_scoring_path(self) -> None:
        # DE/FR/IT/ES leading articles are prose, not names (eval-only drop).
        text = "Die Gebühr wird erhoben."
        assert "Die Gebühr" not in _persons(_SCORE, text)
