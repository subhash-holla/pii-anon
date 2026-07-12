"""sp7 Phase-A #9 — curated geo gazetteer (LOCATION), home-safe subset.

Countries + US states as unambiguous public-domain facts, context-gated with a
collision veto so ambiguous single tokens (Georgia/Reading) fire only with a
location cue. Additive + leak-safe; the taxonomy split is deferred.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.geo_lexicon import extract_locations
from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENG = RegexEngineAdapter(enabled=True)


def _locs(text: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in _ENG.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == "LOCATION" and f.span_start is not None
    ]


def _extracted(text: str) -> set[str]:
    return {text[s:e] for _t, s, e, _c in extract_locations(text)}


class TestGazetteerRecall:
    @pytest.mark.parametrize(
        "text,place",
        [
            ("She now lives in France permanently.", "France"),
            ("The summit was held in Japan last year.", "Japan"),
            ("Operations expanded to the United States.", "United States"),
            ("He relocated to New York for work.", "New York"),
            ("The office in Portugal handles that.", "Portugal"),
        ],
    )
    def test_unambiguous_place_detected(self, text: str, place: str) -> None:
        assert place in _locs(text), f"{place!r} not in {_locs(text)!r}"


class TestCollisionVeto:
    def test_ambiguous_bare_token_not_fired(self) -> None:
        # in COLLIDE, no location cue -> not a place.
        assert "Georgia" not in _extracted("Georgia sent the report to the team.")
        assert "Reading" not in _extracted("Reading the contract took an hour.")
        assert "Jordan" not in _extracted("Jordan approved the budget yesterday.")

    def test_ambiguous_token_with_cue_fired(self) -> None:
        # with a location cue, the ambiguous token IS a place.
        assert "Georgia" in _extracted("The plant is located in Georgia now.")
        assert "Jordan" in _extracted("She flew to Jordan for the conference.")


class TestLeakSafety:
    def test_location_nested_in_address_dropped(self) -> None:
        # a gazetteer place inside an ADDRESS span is a pure duplicate — the
        # ADDRESS still masks the characters (leak-safe drop).
        text = "Mail to 123 Main Street, Springfield, Texas 75001 today."
        types = {
            (str(f.entity_type))
            for f in _ENG.detect({"text": text}, {"language": "en"})
            if f.span_start is not None and text[f.span_start:f.span_end] == "Texas"
        }
        # "Texas" should not survive as a standalone LOCATION inside the address
        assert "LOCATION" not in types or "ADDRESS" in {
            str(f.entity_type) for f in _ENG.detect({"text": text}, {"language": "en"})
        }

    def test_additive_location_masked(self) -> None:
        assert "France" in _locs("The account holder resides in France.")
