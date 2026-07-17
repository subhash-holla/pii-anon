"""sp7 geo taxonomy — STATE/COUNTRY subtype as an ADVISORY signal.

A library-side type-split (emitting STATE/COUNTY/COUNTRY as the scored
entity_type) would drop the 1251 home LOCATION gold spans under strict
exact-type scoring (measured -0.0021 home F2) — a HARD-gate violation. So the
subtype is exposed on the finding's ``explanation`` while ``entity_type`` stays
LOCATION: the scored (type,start,end) tuple is byte-identical (home-neutral by
construction), and downstream masking/policy + the eval taxonomy relabel get a
library-native geo subtype signal.
"""
from __future__ import annotations

from pii_anon.engines.regex.geo_lexicon import geo_subtype
from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENGINE = RegexEngineAdapter(enabled=True)


class TestGeoSubtypeClassifier:
    def test_us_state(self) -> None:
        assert geo_subtype("California") == "STATE"
        assert geo_subtype("New York") == "STATE"

    def test_country(self) -> None:
        assert geo_subtype("France") == "COUNTRY"
        assert geo_subtype("United States") == "COUNTRY"

    def test_unknown_place_is_generic_location(self) -> None:
        assert geo_subtype("Springfield") == "LOCATION"

    def test_no_county_fabrication(self) -> None:
        # the lexicon has no county data — COUNTY is never returned
        assert geo_subtype("Orange") != "COUNTY"


class TestSubtypeOnFindings:
    def _locs(self, text: str) -> list:
        return [
            f for f in _ENGINE.detect({"text": text}, {"language": "en"})
            if str(f.entity_type) == "LOCATION"
        ]

    def test_entity_type_stays_location(self) -> None:
        # scored type is LOCATION for every geo finding (home-neutral)
        locs = self._locs("She lived in California and visited France.")
        assert locs and all(str(f.entity_type) == "LOCATION" for f in locs)

    def test_subtype_surfaced_on_explanation(self) -> None:
        locs = self._locs("She lived in California and visited France.")
        expl = {f.explanation for f in locs}
        assert any("subtype=STATE" in e for e in expl)
        assert any("subtype=COUNTRY" in e for e in expl)

    def test_scored_tuple_byte_identical_to_plain_location(self) -> None:
        # (type,start,end) is unaffected by the subtype annotation
        text = "based in California near France"
        tuples = {
            (str(f.entity_type), f.span_start, f.span_end)
            for f in self._locs(text)
        }
        # every scored tuple carries the LOCATION type (not STATE/COUNTRY)
        assert all(t[0] == "LOCATION" for t in tuples)
