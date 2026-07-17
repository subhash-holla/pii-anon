"""Roadmap lever #8 arm 2 — demonym / nationality-mention grammar.

TAB/ECHR-style demographic mentions shaped ``a <Demonym> national`` project
onto TAB's DEM gold class (NATIONALITY -> DEM in ds_tab PRED_MAP). Mirrors
the docket grammar (lever #8 arm 1, test_docket_grammar.py).

Grounding — REAL surface forms mined from the TAB DEV split ONLY
(external_eval/data/tab/echr_dev.json; the test split was never opened).
Dev DEM maskable gold census (identifier_type != NO_MASK, union over
annotators, deduped on (start, end)): 590 spans, 294 unique surfaces.
Shape classes: 151 single demonym words ("British" x46, "Swedish" x36,
"Turkish" x18, "Irish"/"Danish" x10 ...), 28 "X national(s)" phrases,
2 "X citizen(s)", plus multiword variants ("United Kingdom national" x10,
"Sierra Leonean national" x2); the remaining ~370 are OTHER DEM subclasses
(kinship "widows"/"father", occupations "Chief Constable", diagnoses
"arthritis") that this grammar deliberately does not chase. The cue-gated
grammar overlaps 161 of the 224 demonym/nationality-flavored dev golds
(72%); the 63 misses are BARE demonyms with no person cue ("the Swedish
company", "under Irish law") — the FP-bomb class the sp3 label-gating
lesson forbids chasing. Cited dev examples:

  1. [001-79012]  "by a British national, Mr Wayne Thomas Black (“the
     applicant”)" — the dominant lead-in; gold carries BOTH extents
     ("British" and "British national") via the annotator union.
  2. [001-114244] "by a Somali national, Mr Ilyas Elmi Hode, and a
     Djibouti national, Ms Hawa Aden Abdi" — country NAME as the
     modifier (no derived demonym exists for Djibouti in the text).
  3. [001-86146]  "by two British nationals, Ms J.M. Burden and Ms S.D.
     Burden" — plural cue noun.
  4. [001-57508]  "Roy H.W. Johnston, an Irish citizen,
     Janice Williams-Johnston, a British citizen" — citizen cue.
  5. [001-57508]  "Mr. B. Walsh, the elected judge of Irish nationality" —
     "nationality" right-noun; gold is the bare demonym "Irish".
  6. [001-106448] "by a Sierra Leonean national, Ms Husenatu Bah" —
     multiword demonym.
  7. [001-67466]  "by a United Kingdom national, Mr Ivan Hooper" —
     multiword country-name modifier.
  8. [001-91286]  "He is of Kurdish origin." — the "of X origin" frame
     (stateless ethnonym; the TAB guideline example class "a journalist
     of Roma origin").

Discipline:
  * ADDITIVE — new NATIONALITY emits only; the existing label-gated
    _NATIONALITY ("Nationality: American") is untouched and keeps firing.
    NATIONALITY is dropped by the production census filter
    (orchestrator.SUPPORTED_ENTITY_TYPES) but scored on the eval path.
  * Cue-gated — a bare demonym NEVER fires ("Turkish delight", "French
    press", "Dutch auction", "the Swedish company" stay clean); a fire
    requires a person cue: "<X> national(s)/citizen(s)/nationality" or
    "of <X> origin/descent/extraction".
  * Case-sensitive cue nouns — "British National Party" (capital N) does
    not fire.
  * Closed modifier lexicon (static demonym tuple + the geo_lexicon
    country gazetteer) for the person-cue form; the "of X origin" frame
    additionally accepts a single capitalized token ("Roma") because the
    frame itself is strongly demographic.
"""
from __future__ import annotations

import os
import time

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENG = RegexEngineAdapter(enabled=True)


def _find(text: str, etype: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in _ENG.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == etype and f.span_start is not None
    ]


class TestPersonCueForms:
    """Dev forms 1/2/3/4/6/7: '<Demonym|Country> national(s)/citizen(s)'."""

    @pytest.mark.parametrize(
        "text,value",
        [
            # dev [001-79012] — the dominant lead-in
            (
                "lodged with the Court by a British national, "
                "Mr Wayne Thomas Black (“the applicant”).",
                "British national",
            ),
            # dev [001-114244] — demonym + country-name modifier pair
            (
                "by a Somali national, Mr Ilyas Elmi Hode, and a "
                "Djibouti national, Ms Hawa Aden Abdi.",
                "Somali national",
            ),
            (
                "by a Somali national, Mr Ilyas Elmi Hode, and a "
                "Djibouti national, Ms Hawa Aden Abdi.",
                "Djibouti national",
            ),
            # dev [001-86146] — plural
            (
                "by two British nationals, Ms J.M. Burden and "
                "Ms S.D. Burden.",
                "British nationals",
            ),
            # dev [001-57508] — citizen cue
            (
                "Roy H.W. Johnston, an Irish citizen, and "
                "Janice Williams-Johnston, a British citizen.",
                "Irish citizen",
            ),
            # dev [001-106448] — multiword demonym
            (
                "by a Sierra Leonean national, Ms Husenatu Bah "
                "(“the applicant”).",
                "Sierra Leonean national",
            ),
            # dev [001-67466] — multiword country-name modifier
            (
                "by a United Kingdom national, Mr Ivan Hooper "
                "(“the applicant”).",
                "United Kingdom national",
            ),
            # dev [001-57508] — "nationality" right-noun
            (
                "Mr. B. Walsh, the elected judge of Irish nationality.",
                "Irish nationality",
            ),
            # dev [001-91286]-adjacent form: granted-nationality phrasing
            (
                "He and three of his sisters were granted British "
                "nationality.",
                "British nationality",
            ),
        ],
    )
    def test_person_cue_detected(self, text: str, value: str) -> None:
        got = _find(text, "NATIONALITY")
        assert value in got, f"{value!r} not in NATIONALITY spans {got!r}"

    def test_both_conjuncts_of_a_pair_detected(self) -> None:
        # dev [001-57508]: two independent cue sites in one sentence
        text = (
            "Roy H.W. Johnston, an Irish citizen, and "
            "Janice Williams-Johnston, a British citizen."
        )
        got = _find(text, "NATIONALITY")
        assert "Irish citizen" in got
        assert "British citizen" in got


class TestOfOriginForms:
    """Dev form 8: 'of <X> origin/descent' (open capitalized token)."""

    @pytest.mark.parametrize(
        "text,value",
        [
            # dev [001-91286]
            ("He is of Kurdish origin.", "Kurdish"),
            # the TAB guideline class ("a journalist of Roma origin") —
            # "Roma" is NOT in the demonym lexicon; the frame accepts it.
            ("The complaint concerned a journalist of Roma origin.", "Roma"),
            ("She is a British citizen of Indian descent.", "Indian"),
        ],
    )
    def test_of_origin_detected(self, text: str, value: str) -> None:
        got = _find(text, "NATIONALITY")
        assert value in got, f"{value!r} not in NATIONALITY spans {got!r}"


class TestNegativeCases:
    """Bare demonyms and lexicalized compounds must NOT fire."""

    @pytest.mark.parametrize(
        "text",
        [
            # lexicalized demonym compounds (the FP-bomb class)
            "She ordered a box of Turkish delight from the bazaar.",
            "He brewed coffee in a French press every morning.",
            "The estate was sold in a Dutch auction last spring.",
            # bare demonym + non-person noun (dev NO-chase class)
            "The programme was produced by the Swedish company Strix.",
            "There was no remedy under English law.",
            "He was a Lance Corporal in the British army.",
            # capitalized "National" — proper-noun compound, not a cue
            "He joined the British National Party in 1990.",
            "She deposited the cheque at First National Bank.",
            # cue noun with no demonym before it
            "The national anthem was played before the match.",
            "All citizens of the world deserve dignity.",
            # "origin" frames without a capitalized ethnonym
            "The fever was of unknown origin.",
            "The parties disputed the country of origin labelling.",
        ],
    )
    def test_no_false_positive(self, text: str) -> None:
        assert _find(text, "NATIONALITY") == [], (
            f"unexpected NATIONALITY in {text!r}"
        )


class TestAdditiveNoShadowing:
    """The existing label-gated NATIONALITY grammar keeps firing."""

    def test_labeled_nationality_still_detected(self) -> None:
        got = _find("Nationality: American", "NATIONALITY")
        assert "American" in got

    def test_geo_location_still_detected(self) -> None:
        # the geo gazetteer LOCATION channel is untouched
        got = _find("She was born in Germany and lives there.", "LOCATION")
        assert "Germany" in got


_REDOS_BLOCKS = [
    # repeated cue words with no demonym
    "national nationals nationality citizen citizens of ",
    # repeated demonyms with no cue
    "British Swedish Turkish Irish Danish Finnish Polish ",
    # near-miss cue chains (demonym + capitalized non-cue)
    "British National British Nationals of British Origin ",
    # dense REAL cue sites (worst-case many matches)
    "a British national, an Irish citizen, of Kurdish origin ",
]


def _pathological(block: str) -> str:
    return (block * (30_000 // len(block) + 1))[:30_000]


class TestReDoS:
    """Pathological 30k-char inputs: the NEW patterns must be linear.

    The adapter-level budget is asserted only on the blocks the PRE-change
    adapter already handled inside 2s. The dense bare-demonym block
    ("British Swedish Turkish ...") took 9.9s on the UNCHANGED adapter —
    a pre-existing hotspot in two ORGANIZATION patterns ("regex
    organization industry" 4.9s + "regex organization" 3.6s, measured
    2026-07-17) that this additive change must not be gated on; for that
    block the new patterns are timed directly.
    """

    @pytest.mark.parametrize("block", _REDOS_BLOCKS)
    def test_new_patterns_linear(self, block: str) -> None:
        from pii_anon.engines.regex.patterns import (
            _NATIONALITY_OF_ORIGIN,
            _NATIONALITY_PERSON_CUE,
        )

        text = _pathological(block)
        t0 = time.perf_counter()
        for pat in (_NATIONALITY_PERSON_CUE, _NATIONALITY_OF_ORIGIN):
            for _ in pat.finditer(text):
                pass
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"new patterns took {elapsed:.2f}s on {block[:30]!r}"

    @pytest.mark.parametrize("block", [_REDOS_BLOCKS[0], _REDOS_BLOCKS[2], _REDOS_BLOCKS[3]])
    def test_adapter_stays_within_budget(self, block: str) -> None:
        text = _pathological(block)
        t0 = time.perf_counter()
        _ENG.detect({"text": text}, {"language": "en"})
        elapsed = time.perf_counter() - t0
        # 2s is the single-process budget; under a parallel test run (xdist
        # workers + model loads) wall-clock stretches ~2-3x on shared cores —
        # same load-headroom treatment as the CI perf SLAs.
        budget = 2.0 if os.getenv("PYTEST_XDIST_WORKER") is None else 6.0
        assert elapsed < budget, f"detect took {elapsed:.2f}s on pathological input"
