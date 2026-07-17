"""sp7 Phase-A A4 — address grammar rework (sp6 mining candidate #10).

Two-tier evidence-gated grammar. The current pattern lists ~24 suffixes and
runs under global re.IGNORECASE, which makes the street-token class slurp
lowercase prose ("1997 the ... Court"). The rework:

  * drops global IGNORECASE (case-insensitivity re-applied ONLY to the suffix),
  * adds a first-token function-word guard,
  * tier-1: full-USPS unambiguous suffix set (ADDITIVE recall — Harbors, Route,
    Squares, Mountains, … — the largest gretel FN class),
  * tier-2: common-noun suffixes accepted ONLY with unit/postcode evidence.

Leak-direction: real addresses (any case, diacritics, lowercase particles) must
still be caught (no lost PII); only function-word-led prose is dropped.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENG = RegexEngineAdapter(enabled=True)


def _addr(text: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in _ENG.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == "ADDRESS" and f.span_start is not None
    ]


def _has_addr(text: str, needle: str) -> bool:
    return any(needle in a for a in _addr(text))


class TestTier1SuffixRecall:
    @pytest.mark.parametrize(
        "text,needle",
        [
            ("Ship to 31461 Matthew Harbors, Suite 297 please.", "31461 Matthew Harbors"),
            ("Delivered to 441 JULIE SQUARES on Monday.", "441 JULIE SQUARES"),
            ("The parcel went to 568 Rogers Route today.", "568 Rogers Route"),
            ("Return to 9714 Becker Mountains by Friday.", "9714 Becker Mountains"),
            ("Mail 7150 Soto Roads to the office.", "7150 Soto Roads"),
        ],
    )
    def test_expanded_suffix_addresses_detected(self, text: str, needle: str) -> None:
        assert _has_addr(text, needle), f"{needle!r} not in {_addr(text)!r}"


class TestTier2EvidenceGate:
    def test_ambiguous_suffix_with_evidence_detected(self) -> None:
        assert _has_addr("Invoice to 999 Katherine Locks, 34871, South Nicholasbury.", "999 Katherine Locks")
        assert _has_addr("Ship 2901 Michelle Spurs, Apt. 99508 overnight.", "2901 Michelle Spurs")

    def test_ambiguous_suffix_without_evidence_rejected(self) -> None:
        # bare "N <Title> <common-noun>" prose must NOT become an address.
        assert _addr("There are 3 Main Points to consider here.") == []
        assert _addr("We reviewed 5 Key Findings in the report.") == []


class TestPrecisionGuards:
    def test_function_word_led_prose_rejected(self) -> None:
        # dropping global IGNORECASE + the func-guard stops prose being read as
        # an address ("1997 the Regional Court" etc.).
        assert _addr("On 1997 the Regional Court reviewed the appeal.") == []
        assert _addr("In 1 of the Rules of Court it is stated.") == []


class TestLeakSafetyRealAddressesPreserved:
    @pytest.mark.parametrize(
        "text,needle",
        [
            ("She lives at 45 King's Road, London now.", "45 King's Road"),
            ("Mail to 123 de la Cruz Street downtown.", "123 de la Cruz Street"),
            ("Deliver to 123 main street tomorrow.", "123 main street"),
            ("The White House at 1600 Pennsylvania Avenue.", "1600 Pennsylvania Avenue"),
            ("Ship 123 Main St, Springfield, IL 62704 today.", "123 Main St"),
        ],
    )
    def test_real_addresses_still_detected(self, text: str, needle: str) -> None:
        assert _has_addr(text, needle), f"LOST real address {needle!r}: {_addr(text)!r}"
