"""sp7 #6 — surname mention-propagation (additive detection post-pass).

A multi-token PERSON_NAME ("Alastair Fennimore") is detected once, but a later
bare "Fennimore" the grammar missed leaks. This post-pass propagates the
detected surname to every standalone verbatim occurrence in the same field,
emitting a new PERSON_NAME span. ADDITIVE (emits only new, non-overlapping
spans; never drops/narrows) => leak-safe; masks MORE partial mentions.

Measured full-50k home strict-F2: 0.9084 -> 0.9114 (+0.0030), PERSON_NAME
recall 0.8004 -> 0.8160; the HARD home no-regression gate passes with margin.
"""
from __future__ import annotations

from pii_anon.engines.regex_adapter import (
    RegexEngineAdapter,
    _propagate_surname_mentions,
    _standalone_occurrences,
)
from pii_anon.types import EngineFinding

_ENGINE = RegexEngineAdapter(enabled=True)


def _person(text: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in _ENGINE.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == "PERSON_NAME" and f.span_start is not None
    ]


def _pn(field: str, s: int, e: int, conf: float = 0.9) -> EngineFinding:
    return EngineFinding(
        entity_type="PERSON_NAME", confidence=conf, field_path=field,
        span_start=s, span_end=e, engine_id="regex-oss", language="en",
    )


class TestStandaloneOccurrences:
    def test_whole_word_only(self) -> None:
        # "Fennimore" inside "Fennimoreville" must NOT match
        text = "Fennimore met Fennimoreville and Fennimore again"
        occ = _standalone_occurrences(text, "Fennimore")
        assert occ == [(0, 9), (33, 42)]

    def test_possessive_is_a_boundary(self) -> None:
        text = "Fennimore's report"
        assert _standalone_occurrences(text, "Fennimore") == [(0, 9)]

    def test_absent_token(self) -> None:
        assert _standalone_occurrences("nothing here", "Fennimore") == []


class TestPropagateHelper:
    def _texts(self, t: str) -> dict:
        return {"f": t}

    def test_surname_propagated_to_bare_mention(self) -> None:
        text = "Alastair Fennimore signed. Later Fennimore left."
        findings = [_pn("f", 0, 18)]  # "Alastair Fennimore"
        out = _propagate_surname_mentions(
            findings, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        added = [f for f in out if f.explanation and "propagated" in f.explanation]
        assert len(added) == 1
        s, e = added[0].span_start, added[0].span_end
        assert text[s:e] == "Fennimore"
        assert added[0].entity_type == "PERSON_NAME"

    def test_additive_never_drops_existing(self) -> None:
        text = "Alastair Fennimore and Fennimore"
        findings = [_pn("f", 0, 18)]
        out = _propagate_surname_mentions(
            findings, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        # original finding is preserved verbatim, new ones only appended
        assert out[0] is findings[0]
        assert len(out) >= len(findings)

    def test_no_overlap_with_existing_finding(self) -> None:
        # the surname already covered by a finding is not re-emitted at that span
        text = "Alastair Fennimore and Fennimore"
        existing = [_pn("f", 0, 18), _pn("f", 23, 32)]  # both mentions already found
        out = _propagate_surname_mentions(
            existing, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        added = [f for f in out if f.explanation and "propagated" in f.explanation]
        assert added == []  # nothing to add; both already covered

    def test_stopword_surname_not_propagated(self) -> None:
        # "Green" is a common-word stopword — a bare "Green" is NOT propagated
        text = "John Green reviewed it. The Green field was empty."
        findings = [_pn("f", 0, 10)]  # "John Green"
        out = _propagate_surname_mentions(
            findings, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        added = [f for f in out if f.explanation and "propagated" in f.explanation]
        assert added == []

    def test_denylisted_surname_not_propagated(self) -> None:
        text = "Alastair Fennimore and Fennimore"
        findings = [_pn("f", 0, 18)]
        out = _propagate_surname_mentions(
            findings, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: s == "Fennimore",
        )
        assert [f for f in out if f.explanation and "propagated" in f.explanation] == []

    def test_short_surname_not_propagated(self) -> None:
        # surname len < 4 is below the propagation floor
        text = "Alastair Ito and Ito"
        findings = [_pn("f", 0, 12)]  # "Alastair Ito"
        out = _propagate_surname_mentions(
            findings, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        assert [f for f in out if f.explanation and "propagated" in f.explanation] == []

    def test_single_token_name_not_a_source(self) -> None:
        # a single-token PERSON_NAME has no surname to propagate
        text = "Fennimore and Fennimore"
        findings = [_pn("f", 0, 9)]
        out = _propagate_surname_mentions(
            findings, self._texts(text), adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        assert [f for f in out if f.explanation and "propagated" in f.explanation] == []

    def test_per_field_isolation(self) -> None:
        # a surname detected in field a does NOT propagate into field b
        findings = [_pn("a", 0, 18)]  # "Alastair Fennimore" in field a
        texts = {"a": "Alastair Fennimore", "b": "Fennimore in another field"}
        out = _propagate_surname_mentions(
            findings, texts, adapter_id="regex-oss",
            language="en", is_denied=lambda t, s: False,
        )
        added = [f for f in out if f.explanation and "propagated" in f.explanation]
        assert all(f.field_path == "a" for f in added)


class TestEndToEndAdapter:
    def test_bare_surname_now_detected(self) -> None:
        text = "Alastair Fennimore reviewed the case. Fennimore signed it later."
        spans = _person(text)
        # both the full name AND the bare surname mention are covered
        assert any("Alastair Fennimore" in s for s in spans)
        assert "Fennimore" in spans

    def test_additive_coverage_superset(self) -> None:
        # propagation ON must never remove a span the base engine detected
        off = RegexEngineAdapter(enabled=True)
        import pii_anon.engines.regex_adapter as ra
        text = "Alastair Fennimore reviewed. Fennimore then left the building."
        # spans with propagation ON (production default)
        on_spans = {(f.span_start, f.span_end) for f in off.detect({"text": text}, {"language": "en"}) if str(f.entity_type) == "PERSON_NAME"}
        # neutralize propagation to get the baseline span set
        orig = ra._propagate_surname_mentions
        ra._propagate_surname_mentions = lambda findings, texts, **kw: findings
        try:
            base_spans = {(f.span_start, f.span_end) for f in RegexEngineAdapter(enabled=True).detect({"text": text}, {"language": "en"}) if str(f.entity_type) == "PERSON_NAME"}
        finally:
            ra._propagate_surname_mentions = orig
        assert base_spans <= on_spans  # ON is a superset (additive)
