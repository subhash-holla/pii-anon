"""sp4 external-validity fix — GLiNER long-document windowing.

Root cause (measured on real TAB/ECHR court documents, 2026-07-10): the
GLiNER model's effective detection collapses with input length — on a real
judgment text the adapter returned 3 findings on the first 500 chars, 1 on
the first 1,000, and ZERO at >=2,000 chars — so every long document silently
lost ALL NER contribution (the home corpus's short records never exposed
this). The adapter must window long inputs (whitespace-aligned chunks with
overlap), re-base offsets, and de-duplicate — additive-only (leak-safe).
"""
from __future__ import annotations

import pytest

pytest.importorskip("gliner")

from pii_anon.engines.gliner_adapter import GLiNERAdapter

_ENGINE = GLiNERAdapter(enabled=True)

# Filler paragraph with no PII (legal-register prose).
_FILLER = (
    "The court considered the procedural history of the application at "
    "length, having regard to the relevant provisions of the Convention "
    "and the established case-law on admissibility criteria. "
)


def _detect(text: str) -> list[tuple[str, int, int]]:
    return [
        (str(f.entity_type), f.span_start, f.span_end)
        for f in _ENGINE.detect({"text": text}, {"language": "en"})
    ]


def test_person_deep_in_long_document_is_detected() -> None:
    """A name at ~6,000 chars depth must still be detected (was: 0 findings)."""
    text = _FILLER * 40 + "The applicant, Mr Kamil Prus, lodged the complaint. " + _FILLER * 10
    name_at = text.index("Kamil Prus")
    assert name_at > 5000, "fixture must place the name deep in the text"
    found = _detect(text)
    person_spans = [(s, e) for lbl, s, e in found if lbl == "PERSON_NAME"]
    assert any(s <= name_at < e for s, e in person_spans), (
        f"PERSON_NAME at offset {name_at} not detected in a "
        f"{len(text)}-char document; got {found[:8]!r}"
    )


def test_offsets_are_rebased_to_the_full_text() -> None:
    """Windowed offsets must index into the ORIGINAL text exactly."""
    text = _FILLER * 40 + "Contact sarah.connor@example.com for details. " + _FILLER * 10
    for lbl, s, e in _detect(text):
        if lbl == "EMAIL_ADDRESS":
            assert text[s:e] == "sarah.connor@example.com"
            break
    else:
        pytest.fail("EMAIL_ADDRESS deep in a long document not detected")


def test_short_text_behavior_unchanged() -> None:
    """Short inputs (below one window) keep the single-call path."""
    found = _detect("John Smith lives in Paris.")
    assert any(lbl == "PERSON_NAME" for lbl, _, _ in found)


def test_no_duplicate_spans_from_window_overlap() -> None:
    """A finding inside the overlap region must not be emitted twice."""
    text = _FILLER * 40 + "The witness, Maria Fernandez, testified. " + _FILLER * 40
    found = _detect(text)
    assert len(found) == len(set(found)), f"duplicate spans emitted: {found!r}"
