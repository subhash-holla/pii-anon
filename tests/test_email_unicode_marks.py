"""EMAIL local parts with combining marks (roadmap lever #4, 2026-07-17).

``\\w`` excludes category-M combining marks, so an abugida/harakat local part
(Thai vowel signs, Bengali matras, Arabic diacritics) terminated the match at
the first mark — every such email scored as BOTH a missed-detection FN and a
truncated-span FP (measured: 16,436 FN+FP pairs on the multilingual test
split, th 0.322 / bn 0.328 EMAIL F2). The fix widens the local-part
continuation class with the category-M ranges; the FIRST character class is
unchanged, so every previous match start is preserved and the change is
strictly additive (over-masking-safe leak direction).
"""
from __future__ import annotations

import pytest

from pii_anon.engines import RegexEngineAdapter


def _detect_emails(text: str) -> list[tuple[int, int, str]]:
    adapter = RegexEngineAdapter(enabled=True)
    findings = adapter.detect({"text": text}, {"language": "xx", "policy_mode": "balanced"})
    return sorted(
        (f.span_start, f.span_end, text[f.span_start:f.span_end])
        for f in findings
        if f.entity_type == "EMAIL_ADDRESS" and f.span_start is not None
    )


@pytest.mark.parametrize(
    "email",
    [
        "สมชาย.ทองดี88@outlook.com",  # Thai: SARA II (U+0E35, Mn) in the local part
        "মৌসুমী.দাস@example.com",  # Bengali: vowel signs (Mc/Mn) throughout
        "مُحمد@example.com",  # Arabic: DAMMA (U+064F, Mn) after the first char
        "예진.박86@outlook.com",  # Hangul (no marks) — the documented pre-fix case
        "alice@example.com",  # ASCII regression anchor
    ],
)
def test_email_with_combining_marks_is_detected_in_full(email: str) -> None:
    text = f"contact {email} today"
    spans = _detect_emails(text)
    assert spans, f"no EMAIL_ADDRESS detected in {text!r}"
    values = [v for _, _, v in spans]
    assert email in values, f"expected full-span {email!r}, got {values!r}"


def test_truncated_mark_split_span_is_not_emitted() -> None:
    """The old failure mode emitted the post-mark tail ('88@outlook.com')."""
    text = "ติดต่อ สมชาย.ทองดี88@outlook.com ครับ"
    spans = _detect_emails(text)
    values = [v for _, _, v in spans]
    assert "88@outlook.com" not in values
    assert "สมชาย.ทองดี88@outlook.com" in values


def test_ascii_email_span_is_byte_identical_to_before() -> None:
    text = "reach alice.smith+tag@example.co.uk or bob_1%x@mail.example.com now"
    spans = _detect_emails(text)
    assert [v for _, _, v in spans] == [
        "alice.smith+tag@example.co.uk",
        "bob_1%x@mail.example.com",
    ]
