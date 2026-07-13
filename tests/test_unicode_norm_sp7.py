"""sp7 panel (robustness) — unicode-normalization pre-pass with offset mapping.

Zero-width and fullwidth chars let PII evade the masking path. The pre-pass
normalizes text for DETECTION while remapping spans to ORIGINAL offsets, so the
masked region covers the exact original text INCLUDING interior obfuscation
chars. ASCII text is byte-identical (fast path); the original text is never
mutated.
"""
from __future__ import annotations

from pii_anon.engines.regex.unicode_norm import normalize_for_detection, remap_span
from pii_anon.engines.regex_adapter import RegexEngineAdapter

_PROD = RegexEngineAdapter(enabled=True)
_OFF = RegexEngineAdapter(enabled=True, unicode_normalize_detection=False)

_ZWSP = "​"
_BOM = "﻿"


def _spans(engine, text: str, etype: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == etype and f.span_start is not None
    ]


class TestNormalizer:
    def test_ascii_fast_path_identity(self) -> None:
        t = "SSN: 123-45-6789"
        scan, m = normalize_for_detection(t)
        assert scan is t and m is None

    def test_zero_width_stripped_map_covers_original(self) -> None:
        t = f"12{_ZWSP}3"
        scan, m = normalize_for_detection(t)
        assert scan == "123"
        # remapping the normalized "123" [0,3) covers the ZWSP too (orig [0,4))
        assert remap_span(m, 0, 3) == (0, 4)

    def test_fullwidth_folded(self) -> None:
        scan, _m = normalize_for_detection("１２３")  # fullwidth 123
        assert scan == "123"

    def test_diacritics_preserved_identity(self) -> None:
        # legitimate precomposed accents must not be mangled → identity map.
        scan, m = normalize_for_detection("café señor")
        assert scan == "café señor" and m is None


class TestEvasionHardening:
    def test_zero_width_ssn_fully_masked(self) -> None:
        text = f"SSN: 123{_ZWSP}-45-6789 today"
        spans = _spans(_PROD, text, "US_SSN")
        assert spans, "zero-width SSN evaded detection"
        # the emitted span covers the ORIGINAL region including the ZWSP
        assert "6789" in spans[0] and "123" in spans[0]
        # ...and it did evade the un-normalized engine (proves the fix bites)
        assert not any("6789" in s and "123" in s for s in _spans(_OFF, text, "US_SSN"))

    def test_fullwidth_phone_detected(self) -> None:
        text = "call ４１５-５５５-０１９８ now"  # 415-555-0198
        assert _PROD.detect({"text": text}, {"language": "en"}), "fullwidth phone evaded"

    def test_bom_email_detected(self) -> None:
        text = f"mail john{_BOM}.doe@example.com please"
        spans = _spans(_PROD, text, "EMAIL_ADDRESS")
        assert any("example.com" in s for s in spans), spans

    def test_original_text_untouched(self) -> None:
        # detection must never mutate the caller's payload text.
        text = f"SSN 123{_ZWSP}-45-6789"
        payload = {"text": text}
        _PROD.detect(payload, {"language": "en"})
        assert payload["text"] == text
