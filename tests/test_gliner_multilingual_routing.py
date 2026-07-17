"""Language/script-aware GLiNER model routing (round-2 structural change).

MEASURED basis (2026-07-17, home multilingual test sample, 960 records / 8
languages): base checkpoint PERSON relaxed-recall 0.505 vs the multilingual
companion (urchade/gliner_multi_pii-v1) at 0.965 WITH ITS NATIVE label
vocabulary — he 0.15->1.0, el 0.53->1.0, ar 0.58->1.0, zh 0.36->0.94,
th 0.0->0.85. GLiNER is prompt-conditioned: each checkpoint family needs its
own label vocabulary (the earlier flat-looking checkpoint comparison prompted
urchade models with knowledgator labels and got near-zero output).

Routing contract: declared non-English language OR predominantly non-Latin
text -> multilingual companion; English/Latin text NEVER leaves the base
model (EN behavior byte-identical by construction).
"""
from __future__ import annotations

from pii_anon.engines.gliner_adapter import GLiNERAdapter


def _adapter(**kw) -> GLiNERAdapter:
    return GLiNERAdapter(enabled=False, **kw)


class TestRuntimeSelection:
    def test_english_language_uses_base(self) -> None:
        kind, labels, label_map = _adapter()._runtime_for("en", "Alice met Bob at the office.")
        assert kind == "base"
        assert "name" in labels and label_map["name"] == "PERSON_NAME"

    def test_declared_nonenglish_language_uses_multi(self) -> None:
        kind, labels, label_map = _adapter()._runtime_for("he", "שלום עולם")
        assert kind == "multi"
        assert "person" in labels and label_map["person"] == "PERSON_NAME"

    def test_nonlatin_script_fallback_when_language_missing(self) -> None:
        # language defaults to "en" upstream when absent — the script probe
        # must still route Thai text to the companion.
        kind, _, _ = _adapter()._runtime_for("en", "ติดต่อ สมชาย ทองดี ได้ที่สำนักงาน")
        assert kind == "multi"

    def test_latin_text_never_leaves_base_even_with_odd_language(self) -> None:
        # a Latin-script value under a declared non-EN language: declared
        # language wins (multi handles Latin languages well per its card).
        kind, _, _ = _adapter()._runtime_for("de", "Herr Schmidt wohnt in Berlin.")
        assert kind == "multi"

    def test_ascii_with_en_stays_base(self) -> None:
        kind, _, _ = _adapter()._runtime_for("en", "a" * 500)
        assert kind == "base"

    def test_empty_text_stays_base(self) -> None:
        kind, _, _ = _adapter()._runtime_for("en", "")
        assert kind == "base"


class TestDisableSwitch:
    def test_constructor_off_disables_routing(self) -> None:
        a = _adapter(multilingual_model_name="off")
        kind, _, _ = a._runtime_for("he", "שלום עולם")
        assert kind == "base"
        assert a.capabilities().supports_languages == ["en"]

    def test_env_off_disables_routing(self, monkeypatch) -> None:
        monkeypatch.setenv("PII_ANON_GLINER_MULTI_MODEL", "off")
        a = _adapter()
        kind, _, _ = a._runtime_for("th", "ทดสอบ")
        assert kind == "base"

    def test_custom_multi_model_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PII_ANON_GLINER_MULTI_MODEL", "urchade/gliner_multi-v2.1")
        a = _adapter()
        assert a._multi_model_name == "urchade/gliner_multi-v2.1"


class TestCapabilities:
    def test_multilingual_capabilities_declared(self) -> None:
        caps = _adapter().capabilities()
        for lang in ("he", "zh", "th", "ar", "ru"):
            assert lang in caps.supports_languages
        assert "en" in caps.supports_languages


class TestScriptProbe:
    def test_mixed_mostly_latin_is_base(self) -> None:
        text = "The email of Иван is ivan@example.com and he works at Acme Corp in Berlin."
        assert GLiNERAdapter._predominantly_non_latin(text) is False

    def test_mostly_cyrillic_is_multi(self) -> None:
        assert GLiNERAdapter._predominantly_non_latin("Иван Петров работает в компании Яндекс") is True

    def test_digits_only_is_base(self) -> None:
        assert GLiNERAdapter._predominantly_non_latin("123-45-6789 555 0100") is False
