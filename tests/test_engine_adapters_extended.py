from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from pii_anon.engines import (
    EngineAdapter,
    GLiNERAdapter,
    LLMGuardAdapter,
    PresidioAdapter,
    ScrubadubAdapter,
    SpacyNERAdapter,
    StanzaNERAdapter,
)
from pii_anon.types import EngineFinding, Payload


class DummyBaseEngine(EngineAdapter):
    adapter_id = "dummy"
    native_dependency = "definitely_not_installed_dependency"

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        return []


def test_engine_base_health_and_capabilities() -> None:
    engine = DummyBaseEngine(enabled=True)
    caps = engine.capabilities()
    assert caps.adapter_id == "dummy"
    assert caps.dependency_available is False

    health = engine.health_check()
    assert health["healthy"] is True
    assert health["details"] == "fallback"

    engine.initialize({"enabled": False})
    disabled_health = engine.health_check()
    assert disabled_health["details"] == "disabled"


def test_presidio_adapter_fallback_and_native_paths(monkeypatch) -> None:
    adapter = PresidioAdapter(enabled=True)
    adapter.initialize({"entities": ["EMAIL_ADDRESS"]})

    monkeypatch.setattr(adapter, "_get_analyzer", lambda: None)
    fallback = adapter.detect({"text": "Contact alice@example.com"}, {"language": "en"})
    assert any(item.entity_type == "EMAIL_ADDRESS" for item in fallback)

    class NativeResult:
        entity_type = "PHONE_NUMBER"
        score = 0.77
        start = 8
        end = 20

    class NativeAnalyzer:
        def analyze(self, text: str, language: str, entities: list[str] | None = None) -> list[NativeResult]:
            _ = text, language, entities
            return [NativeResult()]

    monkeypatch.setattr(adapter, "_get_analyzer", lambda: NativeAnalyzer())
    native = adapter.detect({"text": "x" * 40}, {"language": "en"})
    assert native[0].entity_type == "PHONE_NUMBER"

    class BrokenAnalyzer:
        def analyze(self, text: str, language: str, entities: list[str] | None = None) -> list[NativeResult]:
            _ = text, language, entities
            raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_get_analyzer", lambda: BrokenAnalyzer())
    recovered = adapter.detect({"text": "alice@example.com"}, {"language": "en"})
    assert any(item.entity_type == "EMAIL_ADDRESS" for item in recovered)


def test_scrubadub_adapter_fallback_and_native_paths(monkeypatch) -> None:
    adapter = ScrubadubAdapter(enabled=True)

    monkeypatch.setattr(adapter, "_get_scrubber", lambda: None)
    fallback = adapter.detect({"text": "Dr Smith"}, {"language": "en"})
    assert any(item.entity_type == "PERSON_NAME" for item in fallback)

    class Filth:
        beg = 0
        end = 8

    class NativeScrubber:
        def iter_filth(self, text: str) -> list[Filth]:
            _ = text
            return [Filth()]

    monkeypatch.setattr(adapter, "_get_scrubber", lambda: NativeScrubber())
    native = adapter.detect({"text": "Dr Smith"}, {"language": "en"})
    assert native[0].entity_type == "FILTH"

    class BrokenScrubber:
        def iter_filth(self, text: str) -> list[Filth]:
            _ = text
            raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_get_scrubber", lambda: BrokenScrubber())
    recovered = adapter.detect({"text": "Mr Davis"}, {"language": "en"})
    assert any(item.entity_type == "PERSON_NAME" for item in recovered)


def test_llm_guard_adapter_fallback_and_native_paths(monkeypatch) -> None:
    adapter = LLMGuardAdapter(enabled=True)

    monkeypatch.setattr(adapter, "_load_native_scanner", lambda: None)
    fallback = adapter.detect({"text": "SSN 123-45-6789 and alice@example.com"}, {"language": "en"})
    types = {item.entity_type for item in fallback}
    assert "US_SSN" in types
    assert "EMAIL_ADDRESS" in types

    class ScannerTuple:
        def scan(self, value: str) -> tuple[str, bool, float]:
            _ = value
            return ("masked", False, 0.1)

    monkeypatch.setattr(adapter, "_load_native_scanner", lambda: ScannerTuple())
    flagged = adapter.detect({"text": "hello"}, {"language": "en"})
    assert flagged and flagged[0].entity_type == "SENSITIVE_PII"

    class ScannerBool:
        def scan(self, value: str) -> bool:
            _ = value
            return True

    monkeypatch.setattr(adapter, "_load_native_scanner", lambda: ScannerBool())
    clean = adapter.detect({"text": "hello"}, {"language": "en"})
    assert clean == []

    class BrokenScanner:
        def scan(self, value: str) -> bool:
            _ = value
            raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_load_native_scanner", lambda: BrokenScanner())
    recovered = adapter.detect({"text": "alice@example.com"}, {"language": "en"})
    assert any(item.entity_type == "EMAIL_ADDRESS" for item in recovered)


def test_spacy_adapter_fallback_and_native_paths(monkeypatch) -> None:
    adapter = SpacyNERAdapter(enabled=True)

    monkeypatch.setattr(adapter, "_get_nlp", lambda: None)
    fallback = adapter.detect({"text": "Contact alice@example.com"}, {"language": "en"})
    assert any(item.entity_type == "EMAIL_ADDRESS" for item in fallback)

    class FakeEnt:
        label_ = "PERSON"
        start_char = 8
        end_char = 13

    class FakeDoc:
        ents = [FakeEnt()]

    class FakeNLP:
        def __call__(self, text: str) -> FakeDoc:
            _ = text
            return FakeDoc()

    monkeypatch.setattr(adapter, "_get_nlp", lambda: FakeNLP())
    native = adapter.detect({"text": "hello Alice"}, {"language": "en"})
    assert native and native[0].entity_type == "PERSON_NAME"

    disabled = SpacyNERAdapter(enabled=False)
    assert disabled.detect({"text": "alice@example.com"}, {"language": "en"}) == []


def test_spacy_adapter_import_failure_returns_none(monkeypatch) -> None:
    adapter = SpacyNERAdapter(enabled=True)
    monkeypatch.setattr(adapter, "_nlp", None)
    monkeypatch.setitem(sys.modules, "spacy", ModuleType("spacy"))
    assert adapter._get_nlp() is None


def test_stanza_adapter_fallback_and_native_paths(monkeypatch) -> None:
    adapter = StanzaNERAdapter(enabled=True)

    monkeypatch.setattr(adapter, "_get_pipeline", lambda: None)
    fallback = adapter.detect({"text": "Call 415-555-0100"}, {"language": "en"})
    assert any(item.entity_type == "PHONE_NUMBER" for item in fallback)

    class FakeEnt:
        type = "PER"
        start_char = 6
        end_char = 10

    class FakeDoc:
        entities = [FakeEnt()]

    class FakePipeline:
        def __call__(self, text: str) -> FakeDoc:
            _ = text
            return FakeDoc()

    monkeypatch.setattr(adapter, "_get_pipeline", lambda: FakePipeline())
    native = adapter.detect({"text": "Name Jack"}, {"language": "en"})
    assert native and native[0].entity_type == "PERSON_NAME"

    disabled = StanzaNERAdapter(enabled=False)
    assert disabled.detect({"text": "415-555-0100"}, {"language": "en"}) == []


def test_stanza_adapter_import_failure_returns_none(monkeypatch) -> None:
    adapter = StanzaNERAdapter(enabled=True)
    module = ModuleType("stanza")
    monkeypatch.setitem(sys.modules, "stanza", module)
    monkeypatch.setattr(
        module,
        "Pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )
    assert adapter._get_pipeline() is None


def test_gliner_adapter_fallback_and_native_paths(monkeypatch) -> None:
    adapter = GLiNERAdapter(enabled=True)

    # Fallback path: model unavailable → regex fallback
    monkeypatch.setattr(adapter, "_load_model", lambda: None)
    fallback = adapter.detect(
        {"text": "SSN 123-45-6789 and alice@example.com"}, {"language": "en"}
    )
    types = {item.entity_type for item in fallback}
    assert "US_SSN" in types
    assert "EMAIL_ADDRESS" in types

    # Native path: model returns entity dicts
    class FakeModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict[str, Any]]:
            _ = text, labels, threshold
            return [
                {"label": "name", "start": 0, "end": 10, "score": 0.92},
                {"label": "email address", "start": 15, "end": 35, "score": 0.88},
            ]

    monkeypatch.setattr(adapter, "_load_model", lambda: FakeModel())
    native = adapter.detect({"text": "Jack Davis alice@example.com"}, {"language": "en"})
    assert len(native) == 2
    assert native[0].entity_type == "PERSON_NAME"
    assert native[1].entity_type == "EMAIL_ADDRESS"

    # Broken model path: falls back to regex
    class BrokenModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict[str, Any]]:
            raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_load_model", lambda: BrokenModel())
    recovered = adapter.detect({"text": "alice@example.com"}, {"language": "en"})
    assert any(item.entity_type == "EMAIL_ADDRESS" for item in recovered)

    # Disabled path: returns nothing
    disabled = GLiNERAdapter(enabled=False)
    assert disabled.detect({"text": "alice@example.com"}, {"language": "en"}) == []


def test_gliner_adapter_capabilities() -> None:
    adapter = GLiNERAdapter(enabled=False)
    caps = adapter.capabilities()
    assert caps.adapter_id == "gliner-compatible"
    assert "en" in caps.supports_languages


