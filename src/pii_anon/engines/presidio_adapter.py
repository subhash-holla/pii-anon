from __future__ import annotations

import re
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class PresidioAdapter(EngineAdapter):
    adapter_id = "presidio-compatible"
    native_dependency = "presidio_analyzer"

    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._analyzer: Any | None = None
        self._entities: list[str] | None = None

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        super().initialize(config)
        if config and isinstance(config.get("entities"), list):
            self._entities = [str(v) for v in config["entities"]]

    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities()
        caps.supports_languages = ["en", "es", "fr", "de", "it"]
        return caps

    def _get_analyzer(self) -> Any | None:
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Explicitly use en_core_web_sm to avoid downloading the 400 MB
            # en_core_web_lg model that Presidio defaults to.
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            return self._analyzer
        except Exception:
            return None

    def _fallback_detect(self, payload: Payload, language: str) -> list[EngineFinding]:
        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            for match in self.EMAIL.finditer(value):
                findings.append(
                    EngineFinding(
                        entity_type="EMAIL_ADDRESS",
                        confidence=0.9,
                        field_path=key,
                        span_start=match.start(),
                        span_end=match.end(),
                        engine_id=self.adapter_id,
                        explanation="fallback presidio-compatible regex",
                        language=language,
                    )
                )
        return findings

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        if not self.enabled:
            return []

        language = str(context.get("language", "en")).lower()
        analyzer = self._get_analyzer()
        if analyzer is None:
            return self._fallback_detect(payload, language)

        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            try:
                results = analyzer.analyze(
                    text=value,
                    language=language,
                    entities=self._entities,
                )
            except Exception:
                findings.extend(self._fallback_detect({key: value}, language))
                continue

            for item in results:
                findings.append(
                    EngineFinding(
                        entity_type=str(getattr(item, "entity_type", "UNKNOWN")),
                        confidence=float(getattr(item, "score", 0.7)),
                        field_path=key,
                        span_start=int(getattr(item, "start", 0)),
                        span_end=int(getattr(item, "end", 0)),
                        engine_id=self.adapter_id,
                        explanation="presidio native",
                        language=language,
                    )
                )
        return findings
