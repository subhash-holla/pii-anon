from __future__ import annotations

import re
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class StanzaNERAdapter(EngineAdapter):
    adapter_id = "stanza-ner-compatible"
    native_dependency = "stanza"

    PHONE = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)")

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._pipeline: Any | None = None

    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities()
        caps.supports_languages = ["en"]
        return caps

    def _get_pipeline(self) -> Any | None:
        if self._pipeline is not None:
            return self._pipeline

        try:
            import importlib

            stanza = importlib.import_module("stanza")

            self._pipeline = stanza.Pipeline(
                "en",
                processors="tokenize,ner",
                verbose=False,
                download_method=None,
            )
            return self._pipeline
        except Exception:
            return None

    def _fallback_detect(self, payload: Payload, language: str) -> list[EngineFinding]:
        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            for match in self.PHONE.finditer(value):
                findings.append(
                    EngineFinding(
                        entity_type="PHONE_NUMBER",
                        confidence=0.82,
                        field_path=key,
                        span_start=match.start(),
                        span_end=match.end(),
                        engine_id=self.adapter_id,
                        explanation="fallback stanza-compatible phone",
                        language=language,
                    )
                )
        return findings

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        if not self.enabled:
            return []

        language = str(context.get("language", "en")).lower()
        pipeline = self._get_pipeline()
        if pipeline is None:
            return self._fallback_detect(payload, language)

        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            doc = pipeline(value)
            for ent in getattr(doc, "entities", []):
                mapped_type = {
                    "PER": "PERSON_NAME",
                    "PERSON": "PERSON_NAME",
                    "ORG": "ORGANIZATION",
                    "LOC": "LOCATION",
                }.get(str(getattr(ent, "type", "UNKNOWN")), str(getattr(ent, "type", "UNKNOWN")))
                start = int(getattr(ent, "start_char", 0))
                end = int(getattr(ent, "end_char", 0))
                findings.append(
                    EngineFinding(
                        entity_type=mapped_type,
                        confidence=0.8,
                        field_path=key,
                        span_start=start,
                        span_end=end,
                        engine_id=self.adapter_id,
                        explanation="stanza native ner",
                        language=language,
                    )
                )
        return findings
