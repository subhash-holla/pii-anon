from __future__ import annotations

import re
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class SpacyNERAdapter(EngineAdapter):
    adapter_id = "spacy-ner-compatible"
    native_dependency = "spacy"

    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._nlp: Any | None = None

    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities()
        caps.supports_languages = ["en"]
        return caps

    def _get_nlp(self) -> Any | None:
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
            return self._nlp
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
                        confidence=0.86,
                        field_path=key,
                        span_start=match.start(),
                        span_end=match.end(),
                        engine_id=self.adapter_id,
                        explanation="fallback spacy-compatible email",
                        language=language,
                    )
                )
        return findings

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        if not self.enabled:
            return []

        language = str(context.get("language", "en")).lower()
        nlp = self._get_nlp()
        if nlp is None:
            return self._fallback_detect(payload, language)

        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            doc = nlp(value)
            for ent in doc.ents:
                mapped_type = {
                    "PERSON": "PERSON_NAME",
                    "EMAIL": "EMAIL_ADDRESS",
                    "GPE": "LOCATION",
                    "ORG": "ORGANIZATION",
                }.get(str(ent.label_), str(ent.label_))
                findings.append(
                    EngineFinding(
                        entity_type=mapped_type,
                        confidence=0.82,
                        field_path=key,
                        span_start=int(ent.start_char),
                        span_end=int(ent.end_char),
                        engine_id=self.adapter_id,
                        explanation="spacy native ner",
                        language=language,
                    )
                )
        return findings
