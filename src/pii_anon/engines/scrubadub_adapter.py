from __future__ import annotations

import re
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class ScrubadubAdapter(EngineAdapter):
    adapter_id = "scrubadub-compatible"
    native_dependency = "scrubadub"

    TITLE_NAME = re.compile(r"\b(Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\b")

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._scrubber: Any | None = None

    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities()
        caps.supports_languages = ["en"]
        return caps

    def _get_scrubber(self) -> Any | None:
        if self._scrubber is not None:
            return self._scrubber
        try:
            import importlib

            scrubadub = importlib.import_module("scrubadub")

            self._scrubber = scrubadub.Scrubber()
            return self._scrubber
        except Exception:
            return None

    def _fallback_detect(self, payload: Payload, language: str) -> list[EngineFinding]:
        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            for match in self.TITLE_NAME.finditer(value):
                findings.append(
                    EngineFinding(
                        entity_type="PERSON_NAME",
                        confidence=0.86,
                        field_path=key,
                        span_start=match.start(),
                        span_end=match.end(),
                        engine_id=self.adapter_id,
                        explanation="fallback scrubadub-compatible detector",
                        language=language,
                    )
                )
        return findings

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        if not self.enabled:
            return []

        language = str(context.get("language", "en")).lower()
        scrubber = self._get_scrubber()
        if scrubber is None:
            return self._fallback_detect(payload, language)

        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            try:
                filths = list(scrubber.iter_filth(value))
            except Exception:
                findings.extend(self._fallback_detect({key: value}, language))
                continue

            for filth in filths:
                findings.append(
                    EngineFinding(
                        entity_type=type(filth).__name__.upper(),
                        confidence=0.84,
                        field_path=key,
                        span_start=int(getattr(filth, "beg", 0)),
                        span_end=int(getattr(filth, "end", 0)),
                        engine_id=self.adapter_id,
                        explanation="scrubadub native",
                        language=language,
                    )
                )
        return findings
