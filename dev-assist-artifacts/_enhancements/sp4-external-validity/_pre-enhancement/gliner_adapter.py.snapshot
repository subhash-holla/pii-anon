from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class GLiNERAdapter(EngineAdapter):
    """Engine adapter for GLiNER PII detection.

    Uses the ``knowledgator/gliner-pii-base-v1.0`` model for span-based
    named-entity recognition over a curated set of PII labels.

    When the native ``gliner`` package is unavailable the adapter falls back
    to lightweight regex patterns for common PII types.
    """

    adapter_id = "gliner-compatible"
    native_dependency = "gliner"

    _PII_LABELS = [
        "name",
        "email address",
        "phone number",
        "credit card number",
        "social security number",
        "date of birth",
        "address",
        "passport number",
        "driver's license number",
        "identity card number",
        "bank account number",
        "username",
        "password",
        "ip address",
    ]

    _LABEL_MAP: dict[str, str] = {
        "name": "PERSON_NAME",
        "first name": "PERSON_NAME",
        "last name": "PERSON_NAME",
        "email address": "EMAIL_ADDRESS",
        "phone number": "PHONE_NUMBER",
        "credit card number": "CREDIT_CARD",
        "social security number": "US_SSN",
        "date of birth": "DATE_OF_BIRTH",
        "address": "ADDRESS",
        "passport number": "PASSPORT",
        "driver's license number": "DRIVERS_LICENSE",
        "identity card number": "NATIONAL_ID",
        "bank account number": "BANK_ACCOUNT",
        "username": "USERNAME",
        "password": "PASSWORD",
        "ip address": "IP_ADDRESS",
    }

    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._model: Any | None = None
        self._native_import_probe_ok: bool | None = None

    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities()
        caps.supports_languages = ["en"]
        return caps

    def _probe_native_import(self) -> bool:
        if self._native_import_probe_ok is not None:
            return self._native_import_probe_ok
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "from gliner import GLiNER"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception:
            self._native_import_probe_ok = False
            return False
        self._native_import_probe_ok = proc.returncode == 0
        return self._native_import_probe_ok

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self._probe_native_import():
            return None
        try:
            import importlib
            import warnings

            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            GLiNER = importlib.import_module("gliner").GLiNER
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
                self._model = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")
            return self._model
        except Exception:
            return None

    def _fallback_detect(self, payload: Payload, language: str) -> list[EngineFinding]:
        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            for match in self.SSN.finditer(value):
                findings.append(
                    EngineFinding(
                        entity_type="US_SSN",
                        confidence=0.9,
                        field_path=key,
                        span_start=match.start(),
                        span_end=match.end(),
                        engine_id=self.adapter_id,
                        explanation="fallback gliner ssn",
                        language=language,
                    )
                )
            for match in self.EMAIL.finditer(value):
                findings.append(
                    EngineFinding(
                        entity_type="EMAIL_ADDRESS",
                        confidence=0.82,
                        field_path=key,
                        span_start=match.start(),
                        span_end=match.end(),
                        engine_id=self.adapter_id,
                        explanation="fallback gliner email",
                        language=language,
                    )
                )
        return findings

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        if not self.enabled:
            return []

        language = str(context.get("language", "en")).lower()
        model = self._load_model()
        if model is None:
            return self._fallback_detect(payload, language)

        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            try:
                entities = model.predict_entities(value, self._PII_LABELS, threshold=0.5)
            except Exception:
                findings.extend(self._fallback_detect({key: value}, language))
                continue

            for entity in entities:
                label = str(entity.get("label", "UNKNOWN")).lower()
                mapped_type = self._LABEL_MAP.get(label, label.upper())
                findings.append(
                    EngineFinding(
                        entity_type=mapped_type,
                        confidence=float(entity.get("score", 0.75)),
                        field_path=key,
                        span_start=int(entity.get("start", 0)),
                        span_end=int(entity.get("end", 0)),
                        engine_id=self.adapter_id,
                        explanation="gliner native ner",
                        language=language,
                    )
                )
        return findings
