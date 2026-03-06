from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class LLMGuardAdapter(EngineAdapter):
    adapter_id = "llm-guard-compatible"
    native_dependency = "llm_guard"

    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled=enabled)
        self._native_scanner: Any | None = None
        self._native_import_probe_ok: bool | None = None

    def capabilities(self) -> EngineCapabilities:
        caps = super().capabilities()
        caps.supports_languages = ["en"]
        return caps

    def _load_native_scanner(self) -> Any | None:
        if self._native_scanner is not None:
            return self._native_scanner
        if not self._probe_native_import():
            return None

        try:
            import importlib

            input_scanners = importlib.import_module("llm_guard").input_scanners
        except Exception:
            return None

        for candidate in ("Sensitive", "Anonymize", "Regex"):
            scanner_cls = getattr(input_scanners, candidate, None)
            if scanner_cls is None:
                continue
            try:
                self._native_scanner = scanner_cls()
                return self._native_scanner
            except Exception:
                continue
        return None

    def _probe_native_import(self) -> bool:
        if self._native_import_probe_ok is not None:
            return self._native_import_probe_ok
        # Probe in a subprocess first so unstable native imports cannot abort
        # the benchmark host process.
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "import llm_guard.input_scanners"],
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
                        explanation="fallback llm-guard ssn",
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
                        explanation="fallback llm-guard email",
                        language=language,
                    )
                )
        return findings

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        if not self.enabled:
            return []

        language = str(context.get("language", "en")).lower()
        scanner = self._load_native_scanner()
        if scanner is None:
            return self._fallback_detect(payload, language)

        findings: list[EngineFinding] = []
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            try:
                scan_output = scanner.scan(value)
            except Exception:
                findings.extend(self._fallback_detect({key: value}, language))
                continue

            flagged = False
            sanitized_text: str | None = None
            if isinstance(scan_output, tuple) and len(scan_output) >= 2:
                # Typical shape: (sanitized_text, is_valid, score) or similar.
                if isinstance(scan_output[0], str):
                    sanitized_text = scan_output[0]
                second = scan_output[1]
                if isinstance(second, bool):
                    flagged = not second
                elif isinstance(second, (int, float)):
                    flagged = float(second) < 0.9
            elif isinstance(scan_output, tuple) and len(scan_output) == 1 and isinstance(scan_output[0], str):
                sanitized_text = scan_output[0]
            elif isinstance(scan_output, bool):
                flagged = not scan_output
            elif isinstance(scan_output, str):
                sanitized_text = scan_output

            if not flagged and sanitized_text is not None:
                flagged = sanitized_text != value

            if flagged:
                findings.append(
                    EngineFinding(
                        entity_type="SENSITIVE_PII",
                        confidence=0.88,
                        field_path=key,
                        span_start=0,
                        span_end=len(value),
                        engine_id=self.adapter_id,
                        explanation="llm-guard native sensitive signal",
                        language=language,
                    )
                )
        return findings
