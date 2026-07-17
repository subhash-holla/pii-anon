from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from pii_anon.engines.base import EngineAdapter, passes_ner_span_hygiene
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

    # sp6 label extension: organization / location / occupation had NO ML
    # channel at all (measured: 594 gretel + ~950 TAB + 294 nemotron + 95
    # home ORGANIZATION gold spans with zero ML coverage; LOCATION similar) —
    # the regex engine cannot enumerate open-vocabulary proper nouns.
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
        "organization",
        "location",
        "occupation",
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
        "organization": "ORGANIZATION",
        "company": "ORGANIZATION",
        "location": "LOCATION",
        "city": "LOCATION",
        "occupation": "JOB_TITLE",
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

    # Long-document windowing (sp4 external-validity fix). Measured on real
    # ECHR judgments: the model's effective detection COLLAPSES with input
    # length (3 findings on the first 500 chars, 1 at 1,000, ZERO at >=2,000),
    # so an unwindowed call silently loses every finding in a long document.
    # Window size swept on the TAB DEV split (tuning-legal; test untouched):
    # gold-PERSON overlap 50/56 at 400 chars vs 43/56 @600 / 35/56 @800 /
    # 18/56 @1200 — monotone, so 400 it is. Windows are whitespace-aligned,
    # overlap so a span straddling a boundary is seen whole by the next
    # window, and offsets are re-based to the full text. Additive-only
    # (leak-safe); texts under one window keep the single unwindowed call.
    _WINDOW_CHARS = 400
    _OVERLAP_CHARS = 100

    def _windows(self, text: str) -> list[tuple[int, str]]:
        """Whitespace-aligned (offset, chunk) windows covering ``text``.

        Both EDGES are word-aligned (sp6): the end retreats to the last
        whitespace, and the overlap re-entry point advances to the next
        post-whitespace boundary — a mid-word window start made the model
        report mid-word spans ("Col⟨leen Redding⟩"), measured in the sp6
        cross-dataset mining.
        """
        if len(text) <= self._WINDOW_CHARS:
            return [(0, text)]
        windows: list[tuple[int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + self._WINDOW_CHARS, len(text))
            if end < len(text):
                # Retreat to the last whitespace so no token is split; a
                # pathological whitespace-free run falls back to the hard cut.
                cut = text.rfind(" ", start + self._OVERLAP_CHARS, end)
                if cut > start:
                    end = cut
            windows.append((start, text[start:end]))
            if end >= len(text):
                break
            nxt = max(end - self._OVERLAP_CHARS, start + 1)
            # Word-align the re-entry: start the next window right after the
            # next space at-or-past the overlap point (searching through the
            # boundary space at `end`, which itself belongs to no entity).
            # A whitespace-free run keeps the unaligned fallback — mid-token
            # windows are unavoidable there and better than skipping text.
            if not text[nxt - 1].isspace():
                boundary = text.find(" ", nxt - 1, end + 1)
                if boundary != -1:
                    nxt = boundary + 1
            start = nxt
        return windows

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
            # (start, end, type) -> best finding; dedupes overlap-region
            # echoes keeping the higher-confidence emission.
            best: dict[tuple[int, int, str], EngineFinding] = {}
            failed = False
            for offset, chunk in self._windows(value):
                try:
                    entities = model.predict_entities(
                        chunk, self._PII_LABELS, threshold=0.5
                    )
                except Exception:
                    failed = True
                    break
                for entity in entities:
                    label = str(entity.get("label", "UNKNOWN")).lower()
                    mapped_type = self._LABEL_MAP.get(label, label.upper())
                    span_start = offset + int(entity.get("start", 0))
                    span_end = offset + int(entity.get("end", 0))
                    # Emission hygiene (sp6): a span must sit on word
                    # boundaries of the FULL text — window-edge artifacts
                    # produced mid-word spans ("Col⟨leen Redding⟩"). Snap
                    # outward (never inward: over-masking is the safe
                    # direction); drop only if the snap degenerates.
                    while (
                        span_start > 0
                        and span_start < len(value)
                        and value[span_start].isalnum()
                        and value[span_start - 1].isalnum()
                    ):
                        span_start -= 1
                    while (
                        0 < span_end < len(value)
                        and value[span_end - 1].isalnum()
                        and value[span_end].isalnum()
                    ):
                        span_end += 1
                    if span_end <= span_start:
                        continue
                    confidence = float(entity.get("score", 0.75))
                    # sp6 general FP hygiene (see passes_ner_span_hygiene:
                    # field-label-position veto + single-token-person bar).
                    if not passes_ner_span_hygiene(
                        value, span_start, span_end, mapped_type, confidence
                    ):
                        continue
                    finding = EngineFinding(
                        entity_type=mapped_type,
                        confidence=confidence,
                        field_path=key,
                        span_start=span_start,
                        span_end=span_end,
                        engine_id=self.adapter_id,
                        explanation="gliner native ner",
                        language=language,
                    )
                    dedupe_key = (span_start, span_end, mapped_type)
                    prior = best.get(dedupe_key)
                    if prior is None or finding.confidence > prior.confidence:
                        best[dedupe_key] = finding
            if failed:
                findings.extend(self._fallback_detect({key: value}, language))
                continue
            findings.extend(best[k] for k in sorted(best))
        return findings
