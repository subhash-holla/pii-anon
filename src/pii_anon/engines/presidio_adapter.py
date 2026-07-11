from __future__ import annotations

import re
from typing import Any

from pii_anon.engines.base import EngineAdapter, passes_ner_span_hygiene
from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class PresidioAdapter(EngineAdapter):
    adapter_id = "presidio-compatible"
    native_dependency = "presidio_analyzer"

    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    # Presidio-native → pool-vocabulary label normalization (sp6). Presidio's
    # raw labels (PERSON, NRP, …) never type-voted with the other engines'
    # vocabulary (PERSON_NAME, …), so in the swarm fusion its findings split
    # the cluster vote and died downstream — measured on TAB/ECHR: presidio
    # solo reached relaxed recall 0.530 and ZERO of it survived fusion.
    # Unmapped labels pass through unchanged.
    _LABEL_MAP: dict[str, str] = {
        "PERSON": "PERSON_NAME",
        "NRP": "NATIONALITY",
        "LOCATION": "LOCATION",
        "DATE_TIME": "DATE_TIME",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "CREDIT_CARD": "CREDIT_CARD",
        "US_SSN": "US_SSN",
        "IP_ADDRESS": "IP_ADDRESS",
        "US_DRIVER_LICENSE": "DRIVERS_LICENSE",
        "US_PASSPORT": "PASSPORT",
        "US_BANK_NUMBER": "BANK_ACCOUNT",
        "IBAN_CODE": "IBAN",
        "CRYPTO": "CRYPTO_WALLET",
        # MEDICAL_LICENSE deliberately NOT remapped (round-2 close): the raw
        # label IS in the orchestrator's SUPPORTED_ENTITY_TYPES while
        # NPI_NUMBER is not — the remap inverted the leak direction on the
        # weighted_consensus/union_high_recall masking paths (a conf-1.0
        # medical-license emission stopped being maskable). Every other remap
        # here is a masking GAIN or neutral (verified against the whitelist).
        "US_ITIN": "TAX_ID",
    }

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
                raw_label = str(getattr(item, "entity_type", "UNKNOWN"))
                mapped_type = self._LABEL_MAP.get(raw_label, raw_label)
                span_start = int(getattr(item, "start", 0))
                span_end = int(getattr(item, "end", 0))
                confidence = float(getattr(item, "score", 0.7))
                # sp6 general FP hygiene (shared with the gliner adapter):
                # field-label-position veto + single-token-person bar. This
                # matters doubly here — once presidio's labels normalize into
                # the pool vocabulary, a presidio+gliner junk PAIR would
                # otherwise pass the fusion corroboration gate (measured:
                # +2,713 home PERSON_NAME FPs without this guard).
                if not passes_ner_span_hygiene(
                    value, span_start, span_end, mapped_type, confidence
                ):
                    continue
                findings.append(
                    EngineFinding(
                        entity_type=mapped_type,
                        confidence=confidence,
                        field_path=key,
                        span_start=span_start,
                        span_end=span_end,
                        engine_id=self.adapter_id,
                        explanation="presidio native",
                        language=language,
                    )
                )
        return findings
