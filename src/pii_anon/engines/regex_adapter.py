"""Regex-based PII detection engine adapter.

``RegexEngineAdapter`` is pii-anon's built-in, zero-dependency detection
engine.  It uses compiled regular expressions, checksum validators, and
context-aware confidence scoring to detect **38+ PII entity types** across
English, Spanish, and French.

Architecture
------------
Detection is driven by a **declarative pattern registry** (see
``engines/regex/patterns.py``).  Each ``PatternSpec`` defines a regex
pattern, base confidence, optional validator, and context-type for
confidence adjustment.  The ``detect()`` method iterates the registry,
running each pattern against every string field in the payload.

Confidence tiers
~~~~~~~~~~~~~~~~
- **Checksum-validated** (0.91–0.99): Luhn, IBAN mod-97, ABA routing,
  VIN check digit, Aadhaar Verhoeff.
- **Context-boosted** (+0.08): Surrounding keywords increase confidence.
- **Context-penalized** (−0.05): High false-positive types without context.
- **Format-only** (0.75–0.85): Pattern matches without validation.

Modularity
~~~~~~~~~~
Logic is split across focused sub-modules in ``engines/regex/``:

- ``patterns`` — PatternSpec registry (single source of truth)
- ``validators`` — Pure-function checksum/format validators
- ``confidence`` — Context-aware confidence adjustment
- ``deny_list`` — Deny-list and allow-list management

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
All validation methods that were previously ``@staticmethod`` on this class
are re-exported as class attributes so that existing code referencing
``RegexEngineAdapter._luhn_checksum(...)`` continues to work.
"""

from __future__ import annotations

import re
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.engines.regex.confidence import (
    CONTEXT_BOOST,
    CONTEXT_PENALTY,
    CONTEXT_WINDOW,
    CONTEXT_WORDS,
    HIGH_FP_TYPES,
    adjust_confidence,
    extract_context,
    has_context_words,
)
from pii_anon.engines.regex.deny_list import DenyListManager
from pii_anon.engines.regex.patterns import PATTERN_REGISTRY, PatternSpec
from pii_anon.engines.regex import validators
from pii_anon.types import EngineCapabilities, EngineFinding, Payload

# Backward-compatible aliases — tests and external code may import these
# with the underscore prefix from this module.
_CONTEXT_BOOST = CONTEXT_BOOST
_CONTEXT_PENALTY = CONTEXT_PENALTY
_CONTEXT_WINDOW = CONTEXT_WINDOW
_CONTEXT_WORDS = CONTEXT_WORDS
_HIGH_FP_TYPES = HIGH_FP_TYPES

# ── Validator dispatch table ──────────────────────────────────────────────
# Maps validator names (from PatternSpec.validator) to callables or sentinel
# strings.  Callable validators receive (candidate_text, match_object) and
# return bool.  String sentinels indicate custom handlers in _run_validator()
# that need access to the full match context (e.g., multi-group extraction,
# checksum-vs-format tiered confidence).
#
# This dispatch table replaces a 136-line if-elif chain in earlier versions,
# providing O(1) lookup instead of O(k) linear scanning through branches.
# Custom handlers (string sentinels) still use if-elif in _run_validator()
# because they require complex multi-step logic.

_VALIDATORS: dict[str, Any] = {
    # Simple bool validators — callable, return True/False
    "ipv4": lambda text, _m: validators.is_valid_ipv4(text),
    "aba_routing": lambda text, _m: validators.is_valid_aba_routing(text),
    "vin": lambda text, _m: validators.is_valid_vin_check_digit(text),
    "npi": lambda text, _m: validators.is_valid_npi(text),
    "dea": lambda text, _m: validators.is_valid_dea_number(text),
    # Custom handlers — need multi-step logic in _run_validator()
    "ssn_dash": "_ssn_dash",       # area-code validation (000, 666, 900+)
    "ssn_space": "_ssn_space",     # area-code validation (space-separated)
    "ssn_nodash": "_ssn_nodash",   # full SSN digit validation
    "credit_card": "_credit_card", # Luhn checksum → tiered confidence
    "iban": "_iban",               # mod-97 checksum → tiered confidence
    "sin_luhn": "_sin_luhn",       # Canadian SIN Luhn check
    "aadhaar": "_aadhaar",         # Indian Aadhaar Verhoeff checksum
    "date_iso": "_date_iso",       # month/day range validation
    "gps": "_gps",                 # lat/lon range check (±90/±180)
    "swift_context": "_swift_ctx", # require bank/wire context keywords
    "age": "_age",                 # age range validation (0-150)
}


class RegexEngineAdapter(EngineAdapter):
    """Zero-dependency regex PII detection engine.

    Detects 38+ entity types using compiled regular expressions with
    checksum validation, context-aware confidence scoring, and
    configurable deny/allow-list filtering.

    Parameters
    ----------
    enabled:
        Whether the engine is active (default *True*).
    deny_list_config:
        Optional dict to configure the deny-list.  Keys: ``enabled`` (bool),
        ``lists`` (dict[str, list[str]]).
    allow_list_config:
        Optional dict to configure the allow-list.  Same structure as deny-list.
    """

    adapter_id = "regex-oss"

    # ── Backward-compatible class attributes ───────────────────────────
    # Expose all compiled patterns from the registry as class variables
    # so existing code like `RegexEngineAdapter.EMAIL` still works.
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    SSN_SPACE = re.compile(r"\b\d{3}\s\d{2}\s\d{4}\b")
    SSN_NODASH = re.compile(r"\b\d{9}\b")
    IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
    IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
    PHONE_PATTERNS = {
        "en": re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)"),
        "es": re.compile(r"(?<!\w)(?:\+34[-.\s]?)?(?:6|7|9)\d{2}[-.\s]?\d{3}[-.\s]?\d{3}(?!\w)"),
        "fr": re.compile(r"(?<!\w)(?:\+33[-.\s]?)?(?:0?[1-9])(?:[-.\s]?\d{2}){4}(?!\w)"),
    }
    PERSON_PATTERNS = {
        "en": re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
        "es": re.compile(r"\b(?:Sr|Sra|Srta|Dra)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
        "fr": re.compile(r"\b(?:M|Mme|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
    }
    PERSON_FULL_NAME = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?![A-Za-z])")
    PERSON_FIRST_INITIAL = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]\.\b")
    SURNAME_CONTEXT = re.compile(
        r"\b(?:for|belongs\s+to|associated\s+with)\s+(?:Mr|Mrs|Ms|Dr|Prof)?\.?\s*([A-Z][a-z]+)\b"
    )
    PERSON_ALIAS_CONTEXT = re.compile(
        r"\b(?:alias|called|named|refer(?:red)?\s+to(?:\s+as)?)\s+([A-Z][a-z]{2,})\b",
        re.IGNORECASE,
    )
    PERSON_CONTEXT_KEYWORD = re.compile(
        r"\b(?:name\s+is|patient|employee|client|resident|member|user|"
        r"account\s+holder|beneficiary|author|sender|recipient|contact|"
        r"applicant|insured|claimant|defendant|plaintiff|witness|tenant|"
        r"signed\s+by|submitted\s+by|prepared\s+by|reviewed\s+by|assigned\s+to)"
        r"\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    )
    PERSON_POSSESSIVE = re.compile(
        r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)'s\s+"
        r"(?:account|email|phone|address|record|file|case|report|application|"
        r"profile|password|card|payment|order|appointment|prescription|"
        r"information|data|details)\b"
    )
    ADDRESS_SUFFIXES = {
        "street", "st", "avenue", "ave", "road", "rd", "way",
        "boulevard", "blvd", "lane", "ln", "circle", "terrace", "drive", "dr",
    }
    DOB_CONTEXT = re.compile(
        r"\b(?:born|DOB|date\s+of\s+birth|birth\s*date|d\.o\.b\.?)\s*[:\-]?\s*"
        r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
        re.IGNORECASE,
    )
    DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    MAC_ADDRESS = re.compile(
        r"\b([0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-]"
        r"[0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2})\b"
    )
    DRIVERS_LICENSE = re.compile(
        r"\b(?:driver'?s?\s*license|DL|license\s*(?:number|no|#))\s*[:\-#]?\s*"
        r"([A-Z]\d{4,15}|\d{1,3}-\d{2,4}-\d{4,6})\b",
        re.IGNORECASE,
    )
    PASSPORT = re.compile(
        r"\b(?:passport)\s*(?:number|no|#)?\s*[:\-#]?\s*([A-Z]{1,2}\d{6,9})\b",
        re.IGNORECASE,
    )
    ROUTING_NUMBER = re.compile(
        r"\b(?:routing|ABA|transit)\s*(?:number|no|#)?\s*[:\-#]?\s*(\d{9})\b",
        re.IGNORECASE,
    )
    LICENSE_PLATE = re.compile(
        r"\b(?:plate|license\s*plate|tag)\s*(?:number|no|#)?\s*[:\-#]?\s*"
        r"([A-Z0-9]{1,4}[\s\-]?[A-Z0-9]{2,5})\b",
        re.IGNORECASE,
    )
    BANK_ACCOUNT = re.compile(
        r"\b(?:account|acct|bank\s*account)\s*(?:number|no|#)?\s*[:\-#]?\s*(\d{8,17})\b",
        re.IGNORECASE,
    )
    NATIONAL_ID = re.compile(
        r"\b(?:national\s*id|national\s*identification|citizen\s*id|ID\s*number)"
        r"\s*[:\-#]?\s*([A-Z0-9]{5,20})\b",
        re.IGNORECASE,
    )
    USERNAME_AT = re.compile(r"(?<!\w)@([A-Za-z][A-Za-z0-9._-]{2,30})(?!\w)")
    USERNAME_CONTEXT = re.compile(
        r"\b(?:username|user\s*name|login|handle|screen\s*name)\s*[:\-]?\s*"
        r"([A-Za-z][A-Za-z0-9._-]{2,30})\b",
        re.IGNORECASE,
    )
    EMPLOYEE_ID = re.compile(
        r"\b(?:employee\s*id|EMP|employee\s*number|emp\s*#|staff\s*id)\s*[:\-#]?\s*"
        r"([A-Z0-9]{3,15})\b",
        re.IGNORECASE,
    )
    MEDICAL_RECORD = re.compile(
        r"\b(?:MRN|medical\s*record|patient\s*id|medical\s*id)\s*(?:number|no|#)?\s*[:\-#]?\s*"
        r"([A-Z0-9]{4,20})\b",
        re.IGNORECASE,
    )
    ORGANIZATION = re.compile(
        r"\b([A-Z][A-Za-z&'.]+(?:\s+[A-Z][A-Za-z&'.]+)*)\s+"
        r"(?:Inc|Corp|Corporation|LLC|Ltd|Limited|GmbH|AG|PLC|Co|Company|Group|Foundation|Association)"
        r"\.?\b"
    )
    ADDRESS_PATTERN = re.compile(
        r"\b(\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,4}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|"
        r"Place|Pl|Circle|Cir|Terrace|Ter|Trail|Trl|Highway|Hwy|Parkway|Pkwy))"
        r"\.?\b",
        re.IGNORECASE,
    )
    LOCATION_CONTEXT = re.compile(
        r"\b(?:city|location|located\s+in|residing\s+in|based\s+in|from)\s*[:\-]?\s*"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
    )
    CRYPTO_BITCOIN = re.compile(r"\b([13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
    CRYPTO_BITCOIN_BECH32 = re.compile(r"\b(bc1[a-z0-9]{39,59})\b")
    CRYPTO_ETHEREUM = re.compile(r"\b(0x[a-fA-F0-9]{40})\b")
    GPS_COORDINATES = re.compile(
        r"(?<![0-9.])"
        r"(-?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?))"
        r"\s*[,/]\s*"
        r"(-?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d{1,2}(?:\.\d+)?))"
        r"(?![0-9.])"
    )
    SWIFT_BIC = re.compile(r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b")
    VIN_NUMBER = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
    ZIP_CODE = re.compile(
        r"\b(?:zip\s*(?:code)?|postal\s*code)\s*[:\-#]?\s*(\d{5}(?:-\d{4})?)\b",
        re.IGNORECASE,
    )
    CANADIAN_SIN = re.compile(
        r"\b(?:SIN|social\s+insurance)\s*(?:number|no|#)?\s*[:\-#]?\s*"
        r"(\d{3}[-\s]?\d{3}[-\s]?\d{3})\b",
        re.IGNORECASE,
    )
    UK_NI_NUMBER = re.compile(r"\b([A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D])\b")
    JWT_TOKEN = re.compile(r"\b(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b")
    API_KEY_CONTEXT = re.compile(
        r"\b(?:api[_\s]?key|api[_\s]?token|bearer|access[_\s]?token|secret[_\s]?key)"
        r"\s*[:\-=]\s*"
        r"([A-Za-z0-9_\-]{32,})\b",
        re.IGNORECASE,
    )
    AADHAAR = re.compile(
        r"\b(?:aadhaar|aadhar|uid)\s*(?:number|no|#)?\s*[:\-#]?\s*"
        r"(\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",
        re.IGNORECASE,
    )

    # ── Default deny-list (backward compat) ────────────────────────────
    _DEFAULT_DENY_LISTS: dict[str, set[str]] = {
        "PERSON_NAME": {
            "new york", "san francisco", "los angeles", "united states",
            "north america", "south america", "east coast", "west coast",
            "great britain", "new zealand", "south africa", "north korea",
            "south korea", "united kingdom", "hong kong", "el salvador",
            "costa rica", "puerto rico", "sri lanka", "saudi arabia",
            "test user", "sample data", "john doe", "jane doe",
        },
    }

    def __init__(
        self,
        *,
        enabled: bool = True,
        deny_list_config: dict[str, Any] | None = None,
        allow_list_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self._list_mgr = DenyListManager(
            deny_config=deny_list_config,
            allow_config=allow_list_config,
        )
        # Backward-compatible attribute
        self._deny_lists = self._list_mgr._deny_lists
        # Fast-path flag: skip allow-list checks when no allow-lists exist
        self._has_allow_lists = bool(self._list_mgr._allow_lists)

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize engine, loading deny/allow-lists from config."""
        super().initialize(config)
        if config:
            self._list_mgr.initialize(config)
            self._deny_lists = self._list_mgr._deny_lists
            self._has_allow_lists = bool(self._list_mgr._allow_lists)

    def capabilities(self) -> EngineCapabilities:
        """Return engine capabilities including supported languages."""
        caps = super().capabilities()
        caps.supports_languages = ["en", "es", "fr"]
        caps.supports_streaming = True
        return caps

    # ── Context-aware confidence (delegated) ───────────────────────────

    @staticmethod
    def _extract_context(text: str, start: int, end: int) -> str:
        """Return lowercased text surrounding the matched span."""
        return extract_context(text, start, end)

    @staticmethod
    def _has_context_words(entity_type: str, context_text: str) -> bool:
        """Check if context keywords appear near the match."""
        return has_context_words(entity_type, context_text)

    @classmethod
    def _adjust_confidence(
        cls, entity_type: str, base_confidence: float,
        text: str, start: int, end: int,
    ) -> float:
        """Boost or penalize confidence based on surrounding context."""
        return adjust_confidence(entity_type, base_confidence, text, start, end)

    # ── Deny-list (delegated) ──────────────────────────────────────────

    def _is_denied(self, entity_type: str, matched_text: str) -> bool:
        """Return True if matched text is in the deny-list."""
        return self._list_mgr.is_denied(entity_type, matched_text)

    def _is_allowed(self, entity_type: str, matched_text: str) -> bool:
        """Return True if matched text is in the allow-list."""
        return self._list_mgr.is_allowed(entity_type, matched_text)

    def _load_deny_lists(self, cfg: dict[str, Any]) -> None:
        """Build deny-list sets from configuration dict."""
        self._list_mgr._load_lists(cfg, target=self._list_mgr._deny_lists)
        self._deny_lists = self._list_mgr._deny_lists

    # ── Backward-compatible validation aliases ─────────────────────────

    _luhn_checksum = staticmethod(validators.luhn_checksum)
    _is_cc_format = staticmethod(validators.is_cc_format)
    _is_valid_credit_card = staticmethod(validators.is_valid_credit_card)
    _is_valid_ipv4 = staticmethod(validators.is_valid_ipv4)
    _is_valid_iban_strict = staticmethod(validators.is_valid_iban_strict)
    _is_valid_iban_format = staticmethod(validators.is_valid_iban_format)
    _is_valid_iban = staticmethod(validators.is_valid_iban)
    _is_valid_ssn_digits = staticmethod(validators.is_valid_ssn_digits)
    _is_valid_aba_routing = staticmethod(validators.is_valid_aba_routing)
    _is_valid_vin_check_digit = staticmethod(validators.is_valid_vin_check_digit)
    _is_valid_aadhaar_verhoeff = staticmethod(validators.is_valid_aadhaar_verhoeff)

    def _looks_like_address_phrase(self, text: str, start: int, end: int) -> bool:
        """Check if span looks like a street address (backward compat)."""
        return validators.looks_like_address_phrase(text, start, end)

    def _looks_like_org_or_common_phrase(self, phrase: str) -> bool:
        """Filter common false positives — delegates to deny-list."""
        return self._is_denied("PERSON_NAME", phrase)

    # ── Registry-driven detect() ───────────────────────────────────────

    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:  # noqa: C901, PLR0912, PLR0915
        """Detect PII entities in all string fields of *payload*.

        Iterates the pattern registry, running each ``PatternSpec`` against
        every string value.  For each match:

        1. Extract the matched span (using ``spec.group``).
        2. Run the optional validator; skip or adjust confidence.
        3. Apply context-aware confidence boosting.
        4. Check deny-list / allow-list filters.
        5. Emit an ``EngineFinding``.

        Parameters
        ----------
        payload:
            Dict of field names to values.  Only ``str`` values are scanned.
        context:
            Runtime context including ``language`` (default ``"en"``).

        Returns
        -------
        list[EngineFinding]
            Detected PII findings with confidence scores.
        """
        findings: list[EngineFinding] = []
        if not self.enabled:
            return findings

        language = str(context.get("language", "en")).lower()

        for key, value in payload.items():
            if not isinstance(value, str):
                continue

            # ── Performance pre-filter signals ─────────────────────
            # Computed once per field to skip patterns that cannot match.
            _has_at = "@" in value
            _has_colon = ":" in value
            _has_dot = "." in value
            _has_eyj = "eyJ" in value
            # Defer .lower() until first pattern actually needs the http check.
            _has_http: bool | None = None

            for spec in PATTERN_REGISTRY:
                # Language filter
                if spec.language and spec.language != language:
                    continue

                # Pre-filter: skip patterns that require a character not in text
                if spec.pre_filter:
                    pf = spec.pre_filter
                    if pf == "@" and not _has_at:
                        continue
                    if pf == ":" and not _has_colon:
                        continue
                    if pf == "." and not _has_dot:
                        continue
                    if pf == "http":
                        if _has_http is None:
                            _has_http = "http" in value.lower()
                        if not _has_http:
                            continue
                    if pf == "eyJ" and not _has_eyj:
                        continue
                    if pf == "0x" and "0x" not in value:
                        continue

                for match in spec.pattern.finditer(value):
                    # ── Extract span ───────────────────────────────
                    try:
                        span_start = match.start(spec.group)
                        span_end = match.end(spec.group)
                    except IndexError:
                        span_start = match.start()
                        span_end = match.end()

                    matched_text = value[span_start:span_end]

                    # ── Allow-list check (skip if allowed) ─────────
                    if self._has_allow_lists and self._is_allowed(spec.entity_type, matched_text):
                        continue

                    # ── Address-phrase filter for PERSON_NAME ──────
                    if spec.entity_type == "PERSON_NAME" and spec.group == 0:
                        if validators.looks_like_address_phrase(value, span_start, span_end):
                            continue

                    # ── Deny-list check ────────────────────────────
                    if spec.deny_check and self._is_denied(spec.entity_type, matched_text):
                        continue

                    # ── Validator dispatch ─────────────────────────
                    confidence = spec.base_confidence
                    explanation = spec.explanation

                    if spec.validator:
                        confidence, explanation, skip = self._run_validator(
                            spec, matched_text, match, value, span_start, span_end,
                        )
                        if skip:
                            continue

                    # ── Context-aware confidence ───────────────────
                    if spec.context_type:
                        confidence = adjust_confidence(
                            spec.context_type, confidence, value, span_start, span_end,
                        )

                    findings.append(
                        EngineFinding(
                            entity_type=spec.entity_type,
                            confidence=confidence,
                            field_path=key,
                            span_start=span_start,
                            span_end=span_end,
                            engine_id=self.adapter_id,
                            explanation=explanation,
                            language=language,
                        )
                    )

        return findings

    # ── Custom validator handlers ──────────────────────────────────────

    def _run_validator(
        self,
        spec: PatternSpec,
        matched_text: str,
        match: re.Match[str],
        value: str,
        span_start: int,
        span_end: int,
    ) -> tuple[float, str, bool]:
        """Run a named validator and return (confidence, explanation, skip).

        Returns
        -------
        tuple[float, str, bool]
            ``(confidence, explanation, should_skip)``.  When ``should_skip``
            is *True*, the match should be discarded.
        """
        v = spec.validator
        confidence = spec.base_confidence
        explanation = spec.explanation

        # ── SSN dash format: area validation ───────────────────────
        if v == "ssn_dash":
            area = int(matched_text[:3])
            if area == 0 or area == 666 or area >= 900:
                return 0, "", True
            return confidence, explanation, False

        # ── SSN space format: area validation ──────────────────────
        if v == "ssn_space":
            digits = matched_text.replace(" ", "")
            area = int(digits[:3])
            if area == 0 or area == 666 or area >= 900:
                return 0, "", True
            return confidence, explanation, False

        # ── SSN no-dash: full validation ───────────────────────────
        if v == "ssn_nodash":
            if not validators.is_valid_ssn_digits(matched_text):
                return 0, "", True
            return confidence, explanation, False

        # ── Credit card: Luhn vs format-only ───────────────────────
        if v == "credit_card":
            digits = "".join(ch for ch in matched_text if ch.isdigit())
            if len(digits) < 13 or len(digits) > 19:
                return 0, "", True
            if validators.luhn_checksum(digits):
                return 0.94, "regex credit card (luhn valid)", False
            if validators.is_cc_format(digits):
                return 0.80, "regex credit card (format match)", False
            return 0, "", True

        # ── IBAN: mod-97 vs format-only ────────────────────────────
        if v == "iban":
            if validators.is_valid_iban_strict(matched_text):
                return 0.93, "regex iban (checksum valid)", False
            if validators.is_valid_iban_format(matched_text):
                return 0.78, "regex iban (format match)", False
            return 0, "", True

        # ── Canadian SIN: Luhn on 9 digits ─────────────────────────
        if v == "sin_luhn":
            digits = "".join(ch for ch in matched_text if ch.isdigit())
            if len(digits) != 9:
                return 0, "", True
            if validators.luhn_checksum(digits):
                return spec.valid_confidence or 0.92, "regex canadian sin (luhn valid)", False
            return spec.invalid_confidence or 0.75, "regex canadian sin (format only)", False

        # ── Aadhaar: Verhoeff on 12 digits ─────────────────────────
        if v == "aadhaar":
            digits = "".join(ch for ch in matched_text if ch.isdigit())
            if len(digits) != 12:
                return 0, "", True
            if validators.is_valid_aadhaar_verhoeff(digits):
                return spec.valid_confidence or 0.91, "regex aadhaar (verhoeff valid)", False
            return spec.invalid_confidence or 0.80, "regex aadhaar (format only)", False

        # ── DATE_ISO: month/day validation ─────────────────────────
        if v == "date_iso":
            try:
                parts = matched_text.split("-")
                month, day = int(parts[1]), int(parts[2])
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    return 0, "", True
            except (ValueError, IndexError):
                return 0, "", True
            return confidence, explanation, False

        # ── GPS: latitude/longitude range check ────────────────────
        if v == "gps":
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    return 0, "", True
                # Override span to cover both groups
                return confidence, explanation, False
            except (ValueError, AttributeError):
                return 0, "", True

        # ── SWIFT/BIC: require bank/wire context ───────────────────
        if v == "swift_context":
            ctx = extract_context(value, span_start, span_end)
            if not has_context_words("IBAN", ctx):
                return 0, "", True
            return confidence, explanation, False

        # ── AGE: range validation (0-150) ──────────────────────────
        if v == "age":
            # AGE pattern has 2 groups; one will be None
            g1 = match.group(1)
            g2 = match.group(2)
            age_str = g1 if g1 else g2
            if not age_str:
                return 0, "", True
            try:
                age_val = int(age_str)
                if not (0 <= age_val <= 150):
                    return 0, "", True
            except ValueError:
                return 0, "", True
            return confidence, explanation, False

        # ── Generic validators (ABA routing, VIN, NPI, DEA) ────────
        handler = _VALIDATORS.get(v) if v is not None else None
        if handler and callable(handler):
            is_valid = handler(matched_text, match)
            if is_valid:
                return spec.valid_confidence or confidence, explanation, False
            if spec.invalid_confidence is not None:
                return spec.invalid_confidence, explanation + " (format only)", False
            return 0, "", True

        # Fallback: no validator matched
        return confidence, explanation, False
