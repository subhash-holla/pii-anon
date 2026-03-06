"""Regex-based PII detection engine package.

This package implements pii-anon's built-in regex engine, which provides
zero-dependency PII detection for 38+ entity types using compiled regular
expressions, checksum validators, context-aware confidence scoring, and
configurable deny/allow-list filtering.

Architecture
------------
The engine is split into focused modules for maintainability:

- **patterns** — Declarative ``PatternSpec`` registry defining every regex
  pattern, its entity type, base confidence, and optional validator/context.
- **validators** — Pure-function checksum and format validators (Luhn,
  mod-97 IBAN, ABA routing, VIN check digit, Verhoeff, SSN area rules).
- **confidence** — Context-aware confidence adjustment: boosts scores when
  surrounding text contains entity-relevant keywords, penalizes high
  false-positive types when context is absent.
- **deny_list** — Configurable per-entity-type deny-list (and allow-list)
  to suppress known false positives or protect known safe values.

The main ``RegexEngineAdapter`` in the parent module orchestrates these
components via a registry-driven ``detect()`` loop.
"""

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
from pii_anon.engines.regex.deny_list import (
    DEFAULT_DENY_LISTS,
    DenyListManager,
)
from pii_anon.engines.regex.patterns import (
    PATTERN_REGISTRY,
    PatternSpec,
)
from pii_anon.engines.regex.validators import (
    is_cc_format,
    is_valid_aadhaar_verhoeff,
    is_valid_aba_routing,
    is_valid_credit_card,
    is_valid_iban,
    is_valid_iban_format,
    is_valid_iban_strict,
    is_valid_ipv4,
    is_valid_ssn_digits,
    is_valid_vin_check_digit,
    looks_like_address_phrase,
    luhn_checksum,
)

__all__ = [
    # patterns
    "PATTERN_REGISTRY",
    "PatternSpec",
    # validators
    "is_cc_format",
    "is_valid_aadhaar_verhoeff",
    "is_valid_aba_routing",
    "is_valid_credit_card",
    "is_valid_iban",
    "is_valid_iban_format",
    "is_valid_iban_strict",
    "is_valid_ipv4",
    "is_valid_ssn_digits",
    "is_valid_vin_check_digit",
    "looks_like_address_phrase",
    "luhn_checksum",
    # confidence
    "CONTEXT_BOOST",
    "CONTEXT_PENALTY",
    "CONTEXT_WINDOW",
    "CONTEXT_WORDS",
    "HIGH_FP_TYPES",
    "adjust_confidence",
    "extract_context",
    "has_context_words",
    # deny_list
    "DEFAULT_DENY_LISTS",
    "DenyListManager",
]
