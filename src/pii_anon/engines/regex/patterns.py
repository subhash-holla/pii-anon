"""Declarative pattern registry for the regex PII detection engine.

Every PII entity type detected by the regex engine is defined here as a
``PatternSpec`` — a frozen dataclass that bundles a compiled regex pattern
with metadata: the entity type it detects, base confidence score, capture
group index, optional validator name, context-type for confidence boosting,
and whether deny-list filtering should be applied.

The ``PATTERN_REGISTRY`` tuple is the single source of truth consumed by
``RegexEngineAdapter.detect()``.  Adding a new entity type is as simple as
appending a ``PatternSpec`` to the registry — no detect() code changes needed
for standard patterns.

Design decisions
----------------
- **Frozen dataclass with __slots__** — immutable, low memory overhead.
- **Validator by name** — validators are referenced as string names and
  resolved at runtime from the validators module, avoiding circular imports
  and keeping the registry purely declarative.
- **``group=0`` default** — most patterns use the full match; patterns with
  context keywords use capture group 1.
- **``pre_filter`` field** — optional character that must appear in the text
  for the pattern to be worth running (performance optimization).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatternSpec:
    """Specification for a single regex-based PII detection pattern.

    Parameters
    ----------
    entity_type:
        The PII entity type label (e.g., ``"EMAIL_ADDRESS"``).
    pattern:
        Compiled regex pattern to match against input text.
    base_confidence:
        Default confidence score when the pattern matches.
    group:
        Capture group index for span extraction (0 = full match).
    validator:
        Name of a validation function in ``validators`` module, or *None*.
        When provided, the validator is called with the matched text;
        returning *False* skips the match, *True* may upgrade confidence.
    context_type:
        Key into ``confidence.CONTEXT_WORDS`` for context-aware scoring.
    explanation:
        Human-readable explanation string for the finding.
    language:
        ISO 639-1 code if pattern is language-specific, or *None* for all.
    deny_check:
        Whether to consult the deny-list before emitting a finding.
    pre_filter:
        A character that must exist in the text for this pattern to be
        worth running.  Set to *None* to always run.
    valid_confidence:
        Confidence override when the validator returns *True*.  If *None*,
        ``base_confidence`` is used.
    invalid_confidence:
        Confidence override when the validator returns *False* but the
        match is still emitted (format-only match).  If *None*, the match
        is skipped entirely on validation failure.
    """

    entity_type: str
    pattern: re.Pattern[str]
    base_confidence: float
    group: int = 0
    validator: str | None = None
    context_type: str | None = None
    explanation: str = ""
    language: str | None = None
    deny_check: bool = False
    pre_filter: str | None = None
    valid_confidence: float | None = None
    invalid_confidence: float | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Pattern Definitions
# ═══════════════════════════════════════════════════════════════════════════
#
# Organized by category.  Each pattern includes an inline comment explaining
# what the regex matches.

# ── Core: Email ────────────────────────────────────────────────────────────
# Standard email: local-part @ domain . TLD (2+ chars).
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# ── Core: SSN (3 formats) ─────────────────────────────────────────────────
# Dash-separated: 123-45-6789.
_SSN_DASH = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Space-separated: 123 45 6789.
_SSN_SPACE = re.compile(r"\b\d{3}\s\d{2}\s\d{4}\b")
# No separator: 9 consecutive digits.
_SSN_NODASH = re.compile(r"\b\d{9}\b")

# ── Core: IP Address ──────────────────────────────────────────────────────
# IPv4: four dot-separated octets (validated in post-processing).
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# IPv6: full or compressed notation with :: shorthand.
_IPV6 = re.compile(
    r"(?<![:\w])"                                              # not preceded by : or word char
    r"(?:"
    r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"             # full 8 groups
    r"|"
    r"(?:[0-9a-fA-F]{1,4}:){1,7}:"                           # trailing ::
    r"|"
    r"(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"          # :: in middle
    r"|"
    r"::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}"         # leading ::
    r"|"
    r"::"                                                      # all-zeros
    r")"
    r"(?![:\w])"                                               # not followed by : or word char
)

# ── Core: Credit Card ─────────────────────────────────────────────────────
# 13-19 digits with optional spaces/dashes between groups.
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

# ── Core: IBAN ─────────────────────────────────────────────────────────────
# 2 uppercase country letters + 2 check digits + 11-30 alphanumeric chars.
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

# ── Core: Phone (multilingual) ────────────────────────────────────────────
_PHONE_EN = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)")
_PHONE_ES = re.compile(r"(?<!\w)(?:\+34[-.\s]?)?(?:6|7|9)\d{2}[-.\s]?\d{3}[-.\s]?\d{3}(?!\w)")
_PHONE_FR = re.compile(r"(?<!\w)(?:\+33[-.\s]?)?(?:0?[1-9])(?:[-.\s]?\d{2}){4}(?!\w)")

# ── Person: Title-prefix names (multilingual) ─────────────────────────────
# "Dr. John Smith", "Mr. Garcia Lopez".
_PERSON_EN = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_PERSON_ES = re.compile(r"\b(?:Sr|Sra|Srta|Dra)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_PERSON_FR = re.compile(r"\b(?:M|Mme|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")

# ── Person: Full name (2-3 capitalized words, no title) ───────────────────
_PERSON_FULL_NAME = re.compile(r"(?<![A-Za-z])[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?![A-Za-z])")

# ── Person: First name + initial ("John D.") ──────────────────────────────
_PERSON_FIRST_INITIAL = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]\.\b")

# ── Person: Surname with context ("for Mr Smith", "belongs to Garcia") ────
_SURNAME_CONTEXT = re.compile(
    r"\b(?:for|belongs\s+to|associated\s+with)\s+(?:Mr|Mrs|Ms|Dr|Prof)?\.?\s*([A-Z][a-z]+)\b"
)

# ── Person: Alias context ("alias Jack", "called Maria") ──────────────────
_PERSON_ALIAS = re.compile(
    r"\b(?:alias|called|named|refer(?:red)?\s+to(?:\s+as)?)\s+([A-Z][a-z]{2,})\b",
    re.IGNORECASE,
)

# ── Person: Keyword context ("name is John", "patient Maria Lopez") ───────
_PERSON_KEYWORD = re.compile(
    r"\b(?:name\s+is|patient|employee|client|resident|member|user|"
    r"account\s+holder|beneficiary|author|sender|recipient|contact|"
    r"applicant|insured|claimant|defendant|plaintiff|witness|tenant|"
    r"signed\s+by|submitted\s+by|prepared\s+by|reviewed\s+by|assigned\s+to)"
    r"\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

# ── Person: Possessive context ("John's account", "Maria's email") ────────
_PERSON_POSSESSIVE = re.compile(
    r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)'s\s+"
    r"(?:account|email|phone|address|record|file|case|report|application|"
    r"profile|password|card|payment|order|appointment|prescription|"
    r"information|data|details)\b"
)

# ── Document IDs ───────────────────────────────────────────────────────────
# Date of birth with context keyword.
_DOB_CONTEXT = re.compile(
    r"\b(?:born|DOB|date\s+of\s+birth|birth\s*date|d\.o\.b\.?)\s*[:\-]?\s*"
    r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
    re.IGNORECASE,
)
# ISO 8601 date: YYYY-MM-DD.
_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# General date patterns: "January 15, 2025", "15/01/2025", "Jan 15 2025".
_DATE_GENERAL = re.compile(
    r"\b("
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|"
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

# MAC address: 6 colon-or-dash-separated hex pairs.
_MAC_ADDRESS = re.compile(
    r"\b([0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-]"
    r"[0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2})\b"
)

# Driver's license: letter + digits with context keyword, or DL-prefixed ID.
_DRIVERS_LICENSE = re.compile(
    r"\b(?:driver'?s?\s*license|DL|license\s*(?:number|no|#))\s*[:\-#]?\s*"
    r"(DL-[A-Z]\d{4,6}-\d{2,4}|[A-Z]\d{4,15}|\d{1,3}-\d{2,4}-\d{4,6})\b",
    re.IGNORECASE,
)

# Passport with context keyword.  Accepts mixed alphanumeric IDs
# (e.g. "P7H104167") as well as letter-prefix + digits ("AB1234567").
_PASSPORT = re.compile(
    r"\b(?:passport)\s*(?:number|no|#)?\s*[:\-#]?\s*([A-Z][A-Z0-9]{5,11})\b",
    re.IGNORECASE,
)

# ABA routing number with context.
_ROUTING_NUMBER = re.compile(
    r"\b(?:routing|ABA|transit)\s*(?:number|no|#)?\s*[:\-#]?\s*(\d{9})\b",
    re.IGNORECASE,
)

# License plate with context keyword.
_LICENSE_PLATE = re.compile(
    r"\b(?:plate|license\s*plate|tag)\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"([A-Z0-9]{1,4}[\s\-]?[A-Z0-9]{2,5})\b",
    re.IGNORECASE,
)

# Bank account with context keyword.
_BANK_ACCOUNT = re.compile(
    r"\b(?:account|acct|bank\s*account)\s*(?:number|no|#)?\s*[:\-#]?\s*(\d{8,17})\b",
    re.IGNORECASE,
)

# National ID with context keyword, or NID-/TAX- prefixed IDs.
_NATIONAL_ID = re.compile(
    r"\b(?:national\s*id|national\s*identification|citizen\s*id|ID\s*number"
    r"|(?:international\s+)?tax\s+id)"
    r"\s*[:\-#]?\s*"
    r"((?:NID|TAX)-\d{6,15}|[A-Z0-9]{5,20})\b",
    re.IGNORECASE,
)

# Username: @-prefixed handle.
_USERNAME_AT = re.compile(r"(?<!\w)@([A-Za-z][A-Za-z0-9._-]{2,30})(?!\w)")
# Username with context keyword (includes log-style "User X" patterns
# and config-style "db_user" keys with JSON quoting).
_USERNAME_CONTEXT = re.compile(
    r"(?:\b(?:username|user\s*name|login|handle|screen\s*name|User)"
    r"|\"db_user\")"
    r"\s*[:\-]?\s*[\"']?\s*"
    r"([A-Za-z][A-Za-z0-9._-]{2,30})\b",
    re.IGNORECASE,
)

# Employee ID with context keyword.  Allows hyphenated IDs like "EMP-20165".
_EMPLOYEE_ID = re.compile(
    r"\b(?:employee\s*id|EMP|employee\s*number|emp\s*#|staff\s*id)\s*[:\-#]?\s*"
    r"(EMP-\d{3,10}|[A-Z0-9]{3,15})\b",
    re.IGNORECASE,
)

# Medical record number with context keyword.
_MEDICAL_RECORD = re.compile(
    r"\b(?:MRN|medical\s*record|patient\s*id|medical\s*id)\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"([A-Z0-9]{4,20})\b",
    re.IGNORECASE,
)

# Organization name: multi-word + corporate suffix.
_ORGANIZATION = re.compile(
    r"\b([A-Z][A-Za-z&'.]+(?:\s+[A-Z][A-Za-z&'.]+)*)\s+"
    r"(?:Inc|Corp|Corporation|LLC|Ltd|Limited|GmbH|AG|PLC|Co|Company|Group|Foundation|Association)"
    r"\.?\b"
)

# Street address: number + words + suffix, optionally followed by
# ", City, ST ZIP" so the captured span matches full mailing addresses.
_ADDRESS = re.compile(
    r"\b(\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|"
    r"Place|Pl|Circle|Cir|Terrace|Ter|Trail|Trl|Highway|Hwy|Parkway|Pkwy)"
    r"\.?"
    r"(?:,?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?)"
    r"\b",
    re.IGNORECASE,
)

# Location with context keyword.
_LOCATION_CONTEXT = re.compile(
    r"\b(?:city|location|located\s+in|residing\s+in|based\s+in|from)\s*[:\-]?\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)

# ── Financial: International ───────────────────────────────────────────────
# Bitcoin legacy (1/3 prefix, 25-34 Base58Check chars).
_CRYPTO_BITCOIN = re.compile(r"\b([13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")
# Bitcoin bech32 (bc1 prefix, 39-59 lowercase alphanumeric).
_CRYPTO_BECH32 = re.compile(r"\b(bc1[a-z0-9]{39,59})\b")
# Ethereum (0x prefix + 40 hex chars).
_CRYPTO_ETHEREUM = re.compile(r"\b(0x[a-fA-F0-9]{40})\b")

# GPS coordinates: decimal lat/lon pair.
_GPS = re.compile(
    r"(?<![0-9.])"
    r"(-?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?))"
    r"\s*[,/]\s*"
    r"(-?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d{1,2}(?:\.\d+)?))"
    r"(?![0-9.])"
)

# SWIFT/BIC: 8 or 11 character bank identifier.
_SWIFT_BIC = re.compile(r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b")

# VIN: 17 characters (I, O, Q excluded).
_VIN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")

# US ZIP code with context.
_ZIP_CODE = re.compile(
    r"\b(?:zip\s*(?:code)?|postal\s*code)\s*[:\-#]?\s*(\d{5}(?:-\d{4})?)\b",
    re.IGNORECASE,
)

# Canadian SIN with context.
_CANADIAN_SIN = re.compile(
    r"\b(?:SIN|social\s+insurance)\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"(\d{3}[-\s]?\d{3}[-\s]?\d{3})\b",
    re.IGNORECASE,
)

# UK National Insurance Number: 2 letters + 6 digits + suffix letter (A-D).
_UK_NI = re.compile(r"\b([A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D])\b")

# JWT: three base64url segments, first starts with "eyJ".
_JWT = re.compile(r"\b(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b")

# API key / bearer token: long alphanumeric after context keyword.
_API_KEY = re.compile(
    r"\b(?:api[_\s]?key|api[_\s]?token|bearer|access[_\s]?token|secret[_\s]?key)"
    r"\s*[:\-=]\s*"
    r"([A-Za-z0-9_\-]{32,})\b",
    re.IGNORECASE,
)

# Aadhaar (Indian UID) with context.
_AADHAAR = re.compile(
    r"\b(?:aadhaar|aadhar|uid)\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"(\d{4}[-\s]?\d{4}[-\s]?\d{4})\b",
    re.IGNORECASE,
)

# ── New entity types (Phase 2) ────────────────────────────────────────────

# URL with embedded PII: URLs containing email-like or query params with PII keywords.
_URL_WITH_PII = re.compile(
    r"\bhttps?://[^\s]+(?:"
    r"[?&](?:email|user|ssn|name|phone|account)=[^\s&]+"
    r"|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r")",
    re.IGNORECASE,
)

# AGE: "age 42", "42 years old", "42-year-old", "aged 65".
_AGE = re.compile(
    r"\b(?:age[d]?\s+(\d{1,3})|(\d{1,3})[-\s]?years?[-\s]?old)\b",
    re.IGNORECASE,
)

# NPI: National Provider Identifier (10 digits) with context.
_NPI = re.compile(
    r"\b(?:NPI|national\s+provider)\s*(?:identifier|number|no|#|id)?\s*[:\-#]?\s*(\d{10})\b",
    re.IGNORECASE,
)

# DEA number: 2 letters + 7 digits with context.
_DEA = re.compile(
    r"\b(?:DEA)\s*(?:number|no|#|registration)?\s*[:\-#]?\s*([A-Za-z]{2}\d{7})\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Pattern Registry
# ═══════════════════════════════════════════════════════════════════════════

PATTERN_REGISTRY: tuple[PatternSpec, ...] = (
    # ── EMAIL ──────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="EMAIL_ADDRESS",
        pattern=_EMAIL,
        base_confidence=0.99,
        context_type="EMAIL_ADDRESS",
        explanation="regex email",
        pre_filter="@",
    ),
    # ── US_SSN (dash) ──────────────────────────────────────────────────
    PatternSpec(
        entity_type="US_SSN",
        pattern=_SSN_DASH,
        base_confidence=0.97,
        validator="ssn_dash",
        context_type="US_SSN",
        explanation="regex ssn",
    ),
    # ── US_SSN (space) ─────────────────────────────────────────────────
    PatternSpec(
        entity_type="US_SSN",
        pattern=_SSN_SPACE,
        base_confidence=0.93,
        validator="ssn_space",
        context_type="US_SSN",
        explanation="regex ssn space",
    ),
    # ── US_SSN (no separator) ──────────────────────────────────────────
    PatternSpec(
        entity_type="US_SSN",
        pattern=_SSN_NODASH,
        base_confidence=0.80,
        validator="ssn_nodash",
        context_type="US_SSN",
        explanation="regex ssn nodash",
    ),
    # ── IPv4 ───────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="IP_ADDRESS",
        pattern=_IPV4,
        base_confidence=0.96,
        validator="ipv4",
        context_type="IP_ADDRESS",
        explanation="regex ipv4",
        pre_filter=".",
    ),
    # ── IPv6 ───────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="IP_ADDRESS",
        pattern=_IPV6,
        base_confidence=0.92,
        context_type="IP_ADDRESS",
        explanation="regex ipv6",
        pre_filter=":",
    ),
    # ── CREDIT_CARD ────────────────────────────────────────────────────
    PatternSpec(
        entity_type="CREDIT_CARD",
        pattern=_CREDIT_CARD,
        base_confidence=0.80,
        validator="credit_card",
        context_type="CREDIT_CARD",
        explanation="regex credit card",
    ),
    # ── IBAN ───────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="IBAN",
        pattern=_IBAN,
        base_confidence=0.78,
        validator="iban",
        context_type="IBAN",
        explanation="regex iban",
    ),
    # ── PHONE (en) ─────────────────────────────────────────────────────
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_EN,
        base_confidence=0.96,
        context_type="PHONE_NUMBER",
        explanation="regex phone (en)",
        language="en",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_ES,
        base_confidence=0.96,
        context_type="PHONE_NUMBER",
        explanation="regex phone (es)",
        language="es",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_FR,
        base_confidence=0.96,
        context_type="PHONE_NUMBER",
        explanation="regex phone (fr)",
        language="fr",
    ),
    # ── PERSON_NAME (title-prefix, multilingual) ───────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_EN,
        base_confidence=0.86,
        context_type="PERSON_NAME",
        explanation="regex person (en)",
        language="en",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_ES,
        base_confidence=0.86,
        context_type="PERSON_NAME",
        explanation="regex person (es)",
        language="es",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FR,
        base_confidence=0.86,
        context_type="PERSON_NAME",
        explanation="regex person (fr)",
        language="fr",
        deny_check=True,
    ),
    # ── PERSON_NAME (full name) ────────────────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FULL_NAME,
        base_confidence=0.84,
        context_type="PERSON_NAME",
        explanation="regex full name",
        deny_check=True,
    ),
    # ── PERSON_NAME (first + initial) ──────────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FIRST_INITIAL,
        base_confidence=0.85,
        context_type="PERSON_NAME",
        explanation="regex first+initial",
        deny_check=True,
    ),
    # ── PERSON_NAME (surname context) ──────────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_SURNAME_CONTEXT,
        base_confidence=0.81,
        group=1,
        explanation="regex surname context",
    ),
    # ── PERSON_NAME (alias context) ────────────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_ALIAS,
        base_confidence=0.82,
        group=1,
        explanation="regex alias name",
    ),
    # ── PERSON_NAME (keyword context) ──────────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_KEYWORD,
        base_confidence=0.83,
        group=1,
        explanation="regex person context",
    ),
    # ── PERSON_NAME (possessive context) ───────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_POSSESSIVE,
        base_confidence=0.84,
        group=1,
        explanation="regex person possessive",
        deny_check=True,
    ),
    # ── DATE_OF_BIRTH ──────────────────────────────────────────────────
    PatternSpec(
        entity_type="DATE_OF_BIRTH",
        pattern=_DOB_CONTEXT,
        base_confidence=0.85,
        group=1,
        explanation="regex dob context",
    ),
    # ── DATE_ISO ───────────────────────────────────────────────────────
    PatternSpec(
        entity_type="DATE_ISO",
        pattern=_DATE_ISO,
        base_confidence=0.85,
        group=1,
        validator="date_iso",
        explanation="regex date iso",
    ),
    # ── DATE_TIME (general) ────────────────────────────────────────────
    PatternSpec(
        entity_type="DATE_TIME",
        pattern=_DATE_GENERAL,
        base_confidence=0.78,
        group=1,
        explanation="regex date general",
    ),
    # ── MAC_ADDRESS ────────────────────────────────────────────────────
    PatternSpec(
        entity_type="MAC_ADDRESS",
        pattern=_MAC_ADDRESS,
        base_confidence=0.92,
        group=1,
        explanation="regex mac address",
        pre_filter=":",
    ),
    # ── DRIVERS_LICENSE ────────────────────────────────────────────────
    PatternSpec(
        entity_type="DRIVERS_LICENSE",
        pattern=_DRIVERS_LICENSE,
        base_confidence=0.80,
        group=1,
        explanation="regex drivers license",
    ),
    # ── PASSPORT ───────────────────────────────────────────────────────
    PatternSpec(
        entity_type="PASSPORT",
        pattern=_PASSPORT,
        base_confidence=0.82,
        group=1,
        explanation="regex passport",
    ),
    # ── ROUTING_NUMBER ─────────────────────────────────────────────────
    PatternSpec(
        entity_type="ROUTING_NUMBER",
        pattern=_ROUTING_NUMBER,
        base_confidence=0.83,
        group=1,
        validator="aba_routing",
        valid_confidence=0.93,
        invalid_confidence=0.83,
        explanation="regex routing number",
    ),
    # ── LICENSE_PLATE ──────────────────────────────────────────────────
    PatternSpec(
        entity_type="LICENSE_PLATE",
        pattern=_LICENSE_PLATE,
        base_confidence=0.78,
        group=1,
        explanation="regex license plate",
    ),
    # ── BANK_ACCOUNT ───────────────────────────────────────────────────
    PatternSpec(
        entity_type="BANK_ACCOUNT",
        pattern=_BANK_ACCOUNT,
        base_confidence=0.79,
        group=1,
        explanation="regex bank account",
    ),
    # ── NATIONAL_ID ────────────────────────────────────────────────────
    PatternSpec(
        entity_type="NATIONAL_ID",
        pattern=_NATIONAL_ID,
        base_confidence=0.78,
        group=1,
        explanation="regex national id",
    ),
    # ── USERNAME (@handle) ─────────────────────────────────────────────
    PatternSpec(
        entity_type="USERNAME",
        pattern=_USERNAME_AT,
        base_confidence=0.82,
        group=0,
        explanation="regex username @handle",
        pre_filter="@",
    ),
    # ── USERNAME (context) ─────────────────────────────────────────────
    PatternSpec(
        entity_type="USERNAME",
        pattern=_USERNAME_CONTEXT,
        base_confidence=0.80,
        group=1,
        explanation="regex username context",
    ),
    # ── EMPLOYEE_ID ────────────────────────────────────────────────────
    PatternSpec(
        entity_type="EMPLOYEE_ID",
        pattern=_EMPLOYEE_ID,
        base_confidence=0.80,
        group=1,
        explanation="regex employee id",
    ),
    # ── MEDICAL_RECORD_NUMBER ──────────────────────────────────────────
    PatternSpec(
        entity_type="MEDICAL_RECORD_NUMBER",
        pattern=_MEDICAL_RECORD,
        base_confidence=0.82,
        group=1,
        explanation="regex medical record",
    ),
    # ── ORGANIZATION ───────────────────────────────────────────────────
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION,
        base_confidence=0.80,
        group=0,
        explanation="regex organization",
        deny_check=True,
    ),
    # ── ADDRESS ────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="ADDRESS",
        pattern=_ADDRESS,
        base_confidence=0.82,
        group=1,
        explanation="regex address",
    ),
    # ── LOCATION ───────────────────────────────────────────────────────
    PatternSpec(
        entity_type="LOCATION",
        pattern=_LOCATION_CONTEXT,
        base_confidence=0.76,
        group=1,
        explanation="regex location context",
        deny_check=True,
    ),
    # ── CRYPTO_WALLET (Bitcoin legacy) ─────────────────────────────────
    PatternSpec(
        entity_type="CRYPTO_WALLET",
        pattern=_CRYPTO_BITCOIN,
        base_confidence=0.95,
        group=1,
        explanation="regex crypto bitcoin",
    ),
    PatternSpec(
        entity_type="CRYPTO_WALLET",
        pattern=_CRYPTO_BECH32,
        base_confidence=0.95,
        group=1,
        explanation="regex crypto bitcoin bech32",
    ),
    PatternSpec(
        entity_type="CRYPTO_WALLET",
        pattern=_CRYPTO_ETHEREUM,
        base_confidence=0.97,
        group=1,
        explanation="regex crypto ethereum",
        pre_filter="0x",
    ),
    # ── GPS_COORDINATES ────────────────────────────────────────────────
    PatternSpec(
        entity_type="GPS_COORDINATES",
        pattern=_GPS,
        base_confidence=0.88,
        validator="gps",
        explanation="regex gps coordinates",
    ),
    # ── SWIFT_BIC ──────────────────────────────────────────────────────
    PatternSpec(
        entity_type="SWIFT_BIC",
        pattern=_SWIFT_BIC,
        base_confidence=0.85,
        group=1,
        validator="swift_context",
        explanation="regex swift bic",
    ),
    # ── VIN ────────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="VIN",
        pattern=_VIN,
        base_confidence=0.80,
        group=1,
        validator="vin",
        valid_confidence=0.93,
        invalid_confidence=0.80,
        explanation="regex vin",
    ),
    # ── ZIP_CODE ───────────────────────────────────────────────────────
    PatternSpec(
        entity_type="ZIP_CODE",
        pattern=_ZIP_CODE,
        base_confidence=0.90,
        group=1,
        explanation="regex zip code",
    ),
    # ── CANADIAN_SIN ───────────────────────────────────────────────────
    PatternSpec(
        entity_type="CANADIAN_SIN",
        pattern=_CANADIAN_SIN,
        base_confidence=0.75,
        group=1,
        validator="sin_luhn",
        valid_confidence=0.92,
        invalid_confidence=0.75,
        explanation="regex canadian sin",
    ),
    # ── UK_NI_NUMBER ───────────────────────────────────────────────────
    PatternSpec(
        entity_type="UK_NI_NUMBER",
        pattern=_UK_NI,
        base_confidence=0.89,
        group=1,
        explanation="regex uk ni number",
    ),
    # ── JWT_TOKEN ──────────────────────────────────────────────────────
    PatternSpec(
        entity_type="JWT_TOKEN",
        pattern=_JWT,
        base_confidence=0.95,
        group=1,
        explanation="regex jwt token",
        pre_filter="eyJ",
    ),
    # ── API_KEY ────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="API_KEY",
        pattern=_API_KEY,
        base_confidence=0.91,
        group=1,
        explanation="regex api key",
    ),
    # ── AADHAAR ────────────────────────────────────────────────────────
    PatternSpec(
        entity_type="AADHAAR",
        pattern=_AADHAAR,
        base_confidence=0.80,
        group=1,
        validator="aadhaar",
        valid_confidence=0.91,
        invalid_confidence=0.80,
        explanation="regex aadhaar",
    ),
    # ── URL_WITH_PII (Phase 2) ─────────────────────────────────────────
    PatternSpec(
        entity_type="URL_WITH_PII",
        pattern=_URL_WITH_PII,
        base_confidence=0.87,
        group=0,
        explanation="regex url with pii",
        pre_filter="http",
    ),
    # ── AGE (Phase 2) ──────────────────────────────────────────────────
    PatternSpec(
        entity_type="AGE",
        pattern=_AGE,
        base_confidence=0.82,
        validator="age",
        context_type="AGE",
        explanation="regex age",
    ),
    # ── MEDICAL_LICENSE / NPI (Phase 2) ────────────────────────────────
    PatternSpec(
        entity_type="MEDICAL_LICENSE",
        pattern=_NPI,
        base_confidence=0.88,
        group=1,
        validator="npi",
        context_type="MEDICAL_LICENSE",
        explanation="regex npi",
    ),
    PatternSpec(
        entity_type="MEDICAL_LICENSE",
        pattern=_DEA,
        base_confidence=0.88,
        group=1,
        validator="dea",
        context_type="MEDICAL_LICENSE",
        explanation="regex dea number",
    ),
)
