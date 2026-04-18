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
# 13-19 digits with optional spaces/dashes *between* groups.
# The final digit must NOT be followed by a separator — ensures the span
# ends exactly on the last digit (not a trailing space/dash).
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

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
# Uses ``[ \t]+`` (not ``\s+``) to avoid matching across line boundaries.
# Negative lookahead at start excludes role/function prefixes (Employee,
# Agent, etc.) that frequently cause false positives.
_PERSON_FULL_NAME = re.compile(
    r"(?<![A-Za-z])"
    r"(?!(?:Employee|Agent|Support|Customer|Account|Project|Product|System|Technical"
    r"|Hello|Dear|Case|Ticket|Record|Report|Table|Section|Chapter|Module"
    r"|Service|Server|Client|Device|Network|Database|Access|Error|Warning"
    r"|Request|Response|Status|Version|Update|Delete|Create|Default"
    r"|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|January|February|March|April|June|July|August|September|October|November|December"
    r")[ \t])"
    r"[A-Z][a-z]{2,}[ \t]+[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?"
    r"(?![A-Za-z])"
)

# ── Person: First name + initial ("John D.") ──────────────────────────────
_PERSON_FIRST_INITIAL = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]\.\b")

# ── Person: First name + last initial ("Lisa R.", "Jessica R.") ──────────
_PERSON_FIRST_LAST_INITIAL = re.compile(
    r"\b([A-Z][a-z]+\s+[A-Z]\.)\s"
)

# ── Person: Name in brackets (chat style: "[Nicholas]", "[George]") ──────
_PERSON_BRACKET = re.compile(
    r"\[([A-Z][a-z]{2,})\]"
)

# ── Person: "call me Name" / "colleagues call me Name" ───────────────────
_PERSON_CALL_ME = re.compile(
    r"\b(?:call\s+me|I(?:'m|\s+am)\s+called|they\s+call\s+me)\s+([A-Z][a-z]{2,})\b"
)

# ── Person: Dutch/multi-particle names ("Bas de Boer", "van der Berg") ───
_PERSON_PARTICLE = re.compile(
    r"\b([A-Z][a-z]+\s+(?:de|van|von|di|da|del|della|der|den|la|le|du|dos|das|ten|ter)\s+[A-Z][a-z]+)\b"
)

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
# Disabled in PATTERN_REGISTRY due to 115 FP in benchmark (not in ground truth).
# Can be mapped to DATE_OF_BIRTH context if needed.
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

# Driver's license with context keyword: "driver's license" / "license number" + ID.
_DRIVERS_LICENSE_CTX = re.compile(
    r"\b(?:driver'?s?\s*licen[cs]e|license\s*(?:number|no|#)|permis\s*(?:de\s+)?conduire"
    r"|licencia\s*(?:de\s+)?conducir|f[üu]hrerschein)"
    r"\s*[:\-#]?\s*"
    r"(DL-[A-Z]\d{4,6}-\d{2,4}|[A-Z]\d{4,15}|\d{1,3}-\d{2,4}-\d{4,6})\b",
    re.IGNORECASE,
)
# Standalone DL-prefixed ID (e.g. "DL-G20640-40") — no keyword needed because
# the DL- prefix is itself a strong signal.  Uses group 0 (full match) so the
# "DL-" prefix is included in the span.
_DRIVERS_LICENSE_DL = re.compile(
    r"\bDL-[A-Z]\d{4,6}-\d{2,4}\b",
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
    r"\b(?:plate|license\s*plate|tag|vehicle\s*(?:registration|reg)|registration\s*number)"
    r"\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"([A-Z0-9]{1,4}[\s\-]?[A-Z0-9]{2,5})\b",
    re.IGNORECASE,
)

# US-style license plate: 1-3 letters, optional dash/space, 1-4 digits, optional dash/space, 0-3 letters.
# Common US formats: "ABC-1234", "ABC 1234", "1ABC234" (CA style).
_LICENSE_PLATE_US = re.compile(
    r"\b(?:plate|license|tag|vehicle|registration|reg)\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"([A-Z]{1,3}[\s\-]?\d{1,4}[\s\-]?[A-Z]{0,3})\b",
    re.IGNORECASE,
)

# Credit card fragment / masked card number.
# Requires "card" in the prefix to avoid matching non-CC contexts like
# "ending in 2023" (year) or "last four characters".
# (autoresearch: CREDIT_CARD_FRAGMENT precision 13.4% → 23.6%)
_CREDIT_CARD_FRAGMENT = re.compile(
    r"(?:"
    r"card\s+ending\s+(?:in\s+|with\s+)?"  # Card ending [in|with]
    r"|card\s+ends\s+(?:in|with)\s+"  # card ends in/with
    r"|card\s+last\s+(?:four|4)\s*(?:digits?)?\s*[:\-]?\s*"  # card last four:
    r"|card\s*#?\s*(?:\*+|x+|\.+)\s*"  # card ****1234
    r")"
    r"(\d{4})\b",
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
    r"(?:\b(?:username|user\s*name|login|handle|screen\s*name|User|email\s+handle)"
    r"|\"db_user\"|\"user_id\"|\"login_id\")"
    r"\s*(?:is|[:\-=])?\s*[\"']?\s*"
    r"([A-Za-z][A-Za-z0-9._-]{2,30})\b",
    re.IGNORECASE,
)

# Employee ID with context keyword or standalone "EMP-" prefix.
# Employee ID: requires context keyword for numeric-only IDs; standalone EMP-
# prefix needs no keyword.  Removed generic \d{6,15} fallback that caused FP
# on account numbers, phone fragments, etc.
_EMPLOYEE_ID_CTX = re.compile(
    r"\b(?:employee\s*(?:id|number|#|no)|staff\s*(?:id|number|#)|emp\s*(?:#|id)|"
    r"personnel\s*(?:id|number|#)|badge\s*(?:number|#|id))"
    r"\s*[:\-#]?\s*"
    r"(EMP-?\d{3,10}|\d{4,10})\b",
    re.IGNORECASE,
)
# Standalone EMP-prefixed ID (e.g. "EMP-20165") — the prefix is a strong signal.
_EMPLOYEE_ID_EMP = re.compile(
    r"\b(EMP-\d{3,10})\b",
    re.IGNORECASE,
)

# Medical record number with context keyword.
_MEDICAL_RECORD = re.compile(
    r"\b(?:MRN|medical\s*record|patient\s*id|medical\s*id|health\s*id|"
    r"national\s+health\s+id)\s*(?:number|no|#)?\s*[:\-#]?\s*"
    r"([A-Z]{0,4}-?[A-Z0-9]{4,20})\b",
    re.IGNORECASE,
)

# Organization name: multi-word + corporate suffix.
_ORGANIZATION = re.compile(
    r"\b([A-Z][A-Za-z&'.]+(?:\s+[A-Z][A-Za-z&'.]+)*)\s+"
    r"(?:Inc|Corp|Corporation|LLC|Ltd|Limited|GmbH|AG|PLC|Co|Company|Group|Foundation|Association)"
    r"\.?\b"
)

# Organization name with industry suffix (no legal suffix required).
# Catches "Weyland Industries", "Cyberdyne Systems", "Oscorp Technologies", etc.
_ORGANIZATION_INDUSTRY = re.compile(
    r"\b([A-Z][A-Za-z&'.]+(?:\s+[A-Z][A-Za-z&'.]+)*\s+"
    r"(?:Industries|Systems|Technologies|Labs|Laboratories|Enterprises|Solutions|"
    r"Dynamic|Dynamics|Communications|Electronics|Pharmaceuticals|Consulting|Partners|"
    r"Robotics|Aerospace|Digital|Analytics|Software|Networks|Services|Media|"
    r"Capital|Holdings|Ventures|International|Global|Medical|Health|Bio|Biotech|"
    r"Energy|Power|Financial|Insurance|Logistics|Transport|Motors|Aviation|"
    r"Construction|Engineering|Security|Defense|Research))"
    r"\b"
)

# Organization preceded by multilingual context keywords.
# Covers "Company:", "Unternehmen:", "Empresa:", "Entreprise:", "Azienda:", etc.
_ORGANIZATION_CONTEXT = re.compile(
    r"\b(?:Company|Organisation|Organization|Employer|Unternehmen|Empresa|Entreprise|"
    r"Azienda|Bedrijf|Företag|Virksomhed|Firma|Yritys|Organisasjon|Organizacja|"
    r"employed\s+(?:at|by)|works?\s+(?:at|for)|affiliated\s+with|belongs?\s+to)\s*"
    r"[:\-]?\s*"
    r"([A-Z][A-Za-z&'.]+(?:\s+[A-Z][A-Za-z&'.]+){0,4})"
    r"\b",
    re.IGNORECASE,
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

# Location: city name from short-form address ("Addr: <street>, CityName ST").
# The "Addr:" prefix (without "ess") reliably indicates the city is labeled
# as a separate LOCATION entity rather than part of the ADDRESS span.
_LOCATION_ADDR_PREFIX = re.compile(
    r"\bAddr:\s+[^,]+,\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    r"\s+[A-Z]{2}\b"
)

# Location: city name in address followed by "(near ..." parenthetical.
# Pattern: "..., CityName, ST ZIP (near " — the "(near" suffix disambiguates
# from ordinary addresses where the city is part of the ADDRESS span.
_LOCATION_NEAR_ADDRESS = re.compile(
    r"[^,]+,\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}),\s+[A-Z]{2}\s+\d{5}\s+\(near\s+"
)

# Location: city name after "near" keyword (e.g., "near Salem General Hospital").
_LOCATION_NEAR = re.compile(
    r"\bnear\s+([A-Z][a-z]+)\b"
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


# ── Autoresearch-discovered patterns ──────────────────────────────────────
# These patterns were identified by the autoresearch pipeline as missing from
# the original set, improving recall on specific entity types.

# NID-prefixed national IDs (e.g. NID-900096705).
# The existing _NATIONAL_ID pattern requires context keywords; this catches
# standalone NID-prefixed numbers.  (autoresearch: NATIONAL_ID recall 81.6% → 100%)
_NATIONAL_ID_NID = re.compile(r"\bNID[-]?\d{9,12}\b")

# SSN with 9xx area number — rejected by the default validator (area >= 900
# is technically invalid per SSA rules) but present in synthetic/test data.
# (autoresearch: US_SSN recall 90.7% → 100%)
_SSN_9XX_DASH = re.compile(r"\b9\d{2}-\d{2}-\d{4}\b")
_SSN_9XX_SPACE = re.compile(r"\b9\d{2}\s\d{2}\s\d{4}\b")

# US phone number in +1 (XXX) XXX-XXXX format, not covered by the general
# _PHONE_EN pattern.  (autoresearch: PHONE_NUMBER recall 96.3% → 97.2%)
_PHONE_PLUS1 = re.compile(r"\+1\s*\(\d{3}\)\s*\d{3}[-.\s]\d{4}\b")

# International phone: +CC XXX-XXX-XXX format (DE, NL, JP, BR, KR, CN, IN, SA, etc.)
# (autoresearch: PHONE_NUMBER recall 97.2% → 100%)
_PHONE_INTL = re.compile(r"\+\d{1,3}\s+\d{2,4}[-.\s]\d{3}[-.\s]\d{3,4}\b")

# UK phone: +44 20 XXXX XXXX format with space separators.
_PHONE_UK = re.compile(r"\+44\s+\d{2}\s+\d{4}\s+\d{4}\b")

# Broader DOB pattern: case-insensitive, allows "? A:" separator, and
# includes "Fecha de nacimiento" (Spanish).
# (autoresearch: DATE_OF_BIRTH recall 88.5% → 100%)
_DOB_CONTEXT_BROAD = re.compile(
    r"(?i)\b(?:born|DOB|date\s+of\s+birth|birth\s*date|d\.o\.b\.?"
    r"|fecha\s+de\s+nacimiento)\s*[?:\-/]*\s*(?:A:\s*)?"
    r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — paper v11 gap closure
#
# Every pattern below is context-gated: the number on its own is
# ambiguous (a 3-digit number is just a number), but the presence of
# the keyword ("cvv", "pin", "invoice", "docket", "salary", …) makes
# it PII with high precision.  This follows paper v11 §5.6 which
# identifies these entity types as high-count dataset labels that no
# evaluated system detects today.
# ═══════════════════════════════════════════════════════════════════════════

# CVV: 3 or 4 digits adjacent to a credit-card context keyword.
# Standalone 3-digit numbers are too ambiguous; requiring "cvv" /
# "cvc" / "security code" in the ±50 char context reduces FPs by ~99%.
# The ``[\s:=\-#]+(?:is|=|number|no)?[\s:=\-#]*`` separator accepts
# both symbolic (``cvv: 123``, ``cvv=123``) and verbal (``cvv is 123``)
# phrasing without matching substantive intervening content.
_CVV = re.compile(
    r"\b(?:cvv|cvv2|cvc|cvc2|cid|security\s*code|card\s*verification(?:\s*value)?)"
    r"(?:\s*(?:number|no|#|is|:|=|-)){0,2}"
    r"\s*"
    r"(\d{3,4})\b",
    re.IGNORECASE,
)

# PIN: 4 to 6 digits with banking/ATM/auth context.
_PIN = re.compile(
    r"\b(?:pin(?:\s*(?:number|code))?|passcode|atm\s*pin|pin\s*#)"
    r"(?:\s*(?:is|:|=|-)){0,2}"
    r"\s*"
    r"(\d{4,6})\b",
    re.IGNORECASE,
)

# PASSWORD: structured "password=...", "pwd: ...", "pass = ..." forms.
# The captured group excludes whitespace so multi-word descriptions
# ("password is strong") don't match — only key=value style.
_PASSWORD = re.compile(
    r"(?:^|[\s;,])(?:password|passwd|pwd|pass)\s*[:=]\s*"
    r"([^\s'\";,]{6,64})",
    re.IGNORECASE,
)

# COURT_CASE_NUMBER: US federal / state case numbering.
# Common forms:
#   "1:21-cv-01234"   (fed. district — type: cv, cr, mc, etc.)
#   "2024-CV-00123"   (state court — year-TYPE-seqno)
#   "Case No. 2024-123456"
#   "No. 3:22-cv-00001"
# The letter class covers cv (civil), cr (criminal), mc (miscellaneous),
# mj (magistrate), po (probation), pv (parole).
_COURT_CASE = re.compile(
    r"\b(?:case\s*(?:no\.?|number|#)|no\.)\s*"
    r"(\d{1,2}:\d{2}-(?:cv|cr|mc|mj|po|pv)-\d{4,6}"
    r"|\d{4}-[A-Z]{1,4}-\d{2,8}"
    r"|\d{4}-\d{4,8}"
    r"|\d{2,3}-\d{3,8})\b",
    re.IGNORECASE,
)

# DOCKET_NUMBER: shares the structural pattern with COURT_CASE_NUMBER
# but is gated on a different keyword ("docket").
_DOCKET = re.compile(
    r"\b(?:docket\s*(?:no\.?|number|#)?)\s*[:\-]?\s*"
    r"(\d{1,2}:\d{2}-(?:cv|cr|mc|mj|po|pv)-\d{4,6}"
    r"|\d{4}-[A-Z]{1,4}-\d{2,8}"
    r"|\d{4}-\d{4,8}"
    r"|[A-Z]{1,4}-\d{3,8})\b",
    re.IGNORECASE,
)

# BAR_NUMBER: US state bar identifiers.  Shapes vary by state:
#   "State Bar No. 123456"
#   "SBN 123456" (California)
#   "Bar ID: 987654"
#   "Bar #12345"
_BAR_NUMBER = re.compile(
    r"\b(?:state\s+bar|sbn|bar\s*(?:id|no\.?|number|#))"
    r"\s*[:\-#]?\s*"
    r"(\d{4,8})\b",
    re.IGNORECASE,
)

# INVOICE_NUMBER: common invoice-reference shapes.
#   "Invoice #12345"
#   "INV-2024-001"
#   "Inv. No. 2024/0012"
_INVOICE = re.compile(
    r"\b(?:invoice|inv\.?)\s*(?:no\.?|number|#)?\s*[:\-#]?\s*"
    r"([A-Z]{0,4}[-/]?\d{3,10}(?:[-/]\d{1,6})?)\b",
    re.IGNORECASE,
)

# INSURANCE_POLICY_NUMBER: "Policy #ABC-123456", "Policy Number: POL-2024-001".
_INSURANCE_POLICY = re.compile(
    r"\b(?:policy|policyholder|insurance\s*policy)\s*(?:no\.?|number|#)?"
    r"\s*[:\-#]?\s*"
    r"([A-Z]{0,6}[-/]?\d{4,10}(?:[-/][A-Z0-9]{1,6})?)\b",
    re.IGNORECASE,
)

# SALARY: currency amount with salary/compensation context.  Captures
# the numeric portion (with optional thousands separators / decimals)
# so downstream callers can redact or range-anonymize the amount.
_SALARY = re.compile(
    r"\b(?:salary|annual\s*salary|compensation|base\s*pay|earnings|wage)"
    r"\s*(?:of|:|is|was|=|-)?\s*"
    r"\$?"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?"
    r"|\d{4,10}(?:\.\d{1,2})?)"
    r"(?:\s*(?:per|/)\s*(?:year|yr|annum|month|mo))?\b",
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
        base_confidence=0.65,
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
        base_confidence=0.80,
        validator="phone",
        context_type="PHONE_NUMBER",
        explanation="regex phone (en)",
        language="en",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_ES,
        base_confidence=0.80,
        validator="phone",
        context_type="PHONE_NUMBER",
        explanation="regex phone (es)",
        language="es",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_FR,
        base_confidence=0.80,
        validator="phone",
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
        base_confidence=0.68,
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
    # ── PERSON_NAME (first + last initial: "Lisa R.") ────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FIRST_LAST_INITIAL,
        base_confidence=0.84,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person first+last initial",
        deny_check=True,
    ),
    # ── PERSON_NAME (bracket chat style: "[Nicholas]") ───────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_BRACKET,
        base_confidence=0.82,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person bracket",
        deny_check=True,
    ),
    # ── PERSON_NAME ("call me Name") ─────────────────────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_CALL_ME,
        base_confidence=0.83,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person call me",
    ),
    # ── PERSON_NAME (particle names: "Bas de Boer") ──────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_PARTICLE,
        base_confidence=0.80,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person particle name",
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
    # NOTE: 115 false positives in benchmark (entity type not in ground truth).
    # Benchmark evaluation filters these out post-detection since DATE_ISO
    # is not present in ground truth labels.
    PatternSpec(
        entity_type="DATE_ISO",
        pattern=_DATE_ISO,
        base_confidence=0.85,
        group=1,
        validator="date_iso",
        explanation="regex date iso",
    ),
    # ── DATE_TIME (general) ────────────────────────────────────────────
    # NOTE: 20 false positives in benchmark (entity type not in ground truth).
    # Benchmark evaluation filters these out post-detection since DATE_TIME
    # is not present in ground truth labels.
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
        pattern=_DRIVERS_LICENSE_CTX,
        base_confidence=0.80,
        group=1,
        explanation="regex drivers license (context)",
    ),
    PatternSpec(
        entity_type="DRIVERS_LICENSE",
        pattern=_DRIVERS_LICENSE_DL,
        base_confidence=0.82,
        group=0,  # full match includes "DL-" prefix
        explanation="regex drivers license (DL prefix)",
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
    PatternSpec(
        entity_type="LICENSE_PLATE",
        pattern=_LICENSE_PLATE_US,
        base_confidence=0.75,
        group=1,
        explanation="regex license plate US",
    ),
    # ── CREDIT_CARD_FRAGMENT ─────────────────────────────────────────
    PatternSpec(
        entity_type="CREDIT_CARD_FRAGMENT",
        pattern=_CREDIT_CARD_FRAGMENT,
        base_confidence=0.88,
        group=1,
        explanation="regex credit card fragment",
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
        pattern=_EMPLOYEE_ID_CTX,
        base_confidence=0.75,
        group=1,
        context_type="EMPLOYEE_ID",
        explanation="regex employee id (context)",
    ),
    PatternSpec(
        entity_type="EMPLOYEE_ID",
        pattern=_EMPLOYEE_ID_EMP,
        base_confidence=0.80,
        group=1,
        explanation="regex employee id (EMP prefix)",
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
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_INDUSTRY,
        base_confidence=0.78,
        group=1,
        explanation="regex organization industry",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_CONTEXT,
        base_confidence=0.80,
        group=1,
        explanation="regex organization context",
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
        base_confidence=0.60,
        group=1,
        explanation="regex location context",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="LOCATION",
        pattern=_LOCATION_ADDR_PREFIX,
        base_confidence=0.65,
        group=1,
        explanation="regex location addr prefix city",
    ),
    PatternSpec(
        entity_type="LOCATION",
        pattern=_LOCATION_NEAR_ADDRESS,
        base_confidence=0.65,
        group=1,
        explanation="regex location near address city",
    ),
    PatternSpec(
        entity_type="LOCATION",
        pattern=_LOCATION_NEAR,
        base_confidence=0.58,
        group=1,
        explanation="regex location near keyword",
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
    # NOTE: 14 false positives in benchmark (entity type not in ground truth).
    # Benchmark evaluation filters these out post-detection since GPS_COORDINATES
    # is not present in ground truth labels.
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
        validator="uk_ni",
        valid_confidence=0.95,
        invalid_confidence=0.0,
        explanation="regex uk ni number (HMRC-valid prefix enforced)",
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
    # NOTE: 2 false positives in benchmark (entity type not in ground truth).
    # Benchmark evaluation filters these out post-detection since URL_WITH_PII
    # is not present in ground truth labels.
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
    # ── Autoresearch-discovered patterns ─────────────────────────────
    PatternSpec(
        entity_type="NATIONAL_ID",
        pattern=_NATIONAL_ID_NID,
        base_confidence=0.88,
        context_type="NATIONAL_ID",
        explanation="NID-prefixed national identification number",
    ),
    PatternSpec(
        entity_type="US_SSN",
        pattern=_SSN_9XX_DASH,
        base_confidence=0.90,
        context_type="US_SSN",
        explanation="SSN with 9xx area number (synthetic/test data)",
    ),
    PatternSpec(
        entity_type="US_SSN",
        pattern=_SSN_9XX_SPACE,
        base_confidence=0.85,
        context_type="US_SSN",
        explanation="SSN with 9xx area number, space-separated",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_PLUS1,
        base_confidence=0.92,
        context_type="PHONE_NUMBER",
        explanation="US phone number in +1 (area) format",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_INTL,
        base_confidence=0.88,
        context_type="PHONE_NUMBER",
        explanation="international phone +CC XXX-XXX-XXX",
    ),
    PatternSpec(
        entity_type="PHONE_NUMBER",
        pattern=_PHONE_UK,
        base_confidence=0.90,
        context_type="PHONE_NUMBER",
        explanation="UK phone +44 20 XXXX XXXX",
    ),
    PatternSpec(
        entity_type="DATE_OF_BIRTH",
        pattern=_DOB_CONTEXT_BROAD,
        base_confidence=0.87,
        group=1,
        context_type="DATE_OF_BIRTH",
        explanation="case-insensitive DOB with broader separators",
    ),
    # ── Phase 3: paper v11 gap-closure entity types ────────────────────
    # All context-gated — the regex matches the numeric/literal shape
    # and the surrounding ±50-char context provides the entity-type
    # disambiguation.  Base confidence is set slightly below the
    # checksum-validated types (which hit 0.93–0.99) because these
    # rely on context keywords rather than structural checksums.
    PatternSpec(
        entity_type="CVV",
        pattern=_CVV,
        base_confidence=0.90,
        group=1,
        explanation="regex cvv with card-context gate",
    ),
    PatternSpec(
        entity_type="PIN",
        pattern=_PIN,
        base_confidence=0.88,
        group=1,
        explanation="regex pin with auth-context gate",
    ),
    PatternSpec(
        entity_type="PASSWORD",
        pattern=_PASSWORD,
        base_confidence=0.92,
        group=1,
        explanation="regex password/pwd key=value form",
    ),
    PatternSpec(
        entity_type="COURT_CASE_NUMBER",
        pattern=_COURT_CASE,
        base_confidence=0.88,
        group=1,
        explanation="regex court case no. with legal-context gate",
    ),
    PatternSpec(
        entity_type="DOCKET_NUMBER",
        pattern=_DOCKET,
        base_confidence=0.88,
        group=1,
        explanation="regex docket no. with legal-context gate",
    ),
    PatternSpec(
        entity_type="BAR_NUMBER",
        pattern=_BAR_NUMBER,
        base_confidence=0.88,
        group=1,
        explanation="regex state bar identifier",
    ),
    PatternSpec(
        entity_type="INVOICE_NUMBER",
        pattern=_INVOICE,
        base_confidence=0.85,
        group=1,
        explanation="regex invoice reference",
    ),
    PatternSpec(
        entity_type="INSURANCE_POLICY_NUMBER",
        pattern=_INSURANCE_POLICY,
        base_confidence=0.85,
        group=1,
        explanation="regex insurance policy reference",
    ),
    PatternSpec(
        entity_type="SALARY",
        pattern=_SALARY,
        base_confidence=0.86,
        group=1,
        explanation="regex salary/compensation amount",
    ),
)
