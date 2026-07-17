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

from pii_anon.engines.regex.unicode_norm import COMBINING_MARKS_CLASS


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
# Standard email: local-part @ domain . TLD (2+ chars).  The local part
# accepts Unicode word characters (\w: any-script letters/digits/_ — corpus
# emails carry Hangul/CJK/Arabic/diacritic local parts, e.g.
# 예진.박86@outlook.com); domain and TLD stay ASCII-only, so IDN domains
# remain out of scope.
# The CONTINUATION class additionally accepts category-M combining marks:
# ``\w`` excludes them, so an abugida/harakat local part (Thai vowel signs,
# Bengali matras, Arabic diacritics — e.g. สมชาย.ทองดี88@outlook.com) used to
# terminate at the first mark, scoring as a missed email AND a truncated-span
# FP. The FIRST character class is unchanged (marks never begin a local part),
# so every previous match start is preserved — strictly additive widening.
_EMAIL = re.compile(
    r"\b[\w.%+-][\w.%+-" + COMBINING_MARKS_CLASS + r"]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

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
# The annotation convention is SPLIT by mention shape (sp2 dev-iteration 2/3
# evidence): a title + FULL name excludes the honorific ("Dr. ⟦Karen
# Anderson⟧") while a title + bare SURNAME includes it ("⟦Ms. Davis⟧" — the
# title+surname pair functions as the name unit). One pattern per shape;
# the full-name tail carries the next-field-label guard.
_PERSON_EN = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?![ \t]*[:=]))\b"
)
_PERSON_EN_SURNAME = re.compile(
    r"\b((?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+)\b(?!\s+[A-Z][a-z]+)"
)
_PERSON_ES = re.compile(
    r"\b(?:Sr|Sra|Srta|Dra)\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?![ \t]*[:=]))\b"
)
_PERSON_ES_SURNAME = re.compile(
    r"\b((?:Sr|Sra|Srta|Dra)\.?\s+[A-Z][a-z]+)\b(?!\s+[A-Z][a-z]+)"
)
_PERSON_FR = re.compile(
    r"\b(?:M|Mme|Dr)\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?![ \t]*[:=]))\b"
)
_PERSON_FR_SURNAME = re.compile(
    r"\b((?:M|Mme|Dr)\.?\s+[A-Z][a-z]+)\b(?!\s+[A-Z][a-z]+)"
)

# ── Person: Full name (2-3 capitalized words, no title) ───────────────────
# Uses ``[ \t]+`` (not ``\s+``) to avoid matching across line boundaries.
# Negative lookahead at start excludes role/function prefixes (Employee,
# Agent, etc.) that frequently cause false positives.
#
# Field-label nouns: a Title-Case candidate whose 2nd/3rd token is one of
# these is a form-field label (e.g. "Bic Code", "Medication Name"),
# not a person name — the single largest PERSON_NAME FP factory on the DATA
# corpus.  "Name" is included beyond the core vocabulary because "<X> Name"
# is itself the most generic field label ("Medication Name", "Patient Name")
# while "Name" as a surname is essentially nonexistent.
#
# Tuple design (one word per element, joined at compile time): avoids
# accidental implicit string concatenation fusing adjacent elements into
# a single token (e.g. "Records""Record" → "RecordsRecord"). Longest-first
# ordering is preserved where prefixes exist (Identifier before Id,
# Records before Record, Licence/License both present).
#
# "Swift" measured 0 FP-delta at n=2000 (removed 2026-06-11): the Bic/Code
# neighbors already cover SWIFT-BIC field labels, while keeping Swift was
# blocking Taylor-Swift-class real names (production text).
#
# Adjacency trade: a real 2-token name immediately followed by a Title-Case
# vocab noun ("John Smith Address: ...") is also suppressed — inherent to the
# trailing-window design, F2-justified.
#
# Plural probe (s? suffix, measured 2026-06-11): FP drop = 3 at n=2000
# (threshold ≥5 not met) — reverted per the decision rule; the guard still
# catches the singular-plural cases present in the 2000-record draw.
_FIELD_LABEL_NOUNS: tuple[str, ...] = (
    "Number",
    "Code",
    "License",
    "Licence",
    "Insurance",
    "Account",
    "Routing",
    "Records",   # longest first (before Record)
    "Record",
    "Medication",
    "Docket",
    "Case",
    "Policy",
    "Invoice",
    "Member",
    "Security",
    "Registration",
    "Identifier",  # longest first (before Id)
    "Id",
    "Bic",
    # "Swift" removed: 0 FP-delta at n=2000; Bic/Code cover SWIFT-BIC labels;
    # keeping it blocked Taylor-Swift-class real person names in production.
    "Notary",
    "Plate",
    "Passport",
    "Diagnosis",
    "Procedure",
    "Condition",
    "Salary",
    "Username",
    "Address",
    "Status",
    "Level",
    "Type",
    "Date",
    "Time",
    "Name",
    "ID",          # uppercase sibling of Id ("Employer Tax ID:")
    "Confirmation",  # document headers ("Wire Transfer Confirmation")
    "Receipt",
    "Statement",
    "Summary",
)

# Field-label words that, directly before a colon, announce the NEXT form
# field — a 2nd name token must not consume them ("Donald Rodriguez Email:").
# Kept to KNOWN labels so the dialogue-speaker form ("Daniel Moore: No
# further questions" — the surname IS gold) is not rejected.
_NEXT_FIELD_LABEL_WORDS = (
    "Email|Phone|Fax|Mobile|Website|Address|Username|Password|Title|Dept|"
    "Department|Position|Manager|Supervisor"
)

_PERSON_FULL_NAME = re.compile(
    r"(?<![A-Za-z])"
    r"(?!(?:Employee|Employer|Agent|Support|Customer|Account|Project|Product|Systems?|Technical"
    r"|Hello|Dear|Case|Ticket|Record|Report|Table|Section|Chapter|Module"
    r"|Service|Server|Client|Device|Network|Database|Access|Error|Warning"
    r"|Request|Response|Status|Version|Update|Delete|Create|Default|Wire"
    # Role nouns that precede a real name in the DATA corpus ("Patient
    # Ronald Jackson", "Contact Robert Anderson"): excluding them here makes
    # the match START at the name, fixing the strict-extent FN+FP pair
    # (sp2 dev-iteration 1 evidence).
    r"|Patient|Contact|Applicant|Resident|Attn|Insured|Claimant|Defendant"
    r"|Plaintiff|Witness|Tenant|Beneficiary|Recipient|Sender|Doctor|Nurse"
    r"|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|January|February|March|April|June|July|August|September|October|November|December"
    r")[ \t])"
    # Trailing-token guard: block candidates whose 2nd or 3rd capitalized
    # word is a field-label noun ("Docket Number", "Health Insurance").
    # (An alternative followed-by-colon guard measured -964 FP but -54 TP at
    # n=2000 — a net micro-F2 loss vs this rule's -932 FP at zero TP loss.)
    r"(?![A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?[ \t]+"
    r"(?:" + "|".join(_FIELD_LABEL_NOUNS) + r")(?![A-Za-z]))"
    # Tail-token guards: the 2nd token must not be a KNOWN field-label word
    # before a colon ("Donald Rodriguez␉Email: ..." — but "Daniel Moore: No
    # further questions" keeps the surname, the dialogue-speaker gold form);
    # the 3rd token must not directly precede a colon at all. A leading
    # street number blocks the candidate ("9953 Dogwood Ct" is an address).
    r"(?<![0-9][ \t])"
    # P4 (sp7 #6): capturing atoms widened to Latin diacritics ("Fabián
    # Montalbán", "François Gilbert"); the guard lookaheads above stay ASCII.
    r"[A-ZÀ-ÖØ-ÞĀ-ſ][a-zß-öø-ÿĀ-ſ]{2,}[ \t]+(?!(?:" + _NEXT_FIELD_LABEL_WORDS + r")[ \t]*[:=])"
    r"[A-ZÀ-ÖØ-ÞĀ-ſ][a-zß-öø-ÿĀ-ſ]+(?:[ \t]+[A-ZÀ-ÖØ-ÞĀ-ſ][a-zß-öø-ÿĀ-ſ]+(?![ \t]*[:=]))?"
    r"(?![A-Za-z])"
)

# ── sp7 #6 — name grammar & span hygiene (SAFE subset, additive) ──────────
# Latin letter classes incl. diacritics (Latin-1 + Latin Extended-A) so
# Çağlayan / Sarısoy / Montalbán / Treves-Torlonia capture in full.
_NAME_U = "A-ZÀ-ÖØ-ÞĀ-ſ"
_NAME_LO = "a-zß-öø-ÿĀ-ſ"
_TOK_TITLED = rf"[{_NAME_U}][{_NAME_U}{_NAME_LO}]+(?:[-'’][{_NAME_U}{_NAME_LO}]+)*"
_HONORIFIC = (
    r"(?:Dr|Mr|Mrs|Ms|Miss|Mx|Prof|Sir|Lord|Lady|Mme|Mlle|Sr|Sra|Srta|Dra|Hon)"
)
# P1 — honorific + Unicode full name (captures the name sans title, group=1).
_PERSON_TITLE_FULL_U = re.compile(
    rf"\b{_HONORIFIC}\.?[ \t]+({_TOK_TITLED}"
    rf"(?:[ \t]+(?!(?:{_NEXT_FIELD_LABEL_WORDS})[ \t]*[:=]){_TOK_TITLED}){{1,2}})"
)
# P2 — honorific + initial(s) + surname ("Mr S. Esmer" -> "S. Esmer").
_PERSON_TITLE_INITIALS = re.compile(
    rf"\b{_HONORIFIC}\.?[ \t]+((?:[{_NAME_U}]\.){{1,3}}[ \t]*{_TOK_TITLED})"
)
# P3 — untitled First + middle-initial + Surname, with a document-structure
# negative guard ("Section A. Overview" is not a person).
_NAME_SECTION_WORDS = (
    r"Section|Chapter|Part|Article|Exhibit|Appendix|Figure|Table|Item|Note|"
    r"Schedule|Annex|Clause|Paragraph|Volume|Page|Line|Step|Phase|Level|Class|"
    r"Type|Form"
)
_PERSON_FIRST_MIDINITIAL_LAST = re.compile(
    rf"(?<![{_NAME_U}])(?!(?:{_NAME_SECTION_WORDS})[ \t])"
    rf"([{_NAME_U}][{_NAME_LO}]{{2,}}[ \t]+[{_NAME_U}]\.[ \t]+[{_NAME_U}][{_NAME_LO}]+)\b"
)
# P3C — ALL-CAPS variant. The middle-initial dot is REQUIRED (this is the only
# anchor that keeps ALL-CAPS from re-opening the A1 header FP flood).
_PERSON_FIRST_MIDINITIAL_LAST_CAPS = re.compile(
    rf"(?<![A-Z])(?!(?:{_NAME_SECTION_WORDS.upper()})[ \t])"
    rf"([{_NAME_U}]{{3,}}[ \t]+[{_NAME_U}]\.[ \t]+[{_NAME_U}]{{3,}})\b"
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
# The keyword block is case-insensitive (scoped ``(?i:...)``): corpus text
# capitalizes role nouns at sentence/field starts ("Patient Ronald Jackson")
# and the case-sensitive variant silently lost the correct-extent match
# (sp2 dev-iteration 1 evidence). The CAPTURE stays case-sensitive — the
# name shape itself must be Title-Case. The optional 2nd captured token
# carries the same next-field-label guard as ``_PERSON_FULL_NAME``.
_PERSON_KEYWORD = re.compile(
    r"\b(?i:name\s+is|patient|employee|client|resident|member|user|"
    r"account\s+holder|beneficiary|author|sender|recipient|contact|"
    r"applicant|insured|claimant|defendant|plaintiff|witness|tenant|"
    r"signed\s+by|submitted\s+by|prepared\s+by|reviewed\s+by|assigned\s+to)"
    # An honorific between keyword and name is skipped, NOT captured
    # ("signed by Dr. Robert Torres" must not emit the bare "Dr").
    r"\s+(?:(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+)?"
    r"([A-Z][a-z]+(?:\s+(?!(?:" + _NEXT_FIELD_LABEL_WORDS + r")[ \t]*[:=])[A-Z][a-z]+)?)\b"
)

# ── Person: field-label name ("Name: John Smith", "Full Name - Jane Doe") ──
# The bare "<...> Name: <value>" form is the dominant shape in form-style
# corpus records; no prior pattern covered it ("name is" required the verb).
# ``\b`` keeps "Username:" safe (no boundary between "User" and "name");
# tail tokens carry the next-field-label guard.
_PERSON_LABELED = re.compile(
    r"\b(?i:(?:full|legal|employee|patient|customer|client|account|user|contact)?"
    r"[ \t]*name)[ \t]*[:\-=][ \t]*"
    r"(?:(?:Dr|Mr|Mrs|Ms|Prof)\.?[ \t]+)?"
    r"([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+(?![ \t]*[:=])){0,2})"
    # The trailing guard stops backtracking from shaving a rejected label
    # token into a sub-token ("Email:" -> "Emai") to dodge the colon guard.
    r"(?![A-Za-z])"
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
    # The dotted/slashed numeric form must not fire on FRAGMENTS of dotted
    # number runs (IP addresses: "208.⟦74.38.190⟧"): reject when preceded by
    # a digit/dot or followed by (optionally a dot then) another digit.
    r"(?<![\d.])\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}(?!\.?\d)"
    r")\b",
    re.IGNORECASE,
)

# sp7 A3 — natural-language / locale prose date grammar (mining candidate #5).
# Covers the day-first, ordinal, legal and Month-year forms the ASCII
# "Month d, yyyy" + numeric grammar above misses. Dominant in TAB/ECHR court
# prose ("27 May 1994", "1st day of January, 2023", "May 1994"). ADDITIVE
# (emits DATE_TIME, benchmark-ignored on home scoring, masked in production).
# Boundary hygiene: the match starts at the day/month token, never a leading
# determiner ("no later than March 20, 2023" -> "March 20, 2023").
_MONTH_NAME = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)"
_DATE_PROSE = re.compile(
    r"(?<![\w.])("
    # legal: "1st day of January, 2023"
    r"\d{1,2}" + _ORD + r"\s+day\s+of\s+" + _MONTH_NAME + r",?\s+\d{4}"
    r"|"
    # day-(of-)Month-(year): "27 May 1994", "27th May 1994", "3rd of April 1980"
    r"\d{1,2}" + _ORD + r"?\s+(?:of\s+)?" + _MONTH_NAME + r"(?:,?\s+\d{2,4})?"
    r"|"
    # Month-day-(year): "May 27, 1994", "Jan 15 2025", "May 27th 1994"
    + _MONTH_NAME + r"\s+\d{1,2}" + _ORD + r"?,?\s+\d{2,4}"
    r"|"
    # Month-year: "May 1994", "January 2023"
    + _MONTH_NAME + r"\s+\d{4}"
    r")(?![\w])",
    re.IGNORECASE,
)

# Space-separated datetime "2021-10-23 09:53:00 UTC" — the log/report shape the
# T-separated ISO pattern below misses.
_DATETIME_SPACE = re.compile(
    r"(?<![\d.])(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?"
    r"(?:\s*(?:[+-]\d{2}:?\d{2}|Z|[A-Z]{2,4}))?)(?!\d)"
)

# ISO-8601 datetime ("2021-10-23T09:53:00Z", "2021-01-06T08:34:00+02:00") —
# the dominant TIMESTAMP shape in log/report corpora. Seconds, fractional
# seconds and timezone designators are optional.
_DATETIME_ISO8601 = re.compile(
    r"(?<![\d.])(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    # Trailing guard rejects only a continuing DIGIT (a longer number), NOT a
    # sentence-ending period — `(?![\d.])` made the engine backtrack and shave
    # the timezone designator off "…00Z." (Z then '.') to satisfy the lookahead.
    r"(?:Z|[+-]\d{2}:?\d{2})?)(?!\d)"
)

# MAC address: 6 colon-or-dash-separated hex pairs.
_MAC_ADDRESS = re.compile(
    r"\b([0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-]"
    r"[0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2}[:\-][0-9A-Fa-f]{2})\b"
)

# Driver's license with driver-specific context keyword.
# Covers: "driver's license", "drivers license", "permis de conduire",
# "licencia de conducir", "führerschein" — all unambiguously driver-document
# keywords.  Includes the hyphenated alpha form ([A-Z]{1,3}-\d{5,9}) because
# these keywords are not shared with software/SPDX license identifiers.
_DRIVERS_LICENSE_CTX = re.compile(
    r"\b(?:driver'?s?\s*licen[cs]e(?:\s*(?:number|no|#))?|permis\s*(?:de\s+)?conduire"
    r"|licencia\s*(?:de\s+)?conducir|f[üu]hrerschein)"
    r"\s*[:\-#]?\s*"
    r"(DL-[A-Z]\d{4,6}-\d{2,4}|[A-Z]{1,3}-\d{5,9}|[A-Z]\d{4,15}|\d{1,3}-\d{2,4}-\d{4,6})\b",
    re.IGNORECASE,
)
# Bare "license number/no/#" keyword (no 'driver' prefix) — ambiguous with
# software/SPDX license identifiers (e.g. "license # MIT-12345").  This path
# deliberately EXCLUDES the hyphenated alpha value form ([A-Z]{1,3}-\d{5,9})
# to prevent SPDX-shaped strings (MIT-12345, GPL-30000) from being tagged as
# DRIVERS_LICENSE.
#
# Double-emission guard: a negative lookbehind prevents this pattern from
# firing inside the driver-specific keyword phrases that _DRIVERS_LICENSE_CTX
# already covers (e.g. "driver's license number: D1234567").  The three
# variants cover driver'?s? (with re.IGNORECASE also applying to lookbehinds):
#   "driver's " (9 chars), "drivers " (8 chars), "driver " (7 chars).
# All are fixed-width, satisfying Python's lookbehind constraint.
_DRIVERS_LICENSE_BARE_KW = re.compile(
    r"(?<!driver's )(?<!drivers )(?<!driver )\blicense\s*(?:number|no|#)"
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
# The optional parenthetical group (up to 24 chars, e.g. "(rental)") handles
# corpus forms like "License Plate (rental): JBL-4117".  The 24-char cap
# prevents bridging across unrelated asides.
# Known FP surface: bare "tag" + parenthetical (e.g. "tag (v2): AB-123") can
# over-trigger; accepted — LICENSE_PLATE is in HIGH_FP_TYPES (context penalty).
_LICENSE_PLATE = re.compile(
    r"\b(?:plate|license\s*plate|tag|vehicle\s*(?:registration|reg)|registration\s*number)"
    r"\s*(?:\([^)\n]{1,24}\)\s*)?(?:number|no|#)?\s*[:\-#]?\s*"
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
# (Those figures are census-lens numbers; production precision of the fragment
# template is higher — census 'card ending NNNN' mentions are often unannotated.)
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
# SP1 Task 8 (2026-06-11): consume the optional "Number|No|#" qualifier so it
# is never captured as the value span (FP class); expand the value sub-pattern
# to allow internal hyphens and require ≥1 digit (FN class A: "37-3808704";
# FN class C: "5T-243002S"; lookahead (?=[A-Za-z0-9\-]*\d) enforces the digit
# requirement without anchoring the match to a specific prefix).
_NATIONAL_ID = re.compile(
    r"\b(?:national\s*id|national\s*identification|citizen\s*id|ID\s*number"
    r"|(?:international\s+)?tax\s+id)"
    r"(?:\s+(?:number|no|#))?"
    r"\s*[:\-#]?\s*"
    r"((?=[A-Za-z0-9\-]*\d)[A-Za-z0-9][A-Za-z0-9\-]{3,23})",
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
# Org token atom: Title-Case/ALL-CAPS word, '&' and apostrophe allowed —
# NO '.' inside the atom (a "Company." atom let matches swallow the sentence
# period and continue into the next sentence) and separators are [ \t]+
# (NOT \s+ — an org name never crosses a line break; sp2 dev-iteration 2).
_ORG_ATOM = r"[A-Z][A-Za-z&']+"

_ORGANIZATION = re.compile(
    r"\b(" + _ORG_ATOM + r"(?:[ \t]+" + _ORG_ATOM + r")*)[ \t]+"
    r"(?:Inc|Corp|Corporation|LLC|Ltd|Limited|GmbH|AG|PLC|Co|Company|Group|Foundation|Association)"
    r"\.?\b"
)

# Organization name with industry suffix (no legal suffix required).
# Catches "Weyland Industries", "Cyberdyne Systems", "Oscorp Technologies", etc.
_ORGANIZATION_INDUSTRY = re.compile(
    r"\b(" + _ORG_ATOM + r"(?:[ \t]+" + _ORG_ATOM + r")*[ \t]+"
    r"(?:Industries|Systems|Technologies|Labs|Laboratories|Enterprises|Solutions|"
    r"Dynamic|Dynamics|Communications|Electronics|Pharmaceuticals|Consulting|Partners|"
    r"Robotics|Aerospace|Digital|Analytics|Software|Networks|Services|Media|"
    r"Capital|Holdings|Ventures|International|Global|Medical|Health|Bio|Biotech|"
    r"Energy|Power|Financial|Insurance|Logistics|Transport|Motors|Aviation|"
    r"Construction|Engineering|Security|Defense|Research|"
    r"Administration|Agency|Bureau|Authority|Commission|Institute|"
    r"Center|Centre|University|Hospital|Clinic|Bank))"
    r"\b"
)

# Organization preceded by multilingual context keywords.
# Covers "Company:", "Unternehmen:", "Empresa:", "Entreprise:", "Azienda:", etc.
# Case-insensitivity is SCOPED to the keyword block — the capture stays
# case-sensitive (full-pattern IGNORECASE let the capture accept arbitrary
# case). Noun keywords REQUIRE a label colon/dash: bare "Employer Tax ID:"
# captured the next field's label. Verb forms keep plain whitespace. The
# capture's tail tokens carry the next-field-label guard ``(?![ \t]*[:=])``.
_ORGANIZATION_CONTEXT = re.compile(
    r"\b(?:(?i:Company|Organisation|Organization|Employer|Unternehmen|Empresa|Entreprise|"
    r"Azienda|Bedrijf|Företag|Virksomhed|Firma|Yritys|Organisasjon|Organizacja)\s*[:\-]"
    r"|(?i:employed\s+(?:at|by)|works?\s+(?:at|for)|working\s+(?:at|for)|"
    r"position\s+at|role\s+at|affiliated\s+with|belongs?\s+to))"
    r"[ \t]*"
    # sp7 panel (multilingual lens): a bare honorific is never an organization
    # — "belongs to Mr. Wu" captured ORGANIZATION 'Mr'. Veto it at the head.
    r"(?!(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sir|Lord|Lady|Mme|Mlle|Sr|Sra|Srta|Dra|Hon)\.?(?![A-Za-z]))"
    r"(" + _ORG_ATOM + r"(?:[ \t]+" + _ORG_ATOM + r"(?![ \t]*[:=])){0,4})"
    r"(?![A-Za-z])"
)

# sp7 #8 — institution / firm ORGANIZATION grammar (mining candidate #8).
# TAB/ECHR court prose is dense with institution names the grammar above
# misses, and the person heuristic simultaneously eats them. ADDITIVE
# (ORGANIZATION is supported + person-shadowing, so the eval-only person drop
# cleans the FP face for free). Diacritic-aware ATOM so İzmir/Kraków/Będzin
# capture in full (Latin-1 + Latin Extended-A).
_ORG_ATOM_UC = r"[A-ZÀ-ÝĀ-Ž][A-Za-zÀ-ÿĀ-ſ'.\-]+"
_ORG_CONN = r"(?:of|and|for|the|de|del|van|von|und|di)"
# A capitalized function word that a real institution/firm name never STARTS
# with (sentence-initial "The"/"Before"/"After"/…). Vetoing it at the run head
# keeps the leading determiner/preposition out of the captured span.
_ORG_LEAD_STOP = (
    r"(?!(?:The|A|An|This|That|These|Those|Before|After|At|In|On|By|For|From|"
    r"With|To|Of|And|Or|But|If|As|Its|Their|Our|Your|His|Her|Which|When|Where|"
    r"While|Also|However|Therefore|Thus|Since|Until|Upon|Under|Over|Between|"
    r"During|Although|Because|Both|Each|Every|Such|Said)\b)"
)
# A trailing run unit that always ends on an ATOM (an optional internal
# connective + a Title-Case token) — prevents a trailing connective from being
# absorbed ("Ministry of Justice for" -> "Ministry of Justice").
_ORG_TRAIL = r"(?:[ \t]+(?:" + _ORG_CONN + r"[ \t]+)?" + _ORG_ATOM_UC + r")"
# Tail-keyword form: Title-Case run (1-5 tokens, connectives allowed) + an
# institution keyword as the TAIL token. Keyword set is the GAP not already
# covered by _ORGANIZATION_INDUSTRY (which has University/Hospital/Commission/
# Authority/Agency/Bureau/Institute). NOTE: "Court" is deliberately EXCLUDED
# here — it doubles as a residential street suffix ("Birch Court") — and is
# handled by the descriptor-gated _ORGANIZATION_COURT below.
_ORGANIZATION_INSTITUTION = re.compile(
    r"\b(" + _ORG_LEAD_STOP + _ORG_ATOM_UC + _ORG_TRAIL + r"{0,4}[ \t]+"
    r"(?:Ministry|Directorate|Tribunal|Prosecutor|Parliament|Government|"
    r"Council|Board|Committee|Chamber|Prison|Assembly|Constabulary|"
    r"Municipality|Secretariat|Penitentiary|Inspectorate|Ombudsman|Presidency))\b"
)
# Court form: a court name is institutional ONLY when a court-type DESCRIPTOR
# sits immediately before "Court" — this fires on "Sinop Assize Court" /
# "Supreme Administrative Court" but NOT on the residential address "Birch
# Court" (which was 100% of the home ORG false positives). The head form
# "Court of X" is handled by _ORGANIZATION_INSTITUTION_OF.
_COURT_DESC = (
    r"(?:Assize|Regional|District|Supreme|Appeal|Appeals|High|Crown|Magistrates?|"
    r"Security|Administrative|Constitutional|Circuit|County|Federal|Superior|"
    r"Juvenile|Family|Criminal|Civil|Commercial|Labour|Labor|Cassation|Justice|"
    r"Martial|Provincial|Municipal|Metropolitan|Central|National|International|"
    r"Special|Military|Revolutionary|State|Peace|Sharia|Ecclesiastical|Arbitration)"
)
_ORGANIZATION_COURT = re.compile(
    r"\b(" + _ORG_LEAD_STOP + r"(?:" + _ORG_ATOM_UC + r"[ \t]+){0,3}"
    + _COURT_DESC + r"[ \t]+Court)\b"
)
# Head-of form: keyword + literal "of" + Title-Case run. Keyword set RESTRICTED
# (Department/Office/Bureau excluded — "Department of Cardiology" is home
# clinical prose, not a PII org).
_ORGANIZATION_INSTITUTION_OF = re.compile(
    r"\b((?:Court|Ministry|Directorate|Tribunal|Council|Board)[ \t]+of[ \t]+"
    + _ORG_ATOM_UC + _ORG_TRAIL + r"{0,3})\b"
)
# Firm form: 1-3 Title-Case tokens + (&|and) + firm-suffix token.
_ORGANIZATION_FIRM = re.compile(
    r"\b(" + _ORG_LEAD_STOP + _ORG_ATOM_UC + r"(?:[ \t]+" + _ORG_ATOM_UC + r"){0,2}[ \t]*(?:&|and)[ \t]*"
    r"(?:Sons|Associates|Partners|Brothers|Bros|Co)\b(?:[ \t]+" + _ORG_ATOM_UC + r"){0,2})"
)

# CONTEXT-ANCHORED CamelCase organization ("from InnovateLabs", "at OpenAI"):
# a bare internal-capital token is NOT a reliable org signal — it fires on
# tech terms (WiFi, JavaScript, PowerPoint, GitHub) and on CamelCase surnames
# (DeAndre, LaToya, DiCaprio). Requiring a preceding affiliation word keeps
# the corpus form ("Daniel Moore from InnovateLabs") while dropping those FPs.
# Mc/Mac/O' surname shapes stay excluded.
_ORGANIZATION_CAMELCASE = re.compile(
    r"\b(?i:at|from|with|for|by|joined|employer|vendor|client)[ \t]+"
    r"(?!(?:Mc|Mac|O')[A-Z])([A-Z][a-z]+[A-Z][A-Za-z]+)\b"
)

# Street address: number + words + suffix, optionally followed by
# ", City, ST ZIP" so the captured span matches full mailing addresses.
# sp7 A4 — two-tier evidence-gated address grammar (mining candidate #10).
# Global re.IGNORECASE is DROPPED (it made the street-token class slurp
# lowercase prose "1997 the ... Court"); case-insensitivity is re-applied ONLY
# to the suffix via (?i:...). A first-token function-word guard rejects
# prose-led matches. Interior tokens are [A-Za-z]-led so lowercase particles
# (de la / van der) and ALL-CAPS street names ("JULIE SQUARES") are preserved.
_ADDR_FUNC = r"the|of|a|an|and|by|at|to|in|on|for|with|was|were|is|are|from"
# Tier-1: unambiguous topographic/road suffixes (full USPS Pub-28 C1 set) —
# matches freely (ADDITIVE recall; the largest gretel FN class).
_USPS_UNAMBIG = (
    r"Alley|Arcade|Avenue|Ave|Boulevard|Blvd|Bypass|Causeway|Circle|Cir|Court|Ct|"
    r"Courts|Crescent|Crossing|Xing|Crossroad|Drive|Dr|Drives|Expressway|Freeway|"
    r"Highway|Hwy|Lane|Ln|Motorway|Overpass|Parkway|Pkwy|Parkways|Place|Pl|Plaza|"
    r"Roads|Road|Rd|Route|Skyway|Squares|Square|Sq|Station|Streets|Street|St|"
    r"Terrace|Ter|Throughway|Trafficway|Trail|Trl|Turnpike|Underpass|Viaduct|"
    r"Harbors|Harbor|Manors|Manor|Estates|Estate|Villages|Village|Vistas|Vista|"
    r"Heights|Mountains|Mountain|Ways|Way"
)
# Tier-2: suffixes that are ALSO common English nouns — accepted ONLY with a
# following unit or postcode token (the evidence gate that resolves the
# gretel-loosen vs prose-tighten tension).
_USPS_AMBIG = (
    r"Bend|Bluffs|Bluff|Branch|Bridge|Brooks|Brook|Camp|Canyon|Cape|Centers|"
    r"Center|Cliffs|Cliff|Club|Commons|Common|Corners|Corner|Course|Coves|Cove|"
    r"Creek|Crest|Curve|Dale|Dam|Divide|Falls|Fall|Ferry|Fields|Field|Flats|Flat|"
    r"Fords|Ford|Forest|Forge|Forks|Fork|Fort|Gardens|Garden|Gateway|Glens|Glen|"
    r"Greens|Green|Groves|Grove|Haven|Hills|Hill|Hollow|Inlet|Islands|Island|Isle|"
    r"Junction|Keys|Key|Knolls|Knoll|Lakes|Lake|Landing|Lights|Light|Loaf|Locks|"
    r"Lock|Lodge|Loop|Mall|Meadows|Meadow|Mews|Mills|Mill|Mission|Mount|Neck|"
    r"Orchard|Oval|Parks|Park|Passage|Pass|Path|Pike|Pines|Pine|Plains|Plain|"
    r"Points|Point|Ports|Port|Prairie|Radial|Ramp|Ranch|Rapids|Rapid|Rest|Ridges|"
    r"Ridge|River|Row|Rue|Run|Shoals|Shoal|Shores|Shore|Springs|Spring|Spur|Spurs|"
    r"Stream|Summit|Trace|Track|Trailer|Tunnel|Unions|Union|Valleys|Valley|Views|"
    r"View|Walks|Walk|Wall|Wells|Well"
)
# Evidence tails for the tier-2 lookahead (unit or postcode adjacent to the
# suffix). Separators are [ \t] — an address never crosses a line break, and
# allowing \n let "9902\nLicense pl" match "…License pl(ace)" as an address.
_ADDR_UNIT = (
    r"(?:,?[ \t]+(?:Apt|Apartment|Suite|Ste|Unit|Floor|Fl|Rm|Room|Building|Bldg|Box|#)"
    r"\.?[ \t]*#?[ \t]*\w+)"
)
_ADDR_ZIP = r"(?:,?[ \t]+\d{4,5}(?:[ \t]*[A-Z]{2})?\b)"
# Tier-1 captured tail: all-or-nothing ", City …, ST 12345" (matches the home
# gold span boundary exactly — independent unit/zip captures over-extended the
# span and broke strict-span scoring).
_ADDR_TAIL = (
    r"(?:,?[ \t]+[A-Za-z][A-Za-z'.\-]+(?:[ \t]+[A-Za-z][A-Za-z'.\-]+)*,?"
    r"[ \t]+[A-Z]{2}[ \t]+\d{5}(?:-\d{4})?)?"
)
_ADDR_HEAD = (
    r"\b(\d{1,6}[ \t]+(?!(?i:" + _ADDR_FUNC + r")\b)"
    r"(?:[A-Za-z][A-Za-z'.\-]*[ \t]+){1,4}"
)
_ADDRESS = re.compile(
    _ADDR_HEAD + r"(?i:" + _USPS_UNAMBIG + r")\.?" + _ADDR_TAIL + r")"
)
# Tier-2 requires evidence (unit or postcode) via lookahead right after the
# ambiguous suffix — the evidence is NOT captured, so it never extends the span.
_ADDRESS_AMBIGUOUS = re.compile(
    _ADDR_HEAD + r"(?i:" + _USPS_AMBIG + r")\.?"
    + r"(?=" + _ADDR_UNIT + r"|" + _ADDR_ZIP + r"))"
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

# GPS coordinates: decimal lat/lon pair. The PATTERN deliberately keeps the
# permissive pre-sp6 form (integer halves admitted): narrowing it here was a
# PRODUCTION LEAK the sp6 close caught — regex-oss is the AX-003 floor
# source, so a pair like "41, -87" that the narrowed pattern dropped reached
# production UNMASKED with no downstream layer able to restore it (the sp2
# showstopper class: an eval precision optimization executing as a drop on
# the masking path). The eval-side precision fix (the "15/09" date-fragment
# FP class; Nemotron P=0.072, home P=0.157) lives in
# regex_adapter._drop_undecimaled_gps under eval_cross_type_arbitration —
# eval-only, never on the masking path.
_GPS = re.compile(
    r"(?<![0-9.])"
    r"(-?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?))"
    r"\s*[,/]\s*"
    r"(-?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d{1,2}(?:\.\d+)?))"
    r"(?![0-9.])"
)

# sp7 #7 — hemisphere-suffixed coordinates ("40.7234 N, 123.1235 W"). ADDITIVE:
# the N/S + E/W markers are themselves the disambiguator, so this pairs with the
# scoring-only non-geo GPS drop to keep GPS coverage non-shrinking. group=0 so
# the span includes the hemisphere markers (exact-match the gold).
_GPS_HEMISPHERE = re.compile(
    r"(?<![0-9A-Za-z.])(-?\d{1,3}(?:\.\d+)?)\s*[°]?\s*[NSns]\s*[,/ ]\s*"
    r"(-?\d{1,3}(?:\.\d+)?)\s*[°]?\s*[EWew](?![A-Za-z0-9])"
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
#   "Bar Number: BAR-866155"   (dominant corpus form; BAR-prefix + digits)
#   "Bar No. FL-530363"        (deposition form; 2-letter state code + digits)
# Value alternation (most-specific first):
#   BAR-\d{4,7}         explicit BAR-prefix (unambiguous)
#   [A-Z]{1,3}-\d{4,7}  state-code prefix (FL-, CA-, NY-, etc.); context-gated
#   \d{4,8}             classic plain-digit form
# The [A-Z]{1,3}-\d{4,7} shape is kept BEHIND specific bar-context keywords
# (bar number/no./id, state bar, sbn) so generic XX-NNNNN refs cannot fire.
_BAR_NUMBER = re.compile(
    r"\b(?:state\s+bar|sbn|bar\s*(?:id|no\.?|number|#))"
    r"\s*[:\-#]?\s*"
    r"(BAR-\d{4,7}|[A-Z]{1,3}-\d{4,7}|\d{4,8})\b",
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
# sp2 external-coverage tranche
# ═══════════════════════════════════════════════════════════════════════════
# Patterns grounded in sampled gold shapes from the pii-anon-eval-data dev
# split (2026-06-12). Value classes tolerate zero-width characters
# (U+200B/200C/200D): the corpus's adversarial records embed them INSIDE
# values ("7​0-632129​2") and gold spans cover the full obfuscated string.
# Most of these labels are census-ignored by the pinned internal authority
# (documented in tests/test_pattern_label_alignment.py); they earn external
# credit through the DATA harness's native->canonical-63 label map.

_ZW = "\u200b\u200c\u200d"  # ZWSP / ZWNJ / ZWJ as escapes (NOT raw invisibles)

_TAX_ID_LABELED = re.compile(
    r"(?i:\b(?:tax[ \t]*id|ein|tin)\b)\s*[:\-]?\s*"
    r"([A-Z0-9" + _ZW + r"]{2,4}-[A-Z0-9" + _ZW + r"]{5,10})(?![A-Za-z0-9])"
)

_JOB_TITLE_LABELED = re.compile(
    r"(?i:\b(?:job|position|title|role|occupation)\b)\s*[:\-]\s*"
    r"([A-Z][A-Za-z]+(?:[ \t][A-Z&][A-Za-z]*){0,3})(?![A-Za-z])"
)
_JOB_TITLE_LEXICON = re.compile(
    r"\b((?:Senior[ \t]|Junior[ \t]|Lead[ \t]|Chief[ \t]|Principal[ \t])?"
    r"(?:Legal[ \t]Counsel|Medical[ \t]Director|Systems?[ \t]Administrator|"
    r"Security[ \t]Analyst|Financial[ \t]Analyst|Data[ \t](?:Scientist|Analyst|Engineer)|"
    r"Software[ \t](?:Engineer|Developer)|Project[ \t]Manager|Product[ \t]Manager|"
    r"Operations[ \t]Manager|Quality[ \t]Assurance[ \t]Lead|Registered[ \t]Nurse|"
    r"Executive[ \t]Officer|Court[ \t]Reporter))\b"
)

# Health condition, TWO disambiguated forms (the bare prose lead-ins below
# precede a PERSON just as often as a condition — "Evaluation of Mark Thompson"
# — so capturing freely there mislabels names; sp2 remediation):
#  (1) the "Diagnosis:" LABEL is unambiguous → free value.
_HEALTH_CONDITION_DIAGNOSIS = re.compile(
    r"(?i:diagnosis)\s*[:\-]\s*"
    r"([A-Z][A-Za-z0-9]*(?:[ \t][A-Z0-9][A-Za-z0-9]*){0,4})(?![A-Za-z])"
)
#  (2) a prose lead-in REQUIRES the value to carry a medical marker (a
#      condition word or a clinical suffix) — "Mark Thompson" carries none.
_CONDITION_MARKER = (
    r"Disease|Syndrome|Disorder|Diabetes|Cancer|Infection|Failure|Hypertension|"
    r"Asthma|Arthritis|Bronchitis|Pneumonia|Mellitus|Lupus|Erythematosus|"
    r"Sclerosis|An[ae]mia|Migraine|Depression|Anxiety|Insufficiency|"
    r"itis|osis|emia|opathy|algia"
)
_HEALTH_CONDITION_LEADIN = re.compile(
    r"(?i:diagnosed[ \t]with|presents[ \t]with|consistent[ \t]with|"
    r"evaluation[ \t]of|consultation[ \t]regarding|history[ \t]of|"
    r"treatment[ \t]for|suffers[ \t]from)\s*[:\-]?\s*"
    r"(?=[A-Za-z0-9 \t]{0,45}(?:" + _CONDITION_MARKER + r"))"
    r"([A-Z][A-Za-z0-9]*(?:[ \t][A-Z0-9][A-Za-z0-9]*){0,4})(?![A-Za-z])"
)

# Drug name + dose is distinctive without a label ("Gabapentin 300mg"). The
# labeled list form REQUIRES the value to end in a dose or a medication-form
# word — a bare "still taking Robert Williams" carries neither (sp2 fix).
_MEDICATION_DOSE = re.compile(r"\b([A-Z][a-z]{3,}[ \t]\d{1,4}[ \t]?mg)\b")
_MEDICATION_LABELED = re.compile(
    r"(?i:medications?[ \t]+include|current[ \t]medications?[ \t]*[:\-]|"
    r"still[ \t]taking|prescribed)\s*"
    r"([A-Z][a-z]{3,}(?:"
    r"[ \t](?:Inhaler|Cream|Injection|Tablets?|Capsules?|Spray|Patch|Solution|Drops)"
    r"|[ \t]\d{1,4}[ \t]?mg))(?![A-Za-z])"
)

_HEALTH_INSURANCE_INS = re.compile(r"\b(INS-[0-9" + _ZW + r"]{6,12})(?![A-Za-z0-9])")
_HEALTH_INSURANCE_LABELED = re.compile(
    r"(?i:insurance(?:[ \t]id)?)\s*(?:is[ \t]+|[:\-][ \t]*)"
    r"([A-Z0-9+/=" + _ZW + r"-]{6,28})(?![A-Za-z0-9])"
)

_CC_FRAGMENT_ENDING = re.compile(r"(?i:card[ \t]+ending(?:[ \t]+in)?[ \t]+)(\d{4})\b")
_CC_FRAGMENT_MASKED = re.compile(
    r"([*" + _ZW + r"]{3,6}-[*" + _ZW + r"]{3,6}-[*" + _ZW + r"]{3,6}"
    r"-[\d*" + _ZW + r"]{3,8})"
)
_CC_FRAGMENT_LABELED = re.compile(
    r"(?i:credit[ \t]card[ \t]fragment)\s*[:\-]\s*"
    r"([A-Za-z0-9+/=*" + _ZW + r"-]{8,40})(?![A-Za-z0-9=])"
)

_VISA_NUMBER_LABELED = re.compile(
    r"(?i:visa[ \t]number)\s*[:\-]\s*([A-Za-z0-9+/=" + _ZW + r"]{7,20})(?![A-Za-z0-9=])"
)

_PRESCRIPTION_RX = re.compile(r"\b(RX-[0-9" + _ZW + r"]{6,10})(?![A-Za-z0-9])")
_PRESCRIPTION_LABELED = re.compile(
    r"(?i:prescription[ \t]number)\s*[:\-]\s*"
    r"([A-Za-z0-9+/=" + _ZW + r"]{6,24})(?![A-Za-z0-9=])"
)

_DEVICE_ID_LABELED = re.compile(
    r"(?i:device[ \t](?:identifier|id)|imei|udid)\s*[:\-]\s*"
    r"([A-Za-z0-9" + _ZW + r"-]{10,40})(?![A-Za-z0-9])"
)
# Uppercase dashed UUID (corpus device IDs); lowercase hex session IDs must
# NOT match, so the class is deliberately case-sensitive.
_DEVICE_ID_UUID = re.compile(
    r"\b([0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})\b"
)

_SOCIAL_MEDIA_HANDLE = re.compile(r"(?<![\w@.])(@[A-Za-z][A-Za-z0-9_]{2,30})\b")

# Unambiguous credentials (PhD/MBA/...) match bare; the common nouns
# Master/Bachelor/Associate REQUIRE an "'s" or "Degree" qualifier, else they
# false-fire on "Master Service Agreement", "Senior Associate", "The Bachelor"
# (sp2 fix).
_EDUCATION_LEVEL = re.compile(
    r"\b((?:PhD|Ph\.D\.?|MBA|MSc|BSc|Doctorate|High[ \t]School[ \t]Diploma|"
    r"(?:Master|Bachelor|Associate)(?:['’]s(?:[ \t]Degree)?|[ \t]Degree)"
    r"(?:[ \t]in[ \t][A-Z][a-z]+(?:[ \t][A-Z][a-z]+){0,3})?))\b"
)

# Categorical demographics: own field label THEN a closed value lexicon (bare
# lexicon words in prose are FP bombs — a field label is required). The
# generator's "Record shows <value>" filler is DELIBERATELY NOT an anchor: it
# is the eval-data template's universal filler (it embeds values of EVERY
# type), so keying on it would memorise the generator, not recognise the
# entity — a benchmark-gaming pattern the eval-integrity axiom forbids. Types
# whose corpus form is ONLY "Record shows X" (nationality / ethnicity /
# political-opinion) therefore score near-zero recall here; that is the honest
# cost of not gaming — the recogniser still fires on a real "Nationality:" label.
def _categorical(label: str, values: str) -> re.Pattern[str]:
    return re.compile(
        r"(?i:\b(?:" + label + r")\b)\s*[:\-]\s*"
        r"(" + values + r")\b"
    )


_GENDER = _categorical("gender", r"Male|Female|Non-?binary")
_NATIONALITY = _categorical(
    "nationality",
    r"American|British|Canadian|Australian|German|French|Italian|Spanish|"
    r"Mexican|Indian|Chinese|Japanese|Brazilian|Korean",
)
_ETHNICITY = _categorical(
    "ethnicity", r"European|Hispanic|Asian|African|Caucasian|Latino|Latina"
)
_POLITICAL_OPINION = _categorical(
    "political[ \t]opinion|political[ \t]affiliation",
    r"Liberal|Conservative|Progressive|Libertarian|Moderate|Socialist|Centrist",
)
_RELIGIOUS_BELIEF = _categorical(
    "religious[ \t]belief|religion",
    r"Christian|Buddhist|Muslim|Jewish|Hindu|Atheist|Catholic|Protestant|Agnostic|Sikh",
)
_MARITAL_STATUS = _categorical(
    "marital[ \t]status", r"Married|Single|Divorced|Widowed|Separated"
)
_HOUSEHOLD_SIZE = re.compile(r"(?i:\bhousehold[ \t]size\b)\s*[:\-]\s*(\d{1,2})\b")

_VEHICLE_MODEL_LABELED = re.compile(
    r"(?i:\bvehicle(?:[ \t]model)?\b)\s*[:\-]\s*"
    r"([A-Z0-9][A-Za-z0-9]*(?:[ \t][A-Z][A-Za-z0-9-]+){0,3})(?![A-Za-z])"
)
_VEHICLE_MODEL_YEAR_MAKE = re.compile(
    r"\b((?:19|20)\d{2}[ \t]"
    r"(?:Toyota|Honda|Ford|Subaru|Chevrolet|Chevy|BMW|Mercedes|Nissan|Hyundai|"
    r"Kia|Tesla|Volkswagen|Audi|Mazda|Jeep|Dodge|Lexus|Volvo)"
    r"[ \t][A-Z][A-Za-z0-9-]+)\b"
)

_PROCEDURE_LABELED = re.compile(
    r"(?i:\bprocedure\b)\s*[:\-]\s*"
    r"([A-Z][A-Za-z0-9-]*(?:[ \t][A-Z][A-Za-z0-9-]+){0,4})(?![A-Za-z])"
)
_PROCEDURE_MODALITY = re.compile(
    r"\b((?:MRI|CT|PET|EKG|ECG|EEG|Ultrasound)[ \t][A-Z][A-Za-z-]+"
    r"|Chest[ \t]X-?[Rr]ay|Pulmonary[ \t]Function[ \t]Test)\b"
)

_BIOMETRIC_BIO = re.compile(
    r"\b(B[" + _ZW + r"]?I[" + _ZW + r"]?O-[A-F0-9" + _ZW + r"]{8,24})(?![A-Za-z0-9])"
)
_BIOMETRIC_LABELED = re.compile(
    r"(?i:biometric[ \t]id)\s*[:\-]\s*"
    r"([A-Za-z0-9+/=" + _ZW + r"-]{8,32})(?![A-Za-z0-9=])"
)

_COURT_CASE_YEAR_FORM = re.compile(
    r"\b(20\d{2}-(?:CIV|CV|CRIM|CR|FAM|PROB)-\d{3,6})\b"
)
_DOCKET_FEDERAL = re.compile(
    r"\b(\d:\d{2}-(?:mj|cr|cv|md|mc)-\d{4,6}(?:-[A-Z]{2,4})?)\b"
)
_INVOICE_INV = re.compile(r"\b(INV-[A-Z0-9" + _ZW + r"]{4,12})(?![A-Za-z0-9])")
_INVOICE_LABELED = re.compile(
    r"(?i:invoice[ \t]number)\s*[:\-]\s*"
    r"([A-Za-z0-9+/=" + _ZW + r"]{4,24})(?![A-Za-z0-9=])"
)
_SWIFT_LABELED = re.compile(
    r"(?i:swift(?:[ \t]bic)?(?:[ \t]code)?)\s*[:\-]\s*"
    r"([A-Z0-9" + _ZW + r"]{8,11})(?![A-Za-z0-9])"
)
_DL_PREFIXED = re.compile(
    r"\b(DL-[A-Z0-9" + _ZW + r"]{4,10}(?:-\d{1,4})?)(?![A-Za-z0-9])"
)
_DL_LABELED = re.compile(
    r"(?i:driver'?s?[ \t]licen[cs]e(?:[ \t]number)?)\s*[:\-]\s*"
    r"([A-Za-z0-9+/=" + _ZW + r"-]{5,20})(?![A-Za-z0-9=])"
)

# Dollar-amount extents INCLUDE the "$" (gold convention); corpus labels are
# Annual income / Wages / Amount / Total, which the legacy _SALARY keyword
# set missed entirely (0 recall on the DATA dev split).
_SALARY_LABELED = re.compile(
    r"(?i:annual[ \t]income|salary|wages|total[ \t]compensation|amount|total)"
    r"\s*[:\-]\s*"
    r"(\$[\d," + _ZW + r"]{3,12}(?:\.\d{2})?)(?!\d)"
)

_API_KEY_SK = re.compile(r"\b(sk-[A-Za-z0-9]{16,48})\b")
_API_KEY_LABELED = re.compile(
    r"(?i:api[_ \t-]?key(?:[ \t]for[ \t][a-z]+)?)\s*[:=]\s*\"?"
    r"([A-Za-z0-9+/=_-]{16,64})\"?(?![A-Za-z0-9])"
)


# ═══════════════════════════════════════════════════════════════════════════
# sp3 v2.2.0 re-baseline tranche
# ═══════════════════════════════════════════════════════════════════════════
# The v2.2.0 corpus obfuscates several secret-like values as base64, short
# alphanumerics, or zero-width-embedded strings BEHIND their specific field
# label ("CVV: MzIx", "PIN: ODQzNw==", "Policy: P0L-2694750"). The legacy
# digit-only value classes could not reach them, so recall on these
# census-external types collapsed on the rebuilt substrate. Each pattern here
# gates on the SAME specific label (leak-safe, additive — never the universal
# "Record shows X" generator filler) and widens the value class to admit
# base64 / _ZW / OCR-style P0L. Values carry no interior space/comma, so a
# maximal value-char run captures the exact obfuscated extent.

# CVV / PIN encoded values: a base64 or short-alnum secret after the field
# label + an explicit ':'/'=' (stricter than the legacy verbal separator, so
# prose like "cvv is fine" cannot fire).
_CVV_ENCODED = re.compile(
    r"(?i:cvv|cvv2|cvc|cvc2|cid|security[ \t]code|card[ \t]verification(?:[ \t]value)?)"
    r"\s*[:=]\s*([A-Za-z0-9+/=" + _ZW + r"]{3,8})(?![A-Za-z0-9+/=])"
)
_PIN_ENCODED = re.compile(
    r"(?i:pin(?:[ \t](?:number|code))?|passcode|atm[ \t]pin|pin[ \t]?#)"
    r"\s*[:=]\s*([A-Za-z0-9+/=" + _ZW + r"]{3,16})(?![A-Za-z0-9+/=])"
)

# PASSWORD in code/config/JSON form: pass := "value" or "password": "value".
# The value is DELIMITED by the surrounding quotes, so special chars
# ($ # ! @ %) that the bare-token _PASSWORD form stops on are captured intact.
_PASSWORD_QUOTED = re.compile(
    r"\b(?i:password|passwd|pwd|pass)\b\"?\s*(?::=|[:=])\s*"
    r"\"([^\"\n" + _ZW + r"]{6,64}|[^\"\n]{6,64})\""
)

# INSURANCE_POLICY_NUMBER obfuscated forms: OCR P0L (zero-for-O), zero-width
# embedded, base64, or an alphanumeric suffix (POL-48BS84B). The value
# lookahead REQUIRES a digit so a prose "Policy: standard terms" cannot fire.
_INSURANCE_POLICY_ENCODED = re.compile(
    r"(?i:policy|policyholder|insurance[ \t]policy(?:[ \t]number)?)"
    r"\s*(?:no\.?|number|#)?\s*[:\-#]?\s*"
    r"(?=[A-Za-z0-9+/=" + _ZW + r"-]*\d)"
    r"([A-Za-z0-9+/=" + _ZW + r"][A-Za-z0-9+/=" + _ZW + r"-]{3,40})(?![A-Za-z0-9+/=])"
)

# AUTHENTICATION_TOKEN: bearer tokens, base64 "Bearer …" (QmVhcmVy = b64
# "Bearer"), truncated-JWT placeholders (eyJ…), and any value behind the
# "Authentication Token:" label. Real 3-segment JWTs stay JWT_TOKEN (_JWT);
# the eyJ form here requires the literal "…" truncation, so the two never
# double-emit on one span. All fold to AUTHENTICATION_TOKEN externally.
#
# The corpus obfuscates the "Bearer" keyword adversarially: B->8 OCR ("8earer")
# and zero-width chars embedded between letters ("Bea<ZWSP>rer") / after it
# ("Bearer<ZWSP> "). _BEARER_KW tolerates both while keeping the keyword in the
# captured gold span. The eyJ placeholder captures EXACTLY three trailing dots
# ('.' is OUT of the value class, so the ellipsis matches and stops; a trailing
# sentence period stays outside the span).
_BEARER_KW = (
    r"[B8][" + _ZW + r"]?e[" + _ZW + r"]?a[" + _ZW + r"]?r[" + _ZW + r"]?e[" + _ZW + r"]?r"
)
_AUTH_TOKEN_LABELED = re.compile(
    r"(?i:authentication[ \t]token|auth[ \t]token|bearer[ \t]token)"
    r"\s*[:\-]\s*"
    r"(" + _BEARER_KW + r"[ \t" + _ZW + r"]+[A-Za-z0-9+/=" + _ZW + r"]{8,}"
    r"|eyJ[A-Za-z0-9_" + _ZW + r"-]{2,}\.\.\."
    r"|[A-Za-z0-9+/=" + _ZW + r"]{12,})(?![A-Za-z0-9+/=])"
)
_AUTH_TOKEN_BEARER = re.compile(
    r"\b(" + _BEARER_KW + r"[ \t" + _ZW + r"]+[A-Za-z0-9+/=" + _ZW + r"]{16,})"
    r"(?![A-Za-z0-9+/=])"
)
_AUTH_TOKEN_B64_BEARER = re.compile(
    r"\b(QmVhcmVy[A-Za-z0-9+/" + _ZW + r"]{4,}={0,2})(?![A-Za-z0-9+/=])"
)
_AUTH_TOKEN_JWT_TRUNC = re.compile(
    r"\b(eyJ[A-Za-z0-9_" + _ZW + r"-]{2,}\.\.\.)"
)

# ── GDPR Article-9 special categories (taxonomy 63 -> 66) ──────────────────
# SEXUAL_ORIENTATION / TRADE_UNION_MEMBERSHIP / GENETIC_DATA. Detection keys on
# a SPECIFIC field label (the blessed _categorical shape) or, for genetic
# data, on intrinsic value structure (gene symbols / dbSNP rs-IDs) — NEVER the
# universal "Record shows X" generator filler (the eval-integrity line at
# ~line 1005). These earn EXTERNAL credit via the DATA LABEL_MAP; internally
# they are census-unreachable (documented in test_pattern_label_alignment.py).

# Sexual orientation: label-gated closed lexicon (a bare orientation word in
# prose is an FP bomb, so the specific "Sexual Orientation:" label is required
# — exactly the _categorical discipline).
_SEXUAL_ORIENTATION = re.compile(
    r"(?i:sexual[ \t]orientation)\s*[:\-]\s*"
    r"(gay|lesbian|bisexual|pansexual|asexual|queer|heterosexual|homosexual|"
    r"straight|questioning|demisexual|omnisexual|polysexual)\b"
)

# Trade-union membership: label + proper-noun value capture (union names are
# open-vocabulary — "IG Metall", "Teamsters Local 25", "NUT member"). The
# corpus form is comma/field terminated; the value class excludes '.'/',' so
# it stops cleanly at the field boundary. Label-gated (FP-safe: a bare "CGT"
# in prose never fires).
_TRADE_UNION = re.compile(
    r"(?i:trade[ \t]union(?:[ \t]membership)?|union[ \t]membership)"
    r"\s*[:\-]\s*"
    r"([A-Za-z0-9](?:[A-Za-z0-9&' -]*[A-Za-z0-9])?)"
    r"(?=\s*(?:[,.|/\n]|contact\b|$))"
)

# Genetic data — TWO honest routes.
#  (1) label + value capture (comma/field terminated; catches STR-profile /
#      karyotype forms without a canonical gene symbol).
_GENETIC_LABELED = re.compile(
    r"(?i:genetic[ \t]data|genetic[ \t]marker)\s*[:\-]\s*"
    r"([^\n,]*?[A-Za-z0-9)])(?=[,\n]|\s+contact\b)"
)
#  (2) intrinsic structural: a canonical gene symbol or dbSNP rs-ID, extended
#      through one-or-more trailing genetic qualifiers (so the FULL gold
#      extent "BRCA1 c.68_69delAG pathogenic variant" is captured, not just to
#      the first qualifier). Self-identifying PII, low-FP, generalizes beyond
#      the template.
_GENETIC_QUAL = (
    r"homozygous|heterozygous|carrier|positive|negative|genotype|variant"
    r"|expansion|pathogenic|benign|mutation|allele|repeat|deletion|duplication"
    r"|profile|haplotype"
)
_GENETIC_INTRINSIC = re.compile(
    r"\b((?:BRCA[12]|CFTR|APOE|HLA-[A-Z0-9*:]+|MTHFR|HTT|Factor[ \t]V[ \t]Leiden"
    r"|rs\d{3,})"
    r"[ \tA-Za-z0-9Ͱ-Ͽ*:()/._#+-]*?"
    r"[ \t](?:" + _GENETIC_QUAL + r")"
    r"(?:[ \t](?:" + _GENETIC_QUAL + r"))*"
    r"(?:[ \t]\(\d+\))?)"
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
    # ── PERSON_NAME (title-prefix, multilingual; group 1 = name sans title) ──
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_EN,
        base_confidence=0.86,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person (en)",
        language="en",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_ES,
        base_confidence=0.86,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person (es)",
        language="es",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FR,
        base_confidence=0.86,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person (fr)",
        language="fr",
        deny_check=True,
    ),
    # ── PERSON_NAME (title + bare surname: title INCLUDED per convention) ──
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_EN_SURNAME,
        base_confidence=0.85,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person title+surname (en)",
        language="en",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_ES_SURNAME,
        base_confidence=0.85,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person title+surname (es)",
        language="es",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FR_SURNAME,
        base_confidence=0.85,
        group=1,
        context_type="PERSON_NAME",
        explanation="regex person title+surname (fr)",
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
    # ── PERSON_NAME (sp7 #6: honorific / initial / diacritic grammar) ──
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_TITLE_FULL_U,
        base_confidence=0.90,
        group=1,
        explanation="regex person title full unicode (sp7 #6)",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_TITLE_INITIALS,
        base_confidence=0.90,
        group=1,
        explanation="regex person title initials (sp7 #6)",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FIRST_MIDINITIAL_LAST,
        base_confidence=0.85,
        group=1,
        explanation="regex person first midinitial last (sp7 #6)",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_FIRST_MIDINITIAL_LAST_CAPS,
        base_confidence=0.85,
        group=1,
        explanation="regex person first midinitial last caps (sp7 #6)",
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
    # ── PERSON_NAME (field label: "Name: John Smith") ──────────────────
    PatternSpec(
        entity_type="PERSON_NAME",
        pattern=_PERSON_LABELED,
        base_confidence=0.86,
        group=1,
        explanation="regex person field label",
        deny_check=True,
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
    # ── DATE_TIME (ISO-8601 datetime — the TIMESTAMP shape) ────────────
    PatternSpec(
        entity_type="DATE_TIME",
        pattern=_DATETIME_ISO8601,
        base_confidence=0.95,
        group=1,
        explanation="regex iso-8601 datetime",
    ),
    # ── DATE_TIME (sp7 A3: natural-language / locale prose date grammar) ─
    PatternSpec(
        entity_type="DATE_TIME",
        pattern=_DATE_PROSE,
        base_confidence=0.80,
        group=1,
        explanation="regex prose date (sp7 A3)",
    ),
    # ── DATE_TIME (sp7 A3: space-separated datetime) ───────────────────
    PatternSpec(
        entity_type="DATE_TIME",
        pattern=_DATETIME_SPACE,
        base_confidence=0.92,
        group=1,
        explanation="regex space datetime (sp7 A3)",
    ),
    # ── sp2 external-coverage tranche ──────────────────────────────────
    PatternSpec(entity_type="TAX_ID", pattern=_TAX_ID_LABELED, base_confidence=0.88, group=1, explanation="regex tax id (labeled)"),
    PatternSpec(entity_type="JOB_TITLE", pattern=_JOB_TITLE_LABELED, base_confidence=0.84, group=1, explanation="regex job title (labeled)"),
    PatternSpec(entity_type="JOB_TITLE", pattern=_JOB_TITLE_LEXICON, base_confidence=0.80, group=1, explanation="regex job title (lexicon)"),
    PatternSpec(entity_type="HEALTH_CONDITION", pattern=_HEALTH_CONDITION_DIAGNOSIS, base_confidence=0.86, group=1, explanation="regex health condition (diagnosis label)"),
    PatternSpec(entity_type="HEALTH_CONDITION", pattern=_HEALTH_CONDITION_LEADIN, base_confidence=0.82, group=1, explanation="regex health condition (lead-in + marker)"),
    PatternSpec(entity_type="MEDICATION_NAME", pattern=_MEDICATION_DOSE, base_confidence=0.86, group=1, explanation="regex medication (name+dose)"),
    PatternSpec(entity_type="MEDICATION_NAME", pattern=_MEDICATION_LABELED, base_confidence=0.82, group=1, explanation="regex medication (labeled)"),
    PatternSpec(entity_type="HEALTH_INSURANCE_ID", pattern=_HEALTH_INSURANCE_INS, base_confidence=0.92, group=1, explanation="regex insurance id (INS-)"),
    PatternSpec(entity_type="HEALTH_INSURANCE_ID", pattern=_HEALTH_INSURANCE_LABELED, base_confidence=0.84, group=1, explanation="regex insurance id (labeled)"),
    PatternSpec(entity_type="CREDIT_CARD_FRAGMENT", pattern=_CC_FRAGMENT_ENDING, base_confidence=0.90, group=1, explanation="regex card fragment (ending)"),
    PatternSpec(entity_type="CREDIT_CARD_FRAGMENT", pattern=_CC_FRAGMENT_MASKED, base_confidence=0.92, group=1, explanation="regex card fragment (masked)"),
    PatternSpec(entity_type="CREDIT_CARD_FRAGMENT", pattern=_CC_FRAGMENT_LABELED, base_confidence=0.86, group=1, explanation="regex card fragment (labeled)"),
    PatternSpec(entity_type="VISA_NUMBER", pattern=_VISA_NUMBER_LABELED, base_confidence=0.88, group=1, explanation="regex visa number (labeled)"),
    PatternSpec(entity_type="PRESCRIPTION_NUMBER", pattern=_PRESCRIPTION_RX, base_confidence=0.92, group=1, explanation="regex prescription (RX-)"),
    PatternSpec(entity_type="PRESCRIPTION_NUMBER", pattern=_PRESCRIPTION_LABELED, base_confidence=0.84, group=1, explanation="regex prescription (labeled)"),
    PatternSpec(entity_type="DEVICE_IDENTIFIER", pattern=_DEVICE_ID_LABELED, base_confidence=0.86, group=1, explanation="regex device id (labeled)"),
    PatternSpec(entity_type="DEVICE_IDENTIFIER", pattern=_DEVICE_ID_UUID, base_confidence=0.84, group=1, explanation="regex device id (uuid)"),
    PatternSpec(entity_type="SOCIAL_MEDIA_HANDLE", pattern=_SOCIAL_MEDIA_HANDLE, base_confidence=0.86, group=1, explanation="regex social media handle"),
    PatternSpec(entity_type="EDUCATION_LEVEL", pattern=_EDUCATION_LEVEL, base_confidence=0.82, group=1, explanation="regex education level"),
    PatternSpec(entity_type="GENDER", pattern=_GENDER, base_confidence=0.86, group=1, explanation="regex gender (contextual)"),
    PatternSpec(entity_type="NATIONALITY", pattern=_NATIONALITY, base_confidence=0.84, group=1, explanation="regex nationality (contextual)"),
    PatternSpec(entity_type="ETHNICITY", pattern=_ETHNICITY, base_confidence=0.84, group=1, explanation="regex ethnicity (contextual)"),
    PatternSpec(entity_type="POLITICAL_OPINION", pattern=_POLITICAL_OPINION, base_confidence=0.84, group=1, explanation="regex political opinion (contextual)"),
    PatternSpec(entity_type="RELIGIOUS_BELIEF", pattern=_RELIGIOUS_BELIEF, base_confidence=0.84, group=1, explanation="regex religious belief (contextual)"),
    PatternSpec(entity_type="MARITAL_STATUS", pattern=_MARITAL_STATUS, base_confidence=0.86, group=1, explanation="regex marital status (contextual)"),
    PatternSpec(entity_type="HOUSEHOLD_SIZE", pattern=_HOUSEHOLD_SIZE, base_confidence=0.86, group=1, explanation="regex household size (labeled)"),
    PatternSpec(entity_type="VEHICLE_MODEL", pattern=_VEHICLE_MODEL_LABELED, base_confidence=0.84, group=1, explanation="regex vehicle model (labeled)"),
    PatternSpec(entity_type="VEHICLE_MODEL", pattern=_VEHICLE_MODEL_YEAR_MAKE, base_confidence=0.86, group=1, explanation="regex vehicle model (year+make)"),
    PatternSpec(entity_type="PROCEDURE_NAME", pattern=_PROCEDURE_LABELED, base_confidence=0.84, group=1, explanation="regex procedure (labeled)"),
    PatternSpec(entity_type="PROCEDURE_NAME", pattern=_PROCEDURE_MODALITY, base_confidence=0.84, group=1, explanation="regex procedure (modality)"),
    PatternSpec(entity_type="BIOMETRIC_ID", pattern=_BIOMETRIC_BIO, base_confidence=0.92, group=1, explanation="regex biometric id (BIO-)"),
    PatternSpec(entity_type="BIOMETRIC_ID", pattern=_BIOMETRIC_LABELED, base_confidence=0.84, group=1, explanation="regex biometric id (labeled)"),
    PatternSpec(entity_type="COURT_CASE_NUMBER", pattern=_COURT_CASE_YEAR_FORM, base_confidence=0.90, group=1, explanation="regex court case (year form)"),
    PatternSpec(entity_type="DOCKET_NUMBER", pattern=_DOCKET_FEDERAL, base_confidence=0.90, group=1, explanation="regex docket (federal form)"),
    PatternSpec(entity_type="INVOICE_NUMBER", pattern=_INVOICE_INV, base_confidence=0.90, group=1, explanation="regex invoice (INV-)"),
    PatternSpec(entity_type="INVOICE_NUMBER", pattern=_INVOICE_LABELED, base_confidence=0.90, group=1, explanation="regex invoice (labeled)"),
    PatternSpec(entity_type="SWIFT_BIC", pattern=_SWIFT_LABELED, base_confidence=0.88, group=1, explanation="regex swift bic (labeled)"),
    PatternSpec(entity_type="DRIVERS_LICENSE", pattern=_DL_PREFIXED, base_confidence=0.90, group=1, explanation="regex drivers license (DL-)"),
    PatternSpec(entity_type="DRIVERS_LICENSE", pattern=_DL_LABELED, base_confidence=0.84, group=1, explanation="regex drivers license (labeled value)"),
    PatternSpec(entity_type="SALARY", pattern=_SALARY_LABELED, base_confidence=0.90, group=1, explanation="regex salary (labeled, $-inclusive)"),
    PatternSpec(entity_type="API_KEY", pattern=_API_KEY_SK, base_confidence=0.92, group=1, explanation="regex api key (sk-)"),
    PatternSpec(entity_type="API_KEY", pattern=_API_KEY_LABELED, base_confidence=0.86, group=1, explanation="regex api key (labeled)"),
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
        pattern=_DRIVERS_LICENSE_BARE_KW,
        base_confidence=0.80,
        group=1,
        explanation="regex drivers license (bare keyword)",
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
    # ── CREDIT_CARD (fragment / masked last-4) ───────────────────────
    # Emits CREDIT_CARD, not CREDIT_CARD_FRAGMENT: the census authority
    # maps CREDIT_CARD_FRAGMENT detections to _BENCHMARK_IGNORE while the
    # corpus loader maps fragment TRUTH to CREDIT_CARD — so the old label
    # was eval-unreachable (the DEA_NUMBER/NPI_NUMBER incident class; see
    # tests/test_pattern_label_alignment.py). No validator on purpose: a
    # 4-digit fragment cannot pass the full-card Luhn check.
    PatternSpec(
        entity_type="CREDIT_CARD",
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
    # ── ORGANIZATION (sp7 #8: institution / firm grammar) ──────────────
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_INSTITUTION,
        base_confidence=0.62,
        group=1,
        explanation="regex organization institution (sp7 #8)",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_COURT,
        base_confidence=0.62,
        group=1,
        explanation="regex organization court (sp7 #8)",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_INSTITUTION_OF,
        base_confidence=0.62,
        group=1,
        explanation="regex organization institution-of (sp7 #8)",
        deny_check=True,
    ),
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_FIRM,
        base_confidence=0.72,
        group=1,
        explanation="regex organization firm (sp7 #8)",
        deny_check=True,
    ),
    # ── ORGANIZATION (single-token CamelCase: "InnovateLabs") ──────────
    PatternSpec(
        entity_type="ORGANIZATION",
        pattern=_ORGANIZATION_CAMELCASE,
        base_confidence=0.72,
        group=1,
        explanation="regex organization camelcase",
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
    PatternSpec(
        entity_type="ADDRESS",
        pattern=_ADDRESS_AMBIGUOUS,
        base_confidence=0.80,
        group=1,
        explanation="regex address ambiguous evidence-gated (sp7 A4)",
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
    PatternSpec(
        entity_type="GPS_COORDINATES",
        pattern=_GPS_HEMISPHERE,
        base_confidence=0.88,
        group=0,
        explanation="regex gps hemisphere (sp7 #7)",
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
    # ── NPI_NUMBER / DEA_NUMBER (Phase 2) ──────────────────────────────
    PatternSpec(
        entity_type="NPI_NUMBER",
        pattern=_NPI,
        base_confidence=0.88,
        group=1,
        validator="npi",
        # Corpus NPIs are mostly Luhn-invalid; demote to 0.70 rather than skip (recall-critical).
        invalid_confidence=0.70,
        # context_type is a shared CONTEXT_WORDS vocabulary key (npi/dea/medical terms), not the emitted label.
        context_type="MEDICAL_LICENSE",
        explanation="regex npi",
    ),
    PatternSpec(
        entity_type="DEA_NUMBER",
        pattern=_DEA,
        base_confidence=0.88,
        group=1,
        validator="dea",
        valid_confidence=0.93,
        # Corpus DEA values are mostly checksum-invalid; demote to 0.70 rather than skip (recall-critical).
        invalid_confidence=0.70,
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
    # All context-gated — the regex REQUIRES the keyword adjacent to the
    # captured group, not merely in the ±50-char context window.  This
    # regex-level gating is a structural guarantee on par with a checksum
    # for credit cards or IBANs: a match cannot happen without the
    # keyword, so false positives on random numeric content are
    # architecturally impossible.
    #
    # Base confidence is therefore set to ≥0.90 so Phase 3 findings
    # qualify for the swarm's Layer 1 fast-pass
    # (``SwarmConfig.fast_pass_threshold = 0.90``) — the baseline
    # catches these patterns and emits them directly without paying for
    # NER fusion.  This is the paper v11 §5.6 recommendation: route
    # rule-based detections through the fast path, let the mixture of
    # experts handle the open-vocabulary tail.
    PatternSpec(
        entity_type="CVV",
        pattern=_CVV,
        base_confidence=0.92,
        group=1,
        explanation="regex cvv with card-context gate",
    ),
    PatternSpec(
        entity_type="PIN",
        pattern=_PIN,
        base_confidence=0.90,
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
        base_confidence=0.90,
        group=1,
        explanation="regex court case no. with legal-context gate",
    ),
    PatternSpec(
        entity_type="DOCKET_NUMBER",
        pattern=_DOCKET,
        base_confidence=0.90,
        group=1,
        explanation="regex docket no. with legal-context gate",
    ),
    PatternSpec(
        entity_type="BAR_NUMBER",
        pattern=_BAR_NUMBER,
        base_confidence=0.90,
        group=1,
        explanation="regex state bar identifier",
    ),
    PatternSpec(
        entity_type="INVOICE_NUMBER",
        pattern=_INVOICE,
        base_confidence=0.90,
        group=1,
        explanation="regex invoice reference",
    ),
    PatternSpec(
        entity_type="INSURANCE_POLICY_NUMBER",
        pattern=_INSURANCE_POLICY,
        base_confidence=0.90,
        group=1,
        explanation="regex insurance policy reference",
    ),
    PatternSpec(
        entity_type="SALARY",
        pattern=_SALARY,
        base_confidence=0.90,
        group=1,
        explanation="regex salary/compensation amount",
    ),
    # ── sp3 v2.2.0 re-baseline tranche ─────────────────────────────────
    PatternSpec(
        entity_type="CVV",
        pattern=_CVV_ENCODED,
        base_confidence=0.90,
        group=1,
        explanation="regex CVV (base64/alnum value behind CVV label)",
    ),
    PatternSpec(
        entity_type="PIN",
        pattern=_PIN_ENCODED,
        base_confidence=0.90,
        group=1,
        explanation="regex PIN (base64/alnum value behind PIN label)",
    ),
    PatternSpec(
        entity_type="PASSWORD",
        pattern=_PASSWORD_QUOTED,
        base_confidence=0.92,
        group=1,
        explanation="regex password (quoted code/config/JSON value)",
    ),
    PatternSpec(
        entity_type="INSURANCE_POLICY_NUMBER",
        pattern=_INSURANCE_POLICY_ENCODED,
        base_confidence=0.90,
        group=1,
        explanation="regex insurance policy (OCR/base64/zero-width value)",
    ),
    PatternSpec(
        entity_type="AUTHENTICATION_TOKEN",
        pattern=_AUTH_TOKEN_LABELED,
        base_confidence=0.90,
        group=1,
        explanation="regex auth token (value behind authentication-token label)",
    ),
    PatternSpec(
        entity_type="AUTHENTICATION_TOKEN",
        pattern=_AUTH_TOKEN_BEARER,
        base_confidence=0.88,
        group=1,
        explanation="regex auth token (intrinsic Bearer <token>)",
    ),
    PatternSpec(
        entity_type="AUTHENTICATION_TOKEN",
        pattern=_AUTH_TOKEN_B64_BEARER,
        base_confidence=0.88,
        group=1,
        explanation="regex auth token (base64-encoded Bearer token)",
    ),
    PatternSpec(
        entity_type="AUTHENTICATION_TOKEN",
        pattern=_AUTH_TOKEN_JWT_TRUNC,
        base_confidence=0.88,
        group=1,
        explanation="regex auth token (truncated JWT placeholder eyJ...)",
    ),
    # ── GDPR Article-9 special categories (63 -> 66) ────────────────────
    PatternSpec(
        entity_type="SEXUAL_ORIENTATION",
        pattern=_SEXUAL_ORIENTATION,
        base_confidence=0.86,
        group=1,
        explanation="regex sexual orientation (labeled lexicon)",
    ),
    PatternSpec(
        entity_type="TRADE_UNION_MEMBERSHIP",
        pattern=_TRADE_UNION,
        base_confidence=0.84,
        group=1,
        explanation="regex trade-union membership (labeled value)",
    ),
    PatternSpec(
        entity_type="GENETIC_DATA",
        pattern=_GENETIC_LABELED,
        base_confidence=0.84,
        group=1,
        explanation="regex genetic data (labeled value)",
    ),
    PatternSpec(
        entity_type="GENETIC_DATA",
        pattern=_GENETIC_INTRINSIC,
        base_confidence=0.86,
        group=1,
        explanation="regex genetic data (intrinsic gene/rs-ID + qualifier)",
    ),
)
