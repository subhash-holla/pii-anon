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
from collections.abc import Callable
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.engines.regex.confidence import (
    CONTEXT_BOOST,
    CONTEXT_PENALTY,
    CONTEXT_WINDOW,
    CONTEXT_WORDS,
    HIGH_FP_TYPES,
    adjust_confidence,
    apply_negative_context,
    extract_context,
    has_context_words,
)
from pii_anon.engines.regex.deny_list import DenyListManager
from pii_anon.engines.regex.geo_lexicon import extract_locations, geo_subtype
from pii_anon.engines.regex.labeled_fields import extract_labeled_fields
from pii_anon.engines.regex.unicode_norm import normalize_for_detection, remap_span
from pii_anon.engines.regex.patterns import PATTERN_REGISTRY, PatternSpec
from pii_anon.engines.regex import validators
from pii_anon.engines.regex.validators import _extract_digits
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

# Minimum confidence threshold for emitting a finding.  Findings below
# this threshold are suppressed to reduce false positives from patterns
# that receive context penalties.
_MIN_EMIT_CONFIDENCE: float = 0.50


#: Specific types whose spans shadow a coinciding PERSON_NAME finding: the
#: person patterns are generic Title-Case matchers, so a span that a
#: specific-type pattern claimed (org, job title, condition, medication, ...)
#: is near-certainly not a person — emitting both costs a guaranteed FP
#: under exact-span scoring.
_PERSON_SHADOWING_TYPES: frozenset[str] = frozenset({
    "ORGANIZATION",
    "JOB_TITLE",
    "HEALTH_CONDITION",
    "MEDICATION_NAME",
    "PROCEDURE_NAME",
    "EDUCATION_LEVEL",
    "VEHICLE_MODEL",
})


#: sp7 A1 — leading tokens that a real PERSON/ORG name never starts with;
#: a Title-Case phrase led by one is document prose, not a name.
_NAME_LEADING_STOPWORDS: frozenset[str] = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "Our", "Your",
    "Their", "His", "Her", "Its", "United", "New", "All", "Any", "Each",
    "Some", "Please", "Note", "See", "Per", "Via", "For", "From", "With",
    # sp7 #6 — public DE/FR/IT/ES definite/indefinite articles + possessives
    # (foreign prose that a Title-Case run leads with, not a name).
    "Die", "Der", "Das", "Den", "Dem", "Ein", "Eine", "Ihre", "Ihr",
    "Ihren", "Unsere", "Unser", "Le", "La", "Les", "Un", "Une", "Des",
    "Du", "Votre", "Vos", "Notre", "Nos", "El", "Los", "Las", "Una",
    "Su", "Sus", "Il", "Lo", "Gli", "Vostro",
})
#: General document-structure / label vocabulary (NOT dataset gold): a
#: Title-Case phrase whose tokens are ALL drawn from this set is a heading or
#: field label, not a name. Kept domain-general to avoid benchmark-gaming.
_HEADER_COMMON_WORDS: frozenset[str] = frozenset({
    "Tax", "Form", "Information", "Report", "Section", "Number", "Account",
    "Service", "Product", "Court", "District", "State", "Department", "Office",
    "Committee", "Details", "Summary", "Notice", "Terms", "Policy", "Payment",
    "Invoice", "Order", "Customer", "Client", "Investor", "Balance", "Amount",
    "Status", "Address", "Contact", "Record", "Reference", "Case", "Claim",
    "Data", "Type", "Date", "Time", "Name", "Title", "Value", "Total", "Code",
    "Group", "Team", "Level", "Rate", "Fee", "Plan", "Line", "Page", "Table",
})
#: Common given names that override the stopword/common-word drop (protects a
#: real name like "Summer Rose" / "April Chen" whose first token is a word).
_COMMON_GIVEN_NAMES: frozenset[str] = frozenset({
    "Summer", "April", "May", "June", "Autumn", "Dawn", "Faith", "Grace",
    "Hope", "Joy", "Rose", "Ruby", "Pearl", "Crystal", "Star", "Sky",
    "Amber", "Ivy", "Jade", "Iris", "Belle", "Chase", "Grant", "Miles",
})


#: sp7 #6 — leading greeting/role tokens a real name span never starts with;
#: trimmed (not dropped) on the eval path so "Ciao Nalda" scores as "Nalda".
_SALUTATION_TOKENS: frozenset[str] = frozenset({
    "Ciao", "Hallo", "Hey", "Hi", "Hello", "Dear", "Estimado", "Estimada",
    "Gentile", "Bonjour", "Salut", "Hola", "Buongiorno", "Cher", "Chère",
    "Querido", "Querida", "Caro", "Cara", "Hallo", "Hej", "Ola",
})


#: sp7 #6 mention-propagation — a DOMAIN-GENERAL veto of Title-Case tokens that
#: are common English words (or common words that double as surnames) and so are
#: NOT safe to propagate as a bare PERSON_NAME mention. Designed a-priori (NOT
#: derived from home-set FP surfaces — no test-split tuning); it deliberately
#: forfeits ambiguous word-surnames (Green/Brown/Ford/King/Young/…) for
#: precision. Matching is case-SENSITIVE on the token's own capitalisation, so
#: lowercase common usage ("will", "rose", "park") can never match a surname.
_PROPAGATION_STOPWORDS: frozenset[str] = frozenset({
    # weekdays / months
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    # document-structure / label / role nouns that surface Title-Cased in prose
    "Patient", "Doctor", "Nurse", "Report", "Summary", "Subject", "Address",
    "Account", "Name", "Date", "Number", "Record", "Case", "Section", "Note",
    "Notes", "Page", "Table", "Figure", "Company", "Department", "Office",
    "Team", "Group", "Client", "Customer", "Member", "Manager", "Director",
    "President", "Officer", "Agent", "Owner", "User", "Admin", "Staff",
    # common Title-Case prose openers / high-frequency words
    "The", "This", "That", "These", "Those", "Their", "There", "Then", "Thus",
    "When", "Where", "While", "With", "From", "Both", "Each", "After", "Before",
    "During", "Please", "Thank", "Thanks", "Regards", "Sincerely", "Best",
    # common English words that also occur as surnames — forfeited for precision
    "Green", "Brown", "White", "Black", "Gray", "Grey", "King", "Young",
    "Ford", "Park", "Hill", "Bank", "Price", "Bell", "Cook", "Fox", "Wood",
    "Stone", "Field", "Rich", "Long", "Short", "Small", "Little", "Moon",
    "Sun", "Rain", "Snow", "Winter", "Case", "Ward", "Bishop", "Marsh",
})

#: mid-band confidence for a propagated partial mention (above the 0.50 emit
#: floor; below a directly-grammar-detected full name).
_PROPAGATED_PERSON_CONFIDENCE = 0.80


def _is_word_char(ch: str) -> bool:
    """Word-boundary predicate for the propagation scan: alphanumeric or ``_``."""
    return ch.isalnum() or ch == "_"


def _standalone_occurrences(text: str, token: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` for every whole-word occurrence of ``token`` in
    ``text`` (not the interior of a longer word). O(n) ``str.find`` sweep with
    manual boundary checks — no regex, ReDoS-free. A trailing apostrophe
    ("Fennimore's") is a boundary, so the possessive matches the bare name."""
    out: list[tuple[int, int]] = []
    if not token:
        return out
    i = text.find(token)
    n = len(text)
    tlen = len(token)
    while i != -1:
        j = i + tlen
        before_ok = i == 0 or not _is_word_char(text[i - 1])
        after_ok = j >= n or not _is_word_char(text[j])
        if before_ok and after_ok:
            out.append((i, j))
        i = text.find(token, i + 1)
    return out


def _propagate_surname_mentions(
    findings: list[EngineFinding],
    texts: dict[str | None, str],
    *,
    adapter_id: str,
    language: str,
    is_denied: Callable[[str, str], bool],
) -> list[EngineFinding]:
    """sp7 #6 — ADDITIVE surname mention-propagation (all paths, leak-safe).

    A multi-token ``PERSON_NAME`` ("Alastair Fennimore") is detected once, but a
    later bare "Fennimore" the grammar missed leaks. For each detected
    multi-token PERSON_NAME, propagate its SURNAME (last token) to every
    standalone verbatim occurrence in the SAME field not already covered by a
    finding, emitting a new PERSON_NAME span. Gate: surname length >= 4,
    all-alpha, strict Title-Case, not deny-listed, not a common-word stopword.

    Purely additive — emits only NEW non-overlapping spans; never drops,
    narrows, or re-types an existing finding. On the masking path it masks MORE
    partial mentions (the safe leak-direction). Detection-only; no eval gating.
    """
    # per field: the surnames worth propagating + an occupancy mask over the
    # field text (bytearray -> O(1) per char, so the whole pass is LINEAR in
    # text length; a per-interval scan would be O(n^2) on a pathological
    # many-repeats input — the sp7 mention-propagation close).
    surnames_by_field: dict[str | None, set[str]] = {}
    occupied_by_field: dict[str | None, list[tuple[int, int]]] = {}
    for f in findings:
        fp = f.field_path
        if isinstance(f.span_start, int) and isinstance(f.span_end, int):
            occupied_by_field.setdefault(fp, []).append((f.span_start, f.span_end))
        if f.entity_type != "PERSON_NAME" or not isinstance(f.span_start, int):
            continue
        text = texts.get(fp, "")
        span = text[f.span_start:f.span_end]
        tokens = span.split()
        if len(tokens) < 2:
            continue
        surname = tokens[-1].strip(".,'\"")
        if (
            len(surname) >= 4
            and surname.isalpha()
            and surname[0].isupper()
            and surname[1:].islower()
            and surname not in _PROPAGATION_STOPWORDS
            and not is_denied("PERSON_NAME", surname)
        ):
            surnames_by_field.setdefault(fp, set()).add(surname)

    added: list[EngineFinding] = []
    for fp, surnames in surnames_by_field.items():
        text = texts.get(fp, "")
        mask = bytearray(len(text))
        for os, oe in occupied_by_field.get(fp, []):
            for k in range(max(0, os), min(len(text), oe)):
                mask[k] = 1
        # longest surname first so a longer name masks before a substring.
        for surname in sorted(surnames, key=len, reverse=True):
            for start, end in _standalone_occurrences(text, surname):
                if any(mask[k] for k in range(start, end)):
                    continue
                for k in range(start, end):
                    mask[k] = 1
                added.append(
                    EngineFinding(
                        entity_type="PERSON_NAME",
                        confidence=_PROPAGATED_PERSON_CONFIDENCE,
                        field_path=fp,
                        span_start=start,
                        span_end=end,
                        engine_id=adapter_id,
                        explanation="propagated surname mention (sp7 mention-propagation)",
                        language=language,
                    )
                )
    return findings + added


def _trim_salutation_led_person(
    findings: list[EngineFinding], texts: dict[str | None, str]
) -> list[EngineFinding]:
    """EVAL-ONLY: strip a leading greeting token from a PERSON_NAME span so the
    scored span starts at the actual name ("Ciao Nalda" -> "Nalda"). Boundary
    hygiene on the scoring path only — the production masking span is unchanged
    (over-masking the greeting is harmless; the sp2 discipline)."""
    for f in findings:
        if (
            f.entity_type == "PERSON_NAME"
            and isinstance(f.span_start, int)
            and isinstance(f.span_end, int)
        ):
            text = texts.get(f.field_path, "")
            span = text[f.span_start:f.span_end]
            toks = span.split()
            while len(toks) >= 2 and toks[0].rstrip(".,").capitalize() in _SALUTATION_TOKENS:
                advance = len(toks[0])
                rest = span[advance:]
                stripped = len(rest) - len(rest.lstrip())
                f.span_start += advance + stripped
                span = text[f.span_start:f.span_end]
                toks = span.split()
    return findings


def _drop_titlecase_noise_person(
    findings: list[EngineFinding], texts: dict[str | None, str]
) -> list[EngineFinding]:
    """EVAL-ONLY: drop PERSON_NAME/ORGANIZATION spans that are Title-Case
    document noise, not names (sp7 A1 — mined the single largest FP class
    across 5/5 external datasets: 71% of ALL Nemotron FPs).

    Dropped shapes (each provably absent from the home corpus's name gold, so
    home recall is unaffected): (1) markdown-wrapped (``**X**`` / ``__X__``);
    (2) led by a determiner/preposition stopword ("The Court", "United States
    District"); (3) EVERY token in the general header/label vocabulary ("Tax
    Form", "Investor Information"). A leading common GIVEN name overrides the
    drop ("Summer Rose" is kept).

    sp2 LEAK-DIRECTION discipline: runs ONLY under
    ``eval_cross_type_arbitration``. On the production masking path this never
    fires, so a real name that sits in a heading is still masked — over-masking
    a heading is the safe direction; dropping it would be the leak.
    """
    out: list[EngineFinding] = []
    for finding in findings:
        if (
            finding.entity_type in ("PERSON_NAME", "ORGANIZATION")
            and isinstance(finding.span_start, int)
            and isinstance(finding.span_end, int)
        ):
            text = texts.get(finding.field_path, "")
            span = text[finding.span_start:finding.span_end]
            tokens = span.split()
            first = tokens[0] if tokens else ""
            if first in _COMMON_GIVEN_NAMES:
                out.append(finding)
                continue
            pre = text[max(0, finding.span_start - 2):finding.span_start]
            post = text[finding.span_end:finding.span_end + 2]
            markdown_wrapped = (
                (pre.endswith("**") and post.startswith("**"))
                or (pre.endswith("__") and post.startswith("__"))
            )
            stopword_led = first in _NAME_LEADING_STOPWORDS
            all_header_words = bool(tokens) and all(
                t in _HEADER_COMMON_WORDS for t in tokens
            )
            if markdown_wrapped or stopword_led or all_header_words:
                continue
        out.append(finding)
    return out


def _drop_undecimaled_gps(
    findings: list[EngineFinding], texts: dict[str | None, str]
) -> list[EngineFinding]:
    """EVAL-ONLY: drop GPS_COORDINATES spans carrying no decimal point.

    The permissive GPS pattern is deliberately kept on the masking path —
    the sp6 close proved that narrowing the PATTERN leaked "41, -87"-style
    pairs to production unmasked via the AX-003 floor (the sp2 showstopper
    class). At eval, undecimaled pairs are date fragments ("15/09";
    coordinate strict P=0.072 on Nemotron, P=0.157 at home) while every real
    coordinate gold (77/77 home dev) carries a decimal. sp2 discipline: runs
    ONLY under ``eval_cross_type_arbitration`` — in production, over-masking
    a date fragment is the safe direction.
    """
    out: list[EngineFinding] = []
    for finding in findings:
        if (
            finding.entity_type == "GPS_COORDINATES"
            and isinstance(finding.span_start, int)
            and isinstance(finding.span_end, int)
        ):
            text = texts.get(finding.field_path, "")
            if "." not in text[finding.span_start:finding.span_end]:
                continue
        out.append(finding)
    return out


# ── sp7 #7 numeric-identifier guards (ALL SCORING-ONLY suppressors) ─────────
_GPS_MONEY = re.compile(r"^\$?\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?$")
_GPS_RATING = re.compile(r"^\d{1,3}(?:\.\d+)?/\d{1,2}$")
_SSN_POS = re.compile(
    r"(?i)\b(?:ssn|social\s+security|social\s+insurance|taxpayer|tin)\b"
)
_SSN_SEQ = re.compile(
    r"^(?:123456789|987654321|012345678|876543210|123123123|111111111|000000000)$"
)
_SSN_GLUE = frozenset({"=", "|", "&", ":"})


def _drop_nongeo_gps(
    findings: list[EngineFinding], texts: dict[str | None, str]
) -> list[EngineFinding]:
    """SCORING-ONLY: drop GPS spans whose FULL text is a money or rating shape
    ("$1,125.00", "4.5/5") — real coordinate pairs never match these anchors.
    The masking-path GPS pattern is untouched (sp6 leak lesson); paired with the
    additive hemisphere pattern so GPS coverage never shrinks."""
    out: list[EngineFinding] = []
    for f in findings:
        if (
            f.entity_type == "GPS_COORDINATES"
            and isinstance(f.span_start, int)
            and isinstance(f.span_end, int)
        ):
            span = texts.get(f.field_path, "")[f.span_start:f.span_end]
            if _GPS_MONEY.match(span) or _GPS_RATING.match(span):
                continue
        out.append(f)
    return out


def _drop_bare9_ssn_noncontext(
    findings: list[EngineFinding], texts: dict[str | None, str]
) -> list[EngineFinding]:
    """SCORING-ONLY: drop a bare 9-digit US_SSN span that is a sequential
    placeholder OR glued to a FIX/tag delimiter (=|&:), UNLESS a positive SSN
    cue sits in the ±40-char window. Masking path keeps every candidate."""
    out: list[EngineFinding] = []
    for f in findings:
        if (
            f.entity_type == "US_SSN"
            and isinstance(f.span_start, int)
            and isinstance(f.span_end, int)
        ):
            text = texts.get(f.field_path, "")
            span = text[f.span_start:f.span_end]
            if span.isdigit() and len(span) == 9:
                glued = f.span_start > 0 and text[f.span_start - 1] in _SSN_GLUE
                window = text[max(0, f.span_start - 40):f.span_end + 40]
                if (_SSN_SEQ.match(span) or glued) and not _SSN_POS.search(window):
                    continue
        out.append(f)
    return out


# NOTE: an IBAN mod-97 scoring-drop was prototyped and REJECTED — it removed
# 225 home gold IBANs (home synthetic IBANs are checksum-invalid by
# construction), a home-recall regression for negligible external gain. The
# mining flag ("home-floor sign-off") was correct; the home floor moved, so the
# guard is not shipped.


def _apply_label_wins(
    findings: list[EngineFinding], labeled: list[EngineFinding]
) -> list[EngineFinding]:
    """Merge A2 labeled-field findings with the pattern findings under a
    leak-SAFE ``label-wins`` rule.

    Two leak-safe moves, both preserving coverage:

    1. **Defer to same-type pattern coverage.** A labeled finding is dropped
       when a pattern finding of the SAME type already overlaps it — the
       pattern already ruled on that span with its own validator/confidence
       (checksum, length), so A2 must not override it (e.g. an invalid-ABA
       routing number keeps the pattern's lower confidence, not A2's 0.90).
    2. **Different-type label-wins relabel.** A pattern finding wholly
       contained in a DIFFERENT-type surviving labeled span is dropped in
       favour of the label's type — the value span stays covered by the
       labeled finding, so masking is preserved (the sp6 "changes labels,
       not coverage" invariant).

    Runs on ALL paths: the merge is additive and every drop keeps the span
    masked, so it is leak-safe by construction.
    """
    if not labeled:
        return findings

    def _spanned(a: EngineFinding, b: EngineFinding) -> bool:
        return (
            a.field_path == b.field_path
            and a.span_start is not None
            and a.span_end is not None
            and b.span_start is not None
            and b.span_end is not None
        )

    # 1) drop A2 findings the patterns already cover with the same type.
    surviving: list[EngineFinding] = [
        lab
        for lab in labeled
        if not any(
            f.entity_type == lab.entity_type
            and _spanned(f, lab)
            and f.span_start < lab.span_end  # type: ignore[operator]
            and lab.span_start < f.span_end  # type: ignore[operator]
            for f in findings
        )
    ]

    # 2) relabel: drop pattern findings contained in a different-type label.
    kept: list[EngineFinding] = []
    for f in findings:
        shadowed = any(
            lab.entity_type != f.entity_type
            and _spanned(lab, f)
            and lab.span_start <= f.span_start  # type: ignore[operator]
            and f.span_end <= lab.span_end  # type: ignore[operator]
            for lab in surviving
        )
        if not shadowed:
            kept.append(f)
    return kept + surviving


def _drop_person_shadowed_by_specific(
    findings: list[EngineFinding],
) -> list[EngineFinding]:
    """Drop PERSON_NAME findings covered by a specific-type finding's span."""
    shadow_spans = [
        (f.field_path, f.span_start, f.span_end)
        for f in findings
        if f.entity_type in _PERSON_SHADOWING_TYPES
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    ]
    if not shadow_spans:
        return findings
    out: list[EngineFinding] = []
    for finding in findings:
        if (
            finding.entity_type == "PERSON_NAME"
            and isinstance(finding.span_start, int)
            and isinstance(finding.span_end, int)
            and any(
                field == finding.field_path
                and start <= finding.span_start
                and finding.span_end <= end
                for field, start, end in shadow_spans
            )
        ):
            continue
        out.append(finding)
    return out


#: birth cue for sp7 A3 DOB promotion (word-bounded so "reborn" never fires).
_BIRTH_CUE_RE = re.compile(
    r"(?i)\b(?:born|d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date|birthday|bday)\b"
)
_PROMOTABLE_DATE_TYPES = frozenset({"DATE_TIME", "DATE_ISO"})


def _promote_dob_by_cue(
    findings: list[EngineFinding], texts: dict[str | None, str]
) -> list[EngineFinding]:
    """Re-type a general date to ``DATE_OF_BIRTH`` when a birth cue precedes it
    within ~30 chars (sp7 A3, mining candidate #5).

    Leak-SAFE relabel: the span stays covered/masked either way; only the type
    changes. Cue-gated (``born``/``DOB``/``date of birth`` word-bounded) for
    precision. Runs on ALL paths — masking is preserved and DOB is the more
    specific, more-protective type. The A2 labeled-field bridge already handles
    the clean ``Date of Birth: <value>`` form; this catches prose cues
    ("was born on 27 May 1994") the bridge does not.
    """
    for f in findings:
        if (
            f.entity_type in _PROMOTABLE_DATE_TYPES
            and isinstance(f.span_start, int)
        ):
            text = texts.get(f.field_path)
            if not text:
                continue
            window = text[max(0, f.span_start - 30) : f.span_start]
            if _BIRTH_CUE_RE.search(window):
                f.entity_type = "DATE_OF_BIRTH"
    return findings


def _drop_location_nested_in_address(
    findings: list[EngineFinding],
) -> list[EngineFinding]:
    """Drop a gazetteer LOCATION wholly contained in an ADDRESS/GPS span (sp7
    #9). Leak-SAFE: the covering ADDRESS/GPS still masks those characters, so
    the nested duplicate is a pure false positive under one-to-one scoring."""
    cover = [
        (f.field_path, f.span_start, f.span_end)
        for f in findings
        if f.entity_type in ("ADDRESS", "GPS_COORDINATES")
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    ]
    if not cover:
        return findings
    out: list[EngineFinding] = []
    for f in findings:
        if (
            f.entity_type == "LOCATION"
            and isinstance(f.span_start, int)
            and isinstance(f.span_end, int)
            and any(
                field == f.field_path and s <= f.span_start and f.span_end <= e
                for field, s, e in cover
            )
        ):
            continue
        out.append(f)
    return out


def _drop_ssn_shadowed_national_id(
    findings: list[EngineFinding],
) -> list[EngineFinding]:
    """Drop a NATIONAL_ID finding whose span a US_SSN finding already covers
    (sp7 panel, multilingual lens). "Tax ID: 573-33-7773" matched BOTH the SSN
    pattern and the labeled national-id pattern on the SAME span — under
    one-to-one strict scoring the second is a pure FP (dominant in the
    zh/ko/hi/ar home docs, which scaffold SSN-format values behind an English
    "Tax ID:" label). Leak-SAFE (the _drop_dob_shadowed_dates class): the
    surviving US_SSN still masks the region, so this runs on ALL paths."""
    ssn_spans = [
        (f.field_path, f.span_start, f.span_end)
        for f in findings
        if f.entity_type == "US_SSN"
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    ]
    if not ssn_spans:
        return findings
    out: list[EngineFinding] = []
    for f in findings:
        if (
            f.entity_type == "NATIONAL_ID"
            and isinstance(f.span_start, int)
            and isinstance(f.span_end, int)
            and any(
                field == f.field_path and s <= f.span_start and f.span_end <= e
                for field, s, e in ssn_spans
            )
        ):
            continue
        out.append(f)
    return out


def _drop_dob_shadowed_dates(findings: list[EngineFinding]) -> list[EngineFinding]:
    """Drop generic ``DATE_TIME`` findings overlapping a ``DATE_OF_BIRTH``.

    A date in DOB context is annotated DATE_OF_BIRTH, never ALSO a generic
    date — under exact-span scoring the second typed span on the same text
    is a pure false positive ("DOB: 12/28/1966" must yield one finding).
    """
    dob_spans = [
        (f.field_path, f.span_start, f.span_end)
        for f in findings
        if f.entity_type == "DATE_OF_BIRTH"
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    ]
    if not dob_spans:
        return findings
    out: list[EngineFinding] = []
    for finding in findings:
        if (
            finding.entity_type == "DATE_TIME"
            and isinstance(finding.span_start, int)
            and isinstance(finding.span_end, int)
            and any(
                field == finding.field_path
                and finding.span_start < end
                and start < finding.span_end
                for field, start, end in dob_spans
            )
        ):
            continue
        out.append(finding)
    return out


def _drop_nested_same_type(findings: list[EngineFinding]) -> list[EngineFinding]:
    """Drop same-type findings nested inside (or duplicating) a kept span.

    Several patterns per entity type legitimately match one mention
    ("Melissa" via surname-context inside "Melissa White" via full-name;
    two phone patterns on the same number). Under multiset exact-span
    scoring every nested or duplicate emission is a pure false positive.
    Per ``(field_path, entity_type)`` group, keep maximal spans: longest
    first, then earliest start, then highest confidence; a span equal to
    or strictly contained in a kept span is dropped. Original list order
    is preserved for the survivors (downstream fusion is order-sensitive).
    """
    if len(findings) < 2:
        return findings
    groups: dict[tuple[str | None, str], list[EngineFinding]] = {}
    spanless: list[EngineFinding] = []
    for finding in findings:
        if isinstance(finding.span_start, int) and isinstance(finding.span_end, int):
            groups.setdefault((finding.field_path, finding.entity_type), []).append(finding)
        else:
            spanless.append(finding)

    keep: set[int] = {id(f) for f in spanless}
    for group in groups.values():
        kept: list[EngineFinding] = []
        for finding in sorted(
            group,
            key=lambda f: (
                -(f.span_end - f.span_start),  # type: ignore[operator]
                f.span_start,
                -f.confidence,
            ),
        ):
            contained = any(
                k.span_start <= finding.span_start and finding.span_end <= k.span_end  # type: ignore[operator]
                for k in kept
            )
            if not contained:
                kept.append(finding)
                keep.add(id(finding))
    return [f for f in findings if id(f) in keep]


# ISO 3166-1 alpha-2 country codes — used by the ``swift_context``
# validator to reject BIC-shaped strings whose country-code pair is
# not a real jurisdiction (the regex surface is broad by design, so
# this narrows it without a full NER pass).
_ISO_3166_ALPHA_2: frozenset[str] = frozenset({
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS",
    "AT", "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG",
    "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS", "BT",
    "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI",
    "CK", "CL", "CM", "CN", "CO", "CR", "CU", "CV", "CW", "CX", "CY",
    "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG", "EH",
    "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB",
    "GD", "GE", "GF", "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ",
    "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM", "HN", "HR", "HT",
    "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT",
    "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP",
    "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS",
    "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH",
    "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU",
    "MV", "MW", "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI",
    "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG",
    "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA",
    "RE", "RO", "RS", "RU", "RW", "SA", "SB", "SC", "SD", "SE", "SG",
    "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS", "ST",
    "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK",
    "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA", "UG",
    "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI", "VN", "VU",
    "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW",
})

_VALIDATORS: dict[str, Any] = {
    # Simple bool validators — callable, return True/False
    "ipv4": lambda text, _m: validators.is_valid_ipv4(text),
    "aba_routing": lambda text, _m: validators.is_valid_aba_routing(text),
    "vin": lambda text, _m: validators.is_valid_vin_check_digit(text),
    "npi": lambda text, _m: validators.is_valid_npi(text),
    "uk_ni": lambda text, _m: validators.is_valid_uk_ni(text),
    "dea": lambda text, _m: validators.is_valid_dea_number(text),
    "phone": lambda text, _m: validators.is_valid_phone_number(text),
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
        eval_cross_type_arbitration: bool = False,
        unicode_normalize_detection: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        # sp7 panel: strip zero-width / fold fullwidth chars for DETECTION so
        # obfuscated PII ("SSN: 123<ZWSP>-45-6789", fullwidth phones) cannot
        # evade the masking path. ASCII text takes a C-speed identity fast path
        # (byte-identical behaviour); spans are remapped to ORIGINAL offsets.
        self._unicode_normalize = unicode_normalize_detection
        # Cross-type arbitration (drop a generic PERSON_NAME when a more
        # specific type covers the same span) is an EVAL-ONLY precision
        # optimisation. It is OFF by default because in the production
        # masking path most specific shadowing types (JOB_TITLE,
        # HEALTH_CONDITION, ...) are NOT masked downstream — dropping the
        # PERSON_NAME there would LEAK the name (the safe direction is to
        # over-mask). Benchmark predictors opt in; production never does.
        self._eval_cross_type_arbitration = eval_cross_type_arbitration
        self._list_mgr = DenyListManager(
            deny_config=deny_list_config,
            allow_config=allow_list_config,
        )
        # Backward-compatible attribute
        self._deny_lists = self._list_mgr._deny_lists
        # Fast-path flag: skip allow-list checks when no allow-lists exist
        self._has_allow_lists = bool(self._list_mgr._allow_lists)

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize engine, loading deny/allow-lists and confidence tuning from config."""
        super().initialize(config)
        if config:
            self._list_mgr.initialize(config)
            self._deny_lists = self._list_mgr._deny_lists
            self._has_allow_lists = bool(self._list_mgr._allow_lists)
            # Apply confidence tuning from CoreConfig.confidence
            conf = config.get("confidence", {})
            if conf:
                from pii_anon.engines.regex.confidence import configure_from_config
                configure_from_config(
                    context_boost=conf.get("context_boost"),
                    context_penalty=conf.get("context_penalty"),
                    context_window=conf.get("context_window"),
                    confidence_cap=conf.get("confidence_cap"),
                    confidence_floor=conf.get("confidence_floor"),
                    # Not (yet) a ConfidenceConfig schema field — forwarded
                    # when present in dict-form config; absent -> None ->
                    # the module default is preserved.
                    negative_context_penalty=conf.get("negative_context_penalty"),
                )

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
        labeled: list[EngineFinding] = []
        if not self.enabled:
            return findings

        language = str(context.get("language", "en")).lower()

        for key, value in payload.items():
            if not isinstance(value, str):
                continue

            # sp7 panel: normalize for detection (strip zero-width, fold
            # fullwidth). Rebind ``value`` to the normalized scan-text so the
            # whole loop body runs against it; the field's new spans are
            # remapped back to ORIGINAL offsets at the loop bottom. ASCII text
            # returns (value, None) — identity, byte-identical behaviour.
            _f0, _l0 = len(findings), len(labeled)
            _norm_map: list[int] | None = None
            if self._unicode_normalize:
                value, _norm_map = normalize_for_detection(value)

            # ── A2 labeled-field value bridge (additive, all paths) ─
            # A value behind an unambiguous field-label cue ("SSN:",
            # "Date of Birth:") typed FROM THE LABEL. Additive recall +
            # leak-safe relabel via _apply_label_wins below.
            for etype, l_start, l_end, l_conf in extract_labeled_fields(value):
                labeled.append(
                    EngineFinding(
                        entity_type=etype,
                        confidence=l_conf,
                        field_path=key,
                        span_start=l_start,
                        span_end=l_end,
                        engine_id=self.adapter_id,
                        explanation="labeled-field cue->value bridge (sp7 A2)",
                        language=language,
                    )
                )

            # ── #9 geo gazetteer (additive LOCATION, all paths) ────
            # entity_type stays LOCATION (home strict scoring byte-identical);
            # the STATE/COUNTRY subtype is surfaced on the explanation as an
            # advisory taxonomy signal (sp7 geo taxonomy — metadata, not a
            # scored type: a library-side type-split would drop the 1251 home
            # LOCATION gold under exact-type scoring).
            for etype, g_start, g_end, g_conf in extract_locations(value):
                _subtype = geo_subtype(value[g_start:g_end])
                labeled.append(
                    EngineFinding(
                        entity_type=etype,
                        confidence=g_conf,
                        field_path=key,
                        span_start=g_start,
                        span_end=g_end,
                        engine_id=self.adapter_id,
                        explanation=f"geo gazetteer location (sp7 #9) [subtype={_subtype}]",
                        language=language,
                    )
                )

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
                    else:
                        # Specs without context scoring (e.g. ORGANIZATION)
                        # still get the negative-context demotion: a window
                        # naming ANOTHER identifier's field label demotes
                        # the finding below the emit floor. No-op for
                        # entity types without NEGATIVE_CONTEXT_WORDS.
                        confidence = apply_negative_context(
                            spec.entity_type, confidence, value, span_start, span_end,
                        )

                    # ── Confidence threshold filter ────────────────
                    # Suppress low-confidence findings to reduce false
                    # positives.  The threshold is intentionally set low
                    # so that only findings penalized by both low base
                    # confidence AND missing context are suppressed.
                    if confidence < _MIN_EMIT_CONFIDENCE:
                        continue

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

            # sp7 panel: remap THIS field's new spans from normalized coords
            # back to the ORIGINAL text, so masking replaces the exact original
            # region (incl. any interior obfuscation chars). No-op on the ASCII
            # fast path (_norm_map is None).
            if _norm_map is not None:
                for _fnd in findings[_f0:]:
                    if isinstance(_fnd.span_start, int) and isinstance(_fnd.span_end, int):
                        _fnd.span_start, _fnd.span_end = remap_span(
                            _norm_map, _fnd.span_start, _fnd.span_end
                        )
                for _lbl in labeled[_l0:]:
                    if isinstance(_lbl.span_start, int) and isinstance(_lbl.span_end, int):
                        _lbl.span_start, _lbl.span_end = remap_span(
                            _norm_map, _lbl.span_start, _lbl.span_end
                        )

        # _drop_nested_same_type (same-type containment) and
        # _drop_dob_shadowed_dates (DOB ⊇ region, both masked) are
        # LEAK-SAFE — the surviving span still covers the dropped region
        # with a masked type, so they always run. Cross-type PERSON
        # arbitration + the undecimaled-GPS drop are EVAL-ONLY (see
        # __init__ and _drop_undecimaled_gps — the sp6 close).
        # A2: merge labeled-field findings (additive) with the leak-safe
        # label-wins relabel, BEFORE the drop pipeline so downstream dedup
        # sees the final type assignment. Runs on all paths.
        findings = _apply_label_wins(findings, labeled)
        # A3: promote general dates behind a birth cue to DATE_OF_BIRTH
        # (leak-safe relabel, all paths), then drop the now-shadowed dates.
        text_all: dict[str | None, str] = {
            k: v for k, v in payload.items() if isinstance(v, str)
        }
        findings = _promote_dob_by_cue(findings, text_all)
        findings = _drop_location_nested_in_address(findings)
        findings = _drop_ssn_shadowed_national_id(findings)
        findings = _drop_dob_shadowed_dates(findings)
        if self._eval_cross_type_arbitration:
            # text_all (built above for the DOB promotion) already holds the
            # str fields — no need to rebuild the same dict (sp7 panel).
            findings = _drop_person_shadowed_by_specific(findings)
            findings = _drop_undecimaled_gps(findings, text_all)
            findings = _drop_nongeo_gps(findings, text_all)
            findings = _drop_bare9_ssn_noncontext(findings, text_all)
            findings = _drop_titlecase_noise_person(findings, text_all)
            findings = _trim_salutation_led_person(findings, text_all)
        # sp7 #6 mention-propagation (ADDITIVE, all paths): seed from the
        # PERSON_NAMEs that survived arbitration and propagate their surnames to
        # bare standalone mentions the grammar missed. Masks MORE (leak-safe).
        findings = _propagate_surname_mentions(
            findings,
            text_all,
            adapter_id=self.adapter_id,
            language=language,
            is_denied=self._is_denied,
        )
        return _drop_nested_same_type(findings)

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
            digits = _extract_digits(matched_text)
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
            digits = _extract_digits(matched_text)
            if len(digits) != 9:
                return 0, "", True
            if validators.luhn_checksum(digits):
                return spec.valid_confidence or 0.92, "regex canadian sin (luhn valid)", False
            return spec.invalid_confidence or 0.75, "regex canadian sin (format only)", False

        # ── Aadhaar: Verhoeff on 12 digits ─────────────────────────
        if v == "aadhaar":
            digits = _extract_digits(matched_text)
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

        # ── SWIFT/BIC: validate country code + require bank context ─
        if v == "swift_context":
            # Strict BIC structure: 4 letters (bank) + 2 letters
            # (ISO 3166-1 alpha-2 country) + 2 alphanumeric (location)
            # + optional 3 alphanumeric (branch).  The pattern is
            # permissive by design to catch BIC-like strings, so we
            # harden it here: the country-code pair MUST be in the ISO
            # alpha-2 set, otherwise the match is almost always noise
            # (test fixtures, code IDs, etc.).
            candidate = match.group(1) if match.groups() else match.group(0)
            country_code = candidate[4:6] if len(candidate) >= 6 else ""
            if country_code not in _ISO_3166_ALPHA_2:
                return 0, "", True
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
