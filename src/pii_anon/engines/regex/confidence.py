"""Context-aware confidence scoring for regex PII detections.

Inspired by Microsoft Presidio's ``LemmaContextAwareEnhancer``, this module
adjusts detection confidence based on surrounding text.  When context keywords
appear near a matched span (e.g., "social security" near a 9-digit number),
confidence is boosted.  When context is absent for high false-positive entity
types (US_SSN, PERSON_NAME), confidence is penalized.

Algorithm
---------
1. Extract a window of *±CONTEXT_WINDOW* characters around the matched span.
2. Tokenize the window into lowercase words.
3. Intersect tokens with the entity-type's keyword set.
4. If intersection is non-empty → boost by *CONTEXT_BOOST* (capped at 0.99).
5. If empty **and** the entity type is in *HIGH_FP_TYPES* → penalize by
   *CONTEXT_PENALTY* (floored at 0.50).
6. Otherwise → return the base confidence unchanged.

This approach avoids NLP dependencies (no lemmatization needed) while still
providing meaningful confidence differentiation.
"""

from __future__ import annotations

import re

# ── Context keyword sets per entity type ───────────────────────────────────
# Each set contains lowercase tokens that commonly appear near genuine PII
# of the given type.

CONTEXT_WORDS: dict[str, set[str]] = {
    "US_SSN": {
        "social", "security", "ssn", "tax", "tin", "taxpayer",
        "identification", "ss#",
    },
    "CREDIT_CARD": {
        "credit", "card", "visa", "mastercard", "amex", "discover",
        "payment", "debit", "charge", "cc", "cardnumber",
    },
    "PHONE_NUMBER": {
        "call", "phone", "tel", "telephone", "mobile", "fax",
        "cell", "contact", "reach", "cellphone",
    },
    "EMAIL_ADDRESS": {
        "email", "mail", "e-mail", "contact", "send", "address",
    },
    "IP_ADDRESS": {
        "ip", "host", "server", "network", "ipv4", "ipv6",
    },
    "IBAN": {
        "iban", "bank", "transfer", "wire", "international", "bic",
    },
    "ROUTING_NUMBER": {
        "routing", "aba", "transit", "bank", "sort",
    },
    "PERSON_NAME": {
        "name", "person", "patient", "employee", "client",
        "member", "user", "resident", "beneficiary", "customer",
    },
    "AGE": {
        "age", "aged", "years", "old", "born", "birthday",
    },
    "MEDICAL_LICENSE": {
        "npi", "dea", "license", "provider", "physician", "prescriber",
    },
    # --- Entity types added for broader context coverage ---
    "DATE_OF_BIRTH": {
        "born", "birth", "dob", "birthday", "birthdate", "date",
        "nacimiento", "naissance",
    },
    "BANK_ACCOUNT": {
        "account", "bank", "checking", "savings", "deposit",
        "acct", "cuenta", "compte",
    },
    "DRIVERS_LICENSE": {
        "driver", "license", "licence", "dl", "driving", "permit",
        "licencia", "permis",
    },
    "PASSPORT": {
        "passport", "pasaporte", "passeport", "travel", "document",
        "visa", "immigration",
    },
    "NATIONAL_ID": {
        "national", "identity", "id", "identification", "cedula",
        "dni", "citizen",
    },
    "VIN": {
        "vin", "vehicle", "car", "automobile", "chassis",
        "identification", "registration",
    },
    "MAC_ADDRESS": {
        "mac", "hardware", "device", "interface", "ethernet",
        "wifi", "adapter", "network",
    },
    "ORGANIZATION": {
        "company", "corporation", "inc", "ltd", "llc", "org",
        "enterprise", "firm", "business", "employer",
    },
    "LOCATION": {
        "location", "city", "state", "country", "region",
        "address", "place", "area", "district", "province",
    },
    "ADDRESS": {
        "address", "street", "avenue", "road", "boulevard",
        "suite", "apt", "apartment", "residence", "domicilio",
    },
}

# Entity types where absence of context should *penalize* confidence.
# Expanded to match the paper's identification of high-FP categories:
# PERSON_NAME, LOCATION, ORGANIZATION, and ADDRESS all produce
# frequent false positives when context is absent (Section 6.3).
HIGH_FP_TYPES: frozenset[str] = frozenset({
    "US_SSN",
    "PERSON_NAME",
    "LOCATION",
    "ORGANIZATION",
    "ADDRESS",
})

# Tuning constants.
CONTEXT_BOOST: float = 0.08
CONTEXT_PENALTY: float = 0.05
CONTEXT_WINDOW: int = 50

# Pre-compiled word tokenizer.
_WORD_RE = re.compile(r"\b\w+\b")


def extract_context(text: str, start: int, end: int) -> str:
    """Return lowercased text surrounding the matched span.

    Parameters
    ----------
    text:
        Full input string.
    start:
        Start offset of the matched span.
    end:
        End offset of the matched span.

    Returns
    -------
    str
        Lowercase substring of *text* from ``max(0, start - CONTEXT_WINDOW)``
        to ``min(len(text), end + CONTEXT_WINDOW)``.
    """
    ctx_start = max(0, start - CONTEXT_WINDOW)
    ctx_end = min(len(text), end + CONTEXT_WINDOW)
    return text[ctx_start:ctx_end].lower()


def has_context_words(entity_type: str, context_text: str) -> bool:
    """Check if any context keywords for *entity_type* appear in *context_text*.

    Uses set intersection of tokenized context against the keyword set.
    Returns *False* if the entity type has no configured keywords.
    """
    words = CONTEXT_WORDS.get(entity_type)
    if not words:
        return False
    # Split is faster than regex tokenization for plain ASCII context text.
    tokens = context_text.split()
    return any(t in words for t in tokens)


def adjust_confidence(
    entity_type: str,
    base_confidence: float,
    text: str,
    start: int,
    end: int,
) -> float:
    """Boost or penalize *base_confidence* based on surrounding context.

    Parameters
    ----------
    entity_type:
        The PII entity type (e.g., ``"US_SSN"``).
    base_confidence:
        Raw confidence from pattern matching or validation.
    text:
        Full input text containing the match.
    start:
        Start offset of the matched span.
    end:
        End offset of the matched span.

    Returns
    -------
    float
        Adjusted confidence in [0.50, 0.99].
    """
    ctx = extract_context(text, start, end)
    if has_context_words(entity_type, ctx):
        return min(0.99, base_confidence + CONTEXT_BOOST)
    if entity_type in HIGH_FP_TYPES:
        return max(0.50, base_confidence - CONTEXT_PENALTY)
    return base_confidence
