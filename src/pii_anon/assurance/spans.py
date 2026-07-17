"""Dependency-free span-matching primitives.

Lives at the package root (not under ``assessors/``) so both ``adjudication`` and
the assessors can import it without triggering the ``assessors`` package __init__
(which would create an import cycle).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .stats import Counts

Span = tuple[str, int, int]

# Common short gold-label aliases beyond the canonical taxonomy.
_TYPE_ALIASES = frozenset({
    "email", "phone", "ssn", "person", "name", "address", "date", "date_time",
    "org", "organization", "location", "loc", "gpe", "url", "ip", "credit_card",
    "iban", "dob", "zip", "postcode", "username", "id", "misc", "other", "entity",
})
_KNOWN_TYPES: frozenset[str] | None = None


def _known_types() -> frozenset[str]:
    global _KNOWN_TYPES
    if _KNOWN_TYPES is None:
        base: set[str] = set(_TYPE_ALIASES)
        try:  # the project taxonomy (48 canonical types) — lazy, light
            from pii_anon.eval_framework.taxonomy import PII_TAXONOMY

            base |= {t.casefold() for t in PII_TAXONOMY.all_types()}
        except Exception:  # noqa: BLE001 - allowlist degrades to aliases only
            pass
        _KNOWN_TYPES = frozenset(base)
    return _KNOWN_TYPES


def canonical_type(etype: str) -> str:
    """Emit a RECOGNIZED entity type verbatim; bucket any unrecognized type — which could
    carry a PII name in a malformed/hostile dataset — to a stable, raw-free ``other:<hash>``.

    This is fail-closed for the type-as-PII vector: a name-shaped type (e.g.
    ``Hernandez_Maiden_Name``) is not in the allowlist, so its raw string never reaches
    an artifact; distinct unknown types still map to distinct buckets (breakdown preserved)."""
    if etype.casefold() in _known_types():
        return etype
    # errors="replace" mirrors every other encode() in this package (dataset._field,
    # provenance.output_hash, pii_egress): a lone UTF-16 surrogate in a hostile/scraped
    # type label must bucket to a stable other:<hash>, never crash the whole run at the
    # UTF-8 boundary (UnicodeEncodeError) — that was a full-report DoV from one bad span.
    return f"other:{hashlib.sha1(etype.encode('utf-8', 'replace')).hexdigest()[:8]}"


# Reconcile common gold-label vocabularies (short NER tags) with pii-anon's native types,
# so an exact-boundary detection under a synonym is a MATCH, not a phantom-0 type artifact.
# Conservative: only well-known synonyms; an unknown type stays its own canonical (casefold).
_TYPE_SYNONYMS = {
    "email": "email", "email_address": "email", "e-mail": "email", "emailaddress": "email", "mail": "email",
    "phone": "phone", "phone_number": "phone", "telephone": "phone", "tel": "phone", "phonenumber": "phone",
    "person": "person", "person_name": "person", "name": "person", "fullname": "person", "per": "person", "pers": "person",
    "ssn": "ssn", "us_ssn": "ssn", "social_security_number": "ssn",
    "address": "address", "street_address": "address", "postal_address": "address", "addr": "address",
    "location": "location", "loc": "location", "gpe": "location",
    "credit_card": "creditcard", "credit_card_number": "creditcard", "cc": "creditcard", "card": "creditcard",
    "date": "date", "date_time": "date", "datetime": "date",
    "date_of_birth": "dob", "dob": "dob", "birthdate": "dob",
    "ip": "ip", "ip_address": "ip",
    "org": "org", "organization": "org", "organisation": "org", "company": "org",
    "iban": "iban", "bank_account_number": "bankaccount", "bank_account": "bankaccount", "account": "bankaccount",
}


def match_type(etype: str) -> str:
    """Canonical key for matching — maps a known synonym to a shared form, else casefold."""
    t = etype.casefold()
    return _TYPE_SYNONYMS.get(t, t)


def _norm(spans: Sequence[Span], *, type_sensitive: bool) -> set[tuple[str, int, int]]:
    return {
        ((match_type(s[0]) if type_sensitive else ""), int(s[1]), int(s[2]))
        for s in spans
    }


def match_counts(pred: Sequence[Span], gold: Sequence[Span], *, type_sensitive: bool = True) -> Counts:
    """Strict confusion counts: a prediction matches gold iff exact boundaries
    (and, when ``type_sensitive``, casefolded entity type). De-duplicates spans."""
    p = _norm(pred, type_sensitive=type_sensitive)
    g = _norm(gold, type_sensitive=type_sensitive)
    return Counts(tp=len(p & g), fp=len(p - g), fn=len(g - p))


def per_type_counts(pred: Sequence[Span], gold: Sequence[Span]) -> dict[str, Counts]:
    """Per-entity-type strict confusion counts (type-sensitive).

    Single-pass bucketing by casefolded type — O(N), not O(types × spans) — so a
    pipeline that assigns a distinct type per entity cannot drive the detection hot
    loop quadratic (a DoV hang)."""
    from collections import defaultdict

    bp: dict[str, list[Span]] = defaultdict(list)
    bg: dict[str, list[Span]] = defaultdict(list)
    for s in pred:
        bp[match_type(s[0])].append(s)  # group by reconciled type so synonyms align
    for s in gold:
        bg[match_type(s[0])].append(s)
    return {
        t: match_counts(bp.get(t, []), bg.get(t, []), type_sensitive=True)
        for t in set(bp) | set(bg)
    }
