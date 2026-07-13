"""sp7 panel #2 — value-consistent (coreference) masking for the production path.

The primary transform masks each DETECTED span. A distinctive value (a name,
account, id) that was detected at one mention but appears verbatim AGAIN
elsewhere — a mention the detector missed, or the same value in another field —
survives in the output and is trivially reconstructable. This pass-2 sweep
redacts every remaining VERBATIM occurrence of a detected sticky-type value with
the SAME replacement that value already received (token consistency).

Design (grounded in the orchestrator):
  - Operate on the ORIGINAL field text with the pass-1 covered spans, NOT the
    transformed text: an HMAC token can itself contain a run of digits/letters
    equal to a short surface, so string-replacing in output space could corrupt
    an inserted token.
  - ADDITIVE on the masking path (redacts MORE with a replacement already
    deemed safe for that value; never narrows a pattern, drops a finding, or
    touches detection) => leak-safe on both paths.
  - Bounded over-mask: only sticky, distinctive types (mirrors the assurance
    report masker) and only surfaces of length >= MIN_LEN.
"""
from __future__ import annotations

#: Sticky, distinctive types where redacting a repeated verbatim value is safe.
#: Mirrors assurance/reconstruction_resistance_cli._COREF_TYPES.
VALUE_CONSISTENT_TYPES = frozenset({
    "PERSON_NAME", "ORGANIZATION", "BANK_ACCOUNT", "PASSPORT", "CUSTOMER_ID",
    "EMPLOYEE_ID", "MEDICAL_RECORD_NUMBER", "IBAN", "CREDIT_CARD", "NATIONAL_ID",
    "US_SSN", "DRIVERS_LICENSE", "EMAIL_ADDRESS",
})
VALUE_CONSISTENT_MIN_LEN = 4


def build_surfaces(
    link_audit: list[dict[str, object]],
) -> list[tuple[str, str, str, str]]:
    """Build the payload-wide ``(surface, replacement, entity_type, cluster_id)``
    list from the pass-1 audit — sticky types only, surface length >= MIN_LEN,
    deduped, LONGEST-surface-first (so a longer value masks before a substring).

    ``cluster_id`` is carried so a residual occurrence is attributed to the SAME
    identity cluster as the value it mirrors (a repeat of "Jack" IS Jack)."""
    seen: set[str] = set()
    surfaces: list[tuple[str, str, str, str]] = []
    for entry in link_audit:
        etype = str(entry.get("entity_type", ""))
        if etype not in VALUE_CONSISTENT_TYPES:
            continue
        replacement = str(entry.get("replacement", ""))
        cluster_id = str(entry.get("cluster_id", ""))
        for key in ("mention_text", "canonical_text"):
            surface = entry.get(key)
            if (
                isinstance(surface, str)
                and len(surface) >= VALUE_CONSISTENT_MIN_LEN
                and surface not in seen
            ):
                seen.add(surface)
                surfaces.append((surface, replacement, etype, cluster_id))
    surfaces.sort(key=lambda s: -len(s[0]))
    return surfaces


def sweep_residual_occurrences(
    text: str,
    surfaces: list[tuple[str, str, str, str]],
    covered: list[tuple[int, int]],
) -> list[tuple[int, int, str, str, str]]:
    """Find residual verbatim occurrences of each surface in ``text``, not
    overlapping ``covered`` (pass-1 spans, original offsets) or each other.

    Returns accepted ``(start, end, replacement, entity_type, cluster_id)`` in
    original offsets, ready to merge with the pass-1 replacements for
    re-assembly (cluster_id attributes the redaction to its identity cluster)."""
    taken: list[tuple[int, int]] = list(covered)
    out: list[tuple[int, int, str, str, str]] = []
    for surface, replacement, etype, cluster_id in surfaces:
        i = text.find(surface)
        while i != -1:
            j = i + len(surface)
            if any(i < ce and cs < j for cs, ce in taken):
                # overlaps a covered/taken span; a covered span may end inside
                # this hit, so the next occurrence can start before j.
                i = text.find(surface, i + 1)
                continue
            taken.append((i, j))
            out.append((i, j, replacement, etype, cluster_id))
            i = text.find(surface, j)
    return out
