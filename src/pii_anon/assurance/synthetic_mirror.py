"""Synthetic-mirror corpus generator.

Produces a no-real-PII corpus that mirrors the dataset's structural distribution
(record count, language, approximate length, per-record entity-type counts) so an
external auditor can re-run the *methodology* end-to-end on non-real data, even
though the headline numbers stay data-bound (AX-001).

Every value is generated, never copied from the real corpus. Reserved/non-routable
ranges are used (``example`` domains, ``555-01xx`` phones, ``900-xx-xxxx`` SSN-shaped
strings that are not valid SSNs) so the mirror itself contains no real PII.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence

from .dataset import AssuranceDataset, AssuranceRecord
from .spans import canonical_type

# The structural signature this generator stamps on every mirror record: a
# fingerprint-derived record_id and the fixed "Synthetic record <i> (<lang>)." head.
# Used by is_synthetic_mirror() to recognize a mirror fed back through the runner.
_MIRROR_ID_RE = re.compile(r"^mirror-[0-9a-f]{8}-\d+$")
_MIRROR_HEAD_RE = re.compile(r"^Synthetic record \d+ \(")

# ISO 639-1 two-letter language codes — the allowlist for the mirror's language tag.
_ISO_639_1 = frozenset((
    "aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr "
    "cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv "
    "ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km "
    "kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na "
    "nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc "
    "sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt "
    "tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu"
).split())


def _safe_type(etype: str) -> str:
    """Canonicalize the type — a recognized type verbatim, any unrecognized (possibly
    PII-bearing) type to a stable ``other:<hash>`` so the mirror never copies a raw name."""
    return canonical_type(etype or "ENTITY")


def _safe_language(language: str) -> str:
    """Canonicalize to a known ISO-639-1 BASE code (dropping any region/variant subtag,
    which could carry PII like ``ng-lee``); an unrecognized tag maps to ``und``. Never
    copies the raw language value into the mirror."""
    base = (language or "").split("-", 1)[0].casefold()
    return base if base in _ISO_639_1 else "und"


def _fake_value(entity_type: str, rng: random.Random, idx: int) -> str:
    t = entity_type.upper()
    if "EMAIL" in t:
        return f"user{idx}{rng.randint(0, 999)}@example.com"
    if "PHONE" in t:
        return f"555-01{rng.randint(0, 99):02d}-{rng.randint(1000, 9999)}"
    if "SSN" in t:
        return f"900-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    if "PERSON" in t or "NAME" in t:
        return f"Synthetic Person {idx}{rng.randint(0, 99)}"
    if "ADDRESS" in t:
        return f"{rng.randint(1, 999)} Example Ave, Testville"
    if "CREDIT" in t or "CARD" in t:
        return f"4000-0000-0000-{rng.randint(1000, 9999)}"
    return f"<{t}-{idx}>"


_FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do".split()


def synthesize_mirror(dataset: AssuranceDataset, *, seed: int) -> list[AssuranceRecord]:
    """Build a structural mirror of ``dataset`` with synthetic PII + matching labels."""
    from .dataset import GoldSpan

    # Salt the RNG with the dataset fingerprint so distinct datasets get distinct synthetic
    # tokens. NOTE: this only REDUCES, it does NOT eliminate, the round-trip collision — the
    # reserved token namespaces are intrinsically small (an email's random part is 1 of 1000;
    # a card is 1 of 9000), so two independent fingerprint-salted draws still collide with
    # non-trivial probability over a realistic record count. The actual guarantee against a
    # mirror-of-a-mirror tripping the egress scan is in the runner: it does NOT regenerate a
    # mirror when the input is already one (see is_synthetic_mirror + runner._run_inner).
    rng = random.Random(f"{seed}:{dataset.fingerprint}")
    # fingerprint-derived id prefix: a mirror gets DIFFERENT record_ids per dataset, and the
    # fixed shape is the structural signature is_synthetic_mirror() recognizes.
    fp8 = hashlib.sha1(dataset.fingerprint.encode("utf-8")).hexdigest()[:8]
    mirror: list[AssuranceRecord] = []
    for i, rec in enumerate(dataset.records):
        # entity types to reproduce: gold labels if present, else a default single EMAIL.
        # Sanitize the type label (it COULD carry PII in a hostile dataset) — keep only a
        # safe identifier-shaped token, else a generic "ENTITY".
        types = [_safe_type(s.entity_type) for s in rec.labels] if rec.labels else ["EMAIL"]
        language = _safe_language(rec.language)
        head = f"Synthetic record {i} ({language})."
        text = head
        labels: list[GoldSpan] = []
        for j, etype in enumerate(types):
            val = _fake_value(etype, rng, i * 100 + j)
            prefix = f" {etype.lower()}: "
            start = len(text) + len(prefix)
            text += prefix + val
            labels.append(GoldSpan(etype, start, start + len(val)))
        # pad toward the original length with deterministic filler (appended, so offsets hold)
        while len(text) < len(rec.text):
            text += " " + rng.choice(_FILLER)
        # record_id is index-based and group is dropped — NEVER copy raw field values into
        # the mirror (those bypassed the egress gate before this fix).
        mirror.append(
            AssuranceRecord(
                record_id=f"mirror-{fp8}-{i}",
                text=text,
                language=language,
                labels=tuple(labels),
                group=None,
            )
        )
    return mirror


def is_synthetic_mirror(dataset: AssuranceDataset) -> bool:
    """Was ``dataset`` produced by :func:`synthesize_mirror` (i.e. fed back through the
    runner)? True only when EVERY record carries BOTH the fingerprint-stamped record_id
    (``mirror-<8hex>-<i>``) AND the fixed ``Synthetic record <i> (<lang>).`` head.

    Deliberately STRICT (all-records, dual-signature): a real dataset essentially cannot
    match it, and a near-miss (partial overlap, a real corpus that merely starts with
    "Synthetic record") returns False so the normal mirror-generation + full egress path
    is taken. This is a generation-skip signal ONLY — it never relaxes the egress gate, so
    a false negative just regenerates a mirror (the safe default) and a false positive only
    skips an artifact whose input is, by construction, already non-real PII."""
    records = dataset.records
    if not records:
        return False
    return all(
        _MIRROR_ID_RE.match(r.record_id) is not None and _MIRROR_HEAD_RE.match(r.text) is not None
        for r in records
    )


def mirror_to_rows(records: Sequence[AssuranceRecord]) -> list[dict[str, object]]:
    """Serialize synthetic-mirror records to JSONL-loadable dict rows (safe: no real PII)."""
    rows: list[dict[str, object]] = []
    for r in records:
        rows.append({
            "id": r.record_id,
            "text": r.text,
            "language": r.language,
            "labels": [[s.entity_type, s.start, s.end] for s in (r.labels or ())],
        })
    return rows
