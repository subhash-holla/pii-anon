"""The PII-egress gate — a SEPARATE, fail-closed, byte-level safety gate.

Distinct from the numeric no-fabrication gate (which only proves a *number* is
well-formed and cannot tell whether a *string* contains PII). This gate runs on
the FINAL serialized bytes of every artifact, as the last step before any write.

Two FP-resistant layers (the blanket character-k-gram scan was removed — it
false-positived on common English shared between the dataset and the report's own
boilerplate):

1. **Structured-token containment** (recall-independent, FP-free): extract the
   PII-SHAPED tokens from every raw string (emails; digit-bearing tokens like
   phones / SSNs / card / account / IDs) and flag if any appears verbatim in the
   output. Pure-alphabetic common words are never extracted, so boilerplate
   overlap cannot trip it; structured PII is the high-risk, high-specificity case.

2. **Detector-surface containment** (covers unstructured PII like names): run the
   detector on each raw string AND on the output; a finding is raised only when an
   actually-DETECTED, *distinctive* PII surface from a raw record appears as a
   DELIMITED TOKEN in the output. Two specificity guards keep this FP-free:

   * **Distinctiveness floor** (:func:`_is_distinctive_surface`): a detector may flag a
     low-entropy surface like a 4-digit invoice number, but a 4-digit string's digits
     appear by chance inside virtually every sha256 digest and numeric metric — flagging
     it makes the gate fire on every realistic dataset and stops being a safety control.
     So a detector surface is treated as a containment risk only when it is specific
     enough that a verbatim appearance is real evidence of a copy: an email, a multi-word
     surface (a full name), a ≥7-digit identifier, or a ≥8-char non-pure-digit token. This
     mirrors Layer 1's own floor (which already excludes <5-char / pure-4-digit tokens) and
     the leakage assessor's ≥7-digit survival rule — the two layers are now consistent.
   * **Token-boundary containment** (:func:`_token_contained`): the surface must appear
     bounded by non-alphanumeric characters (or string edges), so a value embedded inside a
     longer alnum run (e.g. a digit-run inside a hash) is not a match. A genuine verbatim
     copy of the surface preserves its token boundaries, so real-leak recall is unaffected.

Residual limitation (documented honestly): PII that is unstructured AND missed by the
detector — OR a bare, non-distinctive value such as a lone ≤6-digit number copied verbatim
on its own — is not caught by this layer. This is mitigated by the design's primary rule
that NO raw text is emitted into any artifact at all, and by Layer 1 for structured tokens.

CARDINAL: the gate must not itself leak — findings carry record ids / counts ONLY,
never the offending substring.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass


def _rid(record_id: str) -> str:
    """A short, stable, PII-free handle for a record_id (which may itself carry PII) —
    the gate must never self-leak, even in a finding message."""
    return hashlib.sha1(record_id.encode("utf-8", "replace")).hexdigest()[:8]


class PiiEgressError(RuntimeError):
    """Raised when an artifact fails the PII-egress gate. Message is PII-free."""


@dataclass(frozen=True)
class EgressFinding:
    kind: str        # "containment" | "detector" | "detector_error"
    detail: str      # PII-FREE: record id / counts only


Detector = Callable[[str], Iterable[tuple[str, int, int]]]

# Linear-time scanning only (whitespace split + a bounded-char-class digit-run regex —
# no `+@` backtracking), so the FULL record is scanned with no length cap: truncating
# would silently drop coverage of structured PII past the cutoff (the unsafe direction).
_DIGIT_RUN = re.compile(r"\d[\d \-().]{3,}\d")  # phone/SSN runs incl. internal separators
_STRIP = ".,;:!?()[]{}\"'<>"

# A dotted-numeric software version (semver / CalVer + optional pre-release suffix). Used ONLY
# by scrub_label to keep a legitimate pipeline/dataset version readable instead of redacting it
# as a digit-bearing identifier. Tight by construction: requires DOTS (so hyphen/space-separated
# PII shapes — SSN 123-45-6789, phone 555-867-5309, card 4111 1111 1111 — never match) and caps
# each segment at 4 digits and the whole string at 8 total digits (so a long bare ID cannot be
# smuggled as one segment, and a 9+-digit number chunked across dots is still redacted). This is
# presentation-only; the egress gate (scan_output) is unchanged and remains the real guard.
_VERSION_RE = re.compile(
    r"^v?\d{1,4}(?:\.\d{1,4}){1,4}(?:[.\-_]?(?:rc|alpha|beta|a|b|dev|post|pre)\d{0,4})?$",
    re.IGNORECASE,
)
_SUFFIX_STRIP = re.compile(r"[.\-_]?(?:rc|alpha|beta|dev|post|pre|a|b)\d*$", re.IGNORECASE)

# Library-controlled safe-default label constants. They are not user input and are
# definitionally PII-free, so they are exempt from label scrubbing — this keeps the
# fallback dataset/pipeline names readable even though a case-insensitive NER may
# false-flag the common word "dataset"/"pipeline" as a USERNAME surface.
_SAFE_DEFAULT_LABELS = frozenset({"user-dataset", "user-pipeline"})

# A run of >=3 alphabetic tokens (each >=2 letters) joined by single '_' or '-'
# separators — the shape of a separator-joined name like 'john-smith-jones' or
# 'patient_jane_doe' that a case-insensitive NER misses when it is lowercase. See
# scrub_label for why the floor is 3 (2-token runs are the safe DEFAULT labels).
_NAME_RUN_RE = re.compile(r"[A-Za-z]{2,}(?:[_-][A-Za-z]{2,}){2,}")


def _is_version_token(tok: str, detector: Detector | None = None) -> bool:
    """A dotted-numeric version string (not PII) — exempt from label redaction ONLY.

    Guards the calendar-date collision (rounds 20–21). Two layers, because a dotted-numeric
    version and a dotted-numeric date are structurally ambiguous:

    1. A 4-digit-LEADING dotted token (``YYYY.MM[.DD]``) is year-first-date-ambiguous and is never
       exempt. Classic semver has a 1–3-digit leading segment (``2.3.1``, ``10.4.2``, ``v24.04``).
    2. DETECTOR DEFERRAL (the date oracle): rather than hand-enumerate every date ordering, the
       exemption defers to the PII detector — but a ``v`` prefix / pre-release suffix BLINDS the
       detector (``detect('v90.05.21') == []`` while ``detect('90.05.21')`` tags a DATE), so we
       strip that decoration to the bare dotted core first and re-consult. If the detector tags
       the bare core as anything, the token is NOT a version (closes ``v90.05.21`` / ``90.05.21rc1``
       and any other ordering the detector knows). Real version cores (``2.3.1``, ``24.04``) are
       detector-clean, so they stay readable. Fail-closed: a detector error ⇒ not exempt.

    Without a detector only layer 1 applies (best-effort); the runner always supplies one."""
    if not _VERSION_RE.match(tok) or sum(c.isdigit() for c in tok) > 8:
        return False
    core = tok[1:] if tok[:1] in ("v", "V") else tok
    if len(core.split(".", 1)[0]) >= 4:
        return False  # year-first (YYYY.*) date-ambiguous
    if detector is not None:
        bare = _SUFFIX_STRIP.sub("", core)  # strip decoration so the detector sees the bare form
        try:
            if list(detector(bare)):
                return False  # detector tags the bare dotted core (e.g. a date) -> not a version
        except Exception:  # noqa: BLE001 - a detector failure must not crash label scrubbing
            return False  # fail closed: cannot verify -> do not exempt
    return True


def _structured_tokens(text: str, *, min_len: int) -> set[str]:
    """PII-shaped tokens only — emails, digit-bearing identifiers, phone-like runs.
    Pure-alphabetic words (common English) are intentionally excluded. Linear time over
    the FULL string (no truncation)."""
    scan = text
    tokens: set[str] = set()
    for raw_tok in scan.split():  # whitespace split is linear and avoids regex backtracking
        tok = raw_tok.strip(_STRIP)
        if len(tok) < 5:
            continue
        if "@" in tok:  # email-shaped: any non-empty local + domain (incl. TLD-less intranet)
            local, _, domain = tok.partition("@")
            if local and domain:
                tokens.add(tok)
        elif sum(c.isdigit() for c in tok) >= 3:  # ids / cards / account numbers
            tokens.add(tok)
    for m in _DIGIT_RUN.finditer(scan):  # bounded char class -> linear; catches spaced phones
        tok = m.group().strip()
        if sum(c.isdigit() for c in tok) >= 5:
            tokens.add(tok)
    return {t for t in tokens if len(t) >= min(min_len, 5)}


def _is_distinctive_surface(surface: str) -> bool:
    """Is a detector-found surface specific enough that a verbatim appearance in an artifact
    is real evidence of a copy (not a coincidental substring of a hash / numeric metric)?

    Excludes low-entropy surfaces (a bare ≤6-digit number, a single short word): a 4-digit
    invoice number's digits appear by chance in nearly every sha256 digest, so flagging it
    would fire the gate on every realistic dataset. Mirrors Layer 1's <5-char / pure-4-digit
    exclusion and the leakage assessor's ≥7-digit rule, so the two layers stay consistent."""
    s = surface.strip()
    if "@" in s:  # email-shaped
        return True
    if any(ch.isspace() for ch in s):  # multi-word (e.g. a full name)
        return True
    digits = sum(ch.isdigit() for ch in s)
    if digits >= 7:  # long numeric identifier (SSN / card / account / phone)
        return True
    # a long, non-pure-digit token (a long single-word name / handle); a pure-digit run of
    # this length is handled by the ≥7-digit branch above.
    return len(s) >= 8 and digits < len(s)


_ALNUM = re.compile(r"[A-Za-z0-9]")


def _token_contained(output: str, surface: str) -> bool:
    """``surface`` appears in ``output`` delimited by non-alphanumeric boundaries (or string
    edges) — NOT embedded inside a longer alnum run (e.g. a digit-run inside a hash). A real
    verbatim copy preserves its token boundaries, so this strips coincidental-substring FPs
    without reducing real-leak recall."""
    if not surface:
        return False
    n = len(surface)
    start = 0
    while True:
        idx = output.find(surface, start)
        if idx < 0:
            return False
        before = output[idx - 1] if idx > 0 else ""
        after = output[idx + n] if idx + n < len(output) else ""
        if not _ALNUM.match(before or " ") and not _ALNUM.match(after or " "):
            return True
        start = idx + 1


def scrub_label(value: str, detector: Detector | None = None) -> str:
    """Redact PII-shaped tokens from a free-form LABEL (dataset name, pipeline name /
    version) before it is emitted into an artifact. A filename stem like
    ``patient_jane_ssn_123456789`` carries structured PII; this replaces such tokens
    with ``[redacted]`` so a config/label string can never become an egress vector.

    Three passes, each strictly redacting (never un-redacting):

    1. **Structured tokens** — emails / digit-bearing identifiers (a dotted-numeric
       *version* is exempted so it stays readable).
    2. **Detector surfaces** (when a detector is supplied) — name-shaped PII the
       structured pass misses; probes a separator-normalized view (``_``/``-`` -> spaces)
       so NER sees ``Jane_Doe`` / ``Jane-Doe`` as a name, and redacts every separator
       variant.
    3. **Separator-joined name runs** (:data:`_NAME_RUN_RE`, detector-independent) — the
       containment backstop for the residual gap that the NER is *case-sensitive*: it does
       not flag a lowercase ``john smith``, so a label like ``dlp-for-john-smith`` slipped
       through passes 1–2 verbatim. This pass redacts any run of >=3 alphabetic tokens
       joined by ``_``/``-``. The floor is 3 by construction — the library's safe DEFAULT
       labels (``user-dataset`` / ``user-pipeline``) are 2-token runs that must stay
       readable, so a 2-token floor is impossible. A lowercase name and a lowercase phrase
       are not distinguishable, so this over-redacts a long descriptive label
       (``customer-data-export`` -> ``[redacted]``) — the safe direction for a gate.

    Labels are user-controlled metadata, so this is best-effort. **Residual limitation
    (documented honestly):** a *2-token* lowercase name (``john-smith``) or one trailing a
    non-alpha token (``x12-john-smith``) is below the 3-token floor and survives; the
    primary guard remains the design rule that NO raw record text is emitted, plus the
    safe default names. The library-controlled defaults are exempted outright."""
    # The safe DEFAULT labels are library constants, not user input, and definitionally
    # PII-free — return them verbatim so a case-insensitive NER's false "dataset"/"pipeline"
    # = USERNAME flag cannot mangle the readable fallback name.
    if value in _SAFE_DEFAULT_LABELS:
        return value
    # drop lone surrogate code points first — they can't be UTF-8 encoded and would crash
    # the artifact write boundary if they reached an output via a label.
    out = "".join(c for c in value if not 0xD800 <= ord(c) <= 0xDFFF)
    for tok in sorted(_structured_tokens(out, min_len=5), key=len, reverse=True):
        if _is_version_token(tok, detector):  # a dotted-numeric version is not PII; keep it readable
            continue
        out = out.replace(tok, "[redacted]")
    if detector is not None:
        # probe a separator-normalized view (underscores AND hyphens -> spaces) so NER sees
        # 'Olafur_Bjarnason' / 'Olafur-Bjarnason' as a name; redact every separator variant.
        probe = re.sub(r"[_\-]", " ", out)
        surfaces = sorted(
            {s for s in _detected_surfaces(detector, probe) if len(s.strip()) >= 3},
            key=len, reverse=True,
        )
        for surf in surfaces:
            for variant in (surf, surf.replace(" ", "_"), surf.replace(" ", "-")):
                out = out.replace(variant, "[redacted]")
    # Detector-independent backstop for lowercase separator-joined names (pass 3).
    out = _NAME_RUN_RE.sub("[redacted]", out)
    return out


def _detected_surfaces(detector: Detector, text: str) -> list[str]:
    try:
        hits = list(detector(text))
    except Exception:  # noqa: BLE001 - a detector failure on one string is not fatal here
        return []
    out: list[str] = []
    for item in hits:
        try:
            start, end = int(item[1]), int(item[2])
            surface = text[start:end]
        except Exception:  # noqa: BLE001
            continue
        if len(surface.strip()) >= 3:
            out.append(surface)
    return out


def scan_output(
    output: str,
    raw_corpus: Iterable[tuple[str, str]],
    *,
    min_k: int = 8,
    detector: Detector | None = None,
    containment: bool = True,
) -> list[EgressFinding]:
    """Scan one artifact's text for PII egress. Returns findings (empty == clean)."""
    corpus = [(rid, t) for rid, t in raw_corpus if t]
    findings: list[EgressFinding] = []

    if containment:
        # Layer 1 — structured-token containment.
        for record_id, text in corpus:
            if any(tok in output for tok in _structured_tokens(text, min_len=min_k)):
                findings.append(
                    EgressFinding("containment", f"record {_rid(record_id)}: structured raw token present in output")
                )

        # Layer 2 (raw side) — distinctive detector-found PII surfaces from raw records,
        # matched as delimited tokens (the two specificity guards keep this FP-free).
        if detector is not None:
            for record_id, text in corpus:
                surfaces = [s for s in _detected_surfaces(detector, text) if _is_distinctive_surface(s)]
                if any(_token_contained(output, s) for s in surfaces):
                    findings.append(
                        EgressFinding("detector", f"record {_rid(record_id)}: detected PII surface present in output")
                    )

    # Layer 2 (output side) — PII detected IN the output that traces to the corpus.
    if detector is not None:
        try:
            out_hits = list(detector(output))
        except Exception as exc:  # noqa: BLE001 - cannot verify => fail closed
            findings.append(EgressFinding("detector_error", f"detector raised {type(exc).__name__}; cannot verify output"))
        else:
            raw_texts = [t for _cid, t in corpus]
            for item in out_hits:
                try:
                    start, end = int(item[1]), int(item[2])
                    surface = output[start:end]
                except Exception:  # noqa: BLE001
                    continue
                if (
                    len(surface.strip()) >= 3
                    and _is_distinctive_surface(surface)
                    and any(surface in t for t in raw_texts)
                ):
                    findings.append(
                        EgressFinding("detector", "detector-confirmed input-derived PII present in output")
                    )
                    break
    return findings


def assert_safe(
    output: str,
    raw_corpus: Iterable[tuple[str, str]],
    *,
    artifact: str = "artifact",
    min_k: int = 8,
    detector: Detector | None = None,
    containment: bool = True,
) -> None:
    """Fail CLOSED: raise :class:`PiiEgressError` (PII-free message) on any finding."""
    findings = scan_output(
        output, list(raw_corpus), min_k=min_k, detector=detector, containment=containment
    )
    if findings:
        summary = "; ".join(f"[{f.kind}] {f.detail}" for f in findings[:10])
        extra = "" if len(findings) <= 10 else f" (+{len(findings) - 10} more)"
        raise PiiEgressError(
            f"PII-egress gate FAILED for {artifact}: {len(findings)} finding(s): {summary}{extra}"
        )
