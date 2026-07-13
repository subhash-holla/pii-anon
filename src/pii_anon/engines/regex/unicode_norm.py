"""Unicode-normalization pre-pass for detection (sp7 panel: evasion hardening).

Zero-width chars (U+200B ZWSP / U+200C ZWNJ / U+200D ZWJ / U+FEFF BOM /
soft-hyphen) and fullwidth / compatibility forms let PII EVADE the regex
masking path: ``SSN: 123<ZWSP>-45-6789`` masks only ``45-6789``; a fullwidth
phone matches nothing. This normalizes text for DETECTION ONLY — the ORIGINAL
text is never modified, and spans are remapped back to original-text offsets
via the returned index map, so masking still replaces the exact original
region (INCLUDING any interior obfuscation chars).

Direction: ADDITIVE — the pre-pass can only make patterns match text they
previously could not; remapped spans only ever EXPAND to cover the original
region. Leak-safe.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache


@lru_cache(maxsize=8192)
def _fold_char(ch: str) -> str:
    """Fold one non-ASCII char: drop format chars (Cf), else per-char NFKC.

    Returns ``""`` for a stripped format char, else the NFKC-folded string
    (usually 1 char; a few compatibility chars expand). Per-char NFKC never
    composes ACROSS chars, so precomposed accents and combining sequences
    ("café") are left intact — legitimate multilingual text keeps the identity
    map and is unaffected downstream."""
    if unicodedata.category(ch) == "Cf":
        return ""  # ZWSP/ZWNJ/ZWJ/BOM/soft-hyphen/word-joiner/bidi controls
    return unicodedata.normalize("NFKC", ch)


def normalize_for_detection(text: str) -> tuple[str, list[int] | None]:
    """Return ``(scan_text, idx_map)``.

    ``idx_map[i]`` is the ORIGINAL index of normalized char ``i`` (a
    ``len(text)`` sentinel is appended). ``idx_map`` is ``None`` when
    ``scan_text is text`` (nothing changed) — the C-speed fast path for ASCII
    text (NFKC is identity on ASCII), which is the overwhelming common case and
    leaves detection byte-identical.
    """
    if text.isascii():
        return text, None
    out: list[str] = []
    idx_map: list[int] = []
    changed = False
    for orig_i, ch in enumerate(text):
        if ord(ch) < 0x80:
            out.append(ch)
            idx_map.append(orig_i)
            continue
        folded = _fold_char(ch)
        if folded != ch:
            changed = True
        for fc in folded:  # 0 (stripped), 1 (usual), or >1 (expansion) chars
            out.append(fc)
            idx_map.append(orig_i)
    if not changed:
        return text, None
    idx_map.append(len(text))  # end sentinel for exclusive-end remap
    return "".join(out), idx_map


def remap_span(idx_map: list[int] | None, start: int, end: int) -> tuple[int, int]:
    """Map a ``[start, end)`` span in normalized coordinates back to the
    original text. Identity when ``idx_map`` is ``None`` (fast path).

    The original span is ``[idx_map[start], idx_map[end])`` — a contiguous
    original range, so any interior obfuscation chars between mapped positions
    are covered by construction (the leak-safe expansion)."""
    if idx_map is None:
        return start, end
    n = len(idx_map)
    o_start = idx_map[start] if 0 <= start < n else start
    o_end = idx_map[end] if 0 <= end < n else (idx_map[-1] if idx_map else end)
    return o_start, o_end
