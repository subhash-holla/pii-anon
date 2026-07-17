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

# Category-M (Mn/Mc/Me) combining-mark ranges over the BMP, for regex character
# classes: ``\\w`` excludes marks, so abugida/harakat value atoms (Thai vowel
# signs, Indic matras, Arabic diacritics) otherwise split mid-value. Generated
# from unicodedata (Unicode 15.1, Python 3.12):
#   ranges of c in [0, 0x10000) where unicodedata.category(chr(c)).startswith("M")
# Static so imports stay free; SMP marks (musical/ancient notation) are out of
# scope for PII value atoms. Python decodes the escapes to literal characters.
COMBINING_MARKS_CLASS = (
    "\u0300-\u036f\u0483-\u0489\u0591-\u05bd\u05bf\u05c1-\u05c2\u05c4-\u05c5"
    "\u05c7\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4"
    "\u06e7-\u06e8\u06ea-\u06ed\u0711\u0730-\u074a\u07a6-\u07b0\u07eb-\u07f3"
    "\u07fd\u0816-\u0819\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0859-\u085b"
    "\u0898-\u089f\u08ca-\u08e1\u08e3-\u0903\u093a-\u093c\u093e-\u094f"
    "\u0951-\u0957\u0962-\u0963\u0981-\u0983\u09bc\u09be-\u09c4\u09c7-\u09c8"
    "\u09cb-\u09cd\u09d7\u09e2-\u09e3\u09fe\u0a01-\u0a03\u0a3c\u0a3e-\u0a42"
    "\u0a47-\u0a48\u0a4b-\u0a4d\u0a51\u0a70-\u0a71\u0a75\u0a81-\u0a83\u0abc"
    "\u0abe-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ae2-\u0ae3\u0afa-\u0aff"
    "\u0b01-\u0b03\u0b3c\u0b3e-\u0b44\u0b47-\u0b48\u0b4b-\u0b4d\u0b55-\u0b57"
    "\u0b62-\u0b63\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd7"
    "\u0c00-\u0c04\u0c3c\u0c3e-\u0c44\u0c46-\u0c48\u0c4a-\u0c4d\u0c55-\u0c56"
    "\u0c62-\u0c63\u0c81-\u0c83\u0cbc\u0cbe-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd"
    "\u0cd5-\u0cd6\u0ce2-\u0ce3\u0cf3\u0d00-\u0d03\u0d3b-\u0d3c\u0d3e-\u0d44"
    "\u0d46-\u0d48\u0d4a-\u0d4d\u0d57\u0d62-\u0d63\u0d81-\u0d83\u0dca"
    "\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0df2-\u0df3\u0e31\u0e34-\u0e3a"
    "\u0e47-\u0e4e\u0eb1\u0eb4-\u0ebc\u0ec8-\u0ece\u0f18-\u0f19\u0f35\u0f37"
    "\u0f39\u0f3e-\u0f3f\u0f71-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc"
    "\u0fc6\u102b-\u103e\u1056-\u1059\u105e-\u1060\u1062-\u1064\u1067-\u106d"
    "\u1071-\u1074\u1082-\u108d\u108f\u109a-\u109d\u135d-\u135f\u1712-\u1715"
    "\u1732-\u1734\u1752-\u1753\u1772-\u1773\u17b4-\u17d3\u17dd\u180b-\u180d"
    "\u180f\u1885-\u1886\u18a9\u1920-\u192b\u1930-\u193b\u1a17-\u1a1b"
    "\u1a55-\u1a5e\u1a60-\u1a7c\u1a7f\u1ab0-\u1ace\u1b00-\u1b04\u1b34-\u1b44"
    "\u1b6b-\u1b73\u1b80-\u1b82\u1ba1-\u1bad\u1be6-\u1bf3\u1c24-\u1c37"
    "\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced\u1cf4\u1cf7-\u1cf9\u1dc0-\u1dff"
    "\u20d0-\u20f0\u2cef-\u2cf1\u2d7f\u2de0-\u2dff\u302a-\u302f\u3099-\u309a"
    "\ua66f-\ua672\ua674-\ua67d\ua69e-\ua69f\ua6f0-\ua6f1\ua802\ua806\ua80b"
    "\ua823-\ua827\ua82c\ua880-\ua881\ua8b4-\ua8c5\ua8e0-\ua8f1\ua8ff"
    "\ua926-\ua92d\ua947-\ua953\ua980-\ua983\ua9b3-\ua9c0\ua9e5\uaa29-\uaa36"
    "\uaa43\uaa4c-\uaa4d\uaa7b-\uaa7d\uaab0\uaab2-\uaab4\uaab7-\uaab8"
    "\uaabe-\uaabf\uaac1\uaaeb-\uaaef\uaaf5-\uaaf6\uabe3-\uabea\uabec-\uabed"
    "\ufb1e\ufe00-\ufe0f\ufe20-\ufe2f"
)
