"""Pure-stdlib PDF text reader (S7-01, DC-14, FR-031/FR-033-representative).

Extracts text from simple machine-generated PDFs with NO third-party
dependency: it scans the raw bytes for content streams, FlateDecode-inflates
them when ``/Filter /FlateDecode`` is declared (``zlib``), and harvests the
text-show operators (``(literal) Tj``, ``[(a) (b)] TJ``, ``'``/``"``) with a
paren-balanced, backslash-escape-aware literal-string scanner. Each
text-bearing content stream becomes ONE :class:`IngestRecord` — for the
one-content-stream-per-page PDFs this reader targets, record == page — with
``metadata={"modality": "pdf", "source_coords": {"page": n}}`` (the FR-033
representative page-granular fidelity assertion).

Documented limits (deliberate — this is the honest stdlib floor):
no cross-reference-table walk, no encrypted PDFs, no hex strings, no CID /
composite fonts, no encodings beyond Latin-1 literals, no Td/TM positioning
reconstruction (chunks are newline-joined in stream order). Sufficient for
simple machine-generated PDFs and the synthetic fixtures.

# SWITCH-POINT(PDF-LIB): full-fidelity extraction (xref walk, fonts,
# encodings, encrypted docs) swaps `pypdf` in behind this same "pdf" reader
# name + capability row. Pass-2.
# SWITCH-POINT(COORDS): glyph/span-level source coordinates (beyond page
# granularity) for FR-033-full. Pass-2.

Deterministic (AX-002): a byte-driven parse — no wall-clock, no randomness.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterator
from pathlib import Path

from .native import ReaderCapabilities
from .schema import IngestConfig, IngestRecord

__all__ = ["PdfTextReader"]


# The negative lookbehind keeps the keyword scan from matching the tail of
# "endstream\n" — without it every endstream would open a phantom stream
# spanning into the NEXT object (observed as a duplicated final page).
_STREAM_START_RE = re.compile(rb"(?<!end)stream\r?\n")

# Single-character escapes in PDF literal strings (PDF 32000-1 §7.3.4.2).
_LITERAL_ESCAPES: dict[int, str] = {
    ord("n"): "\n",
    ord("r"): "\r",
    ord("t"): "\t",
    ord("b"): "\b",
    ord("f"): "\f",
    ord("("): "(",
    ord(")"): ")",
    ord("\\"): "\\",
}


def _parse_literal_string(content: bytes, start: int) -> tuple[str, int]:
    """Parse a ``(...)`` literal string starting at ``content[start] == '('``.

    Handles backslash escapes (named, octal ``\\ddd``, line continuation)
    and balanced nested parentheses. Returns the decoded text (Latin-1,
    byte-per-char) and the index just past the closing paren.
    """
    out: list[str] = []
    depth = 1
    i = start + 1
    n = len(content)
    while i < n and depth > 0:
        c = content[i]
        if c == 0x5C:  # backslash
            i += 1
            if i >= n:
                break
            e = content[i]
            if 0x30 <= e <= 0x37:  # octal \d, \dd, \ddd
                digits = chr(e)
                for _ in range(2):
                    if i + 1 < n and 0x30 <= content[i + 1] <= 0x37:
                        i += 1
                        digits += chr(content[i])
                out.append(chr(int(digits, 8)))
            elif e in (0x0A, 0x0D):  # line continuation — emits nothing
                if e == 0x0D and i + 1 < n and content[i + 1] == 0x0A:
                    i += 1
            else:
                out.append(_LITERAL_ESCAPES.get(e, chr(e)))
            i += 1
            continue
        if c == 0x28:  # (
            depth += 1
            out.append("(")
        elif c == 0x29:  # )
            depth -= 1
            if depth == 0:
                i += 1
                break
            out.append(")")
        else:
            out.append(chr(c))
        i += 1
    return "".join(out), i


def _extract_text_chunks(content: bytes) -> list[str]:
    """Harvest text shown by ``Tj`` / ``TJ`` / ``'`` / ``\"`` operators.

    Literal strings accumulate in a pending buffer; a show operator flushes
    the buffer as one chunk (so a ``TJ`` array's strings join, kerning
    numbers ignored). Strings never followed by a show operator (e.g.
    annotation payloads) are dropped — only *shown* text counts.
    """
    chunks: list[str] = []
    pending: list[str] = []
    i = 0
    n = len(content)
    while i < n:
        c = content[i]
        if c == 0x28:  # ( — literal string
            text, i = _parse_literal_string(content, i)
            pending.append(text)
            continue
        if c == 0x54 and i + 1 < n and content[i + 1] in (0x6A, 0x4A):  # Tj / TJ
            if pending:
                chunks.append("".join(pending))
                pending = []
            i += 2
            continue
        if c in (0x27, 0x22) and pending:  # ' / " show operators
            chunks.append("".join(pending))
            pending = []
            i += 1
            continue
        i += 1
    return chunks


def _decode_stream(raw: bytes, dict_text: bytes) -> bytes:
    """Inflate a content stream when its object dict declares FlateDecode.

    Uses ``decompressobj`` so trailing bytes after the zlib stream (e.g. the
    newline before ``endstream``) never raise.
    """
    if b"/FlateDecode" in dict_text:
        inflater = zlib.decompressobj()
        return inflater.decompress(raw) + inflater.flush()
    return raw


class PdfTextReader:
    """The REAL stdlib native reader: PDF text extraction (FR-031)."""

    format_name = "pdf"
    native_dependency: str | None = None  # stdlib zlib only

    def capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(
            format_name=self.format_name,
            native_dependency=None,
            dependency_available=True,
            extracts_text=True,
            supports_reconstruction=False,
            notes=(
                "Pure-stdlib Tj/TJ harvesting (uncompressed + FlateDecode); "
                "simple machine-generated PDFs; # SWITCH-POINT(PDF-LIB) for "
                "full-fidelity pypdf extraction."
            ),
        )

    def read(
        self, path: str | Path, config: IngestConfig
    ) -> Iterator[IngestRecord]:
        """Yield one record per text-bearing content stream (== page)."""
        data = Path(path).read_bytes()
        page = 0
        for match in _STREAM_START_RE.finditer(data):
            start = match.end()
            end = data.find(b"endstream", start)
            if end < 0:
                continue
            # The owning object dict runs from the preceding "obj" keyword
            # to "stream" — captured whole so nested dicts cannot hide the
            # /Filter declaration (documented-limits parse).
            obj_head = data.rfind(b"obj", 0, match.start())
            dict_text = data[max(obj_head, 0) : match.start()]
            try:
                decoded = _decode_stream(data[start:end], dict_text)
            except zlib.error:
                # An undecodable stream is skipped, never mis-extracted.
                continue
            text_chunks = _extract_text_chunks(decoded)
            if not text_chunks:
                continue
            page += 1
            text = "\n".join(text_chunks)
            if config.max_record_chars and len(text) > config.max_record_chars:
                text = text[: config.max_record_chars]
            yield IngestRecord(
                record_id=page - 1,
                text=text,
                metadata={"modality": "pdf", "source_coords": {"page": page}},
            )
