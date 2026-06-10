"""S7-01 — native-format readers behind the ``Iterator[IngestRecord]`` contract (DC-14).

Implements FR-031 (uniform native readers — PDF/image/screenshot/DICOM/audio;
MUST), FR-032 (text-format round-trip stays real + pinned; native readers
report supports_reconstruction=False honestly), FR-033 (extraction fidelity —
page-granular source_coords; SHOULD/representative), FR-034 (per-modality
recall scored separately; MUST/representative) and FR-035 (CI gate on reader
recall regression — teeth-proven; MUST). Upholds NFR-026 (optional deps
degrade LOUDLY, never silent empty text), NFR-023 (stream-vs-batch parity
guard), AX-001 (synthetic fixtures only) and AX-002 (deterministic reads).

Anchors A1–A13 per the story contract
(dev-assist-artifacts/04-development/02-stories/sprint-7/
S7-01-native-format-readers.md). Exact-value anchors throughout — never
bounds. No optional dependency (pytesseract/PIL/pydicom) is installed in
this environment; the capability-honesty anchors pin exactly that.
"""
from __future__ import annotations

import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from pii_anon.ingestion import (
    FileFormat,
    IngestConfig,
    IngestRecord,
    detect_format,
    read_file,
    write_results,
)
from pii_anon.ingestion.native import (
    AudioReader,
    DicomReader,
    ImageOcrReader,
    NativeReader,
    NativeReaderRegistry,
    ReaderCapabilities,
    ScreenshotOcrReader,
    default_reader_registry,
    reader_capabilities,
)
from pii_anon.ingestion.native_pdf import PdfTextReader


# ---------------------------------------------------------------------------
# Synthetic PDF fixtures (AX-001 — synthetic-shaped values only)
# ---------------------------------------------------------------------------

_PAGE_1_TEXT = "mail zz-alias@synthetic.test now"
_PAGE_2_TEXT = "call 000-0000 today"


def _pdf_bytes(*, compress: bool = False) -> bytes:
    """Handcraft a minimal 2-page PDF (one content stream per page).

    The streams carry one ``(literal) Tj`` text-show each. With
    ``compress=True`` the identical content streams are FlateDecode-encoded.
    """

    def _content(text: str) -> bytes:
        return b"BT /F1 12 Tf 72 720 Td (" + text.encode("latin-1") + b") Tj ET"

    def _stream_obj(num: int, payload: bytes) -> bytes:
        if compress:
            payload = zlib.compress(payload)
            head = (
                f"{num} 0 obj\n<< /Length {len(payload)} "
                "/Filter /FlateDecode >>\nstream\n"
            ).encode("latin-1")
        else:
            head = (
                f"{num} 0 obj\n<< /Length {len(payload)} >>\nstream\n"
            ).encode("latin-1")
        return head + payload + b"\nendstream\nendobj\n"

    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n",
        _stream_obj(4, _content(_PAGE_1_TEXT)),
        b"5 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 6 0 R >>\nendobj\n",
        _stream_obj(6, _content(_PAGE_2_TEXT)),
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n",
    ]
    return b"".join(parts)


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic.pdf"
    p.write_bytes(_pdf_bytes(compress=False))
    return p


@pytest.fixture
def pdf_flate_path(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic-flate.pdf"
    p.write_bytes(_pdf_bytes(compress=True))
    return p


# ---------------------------------------------------------------------------
# A1 — registry + entry-point discovery
# ---------------------------------------------------------------------------

def test_a1_default_registry_and_entrypoint_discovery() -> None:
    """[UNIT-TEST] A1: the default registry carries exactly the five readers
    (sorted); entry-point discovery on the installed pii_anon.readers group
    returns the same five; every entry satisfies the NativeReader Protocol."""
    registry = default_reader_registry()
    assert registry.names() == ["audio", "dicom", "image", "pdf", "screenshot"]

    discovered = NativeReaderRegistry()
    names = discovered.discover_entrypoint_readers()
    assert sorted(names) == ["audio", "dicom", "image", "pdf", "screenshot"]
    for name in names:
        reader = discovered.get(name)
        assert isinstance(reader, NativeReader)


# ---------------------------------------------------------------------------
# A2 — capability honesty in this environment
# ---------------------------------------------------------------------------

def test_a2_capability_honesty_without_optional_deps() -> None:
    """[UNIT-TEST] A2: exact capability booleans in this env — pdf is
    stdlib-real; image/screenshot/dicom report their missing native deps;
    audio extracts no text. Constructors import no optional dependency."""
    caps = {c.format_name: c for c in reader_capabilities()}
    assert sorted(caps) == ["audio", "dicom", "image", "pdf", "screenshot"]

    assert caps["pdf"].native_dependency is None
    assert caps["pdf"].dependency_available is True
    assert caps["pdf"].extracts_text is True

    assert caps["image"].native_dependency == "pytesseract"
    assert caps["image"].dependency_available is False
    assert caps["screenshot"].native_dependency == "pytesseract"
    assert caps["screenshot"].dependency_available is False
    assert caps["dicom"].native_dependency == "pydicom"
    assert caps["dicom"].dependency_available is False

    assert caps["audio"].extracts_text is False
    assert caps["audio"].dependency_available is False

    # FR-032 honesty: no native reader claims reconstruction.
    for cap in caps.values():
        assert isinstance(cap, ReaderCapabilities)
        assert cap.supports_reconstruction is False


# ---------------------------------------------------------------------------
# A3 / A4 — the REAL stdlib PDF reader
# ---------------------------------------------------------------------------

def test_a3_pdf_reader_extracts_exact_pages_uncompressed(pdf_path: Path) -> None:
    """[UNIT-TEST] A3: the handcrafted 2-page synthetic PDF yields exactly 2
    records with the exact expected strings and record_ids 0,1."""
    records = list(PdfTextReader().read(pdf_path, IngestConfig()))
    assert len(records) == 2
    assert records[0].record_id == 0
    assert records[1].record_id == 1
    assert records[0].text == _PAGE_1_TEXT
    assert records[1].text == _PAGE_2_TEXT


def test_a4_pdf_reader_extracts_exact_pages_flatedecode(
    pdf_path: Path, pdf_flate_path: Path
) -> None:
    """[UNIT-TEST] A4: the zlib/FlateDecode variant yields the exact same
    text as the uncompressed variant."""
    plain = [r.text for r in PdfTextReader().read(pdf_path, IngestConfig())]
    flate = [r.text for r in PdfTextReader().read(pdf_flate_path, IngestConfig())]
    assert flate == plain == [_PAGE_1_TEXT, _PAGE_2_TEXT]


# ---------------------------------------------------------------------------
# A5 — determinism (AX-002)
# ---------------------------------------------------------------------------

def test_a5_pdf_reads_are_deterministic(pdf_flate_path: Path) -> None:
    """[UNIT-TEST] A5: 5 consecutive reads yield byte-identical sequences."""
    reader = PdfTextReader()
    runs = [
        [(r.record_id, r.text, r.metadata) for r in reader.read(pdf_flate_path, IngestConfig())]
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs[1:])


# ---------------------------------------------------------------------------
# A6 — format detection + read_file dispatch
# ---------------------------------------------------------------------------

def test_a6_detect_format_and_read_file_dispatch(pdf_path: Path) -> None:
    """[UNIT-TEST] A6: the new extensions map to the exact FileFormat
    members; read_file dispatches PDFs through the registry; text-format
    detection is unchanged."""
    assert detect_format("a.pdf") is FileFormat.PDF
    assert detect_format("a.png") is FileFormat.PNG
    assert detect_format("a.jpg") is FileFormat.JPEG
    assert detect_format("a.jpeg") is FileFormat.JPEG
    assert detect_format("a.dcm") is FileFormat.DICOM
    assert detect_format("a.wav") is FileFormat.WAV
    # Existing text formats unchanged (exact pins).
    assert detect_format("a.csv") is FileFormat.CSV
    assert detect_format("a.txt") is FileFormat.TXT

    records = list(read_file(pdf_path))
    assert [r.text for r in records] == [_PAGE_1_TEXT, _PAGE_2_TEXT]


# ---------------------------------------------------------------------------
# A7 / A8 — loud degradation (NFR-026: no silent recall loss)
# ---------------------------------------------------------------------------

def test_a7_ocr_reader_raises_loud_importerror_naming_the_extra(
    tmp_path: Path,
) -> None:
    """[UNIT-TEST] A7: with the extra absent, the OCR readers raise an
    ImportError whose message names pii-anon[ocr]; construction and
    capability reporting never raise."""
    png = tmp_path / "synthetic.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    for reader in (ImageOcrReader(), ScreenshotOcrReader()):
        assert reader.capabilities().dependency_available is False
        with pytest.raises(ImportError, match=r"pii-anon\[ocr\]"):
            list(reader.read(png, IngestConfig()))

    dcm = tmp_path / "synthetic.dcm"
    dcm.write_bytes(b"DICM")
    with pytest.raises(ImportError, match=r"pii-anon\[dicom\]"):
        list(DicomReader().read(dcm, IngestConfig()))


def test_a8_audio_reader_never_yields_silent_empty_text(tmp_path: Path) -> None:
    """[SECURITY-TEST] A8: the audio reader RAISES (naming the ASR Pass-2
    seam) — it can never silently yield empty-text records (NFR-026
    'no silent recall loss')."""
    wav = tmp_path / "synthetic.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    with pytest.raises(NotImplementedError, match="ASR"):
        list(AudioReader().read(wav, IngestConfig()))


# ---------------------------------------------------------------------------
# A9 — extraction fidelity (FR-033 representative)
# ---------------------------------------------------------------------------

def test_a9_pdf_records_carry_page_source_coords(pdf_path: Path) -> None:
    """[UNIT-TEST] A9: every PDF record carries modality + 1-based page
    source_coords; offsets used downstream are in-range."""
    records = list(PdfTextReader().read(pdf_path, IngestConfig()))
    for idx, record in enumerate(records):
        assert record.metadata["modality"] == "pdf"
        assert record.metadata["source_coords"] == {"page": idx + 1}
        assert len(record.text) > 0


# ---------------------------------------------------------------------------
# A10 / A11 — per-modality recall (FR-034) + CI-gate teeth (FR-035)
# ---------------------------------------------------------------------------
# Representative harness: gold spans are expected substrings per modality;
# recall = found/total per modality, NEVER merged across modalities. The
# corpus-scale benchmark is # SWITCH-POINT(DATA) Pass-2.

_GOLD: dict[str, list[str]] = {
    "pdf": ["zz-alias@synthetic.test", "000-0000"],
    "txt": ["zz-alias@synthetic.test", "111-1111"],
}

# The pinned per-modality recall baselines (FR-035). Integer-count rates.
_RECALL_BASELINE: dict[str, float] = {"pdf": 1.0, "txt": 0.5}


def _modality_recall(extracted: dict[str, str]) -> dict[str, float]:
    """Per-modality recall of gold substrings over extracted text."""
    recalls: dict[str, float] = {}
    for modality, gold_spans in _GOLD.items():
        text = extracted.get(modality, "")
        found = sum(1 for span in gold_spans if span in text)
        recalls[modality] = found / len(gold_spans)
    return recalls


def _recall_gate_passes(recalls: dict[str, float]) -> bool:
    """FR-035: every modality must meet its pinned recall baseline."""
    return all(recalls[m] >= _RECALL_BASELINE[m] for m in _RECALL_BASELINE)


def test_a10_per_modality_recall_scored_separately(
    pdf_path: Path, tmp_path: Path
) -> None:
    """[UNIT-TEST] A10: pdf recall is exactly 2/2 and txt recall exactly 1/2
    on the synthetic gold — keyed separately per modality, never merged."""
    txt = tmp_path / "synthetic.txt"
    txt.write_text("mail zz-alias@synthetic.test now\n", encoding="utf-8")

    extracted = {
        "pdf": "\n".join(r.text for r in read_file(pdf_path)),
        "txt": "\n".join(r.text for r in read_file(txt)),
    }
    recalls = _modality_recall(extracted)
    assert recalls == {"pdf": 1.0, "txt": 0.5}
    assert _recall_gate_passes(recalls)


def test_a11_recall_regression_gate_has_teeth(pdf_path: Path, tmp_path: Path) -> None:
    """[UNIT-TEST] A11: a deliberately degraded reader (drops page 2 — the
    phone span) pushes pdf recall to 1/2 < the pinned 1.0 baseline and the
    gate FAILS — the FR-035 gate has teeth."""
    txt = tmp_path / "synthetic.txt"
    txt.write_text("mail zz-alias@synthetic.test now\n", encoding="utf-8")

    degraded_pdf_text = "\n".join(
        r.text for r in read_file(pdf_path) if r.record_id == 0
    )
    recalls = _modality_recall(
        {"pdf": degraded_pdf_text, "txt": txt.read_text(encoding="utf-8")}
    )
    assert recalls["pdf"] == 0.5
    assert not _recall_gate_passes(recalls)


# ---------------------------------------------------------------------------
# A12 — stream/batch parity guard (FR-036 / NFR-023 representative)
# ---------------------------------------------------------------------------

def test_a12_stream_vs_batch_parity_and_text_round_trip(
    pdf_path: Path, tmp_path: Path
) -> None:
    """[UNIT-TEST] A12: lazy iteration vs full materialization yield
    identical records for pdf + txt; the existing text-format round-trip
    (writers) byte-identity is re-pinned on a synthetic fixture."""
    lazy_pdf: Iterator[IngestRecord] = read_file(pdf_path)
    assert [(r.record_id, r.text) for r in lazy_pdf] == [
        (r.record_id, r.text) for r in list(read_file(pdf_path))
    ]

    txt = tmp_path / "synthetic.txt"
    txt.write_text("alpha line one\nbeta line two\n", encoding="utf-8")
    assert [(r.record_id, r.text) for r in read_file(txt)] == [
        (r.record_id, r.text) for r in list(read_file(txt))
    ]

    # Text round-trip stays REAL (FR-032 for text formats): write records
    # back out in the writers' result shape and re-read byte-identical texts.
    out = tmp_path / "round-trip.jsonl"
    records = list(read_file(txt))
    write_results(
        iter({"transformed_payload": {"text": r.text}} for r in records),
        str(out),
    )
    reread = list(read_file(out, IngestConfig(text_column="transformed_text")))
    assert [r.text for r in reread] == [r.text for r in records]


# ---------------------------------------------------------------------------
# A14 — bounded FlateDecode inflate (gate remediation: security-sast MAJOR-1)
# ---------------------------------------------------------------------------

def test_a14_flatedecode_bomb_stream_is_skipped_not_inflated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[SECURITY-TEST] A14: a FlateDecode stream whose inflated size exceeds
    the per-stream ceiling is treated as undecodable and SKIPPED (no record,
    no unbounded memory) while an honest page in the same PDF still
    extracts. Honest-input behavior unchanged (A4 still passes)."""
    import pii_anon.ingestion.native_pdf as native_pdf

    monkeypatch.setattr(native_pdf, "_MAX_DECOMPRESSED_STREAM_BYTES", 4096)

    bomb_payload = zlib.compress(b"BT (" + b"A" * 100_000 + b") Tj ET")
    honest_payload = b"BT (" + _PAGE_2_TEXT.encode("latin-1") + b") Tj ET"
    parts = [
        b"%PDF-1.4\n",
        (
            f"4 0 obj\n<< /Length {len(bomb_payload)} "
            "/Filter /FlateDecode >>\nstream\n"
        ).encode("latin-1"),
        bomb_payload,
        b"\nendstream\nendobj\n",
        (f"6 0 obj\n<< /Length {len(honest_payload)} >>\nstream\n").encode(
            "latin-1"
        ),
        honest_payload,
        b"\nendstream\nendobj\n%%EOF\n",
    ]
    p = tmp_path / "bomb.pdf"
    p.write_bytes(b"".join(parts))

    records = list(PdfTextReader().read(p, IngestConfig()))
    assert [r.text for r in records] == [_PAGE_2_TEXT]

    # Direct unit pin: the bounded decoder refuses the bomb with zlib.error.
    with pytest.raises(zlib.error, match="ceiling"):
        native_pdf._decode_stream(
            bomb_payload, b"<< /Filter /FlateDecode >>", max_bytes=4096
        )


# ---------------------------------------------------------------------------
# A15 / A16 / A17 — parser edge paths + registry guards (code-quality MAJOR-2)
# ---------------------------------------------------------------------------

def test_a15_literal_string_escapes_nested_parens_and_quote_ops() -> None:
    """[UNIT-TEST] A15: octal/named/escaped-paren escapes, CRLF line
    continuation, balanced nested parentheses, and the ' / " show operators
    all decode to the exact expected chunks."""
    from pii_anon.ingestion.native_pdf import _extract_text_chunks

    content = (
        b"BT (Esc\\101ped\\n\\(line\\)) Tj "
        b"(outer (inner) tail) Tj "
        b"(ab\\\r\ncd) Tj "
        b"(quoted) ' "
        b"(dq) \" ET"
    )
    assert _extract_text_chunks(content) == [
        "EscAped\n(line)",
        "outer (inner) tail",
        "abcd",
        "quoted",
        "dq",
    ]
    # A string never followed by a show operator is dropped.
    assert _extract_text_chunks(b"BT (orphan) ET") == []


def test_a16_corrupt_zlib_empty_streams_and_truncation(
    tmp_path: Path, pdf_path: Path
) -> None:
    """[UNIT-TEST] A16: a corrupt FlateDecode stream is skipped while honest
    pages survive; a no-text content stream yields no record; and
    max_record_chars truncates the page text exactly."""
    honest_payload = b"BT (" + _PAGE_1_TEXT.encode("latin-1") + b") Tj ET"
    drawing_payload = b"0 0 m 10 10 l S"
    parts = [
        b"%PDF-1.4\n",
        b"4 0 obj\n<< /Length 8 /Filter /FlateDecode >>\nstream\n",
        b"not-zlib",
        b"\nendstream\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(drawing_payload)} >>\nstream\n"
        ).encode("latin-1"),
        drawing_payload,
        b"\nendstream\nendobj\n",
        (f"6 0 obj\n<< /Length {len(honest_payload)} >>\nstream\n").encode(
            "latin-1"
        ),
        honest_payload,
        b"\nendstream\nendobj\n%%EOF\n",
    ]
    p = tmp_path / "mixed.pdf"
    p.write_bytes(b"".join(parts))
    records = list(PdfTextReader().read(p, IngestConfig()))
    assert [r.text for r in records] == [_PAGE_1_TEXT]

    truncated = list(
        PdfTextReader().read(pdf_path, IngestConfig(max_record_chars=4))
    )
    assert [r.text for r in truncated] == [
        _PAGE_1_TEXT[:4],
        _PAGE_2_TEXT[:4],
    ]


def test_a16b_parser_and_decoder_edge_paths(tmp_path: Path) -> None:
    """[UNIT-TEST] A16b: truncation/edge guards — a literal string cut off
    mid-escape, a multi-megabyte (within-ceiling) FlateDecode stream that
    exercises the chunked-inflate loop + flush tail, and a stream keyword
    with no endstream (skipped while honest pages survive)."""
    import pii_anon.ingestion.native_pdf as native_pdf
    from pii_anon.ingestion.native_pdf import _parse_literal_string

    # Backslash at end-of-content: parser returns what it has, no crash.
    text, idx = _parse_literal_string(b"(abc\\", 0)
    assert text == "abc"
    assert idx == 5

    # A 3 MiB-inflating stream WITHIN the ceiling extracts fully (the
    # chunked loop runs multiple 1 MiB steps; the flush tail is appended).
    big = b"A" * (3 * 1024 * 1024)
    payload = zlib.compress(b"BT (" + big + b") Tj ET")
    decoded = native_pdf._decode_stream(
        payload, b"<< /Filter /FlateDecode >>"
    )
    assert decoded == b"BT (" + big + b") Tj ET"

    # A `stream` keyword with no matching endstream is skipped; the honest
    # page in the same file still extracts.
    honest_payload = b"BT (" + _PAGE_1_TEXT.encode("latin-1") + b") Tj ET"
    parts = [
        b"%PDF-1.4\n",
        (f"4 0 obj\n<< /Length {len(honest_payload)} >>\nstream\n").encode(
            "latin-1"
        ),
        honest_payload,
        b"\nendstream\nendobj\n",
        b"6 0 obj\n<< /Length 5 >>\nstream\n",  # no endstream follows
        b"(x) Tj%%EOF\n",
    ]
    p = tmp_path / "no-endstream.pdf"
    p.write_bytes(b"".join(parts))
    records = list(PdfTextReader().read(p, IngestConfig()))
    assert [r.text for r in records] == [_PAGE_1_TEXT]


def test_a17_registry_fail_closed_and_round_trip() -> None:
    """[UNIT-TEST] A17: register() refuses a non-conformant object with
    TypeError (fail-closed); register/get/unregister round-trip works."""
    registry = NativeReaderRegistry()
    with pytest.raises(TypeError, match="NativeReader"):
        registry.register("bad", object())  # type: ignore[arg-type]

    reader = PdfTextReader()
    registry.register("pdf", reader)
    assert registry.get("pdf") is reader
    assert registry.names() == ["pdf"]
    registry.unregister("pdf")
    registry.unregister("pdf")  # no-op when absent
    assert registry.get("pdf") is None
    assert registry.names() == []


# ---------------------------------------------------------------------------
# A13 — import-boundary audit
# ---------------------------------------------------------------------------

def test_a13_modules_import_nothing_forbidden() -> None:
    """[AUDIT] A13: native.py + native_pdf.py import nothing from
    swarm/moe/fusion/orchestrator."""
    import ast

    import pii_anon.ingestion.native as native_mod
    import pii_anon.ingestion.native_pdf as pdf_mod

    forbidden = {"swarm", "moe", "fusion", "orchestrator"}
    for mod in (native_mod, pdf_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                head = node.module.split(".")
                assert not (
                    len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden
                ), node.module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    head = alias.name.split(".")
                    assert not (
                        len(head) >= 2
                        and head[0] == "pii_anon"
                        and head[1] in forbidden
                    ), alias.name
