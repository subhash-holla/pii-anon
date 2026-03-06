"""Lazy file readers for CSV, JSON, JSONL, plain-text, Parquet, XML, and HTML.

Every reader returns ``Iterator[IngestRecord]`` so that callers can process
arbitrarily large files without loading them entirely into memory.

Additional formats (Parquet, XML, HTML) use optional dependencies that
degrade gracefully with clear error messages when unavailable.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import FileFormat, IngestConfig, IngestRecord, detect_format


def read_file(
    path: str | Path,
    config: IngestConfig | None = None,
) -> Iterator[IngestRecord]:
    """Read records from *path* according to *config*.

    If ``config.format`` is ``None`` the format is auto-detected from the file
    extension.
    """
    config = config or IngestConfig()
    resolved = Path(path)
    fmt = config.format or detect_format(str(resolved))

    reader_map = {
        FileFormat.CSV: _read_csv,
        FileFormat.JSON: _read_json,
        FileFormat.JSONL: _read_jsonl,
        FileFormat.TXT: _read_txt,
        FileFormat.PARQUET: _read_parquet,
        FileFormat.XML: _read_xml,
        FileFormat.HTML: _read_html,
    }
    reader = reader_map.get(fmt)
    if reader is None:
        raise ValueError(f"Unsupported file format: {fmt}")
    yield from reader(resolved, config)


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------


def _read_csv(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    with path.open("r", encoding=config.encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=config.csv_delimiter)
        for idx, row in enumerate(reader):
            text = row.get(config.text_column, "")
            if text is None:
                text = ""
            metadata = {k: v for k, v in row.items() if k != config.text_column}
            if config.max_record_chars and len(text) > config.max_record_chars:
                text = text[: config.max_record_chars]
            yield IngestRecord(record_id=idx, text=text, metadata=metadata)


def _read_json(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    raw = path.read_text(encoding=config.encoding)
    data = json.loads(raw)
    records: list[dict[str, Any]]
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Support {"records": [...]} wrapper or fall back to single-record
        if "records" in data and isinstance(data["records"], list):
            records = data["records"]
        else:
            records = [data]
    else:
        raise ValueError(f"JSON root must be an array or object, got {type(data).__name__}")

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        text = str(item.get(config.text_column, ""))
        metadata = {k: v for k, v in item.items() if k != config.text_column}
        if config.max_record_chars and len(text) > config.max_record_chars:
            text = text[: config.max_record_chars]
        yield IngestRecord(record_id=idx, text=text, metadata=metadata)


def _read_jsonl(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    with path.open("r", encoding=config.encoding) as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                continue
            text = str(item.get(config.text_column, ""))
            metadata = {k: v for k, v in item.items() if k != config.text_column}
            if config.max_record_chars and len(text) > config.max_record_chars:
                text = text[: config.max_record_chars]
            yield IngestRecord(record_id=idx, text=text, metadata=metadata)


def _read_txt(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    if config.whole_file:
        text = path.read_text(encoding=config.encoding)
        if config.max_record_chars and len(text) > config.max_record_chars:
            text = text[: config.max_record_chars]
        yield IngestRecord(record_id=0, text=text, metadata={"source": str(path)})
        return

    with path.open("r", encoding=config.encoding) as fh:
        idx = 0
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if config.max_record_chars and len(stripped) > config.max_record_chars:
                stripped = stripped[: config.max_record_chars]
            yield IngestRecord(record_id=idx, text=stripped, metadata={})
            idx += 1


def _read_parquet(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    """Read records from a Parquet file (batch iterator for memory efficiency).

    Requires optional dependency ``pyarrow``.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Parquet support requires the 'pyarrow' package. "
            "Install with: pip install pii-anon[parquet]  (or: pip install pyarrow)"
        ) from None

    table = pq.read_table(str(path))
    col_names = table.column_names

    if config.text_column not in col_names:
        raise ValueError(
            f"Text column {config.text_column!r} not found in Parquet file. "
            f"Available columns: {col_names}"
        )

    record_id = 0
    batch_size = 10_000
    for batch in table.to_batches(max_chunksize=batch_size):
        batch_dict = batch.to_pydict()
        texts = batch_dict.get(config.text_column, [])
        for i in range(len(texts)):
            text = str(texts[i]) if texts[i] is not None else ""
            if config.max_record_chars and len(text) > config.max_record_chars:
                text = text[: config.max_record_chars]
            metadata = {
                col: batch_dict[col][i]
                for col in col_names
                if col != config.text_column
            }
            yield IngestRecord(record_id=record_id, text=text, metadata=metadata)
            record_id += 1


def _read_xml(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    """Read text content from an XML file.

    Uses stdlib ``xml.etree.ElementTree``.  Each text-bearing element
    becomes one ``IngestRecord``.  If ``config.whole_file`` is True, all
    text is concatenated into a single record.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(path))
    root = tree.getroot()

    if config.whole_file:
        # Concatenate all text content
        parts: list[str] = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                parts.append(elem.tail.strip())
        full_text = " ".join(parts)
        if config.max_record_chars and len(full_text) > config.max_record_chars:
            full_text = full_text[: config.max_record_chars]
        yield IngestRecord(
            record_id=0,
            text=full_text,
            metadata={"source": str(path), "format": "xml"},
        )
        return

    # One record per text-bearing element
    record_id = 0
    for elem in root.iter():
        text_parts: list[str] = []
        if elem.text and elem.text.strip():
            text_parts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            text_parts.append(elem.tail.strip())
        if not text_parts:
            continue
        text = " ".join(text_parts)
        if config.max_record_chars and len(text) > config.max_record_chars:
            text = text[: config.max_record_chars]
        yield IngestRecord(
            record_id=record_id,
            text=text,
            metadata={"tag": elem.tag, "attrib": dict(elem.attrib)},
        )
        record_id += 1


def _read_html(path: Path, config: IngestConfig) -> Iterator[IngestRecord]:
    """Read text content from an HTML file.

    Uses stdlib ``html.parser`` — no external dependencies required.
    Strips tags and extracts visible text content.
    """
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        """Simple HTML text extractor that skips script/style content."""

        def __init__(self) -> None:
            super().__init__()
            self.texts: list[str] = []
            self._skip = False
            self._skip_tags = {"script", "style", "noscript"}

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag.lower() in self._skip_tags:
                self._skip = True

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() in self._skip_tags:
                self._skip = False

        def handle_data(self, data: str) -> None:
            if not self._skip:
                stripped = data.strip()
                if stripped:
                    self.texts.append(stripped)

    raw = path.read_text(encoding=config.encoding)
    extractor = _TextExtractor()
    extractor.feed(raw)

    if config.whole_file or len(extractor.texts) == 0:
        full_text = " ".join(extractor.texts)
        if config.max_record_chars and len(full_text) > config.max_record_chars:
            full_text = full_text[: config.max_record_chars]
        yield IngestRecord(
            record_id=0,
            text=full_text,
            metadata={"source": str(path), "format": "html"},
        )
        return

    for idx, text in enumerate(extractor.texts):
        if config.max_record_chars and len(text) > config.max_record_chars:
            text = text[: config.max_record_chars]
        yield IngestRecord(record_id=idx, text=text, metadata={"format": "html"})
