"""Data models for file-based ingestion.

Supports CSV, JSON, JSONL, and plain-text file formats with lazy
iteration so that memory stays flat even for GB-scale datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FileFormat(str, Enum):
    """Supported input/output file formats."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    TXT = "txt"
    PARQUET = "parquet"
    XML = "xml"
    HTML = "html"


@dataclass
class IngestConfig:
    """Configuration for file ingestion."""

    format: FileFormat | None = None  # auto-detect from extension if None
    text_column: str = "text"
    encoding: str = "utf-8"
    csv_delimiter: str = ","
    whole_file: bool = False  # TXT only: treat entire file as one record
    max_record_chars: int = 0  # 0 = no limit; capped for safety


@dataclass
class IngestRecord:
    """A single record read from a file."""

    record_id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileIngestResult:
    """Summary of a file ingestion + processing run."""

    input_path: str
    output_path: str | None
    format: str
    records_processed: int
    records_failed: int
    total_chars: int
    total_chunks: int
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def records_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.records_processed / self.elapsed_seconds


def detect_format(path: str) -> FileFormat:
    """Detect file format from extension."""
    lower = path.lower()
    if lower.endswith(".csv"):
        return FileFormat.CSV
    if lower.endswith(".jsonl") or lower.endswith(".ndjson"):
        return FileFormat.JSONL
    if lower.endswith(".json"):
        return FileFormat.JSON
    if lower.endswith(".txt") or lower.endswith(".text"):
        return FileFormat.TXT
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return FileFormat.PARQUET
    if lower.endswith(".xml"):
        return FileFormat.XML
    if lower.endswith(".html") or lower.endswith(".htm"):
        return FileFormat.HTML
    raise ValueError(
        f"Cannot detect file format from extension: {path!r}. "
        f"Supported: .csv, .json, .jsonl, .ndjson, .txt, .text, "
        f".parquet, .pq, .xml, .html, .htm"
    )
