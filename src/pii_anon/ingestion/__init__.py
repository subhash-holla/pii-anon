"""File-based ingestion for batch PII processing.

Supports CSV, JSON, JSONL, plain-text, Parquet, XML, and HTML files
with lazy iteration so that memory usage stays flat regardless of file size.
Also provides DataFrame adapters for pandas/polars integration, and the
native-format reader surface (PDF / image-OCR / screenshot / DICOM / audio
— S7-01, DC-14) behind the same ``Iterator[IngestRecord]`` contract.
"""

from .dataframe import read_dataframe, results_to_dataframe
from .native import (
    NativeReader,
    NativeReaderRegistry,
    ReaderCapabilities,
    default_reader_registry,
    reader_capabilities,
)
from .readers import read_file
from .schema import (
    FileFormat,
    FileIngestResult,
    IngestConfig,
    IngestRecord,
    detect_format,
)
from .writers import write_results

__all__ = [
    "FileFormat",
    "FileIngestResult",
    "IngestConfig",
    "IngestRecord",
    "NativeReader",
    "NativeReaderRegistry",
    "ReaderCapabilities",
    "default_reader_registry",
    "detect_format",
    "read_file",
    "read_dataframe",
    "reader_capabilities",
    "results_to_dataframe",
    "write_results",
]
