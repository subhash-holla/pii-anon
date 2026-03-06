"""Streaming writers for CSV, JSON, JSONL, plain-text, and Parquet output.

Each writer accepts an iterator of result dicts and writes them
incrementally so that output-file size is not limited by RAM.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import FileFormat, detect_format


def write_results(
    results: Iterator[dict[str, Any]],
    path: str | Path,
    *,
    fmt: FileFormat | None = None,
    text_key: str = "transformed_text",
    encoding: str = "utf-8",
) -> int:
    """Write processing results to *path*.

    Returns the number of records written.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or detect_format(str(resolved))

    writer_map = {
        FileFormat.CSV: _write_csv,
        FileFormat.JSON: _write_json,
        FileFormat.JSONL: _write_jsonl,
        FileFormat.TXT: _write_txt,
        FileFormat.PARQUET: _write_parquet,
    }
    writer = writer_map.get(fmt)
    if writer is None:
        raise ValueError(f"Unsupported output format: {fmt}")
    return writer(results, resolved, text_key=text_key, encoding=encoding)


# ---------------------------------------------------------------------------
# Format-specific writers
# ---------------------------------------------------------------------------


def _write_csv(
    results: Iterator[dict[str, Any]],
    path: Path,
    *,
    text_key: str,
    encoding: str,
) -> int:
    count = 0
    fieldnames: list[str] | None = None
    with path.open("w", encoding=encoding, newline="") as fh:
        writer: csv.DictWriter[str] | None = None
        for record in results:
            flat = _flatten_result(record, text_key)
            if writer is None:
                fieldnames = list(flat.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            assert fieldnames is not None  # guaranteed by writer init above
            writer.writerow({k: flat.get(k, "") for k in fieldnames})
            count += 1
    return count


def _write_json(
    results: Iterator[dict[str, Any]],
    path: Path,
    *,
    text_key: str,
    encoding: str,
) -> int:
    # JSON array requires materializing all results — unavoidable for valid JSON
    items: list[dict[str, Any]] = []
    for record in results:
        items.append(_flatten_result(record, text_key))
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding=encoding)
    return len(items)


def _write_jsonl(
    results: Iterator[dict[str, Any]],
    path: Path,
    *,
    text_key: str,
    encoding: str,
) -> int:
    count = 0
    with path.open("w", encoding=encoding) as fh:
        for record in results:
            flat = _flatten_result(record, text_key)
            fh.write(json.dumps(flat, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_txt(
    results: Iterator[dict[str, Any]],
    path: Path,
    *,
    text_key: str,
    encoding: str,
) -> int:
    count = 0
    with path.open("w", encoding=encoding) as fh:
        for record in results:
            flat = _flatten_result(record, text_key)
            text = str(flat.get(text_key, ""))
            fh.write(text + "\n")
            count += 1
    return count


def _write_parquet(
    results: Iterator[dict[str, Any]],
    path: Path,
    *,
    text_key: str,
    encoding: str,
) -> int:
    """Write results to a Parquet file.

    Requires optional dependency ``pyarrow``.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Parquet output requires the 'pyarrow' package. "
            "Install with: pip install pii-anon[parquet]  (or: pip install pyarrow)"
        ) from None

    # Build column-oriented data directly to avoid double-materializing rows.
    columns: dict[str, list[Any]] = {}
    count = 0
    for record in results:
        flat = _flatten_result(record, text_key)
        # Initialise columns on the first row.
        if count == 0:
            for key in flat:
                columns[key] = []
        for key in columns:
            columns[key].append(flat.get(key))
        count += 1

    if count == 0:
        # Create empty parquet with at least one column
        table = pa.table({text_key: pa.array([], type=pa.string())})
        pq.write_table(table, str(path))
        return 0

    table = pa.table(columns)
    pq.write_table(table, str(path))
    return count


def _flatten_result(record: dict[str, Any], text_key: str) -> dict[str, Any]:
    """Flatten a processing result dict for tabular output."""
    flat: dict[str, Any] = {}

    # Preserve original metadata
    for k, v in record.get("metadata", {}).items():
        flat[k] = v

    # Core output fields
    transformed = record.get("transformed_payload", {})
    if isinstance(transformed, dict):
        flat[text_key] = transformed.get("text", str(transformed))
    else:
        flat[text_key] = str(transformed)

    # Summary stats
    findings = record.get("ensemble_findings", [])
    flat["entities_found"] = len(findings) if isinstance(findings, list) else 0
    envelope = record.get("confidence_envelope", {})
    if isinstance(envelope, dict):
        flat["confidence_score"] = envelope.get("score", 0.0)
        flat["risk_level"] = envelope.get("risk_level", "unknown")
    return flat
