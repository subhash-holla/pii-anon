"""DataFrame integration for PII processing pipelines.

Provides zero-dependency adapters between pandas-like DataFrames and the
ingestion pipeline.  All DataFrame interactions are duck-typed — any object
with ``iterrows()`` or ``__iter__`` works, so polars, vaex, and modin
DataFrames are also supported.

Usage::

    import pandas as pd
    from pii_anon.ingestion.dataframe import read_dataframe, results_to_dataframe

    df = pd.read_csv("data.csv")
    for record in read_dataframe(df, text_column="message"):
        ...

    out_df = results_to_dataframe(results)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .schema import IngestRecord


def read_dataframe(
    df: Any,
    *,
    text_column: str = "text",
    max_record_chars: int = 0,
) -> Iterator[IngestRecord]:
    """Convert a DataFrame to an iterator of ``IngestRecord``.

    Works with any pandas-like DataFrame (duck-typed: uses ``iterrows()``
    or falls back to column-based iteration).  No pandas import required.

    Parameters
    ----------
    df : DataFrame-like
        Any object with ``iterrows()`` or column access.
    text_column : str
        Name of the column containing text to process.
    max_record_chars : int
        Maximum characters per record (0 = no limit).

    Yields
    ------
    IngestRecord
        One record per row.

    Raises
    ------
    ValueError
        If ``text_column`` is not found in the DataFrame.
    """
    # Validate column existence
    columns = _get_columns(df)
    if columns is not None and text_column not in columns:
        raise ValueError(
            f"Text column {text_column!r} not found in DataFrame. "
            f"Available columns: {list(columns)}"
        )

    # Prefer iterrows() (pandas/polars compat)
    if hasattr(df, "iterrows"):
        for idx, row in df.iterrows():
            text = str(row.get(text_column, "") or "")
            if max_record_chars and len(text) > max_record_chars:
                text = text[:max_record_chars]
            metadata = {
                k: _safe_value(v)
                for k, v in row.items()
                if k != text_column
            }
            yield IngestRecord(
                record_id=int(idx) if isinstance(idx, (int, float)) else hash(idx),
                text=text,
                metadata=metadata,
            )
    elif hasattr(df, "iter_rows"):
        # Polars-style iteration
        for idx, row in enumerate(df.iter_rows(named=True)):
            text = str(row.get(text_column, "") or "")
            if max_record_chars and len(text) > max_record_chars:
                text = text[:max_record_chars]
            metadata = {
                k: _safe_value(v)
                for k, v in row.items()
                if k != text_column
            }
            yield IngestRecord(record_id=idx, text=text, metadata=metadata)
    else:
        raise TypeError(
            f"Cannot iterate over {type(df).__name__}. "
            f"Expected a DataFrame with iterrows() or iter_rows() method."
        )


def results_to_dataframe(
    results: list[dict[str, Any]],
    *,
    text_key: str = "transformed_text",
) -> Any:
    """Convert orchestrator results to a pandas DataFrame.

    Requires ``pandas`` to be installed.

    Parameters
    ----------
    results : list[dict]
        List of orchestrator result dicts.
    text_key : str
        Key name for the transformed text column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per result.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "results_to_dataframe() requires pandas. "
            "Install with: pip install pandas"
        ) from None

    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {}

        # Metadata
        for k, v in result.get("metadata", {}).items():
            row[k] = v

        # Transformed text
        transformed = result.get("transformed_payload", {})
        if isinstance(transformed, dict):
            row[text_key] = transformed.get("text", str(transformed))
        else:
            row[text_key] = str(transformed)

        # Detection summary
        findings = result.get("ensemble_findings", [])
        row["entities_found"] = len(findings) if isinstance(findings, list) else 0

        envelope = result.get("confidence_envelope", {})
        if isinstance(envelope, dict):
            row["confidence_score"] = envelope.get("score", 0.0)
            row["risk_level"] = envelope.get("risk_level", "unknown")

        rows.append(row)

    return pd.DataFrame(rows)


def _get_columns(df: Any) -> list[str] | None:
    """Extract column names from a DataFrame-like object."""
    if hasattr(df, "columns"):
        cols = df.columns
        if hasattr(cols, "tolist"):
            return list(cols.tolist())
        return list(cols)
    return None


def _safe_value(v: Any) -> Any:
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if hasattr(v, "item"):
        return v.item()
    return v
