"""Dual-mode dataset loading with PII-safe handling + a data-bound fingerprint.

The raw text lives ONLY in memory (``AssuranceRecord.text``). It is never
serialized; the report carries the dataset *fingerprint* (a sha256), never the
data (AX-001 + the reproducibility claim in §9 of the design spec). Every
raw-bearing dataclass overrides ``__repr__`` to elide text.

Mode rule (red-team: never blend claim strengths): the dataset is ``LABELED``
(publishable path) IFF *every* record carries gold labels; otherwise ``SILVER``
(advisory path). A mixed dataset is treated as silver — a partial-gold subset is
not promoted to a publishable headline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .claim_strength import MeasurementMode


@dataclass(frozen=True)
class GoldSpan:
    """A ground-truth PII span (0-indexed, half-open)."""

    entity_type: str
    start: int
    end: int


@dataclass
class AssuranceRecord:
    """One dataset record. ``text`` is RAW PII — in-memory only, never serialized."""

    record_id: str
    text: str
    language: str = "en"
    labels: tuple[GoldSpan, ...] | None = None  # None => unlabeled
    group: str | None = None  # optional strata for fairness (e.g. locale, cohort)

    def __repr__(self) -> str:  # defensive: never echo raw text / user-controlled fields
        n_labels = "unlabeled" if self.labels is None else f"{len(self.labels)} labels"
        return f"AssuranceRecord(len(text)={len(self.text)}, {n_labels})"


@dataclass(frozen=True)
class AssuranceDataset:
    """A loaded dataset + its mode + fingerprint. Holds raw records in memory."""

    records: tuple[AssuranceRecord, ...]
    mode: MeasurementMode
    fingerprint: str
    name: str = "user-dataset"

    def __repr__(self) -> str:
        return (
            f"AssuranceDataset(name={self.name!r}, n={len(self.records)}, "
            f"mode={self.mode.value}, fingerprint={self.fingerprint[:12]}…)"
        )

    @property
    def has_groups(self) -> bool:
        return any(r.group is not None for r in self.records)

    @property
    def languages(self) -> set[str]:
        return {r.language for r in self.records}


def _parse_labels(raw: Any, text_len: int) -> tuple[GoldSpan, ...] | None:
    """Parse a record's labels from dict-form or list-form. ``None``/absent → None.

    A malformed OR out-of-range label is DROPPED (mirrors the predictor span-normalization
    contract: ``0 <= start < end <= text_len``), never raised — and a parse error never
    echoes the offending value (PII-leak via traceback). The range bound also rejects a
    huge-int coordinate that would otherwise crash fingerprinting (``json.dumps`` of a
    >4300-digit int)."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return ()
    spans: list[GoldSpan] = []
    for item in raw:
        try:
            if isinstance(item, Mapping):
                etype = str(item.get("entity_type", item.get("type", "")))
                start = int(item["start"])
                end = int(item["end"])
            else:  # [type, start, end] tuple/list
                etype, start, end = str(item[0]), int(item[1]), int(item[2])
        except Exception:  # noqa: BLE001 - drop malformed label; never echo its value
            continue
        if etype and 0 <= start < end <= text_len:  # bounded to the record's text
            spans.append(GoldSpan(etype, start, end))
    return tuple(spans)


def _safe_str(value: Any, fallback: str) -> str:
    """str() that survives a huge int (``str(10**5000)`` raises ValueError) — a
    malformed scalar field degrades to a fallback rather than aborting the whole load."""
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return fallback


def _record_from_dict(row: Mapping[str, Any], index: int) -> AssuranceRecord:
    default_id = f"rec-{index + 1:06d}"
    record_id = _safe_str(row.get("record_id", row.get("id", default_id)), default_id)
    text = row.get("text")
    if not isinstance(text, str) or not text:
        # never echo record_id (may itself carry PII) — use the positional index
        raise ValueError(f"record #{index + 1} has empty/invalid text")
    language = _safe_str(row.get("language", "en"), "und")
    labels = _parse_labels(row.get("labels", row.get("spans")), len(text))
    group = row.get("group")
    return AssuranceRecord(
        record_id=record_id,
        text=text,
        language=language,
        labels=labels,
        group=_safe_str(group, "[group]") if group is not None else None,
    )


def _field(h: Any, value: str) -> None:
    """Length-prefixed field update so no byte sequence in one field can be confused
    with a field boundary (a plain NUL separator collides when text itself has NULs).

    Encodes with ``errors="replace"`` so a lone UTF-16 surrogate (which occurs in real
    scraped/corrupted data) cannot raise mid-run inside the fingerprint — deterministic
    and crash-free on adversarial text."""
    raw = value.encode("utf-8", "replace")
    h.update(len(raw).to_bytes(8, "big"))
    h.update(raw)


def compute_fingerprint(records: Sequence[AssuranceRecord]) -> str:
    """sha256 over canonical content. Data-bound (changes if data changes) and
    one-way (safe to emit). Record order is part of the identity. Every variable-length
    field is length-prefixed so distinct datasets cannot collide via separator bytes."""
    h = hashlib.sha256()
    for r in records:
        _field(h, r.record_id)
        _field(h, r.text)
        _field(h, r.language)
        _field(h, r.group or "")
        if r.labels is None:
            _field(h, "NONE")
        else:
            _field(h, json.dumps(
                [(s.entity_type, s.start, s.end) for s in r.labels], sort_keys=True
            ))
    return h.hexdigest()


def _build(records: Sequence[AssuranceRecord], name: str) -> AssuranceDataset:
    if not records:
        raise ValueError("dataset is empty")
    # LABELED (publishable) requires every record to carry a labels field AND the dataset
    # to contain at least one gold span — an all-empty/all-malformed label set is treated
    # as SILVER so the silver self-reference disclosure is not silently omitted.
    all_labeled = all(r.labels is not None for r in records) and any(r.labels for r in records)
    mode = MeasurementMode.LABELED if all_labeled else MeasurementMode.SILVER
    return AssuranceDataset(
        records=tuple(records),
        mode=mode,
        fingerprint=compute_fingerprint(records),
        name=name,
    )


def load_records(rows: Iterable[Mapping[str, Any]], *, name: str = "user-dataset") -> AssuranceDataset:
    """Build a dataset from in-memory dict rows."""
    records = [_record_from_dict(row, i) for i, row in enumerate(rows)]
    return _build(records, name)


def load_jsonl(path: str | Path, *, name: str | None = None, max_records: int | None = None) -> AssuranceDataset:
    """Load a JSONL dataset file. Each line: {id|record_id, text, language?, labels|spans?, group?}."""
    p = Path(path)
    rows: list[Mapping[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_records is not None and len(rows) >= max_records:
                break
    # Default to a generic name, NOT the raw filename stem — a filename like
    # `patient_jane_ssn_123456789.jsonl` would otherwise leak that identifier into
    # every artifact. An explicit `name` (scrubbed downstream) overrides.
    return load_records(rows, name=name or "user-dataset")
