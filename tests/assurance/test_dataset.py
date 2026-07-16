"""Tests for dual-mode dataset loading, fingerprinting, and PII-safe repr."""

from __future__ import annotations

import json

import pytest

from pii_anon.assurance.claim_strength import MeasurementMode
from pii_anon.assurance.dataset import (
    AssuranceRecord,
    GoldSpan,
    compute_fingerprint,
    load_jsonl,
    load_records,
)

_LABELED = [
    {"id": "r1", "text": "Email alice at a@b.com", "labels": [{"entity_type": "EMAIL", "start": 15, "end": 22}]},
    {"id": "r2", "text": "Call Bob now", "labels": [["PERSON", 5, 8]]},
]
_UNLABELED = [
    {"id": "r1", "text": "Email alice at a@b.com"},
    {"id": "r2", "text": "Call Bob now"},
]


def test_labeled_dataset_is_labeled_mode() -> None:
    ds = load_records(_LABELED)
    assert ds.mode is MeasurementMode.LABELED
    assert len(ds.records) == 2
    assert ds.fingerprint


def test_any_missing_labels_falls_back_to_silver() -> None:
    rows = [_LABELED[0], _UNLABELED[1]]  # one labeled, one not
    ds = load_records(rows)
    assert ds.mode is MeasurementMode.SILVER  # never blend -> silver


def test_unlabeled_is_silver() -> None:
    ds = load_records(_UNLABELED)
    assert ds.mode is MeasurementMode.SILVER


def test_labels_parse_both_forms() -> None:
    ds = load_records(_LABELED)
    assert ds.records[0].labels == (GoldSpan("EMAIL", 15, 22),)
    assert ds.records[1].labels == (GoldSpan("PERSON", 5, 8),)


def test_fingerprint_deterministic_and_data_bound() -> None:
    a = compute_fingerprint(load_records(_LABELED).records)
    b = compute_fingerprint(load_records(_LABELED).records)
    assert a == b
    mutated = [dict(_LABELED[0], text="Email alice at z@b.com"), _LABELED[1]]
    assert compute_fingerprint(load_records(mutated).records) != a


def test_record_repr_does_not_leak_text() -> None:
    r = AssuranceRecord(record_id="r1", text="SSN 123-45-6789 of John Smith")
    assert "123-45-6789" not in repr(r)
    assert "John Smith" not in repr(r)


def test_dataset_repr_does_not_leak_text() -> None:
    ds = load_records([{"id": "r1", "text": "SSN 123-45-6789"}])
    assert "123-45-6789" not in repr(ds)


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError):
        load_records([])


def test_invalid_text_raises() -> None:
    with pytest.raises(ValueError):
        load_records([{"id": "r1", "text": ""}])


def test_load_jsonl(tmp_path) -> None:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _LABELED) + "\n", encoding="utf-8")
    ds = load_jsonl(p)
    assert ds.mode is MeasurementMode.LABELED
    assert len(ds.records) == 2
