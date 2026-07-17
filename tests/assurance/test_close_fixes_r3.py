"""Regression tests for the upheld findings from the round-3 adversarial close."""

from __future__ import annotations

import itertools
import json
import time

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.provenance import output_hash, transform_hash
from pii_anon.assurance.spans import per_type_counts


def _records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _adapter(name, detect, transform):
    return PipelineAdapter(name=name, detect=detect, transform=transform, version="1", deterministic=True)


# --- #1 dataset_name never leaks a PII filename / explicit PII name ---------------
def test_dataset_name_defaults_safe(tmp_path) -> None:
    p = tmp_path / "patient_jane_ssn_123456789.jsonl"
    p.write_text(json.dumps(_records(20)[0]) + "\n" + "\n".join(json.dumps(r) for r in _records(20)[1:]),
                 encoding="utf-8")
    out = tmp_path / "out"
    rep = run_assurance_report(AssuranceConfig(
        dataset=str(p), pipeline=_adapter("u", lambda t: [], lambda t: t),
        dimensions=("detection",), outputs=("json",), out_dir=str(out), seed=1, n_resamples=20))
    assert "123456789" not in rep.dataset_name  # filename stem not used
    assert "123456789" not in (out / "report.json").read_text()


def test_explicit_pii_dataset_name_scrubbed(tmp_path) -> None:
    rep = run_assurance_report(AssuranceConfig(
        records=_records(20), dataset_name="patient_jane_ssn_123456789",
        pipeline=_adapter("u", lambda t: [], lambda t: t), dimensions=("detection",),
        outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=20))
    assert "123456789" not in rep.dataset_name
    assert "123456789" not in (tmp_path / "report.json").read_text()


# --- #2 per_type_counts is O(N), not O(N^2) --------------------------------------
def test_per_type_counts_is_linear() -> None:
    spans = [(f"T{i}", i, i + 1) for i in range(20000)]
    start = time.time()
    result = per_type_counts(spans, spans)
    assert time.time() - start < 3.0  # was ~18s quadratic
    assert len(result) == 20000
    assert all(c.tp == 1 for c in result.values())


# --- #3 period-aligned non-determinism is caught (reversed-order probe) -----------
def test_period_aligned_nondeterminism_caught(tmp_path) -> None:
    counter = itertools.count()

    def period2(text: str):
        return [("EMAIL_ADDRESS", 0, 3)] if next(counter) % 2 == 0 else []

    user = _adapter("p2", period2, lambda t: t)
    run_assurance_report(AssuranceConfig(
        records=_records(60), pipeline=user, dimensions=("detection",), outputs=("json",),
        out_dir=str(tmp_path), seed=1, n_resamples=80))  # even N: forward re-run would alias
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["provenance"]["user_pipeline"]["deterministic"] is False


# --- #4 output_hash is injective across separator-bearing entity types ------------
def test_output_hash_injective_on_separators() -> None:
    a = [[("A", 1, 2), ("B", 3, 4)]]
    b = [[("A:1:2|B", 3, 4)]]  # collides under naive ':'/'|' concatenation
    assert output_hash(a) != output_hash(b)


# --- minor: transform_hash distinguishes None from the literal string ------------
def test_transform_hash_none_distinct_from_string() -> None:
    assert transform_hash([None]) != transform_hash(["\x00none"])
    assert transform_hash([None]) != transform_hash([""])


# --- minor: huge-int record_id does not abort the load ---------------------------
def test_huge_int_record_id_does_not_crash() -> None:
    from pii_anon.assurance.dataset import load_records
    ds = load_records([{"id": 10**5000, "text": "hello world here", "labels": [["EMAIL", 0, 5]]}])
    assert len(ds.records) == 1  # huge-int id degraded to a fallback, no crash
