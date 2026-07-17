"""Regression tests for the upheld findings from the round-9 (certified-path) close."""

from __future__ import annotations

import json
import re

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adapters import pii_anon_adapter
from pii_anon.assurance.pii_egress import scrub_label

_EMAIL = re.compile(r"\S+@\S+\.\S+")


def _email_records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _email_pipeline(name="acme"):
    return PipelineAdapter(
        name=name, detect=lambda t: [("EMAIL_ADDRESS", m.start(), m.end()) for m in _EMAIL.finditer(t)],
        transform=lambda t: _EMAIL.sub("[EMAIL]", t), version="1", deterministic=True,
    )


# --- #1 case-altering transform must score ~100% leakage (PII still readable) -----
def test_case_altering_transform_scores_leak(tmp_path) -> None:
    rows = []
    for i in range(60):
        name = "Jonathan Smith"
        text = f"Record {i}: patient {name} was admitted."
        st = text.index(name)
        rows.append({"id": f"r{i}", "text": text, "labels": [["PERSON_NAME", st, st + len(name)]]})
    user = PipelineAdapter(name="lower", detect=lambda t: [], transform=lambda t: t.lower(), deterministic=True)
    rep = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=user, dimensions=("leakage",), outputs=("json",),
        out_dir=str(tmp_path), seed=1, n_resamples=80))
    # the name is fully readable after .lower() -> must NOT score 0.0
    assert rep.dimensions["leakage"]["lower"].value == 1.0


# --- #2 round-trip on the shipped synthetic mirror must not crash -----------------
def test_synthetic_mirror_roundtrip_no_crash(tmp_path) -> None:
    out_a = tmp_path / "a"
    run_assurance_report(AssuranceConfig(
        records=_email_records(60), pipeline=_email_pipeline(), dimensions=("detection",),
        outputs=("json",), out_dir=str(out_a), seed=7, n_resamples=60))
    mirror = out_a / "repro-bundle" / "synthetic-mirror.jsonl"
    mirror_rows = [json.loads(line) for line in mirror.read_text().splitlines() if line.strip()]
    # feed the shipped mirror back through the runner with JSON output — must not crash
    out_b = tmp_path / "b"
    run_assurance_report(AssuranceConfig(
        records=mirror_rows, pipeline=_email_pipeline(), dimensions=("detection",),
        outputs=("json",), out_dir=str(out_b), seed=7, n_resamples=60))
    assert (out_b / "report.json").is_file()


# --- minor: hyphenated name in a label is scrubbed --------------------------------
def test_scrub_label_catches_hyphenated_name() -> None:
    det = pii_anon_adapter().detect
    out = scrub_label("export-by-John-Smith-v2", det)
    assert "John" not in out and "Smith" not in out


# --- minor: a huge-int seed is rejected, not crashed ------------------------------
def test_huge_seed_rejected() -> None:
    cfg = AssuranceConfig(records=[{"id": "r", "text": "hello world"}],
                          pipeline=PipelineAdapter(name="u", detect=lambda t: []), seed=10**5000)
    with pytest.raises(ValueError):
        cfg.validate()
