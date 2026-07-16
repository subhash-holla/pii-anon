"""Regression tests for the upheld findings from the round-15 (certified-path) close."""

from __future__ import annotations

import re

from pii_anon.assurance import (
    AssuranceConfig,
    PipelineAdapter,
    run_assurance_report,
    verify_report_files,
    verify_reproduction,
)
from pii_anon.assurance.__main__ import main as cli_main

_RX = re.compile(r"\S+@\S+")


def _records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _email_pipeline(name="acme"):
    return PipelineAdapter(
        name=name, detect=lambda t: [("EMAIL_ADDRESS", m.start(), m.end()) for m in _RX.finditer(t)],
        transform=lambda t: _RX.sub("[E]", t), version="1", deterministic=True,
    )


# --- #1 the external verification mode now EXISTS and works -----------------------
def test_verify_reproduction_match_mismatch_and_fail_closed() -> None:
    a = {"provenance": {"dataset_fingerprint": "F",
                        "user_pipeline": {"output_hash": "O", "transform_output_hash": "T"}}}
    assert verify_reproduction(a, a).reproduced is True

    mismatch = {"provenance": {"dataset_fingerprint": "F",
                               "user_pipeline": {"output_hash": "DIFFERENT", "transform_output_hash": "T"}}}
    r = verify_reproduction(a, mismatch)
    assert r.reproduced is False and r.checks["output_hash"] is False

    missing = {"provenance": {"dataset_fingerprint": None, "user_pipeline": {}}}
    assert verify_reproduction(missing, missing).reproduced is False  # fail closed on missing hash


def test_deterministic_run_verifies(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    run_assurance_report(AssuranceConfig(records=_records(60), pipeline=_email_pipeline(),
                                         dimensions=("detection",), outputs=("json",), out_dir=str(a),
                                         seed=7, n_resamples=200))
    run_assurance_report(AssuranceConfig(records=_records(60), pipeline=_email_pipeline(),
                                         dimensions=("detection",), outputs=("json",), out_dir=str(b),
                                         seed=7, n_resamples=200))
    result = verify_report_files(a / "report.json", b / "report.json")
    assert result.reproduced is True
    # and via the CLI
    rc = cli_main(["verify", "--report", str(a / "report.json"), "--against", str(b / "report.json")])
    assert rc == 0


def test_nonreproducing_run_fails_verify(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    # two pipelines that differ -> different output_hash -> NOT reproduced
    run_assurance_report(AssuranceConfig(records=_records(60), pipeline=_email_pipeline(),
                                         dimensions=("detection",), outputs=("json",), out_dir=str(a),
                                         seed=7, n_resamples=200))
    weak = PipelineAdapter(name="acme", detect=lambda t: [], transform=lambda t: t, version="1", deterministic=True)
    run_assurance_report(AssuranceConfig(records=_records(60), pipeline=weak,
                                         dimensions=("detection",), outputs=("json",), out_dir=str(b),
                                         seed=7, n_resamples=200))
    assert verify_report_files(a / "report.json", b / "report.json").reproduced is False
    rc = cli_main(["verify", "--report", str(a / "report.json"), "--against", str(b / "report.json")])
    assert rc == 1


# --- minor: TLD-less / intranet emails in labels are scrubbed ---------------------
def test_tld_less_email_in_label_scrubbed(tmp_path) -> None:
    user = _email_pipeline(name="dlp owned by carol@internal")
    run_assurance_report(AssuranceConfig(
        records=_records(60), dataset_name="run for dave.smith@corp", pipeline=user,
        dimensions=("detection",), outputs=("json", "markdown"), out_dir=str(tmp_path),
        seed=1, n_resamples=200))
    for fname in ("report.json", "report.md"):
        c = (tmp_path / fname).read_text(encoding="utf-8")
        assert "carol@internal" not in c and "dave.smith@corp" not in c
