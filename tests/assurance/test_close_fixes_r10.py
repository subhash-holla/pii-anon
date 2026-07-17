"""Regression tests for the upheld findings from the round-10 (certified-path) close."""

from __future__ import annotations

import json
import re

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.runner import AssuranceRunError

_EMAIL = re.compile(r"\S+@\S+\.\S+")


def _email_records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


# --- #1 a user-pipeline BaseException is scrubbed (no raw PII leak) ----------------
@pytest.mark.parametrize("exc_factory", [
    lambda text: SystemExit(f"died on: {text}"),
    lambda text: KeyboardInterrupt(),
])
def test_base_exception_is_scrubbed(tmp_path, exc_factory) -> None:
    secret = "ZZSECRET-9000@victim.example"
    rows = [{"id": f"r{i}", "text": f"x {secret} y", "labels": [["EMAIL_ADDRESS", 2, 2 + len(secret)]]}
            for i in range(5)]

    def detect(text):
        raise exc_factory(text)

    cfg = AssuranceConfig(records=rows, pipeline=PipelineAdapter(name="u", detect=detect, transform=lambda t: t),
                          dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=5)
    # the adapter scrubs the BaseException, so the run actually COMPLETES (no leak, no crash)
    run_assurance_report(cfg)
    content = (tmp_path / "report.json").read_text()
    assert "ZZSECRET" not in content


def test_base_exception_outside_adapter_is_pii_free(tmp_path, monkeypatch) -> None:
    secret = "ZZSECRET-9001@victim.example"
    rows = [{"id": f"r{i}", "text": f"x {secret} y", "labels": [["EMAIL_ADDRESS", 2, 2 + len(secret)]]}
            for i in range(5)]
    import pii_anon.assurance.runner as runner_mod

    def boom(*a, **k):
        raise SystemExit(f"internal death: {rows}")  # BaseException carrying raw rows

    monkeypatch.setattr(runner_mod, "assess_detection", boom)
    cfg = AssuranceConfig(records=rows, pipeline=PipelineAdapter(name="u", detect=lambda t: [], transform=lambda t: t),
                          dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=5)
    with pytest.raises(AssuranceRunError) as ei:
        run_assurance_report(cfg)
    assert "ZZSECRET" not in str(ei.value)
    tb = ei.value.__traceback__
    while tb is not None:
        if "/src/pii_anon/" in tb.tb_frame.f_code.co_filename:
            for v in tb.tb_frame.f_locals.values():
                assert "ZZSECRET" not in repr(v)
        tb = tb.tb_next


# --- #2 period-2 toggle pipeline is caught (not MEASURED, deterministic=False) ----
def test_period_two_toggle_caught(tmp_path) -> None:
    state = {"c": 0}

    def detect(text):
        state["c"] += 1
        if state["c"] % 2 == 0:
            return []
        return [("EMAIL_ADDRESS", m.start(), m.end()) for m in _EMAIL.finditer(text)]

    def transform(text):
        out = text
        for _t, s, e in reversed(detect(text)):
            out = out[:s] + "[X]" + out[e:]
        return out

    user = PipelineAdapter(name="parity", detect=detect, transform=transform, version="1", deterministic=True)
    rep = run_assurance_report(AssuranceConfig(
        records=_email_records(60), pipeline=user, dimensions=("detection",), outputs=("json",),
        out_dir=str(tmp_path), seed=42, n_resamples=100))
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["provenance"]["user_pipeline"]["deterministic"] is False
    assert rep.dimensions["detection"]["parity"].claim_strength.value != "measured"


# --- minor: power thresholds are disclosed in the statistics block ----------------
def test_power_thresholds_disclosed(tmp_path) -> None:
    # a STRICTER floor (tighten-only) is allowed and disclosed
    run_assurance_report(AssuranceConfig(
        records=_email_records(60), pipeline=PipelineAdapter(name="u", detect=lambda t: [], transform=lambda t: t,
                                                             version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=60,
        power_min_clusters=30))
    pt = json.loads((tmp_path / "report.json").read_text())["statistics"]["power_thresholds"]
    assert pt["min_clusters"] == 30  # active floor visible in the report
    assert pt["min_support"] == 50


# --- minor: egress scans the FULL record (no silent truncation past 1M) -----------
def test_egress_scans_beyond_one_million_chars() -> None:
    from pii_anon.assurance.pii_egress import scan_output
    secret = "user@victim.example"
    big = ("a " * 600_000) + secret  # secret past the old 1M cutoff
    findings = scan_output("leak " + secret, [("r1", big)])
    assert any(f.kind == "containment" for f in findings)
