"""Regression tests for the upheld findings from the round-11 (certified-path) close."""

from __future__ import annotations

import json

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report


def _email_records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _cfg(tmp_path, **kw):
    d = dict(records=_email_records(40),
             pipeline=PipelineAdapter(name="u", detect=lambda t: [], transform=lambda t: t,
                                      version="1", deterministic=True),
             dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=60)
    d.update(kw)
    return AssuranceConfig(**d)


# --- #1 power thresholds are tighten-only: weakening is rejected ------------------
def test_weakened_ci_halfwidth_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        _cfg(tmp_path, power_max_ci_halfwidth=1e9).validate()


def test_weakened_min_support_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        _cfg(tmp_path, power_min_support=0).validate()


def test_weakened_min_clusters_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        _cfg(tmp_path, power_min_clusters=0).validate()


def test_stricter_thresholds_allowed(tmp_path) -> None:
    # tightening is fine
    _cfg(tmp_path, power_min_support=100, power_max_ci_halfwidth=0.02, power_min_clusters=50).validate()


# --- the runner cannot be tricked into a no-op gate (end-to-end) ------------------
def test_runner_rejects_weakened_gate(tmp_path) -> None:
    from pii_anon.assurance.runner import AssuranceRunError
    # config.validate runs first (before any PII work) -> raises ValueError, not AssuranceRunError
    with pytest.raises((ValueError, AssuranceRunError)):
        run_assurance_report(_cfg(tmp_path, power_max_ci_halfwidth=1e9))


# --- power thresholds are disclosed on the human-facing Markdown surface ----------
def test_power_thresholds_in_markdown(tmp_path) -> None:
    run_assurance_report(_cfg(tmp_path, records=_email_records(60), outputs=("markdown",),
                              power_min_clusters=30))
    md = (tmp_path / "report.md").read_text()
    assert "Power gate" in md
    assert "clusters ≥ 30" in md


# --- the best-effort label-scrubbing limitation is disclosed ----------------------
def test_label_limitation_disclosed(tmp_path) -> None:
    run_assurance_report(_cfg(tmp_path, records=_email_records(60), outputs=("json",)))
    limitations = " ".join(json.loads((tmp_path / "report.json").read_text())["limitations"])
    assert "best-effort scrubbed" in limitations
