"""Regression tests for the upheld findings from the round-14 (certified-path) close."""

from __future__ import annotations

import re

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.claim_strength import ClaimStrength
from pii_anon.assurance.report import NoFabricationError, _sanitize

_RX = re.compile(r"\S+@\S+\.\S+")


def _records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _detect(text):
    return [("EMAIL_ADDRESS", m.start(), m.end()) for m in _RX.finditer(text)]


# --- #1 a lone surrogate in a user label must not crash the write -----------------
@pytest.mark.parametrize("field", ["name", "version", "dataset_name"])
def test_lone_surrogate_in_label_completes(tmp_path, field) -> None:
    name = "acme\udc80dlp" if field == "name" else "acme"
    version = "v1\udc80" if field == "version" else "v1"
    dataset_name = "data\udc80set" if field == "dataset_name" else None
    pipe = PipelineAdapter(name=name, detect=_detect, transform=lambda t: t, version=version, deterministic=True)
    run_assurance_report(AssuranceConfig(
        records=_records(60), dataset_name=dataset_name, pipeline=pipe,
        dimensions=("detection",), outputs=("json", "markdown"), out_dir=str(tmp_path),
        seed=1, n_resamples=200))
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()


# --- OBSERVATION #1 hardened: deny-list range-checks any float metadata key -------
def test_sanitize_denylist_gates_unknown_unit_key() -> None:
    with pytest.raises(NoFabricationError):
        _sanitize({"survival_rate": 1.7}, "ctx")  # >1 under a NEW key is now gated
    with pytest.raises(NoFabricationError):
        _sanitize({"per_entity": {"X": {"leak": 1.4}}}, "ctx")  # nested too
    with pytest.raises(NoFabricationError):
        _sanitize({"precision": True}, "ctx")  # bool-as-score still rejected
    assert _sanitize({"lower_bound": True}, "ctx") == {"lower_bound": True}  # flag kept
    assert _sanitize({"effect_size_vs_baseline": -3.5}, "ctx")["effect_size_vs_baseline"] == -3.5  # unbounded


# --- OBSERVATION #2 hardened: certification flag has a single source of truth ------
def test_certification_flag_single_source(tmp_path, monkeypatch) -> None:
    # patching ONLY the provenance module flips the runner's demotion gate
    import pii_anon.assurance.provenance as prov_mod
    monkeypatch.setattr(prov_mod, "HARNESS_CLOSE_CERTIFIED", False)
    report = run_assurance_report(AssuranceConfig(
        records=_records(80), pipeline=PipelineAdapter(name="perfect", detect=_detect, transform=lambda t: t,
                                                       version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=200))
    pa = report.dimensions["detection"]["pii-anon"]
    assert pa.claim_strength is ClaimStrength.ADVISORY  # demoted by the provenance flag alone
    assert any("adversarial close" in r for r in pa.reasons)
