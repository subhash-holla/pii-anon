"""Regression tests for the upheld findings from the round-6 adversarial close."""

from __future__ import annotations

import json
import re

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.claim_strength import ClaimStrength


def _aligned(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now please for details."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


# --- #1 detector-less user -> detection NOT_ASSESSABLE, NO phantom-0 win ----------
def test_detectorless_user_no_phantom_win(tmp_path, monkeypatch) -> None:
    # Even with the harness CERTIFIED (the eventual end-state), a transform-only user
    # must not hand pii-anon a publishable detection win.
    import pii_anon.assurance.provenance as prov_mod
    monkeypatch.setattr(prov_mod, "HARNESS_CLOSE_CERTIFIED", True)

    user = PipelineAdapter(name="acme", detect=None, transform=lambda t: t, version="1", deterministic=True)
    rep = run_assurance_report(AssuranceConfig(
        records=_aligned(60), pipeline=user, dimensions=("detection",), outputs=("json",),
        out_dir=str(tmp_path), seed=1, n_resamples=80))
    assert rep.dimensions["detection"]["acme"].claim_strength is ClaimStrength.NOT_ASSESSABLE
    data = json.loads((tmp_path / "report.json").read_text())
    assert "acme" not in data["detection_comparisons"]  # no head-to-head winner rendered


# --- #2 short-but-leaky transform is scored as a leak, never laundered to 0 -------
def test_short_leaky_transform_scored_as_leak(tmp_path) -> None:
    rows = _aligned(60)

    def leaky(text: str) -> str:
        # outputs ONLY the email (short — below the degeneracy length cutoff) — a real leak
        m = re.search(r"\S+@\S+\.\S+", text)
        return m.group() if m else ""

    user = PipelineAdapter(name="leaky", detect=lambda t: [], transform=leaky, deterministic=True)
    rep = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=user, dimensions=("leakage",), outputs=("json",),
        out_dir=str(tmp_path), seed=1, n_resamples=80))
    res = rep.dimensions["leakage"]["leaky"]
    assert res.value == 1.0  # all emails survive verbatim -> 100% leakage, not 0.0 / NOT_ASSESSABLE


def test_destroy_everything_transform_not_assessable(tmp_path) -> None:
    # a content-destroying transform with NO surviving PII is still NOT_ASSESSABLE (not 0)
    user = PipelineAdapter(name="destroyer", detect=lambda t: [], transform=lambda t: ".", deterministic=True)
    rep = run_assurance_report(AssuranceConfig(
        records=_aligned(60), pipeline=user, dimensions=("leakage",), outputs=("json",),
        out_dir=str(tmp_path), seed=1, n_resamples=80))
    assert rep.dimensions["leakage"]["destroyer"].claim_strength is ClaimStrength.NOT_ASSESSABLE
