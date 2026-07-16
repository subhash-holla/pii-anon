"""Regression tests for the upheld findings from the round-8 (certified-path) close."""

from __future__ import annotations

import json

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.claim_strength import ClaimStrength


def _records(n, spans_per_record=40):
    """n records, each carrying many dense gold spans (to inflate span support)."""
    rows = []
    for i in range(n):
        labels: list[list[object]] = []
        text = ""
        for j in range(spans_per_record):
            tok = f"ID{i:03d}{j:03d}99"
            text += f"rec {i} field {j}: "
            start = len(text)
            text += tok + " "
            labels.append(["EMPLOYEE_ID", start, start + len(tok)])
        rows.append({"id": f"r{i}", "text": text, "labels": labels})
    return rows


# --- #1 few clusters (records) -> ADVISORY, never a MEASURED publishable win ------
def test_few_records_not_measured_even_certified(tmp_path) -> None:
    # 2 records with 40 dense gold spans each: span support=80 >= 50, but only 2 clusters.
    rows = _records(2, spans_per_record=40)

    def perfect(text: str):
        import re
        return [("EMPLOYEE_ID", m.start(), m.end()) for m in re.finditer(r"ID\d{6}99", text)]

    report = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="acme", detect=perfect, transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=200))
    user = report.dimensions["detection"]["acme"]
    assert user.claim_strength is ClaimStrength.ADVISORY
    assert any("cluster" in r for r in user.reasons)
    data = json.loads((tmp_path / "report.json").read_text())
    # no publishable head-to-head win on 2 records
    cmp = data["detection_comparisons"].get("acme")
    if cmp is not None:
        assert cmp["claim_strength"] != "measured"
        assert cmp["significant_holm"] is False


# --- #2 bootstrap B, alpha, family size are recorded for reproducibility ----------
def test_statistics_block_is_recorded(tmp_path) -> None:
    rows = []
    for i in range(60):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="u", detect=lambda t: [], transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=7, n_resamples=137))
    data = json.loads((tmp_path / "report.json").read_text())
    stats = data["statistics"]
    assert stats["bootstrap_resamples"] == 137
    assert stats["alpha"] == 0.05
    assert stats["comparison_family_size"] >= 1
    assert stats["seed"] == 7
    # methodology states B so the CI/p-values are recomputable
    methodology = " ".join(data["methodology"])
    assert "B=137" in methodology
