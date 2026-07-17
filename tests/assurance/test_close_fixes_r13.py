"""Regression tests for the upheld findings from the round-13 (certified-path) close."""

from __future__ import annotations

import re

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.claim_strength import ClaimStrength
from pii_anon.assurance.dataset import load_records
from pii_anon.assurance.stats import Counts, cluster_bootstrap_ci, pooled_f1
from pii_anon.assurance.synthetic_mirror import synthesize_mirror


# --- #1 a PII-shaped language tag is canonicalized, never copied to the mirror ----
def test_language_field_not_leaked_to_mirror() -> None:
    ds = load_records([
        {"id": "r1", "text": "x a@b.com y", "language": "ng-lee", "labels": [["EMAIL_ADDRESS", 2, 9]]},
        {"id": "r2", "text": "x c@d.com y", "language": "kim", "labels": [["EMAIL_ADDRESS", 2, 9]]},
    ])
    mirror = synthesize_mirror(ds, seed=1)
    assert mirror[0].language == "ng"   # region subtag "lee" dropped
    assert mirror[1].language == "und"  # unknown base -> und
    for rec in mirror:
        assert "lee" not in rec.text and "kim" not in rec.text


# --- #2 too-few bootstrap resamples cannot forge a MEASURED (zero-width CI) --------
def test_low_resamples_yield_no_ci() -> None:
    recs = [Counts(tp=3, fp=1, fn=1) for _ in range(60)]
    _pt, ci = cluster_bootstrap_ci(recs, pooled_f1, seed=1, n_resamples=1)
    assert ci is None  # below the committed floor (would have been zero-width)


def test_low_resamples_not_measured(tmp_path) -> None:
    rows = []
    for i in range(80):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    rx = re.compile(r"\S+@\S+\.\S+")

    def perfect(text: str):
        return [("EMAIL_ADDRESS", m.start(), m.end()) for m in rx.finditer(text)]

    # a perfect, well-powered detector — but only 5 bootstrap resamples
    report = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="perfect", detect=perfect, transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=5))
    res = report.dimensions["detection"]["perfect"]
    assert res.claim_strength is not ClaimStrength.MEASURED  # no forged precision from B=5
    assert any("confidence interval" in r for r in res.reasons)
