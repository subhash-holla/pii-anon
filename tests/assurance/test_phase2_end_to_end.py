"""Phase-2 end-to-end: all four dimensions + all four outputs, both gates, PII-safety."""

from __future__ import annotations

import json
import re

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report

_EMAIL = re.compile(r"[A-Za-z0-9.]+@[A-Za-z0-9.]+")


def _rows(n: int = 120) -> list[dict]:
    rows = []
    for i in range(n):
        name = f"Person{i} Lastname{i}"
        text = f"Ticket {i}: customer {name} email p{i}@host.example wrote in."
        ns = text.index(name)
        es = text.index(f"p{i}@host.example")
        rows.append({
            "id": f"t{i:04d}", "text": text, "language": "en" if i % 2 == 0 else "de",
            "labels": [["PERSON_NAME", ns, ns + len(name)],
                       ["EMAIL_ADDRESS", es, es + len(f"p{i}@host.example")]],
        })
    return rows


def _email_only_pipe() -> PipelineAdapter:
    return PipelineAdapter(
        name="acme-dlp",
        detect=lambda t: [("EMAIL_ADDRESS", m.start(), m.end()) for m in _EMAIL.finditer(t)],
        transform=lambda t: _EMAIL.sub("[EMAIL]", t),
        version="2.3.1", deterministic=True,
    )


def test_all_dimensions_all_outputs(tmp_path) -> None:
    rep = run_assurance_report(AssuranceConfig(
        records=_rows(), pipeline=_email_only_pipe(), dataset_name="acme-tix",
        dimensions=("detection", "leakage", "reidentification", "utility_fairness_compliance"),
        outputs=("json", "markdown", "html", "summary"),
        out_dir=str(tmp_path), seed=7, n_resamples=300))
    for f in ("report.json", "report.md", "report.html", "summary.html",
              "repro-bundle/provenance.json", "repro-bundle/synthetic-mirror.jsonl"):
        assert (tmp_path / f).is_file(), f
    # the user-facing dimension expanded into utility / fairness / compliance
    assert {"detection", "leakage", "reidentification", "utility", "fairness", "compliance"} == set(rep.dimensions)
    blob = json.loads((tmp_path / "report.json").read_text())
    assert set(blob["dimensions"]) >= {"reidentification", "utility", "fairness", "compliance"}


def test_phase2_egress_gate_blocks_planted_pii_in_html(tmp_path) -> None:
    # a planted distinctive token in a record must never appear in ANY artifact, incl. html/summary
    planted = "Zebediah Q. Worthington-McfakePerson"
    rows = _rows(60)
    rows[0]["text"] = f"Ticket: contact {planted} now."
    rows[0]["labels"] = [["PERSON_NAME", len("Ticket: contact "),
                          len("Ticket: contact ") + len(planted)]]
    # a no-op transform leaves the planted name in the output, but the renderers never emit raw
    # record text, so the artifacts stay clean (and the egress gate is the byte-level backstop).
    rep = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=_email_only_pipe(), dataset_name="planted",
        dimensions=("detection", "leakage", "reidentification", "utility_fairness_compliance"),
        outputs=("json", "markdown", "html", "summary"),
        out_dir=str(tmp_path), seed=3, n_resamples=300))
    assert rep is not None
    for f in ("report.json", "report.md", "report.html", "summary.html"):
        assert planted not in (tmp_path / f).read_text(), f


def test_reid_and_compliance_never_measured_in_output(tmp_path) -> None:
    run_assurance_report(AssuranceConfig(
        records=_rows(), pipeline=_email_only_pipe(), dataset_name="acme-tix",
        dimensions=("reidentification", "utility_fairness_compliance"),
        outputs=("json",), out_dir=str(tmp_path), seed=7, n_resamples=300))
    blob = json.loads((tmp_path / "report.json").read_text())
    for sysmap in (blob["dimensions"]["reidentification"], blob["dimensions"]["compliance"]):
        for r in sysmap.values():
            assert r["claim_strength"] != "measured"  # never publishable head-to-head
