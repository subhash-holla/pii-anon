"""CLI smoke test, synthetic-mirror safety, and numeric-gate rejection."""

from __future__ import annotations

import json

import pytest

from pii_anon.assurance.__main__ import main
from pii_anon.assurance.claim_strength import ClaimStrength, DimensionResult
from pii_anon.assurance.dataset import load_records
from pii_anon.assurance.provenance import build_provenance
from pii_anon.assurance.report import AssuranceReport, NoFabricationError
from pii_anon.assurance.synthetic_mirror import mirror_to_rows, synthesize_mirror


def test_cli_runs_detection_only(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    rows = [
        {"id": f"r{i}", "text": f"Record {i}: contact u{i}@host.example for info.",
         "labels": [["EMAIL", 20, 20 + len(f"u{i}@host.example")]]}
        for i in range(10)
    ]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = main([
        "--dataset", str(data),
        "--pipeline-entrypoint", "pii_anon.eval_framework.first_party:pii_anon_predictor",
        "--pipeline-name", "first-party-clone",
        "--dimensions", "detection,leakage",
        "--outputs", "json",
        "--out", str(out),
        "--seed", "1",
    ])
    assert rc == 0
    assert (out / "report.json").is_file()


def test_cli_requires_an_entrypoint(tmp_path) -> None:
    data = tmp_path / "d.jsonl"
    data.write_text('{"id":"r1","text":"hi there friend"}\n', encoding="utf-8")
    rc = main(["--dataset", str(data), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_synthetic_mirror_contains_no_raw_text() -> None:
    distinctive = "ZZQX-SECRET-TOKEN-7777"
    ds = load_records([
        {"id": "r1", "text": f"the value is {distinctive} here",
         "labels": [["SECRET", 13, 13 + len(distinctive)]]},
        {"id": "r2", "text": f"another {distinctive} appears", "labels": [["SECRET", 8, 8 + len(distinctive)]]},
    ])
    mirror = synthesize_mirror(ds, seed=7)
    for rec in mirror:
        assert "ZZQX-SECRET" not in rec.text  # generator never copies raw text
        # labels point at the synthetic value
        for span in rec.labels or ():
            assert rec.text[span.start:span.end].strip()
    rows = mirror_to_rows(mirror)
    assert all("ZZQX-SECRET" not in r["text"] for r in rows)


def _minimal_report(value) -> AssuranceReport:
    prov = build_provenance(
        dataset_name="d", dataset_fingerprint="abc123", seed=1, pii_anon_version="0.0.0",
        user_pipeline_name="u", user_pipeline_version="1", user_pipeline_deterministic=True,
        user_pipeline_output_hash="hash", timestamp="2026-06-17T00:00:00Z",
    )
    res = DimensionResult(
        dimension="detection", system="u", claim_strength=ClaimStrength.ADVISORY, value=value, support=10,
    )
    return AssuranceReport(
        dataset_name="d", dataset_fingerprint="abc123", mode="labeled", reference_kind="gold",
        n_records=10, systems=("u", "pii-anon"), baseline="pii-anon", provenance=prov,
        dimensions={"detection": {"u": res}},
    )


def test_numeric_gate_rejects_nan() -> None:
    with pytest.raises(NoFabricationError):
        _minimal_report(float("nan")).to_dict()


def test_numeric_gate_rejects_out_of_range() -> None:
    with pytest.raises(NoFabricationError):
        _minimal_report(1.5).to_dict()


def test_numeric_gate_allows_none_value() -> None:
    d = _minimal_report(None).to_dict()
    assert d["dimensions"]["detection"]["u"]["value"] is None


def test_numeric_gate_rejects_nan_in_metadata() -> None:
    rep = _minimal_report(0.5)
    rep.dimensions["detection"]["u"].metadata["bad"] = float("inf")
    with pytest.raises(NoFabricationError):
        rep.to_dict()
