"""Regression tests for the upheld findings from the round-5 adversarial close."""

from __future__ import annotations

import json

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adjudication import RecordEval
from pii_anon.assurance.claim_strength import ClaimStrength, DimensionResult
from pii_anon.assurance.provenance import build_provenance, provenance_gate
from pii_anon.assurance.report import AssuranceReport, NoFabricationError
from pii_anon.assurance.stats import Counts, cluster_bootstrap_ci, pooled_f1


def _prov(**kw):
    base = dict(dataset_name="d", dataset_fingerprint="abc", seed=1, pii_anon_version="0",
                user_pipeline_name="u", user_pipeline_version="1", user_pipeline_deterministic=True,
                user_pipeline_output_hash="h", timestamp="t")
    base.update(kw)
    return build_provenance(**base)


def _report_with_metadata(meta) -> AssuranceReport:
    res = DimensionResult("detection", "u", ClaimStrength.ADVISORY, value=0.5, support=10, metadata=meta)
    return AssuranceReport("d", "abc", "labeled", "gold", 10, ("u", "pii-anon"), "pii-anon", _prov(),
                           {"detection": {"u": res}})


# --- #1 bool in a unit-score field is rejected (flat + nested) --------------------
def test_bool_in_unit_field_flat_rejected() -> None:
    with pytest.raises(NoFabricationError):
        _report_with_metadata({"precision": True}).to_dict()


def test_bool_in_unit_field_nested_rejected() -> None:
    with pytest.raises(NoFabricationError):
        _report_with_metadata({"per_entity_type": {"EMAIL": {"precision": True, "recall": False}}}).to_dict()


def test_bool_in_nonunit_flag_ok() -> None:
    d = _report_with_metadata({"lower_bound": True, "direction": "lower_is_better"}).to_dict()
    assert d["dimensions"]["detection"]["u"]["metadata"]["lower_bound"] is True


# --- #2 determinism probe catches a per-call flaky detector ----------------------
def test_multi_pass_probe_catches_flaky(tmp_path) -> None:
    counts: dict[str, int] = {}

    def flaky(text: str):
        counts[text] = counts.get(text, 0) + 1
        return [("EMAIL_ADDRESS", 0, 3)] if counts[text] % 2 == 1 else []

    rows = [{"id": f"r{i}", "text": f"Record {i} contact here now", "labels": [["EMAIL_ADDRESS", 0, 3]]}
            for i in range(60)]
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="flaky", detect=flaky, transform=lambda t: t, deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=50))
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["provenance"]["user_pipeline"]["deterministic"] is False


# --- #3 transform-only pipeline binds the provenance descriptor via transform hash
def test_transform_only_binds_via_transform_hash() -> None:
    p = _prov(user_pipeline_version="", user_pipeline_output_hash=None, user_pipeline_transform_hash="thash")
    _ok, reasons = provenance_gate(p)
    assert not any("descriptor" in r for r in reasons)


def test_no_binding_descriptor_fails() -> None:
    p = _prov(user_pipeline_version="", user_pipeline_output_hash=None, user_pipeline_transform_hash=None)
    _ok, reasons = provenance_gate(p)
    assert any("descriptor" in r for r in reasons)


# --- minor: RecordEval repr never echoes record_id -------------------------------
def test_recordeval_repr_elides_record_id() -> None:
    r = RecordEval(record_id="patient_jane_secret", original_text="hi there", language="en",
                   reference=(), pred_by_system={"u": ()}, transform_by_system={"u": None})
    assert "patient_jane" not in repr(r)


# --- minor: n_resamples guards ---------------------------------------------------
def test_config_rejects_bad_n_resamples() -> None:
    cfg = AssuranceConfig(records=[{"id": "r", "text": "hello world"}],
                          pipeline=PipelineAdapter(name="u", detect=lambda t: []), n_resamples=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_cluster_bootstrap_zero_resamples_no_ci() -> None:
    _pt, ci = cluster_bootstrap_ci([Counts(1, 0, 0)] * 10, pooled_f1, seed=1, n_resamples=0)
    assert ci is None  # fail closed, no fabricated zero-width CI
