"""Regression tests for the upheld findings from the round-2 adversarial close."""

from __future__ import annotations

import json

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adjudication import RecordEval, Substrate, agreement_report
from pii_anon.assurance.claim_strength import (
    ClaimStrength,
    DimensionResult,
    MeasurementMode,
    PowerThresholds,
    power_gate,
)
from pii_anon.assurance.dataset import compute_fingerprint, load_records
from pii_anon.assurance.provenance import build_provenance
from pii_anon.assurance.report import AssuranceReport, NoFabricationError
from pii_anon.assurance.runner import AssuranceRunError
from pii_anon.assurance.stats import Counts, paired_bootstrap, pooled_f1


def _records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _adapter(name, detect, transform):
    return PipelineAdapter(name=name, detect=detect, transform=transform, version="1", deterministic=True)


def _cfg(tmp_path, pipeline, **kw):
    d = dict(records=_records(), pipeline=pipeline, dimensions=("detection",), outputs=("json",),
             out_dir=str(tmp_path), seed=1, n_resamples=120)
    d.update(kw)
    return AssuranceConfig(**d)


def _report_with_metadata(meta) -> AssuranceReport:
    prov = build_provenance(
        dataset_name="d", dataset_fingerprint="abc", seed=1, pii_anon_version="0",
        user_pipeline_name="u", user_pipeline_version="1", user_pipeline_deterministic=True,
        user_pipeline_output_hash="h", timestamp="t",
    )
    res = DimensionResult("detection", "u", ClaimStrength.ADVISORY, value=0.5, support=10, metadata=meta)
    return AssuranceReport("d", "abc", "labeled", "gold", 10, ("u", "pii-anon"), "pii-anon", prov,
                           {"detection": {"u": res}})


# --- #1 numeric gate range-checks metadata (nested + flat) ------------------------
def test_metadata_out_of_range_nested_rejected() -> None:
    with pytest.raises(NoFabricationError):
        _report_with_metadata({"per_entity_type": {"EMAIL": {"precision": 2.5, "recall": 0.3, "f1": 0.4}}}).to_dict()


def test_metadata_out_of_range_flat_rejected() -> None:
    with pytest.raises(NoFabricationError):
        _report_with_metadata({"precision": 1.7}).to_dict()


def test_metadata_in_range_ok() -> None:
    d = _report_with_metadata({"precision": 0.9, "effect_size_vs_baseline": -3.5}).to_dict()
    assert d["dimensions"]["detection"]["u"]["metadata"]["precision"] == 0.9
    assert d["dimensions"]["detection"]["u"]["metadata"]["effect_size_vs_baseline"] == -3.5  # unbounded ok


# --- #2 PII never leaks via repr or a locals-capturing traceback ------------------
def test_config_repr_does_not_leak_records() -> None:
    pii = "Jane Victim 123-45-6789 jane@vic.example"
    cfg = AssuranceConfig(records=[{"id": "r1", "text": pii, "group": pii}],
                          pipeline=PipelineAdapter(name="u"))
    assert "Jane Victim" not in repr(cfg) and "123-45-6789" not in repr(cfg)


def test_internal_error_is_scrubbed_pii_free(tmp_path, monkeypatch) -> None:
    secret = "ZZSECRET-7777@victim.example"
    rows = [{"id": f"r{i}", "text": f"x {secret} y", "labels": [["EMAIL_ADDRESS", 2, 2 + len(secret)]]}
            for i in range(5)]
    # force an internal failure deep in the run
    import pii_anon.assurance.runner as runner_mod

    def boom(*a, **k):
        raise RuntimeError("internal kaboom")

    monkeypatch.setattr(runner_mod, "assess_detection", boom)
    cfg = AssuranceConfig(records=rows, pipeline=_adapter("u", lambda t: [("EMAIL_ADDRESS", 2, 2 + len(secret))], lambda t: t),
                          dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=5)
    with pytest.raises(AssuranceRunError) as ei:
        run_assurance_report(cfg)
    # message is PII-free
    assert "ZZSECRET" not in str(ei.value)
    # the inner (PII-bearing) frames are suppressed (raise ... from None) and every
    # LIBRARY frame in the surviving traceback has PII-free locals (config elides records).
    tb = ei.value.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if "/src/pii_anon/" in frame.f_code.co_filename:  # library frames only (not the test frame)
            for val in frame.f_locals.values():
                assert "ZZSECRET" not in repr(val), f"PII leaked in {frame.f_code.co_name} locals"
            # the deep assessor frame must NOT be present (chain suppressed)
            assert frame.f_code.co_name not in {"_run_inner", "assess_detection", "boom"}
        tb = tb.tb_next


# --- #3/#4 leakage format/position robustness ------------------------------------
def test_leakage_reformatted_ssn_is_leaked(tmp_path) -> None:
    # gold SSN; user transform reinserts the SSN with different separators -> still a leak
    rows = []
    for i in range(60):
        ssn = f"5{i:02d}-12-3456"
        text = f"Record {i}: ssn {ssn} end"
        st = text.index(ssn)
        rows.append({"id": f"r{i}", "text": text, "labels": [["US_SSN", st, st + len(ssn)]]})

    def reformat(text: str) -> str:
        # remove dashes from any digit run -> "reformats" the SSN but keeps the digits
        import re
        return re.sub(r"(\d)-(\d)", r"\1 \2", text)

    user = _adapter("reformatter", lambda t: [], reformat)
    rep = run_assurance_report(AssuranceConfig(records=rows, pipeline=user, dimensions=("leakage",),
                                               outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=100))
    # reformatted SSNs must NOT score 0% leakage (digit-normalized match catches them)
    assert rep.dimensions["leakage"]["reformatter"].value == 1.0


# --- #4 huge-int label coordinate is dropped (no crash) --------------------------
def test_huge_int_label_dropped() -> None:
    ds = load_records([{"id": "r1", "text": "short text here", "labels": [["EMAIL", 0, 10**5000]]},
                       {"id": "r2", "text": "another record now", "labels": [["EMAIL", 0, 5]]}])
    assert ds.records[0].labels == ()  # out-of-range coord dropped
    assert compute_fingerprint(ds.records)  # does not crash on the huge int


# --- #6 egress structured-token scan is fast on a large no-@ alphanumeric run -----
def test_egress_no_catastrophic_backtracking() -> None:
    from pii_anon.assurance.pii_egress import scan_output
    big = "a1b2c3" * 50_000  # 300k chars, digit-bearing, no '@'
    # must return promptly (linear), not hang
    scan_output("clean report text", [("r1", big)])


# --- #7 transform non-determinism caps the run -----------------------------------
def test_transform_nondeterminism_caps(tmp_path) -> None:
    seen: set[str] = set()

    def flaky_t(text: str) -> str:
        if text in seen:
            return text + " EXTRA"
        seen.add(text)
        return text

    user = _adapter("flaky-t", lambda t: [], flaky_t)  # detect deterministic, transform not
    run_assurance_report(_cfg(tmp_path, user))
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["provenance"]["user_pipeline"]["deterministic"] is False


# --- minor: all-malformed labels -> SILVER ---------------------------------------
def test_all_malformed_labels_is_silver() -> None:
    ds = load_records([{"id": "r1", "text": "hello there", "labels": [["EMAIL", "x", "y"]]},
                       {"id": "r2", "text": "world here", "labels": [["EMAIL", 99, 200]]}])
    assert ds.mode is MeasurementMode.SILVER  # no usable gold -> silver


# --- minor: power gate rejects an inverted CI ------------------------------------
def test_power_gate_rejects_inverted_ci() -> None:
    ok, reasons = power_gate(support=500, n_clusters=500, confidence_interval=(0.9, 0.1), significant=True,
                             thresholds=PowerThresholds(min_support=50, max_ci_halfwidth=0.05))
    assert ok is False
    assert any("inverted" in r for r in reasons)


# --- minor: paired-bootstrap p is floored at 1/B ---------------------------------
def test_paired_p_value_floored() -> None:
    a = [Counts(tp=5, fp=0, fn=0) for _ in range(40)]  # perfect
    b = [Counts(tp=0, fp=0, fn=5) for _ in range(40)]  # misses all
    res = paired_bootstrap(a, b, pooled_f1, seed=1, n_resamples=100)
    assert res.p_value >= 1.0 / 100
    assert res.p_value > 0.0  # never an impossible exact 0


# --- minor: vacuous silver agreement is None, not a fake perfect score ------------
def test_vacuous_agreement_is_none() -> None:
    recs = tuple(
        RecordEval(record_id=f"r{i}", original_text="nothing to see", language="en", reference=(),
                   pred_by_system={"u": (), "pii-anon": ()}, transform_by_system={"u": None, "pii-anon": None})
        for i in range(5)
    )
    sub = Substrate(recs, ("u", "pii-anon"), MeasurementMode.SILVER, "silver")
    rep = agreement_report(sub.records, "u", "pii-anon")
    assert rep.span_jaccard is None  # no spans at all -> undefined, not 1.0
    assert rep.char_kappa is None
