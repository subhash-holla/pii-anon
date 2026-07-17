"""Regression tests for every upheld finding from the Phase 1 adversarial close."""

from __future__ import annotations

import json

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adjudication import RecordEval, Substrate
from pii_anon.assurance.claim_strength import ClaimStrength, MeasurementMode
from pii_anon.assurance.dataset import load_records
from pii_anon.assurance.synthetic_mirror import synthesize_mirror


def _records_aligned_to_pii_anon(n: int = 60):
    rows = []
    for i in range(n):
        surface = f"user{i}@host.example.com"
        text = f"Record {i}: please contact {surface} now."
        st = text.index(surface)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(surface)]]})
    return rows


def _adapter(name: str, detect, transform):
    return PipelineAdapter(name=name, detect=detect, transform=transform, version="1.0", deterministic=True)


def _cfg(tmp_path, pipeline, **kw):
    defaults = dict(
        records=_records_aligned_to_pii_anon(), pipeline=pipeline,
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=200,
    )
    defaults.update(kw)
    return AssuranceConfig(**defaults)


# --- #2/#4 negative delta CI must serialize (pii-anon outperforms the user) -------
def test_negative_delta_ci_serializes(tmp_path) -> None:
    weak = _adapter("weak", lambda t: [], lambda t: t)
    run_assurance_report(_cfg(tmp_path, weak))
    data = json.loads((tmp_path / "report.json").read_text())
    cmp = data["detection_comparisons"]["weak"]
    assert cmp["delta"] < 0  # pii-anon wins
    assert cmp["ci"][0] <= cmp["ci"][1] <= 0  # negative CI serialized, not crashed/clamped


# --- #1 comparison cannot be an unqualified significant win while uncertified -----
def test_comparison_not_publishable_when_uncertified(tmp_path, monkeypatch) -> None:
    import pii_anon.assurance.provenance as prov_mod
    monkeypatch.setattr(prov_mod, "HARNESS_CLOSE_CERTIFIED", False)
    weak = _adapter("weak", lambda t: [], lambda t: t)
    run_assurance_report(_cfg(tmp_path, weak))
    data = json.loads((tmp_path / "report.json").read_text())
    cmp = data["detection_comparisons"]["weak"]
    assert cmp["claim_strength"] != "measured"
    assert cmp["significant_holm"] is False           # hedged
    assert "significant_holm_raw" in cmp              # raw verdict preserved for transparency


def test_comparison_publishable_when_certified(tmp_path) -> None:
    # With the harness CERTIFIED (shipped default), a genuine powered significant gap
    # IS a publishable head-to-head verdict.
    weak = _adapter("weak", lambda t: [], lambda t: t)
    run_assurance_report(_cfg(tmp_path, weak))
    data = json.loads((tmp_path / "report.json").read_text())
    cmp = data["detection_comparisons"]["weak"]
    assert cmp["claim_strength"] == "measured"
    assert cmp["significant_holm"] is True


# --- #6 dict-shaped predictor spans must not crash the run ------------------------
def test_dict_span_predictor_does_not_crash(tmp_path) -> None:
    dicty = _adapter("dicty", lambda t: [{"type": "EMAIL", "start": 0, "end": 3}], lambda t: t)
    run_assurance_report(_cfg(tmp_path, dicty, records=_records_aligned_to_pii_anon(10), n_resamples=80))
    assert (tmp_path / "report.json").is_file()


# --- #5 malformed gold labels: dropped, never crash, never echo value -------------
def test_malformed_labels_dropped_not_crashed() -> None:
    ds = load_records([
        {"id": "r1", "text": "hello there friend", "labels": [["EMAIL", "bad", "x"], {"type": "X"}]},
        {"id": "r2", "text": "another record here", "labels": [["EMAIL", 0, 5]]},
    ])
    assert ds.records[0].labels == ()  # both malformed labels dropped, no crash


# --- #7 destructive transform: NOT_ASSESSABLE, never 0 leakage --------------------
def test_degenerate_transform_leakage_not_assessable(tmp_path) -> None:
    destroyer = _adapter("destroyer", lambda t: [], lambda t: ".")
    report = run_assurance_report(_cfg(tmp_path, destroyer, dimensions=("leakage",)))
    assert report.dimensions["leakage"]["destroyer"].claim_strength is ClaimStrength.NOT_ASSESSABLE


# --- #9 observed non-determinism caps the run regardless of the declared flag -----
def test_nondeterministic_pipeline_capped(tmp_path) -> None:
    seen: set[str] = set()

    def flaky(text: str):
        if text in seen:
            return []
        seen.add(text)
        return [("EMAIL_ADDRESS", 0, 3)]

    user = _adapter("flaky", flaky, lambda t: t)  # DECLARES deterministic=True (a lie)
    run_assurance_report(_cfg(tmp_path, user))
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["provenance"]["user_pipeline"]["deterministic"] is False  # probe caught the lie


# --- #3 PII in non-text fields is in the egress corpus + not copied to the mirror -
def test_pii_in_group_field_is_in_raw_corpus() -> None:
    rec = RecordEval(
        record_id="r1", original_text="hi there", language="en", reference=(),
        pred_by_system={"u": ()}, transform_by_system={"u": None}, group="alice@secret.example",
    )
    sub = Substrate((rec,), ("u",), MeasurementMode.LABELED, "gold")
    values = [v for _k, v in sub.raw_corpus()]
    assert "alice@secret.example" in values  # group field covered by the egress scan


def test_synthetic_mirror_drops_raw_fields() -> None:
    ds = load_records([{
        "id": "SECRET-ID-9999", "text": "x ZZTOP-TOKEN-4242 y",
        "labels": [["EMAIL_ADDRESS", 2, 18]], "group": "cohort-SECRETGROUP",
    }])
    mirror = synthesize_mirror(ds, seed=1)
    for rec in mirror:
        assert "SECRET-ID" not in rec.record_id
        assert rec.group is None
        assert "ZZTOP" not in rec.text and "SECRETGROUP" not in rec.text
