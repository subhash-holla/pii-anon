"""Regression tests for the upheld findings from the Phase-2 adversarial close."""

from __future__ import annotations

import json

import pytest

from pii_anon.assurance.adjudication import RecordEval, Substrate
from pii_anon.assurance.assessors import assess_fairness, assess_reidentification, assess_utility
from pii_anon.assurance.claim_strength import (
    ClaimStrength,
    DimensionResult,
    MeasurementMode,
    PowerThresholds,
)
from pii_anon.assurance.provenance import build_provenance
from pii_anon.assurance.render import render_summary
from pii_anon.assurance.report import AssuranceReport, NoFabricationError
from pii_anon.eval_framework.attacks.reid import ANTI_ANONYMITY_CAVEAT

_THR = PowerThresholds()


def _prov():
    return build_provenance(
        dataset_name="d", dataset_fingerprint="abc", seed=1, pii_anon_version="1",
        user_pipeline_name="u", user_pipeline_version="1", user_pipeline_deterministic=True,
        user_pipeline_output_hash="h", timestamp="t",
    )


# --- SHOWSTOPPER: empty/content-destroying transform must NOT score MEASURED-perfect utility ---
def test_empty_transform_utility_is_not_assessable_not_phantom_perfect() -> None:
    recs = [
        RecordEval(f"r{i}", f"Ticket {i}: customer Alice{i} email a{i}@x.example here ok.", "en",
                   (("EMAIL", 30, 40),), {"u": (), "pii-anon": ()}, {"u": "", "pii-anon": "masked out"})
        for i in range(60)
    ]
    sub = Substrate(tuple(recs), ("u", "pii-anon"), MeasurementMode.LABELED, "gold")
    res = assess_utility(sub, thresholds=_THR, seed=1, n_resamples=300)["u"]
    assert res.claim_strength is ClaimStrength.NOT_ASSESSABLE
    assert res.value is None  # never a phantom-perfect 1.0


def test_wholly_pii_record_utility_not_phantom_perfect() -> None:
    # round-3 sibling: a wholly-PII record has an EMPTY non-PII skeleton, on which the preservation
    # metric returns 1.0 regardless of output -> a zero-anonymization echo must NOT score MEASURED 1.0
    txt = "Johnathan Smith Esquire"
    ref = (("PERSON_NAME", 0, len(txt)),)
    recs = [
        RecordEval(f"r{i}", txt, "en", ref, {"u": ref, "pii-anon": ref}, {"u": txt, "pii-anon": txt})
        for i in range(60)
    ]
    sub = Substrate(tuple(recs), ("u", "pii-anon"), MeasurementMode.LABELED, "gold")
    res = assess_utility(sub, thresholds=_THR, seed=1, n_resamples=300)["u"]
    assert res.claim_strength is ClaimStrength.NOT_ASSESSABLE
    assert res.value is None


def test_content_destroying_transform_utility_not_assessable() -> None:
    # output far shorter than input (destroys content) is unscoreable, never a high utility
    recs = [
        RecordEval(f"r{i}", f"A long ticket {i} with lots of preserved non-PII context here.", "en",
                   (), {"u": (), "pii-anon": ()}, {"u": "x", "pii-anon": "ok"})
        for i in range(60)
    ]
    sub = Substrate(tuple(recs), ("u", "pii-anon"), MeasurementMode.LABELED, "gold")
    res = assess_utility(sub, thresholds=_THR, seed=1, n_resamples=300)["u"]
    assert res.claim_strength is ClaimStrength.NOT_ASSESSABLE


# --- MAJOR: a detector matching no gold must not be "perfectly fair" (phantom-0) ---
def test_zero_recall_detector_fairness_not_assessable() -> None:
    recs = [
        RecordEval(f"r{i}", f"text {i} here", "en" if i % 2 == 0 else "de",
                   (("EMAIL", 0, 4),), {"u": (("WRONG", 100, 200),), "pii-anon": ()},
                   {"u": "x", "pii-anon": "x"})
        for i in range(160)
    ]
    sub = Substrate(tuple(recs), ("u", "pii-anon"), MeasurementMode.LABELED, "gold")
    res = assess_fairness(sub, min_support=50)["u"]
    assert res.claim_strength is ClaimStrength.NOT_ASSESSABLE  # never a 0-gap "fair" PASS


# --- MAJOR: the numeric gate rejects a forged bool under ANY new score key ---
@pytest.mark.parametrize("metadata", [
    {"reid_recall": True},
    {"reid_precision": True},
    {"reid_success_rate": True},
    {"overall_coverage_ratio": True},
    {"gap_threshold": True},
    {"per_standard_coverage": {"nist": True}},
    {"per_group_recall": {"en": True}},
])
def test_numeric_gate_rejects_forged_bool_in_new_score_keys(metadata) -> None:
    res = DimensionResult("reidentification", "u", ClaimStrength.ADVISORY, value=0.5, support=60,
                          metadata=metadata)
    report = AssuranceReport("d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon",
                             _prov(), {"reidentification": {"u": res}})
    with pytest.raises(NoFabricationError):
        report.to_dict()


def test_numeric_gate_still_keeps_known_bool_flags() -> None:
    # the inverted rule must not reject the legitimate flags
    res = DimensionResult("leakage", "u", ClaimStrength.ADVISORY, value=0.5, support=60,
                          metadata={"lower_bound": True, "type_schema_mismatch": True})
    report = AssuranceReport("d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon",
                             _prov(), {"leakage": {"u": res}})
    d = report.to_dict()
    assert d["dimensions"]["leakage"]["u"]["metadata"]["lower_bound"] is True


# --- MAJOR: the one-page summary certificate must surface the anti-anonymity caveat ---
# --- round 2 MAJOR: re-id DIRECTION — a cross-record identifier permutation leaks readable real
#     PII and must report HIGH re-id risk, never the safest 0.0 (the leak-direction class) --------
def _reid_dir_substrate() -> Substrate:
    def mk(rid, text, spans, tbs):
        return RecordEval(record_id=rid, original_text=text, language="en", reference=tuple(spans),
                          pred_by_system={s: () for s in tbs}, transform_by_system=tbs)
    recs = [
        mk("r0", "NAMEA here today", [("PERSON", 0, 5)],
           {"identity": "NAMEA here today", "strong": "[REDACTED] here today", "swap": "NAMEB here today"}),
        mk("r1", "NAMEB here today", [("PERSON", 0, 5)],
           {"identity": "NAMEB here today", "strong": "[REDACTED] here today", "swap": "NAMEA here today"}),
    ]
    return Substrate(tuple(recs), ("identity", "strong", "swap"), MeasurementMode.LABELED, "gold")


def test_reid_cross_record_permutation_reports_high_risk_not_zero() -> None:
    r = assess_reidentification(_reid_dir_substrate(), thresholds=_THR, seed=0)
    # a transform that swaps real identifiers across records leaks readable real PII -> HIGH risk
    assert r["swap"].value == 1.0
    # monotonicity sanity: identity (readable) HIGH, strong redaction LOW
    assert r["identity"].value == 1.0
    assert r["strong"].value == 0.0
    # the swap (leaks 100% readable PII) must NOT look safer than the strong masker
    assert r["swap"].value > r["strong"].value


def test_reid_character_dotting_not_safer_than_redaction() -> None:
    # round-5: separator-obfuscated PII (A.l.p.h.o.n.s.e) stays human-readable, so a weak
    # dotting transform must NOT report LOWER re-id risk than a strong redactor (no inversion).
    def mk(rid, text, spans, tbs):
        return RecordEval(rid, text, "en", tuple(spans), {s: () for s in tbs}, tbs)
    names = ["Alphonse", "Bernadette", "Cornelius", "Demetria", "Eustace"]
    recs = [
        mk(f"r{i}", f"Patient {nm} here.",
           [("person", len("Patient "), len("Patient ") + len(nm))],
           {"dotted": f"Patient {nm} here.".replace(nm, ".".join(nm)),
            "strong": (f"Patient {nm} here." if i == 0 else f"Patient {nm} here.".replace(nm, "[REDACTED]"))})
        for i, nm in enumerate(names)
    ]
    sub = Substrate(tuple(recs), ("dotted", "strong"), MeasurementMode.LABELED, "gold")
    r = assess_reidentification(sub, thresholds=_THR, seed=1)
    assert r["dotted"].value == 1.0          # all 5 names recoverable from the compacted view
    assert r["dotted"].value >= r["strong"].value  # weaker obfuscation never looks safer


# --- round 2 MAJOR: numeric gate must reject an out-of-[0,1] INT rate (not just the float twin) --
@pytest.mark.parametrize("metadata", [
    {"overall_coverage_ratio": 42},
    {"reid_success_rate": 88},
    {"gap_threshold": -3},
    {"per_group_recall": {"en": 7}},
    {"per_standard_coverage": {"nist": 9}},
])
def test_numeric_gate_rejects_out_of_range_int_rate(metadata) -> None:
    res = DimensionResult("compliance", "u", ClaimStrength.ADVISORY, value=0.5, support=60,
                          metadata=metadata)
    report = AssuranceReport("d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon",
                             _prov(), {"compliance": {"u": res}})
    with pytest.raises(NoFabricationError):
        report.to_dict()


# --- round 4 OBSERVATION: counts (support + count metadata) must fail closed, never serialize a
#     negative / huge-int-DoV value nor crash json.dumps with a raw ValueError -------------------
def test_numeric_gate_rejects_bad_support() -> None:
    # NB: 10**5000 must NOT be a parametrize id (pytest stringifies it -> int->str digit-limit
    # crash at COLLECTION) — loop the battery inside the test instead.
    for support in (-99, 10 ** 5000, True):
        res = DimensionResult("reidentification", "u", ClaimStrength.ADVISORY, value=0.5,
                              support=support, metadata={})
        report = AssuranceReport("d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon",
                                 _prov(), {"reidentification": {"u": res}})
        with pytest.raises(NoFabricationError):
            json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("metadata", [
    {"candidate_set_size": -5}, {"n_targets": -3}, {"linkable_targets": -7},
])
def test_numeric_gate_rejects_negative_count_metadata(metadata) -> None:
    res = DimensionResult("reidentification", "u", ClaimStrength.ADVISORY, value=0.5, support=60,
                          metadata=metadata)
    report = AssuranceReport("d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon",
                             _prov(), {"reidentification": {"u": res}})
    with pytest.raises(NoFabricationError):
        report.to_dict()


def test_numeric_gate_keeps_legitimate_int_counts() -> None:
    res = DimensionResult("reidentification", "u", ClaimStrength.ADVISORY, value=0.5, support=60,
                          metadata={"candidate_set_size": 120, "n_targets": 140, "correct": 12,
                                    "linkable_targets": 53})
    report = AssuranceReport("d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon",
                             _prov(), {"reidentification": {"u": res}})
    d = report.to_dict()
    assert d["dimensions"]["reidentification"]["u"]["metadata"]["candidate_set_size"] == 120


def test_summary_surfaces_anti_anonymity_caveat() -> None:
    report_dict = {
        "generated_at": "t",
        "dataset": {"name": "d", "fingerprint": "abc" * 16, "n_records": 60, "mode": "labeled",
                    "reference_kind": "gold"},
        "systems": ["u", "pii-anon"], "baseline": "pii-anon",
        "provenance": {"pii_anon_version": "1", "seed": 7, "user_pipeline": {"name": "u", "version": "1"}},
        "dimensions": {"reidentification": {
            "u": {"system": "u", "claim_strength": "advisory", "value": 0.0,
                  "caveats": [ANTI_ANONYMITY_CAVEAT]}}},
    }
    out = render_summary(report_dict)["summary.html"]
    assert "ANTI-ANONYMITY CAVEAT" in out
    assert "anonymity" in out.lower()
