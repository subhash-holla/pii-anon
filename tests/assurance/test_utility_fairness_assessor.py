"""Tests for the utility / fairness / compliance assessors (spec §6.4)."""

from __future__ import annotations

from pii_anon.assurance.adapters import redact_spans
from pii_anon.assurance.adjudication import RecordEval, Substrate
from pii_anon.assurance.assessors import assess_compliance, assess_fairness, assess_utility
from pii_anon.assurance.claim_strength import ClaimStrength, MeasurementMode, PowerThresholds

_THR = PowerThresholds()


def _rec(i: int, *, lang: str, user_transform, user_pred_type: str | None) -> RecordEval:
    name = f"Person{i}"
    text = f"Ticket {i}: customer {name} email p{i}@host.example wrote in."
    ns = text.index(name)
    es = text.index(f"p{i}@host.example")
    ref = (("PERSON_NAME", ns, ns + len(name)), ("EMAIL_ADDRESS", es, es + len(f"p{i}@host.example")))
    pred = () if user_pred_type is None else tuple(s for s in ref if s[0] == user_pred_type)
    return RecordEval(
        record_id=f"r{i}", original_text=text, language=lang, reference=ref,
        pred_by_system={"user": pred, "pii-anon": ref},
        transform_by_system={
            "user": user_transform(text, ref),
            "pii-anon": redact_spans(text, ref),
        },
    )


def _sub(n: int, *, user_transform, user_pred_type, mode=MeasurementMode.LABELED) -> Substrate:
    records = [
        _rec(i, lang=("en" if i % 2 == 0 else "de"),
             user_transform=user_transform, user_pred_type=user_pred_type)
        for i in range(n)
    ]
    return Substrate(tuple(records), ("user", "pii-anon"), mode,
                     "gold" if mode is MeasurementMode.LABELED else "silver")


# --- utility ----------------------------------------------------------------
def test_utility_measured_and_preserving_transform_scores_higher() -> None:
    # a transform that only redacts PII preserves the non-PII skeleton well
    sub = _sub(80, user_transform=redact_spans, user_pred_type="EMAIL_ADDRESS")
    res = assess_utility(sub, thresholds=_THR, seed=1, n_resamples=300)
    u = res["user"]
    assert u.claim_strength is ClaimStrength.MEASURED
    assert u.value is not None and 0.0 <= u.value <= 1.0
    assert u.metadata["direction"] == "higher_is_better"


def test_utility_not_assessable_without_transform() -> None:
    sub = _sub(80, user_transform=lambda t, r: None, user_pred_type="EMAIL_ADDRESS")
    res = assess_utility(sub, thresholds=_THR, seed=1, n_resamples=300)
    assert res["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE


def test_utility_advisory_in_silver_mode() -> None:
    sub = _sub(80, user_transform=redact_spans, user_pred_type="EMAIL_ADDRESS",
               mode=MeasurementMode.SILVER)
    res = assess_utility(sub, thresholds=_THR, seed=1, n_resamples=300)
    assert res["user"].claim_strength is ClaimStrength.ADVISORY


# --- fairness ---------------------------------------------------------------
def test_fairness_advisory_with_powered_groups() -> None:
    sub = _sub(160, user_transform=redact_spans, user_pred_type="EMAIL_ADDRESS")
    res = assess_fairness(sub, min_support=50)
    f = res["pii-anon"]  # pii-anon detects everything -> powered en + de groups
    assert f.claim_strength is ClaimStrength.ADVISORY
    assert f.value is not None and 0.0 <= f.value <= 1.0
    assert set(f.metadata["per_group_recall"]) == {"en", "de"}
    assert f.metadata["direction"] == "lower_is_better"


def test_fairness_not_assessable_without_group_metadata() -> None:
    records = []
    for i in range(120):
        r = _rec(i, lang="", user_transform=redact_spans, user_pred_type="EMAIL_ADDRESS")
        records.append(r)
    sub = Substrate(tuple(records), ("user", "pii-anon"), MeasurementMode.LABELED, "gold")
    res = assess_fairness(sub, min_support=50)
    assert res["pii-anon"].claim_strength is ClaimStrength.NOT_ASSESSABLE


def test_fairness_not_assessable_when_system_predicts_nothing() -> None:
    sub = _sub(160, user_transform=redact_spans, user_pred_type=None)
    res = assess_fairness(sub, min_support=50)
    assert res["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE


def test_fairness_not_assessable_in_silver_mode() -> None:
    sub = _sub(160, user_transform=redact_spans, user_pred_type="EMAIL_ADDRESS",
               mode=MeasurementMode.SILVER)
    res = assess_fairness(sub, min_support=50)
    assert all(r.claim_strength is ClaimStrength.NOT_ASSESSABLE for r in res.values())


# --- compliance -------------------------------------------------------------
def test_compliance_advisory_coverage_in_unit_interval() -> None:
    sub = _sub(40, user_transform=redact_spans, user_pred_type="EMAIL_ADDRESS")
    res = assess_compliance(sub)
    pa = res["pii-anon"]
    assert pa.claim_strength is ClaimStrength.ADVISORY  # never a head-to-head winner / MEASURED
    assert pa.value is not None and 0.0 <= pa.value <= 1.0
    assert "per_standard_coverage" in pa.metadata
    assert all(0.0 <= v <= 1.0 for v in pa.metadata["per_standard_coverage"].values())


def test_compliance_not_assessable_without_detections() -> None:
    sub = _sub(40, user_transform=redact_spans, user_pred_type=None)
    res = assess_compliance(sub)
    assert res["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE
