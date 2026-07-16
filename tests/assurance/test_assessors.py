"""Tests for the detection + leakage assessors and silver-mode adjudication."""

from __future__ import annotations

from pii_anon.assurance.adjudication import (
    RecordEval,
    Substrate,
    agreement_report,
    union_reference,
)
from pii_anon.assurance.adapters import redact_spans
from pii_anon.assurance.assessors import assess_detection, assess_leakage
from pii_anon.assurance.claim_strength import ClaimStrength, MeasurementMode, PowerThresholds

_THR = PowerThresholds(min_support=50, max_ci_halfwidth=0.05)


def _make_record(i: int, *, user_detects: bool, user_transform: str | None, redact_user_via=None):
    text = f"Record {i}: email u{i}@host.example here ok."
    surface = f"u{i}@host.example"
    start = text.index(surface)
    ref = (("EMAIL", start, start + len(surface)),)
    user_pred = ref if user_detects else ()
    return RecordEval(
        record_id=f"r{i}",
        original_text=text,
        language="en",
        reference=ref,
        pred_by_system={"user": user_pred, "pii-anon": ref},
        transform_by_system={
            "user": user_transform if user_transform != "REDACT" else redact_spans(text, ref),
            "pii-anon": redact_spans(text, ref),
        },
    )


def _labeled_substrate(n: int = 60, *, user_transform_mode: str = "REDACT") -> Substrate:
    records = []
    for i in range(n):
        ut = None
        if user_transform_mode == "REDACT":
            ut = "REDACT"
        elif user_transform_mode == "NOOP":
            ut = f"Record {i}: email u{i}@host.example here ok."  # identity -> leaks
        elif user_transform_mode == "NONE":
            ut = None
        records.append(_make_record(i, user_detects=(i % 2 == 0), user_transform=ut))
    return Substrate(tuple(records), ("user", "pii-anon"), MeasurementMode.LABELED, "gold")


def test_detection_perfect_baseline_is_measured() -> None:
    sub = _labeled_substrate()
    results, comparisons = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=1)
    pa = results["pii-anon"]
    assert pa.value == 1.0
    assert pa.claim_strength is ClaimStrength.MEASURED
    assert pa.support == 60


def test_detection_weaker_user_is_significantly_worse() -> None:
    sub = _labeled_substrate()
    results, comparisons = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=1)
    user = results["user"]
    assert user.value < 0.8  # misses half -> recall 0.5 -> f1 ~0.667
    assert comparisons["user"].delta < 0  # user worse than pii-anon
    assert comparisons["user"].p_value < 0.05


def test_detection_is_deterministic() -> None:
    sub = _labeled_substrate()
    r1, _ = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=42)
    r2, _ = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=42)
    assert r1["user"].value == r2["user"].value
    assert r1["user"].confidence_interval == r2["user"].confidence_interval


def test_detection_silver_is_never_measured() -> None:
    sub = _labeled_substrate()
    silver = Substrate(sub.records, sub.systems, MeasurementMode.SILVER, "silver")
    results, _ = assess_detection(silver, baseline_system="pii-anon", thresholds=_THR, seed=1)
    assert results["pii-anon"].claim_strength is ClaimStrength.ADVISORY


def test_leakage_perfect_redaction_is_zero() -> None:
    sub = _labeled_substrate(user_transform_mode="REDACT")
    results = assess_leakage(sub, thresholds=_THR, seed=1)
    assert results["pii-anon"].value == 0.0
    assert any("lower bound" in c.lower() or "lower_bound" in str(results["pii-anon"].metadata) for c in results["pii-anon"].caveats)


def test_leakage_noop_transform_is_near_total() -> None:
    sub = _labeled_substrate(user_transform_mode="NOOP")
    results = assess_leakage(sub, thresholds=_THR, seed=1)
    assert results["user"].value == 1.0  # identity transform leaks everything


def test_leakage_no_transform_is_not_assessable() -> None:
    sub = _labeled_substrate(user_transform_mode="NONE")
    results = assess_leakage(sub, thresholds=_THR, seed=1)
    assert results["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE
    assert results["user"].value is None  # never 0


def test_leakage_empty_output_is_not_assessable() -> None:
    records = []
    for i in range(60):
        rec = _make_record(i, user_detects=True, user_transform="")  # degenerate empty
        records.append(rec)
    sub = Substrate(tuple(records), ("user", "pii-anon"), MeasurementMode.LABELED, "gold")
    results = assess_leakage(sub, thresholds=_THR, seed=1)
    assert results["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE


def test_silver_agreement_report() -> None:
    # user detects evens, pii-anon detects all -> partial agreement.
    sub = _labeled_substrate()
    rep = agreement_report(sub.records, "user", "pii-anon")
    assert 0.0 < rep.span_jaccard < 1.0   # partial overlap
    assert rep.both == 30 and rep.only_b == 30 and rep.only_a == 0
    assert rep.char_kappa is not None


def test_union_reference_is_recall_favouring() -> None:
    sub = _labeled_substrate()
    rec = sub.records[1]  # odd -> user missed, pii-anon found
    ref = union_reference(rec.pred_by_system)
    assert len(ref) == 1  # union keeps the span pii-anon found even though user missed it
