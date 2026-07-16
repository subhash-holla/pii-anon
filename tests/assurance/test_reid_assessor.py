"""Tests for the re-identification assessor (honestly-labeled stand-in, spec §6.3)."""

from __future__ import annotations

from pii_anon.assurance.adjudication import RecordEval, Substrate
from pii_anon.assurance.assessors import assess_reidentification
from pii_anon.assurance.claim_strength import ClaimStrength, MeasurementMode, PowerThresholds
from pii_anon.eval_framework.attacks.reid import ANTI_ANONYMITY_CAVEAT

_THR = PowerThresholds()


def _rec(i: int, *, transform: str | None) -> RecordEval:
    name = f"Alice{i} Smith{i}"
    text = f"Record {i}: contact {name} for details."
    start = text.index(name)
    ref = (("PERSON", start, start + len(name)),)
    return RecordEval(
        record_id=f"r{i}", original_text=text, language="en", reference=ref,
        pred_by_system={"user": ref, "pii-anon": ref},
        transform_by_system={"user": transform, "pii-anon": "Record: contact [PERSON] for details."},
    )


def _labeled(n: int, *, user_transform) -> Substrate:
    records = [_rec(i, transform=user_transform(i)) for i in range(n)]
    return Substrate(tuple(records), ("user", "pii-anon"), MeasurementMode.LABELED, "gold")


def test_reid_never_measured_and_carries_caveat() -> None:
    # a no-op (identity) transform leaves the names fully re-identifiable
    sub = _labeled(40, user_transform=lambda i: f"Record {i}: contact Alice{i} Smith{i} for details.")
    res = assess_reidentification(sub, thresholds=_THR, seed=1, n_resamples=300)
    user = res["user"]
    assert user.claim_strength is ClaimStrength.ADVISORY  # NEVER measured
    assert ANTI_ANONYMITY_CAVEAT in user.caveats
    assert user.metadata["candidate_set_size"] == 40
    assert user.metadata["direction"] == "lower_is_better"


def test_reid_noop_transform_is_more_reidentifiable_than_masking() -> None:
    leaky = _labeled(40, user_transform=lambda i: f"Record {i}: contact Alice{i} Smith{i} for details.")
    res = assess_reidentification(leaky, thresholds=_THR, seed=1, n_resamples=300)
    # the leaky (identity) user transform leaves identifying tokens -> high re-id success;
    # the pii-anon masking removes them -> ~0 re-id success. Lower is better.
    assert res["user"].value is not None and res["user"].value > 0.0
    assert res["pii-anon"].value == 0.0


def test_reid_not_assessable_without_transform() -> None:
    sub = _labeled(40, user_transform=lambda i: None)
    res = assess_reidentification(sub, thresholds=_THR, seed=1, n_resamples=300)
    assert res["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE
    assert res["user"].value is None


def test_reid_not_assessable_in_silver_mode() -> None:
    records = [_rec(i, transform=f"masked {i}") for i in range(40)]
    sub = Substrate(tuple(records), ("user", "pii-anon"), MeasurementMode.SILVER, "silver")
    res = assess_reidentification(sub, thresholds=_THR, seed=1, n_resamples=300)
    assert all(r.claim_strength is ClaimStrength.NOT_ASSESSABLE for r in res.values())


def test_reid_not_assessable_tiny_closed_world() -> None:
    # only one record carries PII -> a 1-candidate closed world is a trivial link
    one = RecordEval(
        record_id="r0", original_text="contact Bob Jones now", language="en",
        reference=(("PERSON", 8, 17),),
        pred_by_system={"user": (), "pii-anon": ()},
        transform_by_system={"user": "contact Bob Jones now", "pii-anon": "contact [X] now"},
    )
    plain = RecordEval(
        record_id="r1", original_text="no pii here at all", language="en", reference=(),
        pred_by_system={"user": (), "pii-anon": ()},
        transform_by_system={"user": "no pii here at all", "pii-anon": "no pii here at all"},
    )
    sub = Substrate((one, plain), ("user", "pii-anon"), MeasurementMode.LABELED, "gold")
    res = assess_reidentification(sub, thresholds=_THR, seed=1, n_resamples=300)
    assert res["user"].claim_strength is ClaimStrength.NOT_ASSESSABLE


def test_reid_is_deterministic() -> None:
    sub = _labeled(40, user_transform=lambda i: f"Record {i}: contact Alice{i} Smith{i} for details.")
    a = assess_reidentification(sub, thresholds=_THR, seed=5, n_resamples=300)
    b = assess_reidentification(sub, thresholds=_THR, seed=5, n_resamples=300)
    assert a["user"].value == b["user"].value
