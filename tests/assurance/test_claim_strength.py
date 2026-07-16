"""Tests for the three-valued claim-strength engine and the power gate."""

from __future__ import annotations

from pii_anon.assurance.claim_strength import (
    ClaimStrength,
    DimensionResult,
    MeasurementMode,
    PowerThresholds,
    advisory,
    measured,
    not_assessable,
    power_gate,
    resolve,
)

_THR = PowerThresholds(min_support=50, max_ci_halfwidth=0.05)


def test_constructors_set_state() -> None:
    na = not_assessable("leakage", "acme", "pipeline has no transform")
    assert na.claim_strength is ClaimStrength.NOT_ASSESSABLE
    assert na.value is None
    assert na.reasons == ("pipeline has no transform",)

    adv = advisory("detection", "acme", 0.8, reasons=("silver",))
    assert adv.claim_strength is ClaimStrength.ADVISORY
    assert adv.value == 0.8

    m = measured("detection", "acme", 0.8, support=100)
    assert m.claim_strength is ClaimStrength.MEASURED
    assert m.is_publishable is True


def test_not_assessable_value_is_never_zero() -> None:
    na = not_assessable("reid", "acme")
    assert na.value is None  # not 0.0 — a missing dimension must not read as a score


def test_power_gate_passes_when_powered() -> None:
    ok, reasons = power_gate(
        support=200, n_clusters=200, confidence_interval=(0.78, 0.82), significant=True, thresholds=_THR
    )
    assert ok is True
    assert reasons == ()


def test_power_gate_fails_on_low_support() -> None:
    ok, reasons = power_gate(
        support=12, n_clusters=200, confidence_interval=(0.78, 0.82), significant=True, thresholds=_THR
    )
    assert ok is False
    assert any("support" in r for r in reasons)


def test_power_gate_fails_on_wide_ci() -> None:
    ok, reasons = power_gate(
        support=500, n_clusters=500, confidence_interval=(0.4, 0.9), significant=True, thresholds=_THR
    )
    assert ok is False
    assert any("CI too wide" in r for r in reasons)


def test_power_gate_fails_on_missing_ci() -> None:
    ok, reasons = power_gate(
        support=500, n_clusters=500, confidence_interval=None, significant=True, thresholds=_THR
    )
    assert ok is False
    assert any("confidence interval" in r for r in reasons)


def test_power_gate_significance_optional() -> None:
    thr = PowerThresholds(min_support=50, max_ci_halfwidth=0.05, require_significance=True)
    ok, reasons = power_gate(
        support=500, n_clusters=500, confidence_interval=(0.78, 0.82), significant=False, thresholds=thr
    )
    assert ok is False
    assert any("significant" in r for r in reasons)


def test_resolve_none_value_is_not_assessable() -> None:
    r = resolve(
        dimension="leakage", system="acme", mode=MeasurementMode.LABELED, value=None,
        support=0, n_clusters=0, confidence_interval=None, significant=None, thresholds=_THR,
    )
    assert r.claim_strength is ClaimStrength.NOT_ASSESSABLE


def test_resolve_silver_is_never_measured() -> None:
    # Even with perfect power, silver mode can never be publishable (circular reference).
    r = resolve(
        dimension="detection", system="acme", mode=MeasurementMode.SILVER, value=0.9,
        support=10_000, n_clusters=10_000, confidence_interval=(0.899, 0.901), significant=True, thresholds=_THR,
    )
    assert r.claim_strength is ClaimStrength.ADVISORY


def test_resolve_labeled_powered_is_measured() -> None:
    r = resolve(
        dimension="detection", system="acme", mode=MeasurementMode.LABELED, value=0.9,
        support=1000, n_clusters=1000, confidence_interval=(0.89, 0.91), significant=True, thresholds=_THR,
    )
    assert r.claim_strength is ClaimStrength.MEASURED


def test_resolve_labeled_underpowered_is_advisory_with_reasons() -> None:
    r = resolve(
        dimension="detection", system="acme", mode=MeasurementMode.LABELED, value=0.9,
        support=5, n_clusters=5, confidence_interval=(0.5, 0.99), significant=False, thresholds=_THR,
    )
    assert r.claim_strength is ClaimStrength.ADVISORY
    assert r.reasons  # carries the demotion reason


def test_power_gate_fails_on_few_clusters() -> None:
    # high span support but only 2 records (clusters) -> not powered (degenerate bootstrap)
    ok, reasons = power_gate(
        support=200, n_clusters=2, confidence_interval=(1.0, 1.0), significant=True, thresholds=_THR
    )
    assert ok is False
    assert any("clusters" in r for r in reasons)


def test_repr_does_not_leak_metadata() -> None:
    r = DimensionResult(
        dimension="detection", system="acme", claim_strength=ClaimStrength.MEASURED,
        value=0.9, metadata={"secret": "John Smith 123-45-6789"},
    )
    assert "John Smith" not in repr(r)
    assert "123-45-6789" not in repr(r)
