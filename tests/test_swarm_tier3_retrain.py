"""Tests for the Tier 3 retrain-readiness changes to pii-anon-swarm.

Covers the new surface added so a post-session swarm retrain picks up
v1.3.0 Tier 3 signals without inference-time regressions:

* ``TrainingRecord`` Tier 3 fields populate from pii-anon-datasets records
* ``compute_sample_weights_from_records`` maps RRS → loss weight
* ``select_f2_threshold`` picks the emission threshold that maximises F2
* ``FEATURE_VERSION`` is bumped and ``extract_features`` emits 21 values
* ``FloorGateConfig.industry_leadership`` and
  ``GovernanceThresholds.industry_leadership`` match the v10 bar
* ``evaluate_floor_gates`` honours the new ``min_f2`` gate
* ``SEMANTIC_TYPES`` includes ``EMAIL_ADDRESS`` and ``CREDIT_CARD``
"""
from __future__ import annotations

import pytest

from pii_anon.eval_framework import (
    CompositeConfig,
    FloorGateConfig,
    GovernanceThresholds,
    compute_composite,
)
from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates
from pii_anon.swarm import SEMANTIC_TYPES
from pii_anon.swarm_datasets import TrainingRecord
from pii_anon.swarm_learner import (
    FEATURE_VERSION,
    compute_sample_weights_from_records,
    select_f2_threshold,
)


# ---------------------------------------------------------------------------
# SEMANTIC_TYPES — the precision-boosting corroboration gate
# ---------------------------------------------------------------------------

def test_semantic_types_includes_email_and_credit_card():
    """v10 regression: EMAIL_ADDRESS and CREDIT_CARD had swarm precision
    below 0.5 on the benchmark because they bypassed the corroboration
    gate.  Adding them to SEMANTIC_TYPES clamps false positives.
    """
    assert "EMAIL_ADDRESS" in SEMANTIC_TYPES
    assert "CREDIT_CARD" in SEMANTIC_TYPES


def test_semantic_types_preserves_original_entries():
    """Adding EMAIL/CREDIT_CARD must not remove the original entries."""
    for et in (
        "PERSON_NAME", "ORGANIZATION", "LOCATION", "DATE_OF_BIRTH",
        "ADDRESS", "USERNAME", "PHONE_NUMBER",
    ):
        assert et in SEMANTIC_TYPES


# ---------------------------------------------------------------------------
# FEATURE_VERSION — trained artifacts are keyed to a specific shape
# ---------------------------------------------------------------------------

def test_feature_version_is_bumped():
    """Guards the artifact-shape contract across version bumps.

    v2 introduced feature 21 (multilingual context keywords).
    v3 added distinct ENTITY_TYPE_ENCODING indices for the Phase 3
    paper-v11 gap-closure types (CVV / PIN / PASSWORD / COURT_CASE /
    DOCKET / BAR / INVOICE / INSURANCE_POLICY / SALARY).
    """
    assert FEATURE_VERSION >= 3


# ---------------------------------------------------------------------------
# TrainingRecord — Tier 3 fields
# ---------------------------------------------------------------------------

def test_training_record_tier3_defaults():
    """Existing loaders that don't populate Tier 3 fields must still work."""
    rec = TrainingRecord(record_id="r", text="hello", labels=[])
    assert rec.behavioral_signal_density == 0.0
    assert rec.re_identification_resistance_score is None
    assert rec.persona_id is None
    assert rec.is_paired_profile is False


def test_training_record_tier3_explicit():
    rec = TrainingRecord(
        record_id="r", text="hello", labels=[],
        behavioral_signal_density=0.42,
        re_identification_resistance_score=0.63,
        persona_id="persona_0001",
        is_paired_profile=True,
    )
    assert rec.behavioral_signal_density == 0.42
    assert rec.re_identification_resistance_score == 0.63
    assert rec.persona_id == "persona_0001"
    assert rec.is_paired_profile is True


# ---------------------------------------------------------------------------
# compute_sample_weights_from_records — RRS → loss weight
# ---------------------------------------------------------------------------

def test_sample_weights_none_rrs_uses_default_weight():
    recs = [TrainingRecord(record_id="r", text="x", labels=[])]
    weights = compute_sample_weights_from_records(recs, default_weight=1.0)
    assert weights == [1.0]


def test_sample_weights_low_rrs_is_upweighted():
    """RRS=0 (trivially re-identifiable) → maximum weight."""
    recs = [
        TrainingRecord(record_id="easy", text="x", labels=[],
                       re_identification_resistance_score=1.0),
        TrainingRecord(record_id="hard", text="x", labels=[],
                       re_identification_resistance_score=0.0),
    ]
    weights = compute_sample_weights_from_records(recs, rrs_boost=2.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(2.0)


def test_sample_weights_paired_profile_stacks():
    recs = [
        TrainingRecord(record_id="r", text="x", labels=[],
                       re_identification_resistance_score=0.0,
                       is_paired_profile=True),
    ]
    weights = compute_sample_weights_from_records(
        recs, rrs_boost=2.0, paired_profile_boost=1.5,
    )
    assert weights[0] == pytest.approx(3.0)


def test_sample_weights_reject_bad_boosts():
    with pytest.raises(ValueError, match=">= 1.0"):
        compute_sample_weights_from_records([], rrs_boost=0.5)
    with pytest.raises(ValueError, match=">= 1.0"):
        compute_sample_weights_from_records([], paired_profile_boost=0.5)


def test_sample_weights_clamp_rrs_out_of_range():
    """RRS values outside [0, 1] clamp instead of producing negative weights."""
    recs = [
        TrainingRecord(record_id="a", text="x", labels=[],
                       re_identification_resistance_score=-0.5),
        TrainingRecord(record_id="b", text="x", labels=[],
                       re_identification_resistance_score=1.5),
    ]
    weights = compute_sample_weights_from_records(recs, rrs_boost=2.0)
    # -0.5 → clamped to 0 → rrs_boost (2.0)
    assert weights[0] == pytest.approx(2.0)
    # 1.5 → clamped to 1 → default_weight (1.0)
    assert weights[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_f2_threshold — F2-optimising emission threshold
# ---------------------------------------------------------------------------

def test_select_f2_threshold_prefers_recall_over_precision():
    """F2 weighs recall 4× precision.  On a synthetic set where the
    0.35 threshold catches all TPs but admits 1 FP, and the 0.65
    threshold catches only half the TPs with zero FPs, F2 should pick
    the lower threshold.
    """
    scores = [0.9, 0.85, 0.4, 0.38, 0.1]
    labels = [1, 1, 1, 0, 0]
    # Threshold 0.35 → TP=3, FP=1 → P=0.75, R=1.0, F2≈0.94
    # Threshold 0.50 → TP=2, FP=0 → P=1.0, R=0.67, F2≈0.73
    threshold, f2 = select_f2_threshold(
        scores, labels, min_threshold=0.30, max_threshold=0.70, step=0.05,
    )
    assert threshold == pytest.approx(0.35, abs=0.06)
    assert f2 > 0.90


def test_select_f2_threshold_empty_scores_returns_default():
    threshold, f2 = select_f2_threshold([], [])
    assert threshold == 0.5
    assert f2 == 0.0


def test_select_f2_threshold_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        select_f2_threshold([0.5], [1, 0])


def test_select_f2_threshold_invalid_grid_raises():
    with pytest.raises(ValueError):
        select_f2_threshold([0.5], [1], min_threshold=0.5, max_threshold=0.5)
    with pytest.raises(ValueError):
        select_f2_threshold([0.5], [1], step=0)


def test_select_f2_threshold_all_fp_no_positive():
    """When no threshold yields any TP, return the fallback default."""
    threshold, f2 = select_f2_threshold([0.9, 0.8], [0, 0])
    assert threshold == 0.5
    assert f2 == 0.0


# ---------------------------------------------------------------------------
# Industry-leadership presets
# ---------------------------------------------------------------------------

def test_floor_gate_industry_leadership_preset():
    gate = FloorGateConfig.industry_leadership()
    assert gate.enabled is True
    assert gate.min_f1 == 0.60
    assert gate.min_f2 == 0.65
    assert gate.min_privacy == 0.70
    assert gate.min_fairness == 0.50
    assert gate.min_entity_coverage == 0.80


def test_governance_thresholds_industry_leadership_preset():
    thr = GovernanceThresholds.industry_leadership()
    assert thr.min_rating == 1600.0
    assert thr.max_rd == 80.0
    assert thr.min_matches == 10


def test_f2_gate_caps_composite_when_f2_below_threshold():
    """A system with F1=0.80 and F2=0.50 fails the leadership F2 gate
    even though it would pass the F1 gate on its own.
    """
    cfg = CompositeConfig(
        floor_gates=FloorGateConfig.industry_leadership(),
    )
    score = compute_composite(
        f1=0.80, precision=0.95, recall=0.50,   # F2 ~= 0.54
        latency_ms=5.0, docs_per_hour=500_000,
        privacy_score=0.85, utility_score=0.80, fairness_score=0.70,
        entity_types_detected=20, entity_types_total=22,
        config=cfg,
    )
    assert score.floor_gate_result is not None
    assert score.floor_gate_result.gates["f2"]["passed"] is False
    assert score.floor_gate_result.capped is True
    assert score.score == pytest.approx(0.40)


def test_f2_gate_is_noop_when_min_f2_is_zero():
    """Older FloorGateConfig callers with min_f2=0 don't see the gate."""
    cfg = CompositeConfig(floor_gates=FloorGateConfig(enabled=True))
    score = compute_composite(
        f1=0.80, precision=0.95, recall=0.50,
        latency_ms=5.0, docs_per_hour=500_000,
        privacy_score=0.85, utility_score=0.80, fairness_score=0.70,
        entity_types_detected=20, entity_types_total=22,
        config=cfg,
    )
    assert score.floor_gate_result is not None
    assert "f2" not in score.floor_gate_result.gates


def test_evaluate_floor_gates_omits_f2_when_input_missing():
    """Supplying a min_f2 threshold without an f2 score is still valid."""
    cfg = FloorGateConfig.industry_leadership()
    result = evaluate_floor_gates(
        f1=0.70, privacy_score=0.80, fairness_score=0.60,
        entity_coverage=0.85, config=cfg,
    )
    # min_f2 > 0 but no f2 supplied → gate not recorded, not counted.
    assert "f2" not in result.gates
