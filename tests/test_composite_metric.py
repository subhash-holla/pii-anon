"""Tests for the PII-Rate-Elo composite metric module.

Tests cover normalization functions, configuration validation, Tier 1/Tier 2
computation, weight customization, edge cases, and serialization.
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.metrics.composite import (
    CompositeConfig,
    compute_composite,
    compute_composite_from_benchmark_result,
    normalize_identity,
    normalize_latency,
    normalize_throughput,
)


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------

class TestNormalizeIdentity:
    def test_clamp_within_range(self):
        assert normalize_identity(0.5) == 0.5

    def test_clamp_zero(self):
        assert normalize_identity(0.0) == 0.0

    def test_clamp_one(self):
        assert normalize_identity(1.0) == 1.0

    def test_clamp_above_one(self):
        assert normalize_identity(1.5) == 1.0

    def test_clamp_below_zero(self):
        assert normalize_identity(-0.3) == 0.0


class TestNormalizeLatency:
    def test_zero_latency_returns_one(self):
        assert normalize_latency(0.0) == 1.0

    def test_negative_latency_returns_one(self):
        assert normalize_latency(-5.0) == 1.0

    def test_at_reference_returns_half(self):
        result = normalize_latency(100.0, reference_ms=100.0)
        assert abs(result - 0.5) < 1e-9

    def test_very_high_latency_approaches_zero(self):
        result = normalize_latency(10000.0, reference_ms=100.0)
        assert result < 0.001

    def test_very_low_latency_approaches_one(self):
        result = normalize_latency(0.01, reference_ms=100.0)
        assert result > 0.999

    def test_custom_reference(self):
        result = normalize_latency(50.0, reference_ms=50.0)
        assert abs(result - 0.5) < 1e-9

    def test_monotone_decreasing(self):
        """Higher latency should always produce lower score."""
        scores = [normalize_latency(ms) for ms in [1, 10, 50, 100, 500, 1000]]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


class TestNormalizeThroughput:
    def test_zero_returns_zero(self):
        assert normalize_throughput(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert normalize_throughput(-100.0) == 0.0

    def test_at_reference_returns_half(self):
        result = normalize_throughput(1_000_000.0, reference_dph=1_000_000.0)
        assert abs(result - 0.5) < 1e-9

    def test_very_high_throughput_approaches_one(self):
        result = normalize_throughput(1e12, reference_dph=1_000_000.0)
        assert result > 0.999

    def test_monotone_increasing(self):
        """Higher throughput should always produce higher score."""
        scores = [normalize_throughput(dph) for dph in [100, 1000, 10000, 1e6, 1e9]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


# ---------------------------------------------------------------------------
# CompositeConfig
# ---------------------------------------------------------------------------

class TestCompositeConfig:
    def test_default_config_validates(self):
        cfg = CompositeConfig()
        cfg.validate()  # should not raise

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha_privacy"):
            CompositeConfig(alpha_privacy=1.5).validate()

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha_privacy"):
            CompositeConfig(alpha_privacy=-0.1).validate()

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight_detection_f1"):
            CompositeConfig(weight_detection_f1=-1.0).validate()

    def test_negative_reference_latency_raises(self):
        with pytest.raises(ValueError, match="reference_latency_ms"):
            CompositeConfig(reference_latency_ms=0.0).validate()

    def test_negative_reference_throughput_raises(self):
        with pytest.raises(ValueError, match="reference_throughput_dph"):
            CompositeConfig(reference_throughput_dph=-1.0).validate()

    def test_total_weight_default(self):
        cfg = CompositeConfig()
        assert abs(cfg.total_weight - 1.0) < 1e-9

    def test_tier1_weight(self):
        cfg = CompositeConfig()
        assert abs(cfg.tier1_weight - 1.0) < 1e-9

    def test_tier2_weight_default_zero(self):
        cfg = CompositeConfig()
        assert cfg.tier2_weight == 0.0

    def test_tier2_weight_nonzero(self):
        cfg = CompositeConfig(weight_privacy=0.1, weight_utility=0.1, weight_fairness=0.1)
        assert abs(cfg.tier2_weight - 0.3) < 1e-9


# ---------------------------------------------------------------------------
# Tier 1 composite computation
# ---------------------------------------------------------------------------

class TestTier1Composite:
    def test_perfect_scores(self):
        """Perfect detection + fast system should yield high composite."""
        result = compute_composite(
            f1=1.0,
            precision=1.0,
            recall=1.0,
            latency_ms=0.01,
            docs_per_hour=1e12,
        )
        assert result.score > 0.95

    def test_zero_scores(self):
        """Zero detection + zero throughput should yield very low composite."""
        result = compute_composite(
            f1=0.0,
            precision=0.0,
            recall=0.0,
            latency_ms=10000.0,
            docs_per_hour=0.0,
        )
        assert result.score < 0.01

    def test_score_in_range(self):
        """Composite must always be in [0, 1]."""
        result = compute_composite(
            f1=0.7,
            precision=0.8,
            recall=0.6,
            latency_ms=50.0,
            docs_per_hour=500_000.0,
        )
        assert 0.0 <= result.score <= 1.0

    def test_detection_dominated(self):
        """With default weights, detection quality should dominate."""
        high_detection = compute_composite(
            f1=0.9, precision=0.9, recall=0.9,
            latency_ms=100.0, docs_per_hour=500_000.0,
        )
        low_detection = compute_composite(
            f1=0.3, precision=0.3, recall=0.3,
            latency_ms=100.0, docs_per_hour=500_000.0,
        )
        assert high_detection.score > low_detection.score

    def test_sub_scores_populated(self):
        result = compute_composite(
            f1=0.8, precision=0.85, recall=0.75,
            latency_ms=50.0, docs_per_hour=2_000_000.0,
        )
        assert result.detection_sub > 0.0
        assert result.efficiency_sub > 0.0

    def test_raw_inputs_stored(self):
        result = compute_composite(
            f1=0.8, precision=0.85, recall=0.75,
            latency_ms=50.0, docs_per_hour=2_000_000.0,
        )
        assert result.raw_inputs["f1"] == 0.8
        assert result.raw_inputs["latency_ms"] == 50.0

    def test_components_all_present(self):
        result = compute_composite(
            f1=0.8, precision=0.85, recall=0.75,
            latency_ms=50.0, docs_per_hour=2_000_000.0,
        )
        expected_keys = {
            "f1_normalized", "precision_normalized", "recall_normalized",
            "latency_normalized", "throughput_normalized",
            "coverage_normalized",
            "privacy_normalized", "utility_normalized", "fairness_normalized",
            "tier1_score", "tier2_score",
        }
        assert set(result.components.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tier 2 composite computation
# ---------------------------------------------------------------------------

class TestTier2Composite:
    def test_privacy_utility_affect_score(self):
        """With Tier 2 weights and β<1, privacy/utility scores should shift the composite."""
        cfg = CompositeConfig(
            weight_privacy=0.15,
            weight_utility=0.10,
            weight_fairness=0.05,
            beta_tier_balance=0.5,
        )
        high_privacy = compute_composite(
            f1=0.8, precision=0.8, recall=0.8,
            latency_ms=50.0, docs_per_hour=1e6,
            privacy_score=1.0, utility_score=1.0, fairness_score=1.0,
            config=cfg,
        )
        low_privacy = compute_composite(
            f1=0.8, precision=0.8, recall=0.8,
            latency_ms=50.0, docs_per_hour=1e6,
            privacy_score=0.0, utility_score=0.0, fairness_score=0.0,
            config=cfg,
        )
        assert high_privacy.score > low_privacy.score

    def test_alpha_privacy_weighting(self):
        """Higher alpha should weight privacy more than utility."""
        cfg_high_alpha = CompositeConfig(
            alpha_privacy=0.9,
            weight_privacy=0.15,
            weight_utility=0.15,
            beta_tier_balance=0.5,
        )
        cfg_low_alpha = CompositeConfig(
            alpha_privacy=0.1,
            weight_privacy=0.15,
            weight_utility=0.15,
            beta_tier_balance=0.5,
        )
        # High privacy, low utility
        high_alpha_result = compute_composite(
            f1=0.8, precision=0.8, recall=0.8,
            latency_ms=50.0, docs_per_hour=1e6,
            privacy_score=1.0, utility_score=0.0,
            config=cfg_high_alpha,
        )
        low_alpha_result = compute_composite(
            f1=0.8, precision=0.8, recall=0.8,
            latency_ms=50.0, docs_per_hour=1e6,
            privacy_score=1.0, utility_score=0.0,
            config=cfg_low_alpha,
        )
        assert high_alpha_result.score > low_alpha_result.score


# ---------------------------------------------------------------------------
# compute_composite_from_benchmark_result
# ---------------------------------------------------------------------------

class TestFromBenchmarkResult:
    def test_basic_usage(self):
        class MockResult:
            f1 = 0.8
            precision = 0.85
            recall = 0.75
            latency_p50_ms = 50.0
            docs_per_hour = 2_000_000.0

        result = compute_composite_from_benchmark_result(MockResult())
        assert 0.0 <= result.score <= 1.0

    def test_missing_attributes_default_zero(self):
        class EmptyResult:
            pass

        result = compute_composite_from_benchmark_result(EmptyResult())
        assert result.score >= 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestCompositeScoreSerialization:
    def test_to_dict_keys(self):
        result = compute_composite(
            f1=0.8, precision=0.85, recall=0.75,
            latency_ms=50.0, docs_per_hour=2_000_000.0,
        )
        d = result.to_dict()
        assert "score" in d
        assert "detection_sub" in d
        assert "efficiency_sub" in d
        assert "components" in d
        assert "raw_inputs" in d

    def test_to_dict_values_rounded(self):
        result = compute_composite(
            f1=0.8, precision=0.85, recall=0.75,
            latency_ms=50.0, docs_per_hour=2_000_000.0,
        )
        d = result.to_dict()
        # All values should be rounded to 6 decimal places
        for key in ("score", "detection_sub", "efficiency_sub"):
            val_str = str(d[key])
            if "." in val_str:
                decimals = len(val_str.split(".")[1])
                assert decimals <= 6
