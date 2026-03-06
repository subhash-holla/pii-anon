"""Extended tests for eval_framework/metrics/composite.py uncovered branches."""

from __future__ import annotations

import math

import pytest

from pii_anon.eval_framework.metrics.composite import (
    CompositeConfig,
    normalize_attack_success_rate,
    normalize_canary_exposure,
    normalize_epsilon_dp,
    normalize_identity,
    normalize_k_anonymity,
    normalize_latency,
    normalize_mia_auc,
    normalize_throughput,
)


class TestNormalizeIdentity:
    """Test normalize_identity function."""

    def test_clamps_below_zero(self) -> None:
        """Test clamping of negative values."""
        assert normalize_identity(-0.5) == 0.0

    def test_clamps_above_one(self) -> None:
        """Test clamping of values above 1."""
        assert normalize_identity(1.5) == 1.0

    def test_preserves_valid_range(self) -> None:
        """Test values in [0, 1] are preserved."""
        assert normalize_identity(0.5) == 0.5
        assert normalize_identity(0.0) == 0.0
        assert normalize_identity(1.0) == 1.0


class TestNormalizeLatency:
    """Test normalize_latency function."""

    def test_latency_zero_returns_one(self) -> None:
        """Test zero latency returns perfect score."""
        assert normalize_latency(0.0) == 1.0

    def test_latency_negative_returns_one(self) -> None:
        """Test negative latency returns perfect score."""
        assert normalize_latency(-10.0) == 1.0

    def test_latency_at_reference_returns_half(self) -> None:
        """Test latency at reference point returns 0.5."""
        score = normalize_latency(100.0, reference_ms=100.0)
        assert score == 0.5

    def test_latency_below_reference_returns_above_half(self) -> None:
        """Test latency below reference returns > 0.5."""
        score = normalize_latency(50.0, reference_ms=100.0)
        assert score > 0.5

    def test_latency_above_reference_returns_below_half(self) -> None:
        """Test latency above reference returns < 0.5."""
        score = normalize_latency(200.0, reference_ms=100.0)
        assert score < 0.5

    def test_latency_custom_reference(self) -> None:
        """Test custom reference point."""
        score = normalize_latency(50.0, reference_ms=50.0)
        assert score == 0.5


class TestNormalizeThroughput:
    """Test normalize_throughput function."""

    def test_throughput_zero_returns_zero(self) -> None:
        """Test zero throughput returns 0."""
        assert normalize_throughput(0.0) == 0.0

    def test_throughput_negative_returns_zero(self) -> None:
        """Test negative throughput returns 0."""
        assert normalize_throughput(-100.0) == 0.0

    def test_throughput_at_reference_returns_half(self) -> None:
        """Test throughput at reference returns 0.5."""
        score = normalize_throughput(1_000_000.0, reference_dph=1_000_000.0)
        assert score == 0.5

    def test_throughput_exceeds_reference_approaches_one(self) -> None:
        """Test very high throughput approaches 1."""
        score = normalize_throughput(10_000_000.0, reference_dph=1_000_000.0)
        assert 0.5 < score < 1.0

    def test_throughput_below_reference_approaches_zero(self) -> None:
        """Test very low throughput approaches 0."""
        score = normalize_throughput(100_000.0, reference_dph=1_000_000.0)
        assert 0.0 < score < 0.5


class TestNormalizeAttackSuccessRate:
    """Test normalize_attack_success_rate function."""

    def test_asr_zero_returns_one(self) -> None:
        """Test ASR=0 returns perfect score."""
        assert normalize_attack_success_rate(0.0) == 1.0

    def test_asr_one_returns_zero(self) -> None:
        """Test ASR=1 returns zero."""
        assert normalize_attack_success_rate(1.0) == 0.0

    def test_asr_half_returns_half(self) -> None:
        """Test ASR=0.5 returns 0.5."""
        assert normalize_attack_success_rate(0.5) == 0.5

    def test_asr_clamps_below_zero(self) -> None:
        """Test ASR < 0 is clamped to 0."""
        assert normalize_attack_success_rate(-0.5) == 1.0

    def test_asr_clamps_above_one(self) -> None:
        """Test ASR > 1 is clamped to 1."""
        assert normalize_attack_success_rate(1.5) == 0.0


class TestNormalizeMIAAUC:
    """Test normalize_mia_auc function."""

    def test_mia_auc_half_returns_one(self) -> None:
        """Test MIA AUC=0.5 returns 1.0 (no adversarial advantage)."""
        score = normalize_mia_auc(0.5)
        assert score == 1.0

    def test_mia_auc_one_returns_zero(self) -> None:
        """Test MIA AUC=1.0 returns 0.0 (perfect attack)."""
        score = normalize_mia_auc(1.0)
        assert score == 0.0

    def test_mia_auc_zero_returns_one(self) -> None:
        """Test MIA AUC=0 returns 1.0 (random guessing)."""
        score = normalize_mia_auc(0.0)
        assert score == 1.0

    def test_mia_auc_clamps_to_valid_range(self) -> None:
        """Test MIA AUC scores are clamped to [0, 1]."""
        score_below = normalize_mia_auc(-0.5)
        score_above = normalize_mia_auc(1.5)
        assert 0.0 <= score_below <= 1.0
        assert 0.0 <= score_above <= 1.0


class TestNormalizeCanaryExposure:
    """Test normalize_canary_exposure function."""

    def test_canary_zero_exposure_returns_one(self) -> None:
        """Test zero exposure returns perfect score."""
        assert normalize_canary_exposure(0.0) == 1.0

    def test_canary_negative_exposure_returns_one(self) -> None:
        """Test negative exposure returns perfect score."""
        assert normalize_canary_exposure(-5.0) == 1.0

    def test_canary_large_exposure_approaches_zero(self) -> None:
        """Test large exposure approaches 0."""
        score = normalize_canary_exposure(100.0, c=5.0)
        assert 0.0 <= score < 0.1

    def test_canary_custom_c_parameter(self) -> None:
        """Test custom c parameter affects decay."""
        score_c5 = normalize_canary_exposure(5.0, c=5.0)
        score_c10 = normalize_canary_exposure(5.0, c=10.0)
        # Higher c means slower decay
        assert score_c10 > score_c5

    def test_canary_exposure_formula(self) -> None:
        """Test canary exposure follows exp(-E/c) formula."""
        exposure = 5.0
        c = 5.0
        expected = math.exp(-exposure / c)
        assert normalize_canary_exposure(exposure, c=c) == expected


class TestNormalizeKAnonymity:
    """Test normalize_k_anonymity function."""

    def test_k_one_returns_zero(self) -> None:
        """Test k=1 (no anonymity) returns 0."""
        assert normalize_k_anonymity(1) == 0.0

    def test_k_zero_returns_zero(self) -> None:
        """Test k=0 returns 0."""
        assert normalize_k_anonymity(0) == 0.0

    def test_k_max_returns_one(self) -> None:
        """Test k=k_max returns 1.0."""
        score = normalize_k_anonymity(100, k_max=100)
        assert score == 1.0

    def test_k_half_max_returns_half(self) -> None:
        """Test k=sqrt(k_max) returns ~0.5."""
        k_max = 100
        k_half = int(math.sqrt(k_max))
        score = normalize_k_anonymity(k_half, k_max=k_max)
        assert 0.4 < score < 0.6

    def test_k_max_zero_returns_one(self) -> None:
        """Test k_max=0 returns 1.0 (degenerate case)."""
        assert normalize_k_anonymity(50, k_max=0) == 1.0

    def test_k_clamps_to_valid_range(self) -> None:
        """Test output is clamped to [0, 1]."""
        score_high = normalize_k_anonymity(1000000, k_max=100)
        assert 0.0 <= score_high <= 1.0


class TestNormalizeEpsilonDP:
    """Test normalize_epsilon_dp function."""

    def test_epsilon_zero_returns_one(self) -> None:
        """Test ε=0 returns perfect score."""
        assert normalize_epsilon_dp(0.0) == 1.0

    def test_epsilon_at_reference_returns_exp_minus_one(self) -> None:
        """Test ε=ε₀ returns exp(-1) ≈ 0.368."""
        score = normalize_epsilon_dp(1.0, epsilon_0=1.0)
        assert abs(score - math.exp(-1.0)) < 0.01

    def test_epsilon_negative_returns_one(self) -> None:
        """Test ε < 0 returns 1.0."""
        assert normalize_epsilon_dp(-1.0) == 1.0

    def test_epsilon_with_delta_penalizes(self) -> None:
        """Test δ > threshold penalizes by 50%."""
        score_no_delta = normalize_epsilon_dp(1.0, epsilon_0=1.0, delta=None)
        score_with_delta = normalize_epsilon_dp(1.0, epsilon_0=1.0, delta=1e-4)
        # With weak δ (> threshold), score should be halved
        assert abs(score_with_delta - score_no_delta * 0.5) < 0.01

    def test_epsilon_with_small_delta_no_penalty(self) -> None:
        """Test δ ≤ threshold has no penalty."""
        score_no_delta = normalize_epsilon_dp(1.0, epsilon_0=1.0, delta=None)
        score_with_small_delta = normalize_epsilon_dp(1.0, epsilon_0=1.0, delta=1e-6)
        # Small δ should not penalize significantly
        assert score_with_small_delta >= score_no_delta * 0.9

    def test_epsilon_custom_reference(self) -> None:
        """Test custom epsilon_0 reference."""
        score_ref1 = normalize_epsilon_dp(1.0, epsilon_0=1.0)
        score_ref2 = normalize_epsilon_dp(1.0, epsilon_0=2.0)
        # Higher reference = higher score for same epsilon
        assert score_ref2 > score_ref1


class TestCompositeConfigValidation:
    """Test CompositeConfig validation."""

    def test_config_valid_defaults(self) -> None:
        """Test default config is valid."""
        config = CompositeConfig()
        config.validate()  # Should not raise

    def test_config_invalid_alpha_below_zero(self) -> None:
        """Test invalid alpha_privacy < 0."""
        config = CompositeConfig(alpha_privacy=-0.1)
        with pytest.raises(ValueError, match="alpha_privacy"):
            config.validate()

    def test_config_invalid_alpha_above_one(self) -> None:
        """Test invalid alpha_privacy > 1."""
        config = CompositeConfig(alpha_privacy=1.1)
        with pytest.raises(ValueError, match="alpha_privacy"):
            config.validate()

    def test_config_invalid_beta_below_zero(self) -> None:
        """Test invalid beta_tier_balance < 0."""
        config = CompositeConfig(beta_tier_balance=-0.1)
        with pytest.raises(ValueError, match="beta_tier_balance"):
            config.validate()

    def test_config_invalid_beta_above_one(self) -> None:
        """Test invalid beta_tier_balance > 1."""
        config = CompositeConfig(beta_tier_balance=1.1)
        with pytest.raises(ValueError, match="beta_tier_balance"):
            config.validate()

    def test_config_invalid_weight_negative(self) -> None:
        """Test negative weight raises error."""
        config = CompositeConfig(weight_detection_f1=-0.1)
        with pytest.raises(ValueError, match="weight_detection_f1"):
            config.validate()

    def test_config_invalid_reference_latency_zero(self) -> None:
        """Test zero reference latency raises error."""
        config = CompositeConfig(reference_latency_ms=0.0)
        with pytest.raises(ValueError, match="reference_latency_ms"):
            config.validate()

    def test_config_invalid_reference_latency_negative(self) -> None:
        """Test negative reference latency raises error."""
        config = CompositeConfig(reference_latency_ms=-10.0)
        with pytest.raises(ValueError, match="reference_latency_ms"):
            config.validate()

    def test_config_invalid_reference_throughput_zero(self) -> None:
        """Test zero reference throughput raises error."""
        config = CompositeConfig(reference_throughput_dph=0.0)
        with pytest.raises(ValueError, match="reference_throughput_dph"):
            config.validate()

    def test_config_invalid_reference_throughput_negative(self) -> None:
        """Test negative reference throughput raises error."""
        config = CompositeConfig(reference_throughput_dph=-100.0)
        with pytest.raises(ValueError, match="reference_throughput_dph"):
            config.validate()


class TestCompositeConfigWeights:
    """Test CompositeConfig weight calculations."""

    def test_total_weight_all_zero(self) -> None:
        """Test total weight with all zero weights."""
        config = CompositeConfig(
            weight_detection_f1=0.0,
            weight_detection_precision=0.0,
            weight_detection_recall=0.0,
            weight_latency=0.0,
            weight_throughput=0.0,
            weight_coverage=0.0,
            weight_privacy=0.0,
            weight_utility=0.0,
            weight_fairness=0.0,
        )
        assert config.total_weight == 0.0

    def test_total_weight_summed_correctly(self) -> None:
        """Test total weight is sum of all weights."""
        config = CompositeConfig(
            weight_detection_f1=0.5,
            weight_detection_precision=0.15,
            weight_detection_recall=0.15,
            weight_latency=0.1,
            weight_throughput=0.1,
        )
        expected = 0.5 + 0.15 + 0.15 + 0.1 + 0.1
        assert config.total_weight == expected

    def test_tier1_weight_excludes_tier2(self) -> None:
        """Test tier1_weight excludes tier 2 weights."""
        config = CompositeConfig(
            weight_detection_f1=0.0,
            weight_detection_precision=0.0,
            weight_detection_recall=0.0,
            weight_latency=0.0,
            weight_throughput=0.0,
            weight_coverage=0.0,
            weight_privacy=0.25,
            weight_utility=0.25,
        )
        assert config.tier1_weight == 0.0
        assert config.tier2_weight == 0.5

    def test_tier2_weight_only_includes_tier2(self) -> None:
        """Test tier2_weight includes only tier 2 weights."""
        config = CompositeConfig(
            weight_detection_f1=0.5,
            weight_privacy=0.1,
            weight_utility=0.2,
            weight_fairness=0.2,
        )
        assert config.tier2_weight == 0.5


class TestCompositeConfigBoundaryValues:
    """Test CompositeConfig with boundary values."""

    def test_config_alpha_boundaries(self) -> None:
        """Test alpha can be 0 or 1."""
        config0 = CompositeConfig(alpha_privacy=0.0)
        config0.validate()
        config1 = CompositeConfig(alpha_privacy=1.0)
        config1.validate()

    def test_config_beta_boundaries(self) -> None:
        """Test beta can be 0 or 1."""
        config0 = CompositeConfig(beta_tier_balance=0.0)
        config0.validate()
        config1 = CompositeConfig(beta_tier_balance=1.0)
        config1.validate()

    def test_config_zero_weights(self) -> None:
        """Test zero weights are valid."""
        config = CompositeConfig(weight_detection_f1=0.0)
        config.validate()

    def test_config_large_weights(self) -> None:
        """Test large weights are valid."""
        config = CompositeConfig(weight_detection_f1=1000.0)
        config.validate()
