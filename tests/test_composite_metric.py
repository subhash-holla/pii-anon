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


# ---------------------------------------------------------------------------
# Adversarial normalization functions
# ---------------------------------------------------------------------------

class TestNormalizeAttackSuccessRate:
    def test_zero_success_yields_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(0.0) == 1.0

    def test_full_success_yields_zero(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(1.0) == 0.0

    def test_half_success_yields_half(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(0.5) == 0.5

    def test_clamped_negative_input(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        # Negative input gets 1 - (-0.5) = 1.5, clamped to 1.0
        assert normalize_attack_success_rate(-0.5) == 1.0

    def test_clamped_high_input(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        # Input > 1.0 gets 1 - 1.5 = -0.5, clamped to 0.0
        assert normalize_attack_success_rate(1.5) == 0.0


class TestNormalizeMiaAuc:
    def test_random_guessing_yields_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        assert normalize_mia_auc(0.5) == 1.0

    def test_perfect_attack_yields_zero(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        assert normalize_mia_auc(1.0) == 0.0

    def test_medium_auc(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        result = normalize_mia_auc(0.75)
        assert 0.4 < result < 0.6

    def test_clamped_to_range(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        assert 0.0 <= normalize_mia_auc(2.0) <= 1.0


class TestNormalizeCanaryExposure:
    def test_zero_exposure_yields_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        assert normalize_canary_exposure(0.0) == 1.0

    def test_negative_exposure_yields_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        assert normalize_canary_exposure(-5.0) == 1.0

    def test_high_exposure_approaches_zero(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        result = normalize_canary_exposure(50.0, c=5.0)
        assert result < 0.001

    def test_custom_c_parameter(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        result_c5 = normalize_canary_exposure(5.0, c=5.0)
        result_c10 = normalize_canary_exposure(5.0, c=10.0)
        assert result_c10 > result_c5


class TestNormalizeKAnonymity:
    def test_k_one_yields_zero(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(1) == 0.0

    def test_k_max_yields_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(100, k_max=100) == 1.0

    def test_logarithmic_scaling(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        result_10 = normalize_k_anonymity(10, k_max=100)
        result_100 = normalize_k_anonymity(100, k_max=100)
        assert 0 < result_10 < result_100

    def test_low_k_max(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(2, k_max=1) == 1.0


class TestNormalizeEpsilonDp:
    def test_zero_epsilon_yields_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        assert normalize_epsilon_dp(0.0) == 1.0

    def test_with_delta_penalty(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        without_delta = normalize_epsilon_dp(1.0, delta=None)
        with_delta = normalize_epsilon_dp(1.0, delta=1e-4)
        assert with_delta < without_delta

    def test_high_epsilon_low_privacy(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        result = normalize_epsilon_dp(10.0, epsilon_0=1.0)
        assert result < 0.001


# ---------------------------------------------------------------------------
# Floor gates
# ---------------------------------------------------------------------------

class TestFloorGates:
    def test_all_gates_pass(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(
            min_f1=0.5, min_privacy=0.5, min_fairness=0.5, min_entity_coverage=0.5,
            enabled=True
        )
        result = evaluate_floor_gates(
            f1=0.8, privacy_score=0.8, fairness_score=0.8, entity_coverage=0.8,
            config=config
        )
        assert result.all_passed
        assert not result.capped

    def test_f1_gate_fails(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(min_f1=0.8, enabled=True)
        result = evaluate_floor_gates(
            f1=0.5, privacy_score=0.9, fairness_score=0.9, entity_coverage=0.9,
            config=config
        )
        assert not result.all_passed
        assert result.capped
        assert "f1" in [g for g in result.gates if not result.gates[g]["passed"]]

    def test_privacy_gate_fails(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(min_privacy=0.8, enabled=True)
        result = evaluate_floor_gates(
            f1=0.9, privacy_score=0.5, fairness_score=0.9, entity_coverage=0.9,
            config=config
        )
        assert not result.all_passed
        assert "privacy" in result.gates

    def test_fairness_gate_fails(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(min_fairness=0.8, enabled=True)
        result = evaluate_floor_gates(
            f1=0.9, privacy_score=0.9, fairness_score=0.5, entity_coverage=0.9,
            config=config
        )
        assert not result.all_passed

    def test_coverage_gate_fails(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(min_entity_coverage=0.8, enabled=True)
        result = evaluate_floor_gates(
            f1=0.9, privacy_score=0.9, fairness_score=0.9, entity_coverage=0.5,
            config=config
        )
        assert not result.all_passed

    def test_remediation_suggestions(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(min_f1=0.8, enabled=True)
        result = evaluate_floor_gates(
            f1=0.5, privacy_score=0.9, fairness_score=0.9, entity_coverage=0.9,
            config=config
        )
        assert len(result.remediation) > 0
        assert any("f1" in r.lower() for r in result.remediation)

    def test_floor_gate_result_serialization(self):
        from pii_anon.eval_framework.metrics.composite import evaluate_floor_gates, FloorGateConfig
        config = FloorGateConfig(min_f1=0.8, enabled=True)
        result = evaluate_floor_gates(
            f1=0.5, privacy_score=0.9, fairness_score=0.9, entity_coverage=0.9,
            config=config
        )
        d = result.to_dict()
        assert "all_passed" in d
        assert "gates" in d
        assert "capped" in d
        assert "remediation" in d


# ---------------------------------------------------------------------------
# Entity coverage normalization
# ---------------------------------------------------------------------------

class TestNormalizeEntityCoverage:
    def test_perfect_coverage(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        result = normalize_entity_coverage(13, 13)
        assert result == 1.0

    def test_zero_total_returns_zero(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        result = normalize_entity_coverage(5, 0)
        assert result == 0.0

    def test_partial_coverage(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        result = normalize_entity_coverage(5, 10)
        assert result == 0.5

    def test_negative_detected_clamped(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        result = normalize_entity_coverage(-1, 10)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Pareto frontier analysis
# ---------------------------------------------------------------------------

class TestParetoFrontierAnalyzer:
    def test_empty_analyzer(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        assert analyzer.compute_frontier() == []

    def test_single_system_on_frontier(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.7, 0.8)
        frontier = analyzer.compute_frontier()
        assert frontier == ["a"]

    def test_dominated_system(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.9, 0.9)  # Better on both dimensions
        analyzer.add_system("b", 0.7, 0.8)  # Worse on both
        frontier = analyzer.compute_frontier()
        assert frontier == ["a"]
        assert not analyzer.is_dominated("a")
        assert analyzer.is_dominated("b")

    def test_trade_off_both_on_frontier(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.9, 0.6)  # High privacy, low utility
        analyzer.add_system("b", 0.6, 0.9)  # Low privacy, high utility
        frontier = analyzer.compute_frontier()
        assert set(frontier) == {"a", "b"}

    def test_distance_to_frontier_on_frontier(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.8, 0.8)
        distance = analyzer.distance_to_frontier("a")
        assert distance == 0.0

    def test_distance_to_frontier_off_frontier(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.9, 0.9)
        analyzer.add_system("b", 0.5, 0.5)
        distance = analyzer.distance_to_frontier("b")
        assert distance > 0.0

    def test_distance_unknown_system(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        distance = analyzer.distance_to_frontier("nonexistent")
        assert distance == float("inf")

    def test_frontier_data_serialization(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.8, 0.7)
        analyzer.add_system("b", 0.6, 0.9)
        data = analyzer.frontier_data()
        assert "systems" in data
        assert "frontier" in data
        assert "dominated" in data
        assert "a" in data["systems"]
        assert "on_frontier" in data["systems"]["a"]

    def test_reset_clears_systems(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 0.8, 0.7)
        analyzer.reset()
        assert analyzer.compute_frontier() == []

    def test_clipping_values(self):
        from pii_anon.eval_framework.metrics.composite import ParetoFrontierAnalyzer
        analyzer = ParetoFrontierAnalyzer()
        analyzer.add_system("a", 1.5, -0.5)  # Out of bounds
        assert analyzer._systems["a"][0] == 1.0  # Clipped to [0, 1]
        assert analyzer._systems["a"][1] == 0.0  # Clipped to [0, 1]
