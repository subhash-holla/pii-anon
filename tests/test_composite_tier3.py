"""Tests for Tier 3 composite metric extensions.

Covers:
- F_beta / F2 score computation
- Re-identification Resistance Score (RRS)
- Quasi-Identifier Coverage (QIC)
- Behavioral Signal Leakage (BSL)
- Deployment profile presets (standard, high_security, high_throughput)
- F2-based privacy-first composite preset
- End-to-end composite with Tier 3 inputs

Reference: Lermen et al. (2026); recommendations-dataset-metric-evolution.md
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.metrics.composite import (
    CompositeConfig,
    compute_composite,
    fbeta_score,
    normalize_behavioral_signal_leakage,
    normalize_quasi_identifier_coverage,
    normalize_reidentification_resistance,
)


# ---------------------------------------------------------------------------
# F_beta score
# ---------------------------------------------------------------------------


class TestFBetaScore:
    def test_f1_when_beta_is_one(self) -> None:
        # Standard F1 formula: 2·P·R / (P + R)
        assert abs(fbeta_score(0.8, 0.6, beta=1.0) - 0.6857142857) < 1e-6

    def test_f2_double_weights_recall(self) -> None:
        # F2 should be closer to recall when recall is lower.
        f2 = fbeta_score(precision=0.8, recall=0.6, beta=2.0)
        f1 = fbeta_score(0.8, 0.6, beta=1.0)
        # F2 < F1 when P > R (recall penalized more in F2).
        assert f2 < f1

    def test_f_half_double_weights_precision(self) -> None:
        f_half = fbeta_score(0.8, 0.6, beta=0.5)
        f1 = fbeta_score(0.8, 0.6, beta=1.0)
        assert f_half > f1  # F0.5 emphasizes precision, which is higher.

    def test_zero_precision_and_recall_returns_zero(self) -> None:
        assert fbeta_score(0.0, 0.0, beta=2.0) == 0.0

    def test_equal_precision_recall_gives_same_value(self) -> None:
        # F_beta(P, P, any β) = P exactly when P = R.
        assert abs(fbeta_score(0.7, 0.7, beta=2.0) - 0.7) < 1e-9
        assert abs(fbeta_score(0.7, 0.7, beta=0.5) - 0.7) < 1e-9

    def test_perfect_scores(self) -> None:
        assert abs(fbeta_score(1.0, 1.0, beta=2.0) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Re-identification Resistance Score (RRS)
# ---------------------------------------------------------------------------


class TestRRS:
    def test_lermen_baseline(self) -> None:
        """Lermen et al. 2026: 67% recall × 90% precision → RRS = 0.397."""
        rrs = normalize_reidentification_resistance(0.67, 0.90)
        assert abs(rrs - 0.397) < 1e-6

    def test_no_attack_success(self) -> None:
        assert normalize_reidentification_resistance(0.0, 0.0) == 1.0

    def test_perfect_attack(self) -> None:
        assert normalize_reidentification_resistance(1.0, 1.0) == 0.0

    def test_clamping_inputs(self) -> None:
        # Values outside [0,1] are clamped before computation.
        # (-0.5, 1.5) → (0, 1) → RRS = 1 - (0 * 1) = 1.0 (no attack success).
        assert normalize_reidentification_resistance(-0.5, 1.5) == 1.0
        # (2.0, 2.0) → (1, 1) → RRS = 1 - (1 * 1) = 0.0 (perfect attack).
        assert normalize_reidentification_resistance(2.0, 2.0) == 0.0

    def test_monotone_in_recall(self) -> None:
        # Higher attack recall → lower RRS.
        high = normalize_reidentification_resistance(0.3, 0.8)
        low = normalize_reidentification_resistance(0.9, 0.8)
        assert high > low


# ---------------------------------------------------------------------------
# Quasi-Identifier Coverage (QIC)
# ---------------------------------------------------------------------------


class TestQIC:
    def test_half_removed(self) -> None:
        assert normalize_quasi_identifier_coverage(5, 10) == 0.5

    def test_all_removed(self) -> None:
        assert normalize_quasi_identifier_coverage(10, 10) == 1.0

    def test_none_removed(self) -> None:
        assert normalize_quasi_identifier_coverage(0, 10) == 0.0

    def test_zero_signals_returns_zero(self) -> None:
        assert normalize_quasi_identifier_coverage(0, 0) == 0.0

    def test_weighted_coverage(self) -> None:
        # Weighted: if total weight is 10 and removed weight is 3 → 0.3.
        weights = [2.0, 3.0, 1.0, 4.0]  # total = 10
        assert abs(normalize_quasi_identifier_coverage(3, 4, weights) - 0.3) < 1e-6

    def test_coverage_clamped_to_one(self) -> None:
        # Even if removed > total (pathological), clamped to 1.
        assert normalize_quasi_identifier_coverage(15, 10) == 1.0


# ---------------------------------------------------------------------------
# Behavioral Signal Leakage (BSL)
# ---------------------------------------------------------------------------


class TestBSL:
    def test_no_leakage(self) -> None:
        # Zero similarity = perfect obfuscation = BSL 1.0.
        assert normalize_behavioral_signal_leakage(0.0) == 1.0

    def test_perfect_leakage(self) -> None:
        # Full similarity = complete leak = BSL 0.0.
        assert normalize_behavioral_signal_leakage(1.0) == 0.0

    def test_partial_leakage(self) -> None:
        assert abs(normalize_behavioral_signal_leakage(0.7) - 0.3) < 1e-6

    def test_clamping(self) -> None:
        # Cosine similarity can be outside [0, 1] for edge cases.
        assert normalize_behavioral_signal_leakage(-0.5) == 1.0
        assert normalize_behavioral_signal_leakage(1.5) == 0.0


# ---------------------------------------------------------------------------
# Deployment profile presets
# ---------------------------------------------------------------------------


class TestDeploymentProfile:
    def test_standard_profile(self) -> None:
        cfg = CompositeConfig.for_deployment("standard")
        # Tier 1 gets 50% of weight, Tier 3 gets 30%.
        assert abs(cfg.tier1_weight - 0.70) < 1e-6  # 0.5 (D) + 0.2 (O)
        assert abs(cfg.tier3_weight - 0.30) < 1e-6
        assert cfg.deployment_profile == "standard"

    def test_high_security_profile(self) -> None:
        cfg = CompositeConfig.for_deployment("high_security")
        assert abs(cfg.tier1_weight - 0.40) < 1e-6
        assert abs(cfg.tier3_weight - 0.60) < 1e-6
        # In high-security, RRS is the dominant component.
        assert cfg.weight_reidentification_resistance > cfg.weight_detection_f2

    def test_high_throughput_profile(self) -> None:
        cfg = CompositeConfig.for_deployment("high_throughput")
        assert abs(cfg.tier1_weight - 0.80) < 1e-6  # 0.4 (D) + 0.4 (O)
        assert abs(cfg.tier3_weight - 0.20) < 1e-6

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown deployment profile"):
            CompositeConfig.for_deployment("invalid_profile")  # type: ignore[arg-type]

    def test_validates_cleanly(self) -> None:
        for profile in ("standard", "high_security", "high_throughput"):
            CompositeConfig.for_deployment(profile).validate()  # no exception


# ---------------------------------------------------------------------------
# F2 privacy-first preset
# ---------------------------------------------------------------------------


class TestF2Preset:
    def test_f2_weights_match_paper_v9(self) -> None:
        cfg = CompositeConfig.f2_privacy_first()
        # Paper v9: F2=0.40, P=0.10, R=0.20, latency=0.15, throughput=0.15.
        assert cfg.weight_detection_f2 == 0.40
        assert cfg.weight_detection_precision == 0.10
        assert cfg.weight_detection_recall == 0.20
        assert cfg.weight_latency == 0.15
        assert cfg.weight_throughput == 0.15
        assert cfg.weight_detection_f1 == 0.0  # F1 is OFF when F2 is primary.

    def test_validates_cleanly(self) -> None:
        CompositeConfig.f2_privacy_first().validate()


# ---------------------------------------------------------------------------
# End-to-end composite with Tier 3 inputs
# ---------------------------------------------------------------------------


class TestComputeCompositeTier3:
    def test_composite_with_tier3_inputs(self) -> None:
        score = compute_composite(
            f1=0.80, precision=0.75, recall=0.85,
            latency_ms=50.0, docs_per_hour=500_000,
            reidentification_recall=0.40,
            reidentification_precision=0.85,
            quasi_identifiers_removed=7,
            quasi_identifiers_total=10,
            behavioral_signal_similarity=0.30,
            config=CompositeConfig.for_deployment("high_security"),
        )
        # Score is in [0, 1].
        assert 0.0 <= score.score <= 1.0
        # Tier 3 sub-score is populated.
        assert score.reidentification_sub > 0.0
        # Components contain the new Tier 3 metrics.
        assert "reidentification_resistance_normalized" in score.components
        assert "quasi_identifier_coverage_normalized" in score.components
        assert "behavioral_signal_leakage_normalized" in score.components
        assert "f2_normalized" in score.components

    def test_composite_without_tier3_inputs(self) -> None:
        """Tier 3 defaults to 0 when inputs are omitted."""
        score = compute_composite(
            f1=0.80, precision=0.75, recall=0.85,
            latency_ms=50.0, docs_per_hour=500_000,
        )
        assert score.reidentification_sub == 0.0
        assert score.components["reidentification_resistance_normalized"] == 0.0
        assert score.components["quasi_identifier_coverage_normalized"] == 0.0
        assert score.components["behavioral_signal_leakage_normalized"] == 0.0

    def test_f2_preset_rewards_recall_over_precision(self) -> None:
        """F2 preset ranks high-recall systems above high-precision systems."""
        # System A: high precision, low recall.
        a = compute_composite(
            f1=0.60, precision=0.95, recall=0.44,
            latency_ms=10.0, docs_per_hour=1_000_000,
            config=CompositeConfig.f2_privacy_first(),
        )
        # System B: lower precision, higher recall.
        b = compute_composite(
            f1=0.60, precision=0.50, recall=0.75,
            latency_ms=10.0, docs_per_hour=1_000_000,
            config=CompositeConfig.f2_privacy_first(),
        )
        # Under F2 weighting, B should score higher because recall is double-weighted.
        assert b.score > a.score

    def test_high_security_penalizes_reidentification(self) -> None:
        """high_security profile: vulnerable systems score lower."""
        # System A: great detection, BUT vulnerable to re-identification.
        a = compute_composite(
            f1=0.90, precision=0.90, recall=0.90,
            latency_ms=20.0, docs_per_hour=500_000,
            reidentification_recall=0.85,  # high attack success
            reidentification_precision=0.90,
            quasi_identifiers_removed=2, quasi_identifiers_total=10,
            behavioral_signal_similarity=0.80,
            config=CompositeConfig.for_deployment("high_security"),
        )
        # System B: slightly weaker detection, but much more resistant.
        b = compute_composite(
            f1=0.80, precision=0.80, recall=0.80,
            latency_ms=20.0, docs_per_hour=500_000,
            reidentification_recall=0.10,  # low attack success
            reidentification_precision=0.30,
            quasi_identifiers_removed=9, quasi_identifiers_total=10,
            behavioral_signal_similarity=0.10,
            config=CompositeConfig.for_deployment("high_security"),
        )
        assert b.score > a.score

    def test_serialization_includes_tier3(self) -> None:
        score = compute_composite(
            f1=0.80, precision=0.75, recall=0.85,
            latency_ms=50.0, docs_per_hour=500_000,
            reidentification_recall=0.40,
            reidentification_precision=0.85,
            config=CompositeConfig.for_deployment("standard"),
        )
        d = score.to_dict()
        assert "reidentification_sub" in d
        assert "deployment_profile" in d
        assert d["deployment_profile"] == "standard"


# ---------------------------------------------------------------------------
# Elo tournament extension
# ---------------------------------------------------------------------------


class TestReidentificationTournament:
    def test_rrs_tournament_ranks_resistant_higher(self) -> None:
        from pii_anon.eval_framework.rating.elo import PIIRateEloEngine

        engine = PIIRateEloEngine()
        rrs_scores = {
            "resistant": 0.95,
            "moderate": 0.50,
            "vulnerable": 0.10,
        }
        engine.run_reidentification_tournament(rrs_scores)
        leaderboard = engine.get_leaderboard()
        assert leaderboard[0].system_name == "resistant"
        assert leaderboard[-1].system_name == "vulnerable"
