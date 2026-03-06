"""Tests for paper-aligned enhancements to the pii-anon library.

Covers:
  1. Tier 2 adversarial normalization functions (Section 4.4)
  2. β-weighted Tier 1/Tier 2 composite composition (Section 4.4)
  3. Entity-type coverage breadth metric (Section 1)
  4. Governance thresholds (Section 4.7)
  5. Expanded confidence scoring (HIGH_FP_TYPES, context keywords)
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# 1. Tier 2 adversarial normalization functions
# ---------------------------------------------------------------------------

class TestNormalizeAttackSuccessRate:
    """N(ASR) = 1 - ASR  (lower ASR → higher score)."""

    def test_zero_asr_perfect_defense(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(0.0) == 1.0

    def test_full_extraction(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(1.0) == 0.0

    def test_midpoint(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(0.5) == pytest.approx(0.5)

    def test_clamps_negative(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(-0.1) == 1.0

    def test_clamps_over_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_attack_success_rate
        assert normalize_attack_success_rate(1.5) == 0.0


class TestNormalizeMiaAuc:
    """N(AUC) = clip(2(1-AUC), 0, 1).  AUC=0.5 → 1.0; AUC=1.0 → 0.0."""

    def test_random_baseline_auc(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        assert normalize_mia_auc(0.5) == pytest.approx(1.0)

    def test_perfect_attack_auc(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        assert normalize_mia_auc(1.0) == pytest.approx(0.0)

    def test_midpoint_auc(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        assert normalize_mia_auc(0.75) == pytest.approx(0.5)

    def test_below_random_clamps(self):
        from pii_anon.eval_framework.metrics.composite import normalize_mia_auc
        # AUC < 0.5 means worse than random → max score = 1.0
        assert normalize_mia_auc(0.3) == 1.0


class TestNormalizeCanaryExposure:
    """N(E) = exp(-E/c).  E=0 → 1.0; large E → 0.0."""

    def test_zero_exposure(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        assert normalize_canary_exposure(0.0) == 1.0

    def test_at_c(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        result = normalize_canary_exposure(5.0, c=5.0)
        assert result == pytest.approx(math.exp(-1.0))

    def test_large_exposure(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        result = normalize_canary_exposure(50.0, c=5.0)
        assert result < 0.001

    def test_custom_c(self):
        from pii_anon.eval_framework.metrics.composite import normalize_canary_exposure
        result = normalize_canary_exposure(10.0, c=10.0)
        assert result == pytest.approx(math.exp(-1.0))


class TestNormalizeKAnonymity:
    """N(k) = clip(log(k)/log(k_max), 0, 1).  k=1 → 0; k=k_max → 1."""

    def test_k_equals_one(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(1) == 0.0

    def test_k_equals_kmax(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(100, k_max=100) == pytest.approx(1.0)

    def test_k_midpoint(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        result = normalize_k_anonymity(10, k_max=100)
        assert result == pytest.approx(0.5)

    def test_k_zero(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(0) == 0.0

    def test_k_exceeds_kmax(self):
        from pii_anon.eval_framework.metrics.composite import normalize_k_anonymity
        assert normalize_k_anonymity(200, k_max=100) == 1.0


class TestNormalizeEpsilonDP:
    """N(ε) = exp(-ε/ε₀) with optional δ gate."""

    def test_zero_epsilon(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        assert normalize_epsilon_dp(0.0) == 1.0

    def test_at_epsilon_0(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        result = normalize_epsilon_dp(1.0, epsilon_0=1.0)
        assert result == pytest.approx(math.exp(-1.0))

    def test_large_epsilon(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        result = normalize_epsilon_dp(10.0, epsilon_0=1.0)
        assert result < 0.001

    def test_delta_gate_penalty(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        without_delta = normalize_epsilon_dp(1.0, epsilon_0=1.0)
        with_delta = normalize_epsilon_dp(
            1.0, epsilon_0=1.0, delta=1e-3, delta_threshold=1e-5,
        )
        assert with_delta == pytest.approx(without_delta * 0.5)

    def test_delta_below_threshold_no_penalty(self):
        from pii_anon.eval_framework.metrics.composite import normalize_epsilon_dp
        without_delta = normalize_epsilon_dp(1.0, epsilon_0=1.0)
        with_delta = normalize_epsilon_dp(
            1.0, epsilon_0=1.0, delta=1e-7, delta_threshold=1e-5,
        )
        assert with_delta == pytest.approx(without_delta)


# ---------------------------------------------------------------------------
# 2. β-weighted Tier 1/Tier 2 composite composition
# ---------------------------------------------------------------------------

class TestBetaWeightedComposition:
    """C_s = β · C_s^(1) + (1-β) · C_s^(2)."""

    def test_beta_one_is_tier1_only(self):
        from pii_anon.eval_framework.metrics.composite import (
            CompositeConfig, compute_composite,
        )
        cfg = CompositeConfig(
            beta_tier_balance=1.0,
            weight_privacy=0.3,
            weight_utility=0.2,
        )
        result = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            privacy_score=1.0, utility_score=1.0,
            config=cfg,
        )
        # β=1 means Tier 2 should be ignored
        result_t1_only = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            config=CompositeConfig(),  # default: no Tier 2
        )
        assert result.score == pytest.approx(result_t1_only.score, abs=1e-5)

    def test_beta_zero_is_tier2_only(self):
        from pii_anon.eval_framework.metrics.composite import (
            CompositeConfig, compute_composite,
        )
        cfg = CompositeConfig(
            beta_tier_balance=0.0,
            weight_privacy=0.5,
            weight_utility=0.5,
            alpha_privacy=0.5,
        )
        result = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            privacy_score=0.8, utility_score=0.6,
            config=cfg,
        )
        # β=0 means score = pure Tier 2 = α·P + (1-α)·U = 0.5*0.8 + 0.5*0.6 = 0.7
        assert result.score == pytest.approx(0.7, abs=1e-5)

    def test_beta_half_blends_tiers(self):
        from pii_anon.eval_framework.metrics.composite import (
            CompositeConfig, compute_composite,
        )
        cfg = CompositeConfig(
            beta_tier_balance=0.5,
            weight_privacy=0.5,
            weight_utility=0.5,
            alpha_privacy=0.5,
        )
        result = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            privacy_score=0.8, utility_score=0.6,
            config=cfg,
        )
        # Score should be between pure Tier 1 and pure Tier 2
        t1 = result.components["tier1_score"]
        t2 = result.components["tier2_score"]
        expected = 0.5 * t1 + 0.5 * t2
        assert result.score == pytest.approx(expected, abs=1e-5)

    def test_beta_validation(self):
        from pii_anon.eval_framework.metrics.composite import CompositeConfig
        cfg = CompositeConfig(beta_tier_balance=1.5)
        with pytest.raises(ValueError, match="beta_tier_balance"):
            cfg.validate()


# ---------------------------------------------------------------------------
# 3. Entity-type coverage breadth metric
# ---------------------------------------------------------------------------

class TestEntityCoverage:
    """normalize_entity_coverage(detected, total)."""

    def test_full_coverage(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        assert normalize_entity_coverage(13, 13) == pytest.approx(1.0)

    def test_single_type(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        assert normalize_entity_coverage(1, 13) == pytest.approx(1 / 13)

    def test_zero_detected(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        assert normalize_entity_coverage(0, 13) == 0.0

    def test_zero_total(self):
        from pii_anon.eval_framework.metrics.composite import normalize_entity_coverage
        assert normalize_entity_coverage(5, 0) == 0.0

    def test_coverage_in_composite(self):
        from pii_anon.eval_framework.metrics.composite import (
            CompositeConfig, compute_composite,
        )
        # With coverage enabled, more entity types → higher score
        cfg = CompositeConfig(weight_coverage=0.10, weight_throughput=0.00)
        broad = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            entity_types_detected=13, entity_types_total=13,
            config=cfg,
        )
        narrow = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            entity_types_detected=1, entity_types_total=13,
            config=cfg,
        )
        assert broad.score > narrow.score

    def test_coverage_disabled_by_default(self):
        from pii_anon.eval_framework.metrics.composite import compute_composite
        # weight_coverage=0 by default, so coverage shouldn't affect score
        result = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            entity_types_detected=13, entity_types_total=13,
        )
        result_nocov = compute_composite(
            f1=0.6, precision=0.5, recall=0.7,
            latency_ms=5.0, docs_per_hour=500_000,
            entity_types_detected=0, entity_types_total=13,
        )
        assert result.score == pytest.approx(result_nocov.score, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Governance thresholds (Section 4.7)
# ---------------------------------------------------------------------------

class TestGovernanceThresholds:
    """R > 1500, RD < 100 as production-grade gate."""

    def test_system_passes_governance(self):
        from pii_anon.eval_framework.rating.elo import (
            PIIRateEloEngine, GovernanceThresholds,
        )
        engine = PIIRateEloEngine()
        # Play enough rounds to lower RD
        composites = {"good": 0.9, "bad": 0.1}
        for _ in range(20):
            engine.run_round_robin(composites)
        result = engine.evaluate_governance(
            "good",
            thresholds=GovernanceThresholds(
                min_rating=1500.0, max_rd=200.0, min_matches=6,
            ),
        )
        assert result.passed is True
        assert result.min_rating_met is True
        assert result.min_matches_met is True

    def test_system_fails_rating(self):
        from pii_anon.eval_framework.rating.elo import (
            PIIRateEloEngine, GovernanceThresholds,
        )
        engine = PIIRateEloEngine()
        composites = {"good": 0.9, "bad": 0.1}
        for _ in range(20):
            engine.run_round_robin(composites)
        result = engine.evaluate_governance(
            "bad",
            thresholds=GovernanceThresholds(
                min_rating=1500.0, max_rd=200.0, min_matches=6,
            ),
        )
        assert result.passed is False
        assert result.min_rating_met is False

    def test_system_fails_rd(self):
        from pii_anon.eval_framework.rating.elo import (
            PIIRateEloEngine, GovernanceThresholds,
        )
        engine = PIIRateEloEngine()
        composites = {"a": 0.7, "b": 0.6}
        engine.run_round_robin(composites)  # only 1 match each
        result = engine.evaluate_governance(
            "a",
            thresholds=GovernanceThresholds(
                min_rating=1400.0, max_rd=50.0, min_matches=1,
            ),
        )
        assert result.max_rd_met is False

    def test_system_fails_min_matches(self):
        from pii_anon.eval_framework.rating.elo import (
            PIIRateEloEngine, GovernanceThresholds,
        )
        engine = PIIRateEloEngine()
        composites = {"a": 0.8, "b": 0.2}
        engine.run_round_robin(composites)
        result = engine.evaluate_governance(
            "a",
            thresholds=GovernanceThresholds(
                min_rating=1400.0, max_rd=400.0, min_matches=10,
            ),
        )
        assert result.min_matches_met is False
        assert result.passed is False

    def test_unknown_system_fails(self):
        from pii_anon.eval_framework.rating.elo import PIIRateEloEngine
        engine = PIIRateEloEngine()
        result = engine.evaluate_governance("nonexistent")
        assert result.passed is False
        assert "not found" in result.notes[0].lower()

    def test_evaluate_all_governance(self):
        from pii_anon.eval_framework.rating.elo import (
            PIIRateEloEngine, GovernanceThresholds,
        )
        engine = PIIRateEloEngine()
        composites = {"a": 0.9, "b": 0.5, "c": 0.1}
        for _ in range(10):
            engine.run_round_robin(composites)
        results = engine.evaluate_all_governance(
            thresholds=GovernanceThresholds(
                min_rating=1400.0, max_rd=300.0, min_matches=5,
            ),
        )
        assert len(results) == 3
        # Results should be sorted by rating descending
        assert results[0].rating >= results[1].rating >= results[2].rating

    def test_governance_result_to_dict(self):
        from pii_anon.eval_framework.rating.elo import GovernanceResult
        result = GovernanceResult(
            system_name="test",
            passed=True,
            rating=1550.0,
            rd=80.0,
            min_rating_met=True,
            max_rd_met=True,
            min_matches_met=True,
            notes=["OK"],
        )
        d = result.to_dict()
        assert d["system_name"] == "test"
        assert d["passed"] is True
        assert d["rating"] == 1550.0

    def test_governance_thresholds_to_dict(self):
        from pii_anon.eval_framework.rating.elo import GovernanceThresholds
        t = GovernanceThresholds(min_rating=1600.0, max_rd=80.0, min_matches=12)
        d = t.to_dict()
        assert d["min_rating"] == 1600.0
        assert d["max_rd"] == 80.0
        assert d["min_matches"] == 12


# ---------------------------------------------------------------------------
# 5. Expanded confidence scoring
# ---------------------------------------------------------------------------

class TestExpandedHighFPTypes:
    """HIGH_FP_TYPES should include LOCATION, ORGANIZATION, ADDRESS."""

    def test_high_fp_types_expanded(self):
        from pii_anon.engines.regex.confidence import HIGH_FP_TYPES
        assert "PERSON_NAME" in HIGH_FP_TYPES
        assert "US_SSN" in HIGH_FP_TYPES
        assert "LOCATION" in HIGH_FP_TYPES
        assert "ORGANIZATION" in HIGH_FP_TYPES
        assert "ADDRESS" in HIGH_FP_TYPES

    def test_location_penalized_without_context(self):
        from pii_anon.engines.regex.confidence import adjust_confidence
        result = adjust_confidence("LOCATION", 0.75, "some random text here", 5, 10)
        assert result < 0.75  # penalized

    def test_organization_penalized_without_context(self):
        from pii_anon.engines.regex.confidence import adjust_confidence
        result = adjust_confidence("ORGANIZATION", 0.75, "some random text here", 5, 10)
        assert result < 0.75

    def test_address_penalized_without_context(self):
        from pii_anon.engines.regex.confidence import adjust_confidence
        result = adjust_confidence("ADDRESS", 0.75, "some random text here", 5, 10)
        assert result < 0.75


class TestExpandedContextWords:
    """New entity types should have context keywords."""

    def test_date_of_birth_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("DATE_OF_BIRTH", "date of birth dob")

    def test_bank_account_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("BANK_ACCOUNT", "bank account number")

    def test_drivers_license_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("DRIVERS_LICENSE", "driver license number")

    def test_passport_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("PASSPORT", "passport number travel")

    def test_national_id_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("NATIONAL_ID", "national identity card")

    def test_vin_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("VIN", "vehicle identification number vin")

    def test_mac_address_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("MAC_ADDRESS", "mac address hardware device")

    def test_organization_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("ORGANIZATION", "company corporation inc")

    def test_location_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("LOCATION", "city location region")

    def test_address_context(self):
        from pii_anon.engines.regex.confidence import has_context_words
        assert has_context_words("ADDRESS", "street address avenue")

    def test_context_boost_for_date_of_birth(self):
        from pii_anon.engines.regex.confidence import adjust_confidence
        result = adjust_confidence(
            "DATE_OF_BIRTH", 0.80,
            "the date of birth is 1990-01-15 in our records", 21, 31,
        )
        assert result > 0.80  # boosted


# ---------------------------------------------------------------------------
# 6. Composite backward compatibility
# ---------------------------------------------------------------------------

class TestCompositeBackwardCompatibility:
    """Existing compute_composite calls must still work without new args."""

    def test_default_call_unchanged(self):
        from pii_anon.eval_framework.metrics.composite import compute_composite
        result = compute_composite(
            f1=0.612, precision=0.580, recall=0.647,
            latency_ms=6.94, docs_per_hour=514_000,
        )
        assert 0.0 <= result.score <= 1.0
        assert result.components["tier1_score"] > 0
        assert result.components["tier2_score"] == 0.0

    def test_compute_composite_from_benchmark_result_unchanged(self):
        from pii_anon.eval_framework.metrics.composite import (
            compute_composite_from_benchmark_result,
        )

        class FakeResult:
            f1 = 0.612
            precision = 0.580
            recall = 0.647
            latency_p50_ms = 6.94
            docs_per_hour = 514_000

        result = compute_composite_from_benchmark_result(FakeResult())
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# 7. Top-level import smoke test
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    """New exports should be importable from eval_framework."""

    def test_tier2_normalizers_importable(self):
        from pii_anon.eval_framework import (  # noqa: F401
            normalize_attack_success_rate,
            normalize_canary_exposure,
            normalize_entity_coverage,
            normalize_epsilon_dp,
            normalize_k_anonymity,
            normalize_mia_auc,
        )

    def test_governance_importable(self):
        from pii_anon.eval_framework import (  # noqa: F401
            GovernanceResult,
            GovernanceThresholds,
        )
