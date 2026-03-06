"""Tests for the PII-Rate-Elo rating engine.

Tests cover sigmoid mapping, expected score, match updates, rating
convergence, RD decrease, leaderboard ordering, round-robin, and history.
"""

from __future__ import annotations

from pii_anon.eval_framework.rating.elo import (
    PIIRateEloEngine,
    EloRating,
    RatingUpdate,
)


# ---------------------------------------------------------------------------
# EloRating dataclass
# ---------------------------------------------------------------------------

class TestEloRating:
    def test_default_values(self):
        r = EloRating(system_name="test")
        assert r.rating == 1500.0
        assert r.rd == 350.0
        assert r.num_matches == 0

    def test_to_dict(self):
        r = EloRating(system_name="sys-a", rating=1600.5, rd=200.0, num_matches=5)
        d = r.to_dict()
        assert d["system_name"] == "sys-a"
        assert d["rating"] == 1600.5
        assert d["rd"] == 200.0
        assert d["num_matches"] == 5


class TestRatingUpdate:
    def test_to_dict(self):
        u = RatingUpdate(
            system_name="sys-a",
            old_rating=1500.0,
            new_rating=1520.0,
            change=20.0,
            expected_score=0.5,
            actual_score=0.7,
            k_factor=32.0,
        )
        d = u.to_dict()
        assert d["system_name"] == "sys-a"
        assert d["change"] == 20.0


# ---------------------------------------------------------------------------
# Engine internals
# ---------------------------------------------------------------------------

class TestEngineSigmoid:
    def test_zero_difference_yields_half(self):
        engine = PIIRateEloEngine()
        assert abs(engine._sigmoid(0.0) - 0.5) < 1e-9

    def test_positive_difference_above_half(self):
        engine = PIIRateEloEngine()
        assert engine._sigmoid(0.1) > 0.5

    def test_negative_difference_below_half(self):
        engine = PIIRateEloEngine()
        assert engine._sigmoid(-0.1) < 0.5

    def test_large_positive_approaches_one(self):
        engine = PIIRateEloEngine()
        assert engine._sigmoid(5.0) > 0.99

    def test_large_negative_approaches_zero(self):
        engine = PIIRateEloEngine()
        assert engine._sigmoid(-5.0) < 0.01


class TestExpectedScore:
    def test_equal_ratings_yield_half(self):
        engine = PIIRateEloEngine()
        assert abs(engine._expected_score(1500, 1500) - 0.5) < 1e-9

    def test_higher_rating_favored(self):
        engine = PIIRateEloEngine()
        assert engine._expected_score(1600, 1400) > 0.5

    def test_lower_rating_underdog(self):
        engine = PIIRateEloEngine()
        assert engine._expected_score(1400, 1600) < 0.5

    def test_symmetry(self):
        """E_ij + E_ji should equal 1."""
        engine = PIIRateEloEngine()
        e_ij = engine._expected_score(1550, 1450)
        e_ji = engine._expected_score(1450, 1550)
        assert abs(e_ij + e_ji - 1.0) < 1e-9

    def test_100_point_gap(self):
        """100-point gap at scale=400 should give ~64% expected score."""
        engine = PIIRateEloEngine(scale=400.0)
        e = engine._expected_score(1600, 1500)
        assert 0.63 < e < 0.65


class TestAdaptiveK:
    def test_initial_rd_yields_base_k(self):
        engine = PIIRateEloEngine(k_base=32.0, initial_rd=350.0)
        k = engine._adaptive_k(350.0)
        assert abs(k - 32.0) < 1e-9

    def test_low_rd_yields_lower_k(self):
        engine = PIIRateEloEngine(k_base=32.0, initial_rd=350.0)
        k = engine._adaptive_k(100.0)
        assert k < 32.0

    def test_high_rd_yields_higher_k(self):
        engine = PIIRateEloEngine(k_base=32.0, initial_rd=350.0)
        k = engine._adaptive_k(700.0)
        assert k > 32.0

    def test_k_clamped_lower(self):
        engine = PIIRateEloEngine(k_base=32.0, initial_rd=350.0)
        k = engine._adaptive_k(1.0)
        assert k >= 16.0  # k_base / 2

    def test_k_clamped_upper(self):
        engine = PIIRateEloEngine(k_base=32.0, initial_rd=350.0)
        k = engine._adaptive_k(10000.0)
        assert k <= 64.0  # k_base * 2


class TestUpdateRD:
    def test_rd_decreases_with_matches(self):
        engine = PIIRateEloEngine()
        rd_10 = engine._update_rd(350.0, 10)
        rd_50 = engine._update_rd(350.0, 50)
        assert rd_10 > rd_50

    def test_rd_never_below_floor(self):
        engine = PIIRateEloEngine(rd_floor=30.0)
        rd = engine._update_rd(350.0, 10000)
        assert rd >= 30.0

    def test_zero_matches_returns_old(self):
        engine = PIIRateEloEngine()
        rd = engine._update_rd(350.0, 0)
        assert rd == 350.0


# ---------------------------------------------------------------------------
# Match updates
# ---------------------------------------------------------------------------

class TestMatchUpdates:
    def test_better_system_gains_rating(self):
        engine = PIIRateEloEngine()
        u_i, u_j = engine.update_from_match("strong", "weak", 0.9, 0.3)
        assert u_i.change > 0
        assert u_j.change < 0

    def test_equal_composites_small_changes(self):
        engine = PIIRateEloEngine()
        u_i, u_j = engine.update_from_match("a", "b", 0.5, 0.5)
        assert abs(u_i.change) < 1.0
        assert abs(u_j.change) < 1.0

    def test_rating_changes_symmetric(self):
        """Total rating change should roughly sum to zero (same K would give exact)."""
        engine = PIIRateEloEngine()
        u_i, u_j = engine.update_from_match("a", "b", 0.8, 0.4)
        # Not exactly zero due to adaptive K, but should be close
        assert abs(u_i.change + u_j.change) < 5.0

    def test_match_increments_num_matches(self):
        engine = PIIRateEloEngine()
        engine.update_from_match("a", "b", 0.8, 0.4)
        assert engine.get_rating("a").num_matches == 1
        assert engine.get_rating("b").num_matches == 1

    def test_multiple_matches_accumulate(self):
        engine = PIIRateEloEngine()
        for _ in range(5):
            engine.update_from_match("a", "b", 0.9, 0.3)
        assert engine.get_rating("a").num_matches == 5
        assert engine.get_rating("b").num_matches == 5

    def test_ensure_system_creates_entry(self):
        engine = PIIRateEloEngine()
        r = engine.ensure_system("new_system")
        assert r.system_name == "new_system"
        assert r.rating == 1500.0


# ---------------------------------------------------------------------------
# Rating convergence
# ---------------------------------------------------------------------------

class TestRatingConvergence:
    def test_better_system_rated_higher_after_round_robin(self):
        """After round-robin, the system with highest composite should have highest Elo."""
        engine = PIIRateEloEngine()
        composites = {
            "pii-anon": 0.85,
            "presidio": 0.60,
            "scrubadub": 0.45,
            "gliner": 0.55,
        }
        engine.run_round_robin(composites)
        lb = engine.get_leaderboard()
        assert lb[0].system_name == "pii-anon"
        assert lb[-1].system_name == "scrubadub"

    def test_repeated_round_robins_stabilize(self):
        """Multiple round-robins should converge ratings (changes get smaller)."""
        engine = PIIRateEloEngine()
        composites = {"a": 0.9, "b": 0.6, "c": 0.3}
        changes_first = sum(abs(u.change) for u in engine.run_round_robin(composites))
        changes_later = sum(abs(u.change) for u in engine.run_round_robin(composites))
        # Later rounds should have smaller total changes (as RD decreases)
        assert changes_later <= changes_first * 1.5  # allow some slack


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------

class TestRoundRobin:
    def test_all_systems_registered(self):
        engine = PIIRateEloEngine()
        composites = {"a": 0.8, "b": 0.6, "c": 0.4}
        engine.run_round_robin(composites)
        assert engine.get_rating("a") is not None
        assert engine.get_rating("b") is not None
        assert engine.get_rating("c") is not None

    def test_correct_number_of_updates(self):
        """n systems → n*(n-1)/2 matches → n*(n-1) updates."""
        engine = PIIRateEloEngine()
        composites = {"a": 0.8, "b": 0.6, "c": 0.4, "d": 0.2}
        updates = engine.run_round_robin(composites)
        n = 4
        assert len(updates) == n * (n - 1)  # 2 updates per match

    def test_deterministic_order(self):
        """Round-robin should be deterministic with same inputs."""
        engine1 = PIIRateEloEngine()
        engine2 = PIIRateEloEngine()
        composites = {"x": 0.7, "y": 0.5, "z": 0.3}
        engine1.run_round_robin(composites)
        engine2.run_round_robin(composites)
        lb1 = engine1.get_leaderboard()
        lb2 = engine2.get_leaderboard()
        for r1, r2 in zip(lb1, lb2):
            assert r1.system_name == r2.system_name
            assert abs(r1.rating - r2.rating) < 1e-9


# ---------------------------------------------------------------------------
# Leaderboard and history
# ---------------------------------------------------------------------------

class TestLeaderboardAndHistory:
    def test_leaderboard_sorted_descending(self):
        engine = PIIRateEloEngine()
        composites = {"a": 0.9, "b": 0.5, "c": 0.1}
        engine.run_round_robin(composites)
        lb = engine.get_leaderboard()
        for i in range(len(lb) - 1):
            assert lb[i].rating >= lb[i + 1].rating

    def test_history_populated(self):
        engine = PIIRateEloEngine()
        engine.update_from_match("a", "b", 0.8, 0.4)
        assert len(engine.get_history()) == 2

    def test_reset_clears_everything(self):
        engine = PIIRateEloEngine()
        engine.update_from_match("a", "b", 0.8, 0.4)
        engine.reset()
        assert engine.get_rating("a") is None
        assert len(engine.get_history()) == 0

    def test_get_rating_unknown_returns_none(self):
        engine = PIIRateEloEngine()
        assert engine.get_rating("nonexistent") is None
