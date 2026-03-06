"""Tests for Elo/Glicko-style rating engine.

Tests cover rating initialization, pairwise matches, round-robin tournaments,
rating decay, and auto-calibration.
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.rating.elo import (
    EloRating,
    RatingUpdate,
    CalibrationResult,
    PIIRateEloEngine,
)


class TestEloRating:
    """Test EloRating dataclass."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        rating = EloRating(system_name="system_a")

        assert rating.system_name == "system_a"
        assert rating.rating == 1500.0
        assert rating.rd == 350.0
        assert rating.num_matches == 0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        rating = EloRating(
            system_name="custom",
            rating=1600.0,
            rd=200.0,
            num_matches=10,
        )

        assert rating.rating == 1600.0
        assert rating.rd == 200.0
        assert rating.num_matches == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rating = EloRating(
            system_name="test",
            rating=1550.5,
            rd=325.3,
            num_matches=5,
        )

        d = rating.to_dict()

        assert d["system_name"] == "test"
        assert d["rating"] == 1550.5
        assert d["rd"] == 325.3
        assert d["num_matches"] == 5

    def test_to_dict_rounds_values(self):
        """Test that to_dict rounds values."""
        rating = EloRating(
            system_name="test",
            rating=1500.123456,
            rd=350.987654,
        )

        d = rating.to_dict()

        # Values should be rounded to 2 decimal places
        assert len(str(d["rating"]).split(".")[-1]) <= 2


class TestRatingUpdate:
    """Test RatingUpdate dataclass."""

    def test_init(self):
        """Test initialization."""
        update = RatingUpdate(
            system_name="system_a",
            old_rating=1500.0,
            new_rating=1520.0,
            change=20.0,
            expected_score=0.45,
            actual_score=0.6,
            k_factor=32.0,
        )

        assert update.system_name == "system_a"
        assert update.change == 20.0
        assert update.expected_score == 0.45

    def test_to_dict(self):
        """Test conversion to dictionary."""
        update = RatingUpdate(
            system_name="test",
            old_rating=1500.0,
            new_rating=1520.5,
            change=20.5,
            expected_score=0.45123,
            actual_score=0.60789,
            k_factor=32.0,
        )

        d = update.to_dict()

        assert d["system_name"] == "test"
        assert d["old_rating"] == 1500.0
        assert d["new_rating"] == 1520.5


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""

    def test_init(self):
        """Test initialization."""
        result = CalibrationResult(
            old_parameters={"gamma": 10.0, "scale": 400.0},
            new_parameters={"gamma": 12.0, "scale": 400.0},
            score_distribution={"mean": 0.75, "std": 0.1},
            recommendation="Increase gamma for sharper sigmoid",
        )

        assert result.old_parameters["gamma"] == 10.0
        assert result.new_parameters["gamma"] == 12.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = CalibrationResult(
            old_parameters={"gamma": 10.0},
            new_parameters={"gamma": 12.0},
            score_distribution={"mean": 0.75},
        )

        d = result.to_dict()

        assert "old_parameters" in d
        assert "new_parameters" in d
        assert "score_distribution" in d


class TestPIIRateEloEngineInit:
    """Test PIIRateEloEngine initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        engine = PIIRateEloEngine()

        assert engine._initial_rating == 1500.0
        assert engine._initial_rd == 350.0
        assert engine._k_base == 32.0
        assert engine._scale == 400.0
        assert engine._gamma == 10.0
        assert engine._rd_floor == 30.0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        engine = PIIRateEloEngine(
            initial_rating=1600.0,
            initial_rd=300.0,
            k_base=24.0,
            scale=500.0,
            gamma=8.0,
            rd_floor=20.0,
        )

        assert engine._initial_rating == 1600.0
        assert engine._initial_rd == 300.0
        assert engine._k_base == 24.0
        assert engine._scale == 500.0
        assert engine._gamma == 8.0
        assert engine._rd_floor == 20.0

    def test_init_empty_ratings_and_history(self):
        """Test that engine starts with empty ratings and history."""
        engine = PIIRateEloEngine()

        assert len(engine._ratings) == 0
        assert len(engine._history) == 0


class TestPIIRateEloEngineSigmoid:
    """Test sigmoid function."""

    def test_sigmoid_zero(self):
        """Test sigmoid at x=0."""
        engine = PIIRateEloEngine(gamma=10.0)

        result = engine._sigmoid(0.0)

        assert result == pytest.approx(0.5, abs=1e-6)

    def test_sigmoid_positive(self):
        """Test sigmoid with positive input."""
        engine = PIIRateEloEngine(gamma=10.0)

        result = engine._sigmoid(0.1)

        assert result > 0.5
        assert 0.0 <= result <= 1.0

    def test_sigmoid_negative(self):
        """Test sigmoid with negative input."""
        engine = PIIRateEloEngine(gamma=10.0)

        result = engine._sigmoid(-0.1)

        assert result < 0.5
        assert 0.0 <= result <= 1.0

    def test_sigmoid_bounds(self):
        """Test that sigmoid output is in [0, 1]."""
        engine = PIIRateEloEngine(gamma=10.0)

        for x in [-10, -1, 0, 1, 10]:
            result = engine._sigmoid(x)
            assert 0.0 <= result <= 1.0

    def test_sigmoid_monotone_increasing(self):
        """Test that sigmoid is monotone increasing."""
        engine = PIIRateEloEngine(gamma=10.0)

        values = [-5, -2, 0, 2, 5]
        results = [engine._sigmoid(x) for x in values]

        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


class TestPIIRateEloEngineExpectedScore:
    """Test expected score calculation."""

    def test_expected_score_equal_ratings(self):
        """Test expected score when ratings are equal."""
        engine = PIIRateEloEngine()

        result = engine._expected_score(1500.0, 1500.0)

        assert result == pytest.approx(0.5, abs=1e-6)

    def test_expected_score_higher_rating_favored(self):
        """Test that higher rating gives higher expected score."""
        engine = PIIRateEloEngine()

        higher_rating = engine._expected_score(1600.0, 1500.0)
        lower_rating = engine._expected_score(1500.0, 1600.0)

        assert higher_rating > 0.5
        assert lower_rating < 0.5
        assert higher_rating > lower_rating

    def test_expected_score_bounds(self):
        """Test that expected score is in [0, 1]."""
        engine = PIIRateEloEngine()

        for r1 in [1000, 1500, 2000]:
            for r2 in [1000, 1500, 2000]:
                result = engine._expected_score(r1, r2)
                assert 0.0 <= result <= 1.0

    def test_expected_score_symmetric(self):
        """Test that expected scores sum to 1."""
        engine = PIIRateEloEngine()

        e1 = engine._expected_score(1600.0, 1500.0)
        e2 = engine._expected_score(1500.0, 1600.0)

        assert (e1 + e2) == pytest.approx(1.0, abs=1e-6)


class TestPIIRateEloEngineAdaptiveK:
    """Test adaptive K-factor calculation."""

    def test_adaptive_k_default_at_initial_rd(self):
        """Test adaptive K at initial RD."""
        engine = PIIRateEloEngine(initial_rd=350.0, k_base=32.0)

        k = engine._adaptive_k(350.0)

        assert k == pytest.approx(32.0, abs=1e-6)

    def test_adaptive_k_higher_at_higher_rd(self):
        """Test that K increases with RD."""
        engine = PIIRateEloEngine(initial_rd=350.0, k_base=32.0)

        k_low = engine._adaptive_k(100.0)
        k_high = engine._adaptive_k(700.0)

        assert k_high > k_low

    def test_adaptive_k_bounded(self):
        """Test that K is bounded by min and max."""
        engine = PIIRateEloEngine(k_base=32.0)

        k_very_high = engine._adaptive_k(10000.0)
        k_very_low = engine._adaptive_k(0.1)

        assert k_very_high <= 32.0 * 2.0
        assert k_very_low >= 32.0 / 2.0


class TestPIIRateEloEngineUpdateRd:
    """Test RD update calculation."""

    def test_update_rd_no_matches_unchanged(self):
        """Test that RD doesn't change with 0 matches."""
        engine = PIIRateEloEngine(initial_rd=350.0, rd_floor=30.0)

        new_rd = engine._update_rd(350.0, 0)

        assert new_rd == 350.0

    def test_update_rd_decreases_with_matches(self):
        """Test that RD decreases with more matches."""
        engine = PIIRateEloEngine(initial_rd=350.0, rd_floor=30.0)

        rd_1 = engine._update_rd(350.0, 1)
        rd_10 = engine._update_rd(350.0, 10)

        assert rd_10 < rd_1
        assert rd_1 < 350.0

    def test_update_rd_bounded_by_floor(self):
        """Test that RD is bounded below by floor."""
        engine = PIIRateEloEngine(initial_rd=350.0, rd_floor=30.0)

        new_rd = engine._update_rd(350.0, 1000)

        assert new_rd >= 30.0

    def test_update_rd_approaches_floor(self):
        """Test that RD approaches floor with many matches."""
        engine = PIIRateEloEngine(initial_rd=350.0, rd_floor=30.0)

        # With many matches, RD should approach floor
        rd_final = engine._update_rd(350.0, 100000)

        assert rd_final == pytest.approx(30.0, abs=1.0)


class TestPIIRateEloEngineEnsureSystem:
    """Test ensure_system method."""

    def test_ensure_system_creates_new(self):
        """Test that ensure_system creates new system."""
        engine = PIIRateEloEngine()

        rating = engine.ensure_system("system_a")

        assert rating.system_name == "system_a"
        assert rating.rating == 1500.0

    def test_ensure_system_returns_existing(self):
        """Test that ensure_system returns existing system."""
        engine = PIIRateEloEngine()

        r1 = engine.ensure_system("system_a")
        r2 = engine.ensure_system("system_a")

        assert r1 is r2

    def test_ensure_system_modifies_returned_object(self):
        """Test that modifying returned rating affects engine."""
        engine = PIIRateEloEngine()

        rating = engine.ensure_system("system_a")
        rating.rating = 1600.0

        retrieved = engine.get_rating("system_a")

        assert retrieved.rating == 1600.0


class TestPIIRateEloEngineUpdateFromMatch:
    """Test update_from_match method."""

    def test_update_from_match_returns_two_updates(self):
        """Test that update returns two RatingUpdate objects."""
        engine = PIIRateEloEngine()

        u1, u2 = engine.update_from_match(
            "system_a", "system_b", 0.8, 0.7
        )

        assert isinstance(u1, RatingUpdate)
        assert isinstance(u2, RatingUpdate)
        assert u1.system_name == "system_a"
        assert u2.system_name == "system_b"

    def test_update_from_match_winner_gains_rating(self):
        """Test that winner gains rating and loser loses it."""
        engine = PIIRateEloEngine()

        # System A has higher composite score (0.9 vs 0.1)
        u_a, u_b = engine.update_from_match(
            "system_a", "system_b", 0.9, 0.1
        )

        assert u_a.change > 0.0  # Winner gains
        assert u_b.change < 0.0  # Loser loses

    def test_update_from_match_creates_systems(self):
        """Test that match creates systems if they don't exist."""
        engine = PIIRateEloEngine()

        assert engine.get_rating("new_a") is None
        assert engine.get_rating("new_b") is None

        engine.update_from_match("new_a", "new_b", 0.8, 0.7)

        assert engine.get_rating("new_a") is not None
        assert engine.get_rating("new_b") is not None

    def test_update_from_match_updates_num_matches(self):
        """Test that num_matches is incremented."""
        engine = PIIRateEloEngine()

        engine.update_from_match("system_a", "system_b", 0.8, 0.7)

        assert engine.get_rating("system_a").num_matches == 1
        assert engine.get_rating("system_b").num_matches == 1

    def test_update_from_match_updates_rd(self):
        """Test that RD is updated after match."""
        engine = PIIRateEloEngine()

        r_a = engine.ensure_system("system_a")
        old_rd = r_a.rd

        engine.update_from_match("system_a", "system_b", 0.8, 0.7)

        # RD should decrease with the match
        assert engine.get_rating("system_a").rd < old_rd

    def test_update_from_match_adds_to_history(self):
        """Test that match is added to history."""
        engine = PIIRateEloEngine()

        assert len(engine.get_history()) == 0

        engine.update_from_match("system_a", "system_b", 0.8, 0.7)

        assert len(engine.get_history()) == 2


class TestPIIRateEloEngineRunRoundRobin:
    """Test run_round_robin method."""

    def test_run_round_robin_single_pair(self):
        """Test round-robin with two systems."""
        engine = PIIRateEloEngine()

        composites = {"system_a": 0.8, "system_b": 0.7}

        updates = engine.run_round_robin(composites)

        assert len(updates) == 2

    def test_run_round_robin_three_systems(self):
        """Test round-robin with three systems."""
        engine = PIIRateEloEngine()

        composites = {"a": 0.9, "b": 0.8, "c": 0.7}

        updates = engine.run_round_robin(composites)

        # 3 systems -> 3 pairs -> 6 updates
        assert len(updates) == 6

    def test_run_round_robin_deterministic(self):
        """Test that round-robin is deterministic."""
        engine1 = PIIRateEloEngine()
        engine2 = PIIRateEloEngine()

        composites = {"b": 0.8, "a": 0.7, "c": 0.9}

        updates1 = engine1.run_round_robin(composites)
        updates2 = engine2.run_round_robin(composites)

        # Results should be identical
        assert len(updates1) == len(updates2)
        for u1, u2 in zip(updates1, updates2):
            assert u1.system_name == u2.system_name
            assert u1.change == pytest.approx(u2.change, abs=1e-6)

    def test_run_round_robin_creates_systems(self):
        """Test that round-robin creates systems."""
        engine = PIIRateEloEngine()

        composites = {"new_a": 0.8, "new_b": 0.7}

        engine.run_round_robin(composites)

        assert engine.get_rating("new_a") is not None
        assert engine.get_rating("new_b") is not None


class TestPIIRateEloEngineGetRating:
    """Test get_rating method."""

    def test_get_rating_nonexistent_returns_none(self):
        """Test that nonexistent system returns None."""
        engine = PIIRateEloEngine()

        rating = engine.get_rating("nonexistent")

        assert rating is None

    def test_get_rating_returns_existing(self):
        """Test that existing system is returned."""
        engine = PIIRateEloEngine()
        engine.ensure_system("system_a")

        rating = engine.get_rating("system_a")

        assert rating is not None
        assert rating.system_name == "system_a"


class TestPIIRateEloEngineGetLeaderboard:
    """Test get_leaderboard method."""

    def test_get_leaderboard_empty(self):
        """Test leaderboard with no systems."""
        engine = PIIRateEloEngine()

        leaderboard = engine.get_leaderboard()

        assert len(leaderboard) == 0

    def test_get_leaderboard_single_system(self):
        """Test leaderboard with one system."""
        engine = PIIRateEloEngine()
        engine.ensure_system("system_a")

        leaderboard = engine.get_leaderboard()

        assert len(leaderboard) == 1
        assert leaderboard[0].system_name == "system_a"

    def test_get_leaderboard_sorted_descending(self):
        """Test that leaderboard is sorted by rating descending."""
        engine = PIIRateEloEngine()

        engine.ensure_system("system_a").rating = 1600.0
        engine.ensure_system("system_b").rating = 1500.0
        engine.ensure_system("system_c").rating = 1550.0

        leaderboard = engine.get_leaderboard()

        ratings = [r.rating for r in leaderboard]

        assert ratings == sorted(ratings, reverse=True)

    def test_get_leaderboard_reflects_updates(self):
        """Test that leaderboard reflects match updates."""
        engine = PIIRateEloEngine()

        engine.update_from_match("a", "b", 0.9, 0.1)

        leaderboard = engine.get_leaderboard()

        # System a should be first (higher rating after winning)
        assert leaderboard[0].system_name == "a"


class TestPIIRateEloEngineHistory:
    """Test history tracking."""

    def test_get_history_empty_initially(self):
        """Test that history is empty initially."""
        engine = PIIRateEloEngine()

        history = engine.get_history()

        assert len(history) == 0

    def test_get_history_populated_after_match(self):
        """Test that history is populated after matches."""
        engine = PIIRateEloEngine()

        engine.update_from_match("a", "b", 0.8, 0.7)

        history = engine.get_history()

        assert len(history) == 2

    def test_get_history_returns_copy(self):
        """Test that get_history returns a copy."""
        engine = PIIRateEloEngine()

        engine.update_from_match("a", "b", 0.8, 0.7)

        history1 = engine.get_history()
        history2 = engine.get_history()

        assert history1 == history2
        assert history1 is not history2


class TestPIIRateEloEngineReset:
    """Test reset method."""

    def test_reset_clears_ratings(self):
        """Test that reset clears ratings."""
        engine = PIIRateEloEngine()

        engine.ensure_system("system_a")
        assert len(engine.get_leaderboard()) == 1

        engine.reset()

        assert len(engine.get_leaderboard()) == 0

    def test_reset_clears_history(self):
        """Test that reset clears history."""
        engine = PIIRateEloEngine()

        engine.update_from_match("a", "b", 0.8, 0.7)
        assert len(engine.get_history()) == 2

        engine.reset()

        assert len(engine.get_history()) == 0

    def test_reset_allows_reuse(self):
        """Test that engine can be reused after reset."""
        engine = PIIRateEloEngine()

        engine.update_from_match("a", "b", 0.8, 0.7)
        engine.reset()

        engine.update_from_match("c", "d", 0.9, 0.6)

        assert len(engine.get_leaderboard()) == 2
