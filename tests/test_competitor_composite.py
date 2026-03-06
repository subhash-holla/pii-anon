"""Tests for the competitor-composite integration bridge.

Tests cover scoring, Elo rating, and end-to-end leaderboard generation
from mock CompetitorComparisonReport data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pii_anon.eval_framework.evaluation.competitor_composite import (
    build_leaderboard,
    rate_comparison_report,
    score_comparison_report,
)
from pii_anon.eval_framework.metrics.composite import CompositeConfig


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

@dataclass
class MockSystemBenchmarkResult:
    """Minimal mock mirroring SystemBenchmarkResult fields."""

    system: str
    available: bool = True
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    latency_p50_ms: float = 100.0
    docs_per_hour: float = 500_000.0
    samples: int = 100
    evaluation_track: str = "detect_only"
    license_name: str | None = "MIT"
    skipped_reason: str | None = None


@dataclass
class MockCompetitorComparisonReport:
    """Minimal mock mirroring CompetitorComparisonReport."""

    systems: list[MockSystemBenchmarkResult] = field(default_factory=list)


def _make_mock_report() -> MockCompetitorComparisonReport:
    return MockCompetitorComparisonReport(systems=[
        MockSystemBenchmarkResult(
            system="pii-anon",
            f1=0.85,
            precision=0.90,
            recall=0.80,
            latency_p50_ms=0.01,
            docs_per_hour=168_000_000.0,
        ),
        MockSystemBenchmarkResult(
            system="presidio",
            f1=0.70,
            precision=0.75,
            recall=0.65,
            latency_p50_ms=10.0,
            docs_per_hour=300_000.0,
        ),
        MockSystemBenchmarkResult(
            system="scrubadub",
            f1=0.45,
            precision=0.50,
            recall=0.40,
            latency_p50_ms=15.0,
            docs_per_hour=200_000.0,
        ),
        MockSystemBenchmarkResult(
            system="gliner",
            f1=0.55,
            precision=0.60,
            recall=0.50,
            latency_p50_ms=5.0,
            docs_per_hour=400_000.0,
        ),
        MockSystemBenchmarkResult(
            system="unavailable_system",
            available=False,
            skipped_reason="not installed",
        ),
    ])


# ---------------------------------------------------------------------------
# score_comparison_report
# ---------------------------------------------------------------------------

class TestScoreComparisonReport:
    def test_available_systems_scored(self):
        report = _make_mock_report()
        scores = score_comparison_report(report)
        assert "pii-anon" in scores
        assert "presidio" in scores
        assert "scrubadub" in scores
        assert "gliner" in scores

    def test_unavailable_systems_excluded(self):
        report = _make_mock_report()
        scores = score_comparison_report(report)
        assert "unavailable_system" not in scores

    def test_scores_in_range(self):
        report = _make_mock_report()
        scores = score_comparison_report(report)
        for name, cs in scores.items():
            assert 0.0 <= cs.score <= 1.0, f"{name} score {cs.score} out of range"

    def test_pii_anon_highest_composite(self):
        """pii-anon should have the highest composite given its superior metrics."""
        report = _make_mock_report()
        scores = score_comparison_report(report)
        pii_anon_score = scores["pii-anon"].score
        for name, cs in scores.items():
            if name != "pii-anon":
                assert pii_anon_score >= cs.score, (
                    f"pii-anon ({pii_anon_score}) should beat {name} ({cs.score})"
                )

    def test_custom_config(self):
        report = _make_mock_report()
        cfg = CompositeConfig(
            weight_detection_f1=0.90,
            weight_detection_precision=0.05,
            weight_detection_recall=0.05,
            weight_latency=0.0,
            weight_throughput=0.0,
        )
        scores = score_comparison_report(report, config=cfg)
        assert "pii-anon" in scores


# ---------------------------------------------------------------------------
# rate_comparison_report
# ---------------------------------------------------------------------------

class TestRateComparisonReport:
    def test_engine_returned(self):
        report = _make_mock_report()
        engine = rate_comparison_report(report)
        assert engine is not None

    def test_all_systems_rated(self):
        report = _make_mock_report()
        engine = rate_comparison_report(report)
        for name in ["pii-anon", "presidio", "scrubadub", "gliner"]:
            assert engine.get_rating(name) is not None

    def test_best_system_highest_elo(self):
        report = _make_mock_report()
        engine = rate_comparison_report(report)
        lb = engine.get_leaderboard()
        assert lb[0].system_name == "pii-anon"

    def test_worst_system_lowest_elo(self):
        report = _make_mock_report()
        engine = rate_comparison_report(report)
        lb = engine.get_leaderboard()
        assert lb[-1].system_name == "scrubadub"

    def test_history_populated(self):
        report = _make_mock_report()
        engine = rate_comparison_report(report)
        # 4 available systems → 6 matches → 12 updates
        assert len(engine.get_history()) == 12


# ---------------------------------------------------------------------------
# build_leaderboard
# ---------------------------------------------------------------------------

class TestBuildLeaderboard:
    def test_leaderboard_returned(self):
        report = _make_mock_report()
        lb = build_leaderboard(report)
        assert lb is not None
        assert lb.benchmark_name == "pii-anon-benchmark"

    def test_all_systems_in_leaderboard(self):
        """All systems (including unavailable) should appear in leaderboard."""
        report = _make_mock_report()
        lb = build_leaderboard(report)
        names = {sc.system_name for sc in lb.systems}
        assert "pii-anon" in names
        assert "unavailable_system" in names

    def test_sorted_by_composite(self):
        report = _make_mock_report()
        lb = build_leaderboard(report)
        # Available systems should be sorted by composite (descending)
        available = [sc for sc in lb.systems if sc.available]
        for i in range(len(available) - 1):
            assert available[i].composite_score >= available[i + 1].composite_score

    def test_elo_ratings_attached(self):
        report = _make_mock_report()
        lb = build_leaderboard(report)
        pii_anon = next(sc for sc in lb.systems if sc.system_name == "pii-anon")
        assert pii_anon.elo_rating != 1500.0  # Should have been updated

    def test_custom_benchmark_name(self):
        report = _make_mock_report()
        lb = build_leaderboard(report, benchmark_name="custom-bench")
        assert lb.benchmark_name == "custom-bench"

    def test_markdown_export(self):
        report = _make_mock_report()
        lb = build_leaderboard(report)
        md = lb.to_markdown()
        assert "pii-anon" in md
        assert "Leaderboard" in md

    def test_json_export(self):
        import json
        report = _make_mock_report()
        lb = build_leaderboard(report)
        j = lb.to_json()
        data = json.loads(j)
        assert "systems" in data

    def test_unavailable_system_default_scores(self):
        report = _make_mock_report()
        lb = build_leaderboard(report)
        unavail = next(sc for sc in lb.systems if sc.system_name == "unavailable_system")
        assert not unavail.available
        assert unavail.composite_score == 0.0
        assert unavail.elo_rating == 1500.0  # default, never played
