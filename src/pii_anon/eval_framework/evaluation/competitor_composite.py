"""Bridge between CompetitorComparisonReport and the composite/Elo system.

Provides convenience functions to:
1. Score every system in a competitor comparison report with the composite metric.
2. Run Elo round-robin ratings from those composite scores.
3. Build a complete leaderboard from a single comparison report.

Usage::

    from pii_anon.evaluation import compare_competitors
    from pii_anon.eval_framework.evaluation.competitor_composite import build_leaderboard

    report = compare_competitors(dataset="pii_anon_benchmark_v1")
    leaderboard = build_leaderboard(report)
    print(leaderboard.to_markdown())
"""

from __future__ import annotations

from typing import Any

from ..metrics.composite import (
    CompositeConfig,
    CompositeScore,
    compute_composite_from_benchmark_result,
)
from ..rating.elo import PIIRateEloEngine
from ..rating.leaderboard import Leaderboard
from ..rating.scorecard import SystemScorecard


def score_comparison_report(
    report: Any,
    config: CompositeConfig | None = None,
) -> dict[str, CompositeScore]:
    """Compute Tier 1 composite score for every system in *report*.

    Parameters
    ----------
    report:
        A ``CompetitorComparisonReport`` (or any object with a
        ``systems`` list of ``SystemBenchmarkResult``-like objects).
    config:
        Composite weighting/normalization config.  If ``None``, uses
        default Tier 1 weights.

    Returns
    -------
    dict[str, CompositeScore]
        Mapping of system name → composite score result.
    """
    scores: dict[str, CompositeScore] = {}
    for system in report.systems:
        if not getattr(system, "available", False):
            continue
        scores[system.system] = compute_composite_from_benchmark_result(
            system, config=config,
        )
    return scores


def rate_comparison_report(
    report: Any,
    config: CompositeConfig | None = None,
    *,
    elo_engine: PIIRateEloEngine | None = None,
) -> PIIRateEloEngine:
    """Compute Elo ratings from a competitor comparison report.

    1. Computes composite scores for all available systems.
    2. Runs a round-robin of pairwise matches.

    Parameters
    ----------
    report:
        A ``CompetitorComparisonReport``.
    config:
        Composite weighting/normalization config.
    elo_engine:
        Optional pre-configured engine.  If ``None``, a fresh engine
        is created with default parameters.

    Returns
    -------
    PIIRateEloEngine
        The engine with updated ratings after round-robin.
    """
    scores = score_comparison_report(report, config=config)
    composites = {name: cs.score for name, cs in scores.items()}

    engine = elo_engine or PIIRateEloEngine()
    engine.run_round_robin(composites)
    return engine


def build_leaderboard(
    report: Any,
    config: CompositeConfig | None = None,
    *,
    benchmark_name: str = "pii-anon-benchmark",
    elo_engine: PIIRateEloEngine | None = None,
) -> Leaderboard:
    """End-to-end: score → rate → leaderboard from a comparison report.

    Parameters
    ----------
    report:
        A ``CompetitorComparisonReport``.
    config:
        Composite weighting/normalization config.
    benchmark_name:
        Name for the leaderboard header.
    elo_engine:
        Optional pre-configured Elo engine.

    Returns
    -------
    Leaderboard
        Sorted by composite score (descending), with Elo ratings attached.
    """
    scores = score_comparison_report(report, config=config)
    composites = {name: cs.score for name, cs in scores.items()}

    engine = elo_engine or PIIRateEloEngine()
    engine.run_round_robin(composites)

    scorecards: list[SystemScorecard] = []
    for system in report.systems:
        name = system.system
        available = getattr(system, "available", False)

        cs = scores.get(name)
        elo = engine.get_rating(name)

        scorecards.append(SystemScorecard(
            system_name=name,
            available=available,
            f1=getattr(system, "f1", 0.0),
            precision=getattr(system, "precision", 0.0),
            recall=getattr(system, "recall", 0.0),
            latency_p50_ms=getattr(system, "latency_p50_ms", 0.0),
            docs_per_hour=getattr(system, "docs_per_hour", 0.0),
            composite_score=cs.score if cs else 0.0,
            elo_rating=elo.rating if elo else 1500.0,
            elo_rd=elo.rd if elo else 350.0,
            samples=getattr(system, "samples", 0),
            evaluation_track=getattr(system, "evaluation_track", "detect_only"),
            license_name=getattr(system, "license_name", None),
        ))

    leaderboard = Leaderboard(
        benchmark_name=benchmark_name,
        systems=scorecards,
    )
    leaderboard.sort_by_composite()
    return leaderboard
