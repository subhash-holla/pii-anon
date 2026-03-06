"""PII-Rate-Elo rating engine for PII de-identification system comparison.

Provides Elo/Glicko-style pairwise ratings, scorecards, and leaderboards
for comparing PII detection and anonymization systems on a single composite
metric.

Evidence basis:
    - Elo (1978) "The Rating of Chessplayers, Past and Present"
    - Glickman (2001) "Parameter estimation in large dynamic paired comparison experiments"
    - Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs"

See ``docs/composite-metric-evidence.md`` for full research backing.
"""

from __future__ import annotations

from .elo import (
    PIIRateEloEngine,
    EloRating,
    GovernanceResult,
    GovernanceThresholds,
    RatingUpdate,
)
from .leaderboard import Leaderboard, LeaderboardExporter
from .scorecard import BenchmarkScorecard, SystemScorecard

__all__ = [
    "PIIRateEloEngine",
    "EloRating",
    "GovernanceResult",
    "GovernanceThresholds",
    "RatingUpdate",
    "SystemScorecard",
    "BenchmarkScorecard",
    "Leaderboard",
    "LeaderboardExporter",
]
