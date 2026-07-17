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

from .bayes_bt import (
    BayesBTEngine,
    MissingOptionalDependencyError,
    PairedCountsWithTies,
    Posterior,
)
from .bradley_terry import BradleyTerryConvergenceWarning, BradleyTerryMLEEngine
from .convergence import ConvergenceError, ConvergenceReport
from .paired_set import PairedComparisonSet, assemble_paired_set
from .significance import (
    PairwiseVerdict,
    davidson_p_i_beats_j,
    davidson_p_j_beats_i,
    davidson_p_tie,
    pairwise_significance,
    rank_one_distribution,
    rank_one_probability,
)
from .elo import (
    PIIRateEloEngine,
    EloRating,
    GovernanceResult,
    GovernanceThresholds,
    RatingUpdate,
)
from .leaderboard import Leaderboard, LeaderboardExporter
from .port import RatingEnginePort
from .registry import RatingEngineRegistry
from .scorecard import BenchmarkScorecard, SystemScorecard

__all__ = [
    "PIIRateEloEngine",
    "BradleyTerryMLEEngine",
    "BradleyTerryConvergenceWarning",
    "BayesBTEngine",
    "Posterior",
    "PairedCountsWithTies",
    "MissingOptionalDependencyError",
    "ConvergenceReport",
    "ConvergenceError",
    "PairwiseVerdict",
    "pairwise_significance",
    "rank_one_probability",
    "rank_one_distribution",
    "davidson_p_i_beats_j",
    "davidson_p_tie",
    "davidson_p_j_beats_i",
    "PairedComparisonSet",
    "assemble_paired_set",
    "EloRating",
    "GovernanceResult",
    "GovernanceThresholds",
    "RatingUpdate",
    "RatingEnginePort",
    "RatingEngineRegistry",
    "SystemScorecard",
    "BenchmarkScorecard",
    "Leaderboard",
    "LeaderboardExporter",
]
