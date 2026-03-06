"""Elo/Glicko-style pairwise rating engine for PII de-identification systems.

Treats each system as a "player" and each benchmark evaluation as a "match".
Composite metric differences are mapped to match outcomes via a sigmoid
function, then standard Elo updates are applied with Glicko-style Rating
Deviation (RD) for uncertainty quantification.

Theory:
    Expected score: E_ij = 1 / (1 + 10^((R_j - R_i) / scale))
    Match outcome:  S_ij = σ(γ · (C_i - C_j))
    Rating update:  R_i' = R_i + K · (S_ij - E_ij)

    where K is adaptive based on RD (higher uncertainty → larger updates).

Evidence basis:
    - Elo (1978) — original rating system
    - Glickman (2001) — Rating Deviation for uncertainty
    - Bradley & Terry (1952) — paired-comparison probability model
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EloRating:
    """Current Elo rating for a single system.

    Attributes
    ----------
    system_name:
        Human-readable identifier for the system.
    rating:
        Current Elo rating (initial default 1500).
    rd:
        Rating Deviation — quantifies uncertainty.  High RD means few
        matches have been played.  Decreases toward *rd_floor* with more
        matches.
    num_matches:
        Number of pairwise matches played.
    """

    system_name: str
    rating: float = 1500.0
    rd: float = 350.0
    num_matches: int = 0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "system_name": self.system_name,
            "rating": round(self.rating, 2),
            "rd": round(self.rd, 2),
            "num_matches": self.num_matches,
        }


@dataclass
class RatingUpdate:
    """Record of a single rating change from one match.

    Stored in the engine history for auditability.
    """

    system_name: str
    old_rating: float
    new_rating: float
    change: float
    expected_score: float
    actual_score: float
    k_factor: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "system_name": self.system_name,
            "old_rating": round(self.old_rating, 2),
            "new_rating": round(self.new_rating, 2),
            "change": round(self.change, 4),
            "expected_score": round(self.expected_score, 6),
            "actual_score": round(self.actual_score, 6),
            "k_factor": round(self.k_factor, 2),
        }


@dataclass
class CalibrationResult:
    """Result of engine auto-calibration.

    Attributes
    ----------
    old_parameters:
        Dictionary of previous parameters.
    new_parameters:
        Dictionary of adjusted parameters.
    score_distribution:
        Statistics about the score distribution used for calibration.
    recommendation:
        Human-readable recommendation string.
    """
    old_parameters: dict[str, float]
    new_parameters: dict[str, float]
    score_distribution: dict[str, float] = dataclass_field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "old_parameters": {k: round(v, 4) for k, v in self.old_parameters.items()},
            "new_parameters": {k: round(v, 4) for k, v in self.new_parameters.items()},
            "score_distribution": {k: round(v, 6) for k, v in self.score_distribution.items()},
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PIIRateEloEngine:
    """Elo/Glicko-style rating engine for PII de-identification systems.

    Parameters
    ----------
    initial_rating:
        Starting Elo rating for new systems (default 1500, chess convention).
    initial_rd:
        Starting Rating Deviation (default 350, high uncertainty).
    k_base:
        Base K-factor for rating updates (default 32).
    scale:
        Elo scale parameter (default 400).  A 100-point gap corresponds
        to approximately 64% expected win probability.
    gamma:
        Sigmoid steepness for composite-to-outcome mapping (default 10).
        Controls how sharply composite score differences map to wins.
    rd_floor:
        Minimum Rating Deviation (default 30).  Prevents RD from
        decreasing below this value.
    """

    def __init__(
        self,
        *,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        k_base: float = 32.0,
        scale: float = 400.0,
        gamma: float = 10.0,
        rd_floor: float = 30.0,
    ) -> None:
        self._initial_rating = initial_rating
        self._initial_rd = initial_rd
        self._k_base = k_base
        self._scale = scale
        self._gamma = gamma
        self._rd_floor = rd_floor

        self._ratings: dict[str, EloRating] = {}
        self._history: list[RatingUpdate] = []

    # -- Internals ----------------------------------------------------------

    def _sigmoid(self, x: float) -> float:
        """Map composite score difference to [0, 1] outcome via logistic sigmoid.

        S_ij = 1 / (1 + exp(-γ · x))

        A positive *x* (system_i has higher composite) yields S > 0.5.
        """
        clamped = max(-20.0, min(20.0, self._gamma * x))
        return 1.0 / (1.0 + math.exp(-clamped))

    def _expected_score(self, rating_i: float, rating_j: float) -> float:
        """Elo expected score: E_ij = 1 / (1 + 10^((R_j - R_i) / scale))."""
        exponent = (rating_j - rating_i) / self._scale
        exponent = max(-10.0, min(10.0, exponent))
        result: float = 1.0 / (1.0 + 10.0 ** exponent)
        return result

    def _adaptive_k(self, rd: float) -> float:
        """Compute adaptive K-factor based on Rating Deviation.

        Higher RD → higher K (more aggressive updates for uncertain ratings).
        K = k_base · (rd / initial_rd), clamped to [k_base/2, k_base·2].
        """
        ratio = rd / self._initial_rd
        k = self._k_base * ratio
        return max(self._k_base / 2.0, min(self._k_base * 2.0, k))

    def _update_rd(self, old_rd: float, num_matches: int) -> float:
        """Decrease RD with more matches, approaching rd_floor.

        rd_new = max(rd_floor, initial_rd / sqrt(1 + num_matches / 5))

        The divisor of 5 means RD halves after ~15 matches.
        """
        if num_matches <= 0:
            return old_rd
        decay = self._initial_rd / math.sqrt(1.0 + num_matches / 5.0)
        return max(self._rd_floor, decay)

    # -- Public API ---------------------------------------------------------

    def ensure_system(self, name: str) -> EloRating:
        """Create a rating entry for *name* if it doesn't exist yet.

        Returns the current (or newly created) rating.
        """
        if name not in self._ratings:
            self._ratings[name] = EloRating(
                system_name=name,
                rating=self._initial_rating,
                rd=self._initial_rd,
            )
        return self._ratings[name]

    def update_from_match(
        self,
        system_i: str,
        system_j: str,
        composite_i: float,
        composite_j: float,
    ) -> tuple[RatingUpdate, RatingUpdate]:
        """Update ratings from a single pairwise match.

        The match outcome is derived from composite score difference
        via sigmoid mapping.

        Returns a tuple of (update_i, update_j).
        """
        ri = self.ensure_system(system_i)
        rj = self.ensure_system(system_j)

        # Match outcome from composite scores
        actual_i = self._sigmoid(composite_i - composite_j)
        actual_j = 1.0 - actual_i

        # Expected scores from current ratings
        expected_i = self._expected_score(ri.rating, rj.rating)
        expected_j = 1.0 - expected_i

        # Adaptive K-factors
        k_i = self._adaptive_k(ri.rd)
        k_j = self._adaptive_k(rj.rd)

        # Rating updates
        change_i = k_i * (actual_i - expected_i)
        change_j = k_j * (actual_j - expected_j)

        old_i = ri.rating
        old_j = rj.rating

        ri.rating += change_i
        rj.rating += change_j
        ri.num_matches += 1
        rj.num_matches += 1
        ri.rd = self._update_rd(ri.rd, ri.num_matches)
        rj.rd = self._update_rd(rj.rd, rj.num_matches)

        update_i = RatingUpdate(
            system_name=system_i,
            old_rating=old_i,
            new_rating=ri.rating,
            change=change_i,
            expected_score=expected_i,
            actual_score=actual_i,
            k_factor=k_i,
        )
        update_j = RatingUpdate(
            system_name=system_j,
            old_rating=old_j,
            new_rating=rj.rating,
            change=change_j,
            expected_score=expected_j,
            actual_score=actual_j,
            k_factor=k_j,
        )

        self._history.append(update_i)
        self._history.append(update_j)

        return update_i, update_j

    def run_round_robin(
        self,
        composites: dict[str, float],
    ) -> list[RatingUpdate]:
        """Run all-pairs round-robin from a composite score dictionary.

        Every pair of systems plays one match.  Updates are applied
        sequentially (order: sorted system names for determinism).

        Parameters
        ----------
        composites:
            Mapping of system name → composite score.

        Returns
        -------
        list[RatingUpdate]
            All rating updates from this round-robin.
        """
        systems = sorted(composites.keys())
        updates: list[RatingUpdate] = []

        for idx_i in range(len(systems)):
            for idx_j in range(idx_i + 1, len(systems)):
                name_i = systems[idx_i]
                name_j = systems[idx_j]
                u_i, u_j = self.update_from_match(
                    name_i, name_j, composites[name_i], composites[name_j],
                )
                updates.extend([u_i, u_j])

        return updates

    def get_rating(self, name: str) -> EloRating | None:
        """Return the current rating for *name*, or ``None`` if unknown."""
        return self._ratings.get(name)

    def get_leaderboard(self) -> list[EloRating]:
        """Return all ratings sorted by descending Elo rating."""
        return sorted(
            self._ratings.values(),
            key=lambda r: r.rating,
            reverse=True,
        )

    def get_history(self) -> list[RatingUpdate]:
        """Return the full update history."""
        return list(self._history)

    def reset(self) -> None:
        """Clear all ratings and history."""
        self._ratings.clear()
        self._history.clear()

    def auto_calibrate(self, composites: dict[str, float]) -> CalibrationResult:
        """Auto-calibrate engine parameters based on score distribution.

        Analyzes the distribution of composite scores to recommend parameter
        adjustments:
        - Adjusts scale based on score spread (standard deviation)
        - Adjusts k_base based on number of systems

        Parameters
        ----------
        composites:
            Mapping of system name → composite score [0, 1].

        Returns
        -------
        CalibrationResult
            Dictionary with old/new parameters and recommendations.
        """
        if not composites:
            return CalibrationResult(
                old_parameters={
                    "scale": self._scale,
                    "k_base": self._k_base,
                    "gamma": self._gamma,
                },
                new_parameters={
                    "scale": self._scale,
                    "k_base": self._k_base,
                    "gamma": self._gamma,
                },
                score_distribution={},
                recommendation="No systems provided; no calibration applied.",
            )

        scores = list(composites.values())
        num_systems = len(scores)

        # Compute distribution statistics
        mean_score = sum(scores) / num_systems if num_systems > 0 else 0.5
        variance = sum((s - mean_score) ** 2 for s in scores) / num_systems if num_systems > 1 else 0.0
        std_score = math.sqrt(variance)
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        # Store old parameters
        old_params = {
            "scale": self._scale,
            "k_base": self._k_base,
            "gamma": self._gamma,
        }

        # Adjust scale based on standard deviation
        # Higher std → larger spread → smaller scale (more sensitive to differences)
        # Lower std → tighter cluster → larger scale (less sensitive)
        target_std = 0.15  # Target standard deviation for well-spread scores
        if std_score > 0.001:
            scale_factor = target_std / std_score
            new_scale = self._scale * scale_factor
        else:
            new_scale = self._scale

        # Adjust k_base based on number of systems
        # Fewer systems → higher k (faster convergence)
        # More systems → lower k (more stable with many comparisons)
        base_k_systems = 4
        k_factor = base_k_systems / max(num_systems, 1)
        new_k_base = self._k_base * k_factor

        # Clamp new parameters to reasonable ranges
        new_scale = max(100.0, min(1000.0, new_scale))
        new_k_base = max(8.0, min(64.0, new_k_base))

        new_params = {
            "scale": new_scale,
            "k_base": new_k_base,
            "gamma": self._gamma,
        }

        # Generate recommendation
        scale_change = new_scale - self._scale
        k_change = new_k_base - self._k_base

        recommendations = []
        if abs(scale_change) > 10:
            if scale_change > 0:
                recommendations.append(
                    f"Increase scale to {new_scale:.1f} (scores are tightly clustered, std={std_score:.3f})"
                )
            else:
                recommendations.append(
                    f"Decrease scale to {new_scale:.1f} (scores are spread out, std={std_score:.3f})"
                )
        if abs(k_change) > 2:
            if k_change > 0:
                recommendations.append(
                    f"Increase k_base to {new_k_base:.1f} (few systems, needs faster convergence)"
                )
            else:
                recommendations.append(
                    f"Decrease k_base to {new_k_base:.1f} (many systems, needs stability)"
                )
        if not recommendations:
            recommendations.append("Current parameters are well-calibrated.")

        recommendation = " ".join(recommendations)

        return CalibrationResult(
            old_parameters=old_params,
            new_parameters=new_params,
            score_distribution={
                "mean": round(mean_score, 6),
                "std": round(std_score, 6),
                "min": round(min_score, 6),
                "max": round(max_score, 6),
                "range": round(score_range, 6),
                "num_systems": num_systems,
            },
            recommendation=recommendation,
        )

    def check_convergence(self, threshold_rd: float | None = None) -> bool:
        """Check if all systems have converged (RD below threshold).

        Parameters
        ----------
        threshold_rd:
            Maximum allowed RD for convergence. If ``None``, defaults to
            2 * rd_floor.

        Returns
        -------
        bool
            True if all systems have RD below threshold.
        """
        if threshold_rd is None:
            threshold_rd = 2.0 * self._rd_floor

        for rating in self._ratings.values():
            if rating.rd > threshold_rd:
                return False
        return True

    def tournament_summary(self) -> dict[str, Any]:
        """Generate comprehensive tournament summary with rankings and statistics.

        Returns
        -------
        dict
            Dictionary with:
            - "rankings": sorted list of system records with confidence intervals
            - "pairwise_significance": significance tests for all pairs
            - "min_distinguishable_diff": smallest meaningful rating difference
            - "converged": whether all systems have converged
        """
        # Get leaderboard
        leaderboard = self.get_leaderboard()

        # Build rankings with confidence intervals
        rankings = []
        for rating in leaderboard:
            # 95% confidence interval: rating ± 1.96 * rd
            ci_lower = rating.rating - 1.96 * rating.rd
            ci_upper = rating.rating + 1.96 * rating.rd
            rankings.append({
                "name": rating.system_name,
                "rating": round(rating.rating, 2),
                "rd": round(rating.rd, 2),
                "ci_lower": round(ci_lower, 2),
                "ci_upper": round(ci_upper, 2),
                "num_matches": rating.num_matches,
            })

        # Compute pairwise significance tests
        pairwise_significance: dict[str, dict[str, Any]] = {}
        for i, rating_i in enumerate(leaderboard):
            for j, rating_j in enumerate(leaderboard):
                if i < j:
                    name_i = rating_i.system_name
                    name_j = rating_j.system_name
                    rating_diff = abs(rating_i.rating - rating_j.rating)

                    # Significance threshold: 2 * sqrt(RD_i^2 + RD_j^2)
                    rd_combined = 2.0 * math.sqrt(rating_i.rd ** 2 + rating_j.rd ** 2)
                    significant = rating_diff > rd_combined

                    key = f"{name_i}_vs_{name_j}"
                    pairwise_significance[key] = {
                        "rating_diff": round(rating_diff, 2),
                        "significance_threshold": round(rd_combined, 2),
                        "significant": significant,
                    }

        # Compute minimum distinguishable difference
        # This is the smallest difference that would be statistically significant
        # given the current RD values
        if leaderboard:
            avg_rd = sum(r.rd for r in leaderboard) / len(leaderboard)
            min_distinguishable_diff = 2.0 * math.sqrt(2.0 * avg_rd ** 2)
        else:
            min_distinguishable_diff = 0.0

        # Check convergence
        converged = self.check_convergence()

        return {
            "rankings": rankings,
            "pairwise_significance": pairwise_significance,
            "min_distinguishable_diff": round(min_distinguishable_diff, 2),
            "converged": converged,
            "num_systems": len(leaderboard),
            "total_matches": len(self._history) // 2,
        }

    def evaluate_governance(
        self,
        system_name: str,
        *,
        thresholds: GovernanceThresholds | None = None,
    ) -> GovernanceResult:
        """Evaluate whether a system meets governance thresholds.

        Implements Section 4.7 of the PII-Rate-Elo paper: a production-grade
        gate requiring ``R > min_rating`` with ``RD < max_rd`` as
        indicators that a system is both high-performing and
        well-evaluated.

        Parameters
        ----------
        system_name:
            The system to evaluate.
        thresholds:
            Governance thresholds to apply.  If ``None``, defaults are
            used (R > 1500, RD < 100).

        Returns
        -------
        GovernanceResult
            Detailed pass/fail result with per-criterion breakdown.
        """
        thresh = thresholds or GovernanceThresholds()
        rating = self._ratings.get(system_name)

        if rating is None:
            return GovernanceResult(
                system_name=system_name,
                passed=False,
                rating=0.0,
                rd=self._initial_rd,
                min_rating_met=False,
                max_rd_met=False,
                min_matches_met=False,
                notes=["System not found in rating engine."],
            )

        rating_ok = rating.rating >= thresh.min_rating
        rd_ok = rating.rd <= thresh.max_rd
        matches_ok = rating.num_matches >= thresh.min_matches
        passed = rating_ok and rd_ok and matches_ok

        notes: list[str] = []
        if not rating_ok:
            notes.append(
                f"Rating {rating.rating:.1f} < threshold {thresh.min_rating}"
            )
        if not rd_ok:
            notes.append(
                f"RD {rating.rd:.1f} > max allowed {thresh.max_rd}"
            )
        if not matches_ok:
            notes.append(
                f"Matches {rating.num_matches} < minimum {thresh.min_matches}"
            )
        if passed:
            notes.append("System meets all governance thresholds.")

        return GovernanceResult(
            system_name=system_name,
            passed=passed,
            rating=rating.rating,
            rd=rating.rd,
            min_rating_met=rating_ok,
            max_rd_met=rd_ok,
            min_matches_met=matches_ok,
            notes=notes,
        )

    def evaluate_all_governance(
        self,
        *,
        thresholds: GovernanceThresholds | None = None,
    ) -> list[GovernanceResult]:
        """Evaluate governance thresholds for all registered systems.

        Returns results sorted by rating (descending).
        """
        results = [
            self.evaluate_governance(name, thresholds=thresholds)
            for name in self._ratings
        ]
        results.sort(key=lambda r: r.rating, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Governance thresholds (Section 4.7)
# ---------------------------------------------------------------------------

@dataclass
class GovernanceThresholds:
    """Configurable governance thresholds for production-grade deployment.

    Per Section 4.7 of the PII-Rate-Elo paper, systems must exceed a minimum
    Elo rating with sufficiently low Rating Deviation before being
    considered production-grade.

    Attributes
    ----------
    min_rating:
        Minimum Elo rating for production-grade gate (default 1500).
    max_rd:
        Maximum Rating Deviation for confidence in ranking (default 100).
    min_matches:
        Minimum number of pairwise matches for rating stability
        (default 6 — two full round-robins with 4 systems).
    """

    min_rating: float = 1500.0
    max_rd: float = 100.0
    min_matches: int = 6

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_rating": self.min_rating,
            "max_rd": self.max_rd,
            "min_matches": self.min_matches,
        }


@dataclass
class GovernanceResult:
    """Result of governance threshold evaluation for a single system.

    Attributes
    ----------
    system_name:
        Identifier of the evaluated system.
    passed:
        ``True`` if all thresholds are met.
    rating:
        Current Elo rating.
    rd:
        Current Rating Deviation.
    min_rating_met:
        Whether the minimum rating threshold was met.
    max_rd_met:
        Whether the RD is below the maximum.
    min_matches_met:
        Whether sufficient matches have been played.
    notes:
        Human-readable explanation of the result.
    """

    system_name: str
    passed: bool
    rating: float
    rd: float
    min_rating_met: bool
    max_rd_met: bool
    min_matches_met: bool
    notes: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "passed": self.passed,
            "rating": round(self.rating, 2),
            "rd": round(self.rd, 2),
            "min_rating_met": self.min_rating_met,
            "max_rd_met": self.max_rd_met,
            "min_matches_met": self.min_matches_met,
            "notes": self.notes,
        }
