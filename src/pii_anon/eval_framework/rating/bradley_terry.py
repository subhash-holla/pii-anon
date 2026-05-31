"""Bradley-Terry MLE rating engine — SECOND ladder tier (S3-02, FR-003).

Fits Bradley-Terry (1952) strengths by **minorization-maximization** (Hunter
2004) using ONLY the Python standard library (``math`` + ``random.Random``),
and quantifies uncertainty by **paired (record-resampling) bootstrap**. The
engine satisfies the structural :class:`RatingEnginePort` with ZERO call-site
changes and is registered as the ``bradley-terry-mle`` entry point in group
``pii_anon.rating_engines``. This is the fast PR-CI / smoke tier of the
3-tier ladder (``glicko-legacy`` → ``bradley-terry-mle`` → ``bayes-bt``); the
claim-grade default (``bayes-bt``, NumPyro NUTS) follows in S3-03.

Theory
------
Each system *i* has strength ``π_i = exp(θ_i)``. For paired comparisons the
BT win probability is ``P(i beats j) = π_i / (π_i + π_j)``. The MLE maximizes
the paired-comparison likelihood; Hunter's MM surrogate gives the closed-form,
guaranteed-ascent update::

    π_i ← W_i / Σ_{j≠i} n_ij / (π_i + π_j)

where ``W_i`` is *i*'s total (possibly fractional) wins and ``n_ij`` the number
of comparisons between *i* and *j*. After each sweep the strengths are
renormalized to geometric-mean 1 (⇔ ``Σ_i θ_i = 0``), anchoring identifiability
("sum-to-zero"). Iteration stops at ``max|Δ log π| < tol``.

Strengths are reported on the **Elo scale** ``rating = 1500 + 400·θ_i/ln(10)``
so deltas read like Elo points (``scale=400`` matches
:mod:`pii_anon.eval_framework.rating.elo`).

Port-compat path
----------------
:meth:`run_round_robin` takes ``name -> composite score``. Point composites are
a *perfect total order*, for which the raw BT MLE diverges (strengths → ±∞). We
map composite gaps to **fractional** pairwise outcomes via
``s_ij = sigmoid(γ·(C_i − C_j))`` (the same logistic shape as
:meth:`elo.PIIRateEloEngine._sigmoid`, ``γ=10``), giving a complete soft design
on which the MM fit is finite and well-conditioned. A small symmetric smoothing
prior guarantees a finite MLE even for a separable design.

Claim-grade path (NOT part of the port)
---------------------------------------
:meth:`fit_paired` accepts record-level paired *counts*
``{(sys_i, sys_j): (wins_i, wins_j, n)}`` and fits MM directly on integers (real
designs are not perfectly separable, so no soft mapping is needed).
:meth:`paired_bootstrap` resamples *records* with replacement, refits, and
returns deterministic percentile CIs (the frequentist foundation of NFR-002/003
significance coherence). Ties are out of scope here (plain BT) — the Davidson
tie term is S3-04.

# SWITCH-POINT(S6): the canonical BT primitive is ``pii-anon-eval-data``
# ``stats/bradley_terry.py`` (verified ABSENT today; DATA repo lands it at S6).
# When it lands, replace the local ``_fit_mm`` body with a thin adapter over the
# frozen primitive — this port keeps both call-sites stable.

Evidence basis:
    - Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs"
    - Hunter (2004) "MM algorithms for generalized Bradley-Terry models"
"""

from __future__ import annotations

import math
import random

from .elo import EloRating, RatingUpdate

__all__ = ["BradleyTerryMLEEngine"]

# Elo-scale reporting constants — kept in lock-step with ``elo.py``.
_ELO_ANCHOR = 1500.0
_ELO_SCALE = 400.0
_LN10 = math.log(10.0)

# Comparison-shape aliases.
PairedCounts = dict[tuple[str, str], tuple[float, float, int]]
Record = tuple[str, str]


class BradleyTerryMLEEngine:
    """Bradley-Terry MLE engine (MM / Hunter 2004) — pure stdlib.

    Satisfies :class:`~pii_anon.eval_framework.rating.port.RatingEnginePort`
    structurally (``run_round_robin`` + ``get_rating``); the richer
    ``fit_paired`` / ``paired_bootstrap`` API is engine-only and intentionally
    NOT part of the port (so the ladder tiers stay swappable).

    Parameters
    ----------
    gamma:
        Logistic steepness for the composite-to-outcome map in the port path
        (default 10, matching ``elo.PIIRateEloEngine``).
    scale:
        Elo scale for rating reporting (default 400, matching ``elo``).
    initial_rating:
        Anchor rating for ``θ = 0`` (default 1500).
    smoothing:
        Symmetric Beta-style pseudo-count added to BOTH directions of every
        modelled pair (default 1e-3). Guarantees a finite MLE on a separable
        design (standard BT regularization) while barely perturbing ordering.
    tol:
        Convergence threshold on ``max|Δ log π|`` (default 1e-9).
    max_iter:
        Maximum MM sweeps (default 1000).
    """

    def __init__(
        self,
        *,
        gamma: float = 10.0,
        scale: float = 400.0,
        initial_rating: float = 1500.0,
        smoothing: float = 1e-3,
        tol: float = 1e-9,
        max_iter: int = 1000,
    ) -> None:
        self._gamma = gamma
        self._scale = scale
        self._initial_rating = initial_rating
        self._smoothing = smoothing
        self._tol = tol
        self._max_iter = max_iter

        self._ratings: dict[str, EloRating] = {}
        self._history: list[RatingUpdate] = []

    # -- Internals ----------------------------------------------------------

    def _sigmoid(self, x: float) -> float:
        """Logistic map of a composite gap to a [0, 1] soft outcome.

        Mirrors :meth:`elo.PIIRateEloEngine._sigmoid` (``σ(γ·x)``) so the BT
        port path is a faithful drop-in for the Elo engine's outcome model.
        """
        clamped = max(-20.0, min(20.0, self._gamma * x))
        return 1.0 / (1.0 + math.exp(-clamped))

    def _to_rating(self, theta: float) -> float:
        """Map a sum-to-zero log-strength to the Elo scale."""
        return _ELO_ANCHOR + self._scale * theta / _LN10

    def _fit_mm(
        self,
        names: list[str],
        wins: dict[str, float],
        games: dict[tuple[str, str], float],
    ) -> dict[str, float]:
        """Run the MM iteration to convergence; return sum-to-zero θ.

        ``wins[i]`` is W_i (total fractional wins of *i*); ``games`` is the
        SYMMETRIC comparison count, i.e. both ``(i, j)`` and ``(j, i)`` map to
        ``n_ij``. Strengths are renormalized to geometric-mean 1 after every
        sweep (sum-to-zero in θ). Systems that played no comparison stay at the
        anchor (θ = 0).
        """
        pi = {name: 1.0 for name in names}
        for _ in range(self._max_iter):
            updated: dict[str, float] = {}
            for name in names:
                denom = 0.0
                for other in names:
                    if other == name:
                        continue
                    n_ij = games.get((name, other), 0.0)
                    if n_ij:
                        denom += n_ij / (pi[name] + pi[other])
                if denom > 0.0 and wins[name] > 0.0:
                    updated[name] = wins[name] / denom
                else:
                    # No wins or no comparisons: leave strength unmoved so an
                    # unplayed system anchors at θ = 0 after renormalization.
                    updated[name] = pi[name]

            # Renormalize to geometric mean 1  ⇔  Σ θ = 0.
            log_mean = sum(math.log(v) for v in updated.values()) / len(updated)
            geo_mean = math.exp(log_mean)
            for name in names:
                updated[name] /= geo_mean

            delta = max(
                abs(math.log(updated[name]) - math.log(pi[name])) for name in names
            )
            pi = updated
            if delta < self._tol:
                break

        return {name: math.log(value) for name, value in pi.items()}

    def _store_ratings(
        self, theta: dict[str, float], num_matches: dict[str, int]
    ) -> None:
        """Materialize Elo-scale :class:`EloRating` records from fitted θ."""
        for name, value in theta.items():
            self._ratings[name] = EloRating(
                system_name=name,
                rating=self._to_rating(value),
                rd=self._ratings.get(name, EloRating(name)).rd,
                num_matches=num_matches.get(name, 0),
            )

    # -- Public API: structural port (FR-003) -------------------------------

    def run_round_robin(self, composites: dict[str, float]) -> list[RatingUpdate]:
        """Run an all-pairs round-robin from ``name -> composite score``.

        Composite gaps are mapped to fractional pairwise outcomes
        ``s_ij = sigmoid(γ·(C_i − C_j))`` over the complete design, the MM fit
        produces sum-to-zero log-strengths, and ratings are reported on the Elo
        scale. One :class:`RatingUpdate` is emitted per system per match
        (old = prior rating, new = fitted rating) for audit parity with the
        Elo engine's history contract.

        Systems are iterated in **sorted name order** for determinism (matching
        :meth:`elo.PIIRateEloEngine.run_round_robin`).
        """
        systems = sorted(composites.keys())
        wins: dict[str, float] = {name: 0.0 for name in systems}
        games: dict[tuple[str, str], float] = {}
        # Record the prior (pre-fit) ratings for the audit history.
        old_ratings = {
            name: (self._ratings[name].rating if name in self._ratings else self._initial_rating)
            for name in systems
        }
        num_matches: dict[str, int] = {name: 0 for name in systems}

        for idx_i in range(len(systems)):
            for idx_j in range(idx_i + 1, len(systems)):
                name_i = systems[idx_i]
                name_j = systems[idx_j]
                s_ij = self._sigmoid(composites[name_i] - composites[name_j])
                # Symmetric smoothing → finite MLE even for a separable order.
                wins[name_i] += s_ij + self._smoothing
                wins[name_j] += (1.0 - s_ij) + self._smoothing
                n_ij = 1.0 + 2.0 * self._smoothing
                games[(name_i, name_j)] = n_ij
                games[(name_j, name_i)] = n_ij
                num_matches[name_i] += 1
                num_matches[name_j] += 1

        theta = self._fit_mm(systems, wins, games) if systems else {}
        self._store_ratings(theta, num_matches)

        updates: list[RatingUpdate] = []
        for idx_i in range(len(systems)):
            for idx_j in range(idx_i + 1, len(systems)):
                name_i = systems[idx_i]
                name_j = systems[idx_j]
                s_ij = self._sigmoid(composites[name_i] - composites[name_j])
                updates.append(
                    self._make_update(name_i, old_ratings[name_i], s_ij)
                )
                updates.append(
                    self._make_update(name_j, old_ratings[name_j], 1.0 - s_ij)
                )

        self._history.extend(updates)
        return updates

    def _make_update(
        self, name: str, old_rating: float, actual_score: float
    ) -> RatingUpdate:
        """Build a per-match :class:`RatingUpdate` against the fitted rating."""
        new_rating = self._ratings[name].rating
        return RatingUpdate(
            system_name=name,
            old_rating=old_rating,
            new_rating=new_rating,
            change=new_rating - old_rating,
            expected_score=0.5,
            actual_score=actual_score,
            k_factor=0.0,
        )

    def get_rating(self, name: str) -> EloRating | None:
        """Return the current rating for ``name``, or ``None`` if unknown."""
        return self._ratings.get(name)

    # -- Public API: claim-grade paired fit (engine-only) -------------------

    def fit_paired(self, comparisons: PairedCounts) -> dict[str, float]:
        """Fit BT strengths from record-level paired *counts*.

        ``comparisons`` maps an ordered key ``(sys_i, sys_j)`` to
        ``(wins_i, wins_j, n)``: *i* won ``wins_i`` of ``n`` comparisons, *j*
        won ``wins_j``. The MM fit runs directly on the integer counts (no soft
        mapping — real paired designs are not perfectly separable). Returns
        sum-to-zero log-strengths θ and stores the corresponding Elo-scale
        ratings (so :meth:`get_rating` works afterwards).

        # DAVIDSON(S3-04): ties are out of scope here (plain BT). The Davidson
        # tie term (and the one-joint-posterior coherence) lands in S3-04; this
        # is the seam where a ``ties_ij`` count and the ``ν`` tie parameter wire
        # in without disturbing the win/total bookkeeping below.
        """
        names: list[str] = []
        seen: set[str] = set()
        wins: dict[str, float] = {}
        games: dict[tuple[str, str], float] = {}

        for (sys_i, sys_j), (wins_i, wins_j, n) in comparisons.items():
            for name in (sys_i, sys_j):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
                    wins[name] = 0.0
            wins[sys_i] += wins_i
            wins[sys_j] += wins_j
            games[(sys_i, sys_j)] = games.get((sys_i, sys_j), 0.0) + float(n)
            games[(sys_j, sys_i)] = games.get((sys_j, sys_i), 0.0) + float(n)

        names.sort()  # deterministic iteration order
        theta = self._fit_mm(names, wins, games) if names else {}

        num_matches: dict[str, int] = {name: 0 for name in names}
        for (sys_i, sys_j), (_wi, _wj, n) in comparisons.items():
            num_matches[sys_i] += n
            num_matches[sys_j] += n
        self._store_ratings(theta, num_matches)
        return theta

    def fit_records(self, records: list[Record]) -> dict[str, float]:
        """Fit BT strengths from a list of ``(winner, loser)`` records.

        Each record is one paired comparison the winner won. Returns sum-to-zero
        log-strengths θ. This is the unit :meth:`paired_bootstrap` resamples.
        """
        names: list[str] = []
        seen: set[str] = set()
        wins: dict[str, float] = {}
        games: dict[tuple[str, str], float] = {}

        for winner, loser in records:
            for name in (winner, loser):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
                    wins[name] = 0.0
            wins[winner] += 1.0
            games[(winner, loser)] = games.get((winner, loser), 0.0) + 1.0
            games[(loser, winner)] = games.get((loser, winner), 0.0) + 1.0

        names.sort()
        return self._fit_mm(names, wins, games) if names else {}

    def paired_bootstrap(
        self, records: list[Record], b: int, *, seed: int
    ) -> dict[str, tuple[float, float]]:
        """Percentile 95% CIs on θ via record-resampling bootstrap.

        Resamples ``records`` with replacement ``b`` times using
        ``random.Random(seed)``, refits BT each time, and returns per-system
        ``(ci_lower, ci_upper)`` 2.5/97.5 percentile intervals on the
        log-strength scale. **Deterministic** for a fixed ``seed`` (pure
        ``random.Random`` draws; sorted system iteration). This is the
        frequentist foundation of NFR-002/003 significance coherence: a point
        estimate always lies inside its CI, and a strict CI separation can never
        contradict the point ordering.
        """
        rng = random.Random(seed)
        names = sorted({name for record in records for name in record})
        samples: dict[str, list[float]] = {name: [] for name in names}
        n_rec = len(records)

        for _ in range(b):
            resampled = [records[rng.randrange(n_rec)] for _ in range(n_rec)]
            theta = self.fit_records(resampled)
            for name in names:
                # A system absent from a resample anchors at θ = 0.
                samples[name].append(theta.get(name, 0.0))

        cis: dict[str, tuple[float, float]] = {}
        for name in names:
            ordered = sorted(samples[name])
            cis[name] = (
                _percentile(ordered, 0.025),
                _percentile(ordered, 0.975),
            )
        return cis


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile of a pre-sorted list (deterministic).

    ``q`` in [0, 1]. Uses linear interpolation between the two nearest ranks so
    the point estimate reliably lies within the resulting interval.
    """
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac
