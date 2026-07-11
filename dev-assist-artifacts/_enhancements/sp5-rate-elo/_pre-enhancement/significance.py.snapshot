"""Coherent pairwise significance BY CONSTRUCTION + rank-1 probability (S3-04).

PURE NUMPY. This module imports **nothing** beyond numpy + the stdlib, so it is
fully testable in any environment (the dev ``.venv`` has numpy but not
numpyro/jax/arviz) and is import-isolated from the detection / orchestration /
``evaluation`` layers (S3-05 boundary). It takes a *joint posterior over
per-system strengths* — the ``(n_draws, n_systems)`` ``theta_samples`` array S3-03
:class:`~pii_anon.eval_framework.rating.bayes_bt.Posterior` produces — and reads
every significance statement off that ONE sample.

Why "by construction" eliminates the three verified ``elo.py`` defects
---------------------------------------------------------------------
``elo.py`` computed three significance-adjacent quantities by three *independent*
routes that can disagree:

* ``elo.py:243`` — the match "outcome" is fabricated from a composite gap by a
  ``σ(γ·Δ)`` sigmoid (an outcome that never happened);
* ``elo.py:542`` — the reported CI is a normal-approx ``rating ± 1.96·rd``
  (decoupled from any data; ``rd`` is a match-count heuristic, not a posterior
  SD);
* ``elo.py:561`` — "significant" is a *third* computation,
  ``|rating_i − rating_j| > 2·√(rd_i² + rd_j²)``, which can disagree with both.

Here all three derive from the SAME difference vector::

    d = θ_samples[:, col_i] − θ_samples[:, col_j]        # one vector, one source
    point       = mean(d)
    (ci_lo, ci_hi) = percentile(d, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)             # CI excludes 0
    p_i_beats_j = mean(d > 0)

Because ``point`` is the mean of ``d`` and ``(ci_lo, ci_hi)`` are percentiles of
the SAME ``d``, ``point`` is *necessarily* inside ``[ci_lo, ci_hi]``; and
``significant`` is *defined* as the interval excluding 0, so it cannot contradict
the sign of ``point`` or of ``p_i_beats_j − 0.5``. The three statements are one
object viewed three ways — that single-source derivation is the NFR-002
by-construction teeth. (``elo.py`` stays untouched as the legacy/glicko tier;
this is the claim-grade replacement path consumed by S4.)

rank-1 probability = the SDO J-meter
------------------------------------
:func:`rank_one_probability` returns ``P(rank(name) = 1 | joint posterior)`` — the
fraction of joint draws in which ``name`` has the maximum θ. Over all systems it
is a proper distribution (sums to 1). This is literally the ``J`` the SDO
``CompetitiveSupremacyGate`` optimizes, computed coherently from the one joint
sample rather than from decoupled marginals.

Davidson (1970) tie closed form
--------------------------------
The pure-numpy Davidson three-way outcome probabilities (used as in-tree teeth
for the NumPyro Davidson likelihood in :mod:`bayes_bt`). With ``δ = θ_i − θ_j``
and tie parameter ``ν ≥ 0`` the unnormalized terms are
``[exp(δ/2), ν, exp(−δ/2)]`` (a common ``exp((θ_i+θ_j)/2)`` factors out), i.e. the
softmax of logits ``[δ/2, ln ν, −δ/2]`` gives ``(P(i>j), P(tie), P(j>i))``. At
``ν = 0`` this nests plain Bradley-Terry (``P(i>j) = σ(δ)``, ``P(tie) = 0``).

Evidence basis:
    - Davidson (1970) "On Extending the Bradley-Terry Model to Accommodate Ties
      in Paired Comparison Experiments", JASA 65(329):317-328.
    - Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PairwiseVerdict",
    "pairwise_significance",
    "rank_one_distribution",
    "rank_one_probability",
    "davidson_p_i_beats_j",
    "davidson_p_tie",
    "davidson_p_j_beats_i",
]

# Credible-interval mass: the central 95% interval (2.5 / 97.5 percentiles).
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5


@dataclass(frozen=True)
class PairwiseVerdict:
    """One coherent pairwise significance verdict, read off ONE joint posterior.

    All five quantities below derive from the SAME difference vector
    ``d = θ_i − θ_j`` over the joint draws, so they cannot disagree (NFR-002 by
    construction).

    Attributes
    ----------
    i, j:
        The two system names (``i`` is the first, ``j`` the second of a sorted
        pair). ``point``/``ci``/``p_i_beats_j`` are oriented as *i relative to j*.
    point:
        ``mean(θ_i − θ_j)`` — the posterior-mean strength gap (Elo-free log scale).
        Positive ⇒ ``i`` is stronger.
    ci_lo, ci_hi:
        The central 95% credible interval of ``θ_i − θ_j`` (2.5 / 97.5
        percentiles). ``point`` is necessarily within ``[ci_lo, ci_hi]``.
    p_i_beats_j:
        ``mean(θ_i > θ_j)`` over the joint draws — the posterior probability that
        ``i`` is the stronger system.
    significant:
        ``True`` iff the credible interval excludes 0 (``ci_lo > 0`` or
        ``ci_hi < 0``). Defined ONLY from the CI, so it cannot contradict
        ``point`` or ``p_i_beats_j``.
    """

    i: str
    j: str
    point: float
    ci_lo: float
    ci_hi: float
    p_i_beats_j: float
    significant: bool


def _validate_samples(
    theta_samples: NDArray[np.floating[Any]], names: list[str]
) -> NDArray[np.float64]:
    """Coerce ``theta_samples`` to ``(n_draws, n_systems)`` float64 and check shape.

    The orientation contract (bayes_bt.py:140) is ``(n_draws, n_systems)`` with
    columns in the order of ``names`` (sorted system list). A mismatch between
    the column count and ``len(names)`` — e.g. a transposed array — raises
    loudly, so an orientation bug can never pass silently.
    """
    arr = np.asarray(theta_samples, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "theta_samples must be 2-D (n_draws, n_systems); "
            f"got ndim={arr.ndim}"
        )
    n_draws, n_systems = arr.shape
    if n_systems != len(names):
        raise ValueError(
            "theta_samples column count must equal len(names) "
            f"(orientation is (n_draws, n_systems)); got {n_systems} columns "
            f"for {len(names)} names — a transposed array would trip this"
        )
    if n_draws < 1:
        raise ValueError("theta_samples must have at least one draw")
    return arr


def pairwise_significance(
    theta_samples: NDArray[np.floating[Any]], names: list[str]
) -> list[PairwiseVerdict]:
    """Coherent pairwise significance for every pair, off ONE joint posterior.

    Parameters
    ----------
    theta_samples:
        ``(n_draws, n_systems)`` joint posterior over per-system log-strengths
        (e.g. :attr:`Posterior.theta_samples`). Column ``k`` is ``names[k]``.
    names:
        System names in column order (the sorted system list).

    Returns
    -------
    One :class:`PairwiseVerdict` per unordered pair ``(names[a], names[b])`` with
    ``a < b`` — ``C(n_systems, 2)`` verdicts in sorted pair order. Each verdict's
    ``point``, ``ci``, ``p_i_beats_j`` and ``significant`` are read off the SAME
    difference vector, so they are coherent by construction (NFR-002). Pure
    function of the samples → deterministic (AX-002).
    """
    arr = _validate_samples(theta_samples, names)
    n_systems = arr.shape[1]
    verdicts: list[PairwiseVerdict] = []
    for a in range(n_systems):
        for b in range(a + 1, n_systems):
            d = arr[:, a] - arr[:, b]  # the ONE vector all four read off
            lo, hi = np.percentile(d, [_CI_LOW_PCT, _CI_HIGH_PCT])
            lo_f = float(lo)
            hi_f = float(hi)
            verdicts.append(
                PairwiseVerdict(
                    i=names[a],
                    j=names[b],
                    point=float(np.mean(d)),
                    ci_lo=lo_f,
                    ci_hi=hi_f,
                    p_i_beats_j=float(np.mean(d > 0.0)),
                    significant=(lo_f > 0.0) or (hi_f < 0.0),
                )
            )
    return verdicts


def rank_one_distribution(
    theta_samples: NDArray[np.floating[Any]], names: list[str]
) -> dict[str, float]:
    """Distribution over "who is #1": ``P(rank(name) = 1 | joint posterior)``.

    For each joint draw, the system with the maximum θ scores a "win"; the
    returned fractions are the per-system win shares and sum to 1.0 (every draw
    has exactly one argmax). This is the proper distribution the SDO J-meter
    reads :func:`rank_one_probability` from. Pure-numpy, deterministic.
    """
    arr = _validate_samples(theta_samples, names)
    winners = np.argmax(arr, axis=1)  # (n_draws,) winning column per draw
    counts = np.bincount(winners, minlength=len(names))
    total = float(arr.shape[0])
    return {name: float(counts[k]) / total for k, name in enumerate(names)}


def rank_one_probability(
    theta_samples: NDArray[np.floating[Any]], names: list[str], name: str
) -> float:
    """``P(rank(name) = 1 | joint posterior)`` — the SDO J for one system.

    The fraction of joint draws in which ``name`` has the maximum θ. Raises
    :class:`KeyError` for an unknown ``name`` (no silent 0.0 — a typo in the
    J-meter call must fail loud). Pure-numpy, deterministic.
    """
    if name not in names:
        raise KeyError(
            f"{name!r} is not among the posterior systems {names!r}; "
            "cannot compute its rank-1 probability"
        )
    return rank_one_distribution(theta_samples, names)[name]


# ---------------------------------------------------------------------------
# Davidson (1970) tie closed form — pure-numpy teeth for the NumPyro likelihood.
# ---------------------------------------------------------------------------


def _davidson_probs(delta: float, nu: float) -> tuple[float, float, float]:
    """``(P(i>j), P(tie), P(j>i))`` from δ = θ_i − θ_j and tie parameter ν ≥ 0.

    Unnormalized terms ``[exp(δ/2), ν, exp(−δ/2)]`` (the common
    ``exp((θ_i+θ_j)/2)`` factor cancels), normalized — equivalently the softmax
    of logits ``[δ/2, ln ν, −δ/2]``. ``ν = 0`` nests plain Bradley-Terry. Computed
    in a numerically stable, normalization-by-max way.
    """
    if nu < 0.0:
        raise ValueError(f"tie parameter nu must be >= 0; got {nu}")
    half = delta / 2.0
    # Stable softmax over the three unnormalized log-terms. For ν = 0 the tie
    # term is exp(-inf) = 0, which correctly nests plain BT.
    log_nu = math.log(nu) if nu > 0.0 else -math.inf
    logits = (half, log_nu, -half)
    m = max(logits)
    exps = [math.exp(x - m) if x != -math.inf else 0.0 for x in logits]
    denom = sum(exps)
    p_i, p_tie, p_j = (e / denom for e in exps)
    return p_i, p_tie, p_j


def davidson_p_i_beats_j(delta: float, nu: float) -> float:
    """Davidson ``P(i > j)`` for strength gap ``δ = θ_i − θ_j`` and tie param ``ν``."""
    return _davidson_probs(delta, nu)[0]


def davidson_p_tie(delta: float, nu: float) -> float:
    """Davidson ``P(tie)`` for strength gap ``δ = θ_i − θ_j`` and tie param ``ν``."""
    return _davidson_probs(delta, nu)[1]


def davidson_p_j_beats_i(delta: float, nu: float) -> float:
    """Davidson ``P(j > i)`` for strength gap ``δ = θ_i − θ_j`` and tie param ``ν``."""
    return _davidson_probs(delta, nu)[2]
