"""MCMC convergence diagnostics gate — the hard NFR-001 teeth (S3-03).

PURE NUMPY. This module imports **nothing** beyond numpy, so it is fully
testable in any environment (the dev ``.venv`` has numpy but not
numpyro/jax/arviz). It is the load-bearing NFR-001 artifact: it decides whether
a fitted posterior is *claim-grade* and **fails loud** when it is not.

Claim-grade gate (NFR-001 literals — resolves SME CATASTROPHIC eval-01)::

    split-R̂ ≤ 1.01   (all parameters)
    bulk-ESS ≥ 400    (per parameter)
    divergences == 0

Only the Bayesian ``bayes-bt`` tier runs through this gate; the frequentist
tiers (``glicko-legacy``, ``bradley-terry-mle``) satisfy NFR-001 only
*by-substitution* and are therefore smoke/fallback only. When a fit misses the
bar, :func:`assert_claim_grade` raises :class:`ConvergenceError` naming the
*binding constraint* — the single next thing to fix — so the program always
knows the gap (the SDO philosophy).

Diagnostic definitions
----------------------
* :func:`split_rhat` — the **split** potential-scale-reduction factor R̂. Each
  chain is split in half (doubling the chain count) so that *within-chain*
  non-stationarity (a slowly drifting chain) inflates the between-chain term and
  is caught. This is the split Gelman-Rubin statistic.

  Evidence basis:
      - Gelman & Rubin (1992) "Inference from iterative simulation using
        multiple sequences", Statist. Sci. 7(4):457-472.
      - Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021) "Rank-normalization,
        folding, and localization: An improved R̂ for assessing convergence of
        MCMC", Bayesian Analysis 16(2):667-718.

* :func:`bulk_ess` — the bulk **effective sample size** via the
  autocorrelation/variogram estimator with Geyer's initial-monotone-sequence
  truncation (the estimator used by Stan / ArviZ). For iid draws ESS approaches
  ``n_chains · n_draws``; for a highly autocorrelated chain it collapses toward
  a handful of effective draws.

  Evidence basis:
      - Geyer (1992) "Practical Markov chain Monte Carlo", Statist. Sci.
        7(4):473-483 (initial monotone sequence estimator).
      - Vehtari et al. (2021), §3 (multi-chain ESS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RHAT_MAX",
    "ESS_MIN_PER_PARAM",
    "MAX_DIVERGENCES",
    "ConvergenceError",
    "ConvergenceReport",
    "split_rhat",
    "bulk_ess",
    "count_divergences",
    "assert_claim_grade",
]

# -- NFR-001 thresholds: these literals ARE the claim-grade contract. --------
RHAT_MAX: float = 1.01
ESS_MIN_PER_PARAM: int = 400
MAX_DIVERGENCES: int = 0


class ConvergenceError(Exception):
    """Raised by :func:`assert_claim_grade` when a fit is NOT claim-grade.

    The message names the *binding constraint* (which diagnostic failed and by
    how much). Catch this specifically at the leaderboard-emission boundary (S4
    CanonicalRunGate) to refuse claim-grade emission and fail loud.
    """


def _as_chains_draws_params(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    """Coerce input to a float64 ``(n_chains, n_draws, n_params)`` array.

    A 2-D ``(n_chains, n_draws)`` input is treated as a single parameter.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(
            "samples must be shaped (n_chains, n_draws[, n_params]); "
            f"got ndim={arr.ndim}"
        )
    return arr


def _split_chains(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Split each chain in half → ``(2·n_chains, n_draws//2, n_params)``.

    The split is what makes R̂ sensitive to *within-chain* drift: a chain whose
    mean wanders between its first and second half produces two half-chains with
    different means, inflating the between-chain variance.
    """
    n_chains, n_draws, n_params = arr.shape
    half = n_draws // 2
    if half < 1:
        # Too few draws to split; fall back to the unsplit chains.
        return arr
    first = arr[:, :half, :]
    second = arr[:, half : 2 * half, :]
    return np.concatenate([first, second], axis=0)


def split_rhat(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    """Per-parameter **split** R̂ (split Gelman-Rubin potential scale reduction).

    Parameters
    ----------
    samples:
        Array shaped ``(n_chains, n_draws, n_params)`` (a 2-D
        ``(n_chains, n_draws)`` input is treated as one parameter).

    Returns
    -------
    ``(n_params,)`` array of R̂ values. R̂ → 1 as chains mix; R̂ > 1 signals the
    chains have not converged to a common distribution. NFR-001 requires
    ``R̂ ≤ 1.01`` for every parameter.

    Notes
    -----
    With ``m`` half-chains of length ``n`` and per-half means ``x̄_j`` /
    variances ``s²_j``::

        B = n/(m-1) · Σ_j (x̄_j − x̄)²            (between-chain variance)
        W = (1/m) · Σ_j s²_j                       (within-chain variance)
        var⁺ = (n-1)/n · W + B/n                    (marginal posterior var est.)
        R̂ = sqrt(var⁺ / W)
    """
    arr = _as_chains_draws_params(samples)
    split = _split_chains(arr)
    m, n, n_params = split.shape

    if m < 2 or n < 2:
        # Cannot estimate between/within variance — report 1.0 (no evidence of
        # non-convergence rather than a spurious failure).
        return np.ones(n_params, dtype=np.float64)

    chain_means = split.mean(axis=1)  # (m, n_params)
    chain_vars = split.var(axis=1, ddof=1)  # (m, n_params)
    grand_mean = chain_means.mean(axis=0)  # (n_params,)

    between = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    within = chain_means.shape[0] and chain_vars.mean(axis=0)  # (n_params,)

    var_plus = (n - 1) / n * within + between / n
    # Guard a degenerate within-chain variance of exactly 0 (constant chains):
    # if both within and the marginal estimate are ~0 the chains are identical
    # constants → R̂ = 1; if only within is 0 but means differ → +inf (correctly
    # non-convergent).
    rhat = np.empty(n_params, dtype=np.float64)
    for p in range(n_params):
        w = within[p]
        vp = var_plus[p]
        if w <= 0.0:
            rhat[p] = 1.0 if vp <= 0.0 else np.inf
        else:
            rhat[p] = float(np.sqrt(vp / w))
    return rhat


def _autocov(chain: NDArray[np.float64], n_lags: int) -> NDArray[np.float64]:
    """Biased autocovariance of a 1-D ``chain`` for lags ``0..n_lags-1``.

    Uses the FFT for O(n log n); the ``1/n`` (biased) normalization is the one
    the Geyer/Vehtari ESS estimator assumes.
    """
    n = chain.shape[0]
    centered = chain - chain.mean()
    # Zero-pad to >= 2n for linear (non-circular) correlation via FFT.
    fft_len = int(2 ** np.ceil(np.log2(2 * n)))
    freq = np.fft.rfft(centered, n=fft_len)
    acov_full = np.fft.irfft(freq * np.conjugate(freq), n=fft_len)[:n]
    acov_full /= n
    return acov_full[:n_lags]


def bulk_ess(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    """Per-parameter bulk effective sample size (autocorrelation estimator).

    Parameters
    ----------
    samples:
        Array shaped ``(n_chains, n_draws, n_params)``.

    Returns
    -------
    ``(n_params,)`` array of effective sample sizes. For iid draws
    ESS ≈ ``n_chains · n_draws``; under strong positive autocorrelation it
    collapses. NFR-001 requires ``ESS ≥ 400`` per parameter.

    Notes
    -----
    Combined multi-chain autocorrelation per Vehtari et al. (2021)::

        ρ_t = 1 − (W − (1/m) Σ_c acov_c(t)) / var⁺

    summed with Geyer's initial-monotone-sequence truncation::

        τ = 1 + 2 Σ_{k≥1} (ρ_{2k} + ρ_{2k+1})   (truncate when a pair sum < 0,
                                                  then enforce monotone decrease)
        ESS = (m · n) / τ
    """
    arr = _as_chains_draws_params(samples)
    m, n, n_params = arr.shape
    total = m * n

    if m < 1 or n < 2:
        return np.full(n_params, float(total), dtype=np.float64)

    ess = np.empty(n_params, dtype=np.float64)
    for p in range(n_params):
        chains = arr[:, :, p]  # (m, n)
        chain_means = chains.mean(axis=1)
        chain_vars = chains.var(axis=1, ddof=1)
        within = float(chain_vars.mean())

        if within <= 0.0:
            # All chains constant → no autocorrelation structure; treat as the
            # full sample size (degenerate but not a failure of mixing).
            ess[p] = float(total)
            continue

        between = n / (m - 1) * float(np.sum((chain_means - chain_means.mean()) ** 2)) \
            if m > 1 else 0.0
        var_plus = (n - 1) / n * within + between / n
        if var_plus <= 0.0:
            ess[p] = float(total)
            continue

        # Mean (over chains) biased autocovariance at each lag.
        n_lags = n
        acov_sum = np.zeros(n_lags, dtype=np.float64)
        for c in range(m):
            acov_sum += _autocov(chains[c], n_lags)
        acov_mean = acov_sum / m  # (n_lags,)

        # ρ_t per Vehtari 2021 (acov_mean[0] == within-style variance estimate).
        rho = 1.0 - (within - acov_mean) / var_plus
        rho[0] = 1.0

        # Geyer initial monotone sequence: sum consecutive pairs, truncate at the
        # first non-positive pair, then enforce a monotone-decreasing envelope.
        tau = 1.0
        max_t = n_lags - 1
        t = 1
        prev_pair = np.inf
        while t + 1 <= max_t:
            pair = rho[t] + rho[t + 1]
            if pair < 0.0:
                break
            pair = min(pair, prev_pair)  # monotone envelope
            tau += 2.0 * pair
            prev_pair = pair
            t += 2
        # A single trailing lag (odd n) contributes ρ_t alone if still positive.
        if t == max_t and rho[t] > 0.0:
            tau += 2.0 * min(rho[t], prev_pair if np.isfinite(prev_pair) else rho[t])

        tau = max(tau, 1.0)  # ESS can never exceed the raw sample count.
        ess[p] = total / tau
    return ess


def count_divergences(
    divergences: NDArray[np.generic] | int | bool | None,
) -> int:
    """Total NUTS divergences as a single non-negative integer.

    Accepts the NUTS ``diverging`` extra field (a boolean/int array over draws),
    a pre-summed count, or ``None`` (→ 0). NFR-001 requires this to be 0 for a
    claim-grade fit (a divergence means the sampler hit a region of high
    curvature it could not integrate — the posterior is untrustworthy there).
    """
    if divergences is None:
        return 0
    if isinstance(divergences, (int, np.integer)):
        return int(divergences)
    if isinstance(divergences, (bool, np.bool_)):
        return int(bool(divergences))
    return int(np.sum(np.asarray(divergences) != 0))


@dataclass(frozen=True)
class ConvergenceReport:
    """Verdict of the NFR-001 convergence gate over one fitted posterior.

    Attributes
    ----------
    max_rhat:
        Worst (largest) split-R̂ across all parameters.
    min_bulk_ess:
        Worst (smallest) bulk-ESS across all parameters.
    n_divergences:
        Total NUTS divergences.
    n_params:
        Number of parameters diagnosed.
    claim_grade:
        ``True`` iff ALL of: ``max_rhat ≤ RHAT_MAX`` ∧
        ``min_bulk_ess ≥ ESS_MIN_PER_PARAM`` ∧ ``n_divergences ≤ MAX_DIVERGENCES``.
    binding_constraint:
        Human-readable name of the single worst-violating check (the next thing
        to fix), or ``""`` when claim-grade. Divergences are reported first
        (an absolute veto), then R̂, then ESS.
    """

    max_rhat: float
    min_bulk_ess: float
    n_divergences: int
    n_params: int
    claim_grade: bool
    binding_constraint: str

    @classmethod
    def from_samples(
        cls,
        samples: NDArray[np.floating],
        *,
        n_divergences: NDArray[np.generic] | int | None = 0,
    ) -> ConvergenceReport:
        """Compute the gate verdict from posterior ``samples``.

        ``samples`` is shaped ``(n_chains, n_draws, n_params)``; ``n_divergences``
        is the NUTS divergence count (array or pre-summed int).
        """
        rhat = split_rhat(samples)
        ess = bulk_ess(samples)
        divergences = count_divergences(n_divergences)

        max_rhat = float(np.max(rhat)) if rhat.size else 1.0
        min_ess = float(np.min(ess)) if ess.size else 0.0
        n_params = int(rhat.size)

        binding = _binding_constraint(max_rhat, min_ess, divergences)
        return cls(
            max_rhat=max_rhat,
            min_bulk_ess=min_ess,
            n_divergences=divergences,
            n_params=n_params,
            claim_grade=(binding == ""),
            binding_constraint=binding,
        )


def _binding_constraint(max_rhat: float, min_ess: float, n_div: int) -> str:
    """Name the single binding NFR-001 violation, or ``""`` if claim-grade.

    Priority (worst first): divergences (absolute veto) → R̂ → ESS. This gives
    the program one unambiguous next-thing-to-fix per the SDO philosophy.
    """
    if n_div > MAX_DIVERGENCES:
        return (
            f"divergences={n_div} > MAX_DIVERGENCES={MAX_DIVERGENCES} "
            f"(NUTS hit non-integrable curvature; posterior untrustworthy)"
        )
    if max_rhat > RHAT_MAX:
        return (
            f"max_rhat={max_rhat:.4g} > RHAT_MAX={RHAT_MAX} "
            f"(chains have not mixed to a common distribution)"
        )
    if min_ess < ESS_MIN_PER_PARAM:
        return (
            f"min_bulk_ess={min_ess:.4g} < ESS_MIN_PER_PARAM={ESS_MIN_PER_PARAM} "
            f"(too few effective draws; estimates are autocorrelation-starved)"
        )
    return ""


def assert_claim_grade(report: ConvergenceReport) -> None:
    """Raise :class:`ConvergenceError` unless ``report`` is claim-grade.

    The exception message is the report's ``binding_constraint`` — the single
    next thing to fix. This is the NFR-001 *fail-loud* boundary: a non-converged
    fit can never be silently treated as claim-grade.
    """
    if not report.claim_grade:
        raise ConvergenceError(
            "posterior is NOT claim-grade (NFR-001): " + report.binding_constraint
        )
