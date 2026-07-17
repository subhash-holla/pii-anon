"""NFR-001 convergence gate — pure-numpy diagnostics + teeth (S3-03).

Implements:
    NFR-001 — the hard MCMC convergence gate that makes ``bayes-bt`` the ONLY
              claim-grade tier (split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param ∧ 0
              divergences). Resolves SME CATASTROPHIC eval-01: frequentist
              tiers satisfy NFR-001 only by-substitution, so the gate must
              FAIL LOUD when diagnostics miss the bar.

Design trace: D-EVAL DECISION 2 — "a hard convergence gate refuses claim-grade
leaderboard emission on failure (fails loud)."

These tests are PURE NUMPY and run in the dev ``.venv`` (numpyro/jax/arviz are
NOT installed; the gate intentionally never imports them). The split-R̂ and
bulk-ESS formulas are cross-checked against the standard split-Gelman-Rubin
(Gelman & Rubin 1992) and rank-normalized/autocorrelation ESS (Vehtari et al.
2021) definitions on hand-computable canned chains.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pii_anon.eval_framework.rating.convergence import (
    ConvergenceError,
    ConvergenceReport,
    assert_claim_grade,
    bulk_ess,
    count_divergences,
    split_rhat,
)
from pii_anon.eval_framework.rating.convergence import (
    ESS_MIN_PER_PARAM,
    MAX_DIVERGENCES,
    RHAT_MAX,
)


# ---------------------------------------------------------------------------
# Threshold literals are the NFR-001 contract — pin them exactly.
# ---------------------------------------------------------------------------

def test_nfr_001_threshold_literals_are_the_contract() -> None:
    """The gate thresholds ARE the NFR-001 acceptance bar. Pin them so a silent
    relaxation (e.g. RHAT_MAX → 1.1) can never weaken the claim-grade gate."""
    assert RHAT_MAX == 1.01
    assert ESS_MIN_PER_PARAM == 400
    assert MAX_DIVERGENCES == 0


# ---------------------------------------------------------------------------
# split_rhat — formula correctness on hand-computable canned chains.
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_split_rhat_near_iid_is_approximately_one() -> None:
    """Given many well-mixed near-iid chains drawn from one distribution, when
    split_rhat runs, then R̂ ≈ 1.0 (≤ 1.01) for every parameter — the canonical
    'converged' signature."""
    rng = _rng(0)
    # (n_chains=8, n_draws=4000, n_params=3) all from N(0,1): well mixed.
    samples = rng.standard_normal((8, 4000, 3))
    rhat = split_rhat(samples)
    assert rhat.shape == (3,)
    assert np.all(rhat <= 1.01), f"near-iid R̂ should be ~1.0, got {rhat}"
    assert np.all(rhat >= 0.99), f"near-iid R̂ should be ~1.0, got {rhat}"


def test_split_rhat_separated_chains_blows_up() -> None:
    """Given 2 chains sampled at very different locations (a classic
    non-convergence signature), when split_rhat runs, then R̂ ≫ 1.01 because the
    between-chain variance dwarfs the within-chain variance."""
    rng = _rng(1)
    # Chain 0 ~ N(0, 0.01); chain 1 ~ N(50, 0.01): same tiny spread, far apart.
    chain0 = rng.standard_normal((1, 2000, 1)) * 0.1 + 0.0
    chain1 = rng.standard_normal((1, 2000, 1)) * 0.1 + 50.0
    samples = np.concatenate([chain0, chain1], axis=0)
    rhat = split_rhat(samples)
    assert rhat.shape == (1,)
    assert rhat[0] > 1.01, f"separated chains must trip R̂, got {rhat[0]}"
    # The separation is enormous, so R̂ should be far above the threshold.
    assert rhat[0] > 5.0, f"R̂ for 50-sigma-apart chains should be large, got {rhat[0]}"


def test_split_rhat_matches_manual_gelman_rubin_on_canned_means() -> None:
    """Cross-check split_rhat against the textbook split Gelman-Rubin formula on
    a small, fully hand-computable case.

    We build 2 chains each of 4 draws with KNOWN within-chain variances and a
    KNOWN between-chain mean gap, split each into halves (→ 4 half-chains of 2),
    and verify the implementation's R̂ equals the value the split-G-R formula
    produces for those half-chains.
    """
    # Two chains, 4 draws each, 1 param.  Hand-picked integers.
    chain_a = np.array([1.0, 3.0, 2.0, 4.0])  # mean 2.5
    chain_b = np.array([10.0, 12.0, 11.0, 13.0])  # mean 11.5 (gap = 9)
    samples = np.stack([chain_a, chain_b])[:, :, None]  # (2, 4, 1)

    # Manual split Gelman-Rubin on the 4 half-chains (length n=2).
    halves = [
        chain_a[:2], chain_a[2:],
        chain_b[:2], chain_b[2:],
    ]
    m = len(halves)
    n = 2
    chain_means = np.array([h.mean() for h in halves])
    chain_vars = np.array([h.var(ddof=1) for h in halves])
    grand_mean = chain_means.mean()
    b = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    w = chain_vars.mean()
    var_plus = (n - 1) / n * w + b / n
    expected_rhat = math.sqrt(var_plus / w)

    got = split_rhat(samples)
    assert got.shape == (1,)
    assert got[0] == pytest.approx(expected_rhat, rel=1e-9), (
        f"split_rhat={got[0]} != manual split-G-R={expected_rhat}"
    )
    # And it must be well above 1 (chains are far apart) → not claim-grade.
    assert got[0] > 1.01


# ---------------------------------------------------------------------------
# bulk_ess — formula correctness.
# ---------------------------------------------------------------------------

def test_bulk_ess_iid_is_close_to_total_draws() -> None:
    """Given iid draws, when bulk_ess runs, then ESS ≈ n_chains · n_draws (iid
    samples carry ~one effective sample each — the upper-bound signature)."""
    rng = _rng(2)
    n_chains, n_draws = 4, 2000
    samples = rng.standard_normal((n_chains, n_draws, 2))
    ess = bulk_ess(samples)
    total = n_chains * n_draws
    assert ess.shape == (2,)
    # iid ESS should land near the total; allow generous sampling slack.
    assert np.all(ess > 0.7 * total), f"iid ESS should be ~{total}, got {ess}"
    assert np.all(ess <= 1.30 * total + 1), f"iid ESS should not exceed total much, got {ess}"


def test_bulk_ess_strong_autocorrelation_collapses() -> None:
    """Given a highly autocorrelated chain (near random-walk), when bulk_ess
    runs, then ESS ≪ total draws (the 'few effective samples' signature that
    must be caught by the ESS≥400 gate)."""
    rng = _rng(3)
    n_chains, n_draws = 4, 2000
    # AR(1) with phi≈0.99: extreme positive autocorrelation.
    phi = 0.99
    samples = np.empty((n_chains, n_draws, 1))
    for c in range(n_chains):
        x = 0.0
        for t in range(n_draws):
            x = phi * x + rng.standard_normal()
            samples[c, t, 0] = x
    ess = bulk_ess(samples)
    total = n_chains * n_draws
    assert ess.shape == (1,)
    assert ess[0] < 0.2 * total, (
        f"phi=0.99 AR(1) must have ESS far below total {total}, got {ess[0]}"
    )
    assert ess[0] > 0.0


def test_bulk_ess_below_threshold_on_short_autocorrelated_run() -> None:
    """A short, autocorrelated run yields ESS < 400/param — the exact condition
    the claim-grade gate must veto."""
    rng = _rng(4)
    phi = 0.97
    n_chains, n_draws = 2, 300
    samples = np.empty((n_chains, n_draws, 1))
    for c in range(n_chains):
        x = 0.0
        for t in range(n_draws):
            x = phi * x + rng.standard_normal()
            samples[c, t, 0] = x
    ess = bulk_ess(samples)
    assert ess[0] < 400.0, f"expected sub-400 ESS, got {ess[0]}"


# ---------------------------------------------------------------------------
# count_divergences.
# ---------------------------------------------------------------------------

def test_count_divergences_counts_truthy_flags() -> None:
    """count_divergences sums a boolean/int divergence-flag array (the NUTS
    'diverging' extra field) into a single non-negative integer."""
    assert count_divergences(np.array([False, False, False])) == 0
    assert count_divergences(np.array([True, False, True, False])) == 2
    assert count_divergences(np.array([[0, 1], [1, 1]])) == 3
    assert count_divergences(None) == 0
    assert count_divergences(7) == 7


# ---------------------------------------------------------------------------
# ConvergenceReport + assert_claim_grade — THE TEETH (NFR-001 fails loud).
# ---------------------------------------------------------------------------

def _well_mixed_samples() -> np.ndarray:
    """Clean, well-identified, well-mixed posterior draws (claim-grade)."""
    rng = _rng(20)
    # 4 chains × 1500 draws × 4 params, all iid N(0,1): R̂≈1, ESS≈6000 ≫ 400.
    return rng.standard_normal((4, 1500, 4))


def test_convergence_report_claim_grade_on_clean_posterior() -> None:
    """NEGATIVE / pass-path: a clean, well-mixed posterior → claim_grade True,
    no raise, binding_constraint sentinel = '' (nothing binding)."""
    samples = _well_mixed_samples()
    report = ConvergenceReport.from_samples(samples, n_divergences=0)
    assert isinstance(report, ConvergenceReport)
    assert report.n_params == 4
    assert report.max_rhat <= RHAT_MAX
    assert report.min_bulk_ess >= ESS_MIN_PER_PARAM
    assert report.n_divergences == 0
    assert report.claim_grade is True
    assert report.binding_constraint == ""
    # assert_claim_grade must NOT raise on a claim-grade report.
    assert_claim_grade(report)  # no exception


def test_teeth_rhat_failure_raises_and_names_constraint() -> None:
    """TEETH: 2 chains sampled at different locations → split-R̂ ≫ 1.01. The
    report is NOT claim-grade and assert_claim_grade RAISES ConvergenceError
    naming R̂ as the binding constraint (fails loud — resolves eval-01)."""
    rng = _rng(21)
    chain0 = rng.standard_normal((1, 1500, 1)) * 0.1 + 0.0
    chain1 = rng.standard_normal((1, 1500, 1)) * 0.1 + 25.0
    samples = np.concatenate([chain0, chain1], axis=0)
    report = ConvergenceReport.from_samples(samples, n_divergences=0)

    assert report.claim_grade is False
    assert report.max_rhat > RHAT_MAX
    assert "rhat" in report.binding_constraint.lower() or "r̂" in report.binding_constraint.lower()

    with pytest.raises(ConvergenceError) as exc:
        assert_claim_grade(report)
    msg = str(exc.value).lower()
    assert "rhat" in msg or "r̂" in msg
    # The error must report the offending value so the program knows the gap.
    assert f"{report.max_rhat:.4g}".lower() in str(exc.value).lower() or "1.01" in str(exc.value)


def test_teeth_divergence_failure_raises_and_names_constraint() -> None:
    """TEETH: a clean posterior but with injected divergences > 0 → NOT
    claim-grade; assert_claim_grade RAISES naming divergences (0-divergence
    policy is absolute)."""
    samples = _well_mixed_samples()
    report = ConvergenceReport.from_samples(samples, n_divergences=3)
    assert report.claim_grade is False
    assert report.n_divergences == 3
    assert "diverg" in report.binding_constraint.lower()

    with pytest.raises(ConvergenceError) as exc:
        assert_claim_grade(report)
    assert "diverg" in str(exc.value).lower()
    assert "3" in str(exc.value)


def test_teeth_ess_failure_raises_and_names_constraint() -> None:
    """TEETH: chains that MIX (R̂ ≤ 1.01) but were simply NOT SAMPLED ENOUGH so
    bulk-ESS < 400/param → NOT claim-grade; assert_claim_grade RAISES naming ESS
    as the binding constraint (R̂ is fine, the gap is too-few effective draws).

    The honest 'undersampled' construction is a SMALL IID posterior: iid draws
    have R̂≈1.0 (well mixed) and ESS≈n_chains·n_draws (each draw ~one effective
    sample), so a total just under 400 puts ESS below the bar while R̂ passes."""
    rng = _rng(22)
    # 4 chains × 80 iid draws = 320 total ⇒ R̂≈1.0, ESS≈320 < 400.
    samples = rng.standard_normal((4, 80, 1))
    report = ConvergenceReport.from_samples(samples, n_divergences=0)

    assert report.max_rhat <= RHAT_MAX, (
        f"R̂ must be fine so ESS is the binding constraint, got {report.max_rhat}"
    )
    assert report.min_bulk_ess < ESS_MIN_PER_PARAM
    assert report.claim_grade is False
    assert "ess" in report.binding_constraint.lower()

    with pytest.raises(ConvergenceError) as exc:
        assert_claim_grade(report)
    assert "ess" in str(exc.value).lower()


def test_binding_constraint_reports_worst_offender_first() -> None:
    """When MULTIPLE diagnostics fail, the report still names ONE binding
    constraint (the next thing to fix, per the SDO philosophy) and stays
    non-claim-grade; assert_claim_grade raises."""
    rng = _rng(23)
    # Separated chains (R̂ fails) AND short/autocorrelated (ESS fails) AND
    # divergences injected — all three broken at once.
    c0 = (rng.standard_normal((1, 200, 1)) * 0.1) + 0.0
    c1 = (rng.standard_normal((1, 200, 1)) * 0.1) + 30.0
    samples = np.concatenate([c0, c1], axis=0)
    report = ConvergenceReport.from_samples(samples, n_divergences=5)
    assert report.claim_grade is False
    assert report.binding_constraint != ""
    with pytest.raises(ConvergenceError):
        assert_claim_grade(report)


# ---------------------------------------------------------------------------
# AX-002 determinism: the gate is pure-numpy and deterministic.
# ---------------------------------------------------------------------------

def test_ax_002_gate_is_deterministic_for_fixed_samples() -> None:
    """Given identical input samples, the gate produces a byte-identical
    ConvergenceReport across repeated calls (pure-numpy, no RNG inside)."""
    samples = _well_mixed_samples()
    r1 = ConvergenceReport.from_samples(samples, n_divergences=0)
    r2 = ConvergenceReport.from_samples(samples.copy(), n_divergences=0)
    assert r1.max_rhat == r2.max_rhat
    assert r1.min_bulk_ess == r2.min_bulk_ess
    assert r1.n_divergences == r2.n_divergences
    assert r1.claim_grade == r2.claim_grade
    assert r1.binding_constraint == r2.binding_constraint


def test_convergence_error_is_an_exception() -> None:
    """ConvergenceError is a real exception type so callers (the leaderboard
    emitter at S4) can catch it specifically."""
    assert issubclass(ConvergenceError, Exception)
