"""Tests for the reproducible statistics layer."""

from __future__ import annotations

from pii_anon.assurance.stats import (
    Counts,
    cluster_bootstrap_ci,
    holm_bonferroni,
    paired_bootstrap,
    pooled_f1,
    prf1_from_counts,
)


def test_prf1_known_values() -> None:
    p, r, f1 = prf1_from_counts(tp=8, fp=2, fn=2)
    assert p == 0.8
    assert r == 0.8
    assert abs(f1 - 0.8) < 1e-9


def test_prf1_empty_is_zero_not_one() -> None:
    assert prf1_from_counts(0, 0, 0) == (0.0, 0.0, 0.0)


def test_pooled_f1_pools_counts() -> None:
    recs = [Counts(4, 1, 1), Counts(4, 1, 1)]  # pooled tp=8 fp=2 fn=2
    assert abs(pooled_f1(recs) - 0.8) < 1e-9


def test_cluster_bootstrap_is_deterministic() -> None:
    recs = [Counts(tp=3, fp=1, fn=1) for _ in range(40)]
    a = cluster_bootstrap_ci(recs, pooled_f1, seed=123, n_resamples=200)
    b = cluster_bootstrap_ci(recs, pooled_f1, seed=123, n_resamples=200)
    assert a == b  # byte-identical given the seed


def test_cluster_bootstrap_point_and_ci_bracket() -> None:
    recs = [Counts(tp=3, fp=1, fn=1) for _ in range(60)]
    point, ci = cluster_bootstrap_ci(recs, pooled_f1, seed=7, n_resamples=300)
    assert ci is not None
    lo, hi = ci
    assert lo <= point <= hi


def test_cluster_bootstrap_too_few_records_no_ci() -> None:
    point, ci = cluster_bootstrap_ci([Counts(1, 0, 0)], pooled_f1, seed=1)
    assert ci is None  # fail closed -> power gate demotes


def test_paired_bootstrap_identical_systems_delta_zero_high_p() -> None:
    recs = [Counts(tp=3, fp=1, fn=1) for _ in range(50)]
    res = paired_bootstrap(recs, list(recs), pooled_f1, seed=5, n_resamples=300)
    assert abs(res.delta) < 1e-9
    assert res.p_value > 0.5  # no real difference


def test_paired_bootstrap_a_strictly_better_low_p() -> None:
    a = [Counts(tp=5, fp=0, fn=0) for _ in range(50)]   # perfect
    b = [Counts(tp=1, fp=4, fn=4) for _ in range(50)]   # poor
    res = paired_bootstrap(a, b, pooled_f1, seed=9, n_resamples=300)
    assert res.delta > 0.5
    assert res.p_value < 0.05


def test_holm_bonferroni_controls_family() -> None:
    # 3 comparisons; only the smallest p should survive Holm at alpha=0.05.
    pvals = {"c1": 0.001, "c2": 0.04, "c3": 0.20}
    sig = holm_bonferroni(pvals, alpha=0.05)
    assert sig["c1"] is True       # 0.001 <= 0.05/3
    assert sig["c2"] is False      # 0.04 > 0.05/2
    assert sig["c3"] is False


def test_holm_bonferroni_empty() -> None:
    assert holm_bonferroni({}) == {}
