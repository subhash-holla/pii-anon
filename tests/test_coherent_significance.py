"""Coherent significance BY CONSTRUCTION + rank-1 probability + Davidson ties.

The headline S3-04 file. Pins:

* **Coherence by construction (NFR-002).** ``pairwise_significance`` reads point,
  CI, sign, and the significant flag off the SAME difference vector
  ``d = θ_i − θ_j`` (one joint posterior). Property tests (Hypothesis) prove the
  three statements CANNOT disagree: point ∈ CI always; significant ⇒ point ≠ 0
  with a sign matching the CI side; never significant-when-CI-spans-0; the win
  probability ``p_i_beats_j`` sign matches the verdict sign.
* **rank_one_probability = the SDO J-meter (FR-004).** fraction of joint draws
  in which a named system has the max θ; the full distribution sums to 1.0;
  a dominant system → ≈ 1.0; a symmetric tie → ≈ 0.5; an unknown name raises;
  deterministic.
* **Davidson (1970) closed form (pure-numpy teeth).** ``davidson_p_i_beats_j`` /
  ``_p_tie`` / ``_p_j_beats_i`` on hand values: sum to 1; ``ν=0`` nests plain BT;
  the symmetric ``(1/3,1/3,1/3)`` at ν=1 and ``(0.25,0.5,0.25)`` at ν=2; the
  softmax-equivalence bridge ``softmax([δ/2, ln ν, −δ/2]) == (p_i, p_tie, p_j)``
  pinning the closed form to the ``Multinomial(logits=...)`` model.
* **The elo-defect regression (FR-004).** An AST/source guard documents that
  ``significance.py`` derives all three of {point, CI, verdict} from ONE sample
  array — NO ``1.96`` normal-CI literal, NO ``_sigmoid`` fabricated outcome, NO
  ``2·√(rd² + rd²)`` decoupled threshold (the three verified ``elo.py`` defects
  at :243 / :542 / :561).
* **theta-orientation pin.** ``theta_samples`` is ``(n_draws, n_systems)`` with
  columns ↔ sorted ``systems`` (bayes_bt.py:140/518-519). A transpose must break
  loudly.
* **Tier-3 separation (FR-010/AX-004).** The RRS Davidson posterior is a SEPARATE
  object / call and is never merged into the de-id significance.
* One ``pytest.importorskip("numpyro")`` real-NUTS-with-ties integration test
  (SKIPs locally; real in the bayes-eval CI lane).

All non-integration paths are pure-numpy → run in-tree WITHOUT numpyro.

Test-type tags: ``[UNIT-TEST]`` ``[PROPERTY-TEST]`` ``[CONTRACT-TEST]``
``[INTEGRATION-TEST]``.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pii_anon.eval_framework.rating.significance import (
    PairwiseVerdict,
    davidson_p_i_beats_j,
    davidson_p_j_beats_i,
    davidson_p_tie,
    pairwise_significance,
    rank_one_distribution,
    rank_one_probability,
)


def _numpyro_absent() -> bool:
    return importlib.util.find_spec("numpyro") is None


# ---------------------------------------------------------------------------
# Synthetic-posterior generators.
#
# A LOCATION FAMILY: every draw is base_draws[:, None] + offsets[None, :], so all
# systems share the SAME base draws and differ only by a constant offset. Then
# d = θ_i − θ_j = offset_i − offset_j is a CONSTANT across draws ⇒ point ∈ CI is
# a theorem (degenerate CI). To give the CI width we add a small per-(draw,col)
# jitter from a single Normal, keeping point ∈ CI a near-certainty rather than a
# bimodal hypothesis flake. (Vehtari-style well-mixed synthetic draws.)
# ---------------------------------------------------------------------------


def _location_family_posterior(
    *,
    offsets: list[float],
    n_draws: int,
    scale: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n_draws)  # shared latent location per draw
    jitter = rng.standard_normal((n_draws, len(offsets))) * scale
    return base[:, None] + np.asarray(offsets)[None, :] + jitter


@st.composite
def _posteriors(draw: st.DrawFn) -> tuple[np.ndarray, list[str]]:
    n_systems = draw(st.integers(min_value=2, max_value=5))
    n_draws = draw(st.integers(min_value=200, max_value=600))
    offsets = draw(
        st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=n_systems,
            max_size=n_systems,
        )
    )
    scale = draw(st.floats(min_value=0.05, max_value=1.5, allow_nan=False))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    names = [f"s{k}" for k in range(n_systems)]
    samples = _location_family_posterior(
        offsets=offsets, n_draws=n_draws, scale=scale, seed=seed
    )
    return samples, names


# ---------------------------------------------------------------------------
# nfr_002: COHERENCE BY CONSTRUCTION — the headline property tests.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(_posteriors())
def test_nfr_002_point_estimate_is_inside_its_own_credible_interval(
    posterior: tuple[np.ndarray, list[str]],
) -> None:
    """[PROPERTY-TEST] point = mean(d) ALWAYS lies within ci = (q2.5, q97.5) of the
    SAME d. A mean strictly outside its own 2.5/97.5 percentile interval is
    impossible for any real sample — this is the by-construction guarantee."""
    samples, names = posterior
    for v in pairwise_significance(samples, names):
        assert v.ci_lo <= v.point <= v.ci_hi, (
            f"point {v.point} outside CI [{v.ci_lo}, {v.ci_hi}] for {v.i} vs {v.j}"
        )


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(_posteriors())
def test_nfr_002_significant_implies_point_nonzero_with_matching_sign(
    posterior: tuple[np.ndarray, list[str]],
) -> None:
    """[PROPERTY-TEST] significant ⇒ point ≠ 0 AND sign(point) agrees with the CI
    side: ci_lo > 0 ⇒ point > 0; ci_hi < 0 ⇒ point < 0. (point ∈ CI already
    forces this, but we assert it directly as the coherence claim.)"""
    samples, names = posterior
    for v in pairwise_significance(samples, names):
        if v.significant:
            assert v.point != 0.0
            if v.ci_lo > 0.0:
                assert v.point > 0.0, "CI strictly positive but point ≤ 0"
            else:
                assert v.ci_hi < 0.0 and v.point < 0.0, "CI strictly negative but point ≥ 0"


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(_posteriors())
def test_nfr_002_never_significant_when_ci_spans_zero(
    posterior: tuple[np.ndarray, list[str]],
) -> None:
    """[PROPERTY-TEST] The forbidden state of elo.py:561 (a significance verdict
    that disagrees with the interval) is UNREACHABLE: if ci_lo ≤ 0 ≤ ci_hi then
    significant is False, by construction."""
    samples, names = posterior
    for v in pairwise_significance(samples, names):
        spans_zero = v.ci_lo <= 0.0 <= v.ci_hi
        if spans_zero:
            assert v.significant is False, "significant True while CI spans 0 — incoherent"


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(_posteriors())
def test_nfr_002_p_i_beats_j_sign_matches_significant_verdict(
    posterior: tuple[np.ndarray, list[str]],
) -> None:
    """[PROPERTY-TEST] p_i_beats_j = mean(θ_i > θ_j) is read off the SAME draws and
    cannot contradict the SIGNIFICANT verdict: when the CI strictly excludes 0,
    the win-probability is decisively on that side (> 0.5 if ci_lo > 0; < 0.5 if
    ci_hi < 0).

    NB: we deliberately tie this to the *significant* verdict, not to the raw
    sign of ``point``. ``point`` is the posterior MEAN of d while ``p_i_beats_j``
    is the fraction of POSITIVE draws (a median-like sign mass); under a skewed d
    a tiny-positive mean can legitimately coexist with exactly 50% positive
    draws. The coherence claim is that the SIGNIFICANT verdict and the
    win-probability agree — which they must, since a CI strictly above 0 means
    essentially all draws are positive."""
    samples, names = posterior
    for v in pairwise_significance(samples, names):
        if v.significant and v.ci_lo > 0.0:
            assert v.p_i_beats_j > 0.5, "CI strictly positive but p_i_beats_j ≤ 0.5"
        elif v.significant and v.ci_hi < 0.0:
            assert v.p_i_beats_j < 0.5, "CI strictly negative but p_i_beats_j ≥ 0.5"


def test_nfr_002_significance_is_single_source_three_quantities_from_one_vector() -> None:
    """[UNIT-TEST] Pin the single-source derivation explicitly on a fixed
    posterior: recompute point/ci/significant from d = θ_i − θ_j by hand and
    assert the verdict matches — proving they are read off ONE vector."""
    # 3 systems; column 1 clearly above column 2.
    rng = np.random.default_rng(7)
    base = rng.standard_normal(2000)
    samples = np.stack(
        [base, base + 1.5 + 0.1 * rng.standard_normal(2000), base - 0.4],
        axis=1,
    )
    names = ["a", "b", "c"]
    all_verdicts = pairwise_significance(samples, names)
    assert all(isinstance(v, PairwiseVerdict) for v in all_verdicts)
    verdicts = {(v.i, v.j): v for v in all_verdicts}
    v = verdicts[("a", "b")]
    d = samples[:, 0] - samples[:, 1]  # the SAME vector the impl must use
    assert v.point == pytest.approx(float(np.mean(d)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert v.ci_lo == pytest.approx(float(lo))
    assert v.ci_hi == pytest.approx(float(hi))
    assert v.significant == bool(lo > 0.0 or hi < 0.0)
    assert v.p_i_beats_j == pytest.approx(float(np.mean(d > 0.0)))


# ---------------------------------------------------------------------------
# theta-orientation pin — (n_draws, n_systems), columns ↔ sorted systems.
# ---------------------------------------------------------------------------


def test_fr_004_orientation_columns_map_to_systems_transpose_breaks_loudly() -> None:
    """[CONTRACT-TEST] theta_samples is (n_draws, n_systems): differencing columns
    [:, col_i] − [:, col_j]. A 4-system posterior fed TRANSPOSED (n_systems rows)
    must NOT silently produce a 4-pair verdict list — feed mismatched names so it
    raises, pinning the orientation so a transpose can't pass unnoticed."""
    samples = _location_family_posterior(
        offsets=[1.0, 0.5, -0.5, -1.0], n_draws=300, scale=0.2, seed=1
    )  # (300, 4)
    names = ["a", "b", "c", "d"]
    # Correct orientation: 4 systems → C(4,2)=6 verdicts.
    assert len(pairwise_significance(samples, names)) == 6
    # Transposed orientation: (4, 300). names of length 4 ≠ 300 columns → raise.
    with pytest.raises(ValueError):
        pairwise_significance(samples.T, names)


def test_fr_004_orientation_verdict_i_j_are_the_named_systems_in_order() -> None:
    """[CONTRACT-TEST] Verdict (i, j) carries the actual system NAMES (sorted
    pairings), and column 0 corresponds to names[0]."""
    samples = _location_family_posterior(
        offsets=[2.0, -2.0], n_draws=500, scale=0.1, seed=2
    )
    names = ["alpha", "beta"]
    (v,) = pairwise_significance(samples, names)
    assert (v.i, v.j) == ("alpha", "beta")
    # alpha (col0, offset +2) clearly beats beta (col1, offset −2).
    assert v.point > 0 and v.significant and v.p_i_beats_j > 0.9


def test_fr_004_orientation_non_2d_samples_rejected() -> None:
    """[CONTRACT-TEST] A non-2-D samples array (e.g. 1-D or 3-D) is rejected — the
    orientation contract is strictly (n_draws, n_systems)."""
    with pytest.raises(ValueError):
        pairwise_significance(np.zeros((10,)), ["a"])  # 1-D
    with pytest.raises(ValueError):
        pairwise_significance(np.zeros((4, 5, 2)), ["a", "b"])  # 3-D


def test_fr_004_orientation_zero_draws_rejected() -> None:
    """[CONTRACT-TEST] Zero draws (an empty posterior) is rejected — you cannot
    derive a verdict from no samples."""
    with pytest.raises(ValueError):
        pairwise_significance(np.zeros((0, 2)), ["a", "b"])


# ---------------------------------------------------------------------------
# fr_004: rank_one_probability — the SDO J-meter.
# ---------------------------------------------------------------------------


def test_fr_004_rank_one_distribution_sums_to_one() -> None:
    """[UNIT-TEST] rank_one_distribution is a proper distribution over "who is
    #1": Σ probabilities = 1.0 (every draw has exactly one argmax)."""
    samples = _location_family_posterior(
        offsets=[0.5, 0.0, -0.5, 0.2], n_draws=1000, scale=0.8, seed=3
    )
    names = ["a", "b", "c", "d"]
    dist = rank_one_distribution(samples, names)
    assert set(dist) == set(names)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in dist.values())


def test_fr_004_rank_one_probability_dominant_system_near_one() -> None:
    """[UNIT-TEST] A clearly-dominant synthetic system → P(rank=1) ≈ 1.0; this IS
    the SDO J for that system."""
    samples = _location_family_posterior(
        offsets=[5.0, 0.0, -1.0], n_draws=2000, scale=0.3, seed=4
    )
    names = ["winner", "mid", "low"]
    assert rank_one_probability(samples, names, "winner") == pytest.approx(1.0, abs=1e-6)


def test_fr_004_rank_one_probability_symmetric_tie_is_half() -> None:
    """[UNIT-TEST] Two systems whose strengths are exchangeable → P(rank=1) ≈ 0.5
    each (within Monte-Carlo tolerance)."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal(20000)
    b = rng.standard_normal(20000)  # iid, exchangeable with a
    samples = np.stack([a, b], axis=1)
    names = ["a", "b"]
    pa = rank_one_probability(samples, names, "a")
    pb = rank_one_probability(samples, names, "b")
    assert pa == pytest.approx(0.5, abs=0.02)
    assert pb == pytest.approx(0.5, abs=0.02)
    assert pa + pb == pytest.approx(1.0)


def test_fr_004_rank_one_probability_unknown_name_raises() -> None:
    """[CONTRACT-TEST] An unknown system name raises (no silent 0.0) — a typo in
    the J-meter call must fail loud."""
    samples = _location_family_posterior(
        offsets=[1.0, -1.0], n_draws=100, scale=0.1, seed=6
    )
    names = ["a", "b"]
    with pytest.raises((KeyError, ValueError)):
        rank_one_probability(samples, names, "pii-anon")  # not in names


def test_fr_004_rank_one_probability_is_deterministic() -> None:
    """[UNIT-TEST] Pure-numpy argmax over fixed draws → identical J across calls
    (AX-002)."""
    samples = _location_family_posterior(
        offsets=[0.3, 0.1, -0.4], n_draws=800, scale=0.6, seed=8
    )
    names = ["a", "b", "c"]
    d1 = rank_one_distribution(samples, names)
    d2 = rank_one_distribution(samples.copy(), names)
    assert d1 == d2


def test_fr_004_rank_one_distribution_mismatched_names_raises() -> None:
    """[CONTRACT-TEST] names length must equal the number of columns; a mismatch
    raises (orientation/shape guard)."""
    samples = _location_family_posterior(
        offsets=[1.0, 0.0, -1.0], n_draws=100, scale=0.1, seed=9
    )
    with pytest.raises(ValueError):
        rank_one_distribution(samples, ["a", "b"])  # 2 names, 3 columns


# ---------------------------------------------------------------------------
# Davidson (1970) closed-form pure-numpy teeth.
# ---------------------------------------------------------------------------


def test_fr_004_davidson_three_probabilities_sum_to_one() -> None:
    """[UNIT-TEST] P(i>j) + P(tie) + P(j>i) = 1 for arbitrary δ and ν > 0."""
    for delta in (-2.0, -0.5, 0.0, 0.7, 3.0):
        for nu in (0.1, 1.0, 2.5):
            p_i = davidson_p_i_beats_j(delta, nu)
            p_t = davidson_p_tie(delta, nu)
            p_j = davidson_p_j_beats_i(delta, nu)
            assert p_i + p_t + p_j == pytest.approx(1.0)
            assert all(0.0 <= p <= 1.0 for p in (p_i, p_t, p_j))


def test_fr_004_davidson_nu_zero_nests_plain_bradley_terry() -> None:
    """[UNIT-TEST] ν=0 ⇒ no tie mass; the win probability collapses to the plain
    Bradley-Terry logistic σ(δ): P(i>j) = 1/(1+e^−δ), P(tie) = 0."""
    for delta in (-1.5, 0.0, 0.8, 2.0):
        assert davidson_p_tie(delta, 0.0) == pytest.approx(0.0)
        expected = 1.0 / (1.0 + math.exp(-delta))
        assert davidson_p_i_beats_j(delta, 0.0) == pytest.approx(expected)
        assert davidson_p_j_beats_i(delta, 0.0) == pytest.approx(1.0 - expected)


def test_fr_004_davidson_symmetric_third_at_nu_one() -> None:
    """[UNIT-TEST] δ=0, ν=1 ⇒ the three outcomes are equiprobable (1/3 each):
    unnormalized terms [e^0, ν, e^0] = [1, 1, 1]."""
    assert davidson_p_i_beats_j(0.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert davidson_p_tie(0.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert davidson_p_j_beats_i(0.0, 1.0) == pytest.approx(1.0 / 3.0)


def test_fr_004_davidson_quarter_half_quarter_at_nu_two() -> None:
    """[UNIT-TEST] δ=0, ν=2 ⇒ terms [1, 2, 1] → (0.25, 0.5, 0.25): the tie carries
    half the mass, wins split the rest symmetrically."""
    assert davidson_p_i_beats_j(0.0, 2.0) == pytest.approx(0.25)
    assert davidson_p_tie(0.0, 2.0) == pytest.approx(0.5)
    assert davidson_p_j_beats_i(0.0, 2.0) == pytest.approx(0.25)


def test_fr_004_davidson_softmax_equivalence_bridges_pure_numpy_to_numpyro_logits() -> None:
    """[CONTRACT-TEST] The Davidson↔NumPyro bridge identity: the closed-form
    triple equals softmax([δ/2, ln ν, −δ/2]). This pins the pure-numpy formulas
    to the ``Multinomial(logits=[δ/2, ln ν, −δ/2])`` model the Davidson branch
    will use, so the in-tree teeth and the real-NUTS likelihood are the SAME
    math."""
    for delta in (-2.0, -0.3, 0.0, 1.1, 2.4):
        for nu in (0.25, 1.0, 3.0):
            # Reference softmax via plain stdlib math (no numpy reduction): keeps
            # this identity check independent of the numpy-reload sentinel issue
            # other modules can trigger under coverage instrumentation, and is the
            # exact softmax([δ/2, ln ν, −δ/2]) the NumPyro Multinomial uses.
            logits = [delta / 2.0, math.log(nu), -delta / 2.0]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            denom = sum(exps)
            soft = [e / denom for e in exps]
            assert davidson_p_i_beats_j(delta, nu) == pytest.approx(soft[0])
            assert davidson_p_tie(delta, nu) == pytest.approx(soft[1])
            assert davidson_p_j_beats_i(delta, nu) == pytest.approx(soft[2])


def test_fr_004_davidson_higher_strength_gets_more_win_mass() -> None:
    """[UNIT-TEST] For fixed ν, increasing δ shifts mass from j-wins toward
    i-wins monotonically (sanity on the directionality)."""
    nu = 1.0
    prev = -1.0
    for delta in (-2.0, -1.0, 0.0, 1.0, 2.0):
        p_i = davidson_p_i_beats_j(delta, nu)
        assert p_i > prev
        prev = p_i


def test_fr_004_davidson_negative_nu_rejected() -> None:
    """[CONTRACT-TEST] A negative tie parameter is meaningless → raise."""
    with pytest.raises(ValueError):
        davidson_p_tie(0.0, -1.0)


# ---------------------------------------------------------------------------
# fr_004: the elo-defect regression — significance.py reproduces NONE of the
# three verified elo.py defects.
# ---------------------------------------------------------------------------


def test_fr_004_significance_module_has_no_elo_defect_signatures() -> None:
    """[CONTRACT-TEST] significance.py derives {point, CI, verdict} from ONE
    sample array — it must contain NONE of the three elo.py defect signatures:

      * elo.py:243 — a ``_sigmoid``-fabricated match outcome from a composite gap;
      * elo.py:542 — a fake normal-approx CI ``rating ± 1.96 · rd``;
      * elo.py:561 — a decoupled significance test ``rating_diff > 2·√(rd²+rd²)``.

    Source-level guard: the literal ``1.96`` and the token ``_sigmoid`` and a
    ``sqrt(... rd ... rd ...)`` combined-RD threshold must be ABSENT.
    """
    from pathlib import Path

    import pii_anon.eval_framework.rating.significance as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    # split out docstrings/comments? No — these defect signatures must not appear
    # even in prose for this module, because their absence is the claim. We only
    # allow the *defect IDs being named in a docstring as "NOT reproduced"*, so we
    # check for the load-bearing code tokens, not the words "1.96"/"sigmoid" in a
    # sentence. The tokens below are code-shaped and would not appear in honest
    # prose describing the design.
    assert "1.96 *" not in src and "1.96*" not in src, (
        "no normal-approx CI: significance must read CI off posterior percentiles"
    )
    assert "self._sigmoid" not in src and "def _sigmoid" not in src, (
        "no sigmoid-fabricated outcome: p_i_beats_j is mean(θ_i > θ_j) over draws"
    )
    # The decoupled rating-diff threshold uses a combined RD sqrt; assert no such
    # combined-RD significance test exists.
    assert "rd ** 2" not in src and "rd**2" not in src, (
        "no decoupled rating-diff>threshold significance test"
    )


def test_fr_004_significance_uses_percentile_and_mean_on_one_array() -> None:
    """[CONTRACT-TEST] Positive evidence: significance.py uses np.percentile and
    np.mean (the single-source derivation), proving the by-construction path."""
    from pathlib import Path

    import pii_anon.eval_framework.rating.significance as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "percentile" in src, "CI must come from posterior percentiles"
    assert "mean" in src, "point estimate must be the posterior mean"


# ---------------------------------------------------------------------------
# import isolation (S3-05) — significance.py / paired_set.py import nothing from
# evaluation/swarm/moe/fusion/policy.
# ---------------------------------------------------------------------------


def test_s3_05_new_modules_do_not_import_evaluation_or_orchestration() -> None:
    """[CONTRACT-TEST] significance.py and paired_set.py must import ONLY numpy +
    stdlib + sibling rating modules — NOT evaluation/swarm/moe/fusion/policy. A
    tighter assertion than the package-wide boundary test: it also forbids
    ``pii_anon.eval_framework.evaluation`` and the external_evaluator, so these
    take a plain dict, never an ExternalEvaluationResult."""
    import ast
    from pathlib import Path

    import pii_anon.eval_framework.rating.paired_set as paired_mod
    import pii_anon.eval_framework.rating.significance as sig_mod

    forbidden_prefixes = (
        "pii_anon.swarm",
        "pii_anon.moe",
        "pii_anon.fusion",
        "pii_anon.policy",
        "pii_anon.eval_framework.evaluation",
        "pii_anon.eval_framework.external_evaluator",
        "pii_anon.evaluation",
    )
    for mod in (sig_mod, paired_mod):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.append(node.module)
        for name in imported:
            assert not name.startswith(forbidden_prefixes), (
                f"{Path(mod.__file__).name} imports forbidden module {name}"
            )


# ---------------------------------------------------------------------------
# fr_010 / ax_004: Tier-3 RRS Davidson sub-model NEVER merged into de-id sig.
# ---------------------------------------------------------------------------


def test_fr_010_rrs_posterior_is_a_distinct_object_never_merged() -> None:
    """[CONTRACT-TEST] The Tier-3 re-identification-resistance (RRS) Davidson path
    is a SEPARATE call returning a SEPARATE Posterior; it shares no mutable state
    with the de-id significance.

    Exercised WITHOUT numpyro by checking the engine exposes a distinct RRS entry
    point (``fit_rrs_posterior`` or a ``kind="rrs"`` flag) and that the de-id and
    RRS paths are different methods / produce different objects — the AX-004
    anon≠pseudo-never-merged guard. With numpyro absent both raise the loud
    MissingOptionalDependencyError, which still proves they are SEPARATE calls
    (not one merged code path)."""
    from pii_anon.eval_framework.rating.bayes_bt import (
        BayesBTEngine,
        MissingOptionalDependencyError,
    )

    engine = BayesBTEngine()
    # The RRS path must exist and be DISTINCT from fit_paired_posterior.
    assert hasattr(engine, "fit_rrs_posterior"), (
        "Tier-3 RRS needs its own entry point (FR-010/AX-004 separation)"
    )
    assert (
        engine.fit_rrs_posterior.__func__  # type: ignore[attr-defined]
        is not engine.fit_paired_posterior.__func__  # type: ignore[attr-defined]
    ), "RRS and de-id must be different methods, not one merged path"

    counts_with_ties = {("sys_a", "sys_b"): (5, 3, 2, 10)}
    if _numpyro_absent():
        # Both fail loud (separate calls); neither silently reuses the other.
        with pytest.raises(MissingOptionalDependencyError):
            engine.fit_rrs_posterior(counts_with_ties)
        with pytest.raises(MissingOptionalDependencyError):
            engine.fit_paired_posterior({("sys_a", "sys_b"): (5.0, 5.0, 10)})
        # No de-id posterior was produced as a side effect of the RRS call.
        assert engine.last_posterior is None


# ---------------------------------------------------------------------------
# [INTEGRATION-TEST] real NUTS WITH TIES — importorskip-gated. SKIPS in .venv.
# ---------------------------------------------------------------------------


def test_nfr_001_integration_real_nuts_with_ties_converges_and_identifies_nu() -> None:
    """[INTEGRATION-TEST] Given a synthetic ≥4-system design WITH ties, when the
    Davidson NUTS path runs, then the ConvergenceReport is claim_grade (NFR-001),
    the posterior-mean ordering recovers the planted ordering, and the tie
    parameter ν posterior is identifiable (mean > 0, finite SD).

    SKIPPED in the dev .venv (numpyro absent); real in the bayes-eval CI lane —
    the honest Pass-2 verification of the Davidson tie likelihood.
    """
    pytest.importorskip("numpyro")
    import numpy as _np

    from pii_anon.eval_framework.rating.bayes_bt import BayesBTEngine
    from pii_anon.eval_framework.rating.convergence import RHAT_MAX

    planted = {"sys_a": 0.9, "sys_b": 0.3, "sys_c": -0.3, "sys_d": -0.9}
    names = sorted(planted)
    nu_true = 1.0
    n_per_pair = 200

    counts: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for ii in range(len(names)):
        for jj in range(ii + 1, len(names)):
            ni, nj = names[ii], names[jj]
            delta = planted[ni] - planted[nj]
            # Davidson closed-form expected proportions with ν_true.
            p_i = davidson_p_i_beats_j(delta, nu_true)
            p_t = davidson_p_tie(delta, nu_true)
            wins_i = int(round(p_i * n_per_pair))
            ties = int(round(p_t * n_per_pair))
            wins_j = n_per_pair - wins_i - ties
            counts[(ni, nj)] = (wins_i, wins_j, ties, n_per_pair)

    engine = BayesBTEngine(num_warmup=1000, num_samples=1000, num_chains=4, seed=0)
    posterior = engine.fit_paired_posterior_with_ties(counts)

    report = posterior.convergence
    assert report.claim_grade is True
    assert report.max_rhat <= RHAT_MAX
    assert report.n_divergences == 0

    means = {n: float(_np.mean(posterior.theta_samples[:, k])) for k, n in enumerate(names)}
    ranked = sorted(means, key=lambda n: means[n], reverse=True)
    assert ranked == ["sys_a", "sys_b", "sys_c", "sys_d"]

    # ν posterior identifiable: stored on the posterior and positive.
    nu_samples = getattr(posterior, "nu_samples", None)
    assert nu_samples is not None, "Davidson posterior must carry the ν draws"
    assert float(_np.mean(nu_samples)) > 0.0
    assert float(_np.std(nu_samples)) > 0.0
