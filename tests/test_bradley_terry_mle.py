"""Tests for S3-02: BradleyTerryMLEEngine (pure-stdlib MM + paired bootstrap).

Implements / pins:
    FR-003  — rating-engine abstraction, SECOND ladder tier (``bradley-terry-mle``)
              fitted by minorization-maximization (Hunter 2004), structurally
              satisfying ``RatingEnginePort`` with ZERO call-site changes.
    NFR-002 / NFR-003 — significance-coherence FOUNDATION via paired
              (record-resampling) bootstrap: deterministic CIs, point ∈ CI,
              and rank↔CI coherence (sign↔verdict cannot disagree).
    NFR-026 — graceful degradation: the ``bradley-terry-mle`` entry point is
              discovered alongside ``glicko-legacy`` and resolves to a
              ``RatingEnginePort``; absent optional tiers never raise.
    AX-002  — determinism: same input + same seed ⇒ byte-identical output.

Design trace: D-EVAL DECISION 2 — RatingEnginePort 3-tier ladder
(``glicko-legacy`` → ``bradley-terry-mle`` → ``bayes-bt``). This is the fast
PR-CI / smoke tier (pure ``math`` + ``random.Random(seed)``; no numpy/scipy).

Mathematical contracts encoded here
-----------------------------------
* MM recovers the planted ordering from an asymmetric win-count design.
* The fitted log-strengths satisfy the BT stationarity condition
  ``Σ_j n_ij · π_i/(π_i + π_j) == W_i`` within tol (the MLE fixed point).
* A perfect total order (degenerate point composites) does NOT diverge: the
  soft-outcome mapping ``sigmoid(γ·(C_i − C_j))`` keeps strengths finite and
  convergence is reached within ``max_iter``.
* Sum-to-zero identifiability: ``Σ_i θ_i == 0``.
* Ratings are reported on the Elo scale ``1500 + 400·θ/ln(10)`` so deltas read
  like Elo points (``scale=400`` matches ``elo.py``).
"""

from __future__ import annotations

import math
import warnings

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pii_anon.eval_framework.rating import (
    BradleyTerryConvergenceWarning,
    RatingEnginePort,
    RatingEngineRegistry,
)
from pii_anon.eval_framework.rating.bradley_terry import BradleyTerryMLEEngine
from pii_anon.eval_framework.rating.elo import EloRating, RatingUpdate

RATING_GROUP = "pii_anon.rating_engines"
ELO_SCALE = 400.0
ELO_ANCHOR = 1500.0


def _theta_from_rating(rating: float) -> float:
    """Invert the Elo-scale mapping back to a sum-to-zero log-strength."""
    return (rating - ELO_ANCHOR) * math.log(10.0) / ELO_SCALE


# ---------------------------------------------------------------------------
# fr_003: structural port conformance (zero call-site changes)
# ---------------------------------------------------------------------------

def test_fr_003_bt_engine_is_a_rating_engine_port_runtime_checkable() -> None:
    """Given the runtime-checkable Protocol, when a BT engine instance is
    isinstance-checked, then it satisfies RatingEnginePort with ZERO
    inheritance (structural) — the port keeps both ladder tiers stable."""
    engine = BradleyTerryMLEEngine()
    assert isinstance(engine, RatingEnginePort)


def test_fr_003_bt_engine_does_not_inherit_the_port() -> None:
    """Conformance must be STRUCTURAL, not nominal — the engine must not
    subclass the Protocol (that would defeat the zero-coupling contract)."""
    assert RatingEnginePort not in type(BradleyTerryMLEEngine()).__mro__


def test_fr_003_run_round_robin_returns_rating_updates() -> None:
    """When run_round_robin runs over a composite map, then it returns a
    list[RatingUpdate] (audit-history parity with the Elo engine)."""
    port: RatingEnginePort = BradleyTerryMLEEngine()
    updates = port.run_round_robin({"sys_a": 0.9, "sys_b": 0.6, "sys_c": 0.3})
    assert isinstance(updates, list)
    assert updates, "round-robin over 3 systems must yield updates"
    assert all(isinstance(u, RatingUpdate) for u in updates)


def test_fr_003_get_rating_returns_elo_rating_for_every_system() -> None:
    """After a round-robin, get_rating returns an EloRating for every system
    present in composites, and None for an unknown system."""
    engine = BradleyTerryMLEEngine()
    composites = {"sys_a": 0.9, "sys_b": 0.6, "sys_c": 0.3}
    engine.run_round_robin(composites)
    for name in composites:
        rating = engine.get_rating(name)
        assert isinstance(rating, EloRating)
        assert rating.system_name == name
        assert math.isfinite(rating.rating)
    assert engine.get_rating("does-not-exist") is None


def test_fr_003_port_typed_object_is_fully_usable_through_the_port() -> None:
    """An object typed as RatingEnginePort can be driven entirely through the
    port surface (run_round_robin + get_rating), proving signature alignment."""
    port: RatingEnginePort = BradleyTerryMLEEngine()
    port.run_round_robin({"a": 0.8, "b": 0.2})
    assert isinstance(port.get_rating("a"), EloRating)
    assert port.get_rating("missing") is None


# ---------------------------------------------------------------------------
# fr_003: MM correctness — recovers planted ordering + Elo-scale mapping
# ---------------------------------------------------------------------------

def test_fr_003_run_round_robin_recovers_composite_ordering_on_elo_scale() -> None:
    """The port path maps composites to soft pairwise outcomes; the higher
    composite must receive the higher Elo-scale rating (faithful drop-in)."""
    engine = BradleyTerryMLEEngine()
    engine.run_round_robin({"top": 0.95, "mid": 0.55, "low": 0.15})
    r_top = engine.get_rating("top")
    r_mid = engine.get_rating("mid")
    r_low = engine.get_rating("low")
    assert r_top is not None and r_mid is not None and r_low is not None
    assert r_top.rating > r_mid.rating > r_low.rating


def test_fr_003_fit_paired_recovers_known_asymmetric_ordering() -> None:
    """Given a paired-outcome design with known asymmetric win counts
    (A>B>C), when MM runs to convergence, then the recovered log-strength
    ordering matches the true ordering."""
    engine = BradleyTerryMLEEngine()
    comparisons = {
        ("A", "B"): (8.0, 2.0, 10),
        ("B", "C"): (7.0, 3.0, 10),
        ("A", "C"): (9.0, 1.0, 10),
    }
    theta = engine.fit_paired(comparisons)
    assert theta["A"] > theta["B"] > theta["C"]


def test_fr_003_fit_paired_satisfies_bt_stationarity_condition() -> None:
    """The fitted strengths must satisfy the BT MLE stationary equation
    Σ_j n_ij · π_i/(π_i+π_j) == W_i within tol — i.e. it is the actual MLE
    fixed point, not merely a monotone heuristic."""
    engine = BradleyTerryMLEEngine()
    comparisons = {
        ("A", "B"): (8.0, 2.0, 10),
        ("B", "C"): (7.0, 3.0, 10),
        ("A", "C"): (9.0, 1.0, 10),
    }
    theta = engine.fit_paired(comparisons)
    pi = {name: math.exp(t) for name, t in theta.items()}

    # Reconstruct W_i (fractional wins) and n_ij (symmetric games) from input.
    wins: dict[str, float] = {name: 0.0 for name in theta}
    games: dict[tuple[str, str], float] = {}
    for (i, j), (wi, wj, n) in comparisons.items():
        wins[i] += wi
        wins[j] += wj
        games[(i, j)] = float(n)
        games[(j, i)] = float(n)

    max_resid = 0.0
    for i in theta:
        expected = 0.0
        for j in theta:
            if j == i:
                continue
            n_ij = games.get((i, j), 0.0)
            if n_ij:
                expected += n_ij * pi[i] / (pi[i] + pi[j])
        max_resid = max(max_resid, abs(wins[i] - expected))
    assert max_resid < 1e-6, f"stationarity residual too large: {max_resid}"


def test_fr_003_fit_paired_is_sum_to_zero_anchored() -> None:
    """Identifiability is anchored by Σ_i θ_i == 0 (geometric-mean-1 in π)."""
    engine = BradleyTerryMLEEngine()
    theta = engine.fit_paired(
        {
            ("A", "B"): (6.0, 4.0, 10),
            ("B", "C"): (5.0, 5.0, 10),
            ("A", "C"): (7.0, 3.0, 10),
        }
    )
    assert abs(sum(theta.values())) < 1e-9


def test_nfr_026_run_round_robin_ratings_lie_on_the_elo_scale() -> None:
    """Ratings are reported as 1500 + 400·θ/ln10. Inverting a reported rating
    must recover a sum-to-zero θ — proving the documented Elo-scale mapping."""
    engine = BradleyTerryMLEEngine()
    composites = {"a": 0.9, "b": 0.5, "c": 0.1}
    engine.run_round_robin(composites)
    thetas = []
    for name in composites:
        r = engine.get_rating(name)
        assert r is not None
        thetas.append(_theta_from_rating(r.rating))
    assert abs(sum(thetas)) < 1e-6


# ---------------------------------------------------------------------------
# fr_003 (negative): degenerate perfect total order does NOT diverge
# ---------------------------------------------------------------------------

def test_fr_003_degenerate_total_order_does_not_diverge() -> None:
    """Negative case: point composites are a PERFECT total order ⇒ a raw BT
    MLE would send strengths to ±∞. The soft-outcome mapping (sigmoid of
    composite gaps) + smoothing must keep every rating finite and ordered.

    This widely-separated 6-system design is *near-separable*: the MM fit
    exhausts max_iter (honestly flagged via BradleyTerryConvergenceWarning —
    asserted by ``test_code_quality_*``), but the result stays finite + strictly
    monotone, which is the contract this test pins. The convergence warning is
    suppressed here so this assertion set is unaffected by the diagnostics."""
    engine = BradleyTerryMLEEngine()
    # A widely separated, strictly monotone total order.
    composites = {f"s{idx}": float(idx) for idx in range(6)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BradleyTerryConvergenceWarning)
        updates = engine.run_round_robin(composites)
    assert updates
    for name in composites:
        rating = engine.get_rating(name)
        assert rating is not None
        assert math.isfinite(rating.rating), f"{name} rating diverged"
        assert math.isfinite(rating.rd)
    # Strict monotonicity preserved despite separability.
    ordered = [engine.get_rating(f"s{idx}") for idx in range(6)]
    values = [r.rating for r in ordered if r is not None]
    assert values == sorted(values)


def test_fr_003_two_system_degenerate_design_is_finite() -> None:
    """The minimal separable design (one pair, one wins every notional game)
    must still produce a finite, ordered result via the soft mapping."""
    engine = BradleyTerryMLEEngine()
    engine.run_round_robin({"winner": 1.0, "loser": 0.0})
    rw = engine.get_rating("winner")
    rl = engine.get_rating("loser")
    assert rw is not None and rl is not None
    assert math.isfinite(rw.rating) and math.isfinite(rl.rating)
    assert rw.rating > rl.rating


# ---------------------------------------------------------------------------
# code-quality-story-01: MM convergence MUST be observable, never silent
#
# The MM fit can exhaust max_iter WITHOUT reaching tol on near-separable soft
# designs (≥6 systems, large composite gaps). Ratings stay finite + monotone
# (so the result is serviceable), but a non-converged fit must be inspectable
# and honestly flagged — this engine feeds the SDO competitive-supremacy
# J-meter, where a silently non-converged fit would be undetectable.
# ---------------------------------------------------------------------------

# An easy, moderate-gap 3-system design (verified to converge well inside
# max_iter) and a near-separable ≥6-system design (verified to exhaust it).
_EASY_CONVERGENT_DESIGN = {"a": 0.6, "b": 0.5, "c": 0.4}
_NEAR_SEPARABLE_DESIGN = {f"s{idx}": float(idx) for idx in range(6)}


def test_code_quality_fit_exposes_convergence_diagnostics() -> None:
    """An easy moderate 3-system round-robin converges; the engine must expose
    that fact through INSPECTABLE read-only diagnostics — `last_fit_converged`
    True and `last_fit_iterations` strictly below max_iter."""
    engine = BradleyTerryMLEEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an easy design must NOT warn
        engine.run_round_robin(_EASY_CONVERGENT_DESIGN)
    assert engine.last_fit_converged is True
    assert isinstance(engine.last_fit_iterations, int)
    assert 0 < engine.last_fit_iterations < engine.max_iter


def test_code_quality_fit_paired_also_exposes_convergence_diagnostics() -> None:
    """The diagnostics are populated on the claim-grade `fit_paired` path too
    (not only the port path) — a real paired design converges and is flagged
    converged with iterations < max_iter."""
    engine = BradleyTerryMLEEngine()
    engine.fit_paired(
        {
            ("A", "B"): (8.0, 2.0, 10),
            ("B", "C"): (7.0, 3.0, 10),
            ("A", "C"): (9.0, 1.0, 10),
        }
    )
    assert engine.last_fit_converged is True
    assert 0 < engine.last_fit_iterations < engine.max_iter


def test_code_quality_near_separable_soft_design_is_flagged_not_silent() -> None:
    """A ≥6-system round-robin with large composite gaps exhausts max_iter; the
    fit must be HONESTLY flagged `last_fit_converged is False` (iterations hit
    max_iter) — yet the ratings remain finite + strictly monotone in composite
    order, so the result is still serviceable (not discarded, just flagged)."""
    engine = BradleyTerryMLEEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BradleyTerryConvergenceWarning)
        engine.run_round_robin(_NEAR_SEPARABLE_DESIGN)
    assert engine.last_fit_converged is False
    assert engine.last_fit_iterations == engine.max_iter
    # The flagged result is still usable: finite + strictly monotone.
    ratings = [
        engine.get_rating(name) for name in sorted(_NEAR_SEPARABLE_DESIGN)
    ]
    values = [r.rating for r in ratings if r is not None]
    assert len(values) == len(_NEAR_SEPARABLE_DESIGN)
    assert all(math.isfinite(v) for v in values)
    assert values == sorted(values)
    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


def test_code_quality_nonconvergence_emits_warning() -> None:
    """Non-convergence must SURFACE: a BradleyTerryConvergenceWarning fires on
    the near-separable design, and the message carries the diagnostic context
    (n_systems, final delta, max_iter)."""
    engine = BradleyTerryMLEEngine()
    with pytest.warns(BradleyTerryConvergenceWarning) as record:
        engine.run_round_robin(_NEAR_SEPARABLE_DESIGN)
    assert len(record) >= 1
    message = str(record[0].message)
    assert str(len(_NEAR_SEPARABLE_DESIGN)) in message  # n_systems
    assert str(engine.max_iter) in message  # max_iter


def test_code_quality_convergent_design_emits_no_warning() -> None:
    """The converse: an easy convergent design must NOT emit the convergence
    warning at all (recorded-warning list is empty of our warning type)."""
    engine = BradleyTerryMLEEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.run_round_robin(_EASY_CONVERGENT_DESIGN)
    convergence_warnings = [
        w for w in caught
        if issubclass(w.category, BradleyTerryConvergenceWarning)
    ]
    assert convergence_warnings == [], (
        "a convergent design must not emit a convergence warning"
    )


def test_code_quality_convergence_warning_is_a_user_warning() -> None:
    """The warning is a UserWarning subclass exported from the rating package
    (so callers can filter it precisely without importing the engine module)."""
    assert issubclass(BradleyTerryConvergenceWarning, UserWarning)
    import pii_anon.eval_framework.rating as rating_pkg

    assert "BradleyTerryConvergenceWarning" in rating_pkg.__all__
    assert rating_pkg.BradleyTerryConvergenceWarning is BradleyTerryConvergenceWarning


# ---------------------------------------------------------------------------
# nfr_002 / nfr_003: paired bootstrap — determinism, point∈CI, coherence
# ---------------------------------------------------------------------------

def _planted_records() -> list[tuple[str, str]]:
    """A strongly-ordered but NOT perfectly-separable record set (A>B>C)."""
    records: list[tuple[str, str]] = []
    records += [("A", "B")] * 40 + [("B", "A")] * 10
    records += [("B", "C")] * 38 + [("C", "B")] * 12
    records += [("A", "C")] * 45 + [("C", "A")] * 5
    return records


def test_nfr_002_paired_bootstrap_is_deterministic_for_fixed_seed() -> None:
    """Same records + same seed ⇒ byte-identical CIs across independent runs
    (no set/dict-order nondeterminism)."""
    engine = BradleyTerryMLEEngine()
    records = _planted_records()
    ci_a = engine.paired_bootstrap(records, b=200, seed=42)
    ci_b = engine.paired_bootstrap(records, b=200, seed=42)
    assert ci_a == ci_b


def test_nfr_002_paired_bootstrap_differs_across_seeds() -> None:
    """Different seeds should generally yield different resamples (guards
    against a constant CI that ignores the seed)."""
    engine = BradleyTerryMLEEngine()
    records = _planted_records()
    ci_1 = engine.paired_bootstrap(records, b=200, seed=1)
    ci_2 = engine.paired_bootstrap(records, b=200, seed=2)
    assert ci_1 != ci_2


def test_nfr_003_paired_bootstrap_point_estimate_lies_within_ci() -> None:
    """Monotone-consistency: the point MLE θ_i must lie inside its own 95% CI
    for every system."""
    engine = BradleyTerryMLEEngine()
    records = _planted_records()
    names = ["A", "B", "C"]
    point = engine.fit_records(records)
    ci = engine.paired_bootstrap(records, b=300, seed=7)
    for name in names:
        lo, hi = ci[name]
        assert lo <= point[name] <= hi, f"{name}: {point[name]} not in [{lo}, {hi}]"


def test_nfr_002_paired_bootstrap_rank_ci_coherence() -> None:
    """Coherence (sign↔verdict): if system X's CI strictly exceeds system Y's
    (lo_X > hi_Y), then X must be ranked above Y by the point estimate — the
    frequentist foundation of NFR-002. The converse implication is checked:
    a strict CI separation can never contradict the point ordering."""
    engine = BradleyTerryMLEEngine()
    records = _planted_records()
    point = engine.fit_records(records)
    ci = engine.paired_bootstrap(records, b=300, seed=11)
    names = list(point)
    for x in names:
        for y in names:
            if x == y:
                continue
            lo_x, _hi_x = ci[x]
            _lo_y, hi_y = ci[y]
            if lo_x > hi_y:  # CI of x strictly above CI of y
                assert point[x] > point[y], (
                    f"CI-coherence violated: {x} CI strictly > {y} CI "
                    f"but point[{x}]={point[x]} !> point[{y}]={point[y]}"
                )


def test_nfr_003_paired_bootstrap_separates_clearly_distinct_systems() -> None:
    """With a strongly-ordered design, the top system's CI should strictly
    exceed the bottom system's CI (a meaningful significance verdict exists)."""
    engine = BradleyTerryMLEEngine()
    records = _planted_records()
    ci = engine.paired_bootstrap(records, b=300, seed=3)
    lo_a, _ = ci["A"]
    _, hi_c = ci["C"]
    assert lo_a > hi_c, "A and C should be significantly separated"


# ---------------------------------------------------------------------------
# nfr_026: registry discovery lists BOTH tiers + resolves to the port
# ---------------------------------------------------------------------------

def test_nfr_026_discovery_lists_all_ladder_tiers_sorted() -> None:
    """Given the bradley-terry-mle entry point, when discovery runs after the
    editable install, then the registry lists all three installed ladder tiers
    sorted: ['bayes-bt', 'bradley-terry-mle', 'glicko-legacy']. (Updated in S3-03:
    the claim-grade ``bayes-bt`` tier is import-safe so it is discovered even
    without the ``bayes-eval`` extra installed.)"""
    registry = RatingEngineRegistry()
    discovered = registry.discover_entrypoint_engines(RATING_GROUP)
    assert "bradley-terry-mle" in discovered, (
        "entry point not found — run `pip install -e . --no-deps` after "
        "editing pyproject so entry_points metadata regenerates"
    )
    assert "glicko-legacy" in discovered
    assert sorted(discovered) == [
        "bayes-bt",
        "bradley-terry-mle",
        "glicko-legacy",
    ]


def test_nfr_026_discovered_bt_engine_resolves_to_the_port() -> None:
    """The discovered bradley-terry-mle engine must resolve to a concrete
    BradleyTerryMLEEngine that satisfies RatingEnginePort."""
    registry = RatingEngineRegistry()
    registry.discover_entrypoint_engines(RATING_GROUP)
    engine = registry.get("bradley-terry-mle")
    assert isinstance(engine, BradleyTerryMLEEngine)
    assert isinstance(engine, RatingEnginePort)


def test_nfr_026_additive_export_from_rating_package() -> None:
    """The engine is additively exported from the rating package __init__
    (does not disturb existing exports)."""
    import pii_anon.eval_framework.rating as rating_pkg

    assert "BradleyTerryMLEEngine" in rating_pkg.__all__
    assert rating_pkg.BradleyTerryMLEEngine is BradleyTerryMLEEngine
    # Pre-existing exports remain present (additive, not replacing).
    for kept in ("PIIRateEloEngine", "RatingEnginePort", "RatingEngineRegistry"):
        assert kept in rating_pkg.__all__


# ---------------------------------------------------------------------------
# ax_002: seeded determinism property (@given)
# ---------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    composites=st.dictionaries(
        keys=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6
        ),
        values=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        min_size=2,
        max_size=6,
    ),
)
def test_ax_002_run_round_robin_is_deterministic(composites: dict[str, float]) -> None:
    """Property: two independent engines fed identical composites produce
    byte-identical ratings — no seed/order nondeterminism in the port path.

    Some Hypothesis-drawn composite maps are near-separable and legitimately
    exhaust max_iter (flagged via BradleyTerryConvergenceWarning); that warning
    is orthogonal to the determinism contract under test and is suppressed here.
    The convergence DIAGNOSTICS are themselves asserted deterministic."""
    e1 = BradleyTerryMLEEngine()
    e2 = BradleyTerryMLEEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BradleyTerryConvergenceWarning)
        e1.run_round_robin(composites)
        e2.run_round_robin(composites)
    for name in composites:
        r1 = e1.get_rating(name)
        r2 = e2.get_rating(name)
        assert r1 is not None and r2 is not None
        assert r1.rating == r2.rating
        assert r1.rd == r2.rd
        assert math.isfinite(r1.rating)
    # Diagnostics are a pure function of the input too.
    assert e1.last_fit_converged == e2.last_fit_converged
    assert e1.last_fit_iterations == e2.last_fit_iterations


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_ax_002_paired_bootstrap_is_deterministic_per_seed(seed: int) -> None:
    """Property: paired_bootstrap is a pure function of (records, B, seed) —
    repeating with the same seed yields identical CIs for any seed."""
    engine = BradleyTerryMLEEngine()
    records = _planted_records()
    first = engine.paired_bootstrap(records, b=60, seed=seed)
    second = engine.paired_bootstrap(records, b=60, seed=seed)
    assert first == second


# ---------------------------------------------------------------------------
# S4-CS-01: paired_bootstrap_draws — raw (b, n_systems) θ rows for the SDO J.
# ---------------------------------------------------------------------------


def test_s4_paired_bootstrap_draws_shape_and_column_order() -> None:
    """[CONTRACT-TEST] paired_bootstrap_draws returns a (b, n_systems) float64
    array whose columns are the SORTED system names — the orientation
    rank_one_probability consumes directly."""
    import numpy as np

    engine = BradleyTerryMLEEngine()
    draws, names = engine.paired_bootstrap_draws(_planted_records(), 50, seed=3)
    assert names == ["A", "B", "C"]  # sorted
    assert draws.shape == (50, 3)
    assert draws.dtype == np.float64


def test_s4_paired_bootstrap_draws_is_deterministic_for_fixed_seed() -> None:
    """[PROPERTY-TEST] Same records + seed ⇒ byte-identical θ rows (AX-002)."""
    import numpy as np

    engine = BradleyTerryMLEEngine()
    d1, n1 = engine.paired_bootstrap_draws(_planted_records(), 40, seed=9)
    d2, n2 = engine.paired_bootstrap_draws(_planted_records(), 40, seed=9)
    assert n1 == n2
    assert np.array_equal(d1, d2)


def test_s4_paired_bootstrap_draws_feed_rank_one_probability() -> None:
    """[INTEGRATION-TEST] The draws feed rank_one_probability: the planted
    strongest system (A) wins rank-1 in (almost) every resample → J(A) high."""
    from pii_anon.eval_framework.rating.significance import rank_one_probability

    engine = BradleyTerryMLEEngine()
    draws, names = engine.paired_bootstrap_draws(_planted_records(), 200, seed=11)
    j_a = rank_one_probability(draws, names, "A")
    assert j_a > 0.9  # A is the planted winner


def test_s4_paired_bootstrap_draws_does_not_clobber_point_fit_diagnostics() -> None:
    """[PROPERTY-TEST] Like paired_bootstrap, the draws variant snapshots and
    restores the engine's point-fit convergence diagnostics."""
    engine = BradleyTerryMLEEngine()
    point = engine.fit_records(_planted_records())  # establishes diagnostics
    assert point  # non-empty
    saved = (engine.last_fit_converged, engine.last_fit_iterations)
    engine.paired_bootstrap_draws(_planted_records(), 30, seed=5)
    assert (engine.last_fit_converged, engine.last_fit_iterations) == saved
