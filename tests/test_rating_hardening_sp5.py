"""sp5 rating-stack hardening — fixes for the confirmed defect-hunt findings.

Findings fixed here (adversarially verified 2026-07-10, workflow wf_ea1ec403):

F1 (CATASTROPHIC) — the NFR-001 convergence gate was NaN/inf-BLIND: a
posterior containing any non-finite draw produced max_rhat=NaN, and both
threshold comparisons (`NaN > 1.01`, `NaN < 400`) are False, so
claim_grade=True with binding_constraint='' — silently. Downstream,
rank_one_probability on an all-NaN posterior returned J=1.0 for the
first-listed system: a gate-passing NaN posterior fabricates a perfect
claim-grade J. The gate must treat non-finite draws as a first-class
binding constraint.

F3 (MAJOR) — the claim-grade bayes-bt NUTS model used a CENTERED
hierarchical parameterization (sigma ~ HalfNormal, theta_raw ~ N(0, sigma)),
which funnels as sigma -> 0 on tied/near-tied designs: 50/50 tie -> 80
divergences, Davidson tie-heavy -> 158 at default config, so the ONLY
claim-grade tier was structurally unavailable exactly on the program's
core-vs-swarm near-tie shape. Non-centered (theta = sigma * z, z ~ N(0,1))
+ target_accept >= 0.95 drops divergences to ~0.

F4/F5 (MAJOR) — assessment ingestion never reconciled the shared-gold
invariant: players reporting contradictory per-type gold (tp+fn) for the
same dataset, or phantom types only one player asserts, or a scored player
with an EMPTY by_entity_type, were all admitted and could move rankings.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pii_anon.eval_framework.rating.convergence import (
    ConvergenceError,
    ConvergenceReport,
    assert_claim_grade,
)

# ---------------------------------------------------------------------------
# F1 — non-finite draws must fail the gate loud
# ---------------------------------------------------------------------------


class TestNonFiniteGate:
    def _good_chains(self) -> np.ndarray:
        rng = np.random.default_rng(20260710)
        return rng.normal(size=(4, 500, 2))

    def test_all_nan_posterior_is_not_claim_grade(self) -> None:
        report = ConvergenceReport.from_samples(np.full((4, 500, 2), np.nan))
        assert report.claim_grade is False
        assert "finite" in report.binding_constraint.lower()
        with pytest.raises(ConvergenceError):
            assert_claim_grade(report)

    def test_single_nan_draw_is_not_claim_grade(self) -> None:
        samples = self._good_chains()
        samples[1, 200, 0] = np.nan
        report = ConvergenceReport.from_samples(samples)
        assert report.claim_grade is False
        with pytest.raises(ConvergenceError):
            assert_claim_grade(report)

    def test_single_inf_draw_is_not_claim_grade(self) -> None:
        samples = self._good_chains()
        samples[2, 10, 1] = np.inf
        report = ConvergenceReport.from_samples(samples)
        assert report.claim_grade is False

    def test_finite_healthy_chains_still_pass(self) -> None:
        report = ConvergenceReport.from_samples(self._good_chains())
        assert report.claim_grade is True
        assert report.binding_constraint == ""


# ---------------------------------------------------------------------------
# F3 — tied/near-tied designs must fit claim-grade (non-centered model)
# ---------------------------------------------------------------------------


class TestTiedDesignConvergence:
    @pytest.mark.parametrize(
        "counts",
        [
            {("a", "b"): (50, 50, 100)},
            {("a", "b"): (52, 48, 100)},
        ],
        ids=["exact-tie", "near-tie"],
    )
    def test_two_system_tied_designs_fit_claim_grade(self, counts) -> None:
        pytest.importorskip("numpyro")
        from pii_anon.eval_framework.rating.bayes_bt import BayesBTEngine

        engine = BayesBTEngine()
        posterior = engine.fit_paired_posterior(counts)  # raises pre-fix
        assert posterior.convergence.claim_grade is True

    def test_tie_heavy_davidson_design_fits_claim_grade(self) -> None:
        pytest.importorskip("numpyro")
        from pii_anon.eval_framework.rating.bayes_bt import BayesBTEngine

        engine = BayesBTEngine()
        posterior = engine.fit_paired_posterior_with_ties(
            {("a", "b"): (10, 10, 80, 100)}
        )
        assert posterior.convergence.claim_grade is True


# ---------------------------------------------------------------------------
# F4/F5 — shared-gold reconciliation + empty-player guard
# ---------------------------------------------------------------------------


def _artifact(detectors: dict) -> dict:
    return {
        "schema": "pii-anon-baseline-results/v1",
        "matching_policy": "strict-v1",
        "dataset": {"split": "test", "language": "en", "n_records": 10, "n_gold": 30},
        "detectors": detectors,
    }


def _implied_f2(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 5.0 * p * r / (4.0 * p + r) if (4.0 * p + r) > 0.0 else 0.0


def _player(
    per_type: dict[str, tuple[int, int, float]], *, n_gold: int | None = None
) -> dict:
    # The 3rd tuple slot is legacy (a free-form f2); the counts-consistency
    # check (sp5 close) requires f2 == the F2 the counts imply, so it is
    # DERIVED here and the slot ignored.
    by_entity = {
        t: {"counts": {"tp": tp, "fn": fn, "fp": 0}, "f2": _implied_f2(tp, 0, fn)}
        for t, (tp, fn, _f2) in per_type.items()
    }
    gold_total = sum(tp + fn for tp, fn, _ in per_type.values())
    return {
        "status": "scored",
        "micro": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "f2": 0.9},
        "macro": {"f2": 0.5},
        "coverage": {"reachable": max(len(per_type), 1), "of_total": 66},
        "by_entity_type": by_entity,
        "n_gold": gold_total if n_gold is None else n_gold,
        "n_pred": gold_total,
    }


class TestSharedGoldInvariant:
    def test_contradictory_gold_counts_are_rejected(self, tmp_path) -> None:
        from pii_anon.eval_framework.rating.assessment_ingest import load_assessment

        art = _artifact({
            "a": _player({"EMAIL": (18, 2, 0.9)}),   # gold=20
            "b": _player({"EMAIL": (1, 0, 1.0)}),    # gold=1 — contradicts
        })
        p = tmp_path / "art.json"
        p.write_text(json.dumps(art))
        with pytest.raises(ValueError, match="gold"):
            load_assessment(p)

    def test_consistent_gold_counts_load(self, tmp_path) -> None:
        from pii_anon.eval_framework.rating.assessment_ingest import load_assessment

        art = _artifact({
            "a": _player({"EMAIL": (18, 2, 0.9)}),   # gold=20
            "b": _player({"EMAIL": (5, 15, 0.4)}),   # gold=20 — consistent
        })
        p = tmp_path / "art.json"
        p.write_text(json.dumps(art))
        report = load_assessment(p)
        assert len(report.players) == 2

    def test_scored_player_with_no_gold_supported_types_is_rejected(self, tmp_path) -> None:
        from pii_anon.eval_framework.rating.assessment_ingest import load_assessment

        art = _artifact({
            "a": _player({"EMAIL": (18, 2, 0.9)}),
            "b": _player({}),                        # empty by_entity_type
        })
        p = tmp_path / "art.json"
        p.write_text(json.dumps(art))
        with pytest.raises(ValueError, match="gold-supported"):
            load_assessment(p)


# ---------------------------------------------------------------------------
# F2 — rank-1 tie mass must be split, not awarded to the first column
# ---------------------------------------------------------------------------


class TestRankOneTieSplit:
    """F2 (MAJOR): ``rank_one_distribution`` awarded ALL exact-tie draws to the
    alphabetically-first column (``np.argmax`` first-index tie-break), so the
    J primitive was not invariant under system relabeling — and 'pii-anon'
    sorts before 'pii-anon-swarm', biasing the actively-tracked J race in the
    fabrication direction. Tie mass must be split fractionally; non-tied draws
    are untouched (measured: 0/400 tied draws on the committed canonical
    artifact ⇒ today's binding J=0.2775 stays byte-identical).
    """

    def test_exact_tie_posterior_splits_mass_equally(self) -> None:
        from pii_anon.eval_framework.rating.significance import rank_one_distribution

        dist = rank_one_distribution(np.zeros((1000, 2)), ["alpha", "zeta"])
        assert dist["alpha"] == pytest.approx(0.5)
        assert dist["zeta"] == pytest.approx(0.5)

    def test_distribution_is_relabeling_invariant(self) -> None:
        from pii_anon.eval_framework.rating.significance import rank_one_distribution

        rng = np.random.default_rng(20260710)
        arr = rng.normal(size=(4000, 3))
        arr[::5, 0] = arr[::5, 1]  # inject exact two-way ties on 20% of draws
        d1 = rank_one_distribution(arr, ["a", "b", "c"])
        d2 = rank_one_distribution(arr[:, [1, 0, 2]], ["b", "a", "c"])
        for name in ("a", "b", "c"):
            assert d1[name] == pytest.approx(d2[name]), (
                f"rank-1 mass for {name} changed under relabeling: "
                f"{d1[name]} vs {d2[name]}"
            )

    def test_three_way_tie_splits_thirds_and_sums_to_one(self) -> None:
        from pii_anon.eval_framework.rating.significance import rank_one_distribution

        dist = rank_one_distribution(np.zeros((300, 3)), ["a", "b", "c"])
        for share in dist.values():
            assert share == pytest.approx(1 / 3)
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_untied_posterior_unchanged(self) -> None:
        from pii_anon.eval_framework.rating.significance import rank_one_distribution

        rng = np.random.default_rng(7)
        arr = rng.normal(size=(2000, 4))  # continuous — ties have measure zero
        dist = rank_one_distribution(arr, ["a", "b", "c", "d"])
        winners = np.argmax(arr, axis=1)
        counts = np.bincount(winners, minlength=4)
        for k, name in enumerate(["a", "b", "c", "d"]):
            assert dist[name] == pytest.approx(counts[k] / arr.shape[0])


# ---------------------------------------------------------------------------
# sp5 CLOSE remediations (wf_f5c661f6 upheld findings — regression pins)
# ---------------------------------------------------------------------------


class TestCloseRemediations:
    """Pins for the adversarial-close findings on the sp5 diffs themselves."""

    def test_nan_poisoned_posterior_is_refused_by_the_j_primitives(self) -> None:
        """CLOSE MAJOR (J-fabrication): one NaN draw made rank_one_distribution
        emit all-NaN shares, and downstream ``NaN < J_BAR`` is False — the J
        bar silently VACATED and a claim-grade verdict was forged. The shared
        validator must refuse non-finite draws loudly."""
        from pii_anon.eval_framework.rating.significance import (
            pairwise_significance,
            rank_one_distribution,
            rank_one_probability,
        )

        rng = np.random.default_rng(0)
        arr = rng.normal(size=(400, 4))
        arr[7, 3] = np.nan
        names = ["gliner", "pii-anon", "pii-anon-swarm", "presidio"]
        with pytest.raises(ValueError, match="non-finite"):
            rank_one_distribution(arr, names)
        with pytest.raises(ValueError, match="non-finite"):
            rank_one_probability(arr, names, "pii-anon")
        with pytest.raises(ValueError, match="non-finite"):
            pairwise_significance(arr, names)

    def test_overflow_to_nan_diagnostics_fail_the_gate(self) -> None:
        """CLOSE MAJOR (gate evasion): all-FINITE draws of magnitude ~1e308
        overflow the within-chain variance, making rhat/ESS NaN — which passed
        the input-finiteness pre-check and then bypassed both comparisons."""
        from pii_anon.eval_framework.rating.convergence import (
            ConvergenceError,
            ConvergenceReport,
            assert_claim_grade,
        )

        s = np.empty((2, 1000, 1))
        s[0, :, 0] = np.tile([1e308, -1e308], 500)
        s[1, :, 0] = -s[0, :, 0]
        assert np.isfinite(s).all()
        report = ConvergenceReport.from_samples(s)
        assert report.claim_grade is False
        assert "non-finite diagnostics" in report.binding_constraint
        with pytest.raises(ConvergenceError):
            assert_claim_grade(report)

    def test_subnormal_unmixed_chains_fail_the_gate(self) -> None:
        """CLOSE MINOR: chains parked at DIFFERENT subnormal constants had
        both variances underflow to 0 and read rhat=1.0 (converged)."""
        from pii_anon.eval_framework.rating.convergence import ConvergenceReport

        s = np.empty((2, 1000, 1))
        s[0, :, 0] = 1e-320
        s[1, :, 0] = 9e-320
        report = ConvergenceReport.from_samples(s)
        assert report.claim_grade is False

    def test_identical_constant_chains_still_read_converged(self) -> None:
        """Genuinely-identical constant chains keep the rhat=1.0 branch."""
        from pii_anon.eval_framework.rating.convergence import ConvergenceReport

        s = np.full((4, 500, 2), 0.25)
        report = ConvergenceReport.from_samples(s)
        assert np.isclose(report.max_rhat, 1.0)

    def test_nonfinite_failure_report_n_params_matches_healthy_report(self) -> None:
        """CLOSE MINOR: the non-finite early return misreported n_params for
        2-D input (raw shape[-1] instead of the normalized param count)."""
        from pii_anon.eval_framework.rating.convergence import ConvergenceReport

        healthy = np.full((4, 1000), 0.5)
        n_healthy = ConvergenceReport.from_samples(healthy).n_params
        poisoned = healthy.copy()
        poisoned[0, 0] = np.nan
        assert ConvergenceReport.from_samples(poisoned).n_params == n_healthy

    def test_negative_divergence_count_is_refused(self) -> None:
        """CLOSE MINOR: -5 > MAX_DIVERGENCES is False — a negative count
        silently neutralized the divergence veto."""
        from pii_anon.eval_framework.rating.convergence import ConvergenceReport

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="negative"):
            ConvergenceReport.from_samples(
                rng.standard_normal((4, 1000, 3)), n_divergences=-5
            )

    def test_counts_impossible_f2_is_rejected(self, tmp_path) -> None:
        """CLOSE MAJOR (assessment surface): a gold-consistent row whose
        reported F2 contradicts its own tp/fp/fn moved rankings."""
        from pii_anon.eval_framework.rating.assessment_ingest import load_assessment

        good = _player({"EMAIL": (18, 2, 0.0)})
        forged = _player({"EMAIL": (5, 15, 0.0)})
        forged["by_entity_type"]["EMAIL"]["f2"] = 0.99  # counts imply ~0.29
        art = _artifact({"a": good, "b": forged})
        p = tmp_path / "art.json"
        p.write_text(json.dumps(art))
        with pytest.raises(ValueError, match="counts-impossible"):
            load_assessment(p)

    def test_colluding_phantom_type_refuted_by_n_gold(self, tmp_path) -> None:
        """CLOSE MINOR (defense-in-depth): players colluding on a phantom type
        are refuted by the artifact's OWN totals (sum(tp+fn) != n_gold)."""
        from pii_anon.eval_framework.rating.assessment_ingest import load_assessment

        a = _player({"EMAIL": (18, 2, 0.0), "PHANTOM": (5, 0, 0.0)}, n_gold=20)
        b = _player({"EMAIL": (5, 15, 0.0), "PHANTOM": (5, 0, 0.0)}, n_gold=20)
        art = _artifact({"a": a, "b": b})
        p = tmp_path / "art.json"
        p.write_text(json.dumps(art))
        with pytest.raises(ValueError, match="phantom or truncated"):
            load_assessment(p)

    def test_tournament_entry_reruns_reconciliation(self) -> None:
        """CLOSE documented seam: a hand-built AssessmentReport bypassed the
        load-path invariant; run_assessment_tournament re-checks at entry."""
        from pii_anon.eval_framework.rating.assessment_ingest import (
            AssessmentPlayer,
            AssessmentReport,
            run_assessment_tournament,
        )

        def mk(name: str, gold: int) -> AssessmentPlayer:
            return AssessmentPlayer(
                name=name, precision=0.9, recall=0.9, f1=0.9, f2=0.9,
                f2_macro=0.5, coverage_reachable=1, coverage_total=66,
                n_gold=gold, n_pred=gold,
                per_entity_f2={"EMAIL": 0.9}, per_entity_gold={"EMAIL": gold},
            )

        report = AssessmentReport(
            players=(mk("a", 20), mk("b", 1)),  # contradictory shared gold
            excluded=(), dataset={}, matching_policy="strict-v1",
            source_path="hand-built",
        )
        with pytest.raises(ValueError, match="gold"):
            run_assessment_tournament(report)

    def test_tournament_rejects_phantom_f2_key_and_nan_f2(self) -> None:
        """ROUND-2 close upheld MINOR: the entry reconciliation checked gold
        only, so a hand-built report with a phantom per_entity_f2 key gained
        +73 Elo on free 1.0-vs-0.0 wins, and a NaN f2 was silently converted
        into guaranteed wins by the elo sigmoid clamp."""
        from pii_anon.eval_framework.rating.assessment_ingest import (
            AssessmentPlayer,
            AssessmentReport,
            run_assessment_tournament,
        )

        def mk(name: str, f2_map: dict, gold_map: dict) -> AssessmentPlayer:
            return AssessmentPlayer(
                name=name, precision=0.9, recall=0.9, f1=0.9, f2=0.9,
                f2_macro=0.5, coverage_reachable=1, coverage_total=66,
                n_gold=sum(gold_map.values()), n_pred=10,
                per_entity_f2=f2_map, per_entity_gold=gold_map,
            )

        honest = mk("a", {"EMAIL": 0.9}, {"EMAIL": 20})
        phantom = mk("b", {"EMAIL": 0.5, "ZZ_PHANTOM": 1.0}, {"EMAIL": 20})
        report = AssessmentReport(
            players=(honest, phantom), excluded=(), dataset={},
            matching_policy="strict-v1", source_path="hand-built",
        )
        with pytest.raises(ValueError, match="phantom f2 fields"):
            run_assessment_tournament(report)

        nan_player = mk("b", {"EMAIL": float("nan")}, {"EMAIL": 20})
        report2 = AssessmentReport(
            players=(honest, nan_player), excluded=(), dataset={},
            matching_policy="strict-v1", source_path="hand-built",
        )
        with pytest.raises(ValueError, match="finite match score"):
            run_assessment_tournament(report2)

    def test_tournament_rejects_zero_gold_colluded_phantom(self) -> None:
        """ROUND-3 close upheld MINOR: a gold=0 phantom colluded into BOTH
        dicts of ALL players kept every sum intact and moved rankings ±75
        Elo. Gold-supported means tp+fn >= 1 — enforce it on the values."""
        from pii_anon.eval_framework.rating.assessment_ingest import (
            AssessmentPlayer,
            AssessmentReport,
            run_assessment_tournament,
        )

        def mk(name: str, f2: float) -> AssessmentPlayer:
            return AssessmentPlayer(
                name=name, precision=0.9, recall=0.9, f1=0.9, f2=0.9,
                f2_macro=0.5, coverage_reachable=1, coverage_total=66,
                n_gold=20, n_pred=20,
                per_entity_f2={"EMAIL": 0.9, "ZZ_COLLUDE": f2},
                per_entity_gold={"EMAIL": 20, "ZZ_COLLUDE": 0},
            )

        report = AssessmentReport(
            players=(mk("a", 0.0), mk("b", 1.0)), excluded=(), dataset={},
            matching_policy="strict-v1", source_path="hand-built",
        )
        with pytest.raises(ValueError, match="tp\\+fn >= 1"):
            run_assessment_tournament(report)
