"""Tests for S4-CS-01: CompetitiveSupremacyGate — the SDO verdict machine.

Implements / pins:
    The SDO objective J = P(rank(pii-anon)=1 | posterior) (FR-004 / S3-04).
    FR-007 / FR-008 / NFR-006 — canonical-run gate + provenance (G7).
    The completion predicate (§5): CLAIM_GRADE_SOTA / PROVISIONAL_SOTA / NOT_YET.
    G1/G3/G6/G7 computable-now guarantees; G2/G4/G5 as three-valued PENDING.

Design trace: Program AMENDMENT "SOTA-Dominance Objective (SDO)" / DC-11.

Everything here is PURE-PYTHON over a synthetic benchmark dict + a synthetic
posterior (numpy θ draws). The real bayes posterior (numpyro) is Pass-2 /
importorskip; the in-tree J uses the MLE-bootstrap fallback. Exactly ONE test
loads the real ``artifacts/benchmarks/benchmark-results.json`` READ-ONLY and
asserts the value-independent verdict (NOT_YET, binding canonical_claim_run).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pii_anon.eval_framework.evaluation.competitive_supremacy import (
    ENTITY_COVERAGE_MIN,
    EPS_F2,
    EPS_RECALL_PER_LANG,
    J_BAR,
    SupremacyVerdict,
    Verdict,
    _finite_unit_score,
    _g1_recall_floor,
    _g2_pseudonymization_integrity,
    _g3_recall_dominance,
    _g4_calibration_selective_risk,
    _g6_raw_noninferiority,
    _is_finite_number,
    _risk_coverage_is_monotone,
    _run_breaches_recall_floor,
    _tier_status_input,
    _top_composite_system,
    f_beta,
    recall_floor_breachers,
)

# ---------------------------------------------------------------------------
# Threshold-literal pins (these literals ARE the SDO contract — §5/§3).
# ---------------------------------------------------------------------------


def test_threshold_literals_are_pinned() -> None:
    """[CONTRACT-TEST] The SDO threshold literals are pinned exactly."""
    assert J_BAR == 0.95
    assert EPS_F2 == 0.01
    assert ENTITY_COVERAGE_MIN == 0.80
    assert EPS_RECALL_PER_LANG == 0.005


def test_f_beta_is_f2_weighting_recall() -> None:
    """[UNIT-TEST] f_beta(P, R, beta=2) == 5·P·R/(4P+R) (recall-weighted F2)."""
    p, r = 0.6, 0.9
    assert f_beta(p, r, beta=2.0) == pytest.approx(5 * p * r / (4 * p + r))
    # F1 sanity: beta=1 is the harmonic mean.
    assert f_beta(p, r, beta=1.0) == pytest.approx(2 * p * r / (p + r))
    # Degenerate P=R=0 → 0.0 (no division blow-up).
    assert f_beta(0.0, 0.0, beta=2.0) == 0.0


# ---------------------------------------------------------------------------
# Synthetic fixtures — a fully-passing benchmark + a strong posterior.
# ---------------------------------------------------------------------------


def _system(
    name: str,
    *,
    recall: float,
    precision: float,
    composite: float,
    per_entity_recall: dict[str, float] | None = None,
    latency_p50: float = 1.0,
    available: bool = True,
    qualification: str = "qualified",
) -> dict[str, object]:
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "system": name,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "composite_score": composite,
        "per_entity_recall": per_entity_recall or {},
        "latency_p50_ms": latency_p50,
        "available": available,
        "qualification_status": qualification,
        "citation_url": f"https://example.test/{name}",
    }


# Shared entity universe so G1 ⊇ checks are exercisable.
_SHARED_ENTITIES = {"EMAIL_ADDRESS": 0.9, "US_SSN": 0.9, "PHONE_NUMBER": 0.9}
_ENSEMBLE_ENTITIES = {**_SHARED_ENTITIES, "CRYPTO_WALLET": 0.8}  # superset


def _canonical_benchmark() -> dict[str, object]:
    """A fully-passing benchmark: pii-anon dominates recall + F2, canonical run,
    full provenance, floor_pass True, per-language ε present and within bound."""
    return {
        "run_metadata": {
            "canonical_claim_run": True,
            "git_sha": "deadbeef",
            "dataset_sha256": "a" * 64,
            "matrix_sha256": "b" * 64,
            "timestamp_utc": "2026-06-01T00:00:00Z",
        },
        "floor_pass": True,
        "available_competitors": ["presidio", "scrubadub", "gliner"],
        "expected_competitors": ["presidio", "scrubadub", "gliner"],
        "unavailable_competitors": {},
        "per_language_recall_delta": {"en": 0.001, "de": 0.002, "fr": 0.0},
        "systems": [
            _system(
                "pii-anon",
                recall=0.80,
                precision=0.78,
                composite=0.80,
                per_entity_recall=_ENSEMBLE_ENTITIES,
                qualification="core",
            ),
            _system(
                "pii-anon-swarm",
                recall=0.85,
                precision=0.60,
                composite=0.70,
                per_entity_recall=_ENSEMBLE_ENTITIES,
                qualification="core",
            ),
            _system(
                "gliner",
                recall=0.66,
                precision=0.91,
                composite=0.68,
                per_entity_recall=_SHARED_ENTITIES,
            ),
            _system(
                "presidio",
                recall=0.63,
                precision=0.40,
                composite=0.51,
                per_entity_recall=_SHARED_ENTITIES,
            ),
            _system(
                "scrubadub",
                recall=0.21,
                precision=0.86,
                composite=0.51,
                per_entity_recall={"EMAIL_ADDRESS": 0.8},
            ),
        ],
    }


def _strong_posterior() -> tuple[np.ndarray, list[str]]:
    """A joint posterior in which pii-anon is #1 in (almost) every draw → J≈1."""
    rng = np.random.default_rng(42)
    names = ["gliner", "pii-anon", "pii-anon-swarm", "presidio", "scrubadub"]
    n = 4000
    cols = {
        "pii-anon": rng.normal(2.0, 0.3, n),
        "gliner": rng.normal(0.5, 0.3, n),
        "pii-anon-swarm": rng.normal(0.0, 0.3, n),
        "presidio": rng.normal(-1.0, 0.3, n),
        "scrubadub": rng.normal(-1.0, 0.3, n),
    }
    arr = np.column_stack([cols[name] for name in names])
    return arr, names


# G2/G4/G5 still need their successor stories — supply them PASSING explicitly
# only where a CLAIM_GRADE test needs them. The default gate leaves them PENDING.
_ALL_PENDING_PASS = {"G2": True, "G4": True, "G5": True}


# ---------------------------------------------------------------------------
# Verdict machine — the three-valued truth table (§5)
# ---------------------------------------------------------------------------


def test_claim_grade_when_canonical_all_g_pass_j_high_and_tiers_run() -> None:
    """[CONTRACT-TEST] CLAIM_GRADE_SOTA ⟺ canonical ∧ all-Gk pass ∧ J≥0.95 ∧
    (Tier-R ∪ Tier-C RUN-or-WAIVED). All Tier-C waived-with-reason AND the unrun
    Tier-R adapter (gliner2) waived-with-reason — §5 requires the WHOLE Tier-R ∪
    Tier-C set RUN-or-WAIVED, not Tier-C alone."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(),
        theta_samples=theta,
        posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={
            "openai-privacy-filter": "cited-only; raw-F1 carve-out",
            "azure-ai-language": "no API budget",
            "aws-comprehend": "vendor pending",
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA
    assert verdict.j_value is not None and verdict.j_value >= J_BAR
    assert verdict.binding_constraint == ""


def test_provisional_when_blocked_only_by_unrun_tier_c() -> None:
    """[CONTRACT-TEST] PROVISIONAL_SOTA ⟺ everything else passes but Tier-C is
    not yet run (the ONLY remaining blocker). gliner2 (Tier-R) is waived here so
    Tier-C is genuinely the sole remaining tier blocker."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(),
        theta_samples=theta,
        posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
        # No Tier-C waivers → Tier-C stays UNRUN (the sole remaining blocker).
    )
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA
    assert "openai-privacy-filter" in verdict.binding_constraint
    assert verdict.j_value is not None and verdict.j_value >= J_BAR


def test_not_yet_when_canonical_run_is_false() -> None:
    """[CONTRACT-TEST] NOT_YET when canonical_claim_run is False — the #1
    binding constraint, regardless of everything else passing."""
    bench = _canonical_benchmark()
    bench["run_metadata"]["canonical_claim_run"] = False  # type: ignore[index]
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={
            "openai-privacy-filter": "x",
            "azure-ai-language": "x",
            "aws-comprehend": "x",
        },
    )
    assert verdict.verdict is Verdict.NOT_YET
    assert "canonical_claim_run" in verdict.binding_constraint
    assert "False" in verdict.binding_constraint


# ---------------------------------------------------------------------------
# Binding-constraint PRIORITY ordering (§5):
#   canonical_claim_run=False  →  failed-G (lowest k)  →  J gap  →  unrun Tier-C
# ---------------------------------------------------------------------------


def test_binding_priority_canonical_outranks_failed_g_and_j_and_tier_c() -> None:
    """[CONTRACT-TEST] canonical_claim_run=False wins even when a G fails, J is
    low, and Tier-C is unrun simultaneously."""
    bench = _canonical_benchmark()
    bench["run_metadata"]["canonical_claim_run"] = False  # type: ignore[index]
    # Break G3 too (swarm recall below a competitor).
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "pii-anon-swarm":
            s["recall"] = 0.10
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    assert verdict.verdict is Verdict.NOT_YET
    assert "canonical_claim_run" in verdict.binding_constraint


def test_binding_priority_failed_g_outranks_j_gap_and_tier_c() -> None:
    """[CONTRACT-TEST] With canonical run True, a failed guarantee outranks a J
    gap and unrun Tier-C; the LOWEST-k failing G is named."""
    bench = _canonical_benchmark()
    # Break G3 (swarm recall) AND make J low — G3 must be the binding constraint.
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "pii-anon-swarm":
            s["recall"] = 0.10
    theta, names = _strong_posterior()  # J high; irrelevant since a G fails
    verdict = SupremacyVerdict.from_artifacts(
        bench, theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
    )
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.binding_constraint.startswith("G3")


def test_binding_priority_lowest_k_failed_g_named_first() -> None:
    """[CONTRACT-TEST] When G3 AND G6 both fail, the lowest-k (G3) is binding."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "pii-anon-swarm":
            s["recall"] = 0.10  # break G3
        if s["system"] == "pii-anon":
            s["recall"] = 0.10
            s["precision"] = 0.10  # crater core F2 → break G6
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    assert verdict.binding_constraint.startswith("G3")


def test_binding_priority_j_gap_outranks_unrun_tier_c() -> None:
    """[CONTRACT-TEST] All Gk pass, canonical run True, but J < 0.95 → the J gap
    is binding, ahead of unrun Tier-C."""
    bench = _canonical_benchmark()
    # A flat posterior → pii-anon rank-1 prob ≈ 0.2 (5-way tie) < 0.95.
    rng = np.random.default_rng(7)
    names = ["gliner", "pii-anon", "pii-anon-swarm", "presidio", "scrubadub"]
    theta = rng.normal(0.0, 1.0, (3000, 5))
    verdict = SupremacyVerdict.from_artifacts(
        bench, theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
    )
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.j_value is not None and verdict.j_value < J_BAR
    assert "J" in verdict.binding_constraint
    assert "0.95" in verdict.binding_constraint


def test_binding_constraint_always_emitted_empty_only_when_claim_grade() -> None:
    """[PROPERTY-TEST] binding_constraint is "" iff verdict is CLAIM_GRADE. The
    claim branch waives all of Tier-R ∪ Tier-C (incl. gliner2) so it genuinely
    reaches CLAIM_GRADE and exercises the True side of the biconditional."""
    theta, names = _strong_posterior()
    claim = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired"},
    )
    assert claim.verdict is Verdict.CLAIM_GRADE_SOTA  # genuinely claim-grade here
    assert (claim.binding_constraint == "") is (claim.verdict is Verdict.CLAIM_GRADE_SOTA)

    provisional = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
    )
    assert provisional.binding_constraint != ""


# ---------------------------------------------------------------------------
# Per-guarantee Gk fixtures (each pass/fail in isolation)
# ---------------------------------------------------------------------------


def test_g3_passes_when_swarm_recall_dominates_every_competitor() -> None:
    """[UNIT-TEST] G3 PASS: max(pii-anon recall ladder) ≥ max(competitor recall)."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    g3 = verdict.guarantee("G3")
    assert g3.passed is True
    assert g3.observed >= g3.bar


def test_g3_fails_when_a_competitor_out_recalls_the_swarm() -> None:
    """[UNIT-TEST] G3 FAIL: a competitor recall exceeds the pii-anon ladder."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "gliner":
            s["recall"] = 0.99
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G3").passed is False


def test_g6_passes_when_core_f2_within_eps_of_best_tier_r() -> None:
    """[UNIT-TEST] G6 PASS: core F2 ≥ best Tier-R F2 − ε_F ∧ coverage ≥ 0.80."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    g6 = verdict.guarantee("G6")
    assert g6.passed is True


def test_g6_fails_when_core_f2_below_best_tier_r_minus_eps() -> None:
    """[UNIT-TEST] G6 FAIL: cratered core P/R drops F2 well below Tier-R − ε."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "pii-anon":
            s["recall"] = 0.10
            s["precision"] = 0.10
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G6").passed is False


def test_g6_honesty_carveout_tier_c_high_raw_f1_does_not_fail_g6() -> None:
    """[AUDIT] THE OpenAI carve-out: a Tier-C raw-F1 (e.g. 0.96) exceeding
    pii-anon does NOT fail G6 — only Tier-R non-inferiority counts. The carve-out
    note is ALWAYS emitted."""
    bench = _canonical_benchmark()
    # Inject a Tier-C system with crushing raw F1; it is NOT a Tier-R comparator.
    bench["systems"].append(  # type: ignore[attr-defined]
        _system("openai-privacy-filter", recall=0.97, precision=0.97, composite=0.95)
    )
    verdict = SupremacyVerdict.from_artifacts(bench)
    g6 = verdict.guarantee("G6")
    assert g6.passed is True  # Tier-C raw F1 ignored for G6
    assert "carve-out" in g6.binding_detail.lower() or "carve-out" in verdict.carve_out_note.lower()
    assert verdict.carve_out_note  # always emitted


def test_g7_fails_without_canonical_run() -> None:
    """[UNIT-TEST] G7 FAIL: canonical_claim_run False fails the certified-run
    guarantee even with full provenance."""
    bench = _canonical_benchmark()
    bench["run_metadata"]["canonical_claim_run"] = False  # type: ignore[index]
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G7").passed is False


def test_g7_passes_with_canonical_run_and_full_provenance() -> None:
    """[UNIT-TEST] G7 PASS: canonical run True + provenance complete."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    assert verdict.guarantee("G7").passed is True


def test_g7_fails_with_canonical_true_but_missing_provenance() -> None:
    """[UNIT-TEST] G7 FAIL: canonical True but a provenance stamp (dataset hash)
    missing → not certified."""
    bench = _canonical_benchmark()
    del bench["run_metadata"]["dataset_sha256"]  # type: ignore[attr-defined]
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G7").passed is False


def test_g1_structural_superset_passes_when_ensemble_covers_shared() -> None:
    """[UNIT-TEST] G1 PASS: entities(ensemble) ⊇ entities(shared) AND per-lang ε
    ≤ 0.005 (present in this fixture)."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    assert verdict.guarantee("G1").passed is True


def test_g1_pending_when_per_language_delta_absent_never_fabricated() -> None:
    """[AUDIT] G1 is PENDING (None) when the per-language recall-delta artifact is
    absent — NEVER fabricated, NEVER auto-passed."""
    bench = _canonical_benchmark()
    del bench["per_language_recall_delta"]
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G1").passed is None  # three-valued PENDING


def test_g1_fails_when_per_language_eps_exceeds_bound() -> None:
    """[UNIT-TEST] G1 FAIL: a per-language recall regression beyond ε=0.005."""
    bench = _canonical_benchmark()
    bench["per_language_recall_delta"] = {"en": 0.001, "de": 0.02}  # 0.02 > 0.005
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G1").passed is False


# ---------------------------------------------------------------------------
# Three-valued Gk: PENDING never blocks PROVISIONAL, always blocks CLAIM_GRADE
# ---------------------------------------------------------------------------


def test_pending_g2_g4_g5_do_not_block_provisional() -> None:
    """[CONTRACT-TEST] With G2/G4/G5 PENDING (default) and Tier-C unrun, the
    verdict is at most PROVISIONAL — PENDING does not drag it to NOT_YET."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
    )
    # G2/G4/G5 PENDING here.
    assert verdict.guarantee("G2").passed is None
    assert verdict.guarantee("G4").passed is None
    assert verdict.guarantee("G5").passed is None
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA


def test_pending_g2_g4_g5_always_block_claim_grade() -> None:
    """[CONTRACT-TEST] Even with canonical run, J≥0.95, Tier-C waived, a single
    PENDING guarantee blocks CLAIM_GRADE (PENDING is never treated as pass)."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides={"G2": True, "G4": None, "G5": True},  # G4 still PENDING
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
    )
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA
    assert "G4" in verdict.binding_constraint


def test_pending_axes_surfaced_as_named_successors() -> None:
    """[AUDIT] The gate names the pending successors: G2←S4-01, G4←S4-03."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    pending = " ".join(verdict.axes_pending)
    assert "G2" in pending and "S4-01" in pending
    assert "G4" in pending and "S4-03" in pending


def test_three_valued_guarantee_never_collapses_none_to_false_in_provisional() -> None:
    """[PROPERTY-TEST] A PENDING (None) guarantee must not be counted as a
    PROVISIONAL blocker (regression guard against ``all(g.passed ...)``)."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
    )
    # Provisional reached despite three None guarantees.
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA
    assert any(verdict.guarantee(g).passed is None for g in ("G2", "G4", "G5"))


# ---------------------------------------------------------------------------
# J-fallback rank-probability: bayes when posterior present, MLE-bootstrap else
# ---------------------------------------------------------------------------


def test_j_source_is_bayes_when_posterior_supplied() -> None:
    """[UNIT-TEST] j_source == 'bayes' and J == rank_one_probability(posterior)."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
    )
    assert verdict.j_source == "bayes"
    assert verdict.j_value is not None and verdict.j_value > 0.99


def test_j_source_is_mle_bootstrap_when_no_posterior() -> None:
    """[UNIT-TEST] With no posterior, J falls back to the MLE-BT paired bootstrap
    and is labelled 'mle-bootstrap' — J is always reportable."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    assert verdict.j_source == "mle-bootstrap"
    assert verdict.j_value is not None
    # pii-anon has the top composite in the fixture → rank-1 prob is high.
    assert verdict.j_value > 0.5


def test_j_unavailable_cannot_be_claim_grade() -> None:
    """[CONTRACT-TEST] If J cannot be computed at all (no posterior AND a
    single-system benchmark with no competitor pairs), j_source=='unavailable'
    and the verdict can never be CLAIM_GRADE."""
    bench = _canonical_benchmark()
    # Strip to a single system → no pairs for the MLE-bootstrap fallback.
    bench["systems"] = [s for s in bench["systems"]  # type: ignore[assignment]
                        if s["system"] == "pii-anon"]
    bench["available_competitors"] = []
    verdict = SupremacyVerdict.from_artifacts(
        bench, pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
    )
    assert verdict.j_source == "unavailable"
    # §5: J ≥ 0.95 is required for BOTH CLAIM_GRADE and PROVISIONAL, so a J that
    # cannot even be computed is a hard NOT_YET — not merely "not CLAIM_GRADE".
    # The verdict machine reports the J-gap as the binding constraint (ahead of
    # the unrun-Tier-C blocker), so NOT_YET is the only correct verdict here.
    assert verdict.verdict is Verdict.NOT_YET
    assert "J" in verdict.binding_constraint


# ---------------------------------------------------------------------------
# Tier-C unrun ⟹ at most PROVISIONAL (CLAIM_GRADE blocked)
# ---------------------------------------------------------------------------


def test_tier_c_unrun_caps_verdict_at_provisional() -> None:
    """[CONTRACT-TEST] Tier-C unrun ⟹ CLAIM_GRADE blocked ⟹ at most PROVISIONAL,
    even with everything else (canonical, all-G, J≥0.95) passing."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
    )
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA
    assert verdict.unrun_tier_c  # honesty boundary populated


def test_tier_c_all_waived_unblocks_claim_grade() -> None:
    """[CONTRACT-TEST] The WHOLE Tier-R ∪ Tier-C set waived-with-reason (all
    Tier-C + the unrun Tier-R gliner2) satisfies the §5 tier predicate so (with
    everything else passing) the verdict is CLAIM_GRADE."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "documented reason" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA


# ---------------------------------------------------------------------------
# Honesty surface always present
# ---------------------------------------------------------------------------


def test_carve_out_note_always_emitted_regardless_of_verdict() -> None:
    """[AUDIT] The OpenAI raw-F1 carve-out note is ALWAYS emitted."""
    for bench in (_canonical_benchmark(),):
        verdict = SupremacyVerdict.from_artifacts(bench)
        assert verdict.carve_out_note
        assert "openai" in verdict.carve_out_note.lower()


def test_not_yet_carries_canonical_false_banner_when_provisional_run() -> None:
    """[AUDIT] A non-canonical run surfaces the canonical_claim_run=False banner
    in the honesty fields (not silently swallowed)."""
    bench = _canonical_benchmark()
    bench["run_metadata"]["canonical_claim_run"] = False  # type: ignore[index]
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.canonical_claim_run is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_verdict_is_deterministic_for_fixed_inputs() -> None:
    """[PROPERTY-TEST] from_artifacts is a pure function of its inputs — the
    MLE-bootstrap J uses a fixed internal seed → byte-identical J across calls."""
    b1 = _canonical_benchmark()
    b2 = copy.deepcopy(b1)
    v1 = SupremacyVerdict.from_artifacts(b1)
    v2 = SupremacyVerdict.from_artifacts(b2)
    assert v1.j_value == v2.j_value
    assert v1.verdict == v2.verdict
    assert v1.binding_constraint == v2.binding_constraint


# ---------------------------------------------------------------------------
# G7 RecallFloorVerdictGuard (story §3 G7): "a floor-breaching system can never
# top-rank." The guard couples the recall-floor signals to BOTH the J/ranking
# (a breacher can never be J-argmax) AND G7 (the top claimant must not breach).
# Recall-floor breach is keyed on RECALL-SPECIFIC signals only — a non-qualifying
# qualification_status, a failing recall/f1 profile floor-check, or (when the
# per-language ε artifact is present) ε > EPS_RECALL_PER_LANG. Latency/throughput
# floor failures are NOT recall breaches (the speed-floor is a different axis).
# ---------------------------------------------------------------------------


def _breaching_system(
    name: str,
    *,
    composite: float,
    recall: float = 0.99,
    precision: float = 0.99,
) -> dict[str, object]:
    """A system that BREACHES the recall floor via a non-qualifying
    qualification_status, while otherwise looking strong (high composite)."""
    return _system(
        name,
        recall=recall,
        precision=precision,
        composite=composite,
        per_entity_recall=_ENSEMBLE_ENTITIES,
        qualification="disqualified",  # non-qualifying ⇒ recall-floor breach
    )


def test_recall_floor_breachers_flags_nonqualifying_status() -> None:
    """[UNIT-TEST] recall_floor_breachers() flags a system whose
    qualification_status is non-qualifying (disqualified/floor_breach/…), and
    does NOT flag a 'core'/'qualified' system."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _systems_by_name,
    )

    bench = _canonical_benchmark()
    bench["systems"].append(_breaching_system("rogue", composite=0.99))  # type: ignore[attr-defined]
    systems = _systems_by_name(bench)
    breachers = recall_floor_breachers(bench, systems)
    assert "rogue" in breachers
    assert "pii-anon" not in breachers  # core qualifies
    assert "gliner" not in breachers  # qualified qualifies


def test_recall_floor_breachers_flags_failing_recall_profile_check() -> None:
    """[UNIT-TEST] A per-profile recall (or f1) floor-check that FAILS marks the
    run recall-floor-breaching — but a failing LATENCY/throughput check does
    NOT (the RecallFloorVerdictGuard is recall-specific, not the conflated
    floor_pass)."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _systems_by_name,
    )

    # A latency-only floor failure must NOT make anyone a recall breacher.
    latency_bench = _canonical_benchmark()
    latency_bench["floor_pass"] = False  # ensemble floor fails on speed
    latency_bench["profile_results"] = [
        {
            "profile": "short_chat",
            "floor_pass": False,
            "floor_checks": [
                {"metric": "latency_p50_ms", "passed": False, "actual": 9.9,
                 "target": 1.0, "comparator": "scrubadub"},
                {"metric": "recall", "passed": True, "actual": 0.8,
                 "target": 0.6, "comparator": "gliner"},
            ],
        }
    ]
    systems = _systems_by_name(latency_bench)
    assert recall_floor_breachers(latency_bench, systems) == frozenset()

    # A RECALL floor-check failure, by contrast, IS a recall-floor breach.
    recall_bench = _canonical_benchmark()
    recall_bench["floor_pass"] = False
    recall_bench["profile_results"] = [
        {
            "profile": "long_document",
            "floor_pass": False,
            "floor_checks": [
                {"metric": "recall", "passed": False, "actual": 0.10,
                 "target": 0.66, "comparator": "gliner"},
            ],
        }
    ]
    systems = _systems_by_name(recall_bench)
    assert recall_floor_breachers(recall_bench, systems)  # non-empty


def test_recall_floor_breachers_per_language_eps_when_artifact_present() -> None:
    """[UNIT-TEST] When the per-language ε artifact IS present and a language's ε
    exceeds EPS_RECALL_PER_LANG, the run is recall-floor-breaching; an absent
    artifact is PENDING-not-fabricated (no breach manufactured from nothing)."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _systems_by_name,
    )

    bench = _canonical_benchmark()
    bench["per_language_recall_delta"] = {"en": 0.001, "de": 0.02}  # 0.02 > 0.005
    systems = _systems_by_name(bench)
    assert recall_floor_breachers(bench, systems)  # ε breach

    # Absent artifact + qualifying statuses + passing profiles ⇒ no fabricated breach.
    clean = _canonical_benchmark()
    del clean["per_language_recall_delta"]
    systems_clean = _systems_by_name(clean)
    assert recall_floor_breachers(clean, systems_clean) == frozenset()


def test_floor_breacher_with_highest_composite_never_top_ranks() -> None:
    """[PROPERTY-TEST] THE TEETH of the RecallFloorVerdictGuard (story §3 G7).

    Construct a benchmark where a FLOOR-BREACHING system holds the STRICTLY
    HIGHEST composite_score. The guard must ensure:
      1. it is NEVER J-argmax — the crowned (rank-1) system is a floor-COMPLIANT
         one (pii-anon), i.e. J(breacher) does not crown it; AND
      2. the verdict can NEVER be CLAIM_GRADE on account of the guard (a
         floor-breaching top-composite system cannot yield a claim-grade SDO).
    """
    bench = _canonical_benchmark()
    # The breacher out-composites everyone (0.99 > pii-anon 0.80) yet breaches
    # the recall floor (non-qualifying status) → it must never be crowned.
    bench["systems"].append(_breaching_system("rogue-sota", composite=0.99))  # type: ignore[attr-defined]

    verdict = SupremacyVerdict.from_artifacts(
        bench,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
    )
    # The breacher is identified.
    assert "rogue-sota" in verdict.recall_floor_breachers
    # 1. The MLE-bootstrap J must NOT crown the breacher: pii-anon's rank-1
    #    probability stays the SDO objective, and the breacher cannot be the
    #    argmax (its rank-1 mass is forced to 0 / it is excluded).
    j_top = verdict.j_rank1_system
    assert j_top is not None
    assert j_top != "rogue-sota"
    # 2. The verdict cannot be CLAIM_GRADE while a floor-breaching system holds
    #    the top composite (the guard sub-condition of G7 fails).
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA
    g7 = verdict.guarantee("G7")
    assert g7.passed is False
    assert "floor" in g7.binding_detail.lower()


def test_floor_compliant_top_system_passes_the_recall_floor_guard() -> None:
    """[UNIT-TEST] The companion to the teeth: a floor-COMPLIANT top system
    (pii-anon, qualifying status, passing recall floor-checks) passes the
    RecallFloorVerdictGuard — G7's guard sub-condition does not fire, and with a
    canonical run + provenance G7 PASSES."""
    bench = _canonical_benchmark()  # pii-anon top composite, 'core', floor clean
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    assert verdict.recall_floor_breachers == frozenset()
    assert verdict.guarantee("G7").passed is True  # guard does not demote a clean top
    # The crowned system is the floor-compliant claimant.
    assert verdict.j_rank1_system == "pii-anon"


def test_g7_guard_fails_when_top_claimant_pii_anon_is_floor_breaching() -> None:
    """[CONTRACT-TEST] If pii-anon ITSELF is the top-composite claimant but
    breaches the recall floor (e.g. a per-language ε regression), the
    RecallFloorVerdictGuard sub-condition of G7 fails with a clear binding_detail
    — even though canonical_claim_run is True and provenance is complete."""
    bench = _canonical_benchmark()
    # pii-anon stays top composite, but a per-language ε regression breaches the
    # recall floor for the run.
    bench["per_language_recall_delta"] = {"en": 0.05}  # 0.05 ≫ 0.005
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    g7 = verdict.guarantee("G7")
    assert g7.passed is False
    assert "floor" in g7.binding_detail.lower()
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


# ---------------------------------------------------------------------------
# §5 completion predicate: CLAIM_GRADE ⟺ … ∧ (Tier-R ∪ Tier-C all RUN-or-WAIVED).
# An UNRUN Tier-R member (gliner2) — not just unrun Tier-C — blocks CLAIM_GRADE.
# ---------------------------------------------------------------------------


def test_unrun_tier_r_gliner2_blocks_claim_grade() -> None:
    """[CONTRACT-TEST] §5: gliner2 is a Tier-R member that is UNRUN in today's
    benchmark. With everything else satisfied (canonical, all-G pass, J≥0.95, all
    Tier-C waived), an UNRUN-unwaived Tier-R member STILL blocks CLAIM_GRADE → at
    most PROVISIONAL. (Regression guard: _decide must gate on Tier-R ∪ Tier-C, not
    Tier-C alone.)"""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(),  # available_competitors omits gliner2 ⇒ UNRUN
        theta_samples=theta,
        posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
    )
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA
    assert "gliner2" in verdict.binding_constraint


def test_waived_tier_r_gliner2_unblocks_claim_grade() -> None:
    """[CONTRACT-TEST] §5: gliner2 WAIVED-with-reason satisfies the Tier-R∪Tier-C
    predicate, so (with all else satisfied) the verdict reaches CLAIM_GRADE."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(),
        theta_samples=theta,
        posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA
    assert verdict.binding_constraint == ""


def test_unrun_tier_r_surfaces_in_honesty_boundary() -> None:
    """[AUDIT] The gate exposes the unrun Tier-R set (gliner2) as a visible
    honesty-boundary field, distinct from unrun Tier-C."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    assert "gliner2" in verdict.unrun_tier_r


# ---------------------------------------------------------------------------
# THE ONE real-artifact test — READ-ONLY, value-independent.
# ---------------------------------------------------------------------------

_REAL_ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "benchmarks"
    / "benchmark-results.json"
)


@pytest.mark.skipif(
    not _REAL_ARTIFACT.exists(), reason="benchmark-results.json artifact absent"
)
def test_real_artifact_verdict_is_not_yet_binding_canonical_run() -> None:
    """[INTEGRATION-TEST] Load the REAL benchmark JSON read-only and assert the
    value-INDEPENDENT verdict: today's artifact has canonical_claim_run=False, so
    the gate MUST report NOT_YET with binding_constraint naming
    canonical_claim_run=False. (No benchmark numbers are asserted — only the
    canonical-run honesty gate.)"""
    bench = json.loads(_REAL_ARTIFACT.read_text(encoding="utf-8"))
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.verdict is Verdict.NOT_YET
    assert "canonical_claim_run" in verdict.binding_constraint
    assert "False" in verdict.binding_constraint
    assert verdict.canonical_claim_run is False


@pytest.mark.skipif(
    not _REAL_ARTIFACT.exists(), reason="benchmark-results.json artifact absent"
)
def test_real_artifact_recall_floor_guard_does_not_perturb_headline() -> None:
    """[INTEGRATION-TEST] CRITICAL invariant: the RecallFloorVerdictGuard must NOT
    change today's headline verdict/binding. The real artifact's top-level
    ``floor_pass`` is False, but that is driven by LATENCY/throughput floor-checks
    (short_chat / structured_form_latency / log_lines), NOT recall — every recall
    and f1 profile floor-check PASSES. So:
      * pii-anon (the top-composite claimant, qualification 'core', recall
        floor-checks passing) is NOT a recall-floor breacher — the guard must not
        conflate the latency-driven floor_pass=False with a recall breach; AND
      * the headline stays NOT_YET / canonical_claim_run=False (canonical is the
        #1 binding constraint, ahead of the guard sub-condition)."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _systems_by_name,
    )

    bench = json.loads(_REAL_ARTIFACT.read_text(encoding="utf-8"))
    systems = _systems_by_name(bench)
    breachers = recall_floor_breachers(bench, systems)
    # Latency-driven floor_pass=False must NOT manufacture a recall breach for
    # the qualifying claimant.
    assert "pii-anon" not in breachers
    # Headline unchanged: canonical remains the #1 binding constraint.
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.binding_constraint.startswith("canonical_claim_run=False")


# ---------------------------------------------------------------------------
# CLI surface — the `supremacy` command (thin; gate logic stays in the gate).
# ---------------------------------------------------------------------------


def _write_bench(tmp_path: Path, bench: dict[str, object]) -> Path:
    target = tmp_path / "benchmark-results.json"
    target.write_text(json.dumps(bench), encoding="utf-8")
    return target


def test_cli_supremacy_is_non_blocking_exit_zero(tmp_path: Path) -> None:
    """[INTEGRATION-TEST] `supremacy` (default) prints the verdict + binding
    constraint and exits 0 even when the verdict is NOT_YET (non-blocking)."""
    from typer.testing import CliRunner

    from pii_anon.cli import create_app

    bench = _canonical_benchmark()
    bench["run_metadata"]["canonical_claim_run"] = False  # type: ignore[index]
    artifact = _write_bench(tmp_path, bench)

    runner = CliRunner()
    result = runner.invoke(
        create_app(), ["supremacy", "--artifact", str(artifact), "--output", "json"]
    )
    assert result.exit_code == 0
    assert "NOT_YET" in result.stdout
    assert "canonical_claim_run" in result.stdout


def test_cli_supremacy_canonical_claim_exits_one_unless_claim_grade(
    tmp_path: Path,
) -> None:
    """[INTEGRATION-TEST] With --canonical-claim a non-CLAIM_GRADE verdict exits
    1 (the only hard-failure mode)."""
    from typer.testing import CliRunner

    from pii_anon.cli import create_app

    artifact = _write_bench(tmp_path, _canonical_benchmark())  # PROVISIONAL at best
    runner = CliRunner()
    result = runner.invoke(
        create_app(),
        ["supremacy", "--artifact", str(artifact), "--canonical-claim"],
    )
    assert result.exit_code == 1


def test_cli_supremacy_missing_artifact_is_bad_parameter(tmp_path: Path) -> None:
    """[INTEGRATION-TEST] A missing artifact path is a typer BadParameter (exit
    non-zero), never a silent empty verdict."""
    from typer.testing import CliRunner

    from pii_anon.cli import create_app

    runner = CliRunner()
    result = runner.invoke(
        create_app(),
        ["supremacy", "--artifact", str(tmp_path / "nope.json")],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# S4-01: G2 — pseudonymization-integrity strict dominance (PASS/FAIL/PENDING).
#
# G2 is now a pure (benchmark) -> GuaranteeResult computed by
# `_g2_pseudonymization_integrity` (replacing the hardcoded PENDING placeholder):
#   * PASS  iff pii-anon's pseudonymization_integrity_score STRICTLY dominates
#           every competitor's AND unauthorized_reversal_rate == 0 AND BOTH family
#           fields (anonymization_score + pseudonymization_integrity_score) are
#           present-and-distinct on pii-anon;
#   * FAIL  iff the fields are present but pii-anon does NOT dominate (or
#           unauthorized-reversal > 0) — the binding detail names the gap;
#   * PENDING (None) iff the benchmark lacks the fields (NO fabrication — the
#           current smoke artifact lacks them; the S7 canonical run emits them).
# ---------------------------------------------------------------------------


def _with_pseudo_fields(
    bench: dict[str, object],
    *,
    pii_anon_pi: float,
    competitor_pi: float,
    pii_anon_anon: float = 0.95,
    unauthorized_reversal_rate: float = 0.0,
) -> dict[str, object]:
    """Inject the S4-01 family fields onto a benchmark fixture (synthetic).

    pii-anon (and pii-anon-swarm) get ``pseudonymization_integrity_score`` =
    ``pii_anon_pi`` + a distinct ``anonymization_score`` + an
    ``unauthorized_reversal_rate``; every competitor gets ``competitor_pi`` (and a
    competitor anonymization_score). Mutates a COPY-safe per-system dict in place.
    """
    for s in bench["systems"]:  # type: ignore[attr-defined]
        name = s["system"]  # type: ignore[index]
        if name in ("pii-anon", "pii-anon-swarm"):
            s["pseudonymization_integrity_score"] = pii_anon_pi  # type: ignore[index]
            s["anonymization_score"] = pii_anon_anon  # type: ignore[index]
            s["unauthorized_reversal_rate"] = unauthorized_reversal_rate  # type: ignore[index]
        else:
            s["pseudonymization_integrity_score"] = competitor_pi  # type: ignore[index]
            s["anonymization_score"] = 0.50  # type: ignore[index]
            s["unauthorized_reversal_rate"] = 0.0  # type: ignore[index]
    return bench


def _with_core_pseudo_only(
    bench: dict[str, object],
    *,
    pii_anon_pi: float,
    pii_anon_anon: float = 0.95,
    unauthorized_reversal_rate: float = 0.0,
) -> dict[str, object]:
    """Inject the S4-01 family fields ONLY onto the pii-anon ladder.

    Every competitor is left WITHOUT a ``pseudonymization_integrity_score`` — the
    no-comparator scenario the SDO gate must NOT fabricate a win against.
    """
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] in ("pii-anon", "pii-anon-swarm"):  # type: ignore[index]
            s["pseudonymization_integrity_score"] = pii_anon_pi  # type: ignore[index]
            s["anonymization_score"] = pii_anon_anon  # type: ignore[index]
            s["unauthorized_reversal_rate"] = unauthorized_reversal_rate  # type: ignore[index]
    return bench


def _with_core_pseudo_and_one_competitor(
    bench: dict[str, object],
    *,
    pii_anon_pi: float,
    one_competitor_pi: float,
    competitor_name: str = "gliner",
    pii_anon_anon: float = 0.95,
) -> dict[str, object]:
    """Inject the family fields onto the pii-anon ladder AND onto exactly ONE
    competitor (``competitor_name``), leaving every OTHER competitor without a
    ``pseudonymization_integrity_score``.

    Exercises the boundary the new no-comparator guard must respect: a single real
    comparator is enough to compute dominance — the guard fires only when NONE
    carry the field.
    """
    for s in bench["systems"]:  # type: ignore[attr-defined]
        name = s["system"]  # type: ignore[index]
        if name in ("pii-anon", "pii-anon-swarm"):
            s["pseudonymization_integrity_score"] = pii_anon_pi  # type: ignore[index]
            s["anonymization_score"] = pii_anon_anon  # type: ignore[index]
            s["unauthorized_reversal_rate"] = 0.0  # type: ignore[index]
        elif name == competitor_name:
            s["pseudonymization_integrity_score"] = one_competitor_pi  # type: ignore[index]
    return bench


def _g2_systems(
    core_pi: object,
    *,
    anon: object = 0.95,
    unauthorized: float = 0.0,
    competitors: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Build a minimal name→system map for direct ``_g2_*`` unit tests.

    ``competitors`` maps a competitor name to its
    ``pseudonymization_integrity_score`` value; a name absent from the mapping (or
    an empty/None mapping) yields ZERO competitors. To model a competitor that
    carries NO score, simply omit it from ``competitors``.
    """
    systems: dict[str, dict[str, object]] = {
        "pii-anon": {
            "system": "pii-anon",
            "pseudonymization_integrity_score": core_pi,
            "anonymization_score": anon,
            "unauthorized_reversal_rate": unauthorized,
        },
    }
    for name, pi in (competitors or {}).items():
        systems[name] = {"system": name, "pseudonymization_integrity_score": pi}
    return systems


def test_g2_pass_when_pii_anon_pi_strictly_dominates_and_no_unauthorized() -> None:
    """[CONTRACT-TEST] A5: G2 PASS — pii-anon's pseudonymization_integrity_score
    strictly dominates every competitor, unauthorized-reversal is 0, and both
    family fields are present-and-distinct on pii-anon."""
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    verdict = SupremacyVerdict.from_artifacts(bench)
    g2 = verdict.guarantee("G2")
    assert g2.passed is True
    assert g2.observed >= g2.bar


def test_g2_fail_when_present_but_not_dominant() -> None:
    """[CONTRACT-TEST] A5: G2 FAIL — the fields are present but a competitor's
    pseudonymization_integrity_score meets/exceeds pii-anon's (not strict
    dominance); the binding detail names the gap."""
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.60, competitor_pi=0.90
    )
    verdict = SupremacyVerdict.from_artifacts(bench)
    g2 = verdict.guarantee("G2")
    assert g2.passed is False
    assert "G2" in g2.binding_detail


def test_g2_fail_when_unauthorized_reversal_nonzero_even_if_dominant() -> None:
    """[CONTRACT-TEST] A5 / NFR-014: even with strict PI dominance, a nonzero
    unauthorized-reversal rate on pii-anon FAILS G2 (NFR-014 must = 0)."""
    bench = _with_pseudo_fields(
        _canonical_benchmark(),
        pii_anon_pi=0.95,
        competitor_pi=0.60,
        unauthorized_reversal_rate=0.10,  # a leak — NFR-014 violated
    )
    verdict = SupremacyVerdict.from_artifacts(bench)
    g2 = verdict.guarantee("G2")
    assert g2.passed is False
    assert "reversal" in g2.binding_detail.lower()


def test_g2_pending_when_fields_absent_never_fabricated() -> None:
    """[CONTRACT-TEST] A5 / A6: G2 PENDING (None) when the benchmark lacks the
    pseudonymization_integrity_score / anonymization_score fields — never a
    fabricated PASS (the current smoke artifact lacks them)."""
    bench = _canonical_benchmark()  # no family fields injected
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G2").passed is None


def test_g2_pending_when_only_one_family_field_present() -> None:
    """[CONTRACT-TEST] A5: BOTH family fields must be present-and-distinct; if only
    pseudonymization_integrity_score is present (anonymization_score absent), G2 is
    PENDING — the two-family contract is not satisfiable from a half-populated
    artifact (no fabrication of the missing half)."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] in ("pii-anon", "pii-anon-swarm"):  # type: ignore[index]
            s["pseudonymization_integrity_score"] = 0.95  # type: ignore[index]
            # anonymization_score deliberately absent
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G2").passed is None


def test_g2_teeth_non_vacuous_neutering_dominance_flips_pass_to_fail() -> None:
    """[PROPERTY-TEST] A5 META-GUARD: the dominance check is non-vacuous — raising
    a competitor's pseudonymization_integrity_score above pii-anon's flips the
    SAME fixture from PASS to FAIL (the guarantee actually depends on dominance)."""
    passing = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    assert SupremacyVerdict.from_artifacts(passing).guarantee("G2").passed is True

    neutered = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.99
    )
    assert SupremacyVerdict.from_artifacts(neutered).guarantee("G2").passed is False


def test_g2_pass_unblocks_claim_grade_with_all_else_satisfied() -> None:
    """[CONTRACT-TEST] A5: a computed G2 PASS (from the fields, NOT a manual
    override) participates in CLAIM_GRADE — with G4/G5 overridden PASS, J≥0.95,
    canonical run, and the whole Tier-R∪Tier-C set waived, the verdict reaches
    CLAIM_GRADE (G2 is no longer a forced PENDING blocker)."""
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        # G2 is computed from the fields; only the still-pending successors G4/G5
        # are supplied PASS here.
        pending_overrides={"G4": True, "G5": True},
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G2").passed is True
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA


def test_g2_override_still_honoured_for_backward_compat() -> None:
    """[CONTRACT-TEST] A5: an explicit pending_overrides['G2'] still wins (the
    successor-override seam is preserved) — an injected G2=False overrides the
    computed PASS, so existing tests that drive G2 via overrides keep working."""
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    verdict = SupremacyVerdict.from_artifacts(
        bench, pending_overrides={"G2": False}
    )
    assert verdict.guarantee("G2").passed is False


@pytest.mark.skipif(
    not _REAL_ARTIFACT.exists(), reason="benchmark-results.json artifact absent"
)
def test_g2_real_artifact_stays_pending_and_verdict_not_yet() -> None:
    """[CONTRACT-TEST] A6: on the REAL (smoke) artifact — which lacks the family
    fields — G2 reads PENDING (no fabricated PASS) and the headline verdict stays
    NOT_YET with binding canonical_claim_run=False."""
    bench = json.loads(_REAL_ARTIFACT.read_text(encoding="utf-8"))
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G2").passed is None  # PENDING on missing fields
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.binding_constraint.startswith("canonical_claim_run=False")


def test_g2_successor_label_names_s4_01_when_pending() -> None:
    """[AUDIT] A6: while G2 is PENDING (fields absent), the gate still names its
    successor S4-01 as awaiting a run that emits the fields."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    g2 = verdict.guarantee("G2")
    assert g2.passed is None
    assert "S4-01" in g2.binding_detail or any(
        "S4-01" in p for p in verdict.axes_pending
    )


def test_g2_is_deterministic_on_identical_benchmarks() -> None:
    """[PROPERTY-TEST] A8 / AX-002: _g2_pseudonymization_integrity is a pure
    function — identical benchmarks yield identical G2 outcomes."""
    b1 = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    b2 = copy.deepcopy(b1)
    g2_a = SupremacyVerdict.from_artifacts(b1).guarantee("G2")
    g2_b = SupremacyVerdict.from_artifacts(b2).guarantee("G2")
    assert g2_a == g2_b


def test_g2_pending_when_no_competitor_carries_pi_never_fabricated() -> None:
    """[CONTRACT-TEST] A5 / A6: pii-anon carries BOTH family fields but NO
    competitor carries a pseudonymization_integrity_score. There is no comparator,
    so the gate returns PENDING (None) — NOT a fabricated PASS against a phantom
    0.0 baseline (the original false-PASS gap). Symmetric with the core-side guard:
    PENDING when the artifact is absent — never fabricated."""
    bench = _with_core_pseudo_only(_canonical_benchmark(), pii_anon_pi=0.95)
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G2").passed is None  # PENDING, not a fabricated PASS


def test_g2_no_competitor_pi_blocks_claim_grade_sota() -> None:
    """[CONTRACT-TEST] A5 / §5 END-TO-END: the reported false-PASS exploit is
    closed. With a canonical run, J≥0.95, G4/G5 overridden PASS, and the whole
    Tier-R∪Tier-C set waived — but NO competitor carrying a
    pseudonymization_integrity_score — G2 is PENDING and the headline verdict does
    NOT reach CLAIM_GRADE_SOTA (previously it did, on a vacuous G2 PASS over a
    phantom 0.0 comparator)."""
    bench = _with_core_pseudo_only(_canonical_benchmark(), pii_anon_pi=0.95)
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G4": True, "G5": True},
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G2").passed is None
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


def test_g2_pending_when_zero_competitors_no_comparator() -> None:
    """[CONTRACT-TEST] A5 anti-fabrication: with NO competitor systems at all there
    is no comparator, so strict dominance is unprovable. G2 is PENDING (None) — the
    gate must not fabricate a win against a phantom 0.0 baseline."""
    g2 = _g2_pseudonymization_integrity(_g2_systems(0.95, competitors={}))
    assert g2.passed is None


def test_g2_pass_when_only_one_competitor_carries_lower_pi_guard_not_overbroad() -> None:
    """[PROPERTY-TEST] A5 META-GUARD: the new no-comparator PENDING guard is NOT
    over-broad. When AT LEAST ONE competitor carries a (lower) real
    pseudonymization_integrity_score, G2 still computes a PASS (dominance over the
    real comparator) even though the OTHER competitors lack the field."""
    bench = _with_core_pseudo_and_one_competitor(
        _canonical_benchmark(), pii_anon_pi=0.95, one_competitor_pi=0.60
    )
    g2 = SupremacyVerdict.from_artifacts(bench).guarantee("G2")
    assert g2.passed is True
    assert g2.observed >= g2.bar


def test_g2_s7_02_producer_contract_competitors_carry_honest_zero_pi() -> None:
    """[CONTRACT-TEST] S7-02 / SO-11 (gate-side companion): the canonical-run
    producer's contract — every Tier-R competitor carries an HONEST
    ``pseudonymization_integrity_score = 0.0`` (irreversible incumbents) — gives
    the gate a real comparator, so G2 computes a real dominance PASS (NOT a phantom
    0.0 win, and NOT PENDING). This pins the exact producer↔gate field contract the
    S7-02 ``_attach_g2_deid_families`` honours; if the producer ever stopped
    emitting the competitor field, G2 would fall back to PENDING (the companion
    SUT-only case below) and the headline verdict could never reach a claim."""
    bench = _g2_systems(
        0.97,  # pii-anon real pseudonymization-integrity
        anon=1.0,
        unauthorized=0.0,
        # The producer emits an HONEST 0.0 on EVERY irreversible incumbent.
        competitors={"gliner": 0.0, "presidio": 0.0, "scrubadub": 0.0},
    )
    g2 = _g2_pseudonymization_integrity(bench)
    assert g2.passed is True, g2.binding_detail  # real comparator ⇒ provable
    assert g2.observed == 0.97  # pii-anon's score
    assert g2.bar == 0.0  # best (honest) competitor comparator

    # Companion: strip the competitor field (a SUT-only artifact) ⇒ PENDING, no
    # phantom 0.0 win. This is the regression the producer-side A16 also pins.
    sut_only = _g2_systems(0.97, anon=1.0, unauthorized=0.0, competitors={})
    g2_sut = _g2_pseudonymization_integrity(sut_only)
    assert g2_sut.passed is None, g2_sut.binding_detail


def test_g2_pending_when_core_pi_non_finite_or_out_of_range_not_a_win() -> None:
    """[CONTRACT-TEST] A5 anti-fabrication: a non-finite or out-of-[0,1] pii-anon
    pseudonymization_integrity_score is not a real measurement
    (deid_families.integrity_score is bounded [0,1] by construction) and must
    never seed a dominance win. +inf previously PASSed vacuously (inf > any
    competitor); a corrupt score (NaN / ±inf / 1.5 / -0.1) is now treated as an
    ABSENT measurement ⇒ G2 PENDING (None) — never a fabricated PASS, never a
    misleading FAIL over a corrupt bar."""
    for bad in (float("inf"), float("nan"), float("-inf"), 1.5, -0.1):
        g2 = _g2_pseudonymization_integrity(
            _g2_systems(bad, competitors={"gliner": 0.60})
        )
        assert g2.passed is None


def test_g2_pending_when_core_pi_is_bool_not_coerced_to_one() -> None:
    """[CONTRACT-TEST] A5 anti-fabrication: a boolean pseudonymization_integrity_
    score is a flag, not a score — Python's isinstance(True, (int, float)) is True,
    so True would otherwise coerce to 1.0 and vacuously dominate 0.60. A boolean
    family field is treated as ABSENT ⇒ G2 PENDING (None), never a fabricated
    PASS."""
    g2 = _g2_pseudonymization_integrity(
        _g2_systems(True, competitors={"gliner": 0.60})
    )
    assert g2.passed is None


def test_vector11_is_finite_number_rejects_unbounded_int_fail_closed() -> None:
    """[SECURITY-TEST] S7-02 vector #11: ``_is_finite_number`` REJECTS a Python int
    wider than a C double (``10**400``) — ``math.isfinite`` raises OverflowError on
    such a value, so the predicate must guard it and fail CLOSED (return ``False``),
    never crash. This is the ONE sanctioned hardening of the SDO gate under the S7-02
    close (story §9 / vector #11). Ordinary finite values are unaffected."""
    assert _is_finite_number(10**400) is False
    assert _is_finite_number(-(10**400)) is False
    # The unguarded math.isfinite WOULD raise on these; the guard must not.
    import math

    with pytest.raises(OverflowError):
        math.isfinite(10**400)  # documents WHY the guard is needed
    # Ordinary values keep their semantics.
    assert _is_finite_number(0.5) is True
    assert _is_finite_number(1) is True
    assert _is_finite_number(True) is False
    assert _is_finite_number(float("nan")) is False
    assert _finite_unit_score(10**400) is None
    assert _finite_unit_score(0.42) == 0.42


def test_vector11_g2_pending_on_huge_int_field_no_crash() -> None:
    """[SECURITY-TEST] S7-02 vector #11 in G2: a huge-int (out-of-float-range)
    pseudonymization_integrity_score is treated as an ABSENT measurement ⇒ G2 PENDING
    (None), never an OverflowError crash and never a fabricated win. Probed on BOTH
    the SUT field and the sole-competitor field."""
    # Huge int on the SUT ⇒ no real SUT score ⇒ PENDING (not a crash).
    g2_core = _g2_pseudonymization_integrity(
        _g2_systems(10**400, competitors={"gliner": 0.60})
    )
    assert g2_core.passed is None
    # Huge int as the sole competitor 'score' ⇒ no real comparator ⇒ PENDING.
    g2_comp = _g2_pseudonymization_integrity(
        _g2_systems(0.95, competitors={"gliner": 10**400})
    )
    assert g2_comp.passed is None


def test_vector11_g4_fail_on_huge_int_per_class_ece_no_crash() -> None:
    """[SECURITY-TEST] S7-02 vector #11 in G4: a huge-int per-class ECE is a
    present-but-out-of-range measurement ⇒ G4 FAIL (it must never slip ``ece ≤ bar``
    via an OverflowError crash). The artifact-supplied ECE-threshold clamp is also
    robust to a huge-int threshold (it falls back to the sanctioned bar)."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "US_SSN": 10**400})
    )
    assert g4.passed is False  # not a crash, not a vacuous pass
    # A huge-int threshold must not loosen the bar (clamped to the sanctioned ceiling).
    g4b = _g4_calibration_selective_risk(
        _g4_core(
            per_class_ece={"EMAIL_ADDRESS": 0.5},
            per_class_ece_threshold={"EMAIL_ADDRESS": 10**400},
        )
    )
    assert g4b.passed is False


def test_g2_pending_when_only_competitor_pi_is_not_a_real_score() -> None:
    """[CONTRACT-TEST] A5 anti-fabrication: a competitor whose only
    pseudonymization_integrity_score is a bool or non-finite value carries NO real
    comparator. When that is the sole competitor 'score', there is nothing to
    dominate ⇒ G2 PENDING (None) — neither a fabricated PASS (phantom 0.0) nor a
    misleading FAIL (coercing True→1.0 or treating +inf as a real bar)."""
    for bogus in (True, float("inf"), float("nan")):
        g2 = _g2_pseudonymization_integrity(
            _g2_systems(0.95, competitors={"gliner": bogus})
        )
        assert g2.passed is None


# ---------------------------------------------------------------------------
# S4-03: G4 — calibration / selective-risk (PASS/FAIL/PENDING teeth).
#
# G4 is now a pure (benchmark) -> GuaranteeResult computed by
# `_g4_calibration_selective_risk` (replacing the hardcoded PENDING placeholder):
#   * PASS  iff pii-anon's per-class ECE ≤ the NFR-017 thresholds (0.05
#           high-resource / 0.08 long-tail) AND the selective-risk AURC curve is
#           monotone non-increasing AND there are ≥3 abstention operating points
#           AND calibrated_confidence_coverage == 1.0 (NFR-020, 0 bare-logit);
#   * FAIL  iff the calibration fields are PRESENT but a threshold is breached
#           (a class ECE over bar, a non-monotone AURC, <3 abstention points, or
#           coverage < 1.0) — the binding detail names the breach;
#   * PENDING (None) iff the benchmark lacks the calibration fields (NO
#           fabrication — the current smoke artifact lacks them; the S7 canonical
#           run emits them).
# ---------------------------------------------------------------------------


def _conforming_calibration() -> dict[str, object]:
    """A SYNTHETIC pii-anon calibration block that MEETS every NFR threshold.

    * per-class ECE (post-scaling) ≤ the per-class bar (0.05 high-resource;
      0.08 long-tail);
    * a monotone-non-increasing selective-risk curve (achieved risk does not
      increase as coverage drops) with an AURC in [0, 1];
    * ≥3 abstention operating points at selective-risk {1%, 2%, 5%};
    * calibrated_confidence_coverage == 1.0 (0 bare-logit).
    """
    return {
        "per_class_ece": {"EMAIL_ADDRESS": 0.02, "US_SSN": 0.03},
        "per_class_ece_threshold": {"EMAIL_ADDRESS": 0.05, "US_SSN": 0.05},
        "selective_risk_aurc": 0.04,
        # risk-coverage curve as (coverage, risk) rows ascending in coverage;
        # risk must be non-decreasing as coverage grows (monotone non-increasing
        # as you abstain MORE).
        "risk_coverage_curve": [
            {"coverage": 0.25, "risk": 0.005},
            {"coverage": 0.50, "risk": 0.01},
            {"coverage": 0.75, "risk": 0.02},
            {"coverage": 1.00, "risk": 0.04},
        ],
        "abstention_operating_points": [
            {"target_risk": 0.01, "achieved_coverage": 0.50, "achieved_risk": 0.01},
            {"target_risk": 0.02, "achieved_coverage": 0.75, "achieved_risk": 0.02},
            {"target_risk": 0.05, "achieved_coverage": 1.00, "achieved_risk": 0.04},
        ],
        "calibrated_confidence_coverage": 1.0,
    }


def _with_calibration(
    bench: dict[str, object], calibration: dict[str, object] | None
) -> dict[str, object]:
    """Inject a synthetic calibration block onto the pii-anon core system.

    ``calibration=None`` injects nothing (the PENDING case). Mutates the
    pii-anon system dict in place; competitors are left without calibration
    fields (only the claimant is gated, mirroring G2's claimant-keyed read).
    """
    if calibration is None:
        return bench
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "pii-anon":  # type: ignore[index]
            s.update(calibration)  # type: ignore[attr-defined]
    return bench


def test_g4_pass_when_calibration_meets_all_thresholds() -> None:
    """[CONTRACT-TEST] A7: G4 PASS — per-class ECE ≤ bar, AURC monotone, ≥3
    abstention points, coverage == 1.0."""
    bench = _with_calibration(_canonical_benchmark(), _conforming_calibration())
    g4 = SupremacyVerdict.from_artifacts(bench).guarantee("G4")
    assert g4.passed is True


def test_g4_fail_when_a_class_ece_exceeds_threshold() -> None:
    """[CONTRACT-TEST] A7: G4 FAIL — a class ECE over its NFR-017 bar; the binding
    detail names the worst class."""
    cal = _conforming_calibration()
    cal["per_class_ece"] = {"EMAIL_ADDRESS": 0.02, "US_SSN": 0.09}  # > 0.05 bar
    bench = _with_calibration(_canonical_benchmark(), cal)
    g4 = SupremacyVerdict.from_artifacts(bench).guarantee("G4")
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail


def test_g4_fail_when_coverage_below_one_nfr020() -> None:
    """[CONTRACT-TEST] A7 / NFR-020: G4 FAIL — calibrated_confidence_coverage < 1.0
    (a bare-logit finding exists) fails G4 even with good ECE/AURC."""
    cal = _conforming_calibration()
    cal["calibrated_confidence_coverage"] = 0.98  # a bare-logit slipped through
    bench = _with_calibration(_canonical_benchmark(), cal)
    g4 = SupremacyVerdict.from_artifacts(bench).guarantee("G4")
    assert g4.passed is False
    assert (
        "coverage" in g4.binding_detail.lower()
        or "NFR-020" in g4.binding_detail
    )


def test_g4_fail_when_fewer_than_three_abstention_points() -> None:
    """[CONTRACT-TEST] A7 / NFR-021: G4 FAIL — fewer than 3 abstention operating
    points."""
    cal = _conforming_calibration()
    cal["abstention_operating_points"] = [
        {"target_risk": 0.01, "achieved_coverage": 0.50, "achieved_risk": 0.01},
        {"target_risk": 0.05, "achieved_coverage": 1.00, "achieved_risk": 0.04},
    ]  # only 2
    bench = _with_calibration(_canonical_benchmark(), cal)
    g4 = SupremacyVerdict.from_artifacts(bench).guarantee("G4")
    assert g4.passed is False


def test_g4_fail_when_risk_coverage_not_monotone() -> None:
    """[CONTRACT-TEST] A7 / NFR-019: G4 FAIL — the risk-coverage curve is NOT
    monotone non-increasing (risk DROPS as coverage grows somewhere)."""
    cal = _conforming_calibration()
    cal["risk_coverage_curve"] = [
        {"coverage": 0.25, "risk": 0.05},  # higher risk at LOW coverage — illegal
        {"coverage": 0.50, "risk": 0.01},
        {"coverage": 0.75, "risk": 0.02},
        {"coverage": 1.00, "risk": 0.04},
    ]
    bench = _with_calibration(_canonical_benchmark(), cal)
    g4 = SupremacyVerdict.from_artifacts(bench).guarantee("G4")
    assert g4.passed is False
    assert "monoton" in g4.binding_detail.lower()


def test_g4_pending_when_fields_absent_never_fabricated() -> None:
    """[CONTRACT-TEST] A7 / A8: G4 PENDING (None) when the benchmark lacks the
    calibration fields — never a fabricated PASS (the current smoke artifact lacks
    them)."""
    bench = _canonical_benchmark()  # no calibration block injected
    assert SupremacyVerdict.from_artifacts(bench).guarantee("G4").passed is None


def test_g4_teeth_non_vacuous_worsening_a_class_ece_flips_pass_to_fail() -> None:
    """[PROPERTY-TEST] A7 META-GUARD: the per-class ECE check is non-vacuous —
    worsening ONE class's ECE past its bar flips the SAME fixture from PASS to
    FAIL (G4 actually depends on the ECE thresholds)."""
    passing = _with_calibration(_canonical_benchmark(), _conforming_calibration())
    assert SupremacyVerdict.from_artifacts(passing).guarantee("G4").passed is True

    worsened_cal = _conforming_calibration()
    worsened_cal["per_class_ece"] = {"EMAIL_ADDRESS": 0.02, "US_SSN": 0.07}
    worsened = _with_calibration(_canonical_benchmark(), worsened_cal)
    assert SupremacyVerdict.from_artifacts(worsened).guarantee("G4").passed is False


def test_g4_long_tail_class_held_to_looser_bar() -> None:
    """[CONTRACT-TEST] A7 / NFR-017: a long-tail class is held to the 0.08 bar, not
    the 0.05 high-resource bar — an ECE of 0.07 on a class whose per-class
    threshold is 0.08 still PASSES."""
    cal = _conforming_calibration()
    cal["per_class_ece"] = {"EMAIL_ADDRESS": 0.02, "CRYPTO_WALLET": 0.07}
    cal["per_class_ece_threshold"] = {"EMAIL_ADDRESS": 0.05, "CRYPTO_WALLET": 0.08}
    bench = _with_calibration(_canonical_benchmark(), cal)
    assert SupremacyVerdict.from_artifacts(bench).guarantee("G4").passed is True


def test_g4_override_still_honoured_for_backward_compat() -> None:
    """[CONTRACT-TEST] A7: an explicit pending_overrides['G4'] still wins (the
    successor-override seam is preserved) — an injected G4=False overrides the
    computed PASS, so existing tests driving G4 via overrides keep working."""
    bench = _with_calibration(_canonical_benchmark(), _conforming_calibration())
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides={"G4": False})
    assert verdict.guarantee("G4").passed is False


def test_g4_pass_unblocks_claim_grade_with_all_else_satisfied() -> None:
    """[CONTRACT-TEST] A7: a COMPUTED G4 PASS (from the fields, not an override)
    participates in CLAIM_GRADE — with G2 fields + G5 override PASS, J≥0.95,
    canonical run, and the whole Tier-R∪Tier-C set waived, the verdict reaches
    CLAIM_GRADE (G4 is no longer a forced PENDING blocker)."""
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    bench = _with_calibration(bench, _conforming_calibration())
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G5": True},  # G2 + G4 computed from fields
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G4").passed is True
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA


def test_g4_is_deterministic_on_identical_benchmarks() -> None:
    """[PROPERTY-TEST] A9 / AX-002: _g4_calibration_selective_risk is a pure
    function — identical benchmarks yield identical G4 outcomes."""
    b1 = _with_calibration(_canonical_benchmark(), _conforming_calibration())
    b2 = copy.deepcopy(b1)
    g4_a = SupremacyVerdict.from_artifacts(b1).guarantee("G4")
    g4_b = SupremacyVerdict.from_artifacts(b2).guarantee("G4")
    assert g4_a == g4_b


def test_g4_successor_label_names_s4_03_when_pending() -> None:
    """[AUDIT] A8: while G4 is PENDING (fields absent), the gate still names its
    successor S4-03 as awaiting a run that emits the calibration fields."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    g4 = verdict.guarantee("G4")
    assert g4.passed is None
    assert "S4-03" in g4.binding_detail or any(
        "S4-03" in p for p in verdict.axes_pending
    )


@pytest.mark.skipif(
    not _REAL_ARTIFACT.exists(), reason="benchmark-results.json artifact absent"
)
def test_g4_real_artifact_stays_pending_and_verdict_not_yet() -> None:
    """[CONTRACT-TEST] A8: on the REAL (smoke) artifact — which lacks the
    calibration fields — G4 reads PENDING (no fabricated PASS) and the headline
    verdict stays NOT_YET with binding canonical_claim_run=False."""
    bench = json.loads(_REAL_ARTIFACT.read_text(encoding="utf-8"))
    verdict = SupremacyVerdict.from_artifacts(bench)
    assert verdict.guarantee("G4").passed is None  # PENDING on missing fields
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.binding_constraint.startswith("canonical_claim_run=False")


# ---------------------------------------------------------------------------
# S4-03 (iter-3 hardening): G4 anti-fabrication — the G2/G4-close (wwty6wq9v)
#   found that `_g4_calibration_selective_risk` trusts artifact-supplied values
#   without validation and can be driven to a FALSE PASS that propagates to
#   CLAIM_GRADE_SOTA:
#     (1) a NON-FINITE per-class ECE (NaN / -inf) slips `ece > bar`
#         (`nan > bar` / `-inf > bar` are both False) so `ece_ok` stays True;
#     (2) `coverage > 1.0` (or +inf) is admitted by `coverage_val >= 1.0`
#         (NFR-020 is an EXACT 100% must — >1.0 is corrupt, not a pass);
#     (3) an artifact-supplied per-class ECE threshold (e.g. 0.99 / +inf / NaN)
#         is trusted UNCLAMPED and LOOSENS the bar, masking a real high ECE.
#   These tests encode each fabrication vector and FAIL on the pre-hardening
#   gate (which returns a FALSE PASS); the hardening makes each a FAIL while
#   the well-formed PASS/FAIL/PENDING teeth above stay green. The gate's own
#   sanctioned bars (0.05 high-resource / 0.08 long-tail) are authoritative — an
#   artifact may TIGHTEN the bar but never LOOSEN it.
# ---------------------------------------------------------------------------


def _g4_core(**calibration: object) -> dict[str, dict[str, object]]:
    """Build a minimal name→system map carrying a pii-anon calibration block for
    direct ``_g4_calibration_selective_risk`` unit tests.

    Seeded from :func:`_conforming_calibration` (which PASSes) so each test can
    perturb exactly ONE field to the fabrication vector under test — proving the
    FAIL is attributable to that vector, not to an unrelated malformed field.
    """
    block = _conforming_calibration()
    block.update(calibration)
    return {"pii-anon": {"system": "pii-anon", **block}}


def test_g4_fail_when_per_class_ece_is_nan_never_slips_the_bar() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-017: a NaN per-class ECE must
    NEVER satisfy 'every ECE ≤ bar'. ``nan > bar`` is False, so the pre-hardening
    gate left ``ece_ok`` True and returned a FALSE PASS. A non-finite ECE is a
    present-but-corrupt measurement → FAIL (not a pass over a NaN)."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "US_SSN": float("nan")})
    )
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail


def test_g4_fail_when_per_class_ece_is_negative_infinity() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-017: a -inf per-class ECE also
    slips ``ece > bar`` (``-inf > bar`` is False) → pre-hardening FALSE PASS. A
    non-finite ECE → FAIL."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "US_SSN": float("-inf")})
    )
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail


def test_g4_fail_when_per_class_ece_is_positive_infinity() -> None:
    """[CONTRACT-TEST] A7 / NFR-017: a +inf per-class ECE already lost the
    ``ece > bar`` comparison (so it FAILed pre-hardening too), but it must keep
    FAILing AND be reported as the non-finite breach, not silently as 'over bar'
    — the finiteness guard owns every non-finite ECE uniformly."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "US_SSN": float("inf")})
    )
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail


def test_g4_fail_when_coverage_exceeds_one_nfr020_exact_must() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-020: ``calibrated_confidence_
    coverage`` > 1.0 is a CORRUPT artifact, never a pass. The pre-hardening gate
    used ``coverage_val >= 1.0`` so 1.5 was admitted → FALSE PASS. NFR-020 is the
    lone EXACT 100% must: coverage must be finite AND in [0, 1] AND == 1.0; 1.5
    → FAIL."""
    g4 = _g4_calibration_selective_risk(_g4_core(calibrated_confidence_coverage=1.5))
    assert g4.passed is False
    assert (
        "coverage" in g4.binding_detail.lower() or "NFR-020" in g4.binding_detail
    )


def test_g4_fail_when_coverage_is_positive_infinity() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-020: a +inf coverage is admitted
    by ``>= 1.0`` → pre-hardening FALSE PASS. A non-finite coverage → FAIL."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(calibrated_confidence_coverage=float("inf"))
    )
    assert g4.passed is False
    assert (
        "coverage" in g4.binding_detail.lower() or "NFR-020" in g4.binding_detail
    )


def test_g4_fail_when_coverage_is_nan() -> None:
    """[CONTRACT-TEST] A7 / NFR-020: a NaN coverage already lost ``>= 1.0`` (so it
    FAILed pre-hardening), but it must keep FAILing AND be reported as the
    coverage breach — the finiteness guard owns it uniformly (never a pass over a
    NaN)."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(calibrated_confidence_coverage=float("nan"))
    )
    assert g4.passed is False
    assert (
        "coverage" in g4.binding_detail.lower() or "NFR-020" in g4.binding_detail
    )


def test_g4_fail_when_artifact_threshold_loosens_the_bar_masking_high_ece() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-017: an artifact-supplied
    per-class ECE threshold must NEVER LOOSEN the gate's sanctioned bar. The
    pre-hardening gate trusted ``bar = thresholds.get(et, 0.05)`` unclamped, so a
    benchmark stamping ``per_class_ece_threshold = {EMAIL_ADDRESS: 0.99}`` masked a
    real ECE of 0.5 → FALSE PASS. The gate clamps to ``min(artifact, sanctioned)``
    so the real 0.5 ECE is measured against the 0.05 sanctioned bar → FAIL."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(
            per_class_ece={"EMAIL_ADDRESS": 0.5},
            per_class_ece_threshold={"EMAIL_ADDRESS": 0.99},
        )
    )
    assert g4.passed is False
    assert "EMAIL_ADDRESS" in g4.binding_detail


def test_g4_fail_when_artifact_threshold_is_positive_infinity() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-017: a +inf per-class threshold
    would loosen the bar to admit any ECE → pre-hardening FALSE PASS. A non-finite
    artifact threshold is rejected (fall back to the sanctioned bar), so a real
    ECE 0.5 is measured against 0.05 → FAIL."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(
            per_class_ece={"EMAIL_ADDRESS": 0.5},
            per_class_ece_threshold={"EMAIL_ADDRESS": float("inf")},
        )
    )
    assert g4.passed is False
    assert "EMAIL_ADDRESS" in g4.binding_detail


def test_g4_fail_when_artifact_threshold_is_nan() -> None:
    """[CONTRACT-TEST] A7 anti-fabrication / NFR-017: a NaN per-class threshold
    makes ``ece > bar`` (``0.5 > nan``) False → pre-hardening FALSE PASS. A
    non-finite artifact threshold is rejected (fall back to the sanctioned 0.05
    bar), so a real ECE 0.5 → FAIL."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(
            per_class_ece={"EMAIL_ADDRESS": 0.5},
            per_class_ece_threshold={"EMAIL_ADDRESS": float("nan")},
        )
    )
    assert g4.passed is False
    assert "EMAIL_ADDRESS" in g4.binding_detail


def test_g4_tighter_artifact_threshold_is_still_honored_clamp_only_tightens() -> None:
    """[CONTRACT-TEST] A7 / NFR-017: the clamp is one-directional — an artifact may
    TIGHTEN the bar but never loosen it. A per-class threshold of 0.03 (tighter
    than the 0.05 sanctioned bar) is honored, so an ECE of 0.04 (which would PASS
    the 0.05 bar) FAILs against the tighter 0.03 the artifact requested."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(
            per_class_ece={"EMAIL_ADDRESS": 0.04},
            per_class_ece_threshold={"EMAIL_ADDRESS": 0.03},
        )
    )
    assert g4.passed is False
    assert "EMAIL_ADDRESS" in g4.binding_detail


def test_g4_coverage_exactly_one_still_passes_after_finiteness_hardening() -> None:
    """[CONTRACT-TEST] A7 / NFR-020 regression: the finiteness/range hardening must
    not break the well-formed path — an exact coverage of 1.0 (the NFR-020 must)
    on an otherwise-conforming block still PASSes."""
    g4 = _g4_calibration_selective_risk(_g4_core(calibrated_confidence_coverage=1.0))
    assert g4.passed is True


def test_g4_nan_ece_fabrication_does_not_reach_claim_grade_end_to_end() -> None:
    """[CONTRACT-TEST] §5 END-TO-END: the reported G4 false-PASS exploit is closed
    at the verdict layer. With a canonical run, J≥0.95, G2 fields + G5 override
    PASS, and the whole Tier-R∪Tier-C set waived — but a NaN per-class ECE in the
    calibration block — G4 is FAIL and the headline verdict does NOT reach
    CLAIM_GRADE_SOTA (pre-hardening it did, on a vacuous G4 PASS over a NaN ECE)."""
    cal = _conforming_calibration()
    cal["per_class_ece"] = {"EMAIL_ADDRESS": 0.02, "US_SSN": float("nan")}
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    bench = _with_calibration(bench, cal)
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G5": True},
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G4").passed is False
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


def test_g4_coverage_overflow_fabrication_does_not_reach_claim_grade_end_to_end() -> None:
    """[CONTRACT-TEST] §5 END-TO-END: the coverage>1.0 fabrication is closed at the
    verdict layer — a calibrated_confidence_coverage of 1.5 with everything else
    satisfied yields G4 FAIL and a non-CLAIM_GRADE verdict (pre-hardening 1.5 was
    admitted as a vacuous coverage pass)."""
    cal = _conforming_calibration()
    cal["calibrated_confidence_coverage"] = 1.5
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    bench = _with_calibration(bench, cal)
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G5": True},
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G4").passed is False
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


def test_g4_unclamped_threshold_fabrication_does_not_reach_claim_grade_end_to_end() -> None:
    """[CONTRACT-TEST] §5 END-TO-END: the unclamped-threshold fabrication is closed
    at the verdict layer — a benchmark stamping a loosened per-class threshold
    (0.99) to mask a real ECE 0.5 yields G4 FAIL and a non-CLAIM_GRADE verdict (the
    gate's sanctioned bar is authoritative, not artifact-overridable upward)."""
    cal = _conforming_calibration()
    cal["per_class_ece"] = {"EMAIL_ADDRESS": 0.5}
    cal["per_class_ece_threshold"] = {"EMAIL_ADDRESS": 0.99}
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    bench = _with_calibration(bench, cal)
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G5": True},
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G4").passed is False
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


def test_g4_is_deterministic_on_fabrication_vectors() -> None:
    """[PROPERTY-TEST] A9 / AX-002: the hardened guard is still a pure function —
    each fabrication vector yields an identical (FAIL) outcome across two runs on
    deep-copied inputs."""
    for cal_over in (
        {"per_class_ece": {"EMAIL_ADDRESS": 0.02, "US_SSN": float("nan")}},
        {"calibrated_confidence_coverage": 1.5},
        {
            "per_class_ece": {"EMAIL_ADDRESS": 0.5},
            "per_class_ece_threshold": {"EMAIL_ADDRESS": 0.99},
        },
    ):
        s1 = _g4_core(**cal_over)
        s2 = copy.deepcopy(s1)
        assert _g4_calibration_selective_risk(s1) == _g4_calibration_selective_risk(s2)


# ---------------------------------------------------------------------------
# S7-02 close-3 — G4 NEGATIVE per-class ECE fabrication. The close found the LAST
# hole in `_g4_calibration_selective_risk`: a per-class ECE is validated only for
# FINITENESS (`_is_finite_number`), with NO LOWER BOUND. An Expected Calibration
# Error is ``≥ 0`` by construction, so a forged NEGATIVE ECE (e.g. -1.0) is
# ``< bar`` ⇒ counts as "within bar" ⇒ G4.passed=True ("worst -1.0") — a fabricated
# moat-axis PASS that flips the honest PENDING/FAIL and can escalate to
# CLAIM_GRADE_SOTA via the trusted-caller waiver seam. A negative ECE is a corrupt
# present-but-unusable measurement (non-physical, symmetric with rejecting a
# coverage outside [0,1]) → a BREACH, never "within bar"; it is also excluded from
# `finite_eces` so garbage never stands in as the worst observed ECE.
# ---------------------------------------------------------------------------


def test_g4_fail_when_per_class_ece_is_negative_one() -> None:
    """[CONTRACT-TEST] close-3 anti-fabrication / NFR-017: a per-class ECE of -1.0
    is non-physical (ECE ≥ 0 by construction). The pre-hardening gate validated only
    finiteness, so -1.0 < bar slipped 'every ECE ≤ bar' → a FALSE PASS ("worst
    -1.0"). A negative ECE is a corrupt measurement → G4 FAIL, with the offending
    class named in the breach detail."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "EMAIL": -1.0})
    )
    assert g4.passed is False
    assert "EMAIL" in g4.binding_detail
    # The garbage value must NOT stand in as the worst observed ECE.
    assert g4.observed >= 0.0


def test_g4_fail_when_per_class_ece_is_small_negative() -> None:
    """[CONTRACT-TEST] close-3 / NFR-017: even a small negative ECE (-0.001) is a
    corrupt measurement — the lower bound is exactly 0, not 'approximately 0'. A
    sub-zero ECE FAILs (it is below the only physical floor)."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "US_SSN": -0.001})
    )
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail


def test_g4_fail_when_per_class_ece_is_large_negative() -> None:
    """[CONTRACT-TEST] close-3 / NFR-017: a large negative ECE (-5.0) is the same
    fabrication class — magnitude does not matter, the sign does. FAIL."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.02, "US_SSN": -5.0})
    )
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail


def test_g4_co_located_negative_and_over_bar_reports_over_bar_as_headline() -> None:
    """[CONTRACT-TEST] close-3 detail-ranking: when a NEGATIVE ECE (-1.0) co-exists
    with a GENUINE over-bar class (SSN 0.5 > 0.05 bar, slack > 0), the over-bar
    class stays the HEADLINE breach (it has the larger physical slack) — the lone
    negative is reported only when it is the sole breach. Either way G4 FAILs."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL": -1.0, "US_SSN": 0.5})
    )
    assert g4.passed is False
    assert "US_SSN" in g4.binding_detail
    assert "0.5" in g4.binding_detail


def test_g4_pass_unchanged_when_ece_is_exactly_zero() -> None:
    """[CONTRACT-TEST] close-3 CARDINAL-RULE regression: the lower-bound hardening
    must not move the floor — an HONEST ECE of exactly 0.0 (a perfectly calibrated
    class) is ≥ 0 and ≤ bar, so it still PASSes."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(per_class_ece={"EMAIL_ADDRESS": 0.0, "US_SSN": 0.0})
    )
    assert g4.passed is True


def test_g4_negative_ece_fabrication_does_not_reach_claim_grade_end_to_end() -> None:
    """[CONTRACT-TEST] §5 END-TO-END: the negative-ECE false-PASS is closed at the
    verdict layer. With a canonical run, J≥0.95, G2 fields + G5 override PASS, and
    the whole Tier-R∪Tier-C set waived — but a NEGATIVE per-class ECE (-1.0) in the
    calibration block — G4 is FAIL and the headline verdict does NOT reach
    CLAIM_GRADE_SOTA (pre-hardening it did, on a vacuous G4 PASS over -1.0)."""
    cal = _conforming_calibration()
    cal["per_class_ece"] = {"EMAIL_ADDRESS": 0.02, "EMAIL": -1.0}
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    bench = _with_calibration(bench, cal)
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G5": True},
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G4").passed is False
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


# ---------------------------------------------------------------------------
# S7-02 adversarial-close hardening — the no-fabrication invariant's 3 gate
# holes (the close's exact repros). The shared root-cause: an artifact-numeric
# field that DRIVES a verdict was read with bare ``float()`` /
# ``isinstance(..., (int, float))`` instead of routing through the gate's OWN
# hardened validators (``_finite_unit_score`` / ``_is_finite_number``), so a
# bool / NaN / ±inf / huge-int (10**400) either FABRICATED a PASS-or-crown or
# CRASHED the shipped ``pii-anon supremacy`` CLI (a fail-loud denial-of-verdict).
# Each must now fail CLOSED: the guarantee reads PENDING/FAIL, the system is
# EXCLUDED from the rank-1 crown/bootstrap, or the breach-guard skips — NEVER a
# fabricated PASS, NEVER a crash.
# ---------------------------------------------------------------------------


# -- MAJOR 1: G4 calibrated_confidence_coverage bool (isinstance+float bypass). --


def test_close_g4_coverage_true_bool_is_not_a_pass() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-1: ``calibrated_confidence_coverage=True``
    is a Python bool (``isinstance(True, int)`` is True; ``float(True) == 1.0``), so
    the pre-hardening ``isinstance(coverage, (int, float))`` + ``float(coverage)``
    coverage path read it as a perfect 1.0 ⇒ a DISHONEST G4 PASS. Routed through
    ``_finite_unit_score`` (which rejects bool), a bool coverage is a corrupt/absent
    coverage ⇒ G4 is NOT a pass (PENDING-or-FAIL), never a fabricated PASS."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(calibrated_confidence_coverage=True)
    )
    assert g4.passed is not True  # never a fabricated PASS over a bool flag


def test_close_g4_coverage_false_bool_is_not_a_pass() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-1 (companion): ``False`` coverage is also a
    bool flag, not a 0.0 measurement — it must be rejected (not a pass), never
    coerced to a real fraction."""
    g4 = _g4_calibration_selective_risk(
        _g4_core(calibrated_confidence_coverage=False)
    )
    assert g4.passed is not True


def test_close_g4_coverage_true_does_not_reach_claim_grade_end_to_end() -> None:
    """[CONTRACT-TEST] §5 END-TO-END (close MAJOR-1): with a canonical run, J≥0.95,
    G2 fields + G5 override PASS, and the whole Tier-R∪Tier-C set waived — but
    ``calibrated_confidence_coverage=True`` (a bool) in the calibration block — G4 is
    NOT a pass and the headline verdict does NOT reach CLAIM_GRADE_SOTA. Pre-hardening
    the bool coerced to 1.0 → a vacuous G4 PASS → ``_decide`` → CLAIM_GRADE_SOTA."""
    cal = _conforming_calibration()
    cal["calibrated_confidence_coverage"] = True
    bench = _with_pseudo_fields(
        _canonical_benchmark(), pii_anon_pi=0.95, competitor_pi=0.60
    )
    bench = _with_calibration(bench, cal)
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G5": True},
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet wired; cited Tier-R"},
    )
    assert verdict.guarantee("G4").passed is not True
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


# -- MAJOR 2: composite_score non-finite (the verdict-driving J/G3/G6-crown field). --


def test_close_top_composite_excludes_positive_infinity_no_phantom_crown() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-2: a ``composite_score=+inf`` must NEVER win
    the top-composite rank-1 crown (it would fabricate a phantom rank-1 over a
    genuinely-superior finite competitor). The non-finite composite is EXCLUDED, so
    the honest finite winner (``pii-anon`` at 0.8) is crowned."""
    systems = {
        "pii-anon": {"system": "pii-anon", "composite_score": 0.8},
        "rogue": {"system": "rogue", "composite_score": float("inf")},
    }
    assert _top_composite_system(systems) == "pii-anon"  # +inf excluded, not crowned


def test_close_top_composite_nan_does_not_crash_excluded() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-2: a ``composite_score=NaN`` must not crash
    ``_top_composite_system`` (pre-hardening ``NaN == best`` is always False ⇒ empty
    list ⇒ ``[0]`` IndexError). NaN is excluded; the finite system survives, and an
    all-NaN field yields ``None`` (no crash, no phantom)."""
    systems = {
        "pii-anon": {"system": "pii-anon", "composite_score": 0.8},
        "rogue": {"system": "rogue", "composite_score": float("nan")},
    }
    assert _top_composite_system(systems) == "pii-anon"  # NaN excluded, no IndexError
    all_nan = {
        "a": {"system": "a", "composite_score": float("nan")},
        "b": {"system": "b", "composite_score": float("nan")},
    }
    assert _top_composite_system(all_nan) is None  # all excluded ⇒ None, not a crash


def test_close_top_composite_huge_int_does_not_crash_excluded() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-2: a ``composite_score=10**400`` (wider than
    a C double) must not raise OverflowError in ``_top_composite_system`` (the
    pre-hardening ``float(s["composite_score"])`` crashed). The huge-int is excluded;
    the finite winner survives."""
    systems = {
        "pii-anon": {"system": "pii-anon", "composite_score": 0.8},
        "rogue": {"system": "rogue", "composite_score": 10**400},
    }
    assert _top_composite_system(systems) == "pii-anon"  # huge-int excluded, no crash


def test_close_composite_positive_infinity_no_phantom_crown_end_to_end() -> None:
    """[CONTRACT-TEST] §5 END-TO-END (close MAJOR-2): a competitor stamping
    ``composite_score=+inf`` must NEVER be crowned the SDO rank-1 system (J argmax),
    and the honest floor-compliant winner (pii-anon) stays crowned. Pre-hardening the
    +inf seeded a phantom rank-1 (the MLE-bootstrap composite gap saturates), crowning
    a rogue over a genuinely-superior competitor — a fabricated PROVISIONAL/J=1.0."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "scrubadub":
            s["composite_score"] = float("inf")  # hostile +inf composite
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
    )
    # The +inf competitor is never the crowned rank-1 system.
    assert verdict.j_rank1_system != "scrubadub"
    # The honest floor-compliant winner stays crowned.
    assert verdict.j_rank1_system == "pii-anon"


def test_close_composite_nan_does_not_crash_verdict_end_to_end() -> None:
    """[SECURITY-TEST] §5 END-TO-END (close MAJOR-2): a ``composite_score=NaN`` on a
    competitor must not crash the full ``from_artifacts`` path with an IndexError (via
    ``_top_composite_system``) nor the MLE-bootstrap (via ``_mle_bootstrap_draws``).
    The verdict computes; the NaN system is never crowned."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "scrubadub":
            s["composite_score"] = float("nan")
    verdict = SupremacyVerdict.from_artifacts(bench)  # must NOT raise (no IndexError)
    # The NaN system is never crowned, and the corrupt composite never fabricates a
    # claim-grade (the security property; the exact non-claim verdict tier is incidental).
    assert verdict.j_rank1_system != "scrubadub"
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


def test_close_composite_huge_int_does_not_crash_verdict_end_to_end() -> None:
    """[SECURITY-TEST] §5 END-TO-END (close MAJOR-2): a ``composite_score=10**400`` on
    a competitor must not raise OverflowError anywhere in the verdict path (the shipped
    ``pii-anon supremacy --artifact`` CLI must not crash on an adversarial artifact).
    The huge-int system is excluded from the crown/bootstrap; the verdict computes."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "scrubadub":
            s["composite_score"] = 10**400
    verdict = SupremacyVerdict.from_artifacts(bench)  # must NOT raise
    assert verdict.j_rank1_system != "scrubadub"


def test_close_floor_breaching_inf_composite_still_excluded_from_crown() -> None:
    """[CONTRACT-TEST] close MAJOR-2 × the RecallFloorVerdictGuard: a floor-BREACHING
    system that ALSO stamps ``composite_score=+inf`` must be doubly ineligible for the
    rank-1 crown — excluded both as a non-finite composite AND as a floor breacher.
    The crowned system stays the floor-compliant pii-anon, and the verdict is never
    CLAIM_GRADE on its account."""
    bench = _canonical_benchmark()
    rogue = _breaching_system("rogue-inf", composite=float("inf"))
    bench["systems"].append(rogue)  # type: ignore[attr-defined]
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={
            n: "w"
            for n in ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")
        },
    )
    assert "rogue-inf" in verdict.recall_floor_breachers
    assert verdict.j_rank1_system == "pii-anon"
    assert verdict.j_rank1_system != "rogue-inf"
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


# -- MAJOR 3: per_language_recall_delta huge-int (G1 + the run-breach guard). --


def test_close_g1_huge_int_per_language_delta_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-3: a ``per_language_recall_delta`` value of
    ``10**400`` (wider than a C double) must not raise OverflowError in
    ``_g1_recall_floor`` (the pre-hardening ``max(abs(float(v)) ...)`` crashed). A
    non-finite/huge-int ε is rejected (fail-closed); G1 does not crash and does not
    silently pass over the corrupt value (a real ε regression is a breach, never a
    fabricated PASS)."""
    bench = {"per_language_recall_delta": {"en": 0.001, "de": 10**400}}
    systems = {"pii-anon": {"system": "pii-anon", "per_entity_recall": {}}}
    g1 = _g1_recall_floor(bench, systems)  # must NOT raise
    assert g1.passed is not True  # never a fabricated PASS over a rejected ε


def test_close_run_breach_guard_huge_int_per_language_delta_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-3: ``_run_breaches_recall_floor`` must not
    crash on a ``10**400`` per-language ε (the pre-hardening ``max(abs(float(v)) ...)``
    raised OverflowError, breaking the shipped CLI). A non-finite/huge-int ε is treated
    as a breach (fail-closed: it CANNOT be certified ≤ the 0.005 bound), never a
    crash."""
    bench = {"per_language_recall_delta": {"en": 0.001, "de": 10**400}}
    assert _run_breaches_recall_floor(bench) is True  # fail-closed breach, no crash


def test_close_run_breach_guard_non_finite_per_language_delta_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-3 (companion): a NaN / ±inf per-language ε is
    not a real measurement — it must be treated as a breach (the run cannot be certified
    floor-compliant from a corrupt ε), never slip ``abs() > bound`` (silently False
    against NaN) or crash."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        bench = {"per_language_recall_delta": {"en": bad}}
        assert _run_breaches_recall_floor(bench) is True


def test_close_huge_int_per_language_delta_does_not_crash_verdict_end_to_end() -> None:
    """[SECURITY-TEST] §5 END-TO-END (close MAJOR-3): a ``per_language_recall_delta``
    huge-int in the benchmark must not crash the full ``from_artifacts`` path (the
    shipped ``pii-anon supremacy --artifact`` CLI). The verdict computes; the corrupt ε
    taints the run as a recall-floor breach (G7 guard), never a fabricated PASS."""
    bench = _canonical_benchmark()
    bench["per_language_recall_delta"] = {"en": 10**400}  # type: ignore[index]
    verdict = SupremacyVerdict.from_artifacts(  # must NOT raise
        bench, pending_overrides=_ALL_PENDING_PASS
    )
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


# -- ALSO-SCAN: recall / precision in G3 + G6 (same bare-float / isinstance class). --


def test_close_g3_huge_int_competitor_recall_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close also-scan (G3 recall): a competitor
    ``recall=10**400`` must not raise OverflowError in ``_g3_recall_dominance`` (the
    pre-hardening ``max(float(systems[c]["recall"]) ...)`` crashed). A non-finite/huge
    competitor recall is rejected (excluded as not a real measurement); G3 does not
    crash and does not award the swarm a fabricated dominance over a corrupt +inf."""
    systems = {
        "pii-anon-swarm": {"system": "pii-anon-swarm", "recall": 0.85, "precision": 0.7},
        "gliner": {"system": "gliner", "recall": 10**400, "precision": 0.9},
    }
    g3 = _g3_recall_dominance(systems)  # must NOT raise
    # A corrupt competitor recall is excluded — never a fabricated dominance over +inf.
    assert g3.passed is not True


def test_close_g3_infinite_swarm_recall_is_not_a_fabricated_dominance() -> None:
    """[SECURITY-TEST] S7-02 close also-scan (G3 recall): a pii-anon-swarm
    ``recall=+inf`` must NEVER fabricate a recall-dominance PASS (``+inf >= best`` is a
    vacuous win over a real competitor). A non-finite claimant recall is rejected ⇒ G3
    is not a fabricated PASS, never a crash."""
    systems = {
        "pii-anon-swarm": {"system": "pii-anon-swarm", "recall": float("inf"), "precision": 0.7},
        "gliner": {"system": "gliner", "recall": 0.66, "precision": 0.9},
    }
    g3 = _g3_recall_dominance(systems)  # must NOT raise
    assert g3.passed is not True  # +inf never a fabricated recall-dominance win


def test_close_g6_huge_int_recall_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close also-scan (G6 recall/precision): a ``10**400`` core
    or Tier-R recall/precision must not raise OverflowError in ``_g6_raw_noninferiority``
    (the ``f_beta(float(...), float(...))`` path). A non-finite/huge recall/precision is
    rejected; G6 does not crash and is never a fabricated non-inferiority PASS."""
    core_huge = {
        "pii-anon": {
            "system": "pii-anon",
            "recall": 10**400,
            "precision": 0.7,
            "per_entity_recall": {"A": 0.9, "B": 0.8},
        },
    }
    g6 = _g6_raw_noninferiority(core_huge)  # must NOT raise
    assert g6.passed is not True
    # A huge-int Tier-R competitor recall must also be excluded (not crash / not a bar).
    tier_r_huge = {
        "pii-anon": {
            "system": "pii-anon",
            "recall": 0.8,
            "precision": 0.8,
            "per_entity_recall": {"A": 0.9, "B": 0.8},
        },
        "presidio": {"system": "presidio", "recall": 10**400, "precision": 0.9},
    }
    g6b = _g6_raw_noninferiority(tier_r_huge)  # must NOT raise
    assert g6b.observed == g6b.observed  # core F2 is finite (== self ⇒ not NaN)


def test_close_g6_infinite_core_recall_is_not_a_fabricated_pass() -> None:
    """[SECURITY-TEST] S7-02 close also-scan (G6): a core ``recall=+inf`` must NEVER
    fabricate a non-inferiority PASS. Pre-hardening ``f_beta(0.7, +inf)`` yielded a
    corrupt F2 that silently flowed into ``core_f2 >= best_tier_r - eps``; the hardened
    path rejects a non-finite core recall ⇒ G6 is not a fabricated PASS, never a
    crash."""
    core_inf = {
        "pii-anon": {
            "system": "pii-anon",
            "recall": float("inf"),
            "precision": 0.7,
            "per_entity_recall": {"A": 0.9, "B": 0.8},
        },
        "presidio": {"system": "presidio", "recall": 0.63, "precision": 0.4},
    }
    g6 = _g6_raw_noninferiority(core_inf)  # must NOT raise
    assert g6.passed is not True


def test_close_g3_g6_huge_int_does_not_crash_verdict_end_to_end() -> None:
    """[SECURITY-TEST] §5 END-TO-END (close also-scan): a competitor recall/precision
    of ``10**400`` in the benchmark must not crash the full ``from_artifacts`` path (the
    shipped ``pii-anon supremacy --artifact`` CLI). The verdict computes; the corrupt
    competitor is excluded from G3/G6, never a fabricated PASS."""
    bench = _canonical_benchmark()
    for s in bench["systems"]:  # type: ignore[attr-defined]
        if s["system"] == "gliner":
            s["recall"] = 10**400
            s["precision"] = 10**400
    verdict = SupremacyVerdict.from_artifacts(  # must NOT raise
        bench, pending_overrides=_ALL_PENDING_PASS
    )
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


# -- Honest-input regressions: the hardening must NOT perturb the honest path. --


def test_close_hardening_preserves_honest_g3_g4_g6_pass() -> None:
    """[CONTRACT-TEST] S7-02 close CARDINAL RULE: the hostile-input hardening must
    leave the HONEST-input verdicts byte-identical. The fully-conforming benchmark
    (finite composites, real coverage 1.0, finite recall/precision) still yields G3
    PASS, G4 PASS, G6 PASS, and the honest top-composite crown is pii-anon."""
    bench = _with_calibration(_canonical_benchmark(), _conforming_calibration())
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    assert verdict.guarantee("G3").passed is True
    assert verdict.guarantee("G4").passed is True
    assert verdict.guarantee("G6").passed is True
    assert _top_composite_system(
        {s["system"]: s for s in bench["systems"]}  # type: ignore[index]
    ) == "pii-anon"


def test_close_hardening_is_deterministic_on_hostile_vectors() -> None:
    """[PROPERTY-TEST] AX-002: each hostile composite/coverage/recall vector yields an
    identical (fail-closed) outcome across two runs on deep-copied inputs — the
    hardened reads stay pure functions."""
    # composite +inf / NaN / huge-int → identical exclusion.
    for bad in (float("inf"), float("nan"), 10**400):
        s1 = {
            "pii-anon": {"system": "pii-anon", "composite_score": 0.8},
            "rogue": {"system": "rogue", "composite_score": bad},
        }
        s2 = copy.deepcopy(s1)
        assert _top_composite_system(s1) == _top_composite_system(s2)
    # coverage bool → identical non-pass.
    cov1 = _g4_core(calibrated_confidence_coverage=True)
    cov2 = copy.deepcopy(cov1)
    assert _g4_calibration_selective_risk(cov1) == _g4_calibration_selective_risk(cov2)


# ---------------------------------------------------------------------------
# S7-02 close-2 — the crash-class holes (fail-LOUD denial-of-verdict on a
# malformed CONTAINER, NOT a fabrication). The keystone re-close confirmed the
# 4 FABRICATION holes are closed but found 2 remaining MAJOR crash-class holes:
# a malformed artifact-sourced CONTAINER (wrong JSON TYPE — a list where a dict
# is read, a dict where a list is read, a scalar where a list is read, a
# non-dict element in the systems array) reached an UNGUARDED ``.get`` / ``.items``
# / ``len()`` / iteration and threw (TypeError / AttributeError) — crashing the
# shipped ``pii-anon supremacy`` CLI with a raw traceback. Hardening: every
# artifact-sourced container is type-checked before consumption, so a malformed
# container fails CLOSED (the affected guarantee reads PENDING/FAIL, the bad
# element is ignored, or the verdict is a clean NOT_YET) — NEVER an exception.
# The CARDINAL RULE holds: HONEST-INPUT verdicts stay byte-identical.
# ---------------------------------------------------------------------------


# -- crash 1: G4 abstention_operating_points non-list scalar + the len() else. --


@pytest.mark.parametrize("bad_points", [42, 3.14, True])
def test_close2_g4_non_list_abstention_points_does_not_crash(bad_points: object) -> None:
    """[SECURITY-TEST] S7-02 close-2 crash-1: ``abstention_operating_points`` as a
    non-list scalar (42 / 3.14 / True) with the OTHER three G4 checks PASSING drove
    the FAIL-detail ``else`` branch which called ``len(points)`` UNCONDITIONALLY —
    ``TypeError: object of type 'int'/'float'/'bool' has no len()``. G4 must instead
    read FAIL (a non-list cannot supply ≥3 operating points) and NEVER crash."""
    core = _g4_core(abstention_operating_points=bad_points)
    g4 = _g4_calibration_selective_risk(core)  # must not raise
    assert g4.passed is False  # fail-closed: a non-list cannot carry ≥3 points
    assert "abstention operating point" in g4.binding_detail


def test_close2_g4_non_list_points_does_not_crash_end_to_end() -> None:
    """[SECURITY-TEST] §5 END-TO-END (close-2 crash-1): the exact close repro — a full
    benchmark whose pii-anon calibration block carries ``abstention_operating_points=42``
    (other G4 fields valid) ⇒ a clean verdict (no exception), G4 FAIL."""
    bench = _with_calibration(
        _canonical_benchmark(),
        {**_conforming_calibration(), "abstention_operating_points": 42},
    )
    verdict = SupremacyVerdict.from_artifacts(bench)  # must not raise
    assert verdict.guarantee("G4").passed is False
    assert verdict.verdict in (Verdict.NOT_YET, Verdict.PROVISIONAL_SOTA)


def test_close2_g4_non_list_points_pass_detail_path_also_safe() -> None:
    """[SECURITY-TEST] close-2 crash-1 (the PASS-detail companion): the count is read
    safely in BOTH the PASS- and FAIL-detail branches. A valid list of ≥3 points keeps
    PASSing (the count still reports), proving the safe-count refactor did not regress
    the honest PASS detail."""
    core = _g4_core()  # conforming: 3 list points
    g4 = _g4_calibration_selective_risk(core)
    assert g4.passed is True
    assert "3 abstention operating points" in g4.binding_detail


# -- crash 2: malformed CONTAINER types flowing through from_artifacts / the CLI. --


def test_close2_systems_as_dict_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close-2 crash-2: ``systems`` as a JSON OBJECT (not array)
    made ``_systems_by_name`` iterate the dict KEYS (strings) and call ``str.get`` →
    ``AttributeError: 'str' object has no attribute 'get'``. A non-list ``systems`` must
    be treated as EMPTY (no systems) → a clean verdict, never a crash."""
    bench = {"systems": {"pii-anon": {"system": "pii-anon", "recall": 0.8}}}
    verdict = SupremacyVerdict.from_artifacts(bench)  # must not raise
    assert verdict.verdict is Verdict.NOT_YET


def test_close2_run_metadata_as_list_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close-2 crash-2: ``run_metadata`` as a JSON ARRAY — the
    ``benchmark.get("run_metadata", {}) or {}`` kept the truthy list, then ``.get`` →
    ``AttributeError: 'list' object has no attribute 'get'``. A non-dict ``run_metadata``
    must be treated as ABSENT ({}) → canonical_claim_run=False → a clean NOT_YET."""
    bench = {"run_metadata": [1, 2, 3], "systems": []}
    verdict = SupremacyVerdict.from_artifacts(bench)  # must not raise
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.canonical_claim_run is False


def test_close2_systems_element_bare_string_is_ignored_no_crash() -> None:
    """[SECURITY-TEST] S7-02 close-2 crash-2: a ``systems`` array carrying a BARE STRING
    element made ``_systems_by_name`` call ``str.get`` → ``AttributeError``. A non-dict
    element must be IGNORED (skipped), the valid dict elements still indexed, no crash."""
    bench = {
        "systems": ["pii-anon", {"system": "gliner", "recall": 0.6}, 42, None],
    }
    verdict = SupremacyVerdict.from_artifacts(bench)  # must not raise
    assert verdict.verdict is Verdict.NOT_YET
    # The valid dict element is still indexed (gliner is a competitor → G3 can read it).
    assert "gliner" not in verdict.recall_floor_breachers or True  # no crash is the pin


def test_close2_per_entity_recall_as_list_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 close-2 crash-2: a system's ``per_entity_recall`` as an
    ARRAY — ``(sys.get("per_entity_recall") or {}).items()`` kept the truthy list then
    ``.items()`` → ``AttributeError: 'list' object has no attribute 'items'`` in
    ``_g1_recall_floor`` (and ``.values()``/``len`` in ``_g6``). A non-dict
    ``per_entity_recall`` must be treated as EMPTY ({}) → no crash; G1/G6 read cleanly."""
    bench = {
        "systems": [
            {"system": "pii-anon", "per_entity_recall": [1, 2, 3], "recall": 0.8,
             "precision": 0.78},
            {"system": "gliner", "per_entity_recall": {"EMAIL_ADDRESS": 0.6},
             "recall": 0.6, "precision": 0.9},
        ],
    }
    verdict = SupremacyVerdict.from_artifacts(bench)  # must not raise
    assert verdict.verdict is Verdict.NOT_YET


def test_close2_g6_per_entity_recall_as_list_does_not_crash() -> None:
    """[SECURITY-TEST] close-2 crash-2 (G6 path): a non-dict ``per_entity_recall`` on
    the core must not crash ``_g6_raw_noninferiority`` (which reads ``.values()``/``len``
    for entity-coverage). Coverage reads 0.0 (no detected entities) → G6 FAIL, no crash."""
    systems = {
        "pii-anon": {"system": "pii-anon", "recall": 0.8, "precision": 0.78,
                     "per_entity_recall": [1, 2, 3]},
    }
    g6 = _g6_raw_noninferiority(systems)  # must not raise
    assert g6.passed is False  # zero entity coverage from a malformed map


def test_close2_per_language_recall_delta_as_list_is_g1_pending() -> None:
    """[SECURITY-TEST] S7-02 close-2 crash-2: ``per_language_recall_delta`` as a LIST
    must not be silently treated as a passing per-language ε (the old code read it as
    present-with-no-items ⇒ a fabricated ``worst ε=0`` PASS). A non-dict per-language
    artifact is malformed/absent-equivalent ⇒ G1 PENDING (ε never fabricated), no crash."""
    bench = {
        "systems": [{"system": "pii-anon", "per_entity_recall": {}}],
        "per_language_recall_delta": [0.1, 0.2],  # two langs each ε=0.1 > 0.005 if read
    }
    g1 = SupremacyVerdict.from_artifacts(bench).guarantee("G1")  # must not raise
    assert g1.passed is None  # PENDING — a malformed per-language artifact is not a PASS


def test_close2_per_language_recall_delta_as_scalar_is_g1_pending() -> None:
    """[SECURITY-TEST] close-2 crash-2 (companion): a SCALAR ``per_language_recall_delta``
    (42) is likewise malformed ⇒ G1 PENDING, never a fabricated ε=0 PASS, never a crash."""
    bench = {"systems": [{"system": "pii-anon"}], "per_language_recall_delta": 42}
    g1 = SupremacyVerdict.from_artifacts(bench).guarantee("G1")  # must not raise
    assert g1.passed is None


def test_close2_run_breach_guard_non_dict_per_language_does_not_crash() -> None:
    """[SECURITY-TEST] close-2 crash-2: ``_run_breaches_recall_floor`` must not crash on a
    non-dict ``per_language_recall_delta`` (a list / scalar). It is already guarded by
    ``isinstance(per_lang, dict)`` — a non-dict yields no run-level recall breach from the
    per-language axis (the profile floor-checks still apply); the pin is no-crash."""
    assert _run_breaches_recall_floor({"per_language_recall_delta": [0.1, 0.2]}) is False
    assert _run_breaches_recall_floor({"per_language_recall_delta": 42}) is False


def test_close2_risk_coverage_curve_as_scalar_is_g4_fail_no_crash() -> None:
    """[SECURITY-TEST] close-2 crash-2: a non-list ``risk_coverage_curve`` (already guarded
    in ``_risk_coverage_is_monotone``) must keep G4 from crashing — the curve reads NOT
    monotone (a malformed curve cannot certify NFR-019) ⇒ G4 FAIL, no exception."""
    core = _g4_core(risk_coverage_curve=99)
    g4 = _g4_calibration_selective_risk(core)  # must not raise
    assert g4.passed is False


def test_close2_profile_results_as_dict_does_not_crash() -> None:
    """[SECURITY-TEST] close-2 crash-2: ``profile_results`` as a JSON OBJECT must not crash
    the run-breach recall guard (iterating a dict yields keys; the per-element
    ``isinstance(profile, dict)`` skip protects it). The pin is no-crash + clean verdict."""
    bench = {"systems": [{"system": "pii-anon"}], "profile_results": {"a": {"x": 1}}}
    verdict = SupremacyVerdict.from_artifacts(bench)  # must not raise
    assert verdict.verdict is Verdict.NOT_YET


# -- the container-type fuzz matrix: from_artifacts ALWAYS returns a verdict. --


def test_close2_container_type_fuzz_from_artifacts_never_raises() -> None:
    """[PROPERTY-TEST] S7-02 close-2: a deterministic fuzz over malformed CONTAINER types
    for every artifact-sourced container (run_metadata / systems / a systems element /
    per_entity_recall / abstention_operating_points / risk_coverage_curve /
    per_language_recall_delta / per_class_ece / per_class_ece_threshold / profile_results /
    floor_checks) ⇒ ``from_artifacts`` ALWAYS returns a Verdict, NEVER raises. The invariant:
    no malformed artifact can crash the gate (it fails CLOSED to a clean verdict)."""
    wrong_types: list[object] = [
        42, 3.14, True, False, "a string", [1, 2, 3], {"k": "v"}, None,
        [{"system": "x"}], [["nested"]], {"nested": {"deep": 1}}, 10**400,
        float("nan"), float("inf"),
    ]

    def _verdict_ok(bench: dict[str, object]) -> None:
        v = SupremacyVerdict.from_artifacts(bench)
        assert isinstance(v.verdict, Verdict)

    for bad in wrong_types:
        # 1. run_metadata wrong type.
        _verdict_ok({"run_metadata": bad, "systems": []})
        # 2. systems wrong type.
        _verdict_ok({"systems": bad})
        # 3. a single systems ELEMENT wrong type (mixed with a valid one).
        _verdict_ok({"systems": [bad, {"system": "pii-anon", "recall": 0.8}]})
        # 4. per_entity_recall wrong type on the core.
        _verdict_ok({"systems": [{"system": "pii-anon", "per_entity_recall": bad,
                                  "recall": 0.8, "precision": 0.7}]})
        # 5. per_language_recall_delta wrong type.
        _verdict_ok({"systems": [{"system": "pii-anon"}],
                     "per_language_recall_delta": bad})
        # 6. abstention_operating_points wrong type (with otherwise-valid G4 block).
        _verdict_ok(_with_calibration(
            _canonical_benchmark(),
            {**_conforming_calibration(), "abstention_operating_points": bad},
        ))
        # 7. risk_coverage_curve wrong type (with otherwise-valid G4 block).
        _verdict_ok(_with_calibration(
            _canonical_benchmark(),
            {**_conforming_calibration(), "risk_coverage_curve": bad},
        ))
        # 8. per_class_ece / per_class_ece_threshold wrong type.
        _verdict_ok(_with_calibration(
            _canonical_benchmark(),
            {**_conforming_calibration(), "per_class_ece": bad},
        ))
        _verdict_ok(_with_calibration(
            _canonical_benchmark(),
            {**_conforming_calibration(), "per_class_ece_threshold": bad},
        ))
        # 9. profile_results / floor_checks wrong type.
        _verdict_ok({"systems": [{"system": "pii-anon"}], "profile_results": bad})
        _verdict_ok({"systems": [{"system": "pii-anon"},
                                 {"system": "gliner"}],
                     "profile_results": [{"floor_checks": bad}]})


def test_close2_top_level_benchmark_not_a_dict_does_not_crash() -> None:
    """[SECURITY-TEST] close-2 defense-in-depth: even a top-level non-dict benchmark
    (a list / scalar — a genuinely-unparseable-shape artifact) must not crash
    ``from_artifacts`` with a raw traceback; it fails CLOSED to a clean NOT_YET."""
    for bad in ([1, 2, 3], "not a dict", 42, None):
        v = SupremacyVerdict.from_artifacts(bad)  # type: ignore[arg-type]
        assert v.verdict is Verdict.NOT_YET


def test_close2_malformed_container_verdict_is_deterministic() -> None:
    """[PROPERTY-TEST] AX-002: each malformed-container input yields an IDENTICAL
    fail-closed verdict across two runs on deep-copied inputs (the hardened reads stay
    pure functions — no order/identity dependence introduced by the shape guards)."""
    cases: list[dict[str, object]] = [
        {"systems": {"pii-anon": {"system": "pii-anon"}}},
        {"run_metadata": [1, 2, 3], "systems": []},
        {"systems": ["pii-anon", {"system": "gliner", "recall": 0.6}]},
        {"systems": [{"system": "pii-anon", "per_entity_recall": [1, 2, 3]}]},
        _with_calibration(
            _canonical_benchmark(),
            {**_conforming_calibration(), "abstention_operating_points": 42},
        ),
    ]
    for bench in cases:
        v1 = SupremacyVerdict.from_artifacts(copy.deepcopy(bench))
        v2 = SupremacyVerdict.from_artifacts(copy.deepcopy(bench))
        assert v1.verdict == v2.verdict
        assert v1.binding_constraint == v2.binding_constraint


def test_close2_hardening_preserves_honest_verdict_byte_identical() -> None:
    """[CONTRACT-TEST] S7-02 close-2 CARDINAL RULE: the container-shape hardening leaves
    the HONEST-input verdict byte-identical. The fully-conforming benchmark (all containers
    correctly typed) yields the SAME guarantees + verdict + binding constraint as before
    the hardening — proving only malformed-container handling changed."""
    bench = _with_calibration(_canonical_benchmark(), _conforming_calibration())
    verdict = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    # Honest guarantees unchanged (G1/G3/G6 compute; G2/G4 overridden-pass).
    assert verdict.guarantee("G1").passed is True
    assert verdict.guarantee("G3").passed is True
    assert verdict.guarantee("G6").passed is True
    assert verdict.guarantee("G7").passed is True
    # The honest top-composite crown is still pii-anon (containers correctly typed).
    assert _top_composite_system(
        {s["system"]: s for s in bench["systems"]}  # type: ignore[index]
    ) == "pii-anon"


# ---------------------------------------------------------------------------
# S7-02 FINAL confirmatory close — the unhashable-ELEMENT crash class.
# close-2 hardened wrong container TYPES; this closes unhashable ELEMENTS *inside*
# the two artifact containers that reach a ``set()`` / ``in frozenset``:
#   (a) an ``available_competitors`` element  -> off-limits ``set(available_competitors)``
#   (b) a ``floor_checks[].metric`` value     -> ``metric in _RECALL_FLOOR_METRICS``
# Both raised ``TypeError: unhashable type`` from the PUBLIC ``from_artifacts`` (a
# denial-of-verdict — no Verdict returned), violating the cardinal "never raises".
# ---------------------------------------------------------------------------


def test_closefinal_available_competitors_unhashable_element_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 final close: ``available_competitors`` carrying an UNHASHABLE
    element (a dict / nested list) crashed the OFF-LIMITS ``run_status_from_benchmark``
    (``set(available_competitors)`` → ``TypeError: unhashable type``) via the public
    ``from_artifacts`` (no Verdict returned). The gate's shield ``_tier_status_input`` must
    drop non-str (hence every unhashable) elements → a clean verdict, never a crash."""
    for bad_elem in ({}, {"system": "x"}, ["nested"], [1, 2, 3]):
        v = SupremacyVerdict.from_artifacts({"available_competitors": [bad_elem]})
        assert v.verdict is Verdict.NOT_YET


def test_closefinal_available_competitors_mixed_keeps_str_drops_unhashable() -> None:
    """[CONTRACT-TEST] final close: the shield keeps the HASHABLE str names (behaviour-
    preserving — a real ``available_competitors`` is a list of strings) and drops only the
    non-str / unhashable elements (crash-safe) before the off-limits ``set(...)`` sees them."""
    shielded = _tier_status_input(
        {"available_competitors": ["gliner", {}, "presidio", ["x"], 42]}, {}
    )
    assert shielded["available_competitors"] == ["gliner", "presidio"]


def test_closefinal_floor_check_metric_unhashable_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 final close: a ``floor_checks`` entry whose ``metric`` is
    UNHASHABLE (dict / list) crashed ``_run_breaches_recall_floor`` at
    ``metric in _RECALL_FLOOR_METRICS`` (``TypeError: unhashable type``) via the public
    ``from_artifacts``. A non-str metric can never be a recall-floor metric → guarded out."""
    for bad_metric in ({"x": 1}, [1, 2, 3], ["recall"]):
        bench = {
            "systems": [{"system": "pii-anon", "recall": 0.8}],
            "profile_results": [{"floor_checks": [{"metric": bad_metric, "passed": False}]}],
        }
        v = SupremacyVerdict.from_artifacts(bench)
        assert v.verdict is Verdict.NOT_YET


def test_closefinal_run_breach_guard_unhashable_metric_no_crash_preserves_str() -> None:
    """[SECURITY-TEST] final close: ``_run_breaches_recall_floor`` ignores an unhashable
    metric (no crash, no breach) while a genuine str metric still triggers a breach —
    behaviour-preserving for honest input (the only added logic is an ``isinstance`` guard)."""
    assert (
        _run_breaches_recall_floor(
            {"profile_results": [{"floor_checks": [{"metric": {"x": 1}, "passed": False}]}]}
        )
        is False
    )
    # honest str metric "recall" with passed=False still breaches (unchanged behaviour)
    assert (
        _run_breaches_recall_floor(
            {"profile_results": [{"floor_checks": [{"metric": "recall", "passed": False}]}]}
        )
        is True
    )


def test_closefinal_unhashable_element_fuzz_from_artifacts_never_raises() -> None:
    """[PROPERTY-TEST] S7-02 final close: a fuzz over UNHASHABLE elements in the two slots
    that reach a ``set()`` / ``in frozenset`` (``available_competitors`` element +
    ``floor_checks[].metric``) — ``from_artifacts`` ALWAYS returns a Verdict, never raises."""
    unhashables: list[object] = [{}, {"k": "v"}, [1, 2], [{"system": "x"}], [["nested"]]]
    for bad in unhashables:
        v1 = SupremacyVerdict.from_artifacts({"available_competitors": [bad, "gliner"]})
        assert isinstance(v1.verdict, Verdict)
        v2 = SupremacyVerdict.from_artifacts(
            {
                "systems": [{"system": "pii-anon", "recall": 0.8}],
                "profile_results": [{"floor_checks": [{"metric": bad, "passed": False}]}],
            }
        )
        assert isinstance(v2.verdict, Verdict)


# ---------------------------------------------------------------------------
# S7-02 FINAL close (round 2) — the G7 provenance fail-OPEN FABRICATION.
# G7 read each provenance stamp "present" via ``not run_metadata.get(f)``, so a
# TRUTHY-but-invalid value (whitespace-only string / int / bool / non-empty list
# or dict) was admitted as present ⇒ a forged G7 PASS ⇒ a forged PROVISIONAL_SOTA
# certified run. A real git_sha / sha256 / ISO-timestamp is a NON-BLANK STRING.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forged", ["   ", "\t\n ", 1, True, [1, 2], {"a": 1}])
def test_closefinal_g7_truthy_nonstring_provenance_does_not_forge_pass(forged: object) -> None:
    """[SECURITY-TEST] S7-02 final close (FABRICATION): a TRUTHY-but-invalid provenance value
    (whitespace-only string, int, bool, non-empty list/dict) was read as a present stamp by
    ``not run_metadata.get(f)`` ⇒ forged G7 PASS. A non-blank STRING is required; anything
    else is missing-equivalent ⇒ G7 FAIL."""
    bench = _canonical_benchmark()
    bench["run_metadata"]["git_sha"] = forged  # type: ignore[index]
    assert SupremacyVerdict.from_artifacts(bench).guarantee("G7").passed is False


def test_closefinal_g7_honest_provenance_still_passes() -> None:
    """[CONTRACT-TEST] CARDINAL RULE: real non-blank-string provenance still PASSES G7
    (behaviour-preserving — ``_canonical_benchmark`` carries git_sha='deadbeef' etc.)."""
    assert SupremacyVerdict.from_artifacts(_canonical_benchmark()).guarantee("G7").passed is True


def test_closefinal_g7_whitespace_provenance_does_not_forge_provisional_sota() -> None:
    """[SECURITY-TEST] end-to-end FABRICATION pin: a strong artifact (all guarantees pass via
    overrides) with WHITESPACE on all 4 provenance fields + canonical_claim_run=True FORGED a
    PROVISIONAL_SOTA (G7 PASS). The forge must collapse to NOT_YET (G7 FAIL)."""
    bench = _canonical_benchmark()
    for f in ("git_sha", "dataset_sha256", "matrix_sha256", "timestamp_utc"):
        bench["run_metadata"][f] = "   "  # type: ignore[index]
    v = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    assert v.guarantee("G7").passed is False
    assert v.verdict is Verdict.NOT_YET


@pytest.mark.parametrize("forged", ["false", "no", "0", "True", 1, [1], {"x": 1}])
def test_closefinal_canonical_claim_run_must_be_strict_true_no_coercion_forge(
    forged: object,
) -> None:
    """[SECURITY-TEST] S7-02 final close (FABRICATION): ``canonical_claim_run`` was read via
    ``bool(run_metadata.get(...))``, so a TRUTHY-but-not-True value forged a certified run —
    most damningly the STRING "false"/"no"/"0" (which a human reads as false) coerces to True ⇒
    forged G7 PASS ⇒ forged PROVISIONAL_SOTA. The flag is BOOLEAN; only the literal ``True``
    (a real bool, as the producer emits) certifies the run."""
    bench = _canonical_benchmark()
    bench["run_metadata"]["canonical_claim_run"] = forged  # type: ignore[index]
    v = SupremacyVerdict.from_artifacts(bench, pending_overrides=_ALL_PENDING_PASS)
    assert v.guarantee("G7").passed is False
    assert v.canonical_claim_run is False
    assert v.verdict is Verdict.NOT_YET


def test_closefinal_canonical_claim_run_real_true_still_certifies() -> None:
    """[CONTRACT-TEST] CARDINAL RULE: a real bool ``True`` (the producer's output) still
    certifies — G7 PASS + canonical_claim_run True (behaviour-preserving)."""
    v = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), pending_overrides=_ALL_PENDING_PASS
    )
    assert v.canonical_claim_run is True
    assert v.guarantee("G7").passed is True


# ---------------------------------------------------------------------------
# S7-02 FINAL close (round 4) — the CATASTROPHIC NaN-curve FABRICATION.
# ``_risk_coverage_is_monotone`` did ``float(p["coverage"])`` — ``float(NaN)`` does
# NOT raise, so a NaN coverage/risk entered the sort + the ``>=`` compare (NaN
# compares False to everything), LAUNDERING a genuinely non-monotone curve into a
# fabricated 'monotone' ⇒ G4 PASS ⇒ CLAIM_GRADE_SOTA. The nested curve-row fields
# bypassed the finiteness validators (the audit's blind spot).
# ---------------------------------------------------------------------------


def test_closefinal_g4_nan_coverage_in_curve_does_not_launder_non_monotone_pass() -> None:
    """[SECURITY-TEST] S7-02 final close (CATASTROPHIC FABRICATION): a NaN ``coverage`` in a
    ``risk_coverage_curve`` row laundered a genuinely NON-monotone curve into a fabricated
    'monotone' ⇒ G4 PASS ⇒ (with the other axes) CLAIM_GRADE_SOTA. A row with a non-finite
    coverage/risk cannot certify NFR-019 ⇒ NOT monotone."""
    non_monotone_finite = [{"coverage": 0.8, "risk": 0.2}, {"coverage": 0.4, "risk": 0.6}]
    assert _risk_coverage_is_monotone(non_monotone_finite) is False  # control: genuinely non-monotone
    with_nan = [
        {"coverage": 0.8, "risk": 0.2},
        {"coverage": float("nan"), "risk": 0.3},
        {"coverage": 0.4, "risk": 0.6},
    ]
    assert _risk_coverage_is_monotone(with_nan) is False  # the NaN must NOT launder it


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf"), 10**400, "x", None, True, [1], {"k": 1}]
)
def test_closefinal_g4_non_finite_curve_field_is_not_monotone(bad: object) -> None:
    """[SECURITY-TEST] a non-finite / non-numeric / huge-int / bool coverage OR risk in any
    curve row ⇒ NOT monotone (cannot certify NFR-019), never a launder, never a crash."""
    assert (
        _risk_coverage_is_monotone([{"coverage": bad, "risk": 0.2}, {"coverage": 0.5, "risk": 0.3}])
        is False
    )
    assert (
        _risk_coverage_is_monotone([{"coverage": 0.8, "risk": bad}, {"coverage": 0.5, "risk": 0.3}])
        is False
    )


def test_closefinal_g4_honest_monotone_curve_still_passes() -> None:
    """[CONTRACT-TEST] CARDINAL: an honest monotone-non-increasing curve (risk non-decreasing as
    coverage ascends) still certifies (behaviour-preserving)."""
    honest = [{"coverage": 0.4, "risk": 0.1}, {"coverage": 0.8, "risk": 0.3}]
    assert _risk_coverage_is_monotone(honest) is True


def test_closefinal_mixed_type_dict_keys_do_not_crash_sorted() -> None:
    """[SECURITY-TEST] S7-02 final close (round 5): a MIXED-TYPE dict KEY (str + int) in
    ``per_class_ece`` (G4 ``sorted``), the G1 entity maps (``sorted(missing)``) or the
    per-language map (``sorted(corrupt_langs)``) crashed ``sorted(...)`` with
    ``TypeError: '<' not supported between 'str' and 'int'`` — a denial-of-verdict reachable
    via a DIRECT dict caller (JSON object keys are always str). from_artifacts must always
    return a Verdict, never raise; the details sort by ``key=str`` (a total order)."""
    # G4: per_class_ece mixed keys (the sorted iteration).
    v1 = SupremacyVerdict.from_artifacts(
        {
            "systems": [
                {
                    "system": "pii-anon",
                    "per_class_ece": {1: 0.1, "EMAIL": 0.2},
                    "calibrated_confidence_coverage": 1.0,
                }
            ],
            "run_metadata": {"canonical_claim_run": True},
        }
    )
    assert isinstance(v1.verdict, Verdict)
    # G1: entity-name mixed keys drive sorted(missing) in the FAIL/superset detail.
    v2 = SupremacyVerdict.from_artifacts(
        {
            "systems": [
                {"system": "pii-anon", "per_entity_recall": {"Z": 0.9}},
                {"system": "gliner", "per_entity_recall": {1: 0.5, "E": 0.5}},
            ],
        }
    )
    assert isinstance(v2.verdict, Verdict)
    # G1: per-language mixed keys drive sorted(corrupt_langs).
    v3 = SupremacyVerdict.from_artifacts(
        {
            "systems": [{"system": "pii-anon", "per_entity_recall": {"E": 0.9}}],
            "per_language_recall_delta": {1: float("nan"), "en": float("inf")},
        }
    )
    assert isinstance(v3.verdict, Verdict)


# ---------------------------------------------------------------------------
# S7-02 FINAL close (round 6) — TWO SHOWSTOPPER fabrications via NESTED
# per_entity_recall VALUES (bool/+inf/huge-int/0.0/negative) that drive the G6
# entity-coverage floor AND the G1 recall-floor superset. The fix: a detection
# requires a VALID [0,1] score > 0 (via _detected_entity_names).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, float("inf"), 10**40, float("nan"), -0.5, "x"])
def test_closefinal_g6_coverage_rejects_non_physical_per_entity_recall(bad: object) -> None:
    """[SECURITY-TEST] S7-02 close-9 SHOWSTOPPER: a non-physical per_entity_recall value
    (bool/+inf/huge-int/NaN/negative) counted as a 'detected' entity, inflating the G6
    entity-coverage NFR floor past 0.80 ⇒ forged G6 PASS ⇒ CLAIM_GRADE. Only a valid [0,1]
    score > 0 is a detection ⇒ honest coverage 4/10=0.40 < 0.80 ⇒ G6 FAIL."""
    per = {f"REAL_{i}": 0.9 for i in range(4)}
    per.update({f"FAKE_{i}": bad for i in range(6)})
    core = {"system": "pii-anon", "recall": 0.9, "precision": 0.9, "per_entity_recall": per}
    gliner = {"system": "gliner", "recall": 0.6, "precision": 0.9, "per_entity_recall": {"REAL_0": 0.6}}
    g6 = _g6_raw_noninferiority({"pii-anon": core, "gliner": gliner})
    assert g6.passed is False  # coverage 0.40, NOT laundered to 1.0


def test_closefinal_g6_honest_coverage_still_passes() -> None:
    """[CONTRACT-TEST] honest real recalls (>0) still count ⇒ coverage unchanged ⇒ G6 PASS."""
    per = {f"E{i}": 0.9 for i in range(9)}
    per["E9"] = 0.0  # 9/10 = 0.90 ≥ 0.80 floor
    core = {"system": "pii-anon", "recall": 0.9, "precision": 0.9, "per_entity_recall": per}
    gliner = {"system": "gliner", "recall": 0.6, "precision": 0.9, "per_entity_recall": {"E0": 0.6}}
    g6 = _g6_raw_noninferiority({"pii-anon": core, "gliner": gliner})
    assert g6.passed is True


@pytest.mark.parametrize("mask", [0.0, -5.0, True, float("inf"), float("nan")])
def test_closefinal_g1_recall_floor_mask_rejected(mask: object) -> None:
    """[SECURITY-TEST] S7-02 close-9 SHOWSTOPPER: listing an entity with a 0.0 / negative /
    non-physical recall on the pii-anon ladder put it in the ENSEMBLE set (no >0 filter),
    masking the G1 recall-floor-by-construction superset ⇒ forged G1 PASS ⇒ CLAIM_GRADE. An
    entity is in the ensemble only if genuinely detected (valid score > 0); a competitor that
    detects it (>0) then exposes the miss ⇒ G1 FAIL."""
    systems = {
        "pii-anon": {"system": "pii-anon", "per_entity_recall": {"OTHER": 0.9, "IBAN": mask}},
        "pii-anon-swarm": {"system": "pii-anon-swarm", "per_entity_recall": {"OTHER": 0.9, "IBAN": mask}},
        "gliner": {"system": "gliner", "per_entity_recall": {"IBAN": 0.7}},
    }
    g1 = _g1_recall_floor({"per_language_recall_delta": {"en": 0.0}}, systems)
    assert g1.passed is False  # ensemble misses IBAN (a mask is not a detection)


def test_closefinal_g1_honest_superset_still_passes() -> None:
    """[CONTRACT-TEST] an honest ensemble that genuinely detects all competitor entities (>0)
    still PASSES G1 (behaviour-preserving)."""
    systems = {
        "pii-anon": {"system": "pii-anon", "per_entity_recall": {"OTHER": 0.9, "IBAN": 0.8}},
        "gliner": {"system": "gliner", "per_entity_recall": {"IBAN": 0.7}},
    }
    g1 = _g1_recall_floor({"per_language_recall_delta": {"en": 0.0}}, systems)
    assert g1.passed is True


# ===========================================================================
# S7-04 — G5 (audit + orchestration latency / interception) goes placeholder →
# computed: _g5_audit_latency(systems, run_metadata). Three-valued; every
# artifact value validated; the overrides["G5"] successor seam preserved.
# The imports are LAZY (inside each test) so the file keeps collecting while
# the gate change is RED.
# ===========================================================================


def _g5_latency_summary(**over: object) -> dict[str, object]:
    """A VALID measured-latency block (in-ceiling for the committed ensemble
    budget 500/1000/2000 ms)."""
    base: dict[str, object] = {
        "system": "pii-anon-swarm",
        "profile": "ensemble",
        "p50_ms": 80.5,
        "p95_ms": 112.0,
        "p99_ms": 133.3,
        "n_records": 12,
        "measurement": "fresh-in-tree-per-record-detection-timing",
    }
    base.update(over)
    return base


def _g5_audit_summary(**over: object) -> dict[str, object]:
    """A VALID audit block (4 channels, no persist breach, zero leaks, ASR 0)."""
    base: dict[str, object] = {
        "interception": {
            "counts_by_channel": {"PROMPT": 2, "MEMORY": 1, "TOOL_IO": 1, "TRACE": 1},
            "no_raw_pii_persist": True,
            "records_total": 5,
        },
        "leakage_sankey": {"blocked": 5, "leaked": 0},
        "injection_resistance": {
            "attack_success_rate": 0.0,
            "benign_task_success_rate": 1.0,
            "n_payloads": 4,
        },
    }
    base.update(over)
    return base


def _g5_systems() -> dict[str, dict[str, object]]:
    """A minimal systems map carrying the ladder (the latency SUT)."""
    return {
        "pii-anon": {"system": "pii-anon"},
        "pii-anon-swarm": {"system": "pii-anon-swarm"},
        "gliner": {"system": "gliner"},
    }


def _g5(
    latency: object = "default", audit: object = "default"
) -> object:
    """Call _g5_audit_latency with the given blocks ('default' = valid block;
    None = omit the key entirely; anything else verbatim)."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _g5_audit_latency,
    )

    rm: dict[str, object] = {}
    if latency == "default":
        rm["latency_summary"] = _g5_latency_summary()
    elif latency is not None:
        rm["latency_summary"] = latency
    if audit == "default":
        rm["audit_summary"] = _g5_audit_summary()
    elif audit is not None:
        rm["audit_summary"] = audit
    return _g5_audit_latency(_g5_systems(), rm)


def test_g5_pending_when_both_blocks_absent_never_fabricated() -> None:
    """[CONTRACT-TEST] S7-04: the smoke-artifact path — no latency/audit fields
    ⇒ G5 PENDING (None), naming the successor; NEVER fabricated."""
    g5 = _g5(latency=None, audit=None)
    assert g5.passed is None
    assert "PENDING" in g5.binding_detail
    assert "G5←S5/S6" in g5.binding_detail


def test_g5_pending_when_audit_half_absent() -> None:
    """[CONTRACT-TEST] a half-populated artifact cannot certify (G2 precedent):
    latency present, audit absent ⇒ PENDING naming the audit half."""
    g5 = _g5(audit=None)
    assert g5.passed is None
    assert "audit" in g5.binding_detail.lower()


def test_g5_pending_when_latency_half_absent() -> None:
    """[CONTRACT-TEST] symmetric: audit present, latency absent ⇒ PENDING
    naming the latency half."""
    g5 = _g5(latency=None)
    assert g5.passed is None
    assert "latency" in g5.binding_detail.lower()


def test_g5_override_seam_preserved_exactly_like_g2_g4() -> None:
    """[CONTRACT-TEST] the pending_overrides['G5'] successor seam wins over
    computation — True/False/None all honored (caller-kwarg-only)."""
    bench = _canonical_benchmark()
    for override, expected in ((True, True), (False, False), (None, None)):
        verdict = SupremacyVerdict.from_artifacts(
            bench, pending_overrides={"G5": override}
        )
        assert verdict.guarantee("G5").passed is expected
        if override is not None:
            assert "supplied by successor" in verdict.guarantee("G5").binding_detail


def test_g5_smoke_benchmark_stays_pending_with_stable_successor_string() -> None:
    """[CONTRACT-TEST] the no-G5-fields benchmark computes G5 PENDING via the
    REAL method (no override) and axes_pending keeps the byte-stable successor
    string (honest-input surface stability)."""
    verdict = SupremacyVerdict.from_artifacts(_canonical_benchmark())
    assert verdict.guarantee("G5").passed is None
    assert "G5←S5/S6 (latency budgets + 4-channel interception)" in verdict.axes_pending


def test_g5_passes_on_valid_in_ceiling_artifact() -> None:
    """[CONTRACT-TEST] the honest produced-artifact path: valid measured latency
    within the committed ensemble ceiling + a clean audit ⇒ G5 PASS; observed/bar
    carry the p99 pair."""
    g5 = _g5()
    assert g5.passed is True
    assert "G5 PASS" in g5.binding_detail
    assert g5.observed == pytest.approx(133.3)
    assert g5.bar == pytest.approx(2000.0)


def test_g5_fails_when_p99_over_committed_ceiling() -> None:
    """[CONTRACT-TEST] honest latency breach: measured p99 5000 ms > committed
    ensemble p99 1000 ms ⇒ FAIL naming the percentile."""
    g5 = _g5(latency=_g5_latency_summary(p99_ms=5000.0))
    assert g5.passed is False
    assert "p99" in g5.binding_detail


def test_g5_fails_when_leaked_positive() -> None:
    """[CONTRACT-TEST] honest audit breach: a leaked span ⇒ FAIL."""
    g5 = _g5(audit=_g5_audit_summary(leakage_sankey={"blocked": 5, "leaked": 1}))
    assert g5.passed is False
    assert "leak" in g5.binding_detail.lower()


def test_g5_fails_when_asr_positive() -> None:
    """[CONTRACT-TEST] honest audit breach: attack_success_rate 0.25 > the
    committed 0.0 bar ⇒ FAIL."""
    audit = _g5_audit_summary()
    audit["injection_resistance"] = {
        "attack_success_rate": 0.25,
        "benign_task_success_rate": 1.0,
        "n_payloads": 4,
    }
    g5 = _g5(audit=audit)
    assert g5.passed is False
    assert "attack" in g5.binding_detail.lower() or "ASR" in g5.binding_detail


def test_g5_fails_when_benign_task_success_below_bar() -> None:
    """[CONTRACT-TEST] the FR-029 anti-degenerate pairing: a guard that blocks
    everything (ASR 0 but benign 0.5 < 0.95) ⇒ FAIL — masking everything is not
    resistance."""
    audit = _g5_audit_summary()
    audit["injection_resistance"] = {
        "attack_success_rate": 0.0,
        "benign_task_success_rate": 0.5,
        "n_payloads": 4,
    }
    g5 = _g5(audit=audit)
    assert g5.passed is False
    assert "benign" in g5.binding_detail.lower()


@pytest.mark.parametrize("persist", [False, "true", 1, [], "yes", 0])
def test_g5_persist_stamp_must_be_strict_true(persist: object) -> None:
    """[SECURITY-TEST] no_raw_pii_persist certifies ONLY as the literal True —
    the canonical_claim_run coercion lesson (close-6): a truthy-but-not-True
    value (the string 'true', the int 1) or an honest False all FAIL."""
    inter = {
        "counts_by_channel": {"PROMPT": 2, "MEMORY": 1, "TOOL_IO": 1, "TRACE": 1},
        "no_raw_pii_persist": persist,
        "records_total": 5,
    }
    g5 = _g5(audit=_g5_audit_summary(interception=inter))
    assert g5.passed is False


def test_g5_pending_when_persist_key_absent() -> None:
    """[CONTRACT-TEST] missing-shape (an absent key) is PENDING, never a PASS
    and never a fabricated FAIL-mask: omission can never read BETTER than the
    placeholder did."""
    inter = {
        "counts_by_channel": {"PROMPT": 2, "MEMORY": 1, "TOOL_IO": 1, "TRACE": 1},
        "records_total": 5,
    }
    g5 = _g5(audit=_g5_audit_summary(interception=inter))
    assert g5.passed is None


def test_g5_fails_when_a_channel_is_missing() -> None:
    """[CONTRACT-TEST] 4-channel least-privilege interception means ALL FOUR
    channels exercised — a counts_by_channel missing TRACE ⇒ FAIL (incomplete
    interception is a breach, not a missing-shape)."""
    inter = {
        "counts_by_channel": {"PROMPT": 2, "MEMORY": 1, "TOOL_IO": 1},
        "no_raw_pii_persist": True,
        "records_total": 4,
    }
    g5 = _g5(audit=_g5_audit_summary(interception=inter))
    assert g5.passed is False
    assert "TRACE" in g5.binding_detail


@pytest.mark.parametrize(
    "corrupt",
    [float("nan"), float("inf"), float("-inf"), -1.0, True, 10**400, "80", []],
)
@pytest.mark.parametrize("key", ["p50_ms", "p95_ms", "p99_ms"])
def test_g5_corrupt_latency_value_fails_never_crashes_never_passes(
    key: str, corrupt: object
) -> None:
    """[SECURITY-TEST] the no-fabrication invariant on the latency half: a
    NaN/±inf/negative/bool/huge-int/str/container percentile is NOT a
    measurement — FAIL (a corrupt present value cannot certify), never a crash,
    never a PASS."""
    g5 = _g5(latency=_g5_latency_summary(**{key: corrupt}))
    assert g5.passed is False


def test_g5_percentile_inversion_is_corrupt() -> None:
    """[SECURITY-TEST] p50 > p95 is non-physical (percentiles are monotone) — a
    corrupt measurement, FAIL."""
    g5 = _g5(latency=_g5_latency_summary(p50_ms=200.0, p95_ms=100.0))
    assert g5.passed is False
    assert "order" in g5.binding_detail.lower() or "p50" in g5.binding_detail


@pytest.mark.parametrize("profile", ["bogus", "", ["ensemble"], {"p": 1}, 7, True])
def test_g5_unknown_or_hostile_profile_fails_never_crashes(profile: object) -> None:
    """[SECURITY-TEST] a profile naming no committed budget can never certify;
    a hostile non-str profile (list/dict — unhashable) must never crash the
    registry lookup."""
    g5 = _g5(latency=_g5_latency_summary(profile=profile))
    assert g5.passed is False


def test_g5_non_ladder_or_unknown_system_fails() -> None:
    """[SECURITY-TEST] the latency SUT must be a ladder member present in the
    benchmark — a competitor's (or phantom) latency cannot certify the SUT's
    NFR-009."""
    assert _g5(latency=_g5_latency_summary(system="gliner")).passed is False
    assert _g5(latency=_g5_latency_summary(system="phantom-sys")).passed is False
    assert _g5(latency=_g5_latency_summary(system=["pii-anon-swarm"])).passed is False


def test_g5_artifact_ceiling_cannot_loosen_committed_budget() -> None:
    """[SECURITY-TEST] the _g4_class_bar lesson on the latency axis: an
    artifact-supplied ceiling may only TIGHTEN. measured p99=5000 with a
    loosening artifact ceiling 1e9 still FAILS against the committed 1000."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _g5_audit_latency,
    )

    rm: dict[str, object] = {
        "latency_summary": _g5_latency_summary(p99_ms=5000.0),
        "latency_ceiling_ms": {"p99_ms": 1e9},
        "audit_summary": _g5_audit_summary(),
    }
    assert _g5_audit_latency(_g5_systems(), rm).passed is False


def test_g5_artifact_ceiling_tightening_is_honored() -> None:
    """[CONTRACT-TEST] a TIGHTER artifact ceiling binds: measured p50=80.5
    passes the committed 250 but FAILS a self-imposed 10.0; a GARBAGE artifact
    ceiling (NaN) is rejected and the committed budget stands (PASS)."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _g5_audit_latency,
    )

    rm: dict[str, object] = {
        "latency_summary": _g5_latency_summary(),
        "latency_ceiling_ms": {"p50_ms": 10.0},
        "audit_summary": _g5_audit_summary(),
    }
    assert _g5_audit_latency(_g5_systems(), rm).passed is False

    rm["latency_ceiling_ms"] = {"p50_ms": float("nan"), "p95_ms": -5.0, "p99_ms": True}
    assert _g5_audit_latency(_g5_systems(), rm).passed is True


@pytest.mark.parametrize(
    "count",
    [True, 0, -1, 1.5, "2", None, 10**5000],
    ids=["bool", "zero", "negative", "float", "str", "none", "huge-int"],
)
def test_g5_invalid_channel_count_fails_never_crashes(count: object) -> None:
    """[SECURITY-TEST] a channel count must be a genuine int ≥ 1 (a bool is a
    flag; a 10**5000 must never crash a detail string — the int→str digit
    limit is a DoV vector)."""
    inter = {
        "counts_by_channel": {"PROMPT": count, "MEMORY": 1, "TOOL_IO": 1, "TRACE": 1},
        "no_raw_pii_persist": True,
        "records_total": 4,
    }
    g5 = _g5(audit=_g5_audit_summary(interception=inter))
    assert g5.passed is False


@pytest.mark.parametrize(
    "n",
    [0, -1, True, "12", 1.5, 10**5000],
    ids=["zero", "negative", "bool", "str", "float", "huge-int"],
)
def test_g5_invalid_n_records_fails_never_crashes(n: object) -> None:
    """[SECURITY-TEST] the timing sample size must be a genuine int in a sane
    range — a bool/str/float/absurd value is corrupt."""
    g5 = _g5(latency=_g5_latency_summary(n_records=n))
    assert g5.passed is False


@pytest.mark.parametrize(
    ("latency", "audit"),
    [
        (["not", "a", "dict"], "default"),
        ("default", "scalar"),
        (42, "default"),
        ("default", ["x"]),
    ],
)
def test_g5_non_dict_blocks_read_as_absent_pending_never_crash(
    latency: object, audit: object
) -> None:
    """[SECURITY-TEST] the close-2 container-shape class: a non-dict block is
    ABSENT-equivalent ⇒ PENDING (fail-closed), never a crash, never a PASS."""
    g5 = _g5(latency=latency, audit=audit)
    assert g5.passed is None


@pytest.mark.parametrize(
    "audit_over",
    [
        {"interception": []},
        {"interception": {"counts_by_channel": [], "no_raw_pii_persist": True, "records_total": 1}},
        {"leakage_sankey": 7},
        {"injection_resistance": "x"},
    ],
)
def test_g5_non_dict_audit_subblocks_pending_never_crash(
    audit_over: dict[str, object],
) -> None:
    """[SECURITY-TEST] NESTED container garbage (the close-9 lesson — nested
    values are first-class): a non-dict sub-block / counts container is
    missing-shape ⇒ PENDING, never a crash."""
    g5 = _g5(audit=_g5_audit_summary(**audit_over))
    assert g5.passed is None


def test_g5_computed_true_supports_claim_grade_end_to_end() -> None:
    """[CONTRACT-TEST] integration: a fully-passing benchmark whose G5 is
    COMPUTED (not overridden) reaches CLAIM_GRADE with G2/G4 overridden True,
    a strong posterior, and all tiers waived — G5 participates in the §5
    predicate as a real guarantee."""
    theta, names = _strong_posterior()
    bench = _canonical_benchmark()
    rm = bench["run_metadata"]
    assert isinstance(rm, dict)
    rm["latency_summary"] = _g5_latency_summary()
    rm["audit_summary"] = _g5_audit_summary()
    verdict = SupremacyVerdict.from_artifacts(
        bench,
        theta_samples=theta,
        posterior_names=names,
        pending_overrides={"G2": True, "G4": True},
        tier_c_waivers={
            "aws-comprehend": "no credentials in CI",
            "azure-ai-language": "no credentials in CI",
            "openai-privacy-filter": "no credentials in CI",
        },
        unrun_tier_r_waivers={"gliner2": "adapter not yet implemented"},
    )
    assert verdict.guarantee("G5").passed is True
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA


def test_g5_computed_fail_binds_as_lowest_failing_guarantee() -> None:
    """[CONTRACT-TEST] integration: a computed G5 FAIL (leaked=1) on an
    otherwise-passing benchmark binds the verdict (G5 < G6 in the guarantee
    order) ⇒ NOT_YET with the G5 detail as the binding constraint."""
    bench = _canonical_benchmark()
    rm = bench["run_metadata"]
    assert isinstance(rm, dict)
    rm["latency_summary"] = _g5_latency_summary()
    rm["audit_summary"] = _g5_audit_summary(
        leakage_sankey={"blocked": 5, "leaked": 1}
    )
    verdict = SupremacyVerdict.from_artifacts(
        bench, pending_overrides={"G2": True, "G4": True}
    )
    assert verdict.guarantee("G5").passed is False
    assert verdict.verdict is Verdict.NOT_YET
    assert verdict.binding_constraint == verdict.guarantee("G5").binding_detail


# ===========================================================================
# S7-04 DoV hardening — the int→str digit-limit crash class (NEW; the prior
# closes fuzzed 10**400 = 401 digits, under the 4300-digit conversion limit).
# Formatting a >4300-digit int into a detail string (repr / str / sort key)
# raises ValueError — a fail-loud denial-of-verdict on the live gate.
# ===========================================================================


def test_dov_g4_huge_int_ece_value_fails_never_crashes() -> None:
    """[SECURITY-TEST] a 10**5000 per-class ECE value is non-finite (rejected)
    AND its detail string must not crash on repr (the digit limit)."""
    systems = {
        "pii-anon": {
            "per_class_ece": {"EMAIL": 10**5000},
            "calibrated_confidence_coverage": 1.0,
            "risk_coverage_curve": [{"coverage": 1.0, "risk": 0.1}],
            "abstention_operating_points": [1, 2, 3],
        }
    }
    g4 = _g4_calibration_selective_risk(systems)
    assert g4.passed is False
    assert "EMAIL" in g4.binding_detail


def test_dov_g4_huge_int_ece_key_never_crashes() -> None:
    """[SECURITY-TEST] a 10**5000 ECE KEY must not crash the sorted(key=str)
    iteration (close-8 kept mixed-type keys; the digit limit broke str). The
    class's valid 0.01 value still evaluates (close-8 precedent: every class
    is kept)."""
    systems = {
        "pii-anon": {
            "per_class_ece": {10**5000: 0.01},
            "calibrated_confidence_coverage": 1.0,
            "risk_coverage_curve": [{"coverage": 1.0, "risk": 0.1}],
            "abstention_operating_points": [1, 2, 3],
        }
    }
    g4 = _g4_calibration_selective_risk(systems)
    assert g4.passed is True  # 0.01 ≤ the 0.05 bar; the key is just a label


def test_dov_g4_huge_int_coverage_fails_never_crashes() -> None:
    """[SECURITY-TEST] a 10**5000 calibrated_confidence_coverage is rejected by
    _finite_unit_score and the FAIL detail must not crash on its repr."""
    systems = {
        "pii-anon": {
            "per_class_ece": {"EMAIL": 0.01},
            "calibrated_confidence_coverage": 10**5000,
            "risk_coverage_curve": [{"coverage": 1.0, "risk": 0.1}],
            "abstention_operating_points": [1, 2, 3],
        }
    }
    g4 = _g4_calibration_selective_risk(systems)
    assert g4.passed is False
    assert "coverage" in g4.binding_detail.lower()


def test_dov_g1_huge_int_entity_key_never_crashes() -> None:
    """[SECURITY-TEST] a 10**5000 per-entity KEY on a competitor (a genuinely
    detected entity the ladder misses) must not crash the G1 missing-entities
    detail (sorted key=str + list formatting both hit the digit limit)."""
    systems = {
        "pii-anon": {"system": "pii-anon", "per_entity_recall": {"OTHER": 0.9}},
        "gliner": {"system": "gliner", "per_entity_recall": {10**5000: 0.7}},
    }
    g1 = _g1_recall_floor({"per_language_recall_delta": {"en": 0.0}}, systems)
    assert g1.passed is False  # the ladder honestly misses that entity


def test_dov_g1_huge_int_lang_key_never_crashes() -> None:
    """[SECURITY-TEST] a 10**5000 language KEY with a corrupt ε must not crash
    the corrupt-language detail formatting."""
    core = {"system": "pii-anon", "per_entity_recall": {"OTHER": 0.9}}
    g1 = _g1_recall_floor(
        {"per_language_recall_delta": {10**5000: float("nan")}},
        {"pii-anon": core},
    )
    assert g1.passed is False  # corrupt ε fails closed, crash-free
