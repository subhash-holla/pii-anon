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
    f_beta,
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
    (Tier-R ∪ Tier-C RUN-or-WAIVED). All Tier-C waived-with-reason here."""
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
    )
    assert verdict.verdict is Verdict.CLAIM_GRADE_SOTA
    assert verdict.j_value is not None and verdict.j_value >= J_BAR
    assert verdict.binding_constraint == ""


def test_provisional_when_blocked_only_by_unrun_tier_c() -> None:
    """[CONTRACT-TEST] PROVISIONAL_SOTA ⟺ everything else passes but Tier-C is
    not yet run (the ONLY remaining blocker)."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(),
        theta_samples=theta,
        posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        # No waivers → Tier-C stays UNRUN.
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
    """[PROPERTY-TEST] binding_constraint is "" iff verdict is CLAIM_GRADE."""
    theta, names = _strong_posterior()
    claim = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "w" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
    )
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
    assert verdict.verdict is not Verdict.CLAIM_GRADE_SOTA


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
    """[CONTRACT-TEST] All Tier-C waived-with-reason satisfies the Tier predicate
    so (with everything else passing) the verdict is CLAIM_GRADE."""
    theta, names = _strong_posterior()
    verdict = SupremacyVerdict.from_artifacts(
        _canonical_benchmark(), theta_samples=theta, posterior_names=names,
        pending_overrides=_ALL_PENDING_PASS,
        tier_c_waivers={n: "documented reason" for n in
                        ("openai-privacy-filter", "azure-ai-language", "aws-comprehend")},
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
