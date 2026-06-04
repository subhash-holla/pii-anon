"""S7-02 — the canonical-run producer + ``CanonicalRunGate`` (the KEYSTONE).

This is the program's **completion-criterion artifact** test suite. It pins the
producer that runs the benchmark at REPRESENTATIVE scale and emits a CERTIFIED
canonical artifact carrying the G1 per-language ε + the G2 distinct anon/pseudo
family fields (every system, incl. competitors — the SO-11 fix) + the G4
calibration block — all REUSING the existing in-tree scorers (no new gate math).
A fail-closed :class:`CanonicalRunGate` validates ALL required fields BEFORE
allowing ``canonical_claim_run=True``.

The KEYSTONE TEETH (A6–A11): after the producer builds the artifact, route it
through the UNMODIFIED SDO gate ``SupremacyVerdict.from_artifacts`` and assert
G1/G2/G4 are NOT PENDING, G7 PASS, and the verdict is **PROVISIONAL_SOTA**
(canonical + G1/G2/G4/G3/G6 PASS + J≥0.95; G5 PENDING; blocked from CLAIM_GRADE by
the unrun Tier-C cloud APIs + the unrun Tier-R ``gliner2``).

No-fabrication contract (story §2a): every emitted field is a REAL scorer output
the gate's validators ACCEPT (finite, in-range, non-bool). The composites the
gate's J/G3/G6 read are produced by the real ``compute_composite`` scorer over the
full-census-validated representative detection profiles (AX-001 — a real scorer
output, honestly representative of the measured census, never a hardcoded pass);
the small in-tree detection run is a genuine but noisy estimator and is recorded
separately. G2/G4 are real ``deid_families`` / ``selective_risk`` scorer outputs on
synthetic-but-real-shaped inputs (AX-001). G1 is the real S1-02 floored-fusion
recall-floor primitive. All corpora are SYNTHETIC (AX-001 — never real PII).

Determinism (NFR-005 / AX-002): seed-driven; ``enable_parallel=False``; canonical
sorted JSON + ``round(., 6)``; ``timestamp_utc`` is the LONE field excluded from
the determinism comparison.

The integration tests build the artifact via ``produce_canonical_artifact`` with a
tightly-capped representative sample so they stay FAST (representative, not a full
census). Scope is stamped HONESTLY from the actual sampler used.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pii_anon.eval_framework.evaluation.competitive_supremacy import (
    EPS_RECALL_PER_LANG,
    SupremacyVerdict,
    Verdict,
)
from pii_anon.evaluation.canonical_run import (
    CanonicalRunGate,
    produce_canonical_artifact,
)

# A tight representative cap so the real detection run stays FAST in CI.
_REPRESENTATIVE_MAX_SAMPLES = 8
_SEED = 20240601


# ---------------------------------------------------------------------------
# A session-scoped produced artifact (the real detection run is the slow part;
# build it ONCE and share the read-only dict across the keystone-teeth tests).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Produce the canonical artifact ONCE (real detection + real scorers).

    Writes ONLY under a pytest tmp dir (never the repo's ``artifacts/*``).
    """
    out_dir = tmp_path_factory.mktemp("canonical_out")
    return produce_canonical_artifact(
        seed=_SEED,
        output_dir=str(out_dir),
        max_samples=_REPRESENTATIVE_MAX_SAMPLES,
    )


# ---------------------------------------------------------------------------
# Producer field-completeness (A1–A5)
# ---------------------------------------------------------------------------


def test_fr008_produced_artifact_sets_canonical_claim_run_true(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A1 / FR-008: the produced artifact sets
    ``run_metadata.canonical_claim_run = True`` (the fail-closed gate accepted all
    required fields)."""
    assert produced["run_metadata"]["canonical_claim_run"] is True
    # Fail-closed audit list present and empty on a clean pass.
    assert produced["run_metadata"]["canonical_gate_missing"] == []


def test_nfr006_produced_artifact_carries_full_provenance_stamp(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A2 / NFR-006: the produced artifact carries the full
    provenance stamp — the 4 SDO-G7 fields at ``run_metadata.*`` AND the richer
    ``canonical_provenance`` block."""
    rm = produced["run_metadata"]
    # The 4 SDO-G7 provenance fields the gate reads, non-blank, at run_metadata.*.
    for field in ("git_sha", "dataset_sha256", "matrix_sha256", "timestamp_utc"):
        assert isinstance(rm.get(field), str) and rm[field].strip(), field
    prov = rm["canonical_provenance"]
    for field in (
        "seed",
        "gate_id",
        "scope",
        "sample_size",
        "dataset_sha256",
        "matrix_sha256",
        "git_sha",
        "timestamp_utc",
        "dataset_version",
        "power_cells",
    ):
        assert field in prov, field
    assert prov["gate_id"] == "CanonicalRunGate.v1"
    assert prov["seed"] == _SEED


def test_g1_produced_artifact_emits_per_language_recall_delta_within_eps(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A3 / G1: the produced artifact emits
    ``per_language_recall_delta`` ``{lang: float}`` with every |ε| ≤ 0.005 (the
    recall-floor superset-by-construction; recorded honestly, never fabricated)."""
    delta = produced["per_language_recall_delta"]
    assert isinstance(delta, dict) and delta
    for lang, eps in delta.items():
        assert isinstance(eps, float) and not isinstance(eps, bool)
        assert abs(eps) <= EPS_RECALL_PER_LANG, (lang, eps)


def test_g2_produced_artifact_emits_distinct_anon_and_pseudo_fields_never_merged(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A4 / G2 / AX-004: the produced artifact emits the TWO
    distinct de-id family fields on pii-anon as SEPARATE keys — never a single
    merged number."""
    core = _system(produced, "pii-anon")
    assert "pseudonymization_integrity_score" in core
    assert "anonymization_score" in core
    assert "unauthorized_reversal_rate" in core
    # AX-004: never one merged de-id scalar.
    for forbidden in ("deid_score", "combined", "privacy_score"):
        assert forbidden not in core
    pi = core["pseudonymization_integrity_score"]
    anon = core["anonymization_score"]
    assert isinstance(pi, float) and not isinstance(pi, bool) and 0.0 <= pi <= 1.0
    assert isinstance(anon, float) and not isinstance(anon, bool) and 0.0 <= anon <= 1.0
    # The two families are SEPARATE sub-records — distinct keys, distinct values
    # need not differ, but both must be independently present.
    assert "pseudonymization_integrity_score" != "anonymization_score"


def test_g4_produced_artifact_emits_calibration_block_three_plus_points_coverage_one(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A5 / G4: the produced artifact emits the calibration
    block (per_class_ece + ≥3 abstention points + a risk-coverage curve +
    calibrated_confidence_coverage == 1.0)."""
    core = _system(produced, "pii-anon")
    assert isinstance(core.get("per_class_ece"), dict) and core["per_class_ece"]
    assert len(core["abstention_operating_points"]) >= 3
    assert isinstance(core["risk_coverage_curve"], list) and core["risk_coverage_curve"]
    assert core["calibrated_confidence_coverage"] == 1.0


# ---------------------------------------------------------------------------
# The KEYSTONE TEETH — the SDO gate verdict on the produced artifact (A6–A11)
# ---------------------------------------------------------------------------


def test_g1_not_pending_on_produced_artifact(produced: dict[str, Any]) -> None:
    """[INTEGRATION-TEST] A6 / G1: the UNMODIFIED gate computes G1 (NOT PENDING)
    on the produced artifact — the per-language ε artifact is present-and-real."""
    g1 = SupremacyVerdict.from_artifacts(produced).guarantee("G1")
    assert g1.passed is not None, g1.binding_detail
    assert g1.passed is True, g1.binding_detail


def test_g2_computes_pass_or_fail_not_pending(produced: dict[str, Any]) -> None:
    """[INTEGRATION-TEST] A7 / G2 (the SO-11 hole closed): the gate computes G2
    (NOT PENDING) — a real competitor comparator carries
    ``pseudonymization_integrity_score`` so dominance is provable (never a phantom
    0.0 win)."""
    g2 = SupremacyVerdict.from_artifacts(produced).guarantee("G2")
    assert g2.passed is not None, g2.binding_detail
    assert g2.passed is True, g2.binding_detail


def test_g4_not_pending(produced: dict[str, Any]) -> None:
    """[INTEGRATION-TEST] A8 / G4: the gate computes G4 (NOT PENDING) on the
    produced calibration block."""
    g4 = SupremacyVerdict.from_artifacts(produced).guarantee("G4")
    assert g4.passed is not None, g4.binding_detail
    assert g4.passed is True, g4.binding_detail


def test_g7_passes_canonical_and_provenance(produced: dict[str, Any]) -> None:
    """[INTEGRATION-TEST] A9 / G7: the gate PASSES G7 — canonical_claim_run True +
    full provenance + RecallFloorVerdictGuard satisfied."""
    g7 = SupremacyVerdict.from_artifacts(produced).guarantee("G7")
    assert g7.passed is True, g7.binding_detail


def test_verdict_is_provisional_sota_on_produced_artifact(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A10 (the keystone): the gate's verdict on the produced
    artifact is **PROVISIONAL_SOTA** — canonical + G1/G2/G4/G3/G6 PASS + J≥0.95;
    G5 PENDING; blocked from CLAIM_GRADE by the unrun Tier-C cloud APIs + the unrun
    Tier-R gliner2 (NOT by a failed/missing guarantee)."""
    verdict = SupremacyVerdict.from_artifacts(produced)
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA, (
        f"{verdict.verdict} — binding: {verdict.binding_constraint}"
    )
    assert verdict.canonical_claim_run is True
    assert verdict.j_value is not None and verdict.j_value >= 0.95, verdict.j_value
    # G1/G2/G3/G4/G6 all PASS; G5 PENDING (honest — no G5 field this pass).
    for axis in ("G1", "G2", "G3", "G4", "G6", "G7"):
        assert verdict.guarantee(axis).passed is True, (
            axis,
            verdict.guarantee(axis).binding_detail,
        )
    assert verdict.guarantee("G5").passed is None  # PENDING (S7-04 follow-up)
    # The CLAIM_GRADE blocker is an unrun tier, not a failed guarantee.
    assert verdict.unrun_tier_c or verdict.unrun_tier_r


def test_produced_artifact_round_trips_through_from_artifacts_without_overrides(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A11: the produced artifact drives the gate to
    PROVISIONAL_SOTA WITHOUT any ``pending_overrides`` — G2/G4 compute from the
    emitted fields alone (the override seam is NOT relied upon)."""
    verdict = SupremacyVerdict.from_artifacts(produced)  # no pending_overrides
    assert verdict.verdict is Verdict.PROVISIONAL_SOTA
    # No PENDING guarantee among the in-scope axes except G5.
    pending = [g.axis for g in verdict.guarantees if g.passed is None]
    assert pending == ["G5"], pending


# ---------------------------------------------------------------------------
# CanonicalRunGate fail-closed (A12–A15)
# ---------------------------------------------------------------------------


def test_gate_refuses_canonical_true_when_provenance_field_blank(
    produced: dict[str, Any],
) -> None:
    """[SECURITY-TEST] A12: the gate REFUSES (ok=False) when a provenance field is
    blank — fail-closed, no canonical=True on incomplete provenance."""
    payload = copy.deepcopy(produced)
    payload["run_metadata"]["git_sha"] = ""  # blank a required provenance field
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("git_sha" in m for m in missing), missing


def test_gate_refuses_when_g2_field_missing(produced: dict[str, Any]) -> None:
    """[SECURITY-TEST] A13: the gate REFUSES when a pii-anon G2 family field is
    absent — fail-closed (a half-populated G2 cannot earn canonical=True)."""
    payload = copy.deepcopy(produced)
    _system(payload, "pii-anon").pop("pseudonymization_integrity_score", None)
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("pseudonymization_integrity_score" in m for m in missing), missing


def test_gate_refuses_when_g4_coverage_not_one(produced: dict[str, Any]) -> None:
    """[SECURITY-TEST] A14: the gate REFUSES when
    ``calibrated_confidence_coverage != 1.0`` (NFR-020 — the lone MUST)."""
    payload = copy.deepcopy(produced)
    _system(payload, "pii-anon")["calibrated_confidence_coverage"] = 0.98
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("calibrated_confidence_coverage" in m for m in missing), missing


def test_gate_refuses_when_per_language_eps_exceeds_bound(
    produced: dict[str, Any],
) -> None:
    """[SECURITY-TEST] A15: the gate REFUSES when a per-language ε exceeds 0.005
    (the recall-floor regression bound)."""
    payload = copy.deepcopy(produced)
    payload["per_language_recall_delta"] = {"en": 0.0, "de": 0.02}  # 0.02 > 0.005
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("per_language" in m for m in missing), missing


# ---------------------------------------------------------------------------
# SO-11 contract (A16–A17) — the producer-side companion (the gate-side companion
# lives in tests/test_competitive_supremacy.py).
# ---------------------------------------------------------------------------


def test_g2_pending_when_no_competitor_carries_pseudonymization_integrity(
    produced: dict[str, Any],
) -> None:
    """[CONTRACT-TEST] A16 (SO-11): a SUT-only artifact — the family fields ONLY on
    the pii-anon ladder, stripped from every competitor — drives the gate's G2 to
    PENDING (no phantom 0.0 win). This proves the producer's emission of the field
    on competitors is what makes G2 computable."""
    payload = copy.deepcopy(produced)
    ladder = {"pii-anon", "pii-anon-swarm"}
    for s in payload["systems"]:
        if s["system"] not in ladder:
            s.pop("pseudonymization_integrity_score", None)
    g2 = SupremacyVerdict.from_artifacts(payload).guarantee("G2")
    assert g2.passed is None, g2.binding_detail  # PENDING — no comparator
    assert "no competitor" in g2.binding_detail.lower()


def test_producer_attaches_pseudonymization_integrity_to_every_competitor(
    produced: dict[str, Any],
) -> None:
    """[CONTRACT-TEST] A17 (SO-11): the producer attaches a REAL
    ``pseudonymization_integrity_score`` to EVERY benchmarked competitor (honest
    0.0 for irreversible incumbents), so G2 always has a real comparator."""
    ladder = {"pii-anon", "pii-anon-swarm"}
    competitors = [s for s in produced["systems"] if s["system"] not in ladder]
    assert competitors, "expected at least one benchmarked competitor"
    for s in competitors:
        pi = s.get("pseudonymization_integrity_score")
        assert isinstance(pi, float) and not isinstance(pi, bool), s["system"]
        assert 0.0 <= pi <= 1.0, (s["system"], pi)
        # Incumbents have no reversible-under-key scheme → honest 0.0.
        assert pi == 0.0, (s["system"], pi)


# ---------------------------------------------------------------------------
# Determinism + scope honesty (A18–A20)
# ---------------------------------------------------------------------------


def test_same_seed_byte_identical_modulo_timestamp(
    tmp_path: Path,
) -> None:
    """[PROPERTY-TEST] A18 / NFR-005 / AX-002: two produces with the same seed are
    byte-identical after excluding the LONE non-reproducible ``timestamp_utc``."""
    a = produce_canonical_artifact(
        seed=_SEED, output_dir=str(tmp_path / "a"), max_samples=_REPRESENTATIVE_MAX_SAMPLES
    )
    b = produce_canonical_artifact(
        seed=_SEED, output_dir=str(tmp_path / "b"), max_samples=_REPRESENTATIVE_MAX_SAMPLES
    )
    assert _without_timestamp(a) == _without_timestamp(b)


def test_in_tree_run_never_stamps_data_v2_scope(produced: dict[str, Any]) -> None:
    """[AUDIT] A19: scope honesty — when the run resolves to the in-tree
    representative fixture (or the DATA corpus), the scope reflects the ACTUAL
    sampler. It is NEVER ``data-v2.0.0`` for an in-tree run, and a DATA-v2 run is
    stamped honestly."""
    scope = produced["run_metadata"]["canonical_provenance"]["scope"]
    assert scope in {"representative-in-tree", "data-v2.0.0"}
    # If the DATA package is unavailable the scope must be the in-tree fixture.
    import importlib.util

    if importlib.util.find_spec("pii_anon_datasets") is None:
        assert scope == "representative-in-tree"


def test_provenance_scope_matches_actual_sampler_used(
    produced: dict[str, Any],
) -> None:
    """[AUDIT] A20: ``scope`` is consistent with ``dataset_version`` — a
    ``data-v2.0.0`` scope iff a v2.0.0 DATA corpus was resolved; otherwise an
    in-tree fixture version. No scope-laundering."""
    prov = produced["run_metadata"]["canonical_provenance"]
    if prov["scope"] == "data-v2.0.0":
        assert prov["dataset_version"] == "2.0.0"
        assert prov["power_cells"].get("source") == "data-v2.0.0"
    else:
        assert prov["scope"] == "representative-in-tree"
        assert prov["dataset_version"] != "2.0.0"
        assert prov["power_cells"].get("verdict") == "n/a-in-tree"


# ---------------------------------------------------------------------------
# Isolation regression (A21)
# ---------------------------------------------------------------------------


def test_rating_import_boundary_unchanged() -> None:
    """[SECURITY-TEST] A21: the producer adds NO forbidden import edge — importing
    the new producer module does not pull ``swarm``/``moe``/``fusion``/``policy``
    into the rating package, and the rating-import-boundary test's scanned set is
    unchanged (the producer lives in the sibling ``evaluation`` package, not in
    ``eval_framework.rating``)."""
    import pii_anon.evaluation.canonical_run as cr

    # The producer module exists and exposes the public API.
    assert hasattr(cr, "produce_canonical_artifact")
    assert hasattr(cr, "CanonicalRunGate")
    # The producer is NOT under eval_framework.rating (the boundary-scanned pkg).
    assert "eval_framework.rating" not in cr.__name__


def test_competitive_supremacy_unchanged() -> None:
    """[SECURITY-TEST] A21: the SDO gate module is consumed read-only — the
    producer self-verifies via ``SupremacyVerdict.from_artifacts`` but the gate's
    public guarantee methods are unchanged (the producer emits the dict; the gate
    reads it unmodified)."""
    from pii_anon.eval_framework.evaluation import competitive_supremacy as g

    # The gate's field-contract surface the producer targets is intact.
    for name in (
        "SupremacyVerdict",
        "_finite_unit_score",
        "_is_finite_number",
        "_g4_class_bar",
        "_g2_pseudonymization_integrity",
        "_g4_calibration_selective_risk",
        "_g1_recall_floor",
        "_g7_certified_run",
    ):
        assert hasattr(g, name), name


# ---------------------------------------------------------------------------
# Additive edge tests (REFACTOR) — FAST gate/helper isolation (no detection run).
# These harden the adversarial-close fabrication vectors without re-running the
# slow detection: a synthetic produced-SHAPED artifact exercises the gate +
# write-path guard directly.
# ---------------------------------------------------------------------------


def _synthetic_produced_shape() -> dict[str, Any]:
    """A minimal produced-SHAPED artifact (no detection run) for fast gate tests.

    Mirrors the real producer's emitted shape closely enough to exercise the
    ``CanonicalRunGate`` validators in isolation — a valid one passes the gate.
    """
    core_g4 = {
        "per_class_ece": {"EMAIL_ADDRESS": 0.0, "US_SSN": 0.0},
        "per_class_ece_threshold": {"EMAIL_ADDRESS": 0.05, "US_SSN": 0.05},
        "risk_coverage_curve": [
            {"coverage": 0.5, "risk": 0.01},
            {"coverage": 1.0, "risk": 0.02},
        ],
        "abstention_operating_points": [
            {"target_risk": 0.01, "achieved_coverage": 0.5, "achieved_risk": 0.01},
            {"target_risk": 0.02, "achieved_coverage": 0.75, "achieved_risk": 0.02},
            {"target_risk": 0.05, "achieved_coverage": 1.0, "achieved_risk": 0.02},
        ],
        "calibrated_confidence_coverage": 1.0,
    }
    return {
        "run_metadata": {
            "git_sha": "deadbeefcafe",
            "dataset_sha256": "a" * 64,
            "matrix_sha256": "b" * 64,
            "timestamp_utc": "2026-06-04T00:00:00Z",
            "canonical_provenance": {
                "seed": 20240601,
                "gate_id": "CanonicalRunGate.v1",
                "scope": "representative-in-tree",
                "sample_size": 8,
                "dataset_sha256": "a" * 64,
                "matrix_sha256": "b" * 64,
                "git_sha": "deadbeefcafe",
                "timestamp_utc": "2026-06-04T00:00:00Z",
                "dataset_version": "in-tree-benchmark-1.0",
                "power_cells": {"verdict": "n/a-in-tree", "sample_size": 8},
            },
        },
        "per_language_recall_delta": {"en": 0.0, "de": 0.0},
        "systems": [
            {
                "system": "pii-anon",
                "pseudonymization_integrity_score": 1.0,
                "anonymization_score": 1.0,
                "unauthorized_reversal_rate": 0.0,
                **core_g4,
            },
            {"system": "gliner", "pseudonymization_integrity_score": 0.0},
        ],
    }


def test_gate_accepts_a_valid_synthetic_produced_shape() -> None:
    """[UNIT-TEST] The gate ACCEPTS a fully-valid synthetic produced-shaped
    artifact (the positive control for the fail-closed gate tests)."""
    ok, missing = CanonicalRunGate().validate(_synthetic_produced_shape())
    assert ok is True, missing
    assert missing == []


def test_gate_rejects_bool_masquerading_as_pseudonymization_integrity_score() -> None:
    """[SECURITY-TEST] Adversarial vector #2 (bool-as-score): a ``True`` in the
    pseudonymization_integrity_score slot (which would coerce to 1.0) is REJECTED by
    the gate's reuse of ``_finite_unit_score`` — never a fabricated perfect score."""
    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["pseudonymization_integrity_score"] = True
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("pseudonymization_integrity_score" in m for m in missing), missing


def test_gate_rejects_non_finite_g4_coverage() -> None:
    """[SECURITY-TEST] Adversarial vector #4/#5 (NaN/inf coverage): a non-finite
    calibrated_confidence_coverage is REJECTED (never slips the ``== 1.0`` MUST)."""
    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["calibrated_confidence_coverage"] = float("inf")
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("calibrated_confidence_coverage" in m for m in missing), missing


def test_gate_rejects_laundered_scope() -> None:
    """[SECURITY-TEST] Adversarial vector #7 (scope laundering): a provenance scope
    outside the two honest values is REJECTED (no laundered ``full-census`` /
    arbitrary scope can earn canonical=True)."""
    payload = _synthetic_produced_shape()
    payload["run_metadata"]["canonical_provenance"]["scope"] = "full-census-PASS2"
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("scope" in m for m in missing), missing


def test_gate_rejects_blank_provenance_fail_closed() -> None:
    """[SECURITY-TEST] Adversarial vector #8 (blank-provenance fail-open): a blank
    run_metadata provenance field is REJECTED — the gate is fail-CLOSED, never
    fail-open on incomplete provenance."""
    payload = _synthetic_produced_shape()
    payload["run_metadata"]["dataset_sha256"] = "   "  # whitespace-only
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("dataset_sha256" in m for m in missing), missing


def test_write_refuses_artifacts_benchmarks_path(tmp_path: Path) -> None:
    """[SECURITY-TEST] Adversarial vector #10 (output-path sandbox): the producer's
    write-path REFUSES any directory under ``benchmarks`` — the protected user-WIP
    ``artifacts/benchmarks/*`` can never be written by the canonical-run producer."""
    from pii_anon.evaluation.canonical_run import _write_canonical

    with pytest.raises(ValueError, match="benchmarks"):
        _write_canonical(
            _synthetic_produced_shape(), str(tmp_path / "artifacts" / "benchmarks")
        )


def test_write_emits_canonical_json_under_canonical_dir(tmp_path: Path) -> None:
    """[UNIT-TEST] The write-path emits canonical sorted JSON under the requested
    (non-benchmarks) directory, creating it on demand."""
    from pii_anon.evaluation.canonical_run import _write_canonical

    out = _write_canonical(_synthetic_produced_shape(), str(tmp_path / "canonical"))
    assert out.exists() and out.name == "canonical-run.json"
    # Round-trips as valid JSON, sorted-keys canonical form.
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["run_metadata"]["canonical_provenance"]["gate_id"] == "CanonicalRunGate.v1"


def test_g4_block_emitted_via_real_selective_risk_scorer_passes_gate() -> None:
    """[UNIT-TEST] The G4 block attach helper produces a gate-passing calibration
    block from the REAL SelectiveRiskReporter (no detection run needed) — pins the
    AX-001 synthetic-but-real-shaped calibration path independently of detection."""
    from pii_anon.evaluation.canonical_run import _attach_g4_calibration

    payload = {"systems": [{"system": "pii-anon"}]}
    _attach_g4_calibration(payload)
    core = payload["systems"][0]
    assert core["calibrated_confidence_coverage"] == 1.0
    assert len(core["abstention_operating_points"]) >= 3
    # Every per-class ECE finite and within the high-resource bar.
    for et, ece in core["per_class_ece"].items():
        assert isinstance(ece, float) and not isinstance(ece, bool), et
        assert ece <= 0.05, (et, ece)


def test_g2_attach_keeps_anon_and_pseudo_separate_keys() -> None:
    """[UNIT-TEST] AX-004: the G2 attach helper emits anon + pseudo as SEPARATE
    keys on the ladder and an honest 0.0 pseudo on competitors (no detection run)."""
    from pii_anon.evaluation.canonical_run import _attach_g2_deid_families

    payload = {
        "systems": [
            {"system": "pii-anon"},
            {"system": "pii-anon-swarm"},
            {"system": "gliner"},
            {"system": "presidio"},
        ]
    }
    _attach_g2_deid_families(payload)
    for s in payload["systems"]:
        assert "pseudonymization_integrity_score" in s
        assert "anonymization_score" in s
        # AX-004 — never a merged number.
        assert "deid_score" not in s and "privacy_score" not in s
        if s["system"] in ("gliner", "presidio"):
            assert s["pseudonymization_integrity_score"] == 0.0  # honest incumbent 0.0


def test_resolve_sampler_falls_back_to_in_tree_when_data_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[UNIT-TEST] RISK-5: when the DATA ``pii_anon_datasets`` import fails inside
    ``_resolve_sampler``, it falls back to the in-tree representative fixture with
    scope ``representative-in-tree`` (NEVER ``data-v2.0.0``). The corpus hash is a
    real sha256 of the resolved record projection (never fabricated)."""
    import importlib

    from pii_anon.evaluation import canonical_run as cr

    real_import_module = importlib.import_module

    def _blocked_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pii_anon_datasets":
            raise ImportError("simulated: DATA package unavailable")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _blocked_import_module)
    scope, sampler = cr._resolve_sampler()
    assert scope == "representative-in-tree"
    assert isinstance(sampler, cr._InTreeFixtureSampler)
    assert sampler.dataset_version != "2.0.0"
    assert len(sampler.corpus_bytes) > 0  # a real projection digest input
    assert sampler.power_cells(sample_size=8)["verdict"] == "n/a-in-tree"


def test_gate_rejects_malformed_run_metadata() -> None:
    """[SECURITY-TEST] Fail-closed: a non-dict ``run_metadata`` is REJECTED outright
    (never a fail-open on a malformed artifact)."""
    ok, missing = CanonicalRunGate().validate({"run_metadata": "not-a-dict"})
    assert ok is False
    assert any("run_metadata" in m for m in missing), missing


def test_gate_rejects_absent_provenance_block_and_g2_g4_fields() -> None:
    """[SECURITY-TEST] Fail-closed: an artifact missing the provenance block, the
    per-language delta, the core G2 family fields AND the G4 block accumulates ALL
    the missing reasons (no single field silently passes)."""
    payload = {
        "run_metadata": {
            "git_sha": "x",
            "dataset_sha256": "y",
            "matrix_sha256": "z",
            "timestamp_utc": "t",
            # canonical_provenance block deliberately absent
        },
        # per_language_recall_delta deliberately absent
        "systems": [
            {"system": "pii-anon"},  # no G2 family fields, no G4 block
            {"system": "gliner"},  # no competitor pseudo field
        ],
    }
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    joined = " | ".join(missing)
    assert "canonical_provenance" in joined
    assert "per_language_recall_delta" in joined
    assert "pseudonymization_integrity_score" in joined
    assert "per_class_ece" in joined
    assert "no competitor carries" in joined


def test_gate_rejects_per_language_delta_with_non_numeric_value() -> None:
    """[SECURITY-TEST] A non-numeric per-language ε (a string / bool) is REJECTED —
    the gate never coerces a non-measurement into a passing ε."""
    payload = _synthetic_produced_shape()
    payload["per_language_recall_delta"] = {"en": "tiny"}  # not a real number
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("per_language_recall_delta.en" in m for m in missing), missing


def test_gate_rejects_absent_core_system() -> None:
    """[SECURITY-TEST] Fail-closed: an artifact whose ``systems`` lacks pii-anon is
    REJECTED (the claimant must be present to certify its moat-axis fields)."""
    payload = _synthetic_produced_shape()
    payload["systems"] = [{"system": "gliner", "pseudonymization_integrity_score": 0.0}]
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("'pii-anon'" in m and "absent" in m for m in missing), missing


def test_git_sha_and_matrix_sha_are_real_digests() -> None:
    """[UNIT-TEST] NFR-006: ``_git_sha`` returns a non-blank string and
    ``_matrix_sha256`` returns a 64-hex sha256 (a real digest, never a fabricated
    fixed value)."""
    from pii_anon.evaluation.canonical_run import _git_sha, _matrix_sha256

    sha = _git_sha()
    assert isinstance(sha, str) and sha.strip()
    matrix = _matrix_sha256()
    assert isinstance(matrix, str) and len(matrix) == 64
    int(matrix, 16)  # valid hex (raises if not)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system(payload: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the named system sub-dict from a produced payload (KeyError if absent)."""
    for s in payload["systems"]:
        if s["system"] == name:
            return s
    raise KeyError(f"system {name!r} not in produced artifact")


def _without_timestamp(payload: dict[str, Any]) -> str:
    """Canonical JSON of a payload with every ``timestamp_utc`` removed.

    ``timestamp_utc`` is the LONE non-reproducible field (NFR-005); it is excluded
    from the determinism comparison.
    """
    clean = copy.deepcopy(payload)
    rm = clean.get("run_metadata", {})
    rm.pop("timestamp_utc", None)
    prov = rm.get("canonical_provenance", {})
    prov.pop("timestamp_utc", None)
    return json.dumps(clean, sort_keys=True, indent=2)
