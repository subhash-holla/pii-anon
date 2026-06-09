"""S7-02 — the canonical-run producer + ``CanonicalRunGate`` (the KEYSTONE).

This is the program's **completion-criterion artifact** test suite. It pins the
producer that runs the benchmark at REPRESENTATIVE scale and emits a CERTIFIED
canonical artifact carrying the G1 per-language ε + the G2 distinct anon/pseudo
family fields (every system, incl. competitors — the SO-11 fix) + the G4
calibration block — all REUSING the existing in-tree scorers (no new gate math).
A fail-closed :class:`CanonicalRunGate` validates ALL required fields BEFORE
allowing ``canonical_claim_run=True``.

The KEYSTONE TEETH (A6–A11): after the producer builds the artifact, route it
through the UNMODIFIED SDO gate ``SupremacyVerdict.from_artifacts`` and assert the
MACHINERY is honest — G1/G2/G4 are NOT PENDING (they compute from the real
scorers), G7 PASS (canonical_claim_run True + provenance), and the verdict is
whatever ``from_artifacts`` HONESTLY computes from the FRESH-measured detection
metrics. The verdict is scale-dependent and NOT hardcoded: the composite/J
dominance genuinely requires full-census scale (a sub-census sample's
regex-vs-neural composite race is razor-thin / flips to a neural competitor), so at
fast CI scale the honest verdict is one of {PROVISIONAL_SOTA, NOT_YET}, and IF
NOT_YET the binding constraint is an honest raw-detection axis (the composite/J or
a raw-detection guarantee), NEVER a fabrication. PROVISIONAL_SOTA is certified by
the documented full-census re-run on the current code (the user's Pass-2; see the
artifact's ``pass2_full_census_reference`` block).

No-fabrication contract (story §2a): every emitted field is a REAL scorer output
the gate's validators ACCEPT (finite, in-range, non-bool). The gate-read detection
metrics (``recall`` / ``precision`` / ``per_entity_recall`` and the
``composite_score`` derived from them with a fixed reference speed) are the FRESH
measurement on the current code — NO prior-run numbers in the gate-read path. The
documented PRIOR full-census run is surfaced as TRANSPARENT
``pass2_full_census_reference`` data, NEVER gate-read. G2/G4 are real
``deid_families`` / ``selective_risk`` scorer outputs on synthetic-but-real-shaped
inputs (AX-001). G1 is the real S1-02 floored-fusion recall-floor primitive. All
corpora are SYNTHETIC (AX-001 — never real PII).

Determinism (NFR-005 / AX-002): seed-driven; the gate-read detection-quality
metrics are a pure function of the record set (IDENTICAL whether ``enable_parallel``
is ``True`` or ``False``); canonical sorted JSON + ``round(., 6)``; ``timestamp_utc``
is the LONE field excluded from the determinism comparison.

The integration tests build the artifact via ``produce_canonical_artifact`` with a
SMALL ``max_samples`` so they stay FAST (a high-variance estimator — NOT a census);
the REAL produced artifact uses ``max_samples=None`` (the FULL in-tree dataset).
Scope is stamped HONESTLY from the actual sampler used.
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

# A SMALL cap so the real detection run stays FAST in CI (a high-variance estimator,
# NOT a census — the real produced artifact uses max_samples=None / the full dataset).
_REPRESENTATIVE_MAX_SAMPLES = 8
_SEED = 20240601


# ---------------------------------------------------------------------------
# A module-scoped produced artifact (the real detection run is the slow part;
# build it ONCE and share the read-only dict across the keystone-teeth tests).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Produce the canonical artifact ONCE (real detection + real scorers).

    Single-threaded (``enable_parallel=False``) for the deterministic CI run, at a
    SMALL ``max_samples`` so it is FAST. Writes ONLY under a pytest tmp dir (never the
    repo's ``artifacts/*``).
    """
    out_dir = tmp_path_factory.mktemp("canonical_out")
    return produce_canonical_artifact(
        seed=_SEED,
        output_dir=str(out_dir),
        max_samples=_REPRESENTATIVE_MAX_SAMPLES,
        enable_parallel=False,
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


def test_verdict_is_honest_on_produced_artifact(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A10 (the keystone — INTEGRITY OVER OUTCOME): the gate's
    verdict on the produced artifact is whatever ``from_artifacts`` HONESTLY computes
    from the FRESH-measured detection metrics — it is NOT hardcoded to
    PROVISIONAL_SOTA.

    The MACHINERY is asserted (canonical_claim_run True; G7 PASS; G5 PASS — S7-04
    closed the last placeholder). The
    verdict is one of {PROVISIONAL_SOTA, NOT_YET} (scale-dependent — the composite/J
    crown genuinely requires full-census scale, so a small CI sample typically lands
    NOT_YET), and IF NOT_YET the binding constraint is an HONEST raw-detection axis
    (the composite/J gap, or a raw-detection guarantee G1/G3/G6/G7) — NEVER a
    fabrication. PROVISIONAL_SOTA is certified by the documented full-census re-run
    (Pass-2)."""
    verdict = SupremacyVerdict.from_artifacts(produced)
    # The run IS certified (the gate validated the fields) — the machinery is honest.
    assert verdict.canonical_claim_run is True
    assert verdict.guarantee("G7").passed is True, verdict.guarantee("G7").binding_detail
    # G5 COMPUTES on the produced artifact (S7-04): real measured latency within
    # the committed ceiling + a clean audit ⇒ PASS (no longer the placeholder).
    assert verdict.guarantee("G5").passed is True, verdict.guarantee("G5").binding_detail
    # The verdict is honest, scale-dependent — NOT a manufactured PROVISIONAL_SOTA.
    assert verdict.verdict in (Verdict.PROVISIONAL_SOTA, Verdict.NOT_YET), verdict.verdict
    if verdict.verdict is Verdict.PROVISIONAL_SOTA:
        # If it honestly clears the bar at this scale: J ≥ bar + an unrun-tier blocker.
        assert verdict.j_value is not None and verdict.j_value >= 0.95, verdict.j_value
        assert verdict.unrun_tier_c or verdict.unrun_tier_r
    else:
        # NOT_YET must bind on an HONEST axis — the composite/J or a raw-detection
        # guarantee — never a missing/fabricated field. (canonical is True, so the
        # binding is NOT 'canonical_claim_run=False'.)
        binding = verdict.binding_constraint
        assert "canonical_claim_run=False" not in binding, binding
        honest_axis = (
            "J=" in binding  # the composite-rank J gap
            or "J unavailable" in binding
            or any(
                # G5 is unreachable here today (asserted PASS above) but listed
                # so the guard stays complete if a future env honestly breaches
                # the latency ceiling.
                f"{ax} FAIL" in binding for ax in ("G1", "G3", "G5", "G6", "G7")
            )  # an honest measured guarantee
        )
        assert honest_axis, f"NOT_YET must bind on an honest axis, got: {binding}"


def test_produced_artifact_round_trips_through_from_artifacts_without_overrides(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] A11: the produced artifact drives the gate WITHOUT any
    ``pending_overrides`` — G1/G2/G3/G4/G5 ALL COMPUTE from the emitted fields alone
    (the override seam is NOT relied upon; S7-04 closed the last placeholder). NO
    axis is PENDING; the verdict is whatever the fresh metrics honestly yield."""
    verdict = SupremacyVerdict.from_artifacts(produced)  # no pending_overrides
    # Every guarantee computes (NOT PENDING) from the emitted fields (S7-04).
    pending = [g.axis for g in verdict.guarantees if g.passed is None]
    assert pending == [], pending
    # The verdict is honestly computed (one of the two valid endpoints) — not hardcoded.
    assert verdict.verdict in (Verdict.PROVISIONAL_SOTA, Verdict.NOT_YET), verdict.verdict


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
    byte-identical after excluding the TWO sanctioned non-reproducible surfaces —
    ``timestamp_utc`` + the measured wall-clock ``latency_summary`` (S7-04; see
    ``_without_timestamp``). The keyed-deterministic ``audit_summary`` is INCLUDED
    in the comparison."""
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
# Vector #11 — the SDO gate's _is_finite_number hardened against a huge-int field
# (the ONE sanctioned gate change; carried from SO-13/SO-14). A Python int wider
# than a C double raises OverflowError in math.isfinite — a control-path validator
# must REJECT it (fail CLOSED), never crash (a fail-loud denial-of-verdict).
# ---------------------------------------------------------------------------


def test_vector11_is_finite_number_rejects_unbounded_int_no_crash() -> None:
    """[SECURITY-TEST] Vector #11: ``_is_finite_number(10**400)`` returns ``False``
    (fail-CLOSED) rather than raising OverflowError. The huge int is wider than a C
    double, so ``math.isfinite`` would crash without the guard."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _is_finite_number,
    )

    assert _is_finite_number(10**400) is False
    assert _is_finite_number(-(10**400)) is False
    # Ordinary values still behave.
    assert _is_finite_number(0.5) is True
    assert _is_finite_number(1) is True
    assert _is_finite_number(True) is False  # a bool is not a measurement


def test_vector11_finite_unit_score_rejects_unbounded_int_no_crash() -> None:
    """[SECURITY-TEST] Vector #11: ``_finite_unit_score(10**400)`` returns ``None``
    (treat-as-absent) rather than raising — the moat-axis validator is robust to a
    huge-int field."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import (
        _finite_unit_score,
    )

    assert _finite_unit_score(10**400) is None
    assert _finite_unit_score(0.73) == 0.73


def test_vector11_canonical_gate_rejects_huge_int_g2_field_no_crash() -> None:
    """[SECURITY-TEST] Vector #11 reaching the NEW gate: a huge-int
    ``pseudonymization_integrity_score`` makes ``CanonicalRunGate.validate`` return
    ``(False, [...])`` (fail-CLOSED) instead of raising OverflowError."""
    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["pseudonymization_integrity_score"] = 10**400
    ok, missing = CanonicalRunGate().validate(payload)  # must NOT raise
    assert ok is False
    assert any("pseudonymization_integrity_score" in m for m in missing), missing


def test_vector11_sdo_gate_g2_pending_on_huge_int_field_no_crash(
    produced: dict[str, Any],
) -> None:
    """[SECURITY-TEST] Vector #11 in the SDO gate's own G2 path: a huge-int moat
    field on every competitor of the produced artifact drives G2 to PENDING (no valid
    comparator), never a crash — the gate stays robust against an adversarial
    artifact (the full ``from_artifacts`` path, not just the CanonicalRunGate)."""
    payload = copy.deepcopy(produced)
    ladder = {"pii-anon", "pii-anon-swarm"}
    for s in payload["systems"]:
        if s["system"] not in ladder:
            s["pseudonymization_integrity_score"] = 10**400  # wider than a C double
    g2 = SupremacyVerdict.from_artifacts(payload).guarantee("G2")  # must NOT raise
    assert g2.passed is None, g2.binding_detail  # PENDING — every comparator rejected
    assert "no competitor" in g2.binding_detail.lower()


def test_vector11_g4_class_bar_rejects_huge_int_threshold_no_crash() -> None:
    """[SECURITY-TEST] Vector #11 on the G4 ECE-threshold clamp: a huge-int
    artifact-supplied per-class threshold is REJECTED (the conservative sanctioned
    bar stands) rather than crashing ``_g4_class_bar``."""
    from pii_anon.eval_framework.evaluation.competitive_supremacy import _g4_class_bar

    bar = _g4_class_bar(10**400)  # must NOT raise
    assert bar <= 0.08  # falls back to the sanctioned bar, never loosened


# ---------------------------------------------------------------------------
# Scope honesty — the gate-read metrics are FRESH-measured + the documented census
# is a TRANSPARENT Pass-2 reference (NEVER gate-read). The remediation core.
# ---------------------------------------------------------------------------


def test_gate_read_systems_are_fresh_not_census_literals(
    produced: dict[str, Any],
) -> None:
    """[AUDIT] The gate-read ``systems`` recall/precision are the FRESH in-tree
    measurement (no prior-run census numbers in the gate-read path). They match the
    separately-surfaced fresh ``representative_in_tree_detection`` block exactly, and
    do NOT equal the documented census literals (e.g. pii-anon census recall
    0.7958)."""
    fresh = produced["representative_in_tree_detection"]["systems"]
    for s in produced["systems"]:
        name = s["system"]
        # The gate-read recall/precision == the fresh detection measurement verbatim.
        assert s["recall"] == fresh[name]["recall"], name
        assert s["precision"] == fresh[name]["precision"], name
    # The census recall for pii-anon (0.7958) must NOT appear as the gate-read recall
    # (the laundering the remediation removed) — the fresh small-sample value differs.
    core = _system(produced, "pii-anon")
    assert core["recall"] != 0.7958, "gate-read recall must be fresh, not the census literal"


def test_producer_strips_benchmark_ignore_sentinel_from_emitted_per_entity_recall() -> None:
    """[REGRESSION] The benchmark-exclusion sentinel ``_BENCHMARK_IGNORE`` (schema.py maps
    LATITUDE_LONGITUDE / TIMESTAMP / SWIFT_BIC_CODE to it; the gold side skips labels mapped
    to it at schema.py:377) must NEVER leak into a system's EMITTED ``per_entity_recall``.

    The leak made the SDO gate's G1 spuriously FAIL: a competitor that "detects" the sentinel
    (recall > 0, e.g. gliner 0.333) while the ensemble does not (0.0, not a detection) reads as
    ``ensemble misses competitor-detected entities ['_BENCHMARK_IGNORE']`` — a FAIL on a
    non-entity that is BY DEFINITION excluded from benchmarking. The prediction-side
    per_entity_recall the gate reads must mirror the gold-side exclusion. Real entities are
    preserved verbatim (no over-stripping)."""
    from types import SimpleNamespace

    from pii_anon.evaluation.canonical_run import _assemble_base_payload

    report = SimpleNamespace(
        dataset="synthetic",
        dataset_source="in-tree",
        systems=[
            SimpleNamespace(
                system="pii-anon",
                recall=0.9,
                precision=0.9,
                per_entity_recall={"EMAIL": 0.95, "_BENCHMARK_IGNORE": 0.0},
                qualification_status="core",
                available=True,
                samples=8,
            ),
            SimpleNamespace(
                system="gliner",
                recall=0.8,
                precision=0.8,
                per_entity_recall={"EMAIL": 0.7, "_BENCHMARK_IGNORE": 0.333},
                qualification_status="qualified",
                available=True,
                samples=8,
            ),
        ],
    )
    sampler = SimpleNamespace(dataset="synthetic", dataset_source="in-tree")
    payload = _assemble_base_payload(
        report, scope="representative-in-tree", sampler=sampler, max_samples=8
    )

    for s in payload["systems"]:
        assert "_BENCHMARK_IGNORE" not in s["per_entity_recall"], (
            f"{s['system']} leaks the _BENCHMARK_IGNORE sentinel into emitted per_entity_recall"
        )
        # The real entity is preserved verbatim — the filter strips ONLY the sentinel.
        assert s["per_entity_recall"].get("EMAIL") is not None, s["system"]


def test_pass2_full_census_reference_present_and_not_gate_read(
    produced: dict[str, Any],
) -> None:
    """[AUDIT] The documented PRIOR full-census run is surfaced as a TRANSPARENT
    ``pass2_full_census_reference`` block (source git_sha 2761a27,
    canonical_claim_run=False, 148994 records) — clearly labelled Pass-2 reference
    data, and NEVER read by the gate (it lives outside the ``systems`` block)."""
    ref = produced["pass2_full_census_reference"]
    assert ref["source_git_sha"].startswith("2761a27")
    assert ref["source_record_count"] == 148994
    assert ref["source_canonical_claim_run"] is False
    assert ref["source_dataset_source"] == "auto"
    assert ref["pii_anon_composite"] == 0.784583
    assert ref["gliner_composite"] == 0.680213
    assert "Pass-2" in ref["note"] or "Pass-2".lower() in ref["note"].lower()
    # It is NOT in the gate-read systems list (the gate only reads payload["systems"]).
    system_names = {s["system"] for s in produced["systems"]}
    assert "pass2_full_census_reference" not in system_names


def test_no_census_literals_module_attribute() -> None:
    """[AUDIT] The scope-laundering census-profile machinery is REMOVED from the
    gate-read path — ``_CENSUS_PROFILES`` / ``_DetectionProfile`` no longer exist as
    module attributes (the remediation's structural guarantee)."""
    from pii_anon.evaluation import canonical_run as cr

    assert not hasattr(cr, "_CENSUS_PROFILES")
    assert not hasattr(cr, "_DetectionProfile")
    assert not hasattr(cr, "_PII_ANON_PER_ENTITY")


def test_detection_scope_labels_fresh_run_honestly(produced: dict[str, Any]) -> None:
    """[AUDIT] The fresh-detection sub-block is labelled by what it IS — a fresh
    in-tree detection at the actual record count — NOT stamped ``data-v2.0.0`` (the
    MINOR-(b) mislabel the remediation fixed)."""
    block = produced["representative_in_tree_detection"]
    assert block["measurement"] == "fresh-in-tree-detection"
    assert block["detection_scope"].startswith("in-tree-fresh-")
    # The fresh small-sample block is NOT labelled with the corpus scope data-v2.0.0.
    assert block["detection_scope"] != "data-v2.0.0"


def test_gate_rejects_non_finite_per_language_eps_nan() -> None:
    """[SECURITY-TEST] MINOR-(a): a NaN per-language ε is REJECTED by the gate's
    ``_is_finite_number`` guard (it must never slip ``abs(ε) > bound``, whose result
    against NaN is silently False)."""
    payload = _synthetic_produced_shape()
    payload["per_language_recall_delta"] = {"en": 0.0, "de": float("nan")}
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("per_language_recall_delta.de" in m for m in missing), missing


def test_gate_read_composite_is_deterministic_across_parallel_modes(
    tmp_path: Path,
) -> None:
    """[PROPERTY-TEST] NFR-005: the gate-read detection-quality metrics
    (recall / precision / composite_score) are a pure function of the record set —
    IDENTICAL whether ``enable_parallel`` is True or False at the same scale. This is
    what lets the real artifact run multi-threaded while staying byte-deterministic."""
    a = produce_canonical_artifact(
        seed=_SEED,
        output_dir=str(tmp_path / "seq"),
        max_samples=_REPRESENTATIVE_MAX_SAMPLES,
        enable_parallel=False,
    )
    b = produce_canonical_artifact(
        seed=_SEED,
        output_dir=str(tmp_path / "par"),
        max_samples=_REPRESENTATIVE_MAX_SAMPLES,
        enable_parallel=True,
    )

    def _systems(p: dict[str, Any]) -> dict[str, Any]:
        return {
            s["system"]: (s["recall"], s["precision"], s["composite_score"])
            for s in p["systems"]
        }

    assert _systems(a) == _systems(b)


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
            # The S7-04 G5 blocks (latency + audit) — required for certification.
            "latency_summary": {
                "system": "pii-anon-swarm",
                "profile": "ensemble",
                "p50_ms": 80.5,
                "p95_ms": 112.0,
                "p99_ms": 133.3,
                "n_records": 8,
                "measurement": "fresh-in-tree-per-record-detection-timing",
            },
            "audit_summary": {
                "interception": {
                    "counts_by_channel": {
                        "PROMPT": 2,
                        "MEMORY": 1,
                        "TOOL_IO": 1,
                        "TRACE": 1,
                    },
                    "no_raw_pii_persist": True,
                    "records_total": 5,
                },
                "leakage_sankey": {"blocked": 5, "leaked": 0},
                "injection_resistance": {
                    "attack_success_rate": 0.0,
                    "benign_task_success_rate": 1.0,
                    "n_payloads": 4,
                },
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
            {"system": "pii-anon-swarm"},
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


def test_gate_rejects_negative_per_class_ece() -> None:
    """[SECURITY-TEST] close-3 (defense-in-depth, symmetric with the SDO gate): a
    NEGATIVE per_class_ece value is non-physical (ECE ≥ 0 by construction). The real
    selective_risk scorer never emits one, but the fail-closed gate must REJECT it
    so the producer can never certify a sub-zero ECE — mirroring the gate-side fix
    where a negative ECE is a breach, not a vacuous 'within bar' PASS."""
    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["per_class_ece"] = {"EMAIL": -1.0}
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("per_class_ece" in m for m in missing), missing


def test_gate_accepts_zero_per_class_ece() -> None:
    """[UNIT-TEST] close-3 CARDINAL-RULE regression: an HONEST per_class_ece of
    exactly 0.0 (a perfectly calibrated class — what the real scorer emits on the
    synthetic well-calibrated set) is ≥ 0 and still ACCEPTED by the gate."""
    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["per_class_ece"] = {"EMAIL_ADDRESS": 0.0, "US_SSN": 0.0}
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is True, missing
    assert missing == []


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


def test_close_write_refuses_capitalized_benchmarks_path_case_insensitive(
    tmp_path: Path,
) -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-4: the pre-hardening ``"benchmarks" in
    out_dir.parts`` check is CASE-SENSITIVE, so ``artifacts/Benchmarks`` (capital B) on
    a case-insensitive filesystem (APFS) WROTE a real canonical-run.json INTO the
    protected ``artifacts/benchmarks/`` (same inode). The check must be
    case-INSENSITIVE — every casing of ``benchmarks`` is refused, and NOTHING is
    written."""
    from pii_anon.evaluation.canonical_run import _write_canonical

    for variant in ("Benchmarks", "BENCHMARKS", "BenchMarks"):
        target = tmp_path / "artifacts" / variant
        with pytest.raises(ValueError, match="(?i)benchmarks"):
            _write_canonical(_synthetic_produced_shape(), str(target))
        # Fail-closed: nothing written under the refused path.
        assert not (target / "canonical-run.json").exists()
        assert not target.exists()


def test_close_write_refuses_resolved_artifacts_benchmarks_alias(
    tmp_path: Path,
) -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-4: the guard must compare the RESOLVED real
    path, not just the literal string parts — a relative / ``..``-laundered path whose
    RESOLVED location lands under ``.../artifacts/benchmarks`` is refused. A real
    ``artifacts/benchmarks`` directory (created here) aliased via a ``..`` hop must be
    rejected and nothing written into it."""
    from pii_anon.evaluation.canonical_run import _write_canonical

    protected = tmp_path / "artifacts" / "benchmarks"
    protected.mkdir(parents=True)
    sentinel_before = sorted(p.name for p in protected.iterdir())
    # A laundered path: <tmp>/artifacts/canonical/../benchmarks resolves UNDER
    # artifacts/benchmarks — the resolved-real-path guard must refuse it.
    laundered = tmp_path / "artifacts" / "canonical" / ".." / "benchmarks"
    with pytest.raises(ValueError, match="(?i)benchmarks"):
        _write_canonical(_synthetic_produced_shape(), str(laundered))
    # Nothing written into the protected directory.
    assert sorted(p.name for p in protected.iterdir()) == sentinel_before
    assert not (protected / "canonical-run.json").exists()


def test_close_produce_canonical_artifact_refuses_capitalized_benchmarks(
    tmp_path: Path,
) -> None:
    """[SECURITY-TEST] S7-02 close MAJOR-4 (end-to-end producer): the PUBLIC
    ``produce_canonical_artifact(output_dir='artifacts/Benchmarks')`` (capital B) is
    REFUSED before any write — the full producer path enforces the case-insensitive
    sandbox, so nothing is ever written under any ``benchmarks`` casing."""
    target = tmp_path / "artifacts" / "Benchmarks"
    with pytest.raises(ValueError, match="(?i)benchmarks"):
        produce_canonical_artifact(
            seed=_SEED,
            output_dir=str(target),
            max_samples=_REPRESENTATIVE_MAX_SAMPLES,
            enable_parallel=False,
        )
    assert not (target / "canonical-run.json").exists()


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
    """Canonical JSON of a payload with the non-reproducible surfaces removed.

    Two sanctioned exclusions (NFR-005): ``timestamp_utc`` and the MEASURED
    wall-clock ``latency_summary`` (S7-04 — NFR-005 itself excludes wall-clock
    speed from determinism; the timing values vary run-to-run by construction).
    Everything else — including the keyed-deterministic ``audit_summary`` —
    MUST be byte-identical across same-seed produces.
    """
    clean = copy.deepcopy(payload)
    rm = clean.get("run_metadata", {})
    rm.pop("timestamp_utc", None)
    rm.pop("latency_summary", None)
    prov = rm.get("canonical_provenance", {})
    prov.pop("timestamp_utc", None)
    return json.dumps(clean, sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# S7-02 FINAL close (round 2) — CanonicalRunGate.validate must FAIL-CLOSED, not
# crash, on a hostile directly-injected payload (the gate class is exported).
# ---------------------------------------------------------------------------


def test_closefinal_canonical_gate_validate_non_dict_payload_does_not_crash() -> None:
    """[SECURITY-TEST] S7-02 final close: ``CanonicalRunGate.validate`` on a non-dict payload
    (a ``json.loads`` of a top-level array) called ``payload.get`` → ``AttributeError``. It
    must fail CLOSED (ok=False), never raise."""
    ok, missing = CanonicalRunGate().validate([1, 2, 3])  # type: ignore[arg-type]
    assert ok is False
    assert missing


def test_closefinal_canonical_gate_validate_unhashable_scope_does_not_crash() -> None:
    """[SECURITY-TEST] final close: an UNHASHABLE ``canonical_provenance.scope`` (a list)
    crashed ``scope not in {…}`` (``TypeError: unhashable``). A non-str scope is not an honest
    scope ⇒ ok=False + a scope finding, never a crash."""
    payload = {
        "run_metadata": {
            "git_sha": "a",
            "dataset_sha256": "b",
            "matrix_sha256": "c",
            "timestamp_utc": "t",
            "canonical_provenance": {
                "seed": 1,
                "gate_id": "g",
                "scope": ["data-v2.0.0"],
                "sample_size": 8,
                "dataset_sha256": "b",
                "matrix_sha256": "c",
                "git_sha": "a",
                "timestamp_utc": "t",
                "dataset_version": "v",
                "power_cells": {},
            },
        },
        "per_language_recall_delta": {"en": 0.0},
        "systems": [{"system": "pii-anon"}, {"system": "gliner"}],
    }
    ok, missing = CanonicalRunGate().validate(payload)  # must not raise
    assert ok is False
    assert any("scope" in m for m in missing)


@pytest.mark.parametrize(
    "bad_systems",
    [None, 42, 3.14, True, [{"system": ["x"]}], [{"system": {"k": 1}}]],
)
def test_closefinal_canonical_gate_validate_hostile_systems_does_not_crash(
    bad_systems: object,
) -> None:
    """[SECURITY-TEST] S7-02 final close (round 4): ``CanonicalRunGate.validate`` crashed on a
    non-iterable ``systems`` (None/int/float/bool → 'object is not iterable') and on an
    UNHASHABLE system-name (list/dict → 'unhashable' in the ``{s.get("system"): s}`` dict-comp
    key). It must fail CLOSED (ok=False), never raise — the EXPORTED gate is callable directly
    on a hostile dict (the close-5 ``systems`` guard was incomplete)."""
    ok, missing = CanonicalRunGate().validate({"run_metadata": {}, "systems": bad_systems})
    assert ok is False
    assert missing


# ---------------------------------------------------------------------------
# S7-04 — the producer emits the G5 latency + audit blocks; the gate requires
# them (shape + audit-integrity) for certification; the LATENCY CEILING
# comparison stays the SDO gate's job (over-budget = an honest G5 FAIL on a
# still-certified run, like G6's F2 — deliberately NOT a certification defect).
# ---------------------------------------------------------------------------


def test_g5_produced_artifact_emits_latency_and_audit_blocks(
    produced: dict[str, Any],
) -> None:
    """[INTEGRATION-TEST] S7-04: the produced artifact carries BOTH G5 blocks —
    a REAL measured full-swarm latency summary (ordered percentiles, the actual
    timing-sample size) and the REAL S6-02/S6-05 audit summary (4 channels, no
    persist breach, zero leaks, ASR 0 with full benign preservation)."""
    rm = produced["run_metadata"]
    lat = rm["latency_summary"]
    assert lat["system"] == "pii-anon-swarm"
    assert lat["profile"] == "ensemble"
    assert 0.0 <= lat["p50_ms"] <= lat["p95_ms"] <= lat["p99_ms"]
    assert lat["n_records"] == _REPRESENTATIVE_MAX_SAMPLES
    assert isinstance(lat["measurement"], str) and lat["measurement"]

    audit = rm["audit_summary"]
    counts = audit["interception"]["counts_by_channel"]
    assert set(counts) == {"PROMPT", "MEMORY", "TOOL_IO", "TRACE"}
    assert all(isinstance(v, int) and v >= 1 for v in counts.values())
    assert audit["interception"]["no_raw_pii_persist"] is True
    assert audit["leakage_sankey"]["leaked"] == 0
    assert audit["leakage_sankey"]["blocked"] >= 4
    inj = audit["injection_resistance"]
    assert inj["attack_success_rate"] == 0.0
    assert inj["benign_task_success_rate"] == 1.0
    assert inj["n_payloads"] >= 4


def test_g5_audit_summary_carries_no_raw_pii(produced: dict[str, Any]) -> None:
    """[SECURITY-TEST] S7-04 / FR-026: the emitted audit summary is
    surrogate-only — none of the synthetic raw PII values used to exercise the
    guard survive anywhere in the emitted artifact."""
    blob = json.dumps(produced)
    for raw in ("jane.roe@example.test", "555-1000", "555-2000"):
        assert raw not in blob


def test_g5_audit_summary_deterministic_same_seed(tmp_path: Path) -> None:
    """[PROPERTY-TEST] S7-04 / NFR-005 / AX-002: the audit half is
    keyed-deterministic — two same-seed produces yield a byte-identical
    audit_summary (only the wall-clock latency_summary values may differ; the
    main determinism test excludes exactly timestamp_utc + latency_summary)."""
    a = produce_canonical_artifact(
        seed=_SEED, output_dir=str(tmp_path / "a"), max_samples=_REPRESENTATIVE_MAX_SAMPLES
    )
    b = produce_canonical_artifact(
        seed=_SEED, output_dir=str(tmp_path / "b"), max_samples=_REPRESENTATIVE_MAX_SAMPLES
    )
    assert a["run_metadata"]["audit_summary"] == b["run_metadata"]["audit_summary"]


def test_gate_refuses_when_g5_latency_summary_missing() -> None:
    """[SECURITY-TEST] S7-04: a canonical run REQUIRES the latency block — a
    payload without it cannot certify (fail-closed)."""
    payload = _synthetic_produced_shape()
    del payload["run_metadata"]["latency_summary"]
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("latency_summary" in m for m in missing), missing


def test_gate_refuses_when_g5_audit_summary_missing() -> None:
    """[SECURITY-TEST] S7-04: a canonical run REQUIRES the audit block — a
    payload without it cannot certify (fail-closed)."""
    payload = _synthetic_produced_shape()
    del payload["run_metadata"]["audit_summary"]
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("audit_summary" in m for m in missing), missing


def test_gate_refuses_audit_integrity_breach_leak() -> None:
    """[SECURITY-TEST] S7-04: an audit-integrity breach (a leaked span) makes
    the artifact itself untrustworthy — certification REFUSED (like the step-3
    ε bound), not merely a G5 FAIL."""
    payload = _synthetic_produced_shape()
    payload["run_metadata"]["audit_summary"]["leakage_sankey"]["leaked"] = 1
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("leak" in m.lower() for m in missing), missing


def test_gate_refuses_audit_persist_not_strict_true() -> None:
    """[SECURITY-TEST] S7-04: the no_raw_pii_persist stamp certifies only as
    the literal True — the string 'true' is a corrupt stamp (the
    canonical_claim_run coercion lesson), certification REFUSED."""
    payload = _synthetic_produced_shape()
    payload["run_metadata"]["audit_summary"]["interception"]["no_raw_pii_persist"] = "true"
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False


def test_gate_refuses_corrupt_latency_shape() -> None:
    """[SECURITY-TEST] S7-04: a shape-corrupt latency block (NaN percentile /
    inverted order / bogus profile) cannot certify — fail-closed, no crash."""
    for mutation in (
        {"p50_ms": float("nan")},
        {"p50_ms": 200.0, "p95_ms": 100.0},
        {"profile": "bogus"},
        {"system": "gliner"},
        {"n_records": True},
    ):
        payload = _synthetic_produced_shape()
        payload["run_metadata"]["latency_summary"].update(mutation)
        ok, missing = CanonicalRunGate().validate(payload)
        assert ok is False, mutation


def test_gate_accepts_over_budget_latency_shape_g5_fails_honestly() -> None:
    """[CONTRACT-TEST] S7-04 — the deliberate design line: an OVER-BUDGET but
    shape-valid measured latency still CERTIFIES (the run is honest; the
    measurement is real) and G5 then honestly FAILS at the SDO gate — exactly
    like G6's F2 on a certified run. Certification ≠ performance."""
    payload = _synthetic_produced_shape()
    payload["run_metadata"]["latency_summary"]["p99_ms"] = 5000.0  # > 2000 committed
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is True, missing

    payload["run_metadata"]["canonical_claim_run"] = True
    verdict = SupremacyVerdict.from_artifacts(payload)
    assert verdict.canonical_claim_run is True
    assert verdict.guarantee("G5").passed is False
    assert verdict.verdict is Verdict.NOT_YET
    assert "G5 FAIL" in verdict.binding_constraint


def test_gate_validate_huge_int_lang_key_never_crashes() -> None:
    """[SECURITY-TEST] S7-04 DoV class: a >4300-digit int KEY in
    per_language_recall_delta crashed validate's missing-detail f-string (the
    int→str conversion limit) — must fail CLOSED, never raise."""
    payload = _synthetic_produced_shape()
    payload["per_language_recall_delta"] = {10**5000: float("nan")}
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert missing


def test_gate_validate_huge_int_ece_key_and_value_never_crash() -> None:
    """[SECURITY-TEST] S7-04 DoV class: a huge-int per_class_ece KEY (with an
    invalid value, forcing the detail format) and a huge-int ECE VALUE both
    crashed validate's f-strings — must fail CLOSED, never raise."""
    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["per_class_ece"] = {10**5000: float("nan")}
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False

    payload = _synthetic_produced_shape()
    _system(payload, "pii-anon")["per_class_ece"] = {"EMAIL_ADDRESS": 10**5000}
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False


def test_gate_validate_huge_int_scope_never_crashes() -> None:
    """[SECURITY-TEST] S7-04 DoV class: a huge-int canonical_provenance.scope
    crashed the not-an-honest-scope detail f-string — must fail CLOSED."""
    payload = _synthetic_produced_shape()
    payload["run_metadata"]["canonical_provenance"]["scope"] = 10**5000
    ok, missing = CanonicalRunGate().validate(payload)
    assert ok is False
    assert any("scope" in m for m in missing), missing
