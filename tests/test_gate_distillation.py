"""S2-02 — the offline distillation trainer that emits the signed ``gate_v1.json``.

Pins the producer half: ``distill_topk_gate`` distils a frozen survival oracle
(the XGBoost meta-learner, via a ``SurvivalOracle`` Protocol) into a compact
per-(entity_type, expert) gate using RRS up-weighting + the closed-form
BCE/KL-optimal compact student (the RRS-weighted per-cell oracle-survival mean)
→ a temperature-softmax over experts; and ``emit_signed_gate`` signs the payload
via the S2-05 ``KeyRing`` and writes the ``{payload, signature}`` envelope.

Acceptance covered here (the producer half):

* **A5** the closed-form distillation is correct + RRS-weighted: each
  ``weights[et][expert]`` equals the temperature-softmax of the RRS-up-weighted
  oracle-survival MEAN per cell (asserted against an independent recomputation,
  abs 1e-9); raising one cell's oracle survival raises that expert's routing
  weight (monotone teeth); the RRS up-weighting demonstrably shifts a weight vs
  uniform weighting (non-vacuous). ``[UNIT-TEST]`` ``[PROPERTY-TEST]``
* **A6** determinism / replay-equality: same ``(oracle, records, temperature,
  seed)`` ⇒ ``canonical_gate_bytes(payload_1) == canonical_gate_bytes(payload_2)``
  (byte-identical); ``oracle_hash`` + ``gate_feature_version`` stable.
  ``[PROPERTY-TEST]``
* **A7** sign → load round-trip: ``emit_signed_gate`` then ``load_distilled_gate``
  yields a gate whose weights match the distilled payload; re-emitting the same
  payload yields a byte-identical file. ``[INTEGRATION-TEST]``
* **A8** real XGBoost oracle distillation (``importorskip``): a tiny
  ``XGBoostMetaLearner`` trained on synthetic 21-feature spans (seed=42) plugs the
  ``SurvivalOracle`` Protocol — distil, sign, load, re-weight. Skips cleanly if
  xgboost absent. ``[INTEGRATION-TEST]``
* **A10 (trainer slice)** no new HARD dependency: ``gate_distillation.py`` does
  not import xgboost at module top-level (the Protocol decouples it); determinism
  — no random / uuid / time / secrets in the distill path. ``[AUDIT]``

Traces: FR-018 (learned routing — the offline producer) · NFR-005 / AX-002
(determinism) · AX-006 (the artifact is signed) · AX-001 (synthetic). All key
material is the S2-05 NON-PRODUCTION test key (real custody Pass-2).
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from pii_anon.moe_gate_signing import KeyRing, canonical_gate_bytes
from pii_anon.routing.distilled_gate import load_distilled_gate
from pii_anon.routing.gate_distillation import (
    SurvivalOracle,
    distill_topk_gate,
    emit_signed_gate,
)
from pii_anon.swarm_learner import compute_sample_weights_from_records

TEST_KEY = b"S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD"

_TRAINER_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "pii_anon" / "routing" / "gate_distillation.py"
)


def _ring() -> KeyRing:
    return KeyRing(current_key_id="k1", current_key=TEST_KEY)


# --------------------------------------------------------------------------- #
# Synthetic survival oracle + records (AX-001 — no real PII, fully determined).
# --------------------------------------------------------------------------- #


@dataclass
class _DistillRecord:
    """A minimal training record for distillation.

    Carries the routing cell ``(entity_type, expert_id)``, the oracle feature
    vector, and the Tier-3 RRS annotations consumed by
    ``compute_sample_weights_from_records`` (which reads
    ``re_identification_resistance_score`` + ``is_paired_profile``)."""

    entity_type: str
    expert_id: str
    features: list[float]
    re_identification_resistance_score: float | None = None
    is_paired_profile: bool = False


class _LinearSurvivalOracle:
    """A deterministic synthetic ``SurvivalOracle``.

    survival = clamp(first feature) so the per-record teacher target is known and
    an independent recomputation can pin the closed-form student exactly. Stable
    ``model_bytes`` make ``oracle_hash`` reproducible."""

    def predict(self, features: list[list[float]]) -> list[float]:
        return [max(0.0, min(1.0, row[0])) for row in features]

    def model_bytes(self) -> bytes:
        return b"linear-survival-oracle-v1"


def _records() -> list[_DistillRecord]:
    """2 entity types × 3 experts, with some RRS=0.0 hard cases.

    The first feature IS the oracle's survival target, so each cell's expected
    student value is directly computable from these numbers + the RRS weights."""
    return [
        # PERSON_NAME / gliner — two records, one a hard RRS=0 case.
        _DistillRecord("PERSON_NAME", "gliner", [0.9], re_identification_resistance_score=1.0),
        _DistillRecord("PERSON_NAME", "gliner", [0.5], re_identification_resistance_score=0.0),
        # PERSON_NAME / presidio — single moderate record.
        _DistillRecord("PERSON_NAME", "presidio", [0.6], re_identification_resistance_score=0.5),
        # PERSON_NAME / regex — weak survival.
        _DistillRecord("PERSON_NAME", "regex", [0.2], re_identification_resistance_score=1.0),
        # US_SSN / regex — strong survival (structured PII).
        _DistillRecord("US_SSN", "regex", [0.95], re_identification_resistance_score=1.0),
        _DistillRecord("US_SSN", "regex", [0.85], re_identification_resistance_score=0.0),
        # US_SSN / gliner — weak survival.
        _DistillRecord("US_SSN", "gliner", [0.3], re_identification_resistance_score=1.0),
        # US_SSN / presidio — moderate.
        _DistillRecord("US_SSN", "presidio", [0.4], re_identification_resistance_score=0.5),
    ]


def _expected_cell_means(
    records: list[_DistillRecord], oracle: _LinearSurvivalOracle
) -> dict[str, dict[str, float]]:
    """Independent recomputation of the RRS-up-weighted per-cell oracle-survival
    mean — the closed-form BCE/KL-optimal compact student. Deliberately does NOT
    call the production trainer so A5 has real teeth."""
    cells: dict[tuple[str, str], list[_DistillRecord]] = {}
    for rec in records:
        cells.setdefault((rec.entity_type, rec.expert_id), []).append(rec)
    means: dict[str, dict[str, float]] = {}
    for (et, expert), recs in cells.items():
        survivals = oracle.predict([r.features for r in recs])
        weights = compute_sample_weights_from_records(recs)
        wsum = sum(weights)
        wmean = sum(w * s for w, s in zip(weights, survivals)) / wsum if wsum > 0 else 0.0
        means.setdefault(et, {})[expert] = wmean
    return means


def _expected_softmax(
    means: dict[str, dict[str, float]], temperature: float
) -> dict[str, dict[str, float]]:
    """Temperature-softmax across experts within each entity type (independent)."""
    out: dict[str, dict[str, float]] = {}
    for et, cell in means.items():
        experts = sorted(cell)
        logits = [cell[e] / temperature for e in experts]
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        total = sum(exps)
        out[et] = {e: exps[i] / total for i, e in enumerate(experts)}
    return out


# --------------------------------------------------------------------------- #
# A5 — closed-form distillation is correct + RRS-weighted  [UNIT] [PROPERTY]
# --------------------------------------------------------------------------- #


def test_a5_distilled_weights_match_independent_closed_form() -> None:
    """A5: each weights[et][expert] equals the temperature-softmax of the
    RRS-up-weighted oracle-survival mean per cell (abs 1e-9)."""
    oracle = _LinearSurvivalOracle()
    records = _records()
    temperature = 0.7
    payload = distill_topk_gate(oracle, records, temperature=temperature, seed=42)

    expected = _expected_softmax(_expected_cell_means(records, oracle), temperature)
    got = payload["weights"]
    assert set(got) == set(expected)
    for et, cell in expected.items():
        assert set(got[et]) == set(cell), f"expert set mismatch for {et}"
        for expert, w in cell.items():
            assert math.isclose(got[et][expert], w, abs_tol=1e-9), (
                f"weight mismatch {et}/{expert}: {got[et][expert]} != {w}"
            )
        assert math.isclose(sum(got[et].values()), 1.0, abs_tol=1e-9)


def test_a5_raising_cell_survival_raises_routing_weight() -> None:
    """A5 (monotone teeth): raising one cell's oracle survival raises that expert's
    routing weight within its entity type."""
    oracle = _LinearSurvivalOracle()
    base_records = _records()
    base_payload = distill_topk_gate(oracle, base_records, temperature=1.0, seed=42)
    base_w = base_payload["weights"]["US_SSN"]["gliner"]

    # Raise US_SSN/gliner survival (first feature) from 0.3 → 0.99.
    bumped = _records()
    for rec in bumped:
        if rec.entity_type == "US_SSN" and rec.expert_id == "gliner":
            rec.features = [0.99]
    bumped_payload = distill_topk_gate(oracle, bumped, temperature=1.0, seed=42)
    bumped_w = bumped_payload["weights"]["US_SSN"]["gliner"]

    assert bumped_w > base_w, "raising a cell's survival must raise its routing weight"


def test_a5_rrs_upweighting_is_non_vacuous() -> None:
    """A5: the RRS up-weighting demonstrably shifts a weight vs uniform weighting
    (so the Tier-3 hardening is not a no-op)."""
    oracle = _LinearSurvivalOracle()

    # A cell whose two records disagree on survival AND on RRS, so RRS-weighting
    # vs uniform produces a different mean.
    rrs_records = [
        _DistillRecord("PERSON_NAME", "gliner", [0.9], re_identification_resistance_score=1.0),
        _DistillRecord("PERSON_NAME", "gliner", [0.1], re_identification_resistance_score=0.0),
        _DistillRecord("PERSON_NAME", "presidio", [0.5], re_identification_resistance_score=1.0),
    ]
    uniform_records = [
        _DistillRecord("PERSON_NAME", "gliner", [0.9], re_identification_resistance_score=1.0),
        _DistillRecord("PERSON_NAME", "gliner", [0.1], re_identification_resistance_score=1.0),
        _DistillRecord("PERSON_NAME", "presidio", [0.5], re_identification_resistance_score=1.0),
    ]
    rrs_payload = distill_topk_gate(oracle, rrs_records, temperature=1.0, seed=42)
    uniform_payload = distill_topk_gate(oracle, uniform_records, temperature=1.0, seed=42)
    assert rrs_payload["weights"]["PERSON_NAME"]["gliner"] != pytest.approx(
        uniform_payload["weights"]["PERSON_NAME"]["gliner"]
    ), "RRS up-weighting did not shift the weight — the Tier-3 weighting is vacuous"


def test_a5_payload_stamps_required_metadata() -> None:
    """A5: the payload stamps oracle_hash, gate_feature_version, schema_version,
    temperature (the DC-02 artifact contract)."""
    oracle = _LinearSurvivalOracle()
    payload = distill_topk_gate(oracle, _records(), temperature=1.0, seed=42, feature_version=3)
    assert isinstance(payload["oracle_hash"], str) and len(payload["oracle_hash"]) == 64
    assert payload["gate_feature_version"] == 3
    assert payload["schema_version"] == 1
    assert payload["temperature"] == 1.0


# --------------------------------------------------------------------------- #
# A6 — determinism / replay-equality (NFR-005 / AX-002)  [PROPERTY-TEST]
# --------------------------------------------------------------------------- #


def test_a6_distill_is_byte_identical_across_replays() -> None:
    """A6: same (oracle, records, temperature, seed) ⇒ byte-identical
    canonical_gate_bytes; oracle_hash + gate_feature_version stable."""
    oracle = _LinearSurvivalOracle()
    p1 = distill_topk_gate(oracle, _records(), temperature=0.9, seed=42)
    p2 = distill_topk_gate(oracle, _records(), temperature=0.9, seed=42)
    assert canonical_gate_bytes(p1) == canonical_gate_bytes(p2)
    assert p1["oracle_hash"] == p2["oracle_hash"]
    assert p1["gate_feature_version"] == p2["gate_feature_version"]


def test_a6_oracle_hash_changes_with_oracle_bytes() -> None:
    """A6: a different oracle (different model_bytes) ⇒ a different oracle_hash (the
    hash genuinely binds the teacher)."""
    class _OtherOracle:
        def predict(self, features: list[list[float]]) -> list[float]:
            return [max(0.0, min(1.0, row[0])) for row in features]

        def model_bytes(self) -> bytes:
            return b"a-different-oracle-v2"

    p1 = distill_topk_gate(_LinearSurvivalOracle(), _records(), temperature=1.0, seed=42)
    p2 = distill_topk_gate(_OtherOracle(), _records(), temperature=1.0, seed=42)
    assert p1["oracle_hash"] != p2["oracle_hash"]


def test_a6_synthetic_oracle_satisfies_protocol() -> None:
    """A6: the synthetic oracle structurally satisfies the SurvivalOracle Protocol
    (runtime-checkable duck-typing on ``predict``)."""
    assert isinstance(_LinearSurvivalOracle(), SurvivalOracle)
    assert not isinstance(object(), SurvivalOracle)


# --------------------------------------------------------------------------- #
# A7 — sign → load round-trip  [INTEGRATION-TEST]
# --------------------------------------------------------------------------- #


def test_a7_emit_then_load_round_trip(tmp_path: Path) -> None:
    """A7: emit_signed_gate then load_distilled_gate yields a gate whose advisory
    weights equal the distilled payload's (verified on load)."""
    oracle = _LinearSurvivalOracle()
    payload = distill_topk_gate(oracle, _records(), temperature=0.8, seed=42)
    gate_file = tmp_path / "gate_v1.json"
    emit_signed_gate(payload, key_ring=_ring(), path=gate_file)

    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None
    # The loaded gate's advisory factor for a known cell equals the payload weight.
    base = [("gliner", 0.5), ("presidio", 0.5)]
    out = dict(gate.adjust("PERSON_NAME", None, list(base)))
    pn = payload["weights"]["PERSON_NAME"]
    assert math.isclose(out["gliner"], 0.5 * pn["gliner"], abs_tol=1e-9)
    assert math.isclose(out["presidio"], 0.5 * pn["presidio"], abs_tol=1e-9)


def test_a7_emit_writes_signed_envelope(tmp_path: Path) -> None:
    """A7: the emitted file is the {payload, signature} envelope (signed via
    KeyRing.sign_payload — the S2-05 contract)."""
    oracle = _LinearSurvivalOracle()
    payload = distill_topk_gate(oracle, _records(), temperature=1.0, seed=42)
    gate_file = tmp_path / "gate_v1.json"
    emit_signed_gate(payload, key_ring=_ring(), path=gate_file)
    envelope = json.loads(gate_file.read_text(encoding="utf-8"))
    assert set(envelope) == {"payload", "signature"}
    assert envelope["signature"]["scheme"] == "hmac-sha256"
    assert envelope["signature"]["key_id"] == "k1"


def test_a7_re_emit_is_byte_identical(tmp_path: Path) -> None:
    """A7: re-emitting the same payload yields a byte-identical file (deterministic
    signing over canonical bytes)."""
    oracle = _LinearSurvivalOracle()
    payload = distill_topk_gate(oracle, _records(), temperature=1.0, seed=42)
    f1 = tmp_path / "gate1.json"
    f2 = tmp_path / "gate2.json"
    emit_signed_gate(payload, key_ring=_ring(), path=f1)
    emit_signed_gate(payload, key_ring=_ring(), path=f2)
    assert f1.read_bytes() == f2.read_bytes()


# --------------------------------------------------------------------------- #
# A8 — real XGBoost oracle distillation  [INTEGRATION-TEST] (importorskip)
# --------------------------------------------------------------------------- #


def test_a8_real_xgboost_oracle_distillation(tmp_path: Path) -> None:
    """A8: a tiny XGBoostMetaLearner (seed=42) plugs the SurvivalOracle Protocol —
    distil, sign, load, re-weight. Skips cleanly if xgboost is absent."""
    pytest.importorskip("xgboost")
    from pii_anon.swarm_learner import FEATURE_VERSION, XGBoostMetaLearner

    n_features = 21  # FEATURE_VERSION=3 inference-time feature shape
    # Deterministic synthetic 21-feature spans: label = 1 iff feature[0] high.
    features: list[list[float]] = []
    labels: list[int] = []
    for i in range(60):
        high = i % 2 == 0
        base_val = 0.9 if high else 0.1
        row = [base_val] + [(i % 7) / 7.0 for _ in range(n_features - 1)]
        features.append(row)
        labels.append(1 if high else 0)

    learner = XGBoostMetaLearner()
    learner.train(features, labels, n_rounds=12, early_stopping=0)
    assert learner.is_trained
    assert isinstance(learner, SurvivalOracle)  # structural Protocol satisfaction

    # Distill the real oracle over a few cells (features are the trained shape).
    records = [
        _DistillRecord("PERSON_NAME", "gliner", [0.9] + [0.5] * (n_features - 1),
                       re_identification_resistance_score=1.0),
        _DistillRecord("PERSON_NAME", "presidio", [0.2] + [0.5] * (n_features - 1),
                       re_identification_resistance_score=0.0),
        _DistillRecord("US_SSN", "regex", [0.95] + [0.5] * (n_features - 1),
                       re_identification_resistance_score=1.0),
    ]
    payload = distill_topk_gate(learner, records, temperature=1.0, seed=42,
                                feature_version=FEATURE_VERSION)
    gate_file = tmp_path / "gate_v1.json"
    emit_signed_gate(payload, key_ring=_ring(), path=gate_file)

    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None
    base = [("gliner", 0.5), ("presidio", 0.5)]
    out = dict(gate.adjust("PERSON_NAME", None, list(base)))
    # The real oracle gave gliner (high-survival feature) more mass than presidio
    # (low-survival feature), so its advisory factor is the larger of the two.
    assert out["gliner"] > out["presidio"]


# --------------------------------------------------------------------------- #
# A10 (trainer slice) — no new HARD dependency + determinism  [AUDIT]
# --------------------------------------------------------------------------- #


def test_a10_trainer_module_does_not_import_xgboost_at_top_level() -> None:
    """A10: gate_distillation.py does NOT import xgboost at module top-level — the
    SurvivalOracle Protocol decouples the trainer (xgboost stays an optional
    extra, lazy / importorskip-only)."""
    src = _TRAINER_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level_imports: set[str] = set()
    for node in tree.body:  # module body ONLY — not nested function bodies
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "xgboost" not in top_level_imports, (
        f"trainer must not import xgboost at top level: {sorted(top_level_imports)}"
    )


def test_a10_trainer_path_has_no_nondeterminism_sources() -> None:
    """A10: the distill path imports none of random / uuid / time / secrets
    (determinism — grouped means + softmax are math-only, NFR-005/AX-002)."""
    src = _TRAINER_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("random", "uuid", "time", "secrets"):
        assert banned not in imported, f"distill path must not import {banned!r}"


# --------------------------------------------------------------------------- #
# Additive edge coverage — oracle_hash fallbacks + temperature guard.
# --------------------------------------------------------------------------- #


def test_oracle_hash_falls_back_to_repr_without_model_bytes() -> None:
    """An oracle WITHOUT model_bytes() still gets a stable oracle_hash (repr-based),
    deterministic across replays."""
    class _NoBytesOracle:
        def predict(self, features: list[list[float]]) -> list[float]:
            return [0.5 for _ in features]

        def __repr__(self) -> str:
            return "_NoBytesOracle(fixed)"

    p1 = distill_topk_gate(_NoBytesOracle(), _records(), temperature=1.0, seed=42)
    p2 = distill_topk_gate(_NoBytesOracle(), _records(), temperature=1.0, seed=42)
    assert isinstance(p1["oracle_hash"], str) and len(p1["oracle_hash"]) == 64
    assert p1["oracle_hash"] == p2["oracle_hash"]


def test_oracle_hash_handles_non_bytes_model_bytes() -> None:
    """An oracle whose model_bytes() returns a non-bytes object is repr-coerced (the
    hash never raises on an odd teacher serialization)."""
    class _StrBytesOracle:
        def predict(self, features: list[list[float]]) -> list[float]:
            return [0.5 for _ in features]

        def model_bytes(self) -> str:  # intentionally not bytes
            return "not-actually-bytes"

    payload = distill_topk_gate(_StrBytesOracle(), _records(), temperature=1.0, seed=42)
    assert isinstance(payload["oracle_hash"], str) and len(payload["oracle_hash"]) == 64


def test_distill_rejects_non_positive_temperature() -> None:
    """A non-positive temperature ⇒ ValueError (a degenerate softmax temperature is
    a caller error, not a silent clamp)."""
    oracle = _LinearSurvivalOracle()
    with pytest.raises(ValueError, match="temperature"):
        distill_topk_gate(oracle, _records(), temperature=0.0, seed=42)
    with pytest.raises(ValueError, match="temperature"):
        distill_topk_gate(oracle, _records(), temperature=-1.0, seed=42)


def test_distill_records_advisory_emission_threshold() -> None:
    """The payload records an advisory emission_threshold (via select_f2_threshold)
    plus its f_beta — metadata only, never a hard cut in the gate."""
    oracle = _LinearSurvivalOracle()
    payload = distill_topk_gate(oracle, _records(), temperature=1.0, seed=42)
    assert "emission_threshold" in payload and isinstance(payload["emission_threshold"], float)
    assert "emission_f_beta" in payload and isinstance(payload["emission_f_beta"], float)


def test_refactor_producer_stamps_exactly_the_consumer_required_keys() -> None:
    """REFACTOR (drift guard): the producer stamps exactly the runtime validator's
    required key set (shared KEY_* constants), so a distilled payload always parses.
    Round-trips every distilled payload through DistilledGatePayload.from_payload."""
    from pii_anon.routing.distilled_gate import _REQUIRED_KEYS, DistilledGatePayload

    payload = distill_topk_gate(_LinearSurvivalOracle(), _records(), temperature=1.0, seed=42)
    # Every required runtime key is present in what the producer stamped.
    assert set(_REQUIRED_KEYS).issubset(payload), (
        f"producer dropped a required key: {set(_REQUIRED_KEYS) - set(payload)}"
    )
    # And the produced payload validates cleanly (no GatePayloadError).
    view = DistilledGatePayload.from_payload(payload)
    assert view.schema_version == 1
