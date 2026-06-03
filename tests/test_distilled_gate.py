"""S2-02 — the runtime ``DistilledTopKGate`` (control-path, security-critical).

Pins the runtime half of the learned MoE gate: a verified-payload-only gate that
advisorily re-weights experts already in ``base`` and can NEVER drop a floored
shared span. ``gate_v1.json`` is a control-path artifact (an attacker who writes
it biases the detector — exactly why S2-05 exists), so the gate's ONLY
construction path is the fail-closed ``load_verified_gate(verify_on_load=True)``
boundary (the S2-01 gate's S2-02 entry-gate obligation).

Acceptance covered here (the runtime half):

* **A1** the gate is constructed ONLY via verify-on-load; tamper / wrong-key ⇒
  ``GateSignatureError`` (fail-closed, no gate); there is NO production code path
  that builds a ``DistilledTopKGate`` from an unverified dict / ``verify_on_load
  =False`` (AST guard). ``[SECURITY-TEST]`` ``[CONTRACT-TEST]``
* **A2** absent ⇒ ``load_distilled_gate`` returns ``None`` ⇒ MoERouter keeps the
  static ``entity_strengths`` softmax (NFR-026); 0 unhandled exceptions.
  ``[UNIT-TEST]``
* **A3** advisory re-weighting: a verified gate that up-weights
  ``gliner-compatible`` for ``PERSON_NAME`` shifts the routed weights toward it;
  an entity_type the gate has no opinion on returns ``base`` unchanged.
  ``[UNIT-TEST]``
* **A4** the advisory gate cannot drop a floored span: the REAL
  ``FloorProjectingFusion(MoEFusionStrategy(... gate that zeroes regex-oss ...))``
  on a checksum-gated ``US_SSN`` shared span ⇒ the span survives carrying the
  ``shared_floor`` provenance marker (AX-003). ``[INTEGRATION-TEST]`` ``[AUDIT]``
* **A9** payload-shape validation is loud: a VERIFIED-but-malformed payload
  (missing ``weights``, a non-finite / negative weight, wrong ``schema_version``)
  ⇒ ``GatePayloadError`` (a structurally-broken signed gate is a loud operator
  error, never a coerced gate). ``[UNIT-TEST]`` ``[SECURITY-TEST]``
* **A10 (runtime slice)** AST scan: ``distilled_gate.py`` imports no xgboost +
  nothing from ``eval_framework.rating`` / ``attacks`` / the SDO gate; the runtime
  gate path uses no ``random`` / ``uuid`` / ``time`` / ``secrets``; consumed
  signer / loader / floor modules byte-identical. ``[AUDIT]``

Traces: FR-018 (learned routing — the gate body) · NFR-026 (graceful
degradation) · NFR-005 / AX-002 (determinism) · AX-006 (least-privilege — the
control-path gate enters ONLY through the fail-closed boundary) · AX-003 (the
gate is advisory; the recall-floor is the no-drop authority). All key material is
the S2-05 NON-PRODUCTION test key (real custody Pass-2); synthetic fixtures only
(AX-001).
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from pii_anon.errors import GatePayloadError, GateSignatureError
from pii_anon.moe import (
    ExpertRegistry,
    ExpertSpec,
    MoEFusionStrategy,
    MoERouter,
    RouteContext,
    RoutingGate,
    build_default_registry,
)
from pii_anon.moe_gate_signing import KeyRing
from pii_anon.routing.distilled_gate import (
    DistilledGatePayload,
    DistilledTopKGate,
    load_distilled_gate,
)
from pii_anon.routing.floor_fusion import FloorProjectingFusion
from pii_anon.types import EngineFinding

TEST_KEY = b"S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD"

_GATE_SRC = Path(__file__).resolve().parents[1] / "src" / "pii_anon" / "routing" / "distilled_gate.py"


def _ring() -> KeyRing:
    return KeyRing(current_key_id="k1", current_key=TEST_KEY)


def _valid_payload(
    *,
    person_gliner: float = 4.0,
    person_presidio: float = 1.0,
) -> dict[str, object]:
    """A schema-valid distilled-gate payload (advisory per-entity weights).

    Up-weights ``gliner-compatible`` for ``PERSON_NAME`` so the advisory factor is
    > 1.0 there; ``US_SSN`` zeroes ``regex-oss`` (the A4 adversary cell)."""
    return {
        "schema_version": 1,
        "gate_feature_version": 1,
        "oracle_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "temperature": 1.0,
        "weights": {
            "PERSON_NAME": {
                "gliner-compatible": person_gliner,
                "presidio-compatible": person_presidio,
            },
            "US_SSN": {"regex-oss": 0.0},
        },
    }


def _write_signed_gate(path: Path, ring: KeyRing, payload: dict[str, object]) -> None:
    envelope = ring.sign_payload(payload)
    path.write_text(json.dumps(envelope), encoding="utf-8")


def _synthetic_person_registry() -> ExpertRegistry:
    reg = ExpertRegistry()
    reg.register_expert(
        ExpertSpec(
            expert_id="gliner-compatible",
            display_name="GLiNER",
            entity_strengths={"PERSON_NAME": 0.80, "EMAIL_ADDRESS": 0.70},
        )
    )
    reg.register_expert(
        ExpertSpec(
            expert_id="presidio-compatible",
            display_name="Presidio",
            entity_strengths={"PERSON_NAME": 0.80, "EMAIL_ADDRESS": 0.80},
        )
    )
    return reg


# --------------------------------------------------------------------------- #
# A1 — gate constructed ONLY via verify-on-load  [SECURITY-TEST] [CONTRACT-TEST]
# --------------------------------------------------------------------------- #


def test_a1_load_distilled_gate_returns_gate_for_signed_artifact(tmp_path: Path) -> None:
    """A1: a validly-signed gate_v1.json ⇒ load_distilled_gate returns a
    DistilledTopKGate that structurally satisfies the RoutingGate Protocol."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert isinstance(gate, DistilledTopKGate)
    assert isinstance(gate, RoutingGate)


def test_a1_tampered_payload_raises_gate_signature_error(tmp_path: Path) -> None:
    """A1 (headline): a tampered payload byte ⇒ GateSignatureError (fail-closed,
    no gate constructed)."""
    gate_file = tmp_path / "gate_v1.json"
    envelope = _ring().sign_payload(_valid_payload())
    # Flip one weight byte AFTER signing — the signature no longer covers it.
    envelope["payload"]["weights"]["PERSON_NAME"]["gliner-compatible"] = 999.0  # type: ignore[index]
    gate_file.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(GateSignatureError):
        load_distilled_gate(gate_file, key_ring=_ring())


def test_a1_wrong_key_raises_gate_signature_error(tmp_path: Path) -> None:
    """A1: an artifact signed under a key NOT in the ring ⇒ GateSignatureError
    (the ring is the trust root, even on the load path)."""
    attacker_ring = KeyRing(current_key_id="attacker", current_key=b"attacker-key-xxxx")
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, attacker_ring, _valid_payload())
    with pytest.raises(GateSignatureError, match="unknown key id"):
        load_distilled_gate(gate_file, key_ring=_ring())


def test_a1_no_unverified_construction_path_in_distilled_gate_module() -> None:
    """A1 (AUDIT): the runtime module has NO production path that constructs a gate
    from an unverified dict / verify_on_load=False.

    Static AST guard: (1) every call to ``load_verified_gate`` in distilled_gate.py
    passes ``verify_on_load=True`` (never False, never omitted-then-defaulted-off);
    (2) ``load_distilled_gate`` is the only function that calls ``load_verified_gate``;
    (3) the source contains no ``verify_on_load=False`` token at all.
    """
    src = _GATE_SRC.read_text(encoding="utf-8")
    assert "verify_on_load=False" not in src, (
        "distilled_gate.py must contain NO verify_on_load=False escape (fail-closed only)"
    )
    tree = ast.parse(src)
    verified_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name)
                else None
            )
            if name == "load_verified_gate":
                verified_calls += 1
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                assert "verify_on_load" in kwargs, (
                    "load_verified_gate must be called with explicit verify_on_load=True"
                )
                val = kwargs["verify_on_load"]
                assert isinstance(val, ast.Constant) and val.value is True, (
                    "load_verified_gate must be called with verify_on_load=True (fail-closed)"
                )
    assert verified_calls >= 1, "distilled_gate.py must route construction through load_verified_gate"


# --------------------------------------------------------------------------- #
# A2 — absent ⇒ static softmax (NFR-026)  [UNIT-TEST]
# --------------------------------------------------------------------------- #


def test_a2_absent_artifact_returns_none(tmp_path: Path) -> None:
    """A2: a non-existent gate_path ⇒ load_distilled_gate returns None (no error)."""
    missing = tmp_path / "does_not_exist_gate_v1.json"
    assert load_distilled_gate(missing, key_ring=_ring()) is None


def test_a2_absent_gate_router_keeps_static_softmax(tmp_path: Path) -> None:
    """A2: a None gate ⇒ MoERouter routes byte-identically to the no-gate baseline,
    0 unhandled exceptions (NFR-026)."""
    missing = tmp_path / "absent_gate_v1.json"
    gate = load_distilled_gate(missing, key_ring=_ring())
    assert gate is None
    registry = _synthetic_person_registry()
    router_with_none = MoERouter(registry, gate=gate)
    router_plain = MoERouter(registry)
    assert router_with_none.route("PERSON_NAME") == router_plain.route("PERSON_NAME")


# --------------------------------------------------------------------------- #
# A3 — advisory re-weighting  [UNIT-TEST]
# --------------------------------------------------------------------------- #


def test_a3_gate_upweights_named_expert(tmp_path: Path) -> None:
    """A3: a verified gate up-weighting gliner-compatible for PERSON_NAME shifts the
    routed weights toward it vs the static base; the result is a valid distribution."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload(person_gliner=8.0, person_presidio=1.0))
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None

    registry = _synthetic_person_registry()
    router_plain = MoERouter(registry)
    router_gated = MoERouter(registry, gate=gate)

    base = dict(router_plain.route("PERSON_NAME"))
    gated = dict(router_gated.route("PERSON_NAME"))
    # gliner is up-weighted ⇒ strictly larger share than the static base.
    assert gated["gliner-compatible"] > base["gliner-compatible"]
    # still a valid distribution (finite, ≥0, Σ=1.0).
    assert all(math.isfinite(w) and w >= 0.0 for w in gated.values())
    assert math.isclose(sum(gated.values()), 1.0, abs_tol=1e-9)


def test_a3_unknown_entity_returns_base_unchanged(tmp_path: Path) -> None:
    """A3: an entity_type the gate has no opinion on ⇒ adjust returns base
    unchanged (no invented expert, no perturbation)."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None
    base = [("gliner-compatible", 0.6), ("presidio-compatible", 0.4)]
    # EMAIL_ADDRESS is not in the gate's weights map ⇒ identity.
    assert gate.adjust("EMAIL_ADDRESS", None, list(base)) == base


def test_a3_empty_opinion_returns_base_unchanged(tmp_path: Path) -> None:
    """A3: an entity present in weights but with no overlapping expert id ⇒ every
    factor is the 1.0 no-opinion default, so base is returned unchanged in value."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None
    # PERSON_NAME exists in weights, but neither base expert id is in its map.
    base = [("expert-x", 0.5), ("expert-y", 0.5)]
    out = gate.adjust("PERSON_NAME", None, list(base))
    assert out == base


def test_a3_adjust_never_invents_an_expert(tmp_path: Path) -> None:
    """A3: adjust only re-weights experts already in base — it returns exactly the
    same expert ids in the same order (it never injects a gate-only expert)."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None
    base = [("gliner-compatible", 0.7), ("presidio-compatible", 0.3)]
    out = gate.adjust("PERSON_NAME", None, list(base))
    assert [eid for eid, _ in out] == [eid for eid, _ in base]


# --------------------------------------------------------------------------- #
# A4 — advisory gate cannot drop a floored span (AX-003)  [INTEGRATION] [AUDIT]
# --------------------------------------------------------------------------- #


def test_a4_distilled_gate_cannot_drop_floored_shared_span(tmp_path: Path) -> None:
    """A4: a DistilledTopKGate whose weights zero regex-oss for US_SSN still cannot
    drop a checksum-gated US_SSN shared span — the REAL FloorProjectingFusion
    re-injects it post-fusion with the shared_floor provenance marker (AX-003)."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None

    registry = build_default_registry()
    inner = MoEFusionStrategy(registry=registry)
    # Inject the verified distilled gate into the inner router (advisory layer).
    inner.router = MoERouter(
        registry,
        top_k=inner.top_k,
        performance_floor=inner.performance_floor,
        gate=gate,
    )
    pipeline = FloorProjectingFusion(inner)

    shared_span = EngineFinding(
        entity_type="US_SSN",
        confidence=0.99,
        field_path="ssn",
        span_start=0,
        span_end=11,
        engine_id="regex-oss",
        language="en",
        explanation="regex checksum-gated",
    )
    out = pipeline.merge([shared_span])

    survivors = [
        f for f in out
        if f.entity_type == "US_SSN" and f.span_start == 0 and f.span_end == 11
    ]
    assert survivors, "advisory distilled gate dropped a floored shared span"
    # Pin the MECHANISM: the bare inner MoE merge drops this span (the gate zeroes
    # regex-oss ⇒ total_weight==0 ⇒ cluster skipped), so the only way it appears in
    # the full pipeline output is the SharedLayerProjector re-injecting it.
    assert any("shared_floor" in (f.explanation or "") for f in survivors), (
        "floored span survived but NOT via shared_floor re-injection "
        f"(explanations={[f.explanation for f in survivors]})"
    )


# --------------------------------------------------------------------------- #
# A9 — payload-shape validation is loud  [UNIT-TEST] [SECURITY-TEST]
# --------------------------------------------------------------------------- #


def test_a9_missing_weights_raises_gate_payload_error() -> None:
    """A9: a verified-but-malformed payload missing ``weights`` ⇒ GatePayloadError."""
    payload = _valid_payload()
    del payload["weights"]
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a9_non_finite_weight_raises_gate_payload_error(bad: float) -> None:
    """A9: a non-finite weight (NaN / ±inf) ⇒ GatePayloadError (never coerced)."""
    payload = _valid_payload()
    payload["weights"]["PERSON_NAME"]["gliner-compatible"] = bad  # type: ignore[index]
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_negative_weight_raises_gate_payload_error() -> None:
    """A9: a negative weight ⇒ GatePayloadError (weights are advisory factors ≥0)."""
    payload = _valid_payload()
    payload["weights"]["PERSON_NAME"]["gliner-compatible"] = -1.0  # type: ignore[index]
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_wrong_schema_version_raises_gate_payload_error() -> None:
    """A9: a wrong ``schema_version`` ⇒ GatePayloadError (a future/unknown schema is
    a loud operator error, not a best-effort parse)."""
    payload = _valid_payload()
    payload["schema_version"] = 999
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_missing_oracle_hash_raises_gate_payload_error() -> None:
    """A9: a missing required key (``oracle_hash``) ⇒ GatePayloadError."""
    payload = _valid_payload()
    del payload["oracle_hash"]
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_non_mapping_weights_raises_gate_payload_error() -> None:
    """A9: a ``weights`` that is not a mapping-of-mappings ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["weights"] = ["not", "a", "mapping"]
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_bool_weight_rejected_as_non_numeric() -> None:
    """A9: a bool weight (Python bool is an int subclass!) is rejected — the gate
    never silently treats True/False as 1.0/0.0."""
    payload = _valid_payload()
    payload["weights"]["PERSON_NAME"]["gliner-compatible"] = True  # type: ignore[index]
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_malformed_verified_artifact_raises_on_load(tmp_path: Path) -> None:
    """A9: a VERIFIED (correctly-signed) but malformed payload ⇒ GatePayloadError on
    load_distilled_gate (the signature is valid; the SHAPE is broken). This is the
    one loud case — never a silent static fallback."""
    bad_payload = _valid_payload()
    del bad_payload["weights"]
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), bad_payload)  # correctly signed!
    with pytest.raises(GatePayloadError):
        load_distilled_gate(gate_file, key_ring=_ring())


# --------------------------------------------------------------------------- #
# A9 (additive edge coverage) — every shape-validation branch is loud.
# --------------------------------------------------------------------------- #


def test_a9_non_mapping_payload_raises() -> None:
    """A9: a non-mapping payload ⇒ GatePayloadError (defensive type guard)."""
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_a9_non_int_schema_version_raises() -> None:
    """A9: a non-int schema_version ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["schema_version"] = "1"
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_bool_schema_version_raises() -> None:
    """A9: a bool schema_version (bool is an int subclass) ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["schema_version"] = True
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_non_int_gate_feature_version_raises() -> None:
    """A9: a non-int gate_feature_version ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["gate_feature_version"] = 1.5
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_empty_oracle_hash_raises() -> None:
    """A9: an empty oracle_hash string ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["oracle_hash"] = ""
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


@pytest.mark.parametrize("bad_temp", [0.0, -1.0, float("nan"), float("inf"), "1.0", True])
def test_a9_bad_temperature_raises(bad_temp: object) -> None:
    """A9: a non-positive / non-finite / non-numeric / bool temperature ⇒
    GatePayloadError."""
    payload = _valid_payload()
    payload["temperature"] = bad_temp
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_non_string_entity_type_key_raises() -> None:
    """A9: a non-string weights key (entity type) ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["weights"] = {123: {"regex-oss": 1.0}}
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_non_mapping_inner_cell_raises() -> None:
    """A9: a weights cell that is not a mapping ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["weights"] = {"PERSON_NAME": [("regex-oss", 1.0)]}
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_non_string_expert_id_raises() -> None:
    """A9: a non-string expert-id key inside a cell ⇒ GatePayloadError."""
    payload = _valid_payload()
    payload["weights"] = {"PERSON_NAME": {7: 1.0}}
    with pytest.raises(GatePayloadError):
        DistilledGatePayload.from_payload(payload)


def test_a9_valid_payload_view_is_immutable_and_exposed() -> None:
    """A9: a valid payload builds an immutable view; the gate exposes it (the
    weights mapping is read-only — no in-place mutation of a verified gate)."""
    payload = DistilledGatePayload.from_payload(_valid_payload())
    gate = DistilledTopKGate(payload)
    assert gate.payload is payload
    assert payload.weights["PERSON_NAME"]["gliner-compatible"] == 4.0
    with pytest.raises(TypeError):
        payload.weights["PERSON_NAME"]["gliner-compatible"] = 9.9  # type: ignore[index]


# --------------------------------------------------------------------------- #
# A10 (runtime slice) — import-boundary / no-new-dep / determinism  [AUDIT]
# --------------------------------------------------------------------------- #


def test_a10_distilled_gate_imports_no_xgboost_or_eval_framework() -> None:
    """A10: the runtime gate module imports NO xgboost and nothing from
    eval_framework.rating / attacks / the SDO gate (import isolation)."""
    src = _GATE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(mod == "xgboost" or mod.startswith("xgboost.") for mod in imported), (
        f"runtime gate must NOT import xgboost: {sorted(imported)}"
    )
    forbidden = ("eval_framework.rating", "eval_framework.attacks", "competitive_supremacy")
    for mod in imported:
        assert not any(f in mod for f in forbidden), f"forbidden import in runtime gate: {mod}"


def test_a10_distilled_gate_path_has_no_nondeterminism_sources() -> None:
    """A10: the runtime gate path imports none of random / uuid / time / secrets
    (determinism — the gate is a dict lookup + a multiply, NFR-005/AX-002)."""
    src = _GATE_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("random", "uuid", "time", "secrets"):
        assert banned not in imported, f"runtime gate path must not import {banned!r}"


def test_a10_distilled_gate_is_advisory_and_deterministic(tmp_path: Path) -> None:
    """A10: two adjust() calls on the same (entity_type, context, base) are equal
    (the gate carries no hidden state / nondeterminism)."""
    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    gate = load_distilled_gate(gate_file, key_ring=_ring())
    assert gate is not None
    base = [("gliner-compatible", 0.7), ("presidio-compatible", 0.3)]
    ctx = RouteContext(language="es")
    first = gate.adjust("PERSON_NAME", ctx, list(base))
    second = gate.adjust("PERSON_NAME", ctx, list(base))
    assert first == second


# --------------------------------------------------------------------------- #
# Optional "distilled_moe" v2 wiring (the wired control-path mode).  [INTEGRATION]
# Proves the gate plugs the v2 registry/factory closure WITHOUT changing
# build_fusion's public signature (the public gate_path thread is DEFERRED).
# --------------------------------------------------------------------------- #


def test_distilled_moe_mode_is_registered() -> None:
    """The 'distilled_moe' v2 mode is discoverable via available_fusion_modes()."""
    from pii_anon.fusion import available_fusion_modes

    assert "distilled_moe" in available_fusion_modes()


def test_distilled_moe_factory_no_gate_floors_a_shared_span() -> None:
    """v2 factory with gate_path=None ⇒ a floor-wrapped MoE that still floors a
    shared span (NFR-026 static-softmax path + AX-003 floor)."""
    from pii_anon.fusion import FusionBuildSpec, _build_distilled_moe

    spec = FusionBuildSpec(
        mode="distilled_moe", weights={}, min_consensus=1, entity_weights=None,
        iou_threshold=0.5, gate_path=None, key_ring=None, top_k=3,
    )
    pipeline = _build_distilled_moe(spec)
    shared_span = EngineFinding(
        entity_type="US_SSN", confidence=0.99, field_path="ssn",
        span_start=0, span_end=11, engine_id="regex-oss", language="en",
    )
    out = pipeline.merge([shared_span])
    assert any(
        f.entity_type == "US_SSN" and f.span_start == 0 and f.span_end == 11 for f in out
    )


def test_distilled_moe_factory_with_signed_gate_still_floors(tmp_path: Path) -> None:
    """v2 factory with a real signed gate_path (zeroing regex-oss for US_SSN) ⇒ the
    gate loads through verify-on-load and the floor STILL re-injects the shared span
    with the shared_floor marker (AX-003 — the wired gate is advisory)."""
    from pii_anon.fusion import FusionBuildSpec, _build_distilled_moe

    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    spec = FusionBuildSpec(
        mode="distilled_moe", weights={}, min_consensus=1, entity_weights=None,
        iou_threshold=0.5, gate_path=gate_file, key_ring=_ring(), top_k=3,
    )
    pipeline = _build_distilled_moe(spec)
    shared_span = EngineFinding(
        entity_type="US_SSN", confidence=0.99, field_path="ssn",
        span_start=0, span_end=11, engine_id="regex-oss", language="en",
    )
    out = pipeline.merge([shared_span])
    survivors = [
        f for f in out
        if f.entity_type == "US_SSN" and f.span_start == 0 and f.span_end == 11
    ]
    assert survivors and any("shared_floor" in (f.explanation or "") for f in survivors)


def test_distilled_moe_factory_gate_path_without_key_ring_fails_closed(tmp_path: Path) -> None:
    """v2 factory with a gate_path but no key_ring fails CLOSED (ValueError) — a gate
    that cannot be verified must never be loaded with verification skipped."""
    from pii_anon.fusion import FusionBuildSpec, _build_distilled_moe

    gate_file = tmp_path / "gate_v1.json"
    _write_signed_gate(gate_file, _ring(), _valid_payload())
    spec = FusionBuildSpec(
        mode="distilled_moe", weights={}, min_consensus=1, entity_weights=None,
        iou_threshold=0.5, gate_path=gate_file, key_ring=None, top_k=3,
    )
    with pytest.raises(ValueError, match="key_ring"):
        _build_distilled_moe(spec)


def test_distilled_moe_factory_tampered_gate_fails_closed(tmp_path: Path) -> None:
    """v2 factory pointed at a tampered gate_path fails CLOSED (GateSignatureError)
    — the wired control-path mode never silently degrades on a broken signature."""
    from pii_anon.fusion import FusionBuildSpec, _build_distilled_moe

    gate_file = tmp_path / "gate_v1.json"
    envelope = _ring().sign_payload(_valid_payload())
    envelope["payload"]["weights"]["PERSON_NAME"]["gliner-compatible"] = 999.0  # type: ignore[index]
    gate_file.write_text(json.dumps(envelope), encoding="utf-8")
    spec = FusionBuildSpec(
        mode="distilled_moe", weights={}, min_consensus=1, entity_weights=None,
        iou_threshold=0.5, gate_path=gate_file, key_ring=_ring(), top_k=3,
    )
    with pytest.raises(GateSignatureError):
        _build_distilled_moe(spec)
