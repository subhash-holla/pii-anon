"""S2-05 — the ``MoERouter`` verify-on-load construction seam (the teeth).

Pins acceptance A7 + A8 of the S2-05 story:

- A7  ``load_verified_gate(path, key_ring=ring)`` fails CLOSED on a tampered
      ``gate_v1.json`` (raises ``GateSignatureError``, returns NO payload); a valid
      file returns the verified payload. The production construction seam calls the
      loader with ``verify_on_load=True`` (the escape hatch is not the default).
- A8  absence is graceful (``load_verified_gate`` -> ``None`` so ``MoERouter``
      keeps its static ``entity_strengths`` softmax, NFR-026); a present-but-broken
      file RAISES (present-and-broken != absent — never silent-fallback-to-unsigned).

Also pins the additive-only contract: the no-gate ``MoERouter`` constructor /
``route`` / ``route_all`` signatures + behaviour are unchanged (no public-API
regression), and the ``verify_on_load=False`` escape hatch is wired but never the
default and never the production path.

All key material is a deterministic NON-PRODUCTION test key; real custody Pass-2.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from pii_anon.errors import GateSignatureError
from pii_anon.moe import (
    ExpertRegistry,
    ExpertSpec,
    MoERouter,
    load_verified_gate,
)
from pii_anon.moe_gate_signing import KeyRing

TEST_KEY = b"S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD"


def _ring() -> KeyRing:
    return KeyRing(current_key_id="k1", current_key=TEST_KEY)


def _gate_payload() -> dict[str, object]:
    return {
        "gate_feature_version": 1,
        "oracle_hash": "0123456789abcdef0123456789abcdef",
        "entity_expert_weights": {
            "PERSON_NAME": {"gliner-compatible": 0.7, "presidio-compatible": 0.3},
        },
    }


def _write_signed_gate(path: Path, ring: KeyRing) -> None:
    envelope = ring.sign_payload(_gate_payload())
    path.write_text(json.dumps(envelope), encoding="utf-8")


def _write_tampered_gate(path: Path, ring: KeyRing) -> None:
    envelope = ring.sign_payload(_gate_payload())
    envelope["payload"]["oracle_hash"] = "ffffffffffffffffffffffffffffffff"  # type: ignore[index]
    path.write_text(json.dumps(envelope), encoding="utf-8")


# ── A7 — verify-on-load seam fails closed ────────────────────────────────────
def test_a7_valid_file_returns_verified_payload(tmp_path: Path) -> None:
    """A7: a validly-signed gate_v1.json on disk loads and returns the verified
    payload (deep-equal)."""
    gate = tmp_path / "gate_v1.json"
    _write_signed_gate(gate, _ring())
    payload = load_verified_gate(gate, key_ring=_ring())
    assert payload == _gate_payload()


def test_a7_tampered_file_raises_and_returns_nothing(tmp_path: Path) -> None:
    """A7 (headline): a tampered gate_v1.json -> GateSignatureError, NO payload
    leaks back to the caller."""
    gate = tmp_path / "gate_v1.json"
    _write_tampered_gate(gate, _ring())
    with pytest.raises(GateSignatureError):
        load_verified_gate(gate, key_ring=_ring())


def test_a7_unknown_key_file_raises(tmp_path: Path) -> None:
    """A7: a file signed under an out-of-ring key -> raise (the ring is the trust
    root even on the load path)."""
    attacker_ring = KeyRing(current_key_id="attacker", current_key=b"attacker-key-xx")
    gate = tmp_path / "gate_v1.json"
    _write_signed_gate(gate, attacker_ring)
    with pytest.raises(GateSignatureError, match="unknown key id"):
        load_verified_gate(gate, key_ring=_ring())


def test_a7_malformed_json_file_raises(tmp_path: Path) -> None:
    """A7: a present-but-malformed (non-JSON / not-an-envelope) gate file -> raise
    (fail-closed, never silent accept)."""
    gate = tmp_path / "gate_v1.json"
    gate.write_text("this is not json {{{", encoding="utf-8")
    with pytest.raises(GateSignatureError):
        load_verified_gate(gate, key_ring=_ring())


def test_a7_default_verify_on_load_is_true() -> None:
    """A7: the loader's verify_on_load parameter DEFAULTS to True (fail-closed by
    default; the escape hatch must be opt-in)."""
    sig = inspect.signature(load_verified_gate)
    assert sig.parameters["verify_on_load"].default is True


def test_a7_production_seam_uses_verify_on_load_true(tmp_path: Path) -> None:
    """A7: the production MoERouter construction seam loads the gate with
    verify_on_load=True — proven by spying on load_verified_gate and asserting the
    keyword is True (the escape hatch is NOT used on the production path)."""
    import pii_anon.moe as moe_mod

    gate = tmp_path / "gate_v1.json"
    _write_signed_gate(gate, _ring())

    captured: dict[str, object] = {}
    real = moe_mod.load_verified_gate

    def _spy(path, *, key_ring, verify_on_load=True):  # type: ignore[no-untyped-def]
        captured["verify_on_load"] = verify_on_load
        captured["path"] = path
        return real(path, key_ring=key_ring, verify_on_load=verify_on_load)

    # patch the symbol the constructor resolves
    original = moe_mod.load_verified_gate
    moe_mod.load_verified_gate = _spy  # type: ignore[assignment]
    try:
        registry = ExpertRegistry()
        registry.register_expert(
            ExpertSpec(expert_id="gliner-compatible", display_name="g",
                       entity_strengths={"PERSON_NAME": 0.9})
        )
        MoERouter(registry, gate_path=gate, key_ring=_ring())
    finally:
        moe_mod.load_verified_gate = original  # type: ignore[assignment]

    assert captured.get("verify_on_load") is True, (
        "production construction seam must call load_verified_gate(verify_on_load=True)"
    )


def test_a7_constructor_raises_on_tampered_gate(tmp_path: Path) -> None:
    """A7: constructing a MoERouter pointed at a tampered gate_v1.json fails
    closed — the tampered gate never reaches route()."""
    gate = tmp_path / "gate_v1.json"
    _write_tampered_gate(gate, _ring())
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(expert_id="gliner-compatible", display_name="g",
                   entity_strengths={"PERSON_NAME": 0.9})
    )
    with pytest.raises(GateSignatureError):
        MoERouter(registry, gate_path=gate, key_ring=_ring())


# ── A8 — absence graceful, present-broken raises ─────────────────────────────
def test_a8_absent_file_returns_none(tmp_path: Path) -> None:
    """A8: no gate_v1.json at the path -> loader returns None (so MoERouter keeps
    the static entity_strengths softmax, NFR-026) with NO error."""
    missing = tmp_path / "does_not_exist_gate_v1.json"
    assert load_verified_gate(missing, key_ring=_ring()) is None


def test_a8_present_but_broken_raises_not_fallback(tmp_path: Path) -> None:
    """A8: present-but-unverifiable -> RAISE (NOT a degrade to static fallback;
    present-and-broken != absent)."""
    gate = tmp_path / "gate_v1.json"
    _write_tampered_gate(gate, _ring())
    with pytest.raises(GateSignatureError):
        load_verified_gate(gate, key_ring=_ring())


def test_a8_absent_gate_router_uses_static_softmax(tmp_path: Path) -> None:
    """A8: a MoERouter constructed with an ABSENT gate path behaves exactly like
    the no-gate router — static entity_strengths softmax, NFR-026, no error."""
    missing = tmp_path / "absent_gate_v1.json"
    registry = _two_expert_registry()
    router_with_absent = MoERouter(registry, gate_path=missing, key_ring=_ring())
    router_plain = MoERouter(registry)
    assert router_with_absent.route("PERSON_NAME") == router_plain.route("PERSON_NAME")


# ── Additive-only / no public-API regression ─────────────────────────────────
def _two_expert_registry() -> ExpertRegistry:
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(expert_id="gliner-compatible", display_name="GLiNER",
                   entity_strengths={"PERSON_NAME": 0.92, "EMAIL_ADDRESS": 0.70})
    )
    registry.register_expert(
        ExpertSpec(expert_id="presidio-compatible", display_name="Presidio",
                   entity_strengths={"PERSON_NAME": 0.82, "EMAIL_ADDRESS": 0.80})
    )
    return registry


def test_no_gate_constructor_signature_unchanged() -> None:
    """Additive contract: the no-gate constructor still accepts exactly
    (registry, top_k=3, *, performance_floor=True); the new gate params are
    keyword-only with safe defaults so existing callers are unaffected."""
    sig = inspect.signature(MoERouter.__init__)
    params = sig.parameters
    assert params["registry"].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    assert params["top_k"].default == 3
    assert params["performance_floor"].default is True
    assert params["performance_floor"].kind is inspect.Parameter.KEYWORD_ONLY
    # new params are keyword-only with defaults (purely additive)
    assert params["gate_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["gate_path"].default is None
    assert params["key_ring"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["key_ring"].default is None


def test_no_gate_router_behaviour_unchanged() -> None:
    """Additive contract: a MoERouter built the old way (no gate kwargs) routes
    identically — weights sum to 1.0, best expert present (performance floor)."""
    registry = _two_expert_registry()
    router = MoERouter(registry)
    routed = router.route("PERSON_NAME")
    assert routed, "expected experts routed for PERSON_NAME"
    assert abs(sum(w for _, w in routed) - 1.0) < 1e-9
    assert "gliner-compatible" in {eid for eid, _ in routed}  # highest strength


def test_route_all_unchanged_with_no_gate() -> None:
    """Additive contract: route_all() over a no-gate router still returns a dict
    keyed by entity type, each value summing to 1.0."""
    registry = _two_expert_registry()
    router = MoERouter(registry)
    all_routes = router.route_all()
    assert set(all_routes) == {"PERSON_NAME", "EMAIL_ADDRESS"}
    for routed in all_routes.values():
        assert abs(sum(w for _, w in routed) - 1.0) < 1e-9


def test_escape_hatch_exists_but_is_not_default(tmp_path: Path) -> None:
    """Additive contract: verify_on_load=False escape hatch exists for explicit
    offline tooling (returns the raw payload without verification) but is NEVER
    the default — pins it is opt-in only."""
    gate = tmp_path / "gate_v1.json"
    _write_tampered_gate(gate, _ring())
    # default path raises ...
    with pytest.raises(GateSignatureError):
        load_verified_gate(gate, key_ring=_ring())
    # ... explicit opt-out returns the (unverified) payload for offline tooling
    raw = load_verified_gate(gate, key_ring=_ring(), verify_on_load=False)
    assert raw is not None and raw["oracle_hash"] == "ffffffffffffffffffffffffffffffff"


def test_load_verified_gate_reexported_from_moe() -> None:
    """The construction seam exposes load_verified_gate from pii_anon.moe (the
    orchestrator's import surface)."""
    import pii_anon.moe as moe_mod

    assert hasattr(moe_mod, "load_verified_gate")
    assert callable(moe_mod.load_verified_gate)
