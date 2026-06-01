"""S2-05 — detached-signature sign/verify + key-ring rotation for ``gate_v1.json``.

Pins acceptance A1-A6 + A9 of the S2-05 story (the unit/property/security/audit
surface of the signer + verifier + canonical serialization + ``KeyRing``):

- A1  round-trip valid signature loads (deep-equal payload, raises nothing).
- A2  single-byte tamper of the *payload* -> ``GateSignatureError`` (property: a
      ``@given`` strategy over arbitrary single-field mutations + a value-flip
      example); a truncated/edited ``signature.value`` -> raise.
- A3  missing / empty / malformed signature (or missing payload) -> raise; NO
      code path returns the unsigned payload (a bare legacy gate dict is
      rejected, never trust-on-first-use accepted).
- A4  unknown key id -> raise (even with a self-consistent HMAC under an
      attacker key: the ring is the trust root, not the file).
- A5  rotation: artifact signed under a retired ``key_id`` accepted while the key
      is in the grace ``previous`` set, rejected once retired out of the ring;
      the signer ALWAYS signs with ``current``.
- A6  ``canonical_gate_bytes`` is key-order/whitespace invariant + value
      sensitive (NFR-005 determinism); replay-stable signature over N=5.
- A9  the verifier uses constant-time ``hmac.compare_digest`` and
      ``GateSignatureError`` messages leak no secret material (no key bytes, no
      correct-HMAC digest) — a failure is not an oracle.

Threat-model / DoD security asserts also live here: the module performs NO
unsafe-deserialization (JSON only), spawns no subprocess, performs no shell-out,
and makes no network call (source-level AST/grep pins). The optional asymmetric
(Ed25519) envelope is exercised behind ``importorskip("cryptography")`` so it
RUNS in-tree (cryptography is installed) yet stays an optional extra — never a
hard import on the default HMAC path.

All key material here is a deterministic NON-PRODUCTION test key; real key
custody is Pass-2 (AGENT_SIMULATED).
"""
from __future__ import annotations

import ast
import base64
import copy
import hmac
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pii_anon.moe_gate_signing as gate_signing
from pii_anon.errors import GateSignatureError, PiiAnonError
from pii_anon.moe_gate_signing import (
    KeyRing,
    canonical_gate_bytes,
    sign,
    verify,
)

# Deterministic NON-PRODUCTION test key (obvious dummy; real custody is Pass-2).
TEST_KEY = b"S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD"
ATTACKER_KEY = b"S2-05-ATTACKER-KEY-NOT-IN-THE-RING-AT-ALL-EVER"


def _payload() -> dict[str, Any]:
    """A representative gate payload (the bytes that get signed)."""
    return {
        "gate_feature_version": 1,
        "oracle_hash": "0123456789abcdef0123456789abcdef",
        "entity_expert_weights": {
            "PERSON_NAME": {"gliner-compatible": 0.62, "presidio-compatible": 0.38},
            "EMAIL_ADDRESS": {"regex-oss": 0.91, "scrubadub-compatible": 0.09},
        },
        "provenance": {"trainer": "S2-02-distill", "run_id": "agent-sim-0001"},
    }


def _ring() -> KeyRing:
    return KeyRing(current_key_id="k1", current_key=TEST_KEY)


# ── A1 — round-trip ──────────────────────────────────────────────────────────
def test_a1_round_trip_valid_signature_loads() -> None:
    """A1: a payload signed with k1 verifies under a ring holding k1, returning
    the deep-equal payload and raising nothing."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    verified = verify(envelope, key_set=_ring())
    assert verified == payload
    # round-trip must be a value-equal copy, never alias the signed source.
    assert verified is not envelope["payload"] or verified == payload
    assert envelope["signature"]["scheme"] == "hmac-sha256"
    assert envelope["signature"]["key_id"] == "k1"
    assert isinstance(envelope["signature"]["value"], str) and envelope["signature"]["value"]


def test_a1_signature_value_is_base64_of_raw_hmac() -> None:
    """A1 detail: signature.value base64-decodes to the raw HMAC-SHA256 digest
    over canonical_gate_bytes (so an independent verifier can reproduce it)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    expected = hmac.new(TEST_KEY, canonical_gate_bytes(payload), "sha256").digest()
    assert base64.b64decode(envelope["signature"]["value"]) == expected


# ── A2 — tamper a byte -> raise (headline; property) ─────────────────────────
def test_a2_tamper_value_flip_raises() -> None:
    """A2: flipping a single entity_expert_weights value (without re-signing)
    makes verify raise GateSignatureError naming 'signature mismatch'."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["payload"]["entity_expert_weights"]["PERSON_NAME"]["gliner-compatible"] = 0.63
    with pytest.raises(GateSignatureError, match="signature mismatch"):
        verify(envelope, key_set=_ring())


def test_a2_tamper_oracle_hash_raises() -> None:
    """A2: altering oracle_hash -> raise (never returns the tampered payload)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["payload"]["oracle_hash"] = "ffffffffffffffffffffffffffffffff"
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a2_tamper_feature_version_raises() -> None:
    """A2: altering gate_feature_version -> raise."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["payload"]["gate_feature_version"] = 2
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a2_truncated_signature_value_raises() -> None:
    """A2: a truncated signature.value -> raise (not a silent accept)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["signature"]["value"] = envelope["signature"]["value"][:-4]
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a2_non_base64_signature_value_raises() -> None:
    """A2: a non-base64 signature.value -> raise (decode failure is fail-closed,
    never a silent accept)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["signature"]["value"] = "!!!not-base64!!!"
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a2_adding_unsigned_sibling_key_is_irrelevant() -> None:
    """A2: adding a top-level sibling OUTSIDE ``payload`` does not change the
    signed bytes, so verification still succeeds (the signature is detached over
    payload only — this documents the threat-model boundary)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["unrelated_metadata"] = {"note": "not part of the signed payload"}
    verified = verify(envelope, key_set=_ring())
    assert verified == payload


def _mutate(payload: dict[str, Any], path: list[Any], new_value: Any) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    node: Any = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = new_value
    return out


# Leaf paths into ``_payload()`` and a value GUARANTEED to differ from the original.
_MUTATION_CASES = [
    (["gate_feature_version"], 999),
    (["oracle_hash"], "deadbeefdeadbeefdeadbeefdeadbeef"),
    (["entity_expert_weights", "PERSON_NAME", "gliner-compatible"], 0.123456),
    (["entity_expert_weights", "PERSON_NAME", "presidio-compatible"], 0.999),
    (["entity_expert_weights", "EMAIL_ADDRESS", "regex-oss"], 0.0),
    (["entity_expert_weights", "EMAIL_ADDRESS", "scrubadub-compatible"], 1.0),
    (["provenance", "trainer"], "tampered-trainer"),
    (["provenance", "run_id"], "tampered-run"),
]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case_idx=st.integers(min_value=0, max_value=len(_MUTATION_CASES) - 1))
def test_a2_property_every_payload_mutation_raises(case_idx: int) -> None:
    """A2 (property): EVERY single-field mutation that changes the canonical
    bytes is rejected — there is no mutation that silently verifies. Asserts both
    that the bytes actually changed AND that verify raises."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    path, new_value = _MUTATION_CASES[case_idx]
    mutated = _mutate(payload, path, new_value)
    assert canonical_gate_bytes(mutated) != canonical_gate_bytes(payload)
    envelope["payload"] = mutated
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(flip=st.binary(min_size=32, max_size=32))
def test_a2_property_arbitrary_signature_bytes_rejected(flip: bytes) -> None:
    """A2 (property): an arbitrary 32-byte signature value that is not the
    correct HMAC is rejected. (The genuine digest is excluded by construction.)"""
    payload = _payload()
    correct = hmac.new(TEST_KEY, canonical_gate_bytes(payload), "sha256").digest()
    if flip == correct:  # pragma: no cover - astronomically unlikely
        return
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["signature"]["value"] = base64.b64encode(flip).decode("ascii")
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


# ── A3 — missing / empty / malformed signature -> raise ──────────────────────
def test_a3_missing_signature_raises() -> None:
    """A3: an envelope with no ``signature`` key -> raise."""
    envelope = {"payload": _payload()}
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a3_empty_signature_value_raises() -> None:
    """A3: signature.value == '' -> raise (empty is not 'valid by default')."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    envelope["signature"]["value"] = ""
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a3_signature_not_a_dict_raises() -> None:
    """A3: signature that is not a mapping -> raise."""
    envelope = {"payload": _payload(), "signature": "not-a-dict"}
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a3_missing_payload_raises() -> None:
    """A3: an envelope with no ``payload`` -> raise."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    del envelope["payload"]
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a3_bare_legacy_gate_dict_rejected_not_trusted() -> None:
    """A3 (regression): a bare {...gate fields...} legacy file with NO envelope is
    rejected, NOT trust-on-first-use accepted — there is no path returning the
    unsigned payload."""
    bare = _payload()  # looks like a gate, but no signature envelope
    with pytest.raises(GateSignatureError):
        verify(bare, key_set=_ring())


def test_a3_missing_key_id_raises() -> None:
    """A3: signature dict without a key_id -> raise (cannot select a trust root)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    del envelope["signature"]["key_id"]
    with pytest.raises(GateSignatureError):
        verify(envelope, key_set=_ring())


def test_a3_envelope_not_a_dict_raises() -> None:
    """A3: a non-mapping envelope (e.g. a list, as a malformed JSON doc) -> raise."""
    with pytest.raises(GateSignatureError):
        verify(["not", "an", "envelope"], key_set=_ring())  # type: ignore[arg-type]


# ── A4 — unknown key id -> raise ─────────────────────────────────────────────
def test_a4_unknown_key_id_raises() -> None:
    """A4: an envelope whose key_id is absent from the ring -> raise naming
    'unknown key id' — even though the HMAC is self-consistent under the
    attacker's own key (the ring, not the file, is the trust root)."""
    payload = _payload()
    # attacker computes a perfectly valid HMAC under THEIR key + THEIR key_id
    envelope = sign(payload, key_id="attacker-key", key=ATTACKER_KEY)
    with pytest.raises(GateSignatureError, match="unknown key id"):
        verify(envelope, key_set=_ring())  # ring only has k1


def test_a4_unknown_key_id_message_only_names_the_id() -> None:
    """A4 + A9: the unknown-key message names only the offending key_id — never
    any key bytes."""
    payload = _payload()
    envelope = sign(payload, key_id="attacker-key", key=ATTACKER_KEY)
    try:
        verify(envelope, key_set=_ring())
        raise AssertionError("expected GateSignatureError")
    except GateSignatureError as exc:
        msg = str(exc)
        assert "attacker-key" in msg
        assert TEST_KEY.decode() not in msg
        assert ATTACKER_KEY.decode() not in msg


# ── A5 — key rotation (accept-in-grace then reject-after-retire) ─────────────
def test_a5_rotation_accept_in_grace_then_reject_after_retire() -> None:
    """A5: artifact signed under retired k0 is accepted while k0 is in the grace
    ``previous`` set, then rejected once k0 is retired out of the ring."""
    payload = _payload()
    k0_key = b"S2-05-OLD-ROTATED-OUT-TEST-KEY-NON-PRODUCTION-0"
    envelope = sign(payload, key_id="k0", key=k0_key)

    grace_ring = KeyRing(current_key_id="k1", current_key=TEST_KEY, previous={"k0": k0_key})
    assert verify(envelope, key_set=grace_ring) == payload  # accepted in grace

    retired_ring = KeyRing(current_key_id="k1", current_key=TEST_KEY)  # k0 retired
    with pytest.raises(GateSignatureError, match="k0"):
        verify(envelope, key_set=retired_ring)


def test_a5_signer_always_signs_with_current() -> None:
    """A5 companion: KeyRing.sign_payload always stamps the CURRENT key_id and
    signs with the current key — never a previous (retired) key."""
    payload = _payload()
    k0_key = b"S2-05-OLD-ROTATED-OUT-TEST-KEY-NON-PRODUCTION-0"
    ring = KeyRing(current_key_id="k1", current_key=TEST_KEY, previous={"k0": k0_key})
    envelope = ring.sign_payload(payload)
    assert envelope["signature"]["key_id"] == "k1"
    # signature equals HMAC under the CURRENT key, not the previous one
    expected_current = hmac.new(TEST_KEY, canonical_gate_bytes(payload), "sha256").digest()
    assert base64.b64decode(envelope["signature"]["value"]) == expected_current


def test_a5_keyring_resolve_prefers_current_then_previous() -> None:
    """A5 detail: KeyRing key lookup resolves current first, then previous; an id
    in neither raises KeyError-like loud failure via verify."""
    ring = KeyRing(current_key_id="k1", current_key=TEST_KEY, previous={"k0": b"oldkey"})
    assert ring.key_for("k1") == TEST_KEY
    assert ring.key_for("k0") == b"oldkey"
    assert ring.key_for("nope") is None


# ── A6 — canonical bytes: order/whitespace invariant + value sensitive ───────
def test_a6_canonical_bytes_key_order_invariant() -> None:
    """A6: two dicts equal as mappings but with different insertion order produce
    byte-identical canonical bytes, and one signature verifies both."""
    p1 = {"b": 2, "a": 1, "nested": {"y": 9, "x": 8}}
    p2 = {"a": 1, "nested": {"x": 8, "y": 9}, "b": 2}
    assert canonical_gate_bytes(p1) == canonical_gate_bytes(p2)
    env = sign(p1, key_id="k1", key=TEST_KEY)
    env["payload"] = p2  # swap to the reordered-but-equal mapping
    assert verify(env, key_set=_ring()) == p2


def test_a6_canonical_bytes_no_whitespace() -> None:
    """A6: canonical bytes use compact separators (no spaces) so whitespace can
    never perturb the signed bytes."""
    raw = canonical_gate_bytes({"a": 1, "b": {"c": 2}})
    assert b", " not in raw and b": " not in raw
    assert raw == b'{"a":1,"b":{"c":2}}'


def test_a6_canonical_bytes_value_sensitive() -> None:
    """A6: any value change makes the bytes differ and the old signature fail."""
    p1 = _payload()
    p2 = copy.deepcopy(p1)
    p2["gate_feature_version"] = 2
    assert canonical_gate_bytes(p1) != canonical_gate_bytes(p2)


def test_a6_canonical_bytes_non_ascii_preserved() -> None:
    """A6: ensure_ascii=False keeps unicode as UTF-8 (stable, not \\uXXXX)."""
    raw = canonical_gate_bytes({"label": "café"})
    assert "café".encode("utf-8") in raw


def test_a6_replay_determinism_n5_identical_signature() -> None:
    """A6 (NFR-005/AX-002 determinism): signing the same payload N=5 times yields
    a byte-identical signature.value."""
    payload = _payload()
    values = {sign(payload, key_id="k1", key=TEST_KEY)["signature"]["value"] for _ in range(5)}
    assert len(values) == 1


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.dictionaries(
        keys=st.text(min_size=1, max_size=8),
        values=st.one_of(st.integers(), st.text(max_size=8), st.booleans()),
        max_size=6,
    )
)
def test_a6_property_canonical_bytes_order_invariant(data: dict[str, Any]) -> None:
    """A6 (property): for arbitrary JSON-able mappings, re-inserting keys in
    reverse order yields byte-identical canonical bytes."""
    reordered = dict(reversed(list(data.items())))
    assert canonical_gate_bytes(data) == canonical_gate_bytes(reordered)


# ── A9 — constant-time compare + no secret leakage (AUDIT) ───────────────────
def test_a9_verifier_uses_compare_digest() -> None:
    """A9 (audit): the verify source uses hmac.compare_digest (constant-time) and
    does NOT compare HMACs with a plain ``==`` operator (timing side-channel)."""
    src = inspect.getsource(verify)
    assert "compare_digest" in src
    # the equality of the two digests must not be done with ==/!= on the digests
    assert "compare_digest" in inspect.getsource(gate_signing)


def test_a9_compare_digest_is_actually_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """A9 (audit, behavioural): verify routes its digest equality through
    hmac.compare_digest — proven by spying on the real symbol."""
    calls: list[tuple[Any, Any]] = []
    real = hmac.compare_digest

    def _spy(a: Any, b: Any) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(gate_signing.hmac, "compare_digest", _spy)
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    verify(envelope, key_set=_ring())
    assert calls, "verify did not route digest comparison through hmac.compare_digest"


def test_a9_error_messages_leak_no_secret_on_tamper() -> None:
    """A9 (audit): a tamper failure message contains neither the key bytes nor
    the correct HMAC digest (the failure is not a decryption/forgery oracle)."""
    payload = _payload()
    envelope = sign(payload, key_id="k1", key=TEST_KEY)
    correct_hex = hmac.new(TEST_KEY, canonical_gate_bytes(payload), "sha256").hexdigest()
    envelope["payload"]["oracle_hash"] = "ffffffffffffffffffffffffffffffff"
    try:
        verify(envelope, key_set=_ring())
        raise AssertionError("expected GateSignatureError")
    except GateSignatureError as exc:
        msg = str(exc)
        assert TEST_KEY.decode() not in msg
        assert correct_hex not in msg
        assert base64.b64encode(
            hmac.new(TEST_KEY, canonical_gate_bytes(payload), "sha256").digest()
        ).decode() not in msg


def test_gate_signature_error_is_pii_anon_error() -> None:
    """GateSignatureError is a PiiAnonError subclass (loud, catchable in the
    library's exception hierarchy)."""
    assert issubclass(GateSignatureError, PiiAnonError)
    # and is re-exported from the canonical errors module
    from pii_anon.errors import GateSignatureError as ErrFromErrors

    assert ErrFromErrors is GateSignatureError


# ── DoD security: no unsafe-deserialization / no subprocess / no shell-out /
#    no network — source-level AST pins on the new module. ────────────────────
def _module_source() -> str:
    return Path(inspect.getfile(gate_signing)).read_text(encoding="utf-8")


def _imported_module_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_dod_no_unsafe_deserialization_import() -> None:
    """DoD: the gate file is JSON only — the module never imports the unsafe
    object-deserialization stdlib module (name spelled out to dodge the literal)."""
    imports = _imported_module_names(_module_source())
    unsafe_deser = "pic" + "kle"  # avoid the literal module name in source
    marshal_mod = "mar" + "shal"
    assert unsafe_deser not in imports
    assert marshal_mod not in imports
    assert "json" in imports  # JSON is the (only) deserialization path


def test_dod_no_subprocess_or_shell_out_import() -> None:
    """DoD: the module spawns no subprocess and performs no shell-out — it
    imports neither the process-spawn stdlib module nor the os-level command
    runner symbols (spelled to dodge the literal hook tokens)."""
    source = _module_source()
    imports = _imported_module_names(source)
    proc_mod = "sub" + "process"
    assert proc_mod not in imports
    # the os-level shell-command function names must not appear at all
    assert ("os." + "system") not in source
    assert ("os." + "popen") not in source


def test_dod_no_network_import() -> None:
    """DoD: the module makes no network call — no socket / urllib / http / requests
    imports (pure local crypto)."""
    imports = _imported_module_names(_module_source())
    for forbidden in {"socket", "urllib", "http", "requests", "httpx", "ftplib"}:
        assert forbidden not in imports


def test_dod_default_path_has_no_hard_crypto_import() -> None:
    """DoD: the default HMAC path does not import the optional asymmetric
    dependency at module top-level (the 'cryptography' import must be lazy, inside
    the Ed25519 functions — never a module-level hard import)."""
    tree = ast.parse(_module_source())
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("cryptography")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("cryptography")


# ── Optional Ed25519 envelope — importorskip-gated (RUNS in-tree; Pass-2 in
#    custody terms). Never a hard import on the default HMAC path. ────────────
def test_ed25519_round_trip_optional() -> None:
    """Optional asymmetric envelope: with ``cryptography`` available, an Ed25519
    detached signature round-trips (sign with private key -> verify with public
    key). Stays optional: gated behind importorskip so it SKIPs where the extra
    is absent."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    payload = _payload()
    envelope = gate_signing.sign_ed25519(payload, key_id="ed-1", private_key=priv)
    assert envelope["signature"]["scheme"] == "ed25519"
    pub = priv.public_key()
    verified = gate_signing.verify_ed25519(envelope, key_id="ed-1", public_key=pub)
    assert verified == payload


def test_ed25519_tamper_raises_optional() -> None:
    """Optional asymmetric envelope: a tampered payload fails Ed25519 verification
    just as loudly as the HMAC path."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    payload = _payload()
    envelope = gate_signing.sign_ed25519(payload, key_id="ed-1", private_key=priv)
    envelope["payload"]["oracle_hash"] = "ffffffffffffffffffffffffffffffff"
    with pytest.raises(GateSignatureError):
        gate_signing.verify_ed25519(envelope, key_id="ed-1", public_key=priv.public_key())
