"""S2-05 — sign + verify the MoE-router learned-gate artifact ``gate_v1.json``.

``gate_v1.json`` is a *control-path* artifact: when present, the DistilledTopKGate
(DC-02) steers which experts are weighted / early-exited per entity type. An
auto-discovered file on disk that silently re-weights the detector is a
privilege-escalation surface (AX-006 least-privilege): an attacker who can drop
or edit the file can bias routing. This module makes the artifact **unforgeable
and fail-closed**.

Design
------
* **Detached signature over a canonical serialization.** The on-disk file is an
  envelope ``{"payload": {...gate fields...}, "signature": {...}}``. The signature
  covers ``canonical_gate_bytes(payload)`` only — adding/rotating the envelope
  never perturbs the signed bytes. ``canonical_gate_bytes`` is
  ``json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``
  encoded UTF-8: byte-identical across replays (NFR-005 / determinism) so the
  signature is stable, key-order/whitespace invariant, and value sensitive.
* **Default scheme: HMAC-SHA256 (pure-stdlib, zero new dependency).** ``verify``
  recomputes the HMAC and compares with ``hmac.compare_digest`` (constant-time —
  no timing side-channel). Why a shared secret: HMAC needs the same key on the
  signer and the verifier. The optional asymmetric scheme (below) lets a verifier
  hold only a public key, which is preferable once real key custody exists.
* **Optional Ed25519 (asymmetric), behind a lazy import.** ``sign_ed25519`` /
  ``verify_ed25519`` require the optional ``cryptography`` package (the
  ``gate-signing`` extra). The import is **lazy** (inside the functions) so the
  default HMAC path has no hard dependency. These functions are
  ``importorskip``-gated in tests.
* **Key identification + rotation.** The envelope carries ``key_id``. A
  :class:`KeyRing` holds a designated ``current`` key plus zero-or-more
  ``previous`` retired-but-in-grace keys. ``verify`` resolves ``key_id`` against
  the ring; an id in neither set is a loud reject (the *ring*, not the file, is
  the trust root). The signer always signs with ``current``; retiring a key out
  of the ring rejects artifacts signed with it (after grace).
* **Key material is Pass-2.** No real secret is committed. A deterministic,
  clearly NON-PRODUCTION test key (:data:`_TEST_HMAC_KEY`) is sourced through an
  env-overridable provider seam (:func:`gate_signing_key_provider`) so real
  custody (env var / secret store / HSM / KMS) drops in without a code change.

Threat model
------------
The signed bytes cover the gate *payload* (weights / oracle-hash / feature
version), not the filename or filesystem ACLs. This is integrity + authenticity
of the artifact content under the key ring — **not** confidentiality (the gate is
not secret) and **not** a substitute for filesystem permissions. The gate file is
parsed as JSON only; it is never routed through any unsafe object-deserialization
path, never executed, and this module spawns no subprocess, performs no
shell-out, and makes no network call.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any

from pii_anon.errors import GateSignatureError

__all__ = [
    "GateSignatureError",
    "KeyRing",
    "canonical_gate_bytes",
    "gate_signing_key_provider",
    "sign",
    "sign_ed25519",
    "verify",
    "verify_ed25519",
]

HMAC_SCHEME = "hmac-sha256"
ED25519_SCHEME = "ed25519"
ALG_VERSION = 1

# ── NON-PRODUCTION test key ──────────────────────────────────────────────────
# Deterministic, obvious-dummy key for AGENT_SIMULATED in-tree exercise ONLY.
# Real signing-key custody (env var / secret store / HSM / KMS) is a Pass-2
# concern and is injected through ``gate_signing_key_provider`` below — this
# constant is NEVER a production secret.
_TEST_HMAC_KEY: bytes = b"S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD"

# Env var that, when set, overrides the NON-PRODUCTION default with operator-
# supplied key material (still not a real HSM — that is Pass-2 — but the seam is
# here so custody drops in without a code change).
_KEY_ENV_VAR = "PII_ANON_GATE_SIGNING_KEY"


def gate_signing_key_provider() -> bytes:
    """Resolve the HMAC signing key (the env-overridable custody seam).

    Returns the operator-supplied key from the :data:`_KEY_ENV_VAR` environment
    variable (UTF-8 encoded) when set, else the deterministic NON-PRODUCTION test
    key. Real custody (secret store / HSM / KMS) replaces this body in Pass-2
    without changing any caller.
    """
    env_key = os.environ.get(_KEY_ENV_VAR)
    if env_key:
        return env_key.encode("utf-8")
    return _TEST_HMAC_KEY


def canonical_gate_bytes(payload: dict[str, Any]) -> bytes:
    """Return the canonical UTF-8 serialization of a gate ``payload``.

    Deterministic and byte-identical across replays (NFR-005 / determinism):
    keys are sorted, separators are compact (no whitespace), and non-ASCII is
    preserved as UTF-8 (``ensure_ascii=False``). The signature is computed over
    exactly these bytes, so any field reordering / whitespace difference is
    irrelevant and any *value* change is detected.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ── Key ring (current + grace previous) ──────────────────────────────────────
@dataclass(frozen=True)
class KeyRing:
    """A signing/verification key ring: a designated ``current`` key plus
    zero-or-more ``previous`` retired-but-in-grace keys.

    The signer always signs with ``current``. ``verify`` resolves a signature's
    ``key_id`` against the ring (current first, then previous); an id in neither
    set is a loud reject — the ring, not the artifact, is the trust root.
    Rotation retires a key by dropping it from ``previous`` (or by closing the
    grace window), at which point artifacts signed with it are rejected.
    """

    current_key_id: str
    current_key: bytes
    previous: dict[str, bytes] = field(default_factory=dict)

    def key_for(self, key_id: str) -> bytes | None:
        """Return the key bytes for ``key_id`` (current, then previous), or
        ``None`` if the id is not in the ring."""
        if key_id == self.current_key_id:
            return self.current_key
        return self.previous.get(key_id)

    def sign_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sign ``payload`` with the CURRENT key, returning the detached envelope.

        Always stamps ``current_key_id`` and signs with ``current_key`` — never a
        previous (retired) key.
        """
        return sign(payload, key_id=self.current_key_id, key=self.current_key)


def sign(payload: dict[str, Any], *, key_id: str, key: bytes) -> dict[str, Any]:
    """Produce a detached HMAC-SHA256 signature envelope for ``payload``.

    The signature is the base64 of ``HMAC-SHA256(key, canonical_gate_bytes(payload))``.
    Returns ``{"payload": payload, "signature": {scheme, key_id, alg_version, value}}``.
    """
    digest = hmac.new(key, canonical_gate_bytes(payload), hashlib.sha256).digest()
    return {
        "payload": payload,
        "signature": {
            "scheme": HMAC_SCHEME,
            "key_id": key_id,
            "alg_version": ALG_VERSION,
            "value": base64.b64encode(digest).decode("ascii"),
        },
    }


def _require_envelope(envelope: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate envelope shape and return ``(payload, signature)``.

    Raises :class:`GateSignatureError` (fail-closed) on any malformed shape:
    non-mapping envelope, missing/non-mapping ``payload``, missing/non-mapping
    ``signature``, or a ``signature`` lacking a ``key_id`` / a non-empty ``value``.
    There is NO branch that returns an unsigned payload.
    """
    if not isinstance(envelope, dict):
        raise GateSignatureError("malformed gate envelope: expected a mapping")
    if "payload" not in envelope or not isinstance(envelope["payload"], dict):
        raise GateSignatureError("malformed gate envelope: missing or non-mapping 'payload'")
    signature = envelope.get("signature")
    if not isinstance(signature, dict):
        raise GateSignatureError("malformed gate envelope: missing or non-mapping 'signature'")
    key_id = signature.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        raise GateSignatureError("malformed gate envelope: missing 'signature.key_id'")
    value = signature.get("value")
    if not isinstance(value, str) or not value:
        raise GateSignatureError("malformed gate envelope: missing or empty 'signature.value'")
    return envelope["payload"], signature


def verify(envelope: Any, *, key_set: KeyRing) -> dict[str, Any]:
    """Verify a detached HMAC-SHA256 envelope and return the verified payload.

    Fails LOUD (raises :class:`GateSignatureError`) on a tampered byte, a missing
    /empty/malformed signature, an unknown or retired key id, or a malformed
    envelope. Never returns an unverified payload and never silently falls back
    to unsigned content. The digest comparison is constant-time
    (``hmac.compare_digest``) so a verification failure is not a timing oracle,
    and the error messages name only the offending ``key_id`` / ``scheme`` — never
    key bytes or any correct-HMAC material.
    """
    payload, signature = _require_envelope(envelope)

    scheme = signature.get("scheme")
    if scheme != HMAC_SCHEME:
        # Asymmetric envelopes go through verify_ed25519 with an explicit public
        # key; the default verify() path is HMAC only and rejects anything else
        # rather than silently trusting an unexpected scheme.
        raise GateSignatureError(f"unsupported signature scheme: {scheme!r}")

    key_id: str = signature["key_id"]
    key = key_set.key_for(key_id)
    if key is None:
        # The ring is the trust root: an id not in current/previous is rejected
        # even if the file carries a self-consistent HMAC under an attacker key.
        raise GateSignatureError(f"unknown key id: {key_id!r} (not in the key ring)")

    try:
        provided = base64.b64decode(signature["value"], validate=True)
    except (binascii.Error, ValueError) as exc:
        # A non-base64 / truncated signature value is fail-closed, not a silent
        # accept. The exception text carries no secret material.
        raise GateSignatureError("signature mismatch: signature value is not valid base64") from exc

    expected = hmac.new(key, canonical_gate_bytes(payload), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, provided):
        raise GateSignatureError(f"signature mismatch: payload does not verify under key id {key_id!r}")

    return payload


# ── Optional Ed25519 (asymmetric) — lazy import, optional 'gate-signing' extra ─
def sign_ed25519(
    payload: dict[str, Any],
    *,
    key_id: str,
    private_key: Any,
) -> dict[str, Any]:
    """Produce a detached Ed25519 signature envelope (OPTIONAL asymmetric path).

    Requires the optional ``cryptography`` dependency (the ``gate-signing`` extra).
    The import is lazy so the default HMAC path carries no hard dependency.
    ``private_key`` is an ``Ed25519PrivateKey``.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise GateSignatureError(
            "ed25519 signing requires the optional 'gate-signing' extra (cryptography)"
        ) from exc
    if not isinstance(private_key, Ed25519PrivateKey):
        raise GateSignatureError("ed25519 signing requires an Ed25519PrivateKey")
    raw = private_key.sign(canonical_gate_bytes(payload))
    return {
        "payload": payload,
        "signature": {
            "scheme": ED25519_SCHEME,
            "key_id": key_id,
            "alg_version": ALG_VERSION,
            "value": base64.b64encode(raw).decode("ascii"),
        },
    }


def verify_ed25519(
    envelope: Any,
    *,
    key_id: str,
    public_key: Any,
) -> dict[str, Any]:
    """Verify a detached Ed25519 envelope against ``public_key`` (OPTIONAL path).

    Fails LOUD with :class:`GateSignatureError` on a tampered payload, a
    scheme/key-id mismatch, or a malformed envelope. Requires the optional
    ``cryptography`` dependency (lazy import). ``public_key`` is an
    ``Ed25519PublicKey``.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise GateSignatureError(
            "ed25519 verification requires the optional 'gate-signing' extra (cryptography)"
        ) from exc

    payload, signature = _require_envelope(envelope)
    if signature.get("scheme") != ED25519_SCHEME:
        raise GateSignatureError(f"unsupported signature scheme for ed25519 verify: {signature.get('scheme')!r}")
    if signature["key_id"] != key_id:
        raise GateSignatureError(f"unknown key id: {signature['key_id']!r} (expected {key_id!r})")
    if not isinstance(public_key, Ed25519PublicKey):
        raise GateSignatureError("ed25519 verification requires an Ed25519PublicKey")

    try:
        provided = base64.b64decode(signature["value"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GateSignatureError("signature mismatch: signature value is not valid base64") from exc

    try:
        public_key.verify(provided, canonical_gate_bytes(payload))
    except InvalidSignature as exc:
        raise GateSignatureError(f"signature mismatch: payload does not verify under key id {key_id!r}") from exc
    return payload
