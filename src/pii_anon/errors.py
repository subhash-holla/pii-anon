from __future__ import annotations


class PiiAnonError(Exception):
    """Base exception for pii-anon."""


class ConfigurationError(PiiAnonError):
    """Raised when configuration is invalid or cannot be loaded."""


class EngineExecutionError(PiiAnonError):
    """Raised when an engine fails while processing a request."""


class FusionError(PiiAnonError):
    """Raised when fusion strategy execution fails."""


class TokenizationError(PiiAnonError):
    """Raised when tokenization or detokenization fails."""


class TokenStoreIntegrityError(TokenizationError):
    """Raised when an at-rest token-store row fails authenticated decryption.

    The fail-loud signal for the encrypted reversible-pseudonymization vault
    (``EncryptedSQLiteTokenStore``, S6-03). The underlying AEAD primitive
    raises ``InvalidTag`` when a stored ciphertext, nonce, auth tag, or the
    bound associated data (key id / scope / token / version) has been tampered
    with — including a row's ciphertext being relocated to a different
    ``(scope, token)`` slot. This is re-raised as ``TokenStoreIntegrityError``
    rather than ever returning ``None``, the original plaintext, or undecrypted
    bytes: a tampered vault must never silently yield a successful-but-wrong
    reversal (NFR-014 unauthorized-reversal = 0).

    Messages name only the offending ``key_id`` / blind-index prefix and a
    generic reason; they never include key bytes or any plaintext (a failure
    must not become a decryption oracle).
    """


class CalibrationError(PiiAnonError):
    """Raised when calibration data is missing or invalid."""


class ExpertManifestError(PiiAnonError):
    """Raised when an expert manifest is malformed or fails validation."""


class GateSignatureError(PiiAnonError):
    """Raised when a control-path gate artifact (``gate_v1.json``) fails
    signature verification: a tampered byte, a missing/empty/malformed
    signature, an unknown or retired key id, or a malformed envelope.

    This is the fail-loud signal for the MoE-router learned-gate load path
    (S2-05). Verification never silently accepts an unverifiable gate and never
    silently falls back to its unsigned content — it raises this instead.
    Messages name only the offending ``key_id``/``scheme`` and a generic reason;
    they never include key bytes or any correct-HMAC material (a failure must
    not be a forgery oracle).
    """


class GatePayloadError(PiiAnonError):
    """Raised when a *verified* control-path gate payload is structurally
    malformed: a missing required key, a non-finite or negative advisory weight,
    a non-mapping ``weights`` block, or an unknown ``schema_version``.

    This is the fail-loud signal for the S2-02 ``DistilledTopKGate`` payload
    validator (``DistilledGatePayload.from_payload``). The SIGNATURE is checked
    upstream by ``GateSignatureError`` / ``load_verified_gate``; this validates
    the SHAPE of the already-verified content. A structurally-broken *signed*
    gate is a loud operator error — it is never coerced into a partial/best-effort
    gate and never degraded to the static-softmax fallback (present-and-broken
    != absent). Messages name only the offending field and a generic reason; they
    never include key material.
    """
