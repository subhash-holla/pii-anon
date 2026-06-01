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
