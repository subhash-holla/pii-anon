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
