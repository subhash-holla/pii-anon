"""Re-identification service for reversing pseudonymized text.

Provides controlled de-anonymization with full audit logging.  Designed
for enterprise workflows where authorized personnel need to recover
original PII from tokenized documents (e.g., GDPR data subject access
requests, legal discovery, clinical data review).

Usage::

    from pii_anon.tokenization.reidentification import ReidentificationService

    svc = ReidentificationService(token_store, key_manager, tokenizer)
    original = svc.detokenize_text(anonymized_doc, scope="case_42")
    print(svc.audit_log)  # full audit trail
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from pii_anon.tokenization.providers import TokenizerProvider, TokenRecord
from pii_anon.tokenization.store import TokenStore


@dataclass(frozen=True)
class ReidentificationAuditEntry:
    """Record of a single re-identification attempt.

    Attributes
    ----------
    timestamp : float
        Unix timestamp of the attempt.
    scope : str
        Tokenization scope searched.
    token : str
        The token that was looked up.
    entity_type : str
        Entity type of the token (if known).
    version : int
        Key version used.
    success : bool
        Whether the plaintext was recovered.
    reason : str
        Human-readable reason or context for the request.
    """

    timestamp: float
    scope: str
    token: str
    entity_type: str = ""
    version: int = 0
    success: bool = False
    reason: str = ""


# Pattern matching token formats:
#   <ENTITY_TYPE:vN:tok_XXX>   (HMAC tokens)
#   <ENTITY_TYPE:vN:aes_XXX>   (AES-SIV tokens)
_TOKEN_PATTERN = re.compile(
    r"<([A-Z_]+):v(\d+):(tok_[A-Za-z0-9_-]+|aes_[A-Za-z0-9_-]+)>"
)


class ReidentificationService:
    """Controlled de-anonymization with audit logging.

    Looks up tokens in the token store and/or uses the tokenizer's
    ``detokenize`` method to recover original plaintext values.

    Parameters
    ----------
    token_store : TokenStore
        The token storage backend holding mappings.
    key_manager : object | None
        Optional ``KeyManager`` for resolving versioned keys.  If *None*,
        a fixed key must be supplied to detokenize methods.
    tokenizer : TokenizerProvider | None
        Tokenizer for cryptographic detokenization (AES-SIV).
    default_key : str | None
        Fallback key when no ``key_manager`` is provided.

    Example
    -------
    >>> svc = ReidentificationService(store, key_manager=km, tokenizer=tok)
    >>> original = svc.detokenize_text(masked_text, scope="case_1")
    >>> len(svc.audit_log)  # one entry per token attempted
    3
    """

    def __init__(
        self,
        token_store: TokenStore,
        *,
        key_manager: Any | None = None,
        tokenizer: TokenizerProvider | None = None,
        default_key: str | None = None,
    ) -> None:
        self._store = token_store
        self._key_manager = key_manager
        self._tokenizer = tokenizer
        self._default_key = default_key
        self._lock = Lock()
        self._audit: list[ReidentificationAuditEntry] = []

    @property
    def audit_log(self) -> list[ReidentificationAuditEntry]:
        """Return a copy of the audit log."""
        with self._lock:
            return list(self._audit)

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        with self._lock:
            self._audit.clear()

    def _resolve_key(self, version: int) -> str | None:
        """Resolve the secret key for a given version."""
        if self._key_manager is not None:
            try:
                result = self._key_manager.get_key(version)
                return str(result) if result is not None else None
            except (KeyError, ValueError):
                return None
        return self._default_key

    def _record_audit(
        self,
        scope: str,
        token: str,
        entity_type: str,
        version: int,
        success: bool,
        reason: str,
    ) -> None:
        """Append an audit entry (thread-safe)."""
        entry = ReidentificationAuditEntry(
            timestamp=time.time(),
            scope=scope,
            token=token,
            entity_type=entity_type,
            version=version,
            success=success,
            reason=reason,
        )
        with self._lock:
            self._audit.append(entry)

    def detokenize_single(
        self,
        token: str,
        *,
        scope: str,
        reason: str = "",
    ) -> str | None:
        """Reverse a single token to its original plaintext.

        Parameters
        ----------
        token : str
            The full token string (e.g., ``<EMAIL:v1:tok_abc123>``).
        scope : str
            Tokenization scope to search within.
        reason : str
            Audit reason for this lookup.

        Returns
        -------
        str | None
            The original plaintext, or *None* if not recoverable.
        """
        match = _TOKEN_PATTERN.fullmatch(token)
        entity_type = ""
        version = 0

        if match:
            entity_type = match.group(1)
            version = int(match.group(2))

        # Try store lookup first (always fastest)
        mapping = self._store.get(token, scope=scope)
        if mapping is not None:
            self._record_audit(scope, token, entity_type, version, True, reason)
            return mapping.plaintext

        # Try cryptographic detokenization for AES-SIV tokens
        if match and self._tokenizer is not None:
            key = self._resolve_key(version)
            if key is not None:
                token_record = TokenRecord(
                    entity_type=entity_type,
                    version=version,
                    token=token,
                    scope=scope,
                )
                try:
                    result = self._tokenizer.detokenize(
                        token_record, key=key, store=self._store
                    )
                    if result is not None:
                        self._record_audit(scope, token, entity_type, version, True, reason)
                        return result
                except Exception:
                    pass

        self._record_audit(scope, token, entity_type, version, False, reason)
        return None

    def bulk_detokenize(
        self,
        tokens: list[str],
        *,
        scope: str,
        reason: str = "",
    ) -> dict[str, str | None]:
        """Reverse multiple tokens in a single call.

        Parameters
        ----------
        tokens : list[str]
            Token strings to look up.
        scope : str
            Tokenization scope.
        reason : str
            Audit reason for these lookups.

        Returns
        -------
        dict[str, str | None]
            Mapping from token → plaintext (or *None* if not found).
        """
        return {
            token: self.detokenize_single(token, scope=scope, reason=reason)
            for token in tokens
        }

    def detokenize_text(
        self,
        anonymized_text: str,
        *,
        scope: str,
        reason: str = "",
    ) -> str:
        """Reverse all tokens in a full document.

        Scans the text for token patterns (``<ENTITY:vN:tok_XXX>`` or
        ``<ENTITY:vN:aes_XXX>``), looks up each in the store, and
        replaces them with their original plaintext values.

        Parameters
        ----------
        anonymized_text : str
            Text containing pseudonymization tokens.
        scope : str
            Tokenization scope.
        reason : str
            Audit reason for this de-anonymization.

        Returns
        -------
        str
            Text with all recoverable tokens replaced by plaintext.
            Tokens that cannot be reversed are left in place.
        """
        def _replace(m: re.Match[str]) -> str:
            full_token = m.group(0)
            result = self.detokenize_single(full_token, scope=scope, reason=reason)
            return result if result is not None else full_token

        return _TOKEN_PATTERN.sub(_replace, anonymized_text)

    def detokenize_payload(
        self,
        payload: dict[str, Any],
        *,
        scope: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Reverse tokens in all string values of a payload dict.

        Parameters
        ----------
        payload : dict[str, Any]
            Payload with potentially tokenized string values.
        scope : str
            Tokenization scope.
        reason : str
            Audit reason.

        Returns
        -------
        dict[str, Any]
            Payload with string values de-anonymized.
        """
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                result[key] = self.detokenize_text(
                    value, scope=scope, reason=reason
                )
            elif isinstance(value, dict):
                result[key] = self.detokenize_payload(
                    value, scope=scope, reason=reason
                )
            else:
                result[key] = value
        return result
