"""Encrypted-at-rest token storage for reversible pseudonymization (S6-03).

The plain :class:`~pii_anon.tokenization.store.SQLiteTokenStore` persists the
de-tokenization mapping (``scope -> token -> plaintext``) in **cleartext** — a
direct confidentiality hole under FR-026 / AX-006 (no raw PII persisted after
masking) and NFR-014 (unauthorized-reversal must be 0): anyone with the ``.db``
file recovers every original value.

:class:`EncryptedSQLiteTokenStore` is an **additive** drop-in implementation of
the existing :class:`~pii_anon.tokenization.store.TokenStore` ABC that persists
every sensitive field under **authenticated encryption (AEAD)**:

* Each row's payload (``plaintext`` plus, defense-in-depth, ``token`` / ``scope``
  / ``entity_type`` / ``version``) is JSON-serialized over a small explicit
  field dict (never an arbitrary-object loader) and encrypted with
  :class:`~cryptography.hazmat.primitives.ciphers.aead.AESGCM` (default) or
  :class:`~cryptography.hazmat.primitives.ciphers.aead.ChaCha20Poly1305`
  (constructor-selectable). A fresh random 96-bit nonce is generated per
  encryption (no nonce reuse under one data key).
* Row context — ``key_id | blind_index | version`` — is bound as the AEAD
  **associated data (AAD)**, where ``blind_index = HMAC(index_key, scope|token)``
  is the stored primary-key column. Because the AAD includes the blind index,
  a row's ciphertext cannot be relocated to a different ``(scope, token)`` slot
  without the authentication tag failing (this thwarts cut-and-paste / row-swap
  tampering), and the blind index is always available at read time on every
  lookup path, so decryption can always reconstruct the exact AAD.
* The data-encryption key (DEK) is itself **envelope-wrapped** by a
  key-encryption key (KEK) that is supplied by a pluggable
  :class:`EnvelopeKeyProvider` and is **never written to the SQLite file in the
  clear** (NFR-015: the artifact alone, or with the wrong external secret,
  cannot re-join). Rotation stamps each row with its ``key_id`` so old rows
  still decrypt under their original DEK (mirrors ``KeyManager``'s dual-version
  grace pattern); :meth:`EncryptedSQLiteTokenStore.rewrap_all` re-encrypts
  legacy rows under a new key.
* Lookup uses a keyed, non-reversible **blind index**
  ``HMAC-SHA256(index_key, scope | token)`` so ``get`` / ``list`` stay O(1)
  without ever storing ``scope`` / ``token`` in the clear.

On read the AEAD tag is verified; a tampered ciphertext / nonce / tag, or a
relocated row (AAD mismatch), makes decryption **raise**
:class:`~pii_anon.errors.TokenStoreIntegrityError` — never a silent ``None``,
the original plaintext, or undecrypted bytes.

AGENT_SIMULATED honesty boundary
--------------------------------
The in-tree :class:`StaticTestKeyProvider` (which takes raw KEK bytes directly,
in-process) is a **synthetic test stand-in** for a real key custodian. Real
external key custody — cloud KMS, HSM, OS keyring, or any networked secret
manager — is **Pass-2**: it drops in behind the same :class:`EnvelopeKeyProvider`
protocol (a *library* call, never a shell-out / subprocess) and is NOT claimed
here as real-KMS behaviour.

Residual leak (conscious trade-off)
-----------------------------------
The blind index reveals **equality** of ``(scope, token)`` pairs (needed for
O(1) lookup). This is equality over *derived pseudonyms*, not PII, and is
consistent with FR-019's deterministic surrogates. A deployment that rejects
any equality leak can switch to a decrypt-and-scan lookup (no index) at the
cost of O(n) reads — noted here as the switch-point.

Graceful optional-dep degradation
----------------------------------
If the ``cryptography`` package is absent (the ``[crypto]`` extra is not
installed), *importing* this module still succeeds (discovery is never broken),
but constructing :class:`EncryptedSQLiteTokenStore` raises a loud
:class:`~pii_anon.errors.TokenizationError` with the
``pip install pii-anon[crypto]`` remediation hint (mirrors the existing
``AESSIVTokenizer`` guard in ``providers.py``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from pii_anon.errors import TokenizationError, TokenStoreIntegrityError
from pii_anon.tokenization.store import TokenMapping, TokenStore

try:  # optional dependency — the `[crypto]` extra (cryptography>=41.0)
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
except ImportError:  # pragma: no cover - optional dependency
    AESGCM = None  # type: ignore[assignment, misc]
    ChaCha20Poly1305 = None  # type: ignore[assignment, misc]
    InvalidTag = Exception  # type: ignore[assignment, misc]

__all__ = [
    "EncryptedSQLiteTokenStore",
    "EnvelopeKeyProvider",
    "StaticTestKeyProvider",
    "KeyEnvelope",
]

_NONCE_BYTES = 12  # 96-bit AEAD nonce (AES-GCM / ChaCha20-Poly1305 standard)
_DEK_BYTES = 32  # 256-bit data-encryption key
_AAD_SEP = b"\x1f"  # unit separator, unambiguous AAD field delimiter
_BI_SEP = b"\x1f"  # blind-index field delimiter


@dataclass(frozen=True, slots=True)
class KeyEnvelope:
    """An envelope-wrapped data-encryption key (DEK).

    Attributes
    ----------
    key_id : str
        The KEK identifier under which ``wrapped_dek`` was produced.
    wrapped_dek : bytes
        ``AEAD_KEK(dek)`` — the DEK encrypted (and authenticated) under the
        KEK. The bare DEK is **never** stored; only this wrapped form is
        persisted, so the SQLite artifact alone cannot reveal the DEK.
    created_at : float
        Unix timestamp when this envelope was created (audit).
    """

    key_id: str
    wrapped_dek: bytes
    created_at: float


@runtime_checkable
class EnvelopeKeyProvider(Protocol):
    """Pluggable key-encryption-key (KEK) custodian seam.

    The store calls this protocol to wrap / unwrap the per-store data key. The
    in-tree :class:`StaticTestKeyProvider` is the AGENT_SIMULATED stand-in; a
    real custodian (cloud KMS / HSM / OS keyring) implements the same three
    methods at Pass-2 — always as an in-process *library* call, never a
    shell-out / subprocess.
    """

    def current_key_id(self) -> str:
        """Return the identifier of the KEK this provider currently wraps with."""
        ...

    def wrap(self, dek: bytes) -> tuple[str, bytes]:
        """Wrap ``dek`` under the current KEK; return ``(key_id, wrapped_dek)``."""
        ...

    def unwrap(self, key_id: str, wrapped_dek: bytes) -> bytes:
        """Recover the bare DEK from ``wrapped_dek`` produced under ``key_id``.

        Raises if ``key_id`` is unknown or the wrapped bytes fail authentication
        (e.g. the wrong KEK is supplied).
        """
        ...


class StaticTestKeyProvider:
    """In-tree, in-process KEK provider — **AGENT_SIMULATED test stand-in**.

    Wraps the DEK with AEAD under a synthetic test KEK supplied directly as
    bytes. This is NOT a real key custodian: it performs no networking, no
    shell-out, and no subprocess. A real KMS / HSM / OS-keyring custodian is a
    Pass-2 drop-in behind :class:`EnvelopeKeyProvider`.

    Parameters
    ----------
    test_kek : bytes
        The synthetic key-encryption key (16, 24, or 32 bytes for AES-GCM).
    key_id : str
        Identifier stamped onto envelopes produced by this provider.
    """

    def __init__(self, test_kek: bytes, *, key_id: str = "test-kek-v1") -> None:
        if AESGCM is None:  # pragma: no cover - exercised via store ctor guard
            raise TokenizationError(
                "EncryptedSQLiteTokenStore requires optional dependency "
                "`cryptography`. Install with `pip install pii-anon[crypto]`."
            )
        if len(test_kek) not in (16, 24, 32):
            raise TokenizationError(
                "StaticTestKeyProvider KEK must be 16, 24, or 32 bytes "
                f"(got {len(test_kek)})."
            )
        self._kek = bytes(test_kek)
        self._key_id = key_id
        self._aead = AESGCM(self._kek)

    def current_key_id(self) -> str:
        return self._key_id

    def wrap(self, dek: bytes) -> tuple[str, bytes]:
        nonce = os.urandom(_NONCE_BYTES)
        # AAD binds the wrapped DEK to its key_id (envelope cannot be re-labelled).
        wrapped = self._aead.encrypt(nonce, dek, self._key_id.encode("utf-8"))
        return self._key_id, nonce + wrapped

    def unwrap(self, key_id: str, wrapped_dek: bytes) -> bytes:
        nonce, ct = wrapped_dek[:_NONCE_BYTES], wrapped_dek[_NONCE_BYTES:]
        try:
            return self._aead.decrypt(nonce, ct, key_id.encode("utf-8"))
        except InvalidTag as exc:  # wrong KEK or tampered envelope
            raise TokenStoreIntegrityError(
                f"Failed to unwrap data key for key_id={key_id!r}: "
                "wrong key-encryption key or tampered envelope."
            ) from exc


def _new_aead(algorithm: str, key: bytes) -> Any:
    """Construct the selected AEAD primitive over ``key``."""
    algo = algorithm.lower()
    if algo == "aesgcm":
        return AESGCM(key)
    if algo == "chacha20poly1305":
        return ChaCha20Poly1305(key)
    raise TokenizationError(
        f"Unsupported AEAD algorithm {algorithm!r}; "
        "use 'aesgcm' or 'chacha20poly1305'."
    )


class EncryptedSQLiteTokenStore(TokenStore):
    """SQLite-backed token store that encrypts every sensitive field at rest.

    Drop-in for :class:`~pii_anon.tokenization.store.TokenStore` (and therefore
    for the orchestrator's ``token_store: TokenStore | None`` seam) — same
    public surface: :meth:`put`, :meth:`get`, :meth:`list_by_scope`,
    :meth:`count`, :meth:`delete_expired`, :meth:`close`.

    Parameters
    ----------
    db_path : str | Path
        Database file path. ``":memory:"`` is accepted but note the
        not-plaintext-*on-disk* guarantee only matters for file-backed stores.
    key_provider : EnvelopeKeyProvider
        The KEK custodian. The in-tree :class:`StaticTestKeyProvider` is the
        AGENT_SIMULATED test stand-in; real KMS/HSM is Pass-2.
    algorithm : str
        AEAD algorithm for row encryption: ``"aesgcm"`` (default) or
        ``"chacha20poly1305"``.
    """

    _TABLE = "encrypted_token_mappings"
    _KEYS_TABLE = "encrypted_token_keys"

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        key_provider: EnvelopeKeyProvider,
        algorithm: str = "aesgcm",
    ) -> None:
        if AESGCM is None or ChaCha20Poly1305 is None:
            raise TokenizationError(
                "EncryptedSQLiteTokenStore requires optional dependency "
                "`cryptography`. Install with `pip install pii-anon[crypto]`."
            )
        self.db_path = str(db_path)
        self._algorithm = algorithm
        self._provider = key_provider
        self._lock = Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Per-key-id unwrapped DEK cache (in-memory only; never persisted).
        self._dek_cache: dict[str, bytes] = {}
        self._init_tables()
        self._ensure_current_envelope()

    # -- schema ------------------------------------------------------------
    def _init_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    blind_index BLOB NOT NULL PRIMARY KEY,
                    scope_index BLOB NOT NULL,
                    key_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload BLOB NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0.0,
                    expires_at REAL
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_scope
                ON {self._TABLE}(scope_index)
                """
            )
            self._conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_expires
                ON {self._TABLE}(expires_at)
                WHERE expires_at IS NOT NULL
                """
            )
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._KEYS_TABLE} (
                    key_id TEXT NOT NULL PRIMARY KEY,
                    wrapped_dek BLOB NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0.0
                )
                """
            )

    # -- key / envelope management ----------------------------------------
    def _ensure_current_envelope(self) -> None:
        """Ensure an envelope exists for the provider's current key id.

        If the current key id has no envelope yet, mint a fresh DEK and persist
        only its wrapped form. The bare DEK lives only in the in-memory cache.
        """
        key_id = self._provider.current_key_id()
        with self._lock:
            row = self._conn.execute(
                f"SELECT wrapped_dek FROM {self._KEYS_TABLE} WHERE key_id = ?",
                (key_id,),
            ).fetchone()
            if row is None:
                dek = os.urandom(_DEK_BYTES)
                stamped_id, wrapped = self._provider.wrap(dek)
                with self._conn:
                    self._conn.execute(
                        f"INSERT OR REPLACE INTO {self._KEYS_TABLE}"
                        f"(key_id, wrapped_dek, created_at) VALUES (?, ?, ?)",
                        (stamped_id, wrapped, time.time()),
                    )
                self._dek_cache[stamped_id] = dek

    def _dek_for(self, key_id: str) -> bytes:
        """Return the (cached) unwrapped DEK for ``key_id``; unwrap on demand."""
        cached = self._dek_cache.get(key_id)
        if cached is not None:
            return cached
        row = self._conn.execute(
            f"SELECT wrapped_dek FROM {self._KEYS_TABLE} WHERE key_id = ?",
            (key_id,),
        ).fetchone()
        if row is None:
            raise TokenStoreIntegrityError(
                f"No key envelope found for key_id={key_id!r}; cannot decrypt."
            )
        dek = self._provider.unwrap(key_id, bytes(row[0]))
        self._dek_cache[key_id] = dek
        return dek

    @staticmethod
    def _index_key_for(dek: bytes) -> bytes:
        """Derive the blind-index HMAC key from the DEK (domain-separated).

        Distinct from the AEAD encryption key: the index key only ever computes
        a keyed digest of ``scope | token`` and never decrypts anything.
        """
        return hashlib.sha256(b"pii-anon/blind-index-v1" + dek).digest()

    @staticmethod
    def _blind_index(index_key: bytes, scope: str, token: str) -> bytes:
        msg = scope.encode("utf-8") + _BI_SEP + token.encode("utf-8")
        return hmac.new(index_key, msg, hashlib.sha256).digest()

    @staticmethod
    def _scope_index(index_key: bytes, scope: str) -> bytes:
        msg = b"scope" + _BI_SEP + scope.encode("utf-8")
        return hmac.new(index_key, msg, hashlib.sha256).digest()

    # -- AEAD payload helpers ---------------------------------------------
    @staticmethod
    def _aad(key_id: str, blind_index: bytes, version: int) -> bytes:
        """Associated data binding a row to its key id, slot, and version.

        ``blind_index`` is the stored primary-key column ``HMAC(scope|token)``,
        so the AAD is fully reconstructable at read time on every path, and a
        payload moved to a different ``(scope, token)`` slot (different blind
        index) cannot authenticate.
        """
        return _AAD_SEP.join(
            (key_id.encode("utf-8"), blind_index, str(version).encode("utf-8"))
        )

    def _encrypt_payload(
        self, dek: bytes, mapping: TokenMapping, key_id: str, blind_index: bytes
    ) -> bytes:
        """JSON-serialize an explicit field dict and AEAD-encrypt it.

        Returns ``nonce | ciphertext | tag``. The payload is a small explicit
        dict round-tripped via :func:`json.dumps` / :func:`json.loads` only —
        never an arbitrary-object loader.
        """
        plain = json.dumps(
            {
                "scope": mapping.scope,
                "token": mapping.token,
                "plaintext": mapping.plaintext,
                "entity_type": mapping.entity_type,
                "version": mapping.version,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        aead = _new_aead(self._algorithm, dek)
        nonce = os.urandom(_NONCE_BYTES)
        aad = self._aad(key_id, blind_index, mapping.version)
        ct: bytes = aead.encrypt(nonce, plain, aad)
        return nonce + ct

    def _decrypt_payload(
        self, dek: bytes, payload: bytes, key_id: str, blind_index: bytes, version: int
    ) -> dict[str, Any]:
        """Verify + decrypt a row payload; raise loudly on any tamper.

        The AAD (``key_id | blind_index | version``) is reconstructed from the
        row's stored coordinates, so a payload relocated to a different slot
        fails the tag check and raises.
        """
        aead = _new_aead(self._algorithm, dek)
        nonce, ct = payload[:_NONCE_BYTES], payload[_NONCE_BYTES:]
        aad = self._aad(key_id, blind_index, version)
        try:
            plain = aead.decrypt(nonce, ct, aad)
        except InvalidTag as exc:
            raise TokenStoreIntegrityError(
                f"Integrity check failed for a row under key_id={key_id!r}: "
                "ciphertext, nonce, tag, or bound row context (slot/version) "
                "has been tampered with — refusing to return a "
                "successful-but-wrong reversal."
            ) from exc
        decoded: Any = json.loads(plain.decode("utf-8"))
        if not isinstance(decoded, dict):  # pragma: no cover - defensive
            raise TokenStoreIntegrityError(
                f"Decrypted payload for key_id={key_id!r} was not an object."
            )
        return decoded

    @staticmethod
    def _fields_to_mapping(
        fields: dict[str, Any], created_at: float, expires_at: float | None
    ) -> TokenMapping:
        return TokenMapping(
            scope=str(fields["scope"]),
            token=str(fields["token"]),
            plaintext=str(fields["plaintext"]),
            entity_type=str(fields["entity_type"]),
            version=int(fields["version"]),
            created_at=created_at,
            expires_at=expires_at,
        )

    # -- row encode / write (shared by put + rewrap_all) -------------------
    def _encrypt_row(
        self, dek: bytes, mapping: TokenMapping, key_id: str
    ) -> tuple[bytes, bytes, str, int, bytes, float, float | None]:
        """Build the full encrypted row tuple for ``mapping`` under ``key_id``.

        Single source of truth for the (blind_index, scope_index, key_id,
        version, payload, created_at, expires_at) column ordering shared by
        :meth:`put` and :meth:`rewrap_all`.
        """
        index_key = self._index_key_for(dek)
        bi = self._blind_index(index_key, mapping.scope, mapping.token)
        si = self._scope_index(index_key, mapping.scope)
        payload = self._encrypt_payload(dek, mapping, key_id, bi)
        return (bi, si, key_id, mapping.version, payload, mapping.created_at, mapping.expires_at)

    def _write_row(
        self, row: tuple[bytes, bytes, str, int, bytes, float, float | None]
    ) -> None:
        """Persist one prebuilt encrypted row (``INSERT OR REPLACE``)."""
        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self._TABLE}
            (blind_index, scope_index, key_id, version, payload,
             created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )

    # -- TokenStore surface ------------------------------------------------
    def put(self, mapping: TokenMapping) -> None:
        key_id = self._provider.current_key_id()
        with self._lock:
            dek = self._dek_for(key_id)
            row = self._encrypt_row(dek, mapping, key_id)
            with self._conn:
                self._write_row(row)

    def get(self, token: str, *, scope: str | None = None) -> TokenMapping | None:
        now = time.time()
        with self._lock:
            if scope is not None:
                result = self._get_scoped(token, scope, now)
            else:
                result = self._get_scan(token, now)
            if result is not None:
                return result
            # No row decrypted to the requested token. Distinguish a genuinely
            # absent token (return None, per A9) from a wrong-key situation:
            # if any key id that OWNS rows cannot be unwrapped with the supplied
            # KEK, the artifact cannot be reversed and we must FAIL LOUD
            # (NFR-015 — artifact-alone / wrong-secret re-join = FAIL).
            self._raise_if_unreversible()
            return None

    def _get_scoped(self, token: str, scope: str, now: float) -> TokenMapping | None:
        # A (scope, token) may have been written under any live key id (after
        # rotation), so probe each live key's blind index.
        for key_id, dek in self._iter_live_keys():
            index_key = self._index_key_for(dek)
            bi = self._blind_index(index_key, scope, token)
            row = self._conn.execute(
                f"SELECT key_id, version, payload, created_at, expires_at "
                f"FROM {self._TABLE} WHERE blind_index = ? AND key_id = ?",
                (bi, key_id),
            ).fetchone()
            if row is None:
                continue
            expires = row[4]
            if expires is not None and now > float(expires):
                return None
            fields = self._decrypt_payload(dek, bytes(row[2]), key_id, bi, int(row[1]))
            return self._fields_to_mapping(
                fields,
                float(row[3]) if row[3] else 0.0,
                float(expires) if expires is not None else None,
            )
        return None

    def _get_scan(self, token: str, now: float) -> TokenMapping | None:
        # Scope-less lookup mirrors SQLiteTokenStore's token-only scan: walk the
        # rows, decrypt each (the blind index is stored, so AAD is known), and
        # return the first whose decrypted token matches.
        rows = self._conn.execute(
            f"SELECT blind_index, key_id, version, payload, created_at, expires_at "
            f"FROM {self._TABLE}"
        ).fetchall()
        for row in rows:
            expires = row[5]
            if expires is not None and now > float(expires):
                continue
            key_id = str(row[1])
            try:
                dek = self._dek_for(key_id)
            except TokenStoreIntegrityError:
                continue
            fields = self._decrypt_payload(
                dek, bytes(row[3]), key_id, bytes(row[0]), int(row[2])
            )
            if str(fields.get("token")) != token:
                continue
            return self._fields_to_mapping(
                fields,
                float(row[4]) if row[4] else 0.0,
                float(expires) if expires is not None else None,
            )
        return None

    def _raise_if_unreversible(self) -> None:
        """Raise if a key id that owns persisted rows cannot be unwrapped.

        This is the wrong-KEK / wrong-external-secret signal (A4, NFR-015):
        reopening the artifact under a KEK that cannot unwrap the envelope of a
        key id that actually owns rows must fail loudly rather than silently
        report the token as absent. A token that is simply not present (all
        owning envelopes unwrap fine) returns ``None`` instead (A9).
        """
        owning_key_ids = {
            str(r[0])
            for r in self._conn.execute(
                f"SELECT DISTINCT key_id FROM {self._TABLE}"
            ).fetchall()
        }
        for key_id in owning_key_ids:
            try:
                self._dek_for(key_id)
            except TokenStoreIntegrityError as exc:
                raise TokenStoreIntegrityError(
                    f"Cannot reverse the token vault: key envelope for "
                    f"key_id={key_id!r} could not be unwrapped with the "
                    "supplied key-encryption key (wrong external secret)."
                ) from exc

    def list_by_scope(self, scope: str) -> list[TokenMapping]:
        now = time.time()
        results: list[TokenMapping] = []
        with self._lock:
            for key_id, dek in self._iter_live_keys():
                index_key = self._index_key_for(dek)
                si = self._scope_index(index_key, scope)
                rows = self._conn.execute(
                    f"SELECT blind_index, version, payload, created_at, expires_at "
                    f"FROM {self._TABLE} WHERE scope_index = ? AND key_id = ?",
                    (si, key_id),
                ).fetchall()
                for row in rows:
                    expires = row[4]
                    if expires is not None and now > float(expires):
                        continue
                    fields = self._decrypt_payload(
                        dek, bytes(row[2]), key_id, bytes(row[0]), int(row[1])
                    )
                    results.append(
                        self._fields_to_mapping(
                            fields,
                            float(row[3]) if row[3] else 0.0,
                            float(expires) if expires is not None else None,
                        )
                    )
        return results

    def count(self, scope: str | None = None) -> int:
        now = time.time()
        with self._lock:
            if scope is None:
                row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {self._TABLE} "
                    f"WHERE expires_at IS NULL OR expires_at > ?",
                    (now,),
                ).fetchone()
                return int(row[0]) if row else 0
            total = 0
            for key_id, dek in self._iter_live_keys():
                index_key = self._index_key_for(dek)
                si = self._scope_index(index_key, scope)
                row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {self._TABLE} "
                    f"WHERE scope_index = ? AND key_id = ? "
                    f"AND (expires_at IS NULL OR expires_at > ?)",
                    (si, key_id, now),
                ).fetchone()
                total += int(row[0]) if row else 0
            return total

    def delete_expired(self) -> int:
        now = time.time()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM {self._TABLE} "
                f"WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            return int(cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- rotation / audit --------------------------------------------------
    def rotate(self, new_provider: EnvelopeKeyProvider) -> str:
        """Rotate to a new KEK custodian + a fresh DEK.

        Existing rows keep their stamped ``key_id`` (and still decrypt under
        their original DEK); new ``put`` calls encrypt under the new key id.
        Returns the new key id. Mirrors ``KeyManager``'s dual-version grace
        pattern: old rows remain reversible until explicitly rewrapped.
        """
        with self._lock:
            self._provider = new_provider
        self._ensure_current_envelope()
        return new_provider.current_key_id()

    def rewrap_all(self, new_provider: EnvelopeKeyProvider) -> int:
        """Re-encrypt every row under a fresh DEK wrapped by ``new_provider``.

        Returns the number of rows migrated. After this call the only key id in
        active use for the migrated rows is ``new_provider.current_key_id()``;
        the file can no longer be reversed under the old KEK.
        """
        # 1. Decrypt every existing row under its current key id (snapshot).
        with self._lock:
            snapshot = self._decrypt_all_rows()

        # 2. Swap provider + mint the new envelope/DEK.
        with self._lock:
            self._provider = new_provider
        self._ensure_current_envelope()
        new_key_id = new_provider.current_key_id()

        # 3. Atomically clear + re-encrypt every mapping under the new DEK,
        #    reusing the same row-encode / row-write helpers as put().
        with self._lock:
            new_dek = self._dek_for(new_key_id)
            with self._conn:
                self._conn.execute(f"DELETE FROM {self._TABLE}")
                for mapping in snapshot:
                    self._write_row(self._encrypt_row(new_dek, mapping, new_key_id))
        return len(snapshot)

    def list_key_ids(self) -> list[str]:
        """Return all key ids with a persisted envelope (audit enumeration)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key_id FROM {self._KEYS_TABLE} ORDER BY created_at"
            ).fetchall()
        return [str(r[0]) for r in rows]

    def list_key_envelopes(self) -> list[KeyEnvelope]:
        """Return all persisted key envelopes (wrapped DEKs only — audit)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key_id, wrapped_dek, created_at FROM {self._KEYS_TABLE}"
            ).fetchall()
        return [
            KeyEnvelope(key_id=str(r[0]), wrapped_dek=bytes(r[1]), created_at=float(r[2]))
            for r in rows
        ]

    def row_key_id(self, token: str, *, scope: str) -> str | None:
        """Return the ``key_id`` stamped on the row for ``(scope, token)``."""
        with self._lock:
            for key_id, dek in self._iter_live_keys():
                index_key = self._index_key_for(dek)
                bi = self._blind_index(index_key, scope, token)
                row = self._conn.execute(
                    f"SELECT key_id FROM {self._TABLE} "
                    f"WHERE blind_index = ? AND key_id = ?",
                    (bi, key_id),
                ).fetchone()
                if row is not None:
                    return key_id
        return None

    # -- internal helpers --------------------------------------------------
    def _iter_live_keys(self) -> list[tuple[str, bytes]]:
        """Return ``(key_id, dek)`` for every envelope this provider can unwrap.

        Rows written under a key id the current provider cannot unwrap are not
        decryptable; lookups under them return ``None`` (scoped) — the NFR-015
        wrong-key behaviour. A direct read of such a row raises via
        :meth:`_decrypt_payload`.
        """
        result: list[tuple[str, bytes]] = []
        rows = self._conn.execute(
            f"SELECT key_id FROM {self._KEYS_TABLE} ORDER BY created_at"
        ).fetchall()
        for r in rows:
            key_id = str(r[0])
            try:
                result.append((key_id, self._dek_for(key_id)))
            except TokenStoreIntegrityError:
                continue
        return result

    def _decrypt_all_rows(self) -> list[TokenMapping]:
        """Decrypt every stored row into TokenMappings (rewrap snapshot)."""
        rows = self._conn.execute(
            f"SELECT blind_index, key_id, version, payload, created_at, expires_at "
            f"FROM {self._TABLE}"
        ).fetchall()
        out: list[TokenMapping] = []
        for row in rows:
            key_id = str(row[1])
            dek = self._dek_for(key_id)
            fields = self._decrypt_payload(
                dek, bytes(row[3]), key_id, bytes(row[0]), int(row[2])
            )
            out.append(
                self._fields_to_mapping(
                    fields,
                    float(row[4]) if row[4] else 0.0,
                    float(row[5]) if row[5] is not None else None,
                )
            )
        return out

    # -- test-only accessor (AGENT_SIMULATED introspection) ---------------
    def _current_dek_bytes_for_test(self) -> bytes:
        """Return the live DEK bytes (TEST ONLY — never used in production).

        Used by the A2/A5 not-on-disk assertions to prove the bare DEK is never
        written to the SQLite file. Not part of the public ``TokenStore`` API.
        """
        return self._dek_for(self._provider.current_key_id())
