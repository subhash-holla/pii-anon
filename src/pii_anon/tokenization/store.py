"""Token storage backends for pseudonymization.

Provides in-memory and SQLite-backed stores for persisting token-to-plaintext
mappings.  Supports TTL-based expiration, scope enumeration, and bulk
operations for enterprise workflows.

Usage::

    store = SQLiteTokenStore("tokens.db")
    store.put(TokenMapping(scope="s", token="<T>", plaintext="val",
                           entity_type="EMAIL", version=1))
    found = store.get("<T>", scope="s")
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass
class TokenMapping:
    """A single token-to-plaintext mapping.

    Attributes
    ----------
    scope : str
        Tokenization scope (e.g., user ID, document ID).
    token : str
        The pseudonymization token string.
    plaintext : str
        The original plaintext value.
    entity_type : str
        Entity type (e.g., ``"EMAIL_ADDRESS"``).
    version : int
        Key version used for tokenization.
    created_at : float
        Unix timestamp when this mapping was created.
    expires_at : float | None
        Unix timestamp when this mapping expires, or ``None`` for no expiry.
    """

    scope: str
    token: str
    plaintext: str
    entity_type: str
    version: int
    created_at: float = 0.0
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()


class TokenStore:
    """Abstract base for token storage backends."""

    def put(self, mapping: TokenMapping) -> None:
        """Store a token mapping."""
        raise NotImplementedError

    def get(self, token: str, *, scope: str | None = None) -> TokenMapping | None:
        """Retrieve a mapping by token (optionally scoped)."""
        raise NotImplementedError

    def list_by_scope(self, scope: str) -> list[TokenMapping]:
        """Return all mappings for a given scope."""
        raise NotImplementedError

    def count(self, scope: str | None = None) -> int:
        """Count stored mappings, optionally filtered by scope."""
        raise NotImplementedError

    def delete_expired(self) -> int:
        """Remove all expired mappings.  Returns the number removed."""
        raise NotImplementedError

    def close(self) -> None:
        """Close any underlying resources."""
        return None


class InMemoryTokenStore(TokenStore):
    """Thread-safe in-memory token store.

    Suitable for testing, short-lived pipelines, and single-process
    deployments.  All data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], TokenMapping] = {}
        self._lock = Lock()

    def put(self, mapping: TokenMapping) -> None:
        with self._lock:
            self._records[(mapping.scope, mapping.token)] = mapping

    def get(self, token: str, *, scope: str | None = None) -> TokenMapping | None:
        now = time.time()
        with self._lock:
            if scope is not None:
                m = self._records.get((scope, token))
                if m is not None:
                    if m.expires_at is not None and now > m.expires_at:
                        return None
                    return m
                return None
            # Scan all scopes for this token
            for (record_scope, record_token), mapping in self._records.items():
                if record_token == token:
                    if mapping.expires_at is not None and now > mapping.expires_at:
                        continue
                    return mapping
        return None

    def list_by_scope(self, scope: str) -> list[TokenMapping]:
        now = time.time()
        with self._lock:
            return [
                m for (s, _), m in self._records.items()
                if s == scope and (m.expires_at is None or now <= m.expires_at)
            ]

    def count(self, scope: str | None = None) -> int:
        now = time.time()
        with self._lock:
            if scope is not None:
                return sum(
                    1 for (s, _), m in self._records.items()
                    if s == scope and (m.expires_at is None or now <= m.expires_at)
                )
            return sum(
                1 for m in self._records.values()
                if m.expires_at is None or now <= m.expires_at
            )

    def delete_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, m in self._records.items()
                if m.expires_at is not None and now > m.expires_at
            ]
            for k in expired_keys:
                del self._records[k]
            return len(expired_keys)


class SQLiteTokenStore(TokenStore):
    """SQLite-backed persistent token store.

    Stores mappings in a local SQLite database with support for TTL,
    scope enumeration, and expiration cleanup.

    Parameters
    ----------
    db_path : str | Path
        Database file path.  Use ``":memory:"`` for in-memory SQLite.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = Lock()
        self._init_table()

    def _init_table(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_mappings (
                    scope TEXT NOT NULL,
                    token TEXT NOT NULL,
                    plaintext TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0.0,
                    expires_at REAL,
                    PRIMARY KEY (scope, token)
                )
                """
            )
            # Index for expiration cleanup
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_mappings_expires
                ON token_mappings(expires_at)
                WHERE expires_at IS NOT NULL
                """
            )
            # Index for scope enumeration
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_mappings_scope
                ON token_mappings(scope)
                """
            )
            # Migrate existing tables that lack the new columns
            self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add created_at/expires_at columns if missing (backward compat)."""
        try:
            self._conn.execute("SELECT created_at FROM token_mappings LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute(
                "ALTER TABLE token_mappings ADD COLUMN created_at REAL NOT NULL DEFAULT 0.0"
            )
        try:
            self._conn.execute("SELECT expires_at FROM token_mappings LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute(
                "ALTER TABLE token_mappings ADD COLUMN expires_at REAL"
            )

    def put(self, mapping: TokenMapping) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO token_mappings(
                    scope, token, plaintext, entity_type, version, created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mapping.scope, mapping.token, mapping.plaintext,
                    mapping.entity_type, mapping.version,
                    mapping.created_at, mapping.expires_at,
                ),
            )

    def get(self, token: str, *, scope: str | None = None) -> TokenMapping | None:
        now = time.time()
        query = (
            "SELECT scope, token, plaintext, entity_type, version, created_at, expires_at "
            "FROM token_mappings WHERE token = ?"
        )
        params: tuple[object, ...] = (token,)
        if scope is not None:
            query += " AND scope = ?"
            params = (token, scope)

        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        if row is None:
            return None

        expires = row[6]
        if expires is not None and now > expires:
            return None

        return TokenMapping(
            scope=str(row[0]),
            token=str(row[1]),
            plaintext=str(row[2]),
            entity_type=str(row[3]),
            version=int(row[4]),
            created_at=float(row[5]) if row[5] else 0.0,
            expires_at=float(row[6]) if row[6] is not None else None,
        )

    def list_by_scope(self, scope: str) -> list[TokenMapping]:
        now = time.time()
        query = (
            "SELECT scope, token, plaintext, entity_type, version, created_at, expires_at "
            "FROM token_mappings WHERE scope = ? "
            "AND (expires_at IS NULL OR expires_at > ?)"
        )
        with self._lock:
            rows = self._conn.execute(query, (scope, now)).fetchall()

        return [
            TokenMapping(
                scope=str(r[0]),
                token=str(r[1]),
                plaintext=str(r[2]),
                entity_type=str(r[3]),
                version=int(r[4]),
                created_at=float(r[5]) if r[5] else 0.0,
                expires_at=float(r[6]) if r[6] is not None else None,
            )
            for r in rows
        ]

    def count(self, scope: str | None = None) -> int:
        now = time.time()
        if scope is not None:
            query = (
                "SELECT COUNT(*) FROM token_mappings "
                "WHERE scope = ? AND (expires_at IS NULL OR expires_at > ?)"
            )
            params: tuple[object, ...] = (scope, now)
        else:
            query = (
                "SELECT COUNT(*) FROM token_mappings "
                "WHERE expires_at IS NULL OR expires_at > ?"
            )
            params = (now,)

        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return int(row[0]) if row else 0

    def delete_expired(self) -> int:
        now = time.time()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM token_mappings WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
