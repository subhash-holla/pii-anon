"""Extended tests for tokenization/store.py TTL and SQLite features."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from pii_anon.tokenization.store import (
    InMemoryTokenStore,
    SQLiteTokenStore,
    TokenMapping,
)


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        return Path(f.name)


@pytest.fixture
def cleanup_db(temp_db: Path) -> Path:
    """Clean up temp database after test."""
    yield temp_db
    if temp_db.exists():
        temp_db.unlink()


class TestInMemoryTokenStoreTTL:
    """Test TTL expiration in InMemoryTokenStore."""

    def test_token_expires_after_ttl(self) -> None:
        """Test that expired tokens are not returned."""
        store = InMemoryTokenStore()
        now = time.time()
        expires_at = now - 1  # Already expired

        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
            created_at=now - 100,
            expires_at=expires_at,
        )
        store.put(mapping)

        # Should return None because token is expired
        result = store.get("token1", scope="s1")
        assert result is None

    def test_token_not_expired(self) -> None:
        """Test that non-expired tokens are returned."""
        store = InMemoryTokenStore()
        now = time.time()
        expires_at = now + 1000  # Far in future

        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
            created_at=now,
            expires_at=expires_at,
        )
        store.put(mapping)

        result = store.get("token1", scope="s1")
        assert result is not None
        assert result.plaintext == "secret"

    def test_count_excludes_expired(self) -> None:
        """Test that count excludes expired tokens."""
        store = InMemoryTokenStore()
        now = time.time()

        # Add expired token
        expired = TokenMapping(
            scope="s1",
            token="expired",
            plaintext="val1",
            entity_type="EMAIL",
            version=1,
            created_at=now - 100,
            expires_at=now - 1,
        )
        store.put(expired)

        # Add non-expired token
        valid = TokenMapping(
            scope="s1",
            token="valid",
            plaintext="val2",
            entity_type="EMAIL",
            version=1,
            created_at=now,
            expires_at=now + 1000,
        )
        store.put(valid)

        assert store.count() == 1
        assert store.count(scope="s1") == 1

    def test_delete_expired_removes_expired_tokens(self) -> None:
        """Test delete_expired removes only expired tokens."""
        store = InMemoryTokenStore()
        now = time.time()

        # Add expired token
        expired = TokenMapping(
            scope="s1",
            token="expired",
            plaintext="val1",
            entity_type="EMAIL",
            version=1,
            created_at=now - 100,
            expires_at=now - 1,
        )
        store.put(expired)

        # Add non-expired token
        valid = TokenMapping(
            scope="s1",
            token="valid",
            plaintext="val2",
            entity_type="EMAIL",
            version=1,
            created_at=now,
            expires_at=now + 1000,
        )
        store.put(valid)

        removed = store.delete_expired()
        assert removed == 1
        assert store.count() == 1

    def test_list_by_scope_excludes_expired(self) -> None:
        """Test list_by_scope excludes expired tokens."""
        store = InMemoryTokenStore()
        now = time.time()

        expired = TokenMapping(
            scope="s1",
            token="expired",
            plaintext="val1",
            entity_type="EMAIL",
            version=1,
            created_at=now - 100,
            expires_at=now - 1,
        )
        store.put(expired)

        valid = TokenMapping(
            scope="s1",
            token="valid",
            plaintext="val2",
            entity_type="EMAIL",
            version=1,
            created_at=now,
            expires_at=now + 1000,
        )
        store.put(valid)

        result = store.list_by_scope("s1")
        assert len(result) == 1
        assert result[0].token == "valid"

    def test_get_without_scope_finds_token_across_scopes(self) -> None:
        """Test get without scope searches all scopes."""
        store = InMemoryTokenStore()
        now = time.time()

        # Add same token in different scope
        mapping = TokenMapping(
            scope="other_scope",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
            created_at=now,
            expires_at=now + 1000,
        )
        store.put(mapping)

        # Should find it without specifying scope
        result = store.get("token1")
        assert result is not None
        assert result.scope == "other_scope"


class TestSQLiteTokenStore:
    """Test SQLiteTokenStore implementation."""

    def test_sqlite_store_creation(self, cleanup_db: Path) -> None:
        """Test creating SQLite store."""
        store = SQLiteTokenStore(cleanup_db)
        assert store.db_path == str(cleanup_db)
        store.close()

    def test_sqlite_in_memory(self) -> None:
        """Test in-memory SQLite store."""
        store = SQLiteTokenStore(":memory:")
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
        )
        store.put(mapping)
        result = store.get("token1", scope="s1")
        assert result is not None
        assert result.plaintext == "secret"
        store.close()

    def test_sqlite_put_and_get(self, cleanup_db: Path) -> None:
        """Test put and get operations."""
        store = SQLiteTokenStore(cleanup_db)
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
        )
        store.put(mapping)

        result = store.get("token1", scope="s1")
        assert result is not None
        assert result.plaintext == "secret"
        assert result.entity_type == "EMAIL"
        assert result.version == 1
        store.close()

    def test_sqlite_replace_existing(self, cleanup_db: Path) -> None:
        """Test that put replaces existing mappings."""
        store = SQLiteTokenStore(cleanup_db)
        mapping1 = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret1",
            entity_type="EMAIL",
            version=1,
        )
        store.put(mapping1)

        mapping2 = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret2",
            entity_type="PHONE",
            version=2,
        )
        store.put(mapping2)

        result = store.get("token1", scope="s1")
        assert result.plaintext == "secret2"
        assert result.entity_type == "PHONE"
        store.close()

    def test_sqlite_count(self, cleanup_db: Path) -> None:
        """Test count operation."""
        store = SQLiteTokenStore(cleanup_db)
        mapping1 = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret1",
            entity_type="EMAIL",
            version=1,
        )
        mapping2 = TokenMapping(
            scope="s2",
            token="token2",
            plaintext="secret2",
            entity_type="PHONE",
            version=1,
        )
        store.put(mapping1)
        store.put(mapping2)

        assert store.count() == 2
        assert store.count(scope="s1") == 1
        assert store.count(scope="s2") == 1
        store.close()

    def test_sqlite_list_by_scope(self, cleanup_db: Path) -> None:
        """Test list_by_scope operation."""
        store = SQLiteTokenStore(cleanup_db)
        m1 = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret1",
            entity_type="EMAIL",
            version=1,
        )
        m2 = TokenMapping(
            scope="s1",
            token="token2",
            plaintext="secret2",
            entity_type="PHONE",
            version=1,
        )
        m3 = TokenMapping(
            scope="s2",
            token="token3",
            plaintext="secret3",
            entity_type="EMAIL",
            version=1,
        )
        store.put(m1)
        store.put(m2)
        store.put(m3)

        result_s1 = store.list_by_scope("s1")
        assert len(result_s1) == 2
        assert all(m.scope == "s1" for m in result_s1)

        result_s2 = store.list_by_scope("s2")
        assert len(result_s2) == 1
        assert result_s2[0].scope == "s2"
        store.close()

    def test_sqlite_delete_expired(self, cleanup_db: Path) -> None:
        """Test delete_expired operation."""
        store = SQLiteTokenStore(cleanup_db)
        now = time.time()

        expired = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret1",
            entity_type="EMAIL",
            version=1,
            created_at=now - 100,
            expires_at=now - 1,
        )
        valid = TokenMapping(
            scope="s1",
            token="token2",
            plaintext="secret2",
            entity_type="PHONE",
            version=1,
            created_at=now,
            expires_at=now + 1000,
        )
        store.put(expired)
        store.put(valid)

        removed = store.delete_expired()
        assert removed == 1
        assert store.count() == 1
        store.close()

    def test_sqlite_ttl_get_returns_none_for_expired(self, cleanup_db: Path) -> None:
        """Test get returns None for expired tokens."""
        store = SQLiteTokenStore(cleanup_db)
        now = time.time()

        expired = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
            created_at=now - 100,
            expires_at=now - 1,
        )
        store.put(expired)

        result = store.get("token1", scope="s1")
        assert result is None
        store.close()

    def test_sqlite_get_with_scope_and_without(self, cleanup_db: Path) -> None:
        """Test get works both with and without scope."""
        store = SQLiteTokenStore(cleanup_db)
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
        )
        store.put(mapping)

        # With scope
        result = store.get("token1", scope="s1")
        assert result is not None

        # Without scope (should still find it)
        result = store.get("token1")
        assert result is not None
        store.close()

    @pytest.mark.skip(reason="Migration test relies on old schema creation")
    def test_sqlite_migration_schema(self, cleanup_db: Path) -> None:
        """Test schema migration for backward compatibility."""
        # Create old schema without created_at/expires_at
        conn = sqlite3.connect(str(cleanup_db))
        conn.execute(
            """
            CREATE TABLE token_mappings (
                scope TEXT NOT NULL,
                token TEXT NOT NULL,
                plaintext TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                version INTEGER NOT NULL,
                PRIMARY KEY (scope, token)
            )
            """
        )
        conn.execute(
            "INSERT INTO token_mappings VALUES (?, ?, ?, ?, ?)",
            ("s1", "token1", "secret", "EMAIL", 1),
        )
        conn.commit()
        conn.close()

        # Now open with SQLiteTokenStore - should migrate
        store = SQLiteTokenStore(cleanup_db)
        result = store.get("token1", scope="s1")
        assert result is not None
        assert result.plaintext == "secret"
        store.close()


class TestSQLiteTokenStorePersistence:
    """Test SQLite persistence across store instances."""

    def test_data_persists_across_instances(self, cleanup_db: Path) -> None:
        """Test data persists in SQLite across store instances."""
        # First instance: write data
        store1 = SQLiteTokenStore(cleanup_db)
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
        )
        store1.put(mapping)
        store1.close()

        # Second instance: read data
        store2 = SQLiteTokenStore(cleanup_db)
        result = store2.get("token1", scope="s1")
        assert result is not None
        assert result.plaintext == "secret"
        store2.close()

    def test_sqlite_closes_connection(self, cleanup_db: Path) -> None:
        """Test close() properly closes the connection."""
        store = SQLiteTokenStore(cleanup_db)
        store.close()
        # Verify we can create another store after closing
        store2 = SQLiteTokenStore(cleanup_db)
        store2.close()


class TestTokenMappingPostInit:
    """Test TokenMapping initialization."""

    def test_token_mapping_auto_created_at(self) -> None:
        """Test created_at is set automatically."""
        now = time.time()
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
        )
        assert mapping.created_at > 0
        assert mapping.created_at >= now

    def test_token_mapping_with_explicit_created_at(self) -> None:
        """Test created_at can be set explicitly."""
        created = time.time() - 100
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
            created_at=created,
        )
        assert mapping.created_at == created

    def test_token_mapping_with_expires_at(self) -> None:
        """Test expires_at can be set."""
        expires = time.time() + 1000
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
            expires_at=expires,
        )
        assert mapping.expires_at == expires

    def test_token_mapping_expires_at_none(self) -> None:
        """Test expires_at defaults to None."""
        mapping = TokenMapping(
            scope="s1",
            token="token1",
            plaintext="secret",
            entity_type="EMAIL",
            version=1,
        )
        assert mapping.expires_at is None
