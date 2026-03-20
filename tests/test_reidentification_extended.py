"""Extended tests for ReidentificationService to improve code coverage."""

from __future__ import annotations

import time
import threading

import pytest

from pii_anon.tokenization.reidentification import (
    ReidentificationService,
    ReidentificationAuditEntry,
)
from pii_anon.tokenization.store import InMemoryTokenStore, TokenMapping
from pii_anon.tokenization.providers import (
    DeterministicHMACTokenizer,
    TokenRecord,
)
from pii_anon.tokenization.key_manager import KeyManager


class TestResolveKeyWithKeyManager:
    """Test _resolve_key method with key_manager."""

    def test_resolve_key_with_key_manager_success(self):
        """_resolve_key returns key when key_manager.get_key succeeds."""
        km = KeyManager("test-secret")
        store = InMemoryTokenStore()
        svc = ReidentificationService(store, key_manager=km)

        # _resolve_key should call key_manager.get_key
        key = svc._resolve_key(1)
        assert key == "test-secret"

    def test_resolve_key_with_key_manager_returns_none(self):
        """_resolve_key returns None when key_manager.get_key returns None."""
        class MockKeyManager:
            def get_key(self, version):
                return None

        store = InMemoryTokenStore()
        svc = ReidentificationService(store, key_manager=MockKeyManager())

        key = svc._resolve_key(1)
        assert key is None

    def test_resolve_key_with_key_manager_key_error(self):
        """_resolve_key returns None when key_manager raises KeyError."""
        class MockKeyManager:
            def get_key(self, version):
                raise KeyError(f"Version {version} not found")

        store = InMemoryTokenStore()
        svc = ReidentificationService(store, key_manager=MockKeyManager())

        key = svc._resolve_key(99)
        assert key is None

    def test_resolve_key_with_key_manager_value_error(self):
        """_resolve_key returns None when key_manager raises ValueError."""
        class MockKeyManager:
            def get_key(self, version):
                raise ValueError("Invalid version")

        store = InMemoryTokenStore()
        svc = ReidentificationService(store, key_manager=MockKeyManager())

        key = svc._resolve_key(1)
        assert key is None

    def test_resolve_key_with_key_manager_converts_to_string(self):
        """_resolve_key converts non-string results to string."""
        class MockKeyManager:
            def get_key(self, version):
                return 12345  # Non-string

        store = InMemoryTokenStore()
        svc = ReidentificationService(store, key_manager=MockKeyManager())

        key = svc._resolve_key(1)
        assert key == "12345"
        assert isinstance(key, str)

    def test_resolve_key_without_key_manager_returns_default(self):
        """_resolve_key returns default_key when key_manager is None."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store, default_key="fallback-secret")

        key = svc._resolve_key(1)
        assert key == "fallback-secret"

    def test_resolve_key_without_key_manager_no_default(self):
        """_resolve_key returns None when no key_manager and no default."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        key = svc._resolve_key(1)
        assert key is None


class TestCryptographicDetokenization:
    """Test cryptographic detokenization path in detokenize_single."""

    def test_detokenize_single_with_tokenizer_success(self):
        """detokenize_single uses tokenizer when store lookup fails."""
        store = InMemoryTokenStore()
        km = KeyManager("test-secret")

        class MockTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                if token_record.entity_type == "EMAIL":
                    return "john@example.com"
                return None

        svc = ReidentificationService(
            store,
            key_manager=km,
            tokenizer=MockTokenizer(),
        )

        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc>", scope="test", reason="testing"
        )
        assert result == "john@example.com"

        # Should record audit entry with success=True
        log = svc.audit_log
        assert len(log) == 1
        assert log[0].success is True

    def test_detokenize_single_with_tokenizer_no_key(self):
        """detokenize_single skips tokenizer when key resolution fails."""
        store = InMemoryTokenStore()

        class BadKeyManager:
            def get_key(self, version):
                raise KeyError("No key")

        class MockTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                return "should_not_be_called"

        svc = ReidentificationService(
            store,
            key_manager=BadKeyManager(),
            tokenizer=MockTokenizer(),
        )

        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc>", scope="test", reason="testing"
        )
        assert result is None

        # Audit should record failure
        log = svc.audit_log
        assert log[0].success is False

    def test_detokenize_single_with_tokenizer_exception(self):
        """detokenize_single handles tokenizer exceptions gracefully."""
        store = InMemoryTokenStore()
        km = KeyManager("test-secret")

        class BadTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                raise RuntimeError("Tokenizer failed")

        svc = ReidentificationService(
            store,
            key_manager=km,
            tokenizer=BadTokenizer(),
        )

        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc>", scope="test", reason="testing"
        )
        assert result is None

        # Audit should record failure
        log = svc.audit_log
        assert log[0].success is False

    def test_detokenize_single_with_tokenizer_returns_none(self):
        """detokenize_single handles tokenizer returning None."""
        store = InMemoryTokenStore()
        km = KeyManager("test-secret")

        class NoneTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                return None

        svc = ReidentificationService(
            store,
            key_manager=km,
            tokenizer=NoneTokenizer(),
        )

        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc>", scope="test", reason="testing"
        )
        assert result is None

        log = svc.audit_log
        assert log[0].success is False

    def test_detokenize_single_without_tokenizer(self):
        """detokenize_single skips cryptographic path when tokenizer is None."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store, tokenizer=None)

        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc>", scope="test", reason="testing"
        )
        assert result is None

        log = svc.audit_log
        assert log[0].success is False

    def test_detokenize_single_without_store_match(self):
        """detokenize_single tries tokenizer when store doesn't have token."""
        store = InMemoryTokenStore()
        km = KeyManager("test-secret")

        class MockTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                return "recovered"

        svc = ReidentificationService(
            store,
            key_manager=km,
            tokenizer=MockTokenizer(),
        )

        result = svc.detokenize_single(
            "<EMAIL:v1:tok_missing>", scope="test"
        )
        assert result == "recovered"

    def test_detokenize_single_malformed_token_skips_tokenizer(self):
        """detokenize_single skips tokenizer for malformed tokens."""
        store = InMemoryTokenStore()
        km = KeyManager("test-secret")

        class MockTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                return "should_not_be_called"

        svc = ReidentificationService(
            store,
            key_manager=km,
            tokenizer=MockTokenizer(),
        )

        # Malformed token (missing closing >)
        result = svc.detokenize_single(
            "EMAIL:v1:tok_abc", scope="test"
        )
        assert result is None

        log = svc.audit_log
        assert log[0].success is False
        # Should not have entity_type/version for malformed tokens
        assert log[0].entity_type == ""
        assert log[0].version == 0


class TestTokenRecordCreation:
    """Test TokenRecord creation in detokenize_single."""

    def test_detokenize_single_creates_token_record_correctly(self):
        """detokenize_single creates TokenRecord with correct attributes."""
        store = InMemoryTokenStore()
        km = KeyManager("test-secret")

        created_records = []

        class InspectingTokenizer:
            def detokenize(self, token_record, key=None, store=None):
                created_records.append(token_record)
                return "result"

        svc = ReidentificationService(
            store,
            key_manager=km,
            tokenizer=InspectingTokenizer(),
        )

        # Use a token not in the store so it falls through to tokenizer
        # Use version 1 since that's the default in KeyManager
        result = svc.detokenize_single("<PHONE:v1:tok_xyz>", scope="test_scope")

        # Should have called tokenizer
        assert len(created_records) == 1
        rec = created_records[0]
        assert rec.entity_type == "PHONE"
        assert rec.version == 1
        assert rec.token == "<PHONE:v1:tok_xyz>"
        assert rec.scope == "test_scope"
        assert result == "result"


class TestAuditThreadSafety:
    """Test thread-safe audit logging."""

    def test_audit_log_concurrent_writes(self):
        """Audit log is thread-safe with concurrent writes."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        def record_entries(thread_id):
            for i in range(10):
                svc.detokenize_single(
                    f"<EMAIL:v1:tok_{thread_id}_{i}>",
                    scope="test",
                    reason=f"thread_{thread_id}",
                )

        threads = [threading.Thread(target=record_entries, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        log = svc.audit_log
        assert len(log) == 50  # 5 threads * 10 entries

    def test_clear_audit_log_thread_safe(self):
        """Clearing audit log is thread-safe."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        # Add some entries
        for i in range(10):
            svc.detokenize_single(f"<EMAIL:v1:tok_{i}>", scope="test")

        # Clear should be thread-safe
        svc.clear_audit_log()
        assert len(svc.audit_log) == 0

        # Should be able to add more after clearing
        svc.detokenize_single("<EMAIL:v1:tok_new>", scope="test")
        assert len(svc.audit_log) == 1


class TestDetokenizePayloadEdgeCases:
    """Test edge cases in detokenize_payload."""

    def test_detokenize_payload_with_lists(self):
        """detokenize_payload preserves lists in values."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        payload = {
            "emails": ["<EMAIL:v1:tok_abc>", "other"],
            "name": "John",
        }

        result = svc.detokenize_payload(payload, scope="test")
        # Lists should be preserved (not detokenized)
        assert result["emails"] == ["<EMAIL:v1:tok_abc>", "other"]
        assert result["name"] == "John"

    def test_detokenize_payload_deeply_nested(self):
        """detokenize_payload handles deeply nested dicts."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<EMAIL:v1:tok_deep>",
                plaintext="deep@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        payload = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "email": "<EMAIL:v1:tok_deep>"
                        }
                    }
                }
            }
        }

        result = svc.detokenize_payload(payload, scope="test")
        assert result["level1"]["level2"]["level3"]["level4"]["email"] == "deep@example.com"

    def test_detokenize_payload_with_empty_dict(self):
        """detokenize_payload handles empty nested dicts."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        payload = {"empty": {}, "nested": {"also_empty": {}}}
        result = svc.detokenize_payload(payload, scope="test")

        assert result == payload

    def test_detokenize_payload_with_mixed_types(self):
        """detokenize_payload preserves non-string/non-dict types."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        payload = {
            "email": "<EMAIL:v1:tok_abc>",
            "count": 42,
            "active": True,
            "score": 3.14,
            "empty": None,
            "list": [1, 2, 3],
            "bytes": b"data",
        }

        result = svc.detokenize_payload(payload, scope="test")
        assert result["email"] == "john@example.com"
        assert result["count"] == 42
        assert result["active"] is True
        assert result["score"] == 3.14
        assert result["empty"] is None
        assert result["list"] == [1, 2, 3]
        assert result["bytes"] == b"data"


class TestDetokenizeTextPatternMatching:
    """Test pattern matching in detokenize_text."""

    def test_detokenize_text_aes_token_format(self):
        """detokenize_text matches aes_XXX token format."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<PII:v1:aes_xyz123>",
                plaintext="secret",
                entity_type="PII",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_text(
            "Value: <PII:v1:aes_xyz123>",
            scope="test",
        )
        assert result == "Value: secret"

    def test_detokenize_text_multiple_versions(self):
        """detokenize_text handles tokens with different versions."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<EMAIL:v1:tok_v1>",
                plaintext="v1@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )
        store.put(
            TokenMapping(
                scope="test",
                token="<EMAIL:v2:tok_v2>",
                plaintext="v2@example.com",
                entity_type="EMAIL",
                version=2,
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_text(
            "<EMAIL:v1:tok_v1> and <EMAIL:v2:tok_v2>",
            scope="test",
        )
        assert result == "v1@example.com and v2@example.com"

    def test_detokenize_text_uppercase_entity_types(self):
        """detokenize_text matches only uppercase entity type patterns."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        # Pattern requires uppercase
        text = "Email: <email:v1:tok_abc> and <EMAIL:v1:tok_def>"
        result = svc.detokenize_text(text, scope="test")

        # Should only match EMAIL, not email
        assert "<email:v1:tok_abc>" in result
        assert "<EMAIL:v1:tok_def>" in result

    def test_detokenize_text_underscore_in_entity_type(self):
        """detokenize_text matches entity types with underscores."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<CREDIT_CARD:v1:tok_abc>",
                plaintext="1234",
                entity_type="CREDIT_CARD",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_text(
            "Card: <CREDIT_CARD:v1:tok_abc>",
            scope="test",
        )
        assert result == "Card: 1234"

    def test_detokenize_text_special_token_characters(self):
        """detokenize_text matches token IDs with hyphens."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<EMAIL:v1:tok_abc-def-123>",
                plaintext="test@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_text(
            "Email: <EMAIL:v1:tok_abc-def-123>",
            scope="test",
        )
        assert result == "Email: test@example.com"


class TestAuditEntryCreation:
    """Test audit entry creation and attributes."""

    def test_audit_entry_has_timestamp(self):
        """Audit entries have timestamps in audit log."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        before = time.time()
        svc.detokenize_single("<EMAIL:v1:tok_abc>", scope="test")
        after = time.time()

        log = svc.audit_log
        assert len(log) == 1
        assert before <= log[0].timestamp <= after

    def test_audit_entry_preserves_entity_info(self):
        """Audit entries preserve entity info from token."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test",
                token="<PHONE:v5:tok_xyz>",
                plaintext="555-1234",
                entity_type="PHONE",
                version=5,
            )
        )

        svc = ReidentificationService(store)
        svc.detokenize_single(
            "<PHONE:v5:tok_xyz>",
            scope="test",
            reason="legal discovery",
        )

        log = svc.audit_log
        entry = log[0]
        assert entry.entity_type == "PHONE"
        assert entry.version == 5
        assert entry.reason == "legal discovery"
        assert entry.scope == "test"
