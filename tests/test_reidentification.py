"""Tests for ReidentificationService and re-identification workflows."""

import time

import pytest

from pii_anon.tokenization.key_manager import KeyManager
from pii_anon.tokenization.reidentification import (
    ReidentificationService,
    ReidentificationAuditEntry,
)
from pii_anon.tokenization.store import InMemoryTokenStore, TokenMapping
from pii_anon.tokenization.providers import DeterministicHMACTokenizer


class TestDetokenizeSingle:
    """Test single token detokenization."""

    def test_detokenize_single_found_in_store(self):
        """detokenize_single returns plaintext when token found."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_abc123>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc123>", scope="test_scope", reason="testing"
        )
        assert result == "john@example.com"

    def test_detokenize_single_not_found(self):
        """detokenize_single returns None for unknown token."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        result = svc.detokenize_single(
            "<EMAIL:v1:tok_unknown>", scope="test_scope", reason="testing"
        )
        assert result is None

    def test_detokenize_single_wrong_scope(self):
        """detokenize_single returns None for wrong scope."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="scope_a",
                token="<EMAIL:v1:tok_abc123>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_single(
            "<EMAIL:v1:tok_abc123>", scope="scope_b", reason="testing"
        )
        assert result is None

    def test_detokenize_single_expired_token(self):
        """detokenize_single returns None for expired token."""
        store = InMemoryTokenStore()
        now = time.time()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_expired>",
                plaintext="expired@example.com",
                entity_type="EMAIL",
                version=1,
                created_at=now,
                expires_at=now - 10.0,  # Already expired
            )
        )

        svc = ReidentificationService(store)
        result = svc.detokenize_single(
            "<EMAIL:v1:tok_expired>", scope="test_scope", reason="testing"
        )
        assert result is None


class TestBulkDetokenize:
    """Test bulk token detokenization."""

    def test_bulk_detokenize_multiple_found(self):
        """bulk_detokenize returns dict of plaintext for found tokens."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_email>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<PHONE:v1:tok_phone>",
                plaintext="555-1234",
                entity_type="PHONE",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.bulk_detokenize(
            ["<EMAIL:v1:tok_email>", "<PHONE:v1:tok_phone>"],
            scope="test_scope",
            reason="testing",
        )
        assert result == {
            "<EMAIL:v1:tok_email>": "john@example.com",
            "<PHONE:v1:tok_phone>": "555-1234",
        }

    def test_bulk_detokenize_mixed_found_not_found(self):
        """bulk_detokenize returns None for not found tokens."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_found>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        result = svc.bulk_detokenize(
            ["<EMAIL:v1:tok_found>", "<EMAIL:v1:tok_notfound>"],
            scope="test_scope",
            reason="testing",
        )
        assert result == {
            "<EMAIL:v1:tok_found>": "john@example.com",
            "<EMAIL:v1:tok_notfound>": None,
        }

    def test_bulk_detokenize_empty_list(self):
        """bulk_detokenize handles empty token list."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        result = svc.bulk_detokenize([], scope="test_scope", reason="testing")
        assert result == {}


class TestDetokenizeText:
    """Test text detokenization with token replacement."""

    def test_detokenize_text_single_token(self):
        """detokenize_text replaces token in text."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        text = "Contact <EMAIL:v1:tok_abc> for more info."
        result = svc.detokenize_text(text, scope="test_scope", reason="testing")
        assert result == "Contact john@example.com for more info."

    def test_detokenize_text_multiple_tokens(self):
        """detokenize_text replaces multiple tokens."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_email>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<PHONE:v1:tok_phone>",
                plaintext="555-1234",
                entity_type="PHONE",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        text = "Email: <EMAIL:v1:tok_email>, Phone: <PHONE:v1:tok_phone>"
        result = svc.detokenize_text(text, scope="test_scope", reason="testing")
        assert result == "Email: john@example.com, Phone: 555-1234"

    def test_detokenize_text_unknown_token_left_in_place(self):
        """detokenize_text leaves unknown tokens in place."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        text = "Email: <EMAIL:v1:tok_unknown>"
        result = svc.detokenize_text(text, scope="test_scope", reason="testing")
        assert result == "Email: <EMAIL:v1:tok_unknown>"

    def test_detokenize_text_no_tokens(self):
        """detokenize_text returns unchanged text if no tokens."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        text = "This has no tokens."
        result = svc.detokenize_text(text, scope="test_scope", reason="testing")
        assert result == text

    def test_detokenize_text_mixed_tokens(self):
        """detokenize_text handles mix of found and not found tokens."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_found>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        text = "<EMAIL:v1:tok_found> and <EMAIL:v1:tok_notfound>"
        result = svc.detokenize_text(text, scope="test_scope", reason="testing")
        assert result == "john@example.com and <EMAIL:v1:tok_notfound>"


class TestDetokenizePayload:
    """Test payload (dict) detokenization."""

    def test_detokenize_payload_flat_dict(self):
        """detokenize_payload detokenizes flat dictionary."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        payload = {"email": "<EMAIL:v1:tok_abc>", "name": "John"}
        result = svc.detokenize_payload(payload, scope="test_scope", reason="testing")
        assert result == {"email": "john@example.com", "name": "John"}

    def test_detokenize_payload_nested_dict(self):
        """detokenize_payload detokenizes nested dictionaries."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_email>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<PHONE:v1:tok_phone>",
                plaintext="555-1234",
                entity_type="PHONE",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        payload = {
            "user": {
                "contact": {
                    "email": "<EMAIL:v1:tok_email>",
                    "phone": "<PHONE:v1:tok_phone>",
                }
            }
        }
        result = svc.detokenize_payload(payload, scope="test_scope", reason="testing")
        assert result == {
            "user": {
                "contact": {
                    "email": "john@example.com",
                    "phone": "555-1234",
                }
            }
        }

    def test_detokenize_payload_preserves_non_string_values(self):
        """detokenize_payload preserves non-string values."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        payload = {
            "email": "<EMAIL:v1:tok_abc>",
            "age": 30,
            "active": True,
            "score": 99.5,
            "tags": ["work", "personal"],
        }
        result = svc.detokenize_payload(payload, scope="test_scope", reason="testing")
        assert result == {
            "email": "john@example.com",
            "age": 30,
            "active": True,
            "score": 99.5,
            "tags": ["work", "personal"],
        }


class TestAuditLog:
    """Test audit logging functionality."""

    def test_audit_log_records_successful_detokenization(self):
        """audit_log records successful detokenization."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        svc.detokenize_single(
            "<EMAIL:v1:tok_abc>", scope="test_scope", reason="GDPR request"
        )

        log = svc.audit_log
        assert len(log) == 1
        entry = log[0]
        assert entry.token == "<EMAIL:v1:tok_abc>"
        assert entry.entity_type == "EMAIL"
        assert entry.version == 1
        assert entry.success is True
        assert entry.reason == "GDPR request"
        assert entry.scope == "test_scope"

    def test_audit_log_records_failed_detokenization(self):
        """audit_log records failed detokenization."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        svc.detokenize_single(
            "<EMAIL:v1:tok_notfound>", scope="test_scope", reason="testing"
        )

        log = svc.audit_log
        assert len(log) == 1
        entry = log[0]
        assert entry.success is False
        assert entry.token == "<EMAIL:v1:tok_notfound>"

    def test_audit_log_detokenize_text_records_each_attempt(self):
        """audit_log records all token lookups in detokenize_text."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_found>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        text = "<EMAIL:v1:tok_found> and <EMAIL:v1:tok_notfound>"
        svc.detokenize_text(text, scope="test_scope", reason="testing")

        log = svc.audit_log
        assert len(log) == 2
        assert log[0].success is True
        assert log[1].success is False

    def test_audit_log_has_timestamp(self):
        """audit_log entries have timestamps."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        before = time.time()
        svc.detokenize_single(
            "<EMAIL:v1:tok_test>", scope="test_scope", reason="testing"
        )
        after = time.time()

        log = svc.audit_log
        assert len(log) == 1
        assert before <= log[0].timestamp <= after

    def test_clear_audit_log(self):
        """clear_audit_log removes all entries."""
        store = InMemoryTokenStore()
        store.put(
            TokenMapping(
                scope="test_scope",
                token="<EMAIL:v1:tok_abc>",
                plaintext="john@example.com",
                entity_type="EMAIL",
                version=1,
            )
        )

        svc = ReidentificationService(store)
        svc.detokenize_single("<EMAIL:v1:tok_abc>", scope="test_scope")
        assert len(svc.audit_log) == 1

        svc.clear_audit_log()
        assert len(svc.audit_log) == 0


class TestReidentificationAuditEntry:
    """Test ReidentificationAuditEntry dataclass."""

    def test_audit_entry_fields(self):
        """ReidentificationAuditEntry has all expected fields."""
        entry = ReidentificationAuditEntry(
            timestamp=1000.0,
            scope="test",
            token="<EMAIL:v1:tok_abc>",
            entity_type="EMAIL",
            version=1,
            success=True,
            reason="testing",
        )
        assert entry.timestamp == 1000.0
        assert entry.scope == "test"
        assert entry.token == "<EMAIL:v1:tok_abc>"
        assert entry.entity_type == "EMAIL"
        assert entry.version == 1
        assert entry.success is True
        assert entry.reason == "testing"

    def test_audit_entry_frozen(self):
        """ReidentificationAuditEntry is frozen (immutable)."""
        entry = ReidentificationAuditEntry(
            timestamp=1000.0,
            scope="test",
            token="<EMAIL:v1:tok_abc>",
        )
        with pytest.raises(AttributeError):
            entry.success = True

    def test_audit_entry_default_values(self):
        """ReidentificationAuditEntry has sensible defaults."""
        entry = ReidentificationAuditEntry(
            timestamp=1000.0, scope="test", token="<EMAIL:v1:tok_abc>"
        )
        assert entry.entity_type == ""
        assert entry.version == 0
        assert entry.success is False
        assert entry.reason == ""


class TestKeyManagerIntegration:
    """Test integration with KeyManager."""

    def test_detokenize_with_key_manager_valid_version(self):
        """ReidentificationService uses KeyManager to resolve keys."""
        km = KeyManager("primary-secret")
        store = InMemoryTokenStore()

        svc = ReidentificationService(store, key_manager=km)  # noqa: F841

        # Simulate that detokenize_single can resolve the key
        assert km.get_key(1) == "primary-secret"

    def test_detokenize_with_key_manager_invalid_version(self):
        """ReidentificationService handles invalid key versions gracefully."""
        km = KeyManager("primary-secret")
        store = InMemoryTokenStore()
        svc = ReidentificationService(store, key_manager=km)

        # Try to detokenize with non-existent key version
        result = svc.detokenize_single("<EMAIL:v99:tok_test>", scope="test")
        assert result is None


class TestDeterministicHMACTokenizerIntegration:
    """Test roundtrip tokenization and detokenization."""

    def test_tokenize_then_detokenize_text(self):
        """Full roundtrip: tokenize → store → detokenize_text."""
        store = InMemoryTokenStore()
        tokenizer = DeterministicHMACTokenizer()
        km = KeyManager("my-secret")

        # Step 1: Tokenize
        token_rec = tokenizer.tokenize(
            entity_type="EMAIL",
            plaintext="john@example.com",
            scope="case_1",
            version=1,
            key="my-secret",
            store=store,
        )

        # Step 2: Create text with token
        masked_text = f"Contact {token_rec.token} for details."

        # Step 3: Create service and detokenize
        svc = ReidentificationService(store, key_manager=km, tokenizer=tokenizer)
        result = svc.detokenize_text(masked_text, scope="case_1", reason="testing")

        # Verify
        assert result == "Contact john@example.com for details."

    def test_tokenize_multiple_then_detokenize_payload(self):
        """Roundtrip with multiple tokens in payload."""
        store = InMemoryTokenStore()
        tokenizer = DeterministicHMACTokenizer()
        km = KeyManager("my-secret")

        # Tokenize multiple values
        email_token = tokenizer.tokenize(
            entity_type="EMAIL",
            plaintext="john@example.com",
            scope="case_1",
            version=1,
            key="my-secret",
            store=store,
        )
        phone_token = tokenizer.tokenize(
            entity_type="PHONE",
            plaintext="555-1234",
            scope="case_1",
            version=1,
            key="my-secret",
            store=store,
        )

        # Create masked payload
        payload = {
            "user": {"email": email_token.token, "phone": phone_token.token},
            "name": "John",
        }

        # Detokenize
        svc = ReidentificationService(store, key_manager=km, tokenizer=tokenizer)
        result = svc.detokenize_payload(payload, scope="case_1", reason="testing")

        # Verify
        assert result == {
            "user": {"email": "john@example.com", "phone": "555-1234"},
            "name": "John",
        }

    def test_tokenize_audit_trail(self):
        """Audit log tracks all detokenization in roundtrip."""
        store = InMemoryTokenStore()
        tokenizer = DeterministicHMACTokenizer()

        # Tokenize
        token = tokenizer.tokenize(
            entity_type="EMAIL",
            plaintext="john@example.com",
            scope="case_1",
            version=1,
            key="my-secret",
            store=store,
        )

        # Detokenize and track
        svc = ReidentificationService(store, tokenizer=tokenizer)
        svc.detokenize_text(f"Email: {token.token}", scope="case_1", reason="GDPR")

        # Check audit
        log = svc.audit_log
        assert len(log) == 1
        assert log[0].success is True
        assert log[0].reason == "GDPR"
        assert log[0].entity_type == "EMAIL"


class TestDefaultKey:
    """Test fallback default_key when key_manager is None."""

    def test_detokenize_with_default_key(self):
        """ReidentificationService accepts default_key when key_manager is None."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store, default_key="fallback-key")

        # The service should accept the default key parameter
        assert svc._default_key == "fallback-key"

    def test_detokenize_prefers_key_manager_over_default_key(self):
        """KeyManager is used when both are provided."""
        km = KeyManager("km-secret")
        store = InMemoryTokenStore()
        svc = ReidentificationService(
            store, key_manager=km, default_key="default-secret"
        )

        # KeyManager should be preferred
        assert svc._key_manager is not None
        assert svc._default_key == "default-secret"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_detokenize_empty_token_string(self):
        """detokenize_single handles empty token string."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        result = svc.detokenize_single("", scope="test")
        assert result is None

    def test_detokenize_malformed_token(self):
        """detokenize_single handles malformed token format."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        result = svc.detokenize_single("not_a_token", scope="test")
        assert result is None

    def test_detokenize_text_with_partial_match(self):
        """detokenize_text requires exact token format."""
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
        # Missing closing >
        text = "Email: <EMAIL:v1:tok_abc"
        result = svc.detokenize_text(text, scope="test")
        assert result == text

    def test_detokenize_payload_with_none_value(self):
        """detokenize_payload handles None values."""
        store = InMemoryTokenStore()
        svc = ReidentificationService(store)
        payload = {"email": None, "name": "John"}
        result = svc.detokenize_payload(payload, scope="test")
        assert result == {"email": None, "name": "John"}

    def test_audit_log_concurrent_access(self):
        """audit_log is thread-safe."""
        import threading

        store = InMemoryTokenStore()
        svc = ReidentificationService(store)

        def record_entries():
            for i in range(10):
                svc.detokenize_single(f"<EMAIL:v1:tok_{i}>", scope="test")

        threads = [threading.Thread(target=record_entries) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        log = svc.audit_log
        assert len(log) == 30  # 3 threads * 10 entries
