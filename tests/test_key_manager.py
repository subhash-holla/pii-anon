"""Tests for KeyManager key versioning and rotation."""

import time

import pytest

from pii_anon.tokenization.key_manager import KeyManager, KeyVersion


class TestKeyManagerInit:
    """Test KeyManager initialization."""

    def test_init_starts_at_version_1(self):
        """KeyManager starts at version 1."""
        km = KeyManager("primary-secret")
        assert km.current_version() == 1

    def test_init_has_primary_key(self):
        """Initial key is accessible via get_key."""
        km = KeyManager("primary-secret")
        assert km.get_key(1) == "primary-secret"

    def test_init_length_is_1(self):
        """Initial KeyManager has length 1."""
        km = KeyManager("primary-secret")
        assert len(km) == 1


class TestGetKey:
    """Test key retrieval."""

    def test_get_key_valid_version(self):
        """get_key returns key for valid version."""
        km = KeyManager("my-secret")
        assert km.get_key(1) == "my-secret"

    def test_get_key_invalid_version_raises_keyerror(self):
        """get_key raises KeyError for invalid version."""
        km = KeyManager("my-secret")
        with pytest.raises(KeyError, match="Key version 99 not found"):
            km.get_key(99)

    def test_get_key_after_rotation(self):
        """get_key works after rotation."""
        km = KeyManager("primary-secret")
        v2 = km.rotate("new-secret")
        assert km.get_key(1) == "primary-secret"
        assert km.get_key(v2) == "new-secret"


class TestRotate:
    """Test key rotation."""

    def test_rotate_returns_new_version(self):
        """rotate returns incremented version number."""
        km = KeyManager("primary-secret")
        v2 = km.rotate("new-secret")
        assert v2 == 2

    def test_rotate_updates_current_version(self):
        """rotate updates current_version()."""
        km = KeyManager("primary-secret")
        km.rotate("new-secret")
        assert km.current_version() == 2

    def test_rotate_old_key_still_accessible(self):
        """Old key remains accessible after rotation."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        assert km.get_key(1) == "secret-1"
        assert km.get_key(2) == "secret-2"

    def test_rotate_increments_length(self):
        """rotate increments len()."""
        km = KeyManager("secret-1")
        assert len(km) == 1
        km.rotate("secret-2")
        assert len(km) == 2

    def test_multiple_rotations_increment_versions(self):
        """Multiple rotations increment version numbers correctly."""
        km = KeyManager("secret-1")
        v2 = km.rotate("secret-2")
        v3 = km.rotate("secret-3")
        v4 = km.rotate("secret-4")

        assert v2 == 2
        assert v3 == 3
        assert v4 == 4
        assert km.current_version() == 4
        assert len(km) == 4

    def test_multiple_rotations_all_accessible(self):
        """All versions accessible after multiple rotations."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")

        assert km.get_key(1) == "secret-1"
        assert km.get_key(2) == "secret-2"
        assert km.get_key(3) == "secret-3"


class TestCurrentKey:
    """Test current key retrieval."""

    def test_current_key_initial(self):
        """current_key returns primary key initially."""
        km = KeyManager("primary-secret")
        assert km.current_key() == "primary-secret"

    def test_current_key_after_rotation(self):
        """current_key returns latest key after rotation."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        assert km.current_key() == "secret-2"


class TestIsValid:
    """Test key validity checking."""

    def test_is_valid_existing_version(self):
        """is_valid returns True for existing version."""
        km = KeyManager("secret-1")
        assert km.is_valid(1) is True

    def test_is_valid_non_existing_version(self):
        """is_valid returns False for non-existing version."""
        km = KeyManager("secret-1")
        assert km.is_valid(99) is False

    def test_is_valid_after_rotation(self):
        """is_valid works for all versions after rotation."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        assert km.is_valid(1) is True
        assert km.is_valid(2) is True
        assert km.is_valid(3) is False

    def test_is_valid_expired_version(self):
        """is_valid returns False for expired version."""
        km = KeyManager("secret-1", default_ttl_seconds=0.05)
        v1_is_valid_initially = km.is_valid(1)
        assert v1_is_valid_initially is True
        time.sleep(0.1)
        assert km.is_valid(1) is False


class TestExpiration:
    """Test key expiration and TTL."""

    def test_default_ttl_seconds_no_expiry(self):
        """Keys don't expire when TTL is None."""
        km = KeyManager("secret-1", default_ttl_seconds=None)
        time.sleep(0.05)
        assert km.is_valid(1) is True

    def test_default_ttl_seconds_expires(self):
        """Key expires after default TTL."""
        km = KeyManager("secret-1", default_ttl_seconds=0.05)
        assert km.is_valid(1) is True
        time.sleep(0.1)
        assert km.is_valid(1) is False

    def test_get_key_expired_raises_valueerror(self):
        """get_key raises ValueError for expired key."""
        km = KeyManager("secret-1", default_ttl_seconds=0.05)
        time.sleep(0.1)
        with pytest.raises(ValueError, match="Key version 1 has expired"):
            km.get_key(1)

    def test_custom_ttl_on_rotate(self):
        """rotate accepts custom TTL per version."""
        km = KeyManager("secret-1", default_ttl_seconds=10.0)
        v2 = km.rotate("secret-2", ttl_seconds=0.05)

        assert km.is_valid(1) is True
        assert km.is_valid(v2) is True
        time.sleep(0.1)
        assert km.is_valid(1) is True  # Original still valid
        assert km.is_valid(v2) is False  # Rotated one expired


class TestRetire:
    """Test key retirement."""

    def test_retire_removes_version(self):
        """retire removes a version."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")

        km.retire(1)
        assert km.is_valid(1) is False
        assert len(km) == 2

    def test_retire_current_raises_valueerror(self):
        """retire raises ValueError for current version."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")

        with pytest.raises(ValueError, match="Cannot retire the current key version"):
            km.retire(2)

    def test_retire_unknown_raises_keyerror(self):
        """retire raises KeyError for unknown version."""
        km = KeyManager("secret-1")
        with pytest.raises(KeyError, match="Key version 99 not found"):
            km.retire(99)

    def test_retire_allows_rotation_after(self):
        """Can rotate after retiring old versions."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")

        km.retire(1)
        v4 = km.rotate("secret-4")

        assert v4 == 4
        assert km.current_version() == 4
        assert len(km) == 3


class TestListVersions:
    """Test version listing."""

    def test_list_versions_initial(self):
        """list_versions returns initial version."""
        km = KeyManager("secret-1")
        versions = km.list_versions()
        assert len(versions) == 1
        assert versions[0].version == 1
        assert versions[0].key == "secret-1"

    def test_list_versions_sorted(self):
        """list_versions returns sorted by version."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")

        versions = km.list_versions()
        assert [v.version for v in versions] == [1, 2, 3]

    def test_list_versions_all_keys(self):
        """list_versions returns all keys."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")

        versions = km.list_versions()
        keys = [v.key for v in versions]
        assert keys == ["secret-1", "secret-2", "secret-3"]


class TestKeyVersionDataclass:
    """Test KeyVersion frozen dataclass."""

    def test_keyversion_frozen(self):
        """KeyVersion is frozen (immutable)."""
        kv = KeyVersion(version=1, key="secret", created_at=1000.0, expires_at=2000.0)
        with pytest.raises(AttributeError):
            kv.version = 2

    def test_keyversion_has_required_fields(self):
        """KeyVersion has expected fields."""
        kv = KeyVersion(version=1, key="secret", created_at=1000.0)
        assert hasattr(kv, "version")
        assert hasattr(kv, "key")
        assert hasattr(kv, "created_at")
        assert hasattr(kv, "expires_at")

    def test_keyversion_expires_at_optional(self):
        """KeyVersion expires_at defaults to None."""
        kv = KeyVersion(version=1, key="secret", created_at=1000.0)
        assert kv.expires_at is None

    def test_keyversion_attributes_match(self):
        """KeyVersion attributes store correct values."""
        now = time.time()
        expires = now + 100.0
        kv = KeyVersion(version=5, key="test-key", created_at=now, expires_at=expires)
        assert kv.version == 5
        assert kv.key == "test-key"
        assert kv.created_at == now
        assert kv.expires_at == expires


class TestLen:
    """Test __len__ magic method."""

    def test_len_initial(self):
        """len() returns 1 initially."""
        km = KeyManager("secret")
        assert len(km) == 1

    def test_len_after_rotations(self):
        """len() increases with rotations."""
        km = KeyManager("secret-1")
        assert len(km) == 1
        km.rotate("secret-2")
        assert len(km) == 2
        km.rotate("secret-3")
        assert len(km) == 3

    def test_len_after_retire(self):
        """len() decreases after retire."""
        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")
        assert len(km) == 3

        km.retire(1)
        assert len(km) == 2


class TestThreadSafety:
    """Test basic thread-safety (locking behavior)."""

    def test_concurrent_get_key(self):
        """Multiple simultaneous get_key calls work."""
        import threading

        km = KeyManager("secret-1")
        km.rotate("secret-2")
        km.rotate("secret-3")

        results = []

        def read_keys():
            for _ in range(10):
                results.append(km.get_key(1))
                results.append(km.get_key(2))
                results.append(km.get_key(3))

        threads = [threading.Thread(target=read_keys) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 90  # 3 threads * 10 iterations * 3 reads


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_key_string(self):
        """KeyManager accepts empty key string."""
        km = KeyManager("")
        assert km.get_key(1) == ""

    def test_very_long_key(self):
        """KeyManager accepts very long key string."""
        long_key = "x" * 10000
        km = KeyManager(long_key)
        assert km.get_key(1) == long_key

    def test_special_characters_in_key(self):
        """KeyManager accepts special characters in key."""
        special_key = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        km = KeyManager(special_key)
        assert km.get_key(1) == special_key

    def test_unicode_key(self):
        """KeyManager accepts unicode key."""
        unicode_key = "秘密🔐\n\t\r"
        km = KeyManager(unicode_key)
        assert km.get_key(1) == unicode_key
