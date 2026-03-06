"""Enterprise key management for pseudonymization.

Provides key versioning, rotation, and expiration tracking for
deterministic tokenization workflows.  Supports multi-version keys
for graceful rotation (dual-key periods) and audit logging.

Usage::

    km = KeyManager("primary-secret")
    v2 = km.rotate("new-secret")   # → 2
    km.get_key(1)                   # → "primary-secret" (still valid)
    km.get_key(2)                   # → "new-secret"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class KeyVersion:
    """A versioned key with lifecycle metadata.

    Attributes
    ----------
    version : int
        Monotonically increasing version number.
    key : str
        The secret key material.
    created_at : float
        Unix timestamp when this version was created.
    expires_at : float | None
        Unix timestamp when this version expires, or ``None`` for no expiry.
    """

    version: int
    key: str
    created_at: float
    expires_at: float | None = None


class KeyManager:
    """Manage key versions for pseudonymization.

    Supports key rotation with configurable grace periods. Old keys
    remain accessible for de-tokenization until explicitly retired.

    Parameters
    ----------
    primary_key : str
        The initial secret key.
    default_ttl_seconds : float | None
        Default time-to-live for keys. ``None`` means keys don't expire.

    Example
    -------
    >>> km = KeyManager("my-secret")
    >>> km.current_version()
    1
    >>> v2 = km.rotate("new-secret")
    >>> km.get_key(1)  # Still accessible
    'my-secret'
    """

    def __init__(
        self,
        primary_key: str,
        *,
        default_ttl_seconds: float | None = None,
    ) -> None:
        self._lock = Lock()
        self._default_ttl = default_ttl_seconds
        now = time.time()
        expires = now + default_ttl_seconds if default_ttl_seconds else None
        self._versions: dict[int, KeyVersion] = {
            1: KeyVersion(version=1, key=primary_key, created_at=now, expires_at=expires),
        }
        self._current: int = 1

    def rotate(
        self,
        new_key: str,
        *,
        ttl_seconds: float | None = None,
    ) -> int:
        """Rotate to a new key version.

        Parameters
        ----------
        new_key : str
            The new secret key.
        ttl_seconds : float | None
            TTL for this key version. Falls back to ``default_ttl_seconds``.

        Returns
        -------
        int
            The new version number.
        """
        with self._lock:
            new_version = max(self._versions.keys()) + 1
            now = time.time()
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires = now + ttl if ttl else None
            self._versions[new_version] = KeyVersion(
                version=new_version,
                key=new_key,
                created_at=now,
                expires_at=expires,
            )
            self._current = new_version
            return new_version

    def get_key(self, version: int) -> str:
        """Retrieve the key for a specific version.

        Parameters
        ----------
        version : int
            The key version number.

        Returns
        -------
        str
            The secret key.

        Raises
        ------
        KeyError
            If the version does not exist.
        ValueError
            If the version has expired.
        """
        with self._lock:
            kv = self._versions.get(version)
            if kv is None:
                raise KeyError(f"Key version {version} not found.")
            if kv.expires_at is not None and time.time() > kv.expires_at:
                raise ValueError(f"Key version {version} has expired.")
            return kv.key

    def current_version(self) -> int:
        """Return the current (latest) key version number."""
        with self._lock:
            return self._current

    def current_key(self) -> str:
        """Return the current (latest) key."""
        return self.get_key(self.current_version())

    def list_versions(self) -> list[KeyVersion]:
        """Return all key versions sorted by version number."""
        with self._lock:
            return sorted(self._versions.values(), key=lambda kv: kv.version)

    def is_valid(self, version: int) -> bool:
        """Check whether a key version exists and has not expired."""
        with self._lock:
            kv = self._versions.get(version)
            if kv is None:
                return False
            if kv.expires_at is not None and time.time() > kv.expires_at:
                return False
            return True

    def retire(self, version: int) -> None:
        """Permanently remove a key version.

        Parameters
        ----------
        version : int
            The version to retire.

        Raises
        ------
        ValueError
            If attempting to retire the current version.
        KeyError
            If the version does not exist.
        """
        with self._lock:
            if version == self._current:
                raise ValueError("Cannot retire the current key version.")
            if version not in self._versions:
                raise KeyError(f"Key version {version} not found.")
            del self._versions[version]

    def __len__(self) -> int:
        with self._lock:
            return len(self._versions)
