"""Tests for the encrypted-at-rest reversible-pseudonymization token vault.

Story S6-03 (Security MUST, SME Security MAJOR #2). The de-tokenization
mapping ``scope -> token -> plaintext`` must NEVER be persisted unencrypted.
``EncryptedSQLiteTokenStore`` persists every sensitive field under
authenticated encryption (AEAD), with an envelope-wrapped data key and a
keyed blind-index for O(1) lookup, and FAILS LOUD on tamper.

These tests run for REAL in ``.venv`` with the ``[crypto]`` extra (the
``cryptography`` package is present; ``AESGCM``/``ChaCha20Poly1305`` are
import-verified) against a synthetic test KEK and synthetic PII. Real
external key custody (cloud KMS / HSM / OS keyring) is the Pass-2 /
AGENT_SIMULATED honesty boundary: the in-tree ``StaticTestKeyProvider``
stands in for the custodian behind the ``EnvelopeKeyProvider`` protocol.

Test fixtures use ONLY synthetic PII (AX-001): ``"Jane Roe"``,
``"jane.roe@example.test"``, ``"555-0100"`` — never real values. The whole
point of the headline test (A2) is that even these synthetic values are
unrecoverable from the raw ``.db`` bytes.
"""

from __future__ import annotations

import ast
import importlib
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pii_anon.tokenization.encrypted_store as enc_mod
from pii_anon.errors import TokenizationError, TokenStoreIntegrityError
from pii_anon.tokenization.encrypted_store import (
    EncryptedSQLiteTokenStore,
    EnvelopeKeyProvider,
    KeyEnvelope,
    StaticTestKeyProvider,
)
from pii_anon.tokenization.providers import DeterministicHMACTokenizer
from pii_anon.tokenization.store import (
    InMemoryTokenStore,
    SQLiteTokenStore,
    TokenMapping,
    TokenStore,
)

# --- synthetic, never-real fixtures (AX-001) --------------------------------
SYNTH_PLAINTEXT = "jane.roe@example.test"
SYNTH_NAME = "Jane Roe"
SYNTH_PHONE = "555-0100"
SYNTH_TOKEN = "<EMAIL_ADDRESS:v1:tok_abc>"
# Two distinct synthetic test KEKs (32 bytes each for AES-256). Obviously
# synthetic — NOT a real key (security-sast: no hard-coded real key).
KEK_A = b"A" * 32
KEK_B = b"B" * 32


@pytest.fixture
def temp_db_path() -> Iterator[Path]:
    """Yield a fresh temp .db path; clean it up afterwards (deterministic FS)."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "enc_tokens.db"


def _make_store(
    db_path: Path,
    kek: bytes = KEK_A,
    *,
    algorithm: str = "aesgcm",
) -> EncryptedSQLiteTokenStore:
    return EncryptedSQLiteTokenStore(
        db_path,
        key_provider=StaticTestKeyProvider(kek),
        algorithm=algorithm,
    )


def _mapping(
    *,
    scope: str = "s",
    token: str = SYNTH_TOKEN,
    plaintext: str = SYNTH_PLAINTEXT,
    entity_type: str = "EMAIL_ADDRESS",
    version: int = 1,
    created_at: float = 0.0,
    expires_at: float | None = None,
) -> TokenMapping:
    return TokenMapping(
        scope=scope,
        token=token,
        plaintext=plaintext,
        entity_type=entity_type,
        version=version,
        created_at=created_at,
        expires_at=expires_at,
    )


# ===========================================================================
# A1 — round-trip preserves the mapping (FR-019, NFR-014) [UNIT-TEST]
# ===========================================================================
class TestA1RoundTrip:
    def test_fr_019_roundtrip_equality_scoped_get(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        m = _mapping()
        store.put(m)

        got = store.get(SYNTH_TOKEN, scope="s")
        assert got is not None
        assert got.scope == m.scope
        assert got.token == m.token
        assert got.plaintext == m.plaintext == SYNTH_PLAINTEXT
        assert got.entity_type == m.entity_type
        assert got.version == m.version
        assert store.count() == 1
        listed = store.list_by_scope("s")
        assert len(listed) == 1
        assert listed[0].plaintext == SYNTH_PLAINTEXT
        store.close()

    def test_fr_019_roundtrip_survives_reopen(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        store.put(_mapping())
        store.close()
        # Reopen the same file under the same KEK -> still recoverable.
        store2 = _make_store(temp_db_path)
        got = store2.get(SYNTH_TOKEN, scope="s")
        assert got is not None
        assert got.plaintext == SYNTH_PLAINTEXT
        store2.close()

    def test_chacha20_algorithm_roundtrip(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path, algorithm="chacha20poly1305")
        store.put(_mapping())
        got = store.get(SYNTH_TOKEN, scope="s")
        assert got is not None
        assert got.plaintext == SYNTH_PLAINTEXT
        store.close()


# ===========================================================================
# A1/A9 — parametrized shared-behaviour contract vs SQLiteTokenStore
# ===========================================================================
def _build_encrypted(tmp: Path) -> tuple[TokenStore, list]:
    closers: list = []
    store = _make_store(tmp / "enc.db")
    closers.append(store)
    return store, closers


def _build_sqlite(tmp: Path) -> tuple[TokenStore, list]:
    closers: list = []
    store = SQLiteTokenStore(str(tmp / "plain.db"))
    closers.append(store)
    return store, closers


def _build_inmemory(tmp: Path) -> tuple[TokenStore, list]:
    return InMemoryTokenStore(), []


_STORE_BUILDERS = [
    pytest.param(_build_encrypted, id="encrypted"),
    pytest.param(_build_sqlite, id="sqlite"),
    pytest.param(_build_inmemory, id="inmemory"),
]


@pytest.mark.parametrize("builder", _STORE_BUILDERS)
class TestContractSharedBehaviour:
    """The SAME behavioural assertions run against every TokenStore impl."""

    def test_put_get_count(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            store.put(_mapping(scope="s1", token="t1", plaintext="val1"))
            store.put(_mapping(scope="s2", token="t2", plaintext="val2"))
            assert store.count() == 2
            assert store.count(scope="s1") == 1
            assert store.count(scope="s2") == 1
            assert store.get("t1", scope="s1").plaintext == "val1"
        finally:
            for c in closers:
                c.close()

    def test_scope_filtering_list(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            store.put(_mapping(scope="s1", token="t1", plaintext="val1"))
            store.put(_mapping(scope="s1", token="t2", plaintext="val2"))
            store.put(_mapping(scope="s2", token="t3", plaintext="val3"))
            s1 = store.list_by_scope("s1")
            assert len(s1) == 2
            assert all(m.scope == "s1" for m in s1)
            assert {m.plaintext for m in s1} == {"val1", "val2"}
            assert len(store.list_by_scope("s2")) == 1
            assert store.list_by_scope("absent") == []
        finally:
            for c in closers:
                c.close()

    def test_scopeless_get_scan(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            store.put(_mapping(scope="other_scope", token="t1", plaintext="val1"))
            got = store.get("t1")  # no scope -> scan
            assert got is not None
            assert got.scope == "other_scope"
            assert got.plaintext == "val1"
        finally:
            for c in closers:
                c.close()

    def test_absent_token_returns_none(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            assert store.get("nope", scope="s1") is None
            assert store.get("nope") is None
        finally:
            for c in closers:
                c.close()

    def test_replace_existing(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            store.put(_mapping(scope="s1", token="t1", plaintext="secret1", entity_type="EMAIL_ADDRESS", version=1))
            store.put(_mapping(scope="s1", token="t1", plaintext="secret2", entity_type="PHONE_NUMBER", version=2))
            got = store.get("t1", scope="s1")
            assert got is not None
            assert got.plaintext == "secret2"
            assert got.entity_type == "PHONE_NUMBER"
            assert got.version == 2
            assert store.count() == 1
        finally:
            for c in closers:
                c.close()

    def test_ttl_get_returns_none_for_expired(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            # expires_at is compared against the wall clock by every store, so
            # anchor on time.time(): "expired" = far past, "valid" = far future.
            now = time.time()
            store.put(
                _mapping(scope="s1", token="t1", plaintext="v", created_at=now - 100, expires_at=now - 1)
            )
            assert store.get("t1", scope="s1") is None
        finally:
            for c in closers:
                c.close()

    def test_ttl_count_and_list_exclude_expired(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            now = time.time()
            store.put(_mapping(scope="s1", token="exp", plaintext="v1", created_at=now - 100, expires_at=now - 1))
            store.put(_mapping(scope="s1", token="ok", plaintext="v2", created_at=now, expires_at=now + 86_400))
            assert store.count() == 1
            assert store.count(scope="s1") == 1
            listed = store.list_by_scope("s1")
            assert len(listed) == 1
            assert listed[0].token == "ok"
        finally:
            for c in closers:
                c.close()

    def test_delete_expired_removes_only_expired(self, builder, tmp_path: Path) -> None:
        store, closers = builder(tmp_path)
        try:
            now = time.time()
            store.put(_mapping(scope="s1", token="exp", plaintext="v1", created_at=now - 100, expires_at=now - 1))
            store.put(_mapping(scope="s1", token="ok", plaintext="v2", created_at=now, expires_at=now + 86_400))
            removed = store.delete_expired()
            assert removed == 1
            assert store.count() == 1
            assert store.get("ok", scope="s1") is not None
        finally:
            for c in closers:
                c.close()


# ===========================================================================
# A2 — on-disk bytes are NOT plaintext (FR-026 / AX-006) — the headline
# ===========================================================================
class TestA2NotPlaintextOnDisk:
    def test_fr_026_synthetic_plaintext_and_token_absent_from_raw_db(
        self, temp_db_path: Path
    ) -> None:
        store = _make_store(temp_db_path)
        store.put(_mapping(plaintext=SYNTH_PLAINTEXT, token=SYNTH_TOKEN))
        store.put(_mapping(scope="s2", token="<PERSON:v1:tok_def>", plaintext=SYNTH_NAME, entity_type="PERSON"))
        store.close()

        raw = temp_db_path.read_bytes()
        # The synthetic plaintext PII must NOT appear anywhere in the file.
        assert SYNTH_PLAINTEXT.encode("utf-8") not in raw
        assert SYNTH_NAME.encode("utf-8") not in raw
        # The plaintext token string must NOT appear either (defense-in-depth).
        assert SYNTH_TOKEN.encode("utf-8") not in raw
        assert b"<PERSON:v1:tok_def>" not in raw
        # The scope/entity_type cleartext must not leak either.
        assert b"EMAIL_ADDRESS" not in raw

    def test_fr_026_value_columns_are_blob_ciphertext(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        store.put(_mapping())
        store.close()

        conn = sqlite3.connect(str(temp_db_path))
        try:
            # Find the mappings table and assert every sensitive column is BLOB
            # and that no cell anywhere holds the synthetic plaintext/token.
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert tables, "expected at least one table"
            found_ciphertext_blob = False
            for table in tables:
                rows = conn.execute(f"SELECT * FROM '{table}'").fetchall()  # noqa: S608 - table from sqlite_master, not user input
                for row in rows:
                    for cell in row:
                        if isinstance(cell, (bytes, bytearray)):
                            found_ciphertext_blob = True
                            assert SYNTH_PLAINTEXT.encode("utf-8") not in bytes(cell)
                            assert SYNTH_TOKEN.encode("utf-8") not in bytes(cell)
                        if isinstance(cell, str):
                            assert cell != SYNTH_PLAINTEXT
                            assert cell != SYNTH_TOKEN
            assert found_ciphertext_blob, "expected BLOB ciphertext columns"
        finally:
            conn.close()

    def test_nfr_015_kek_and_dek_plaintext_absent_from_disk(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path, kek=KEK_A)
        store.put(_mapping())
        # Capture the live DEK before closing (it must never hit disk).
        dek = store._current_dek_bytes_for_test()  # test-only accessor
        store.close()

        raw = temp_db_path.read_bytes()
        assert KEK_A not in raw  # KEK never persisted
        assert dek not in raw  # DEK never persisted in the clear
        assert len(dek) >= 16


# ===========================================================================
# A3 — tamper a ciphertext byte => decrypt RAISES (NFR-014, integrity)
# ===========================================================================
class TestA3TamperRaises:
    def _corrupt_first_payload_byte(self, db_path: Path) -> None:
        """Flip one byte inside the first row's ciphertext payload blob."""
        conn = sqlite3.connect(str(db_path))
        try:
            table = "encrypted_token_mappings"
            row = conn.execute(
                f"SELECT rowid, payload FROM '{table}'"  # noqa: S608
            ).fetchone()
            assert row is not None, "no row to corrupt"
            rowid, payload = row[0], bytes(row[1])
            assert len(payload) > 0
            mutated = bytearray(payload)
            mutated[0] ^= 0x01
            conn.execute(
                f"UPDATE '{table}' SET payload = ? WHERE rowid = ?",  # noqa: S608
                (bytes(mutated), rowid),
            )
            conn.commit()
        finally:
            conn.close()

    def test_nfr_014_single_byte_tamper_raises_integrity_error(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        store.put(_mapping())
        store.close()

        self._corrupt_first_payload_byte(temp_db_path)

        store2 = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            store2.get(SYNTH_TOKEN, scope="s")
        store2.close()

    def test_integrity_error_is_tokenization_error_subclass(self) -> None:
        assert issubclass(TokenStoreIntegrityError, TokenizationError)

    def test_nfr_014_tamper_never_returns_wrong_plaintext(self, temp_db_path: Path) -> None:
        """Tamper must RAISE; it must never silently return None or wrong bytes."""
        store = _make_store(temp_db_path)
        store.put(_mapping())
        store.close()
        self._corrupt_first_payload_byte(temp_db_path)
        store2 = _make_store(temp_db_path)
        raised = False
        try:
            store2.get(SYNTH_TOKEN, scope="s")
        except TokenStoreIntegrityError:
            raised = True
        assert raised, "tamper must raise, never silently return"
        store2.close()

    def test_nfr_014_row_relocation_aad_mismatch_raises(self, temp_db_path: Path) -> None:
        """Move row A's ciphertext into row B's (scope,token) slot => AAD fail."""
        store = _make_store(temp_db_path)
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:tok_a>", plaintext="a@example.test"))
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:tok_b>", plaintext="b@example.test"))
        store.close()

        # Swap the two rows' encrypted payloads while keeping each row's
        # blind-index column (its AAD binding) in place. Whichever token we
        # then read decrypts the *other* row's payload under *its own* AAD ->
        # the tag fails (cut-and-paste / row-relocation tampering).
        conn = sqlite3.connect(str(temp_db_path))
        try:
            table = "encrypted_token_mappings"
            rows = conn.execute(
                f"SELECT blind_index, payload FROM '{table}'"  # noqa: S608
            ).fetchall()
            assert len(rows) == 2
            (bi_first, payload_first), (bi_second, payload_second) = rows[0], rows[1]
            conn.execute(
                f"UPDATE '{table}' SET payload = ? WHERE blind_index = ?",  # noqa: S608
                (payload_second, bi_first),
            )
            conn.execute(
                f"UPDATE '{table}' SET payload = ? WHERE blind_index = ?",  # noqa: S608
                (payload_first, bi_second),
            )
            conn.commit()
        finally:
            conn.close()

        store2 = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            # Reading row A now decrypts B's relocated payload under A's AAD.
            store2.get("<EMAIL_ADDRESS:v1:tok_a>", scope="s")
        store2.close()

    @settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        flip_pos=st.integers(min_value=0, max_value=10_000),
        flip_mask=st.integers(min_value=1, max_value=255),
    )
    def test_nfr_014_property_any_single_byte_flip_in_payload_raises(
        self, flip_pos: int, flip_mask: int
    ) -> None:
        """Property: flipping ANY byte of the stored payload blob always raises,
        never yields a successful-but-wrong decrypt."""
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "prop.db"
            store = _make_store(db_path)
            store.put(_mapping())
            store.close()

            conn = sqlite3.connect(str(db_path))
            try:
                table = "encrypted_token_mappings"
                row = conn.execute(
                    f"SELECT rowid, payload FROM '{table}'"  # noqa: S608
                ).fetchone()
                rowid, payload = row[0], row[1]
                payload = bytes(payload)
                if len(payload) == 0:
                    return
                pos = flip_pos % len(payload)
                mutated = bytearray(payload)
                mutated[pos] ^= flip_mask
                if bytes(mutated) == payload:
                    return  # no-op flip (mask cancelled) — not a tamper
                conn.execute(
                    f"UPDATE '{table}' SET payload = ? WHERE rowid = ?",  # noqa: S608
                    (bytes(mutated), rowid),
                )
                conn.commit()
            finally:
                conn.close()

            store2 = _make_store(db_path)
            try:
                with pytest.raises(TokenStoreIntegrityError):
                    store2.get(SYNTH_TOKEN, scope="s")
            finally:
                store2.close()


# ===========================================================================
# A4 — wrong key cannot reverse (NFR-014 / NFR-015)
# ===========================================================================
class TestA4WrongKey:
    def test_nfr_015_wrong_kek_cannot_reverse(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path, kek=KEK_A)
        store.put(_mapping())
        store.close()

        # Reopen the SAME .db under a DIFFERENT KEK -> unwrap/AEAD failure.
        store_wrong = _make_store(temp_db_path, kek=KEK_B)
        with pytest.raises((TokenStoreIntegrityError, TokenizationError)):
            store_wrong.get(SYNTH_TOKEN, scope="s")
        store_wrong.close()


# ===========================================================================
# A5 — envelope wrap + key rotation (FR-019 auditable rotation; DC-04)
# ===========================================================================
class TestA5EnvelopeRotation:
    def test_fr_019_rotation_old_rows_still_decrypt_new_rows_under_new_key(
        self, temp_db_path: Path
    ) -> None:
        provider_k1 = StaticTestKeyProvider(KEK_A, key_id="k1")
        store = EncryptedSQLiteTokenStore(temp_db_path, key_provider=provider_k1)
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:tok_old>", plaintext="old@example.test"))

        # Rotate to a new key id wrapping a new DEK.
        provider_k2 = StaticTestKeyProvider(KEK_B, key_id="k2")
        store.rotate(provider_k2)
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:tok_new>", plaintext="new@example.test"))

        # Old row still decrypts (under its stamped key id); new row under k2.
        old = store.get("<EMAIL_ADDRESS:v1:tok_old>", scope="s")
        new = store.get("<EMAIL_ADDRESS:v1:tok_new>", scope="s")
        assert old is not None and old.plaintext == "old@example.test"
        assert new is not None and new.plaintext == "new@example.test"

        # The set of key ids in use is enumerable for audit.
        key_ids = store.list_key_ids()
        assert "k1" in key_ids and "k2" in key_ids
        store.close()

    def test_fr_019_rewrap_all_migrates_legacy_rows(self, temp_db_path: Path) -> None:
        provider_k1 = StaticTestKeyProvider(KEK_A, key_id="k1")
        store = EncryptedSQLiteTokenStore(temp_db_path, key_provider=provider_k1)
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:tok_x>", plaintext=SYNTH_PLAINTEXT))

        provider_k2 = StaticTestKeyProvider(KEK_B, key_id="k2")
        n = store.rewrap_all(provider_k2)
        assert n >= 1  # legacy rows re-encrypted

        # After rewrap, the row carries k2 and still round-trips.
        got = store.get("<EMAIL_ADDRESS:v1:tok_x>", scope="s")
        assert got is not None and got.plaintext == SYNTH_PLAINTEXT
        assert store.row_key_id("<EMAIL_ADDRESS:v1:tok_x>", scope="s") == "k2"
        store.close()

    def test_a5_envelope_stores_only_wrapped_dek_not_bare_dek(self, temp_db_path: Path) -> None:
        provider = StaticTestKeyProvider(KEK_A, key_id="k1")
        store = EncryptedSQLiteTokenStore(temp_db_path, key_provider=provider)
        store.put(_mapping())
        dek = store._current_dek_bytes_for_test()
        envelopes = store.list_key_envelopes()
        assert len(envelopes) >= 1
        for env in envelopes:
            assert isinstance(env, KeyEnvelope)
            # The envelope stores wrapped_dek, never the bare DEK.
            assert dek not in env.wrapped_dek
            assert env.wrapped_dek != dek
            assert env.key_id
        store.close()

    def test_a5_rewrapped_db_decrypts_only_under_new_key(self, temp_db_path: Path) -> None:
        """After rewrap_all to KEK_B, the on-disk file must NOT decrypt under KEK_A."""
        store = EncryptedSQLiteTokenStore(
            temp_db_path, key_provider=StaticTestKeyProvider(KEK_A, key_id="k1")
        )
        store.put(_mapping())
        store.rewrap_all(StaticTestKeyProvider(KEK_B, key_id="k2"))
        store.close()

        # Reopen under the OLD KEK_A -> can no longer reverse.
        stale = EncryptedSQLiteTokenStore(
            temp_db_path, key_provider=StaticTestKeyProvider(KEK_A, key_id="k1")
        )
        with pytest.raises((TokenStoreIntegrityError, TokenizationError)):
            stale.get(SYNTH_TOKEN, scope="s")
        stale.close()


# ===========================================================================
# A6 — deterministic pseudonymization still holds end-to-end (AX-002)
# ===========================================================================
class TestA6DeterminismPreserved:
    def test_ax_002_token_deterministic_through_encrypted_store(self, tmp_path: Path) -> None:
        tokenizer = DeterministicHMACTokenizer()
        key = "synthetic-tokenizer-key"

        store1 = _make_store(tmp_path / "d1.db")
        rec1 = tokenizer.tokenize("EMAIL_ADDRESS", SYNTH_PLAINTEXT, "scopeX", 1, key, store=store1)

        # A separate tokenizer instance (no cache) + fresh encrypted store:
        # the token is the HMAC output, independent of the random AEAD nonce.
        tokenizer2 = DeterministicHMACTokenizer()
        store2 = _make_store(tmp_path / "d2.db")
        rec2 = tokenizer2.tokenize("EMAIL_ADDRESS", SYNTH_PLAINTEXT, "scopeX", 1, key, store=store2)

        assert rec1.token == rec2.token  # determinism preserved (AX-002)

        # Detokenize via the encrypted store -> identical plaintext.
        back1 = tokenizer.detokenize(rec1, key=key, store=store1)
        back2 = tokenizer2.detokenize(rec2, key=key, store=store2)
        assert back1 == back2 == SYNTH_PLAINTEXT
        store1.close()
        store2.close()


# ===========================================================================
# A7 — anon vs pseudo separation preserved (AX-004 / FR-010)
# ===========================================================================
class TestA7AnonPseudoWall:
    def test_ax_004_imports_nothing_from_anon_or_eval_scoring(self) -> None:
        src = Path(enc_mod.__file__).read_text(encoding="utf-8")
        # The encrypted store is a reversible-pseudonymization component and
        # must not import anonymization / eval-scoring packages.
        forbidden = [
            "from pii_anon.anonymizer",
            "import pii_anon.anonymizer",
            "from pii_anon.evaluation",
            "import pii_anon.evaluation",
            "from pii_anon.scoring",
            "import pii_anon.scoring",
            "anonymize_score",
        ]
        for needle in forbidden:
            assert needle not in src, f"anon/eval coupling leaked: {needle!r}"

    def test_ax_004_exposes_only_reversible_mapping_surface(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        # The reversible surface exists ...
        for attr in ("put", "get", "list_by_scope", "count", "delete_expired", "close"):
            assert callable(getattr(store, attr))
        # ... and there is NO anonymization-score emitting surface.
        for forbidden_attr in ("anonymize", "anonymize_score", "score", "irreversible_transform"):
            assert not hasattr(store, forbidden_attr)
        store.close()


# ===========================================================================
# A8 — fail-loud on missing crypto dep (NFR-026 pattern)
# ===========================================================================
class TestA8MissingCryptoDep:
    def test_nfr_026_missing_aead_provider_guard_raises_with_install_hint(
        self, temp_db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Module import must already have succeeded (discovery unaffected) ...
        mod = importlib.import_module("pii_anon.tokenization.encrypted_store")
        # ... now monkeypatch the AEAD symbol to None (mirrors providers.py:13).
        monkeypatch.setattr(mod, "AESGCM", None)
        monkeypatch.setattr(mod, "ChaCha20Poly1305", None, raising=False)
        with pytest.raises(TokenizationError) as ei:
            EncryptedSQLiteTokenStore(temp_db_path, key_provider=StaticTestKeyProvider(KEK_A))
        assert "pii-anon[crypto]" in str(ei.value)

    def test_nfr_026_missing_aead_store_ctor_guard_raises(
        self, temp_db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hit the store constructor's own crypto guard directly.

        Build a real provider FIRST (so it is not the thing that raises), then
        null the AEAD symbols and construct the store — exercising the store's
        independent ``if AESGCM is None or ChaCha20Poly1305 is None`` guard.
        """
        mod = importlib.import_module("pii_anon.tokenization.encrypted_store")
        provider = StaticTestKeyProvider(KEK_A)  # built while crypto present
        monkeypatch.setattr(mod, "AESGCM", None)
        monkeypatch.setattr(mod, "ChaCha20Poly1305", None, raising=False)
        with pytest.raises(TokenizationError) as ei:
            EncryptedSQLiteTokenStore(temp_db_path, key_provider=provider)
        assert "pii-anon[crypto]" in str(ei.value)


# ===========================================================================
# A9 — TokenStore Liskov contract (CONTRACT)
# ===========================================================================
class TestA9LiskovContract:
    def test_isinstance_tokenstore(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        assert isinstance(store, TokenStore)
        store.close()

    def test_all_abstract_methods_overridden(self) -> None:
        for name in ("put", "get", "list_by_scope", "count", "delete_expired"):
            assert getattr(EncryptedSQLiteTokenStore, name) is not getattr(TokenStore, name)

    def test_get_returns_none_for_absent_distinct_from_tamper(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        store.put(_mapping())
        # absent token -> None (NOT a raise; the raise is reserved for tamper)
        assert store.get("<EMAIL_ADDRESS:v1:tok_absent>", scope="s") is None
        # present token -> recovered
        assert store.get(SYNTH_TOKEN, scope="s") is not None
        store.close()


# ===========================================================================
# Provider protocol structural conformance
# ===========================================================================
class TestEnvelopeKeyProviderProtocol:
    def test_static_provider_is_envelope_key_provider(self) -> None:
        provider = StaticTestKeyProvider(KEK_A, key_id="k1")
        assert isinstance(provider, EnvelopeKeyProvider)
        assert provider.current_key_id() == "k1"
        key_id, wrapped = provider.wrap(b"D" * 32)
        assert key_id == "k1"
        unwrapped = provider.unwrap(key_id, wrapped)
        assert unwrapped == b"D" * 32


# ===========================================================================
# Input-validation + robustness guards (defensive error paths)
# ===========================================================================
class TestGuardsAndRobustness:
    def test_provider_rejects_bad_kek_length(self) -> None:
        with pytest.raises(TokenizationError):
            StaticTestKeyProvider(b"short", key_id="k1")  # not 16/24/32 bytes

    def test_store_rejects_unsupported_algorithm_on_put(self, temp_db_path: Path) -> None:
        store = EncryptedSQLiteTokenStore(
            temp_db_path,
            key_provider=StaticTestKeyProvider(KEK_A),
            algorithm="rot13",  # unsupported
        )
        with pytest.raises(TokenizationError):
            store.put(_mapping())
        store.close()

    def test_scopeless_get_skips_expired_then_finds_valid(self, temp_db_path: Path) -> None:
        """Scope-less scan must skip an expired row and still find a live one."""
        now = time.time()
        store = _make_store(temp_db_path)
        store.put(
            _mapping(scope="s1", token="exp", plaintext="v1", created_at=now - 100, expires_at=now - 1)
        )
        store.put(_mapping(scope="s2", token="live", plaintext="v2"))
        # No-scope lookup of the expired token -> None (skipped, then unmatched).
        assert store.get("exp") is None
        # No-scope lookup of the live token -> found.
        got = store.get("live")
        assert got is not None and got.plaintext == "v2"
        store.close()

    def test_scopeless_get_skips_unreversible_row_then_fails_loud(self, temp_db_path: Path) -> None:
        """A row owned by an unwrap-failing key is skipped in the scan loop,
        and because that key owns rows the lookup then FAILS LOUD (NFR-015).

        Exercises both the in-scan skip branch (the row cannot be decrypted) and
        the wrong-key fail-loud guard — a scope-less get under the wrong KEK
        must never silently report a token as absent when an unreadable row set
        is present.
        """
        store = EncryptedSQLiteTokenStore(
            temp_db_path, key_provider=StaticTestKeyProvider(KEK_A, key_id="k1")
        )
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:tok_x>", plaintext=SYNTH_PLAINTEXT))
        store.rewrap_all(StaticTestKeyProvider(KEK_B, key_id="k2"))
        store.close()

        # Reopen under k1/KEK_A. The only row is owned by k2 (unreversible here):
        # the scan skips the undecryptable row, then the wrong-key guard raises.
        stale = EncryptedSQLiteTokenStore(
            temp_db_path, key_provider=StaticTestKeyProvider(KEK_A, key_id="k1")
        )
        with pytest.raises((TokenStoreIntegrityError, TokenizationError)):
            stale.get("<EMAIL_ADDRESS:v1:tok_unrelated>")
        stale.close()

    def test_row_key_id_returns_none_for_absent(self, temp_db_path: Path) -> None:
        store = _make_store(temp_db_path)
        store.put(_mapping())
        assert store.row_key_id("<EMAIL_ADDRESS:v1:tok_absent>", scope="s") is None
        assert store.row_key_id(SYNTH_TOKEN, scope="s") is not None
        store.close()

    def test_dek_for_missing_envelope_raises(self, temp_db_path: Path) -> None:
        """Requesting a DEK for a key id with no persisted envelope raises."""
        store = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            store._dek_for("never-minted-key-id")
        store.close()


# ===========================================================================
# FIX A (iter-2) — committed DoD §12 (d)/(e)/(g) security-acceptance pins
#
# The §12 checkboxes were TRUE by construction but had ZERO committed
# regression tests (unlike S2-05's 4 AST sink tests / S5-04's 8). These pin
# them: an AST/source-scan asserting the module has ZERO unsafe-deserialization
# / subprocess / shell-out / dynamic-code-evaluation / arbitrary-import call
# signatures (mirrors `_scan_unsafe_call_signatures` in test_attack_sandbox.py,
# including the strengthened getattr/import_module string-sink indirection),
# plus a nonce-distinctness assertion (random 96-bit nonce, no reuse / DEK).
#
# Forbidden call/attribute *names* are assembled from parts (token-reword rule)
# so the literal blocked substrings never appear contiguously in this source
# (the PreToolUse Write hook + the AST guard both key on AST node names).
# ===========================================================================
_UNSAFE_DESERIALIZE_MODULE = "p" + "ickle"  # the unsafe-deserialization module
_FORBIDDEN_MODULE_NAMES: frozenset[str] = frozenset(
    {
        _UNSAFE_DESERIALIZE_MODULE,
        "sub" + "process",
        "marshal",
        "shelve",
    }
)
# Forbidden dynamic-code-evaluation / arbitrary-import builtins, assembled from
# parts so the literal blocked substrings stay non-contiguous in this source.
_FORBIDDEN_CALL_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "ev" + "al",
        "ex" + "ec",
        "compile",
        "__import__",
    }
)
# Attribute calls like os.<shell-call> / os.<shell-call-variant>, built from
# parts so the literal os-level shell-call name is not a contiguous substring.
_FORBIDDEN_OS_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "sy" + "stem",
        "po" + "pen",
        "ex" + "ecv",
        "ex" + "ecve",
        "sp" + "awn",
    }
)
_GETATTR_ROUTERS: frozenset[str] = frozenset({"get" + "attr"})
_IMPORT_ROUTERS: frozenset[str] = frozenset({"import_" + "module", "__import__"})
_FORBIDDEN_STRING_SINKS: frozenset[str] = (
    _FORBIDDEN_OS_ATTR_NAMES | _FORBIDDEN_MODULE_NAMES | _FORBIDDEN_CALL_FUNC_NAMES
)


def _imported_module_heads(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append(node.module.split(".")[0])
    return out


def _string_arg_values(node: ast.Call) -> list[str]:
    out: list[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value)
    return out


def _scan_unsafe_call_signatures(tree: ast.AST, source_label: str) -> list[str]:
    """Return unsafe-call-signature violations found in ``tree``.

    Flags (a) NAIVE/CONTIGUOUS signatures — a forbidden builtin call
    (dynamic-code-evaluation family) or a forbidden attribute call (os-level
    shell-call / subprocess attr) — AND (b) getattr/import_module/__import__-
    ROUTED indirection whose STRING argument is a forbidden sink name. Mirrors
    the strengthened matcher in test_attack_sandbox.py so a docstring that
    merely mentions a pattern cannot false-positive (AST node names, not prose).
    """
    violations: list[str] = []
    for head in _imported_module_heads(tree):
        if head in _FORBIDDEN_MODULE_NAMES:
            violations.append(f"{source_label}: imports forbidden module head {head!r}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_FUNC_NAMES:
            violations.append(f"{source_label}: calls forbidden builtin {func.id!r}")
        if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_OS_ATTR_NAMES:
            violations.append(f"{source_label}: calls forbidden attr .{func.attr}")
        router_name: str | None = None
        if isinstance(func, ast.Name):
            router_name = func.id
        elif isinstance(func, ast.Attribute):
            router_name = func.attr
        if router_name in (_GETATTR_ROUTERS | _IMPORT_ROUTERS):
            for sval in _string_arg_values(node):
                if sval in _FORBIDDEN_STRING_SINKS:
                    violations.append(
                        f"{source_label}: {router_name}(...) routes to forbidden sink {sval!r}"
                    )
    return violations


class TestFixASecurityAcceptanceSourceScan:
    """[SECURITY-TEST] FIX A — committed parity pins for DoD §12 (d)/(e)/(g)."""

    def test_dod_d_e_no_unsafe_deserialize_subprocess_shellout_or_dyn_codepath(self) -> None:
        """§12 (d)/(e): an AST scan of encrypted_store.py asserts ZERO
        unsafe-deserialization / subprocess / shell-out / dynamic-code-evaluation
        / arbitrary-import call signatures (and zero getattr/import_module-routed
        string-sink indirection). AST-based, so the module's prose docstring
        (which explicitly states these are NOT used) cannot false-positive."""
        src_path = Path(enc_mod.__file__)
        tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
        violations = _scan_unsafe_call_signatures(tree, src_path.name)
        assert not violations, (
            "encrypted_store.py has unsafe call signatures: " + "; ".join(violations)
        )

    def test_meta_guard_scanner_is_not_a_no_op(self) -> None:
        """Meta-guard: the scanner flags a synthetic forbidden call (keeps the
        §12 (d)/(e) pin honest if names drift). Sink names assembled from parts
        so no literal blocked substring is contiguous in this source."""
        shell_call = "sy" + "stem"
        getattr_router = "get" + "attr"
        snippet = "import os\nf = %s(os, %r)\nf('id')\n" % (getattr_router, shell_call)
        tree = ast.parse(snippet)
        violations = _scan_unsafe_call_signatures(tree, "<meta>")
        assert violations
        assert any("routes to forbidden sink" in v for v in violations)

    def test_dod_d_only_json_serialization_no_arbitrary_loader(self) -> None:
        """§12 (d): the on-disk payload is round-tripped via the json module
        only (json.loads / json.dumps over an explicit field dict) — never an
        arbitrary-object loader. Assert the json calls are present AND that no
        forbidden-deserialization module head is imported."""
        src_path = Path(enc_mod.__file__)
        src = src_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(src_path))
        heads = set(_imported_module_heads(tree))
        assert "json" in heads
        assert heads.isdisjoint(_FORBIDDEN_MODULE_NAMES)
        # json.loads / json.dumps attribute calls are the only (de)serializers.
        json_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "json"
        }
        assert json_calls <= {"loads", "dumps"}
        assert {"loads", "dumps"} <= json_calls

    def test_dod_g_nonce_is_distinct_per_encryption_no_reuse(
        self, temp_db_path: Path
    ) -> None:
        """§12 (g): a fresh random 96-bit nonce per encryption — no reuse under
        one DEK. put N>=50 synthetic mappings, read the stored nonces
        (``payload[:_NONCE_BYTES]``), assert all distinct."""
        n = 60
        store = _make_store(temp_db_path)
        for i in range(n):
            store.put(_mapping(scope="s", token=f"<EMAIL_ADDRESS:v1:tok_{i}>", plaintext=f"u{i}@example.test"))
        store.close()

        nonce_len = enc_mod._NONCE_BYTES
        assert nonce_len == 12  # 96-bit AEAD nonce
        conn = sqlite3.connect(str(temp_db_path))
        try:
            payloads = [
                bytes(r[0])
                for r in conn.execute(
                    "SELECT payload FROM encrypted_token_mappings"  # noqa: S608
                ).fetchall()
            ]
        finally:
            conn.close()
        assert len(payloads) == n
        nonces = [p[:nonce_len] for p in payloads]
        assert all(len(x) == nonce_len for x in nonces)
        assert len(set(nonces)) == n, "nonce reuse detected under a single DEK"

    def test_dod_g_same_mapping_re_put_uses_a_fresh_nonce(self, temp_db_path: Path) -> None:
        """§12 (g): re-encrypting the SAME (scope, token) many times draws a
        fresh nonce each time (INSERT OR REPLACE re-encrypts), so a repeated put
        never reuses a nonce under the live DEK."""
        store = _make_store(temp_db_path)
        seen: set[bytes] = set()
        nonce_len = enc_mod._NONCE_BYTES
        for i in range(50):
            store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=f"v{i}@example.test"))
            conn = sqlite3.connect(str(temp_db_path))
            try:
                row = conn.execute(
                    "SELECT payload FROM encrypted_token_mappings"  # noqa: S608
                ).fetchone()
            finally:
                conn.close()
            seen.add(bytes(row[0])[:nonce_len])
        assert len(seen) == 50, "a re-put of the same mapping reused a nonce"
        store.close()


# ===========================================================================
# FIX B (iter-2) — fold scope_index + expires_at into the AEAD AAD
#
# Pre-fix the AAD binds key_id|blind_index|version only, so a raw sqlite3
# UPDATE can (1) relocate a row to a different scope bucket (audit-enumeration
# mis-bucket) or (2) extend expires_at to resurrect a logically-expired row.
# Each surfaces only the row's OWN genuine plaintext (no forged plaintext, no
# confidentiality break) — but it is a real integrity gap on the audit/TTL
# surface of a security-MUST store. Folding both columns into the AAD makes any
# such tamper fail the tag check -> TokenStoreIntegrityError (fail-loud,
# consistent with the existing single-byte-tamper path). These RED tests FAIL
# on the pre-fix code (the relocated/extended row still decrypts) and PASS after
# the AAD change.
# ===========================================================================
class TestFixBUnauthenticatedColumnsFoldedIntoAad:
    def test_nfr_014_raw_update_relocating_scope_index_raises_on_get(
        self, temp_db_path: Path
    ) -> None:
        """A direct sqlite3 UPDATE that rewrites a row's ``scope_index`` (to
        mis-bucket it under another scope for audit enumeration) must make a
        subsequent scoped ``get`` of that row FAIL LOUD — the stored
        ``scope_index`` is bound in the AEAD AAD, so a forged value breaks the
        tag."""
        store = _make_store(temp_db_path)
        store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT))
        store.close()

        conn = sqlite3.connect(str(temp_db_path))
        try:
            conn.execute(
                "UPDATE encrypted_token_mappings SET scope_index = ?",  # noqa: S608
                (b"\x00" * 32,),  # a different (forged) scope bucket
            )
            conn.commit()
        finally:
            conn.close()

        store2 = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            store2.get(SYNTH_TOKEN, scope="s")
        store2.close()

    def test_nfr_014_raw_update_relocating_scope_index_raises_on_scopeless_get(
        self, temp_db_path: Path
    ) -> None:
        """The scope-less scan path must also fail loud when ``scope_index`` was
        forged (it decrypts each row, so the bound AAD must reject the tamper)."""
        store = _make_store(temp_db_path)
        store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT))
        store.close()

        conn = sqlite3.connect(str(temp_db_path))
        try:
            conn.execute(
                "UPDATE encrypted_token_mappings SET scope_index = ?",  # noqa: S608
                (b"\x07" * 32,),
            )
            conn.commit()
        finally:
            conn.close()

        store2 = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            store2.get(SYNTH_TOKEN)  # no scope -> scan
        store2.close()

    def test_nfr_014_raw_update_extending_expires_at_raises_on_get(
        self, temp_db_path: Path
    ) -> None:
        """A direct sqlite3 UPDATE that extends ``expires_at`` (to resurrect a
        logically-expired row) must make a subsequent ``get`` FAIL LOUD — the
        stored ``expires_at`` is bound in the AEAD AAD, so a forged value breaks
        the tag (rather than silently honouring the extended TTL)."""
        now = time.time()
        store = _make_store(temp_db_path)
        # Row that is ALREADY expired as written.
        store.put(
            _mapping(
                scope="s",
                token=SYNTH_TOKEN,
                plaintext=SYNTH_PLAINTEXT,
                created_at=now - 100,
                expires_at=now - 1,
            )
        )
        store.close()

        conn = sqlite3.connect(str(temp_db_path))
        try:
            # Forge a far-future expiry to "resurrect" the row.
            conn.execute(
                "UPDATE encrypted_token_mappings SET expires_at = ?",  # noqa: S608
                (now + 86_400,),
            )
            conn.commit()
        finally:
            conn.close()

        store2 = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            store2.get(SYNTH_TOKEN, scope="s")
        store2.close()

    def test_nfr_014_raw_update_setting_expires_at_to_null_raises(
        self, temp_db_path: Path
    ) -> None:
        """Clearing ``expires_at`` to NULL (the None sentinel must be a DISTINCT
        bound value from any real expiry) also breaks the tag and fails loud."""
        now = time.time()
        store = _make_store(temp_db_path)
        store.put(
            _mapping(
                scope="s",
                token=SYNTH_TOKEN,
                plaintext=SYNTH_PLAINTEXT,
                created_at=now,
                expires_at=now + 3_600,
            )
        )
        store.close()

        conn = sqlite3.connect(str(temp_db_path))
        try:
            conn.execute(
                "UPDATE encrypted_token_mappings SET expires_at = NULL"  # noqa: S608
            )
            conn.commit()
        finally:
            conn.close()

        store2 = _make_store(temp_db_path)
        with pytest.raises(TokenStoreIntegrityError):
            store2.get(SYNTH_TOKEN, scope="s")
        store2.close()

    def test_legitimate_none_expiry_still_decrypts(self, temp_db_path: Path) -> None:
        """Regression: a row legitimately written with ``expires_at=None`` (the
        canonical-None AAD sentinel) round-trips fine — the None binding must
        not break legitimate decrypts."""
        store = _make_store(temp_db_path)
        store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT, expires_at=None))
        got = store.get(SYNTH_TOKEN, scope="s")
        assert got is not None and got.plaintext == SYNTH_PLAINTEXT
        assert got.expires_at is None
        store.close()

    def test_legitimate_expiry_value_still_decrypts(self, temp_db_path: Path) -> None:
        """Regression: a row with a real future ``expires_at`` round-trips fine
        (the value is bound in the AAD but reconstructed from the stored column
        on read, so a legitimate row authenticates)."""
        now = time.time()
        exp = now + 86_400
        store = _make_store(temp_db_path)
        store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT, created_at=now, expires_at=exp))
        got = store.get(SYNTH_TOKEN, scope="s")
        assert got is not None and got.plaintext == SYNTH_PLAINTEXT
        assert got.expires_at is not None
        store.close()

    def test_legitimate_scope_still_lists_after_aad_fold(self, temp_db_path: Path) -> None:
        """Regression: legitimate rows still enumerate by scope after the
        scope_index fold (the stored scope_index is the AAD binding AND the
        query key, so a genuine row is found and authenticates)."""
        store = _make_store(temp_db_path)
        store.put(_mapping(scope="s1", token="<EMAIL_ADDRESS:v1:t1>", plaintext="a@example.test"))
        store.put(_mapping(scope="s1", token="<EMAIL_ADDRESS:v1:t2>", plaintext="b@example.test"))
        store.put(_mapping(scope="s2", token="<EMAIL_ADDRESS:v1:t3>", plaintext="c@example.test"))
        listed = store.list_by_scope("s1")
        assert {m.plaintext for m in listed} == {"a@example.test", "b@example.test"}
        assert len(store.list_by_scope("s2")) == 1
        store.close()

    def test_rewrap_all_rebinds_aad_and_still_round_trips(self, temp_db_path: Path) -> None:
        """Regression: rewrap_all re-encrypts each row (new DEK, new AAD over the
        same scope_index/expires_at), so migrated rows still decrypt."""
        now = time.time()
        store = EncryptedSQLiteTokenStore(
            temp_db_path, key_provider=StaticTestKeyProvider(KEK_A, key_id="k1")
        )
        store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT, created_at=now, expires_at=now + 3_600))
        store.rewrap_all(StaticTestKeyProvider(KEK_B, key_id="k2"))
        got = store.get(SYNTH_TOKEN, scope="s")
        assert got is not None and got.plaintext == SYNTH_PLAINTEXT
        store.close()


# ===========================================================================
# FIX C (iter-2) — wrong-KEK fail-loud consistency on list_by_scope (NFR-015)
#
# Under a wrong KEK, scoped get() correctly RAISES (A4), but pre-fix
# list_by_scope() returned [] silently and count() returned the raw row count.
# Make list_by_scope fail-loud (matching get) when rows exist by scope_index
# but no key unwraps them. count() with no scope is a deliberate raw COUNT(*)
# (cardinality is already leaked by file size) — pin that conscious choice.
# ===========================================================================
class TestFixCWrongKekListByScopeConsistency:
    def test_nfr_015_list_by_scope_fails_loud_under_wrong_kek(
        self, temp_db_path: Path
    ) -> None:
        """list_by_scope must FAIL LOUD (not return []) when the artifact holds
        rows but the supplied KEK cannot unwrap their envelope — matching the
        scoped get() wrong-KEK behaviour (NFR-015 artifact-alone re-join =
        FAIL)."""
        store = _make_store(temp_db_path, kek=KEK_A)
        store.put(_mapping(scope="s", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT))
        store.close()

        wrong = _make_store(temp_db_path, kek=KEK_B)
        with pytest.raises((TokenStoreIntegrityError, TokenizationError)):
            wrong.list_by_scope("s")
        wrong.close()

    def test_list_by_scope_returns_empty_for_genuinely_absent_scope(
        self, temp_db_path: Path
    ) -> None:
        """Regression: a genuinely-empty scope (correct KEK, no rows) still
        returns [] — the fail-loud path must fire ONLY when rows exist but no
        key unwraps, never for a legitimately-empty scope."""
        store = _make_store(temp_db_path, kek=KEK_A)
        store.put(_mapping(scope="present", token=SYNTH_TOKEN, plaintext=SYNTH_PLAINTEXT))
        assert store.list_by_scope("totally-absent-scope") == []
        store.close()

    def test_list_by_scope_empty_store_returns_empty(self, temp_db_path: Path) -> None:
        """Regression: list_by_scope on an empty store (no rows at all) is [],
        never a raise (no owning key ids => nothing unreversible)."""
        store = _make_store(temp_db_path, kek=KEK_A)
        assert store.list_by_scope("any") == []
        store.close()

    def test_count_no_scope_is_deliberate_raw_count_under_wrong_kek(
        self, temp_db_path: Path
    ) -> None:
        """count() with no scope is a DELIBERATE raw COUNT(*): row cardinality is
        already leaked by file size, so it returns the true (live) row count
        even under a wrong KEK and does NOT raise. Pin this conscious choice."""
        store = _make_store(temp_db_path, kek=KEK_A)
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:t1>", plaintext="a@example.test"))
        store.put(_mapping(scope="s", token="<EMAIL_ADDRESS:v1:t2>", plaintext="b@example.test"))
        store.close()

        wrong = _make_store(temp_db_path, kek=KEK_B)
        # No raise; the raw cardinality is intentionally observable.
        assert wrong.count() == 2
        wrong.close()
