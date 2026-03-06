from pathlib import Path

import pytest

from pii_anon.errors import TokenizationError
from pii_anon.tokenization import AESSIVTokenizer, DeterministicHMACTokenizer, InMemoryTokenStore, SQLiteTokenStore


def test_hmac_tokenization_round_trip_in_store() -> None:
    tokenizer = DeterministicHMACTokenizer()
    store = InMemoryTokenStore()
    token = tokenizer.tokenize("EMAIL_ADDRESS", "alice@example.com", "scope", 1, "key", store=store)

    recovered = tokenizer.detokenize(token, key="key", store=store)
    assert recovered == "alice@example.com"


def test_sqlite_store_round_trip(tmp_path: Path) -> None:
    store = SQLiteTokenStore(tmp_path / "tokens.db")
    tokenizer = DeterministicHMACTokenizer()
    token = tokenizer.tokenize("PHONE_NUMBER", "555-123-4567", "scope", 1, "k", store=store)
    assert tokenizer.detokenize(token, key="k", store=store) == "555-123-4567"


def test_aes_siv_optional_dependency() -> None:
    tokenizer = AESSIVTokenizer()
    try:
        token = tokenizer.tokenize("EMAIL_ADDRESS", "alice@example.com", "scope", 1, "k")
        recovered = tokenizer.detokenize(token, key="k")
        assert recovered == "alice@example.com"
    except TokenizationError:
        pytest.skip("cryptography not installed")
