from __future__ import annotations

import base64

import pytest

from pii_anon.errors import TokenizationError
from pii_anon.tokenization import AESSIVTokenizer, DeterministicHMACTokenizer, TokenRecord
from pii_anon.tokenization.providers import TokenizerProvider
from pii_anon.tokenization.store import InMemoryTokenStore, TokenMapping, TokenStore


class BrokenCipher:
    def decrypt(self, encrypted: bytes, associated_data: list[bytes]) -> bytes:
        _ = encrypted, associated_data
        raise ValueError("cannot decrypt")


def test_tokenizer_provider_base_methods_raise() -> None:
    provider = TokenizerProvider()
    with pytest.raises(NotImplementedError):
        provider.tokenize("EMAIL", "alice@example.com", "scope", 1, "k")
    with pytest.raises(NotImplementedError):
        provider.detokenize(
            TokenRecord(entity_type="EMAIL", version=1, token="t", scope="s"),
            key="k",
        )


def test_hmac_detokenize_mapping_branch() -> None:
    tokenizer = DeterministicHMACTokenizer()
    token = tokenizer.tokenize("EMAIL_ADDRESS", "alice@example.com", "scope", 1, "k")
    recovered = tokenizer.detokenize(token, key="k", mapping={token.token: "mapped@example.com"})
    assert recovered == "mapped@example.com"


def test_aes_detokenize_mapping_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = AESSIVTokenizer()

    # invalid token format should fall back to mapping
    invalid = TokenRecord(entity_type="EMAIL_ADDRESS", version=1, token="not-a-token", scope="scope")
    mapped = tokenizer.detokenize(invalid, key="k", mapping={"not-a-token": "fallback"})
    assert mapped == "fallback"

    # valid token shape but decryption failure should raise TokenizationError
    encoded = base64.urlsafe_b64encode(b"ciphertext").decode("utf-8").rstrip("=")
    token = TokenRecord(entity_type="EMAIL_ADDRESS", version=1, token=f"<EMAIL_ADDRESS:v1:aes_{encoded}>", scope="scope")
    monkeypatch.setattr(AESSIVTokenizer, "_build_cipher", lambda self, key: BrokenCipher())
    with pytest.raises(TokenizationError):
        tokenizer.detokenize(token, key="k")


def test_token_store_base_and_inmemory_paths() -> None:
    store = TokenStore()
    with pytest.raises(NotImplementedError):
        store.put(TokenMapping(scope="s", token="t", plaintext="p", entity_type="E", version=1))
    with pytest.raises(NotImplementedError):
        store.get("t")
    assert store.close() is None

    mem = InMemoryTokenStore()
    mapping = TokenMapping(scope="scope", token="tok", plaintext="plain", entity_type="EMAIL", version=1)
    mem.put(mapping)
    assert mem.get("tok", scope="scope") is not None
    assert mem.get("tok", scope="other") is None
    assert mem.get("tok") is not None
