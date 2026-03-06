from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass

from pii_anon.errors import TokenizationError
from pii_anon.tokenization.store import TokenMapping, TokenStore

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
except ImportError:  # pragma: no cover - optional dependency
    AESSIV = None  # type: ignore


@dataclass
class TokenRecord:
    entity_type: str
    version: int
    token: str
    scope: str


class TokenizerProvider:
    def tokenize(
        self,
        entity_type: str,
        plaintext: str,
        scope: str,
        version: int,
        key: str,
        *,
        store: TokenStore | None = None,
    ) -> TokenRecord:
        raise NotImplementedError

    def detokenize(
        self,
        token: TokenRecord,
        *,
        key: str,
        store: TokenStore | None = None,
        mapping: dict[str, str] | None = None,
    ) -> str | None:
        raise NotImplementedError


class DeterministicHMACTokenizer(TokenizerProvider):
    def tokenize(
        self,
        entity_type: str,
        plaintext: str,
        scope: str,
        version: int,
        key: str,
        *,
        store: TokenStore | None = None,
    ) -> TokenRecord:
        raw = f"{scope}|v{version}|{entity_type}|{plaintext}".encode("utf-8")
        digest = hmac.new(key.encode("utf-8"), raw, hashlib.sha256).digest()
        packed = base64.urlsafe_b64encode(digest[:18]).decode("utf-8").rstrip("=")
        token = f"<{entity_type}:v{version}:tok_{packed}>"
        record = TokenRecord(entity_type=entity_type, version=version, token=token, scope=scope)

        if store is not None:
            store.put(
                TokenMapping(
                    scope=scope,
                    token=token,
                    plaintext=plaintext,
                    entity_type=entity_type,
                    version=version,
                )
            )
        return record

    def detokenize(
        self,
        token: TokenRecord,
        *,
        key: str,
        store: TokenStore | None = None,
        mapping: dict[str, str] | None = None,
    ) -> str | None:
        if store is not None:
            found = store.get(token.token, scope=token.scope)
            if found is not None:
                return found.plaintext
        if mapping:
            return mapping.get(token.token)
        return None


class AESSIVTokenizer(TokenizerProvider):
    def _build_cipher(self, key: str) -> "AESSIV":
        if AESSIV is None:
            raise TokenizationError(
                "AESSIVTokenizer requires optional dependency `cryptography`. Install with `pip install pii-anon[crypto]`."
            )
        material = hashlib.sha512(key.encode("utf-8")).digest()
        return AESSIV(material)

    def tokenize(
        self,
        entity_type: str,
        plaintext: str,
        scope: str,
        version: int,
        key: str,
        *,
        store: TokenStore | None = None,
    ) -> TokenRecord:
        cipher = self._build_cipher(key)
        associated_data = [scope.encode("utf-8"), f"v{version}".encode("utf-8"), entity_type.encode("utf-8")]
        encrypted = cipher.encrypt(plaintext.encode("utf-8"), associated_data)
        packed = base64.urlsafe_b64encode(encrypted).decode("utf-8").rstrip("=")
        token = f"<{entity_type}:v{version}:aes_{packed}>"
        record = TokenRecord(entity_type=entity_type, version=version, token=token, scope=scope)

        if store is not None:
            store.put(
                TokenMapping(
                    scope=scope,
                    token=token,
                    plaintext=plaintext,
                    entity_type=entity_type,
                    version=version,
                )
            )
        return record

    def detokenize(
        self,
        token: TokenRecord,
        *,
        key: str,
        store: TokenStore | None = None,
        mapping: dict[str, str] | None = None,
    ) -> str | None:
        if store is not None:
            found = store.get(token.token, scope=token.scope)
            if found is not None:
                return found.plaintext

        prefix = f"<{token.entity_type}:v{token.version}:aes_"
        if not token.token.startswith(prefix) or not token.token.endswith(">"):
            if mapping:
                return mapping.get(token.token)
            return None

        encoded = token.token[len(prefix) : -1]
        padding = "=" * ((4 - len(encoded) % 4) % 4)

        try:
            encrypted = base64.urlsafe_b64decode(encoded + padding)
            cipher = self._build_cipher(key)
            associated_data = [
                token.scope.encode("utf-8"),
                f"v{token.version}".encode("utf-8"),
                token.entity_type.encode("utf-8"),
            ]
            plaintext = cipher.decrypt(encrypted, associated_data)
            return plaintext.decode("utf-8")
        except Exception as exc:
            raise TokenizationError("Failed to detokenize AES-SIV token") from exc
