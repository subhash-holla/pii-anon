"""Built-in transformation strategies for PII anonymization.

Six concrete strategies covering the full spectrum of privacy-preserving
transformations:

1. **PlaceholderStrategy** — non-reversible placeholder replacement.
2. **TokenizationStrategy** — reversible HMAC/AES-SIV tokenization.
3. **RedactionStrategy** — full/partial character masking.
4. **GeneralizationStrategy** — reduce precision (age→range, zip→prefix).
5. **SyntheticReplacementStrategy** — realistic fake values.
6. **PerturbationStrategy** — calibrated noise for numerical types.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Callable

from pii_anon.transforms.base import (
    StrategyMetadata,
    TransformContext,
    TransformResult,
    TransformStrategy,
)


# ── 1. Placeholder Strategy ──────────────────────────────────────────────


class PlaceholderStrategy(TransformStrategy):
    """Replace PII with a typed placeholder tag.

    Output format: ``<{entity_type}:anon_{index}>`` (configurable template).
    This wraps the existing anonymization logic from the orchestrator.
    """

    strategy_id = "placeholder"

    def __init__(self, template: str = "<{entity_type}:anon_{index}>") -> None:
        self._template = template

    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        template = context.strategy_params.get("template", self._template)
        try:
            replacement = template.format(
                entity_type=entity_type,
                index=context.placeholder_index,
                cluster_id=context.cluster_id,
            )
        except Exception:
            replacement = f"<{entity_type}:anon_{context.placeholder_index}>"

        return TransformResult(
            replacement=replacement,
            strategy_id=self.strategy_id,
            is_reversible=False,
            metadata={"template": template},
        )

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description="Replace PII with typed placeholder tags (non-reversible).",
            reversible=False,
            format_preserving=False,
        )


# ── 2. Tokenization Strategy ─────────────────────────────────────────────


class TokenizationStrategy(TransformStrategy):
    """Replace PII with deterministic HMAC or AES-SIV tokens.

    Delegates to the existing ``TokenizerProvider`` infrastructure.
    Reversible when a token store is configured.
    """

    strategy_id = "tokenize"

    def __init__(self) -> None:
        # Lazy import to avoid circular deps; used at transform time
        self._tokenizer: Any = None
        self._token_store: Any = None

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            from pii_anon.tokenization import DeterministicHMACTokenizer

            self._tokenizer = DeterministicHMACTokenizer()

    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        self._ensure_tokenizer()

        # Allow injecting store via context params
        store = context.strategy_params.get("token_store", self._token_store)
        key = context.token_key or "default-key"
        version = context.token_version

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        record = self._tokenizer.tokenize(
            entity_type,
            plaintext,
            context.scope,
            version,
            key,
            store=store,
        )
        return TransformResult(
            replacement=record.token,
            strategy_id=self.strategy_id,
            is_reversible=True,
            metadata={"version": version, "scope": context.scope},
        )

    def is_reversible(self) -> bool:
        return True

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description="Deterministic HMAC/AES-SIV tokenization (reversible via token store).",
            reversible=True,
            format_preserving=False,
        )


# ── 3. Redaction Strategy ────────────────────────────────────────────────


class RedactionStrategy(TransformStrategy):
    """Mask PII with configurable character-level redaction.

    Modes
    -----
    - ``full``: Replace entire value with mask chars (e.g. ``"████████"``).
    - ``partial_start``: Reveal first N characters (e.g. ``"Jo██ Sm███"``).
    - ``partial_end``: Reveal last N characters (e.g. ``"████ Smith"``).
    - ``length_preserving``: Replace each character with mask char.
    """

    strategy_id = "redact"

    def __init__(
        self,
        mode: str = "full",
        mask_char: str = "█",
        reveal_count: int = 2,
    ) -> None:
        self._mode = mode
        self._mask_char = mask_char
        self._reveal_count = reveal_count

    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        mode = context.strategy_params.get("mode", self._mode)
        mask_char = context.strategy_params.get("mask_char", self._mask_char)
        reveal_count = int(context.strategy_params.get("reveal_count", self._reveal_count))

        replacement = self._apply_redaction(plaintext, mode, mask_char, reveal_count)
        return TransformResult(
            replacement=replacement,
            strategy_id=self.strategy_id,
            is_reversible=False,
            metadata={"mode": mode, "original_length": len(plaintext)},
        )

    @staticmethod
    def _apply_redaction(text: str, mode: str, mask_char: str, reveal_count: int) -> str:
        if not text:
            return text

        if mode == "full":
            return mask_char * len(text)

        if mode == "length_preserving":
            # Preserve spaces and punctuation structure
            return "".join(mask_char if c.isalnum() else c for c in text)

        if mode == "partial_start":
            if len(text) <= reveal_count:
                return text
            return text[:reveal_count] + mask_char * (len(text) - reveal_count)

        if mode == "partial_end":
            if len(text) <= reveal_count:
                return text
            return mask_char * (len(text) - reveal_count) + text[-reveal_count:]

        # Fallback: full redaction
        return mask_char * len(text)

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description="Character-level masking with configurable reveal modes.",
            reversible=False,
            format_preserving=False,
        )


# ── 4. Generalization Strategy ───────────────────────────────────────────


# Default generalization hierarchies per entity type.
_DEFAULT_HIERARCHIES: dict[str, Any] = {
    "AGE": {"type": "numeric_range", "bucket_size": 10},
    "SALARY": {"type": "numeric_range", "bucket_size": 10_000, "prefix": "$", "separator": ","},
    "DATE_OF_BIRTH": {"type": "date_year_only"},
    "DATE": {"type": "date_year_only"},
    "ZIP_CODE": {"type": "prefix_mask", "keep_chars": 3, "mask_char": "*"},
    "ADDRESS": {"type": "prefix_mask", "keep_chars": 0, "mask_char": "*"},
    "PHONE_NUMBER": {"type": "partial_mask", "keep_start": 4, "keep_end": 0, "mask_char": "*"},
    "EMAIL_ADDRESS": {"type": "email_mask"},
    "PERSON_NAME": {"type": "initials"},
    "LOCATION_COORDINATES": {"type": "truncate_precision", "decimal_places": 1},
    "IP_ADDRESS": {"type": "subnet_mask"},
    "CREDIT_CARD_NUMBER": {"type": "partial_mask", "keep_start": 0, "keep_end": 4, "mask_char": "*"},
}


class GeneralizationStrategy(TransformStrategy):
    """Reduce precision of PII values while preserving utility.

    Each entity type has a built-in generalization hierarchy that can be
    overridden via ``custom_hierarchies``.  Examples:

    - Age ``"32"`` → ``"30-39"``
    - Date ``"1990-05-15"`` → ``"1990"``
    - ZIP ``"10001"`` → ``"100**"``
    - Email ``"john.smith@co.com"`` → ``"j***@co.com"``
    - Name ``"Dr. Jane Smith"`` → ``"Dr. J. S."``
    """

    strategy_id = "generalize"

    def __init__(self, custom_hierarchies: dict[str, Any] | None = None) -> None:
        self._hierarchies = dict(_DEFAULT_HIERARCHIES)
        if custom_hierarchies:
            self._hierarchies.update(custom_hierarchies)

    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        # Merge strategy_params overrides
        hierarchy = self._hierarchies.get(entity_type)
        if hierarchy is None:
            # No hierarchy for this type: fall back to initials-style truncation
            replacement = self._generic_generalize(plaintext)
        else:
            gen_type = hierarchy.get("type", "generic")
            replacement = self._dispatch(plaintext, gen_type, hierarchy)

        return TransformResult(
            replacement=replacement,
            strategy_id=self.strategy_id,
            is_reversible=False,
            metadata={"entity_type": entity_type, "hierarchy_type": hierarchy.get("type") if hierarchy else "generic"},
        )

    def _dispatch(self, text: str, gen_type: str, config: dict[str, Any]) -> str:
        dispatch = {
            "numeric_range": self._generalize_numeric_range,
            "date_year_only": self._generalize_date_year,
            "prefix_mask": self._generalize_prefix_mask,
            "partial_mask": self._generalize_partial_mask,
            "email_mask": self._generalize_email,
            "initials": self._generalize_initials,
            "truncate_precision": self._generalize_truncate_precision,
            "subnet_mask": self._generalize_subnet,
        }
        handler = dispatch.get(gen_type, self._generic_generalize_with_config)
        return handler(text, config)

    @staticmethod
    def _generalize_numeric_range(text: str, config: dict[str, Any]) -> str:
        """'32' → '30-39', '$85,000' → '$80,000-$90,000'."""
        # Extract numeric value
        cleaned = re.sub(r"[^\d.\-]", "", text)
        if not cleaned:
            return "[GENERALIZED]"
        try:
            value = float(cleaned)
        except ValueError:
            return "[GENERALIZED]"

        bucket = int(config.get("bucket_size", 10))
        prefix = config.get("prefix", "")
        separator = config.get("separator", "")

        lower = int(value // bucket) * bucket
        upper = lower + bucket - 1

        def _format_num(n: int) -> str:
            s = str(n)
            if separator and len(s) > 3:
                parts = []
                while s:
                    parts.append(s[-3:])
                    s = s[:-3]
                s = separator.join(reversed(parts))
            else:
                s = str(n)
            return f"{prefix}{s}"

        return f"{_format_num(lower)}-{_format_num(upper)}"

    @staticmethod
    def _generalize_date_year(text: str, _config: dict[str, Any]) -> str:
        """'1990-05-15' → '1990', '05/15/1990' → '1990'."""
        match = re.search(r"(\d{4})", text)
        if match:
            return match.group(1)
        return "[YEAR]"

    @staticmethod
    def _generalize_prefix_mask(text: str, config: dict[str, Any]) -> str:
        """'10001' → '100**'."""
        keep = int(config.get("keep_chars", 3))
        mask = str(config.get("mask_char", "*"))
        if keep <= 0:
            return mask * len(text)
        return text[:keep] + mask * max(0, len(text) - keep)

    @staticmethod
    def _generalize_partial_mask(text: str, config: dict[str, Any]) -> str:
        """'+1-555-123-4567' → '+1-5***********'."""
        keep_start = int(config.get("keep_start", 0))
        keep_end = int(config.get("keep_end", 0))
        mask = str(config.get("mask_char", "*"))
        mid_len = max(0, len(text) - keep_start - keep_end)
        start = text[:keep_start] if keep_start else ""
        end = text[-keep_end:] if keep_end else ""
        return start + mask * mid_len + end

    @staticmethod
    def _generalize_email(text: str, _config: dict[str, Any]) -> str:
        """'john.smith@company.com' → 'j***@company.com'."""
        at_idx = text.find("@")
        if at_idx <= 0:
            return "***@[DOMAIN]"
        local = text[:at_idx]
        domain = text[at_idx:]
        if len(local) <= 1:
            return f"{local}***{domain}"
        return f"{local[0]}***{domain}"

    @staticmethod
    def _generalize_initials(text: str, _config: dict[str, Any]) -> str:
        """'Dr. Jane Smith' → 'Dr. J. S.', 'Jane' → 'J.'."""
        parts = text.split()
        if not parts:
            return "[NAME]"

        result = []
        # Common honorifics to preserve
        honorifics = {"mr", "mrs", "ms", "dr", "prof", "sir", "rev"}

        for part in parts:
            cleaned = part.rstrip(".,")
            if cleaned.lower() in honorifics:
                result.append(part)
            elif part and part[0].isalpha():
                result.append(f"{part[0].upper()}.")
            else:
                result.append(part)
        return " ".join(result)

    @staticmethod
    def _generalize_truncate_precision(text: str, config: dict[str, Any]) -> str:
        """'40.7128,-74.0060' → '40.7,-74.0'."""
        places = int(config.get("decimal_places", 1))
        parts = re.split(r"([,\s]+)", text)
        result = []
        for part in parts:
            try:
                val = float(part.strip())
                result.append(f"{val:.{places}f}")
            except ValueError:
                result.append(part)
        return "".join(result)

    @staticmethod
    def _generalize_subnet(text: str, _config: dict[str, Any]) -> str:
        """'192.168.1.100' → '192.168.1.0/24'."""
        parts = text.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        return text  # IPv6 or other format: return unchanged

    @staticmethod
    def _generic_generalize(text: str) -> str:
        """Fallback: truncate to first char + asterisks."""
        if len(text) <= 1:
            return "*"
        return text[0] + "*" * (len(text) - 1)

    @staticmethod
    def _generic_generalize_with_config(text: str, _config: dict[str, Any]) -> str:
        if len(text) <= 1:
            return "*"
        return text[0] + "*" * (len(text) - 1)

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description="Reduce PII precision while preserving utility (ranges, initials, truncation).",
            reversible=False,
            format_preserving=True,
            supports_entity_types=list(self._hierarchies.keys()),
        )


# ── 5. Synthetic Replacement Strategy ────────────────────────────────────

# Built-in name pools (gender-neutral, multi-locale).  Kept small to avoid
# bloating the package; users can extend via ``custom_pools``.

_NAME_POOLS: dict[str, dict[str, list[str]]] = {
    "en": {
        "first": [
            "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
            "Avery", "Harper", "Parker", "Drew", "Logan", "Sage", "Cameron",
            "Emerson", "Hayden", "Rowan", "Ellis", "Finley", "Blake",
            "Oakley", "Skyler", "Reese", "Dakota", "Phoenix", "Adrian",
            "Sawyer", "River", "Marlowe", "Lennox", "Remington", "Sterling",
        ],
        "last": [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor",
            "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson",
            "White", "Harris", "Clark", "Lewis", "Walker", "Hall", "Young",
            "Allen", "King", "Wright", "Scott", "Green", "Baker", "Adams",
        ],
    },
    "es": {
        "first": [
            "Alejandro", "Sofía", "Carlos", "Valentina", "Diego", "Camila",
            "Mateo", "Lucía", "Santiago", "Isabella", "Sebastián", "Mariana",
            "Andrés", "Gabriela", "Nicolás", "Paula", "Daniel", "Andrea",
        ],
        "last": [
            "García", "Rodríguez", "Martínez", "López", "González", "Hernández",
            "Pérez", "Sánchez", "Ramírez", "Torres", "Flores", "Rivera",
        ],
    },
    "fr": {
        "first": [
            "Camille", "Antoine", "Manon", "Lucas", "Chloé", "Hugo",
            "Léa", "Gabriel", "Emma", "Louis", "Jade", "Raphaël",
            "Louise", "Arthur", "Alice", "Jules", "Lina", "Adam",
        ],
        "last": [
            "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard",
            "Petit", "Durand", "Leroy", "Moreau", "Simon", "Laurent",
        ],
    },
    "de": {
        "first": [
            "Maximilian", "Sophie", "Alexander", "Marie", "Paul", "Emma",
            "Leon", "Hannah", "Lukas", "Mia", "Felix", "Lena",
            "Jonas", "Anna", "Tim", "Laura", "Ben", "Lara",
        ],
        "last": [
            "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer",
            "Wagner", "Becker", "Schulz", "Hoffmann", "Koch", "Richter",
        ],
    },
}

_CITY_POOLS: dict[str, list[str]] = {
    "en": [
        "Springfield", "Riverside", "Fairview", "Georgetown", "Clinton",
        "Madison", "Salem", "Franklin", "Greenville", "Arlington",
        "Bristol", "Ashland", "Burlington", "Chester", "Dayton",
    ],
    "es": ["San Miguel", "Santa Cruz", "La Paz", "San José", "Monterrey"],
    "fr": ["Saint-Denis", "Saint-Étienne", "Villeurbanne", "Clermont", "Nanterre"],
    "de": ["Neustadt", "Friedrichshafen", "Waldkirch", "Steinbach", "Rosenheim"],
}


@dataclass
class _SyntheticPools:
    """Aggregated pools for consistent synthetic generation."""

    names: dict[str, dict[str, list[str]]] = field(default_factory=lambda: dict(_NAME_POOLS))
    cities: dict[str, list[str]] = field(default_factory=lambda: dict(_CITY_POOLS))


class SyntheticReplacementStrategy(TransformStrategy):
    """Generate realistic fake PII values.

    Uses deterministic HMAC-keyed pool selection so that the same
    plaintext always maps to the same synthetic value (within a scope).

    Entity-type support:
    - PERSON_NAME: locale-aware fake names
    - EMAIL_ADDRESS: generates email from synthetic name
    - PHONE_NUMBER: format-preserving random digits
    - ADDRESS / ZIP_CODE: fake cities
    - CREDIT_CARD_NUMBER: Luhn-valid fake card numbers
    - DATE_OF_BIRTH: shifted date
    """

    strategy_id = "synthetic"

    def __init__(self, custom_pools: dict[str, Any] | None = None) -> None:
        self._pools = _SyntheticPools()
        if custom_pools:
            if "names" in custom_pools:
                self._pools.names.update(custom_pools["names"])
            if "cities" in custom_pools:
                self._pools.cities.update(custom_pools["cities"])

    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        key = context.token_key or "default-synthetic-key"
        seed = self._derive_seed(key, context.scope, plaintext)

        handler = self._get_handler(entity_type)
        replacement = handler(plaintext, context.language, seed)

        return TransformResult(
            replacement=replacement,
            strategy_id=self.strategy_id,
            is_reversible=False,
            metadata={"locale": context.language, "entity_type": entity_type},
        )

    def _get_handler(self, entity_type: str) -> Callable[[str, str, int], str]:
        handlers = {
            "PERSON_NAME": self._synthetic_name,
            "EMAIL_ADDRESS": self._synthetic_email,
            "PHONE_NUMBER": self._synthetic_phone,
            "ADDRESS": self._synthetic_address,
            "ZIP_CODE": self._synthetic_zip,
            "CREDIT_CARD_NUMBER": self._synthetic_credit_card,
            "DATE_OF_BIRTH": self._synthetic_date,
            "DATE": self._synthetic_date,
        }
        return handlers.get(entity_type, self._synthetic_generic)

    @staticmethod
    def _derive_seed(key: str, scope: str, plaintext: str) -> int:
        """Deterministic seed from key + scope + plaintext."""
        raw = f"{key}|{scope}|{plaintext}".encode("utf-8")
        digest = hmac.new(key.encode("utf-8"), raw, hashlib.sha256).digest()
        return int(struct.unpack(">Q", digest[:8])[0])

    def _synthetic_name(self, plaintext: str, language: str, seed: int) -> str:
        pool = self._pools.names.get(language, self._pools.names.get("en", {"first": ["Alex"], "last": ["Smith"]}))
        first_pool = pool.get("first", ["Alex"])
        last_pool = pool.get("last", ["Smith"])
        first = first_pool[seed % len(first_pool)]
        last = last_pool[(seed >> 16) % len(last_pool)]
        # Try to match structure: if input has honorific, preserve format
        parts = plaintext.split()
        honorifics = {"mr", "mrs", "ms", "dr", "prof", "sir", "rev"}
        if parts and parts[0].rstrip(".,").lower() in honorifics:
            return f"{parts[0]} {first} {last}"
        if len(parts) == 1:
            return first
        return f"{first} {last}"

    def _synthetic_email(self, plaintext: str, language: str, seed: int) -> str:
        pool = self._pools.names.get(language, self._pools.names.get("en", {"first": ["alex"], "last": ["smith"]}))
        first = pool.get("first", ["alex"])[seed % len(pool.get("first", ["alex"]))].lower()
        last = pool.get("last", ["smith"])[(seed >> 16) % len(pool.get("last", ["smith"]))].lower()
        domains = ["example.com", "example.org", "example.net"]
        domain = domains[(seed >> 32) % len(domains)]
        sep = "." if "." in plaintext.split("@")[0] else ""
        return f"{first}{sep}{last}@{domain}"

    @staticmethod
    def _synthetic_phone(plaintext: str, _language: str, seed: int) -> str:
        """Format-preserving: replace digits but keep structure."""
        result = []
        digit_seed = seed
        for ch in plaintext:
            if ch.isdigit():
                result.append(str(digit_seed % 10))
                digit_seed = digit_seed // 10 + 7
            else:
                result.append(ch)
        return "".join(result)

    def _synthetic_address(self, plaintext: str, language: str, seed: int) -> str:
        city_pool = self._pools.cities.get(language, self._pools.cities.get("en", ["Springfield"]))
        city = city_pool[seed % len(city_pool)]
        num = (seed % 9000) + 100
        streets = ["Main St", "Oak Ave", "Elm Dr", "Park Rd", "Cedar Ln", "Maple Blvd"]
        street = streets[(seed >> 16) % len(streets)]
        return f"{num} {street}, {city}"

    @staticmethod
    def _synthetic_zip(plaintext: str, _language: str, seed: int) -> str:
        # Preserve format length
        digits = len(re.sub(r"\D", "", plaintext))
        if digits <= 0:
            digits = 5
        num = seed % (10**digits)
        result = str(num).zfill(digits)
        # If original had a dash (e.g., "12345-6789"), add it back
        if "-" in plaintext:
            parts = plaintext.split("-")
            p1_len = len(re.sub(r"\D", "", parts[0]))
            return f"{result[:p1_len]}-{result[p1_len:]}"
        return result

    @staticmethod
    def _synthetic_credit_card(_plaintext: str, _language: str, seed: int) -> str:
        """Generate a Luhn-valid fake credit card number."""
        # Start with 4 (Visa-like prefix)
        digits = [4]
        s = seed
        for _ in range(14):
            digits.append(s % 10)
            s = s // 10 + 3
        # Compute Luhn check digit
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        check = (10 - (total % 10)) % 10
        digits.append(check)
        card = "".join(str(d) for d in digits)
        return f"{card[:4]}-{card[4:8]}-{card[8:12]}-{card[12:16]}"

    @staticmethod
    def _synthetic_date(plaintext: str, _language: str, seed: int) -> str:
        """Shift date by a deterministic offset (±2 years)."""
        # Try to parse YYYY-MM-DD or similar
        match = re.search(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", plaintext)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            year_offset = (seed % 5) - 2  # -2 to +2
            month_offset = (seed >> 8) % 3
            new_year = year + year_offset
            new_month = max(1, min(12, month + month_offset))
            new_day = max(1, min(28, day))  # safe day
            sep = plaintext[match.start(1) + 4]
            return f"{new_year}{sep}{new_month:02d}{sep}{new_day:02d}"
        return plaintext  # Can't parse: return unchanged

    @staticmethod
    def _synthetic_generic(plaintext: str, _language: str, seed: int) -> str:
        """Fallback: hash-based replacement preserving length."""
        digest = hashlib.sha256(f"{seed}|{plaintext}".encode()).hexdigest()
        # Use hex characters, matching original length
        return digest[: len(plaintext)]

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description="Generate realistic fake PII values (deterministic, locale-aware).",
            reversible=False,
            format_preserving=True,
            supports_entity_types=[
                "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS",
                "ZIP_CODE", "CREDIT_CARD_NUMBER", "DATE_OF_BIRTH", "DATE",
            ],
        )


# ── 6. Perturbation Strategy ─────────────────────────────────────────────


class PerturbationStrategy(TransformStrategy):
    """Add calibrated noise to numerical PII values.

    Supports ε-differential privacy via Laplace mechanism for integer
    values and Gaussian noise for continuous values.

    Supported entity types:
    - AGE: Laplace noise (ε-DP)
    - SALARY: Gaussian noise
    - LOCATION_COORDINATES: random displacement within radius
    - DATE_OF_BIRTH / DATE: shift by random days
    """

    strategy_id = "perturb"

    def __init__(self, epsilon: float = 1.0, sigma: float = 0.1) -> None:
        self._epsilon = epsilon
        self._sigma = sigma

    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        epsilon = float(context.strategy_params.get("epsilon", self._epsilon))
        sigma = float(context.strategy_params.get("sigma", self._sigma))
        seed = self._derive_seed(context)

        handler = self._get_handler(entity_type)
        replacement, noise_meta = handler(plaintext, seed, epsilon, sigma)

        return TransformResult(
            replacement=replacement,
            strategy_id=self.strategy_id,
            is_reversible=False,
            metadata={
                "epsilon": epsilon,
                "sigma": sigma,
                "entity_type": entity_type,
                **noise_meta,
            },
        )

    @staticmethod
    def _derive_seed(context: TransformContext) -> int:
        raw = f"{context.scope}|{context.mention_index}|{context.plaintext}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")

    def _get_handler(self, entity_type: str) -> Callable[[str, int, float, float], tuple[str, dict[str, Any]]]:
        handlers = {
            "AGE": self._perturb_age,
            "SALARY": self._perturb_salary,
            "LOCATION_COORDINATES": self._perturb_coordinates,
            "DATE_OF_BIRTH": self._perturb_date,
            "DATE": self._perturb_date,
        }
        return handlers.get(entity_type, self._perturb_generic)

    @staticmethod
    def _laplace_sample(seed: int, scale: float) -> float:
        """Deterministic Laplace sample from a seed."""
        # Map seed to uniform (0, 1)
        u = ((seed % 1_000_000) / 1_000_000.0) - 0.5
        if u == 0:
            u = 0.001
        sign = 1.0 if u >= 0 else -1.0
        return -scale * sign * math.log(1.0 - 2.0 * abs(u))

    @staticmethod
    def _gaussian_sample(seed: int, sigma: float) -> float:
        """Deterministic Gaussian approximation from seed."""
        # Box-Muller using seed-derived uniforms
        u1 = max(1e-10, ((seed % 1_000_000) + 1) / 1_000_001.0)
        u2 = (((seed >> 20) % 1_000_000) + 1) / 1_000_001.0
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return z * sigma

    @staticmethod
    def _perturb_age(text: str, seed: int, epsilon: float, _sigma: float) -> tuple[str, dict[str, Any]]:
        """Add Laplace noise to age value."""
        cleaned = re.sub(r"\D", "", text)
        if not cleaned:
            return text, {"noise": 0}
        age = int(cleaned)
        sensitivity = 1.0  # age changes by at most 1 per record
        scale = sensitivity / max(epsilon, 0.01)
        noise = PerturbationStrategy._laplace_sample(seed, scale)
        new_age = max(0, min(120, round(age + noise)))
        return str(new_age), {"noise": round(noise, 2), "mechanism": "laplace"}

    @staticmethod
    def _perturb_salary(text: str, seed: int, _epsilon: float, sigma: float) -> tuple[str, dict[str, Any]]:
        """Add Gaussian noise to salary value."""
        cleaned = re.sub(r"[^\d.]", "", text)
        if not cleaned:
            return text, {"noise": 0}
        try:
            salary = float(cleaned)
        except ValueError:
            return text, {"noise": 0}
        noise = PerturbationStrategy._gaussian_sample(seed, sigma * salary)
        new_salary = max(0, round(salary + noise, 2))
        # Reconstruct format
        prefix = "$" if "$" in text else ""
        if "," in text:
            formatted = f"{new_salary:,.0f}"
        else:
            formatted = f"{new_salary:.0f}"
        return f"{prefix}{formatted}", {"noise": round(noise, 2), "mechanism": "gaussian"}

    @staticmethod
    def _perturb_coordinates(text: str, seed: int, _epsilon: float, sigma: float) -> tuple[str, dict[str, Any]]:
        """Add random displacement to GPS coordinates."""
        parts = re.findall(r"-?\d+\.?\d*", text)
        if len(parts) < 2:
            return text, {"noise": 0}
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            return text, {"noise": 0}
        noise_lat = PerturbationStrategy._gaussian_sample(seed, sigma)
        noise_lon = PerturbationStrategy._gaussian_sample(seed >> 20, sigma)
        new_lat = max(-90, min(90, lat + noise_lat))
        new_lon = max(-180, min(180, lon + noise_lon))
        return f"{new_lat:.4f},{new_lon:.4f}", {
            "noise_lat": round(noise_lat, 4),
            "noise_lon": round(noise_lon, 4),
            "mechanism": "gaussian",
        }

    @staticmethod
    def _perturb_date(text: str, seed: int, _epsilon: float, _sigma: float) -> tuple[str, dict[str, Any]]:
        """Shift date by deterministic random days."""
        match = re.search(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", text)
        if not match:
            return text, {"noise": 0}
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        shift_days = (seed % 61) - 30  # ±30 days
        # Simple approximation: shift month boundaries
        day += shift_days
        while day > 28:
            day -= 28
            month += 1
        while day < 1:
            day += 28
            month -= 1
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
        day = max(1, min(28, day))
        sep = text[match.start(1) + 4]
        return f"{year}{sep}{month:02d}{sep}{day:02d}", {
            "shift_days": shift_days,
            "mechanism": "uniform_shift",
        }

    @staticmethod
    def _perturb_generic(text: str, _seed: int, _epsilon: float, _sigma: float) -> tuple[str, dict[str, Any]]:
        """Fallback: no perturbation for unsupported types."""
        return text, {"noise": 0, "mechanism": "none"}

    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description="Add calibrated noise to numerical PII (ε-DP Laplace, Gaussian).",
            reversible=False,
            format_preserving=True,
            supports_entity_types=[
                "AGE", "SALARY", "LOCATION_COORDINATES", "DATE_OF_BIRTH", "DATE",
            ],
        )
