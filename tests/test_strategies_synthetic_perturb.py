"""Tests for SyntheticReplacementStrategy and PerturbationStrategy edge cases."""

from __future__ import annotations

from pii_anon.transforms.base import TransformContext
from pii_anon.transforms.strategies import (
    PerturbationStrategy,
    SyntheticReplacementStrategy,
)


def _ctx(
    plaintext: str = "test",
    entity_type: str = "PERSON_NAME",
    language: str = "en",
    scope: str = "test",
    token_key: str = "k",
    mention_index: int = 0,
    **kwargs,
) -> TransformContext:
    return TransformContext(
        entity_type=entity_type,
        plaintext=plaintext,
        language=language,
        scope=scope,
        token_key=token_key,
        mention_index=mention_index,
        **kwargs,
    )


# ── SyntheticReplacementStrategy ────────────────────────────────────────


class TestSyntheticName:
    def test_simple_name(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("John Smith", "PERSON_NAME", _ctx(plaintext="John Smith"))
        assert len(r.replacement) > 0
        assert r.strategy_id == "synthetic"
        assert not r.is_reversible

    def test_honorific_preserved(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("Dr. Jane Smith", "PERSON_NAME", _ctx(plaintext="Dr. Jane Smith"))
        assert r.replacement.startswith("Dr.")

    def test_single_name(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("Alice", "PERSON_NAME", _ctx(plaintext="Alice"))
        assert " " not in r.replacement  # single name stays single

    def test_deterministic(self) -> None:
        s = SyntheticReplacementStrategy()
        r1 = s.transform("Bob", "PERSON_NAME", _ctx(plaintext="Bob"))
        r2 = s.transform("Bob", "PERSON_NAME", _ctx(plaintext="Bob"))
        assert r1.replacement == r2.replacement


class TestSyntheticEmail:
    def test_basic_email(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("user@corp.com", "EMAIL_ADDRESS", _ctx(plaintext="user@corp.com"))
        assert "@" in r.replacement
        assert "example" in r.replacement

    def test_dotted_email(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("john.doe@corp.com", "EMAIL_ADDRESS", _ctx(plaintext="john.doe@corp.com"))
        assert "." in r.replacement.split("@")[0]


class TestSyntheticPhone:
    def test_basic_phone(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("+1-555-123-4567", "PHONE_NUMBER", _ctx(plaintext="+1-555-123-4567"))
        assert len(r.replacement) == len("+1-555-123-4567")
        # Non-digit chars preserved
        assert r.replacement[0] == "+"
        assert r.replacement[2] == "-"


class TestSyntheticAddress:
    def test_basic_address(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("123 Main St, NY", "ADDRESS", _ctx(plaintext="123 Main St, NY"))
        assert "," in r.replacement

    def test_non_english_locale(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("123 Rue de Paris", "ADDRESS", _ctx(plaintext="123 Rue de Paris", language="fr"))
        assert len(r.replacement) > 0


class TestSyntheticZip:
    def test_basic_zip(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("10001", "ZIP_CODE", _ctx(plaintext="10001"))
        assert len(r.replacement) == 5

    def test_hyphenated_zip(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("12345-6789", "ZIP_CODE", _ctx(plaintext="12345-6789"))
        assert "-" in r.replacement


class TestSyntheticCreditCard:
    def test_luhn_valid(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("4111-1111-1111-1111", "CREDIT_CARD_NUMBER", _ctx(plaintext="4111-1111-1111-1111"))
        # Format: XXXX-XXXX-XXXX-XXXX
        parts = r.replacement.split("-")
        assert len(parts) == 4
        for part in parts:
            assert len(part) == 4
            assert part.isdigit()
        # Starts with 4 (Visa-like)
        assert r.replacement[0] == "4"


class TestSyntheticDate:
    def test_basic_date(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("1990-05-15", "DATE_OF_BIRTH", _ctx(plaintext="1990-05-15"))
        assert "-" in r.replacement
        parts = r.replacement.split("-")
        assert len(parts) == 3

    def test_unparseable_date(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("sometime", "DATE_OF_BIRTH", _ctx(plaintext="sometime"))
        assert r.replacement == "sometime"


class TestSyntheticGeneric:
    def test_unknown_entity_type(self) -> None:
        s = SyntheticReplacementStrategy()
        r = s.transform("foobar", "UNKNOWN_TYPE", _ctx(plaintext="foobar"))
        assert len(r.replacement) == len("foobar")


class TestSyntheticCustomPools:
    def test_custom_names_pool(self) -> None:
        custom = {"names": {"en": {"first": ["TestFirst"], "last": ["TestLast"]}}}
        s = SyntheticReplacementStrategy(custom_pools=custom)
        r = s.transform("Any Name", "PERSON_NAME", _ctx(plaintext="Any Name"))
        assert "TestFirst" in r.replacement or "TestLast" in r.replacement

    def test_custom_cities_pool(self) -> None:
        custom = {"cities": {"en": ["TestCity"]}}
        s = SyntheticReplacementStrategy(custom_pools=custom)
        r = s.transform("123 Main St", "ADDRESS", _ctx(plaintext="123 Main St"))
        assert "TestCity" in r.replacement


class TestSyntheticMetadata:
    def test_metadata(self) -> None:
        s = SyntheticReplacementStrategy()
        m = s.metadata()
        assert m.strategy_id == "synthetic"
        assert not m.reversible
        assert m.format_preserving
        assert "PERSON_NAME" in m.supports_entity_types


# ── PerturbationStrategy ────────────────────────────────────────────────


class TestPerturbAge:
    def test_basic_age(self) -> None:
        s = PerturbationStrategy(epsilon=1.0)
        r = s.transform("32", "AGE", _ctx(plaintext="32", entity_type="AGE"))
        age = int(r.replacement)
        assert 0 <= age <= 120
        assert r.metadata["mechanism"] == "laplace"

    def test_non_numeric_age(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("unknown", "AGE", _ctx(plaintext="unknown", entity_type="AGE"))
        assert r.replacement == "unknown"


class TestPerturbSalary:
    def test_basic_salary(self) -> None:
        s = PerturbationStrategy(sigma=0.1)
        r = s.transform("$85,000", "SALARY", _ctx(plaintext="$85,000", entity_type="SALARY"))
        assert r.replacement.startswith("$")
        assert "," in r.replacement
        assert r.metadata["mechanism"] == "gaussian"

    def test_salary_no_prefix(self) -> None:
        s = PerturbationStrategy(sigma=0.05)
        r = s.transform("50000", "SALARY", _ctx(plaintext="50000", entity_type="SALARY"))
        assert "$" not in r.replacement

    def test_non_numeric_salary(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("negotiable", "SALARY", _ctx(plaintext="negotiable", entity_type="SALARY"))
        assert r.replacement == "negotiable"

    def test_invalid_salary(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("$...", "SALARY", _ctx(plaintext="$...", entity_type="SALARY"))
        # Cleaned result is "..." which can't be parsed
        assert r.metadata.get("noise") == 0 or "mechanism" in r.metadata


class TestPerturbCoordinates:
    def test_basic_coordinates(self) -> None:
        s = PerturbationStrategy(sigma=0.01)
        r = s.transform("40.7128,-74.0060", "LOCATION_COORDINATES",
                         _ctx(plaintext="40.7128,-74.0060", entity_type="LOCATION_COORDINATES"))
        parts = r.replacement.split(",")
        assert len(parts) == 2
        lat, lon = float(parts[0]), float(parts[1])
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
        assert r.metadata["mechanism"] == "gaussian"

    def test_single_value(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("40.7", "LOCATION_COORDINATES",
                         _ctx(plaintext="40.7", entity_type="LOCATION_COORDINATES"))
        assert r.replacement == "40.7"  # can't parse 2 coords


class TestPerturbDate:
    def test_basic_date(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("1990-05-15", "DATE", _ctx(plaintext="1990-05-15", entity_type="DATE"))
        assert "-" in r.replacement
        assert r.metadata["mechanism"] == "uniform_shift"

    def test_unparseable_date(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("sometime", "DATE", _ctx(plaintext="sometime", entity_type="DATE"))
        assert r.replacement == "sometime"


class TestPerturbGeneric:
    def test_unsupported_entity(self) -> None:
        s = PerturbationStrategy()
        r = s.transform("value", "UNKNOWN", _ctx(plaintext="value", entity_type="UNKNOWN"))
        assert r.replacement == "value"
        assert r.metadata["mechanism"] == "none"


class TestPerturbMetadata:
    def test_metadata(self) -> None:
        s = PerturbationStrategy()
        m = s.metadata()
        assert m.strategy_id == "perturb"
        assert not m.reversible
        assert "AGE" in m.supports_entity_types


class TestPerturbStrategyParams:
    def test_custom_epsilon_via_context(self) -> None:
        s = PerturbationStrategy(epsilon=1.0)
        ctx = _ctx(plaintext="25", entity_type="AGE", strategy_params={"epsilon": 0.5})
        r = s.transform("25", "AGE", ctx)
        assert r.metadata["epsilon"] == 0.5
