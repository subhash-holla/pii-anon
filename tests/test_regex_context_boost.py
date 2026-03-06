"""Tests for context-aware confidence boosting (Enhancement 1).

Covers: context word extraction, context detection, confidence boost/penalty,
integration with detect() for multiple entity types, and edge cases.
"""

from __future__ import annotations

from pii_anon.engines.regex_adapter import (
    RegexEngineAdapter,
    _CONTEXT_BOOST,
    _CONTEXT_PENALTY,
    _CONTEXT_WINDOW,
    _CONTEXT_WORDS,
    _HIGH_FP_TYPES,
)


# ---------------------------------------------------------------------------
# _extract_context
# ---------------------------------------------------------------------------


class TestExtractContext:
    def test_basic_extraction(self) -> None:
        text = "This is a sample text with SSN 123-45-6789 inside."
        ctx = RegexEngineAdapter._extract_context(text, 30, 41)
        assert "ssn" in ctx
        assert ctx == ctx.lower()

    def test_clamps_to_start(self) -> None:
        text = "SSN 123-45-6789 is here."
        ctx = RegexEngineAdapter._extract_context(text, 0, 15)
        assert "ssn" in ctx

    def test_clamps_to_end(self) -> None:
        text = "My email is alice@example.com"
        ctx = RegexEngineAdapter._extract_context(text, 12, len(text))
        assert "email" in ctx

    def test_window_size(self) -> None:
        text = "A" * 200
        ctx = RegexEngineAdapter._extract_context(text, 100, 110)
        expected_len = min(len(text), 110 + _CONTEXT_WINDOW) - max(0, 100 - _CONTEXT_WINDOW)
        assert len(ctx) == expected_len


# ---------------------------------------------------------------------------
# _has_context_words
# ---------------------------------------------------------------------------


class TestHasContextWords:
    def test_ssn_context_detected(self) -> None:
        assert RegexEngineAdapter._has_context_words("US_SSN", "social security number is")

    def test_ssn_no_context(self) -> None:
        assert not RegexEngineAdapter._has_context_words("US_SSN", "the quick brown fox")

    def test_credit_card_context(self) -> None:
        assert RegexEngineAdapter._has_context_words("CREDIT_CARD", "credit card number")

    def test_email_context(self) -> None:
        assert RegexEngineAdapter._has_context_words("EMAIL_ADDRESS", "send an email to")

    def test_phone_context(self) -> None:
        assert RegexEngineAdapter._has_context_words("PHONE_NUMBER", "call me at phone")

    def test_ip_context(self) -> None:
        assert RegexEngineAdapter._has_context_words("IP_ADDRESS", "server ip address")

    def test_iban_context(self) -> None:
        assert RegexEngineAdapter._has_context_words("IBAN", "international bank transfer")

    def test_person_context(self) -> None:
        assert RegexEngineAdapter._has_context_words("PERSON_NAME", "patient name is")

    def test_unknown_entity_returns_false(self) -> None:
        assert not RegexEngineAdapter._has_context_words("UNKNOWN_TYPE", "anything here")


# ---------------------------------------------------------------------------
# _adjust_confidence
# ---------------------------------------------------------------------------


class TestAdjustConfidence:
    def test_boost_with_context(self) -> None:
        text = "Social security number: 123-45-6789"
        conf = RegexEngineAdapter._adjust_confidence("US_SSN", 0.97, text, 24, 35)
        assert conf == min(0.99, 0.97 + _CONTEXT_BOOST)

    def test_penalty_high_fp_no_context(self) -> None:
        """US_SSN without context words should be penalized."""
        text = "The value is 123-45-6789 in the database."
        conf = RegexEngineAdapter._adjust_confidence("US_SSN", 0.97, text, 14, 25)
        assert conf == 0.97 - _CONTEXT_PENALTY

    def test_no_penalty_low_fp_no_context(self) -> None:
        """Non-high-FP types without context should retain base confidence."""
        text = "The number is 4111111111111111 here."
        conf = RegexEngineAdapter._adjust_confidence("CREDIT_CARD", 0.94, text, 14, 30)
        # CREDIT_CARD is not in _HIGH_FP_TYPES, so no penalty
        assert conf == 0.94

    def test_boost_capped_at_099(self) -> None:
        text = "email address: test@example.com"
        conf = RegexEngineAdapter._adjust_confidence("EMAIL_ADDRESS", 0.99, text, 15, 31)
        assert conf <= 0.99

    def test_penalty_floor_at_050(self) -> None:
        text = "Random text 123-45-6789 more random text."
        conf = RegexEngineAdapter._adjust_confidence("US_SSN", 0.52, text, 12, 23)
        assert conf >= 0.50

    def test_person_name_penalized_without_context(self) -> None:
        assert "PERSON_NAME" in _HIGH_FP_TYPES
        text = "John Smith went to the store."
        conf = RegexEngineAdapter._adjust_confidence("PERSON_NAME", 0.84, text, 0, 10)
        assert conf < 0.84

    def test_person_name_boosted_with_context(self) -> None:
        text = "Patient name is John Smith in the system."
        conf = RegexEngineAdapter._adjust_confidence("PERSON_NAME", 0.84, text, 16, 26)
        assert conf > 0.84


# ---------------------------------------------------------------------------
# Integration: context boosting in detect()
# ---------------------------------------------------------------------------


class TestContextBoostInDetect:
    def _detect(self, text: str, **ctx: object) -> list:
        adapter = RegexEngineAdapter()
        context = {"language": "en", **ctx}
        return adapter.detect({"text": text}, context)

    def test_ssn_with_context_higher_than_without(self) -> None:
        with_ctx = self._detect("Social security number: 123-45-6789")
        without_ctx = self._detect("Reference code 123-45-6789 in file.")
        ssn_with = [f for f in with_ctx if f.entity_type == "US_SSN"]
        ssn_without = [f for f in without_ctx if f.entity_type == "US_SSN"]
        assert len(ssn_with) >= 1
        assert len(ssn_without) >= 1
        assert ssn_with[0].confidence > ssn_without[0].confidence

    def test_email_with_context_boosted(self) -> None:
        findings = self._detect("Send email to alice@example.com for details.")
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) >= 1
        assert emails[0].confidence >= 0.99

    def test_phone_with_context_boosted(self) -> None:
        findings = self._detect("Call me at phone 555-123-4567 anytime.")
        phones = [f for f in findings if f.entity_type == "PHONE_NUMBER"]
        assert len(phones) >= 1
        # With context, should be boosted above base 0.96
        assert phones[0].confidence >= 0.96


# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------


class TestContextConstants:
    def test_context_words_keys_exist(self) -> None:
        expected_keys = {
            "US_SSN", "CREDIT_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS",
            "IP_ADDRESS", "IBAN", "ROUTING_NUMBER", "PERSON_NAME",
        }
        assert expected_keys <= set(_CONTEXT_WORDS.keys())

    def test_all_context_words_are_lowercase(self) -> None:
        for entity_type, words in _CONTEXT_WORDS.items():
            for word in words:
                assert word == word.lower(), f"{entity_type}: '{word}' not lowercase"

    def test_high_fp_types_are_subset_of_context_words(self) -> None:
        for t in _HIGH_FP_TYPES:
            assert t in _CONTEXT_WORDS, f"{t} in _HIGH_FP_TYPES but not in _CONTEXT_WORDS"

    def test_boost_and_penalty_positive(self) -> None:
        assert _CONTEXT_BOOST > 0
        assert _CONTEXT_PENALTY > 0
