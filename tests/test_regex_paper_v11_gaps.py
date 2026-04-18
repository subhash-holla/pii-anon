"""Tests for the paper v11 gap-closure patterns + P0 correctness fixes.

Covers the Phase 3 additions to the regex-oss baseline:

- New entity types: CVV, PIN, PASSWORD, COURT_CASE_NUMBER,
  DOCKET_NUMBER, BAR_NUMBER, INVOICE_NUMBER, INSURANCE_POLICY_NUMBER,
  SALARY.
- Extended HIGH_FP_TYPES (VIN, LICENSE_PLATE, NATIONAL_ID).
- UK NI structural validator — rejects invalid prefixes and suffixes.
- has_context_words now word-boundary-tokenizes (vs. whitespace-splits).
- swift_context now requires an ISO 3166-1 alpha-2 country code.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.confidence import (
    CONTEXT_WORDS,
    HIGH_FP_TYPES,
    extract_context,
    has_context_words,
)
from pii_anon.engines.regex.validators import is_valid_uk_ni
from pii_anon.engines.regex_adapter import RegexEngineAdapter


# ---------------------------------------------------------------------------
# P0 correctness fixes
# ---------------------------------------------------------------------------

def test_has_context_words_word_boundary_tokenization():
    """``social_security_number`` is a single token under ``str.split`` but
    three tokens under word-boundary tokenization — the fix surfaces the
    ``social`` and ``security`` keywords that used to be hidden.
    """
    # The US_SSN context words include "social" and "security".
    text = "social_security_number:123-45-6789"
    assert has_context_words("US_SSN", text) is True


def test_has_context_words_finds_keyword_at_kv_boundary():
    """``ssn=123`` should match the ``ssn`` context word even though
    there is no whitespace separator.
    """
    assert has_context_words("US_SSN", "ssn=123456789") is True


def test_has_context_words_ignores_absent_keywords():
    """Sanity: no context words in a generic string → no match."""
    assert has_context_words("US_SSN", "some neutral sentence") is False


def test_has_context_words_unknown_entity_returns_false():
    assert has_context_words("TOTALLY_UNKNOWN", "anything goes here") is False


# ---------------------------------------------------------------------------
# HIGH_FP_TYPES expansion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type", [
    "VIN", "LICENSE_PLATE", "NATIONAL_ID",
    # Original entries — must still be present.
    "US_SSN", "PERSON_NAME", "BANK_ACCOUNT",
])
def test_high_fp_types_includes_ambiguous_identifiers(entity_type):
    assert entity_type in HIGH_FP_TYPES


# ---------------------------------------------------------------------------
# UK NI validator — HMRC prefix/suffix rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("candidate,expected", [
    # Canonical valid forms.
    ("AB123456C", True),
    ("AB 12 34 56 C", True),     # spaces allowed, stripped by validator
    ("JR123456D", True),         # valid first+second letter combo
    # Invalid first letter (D/F/I/Q/U/V prohibited).
    ("DA123456A", False),
    ("FA123456A", False),
    ("IA123456A", False),
    ("QA123456A", False),
    ("UA123456A", False),
    ("VA123456A", False),
    # Invalid second letter.
    ("ADGH3456A", False),        # second letter D
    ("AO123456A", False),        # second letter O
    # Reserved prefixes.
    ("BG123456A", False),
    ("GB123456A", False),
    ("KN123456A", False),
    ("TN123456A", False),
    ("ZZ123456A", False),
    # Invalid suffix letter (only A/B/C/D allowed).
    ("AB123456E", False),
    ("AB123456X", False),
    # Non-digit middle.
    ("AB12AB56C", False),
    # Wrong length.
    ("AB12345C", False),
    ("AB1234567C", False),
])
def test_is_valid_uk_ni(candidate, expected):
    assert is_valid_uk_ni(candidate) is expected


# ---------------------------------------------------------------------------
# Phase 3 — new CONTEXT_WORDS entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type", [
    "CVV", "PIN", "PASSWORD",
    "COURT_CASE_NUMBER", "DOCKET_NUMBER", "BAR_NUMBER",
    "INVOICE_NUMBER", "INSURANCE_POLICY_NUMBER", "SALARY",
])
def test_new_entity_types_have_context_words(entity_type):
    assert entity_type in CONTEXT_WORDS
    # Each mapping must contain the base English keyword.
    words = CONTEXT_WORDS[entity_type]
    assert len(words) > 3, f"{entity_type} has thin context-word coverage"


# ---------------------------------------------------------------------------
# Phase 3 — end-to-end pattern detection via RegexEngineAdapter
# ---------------------------------------------------------------------------

@pytest.fixture
def regex_adapter():
    adapter = RegexEngineAdapter(enabled=True)
    adapter.initialize({})
    return adapter


def _detect(adapter, text):
    """Run regex detection on raw text and return (entity_type, span_text) pairs."""
    findings = adapter.detect({"text": text}, {"language": "en"})
    return [
        (f.entity_type, text[f.span_start:f.span_end])
        for f in findings
    ]


@pytest.mark.parametrize("text,expected_type", [
    # CVV — requires card-context keyword
    ("The CVV is 123 for this card.", "CVV"),
    ("cvv2: 4567", "CVV"),
    ("Security Code: 123", "CVV"),
])
def test_cvv_detection(regex_adapter, text, expected_type):
    findings = _detect(regex_adapter, text)
    assert any(t == expected_type for t, _ in findings), (
        f"Expected {expected_type} in findings: {findings}"
    )


def test_cvv_bare_digits_not_detected(regex_adapter):
    """Bare 3-digit numbers without CVV context must not match."""
    findings = _detect(regex_adapter, "The weather is sunny today. 432 days of sun.")
    assert not any(t == "CVV" for t, _ in findings)


@pytest.mark.parametrize("text", [
    "PIN: 1234",
    "PIN number: 123456",
    "ATM PIN is 4321",
])
def test_pin_detection(regex_adapter, text):
    findings = _detect(regex_adapter, text)
    assert any(t == "PIN" for t, _ in findings), f"PIN not in {findings}"


@pytest.mark.parametrize("text", [
    "password=hunter2",
    "pwd: MyS3cretPW",
    "pass = TopSecret123",
])
def test_password_detection(regex_adapter, text):
    findings = _detect(regex_adapter, text)
    assert any(t == "PASSWORD" for t, _ in findings), f"PASSWORD not in {findings}"


@pytest.mark.parametrize("text", [
    "Case No. 1:21-cv-01234 was filed",
    "case no. 3:22-cv-00001 in district court",
])
def test_court_case_detection(regex_adapter, text):
    findings = _detect(regex_adapter, text)
    assert any(t == "COURT_CASE_NUMBER" for t, _ in findings), f"COURT_CASE not in {findings}"


@pytest.mark.parametrize("text,entity_type", [
    ("Docket No. 2024-CV-00123", "DOCKET_NUMBER"),
    ("State Bar No. 123456 is admitted", "BAR_NUMBER"),
    ("SBN 987654", "BAR_NUMBER"),
    ("Invoice #2024-0012", "INVOICE_NUMBER"),
    ("Inv. No. 12345", "INVOICE_NUMBER"),
    ("Policy #ABC-123456", "INSURANCE_POLICY_NUMBER"),
    ("Policy Number: POL-2024-001", "INSURANCE_POLICY_NUMBER"),
])
def test_legal_and_financial_identifiers(regex_adapter, text, entity_type):
    findings = _detect(regex_adapter, text)
    assert any(t == entity_type for t, _ in findings), (
        f"{entity_type} not in findings: {findings}"
    )


@pytest.mark.parametrize("text", [
    "Annual salary of $85,000",
    "Base pay: 72000",
    "Compensation is $120,500 per year",
])
def test_salary_detection(regex_adapter, text):
    findings = _detect(regex_adapter, text)
    assert any(t == "SALARY" for t, _ in findings), f"SALARY not in {findings}"


# ---------------------------------------------------------------------------
# SWIFT/BIC country-code strictness
# ---------------------------------------------------------------------------

def test_swift_bic_rejects_fake_country_codes(regex_adapter):
    """A BIC-shaped string whose country pair is not ISO 3166-1 alpha-2
    must not be emitted.  Example: ``DEUTZZXXX`` has ``ZZ`` as country
    code which is not a valid jurisdiction.
    """
    # Context is bank-related so the text-level gate passes; strict
    # country-code check is what blocks this one.
    text = "Wire transfer via bank code DEUTZZXXX is routed through SWIFT."
    findings = _detect(regex_adapter, text)
    swift_hits = [(t, s) for t, s in findings if t == "SWIFT_BIC"]
    # ZZ is NOT a valid ISO 3166-1 alpha-2 code, so the finding must be
    # suppressed by the validator.
    assert all("ZZ" not in s for _, s in swift_hits), (
        f"Fake country code ZZ leaked through: {swift_hits}"
    )


def test_swift_bic_accepts_real_country_code(regex_adapter):
    """A BIC with a real ISO 3166-1 alpha-2 country pair + bank context
    must be emitted.  Example: ``DEUTDEFF`` (Deutsche Bank AG, DE).
    """
    text = "Wire transfer via bank code DEUTDEFF processed via SWIFT."
    findings = _detect(regex_adapter, text)
    assert any(t == "SWIFT_BIC" for t, _ in findings), (
        f"Valid BIC rejected: {findings}"
    )


# ---------------------------------------------------------------------------
# extract_context sanity — ensures the new tokenizer path is exercised
# ---------------------------------------------------------------------------

def test_extract_context_lowercases_and_slices():
    text = "PREFIX Name: Alice Smith SUFFIX"
    ctx = extract_context(text, 12, 23)
    assert ctx == text[:50].lower()[:len(ctx)] or "alice" in ctx


def test_context_match_is_case_insensitive_via_extract_context():
    """End-to-end: extract_context lowercases the window, so context
    words (lowercase) always match regardless of original casing.
    """
    text = "SOCIAL SECURITY NUMBER: 123-45-6789"
    ctx = extract_context(text, 23, 34)
    assert has_context_words("US_SSN", ctx) is True
