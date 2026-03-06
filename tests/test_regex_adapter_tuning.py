from __future__ import annotations

from pii_anon.engines.regex_adapter import RegexEngineAdapter


def test_regex_adapter_detects_ip_credit_card_and_iban() -> None:
    adapter = RegexEngineAdapter(enabled=True)
    text = (
        "Client IP 192.168.10.42 used card 4111-1111-1111-1111 and "
        "wire transfer IBAN GB82WEST12345698765432."
    )

    findings = adapter.detect({"text": text}, {"language": "en", "policy_mode": "balanced"})
    entity_types = {item.entity_type for item in findings}

    assert "IP_ADDRESS" in entity_types
    assert "CREDIT_CARD" in entity_types
    assert "IBAN" in entity_types


def test_regex_adapter_filters_invalid_ip_but_accepts_format_cc_and_iban() -> None:
    """v1.0.0: Credit cards with valid issuer prefixes and IBANs with valid format
    are now accepted even if checksum fails (medium confidence).  Only truly
    invalid formats are filtered out."""
    adapter = RegexEngineAdapter(enabled=True)

    # Invalid IP: 999 > 255 → should be filtered
    ip_text = "Bad IP 999.168.10.42"
    ip_findings = adapter.detect({"text": ip_text}, {"language": "en", "policy_mode": "balanced"})
    ip_types = {item.entity_type for item in ip_findings}
    assert "IP_ADDRESS" not in ip_types

    # Luhn-invalid but Visa-prefix card → accepted with lower confidence
    cc_text = "card 4111-1111-1111-1112"
    cc_findings = adapter.detect({"text": cc_text}, {"language": "en", "policy_mode": "balanced"})
    cc_cards = [f for f in cc_findings if f.entity_type == "CREDIT_CARD"]
    assert len(cc_cards) == 1
    # format match base 0.80 + context boost 0.08 ("card" keyword) = 0.88
    assert cc_cards[0].confidence == 0.88  # format match, context-boosted

    # Luhn-valid card → high confidence + context boost
    cc_valid_text = "card 4111-1111-1111-1111"
    cc_valid_findings = adapter.detect({"text": cc_valid_text}, {"language": "en", "policy_mode": "balanced"})
    cc_valid = [f for f in cc_valid_findings if f.entity_type == "CREDIT_CARD"]
    assert len(cc_valid) == 1
    assert cc_valid[0].confidence == 0.99  # Luhn-valid + context boost (capped at 0.99)

    # Non-issuer prefix AND Luhn-invalid → filtered
    cc_bad_text = "number 8111-1111-1111-1113"
    cc_bad_findings = adapter.detect({"text": cc_bad_text}, {"language": "en", "policy_mode": "balanced"})
    cc_bad = [f for f in cc_bad_findings if f.entity_type == "CREDIT_CARD"]
    assert len(cc_bad) == 0

    # Short IBAN → should be filtered (too short)
    iban_text = "bad iban GB00TEST123"
    iban_findings = adapter.detect({"text": iban_text}, {"language": "en", "policy_mode": "balanced"})
    iban_types = {item.entity_type for item in iban_findings}
    assert "IBAN" not in iban_types


def test_regex_adapter_skips_non_string_payload_values_and_disabled_mode() -> None:
    adapter = RegexEngineAdapter(enabled=True)
    findings = adapter.detect({"text": "Reach me at alice@example.com", "count": 3}, {"language": "en"})
    entity_types = {item.entity_type for item in findings}
    assert "EMAIL_ADDRESS" in entity_types

    disabled = RegexEngineAdapter(enabled=False)
    assert disabled.detect({"text": "alice@example.com"}, {"language": "en"}) == []


def test_regex_adapter_validation_helpers_cover_edge_cases() -> None:
    assert RegexEngineAdapter._is_valid_ipv4("10.0.0.1") is True
    assert RegexEngineAdapter._is_valid_ipv4("10.0.0") is False
    assert RegexEngineAdapter._is_valid_ipv4("10.0.a.1") is False
    assert RegexEngineAdapter._is_valid_ipv4("10.0.999.1") is False

    # v1.0.0: _is_valid_credit_card now accepts Luhn-valid OR format-match
    assert RegexEngineAdapter._is_valid_credit_card("4111 1111 1111 1111") is True  # Luhn valid
    assert RegexEngineAdapter._is_valid_credit_card("1234") is False  # too short
    assert RegexEngineAdapter._is_valid_credit_card("4111 1111 1111 1112") is True  # format match (Visa prefix)
    assert RegexEngineAdapter._is_valid_credit_card("8111 1111 1111 1113") is False  # no valid prefix + Luhn fail

    # v1.0.0: _is_valid_iban now accepts checksum-valid OR format-valid
    assert RegexEngineAdapter._is_valid_iban("GB82WEST12345698765432") is True
    assert RegexEngineAdapter._is_valid_iban("GB00TEST123") is False  # too short
    assert RegexEngineAdapter._is_valid_iban("12ABWEST12345678901234") is False  # doesn't start with letters

    # v1.0.0: Luhn helper
    assert RegexEngineAdapter._luhn_checksum("4111111111111111") is True
    assert RegexEngineAdapter._luhn_checksum("4111111111111112") is False

    # v1.0.0: CC format helper
    assert RegexEngineAdapter._is_cc_format("4111111111111112") is True  # Visa prefix
    assert RegexEngineAdapter._is_cc_format("5111111111111112") is True  # Mastercard
    assert RegexEngineAdapter._is_cc_format("3411111111111112") is True  # Amex
    assert RegexEngineAdapter._is_cc_format("8111111111111113") is False  # Unknown prefix

    # v1.0.0: IBAN format helper
    assert RegexEngineAdapter._is_valid_iban_format("GB82WEST12345698765432") is True
    assert RegexEngineAdapter._is_valid_iban_format("GB00TEST123") is False  # too short
    assert RegexEngineAdapter._is_valid_iban_strict("GB82WEST12345698765432") is True

    # v1.0.0: SSN digit validator
    assert RegexEngineAdapter._is_valid_ssn_digits("123456789") is True
    assert RegexEngineAdapter._is_valid_ssn_digits("000123456") is False  # area 000
    assert RegexEngineAdapter._is_valid_ssn_digits("666123456") is False  # area 666
    assert RegexEngineAdapter._is_valid_ssn_digits("900123456") is False  # area >= 900


def test_regex_adapter_address_phrase_heuristics_paths() -> None:
    adapter = RegexEngineAdapter(enabled=True)

    # Address suffix in phrase → True
    street_text = "address 42 Main Street, London"
    street_start = street_text.index("Main")
    street_end = street_start + len("Main Street")
    assert adapter._looks_like_address_phrase(street_text, street_start, street_end) is True

    # Digit immediately before match → True
    digit_text = "9John Smith called support"
    digit_start = digit_text.index("John")
    digit_end = digit_start + len("John Smith")
    assert adapter._looks_like_address_phrase(digit_text, digit_start, digit_end) is True

    # No address context → False
    neutral_text = "Customer John Smith called support"
    neutral_start = neutral_text.index("John")
    neutral_end = neutral_start + len("John Smith")
    assert adapter._looks_like_address_phrase(neutral_text, neutral_start, neutral_end) is False

    # "address" keyword with digit prefix → True
    addr_digit_text = "mailing address 7John Smith"
    addr_start = addr_digit_text.index("John")
    addr_end = addr_start + len("John Smith")
    assert adapter._looks_like_address_phrase(addr_digit_text, addr_start, addr_end) is True


def test_regex_adapter_new_entity_types() -> None:
    """v1.0.0: Verify new entity type patterns detect correctly."""
    adapter = RegexEngineAdapter(enabled=True)
    ctx = {"language": "en", "policy_mode": "balanced"}

    # DATE_OF_BIRTH
    dob_findings = adapter.detect({"text": "born 03/15/1990"}, ctx)
    dob_types = {f.entity_type for f in dob_findings}
    assert "DATE_OF_BIRTH" in dob_types

    # MAC_ADDRESS
    mac_findings = adapter.detect({"text": "MAC: 00:1A:2B:3C:4D:5E"}, ctx)
    mac_types = {f.entity_type for f in mac_findings}
    assert "MAC_ADDRESS" in mac_types

    # DRIVERS_LICENSE
    dl_findings = adapter.detect({"text": "driver's license: D1234567"}, ctx)
    dl_types = {f.entity_type for f in dl_findings}
    assert "DRIVERS_LICENSE" in dl_types

    # PASSPORT
    pass_findings = adapter.detect({"text": "passport number: AB1234567"}, ctx)
    pass_types = {f.entity_type for f in pass_findings}
    assert "PASSPORT" in pass_types

    # USERNAME (@handle)
    user_findings = adapter.detect({"text": "Follow @johndoe_42"}, ctx)
    user_types = {f.entity_type for f in user_findings}
    assert "USERNAME" in user_types

    # EMPLOYEE_ID
    emp_findings = adapter.detect({"text": "employee id: EMP12345"}, ctx)
    emp_types = {f.entity_type for f in emp_findings}
    assert "EMPLOYEE_ID" in emp_types

    # MEDICAL_RECORD_NUMBER
    mrn_findings = adapter.detect({"text": "MRN: ABC1234567"}, ctx)
    mrn_types = {f.entity_type for f in mrn_findings}
    assert "MEDICAL_RECORD_NUMBER" in mrn_types

    # ORGANIZATION
    org_findings = adapter.detect({"text": "Acme Corp is hiring"}, ctx)
    org_types = {f.entity_type for f in org_findings}
    assert "ORGANIZATION" in org_types

    # ADDRESS
    addr_findings = adapter.detect({"text": "123 Main Street is the location"}, ctx)
    addr_types = {f.entity_type for f in addr_findings}
    assert "ADDRESS" in addr_types

    # US_SSN variants
    ssn_space = adapter.detect({"text": "SSN is 123 45 6789"}, ctx)
    ssn_types = {f.entity_type for f in ssn_space}
    assert "US_SSN" in ssn_types

    # PERSON_NAME context keyword
    name_ctx = adapter.detect({"text": "patient Maria Lopez was seen"}, ctx)
    name_types = {f.entity_type for f in name_ctx}
    assert "PERSON_NAME" in name_types


def test_regex_adapter_credit_card_tiered_confidence() -> None:
    """v1.0.0: Luhn-valid cards get higher confidence than format-only matches."""
    adapter = RegexEngineAdapter(enabled=True)
    ctx = {"language": "en", "policy_mode": "balanced"}

    # Luhn-valid + context boost ("card" keyword): 0.94 + 0.08 = 0.99 (capped)
    valid_findings = adapter.detect({"text": "card 4111111111111111"}, ctx)
    valid_cc = [f for f in valid_findings if f.entity_type == "CREDIT_CARD"]
    assert len(valid_cc) >= 1
    assert valid_cc[0].confidence == 0.99

    # Luhn-invalid but Visa prefix + context boost: 0.80 + 0.08 = 0.88
    format_findings = adapter.detect({"text": "card 4111111111111112"}, ctx)
    format_cc = [f for f in format_findings if f.entity_type == "CREDIT_CARD"]
    assert len(format_cc) >= 1
    assert format_cc[0].confidence == 0.88


def test_regex_adapter_iban_tiered_confidence() -> None:
    """v1.0.0: Checksum-valid IBANs get higher confidence than format-only matches."""
    adapter = RegexEngineAdapter(enabled=True)
    ctx = {"language": "en", "policy_mode": "balanced"}

    # Valid IBAN format + context boost ("IBAN" keyword): 0.78→0.86 or 0.93→0.99
    format_findings = adapter.detect({"text": "IBAN DE99123456789012345678"}, ctx)
    format_ibans = [f for f in format_findings if f.entity_type == "IBAN"]
    assert len(format_ibans) >= 1
    assert format_ibans[0].confidence in (0.86, 0.99)  # context-boosted
