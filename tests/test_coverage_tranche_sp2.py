"""sp2 dev-iteration 5 — the external-coverage tranche (21 new native labels).

One test per new entity type, each fixture mirroring a REAL gold shape from
the pii-anon-eval-data dev split (sampled 2026-06-12), including the
adversarial zero-width-space obfuscation present in corpus values
("7​0-632129​2"). Value classes must tolerate U+200B/200C/200D and
the captured extent must cover the full obfuscated value (gold spans do).

Internally most of these labels are census-ignored by the PINNED authority
(documented in tests/test_pattern_label_alignment.py's allowlist); they earn
external credit through the DATA harness's native->canonical-63 LABEL_MAP.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENGINE = RegexEngineAdapter(enabled=True)


def _spans(text: str, entity_type: str) -> list[str]:
    return sorted(
        text[f.span_start : f.span_end]
        for f in _ENGINE.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


@pytest.mark.parametrize(
    ("entity_type", "text", "expected"),
    [
        ("TAX_ID", "Length of employment: 9 years\nEmployer Tax ID: 85-5503355", "85-5503355"),
        ("TAX_ID", "Employer: Nordic Analytics\nEIN: 83-9386718", "83-9386718"),
        ("TAX_ID", "Record for Mary Young, Tax Id: 7​0-632129​2 (on file)", "7​0-632129​2"),
        ("JOB_TITLE", "Education: PhD. Job: Systems Administrator. Marital Status: Married.", "Systems Administrator"),
        ("JOB_TITLE", "Employer: Soylent Corp | Position: Medical Director | Annual income: $133,000", "Medical Director"),
        ("JOB_TITLE", "Richard Wright — Legal Counsel at Prestige Worldwide", "Legal Counsel"),
        ("HEALTH_CONDITION", "presents with symptoms consistent with Acute Bronchitis.", "Acute Bronchitis"),
        ("HEALTH_CONDITION", "MRN: MRN-5929414) Diagnosis: Chronic Kidney Disease", "Chronic Kidney Disease"),
        ("HEALTH_CONDITION", "seen for evaluation of Type 2 Diabetes Mellitus.", "Type 2 Diabetes Mellitus"),
        ("MEDICATION_NAME", "Care completed per protocol. Gabapentin 300mg as needed.", "Gabapentin 300mg"),
        ("MEDICATION_NAME", "Current medications include Albuterol Inhaler.", "Albuterol Inhaler"),
        ("HEALTH_INSURANCE_ID", "MRN: MRN-8182535 Insurance ID: INS-880726785", "INS-880726785"),
        ("CREDIT_CARD_FRAGMENT", "Payment method: Card ending 4057.", "4057"),
        ("CREDIT_CARD_FRAGMENT", "Scott, Credit Card Fragment: ****-****-***​*-641​2", "****-****-***​*-641​2"),
        ("VISA_NUMBER", "Record for Sandra Scott, Visa Number: V6AT29900 (verified)", "V6AT29900"),
        ("PRESCRIPTION_NUMBER", "Refill approved. Record shows RX-78901234.", "RX-78901234"),
        ("DEVICE_IDENTIFIER", "Record for Donald Wright, Device Identifier: 1023012g5133025", "1023012g5133025"),
        ("DEVICE_IDENTIFIER", "Record shows AEBE52E7-03EE-455A-B3C4-E57283966239.", "AEBE52E7-03EE-455A-B3C4-E57283966239"),
        ("SOCIAL_MEDIA_HANDLE", "John Lewis, Social Media Handle: @jo7092", "@jo7092"),
        ("EDUCATION_LEVEL", "DOB: 1984-05-09. Education: PhD. Job: Welder.", "PhD"),
        ("EDUCATION_LEVEL", "Record shows Bachelor's Degree in Computer Science.", "Bachelor's Degree in Computer Science"),
        ("GENDER", "Resident: Timothy Harris, 27529. Gender: Male. DOB: 1984-05-09.", "Male"),
        # NB: the generator-filler "Record shows X" anchor was removed as
        # benchmark gaming (sp2 remediation); these types are detected only on
        # a real field label now.
        ("NATIONALITY", "Applicant profile — Nationality: American", "American"),
        ("ETHNICITY", "Ethnicity: European", "European"),
        ("POLITICAL_OPINION", "Political Opinion: Liberal", "Liberal"),
        ("RELIGIOUS_BELIEF", "Paul Gonzalez, Religious Belief: Buddhist", "Buddhist"),
        ("MARITAL_STATUS", "Systems Administrator. Marital Status: Married.", "Married"),
        ("HOUSEHOLD_SIZE", "Kimberly Rodriguez, Household Size: 8", "8"),
        ("VEHICLE_MODEL", "Record for Brian White, Vehicle Model: Subaru Outback", "Subaru Outback"),
        ("VEHICLE_MODEL", "Insured auto: Record shows 2023 Toyota Camry.", "2023 Toyota Camry"),
        ("PROCEDURE_NAME", "Follow-up in 2 weeks. Procedure: Pulmonary Function Test", "Pulmonary Function Test"),
        ("PROCEDURE_NAME", "diagnostic workup, MRI Brain recommended.", "MRI Brain"),
        ("BIOMETRIC_ID", "Kenneth Thomas, Biometric Id: BIO-346673​43D6CF​F239", "BIO-346673​43D6CF​F239"),
        ("COURT_CASE_NUMBER", "Insurance coverage dispute Case No. 2020-CIV-1913", "2020-CIV-1913"),
        ("DOCKET_NUMBER", "DEPOSITION OF Brian Clark\nCase No: 7:24-mj-07176-JBO", "7:24-mj-07176-JBO"),
        ("INVOICE_NUMBER", "SWIFT: ZOYPCHEQ\nAmount: $114,000\nRef: INV-863469", "INV-863469"),
        ("SWIFT_BIC", "IBAN IT60211625943244249156 SWIFT: GKTRCH2B", "GKTRCH2B"),
        ("DRIVERS_LICENSE", "Nguyen, Driver License Number: AC-SE499G", "AC-SE499G"),
        ("DRIVERS_LICENSE", "Cross-ref: 00-0088-9665|10312585|P5V124045|DL-V34045-45", "DL-V34045-45"),
        ("SALARY", "Position: Medical Director | Annual income: $133,000", "$133,000"),
        ("SALARY", "EIN: 83-9386718\nWages: $131,000", "$131,000"),
        ("API_KEY", 'DB_PASS = "zGsoxVZcW3"\nAPI_KEY = "sk-Qxz7QaehBuZfznemMdq9o2vMBYlm2zAG"', "sk-Qxz7QaehBuZfznemMdq9o2vMBYlm2zAG"),
    ],
)
def test_gold_shape_detected_with_exact_extent(
    entity_type: str, text: str, expected: str
) -> None:
    assert expected in _spans(text, entity_type), (
        f"{entity_type}: {expected!r} not in {_spans(text, entity_type)!r}"
    )


class TestTrancheNegatives:
    def test_email_local_part_is_not_social_media_handle(self) -> None:
        text = "Email: dorothy.harris49@fastmail.com is on file."
        assert _spans(text, "SOCIAL_MEDIA_HANDLE") == []

    def test_lowercase_session_hex_is_not_device_identifier(self) -> None:
        text = "Session: ea8378a4bcb4fc48d32f2909d64fffaa active."
        assert _spans(text, "DEVICE_IDENTIFIER") == []

    def test_bare_married_without_context_not_marital_status(self) -> None:
        text = "They got Married in June at the lake house."
        assert _spans(text, "MARITAL_STATUS") == []

    def test_bare_dollar_amount_without_label_not_salary(self) -> None:
        text = "The repair cost was around $1,500 before taxes."
        assert _spans(text, "SALARY") == []
