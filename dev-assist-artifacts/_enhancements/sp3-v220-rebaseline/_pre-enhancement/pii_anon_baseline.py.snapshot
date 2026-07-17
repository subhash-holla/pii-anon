#!/usr/bin/env python3
"""pii-anon (vanilla) detection adapter — the sister library's regex-oss engine.

The first-party detector from the sibling ``pii-anon-code`` library, scored through the SAME
uniform contract as every other baseline — the leaderboard treats the house system as just
another detector, no special-casing. Wraps the library's public BYO predictor seam
(``pii_anon.eval_framework.byo_pipeline.first_party_predictor("pii_anon")``), which emits
NATIVE pii-anon labels; the projection onto the canonical 63 happens HERE in ``label_map``,
like every other adapter (designed in this repo — the sister library is imported only inside
``build()``, exactly as gliner imports gliner; NFR-050). Install: ``pip install -e ../pii-anon-code``.
"""

from __future__ import annotations

import importlib.util

from pii_anon_datasets.baselines.contract import AdapterSpan, coverage_of

# pii-anon NATIVE entity type -> canonical-63 type, or None to drop.
# The trailing alias block (EMAIL/PERSON/PER/ORG/GPE/LOC/UNKNOWN) covers the swarm's
# NER-engine vocabulary so the swarm adapter can share this single projection.
LABEL_MAP: dict[str, str | None] = {
    "AADHAAR": "NATIONAL_ID_NUMBER",
    "ADDRESS": "STREET_ADDRESS",
    "AGE": "AGE",
    "API_KEY": "API_KEY",
    "BANK_ACCOUNT": "BANK_ACCOUNT_NUMBER",
    "BAR_NUMBER": "BAR_NUMBER",
    "CANADIAN_SIN": "NATIONAL_ID_NUMBER",
    "COURT_CASE_NUMBER": "COURT_CASE_NUMBER",
    "CREDIT_CARD": "CREDIT_CARD_NUMBER",
    "CRYPTO_WALLET": "CRYPTOCURRENCY_ADDRESS",
    "CVV": "CVV",
    "DATE_ISO": None,  # ambiguous (DOB vs generic date) — deliberate drop, revisit on dev evidence
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "DATE_TIME": "TIMESTAMP",
    "DEA_NUMBER": "DEA_NUMBER",
    "DOCKET_NUMBER": "DOCKET_NUMBER",
    "DRIVERS_LICENSE": "DRIVER_LICENSE_NUMBER",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
    "GPS_COORDINATES": "LATITUDE_LONGITUDE",
    "IBAN": "IBAN",
    "INSURANCE_POLICY_NUMBER": "INSURANCE_POLICY_NUMBER",
    "INVOICE_NUMBER": "INVOICE_NUMBER",
    "IP_ADDRESS": "IP_ADDRESS",
    "JWT_TOKEN": "AUTHENTICATION_TOKEN",
    "LICENSE_PLATE": "LICENSE_PLATE",
    "LOCATION": "LOCATION_NAME",
    "MAC_ADDRESS": "MAC_ADDRESS",
    "MEDICAL_RECORD_NUMBER": "MEDICAL_RECORD_NUMBER",
    "NATIONAL_ID": "NATIONAL_ID_NUMBER",
    "NPI_NUMBER": "NPI_NUMBER",
    "ORGANIZATION": "ORGANIZATION_NAME",
    "PASSPORT": "PASSPORT_NUMBER",
    "PASSWORD": "PASSWORD",
    "PERSON_NAME": "PERSON_NAME",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PIN": "PIN",
    "ROUTING_NUMBER": "BANK_ROUTING_NUMBER",
    "SALARY": "SALARY",
    "SWIFT_BIC": "SWIFT_BIC_CODE",
    "UK_NI_NUMBER": "NATIONAL_ID_NUMBER",
    "URL_WITH_PII": "URL",
    "USERNAME": "USERNAME",
    "US_SSN": "SOCIAL_SECURITY_NUMBER",
    "VIN": "VEHICLE_IDENTIFICATION_NUMBER",
    "ZIP_CODE": "POSTAL_CODE",
    # sp2 external-coverage tranche (native labels named after the canonical
    # types they project onto — identity mappings).
    "TAX_ID": "TAX_ID",
    "JOB_TITLE": "JOB_TITLE",
    "HEALTH_CONDITION": "HEALTH_CONDITION",
    "MEDICATION_NAME": "MEDICATION_NAME",
    "HEALTH_INSURANCE_ID": "HEALTH_INSURANCE_ID",
    "CREDIT_CARD_FRAGMENT": "CREDIT_CARD_FRAGMENT",
    "VISA_NUMBER": "VISA_NUMBER",
    "PRESCRIPTION_NUMBER": "PRESCRIPTION_NUMBER",
    "DEVICE_IDENTIFIER": "DEVICE_IDENTIFIER",
    "SOCIAL_MEDIA_HANDLE": "SOCIAL_MEDIA_HANDLE",
    "EDUCATION_LEVEL": "EDUCATION_LEVEL",
    "GENDER": "GENDER",
    "NATIONALITY": "NATIONALITY",
    "ETHNICITY": "ETHNICITY",
    "POLITICAL_OPINION": "POLITICAL_OPINION",
    "RELIGIOUS_BELIEF": "RELIGIOUS_BELIEF",
    "MARITAL_STATUS": "MARITAL_STATUS",
    "HOUSEHOLD_SIZE": "HOUSEHOLD_SIZE",
    "VEHICLE_MODEL": "VEHICLE_MODEL",
    "PROCEDURE_NAME": "PROCEDURE_NAME",
    "BIOMETRIC_ID": "BIOMETRIC_ID",
    # NER-engine aliases (swarm pool vocabulary).
    "EMAIL": "EMAIL_ADDRESS",
    "PERSON": "PERSON_NAME",
    "PER": "PERSON_NAME",
    "ORG": "ORGANIZATION_NAME",
    "GPE": "LOCATION_NAME",
    "LOC": "LOCATION_NAME",
    "UNKNOWN": None,
}


class _PiiAnonAdapter:
    name = "pii_anon"
    model_id = "pii-anon vanilla (regex-oss engine)"
    label_map = LABEL_MAP
    deterministic = True

    def available(self) -> bool:
        return importlib.util.find_spec("pii_anon") is not None

    def map_label(self, native: str) -> str | None:
        return LABEL_MAP.get((native or "").strip().upper())

    def build(self) -> object:
        try:
            from pii_anon.eval_framework.byo_pipeline import first_party_predictor
        except ImportError as e:  # pragma: no cover - exercised only on a lib-less checkout
            raise RuntimeError("pii-anon not installed — pip install -e ../pii-anon-code") from e
        return first_party_predictor("pii_anon")

    def detect(self, text: str, model: object) -> list[AdapterSpan]:
        if not text:
            return []
        out: list[AdapterSpan] = []
        for native, start, end in model(text):  # type: ignore[operator]
            et = self.map_label(str(native))
            if et is not None:
                s, e = int(start), int(end)
                out.append(AdapterSpan(s, e, et, text[s:e]))
        return out

    def coverage(self) -> int:
        return coverage_of(LABEL_MAP)


ADAPTER = _PiiAnonAdapter()
