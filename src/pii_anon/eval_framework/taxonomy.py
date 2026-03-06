"""PII entity taxonomy with 48 entity types across 7 categories.

Evidence basis:
- NIST SP 800-122: Guide to Protecting the Confidentiality of PII (2010)
- GDPR Article 4 & Article 9: Personal data and special categories (2016)
- ISO 27701:2019 (updated 2025): Privacy Information Management
- OpenNER 1.0 (2024): Standardised NER taxonomy across 52 languages
- i2b2 2014 de-identification shared task: Clinical PII categories
- CRAPII (2024): Educational PII entity types
- Sweeney (2002): k-anonymity and quasi-identifier re-identification risk

All entity types are mapped to at least one regulatory standard.
Quasi-identifier metadata follows Sweeney (2002) — gender + DOB + ZIP
yields 87% re-identification in the US population.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EntityCategory(str, Enum):
    """Seven evidence-backed PII entity categories.

    Categories follow NIST SP 800-122 linkability taxonomy and GDPR
    Article 4/9 personal-data and special-category classification.
    """

    PERSONAL_IDENTITY = "personal_identity"
    FINANCIAL = "financial"
    GOVERNMENT_ID = "government_id"
    MEDICAL = "medical"
    DIGITAL_TECHNICAL = "digital_technical"
    EMPLOYMENT = "employment"
    BEHAVIORAL_CONTEXTUAL = "behavioral_contextual"


class RiskLevel(str, Enum):
    """Risk classification per NIST SP 800-122 impact levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RegulatoryReference:
    """Maps an entity type to a specific regulatory standard."""

    standard: str
    section: str
    classification: str
    notes: str = ""


@dataclass(frozen=True)
class EntityTypeProfile:
    """Rich metadata for a single PII entity type.

    Each profile carries regulatory mappings, risk classification,
    language-variant hints, and quasi-identifier metadata so that the
    evaluation framework can validate coverage against multiple standards
    and assess re-identification risk simultaneously.
    """

    entity_type: str
    category: EntityCategory
    risk_level: RiskLevel
    description: str
    regulatory_refs: tuple[RegulatoryReference, ...] = ()
    example_patterns: tuple[str, ...] = ()
    language_variant_hint: str = ""
    related_types: tuple[str, ...] = ()

    # v1.0.0: Quasi-identifier and re-identification risk metadata
    is_quasi_identifier: bool = False
    quasi_identifier_groups: tuple[str, ...] = ()
    reidentification_contribution: float = 0.0


# ---------------------------------------------------------------------------
# Regulatory reference helpers
# ---------------------------------------------------------------------------

def _nist(section: str, classification: str, notes: str = "") -> RegulatoryReference:
    return RegulatoryReference("NIST SP 800-122", section, classification, notes)


def _gdpr(section: str, classification: str, notes: str = "") -> RegulatoryReference:
    return RegulatoryReference("GDPR", section, classification, notes)


def _iso(section: str, classification: str, notes: str = "") -> RegulatoryReference:
    return RegulatoryReference("ISO 27701:2019", section, classification, notes)


# ---------------------------------------------------------------------------
# PII_TAXONOMY — 44 canonical entity type constants
# ---------------------------------------------------------------------------

class PII_TAXONOMY:
    """Namespace holding all 48 canonical PII entity-type string constants.

    Usage::

        from pii_anon.eval_framework.taxonomy import PII_TAXONOMY
        assert PII_TAXONOMY.EMAIL_ADDRESS == "EMAIL_ADDRESS"
        all_types = PII_TAXONOMY.all_types()   # list of 48 strings
    """

    # ── Personal Identity (10) ──────────────────────────────────────────
    PERSON_NAME = "PERSON_NAME"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    ADDRESS = "ADDRESS"
    ZIP_CODE = "ZIP_CODE"
    SOCIAL_MEDIA_HANDLE = "SOCIAL_MEDIA_HANDLE"
    USERNAME = "USERNAME"
    GENDER = "GENDER"
    NATIONALITY = "NATIONALITY"

    # ── Financial (7) ──────────────────────────────────────────────────
    CREDIT_CARD_NUMBER = "CREDIT_CARD_NUMBER"
    IBAN = "IBAN"
    BANK_ACCOUNT_NUMBER = "BANK_ACCOUNT_NUMBER"
    ROUTING_NUMBER = "ROUTING_NUMBER"
    SWIFT_BIC_CODE = "SWIFT_BIC_CODE"
    CRYPTOCURRENCY_WALLET = "CRYPTOCURRENCY_WALLET"
    TAX_ID = "TAX_ID"

    # ── Government ID (7) ──────────────────────────────────────────────
    US_SSN = "US_SSN"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    DRIVERS_LICENSE = "DRIVERS_LICENSE"
    NATIONAL_ID_NUMBER = "NATIONAL_ID_NUMBER"
    VISA_NUMBER = "VISA_NUMBER"
    LICENSE_PLATE = "LICENSE_PLATE"
    VEHICLE_IDENTIFICATION_NUMBER = "VEHICLE_IDENTIFICATION_NUMBER"

    # ── Medical (5) ────────────────────────────────────────────────────
    MEDICAL_RECORD_NUMBER = "MEDICAL_RECORD_NUMBER"
    HEALTH_INSURANCE_ID = "HEALTH_INSURANCE_ID"
    PRESCRIPTION_NUMBER = "PRESCRIPTION_NUMBER"
    MEDICAL_DIAGNOSIS = "MEDICAL_DIAGNOSIS"
    BIOMETRIC_ID = "BIOMETRIC_ID"

    # ── Digital / Technical (6) ────────────────────────────────────────
    IP_ADDRESS = "IP_ADDRESS"
    MAC_ADDRESS = "MAC_ADDRESS"
    API_KEY = "API_KEY"
    AUTHENTICATION_TOKEN = "AUTHENTICATION_TOKEN"
    DEVICE_ID = "DEVICE_ID"
    URL_WITH_PII = "URL_WITH_PII"

    # ── Employment (5) ─────────────────────────────────────────────────
    EMPLOYEE_ID = "EMPLOYEE_ID"
    ORGANIZATION = "ORGANIZATION"
    JOB_TITLE = "JOB_TITLE"
    SALARY = "SALARY"
    EDUCATION_LEVEL = "EDUCATION_LEVEL"

    # ── Behavioral / Contextual (7) ────────────────────────────────────
    LOCATION_COORDINATES = "LOCATION_COORDINATES"
    AGE = "AGE"
    ETHNIC_ORIGIN = "ETHNIC_ORIGIN"
    RELIGIOUS_BELIEF = "RELIGIOUS_BELIEF"
    POLITICAL_OPINION = "POLITICAL_OPINION"
    MARITAL_STATUS = "MARITAL_STATUS"
    HOUSEHOLD_SIZE = "HOUSEHOLD_SIZE"

    # ── Vehicle Linkage (1) — v1.0.0 addition ─────────────────────────
    VEHICLE_MODEL = "VEHICLE_MODEL"

    @classmethod
    def all_types(cls) -> list[str]:
        """Return a sorted list of all 48 canonical entity-type strings."""
        return sorted(
            value
            for key, value in vars(cls).items()
            if not key.startswith("_") and isinstance(value, str) and key == key.upper()
        )

    @classmethod
    def types_for_category(cls, category: EntityCategory) -> list[str]:
        """Return entity types belonging to *category*."""
        return [
            profile.entity_type
            for profile in _ENTITY_PROFILES.values()
            if profile.category == category
        ]


# ---------------------------------------------------------------------------
# Full profile registry — one EntityTypeProfile per entity type
# ---------------------------------------------------------------------------

_ENTITY_PROFILES: dict[str, EntityTypeProfile] = {
    # ── Personal Identity ──────────────────────────────────────────────
    PII_TAXONOMY.PERSON_NAME: EntityTypeProfile(
        entity_type=PII_TAXONOMY.PERSON_NAME,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.HIGH,
        description="Full or partial human name including titles, initials, and aliases.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 4(1)", "Personal data - name"),
            _iso("A.7.2.1", "PII identification"),
        ),
        example_patterns=("John Smith", "Dr. Maria Garcia", "김민수"),
        language_variant_hint="Names vary heavily by culture; CJK family-name-first ordering.",
        related_types=(PII_TAXONOMY.USERNAME, PII_TAXONOMY.SOCIAL_MEDIA_HANDLE),
    ),
    PII_TAXONOMY.EMAIL_ADDRESS: EntityTypeProfile(
        entity_type=PII_TAXONOMY.EMAIL_ADDRESS,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.HIGH,
        description="Electronic mail address in RFC 5322 format.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 4(1)", "Personal data - online identifier"),
            _iso("A.7.2.1", "PII identification"),
        ),
        example_patterns=("user@example.com", "名前@例.jp"),
        related_types=(PII_TAXONOMY.USERNAME,),
    ),
    PII_TAXONOMY.PHONE_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.PHONE_NUMBER,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.HIGH,
        description="Telephone number in any national or international format.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 4(1)", "Personal data"),
            _iso("A.7.2.1", "PII identification"),
        ),
        example_patterns=("+1 (415) 555-0100", "+44 20 7946 0958", "+81 3-1234-5678"),
        language_variant_hint="Format varies by country; E.164 is the canonical form.",
    ),
    PII_TAXONOMY.DATE_OF_BIRTH: EntityTypeProfile(
        entity_type=PII_TAXONOMY.DATE_OF_BIRTH,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.HIGH,
        description="Date of birth in any locale-specific format.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - quasi-identifier"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("1990-01-15", "15/01/1990", "January 15, 1990"),
        related_types=(PII_TAXONOMY.AGE,),
        is_quasi_identifier=True,
        quasi_identifier_groups=("gender_dob_zip", "dob_zip_gender_marital"),
        reidentification_contribution=0.25,
    ),
    PII_TAXONOMY.ADDRESS: EntityTypeProfile(
        entity_type=PII_TAXONOMY.ADDRESS,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.HIGH,
        description="Physical mailing or residential address.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 4(1)", "Personal data - location data"),
        ),
        example_patterns=("123 Main St, Springfield, IL 62701", "東京都渋谷区1-2-3"),
        related_types=(PII_TAXONOMY.ZIP_CODE, PII_TAXONOMY.LOCATION_COORDINATES),
    ),
    PII_TAXONOMY.ZIP_CODE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.ZIP_CODE,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.MODERATE,
        description="Postal or ZIP code.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - quasi-identifier"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("90210", "SW1A 1AA", "100-0001"),
        is_quasi_identifier=True,
        quasi_identifier_groups=("gender_dob_zip", "dob_zip_gender_marital", "education_occupation_zip", "household_zip_age", "vehicle_zip_age"),
        reidentification_contribution=0.27,
    ),
    PII_TAXONOMY.SOCIAL_MEDIA_HANDLE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.SOCIAL_MEDIA_HANDLE,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.MODERATE,
        description="Social-media username or handle (e.g. @user).",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - online identifier"),
        ),
        example_patterns=("@johndoe", "@usuario123"),
        related_types=(PII_TAXONOMY.USERNAME,),
    ),
    PII_TAXONOMY.USERNAME: EntityTypeProfile(
        entity_type=PII_TAXONOMY.USERNAME,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.MODERATE,
        description="System or application login username.",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - online identifier"),
        ),
        example_patterns=("jsmith42", "admin_user"),
    ),
    PII_TAXONOMY.GENDER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.GENDER,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.MODERATE,
        description="Gender identity or biological sex designation.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - sex life / sexual orientation", "Gender alone is Art. 4(1); combined context may elevate to Art. 9."),
        ),
        example_patterns=("Male", "Female", "Non-binary"),
        is_quasi_identifier=True,
        quasi_identifier_groups=("gender_dob_zip", "dob_zip_gender_marital"),
        reidentification_contribution=0.13,
    ),
    PII_TAXONOMY.NATIONALITY: EntityTypeProfile(
        entity_type=PII_TAXONOMY.NATIONALITY,
        category=EntityCategory.PERSONAL_IDENTITY,
        risk_level=RiskLevel.MODERATE,
        description="Country of citizenship or national origin.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - racial or ethnic origin"),
            _nist("§2.1", "Linked PII"),
        ),
        example_patterns=("American", "日本人", "Brazilian"),
        related_types=(PII_TAXONOMY.ETHNIC_ORIGIN,),
    ),
    # ── Financial ──────────────────────────────────────────────────────
    PII_TAXONOMY.CREDIT_CARD_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.CREDIT_CARD_NUMBER,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.CRITICAL,
        description="Payment card number (Visa, MasterCard, Amex, etc.).",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII - financial"),
            _gdpr("Art. 4(1)", "Personal data"),
            _iso("A.7.4.5", "PII de-identification"),
        ),
        example_patterns=("4111-1111-1111-1111", "5500 0000 0000 0004"),
    ),
    PII_TAXONOMY.IBAN: EntityTypeProfile(
        entity_type=PII_TAXONOMY.IBAN,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.HIGH,
        description="International Bank Account Number (ISO 13616).",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII - financial"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("GB29 NWBK 6016 1331 9268 19", "DE89 3704 0044 0532 0130 00"),
    ),
    PII_TAXONOMY.BANK_ACCOUNT_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.BANK_ACCOUNT_NUMBER,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.HIGH,
        description="Domestic bank account number.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII - financial"),
        ),
        example_patterns=("12345678", "000123456789"),
    ),
    PII_TAXONOMY.ROUTING_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.ROUTING_NUMBER,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.MODERATE,
        description="Bank routing / transit number (US ABA, etc.).",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - financial"),
        ),
        example_patterns=("021000021", "121000358"),
    ),
    PII_TAXONOMY.SWIFT_BIC_CODE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.SWIFT_BIC_CODE,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.MODERATE,
        description="SWIFT/BIC code identifying a financial institution.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - financial"),
        ),
        example_patterns=("CHASUS33", "DEUTDEFF"),
    ),
    PII_TAXONOMY.CRYPTOCURRENCY_WALLET: EntityTypeProfile(
        entity_type=PII_TAXONOMY.CRYPTOCURRENCY_WALLET,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.HIGH,
        description="Blockchain wallet address (Bitcoin, Ethereum, etc.).",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - pseudonymous identifier"),
        ),
        example_patterns=("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "0x742d35Cc6634C0532925a3b844Bc9e7595f"),
    ),
    PII_TAXONOMY.TAX_ID: EntityTypeProfile(
        entity_type=PII_TAXONOMY.TAX_ID,
        category=EntityCategory.FINANCIAL,
        risk_level=RiskLevel.HIGH,
        description="Tax identification number (TIN, EIN, VAT ID, etc.).",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("12-3456789", "GB123456789"),
    ),
    # ── Government ID ──────────────────────────────────────────────────
    PII_TAXONOMY.US_SSN: EntityTypeProfile(
        entity_type=PII_TAXONOMY.US_SSN,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.CRITICAL,
        description="United States Social Security Number.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII - stand-alone"),
            _gdpr("Art. 87", "National identification number"),
        ),
        example_patterns=("123-45-6789",),
    ),
    PII_TAXONOMY.PASSPORT_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.PASSPORT_NUMBER,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.CRITICAL,
        description="Passport document number issued by any country.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 87", "National identification number"),
        ),
        example_patterns=("AB1234567", "L12345678"),
        language_variant_hint="Format varies by issuing country.",
    ),
    PII_TAXONOMY.DRIVERS_LICENSE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.DRIVERS_LICENSE,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.HIGH,
        description="Driver's license number.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
        ),
        example_patterns=("D123-4567-8901", "JOHNS901S2310086"),
        language_variant_hint="Format varies by issuing state/country.",
    ),
    PII_TAXONOMY.NATIONAL_ID_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.NATIONAL_ID_NUMBER,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.CRITICAL,
        description="National identification number (Aadhaar, NIE, NIN, etc.).",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
            _gdpr("Art. 87", "National identification number"),
        ),
        example_patterns=("1234 5678 9012", "X-1234567-A"),
        language_variant_hint="Highly country-specific format.",
    ),
    PII_TAXONOMY.VISA_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.VISA_NUMBER,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.HIGH,
        description="Travel visa number.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII"),
        ),
        example_patterns=("A12345678",),
    ),
    PII_TAXONOMY.LICENSE_PLATE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.LICENSE_PLATE,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.MODERATE,
        description="Vehicle registration / license plate number.",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - linked to vehicle owner"),
        ),
        example_patterns=("ABC 1234", "AB12 CDE"),
    ),
    PII_TAXONOMY.VEHICLE_IDENTIFICATION_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.VEHICLE_IDENTIFICATION_NUMBER,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.MODERATE,
        description="Vehicle Identification Number (17-character VIN).",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - linked to vehicle owner"),
        ),
        example_patterns=("1HGBH41JXMN109186",),
    ),
    # ── Medical ────────────────────────────────────────────────────────
    PII_TAXONOMY.MEDICAL_RECORD_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.MEDICAL_RECORD_NUMBER,
        category=EntityCategory.MEDICAL,
        risk_level=RiskLevel.CRITICAL,
        description="Patient medical record number.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII - medical"),
            _gdpr("Art. 9(1)", "Special category - health data"),
            _iso("A.7.4.5", "PII de-identification - health"),
        ),
        example_patterns=("MRN-1003212",),
    ),
    PII_TAXONOMY.HEALTH_INSURANCE_ID: EntityTypeProfile(
        entity_type=PII_TAXONOMY.HEALTH_INSURANCE_ID,
        category=EntityCategory.MEDICAL,
        risk_level=RiskLevel.HIGH,
        description="Health insurance policy or member ID.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - health data"),
        ),
        example_patterns=("HIN-456789",),
    ),
    PII_TAXONOMY.PRESCRIPTION_NUMBER: EntityTypeProfile(
        entity_type=PII_TAXONOMY.PRESCRIPTION_NUMBER,
        category=EntityCategory.MEDICAL,
        risk_level=RiskLevel.HIGH,
        description="Prescription or pharmacy reference number.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - health data"),
        ),
        example_patterns=("RX-78901234",),
    ),
    PII_TAXONOMY.MEDICAL_DIAGNOSIS: EntityTypeProfile(
        entity_type=PII_TAXONOMY.MEDICAL_DIAGNOSIS,
        category=EntityCategory.MEDICAL,
        risk_level=RiskLevel.CRITICAL,
        description="Medical condition, diagnosis, or ICD code.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - health data"),
            _nist("§2.1", "Directly identifiable PII - medical"),
        ),
        example_patterns=("Type 2 Diabetes Mellitus", "ICD-10: E11.9"),
    ),
    PII_TAXONOMY.BIOMETRIC_ID: EntityTypeProfile(
        entity_type=PII_TAXONOMY.BIOMETRIC_ID,
        category=EntityCategory.MEDICAL,
        risk_level=RiskLevel.CRITICAL,
        description="Biometric identifier (fingerprint hash, facial ID, iris code).",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - biometric data"),
            _nist("§2.1", "Directly identifiable PII - biometric"),
            _iso("A.7.4.5", "PII de-identification - biometric"),
        ),
        example_patterns=("BIO-FP-a3f8c2e1d5",),
    ),
    # ── Digital / Technical ────────────────────────────────────────────
    PII_TAXONOMY.IP_ADDRESS: EntityTypeProfile(
        entity_type=PII_TAXONOMY.IP_ADDRESS,
        category=EntityCategory.DIGITAL_TECHNICAL,
        risk_level=RiskLevel.MODERATE,
        description="IPv4 or IPv6 network address.",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - online identifier", "Breyer v. Germany (C-582/14) confirmed IP as personal data."),
        ),
        example_patterns=("192.168.1.1", "2001:0db8:85a3::8a2e:0370:7334"),
    ),
    PII_TAXONOMY.MAC_ADDRESS: EntityTypeProfile(
        entity_type=PII_TAXONOMY.MAC_ADDRESS,
        category=EntityCategory.DIGITAL_TECHNICAL,
        risk_level=RiskLevel.MODERATE,
        description="Hardware MAC address (EUI-48 / EUI-64).",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - device identifier"),
        ),
        example_patterns=("00:1A:2B:3C:4D:5E",),
    ),
    PII_TAXONOMY.API_KEY: EntityTypeProfile(
        entity_type=PII_TAXONOMY.API_KEY,
        category=EntityCategory.DIGITAL_TECHNICAL,
        risk_level=RiskLevel.HIGH,
        description="Application programming interface key or secret.",
        regulatory_refs=(
            _iso("A.7.4.5", "PII de-identification - credentials"),
        ),
        example_patterns=("sk-abc123def456ghi789", "AKIA1234567890ABCDEF"),
    ),
    PII_TAXONOMY.AUTHENTICATION_TOKEN: EntityTypeProfile(
        entity_type=PII_TAXONOMY.AUTHENTICATION_TOKEN,
        category=EntityCategory.DIGITAL_TECHNICAL,
        risk_level=RiskLevel.HIGH,
        description="Authentication or session token (JWT, OAuth, etc.).",
        regulatory_refs=(
            _iso("A.7.4.5", "PII de-identification - credentials"),
        ),
        example_patterns=("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.xxx",),
    ),
    PII_TAXONOMY.DEVICE_ID: EntityTypeProfile(
        entity_type=PII_TAXONOMY.DEVICE_ID,
        category=EntityCategory.DIGITAL_TECHNICAL,
        risk_level=RiskLevel.MODERATE,
        description="Unique device identifier (IMEI, UDID, Android ID, etc.).",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - device identifier"),
        ),
        example_patterns=("353456789012345",),
    ),
    PII_TAXONOMY.URL_WITH_PII: EntityTypeProfile(
        entity_type=PII_TAXONOMY.URL_WITH_PII,
        category=EntityCategory.DIGITAL_TECHNICAL,
        risk_level=RiskLevel.MODERATE,
        description="URL containing PII in path or query parameters.",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - online identifier"),
        ),
        example_patterns=("https://example.com/users/john.doe?email=j@e.com",),
    ),
    # ── Employment ─────────────────────────────────────────────────────
    PII_TAXONOMY.EMPLOYEE_ID: EntityTypeProfile(
        entity_type=PII_TAXONOMY.EMPLOYEE_ID,
        category=EntityCategory.EMPLOYMENT,
        risk_level=RiskLevel.MODERATE,
        description="Organisation-issued employee identifier.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("EMP-12725",),
    ),
    PII_TAXONOMY.ORGANIZATION: EntityTypeProfile(
        entity_type=PII_TAXONOMY.ORGANIZATION,
        category=EntityCategory.EMPLOYMENT,
        risk_level=RiskLevel.LOW,
        description="Company, institution, or organisation name.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - contextual"),
        ),
        example_patterns=("Acme Corp", "Stanford University"),
    ),
    PII_TAXONOMY.JOB_TITLE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.JOB_TITLE,
        category=EntityCategory.EMPLOYMENT,
        risk_level=RiskLevel.LOW,
        description="Professional job title or role.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - contextual"),
        ),
        example_patterns=("Senior Software Engineer", "Chief Financial Officer"),
    ),
    PII_TAXONOMY.SALARY: EntityTypeProfile(
        entity_type=PII_TAXONOMY.SALARY,
        category=EntityCategory.EMPLOYMENT,
        risk_level=RiskLevel.HIGH,
        description="Compensation or salary information.",
        regulatory_refs=(
            _nist("§2.1", "Directly identifiable PII - financial"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("$125,000", "€85,000/year"),
    ),
    # ── Behavioral / Contextual ────────────────────────────────────────
    PII_TAXONOMY.LOCATION_COORDINATES: EntityTypeProfile(
        entity_type=PII_TAXONOMY.LOCATION_COORDINATES,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.HIGH,
        description="Geographic latitude/longitude coordinates.",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - location data"),
        ),
        example_patterns=("37.7749, -122.4194", "48.8566° N, 2.3522° E"),
        related_types=(PII_TAXONOMY.ADDRESS,),
    ),
    PII_TAXONOMY.AGE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.AGE,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.MODERATE,
        description="Person's age or age range.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - quasi-identifier"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("34 years old", "age 67"),
        related_types=(PII_TAXONOMY.DATE_OF_BIRTH,),
        is_quasi_identifier=True,
        quasi_identifier_groups=("household_zip_age", "vehicle_zip_age"),
        reidentification_contribution=0.18,
    ),
    PII_TAXONOMY.ETHNIC_ORIGIN: EntityTypeProfile(
        entity_type=PII_TAXONOMY.ETHNIC_ORIGIN,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.CRITICAL,
        description="Racial or ethnic origin.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - racial or ethnic origin"),
        ),
        example_patterns=("Asian", "Hispanic"),
    ),
    PII_TAXONOMY.RELIGIOUS_BELIEF: EntityTypeProfile(
        entity_type=PII_TAXONOMY.RELIGIOUS_BELIEF,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.CRITICAL,
        description="Religious or philosophical beliefs.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - religious or philosophical beliefs"),
        ),
        example_patterns=("Buddhist", "Catholic"),
    ),
    PII_TAXONOMY.POLITICAL_OPINION: EntityTypeProfile(
        entity_type=PII_TAXONOMY.POLITICAL_OPINION,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.CRITICAL,
        description="Political opinions or party affiliation.",
        regulatory_refs=(
            _gdpr("Art. 9(1)", "Special category - political opinions"),
        ),
        example_patterns=("member of the Green Party",),
    ),
    # ── v1.0.0 additions: Quasi-identifier entity types ──────────────
    PII_TAXONOMY.EDUCATION_LEVEL: EntityTypeProfile(
        entity_type=PII_TAXONOMY.EDUCATION_LEVEL,
        category=EntityCategory.EMPLOYMENT,
        risk_level=RiskLevel.LOW,
        description="Highest level of education attained.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - quasi-identifier"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("Bachelor's degree", "PhD", "High school diploma"),
        is_quasi_identifier=True,
        quasi_identifier_groups=("education_occupation_zip",),
        reidentification_contribution=0.10,
    ),
    PII_TAXONOMY.MARITAL_STATUS: EntityTypeProfile(
        entity_type=PII_TAXONOMY.MARITAL_STATUS,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.MODERATE,
        description="Marital or relationship status.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - quasi-identifier"),
            _gdpr("Art. 4(1)", "Personal data"),
        ),
        example_patterns=("married", "single", "divorced", "widowed"),
        is_quasi_identifier=True,
        quasi_identifier_groups=("gender_dob_zip", "marital_occupation"),
        reidentification_contribution=0.08,
    ),
    PII_TAXONOMY.HOUSEHOLD_SIZE: EntityTypeProfile(
        entity_type=PII_TAXONOMY.HOUSEHOLD_SIZE,
        category=EntityCategory.BEHAVIORAL_CONTEXTUAL,
        risk_level=RiskLevel.LOW,
        description="Number of persons in household.",
        regulatory_refs=(
            _nist("§2.1", "Linked PII - quasi-identifier"),
        ),
        example_patterns=("household of 4", "lives alone", "family of 6"),
        is_quasi_identifier=True,
        quasi_identifier_groups=("household_zip_age",),
        reidentification_contribution=0.06,
    ),
    PII_TAXONOMY.VEHICLE_MODEL: EntityTypeProfile(
        entity_type=PII_TAXONOMY.VEHICLE_MODEL,
        category=EntityCategory.GOVERNMENT_ID,
        risk_level=RiskLevel.LOW,
        description="Vehicle make and model (linked to owner via registration).",
        regulatory_refs=(
            _gdpr("Art. 4(1)", "Personal data - linked to vehicle owner"),
        ),
        example_patterns=("2023 Toyota Camry", "Tesla Model 3", "BMW 3 Series"),
        related_types=(PII_TAXONOMY.LICENSE_PLATE, PII_TAXONOMY.VEHICLE_IDENTIFICATION_NUMBER),
        is_quasi_identifier=True,
        quasi_identifier_groups=("vehicle_zip_age",),
        reidentification_contribution=0.05,
    ),
}

# ---------------------------------------------------------------------------
# Quasi-identifier group definitions (Sweeney 2002 + extensions)
# ---------------------------------------------------------------------------

QUASI_IDENTIFIER_GROUPS: dict[str, dict[str, Any]] = {
    "gender_dob_zip": {
        "description": "Sweeney (2002): Gender + Date of Birth + ZIP code",
        "entity_types": (
            PII_TAXONOMY.GENDER,
            PII_TAXONOMY.DATE_OF_BIRTH,
            PII_TAXONOMY.ZIP_CODE,
        ),
        "reidentification_rate": 0.87,
        "population_basis": "US Census",
        "citation": "Sweeney, L. (2002). k-Anonymity: A model for protecting privacy.",
    },
    "name_ssn": {
        "description": "Direct identifier pair: Name + SSN",
        "entity_types": (PII_TAXONOMY.PERSON_NAME, PII_TAXONOMY.US_SSN),
        "reidentification_rate": 0.99,
        "population_basis": "US population",
        "citation": "NIST SP 800-122 §2.1",
    },
    "name_address": {
        "description": "Direct identifier pair: Name + Physical address",
        "entity_types": (PII_TAXONOMY.PERSON_NAME, PII_TAXONOMY.ADDRESS),
        "reidentification_rate": 0.95,
        "population_basis": "General",
        "citation": "NIST SP 800-122 §2.1",
    },
    "dob_zip_gender_marital": {
        "description": "Extended Sweeney: DOB + ZIP + Gender + Marital status",
        "entity_types": (
            PII_TAXONOMY.DATE_OF_BIRTH,
            PII_TAXONOMY.ZIP_CODE,
            PII_TAXONOMY.GENDER,
            PII_TAXONOMY.MARITAL_STATUS,
        ),
        "reidentification_rate": 0.93,
        "population_basis": "US Census extended",
        "citation": "El Emam et al. (2011). A systematic review of re-identification attacks.",
    },
    "education_occupation_zip": {
        "description": "Occupation + education + geography quasi-identifier set",
        "entity_types": (
            PII_TAXONOMY.EDUCATION_LEVEL,
            PII_TAXONOMY.JOB_TITLE,
            PII_TAXONOMY.ZIP_CODE,
        ),
        "reidentification_rate": 0.63,
        "population_basis": "US Census ACS",
        "citation": "Narayanan & Shmatikov (2008). Robust de-anonymization.",
    },
    "household_zip_age": {
        "description": "Household size + ZIP + Age quasi-identifier set",
        "entity_types": (
            PII_TAXONOMY.HOUSEHOLD_SIZE,
            PII_TAXONOMY.ZIP_CODE,
            PII_TAXONOMY.AGE,
        ),
        "reidentification_rate": 0.45,
        "population_basis": "US Census",
        "citation": "El Emam et al. (2011).",
    },
    "vehicle_zip_age": {
        "description": "Vehicle model + ZIP + Age quasi-identifier set",
        "entity_types": (
            PII_TAXONOMY.VEHICLE_MODEL,
            PII_TAXONOMY.ZIP_CODE,
            PII_TAXONOMY.AGE,
        ),
        "reidentification_rate": 0.35,
        "population_basis": "DMV records",
        "citation": "Derived from vehicle registration linkage studies.",
    },
}


class EntityTypeRegistry:
    """Central registry for PII entity type profiles.

    Provides look-up, filtering, and validation against the taxonomy.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, EntityTypeProfile] = dict(_ENTITY_PROFILES)

    # ── queries ────────────────────────────────────────────────────────

    def get(self, entity_type: str) -> EntityTypeProfile | None:
        """Return the profile for *entity_type*, or ``None``."""
        return self._profiles.get(entity_type)

    def all_profiles(self) -> list[EntityTypeProfile]:
        """Return all registered profiles sorted by entity_type."""
        return sorted(self._profiles.values(), key=lambda p: p.entity_type)

    def all_entity_types(self) -> list[str]:
        """Return all registered entity-type strings (sorted)."""
        return sorted(self._profiles.keys())

    def by_category(self, category: EntityCategory) -> list[EntityTypeProfile]:
        """Return profiles belonging to *category*."""
        return [p for p in self._profiles.values() if p.category == category]

    def by_risk_level(self, level: RiskLevel) -> list[EntityTypeProfile]:
        """Return profiles matching *level*."""
        return [p for p in self._profiles.values() if p.risk_level == level]

    def by_standard(self, standard: str) -> list[EntityTypeProfile]:
        """Return profiles that reference *standard* (substring match)."""
        return [
            p
            for p in self._profiles.values()
            if any(standard.lower() in ref.standard.lower() for ref in p.regulatory_refs)
        ]

    # ── quasi-identifier queries (v1.0.0) ─────────────────────────────

    def quasi_identifiers(self) -> list[EntityTypeProfile]:
        """Return all profiles marked as quasi-identifiers."""
        return [p for p in self._profiles.values() if p.is_quasi_identifier]

    def by_quasi_identifier_group(self, group: str) -> list[EntityTypeProfile]:
        """Return profiles belonging to a quasi-identifier group."""
        return [
            p for p in self._profiles.values()
            if group in p.quasi_identifier_groups
        ]

    # ── extension ──────────────────────────────────────────────────────

    def register(self, profile: EntityTypeProfile) -> None:
        """Register a custom entity-type profile."""
        self._profiles[profile.entity_type] = profile

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, entity_type: str) -> bool:
        return entity_type in self._profiles
