"""Dataset loaders and taxonomy mapping for swarm training.

Supports loading from multiple industry-standard PII/NER datasets and
mapping their entity type labels to the pii-anon canonical taxonomy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """Unified training record across all datasets."""

    record_id: str
    text: str
    labels: list[dict[str, Any]]  # [{"entity_type": str, "start": int, "end": int}]
    language: str = "en"
    source_dataset: str = ""


# ── Canonical taxonomy mapping ──────────────────────────────────────────────

TAXONOMY_MAP: dict[str, dict[str, str]] = {
    "pii_anon_eval": {
        # pii-anon-eval-data entity types → canonical benchmark names
        "SOCIAL_SECURITY_NUMBER": "US_SSN",
        "STREET_ADDRESS": "ADDRESS",
        "ORGANIZATION_NAME": "ORGANIZATION",
        "PASSPORT_NUMBER": "PASSPORT",
        "DRIVER_LICENSE_NUMBER": "DRIVERS_LICENSE",
        "BANK_ACCOUNT_NUMBER": "BANK_ACCOUNT",
        "BANK_ROUTING_NUMBER": "ROUTING_NUMBER",
        "NATIONAL_ID_NUMBER": "NATIONAL_ID",
        "LOCATION_NAME": "LOCATION",
        "CRYPTOCURRENCY_ADDRESS": "CRYPTO_WALLET",
        "SOCIAL_MEDIA_HANDLE": "USERNAME",
        "VEHICLE_IDENTIFICATION_NUMBER": "VIN",
        "CREDIT_CARD_NUMBER": "CREDIT_CARD",
        "CREDIT_CARD_FRAGMENT": "CREDIT_CARD",
        "LATITUDE_LONGITUDE": "_IGNORE",
        "TIMESTAMP": "_IGNORE",
        "POSTAL_CODE": "ADDRESS",
        "SWIFT_BIC_CODE": "_IGNORE",
        "HEALTH_INSURANCE_ID": "MEDICAL_RECORD_NUMBER",
        "DEVICE_IDENTIFIER": "MAC_ADDRESS",
        "TAX_ID": "NATIONAL_ID",
        "VISA_NUMBER": "PASSPORT",
    },
    "ai4privacy": {
        "firstname": "PERSON_NAME",
        "lastname": "PERSON_NAME",
        "middlename": "PERSON_NAME",
        "email": "EMAIL_ADDRESS",
        "phone_number": "PHONE_NUMBER",
        "street_address": "ADDRESS",
        "city": "LOCATION",
        "state": "LOCATION",
        "zipcode": "ADDRESS",
        "country": "LOCATION",
        "date": "DATE_OF_BIRTH",
        "date_of_birth": "DATE_OF_BIRTH",
        "ssn": "US_SSN",
        "credit_card_number": "CREDIT_CARD",
        "credit_card_expiry": "_IGNORE",
        "credit_card_cvv": "_IGNORE",
        "iban": "IBAN",
        "ip_address": "IP_ADDRESS",
        "ipv4": "IP_ADDRESS",
        "ipv6": "IP_ADDRESS",
        "mac_address": "MAC_ADDRESS",
        "username": "USERNAME",
        "password": "_IGNORE",
        "company_name": "ORGANIZATION",
        "job_title": "_IGNORE",
        "passport_number": "PASSPORT",
        "driver_license": "DRIVERS_LICENSE",
        "bank_account_number": "BANK_ACCOUNT",
        "routing_number": "ROUTING_NUMBER",
        "swift_bic_code": "_IGNORE",
        "tax_id": "NATIONAL_ID",
        "national_id": "NATIONAL_ID",
        "vehicle_identification_number": "VIN",
        "license_plate": "LICENSE_PLATE",
        "medical_record_number": "MEDICAL_RECORD_NUMBER",
        "employee_id": "EMPLOYEE_ID",
        "cryptocurrency_address": "CRYPTO_WALLET",
    },
    "conll2003": {
        "PER": "PERSON_NAME",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "MISC": "_IGNORE",
    },
    "tab": {
        "PERSON": "PERSON_NAME",
        "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
        "DATE": "DATE_OF_BIRTH",
        "PHONE": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "URL": "_IGNORE",
        "ID": "NATIONAL_ID",
        "CREDIT_CARD": "CREDIT_CARD",
        "FINANCIAL": "BANK_ACCOUNT",
        "HEALTH": "MEDICAL_RECORD_NUMBER",
        "QUANTITY": "_IGNORE",
    },
    "bigcode": {
        "EMAIL": "EMAIL_ADDRESS",
        "IP_ADDRESS": "IP_ADDRESS",
        "KEY": "_IGNORE",
        "NAME": "PERSON_NAME",
        "PASSWORD": "_IGNORE",
        "USERNAME": "USERNAME",
    },
    "i2b2": {
        "PATIENT": "PERSON_NAME",
        "DOCTOR": "PERSON_NAME",
        "USERNAME": "USERNAME",
        "PROFESSION": "_IGNORE",
        "ROOM": "_IGNORE",
        "DEPARTMENT": "ORGANIZATION",
        "HOSPITAL": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "STREET": "ADDRESS",
        "CITY": "LOCATION",
        "STATE": "LOCATION",
        "COUNTRY": "LOCATION",
        "ZIP": "ADDRESS",
        "LOCATION-OTHER": "LOCATION",
        "AGE": "_IGNORE",
        "DATE": "DATE_OF_BIRTH",
        "PHONE": "PHONE_NUMBER",
        "FAX": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "URL": "_IGNORE",
        "SSN": "US_SSN",
        "MEDICALRECORD": "MEDICAL_RECORD_NUMBER",
        "HEALTHPLAN": "MEDICAL_RECORD_NUMBER",
        "ACCOUNT": "BANK_ACCOUNT",
        "LICENSE": "DRIVERS_LICENSE",
        "VEHICLE": "VIN",
        "DEVICE": "MAC_ADDRESS",
        "BIOID": "_IGNORE",
        "IDNUM": "NATIONAL_ID",
    },
}


def map_entity_type(dataset_name: str, raw_type: str) -> str:
    """Map a dataset-specific entity type to the canonical pii-anon taxonomy."""
    mapping = TAXONOMY_MAP.get(dataset_name, {})
    mapped = mapping.get(raw_type, mapping.get(raw_type.lower(), raw_type))
    return mapped


# ── Dataset Loaders ─────────────────────────────────────────────────────────

def load_pii_anon_data(max_records: int | None = None) -> list[TrainingRecord]:
    """Load from the pii-anon-datasets package."""
    try:
        import pii_anon_datasets
    except ImportError:
        logger.warning("pii-anon-datasets not installed; skipping pii_anon_eval")
        return []

    raw = pii_anon_datasets.load_dataset()
    records: list[TrainingRecord] = []
    for i, row in enumerate(raw):
        if max_records is not None and i >= max_records:
            break
        labels = []
        for ann in row.get("annotations", []):
            mapped_type = map_entity_type("pii_anon_eval", ann.get("entity_type", ""))
            if mapped_type == "_IGNORE":
                continue
            labels.append({
                "entity_type": mapped_type,
                "start": ann["start"],
                "end": ann["end"],
            })
        records.append(TrainingRecord(
            record_id=str(row.get("record_id", f"pii-data-{i}")),
            text=str(row.get("text", "")),
            labels=labels,
            language=str(row.get("language", "en")),
            source_dataset="pii_anon_eval",
        ))
    logger.info("Loaded %d records from pii-anon-eval-data", len(records))
    return records


def load_ai4privacy(max_records: int | None = None) -> list[TrainingRecord]:
    """Load from AI4Privacy pii-masking-200k via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("HuggingFace datasets not installed; skipping ai4privacy")
        return []

    try:
        ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    except Exception as exc:
        logger.warning("Failed to load ai4privacy: %s", exc)
        return []

    records: list[TrainingRecord] = []
    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break
        text = row.get("source_text", row.get("masked_text", ""))
        labels = []
        for ann in row.get("privacy_mask", []):
            raw_type = ann.get("label", "")
            mapped_type = map_entity_type("ai4privacy", raw_type)
            if mapped_type == "_IGNORE":
                continue
            labels.append({
                "entity_type": mapped_type,
                "start": ann.get("start", 0),
                "end": ann.get("end", 0),
            })
        records.append(TrainingRecord(
            record_id=f"ai4p-{i}",
            text=text,
            labels=labels,
            language=row.get("language", "en"),
            source_dataset="ai4privacy",
        ))
    logger.info("Loaded %d records from ai4privacy", len(records))
    return records


def load_conll2003(max_records: int | None = None) -> list[TrainingRecord]:
    """Load CoNLL-2003 NER dataset via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("HuggingFace datasets not installed; skipping conll2003")
        return []

    try:
        ds = load_dataset("conll2003", split="train")
    except Exception as exc:
        logger.warning("Failed to load conll2003: %s", exc)
        return []

    ner_tags_map = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
                    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}

    records: list[TrainingRecord] = []
    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break
        tokens = row.get("tokens", [])
        ner_tags = row.get("ner_tags", [])
        text = " ".join(tokens)

        labels = []
        offset = 0
        j = 0
        while j < len(tokens):
            token = tokens[j]
            tag = ner_tags_map.get(ner_tags[j], "O") if j < len(ner_tags) else "O"
            token_start = offset
            token_end = offset + len(token)

            if tag.startswith("B-"):
                bio_type = tag[2:]
                entity_start = token_start
                entity_end = token_end
                j += 1
                while j < len(tokens):
                    next_tag = ner_tags_map.get(ner_tags[j], "O") if j < len(ner_tags) else "O"
                    if next_tag == f"I-{bio_type}":
                        entity_end = offset + 1 + len(tokens[j])
                        offset += 1 + len(tokens[j])
                        j += 1
                    else:
                        break
                mapped_type = map_entity_type("conll2003", bio_type)
                if mapped_type != "_IGNORE":
                    labels.append({
                        "entity_type": mapped_type,
                        "start": entity_start,
                        "end": entity_end,
                    })
                offset = entity_end + 1 if j < len(tokens) else entity_end
                continue

            offset = token_end + 1
            j += 1

        records.append(TrainingRecord(
            record_id=f"conll-{i}",
            text=text,
            labels=labels,
            language="en",
            source_dataset="conll2003",
        ))
    logger.info("Loaded %d records from conll2003", len(records))
    return records


# ── Unified Loader ──────────────────────────────────────────────────────────

DATASET_LOADERS: dict[str, Any] = {
    "pii_anon_eval": load_pii_anon_data,
    "ai4privacy": load_ai4privacy,
    "conll2003": load_conll2003,
}


def load_training_data(
    datasets: list[str] | None = None,
    max_records_per_dataset: int | None = None,
) -> list[TrainingRecord]:
    """Load and combine training data from multiple sources."""
    if datasets is None:
        datasets = list(DATASET_LOADERS.keys())

    all_records: list[TrainingRecord] = []
    for ds_name in datasets:
        loader = DATASET_LOADERS.get(ds_name)
        if loader is None:
            logger.warning("Unknown dataset '%s'; skipping", ds_name)
            continue
        try:
            records = loader(max_records=max_records_per_dataset)
            all_records.extend(records)
        except Exception as exc:
            logger.warning("Failed to load dataset '%s': %s", ds_name, exc)

    logger.info("Total training records loaded: %d from %d datasets", len(all_records), len(datasets))
    return all_records
