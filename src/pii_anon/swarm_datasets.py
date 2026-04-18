"""Dataset loaders and taxonomy mapping for swarm training.

Supports loading from multiple industry-standard PII/NER datasets and
mapping their entity type labels to the pii-anon canonical taxonomy.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """Unified training record across all datasets.

    The base fields (``record_id``, ``text``, ``labels``, ``language``,
    ``source_dataset``) are populated by every loader.  The Tier 3 fields
    below are populated only by :func:`load_pii_anon_data` when the
    ``pii-anon-datasets`` v1.3.0+ behavioral-signal annotations are
    present.  They are exposed as features to
    :class:`~pii_anon.swarm_learner.XGBoostMetaLearner` so the meta-learner
    can learn that high-RRS records deserve a higher precision bar.

    For paired-profile / ESRC-attack records (``pii-anon-datasets`` v1.3.0),
    :attr:`persona_id` lets the trainer group records that share an
    underlying identity so cross-record co-occurrence becomes a feature.
    """

    record_id: str
    text: str
    labels: list[dict[str, Any]]  # [{"entity_type": str, "start": int, "end": int}]
    language: str = "en"
    source_dataset: str = ""

    # Tier 3 (dataset v1.3.0+) — optional, zero-valued when unknown.
    behavioral_signal_density: float = 0.0
    re_identification_resistance_score: float | None = None
    persona_id: str | None = None
    is_paired_profile: bool = False


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
        # Text Anonymization Benchmark (Pilan et al. 2022, arXiv:2202.00443) —
        # curated European Court of Human Rights documents.  Mapping
        # mirrors the pii-rate-elo paper pipeline's converter so training
        # pools stay aligned with the evaluation reference.
        "PERSON": "PERSON_NAME",
        "PER": "PERSON_NAME",
        "NAME": "PERSON_NAME",
        "DEM": "PERSON_NAME",            # demographic marker
        "LOCATION": "LOCATION",
        "LOC": "LOCATION",
        "CITY": "LOCATION",
        "COUNTRY": "LOCATION",
        "GPE": "LOCATION",
        "ORGANIZATION": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "INSTITUTION": "ORGANIZATION",
        "COMPANY": "ORGANIZATION",
        "DATE": "DATE_OF_BIRTH",
        "DATETIME": "DATE_OF_BIRTH",
        "TIME": "DATE_OF_BIRTH",
        "PHONE": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "URL": "_IGNORE",
        "ID": "NATIONAL_ID",
        "CODE": "NATIONAL_ID",
        "MISC": "_IGNORE",
        "CREDIT_CARD": "CREDIT_CARD",
        "FINANCIAL": "BANK_ACCOUNT",
        "HEALTH": "MEDICAL_RECORD_NUMBER",
        "QUANTITY": "_IGNORE",
        "ADDRESS": "ADDRESS",
        "STREET": "ADDRESS",
    },
    "meddocan": {
        # MEDDOCAN (IberLEF 2019, Marimon et al.) — Spanish clinical PHI.
        # ~33K PHI annotations across 29 entity types aligned with HIPAA.
        # Adds non-English clinical coverage our English synthetic
        # corpora don't give us.
        "NOMBRE_SUJETO_ASISTENCIA": "PERSON_NAME",
        "NOMBRE_PERSONAL_SANITARIO": "PERSON_NAME",
        "FAMILIARES_SUJETO_ASISTENCIA": "PERSON_NAME",
        "NOMBRE_INSTITUCION": "ORGANIZATION",
        "INSTITUCION": "ORGANIZATION",
        "HOSPITAL": "ORGANIZATION",
        "CENTRO_SALUD": "ORGANIZATION",
        "EDAD_SUJETO_ASISTENCIA": "_IGNORE",      # age — not a direct identifier in our taxonomy
        "FECHAS": "DATE_OF_BIRTH",
        "TERRITORIO": "LOCATION",
        "PAIS": "LOCATION",
        "PROFESION": "_IGNORE",
        "ID_SUJETO_ASISTENCIA": "MEDICAL_RECORD_NUMBER",
        "ID_ASEGURAMIENTO": "MEDICAL_RECORD_NUMBER",
        "ID_CONTACTO_ASISTENCIAL": "MEDICAL_RECORD_NUMBER",
        "ID_EMPLEO_PERSONAL_SANITARIO": "EMPLOYEE_ID",
        "ID_TITULACION_PERSONAL_SANITARIO": "EMPLOYEE_ID",
        "NUMERO_TELEFONO": "PHONE_NUMBER",
        "NUMERO_FAX": "PHONE_NUMBER",
        "CORREO_ELECTRONICO": "EMAIL_ADDRESS",
        "DIRECCION": "ADDRESS",
        "CALLE": "ADDRESS",
        "SEXO_SUJETO_ASISTENCIA": "_IGNORE",
        "OTRO_NUMERO_IDENTIF": "NATIONAL_ID",
    },
    "ai4privacy_400k": {
        # AI4Privacy PII-Masking-400k (2024) — 17 languages, 54 entity
        # types.  Superset of the 200k release; includes post-2023
        # additions (passport, VIN, crypto wallets).  Mapping mirrors
        # the paper pipeline so our training pool stays consistent with
        # the evaluation reference.
        "FIRSTNAME": "PERSON_NAME",
        "LASTNAME": "PERSON_NAME",
        "MIDDLENAME": "PERSON_NAME",
        "FULLNAME": "PERSON_NAME",
        "NAME": "PERSON_NAME",
        "PREFIX": "_IGNORE",
        "TITLE": "_IGNORE",
        "DISPLAYNAME": "USERNAME",
        "ACCOUNTNAME": "USERNAME",
        "USERNAME": "USERNAME",
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "PHONEIMEI": "_IGNORE",
        "URL": "_IGNORE",
        "STREETADDRESS": "ADDRESS",
        "BUILDINGNUM": "ADDRESS",
        "SECONDARYADDRESS": "ADDRESS",
        "ZIPCODE": "ADDRESS",
        "CITY": "LOCATION",
        "STATE": "LOCATION",
        "COUNTRY": "LOCATION",
        "COUNTY": "LOCATION",
        "ORDINALDIRECTION": "LOCATION",
        "NEARBYGPSCOORDINATE": "LOCATION",
        "COMPANYNAME": "ORGANIZATION",
        "COMPANYSUFFIX": "ORGANIZATION",
        "JOBAREA": "_IGNORE",
        "JOBTITLE": "_IGNORE",
        "JOBDESCRIPTOR": "_IGNORE",
        "JOBTYPE": "_IGNORE",
        "DATE": "DATE_OF_BIRTH",
        "DOB": "DATE_OF_BIRTH",
        "TIME": "DATE_OF_BIRTH",
        "AGE": "_IGNORE",
        "SSN": "US_SSN",
        "SOCIALSECURITYNUMBER": "US_SSN",
        "CREDITCARDNUMBER": "CREDIT_CARD",
        "CREDITCARDCVV": "_IGNORE",
        "CREDITCARDISSUER": "_IGNORE",
        "IBAN": "IBAN",
        "BIC": "_IGNORE",
        "ACCOUNTNUMBER": "BANK_ACCOUNT",
        "AMOUNT": "_IGNORE",
        "CURRENCY": "_IGNORE",
        "CURRENCYCODE": "_IGNORE",
        "CURRENCYNAME": "_IGNORE",
        "CURRENCYSYMBOL": "_IGNORE",
        "PIN": "_IGNORE",
        "PASSWORD": "_IGNORE",
        "MASKEDNUMBER": "NATIONAL_ID",
        "IP": "IP_ADDRESS",
        "IPADDRESS": "IP_ADDRESS",
        "IPV4": "IP_ADDRESS",
        "IPV6": "IP_ADDRESS",
        "MAC": "MAC_ADDRESS",
        "USERAGENT": "_IGNORE",
        "PASSPORTNUMBER": "PASSPORT",
        "DRIVERLICENSE": "DRIVERS_LICENSE",
        "VEHICLEVIN": "VIN",
        "VEHICLEVRM": "VIN",
        "BITCOINADDRESS": "CRYPTO_WALLET",
        "ETHEREUMADDRESS": "CRYPTO_WALLET",
        "LITECOINADDRESS": "CRYPTO_WALLET",
        "IMEI": "_IGNORE",
        "GENDER": "_IGNORE",
        "SEX": "_IGNORE",
        "EYECOLOR": "_IGNORE",
        "HEIGHT": "_IGNORE",
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
        # Tier 3 fields (v1.3.0+) — present as additive, backward-compat
        # additions so older dataset versions still load cleanly.
        behavioral = row.get("behavioral_signals") or {}
        privacy_risk = row.get("privacy_risk") or {}
        tier3 = row.get("tier3_evaluation") or {}
        bsd = float(behavioral.get("behavioral_signal_density", 0.0) or 0.0)
        rrs_raw = privacy_risk.get("re_identification_resistance_score")
        rrs = float(rrs_raw) if rrs_raw is not None else None
        persona = tier3.get("persona_id") if isinstance(tier3, dict) else None
        records.append(TrainingRecord(
            record_id=str(row.get("record_id", f"pii-data-{i}")),
            text=str(row.get("text", "")),
            labels=labels,
            language=str(row.get("language", "en")),
            source_dataset="pii_anon_eval",
            behavioral_signal_density=bsd,
            re_identification_resistance_score=rrs,
            persona_id=str(persona) if persona is not None else None,
            is_paired_profile=bool(tier3.get("is_paired_profile", False))
            if isinstance(tier3, dict) else False,
        ))
    tier3_count = sum(
        1 for r in records if r.re_identification_resistance_score is not None
    )
    logger.info(
        "Loaded %d records from pii-anon-eval-data (Tier 3 scored: %d, paired-profiles: %d)",
        len(records),
        tier3_count,
        sum(1 for r in records if r.is_paired_profile),
    )
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


# ── Industry-reference datasets — loaders ported from the paper pipeline ──
# These mirror the dataset mix used by the pii-rate-elo paper-submission
# evaluation (``../pii-anon-research-paper/pii-rate-elo-pipeline``).
# Adding them to the training pool keeps the swarm's training
# distribution aligned with the reference evaluation corpus:
#
#   pii_anon_eval       — our canonical synthetic benchmark (v1.3.0)
#   ai4privacy_400k     — 400K records, 17 languages, 54 entity types
#                         (2024 superset of the 200k release)
#   tab                 — Text Anonymization Benchmark (Pilan et al. 2022) —
#                         1,268 European Court of Human Rights documents
#                         with peer-reviewed manual annotations
#   meddocan            — MEDDOCAN (Marimon et al., IberLEF 2019) —
#                         Spanish clinical PHI benchmark, adds
#                         non-English clinical coverage
#
# All three are optional HuggingFace loads — they log a warning and
# return ``[]`` when ``datasets`` is missing or the HF hub is unreachable,
# so they are safe to include in the default ``SWARM_DATASETS`` list.


def load_ai4privacy_400k(max_records: int | None = None) -> list[TrainingRecord]:
    """Load AI4Privacy PII-Masking-400k (2024 release, 17 languages).

    Superset of ``load_ai4privacy`` (the 200k release).  Uses the same
    schema — each row has ``source_text`` plus a ``privacy_mask`` JSON
    list of ``{label, start, end, value}`` triples — so the span-parsing
    logic of the 200k loader is reused via delegation.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning(
            "HuggingFace datasets not installed; skipping ai4privacy_400k. "
            "Install with: pip install 'pii-anon[swarm-train]'"
        )
        return []

    try:
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train")
    except Exception as exc:
        logger.warning("Failed to load ai4privacy_400k: %s", exc)
        return []

    records: list[TrainingRecord] = []
    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break
        text = str(row.get("source_text") or row.get("text") or "")
        if not text:
            continue
        # The 400k release uses ``privacy_mask`` (JSON-list).  Older rows
        # may carry ``span_labels`` or raw ``labels`` — accept all three.
        raw_masks = row.get("privacy_mask") or row.get("span_labels") or row.get("labels") or []
        if isinstance(raw_masks, str):
            try:
                raw_masks = json.loads(raw_masks)
            except (json.JSONDecodeError, TypeError):
                raw_masks = []
        labels: list[dict[str, Any]] = []
        for mask in raw_masks or []:
            if not isinstance(mask, dict):
                continue
            raw_type = str(
                mask.get("label") or mask.get("entity_type") or ""
            ).strip()
            if not raw_type:
                continue
            mapped = map_entity_type("ai4privacy_400k", raw_type)
            if mapped in ("_IGNORE", ""):
                continue
            try:
                start = int(mask["start"])
                end = int(mask["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > len(text):
                continue
            labels.append({"entity_type": mapped, "start": start, "end": end})

        records.append(TrainingRecord(
            record_id=str(row.get("record_id") or f"ai4privacy-400k-{i}"),
            text=text,
            labels=labels,
            language=str(row.get("language", "en")),
            source_dataset="ai4privacy_400k",
        ))
    logger.info("Loaded %d records from ai4privacy_400k", len(records))
    return records


def load_tab(max_records: int | None = None) -> list[TrainingRecord]:
    """Load the Text Anonymization Benchmark (TAB — Pilan et al., 2022).

    ~1,268 court documents from the European Court of Human Rights with
    high-quality manual PII annotations.  Peer-reviewed, citable, and
    complements synthetic benchmarks with real-world legal text.

    HuggingFace: ``ildpil/text-anonymization-benchmark`` (primary) with
    a fallback to the ``mattmdjaga`` mirror.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning(
            "HuggingFace datasets not installed; skipping TAB. "
            "Install with: pip install 'pii-anon[swarm-train]'"
        )
        return []

    ds = None
    for hf_path in (
        "ildpil/text-anonymization-benchmark",
        "mattmdjaga/text-anonymization-benchmark-train",
    ):
        try:
            ds = load_dataset(hf_path, split="train")
            logger.info("Loaded TAB from %s", hf_path)
            break
        except Exception as exc:
            logger.debug("TAB load from %s failed: %s", hf_path, exc)
    if ds is None:
        logger.warning("All TAB HuggingFace paths failed; skipping")
        return []

    records: list[TrainingRecord] = []
    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        mentions = row.get("entity_mentions") or []
        if not isinstance(mentions, list):
            continue
        labels: list[dict[str, Any]] = []
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            raw_type = str(mention.get("entity_type") or "").strip()
            if not raw_type:
                continue
            mapped = map_entity_type("tab", raw_type)
            if mapped in ("_IGNORE", ""):
                continue
            raw_start = mention.get("start_offset", mention.get("start", -1))
            raw_end = mention.get("end_offset", mention.get("end", -1))
            if raw_start is None or raw_end is None:
                continue
            try:
                start = int(raw_start)
                end = int(raw_end)
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > len(text):
                continue
            labels.append({"entity_type": mapped, "start": start, "end": end})

        records.append(TrainingRecord(
            record_id=str(row.get("doc_id") or f"tab-{i}"),
            text=text,
            labels=labels,
            language=str(row.get("language", "en")),
            source_dataset="tab",
        ))
    logger.info("Loaded %d records from TAB", len(records))
    return records


def load_meddocan(max_records: int | None = None) -> list[TrainingRecord]:
    """Load MEDDOCAN (Marimon et al., IberLEF 2019) — Spanish clinical PHI.

    ~1,000 synthetic Spanish clinical case reports with ~33K PHI
    annotations across 29 entity types aligned with HIPAA.  Adds
    non-English clinical coverage the synthetic corpora don't give us.

    Requires the ``bigbio`` extension for the HuggingFace schema.  Falls
    back silently when unavailable — meddocan is an opt-in addition to
    the default training pool, not a hard requirement.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning(
            "HuggingFace datasets not installed; skipping MEDDOCAN. "
            "Install with: pip install 'pii-anon[swarm-train]'"
        )
        return []

    ds = None
    for hf_path, kwargs in (
        ("bigbio/meddocan", {"name": "meddocan_bigbio_kb"}),
        ("bigbio/meddocan", {}),
    ):
        try:
            ds = load_dataset(hf_path, split="train", **kwargs)
            logger.info("Loaded MEDDOCAN from %s (%s)", hf_path, kwargs or "default")
            break
        except Exception as exc:
            logger.debug("MEDDOCAN load from %s failed: %s", hf_path, exc)
    if ds is None:
        logger.warning(
            "MEDDOCAN load failed — requires pip install 'bigbio' or a "
            "local BRAT-format copy.  Skipping."
        )
        return []

    records: list[TrainingRecord] = []
    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break
        # BigBio ``bigbio_kb`` schema: passages[].text is the body,
        # entities[] have .type + .offsets[[start, end]].  Fall back to
        # plain ``text`` / ``annotations`` when the schema is flat.
        passages = row.get("passages") or []
        if passages:
            text = " ".join(
                str(p.get("text", [""])[0]) if isinstance(p.get("text"), list)
                else str(p.get("text", ""))
                for p in passages
            )
        else:
            text = str(row.get("text") or "")
        if not text:
            continue

        entities = row.get("entities") or row.get("annotations") or []
        labels: list[dict[str, Any]] = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            raw_type = str(ent.get("type") or ent.get("entity_type") or "").strip()
            if not raw_type:
                continue
            mapped = map_entity_type("meddocan", raw_type)
            if mapped in ("_IGNORE", ""):
                continue
            # BigBio offsets: list of [start, end] pairs (multi-span
            # entities exist); use the outer envelope.
            offsets = ent.get("offsets")
            if isinstance(offsets, list) and offsets:
                try:
                    start = int(offsets[0][0])
                    end = int(offsets[-1][1])
                except (IndexError, TypeError, ValueError):
                    continue
            else:
                try:
                    start = int(ent["start"])
                    end = int(ent["end"])
                except (KeyError, TypeError, ValueError):
                    continue
            if start < 0 or end <= start or end > len(text):
                continue
            labels.append({"entity_type": mapped, "start": start, "end": end})

        records.append(TrainingRecord(
            record_id=str(row.get("id") or row.get("document_id") or f"meddocan-{i}"),
            text=text,
            labels=labels,
            language="es",
            source_dataset="meddocan",
        ))
    logger.info("Loaded %d records from MEDDOCAN", len(records))
    return records


# ── Bring-your-own-data: generic JSONL loader ───────────────────────────────

def load_jsonl(
    path: str | Path,
    *,
    taxonomy_name: str = "custom",
    max_records: int | None = None,
    source_label: str | None = None,
) -> list[TrainingRecord]:
    """Load labeled PII records from a JSONL file.

    This is the extension hook for domain-specific data.  Users who have
    their own labeled PII corpus (medical notes, legal filings, internal
    chat logs) can point the training pipeline at it without writing a
    new dedicated loader.

    Expected per-line schema::

        {
          "record_id": "doc-0001",
          "text": "Patient John Smith, DOB 1985-04-12, ...",
          "annotations": [
            {"entity_type": "PERSON_NAME", "start": 8, "end": 18},
            {"entity_type": "DATE_OF_BIRTH", "start": 24, "end": 34}
          ],
          "language": "en"
        }

    Compatibility shims:

    - The field ``annotations`` is preferred; ``labels`` is accepted as
      an alias for compatibility with the v1.0 schema.
    - ``entity_type`` values are passed through :func:`map_entity_type`
      with *taxonomy_name* so callers can register a private vocabulary
      via :func:`register_taxonomy` instead of pre-processing their data.
    - Annotations mapped to ``"_IGNORE"`` in the taxonomy are dropped
      silently (same policy as the built-in loaders).

    Parameters
    ----------
    path:
        JSONL file path.  ``.gz``-compressed files are auto-decoded.
    taxonomy_name:
        Key into :data:`TAXONOMY_MAP` used to canonicalize entity
        labels.  Defaults to ``"custom"`` — with no matching entry, the
        raw labels pass through unchanged.  Register a mapping once via
        :func:`register_taxonomy` before loading to translate a private
        vocabulary to the canonical ``pii-anon`` taxonomy.
    max_records:
        Cap the number of records loaded.  ``None`` = no cap.
    source_label:
        Value stored in :attr:`TrainingRecord.source_dataset`.  Defaults
        to *taxonomy_name* — this is what the downstream manifest and
        swarm metrics key on, so pick something descriptive.

    Returns
    -------
    list[TrainingRecord]
        Ready for :func:`load_training_data` or the swarm train script.
    """
    import gzip
    import json

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"JSONL training file not found: {source_path}")

    source = source_label or taxonomy_name
    opener: Any = gzip.open if source_path.suffix == ".gz" else open

    records: list[TrainingRecord] = []
    with opener(source_path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSONL line %d in %s: %s",
                    i + 1, source_path, exc,
                )
                continue

            text = str(row.get("text") or "")
            if not text:
                logger.warning(
                    "Skipping row %d in %s: empty or missing 'text' field",
                    i + 1, source_path,
                )
                continue

            raw_annotations = row.get("annotations") or row.get("labels") or []
            labels: list[dict[str, Any]] = []
            for ann in raw_annotations:
                raw_type = str(ann.get("entity_type", "")).strip()
                if not raw_type:
                    continue
                mapped = map_entity_type(taxonomy_name, raw_type)
                if mapped in ("_IGNORE", ""):
                    continue
                try:
                    start = int(ann["start"])
                    end = int(ann["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if start < 0 or end <= start or end > len(text):
                    continue
                labels.append({"entity_type": mapped, "start": start, "end": end})

            records.append(TrainingRecord(
                record_id=str(row.get("record_id") or f"{source}-{i}"),
                text=text,
                labels=labels,
                language=str(row.get("language", "en")),
                source_dataset=source,
            ))
    logger.info(
        "Loaded %d records from %s (taxonomy=%s)",
        len(records), source_path, taxonomy_name,
    )
    return records


# ── Unified Loader ──────────────────────────────────────────────────────────

#: Mapping of dataset-name → loader callable.  Extended at runtime via
#: :func:`register_dataset_loader` so users can add their own domain
#: loaders without forking the library.  See :doc:`/extend-swarm`.
DATASET_LOADERS: dict[str, Any] = {
    "pii_anon_eval": load_pii_anon_data,
    "ai4privacy": load_ai4privacy,
    "ai4privacy_400k": load_ai4privacy_400k,
    "conll2003": load_conll2003,
    "tab": load_tab,
    "meddocan": load_meddocan,
}


def register_dataset_loader(
    name: str,
    loader: Any,
    *,
    replace: bool = False,
) -> None:
    """Register a custom dataset loader under *name*.

    The loader must have the signature
    ``loader(max_records: int | None = None) -> list[TrainingRecord]``.
    Once registered it becomes addressable by the swarm training script:

    .. code-block:: bash

        pii-anon-train-swarm --datasets my_domain,pii_anon_eval

    Parameters
    ----------
    name:
        Identifier used on the CLI and in logs.  Must not contain ``/``
        or ``.jsonl`` — those are reserved for the file-path dispatch
        branch in :func:`load_training_data`.
    loader:
        Callable taking ``max_records`` and returning
        ``list[TrainingRecord]``.
    replace:
        If ``False`` (default) and *name* is already registered, raise
        ``ValueError``.  Set to ``True`` to overwrite.
    """
    if "/" in name or name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        raise ValueError(
            f"dataset name {name!r} looks like a file path — register with a "
            "plain name, e.g. 'my_domain'"
        )
    if not callable(loader):
        raise TypeError(f"loader must be callable, got {type(loader).__name__}")
    if name in DATASET_LOADERS and not replace:
        raise ValueError(
            f"dataset {name!r} already registered; pass replace=True to override"
        )
    DATASET_LOADERS[name] = loader


def register_taxonomy(name: str, mapping: dict[str, str]) -> None:
    """Register a taxonomy mapping for a custom data source.

    Domain datasets often use private entity-type vocabularies
    (``email_addr`` → canonical ``EMAIL_ADDRESS``, ``clin_patient_name``
    → ``PERSON_NAME``).  Register the mapping once at process start and
    the ``taxonomy_name`` argument to :func:`load_jsonl` will pick it up.

    Parameters
    ----------
    name:
        Key used by :func:`map_entity_type`.  Match this to the
        ``taxonomy_name`` you pass into :func:`load_jsonl`.
    mapping:
        ``{raw_entity_type: canonical_entity_type_or_"_IGNORE"}``.
        Labels mapped to ``"_IGNORE"`` are dropped at load time.

    Example
    -------
    .. code-block:: python

        from pii_anon.swarm_datasets import register_taxonomy, load_jsonl

        register_taxonomy("clinical_v1", {
            "patient_name": "PERSON_NAME",
            "patient_dob": "DATE_OF_BIRTH",
            "mrn": "MEDICAL_RECORD_NUMBER",
            "note_timestamp": "_IGNORE",
        })
        records = load_jsonl(
            "my_clinical_data.jsonl",
            taxonomy_name="clinical_v1",
        )
    """
    TAXONOMY_MAP[name] = dict(mapping)


def _looks_like_path(name: str) -> bool:
    """Treat anything with a path separator or a ``.jsonl[.gz]`` suffix as a path."""
    return (
        "/" in name
        or name.endswith(".jsonl")
        or name.endswith(".jsonl.gz")
        or name.endswith(".json")
    )


#: Recommended default training-dataset mix — pii-anon's canonical
#: synthetic corpus plus two industry-leading external corpora.  Mirrors
#: the dataset mix used by the pii-rate-elo research paper's evaluation
#: pipeline (see ``../pii-anon-research-paper/pii-rate-elo-pipeline``) so
#: training and evaluation draw from the same distribution.  ``tab`` and
#: ``ai4privacy_400k`` require ``pip install 'pii-anon[swarm-train]'``
#: for the HuggingFace ``datasets`` dependency — they degrade gracefully
#: (log a warning, contribute zero records) when that dep is absent.
DEFAULT_SWARM_DATASETS: tuple[str, ...] = (
    "pii_anon_eval",       # canonical pii-anon-datasets v1.3.0 (~160K, 60 langs)
    "ai4privacy_400k",     # AI4Privacy 2024 (~400K, 17 langs, 54 entity types)
    "tab",                 # Text Anonymization Benchmark (Pilan 2022, peer-reviewed)
)


def stratified_sample(
    records: list[TrainingRecord],
    n: int,
    *,
    strata_keys: tuple[str, ...] = ("language",),
    seed: int = 42,
) -> list[TrainingRecord]:
    """Return ``n`` records stratified by *strata_keys*.

    Each composite stratum (tuple of ``strata_keys`` values) receives a
    share **proportional to its prevalence** in the input pool.  Small
    strata are protected: every represented stratum gets at least one
    record when ``n >= number_of_strata``.  When the input is smaller
    than *n*, the full input is returned unchanged (no oversampling).

    This is the primary tool for keeping capped training runs
    representative — without it, ``SWARM_MAX_RECORDS=10000`` against a
    60-language corpus collapses to English because the source ordering
    is English-first.  With it, every language present in the pool
    appears in the sample with approximately the same share as in the
    source.

    Parameters
    ----------
    records:
        Input pool.
    n:
        Target sample size.  Must be non-negative.  ``0`` returns ``[]``.
    strata_keys:
        Record attributes used to define strata.  Default: ``("language",)``.
        Pass ``("language", "source_dataset")`` for cross-dataset balance,
        or ``("source_dataset",)`` to balance dataset contributions.
    seed:
        Random seed for reproducibility.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative (got {n})")
    if n == 0 or not records:
        return []
    if n >= len(records):
        return list(records)

    # 1. Bucket records by composite stratum key.
    buckets: dict[tuple[Any, ...], list[TrainingRecord]] = defaultdict(list)
    for rec in records:
        key = tuple(getattr(rec, k, None) for k in strata_keys)
        buckets[key].append(rec)

    # 2. Proportional allocation with a floor of 1 per stratum (when
    #    budget allows) to protect rare strata from disappearing.
    total = len(records)
    alloc: dict[tuple[Any, ...], int] = {}
    for key, bucket in buckets.items():
        share = len(bucket) / total
        raw = int(round(n * share))
        # Guarantee at least one record per stratum when we have budget.
        alloc[key] = max(1, raw) if n >= len(buckets) else raw

    # 3. Adjust to hit exactly ``n`` — allocations can over/undershoot
    #    due to rounding and the ``max(1, ...)`` floor.
    diff = n - sum(alloc.values())
    if diff != 0:
        # Sort by (stratum size, key) so the largest strata absorb the
        # over/undershoot first — deterministic ordering thanks to key.
        ordered_keys = sorted(alloc, key=lambda k: (-len(buckets[k]), k))
        idx = 0
        while diff != 0 and ordered_keys:
            key = ordered_keys[idx % len(ordered_keys)]
            if diff > 0 and alloc[key] < len(buckets[key]):
                alloc[key] += 1
                diff -= 1
            elif diff < 0 and alloc[key] > 0:
                alloc[key] -= 1
                diff += 1
            idx += 1
            if idx > 100_000:  # guardrail — should never fire
                break

    # 4. Sample within each stratum with a dedicated RNG.
    import random as _random
    rng = _random.Random(seed)
    sampled: list[TrainingRecord] = []
    for key in sorted(alloc):  # deterministic ordering across runs
        take = min(alloc[key], len(buckets[key]))
        if take <= 0:
            continue
        sampled.extend(rng.sample(buckets[key], take))
    return sampled


def summarize_training_pool(
    records: list[TrainingRecord],
) -> dict[str, Any]:
    """Return a shape + distribution summary of *records*.

    Covers: total count, per-dataset / per-language / per-entity-type
    counts, label density stats.  Used by the training script to print a
    human-readable pool report before kicking off engine passes so users
    can spot-check that the sample is balanced before committing to a
    multi-hour run.
    """
    total = len(records)
    by_dataset: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    by_entity_type: dict[str, int] = defaultdict(int)
    label_counts: list[int] = []
    for rec in records:
        by_dataset[rec.source_dataset or "unknown"] += 1
        by_language[rec.language or "unknown"] += 1
        n_labels = len(rec.labels)
        label_counts.append(n_labels)
        for lbl in rec.labels:
            by_entity_type[lbl.get("entity_type", "unknown")] += 1

    avg_labels = sum(label_counts) / total if total else 0.0
    return {
        "total_records": total,
        "by_dataset": dict(sorted(by_dataset.items(), key=lambda kv: -kv[1])),
        "by_language": dict(sorted(by_language.items(), key=lambda kv: -kv[1])),
        "by_entity_type": dict(sorted(by_entity_type.items(), key=lambda kv: -kv[1])),
        "avg_labels_per_record": round(avg_labels, 2),
        "total_labels": sum(label_counts),
        "records_without_labels": sum(1 for c in label_counts if c == 0),
    }


def load_training_data(
    datasets: list[str] | None = None,
    max_records_per_dataset: int | None = None,
    *,
    custom_taxonomy_name: str = "custom",
    stratify_by: tuple[str, ...] | None = ("language",),
    seed: int = 42,
) -> list[TrainingRecord]:
    """Load and combine training data from multiple sources.

    The *datasets* list accepts a mix of:

    - **Registered names** — keys in :data:`DATASET_LOADERS` (e.g.
      ``"pii_anon_eval"``, ``"ai4privacy"``, ``"conll2003"``, plus any
      added via :func:`register_dataset_loader`).
    - **File paths** — anything containing ``/`` or ending in
      ``.jsonl`` / ``.jsonl.gz`` / ``.json`` is dispatched to
      :func:`load_jsonl` with *custom_taxonomy_name* applied for entity
      mapping.

    Parameters
    ----------
    datasets:
        List of names or paths.  ``None`` = the recommended
        :data:`DEFAULT_SWARM_DATASETS` mix (pii-anon + 2 industry leaders).
    max_records_per_dataset:
        Per-source cap; ``None`` = no cap.  When set, each dataset is
        stratified-sampled down to the cap using *stratify_by* so a
        multi-language corpus doesn't collapse to English-only after
        capping (which is what happens if we just take the first N).
    custom_taxonomy_name:
        Taxonomy key used when dispatching a file-path entry to
        :func:`load_jsonl`.  Default ``"custom"`` — register your own
        via :func:`register_taxonomy` for domain vocabularies.
    stratify_by:
        Record attributes used for per-dataset stratified sampling when
        *max_records_per_dataset* is set.  Default: ``("language",)``.
        Pass ``None`` to disable stratification (take the first N as
        returned by the source).
    seed:
        Seed for stratified sampling.
    """
    if datasets is None:
        datasets = list(DEFAULT_SWARM_DATASETS)

    # When stratified sampling is active, load the full source and
    # stratify down afterwards.  Naïve oversampling (3× max) would
    # silently drop rare languages that appear at the tail of the
    # source ordering (e.g. ai4privacy's 17th language lives past the
    # first 30K rows).  Full-load costs one HF iteration pass per
    # dataset — tens of seconds — which is negligible compared with
    # hours of engine passes downstream.
    if max_records_per_dataset and stratify_by:
        loader_cap = None
    else:
        loader_cap = max_records_per_dataset

    all_records: list[TrainingRecord] = []
    per_dataset_loaded: dict[str, int] = {}
    per_dataset_final: dict[str, int] = {}
    for ds_name in datasets:
        if _looks_like_path(ds_name):
            try:
                records = load_jsonl(
                    ds_name,
                    taxonomy_name=custom_taxonomy_name,
                    max_records=loader_cap,
                )
            except Exception as exc:
                logger.warning("Failed to load JSONL path '%s': %s", ds_name, exc)
                continue
            per_dataset_loaded[ds_name] = len(records)
            if max_records_per_dataset and stratify_by and len(records) > max_records_per_dataset:
                records = stratified_sample(
                    records, max_records_per_dataset,
                    strata_keys=stratify_by, seed=seed,
                )
            per_dataset_final[ds_name] = len(records)
            all_records.extend(records)
            continue

        loader = DATASET_LOADERS.get(ds_name)
        if loader is None:
            logger.warning(
                "Unknown dataset '%s'; skipping. Known datasets: %s",
                ds_name, sorted(DATASET_LOADERS),
            )
            continue
        try:
            records = loader(max_records=loader_cap)
        except Exception as exc:
            logger.warning("Failed to load dataset '%s': %s", ds_name, exc)
            continue
        per_dataset_loaded[ds_name] = len(records)
        if max_records_per_dataset and stratify_by and len(records) > max_records_per_dataset:
            records = stratified_sample(
                records, max_records_per_dataset,
                strata_keys=stratify_by, seed=seed,
            )
        per_dataset_final[ds_name] = len(records)
        all_records.extend(records)

    # Log a per-dataset breakdown so users see when a default dataset
    # silently contributed zero records (usually means the HF `datasets`
    # extra isn't installed).  The summary is advisory — the training
    # run continues with whatever did load.
    dataset_summary = ", ".join(
        f"{name}={per_dataset_final.get(name, 0):,}" for name in datasets
    )
    logger.info(
        "Total training records loaded: %d from %d datasets (%s)",
        len(all_records), len(datasets), dataset_summary,
    )
    empty_datasets = [
        name for name in datasets
        if not _looks_like_path(name) and per_dataset_loaded.get(name, 0) == 0
    ]
    if empty_datasets:
        logger.warning(
            "Dataset(s) %s contributed zero records. "
            "For the HuggingFace-backed datasets (ai4privacy_400k, tab, meddocan) "
            "install with: pip install 'pii-anon[swarm-train]'.",
            empty_datasets,
        )
    return all_records
