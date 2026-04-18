"""Dataset loaders and taxonomy mapping for swarm training.

Supports loading from multiple industry-standard PII/NER datasets and
mapping their entity type labels to the pii-anon canonical taxonomy.
"""

from __future__ import annotations

import logging
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
    "conll2003": load_conll2003,
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


def load_training_data(
    datasets: list[str] | None = None,
    max_records_per_dataset: int | None = None,
    *,
    custom_taxonomy_name: str = "custom",
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
        List of names or paths.  ``None`` = all registered datasets.
    max_records_per_dataset:
        Per-source cap; ``None`` = no cap.
    custom_taxonomy_name:
        Taxonomy key used when dispatching a file-path entry to
        :func:`load_jsonl`.  Default ``"custom"`` — register your own
        via :func:`register_taxonomy` for domain vocabularies.
    """
    if datasets is None:
        datasets = list(DATASET_LOADERS.keys())

    all_records: list[TrainingRecord] = []
    for ds_name in datasets:
        if _looks_like_path(ds_name):
            try:
                records = load_jsonl(
                    ds_name,
                    taxonomy_name=custom_taxonomy_name,
                    max_records=max_records_per_dataset,
                )
                all_records.extend(records)
            except Exception as exc:
                logger.warning("Failed to load JSONL path '%s': %s", ds_name, exc)
            continue

        loader = DATASET_LOADERS.get(ds_name)
        if loader is None:
            logger.warning(
                "Unknown dataset '%s'; skipping. Known datasets: %s",
                ds_name, sorted(DATASET_LOADERS),
            )
            continue
        try:
            records = loader(max_records=max_records_per_dataset)
            all_records.extend(records)
        except Exception as exc:
            logger.warning("Failed to load dataset '%s': %s", ds_name, exc)

    logger.info("Total training records loaded: %d from %d datasets", len(all_records), len(datasets))
    return all_records
