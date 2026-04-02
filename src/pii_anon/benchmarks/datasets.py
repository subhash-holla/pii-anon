from __future__ import annotations

import gzip
import os
import json
from importlib import resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast


@dataclass
class BenchmarkRecord:
    record_id: str
    text: str
    labels: list[dict[str, Any]]
    language: str = "en"
    source_type: Literal["synthetic", "curated_public"] = "synthetic"
    source_id: str = "generated"
    license: str = "CC0-1.0"
    scenario_id: str = "baseline"
    entity_cluster_id: str = "none"
    mention_variant: str = "none"
    context_group: str = "baseline"
    datatype_group: str = "general"
    difficulty_level: str = "moderate"
    evaluation_dimension: str = "pii_type_coverage"


@dataclass
class UseCaseProfile:
    profile: str
    objective: Literal["accuracy", "balanced", "speed"]
    required: bool = True
    languages: list[str] = field(default_factory=lambda: ["en", "es", "fr"])
    max_samples: int | None = None


_DEFAULT_DATASET: list[BenchmarkRecord] = [
    BenchmarkRecord(
        record_id="default-0001",
        text="Contact alice@example.com and +1 415 555 0100.",
        labels=[
            {"entity_type": "EMAIL_ADDRESS", "start": 8, "end": 25},
            {"entity_type": "PHONE_NUMBER", "start": 30, "end": 45},
        ],
    ),
    BenchmarkRecord(
        record_id="default-0002",
        text="Patient SSN is 123-45-6789 and doctor is Dr Smith.",
        labels=[
            {"entity_type": "US_SSN", "start": 15, "end": 26},
            {"entity_type": "PERSON_NAME", "start": 41, "end": 49},
        ],
    ),
]


DatasetSource = Literal["auto", "package-only"]


def _dataset_file(name: str, *, source: DatasetSource = "auto") -> Path:
    path = resolve_benchmark_dataset_path(name, source=source)
    if path is None:
        source_msg = (
            "from installed `pii-anon-datasets` package resources"
            if source == "package-only"
            else "from configured dataset paths"
        )
        raise FileNotFoundError(
            f"Benchmark dataset `{name}` not found {source_msg}. Install datasets with "
            "`pip install pii-anon-datasets` or set `PII_ANON_DATASET_ROOT`."
        )
    return path


def resolve_benchmark_dataset_path(
    name: str,
    *,
    source: DatasetSource = "auto",
) -> Path | None:
    if source not in {"auto", "package-only"}:
        raise ValueError("source must be one of: auto, package-only")

    # Build candidate paths for both v1 (benchmarks/data/) and v1.1 (data/) layouts.
    # v1 layout:  pii_anon_datasets/benchmarks/data/{name}.jsonl.gz
    # v1.1 layout: pii_anon_datasets/data/pii_anon.jsonl.gz  (canonical name)
    base_dirs = [Path("benchmarks") / "data", Path("data")]

    # Dataset name mapping: pii_anon_benchmark -> pii_anon (v1.1 canonical name)
    _V11_NAME_MAP = {"pii_anon_benchmark": "pii_anon"}
    name_variants = [name]
    if name in _V11_NAME_MAP:
        name_variants.append(_V11_NAME_MAP[name])

    candidates: list[Path] = []

    for dataset_name in name_variants:
        suffixes = [f"{dataset_name}.jsonl.gz", f"{dataset_name}.jsonl"]
        for base_dir in base_dirs:
            for fname in suffixes:
                rel_path = base_dir / fname

                # Installed dataset package path (preferred for wheel installs).
                try:
                    pkg_root = resources.files("pii_anon_datasets")
                    candidates.append(Path(str(pkg_root.joinpath(*rel_path.parts))))
                except Exception:
                    pass

                if source == "auto":
                    env_root = os.getenv("PII_ANON_DATASET_ROOT") or os.getenv("PII_VEIL_DATASET_ROOT")
                    if env_root:
                        candidates.insert(0, Path(env_root) / rel_path)

                    # Sibling repo path (pii-anon-eval-data next to pii-anon-code).
                    code_root = Path(__file__).resolve().parents[3]
                    candidates.append(code_root.parent / "pii-anon-eval-data" / "src" / "pii_anon_datasets" / rel_path)

                    # Local monorepo datasets package path (works in dev without install).
                    candidates.append(code_root / "packages" / "pii_anon_datasets" / "src" / "pii_anon_datasets" / rel_path)

                    # Legacy in-core path for backward compatibility.
                    candidates.append(Path(__file__).resolve().parent / "data" / fname)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _matrix_file(name: str) -> Path:
    return Path(__file__).resolve().parent / "matrix" / name


def _validate_label(record_id: str, text: str, label: dict[str, Any]) -> dict[str, Any]:
    entity_type = str(label.get("entity_type", "")).strip()
    if not entity_type:
        raise ValueError(f"dataset row `{record_id}` has label without entity_type")

    start = int(label.get("start", -1))
    end = int(label.get("end", -1))
    if start < 0 or end < 0 or start >= end or end > len(text):
        raise ValueError(f"dataset row `{record_id}` has invalid label span [{start}, {end})")

    out = {"entity_type": entity_type, "start": start, "end": end}
    for optional_key in ("entity_cluster_id", "mention_variant", "context_group"):
        if optional_key in label and label[optional_key] is not None:
            out[optional_key] = str(label[optional_key])
    return out


def _is_annotations_schema(row: dict[str, Any]) -> bool:
    """Detect pii-anon-eval-data schema (uses 'annotations' instead of 'labels')."""
    return "annotations" in row and "labels" not in row


# Entity type mapping from pii-anon-eval-data names to the canonical benchmark names
# used by the pii-anon detection engine and competitor_compare evaluation.
_EVAL_DATA_ENTITY_TYPE_MAP: dict[str, str] = {
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
    "LATITUDE_LONGITUDE": "_BENCHMARK_IGNORE",
    "TIMESTAMP": "_BENCHMARK_IGNORE",
    "POSTAL_CODE": "ADDRESS",
    "SWIFT_BIC_CODE": "_BENCHMARK_IGNORE",
    "HEALTH_INSURANCE_ID": "MEDICAL_RECORD_NUMBER",
    "DEVICE_IDENTIFIER": "MAC_ADDRESS",
    "TAX_ID": "NATIONAL_ID",
    "VISA_NUMBER": "PASSPORT",
}


def _normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    """Convert a pii-anon-eval-data annotation to the label format expected by BenchmarkRecord."""
    entity_type = annotation.get("entity_type", "")
    entity_type = _EVAL_DATA_ENTITY_TYPE_MAP.get(entity_type, entity_type)
    label: dict[str, Any] = {
        "entity_type": entity_type,
        "start": annotation["start"],
        "end": annotation["end"],
    }
    if annotation.get("cluster_id") is not None:
        label["entity_cluster_id"] = str(annotation["cluster_id"])
    if annotation.get("mention_variant") is not None:
        label["mention_variant"] = str(annotation["mention_variant"])
    return label


def _normalize_row(row: dict[str, Any], index: int) -> BenchmarkRecord:
    # Support both v1 ("id") and v1.1 ("record_id") field names.
    record_id = str(row.get("record_id", row.get("id", f"row-{index + 1:06d}")))
    text = str(row.get("text", ""))
    if not text:
        raise ValueError(f"dataset row `{record_id}` has empty text")

    # Handle pii-anon-eval-data schema: 'annotations' with richer structure
    if _is_annotations_schema(row):
        annotations_raw = list(row.get("annotations", []))
        labels = [
            _validate_label(record_id, text, _normalize_annotation(ann))
            for ann in annotations_raw
        ]
    else:
        labels_raw = list(row.get("labels", []))
        labels = [_validate_label(record_id, text, item) for item in labels_raw]

    # Resolve source_type from v1 field or v1.1 provenance.source_type
    provenance = row.get("provenance", {})
    source_type_raw = str(
        row.get("source_type")
        or (provenance.get("source_type") if isinstance(provenance, dict) else None)
        or "synthetic"
    )
    if source_type_raw not in {"synthetic", "curated_public"}:
        source_type_raw = "synthetic"
    source_type: Literal["synthetic", "curated_public"] = (
        "synthetic" if source_type_raw == "synthetic" else "curated_public"
    )

    # Resolve license from v1 field or v1.1 provenance.license
    license_val = str(
        row.get("license")
        or (provenance.get("license") if isinstance(provenance, dict) else None)
        or "CC0-1.0"
    )

    return BenchmarkRecord(
        record_id=record_id,
        text=text,
        labels=labels,
        language=str(row.get("language", "en")),
        source_type=source_type,
        source_id=str(row.get("source_id", "generated")),
        license=license_val,
        scenario_id=str(row.get("scenario_id", "baseline")),
        entity_cluster_id=str(row.get("entity_cluster_id", "none")),
        mention_variant=str(row.get("mention_variant", "none")),
        context_group=str(row.get("context_group", "baseline")),
        datatype_group=str(row.get("datatype_group", row.get("domain", "general"))),
        difficulty_level=str(row.get("difficulty_level", "moderate")),
        evaluation_dimension=str(row.get("evaluation_dimension", row.get("primary_dimension", "pii_type_coverage"))),
    )


def load_benchmark_dataset(
    name: str = "pii_anon_benchmark",
    *,
    split: Literal["synthetic", "curated_public"] | None = None,
    source: DatasetSource = "auto",
) -> list[BenchmarkRecord]:
    try:
        path = _dataset_file(name, source=source)
    except FileNotFoundError:
        if name == "pii_anon_benchmark":
            raise
        defaults = list(_DEFAULT_DATASET)
        if split is None:
            return defaults
        return [row for row in defaults if row.source_type == split]

    records: list[BenchmarkRecord] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            record = _normalize_row(row, idx)
            if split is None or record.source_type == split:
                records.append(record)
    return records


def summarize_dataset(name: str = "pii_anon_benchmark") -> dict[str, Any]:
    rows = load_benchmark_dataset(name)
    by_language: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    by_datatype: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for item in rows:
        by_language[item.language] = by_language.get(item.language, 0) + 1
        by_source[item.source_type] = by_source.get(item.source_type, 0) + 1
        by_scenario[item.scenario_id] = by_scenario.get(item.scenario_id, 0) + 1
        by_datatype[item.datatype_group] = by_datatype.get(item.datatype_group, 0) + 1
        by_difficulty[item.difficulty_level] = by_difficulty.get(item.difficulty_level, 0) + 1

    return {
        "dataset": name,
        "records": len(rows),
        "language_distribution": by_language,
        "source_distribution": by_source,
        "scenario_distribution": by_scenario,
        "datatype_group_distribution": by_datatype,
        "difficulty_level_distribution": by_difficulty,
    }


def load_use_case_matrix(path: str | None = None) -> list[UseCaseProfile]:
    matrix_path = _matrix_file("use_case_matrix.json") if path is None else Path(path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    profiles_raw = payload.get("profiles", [])
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ValueError(f"invalid use-case matrix in {matrix_path}")

    profiles: list[UseCaseProfile] = []
    for item in profiles_raw:
        if not isinstance(item, dict):
            raise ValueError(f"invalid matrix profile entry: {item!r}")
        objective = str(item.get("objective", "balanced"))
        if objective not in {"accuracy", "balanced", "speed"}:
            raise ValueError(f"invalid objective `{objective}` in {matrix_path}")
        languages_raw = item.get("languages", ["en", "es", "fr"])
        if not isinstance(languages_raw, list) or not languages_raw:
            raise ValueError(f"invalid languages in matrix profile `{item}`")
        profile = UseCaseProfile(
            profile=str(item.get("profile", "unknown")),
            objective=cast(Literal["accuracy", "balanced", "speed"], objective),
            required=bool(item.get("required", True)),
            languages=[str(lang) for lang in languages_raw],
            max_samples=int(item["max_samples"]) if item.get("max_samples") is not None else None,
        )
        profiles.append(profile)
    return profiles
