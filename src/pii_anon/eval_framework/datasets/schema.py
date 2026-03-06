"""Extended benchmark record schema for the PII Anonymization Evaluation Framework v1.0.0.

Backward-compatible with the existing ``BenchmarkRecord`` (all new fields
have defaults) while adding seven evaluation dimensions, quasi-identifier
tracking, adversarial metadata, and statistical stratification fields.

Evaluation Dimensions (with weighting):
    1. Entity Tracking         (20%) — consistent coreference across context
    2. Multilingual & Dialect  (15%) — 52 languages, 17 scripts, locale variants
    3. Context Preservation    (20%) — semantic integrity in dialogues/narratives
    4. Diverse PII Types       (20%) — breadth across 48 entity types / 7 categories
    5. Edge Cases              (10%) — abbreviations, partial PII, ambiguity
    6. Data Format Variations  (10%) — structured/semi-structured/unstructured
    7. Temporal Consistency     (5%) — time-series entity evolution

Evidence basis:
    - Sweeney (2002): k-anonymity and re-identification via quasi-identifiers
    - Gebru et al. (2021): Datasheets for Datasets
    - Cochran (1977): Sampling Techniques (statistical power)
    - TAB (2022), PII-Bench (2025), RAT-Bench (2025): PII benchmark standards
"""

from __future__ import annotations

import gzip
import os
import json
from collections import defaultdict
from importlib import resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Valid dimension tags (the seven core evaluation dimensions)
# ---------------------------------------------------------------------------

VALID_DIMENSION_TAGS = frozenset({
    "entity_tracking",
    "multilingual",
    "context_preservation",
    "diverse_pii_types",
    "edge_cases",
    "data_format_variations",
    "temporal_consistency",
})

# Dimension weights for composite scoring
DIMENSION_WEIGHTS: dict[str, float] = {
    "entity_tracking": 0.20,
    "multilingual": 0.15,
    "context_preservation": 0.20,
    "diverse_pii_types": 0.20,
    "edge_cases": 0.10,
    "data_format_variations": 0.10,
    "temporal_consistency": 0.05,
}


@dataclass
class EvalBenchmarkRecord:
    """A single evaluation benchmark record for PII Anonymization Eval v1.0.0.

    All fields present in the original ``BenchmarkRecord`` are preserved
    (with identical names and semantics).  New fields carry defaults so
    that existing JSONL files load seamlessly.
    """

    # ── from BenchmarkRecord (legacy-compatible) ──────────────────────
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

    # ── data-type & context metadata ──────────────────────────────────
    data_type: Literal[
        "unstructured_text", "structured", "semi_structured", "code", "logs", "mixed",
    ] = "unstructured_text"
    context_length_tier: Literal["short", "medium", "long", "very_long"] = "medium"
    token_count: int = 0
    regulatory_domain: list[str] = field(default_factory=list)
    adversarial_type: str | None = None
    script: str = "Latin"
    entity_types_present: list[str] = field(default_factory=list)

    # ── v1.0.0: Dimension tracking ───────────────────────────────────
    dimension_tags: list[str] = field(default_factory=list)

    # ── v1.0.0 Dim 1: Entity Tracking (20%) ─────────────────────────
    num_repeated_entities: int = 0
    coreference_chains: list[str] = field(default_factory=list)
    entity_tracking_difficulty: str = "none"

    # ── v1.0.0 Dim 2: Multilingual & Dialect (15%) ──────────────────
    language_family: str = ""
    resource_level: str = "high"
    dialect_variant: str | None = None

    # ── v1.0.0 Dim 3: Context Preservation (20%) ────────────────────
    has_conversational_context: bool = False
    context_type: str = "standalone"
    turn_count: int = 1
    preservation_challenge: str = "none"

    # ── v1.0.0 Dim 5: Edge Cases (10%) ──────────────────────────────
    edge_case_types: list[str] = field(default_factory=list)

    # ── v1.0.0 Dim 6: Data Format Variations (10%) ──────────────────
    format_subtype: str | None = None
    format_complexity: str = "simple"

    # ── v1.0.0 Dim 7: Temporal Consistency (5%) ─────────────────────
    is_time_series: bool = False
    time_series_id: str | None = None
    temporal_ordering: int = 0
    temporal_consistency_type: str = "none"

    # ── v1.0.0: Re-identification risk (Sweeney 2002) ───────────────
    quasi_identifiers_present: list[str] = field(default_factory=list)
    reidentification_risk_tier: str = "low"

    # ── v1.0.0: Adversarial robustness ──────────────────────────────
    adversarial_attack_type: str | None = None
    adversarial_difficulty: str = "clean"

    # ── v1.0.0: Stratification ──────────────────────────────────────
    stratum_id: str = ""


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_DATASET = "pii_anon_eval_v1"
_FALLBACK_DATASET = "eval_framework_v1"
DatasetSource = Literal["auto", "package-only"]


def resolve_eval_dataset_path(
    name: str = _DEFAULT_DATASET,
    *,
    source: DatasetSource = "auto",
) -> Path | None:
    if source not in {"auto", "package-only"}:
        raise ValueError("source must be one of: auto, package-only")

    base_dir = Path("eval_framework") / "data"
    # Prefer compressed, fall back to plain JSONL.
    suffixes = [f"{name}.jsonl.gz", f"{name}.jsonl"]
    candidates: list[Path] = []

    for fname in suffixes:
        rel_path = base_dir / fname

        # Installed dataset package path.
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
            code_root = Path(__file__).resolve().parents[4]
            candidates.append(code_root.parent / "pii-anon-eval-data" / "src" / "pii_anon_datasets" / rel_path)

        # Local monorepo datasets package path.
            candidates.append(code_root / "packages" / "pii_anon_datasets" / "src" / "pii_anon_datasets" / rel_path)

        # Legacy in-core path for backward compatibility.
            candidates.append(_DATA_DIR / fname)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: try legacy dataset name for backward compatibility.
    if name == _DEFAULT_DATASET:
        return resolve_eval_dataset_path(_FALLBACK_DATASET, source=source)

    return None


def _classify_context_length(text: str) -> Literal["short", "medium", "long", "very_long"]:
    approx_tokens = len(text.split())
    if approx_tokens < 100:
        return "short"
    if approx_tokens < 1000:
        return "medium"
    if approx_tokens < 10000:
        return "long"
    return "very_long"


def _infer_dimension_tags(row: dict[str, Any], labels: list[dict[str, Any]]) -> list[str]:
    """Infer dimension tags from record content for legacy records."""
    tags: list[str] = list(row.get("dimension_tags", []))
    if tags:
        return tags

    scenario = str(row.get("scenario_id", ""))
    context_group = str(row.get("context_group", ""))
    data_type = str(row.get("data_type", "unstructured_text"))

    # Entity tracking: records with coreference/continuity scenarios
    if "continuity" in scenario or "tracking" in context_group:
        tags.append("entity_tracking")

    # Multilingual: non-English records
    lang = str(row.get("language", "en"))
    if lang != "en":
        tags.append("multilingual")

    # Context preservation: records with context_loss or conversational scenarios
    if "context_loss" in scenario or row.get("has_conversational_context"):
        tags.append("context_preservation")

    # Diverse PII types: records with 5+ entity types
    entity_types = {lbl.get("entity_type") for lbl in labels if lbl.get("entity_type")}
    if len(entity_types) >= 5:
        tags.append("diverse_pii_types")

    # Data format variations: non-plain-text data
    if data_type in {"structured", "semi_structured", "code", "logs", "mixed"}:
        tags.append("data_format_variations")

    # Edge cases: challenging difficulty
    if str(row.get("difficulty_level", "")) == "challenging":
        tags.append("edge_cases")

    # Temporal consistency: time-series records
    if row.get("is_time_series"):
        tags.append("temporal_consistency")

    # Default: diverse PII types for baseline records
    if not tags:
        tags.append("diverse_pii_types")

    return sorted(set(tags))


def _count_repeated_entities(labels: list[dict[str, Any]]) -> int:
    """Count how many entities appear more than once (by cluster or text span)."""
    clusters: dict[str, int] = defaultdict(int)
    for lbl in labels:
        cluster_id = lbl.get("entity_cluster_id", "none")
        if cluster_id and cluster_id != "none":
            clusters[cluster_id] += 1
    return sum(1 for count in clusters.values() if count > 1)


def _normalize_eval_row(row: dict[str, Any], index: int) -> EvalBenchmarkRecord:
    record_id = str(row.get("id", row.get("record_id", f"eval-{index + 1:06d}")))
    text = str(row.get("text", ""))
    if not text:
        raise ValueError(f"eval dataset row `{record_id}` has empty text")

    labels_raw = list(row.get("labels", []))
    labels: list[dict[str, Any]] = []
    for lbl in labels_raw:
        entity_type = str(lbl.get("entity_type", "")).strip()
        start = int(lbl.get("start", -1))
        end = int(lbl.get("end", -1))
        if not entity_type or start < 0 or end <= start or end > len(text):
            continue
        entry: dict[str, Any] = {"entity_type": entity_type, "start": start, "end": end}
        for opt in ("entity_cluster_id", "mention_variant", "context_group"):
            if opt in lbl and lbl[opt] is not None:
                entry[opt] = str(lbl[opt])
        labels.append(entry)

    entity_types_present = sorted({lbl["entity_type"] for lbl in labels})
    language = str(row.get("language", "en"))

    data_type = str(row.get("data_type", "unstructured_text"))
    if data_type not in {"unstructured_text", "structured", "semi_structured", "code", "logs", "mixed"}:
        data_type = "unstructured_text"

    context_length_tier = str(row.get("context_length_tier", ""))
    if context_length_tier not in {"short", "medium", "long", "very_long"}:
        context_length_tier = _classify_context_length(text)

    regulatory_raw = row.get("regulatory_domain", [])
    if isinstance(regulatory_raw, str):
        regulatory_domain = [regulatory_raw] if regulatory_raw else []
    elif isinstance(regulatory_raw, list):
        regulatory_domain = [str(r) for r in regulatory_raw]
    else:
        regulatory_domain = []

    # v1.0.0 fields: infer from content if not explicitly set
    dimension_tags = _infer_dimension_tags(row, labels)
    num_repeated = row.get("num_repeated_entities", _count_repeated_entities(labels))

    coreference_raw = row.get("coreference_chains", [])
    coreference_chains = list(coreference_raw) if isinstance(coreference_raw, list) else []

    entity_tracking_difficulty = str(row.get("entity_tracking_difficulty", "none"))
    if entity_tracking_difficulty not in {"none", "simple", "moderate", "complex"}:
        entity_tracking_difficulty = "none"

    # Quasi-identifier inference
    quasi_ids_raw = row.get("quasi_identifiers_present", [])
    quasi_identifiers = list(quasi_ids_raw) if isinstance(quasi_ids_raw, list) else []

    reid_tier = str(row.get("reidentification_risk_tier", "low"))
    if reid_tier not in {"low", "moderate", "high", "critical"}:
        reid_tier = "low"

    # Edge case types
    edge_raw = row.get("edge_case_types", [])
    edge_case_types = list(edge_raw) if isinstance(edge_raw, list) else []

    return EvalBenchmarkRecord(
        record_id=record_id,
        text=text,
        labels=labels,
        language=language,
        source_type="synthetic" if row.get("source_type", "synthetic") == "synthetic" else "curated_public",
        source_id=str(row.get("source_id", "generated")),
        license=str(row.get("license", "CC0-1.0")),
        scenario_id=str(row.get("scenario_id", "baseline")),
        entity_cluster_id=str(row.get("entity_cluster_id", "none")),
        mention_variant=str(row.get("mention_variant", "none")),
        context_group=str(row.get("context_group", "baseline")),
        datatype_group=str(row.get("datatype_group", "general")),
        difficulty_level=str(row.get("difficulty_level", "moderate")),
        data_type=data_type,  # type: ignore[arg-type]
        context_length_tier=context_length_tier,  # type: ignore[arg-type]
        token_count=int(row.get("token_count", len(text.split()))),
        regulatory_domain=regulatory_domain,
        adversarial_type=row.get("adversarial_type"),
        script=str(row.get("script", "Latin")),
        entity_types_present=entity_types_present,
        # v1.0.0 fields
        dimension_tags=dimension_tags,
        num_repeated_entities=num_repeated,
        coreference_chains=coreference_chains,
        entity_tracking_difficulty=entity_tracking_difficulty,
        language_family=str(row.get("language_family", "")),
        resource_level=str(row.get("resource_level", "high")),
        dialect_variant=row.get("dialect_variant"),
        has_conversational_context=bool(row.get("has_conversational_context", False)),
        context_type=str(row.get("context_type", "standalone")),
        turn_count=int(row.get("turn_count", 1)),
        preservation_challenge=str(row.get("preservation_challenge", "none")),
        edge_case_types=edge_case_types,
        format_subtype=row.get("format_subtype"),
        format_complexity=str(row.get("format_complexity", "simple")),
        is_time_series=bool(row.get("is_time_series", False)),
        time_series_id=row.get("time_series_id"),
        temporal_ordering=int(row.get("temporal_ordering", 0)),
        temporal_consistency_type=str(row.get("temporal_consistency_type", "none")),
        quasi_identifiers_present=quasi_identifiers,
        reidentification_risk_tier=reid_tier,
        adversarial_attack_type=row.get("adversarial_attack_type"),
        adversarial_difficulty=str(row.get("adversarial_difficulty", "clean")),
        stratum_id=str(row.get("stratum_id", "")),
    )


def load_eval_dataset(
    name: str = _DEFAULT_DATASET,
    *,
    language: str | None = None,
    data_type: str | None = None,
    difficulty: str | None = None,
    adversarial_only: bool = False,
    dimension: str | None = None,
    source: DatasetSource = "auto",
) -> list[EvalBenchmarkRecord]:
    """Load an evaluation benchmark dataset from JSONL.

    Parameters
    ----------
    name:
        Dataset filename (without ``.jsonl``).  Defaults to ``pii_anon_eval_v1``
        (the v1.0.0 unified dataset), falling back to ``eval_framework_v1``
        for backward compatibility.
    language:
        Filter by ISO 639-1 code.
    data_type:
        Filter by data type.
    difficulty:
        Filter by difficulty level.
    adversarial_only:
        If ``True``, return only adversarial samples.
    dimension:
        Filter by evaluation dimension tag (e.g. ``"entity_tracking"``).
    """
    path = resolve_eval_dataset_path(name, source=source)
    if path is None:
        source_msg = (
            "from installed `pii-anon-datasets` package resources"
            if source == "package-only"
            else "from configured dataset paths"
        )
        raise FileNotFoundError(
            f"Evaluation dataset `{name}` not found {source_msg}. Install datasets with "
            "`pip install pii-anon-datasets` or set `PII_ANON_DATASET_ROOT`."
        )

    records: list[EvalBenchmarkRecord] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            record = _normalize_eval_row(row, idx)
            if language and record.language != language:
                continue
            if data_type and record.data_type != data_type:
                continue
            if difficulty and record.difficulty_level != difficulty:
                continue
            if adversarial_only and not record.adversarial_type:
                continue
            if dimension and dimension not in record.dimension_tags:
                continue
            records.append(record)
    return records


def summarize_eval_dataset(name: str = _DEFAULT_DATASET) -> dict[str, Any]:
    """Return distribution summary of the evaluation dataset."""
    records = load_eval_dataset(name)
    summary: dict[str, Any] = {
        "dataset": name,
        "version": "1.0.0",
        "total_records": len(records),
        "by_language": {},
        "by_data_type": {},
        "by_difficulty": {},
        "by_context_length": {},
        "by_script": {},
        "by_adversarial": {"adversarial": 0, "normal": 0},
        "by_dimension": {},
        "by_context_type": {},
        "by_reidentification_risk": {},
        "entity_types": set(),
        "regulatory_domains": set(),
    }
    for r in records:
        summary["by_language"][r.language] = summary["by_language"].get(r.language, 0) + 1
        summary["by_data_type"][r.data_type] = summary["by_data_type"].get(r.data_type, 0) + 1
        summary["by_difficulty"][r.difficulty_level] = summary["by_difficulty"].get(r.difficulty_level, 0) + 1
        summary["by_context_length"][r.context_length_tier] = summary["by_context_length"].get(r.context_length_tier, 0) + 1
        summary["by_script"][r.script] = summary["by_script"].get(r.script, 0) + 1
        if r.adversarial_type:
            summary["by_adversarial"]["adversarial"] += 1
        else:
            summary["by_adversarial"]["normal"] += 1
        summary["entity_types"].update(r.entity_types_present)
        summary["regulatory_domains"].update(r.regulatory_domain)

        # v1.0.0 summaries
        for dim in r.dimension_tags:
            summary["by_dimension"][dim] = summary["by_dimension"].get(dim, 0) + 1
        summary["by_context_type"][r.context_type] = summary["by_context_type"].get(r.context_type, 0) + 1
        summary["by_reidentification_risk"][r.reidentification_risk_tier] = (
            summary["by_reidentification_risk"].get(r.reidentification_risk_tier, 0) + 1
        )

    summary["entity_types"] = sorted(summary["entity_types"])
    summary["regulatory_domains"] = sorted(summary["regulatory_domains"])
    return summary
