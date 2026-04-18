"""Extended benchmark record schema for the PII Anonymization Evaluation Framework.

Backward-compatible with the existing ``BenchmarkRecord`` (all new fields
have defaults) while adding seven evaluation dimensions, quasi-identifier
tracking, adversarial metadata, statistical stratification fields, and
(as of dataset v1.3.0) Tier 3 re-identification-resistance signals.

Evaluation Dimensions (with weighting):
    1. Entity Tracking         (20%) — consistent coreference across context
    2. Multilingual & Dialect  (15%) — 60 languages, 32 scripts, locale variants
    3. Context Preservation    (20%) — semantic integrity in dialogues/narratives
    4. Diverse PII Types       (20%) — breadth across 63 entity types / 9 categories
    5. Edge Cases              (10%) — abbreviations, partial PII, ambiguity
    6. Data Format Variations  (10%) — structured/semi-structured/unstructured
    7. Temporal Consistency     (5%) — time-series entity evolution

Tier 3 (dataset v1.3.0) adds behavioral-signal annotations, paired-profile
records, ESRC-attack evaluation targets, a 4th ``anonymized_llm_sanitized``
text variant, and per-record Re-identification Resistance Score (RRS).
These fields feed directly into :func:`pii_anon.eval_framework.compute_composite`
via its Tier 3 kwargs.

Evidence basis:
    - Sweeney (2002): k-anonymity and re-identification via quasi-identifiers
    - Gebru et al. (2021): Datasheets for Datasets
    - Cochran (1977): Sampling Techniques (statistical power)
    - TAB (2022), PII-Bench (2025), RAT-Bench (2025): PII benchmark standards
    - Lermen et al. (2026): Large-scale online deanonymization with LLMs
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
    """A single evaluation benchmark record for PII Anonymization Eval.

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

    # ──: Dimension tracking ───────────────────────────────────
    dimension_tags: list[str] = field(default_factory=list)

    # ── Dim 1: Entity Tracking (20%) ─────────────────────────
    num_repeated_entities: int = 0
    coreference_chains: list[str] = field(default_factory=list)
    entity_tracking_difficulty: str = "none"

    # ── Dim 2: Multilingual & Dialect (15%) ──────────────────
    language_family: str = ""
    resource_level: str = "high"
    dialect_variant: str | None = None

    # ── Dim 3: Context Preservation (20%) ────────────────────
    has_conversational_context: bool = False
    context_type: str = "standalone"
    turn_count: int = 1
    preservation_challenge: str = "none"

    # ── Dim 5: Edge Cases (10%) ──────────────────────────────
    edge_case_types: list[str] = field(default_factory=list)

    # ── Dim 6: Data Format Variations (10%) ──────────────────
    format_subtype: str | None = None
    format_complexity: str = "simple"

    # ── Dim 7: Temporal Consistency (5%) ─────────────────────
    is_time_series: bool = False
    time_series_id: str | None = None
    temporal_ordering: int = 0
    temporal_consistency_type: str = "none"

    # ──: Re-identification risk (Sweeney 2002) ───────────────
    quasi_identifiers_present: list[str] = field(default_factory=list)
    reidentification_risk_tier: str = "low"

    # ──: Adversarial robustness ──────────────────────────────
    adversarial_attack_type: str | None = None
    adversarial_difficulty: str = "clean"

    # ──: Stratification ──────────────────────────────────────
    stratum_id: str = ""

    # ── Tier 3 (dataset v1.3.0+) — behavioral signals ────────
    # Aggregate density of identity-revealing quasi-identifiers
    # that survive entity-level de-identification (writing style,
    # professional domain, interest topics, temporal patterns,
    # location signals, personal anecdotes).  [0.0, 1.0].
    behavioral_signal_density: float = 0.0
    # Categorical risk level: low | moderate | high | critical.
    reidentification_contribution: str = "low"
    # Raw 6-category signal block for specialized Tier 3 consumers.
    behavioral_signals: dict[str, Any] = field(default_factory=dict)

    # ── Tier 3 (dataset v1.3.0+) — re-identification resistance
    # RRS ∈ [0, 1]; higher = more resistant to ESRC-style attacks.
    re_identification_resistance_score: float | None = None
    # Estimated ESRC attack recall against this record ∈ [0, 1].
    estimated_reid_recall: float | None = None
    # Categorical Tier 3 risk: low | moderate | high | critical.
    tier3_risk_level: str = "low"

    # ── Tier 3 (dataset v1.3.0+) — paired-profile / ESRC eval ─
    is_paired_profile: bool = False
    persona_id: str | None = None
    linked_profile_id: str | None = None
    profile_type: str | None = None
    esrc_attack_target: bool = False
    expected_reidentification_difficulty: str = "unknown"
    behavioral_signal_removal_attempted: bool = False

    # ── Tier 3 (dataset v1.3.0+) — anonymized variants ───────
    # Pre-computed anonymized text variants (masked / pseudonymized /
    # generalized / llm_sanitized) and their per-variant utility metrics.
    # Kept as a raw dict because consumers pick which variant to evaluate.
    context_preservation: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"
# v1.1+ canonical name is ``pii_anon``; older v1.0 packages shipped
# ``pii_anon_eval_v1``; very old packages shipped ``pii_anon_eval``.
_DEFAULT_DATASET = "pii_anon"
_LEGACY_DATASET_NAMES: tuple[str, ...] = ("pii_anon_eval_v1", "pii_anon_eval")
DatasetSource = Literal["auto", "package-only"]

# Entity-type aliases used by ``pii-anon-datasets`` ≥ v1.1.
# Mirrors :mod:`pii_anon.benchmarks.datasets._EVAL_DATA_ENTITY_TYPE_MAP`
# so both benchmark and evaluation paths see identical entity labels.
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


def resolve_eval_dataset_path(
    name: str = _DEFAULT_DATASET,
    *,
    source: DatasetSource = "auto",
) -> Path | None:
    if source not in {"auto", "package-only"}:
        raise ValueError("source must be one of: auto, package-only")

    # Dataset layout evolution:
    #   v1.0 — ``eval_framework/data/pii_anon_eval_v1.jsonl.gz``
    #   v1.1+ — ``data/pii_anon.jsonl.gz`` (canonical, no ``eval_framework``
    #           subdirectory)
    # Probe both, preferring the v1.1+ canonical location.
    base_dirs = [Path("data"), Path("eval_framework") / "data"]
    suffixes = [f"{name}.jsonl.gz", f"{name}.jsonl"]
    candidates: list[Path] = []

    env_root = (
        os.getenv("PII_ANON_DATASET_ROOT") or os.getenv("PII_VEIL_DATASET_ROOT")
        if source == "auto"
        else None
    )
    code_root = Path(__file__).resolve().parents[4]

    for base_dir in base_dirs:
        for fname in suffixes:
            rel_path = base_dir / fname

            if env_root:
                candidates.append(Path(env_root) / rel_path)

            try:
                pkg_root = resources.files("pii_anon_datasets")
                candidates.append(Path(str(pkg_root.joinpath(*rel_path.parts))))
            except Exception:
                pass

            if source == "auto":
                # Sibling repo path (pii-anon-eval-data next to pii-anon-code).
                candidates.append(
                    code_root.parent / "pii-anon-eval-data" / "src"
                    / "pii_anon_datasets" / rel_path
                )
                # Local monorepo datasets package path.
                candidates.append(
                    code_root / "packages" / "pii_anon_datasets" / "src"
                    / "pii_anon_datasets" / rel_path
                )
                # Legacy in-core path.
                candidates.append(_DATA_DIR / fname)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: walk the legacy dataset names so callers asking for
    # ``pii_anon`` still locate ``pii_anon_eval_v1`` files in older installs.
    if name == _DEFAULT_DATASET:
        for legacy_name in _LEGACY_DATASET_NAMES:
            fallback = resolve_eval_dataset_path(legacy_name, source=source)
            if fallback is not None:
                return fallback

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
    """Infer dimension tags from record content for legacy records.

    Honors explicit ``dimension_tags`` (legacy) or ``dimensions`` (v1.1+)
    when present; otherwise derives tags heuristically.
    """
    tags: list[str] = list(row.get("dimension_tags") or row.get("dimensions") or [])
    primary = row.get("primary_dimension")
    if primary and primary not in tags:
        tags.insert(0, str(primary))
    if tags:
        return sorted(set(tags))

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

    # v1.1+ uses ``annotations`` with richer metadata; v1.0 used ``labels``.
    # Both are supported — the v1.1 shape is normalized down to the v1.0
    # minimum so downstream metrics keep working unchanged.
    labels_raw = list(row.get("annotations") or row.get("labels", []))
    labels: list[dict[str, Any]] = []
    for lbl in labels_raw:
        raw_type = str(lbl.get("entity_type", "")).strip()
        entity_type = _EVAL_DATA_ENTITY_TYPE_MAP.get(raw_type, raw_type)
        if entity_type == "_BENCHMARK_IGNORE":
            continue
        start = int(lbl.get("start", -1))
        end = int(lbl.get("end", -1))
        if not entity_type or start < 0 or end <= start or end > len(text):
            continue
        entry: dict[str, Any] = {"entity_type": entity_type, "start": start, "end": end}
        # Normalize cluster id from either shape.
        cluster_id = lbl.get("entity_cluster_id") or lbl.get("cluster_id")
        if cluster_id is not None:
            entry["entity_cluster_id"] = str(cluster_id)
        mention_variant = lbl.get("mention_variant")
        if mention_variant is not None:
            entry["mention_variant"] = str(mention_variant)
        context_group = lbl.get("context_group")
        if context_group is not None:
            entry["context_group"] = str(context_group)
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

    # fields: infer from content if not explicitly set
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

    # v1.1+ nests adversarial metadata under an ``adversarial`` block.
    adversarial_raw = row.get("adversarial")
    adversarial_block: dict[str, Any] | None = adversarial_raw if isinstance(adversarial_raw, dict) else None
    adversarial_type = row.get("adversarial_type")
    adversarial_attack_type = row.get("adversarial_attack_type")
    adversarial_difficulty = str(row.get("adversarial_difficulty", "clean"))
    if adversarial_block is not None:
        adversarial_type = adversarial_type or adversarial_block.get("type")
        adversarial_attack_type = adversarial_attack_type or adversarial_block.get("attack_type")
        adversarial_difficulty = str(
            adversarial_block.get("difficulty", adversarial_difficulty)
        )

    # v1.1+ nests re-identification info under ``privacy_risk``.
    privacy_risk_raw = row.get("privacy_risk")
    privacy_risk: dict[str, Any] = privacy_risk_raw if isinstance(privacy_risk_raw, dict) else {}
    if privacy_risk:
        reid_tier = str(
            privacy_risk.get("reidentification_risk")
            or privacy_risk.get("tier3_risk_level")
            or reid_tier
        )
        if reid_tier not in {"low", "moderate", "high", "critical"}:
            reid_tier = "low"
        if not quasi_identifiers and isinstance(privacy_risk.get("quasi_identifiers"), list):
            quasi_identifiers = [str(q) for q in privacy_risk["quasi_identifiers"]]

    # Tier 3 (dataset v1.3.0+) — behavioral signals block.
    behavioral_signals = row.get("behavioral_signals") or {}
    if not isinstance(behavioral_signals, dict):
        behavioral_signals = {}
    behavioral_signal_density = float(
        behavioral_signals.get("behavioral_signal_density", 0.0)
    )
    reidentification_contribution = str(
        behavioral_signals.get("reidentification_contribution", "low")
    )

    # Tier 3 — per-record re-identification resistance scoring.
    rrs_val = privacy_risk.get("re_identification_resistance_score") if privacy_risk else None
    re_identification_resistance_score = (
        float(rrs_val) if rrs_val is not None else None
    )
    reid_recall_val = privacy_risk.get("estimated_reid_recall") if privacy_risk else None
    estimated_reid_recall = float(reid_recall_val) if reid_recall_val is not None else None
    tier3_risk_level = str((privacy_risk or {}).get("tier3_risk_level", "low"))
    if tier3_risk_level not in {"low", "moderate", "high", "critical"}:
        tier3_risk_level = "low"

    # Tier 3 — paired-profile / ESRC evaluation block.
    tier3_block_raw = row.get("tier3_evaluation")
    tier3_block: dict[str, Any] = tier3_block_raw if isinstance(tier3_block_raw, dict) else {}
    is_paired_profile = bool(tier3_block.get("is_paired_profile", False))
    persona_id = tier3_block.get("persona_id")
    linked_profile_id = tier3_block.get("linked_profile_id")
    profile_type = tier3_block.get("profile_type")
    esrc_attack_target = bool(tier3_block.get("esrc_attack_target", False))
    expected_reidentification_difficulty = str(
        tier3_block.get("expected_reidentification_difficulty", "unknown")
    )
    behavioral_signal_removal_attempted = bool(
        tier3_block.get("behavioral_signal_removal_attempted", False)
    )

    # Tier 3 — anonymized variants + per-variant utility metrics.
    context_preservation = row.get("context_preservation") or {}
    if not isinstance(context_preservation, dict):
        context_preservation = {}

    # v1.1+ nests provenance metadata.
    provenance_raw = row.get("provenance")
    provenance: dict[str, Any] = provenance_raw if isinstance(provenance_raw, dict) else {}
    source_type_raw = row.get("source_type") or (provenance.get("source_type") if provenance else None) or "synthetic"
    source_type: Literal["synthetic", "curated_public"] = (
        "curated_public" if source_type_raw == "curated_public" else "synthetic"
    )
    license_val = str(
        row.get("license") or (provenance.get("license") if provenance else None) or "CC0-1.0"
    )
    source_id_val = str(
        row.get("source_id") or (provenance.get("source_id") if provenance else None) or "generated"
    )
    # v1.1+ populates ``domain`` where v1.0 used ``datatype_group``.
    datatype_group_val = str(row.get("datatype_group") or row.get("domain") or "general")

    return EvalBenchmarkRecord(
        record_id=record_id,
        text=text,
        labels=labels,
        language=language,
        source_type=source_type,
        source_id=source_id_val,
        license=license_val,
        scenario_id=str(row.get("scenario_id", "baseline")),
        entity_cluster_id=str(row.get("entity_cluster_id", "none")),
        mention_variant=str(row.get("mention_variant", "none")),
        context_group=str(row.get("context_group", "baseline")),
        datatype_group=datatype_group_val,
        difficulty_level=str(row.get("difficulty_level", "moderate")),
        data_type=data_type,  # type: ignore[arg-type]
        context_length_tier=context_length_tier,  # type: ignore[arg-type]
        token_count=int(row.get("token_count", len(text.split()))),
        regulatory_domain=regulatory_domain,
        adversarial_type=adversarial_type,
        script=str(row.get("script", "Latin")),
        entity_types_present=entity_types_present,
        # fields
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
        adversarial_attack_type=adversarial_attack_type,
        adversarial_difficulty=adversarial_difficulty,
        stratum_id=str(row.get("stratum_id", "")),
        # Tier 3 (v1.3.0+)
        behavioral_signal_density=behavioral_signal_density,
        reidentification_contribution=reidentification_contribution,
        behavioral_signals=behavioral_signals,
        re_identification_resistance_score=re_identification_resistance_score,
        estimated_reid_recall=estimated_reid_recall,
        tier3_risk_level=tier3_risk_level,
        is_paired_profile=is_paired_profile,
        persona_id=str(persona_id) if persona_id is not None else None,
        linked_profile_id=str(linked_profile_id) if linked_profile_id is not None else None,
        profile_type=str(profile_type) if profile_type is not None else None,
        esrc_attack_target=esrc_attack_target,
        expected_reidentification_difficulty=expected_reidentification_difficulty,
        behavioral_signal_removal_attempted=behavioral_signal_removal_attempted,
        context_preservation=context_preservation,
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
        (the unified dataset), falling back to ``pii_anon_eval``
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
    """Return distribution summary of the evaluation dataset.

    As of dataset v1.3.0, the summary additionally reports Tier 3 coverage:
    number of records with behavioral-signal annotations, average RRS, and
    Tier 3 risk-level distribution.
    """
    records = load_eval_dataset(name)
    summary: dict[str, Any] = {
        "dataset": name,
        "version": "latest",
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
        # Tier 3 (dataset v1.3.0+)
        "by_tier3_risk": {},
        "by_reidentification_contribution": {},
        "tier3_evaluation_records": 0,
        "paired_profile_records": 0,
        "esrc_attack_targets": 0,
        "records_with_behavioral_signals": 0,
        "records_with_rrs": 0,
        "rrs_sum": 0.0,
        "behavioral_signal_density_sum": 0.0,
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

        # summaries
        for dim in r.dimension_tags:
            summary["by_dimension"][dim] = summary["by_dimension"].get(dim, 0) + 1
        summary["by_context_type"][r.context_type] = summary["by_context_type"].get(r.context_type, 0) + 1
        summary["by_reidentification_risk"][r.reidentification_risk_tier] = (
            summary["by_reidentification_risk"].get(r.reidentification_risk_tier, 0) + 1
        )

        # Tier 3 distributions
        summary["by_tier3_risk"][r.tier3_risk_level] = (
            summary["by_tier3_risk"].get(r.tier3_risk_level, 0) + 1
        )
        summary["by_reidentification_contribution"][r.reidentification_contribution] = (
            summary["by_reidentification_contribution"].get(r.reidentification_contribution, 0) + 1
        )
        if r.behavioral_signals:
            summary["records_with_behavioral_signals"] += 1
            summary["behavioral_signal_density_sum"] += r.behavioral_signal_density
        if r.re_identification_resistance_score is not None:
            summary["records_with_rrs"] += 1
            summary["rrs_sum"] += r.re_identification_resistance_score
        if r.is_paired_profile:
            summary["paired_profile_records"] += 1
        if r.esrc_attack_target:
            summary["esrc_attack_targets"] += 1
        if r.is_paired_profile or r.esrc_attack_target or r.behavioral_signal_removal_attempted:
            summary["tier3_evaluation_records"] += 1

    summary["entity_types"] = sorted(summary["entity_types"])
    summary["regulatory_domains"] = sorted(summary["regulatory_domains"])
    # Convert running sums to averages.
    rrs_n = summary.pop("records_with_rrs")
    rrs_sum = summary.pop("rrs_sum")
    bsd_n = summary["records_with_behavioral_signals"]
    bsd_sum = summary.pop("behavioral_signal_density_sum")
    summary["avg_re_identification_resistance_score"] = (
        round(rrs_sum / rrs_n, 4) if rrs_n else None
    )
    summary["avg_behavioral_signal_density"] = (
        round(bsd_sum / bsd_n, 4) if bsd_n else None
    )
    summary["records_scored_for_rrs"] = rrs_n
    return summary
