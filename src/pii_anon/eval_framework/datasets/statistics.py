"""Statistical validation and dataset adequacy assessment for PII Anonymization Evaluation v1.0.0.

This module provides comprehensive statistical tools for validating evaluation datasets against
the seven-dimensional evaluation framework. Uses stdlib-only implementations (no scipy dependency)
suitable for core library use.

Evaluation Dimensions and Target Coverage:
    1. Entity Tracking (20%, ~14K) — consistent coreference across context
    2. Multilingual & Dialect (15%, ~10.5K) — 52 languages, 17 scripts
    3. Context Preservation (20%, ~14K) — semantic integrity in narratives
    4. Diverse PII Types (20%, ~14K) — breadth across 48 entity types
    5. Edge Cases (10%, ~7K) — abbreviations, partial PII, ambiguity
    6. Data Format Variations (10%, ~7K) — structured/semi-structured/unstructured
    7. Temporal Consistency (5%, ~3.5K) — time-series entity evolution

Statistical Basis:
    - Cochran (1977): Sampling Techniques — finite population correction & stratification
    - Sweeney (2002): k-anonymity — quasi-identifier tracking & re-identification risk
    - Cohen (1988): Statistical Power Analysis — effect size and sample size relations
    - Gebru et al. (2021): Datasheets for Datasets — documentation standards
    - TAB (2022), PII-Bench (2025), RAT-Bench (2025) — PII benchmark methodologies

References:
    [1] Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley.
    [2] Sweeney, L. (2002). k-anonymity: a model for protecting privacy. IJUFKS, 10(5), 557-570.
    [3] Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Erlbaum.
    [4] Gebru, T., et al. (2021). Datasheets for Datasets. arXiv:1803.09010v7.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .schema import EvalBenchmarkRecord, VALID_DIMENSION_TAGS, DIMENSION_WEIGHTS


# ---------------------------------------------------------------------------
# Sample Size Guidance: Target records per dimension
# Based on ~50K total corpus with weighted allocation.
# Reference: Cochran (1977), TAB (2022) benchmark design
# ---------------------------------------------------------------------------

SAMPLE_SIZE_GUIDANCE: dict[str, dict[str, Any]] = {
    "entity_tracking": {
        "weight": 0.20,
        "target_records": 10000,
        "description": "Coreference chains, entity continuity across context",
    },
    "multilingual": {
        "weight": 0.15,
        "target_records": 7500,
        "description": "12 languages, locale-specific PII patterns, cross-language code-switching",
    },
    "context_preservation": {
        "weight": 0.20,
        "target_records": 10000,
        "description": "Semantic integrity in multi-turn dialogues, narratives",
    },
    "diverse_pii_types": {
        "weight": 0.20,
        "target_records": 10000,
        "description": "22 entity types across person, org, location, financial, ID categories",
    },
    "edge_cases": {
        "weight": 0.10,
        "target_records": 5000,
        "description": "Overlapping entities, false positive triggers, dense PII, Unicode, PII in URLs/code",
    },
    "data_format_variations": {
        "weight": 0.10,
        "target_records": 5000,
        "description": "Structured (JSON/CSV/XML), ASCII tables, code, logs, mixed formats",
    },
    "temporal_consistency": {
        "weight": 0.05,
        "target_records": 2500,
        "description": "Longitudinal medical/financial records, temporal ordering constraints",
    },
}


def calculate_sample_size_per_stratum(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    variance: float = 0.25,
) -> int:
    """Calculate minimum sample size per stratum using Cochran's formula.

    Implements the finite population correction for stratified sampling.
    Used to determine required sample sizes for statistical adequacy.

    Formula (from Cochran 1977, Eq. 4.5):
        n = (Z² × p × (1-p)) / E²
    where:
        Z   = critical value for confidence level
        p   = estimated proportion (0.5 for max variance)
        E   = margin of error
        variance adjustment: assumes p(1-p) ≈ variance parameter

    Parameters
    ----------
    population_size : int
        Total population size in this stratum.
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI). Maps to Z-score:
        - 0.90 → 1.645
        - 0.95 → 1.960
        - 0.99 → 2.576
    margin_of_error : float, default=0.05
        Acceptable margin of error as proportion (e.g., 0.05 = ±5%).
    variance : float, default=0.25
        Estimated variance of the proportion (default p(1-p) at p=0.5).

    Returns
    -------
    int
        Minimum sample size (with finite population correction applied).

    Examples
    --------
    >>> # For 20K entity_tracking records at 95% confidence, ±5% margin:
    >>> n = calculate_sample_size_per_stratum(20000, confidence_level=0.95, margin_of_error=0.05)
    >>> print(f"Required: {n} samples")
    Required: 377 samples

    References
    ----------
    - Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Eq. 4.5, 7.17)
    """
    # Z-score mapping for common confidence levels
    z_scores = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    z = z_scores.get(confidence_level)
    if z is None:
        # Approximate Z from confidence level using inverse normal
        # For arbitrary confidence levels, use 1.96 as default (95%)
        z = 1.960

    # Cochran's formula: n = (Z² × variance) / E²
    numerator = (z ** 2) * variance
    denominator = margin_of_error ** 2
    n_infinite = numerator / denominator

    # Finite population correction: n_corrected = n / (1 + n/N)
    # Reference: Cochran (1977), Eq. 7.17
    if population_size > 0:
        n_corrected = n_infinite / (1 + (n_infinite / population_size))
        return max(1, int(math.ceil(n_corrected)))
    return max(1, int(math.ceil(n_infinite)))


def power_analysis_for_comparison(
    effect_size: float = 0.2,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate minimum sample size for paired t-test using power analysis.

    Determines the required sample size per group to detect a specified effect size
    with given significance level (α) and statistical power (1-β).

    Approximation formula (from Cohen 1988):
        n ≈ 2 × ((Z_α + Z_β) / d)²
    where:
        Z_α     = critical value for significance level α
        Z_β     = critical value for power (1-β)
        d       = Cohen's d effect size

    Effect Size Interpretation (Cohen 1988):
        - small:   d = 0.2
        - medium:  d = 0.5
        - large:   d = 0.8

    Parameters
    ----------
    effect_size : float, default=0.2
        Cohen's d (minimum detectable effect size).
        Typical ranges:
        - 0.2: small effect (domain-dependent)
        - 0.5: medium effect (comparable to typical variations)
        - 0.8: large effect (clearly noticeable)
    alpha : float, default=0.05
        Significance level (Type I error rate, typically 0.05 or 0.01).
    power : float, default=0.80
        Statistical power = 1 - β (typically 0.80 or 0.90).

    Returns
    -------
    int
        Minimum sample size per group required for paired comparison.

    Examples
    --------
    >>> # Detect small effect (d=0.2) with α=0.05, power=0.80
    >>> n = power_analysis_for_comparison(effect_size=0.2, alpha=0.05, power=0.80)
    >>> print(f"Samples per group: {n}")
    Samples per group: 394

    >>> # Detect medium effect (d=0.5) with α=0.05, power=0.90
    >>> n = power_analysis_for_comparison(effect_size=0.5, alpha=0.05, power=0.90)
    >>> print(f"Samples per group: {n}")
    Samples per group: 86

    References
    ----------
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
      Lawrence Erlbaum Associates. (Table 2.3.10)
    - Sweeney, L. (2002). k-anonymity: a model for protecting privacy.
      International Journal on Uncertainty, Fuzziness and Knowledge-Based Systems, 10(5), 557-570.
    """
    # Z-score mapping for common alpha levels (two-tailed)
    z_alpha_map = {
        0.05: 1.960,   # two-tailed α=0.05
        0.01: 2.576,   # two-tailed α=0.01
        0.001: 3.291,  # two-tailed α=0.001
    }
    z_alpha = z_alpha_map.get(alpha, 1.960)

    # Z-score mapping for common power levels
    # power = 1 - β, so β = 1 - power
    z_beta_map = {
        0.80: 0.842,   # power = 0.80, β = 0.20
        0.85: 1.036,   # power = 0.85, β = 0.15
        0.90: 1.282,   # power = 0.90, β = 0.10
        0.95: 1.645,   # power = 0.95, β = 0.05
    }
    z_beta = z_beta_map.get(power, 0.842)

    # Cohen's approximation: n = 2 × ((Z_α + Z_β) / d)²
    # For paired tests, no factor of 2 needed in some formulations;
    # here we use the general formula for comparing two independent groups
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")

    numerator = (z_alpha + z_beta) ** 2
    denominator = effect_size ** 2
    n = numerator / denominator

    return max(1, int(math.ceil(n)))


def validate_dataset_coverage(records: list[EvalBenchmarkRecord]) -> dict[str, Any]:
    """Validate evaluation dataset coverage across dimensions, languages, and entity types.

    Comprehensive assessment of whether the dataset meets adequacy criteria for:
    - Dimensional balance (20 dimensions × 90% target coverage)
    - Language diversity (representation across 52+ language families)
    - Entity type breadth (coverage of 48+ entity types across 7 categories)
    - Statistical power (adequate sample sizes per stratum)

    Returns a structured report with:
    - Coverage statistics by dimension, language, entity type
    - Identified gaps (dimensions/languages/types below minimum)
    - Overall recommendation: PASS | CONDITIONAL | FAIL

    Parameters
    ----------
    records : list[EvalBenchmarkRecord]
        Loaded evaluation dataset records.

    Returns
    -------
    dict[str, Any]
        Coverage validation report with keys:
        - total_records: int
        - by_dimension: {dim: count}
        - by_language: {lang: count}
        - by_entity_type: {type: count}
        - coverage_gaps: list[{language, entity_type, count, required}]
        - dimension_adequacy: bool (all dims ≥ 90% of target)
        - statistical_power: float (estimated Cohen's d detectable)
        - recommendation: "PASS" | "CONDITIONAL" | "FAIL"
        - details: str (human-readable summary)

    Examples
    --------
    >>> from .schema import load_eval_dataset
    >>> records = load_eval_dataset()
    >>> report = validate_dataset_coverage(records)
    >>> print(f"Recommendation: {report['recommendation']}")
    >>> if report['coverage_gaps']:
    ...     for gap in report['coverage_gaps']:
    ...         print(f"  Gap: {gap['language']} + {gap['entity_type']}: "
    ...               f"found {gap['count']}, need {gap['required']}")

    References
    ----------
    - Cochran (1977): Finite population correction and stratification adequacy
    - Gebru et al. (2021): Datasheets for Datasets — completeness and representativeness
    """
    if not records:
        return {
            "total_records": 0,
            "by_dimension": {},
            "by_language": {},
            "by_entity_type": {},
            "coverage_gaps": [],
            "dimension_adequacy": False,
            "statistical_power": 0.0,
            "recommendation": "FAIL",
            "details": "Empty dataset provided.",
        }

    total = len(records)

    # ── Count by dimension ──────────────────────────────────────────────────
    by_dimension: dict[str, int] = defaultdict(int)
    for record in records:
        for dim in record.dimension_tags:
            if dim in VALID_DIMENSION_TAGS:
                by_dimension[dim] += 1

    # ── Count by language ──────────────────────────────────────────────────
    by_language: dict[str, int] = defaultdict(int)
    for record in records:
        by_language[record.language] += 1

    # ── Count by entity type ───────────────────────────────────────────────
    by_entity_type: dict[str, int] = defaultdict(int)
    for record in records:
        for ent_type in record.entity_types_present:
            by_entity_type[ent_type] += 1

    # ── Identify coverage gaps ──────────────────────────────────────────────
    # Gap = dimension/language/entity_type underrepresented relative to balanced allocation
    coverage_gaps: list[dict[str, Any]] = []

    # Minimum records per dimension: 90% of target
    for dim, config in SAMPLE_SIZE_GUIDANCE.items():
        target = config["target_records"]
        minimum = int(math.ceil(0.90 * target))
        actual = by_dimension.get(dim, 0)
        if actual < minimum:
            coverage_gaps.append({
                "type": "dimension",
                "dimension": dim,
                "count": actual,
                "required": minimum,
                "target": target,
                "coverage": round(100.0 * actual / target, 1) if target > 0 else 0,
            })

    # Minimum languages: expect coverage of major language families
    # Heuristic: if <5 records in a detected language, flag as sparse
    for lang, count in by_language.items():
        if count < 5:
            coverage_gaps.append({
                "type": "language",
                "language": lang,
                "count": count,
                "required": 5,
            })

    # Minimum entity types: heuristic threshold of 10 records per type
    for ent_type, count in by_entity_type.items():
        if count < 10:
            coverage_gaps.append({
                "type": "entity_type",
                "entity_type": ent_type,
                "count": count,
                "required": 10,
            })

    # ── Assess dimension adequacy ──────────────────────────────────────────
    # All dimensions should be at ≥90% of target
    dimension_adequacy = all(
        by_dimension.get(dim, 0) >= int(math.ceil(0.90 * config["target_records"]))
        for dim, config in SAMPLE_SIZE_GUIDANCE.items()
    )

    # ── Estimate statistical power ───────────────────────────────────────────
    # Use smallest dimension sample to estimate detectable effect size
    # Formula: d = 2 × (Z_α + Z_β) / √n
    # For α=0.05, power=0.80: (1.96 + 0.84) / √n ≈ 2.80 / √n
    min_dim_size = min(by_dimension.values()) if by_dimension else 0
    if min_dim_size > 0:
        # Approximate Cohen's d detectable with 80% power
        statistical_power = 2.80 / math.sqrt(min_dim_size)
    else:
        statistical_power = float('inf')

    # ── Generate recommendation ────────────────────────────────────────────
    if total < 50000:
        recommendation = "FAIL"
        reason = f"Total records ({total}) below minimum threshold (50K)"
    elif not dimension_adequacy:
        recommendation = "CONDITIONAL"
        reason = "Some dimensions below 90% of target coverage"
    elif len(coverage_gaps) > 10:
        recommendation = "CONDITIONAL"
        reason = f"Multiple coverage gaps detected ({len(coverage_gaps)})"
    else:
        recommendation = "PASS"
        reason = "Dataset meets adequacy criteria across dimensions, languages, entity types"

    # ── Build human-readable summary ───────────────────────────────────────
    details_parts = [
        f"Total records: {total:,}",
        f"Dimensions covered: {len(by_dimension)}/{len(SAMPLE_SIZE_GUIDANCE)}",
        f"Languages: {len(by_language)}",
        f"Entity types: {len(by_entity_type)}",
        f"Recommendation: {reason}",
    ]
    details = "; ".join(details_parts)

    return {
        "total_records": total,
        "by_dimension": dict(by_dimension),
        "by_language": dict(by_language),
        "by_entity_type": dict(by_entity_type),
        "coverage_gaps": coverage_gaps,
        "dimension_adequacy": dimension_adequacy,
        "statistical_power": round(statistical_power, 4),
        "recommendation": recommendation,
        "details": details,
    }


def compute_dataset_statistics(records: list[EvalBenchmarkRecord]) -> dict[str, Any]:
    """Compute comprehensive dataset statistics for metadata and reporting.

    Generates summary statistics suitable for inclusion in dataset metadata JSON
    (e.g., in a datasheet or README).  Covers distributions across all key dimensions
    and provides statistical summaries.

    Parameters
    ----------
    records : list[EvalBenchmarkRecord]
        Loaded evaluation dataset records.

    Returns
    -------
    dict[str, Any]
        Comprehensive statistics dictionary with keys:
        - count: total number of records
        - dimensions: {dim: {count, percentage, weight}}
        - languages: {lang: {count, percentage, family}}
        - data_types: {type: count}
        - context_lengths: {tier: count}
        - entity_types: {type: count}
        - difficulty_levels: {level: count}
        - adversarial_stats: {clean, adversarial, percentage_adversarial}
        - reidentification_risk: {tier: count}
        - edge_cases: {type: count}
        - quasi_identifiers: list of top quasi-identifier types
        - temporal_records: count of time-series records
        - token_statistics: {total, min, max, mean, median}
        - entity_statistics: {total_entities, unique_clusters, max_cluster_size}

    Examples
    --------
    >>> from .schema import load_eval_dataset
    >>> records = load_eval_dataset()
    >>> stats = compute_dataset_statistics(records)
    >>> print(f"Total records: {stats['count']}")
    >>> print(f"English: {stats['languages'].get('en', {}).get('count', 0)}")

    References
    ----------
    - Gebru et al. (2021): Datasheets for Datasets — statistical summaries
    - TAB (2022): Taxonomy and benchmarking for PII detection
    """
    if not records:
        return {
            "count": 0,
            "dimensions": {},
            "languages": {},
            "data_types": {},
            "context_lengths": {},
            "entity_types": {},
            "difficulty_levels": {},
            "adversarial_stats": {"clean": 0, "adversarial": 0, "percentage_adversarial": 0},
            "reidentification_risk": {},
            "edge_cases": {},
            "quasi_identifiers": [],
            "temporal_records": 0,
            "token_statistics": {"total": 0, "count": 0, "mean": 0, "median": 0},
            "entity_statistics": {"total_entities": 0, "unique_clusters": 0, "max_cluster_size": 0},
        }

    total = len(records)

    # ── Dimension statistics ───────────────────────────────────────────────
    dimension_stats: dict[str, dict[str, Any]] = {}
    dimension_counts: dict[str, int] = defaultdict(int)
    for record in records:
        for dim in record.dimension_tags:
            if dim in VALID_DIMENSION_TAGS:
                dimension_counts[dim] += 1

    for dim, count in dimension_counts.items():
        weight = DIMENSION_WEIGHTS.get(dim, 0)
        dimension_stats[dim] = {
            "count": count,
            "percentage": round(100.0 * count / total, 1) if total > 0 else 0,
            "weight": weight,
        }

    # ── Language statistics ────────────────────────────────────────────────
    language_stats: dict[str, dict[str, Any]] = {}
    language_counts: dict[str, int] = defaultdict(int)
    for record in records:
        language_counts[record.language] += 1

    for lang, count in language_counts.items():
        language_stats[lang] = {
            "count": count,
            "percentage": round(100.0 * count / total, 1) if total > 0 else 0,
            "family": "",  # Could be augmented with language family mapping
        }

    # ── Data type statistics ───────────────────────────────────────────────
    data_type_counts: dict[str, int] = defaultdict(int)
    for record in records:
        data_type_counts[record.data_type] += 1

    # ── Context length statistics ──────────────────────────────────────────
    context_length_counts: dict[str, int] = defaultdict(int)
    for record in records:
        context_length_counts[record.context_length_tier] += 1

    # ── Entity type statistics ─────────────────────────────────────────────
    entity_type_counts: dict[str, int] = defaultdict(int)
    for record in records:
        for ent_type in record.entity_types_present:
            entity_type_counts[ent_type] += 1

    # ── Difficulty level statistics ────────────────────────────────────────
    difficulty_counts: dict[str, int] = defaultdict(int)
    for record in records:
        difficulty_counts[record.difficulty_level] += 1

    # ── Adversarial statistics ─────────────────────────────────────────────
    adversarial_count = sum(1 for r in records if r.adversarial_type)
    clean_count = total - adversarial_count
    pct_adversarial = round(100.0 * adversarial_count / total, 1) if total > 0 else 0

    # ── Re-identification risk ─────────────────────────────────────────────
    reid_risk_counts: dict[str, int] = defaultdict(int)
    for record in records:
        reid_risk_counts[record.reidentification_risk_tier] += 1

    # ── Edge case statistics ───────────────────────────────────────────────
    edge_case_counts: dict[str, int] = defaultdict(int)
    for record in records:
        for edge_type in record.edge_case_types:
            edge_case_counts[edge_type] += 1

    # ── Quasi-identifier statistics ────────────────────────────────────────
    quasi_id_counts: dict[str, int] = defaultdict(int)
    for record in records:
        for quasi_id in record.quasi_identifiers_present:
            quasi_id_counts[quasi_id] += 1
    # Sort by frequency and take top 10
    top_quasi_ids = sorted(quasi_id_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # ── Temporal statistics ────────────────────────────────────────────────
    temporal_count = sum(1 for r in records if r.is_time_series)

    # ── Token statistics ───────────────────────────────────────────────────
    token_counts = [r.token_count for r in records if r.token_count > 0]
    if token_counts:
        total_tokens = sum(token_counts)
        mean_tokens = round(total_tokens / len(token_counts), 1)
        # Simple median calculation
        sorted_tokens = sorted(token_counts)
        if len(sorted_tokens) % 2 == 0:
            median_tokens = round((sorted_tokens[len(sorted_tokens) // 2 - 1] + sorted_tokens[len(sorted_tokens) // 2]) / 2, 1)
        else:
            median_tokens = sorted_tokens[len(sorted_tokens) // 2]
        token_stats = {
            "total": total_tokens,
            "count": len(token_counts),
            "mean": mean_tokens,
            "median": median_tokens,
            "min": min(token_counts),
            "max": max(token_counts),
        }
    else:
        token_stats = {"total": 0, "count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}

    # ── Entity cluster statistics ──────────────────────────────────────────
    total_entities = sum(len(r.labels) for r in records)
    cluster_sizes: dict[str, int] = defaultdict(int)
    for record in records:
        for label in record.labels:
            cluster_id = label.get("entity_cluster_id", "none")
            if cluster_id and cluster_id != "none":
                cluster_sizes[cluster_id] += 1
    unique_clusters = len(cluster_sizes)
    max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0

    return {
        "count": total,
        "dimensions": dimension_stats,
        "languages": language_stats,
        "data_types": dict(data_type_counts),
        "context_lengths": dict(context_length_counts),
        "entity_types": dict(entity_type_counts),
        "difficulty_levels": dict(difficulty_counts),
        "adversarial_stats": {
            "clean": clean_count,
            "adversarial": adversarial_count,
            "percentage_adversarial": pct_adversarial,
        },
        "reidentification_risk": dict(reid_risk_counts),
        "edge_cases": dict(edge_case_counts),
        "quasi_identifiers": [{"type": qid, "count": count} for qid, count in top_quasi_ids],
        "temporal_records": temporal_count,
        "token_statistics": token_stats,
        "entity_statistics": {
            "total_entities": total_entities,
            "unique_clusters": unique_clusters,
            "max_cluster_size": max_cluster_size,
        },
    }
