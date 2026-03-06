"""pii-anon Evaluation Framework — industry-leading PII de-identification evaluation.

This module provides a comprehensive, evidence-backed evaluation framework for
PII anonymization and pseudonymization systems.  It covers:

* **44 entity types** across 7 NIST/GDPR/ISO-aligned categories
* **52 languages** aligned with the OpenNER 1.0 benchmark
* **Multi-level evaluation** (token, entity, document, mention)
* **4 SemEval'13 matching modes** (strict, exact, partial, type)
* **Privacy metrics** (re-identification risk, k-anonymity, l-diversity, t-closeness)
* **Utility metrics** (format preservation, semantic preservation, privacy-utility trade-off)
* **Fairness metrics** (cross-language, cross-entity-type, cross-difficulty, cross-script)
* **Standards compliance** (NIST SP 800-122, GDPR, ISO 27701, HIPAA, CCPA)
* **50,000-record benchmark dataset** spanning all languages and entity types

Quick start::

    from pii_anon.eval_framework import (
        EvaluationFramework,
        EvaluationFrameworkConfig,
        PII_TAXONOMY,
        SUPPORTED_LANGUAGES,
        EntityTypeRegistry,
        ComplianceValidator,
    )

    # Run a basic evaluation
    fw = EvaluationFramework()
    report = fw.evaluate(predictions, labels, language="en")

    # Validate compliance
    validator = ComplianceValidator()
    compliance = validator.validate(my_entity_types, standard="gdpr")

Evidence basis: Every component is backed by peer-reviewed research.
See ``pii_anon.eval_framework.research.references`` for full citations.
"""

from __future__ import annotations

# -- Taxonomy ---------------------------------------------------------------
from .taxonomy import (
    EntityCategory,
    EntityTypeProfile,
    EntityTypeRegistry,
    PII_TAXONOMY,
    RiskLevel,
)

# -- Languages --------------------------------------------------------------
from .languages import (
    LanguageProfile,
    ResourceLevel,
    Script,
    SUPPORTED_LANGUAGES,
)

# -- Metrics (base) ---------------------------------------------------------
from .metrics.base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    MultiLevelMetric,
)

# -- Span metrics -----------------------------------------------------------
from .metrics.span_metrics import (
    DocumentLevelConsistencyMetric,
    EntityLevelF1Metric,
    ExactMatchMetric,
    PartialMatchMetric,
    StrictMatchMetric,
    TokenLevelF1Metric,
    TypeMatchMetric,
)

# -- Privacy metrics --------------------------------------------------------
from .metrics.privacy_metrics import (
    KAnonymityMetric,
    LDiversityMetric,
    LeakageDetectionMetric,
    ReidentificationRiskMetric,
    TClosenessMetric,
)

# -- Utility metrics --------------------------------------------------------
from .metrics.utility_metrics import (
    FormatPreservationMetric,
    InformationLossMetric,
    PrivacyUtilityTradeoffMetric,
    SemanticPreservationMetric,
)

# -- Fairness metrics -------------------------------------------------------
from .metrics.fairness_metrics import (
    DifficultyFairnessMetric,
    EntityTypeFairnessMetric,
    LanguageFairnessMetric,
    ScriptFairnessMetric,
)

# -- Datasets ---------------------------------------------------------------
from .datasets.schema import (
    EvalBenchmarkRecord,
    load_eval_dataset,
    resolve_eval_dataset_path,
    summarize_eval_dataset,
)

# -- Evaluation orchestration -----------------------------------------------
from .evaluation.framework import (
    BatchEvaluationReport,
    ContextualEvaluationReport,
    EvaluationFramework,
    EvaluationFrameworkConfig,
    EvaluationReport,
)
from .evaluation.aggregation import MetricAggregator
from .evaluation.context_evaluator import DocumentContextEvaluator
from .evaluation.reporting import ReportGenerator

# -- Standards compliance ---------------------------------------------------
from .standards.compliance import (
    ComplianceReport,
    ComplianceStandard,
    ComplianceValidator,
    CoverageGap,
    MultiStandardComplianceReport,
    validate_all_standards,
    validate_compliance,
)

# -- Composite metric -------------------------------------------------------
from .metrics.composite import (
    CompositeConfig,
    CompositeScore,
    compute_composite,
    compute_composite_from_benchmark_result,
    normalize_attack_success_rate,
    normalize_canary_exposure,
    normalize_entity_coverage,
    normalize_epsilon_dp,
    normalize_k_anonymity,
    normalize_mia_auc,
)

# -- Rating engine ----------------------------------------------------------
from .rating import (
    BenchmarkScorecard,
    PIIRateEloEngine,
    EloRating,
    GovernanceResult,
    GovernanceThresholds,
    Leaderboard,
    LeaderboardExporter,
    RatingUpdate,
    SystemScorecard,
)

# -- Research references ----------------------------------------------------
from .research.references import (
    EVIDENCE_REGISTRY,
    ResearchReference,
    all_references,
    get_references_for,
)


__all__ = [
    # Taxonomy
    "EntityCategory",
    "EntityTypeProfile",
    "EntityTypeRegistry",
    "PII_TAXONOMY",
    "RiskLevel",
    # Languages
    "LanguageProfile",
    "ResourceLevel",
    "Script",
    "SUPPORTED_LANGUAGES",
    # Metrics - base
    "EvalMetricResult",
    "EvaluationLevel",
    "LabeledSpan",
    "MatchMode",
    "MultiLevelMetric",
    # Metrics - span
    "DocumentLevelConsistencyMetric",
    "EntityLevelF1Metric",
    "ExactMatchMetric",
    "PartialMatchMetric",
    "StrictMatchMetric",
    "TokenLevelF1Metric",
    "TypeMatchMetric",
    # Metrics - privacy
    "KAnonymityMetric",
    "LDiversityMetric",
    "LeakageDetectionMetric",
    "ReidentificationRiskMetric",
    "TClosenessMetric",
    # Metrics - utility
    "FormatPreservationMetric",
    "InformationLossMetric",
    "PrivacyUtilityTradeoffMetric",
    "SemanticPreservationMetric",
    # Metrics - fairness
    "DifficultyFairnessMetric",
    "EntityTypeFairnessMetric",
    "LanguageFairnessMetric",
    "ScriptFairnessMetric",
    # Datasets
    "EvalBenchmarkRecord",
    "load_eval_dataset",
    "resolve_eval_dataset_path",
    "summarize_eval_dataset",
    # Evaluation orchestration
    "BatchEvaluationReport",
    "ContextualEvaluationReport",
    "DocumentContextEvaluator",
    "EvaluationFramework",
    "EvaluationFrameworkConfig",
    "EvaluationReport",
    "MetricAggregator",
    "ReportGenerator",
    # Standards compliance
    "ComplianceReport",
    "ComplianceStandard",
    "ComplianceValidator",
    "CoverageGap",
    "MultiStandardComplianceReport",
    "validate_all_standards",
    "validate_compliance",
    # Composite metric
    "CompositeConfig",
    "CompositeScore",
    "compute_composite",
    "compute_composite_from_benchmark_result",
    "normalize_attack_success_rate",
    "normalize_canary_exposure",
    "normalize_entity_coverage",
    "normalize_epsilon_dp",
    "normalize_k_anonymity",
    "normalize_mia_auc",
    # Rating engine
    "BenchmarkScorecard",
    "PIIRateEloEngine",
    "EloRating",
    "GovernanceResult",
    "GovernanceThresholds",
    "Leaderboard",
    "LeaderboardExporter",
    "RatingUpdate",
    "SystemScorecard",
    # Research
    "EVIDENCE_REGISTRY",
    "ResearchReference",
    "all_references",
    "get_references_for",
]
