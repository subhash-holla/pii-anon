"""Evaluation metrics for PII anonymization and pseudonymization assessment.

This sub-package provides multi-level, multi-mode metrics grounded in
academic research (SemEval'13, nervaluate, TAB 2022, RAT-Bench 2025).
"""

from .base import (
    EvalMetricResult,
    EvaluationLevel,
    MatchMode,
    MultiLevelMetric,
)
from .deid_families import (
    AnonymizationScore,
    AnonymizationScorer,
    DeidFamilyScores,
    PseudonymizationIntegrityScore,
    PseudonymizationIntegrityScorer,
)
from .fairness_metrics import (
    DifficultyFairnessMetric,
    EntityTypeFairnessMetric,
    LanguageFairnessMetric,
    ScriptFairnessMetric,
)
from .privacy_metrics import (
    KAnonymityMetric,
    LDiversityMetric,
    LeakageDetectionMetric,
    ReidentificationRiskMetric,
    TClosenessMetric,
)
from .selective_risk import (
    AbstentionOperatingPoint,
    BrierDecomposition,
    RiskCoveragePoint,
    ScoredFinding,
    SelectiveRiskReport,
    SelectiveRiskReporter,
)
from .span_metrics import (
    DocumentLevelConsistencyMetric,
    EntityLevelF1Metric,
    ExactMatchMetric,
    PartialMatchMetric,
    StrictMatchMetric,
    TokenLevelF1Metric,
    TypeMatchMetric,
)
from .utility_metrics import (
    FormatPreservationMetric,
    InformationLossMetric,
    PrivacyUtilityTradeoffMetric,
    SemanticPreservationMetric,
)

__all__ = [
    "EvalMetricResult",
    "EvaluationLevel",
    "MatchMode",
    "MultiLevelMetric",
    # span
    "StrictMatchMetric",
    "ExactMatchMetric",
    "PartialMatchMetric",
    "TypeMatchMetric",
    "EntityLevelF1Metric",
    "TokenLevelF1Metric",
    "DocumentLevelConsistencyMetric",
    # privacy
    "ReidentificationRiskMetric",
    "KAnonymityMetric",
    "LDiversityMetric",
    "TClosenessMetric",
    "LeakageDetectionMetric",
    # de-id families (S4-01 — distinct anon vs pseudo, never merged)
    "AnonymizationScore",
    "AnonymizationScorer",
    "PseudonymizationIntegrityScore",
    "PseudonymizationIntegrityScorer",
    "DeidFamilyScores",
    # utility
    "FormatPreservationMetric",
    "SemanticPreservationMetric",
    "PrivacyUtilityTradeoffMetric",
    "InformationLossMetric",
    # fairness
    "LanguageFairnessMetric",
    "EntityTypeFairnessMetric",
    "DifficultyFairnessMetric",
    "ScriptFairnessMetric",
    # calibration & selective-risk (S4-03 — per-class ECE/Brier/AURC + abstention)
    "SelectiveRiskReporter",
    "SelectiveRiskReport",
    "ScoredFinding",
    "BrierDecomposition",
    "RiskCoveragePoint",
    "AbstentionOperatingPoint",
]
