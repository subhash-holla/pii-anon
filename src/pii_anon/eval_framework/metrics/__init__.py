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
]
