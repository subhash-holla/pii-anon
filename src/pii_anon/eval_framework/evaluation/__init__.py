"""Evaluation framework orchestration and reporting."""

from .aggregation import MetricAggregator
from .context_evaluator import DocumentContextEvaluator
from .framework import (
    BatchEvaluationReport,
    ContextualEvaluationReport,
    EvaluationFramework,
    EvaluationFrameworkConfig,
    EvaluationReport,
)
from .reporting import ReportGenerator

__all__ = [
    "EvaluationFramework",
    "EvaluationFrameworkConfig",
    "EvaluationReport",
    "BatchEvaluationReport",
    "ContextualEvaluationReport",
    "MetricAggregator",
    "DocumentContextEvaluator",
    "ReportGenerator",
]
