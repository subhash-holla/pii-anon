"""Main evaluation framework orchestrator.

Coordinates multi-level, multi-mode evaluation across languages,
entity types, data types, difficulty levels, and context lengths.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from ..datasets.schema import EvalBenchmarkRecord
from ..metrics.base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
)
from ..metrics.fairness_metrics import (
    EntityTypeFairnessMetric,
    LanguageFairnessMetric,
    ComprehensiveFairnessMetric,
)
from ..metrics.privacy_metrics import LeakageDetectionMetric, ReidentificationRiskMetric
from ..metrics.span_metrics import (
    DocumentLevelConsistencyMetric,
    EntityLevelF1Metric,
    TokenLevelF1Metric,
)
from ..metrics.utility_metrics import InformationLossMetric
from ..metrics.composite import CompositeConfig, compute_composite
from ..metrics.streaming import StreamingPipeline
from ..standards.compliance import ComplianceReport, ComplianceValidator
from .aggregation import MetricAggregator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationFrameworkConfig:
    """Configuration for evaluation runs."""

    framework_id: str = "default"
    languages: list[str] = field(default_factory=lambda: ["en"])
    entity_types: list[str] | None = None  # None = all types
    evaluation_levels: list[EvaluationLevel] = field(
        default_factory=lambda: [EvaluationLevel.ENTITY],
    )
    match_modes: list[MatchMode] = field(
        default_factory=lambda: [MatchMode.STRICT, MatchMode.PARTIAL],
    )
    aggregation_methods: list[Literal["micro", "macro", "weighted"]] = field(
        default_factory=lambda: ["micro", "macro", "weighted"],
    )
    include_privacy_metrics: bool = True
    include_utility_metrics: bool = True
    include_fairness_metrics: bool = True
    include_confidence_intervals: bool = True
    include_composite_metric: bool = False
    partial_match_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Results from a single evaluation run."""

    evaluation_id: str
    timestamp: str
    config: EvaluationFrameworkConfig
    metrics_by_level: dict[str, dict[str, EvalMetricResult]] = field(default_factory=dict)
    metrics_by_match_mode: dict[str, dict[str, EvalMetricResult]] = field(default_factory=dict)
    per_entity_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    language: str = "en"
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    privacy_score: float = 0.0
    utility_score: float = 0.0
    fairness_score: float = 0.0
    records_evaluated: int = 0
    composite_score: float | None = None


@dataclass
class BatchEvaluationReport:
    """Aggregated results from batch evaluation."""

    evaluation_id: str
    timestamp: str
    records_evaluated: int
    micro_averaged: dict[str, float] = field(default_factory=dict)
    macro_averaged: dict[str, float] = field(default_factory=dict)
    weighted_averaged: dict[str, float] = field(default_factory=dict)
    per_entity_type: dict[str, dict[str, float]] = field(default_factory=dict)
    per_language: dict[str, dict[str, float]] = field(default_factory=dict)
    per_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    per_data_type: dict[str, dict[str, float]] = field(default_factory=dict)
    per_context_length: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None
    privacy_score: float = 0.0
    fairness_score: float = 0.0
    composite_score: float | None = None


@dataclass
class ContextualEvaluationReport(EvaluationReport):
    """Document-level evaluation with context."""

    mention_clustering_accuracy: float = 0.0
    entity_consistency: float = 0.0
    cross_segment_recall: float = 0.0


@dataclass
class ComprehensiveEvaluationReport:
    """Full evaluation across all metric families."""

    evaluation_id: str
    timestamp: str
    records_evaluated: int
    # Core span metrics
    micro_averaged: dict[str, float] = field(default_factory=dict)
    macro_averaged: dict[str, float] = field(default_factory=dict)
    weighted_averaged: dict[str, float] = field(default_factory=dict)
    per_entity_type: dict[str, dict[str, float]] = field(default_factory=dict)
    # Privacy metrics
    privacy_score: float = 0.0
    leakage_score: float = 0.0
    # Utility metrics
    utility_score: float = 0.0
    format_preservation: float = 0.0
    information_loss: float = 0.0
    # Fairness
    fairness_score: float = 0.0
    fairness_details: dict[str, Any] = field(default_factory=dict)
    # Per-dimension (if benchmark dataset)
    per_dimension: dict[str, dict[str, float]] = field(default_factory=dict)
    # Composite
    composite_score: float | None = None
    floor_gates_passed: bool | None = None
    # CIs
    confidence_interval: tuple[float, float] | None = None
    per_entity_ci: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

class EvaluationFramework:
    """Main evaluation orchestrator.

    Coordinates multi-level, multi-mode evaluation with privacy, utility,
    and fairness metrics.

    Usage::

        from pii_anon.eval_framework import EvaluationFramework

        fw = EvaluationFramework()
        report = fw.evaluate(predictions, labels, language="en")
        batch_report = fw.evaluate_batch(records)
    """

    def __init__(self, config: EvaluationFrameworkConfig | None = None) -> None:
        self.config = config or EvaluationFrameworkConfig()
        self._entity_f1 = EntityLevelF1Metric(
            partial_threshold=self.config.partial_match_threshold,
        )
        self._token_f1 = TokenLevelF1Metric()
        self._doc_consistency = DocumentLevelConsistencyMetric()
        self._reid_risk = ReidentificationRiskMetric()
        self._leakage = LeakageDetectionMetric()
        self._info_loss = InformationLossMetric()
        self._lang_fairness = LanguageFairnessMetric()
        self._entity_fairness = EntityTypeFairnessMetric()
        self._comprehensive_fairness = ComprehensiveFairnessMetric()
        self._aggregator = MetricAggregator()
        self._compliance = ComplianceValidator()

    # ── Core evaluation ────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Run comprehensive evaluation on a single record / batch of spans."""
        eval_id = str(uuid.uuid4())[:12]
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ctx = context or {}

        metrics_by_level: dict[str, dict[str, EvalMetricResult]] = {}
        metrics_by_mode: dict[str, dict[str, EvalMetricResult]] = {}

        # Entity-level across all match modes
        for mode in self.config.match_modes:
            result = self._entity_f1.compute(
                predictions, labels, level=EvaluationLevel.ENTITY, match_mode=mode,
            )
            metrics_by_mode.setdefault(mode.value, {})[self._entity_f1.name] = result

        # Token-level
        if EvaluationLevel.TOKEN in self.config.evaluation_levels:
            tok_result = self._token_f1.compute(predictions, labels)
            metrics_by_level.setdefault("token", {})[self._token_f1.name] = tok_result

        # Document-level consistency
        if EvaluationLevel.DOCUMENT in self.config.evaluation_levels:
            doc_result = self._doc_consistency.compute(
                predictions, labels, context=ctx,
            )
            metrics_by_level.setdefault("document", {})[self._doc_consistency.name] = doc_result

        # Primary F1 (strict mode)
        primary = self._entity_f1.compute(
            predictions, labels, match_mode=MatchMode.STRICT,
        )

        # Privacy metrics
        privacy_score = 0.0
        if self.config.include_privacy_metrics and ctx.get("anonymized_text"):
            reid = self._reid_risk.compute(predictions, labels, context=ctx)
            privacy_score = 1.0 - reid.value  # invert: lower risk = higher score

        # Fairness
        fairness_score = 0.0
        if self.config.include_fairness_metrics:
            ef = self._entity_fairness.compute(predictions, labels)
            fairness_score = 1.0 - ef.value  # lower gap = higher fairness

        # Composite metric (optional, off by default)
        composite_val: float | None = None
        if self.config.include_composite_metric:
            comp_cfg = CompositeConfig(
                weight_privacy=0.15 if self.config.include_privacy_metrics else 0.0,
                weight_utility=0.10 if self.config.include_utility_metrics else 0.0,
                weight_fairness=0.05 if self.config.include_fairness_metrics else 0.0,
            )
            comp = compute_composite(
                f1=primary.f1,
                precision=primary.precision,
                recall=primary.recall,
                latency_ms=0.0,
                docs_per_hour=0.0,
                privacy_score=privacy_score,
                utility_score=0.0,
                fairness_score=fairness_score,
                config=comp_cfg,
            )
            composite_val = comp.score

        return EvaluationReport(
            evaluation_id=eval_id,
            timestamp=ts,
            config=self.config,
            metrics_by_level=metrics_by_level,
            metrics_by_match_mode=metrics_by_mode,
            per_entity_breakdown=primary.per_entity_breakdown,
            language=language,
            precision=primary.precision,
            recall=primary.recall,
            f1=primary.f1,
            privacy_score=round(privacy_score, 6),
            utility_score=0.0,
            fairness_score=round(fairness_score, 6),
            records_evaluated=1,
            composite_score=composite_val,
        )

    # ── Batch evaluation ───────────────────────────────────────────────

    def evaluate_batch(
        self,
        records: list[EvalBenchmarkRecord],
        *,
        predict_fn: Any | None = None,
    ) -> BatchEvaluationReport:
        """Evaluate a batch of records with full aggregation.

        If *predict_fn* is ``None``, labels are used as perfect predictions
        (useful for dataset-only analysis).
        """
        eval_id = str(uuid.uuid4())[:12]
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        all_preds: list[LabeledSpan] = []
        all_labels: list[LabeledSpan] = []
        per_lang_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        per_diff_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        per_dtype_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        per_ctx_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        per_record_f1s: list[float] = []

        for record in records:
            labels = [
                LabeledSpan(
                    entity_type=lbl["entity_type"],
                    start=lbl["start"],
                    end=lbl["end"],
                    record_id=record.record_id,
                )
                for lbl in record.labels
            ]
            if predict_fn is not None:
                preds = predict_fn(record)
            else:
                preds = list(labels)  # perfect predictions for analysis

            all_preds.extend(preds)
            all_labels.extend(labels)

            # Per-record F1
            rec_result = self._entity_f1.compute(preds, labels)
            per_record_f1s.append(rec_result.f1)

            # Group by language
            lang = record.language
            per_lang_groups.setdefault(lang, ([], []))
            per_lang_groups[lang][0].extend(preds)
            per_lang_groups[lang][1].extend(labels)

            # Group by difficulty
            diff = record.difficulty_level
            per_diff_groups.setdefault(diff, ([], []))
            per_diff_groups[diff][0].extend(preds)
            per_diff_groups[diff][1].extend(labels)

            # Group by data type
            dtype = record.data_type
            per_dtype_groups.setdefault(dtype, ([], []))
            per_dtype_groups[dtype][0].extend(preds)
            per_dtype_groups[dtype][1].extend(labels)

            # Group by context length
            ctx_len = record.context_length_tier
            per_ctx_groups.setdefault(ctx_len, ([], []))
            per_ctx_groups[ctx_len][0].extend(preds)
            per_ctx_groups[ctx_len][1].extend(labels)

        # Overall entity-level F1 with per-entity breakdown
        overall = self._entity_f1.compute(all_preds, all_labels)

        # Aggregation
        micro = self._aggregator.compute_micro_averaged(overall.per_entity_breakdown)
        macro = self._aggregator.compute_macro_averaged(overall.per_entity_breakdown)
        weighted = self._aggregator.compute_weighted_averaged(overall.per_entity_breakdown)

        # Per-language
        per_lang: dict[str, dict[str, float]] = {}
        for lang, (pred_group, label_group) in per_lang_groups.items():
            r = self._entity_f1.compute(pred_group, label_group)
            per_lang[lang] = {"precision": r.precision, "recall": r.recall, "f1": r.f1}

        # Per-difficulty
        per_diff: dict[str, dict[str, float]] = {}
        for diff_key, (pred_group, label_group) in per_diff_groups.items():
            r = self._entity_f1.compute(pred_group, label_group)
            per_diff[diff_key] = {"precision": r.precision, "recall": r.recall, "f1": r.f1}

        # Per-data-type
        per_dtype: dict[str, dict[str, float]] = {}
        for data_type_key, (pred_group, label_group) in per_dtype_groups.items():
            r = self._entity_f1.compute(pred_group, label_group)
            per_dtype[data_type_key] = {"precision": r.precision, "recall": r.recall, "f1": r.f1}

        # Per-context-length
        per_ctx: dict[str, dict[str, float]] = {}
        for context_tier, (pred_group, label_group) in per_ctx_groups.items():
            r = self._entity_f1.compute(pred_group, label_group)
            per_ctx[context_tier] = {"precision": r.precision, "recall": r.recall, "f1": r.f1}

        # Confidence interval
        ci = None
        if self.config.include_confidence_intervals and per_record_f1s:
            ci = self._aggregator.compute_confidence_intervals(per_record_f1s)

        # Fairness
        fairness_score = 0.0
        if self.config.include_fairness_metrics:
            ef = self._entity_fairness.compute(all_preds, all_labels)
            fairness_score = 1.0 - ef.value

        # Composite metric (optional, off by default)
        batch_composite: float | None = None
        if self.config.include_composite_metric:
            batch_comp_cfg = CompositeConfig(
                weight_privacy=0.15 if self.config.include_privacy_metrics else 0.0,
                weight_utility=0.10 if self.config.include_utility_metrics else 0.0,
                weight_fairness=0.05 if self.config.include_fairness_metrics else 0.0,
            )
            batch_comp = compute_composite(
                f1=overall.f1,
                precision=overall.precision,
                recall=overall.recall,
                latency_ms=0.0,
                docs_per_hour=0.0,
                privacy_score=0.0,
                utility_score=0.0,
                fairness_score=fairness_score,
                config=batch_comp_cfg,
            )
            batch_composite = batch_comp.score

        return BatchEvaluationReport(
            evaluation_id=eval_id,
            timestamp=ts,
            records_evaluated=len(records),
            micro_averaged=micro,
            macro_averaged=macro,
            weighted_averaged=weighted,
            per_entity_type=overall.per_entity_breakdown,
            per_language=per_lang,
            per_difficulty=per_diff,
            per_data_type=per_dtype,
            per_context_length=per_ctx,
            confidence_interval=ci,
            fairness_score=round(fairness_score, 6),
            composite_score=batch_composite,
        )

    # ── Comprehensive evaluation ───────────────────────────────────────

    def evaluate_comprehensive(
        self,
        records: list[EvalBenchmarkRecord],
        predictions_map: dict[str, list[LabeledSpan]],
        *,
        context_map: dict[str, dict[str, Any]] | None = None,
    ) -> ComprehensiveEvaluationReport:
        """Run all metrics: span, privacy, utility, and fairness.

        Args:
            records: Benchmark records with gold labels and dimension tags.
            predictions_map: Dict of record_id -> predicted spans.
            context_map: Optional dict of record_id -> context dict.

        Returns:
            ComprehensiveEvaluationReport with all metric families.
        """
        eval_id = str(uuid.uuid4())[:12]
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ctx_map = context_map or {}

        all_preds: list[LabeledSpan] = []
        all_labels: list[LabeledSpan] = []
        per_entity_f1s: dict[str, list[float]] = {}
        per_dimension_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
        per_record_f1s: list[float] = []

        for record in records:
            labels = [
                LabeledSpan(
                    entity_type=lbl["entity_type"],
                    start=lbl["start"],
                    end=lbl["end"],
                    record_id=record.record_id,
                )
                for lbl in record.labels
            ]
            preds = predictions_map.get(record.record_id, [])

            all_preds.extend(preds)
            all_labels.extend(labels)

            # Per-record F1
            rec_result = self._entity_f1.compute(preds, labels)
            per_record_f1s.append(rec_result.f1)

            # Per-entity-type F1 accumulation
            for entity_type, metrics in rec_result.per_entity_breakdown.items():
                per_entity_f1s.setdefault(entity_type, []).append(metrics.get("f1", 0.0))

            # Group by dimension if available
            if hasattr(record, "dimension_tag") and record.dimension_tag:
                dim = record.dimension_tag
                per_dimension_groups.setdefault(dim, ([], []))
                per_dimension_groups[dim][0].extend(preds)
                per_dimension_groups[dim][1].extend(labels)

        # Core span metrics
        overall = self._entity_f1.compute(all_preds, all_labels)
        micro = self._aggregator.compute_micro_averaged(overall.per_entity_breakdown)
        macro = self._aggregator.compute_macro_averaged(overall.per_entity_breakdown)
        weighted = self._aggregator.compute_weighted_averaged(overall.per_entity_breakdown)

        # Per-entity confidence intervals
        per_entity_ci: dict[str, dict[str, Any]] = {}
        if self.config.include_confidence_intervals:
            for entity_type, f1_scores in per_entity_f1s.items():
                if f1_scores:
                    ci = self._aggregator.compute_confidence_intervals(f1_scores)
                    per_entity_ci[entity_type] = {
                        "ci_lower": ci[0] if ci else None,
                        "ci_upper": ci[1] if ci else None,
                        "mean": sum(f1_scores) / len(f1_scores),
                    }

        # Privacy metrics
        privacy_score = 0.0
        leakage_score = 0.0
        if self.config.include_privacy_metrics:
            any_context = any(ctx_map.values())
            if any_context:
                reid_scores = []
                for record in records:
                    ctx = ctx_map.get(record.record_id, {})
                    if ctx.get("anonymized_text"):
                        labels = [
                            LabeledSpan(
                                entity_type=lbl["entity_type"],
                                start=lbl["start"],
                                end=lbl["end"],
                                record_id=record.record_id,
                            )
                            for lbl in record.labels
                        ]
                        preds = predictions_map.get(record.record_id, [])
                        reid = self._reid_risk.compute(preds, labels, context=ctx)
                        reid_scores.append(reid.value)
                if reid_scores:
                    privacy_score = 1.0 - (sum(reid_scores) / len(reid_scores))

            leakage_result = self._leakage.compute(all_preds, all_labels)
            leakage_score = 1.0 - leakage_result.value

        # Utility metrics
        utility_score = 0.0
        format_preservation = 0.0
        information_loss = 0.0
        if self.config.include_utility_metrics:
            info_loss = self._info_loss.compute(all_preds, all_labels)
            information_loss = info_loss.value
            utility_score = 1.0 - information_loss
            format_preservation = utility_score

        # Fairness metrics
        fairness_score = 0.0
        fairness_details: dict[str, Any] = {}
        if self.config.include_fairness_metrics:
            comprehensive_fair = self._comprehensive_fairness.compute(all_preds, all_labels)
            fairness_score = 1.0 - comprehensive_fair.value
            fairness_details = {
                "overall_gap": comprehensive_fair.value,
                "by_entity_type": getattr(comprehensive_fair, "per_entity_gap", {}),
            }

        # Per-dimension scores
        per_dimension: dict[str, dict[str, float]] = {}
        for dim, (pred_group, label_group) in per_dimension_groups.items():
            r = self._entity_f1.compute(pred_group, label_group)
            per_dimension[dim] = {
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
            }

        # Overall confidence interval
        overall_ci: tuple[float, float] | None = None
        if self.config.include_confidence_intervals and per_record_f1s:
            overall_ci = self._aggregator.compute_confidence_intervals(per_record_f1s)

        # Composite score
        composite_val: float | None = None
        if self.config.include_composite_metric:
            comp_cfg = CompositeConfig(
                weight_privacy=0.15 if self.config.include_privacy_metrics else 0.0,
                weight_utility=0.10 if self.config.include_utility_metrics else 0.0,
                weight_fairness=0.05 if self.config.include_fairness_metrics else 0.0,
            )
            comp = compute_composite(
                f1=overall.f1,
                precision=overall.precision,
                recall=overall.recall,
                latency_ms=0.0,
                docs_per_hour=0.0,
                privacy_score=privacy_score,
                utility_score=utility_score,
                fairness_score=fairness_score,
                config=comp_cfg,
            )
            composite_val = comp.score

        return ComprehensiveEvaluationReport(
            evaluation_id=eval_id,
            timestamp=ts,
            records_evaluated=len(records),
            micro_averaged=micro,
            macro_averaged=macro,
            weighted_averaged=weighted,
            per_entity_type=overall.per_entity_breakdown,
            privacy_score=round(privacy_score, 6),
            leakage_score=round(leakage_score, 6),
            utility_score=round(utility_score, 6),
            format_preservation=round(format_preservation, 6),
            information_loss=round(information_loss, 6),
            fairness_score=round(fairness_score, 6),
            fairness_details=fairness_details,
            per_dimension=per_dimension,
            composite_score=composite_val,
            floor_gates_passed=None,
            confidence_interval=overall_ci,
            per_entity_ci=per_entity_ci,
        )

    # ── Streaming evaluation ───────────────────────────────────────────

    def evaluate_streaming(
        self,
        window_sizes: list[int] | None = None,
        ewma_alpha: float = 0.05,
    ) -> StreamingPipeline:
        """Create a streaming evaluation pipeline.

        Args:
            window_sizes: Window sizes for sliding metrics (default: [10, 50, 100]).
            ewma_alpha: Exponential weighted moving average alpha (default: 0.05).

        Returns:
            StreamingPipeline instance configured with given parameters.
        """
        return StreamingPipeline(
            window_sizes=window_sizes or [10, 50, 100],
            ewma_alpha=ewma_alpha,
        )

    # ── Dimension-based evaluation ─────────────────────────────────────

    def evaluate_by_dimension(
        self,
        records: list[EvalBenchmarkRecord],
        predictions_map: dict[str, list[LabeledSpan]],
        dimension: str,
        *,
        context_map: dict[str, dict[str, Any]] | None = None,
    ) -> ComprehensiveEvaluationReport:
        """Evaluate records filtered by a specific dimension tag.

        Args:
            records: Benchmark records with dimension_tag attributes.
            predictions_map: Dict of record_id -> predicted spans.
            dimension: The dimension tag value to filter by.
            context_map: Optional dict of record_id -> context dict.

        Returns:
            ComprehensiveEvaluationReport for filtered subset.
        """
        filtered = [
            r for r in records
            if hasattr(r, "dimension_tag") and r.dimension_tag == dimension
        ]
        return self.evaluate_comprehensive(
            filtered, predictions_map, context_map=context_map
        )

    # ── Convenience methods ────────────────────────────────────────────

    def evaluate_by_language(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        languages: list[str] | None = None,
        record_languages: dict[str, str] | None = None,
    ) -> dict[str, EvaluationReport]:
        """Evaluate independently for each language."""
        rl = record_languages or {}
        all_langs = languages or sorted({rl.get(s.record_id, "en") for s in labels})
        results: dict[str, EvaluationReport] = {}
        for lang in all_langs:
            pf = [s for s in predictions if rl.get(s.record_id, "en") == lang]
            lf = [s for s in labels if rl.get(s.record_id, "en") == lang]
            results[lang] = self.evaluate(pf, lf, language=lang)
        return results

    def validate_compliance(
        self,
        entity_types: list[str],
        standard: str = "nist",
        *,
        profile: str = "general",
    ) -> ComplianceReport:
        """Validate entity-type coverage against a regulatory standard.

        Args:
            entity_types: List of entity types to validate.
            standard: Regulatory standard ("nist", "gdpr", etc.).
            profile: Compliance profile ("general", "healthcare", "financial").

        Returns:
            ComplianceReport with validation results.
        """
        # Profile parameter reserved for future profile-specific validation.
        # The underlying ComplianceValidator currently validates against a single standard.
        return self._compliance.validate(entity_types, standard=standard)
