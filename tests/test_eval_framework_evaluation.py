"""Tests for pii_anon.eval_framework.evaluation module.

Validates:
- EvaluationFramework core evaluation pipeline
- Batch evaluation with aggregation
- Context evaluator
- Report generation (JSON, Markdown, CSV)
- MetricAggregator (micro, macro, weighted, CI)
"""

from __future__ import annotations

import json


from pii_anon.eval_framework.evaluation.framework import (
    EvaluationFramework,
    EvaluationFrameworkConfig,
    EvaluationReport,
)
from pii_anon.eval_framework.evaluation.aggregation import MetricAggregator
from pii_anon.eval_framework.evaluation.reporting import ReportGenerator
from pii_anon.eval_framework.metrics.base import (
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(etype: str, start: int, end: int, rid: str = "r1") -> LabeledSpan:
    return LabeledSpan(entity_type=etype, start=start, end=end, record_id=rid)


# ---------------------------------------------------------------------------
# EvaluationFramework
# ---------------------------------------------------------------------------

class TestEvaluationFramework:
    """Core evaluation framework tests."""

    def setup_method(self) -> None:
        self.fw = EvaluationFramework()

    def test_evaluate_perfect_predictions(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10), _span("EMAIL_ADDRESS", 15, 30)]
        labels = list(preds)
        report = self.fw.evaluate(preds, labels, language="en")
        assert isinstance(report, EvaluationReport)
        assert report.f1 == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.language == "en"
        assert report.records_evaluated == 1

    def test_evaluate_zero_recall(self) -> None:
        preds: list[LabeledSpan] = []
        labels = [_span("PERSON_NAME", 0, 10)]
        report = self.fw.evaluate(preds, labels)
        assert report.recall == 0.0
        assert report.f1 == 0.0

    def test_evaluate_zero_precision(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels: list[LabeledSpan] = []
        report = self.fw.evaluate(preds, labels)
        assert report.precision == 0.0

    def test_evaluate_with_match_modes(self) -> None:
        config = EvaluationFrameworkConfig(
            match_modes=[MatchMode.STRICT, MatchMode.PARTIAL, MatchMode.EXACT, MatchMode.TYPE],
        )
        fw = EvaluationFramework(config)
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        report = fw.evaluate(preds, labels)
        assert "strict" in report.metrics_by_match_mode
        assert "partial" in report.metrics_by_match_mode

    def test_evaluate_with_privacy_context(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        ctx = {
            "anonymized_text": "[REDACTED] is a good person",
            "original_texts": ["John Smith"],
        }
        report = self.fw.evaluate(preds, labels, context=ctx)
        assert report.privacy_score >= 0.0

    def test_evaluate_includes_per_entity_breakdown(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10), _span("EMAIL_ADDRESS", 15, 30)]
        report = self.fw.evaluate(preds, labels)
        assert "PERSON_NAME" in report.per_entity_breakdown

    def test_custom_config(self) -> None:
        config = EvaluationFrameworkConfig(
            languages=["en", "es"],
            evaluation_levels=[EvaluationLevel.ENTITY, EvaluationLevel.TOKEN],
            include_privacy_metrics=False,
            include_fairness_metrics=False,
        )
        fw = EvaluationFramework(config)
        report = fw.evaluate(
            [_span("PERSON_NAME", 0, 10)],
            [_span("PERSON_NAME", 0, 10)],
        )
        assert report.privacy_score == 0.0
        assert report.fairness_score == 0.0


class TestEvaluateByLanguage:
    """Per-language evaluation."""

    def test_evaluate_by_language(self) -> None:
        fw = EvaluationFramework()
        preds = [
            _span("PERSON_NAME", 0, 10, "en1"),
            _span("PERSON_NAME", 0, 10, "es1"),
        ]
        labels = list(preds)
        record_languages = {"en1": "en", "es1": "es"}
        results = fw.evaluate_by_language(
            preds, labels,
            languages=["en", "es"],
            record_languages=record_languages,
        )
        assert "en" in results
        assert "es" in results
        assert results["en"].f1 == 1.0
        assert results["es"].f1 == 1.0


class TestComplianceValidation:
    """Framework-level compliance validation."""

    def test_validate_compliance(self) -> None:
        fw = EvaluationFramework()
        report = fw.validate_compliance(
            ["PERSON_NAME", "EMAIL_ADDRESS", "US_SSN"],
            standard="nist",
        )
        assert report.compliant is False  # only 3 of 16 NIST types


# ---------------------------------------------------------------------------
# MetricAggregator
# ---------------------------------------------------------------------------

class TestMetricAggregator:
    """Aggregation methods: micro, macro, weighted."""

    def setup_method(self) -> None:
        self.agg = MetricAggregator()
        self.breakdown = {
            "PERSON_NAME": {"precision": 0.8, "recall": 0.9, "f1": 0.849, "support": 100},
            "EMAIL_ADDRESS": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 50},
        }

    def test_micro_averaged(self) -> None:
        result = self.agg.compute_micro_averaged(self.breakdown)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert 0.0 <= result["f1"] <= 1.0

    def test_macro_averaged(self) -> None:
        result = self.agg.compute_macro_averaged(self.breakdown)
        # Macro = simple average across types
        assert abs(result["precision"] - 0.9) < 0.01
        assert abs(result["recall"] - 0.95) < 0.01

    def test_weighted_averaged(self) -> None:
        result = self.agg.compute_weighted_averaged(self.breakdown)
        assert "f1" in result
        # Weighted by support: PERSON_NAME has 2x the weight
        assert result["f1"] > 0.8

    def test_confidence_intervals(self) -> None:
        scores = [0.85, 0.90, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89, 0.93, 0.84]
        ci = self.agg.compute_confidence_intervals(scores)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        low, high = ci
        assert low <= high
        assert low > 0.0
        assert high <= 1.0

    def test_micro_f1_ci_contains_micro_point_estimate(self) -> None:
        # 1 large imperfect record + 99 tiny perfect records: the macro mean of
        # per-record F1 is ~0.995, but the micro F1 = 2*Sum(tp)/(Sum(pred)+Sum(label))
        # is ~0.749. A CI bootstrapped on per-record F1 (the OLD behaviour)
        # EXCLUDES the micro point; the micro-F1 CI must CONTAIN it.
        counts = [(50, 100, 100)] + [(1, 1, 1)] * 99
        micro_point = 2 * sum(c[0] for c in counts) / (
            sum(c[1] for c in counts) + sum(c[2] for c in counts)
        )
        per_record_f1 = [
            2.0 * tp / (npred + nlab) for tp, npred, nlab in counts
        ]

        micro_lo, micro_hi = self.agg.compute_micro_f1_confidence_interval(
            counts, n_bootstrap=2000,
        )
        macro_lo, macro_hi = self.agg.compute_confidence_intervals(
            per_record_f1, n_bootstrap=2000,
        )

        # The fix: the micro point sits INSIDE the micro CI.
        assert micro_lo <= micro_point <= micro_hi
        # The regression being prevented: the macro CI EXCLUDES the micro point.
        assert not (macro_lo <= micro_point <= macro_hi)

    def test_micro_f1_ci_empty_returns_zero(self) -> None:
        assert self.agg.compute_micro_f1_confidence_interval([]) == (0.0, 0.0)

    def test_micro_f1_ci_is_deterministic(self) -> None:
        counts = [(8, 10, 10), (1, 2, 1), (3, 3, 5), (10, 12, 11)]
        first = self.agg.compute_micro_f1_confidence_interval(
            counts, seed=7, n_bootstrap=500,
        )
        second = self.agg.compute_micro_f1_confidence_interval(
            counts, seed=7, n_bootstrap=500,
        )
        assert first == second


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    """Report generation in multiple formats."""

    def setup_method(self) -> None:
        fw = EvaluationFramework()
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        self.report = fw.evaluate(preds, labels)

    def test_to_json(self) -> None:
        json_str = ReportGenerator.to_json(self.report)
        parsed = json.loads(json_str)
        assert "evaluation_id" in parsed
        assert "f1" in parsed

    def test_to_markdown(self) -> None:
        md = ReportGenerator.to_markdown(self.report)
        assert "# PII Evaluation Report" in md
        assert "F1" in md

    def test_to_csv(self) -> None:
        csv_str = ReportGenerator.to_csv(self.report)
        assert "metric" in csv_str.lower() or "f1" in csv_str.lower()
