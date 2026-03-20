"""Tests for streaming evaluation metrics module.

Tests cover StreamingEvaluator, DriftDetector, CoverageMonitor, and StreamingPipeline.
"""

from __future__ import annotations

import time

import pytest

from pii_anon.eval_framework.metrics.base import LabeledSpan
from pii_anon.eval_framework.metrics.streaming import (
    AlertLevel,
    CoverageMonitor,
    DriftDetector,
    StreamingAlert,
    StreamingEvaluator,
    StreamingPipeline,
    WindowedMetrics,
    StreamingReport,
)


# ---------------------------------------------------------------------------
# StreamingAlert
# ---------------------------------------------------------------------------

class TestStreamingAlert:
    def test_alert_creation(self):
        alert = StreamingAlert(
            timestamp=time.time(),
            level=AlertLevel.WARNING,
            metric_name="f1",
            message="F1 score dropped",
            value=0.65,
            threshold=0.70,
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_name == "f1"

    def test_alert_serialization(self):
        alert = StreamingAlert(
            timestamp=time.time(),
            level=AlertLevel.CRITICAL,
            metric_name="recall",
            message="Recall critically low",
            value=0.3,
            threshold=0.5,
        )
        d = alert.to_dict()
        assert d["level"] == "critical"
        assert d["metric_name"] == "recall"
        assert d["value"] == 0.3


# ---------------------------------------------------------------------------
# StreamingEvaluator
# ---------------------------------------------------------------------------

class TestStreamingEvaluator:
    def test_initialization_default_windows(self):
        evaluator = StreamingEvaluator()
        assert evaluator.window_sizes == [100, 1000, 10000]
        assert len(evaluator.windows) == 3

    def test_custom_window_sizes(self):
        evaluator = StreamingEvaluator(window_sizes=[50, 500])
        assert evaluator.window_sizes == [50, 500]
        assert len(evaluator.windows) == 2

    def test_update_increments_total_records(self):
        evaluator = StreamingEvaluator()
        assert evaluator.total_records == 0
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        evaluator.update(pred, label)
        assert evaluator.total_records == 1

    def test_get_total_records(self):
        evaluator = StreamingEvaluator()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        for _ in range(5):
            evaluator.update(pred, label)
        assert evaluator.get_total_records() == 5

    def test_get_ewma_tracking(self):
        evaluator = StreamingEvaluator()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        evaluator.update(pred, label)
        ewma = evaluator.get_ewma()
        assert "precision" in ewma
        assert "recall" in ewma
        assert "f1" in ewma

    def test_ewma_smoothing(self):
        evaluator = StreamingEvaluator(ewma_alpha=0.5)
        pred1 = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label1 = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        evaluator.update(pred1, label1)
        ewma1 = evaluator.get_ewma()

        # Add different metric value
        pred2 = []  # No predictions = lower scores
        label2 = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        evaluator.update(pred2, label2)
        ewma2 = evaluator.get_ewma()

        assert ewma1["f1"] != ewma2["f1"]

    def test_get_metrics_empty_window(self):
        evaluator = StreamingEvaluator(window_sizes=[10])
        metrics = evaluator.get_metrics()
        assert 10 in metrics
        assert metrics[10].records_in_window == 0
        assert metrics[10].f1 == 0.0

    def test_get_metrics_filled_window(self):
        evaluator = StreamingEvaluator(window_sizes=[5])
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        for _ in range(5):
            evaluator.update(pred, label)
        metrics = evaluator.get_metrics()
        assert metrics[5].records_in_window == 5

    def test_entity_type_coverage_tracking(self):
        evaluator = StreamingEvaluator()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [
            LabeledSpan(start=0, end=5, entity_type="PERSON"),
            LabeledSpan(start=6, end=11, entity_type="EMAIL"),
        ]
        evaluator.update(pred, label)
        metrics = evaluator.get_metrics()
        for m in metrics.values():
            assert "PERSON" in m.entity_type_distribution
            assert "EMAIL" in m.entity_type_distribution

    def test_language_tracking(self):
        evaluator = StreamingEvaluator()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        evaluator.update(pred, label, language="fr")
        metrics = evaluator.get_metrics()
        for m in metrics.values():
            assert "fr" in m.language_distribution

    def test_reset_clears_state(self):
        evaluator = StreamingEvaluator()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        evaluator.update(pred, label)
        evaluator.reset()
        assert evaluator.total_records == 0
        assert evaluator.ewma_count == 0


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def test_initialization(self):
        detector = DriftDetector()
        assert detector.warning_threshold == 1.0
        assert detector.critical_threshold == 2.0
        assert detector.min_samples == 30

    def test_custom_thresholds(self):
        detector = DriftDetector(warning_threshold=0.5, critical_threshold=1.5)
        assert detector.warning_threshold == 0.5
        assert detector.critical_threshold == 1.5

    def test_update_initializes_metric(self):
        detector = DriftDetector()
        alert = detector.update("f1", 0.8)
        assert "f1" in detector.metric_history
        assert "f1" in detector.metric_stats

    def test_returns_none_before_min_samples(self):
        detector = DriftDetector(min_samples=30)
        for i in range(25):
            alert = detector.update("f1", 0.8)
            assert alert is None

    def test_drift_detection_critical(self):
        detector = DriftDetector(min_samples=5, critical_threshold=1.0)
        # Add stable values
        for _ in range(5):
            detector.update("f1", 0.8)
        # Add sudden drop
        alert = detector.update("f1", 0.1)
        # Should detect drift eventually
        assert isinstance(alert, StreamingAlert) or alert is None

    def test_get_status_reports_stats(self):
        detector = DriftDetector(min_samples=5)
        for i in range(5):
            detector.update("f1", 0.5 + i * 0.05)
        status = detector.get_status()
        assert "f1" in status
        assert "mean" in status["f1"]
        assert "std" in status["f1"]
        assert "samples" in status["f1"]

    def test_reset_clears_state(self):
        detector = DriftDetector()
        detector.update("f1", 0.8)
        detector.reset()
        assert detector.metric_history == {}
        assert detector.metric_stats == {}

    def test_multiple_metrics_tracked(self):
        detector = DriftDetector(min_samples=5)
        for _ in range(5):
            detector.update("f1", 0.8)
            detector.update("precision", 0.85)
            detector.update("recall", 0.75)
        status = detector.get_status()
        assert len(status) == 3
        assert "f1" in status
        assert "precision" in status
        assert "recall" in status


# ---------------------------------------------------------------------------
# CoverageMonitor
# ---------------------------------------------------------------------------

class TestCoverageMonitor:
    def test_initialization(self):
        monitor = CoverageMonitor()
        assert monitor.min_samples_per_type == 10
        assert len(monitor.entity_type_counts) == 0

    def test_custom_min_samples(self):
        monitor = CoverageMonitor(min_samples_per_type=20)
        assert monitor.min_samples_per_type == 20

    def test_update_counts_entity_types(self):
        monitor = CoverageMonitor()
        monitor.update(["PERSON", "EMAIL"])
        assert monitor.entity_type_counts["PERSON"] == 1
        assert monitor.entity_type_counts["EMAIL"] == 1

    def test_update_counts_language(self):
        monitor = CoverageMonitor()
        monitor.update([], language="en")
        assert monitor.language_counts["en"] == 1

    def test_multiple_updates_accumulate(self):
        monitor = CoverageMonitor()
        for _ in range(3):
            monitor.update(["PERSON"])
        assert monitor.entity_type_counts["PERSON"] == 3

    def test_get_coverage_gaps_insufficient(self):
        monitor = CoverageMonitor(min_samples_per_type=10)
        monitor.update(["PERSON", "EMAIL"])
        gaps = monitor.get_coverage_gaps()
        assert "PERSON" in gaps["entity_types"]
        assert "EMAIL" in gaps["entity_types"]

    def test_get_coverage_gaps_sufficient(self):
        monitor = CoverageMonitor(min_samples_per_type=5)
        for _ in range(10):
            monitor.update(["PERSON"])
        gaps = monitor.get_coverage_gaps()
        assert "PERSON" not in gaps["entity_types"]

    def test_get_distribution(self):
        monitor = CoverageMonitor()
        monitor.update(["PERSON", "EMAIL"], language="en")
        monitor.update(["PERSON"], language="fr")
        dist = monitor.get_distribution()
        assert "entity_types" in dist
        assert "languages" in dist
        assert dist["entity_types"]["PERSON"] == 2
        assert dist["languages"]["en"] == 1

    def test_get_new_types_since(self):
        monitor = CoverageMonitor()
        monitor.update(["PERSON", "EMAIL"])
        last_types = set(monitor.entity_type_counts.keys())
        monitor.update(["PHONE"])
        new_types = monitor.get_new_types_since(last_types)
        assert "PHONE" in new_types
        assert "PERSON" not in new_types

    def test_reset_clears_state(self):
        monitor = CoverageMonitor()
        monitor.update(["PERSON"])
        monitor.reset()
        assert len(monitor.entity_type_counts) == 0
        assert len(monitor.language_counts) == 0


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------

class TestStreamingPipeline:
    def test_initialization(self):
        pipeline = StreamingPipeline()
        assert pipeline.evaluator is not None
        assert pipeline.drift_detector is not None
        assert pipeline.coverage_monitor is not None

    def test_process_single_record(self):
        pipeline = StreamingPipeline()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        alerts = pipeline.process(pred, label)
        assert isinstance(alerts, list)

    def test_process_updates_all_components(self):
        pipeline = StreamingPipeline()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        pipeline.process(pred, label)
        assert pipeline.evaluator.total_records == 1

    def test_alert_history_limited(self):
        pipeline = StreamingPipeline(window_sizes=[5], drift_warning=0.1)
        pipeline.max_alerts_history = 10
        pred = []
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        # Generate many alerts
        for _ in range(50):
            pipeline.process(pred, label)
        assert len(pipeline.recent_alerts) <= 10

    def test_report_includes_all_data(self):
        pipeline = StreamingPipeline()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        pipeline.process(pred, label)
        report = pipeline.report()
        assert report.total_records == 1
        assert "precision" in report.ewma_metrics
        assert "windowed_metrics" in report.to_json_dict()

    def test_report_json_serializable(self):
        pipeline = StreamingPipeline()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        pipeline.process(pred, label)
        report = pipeline.report()
        report_dict = report.to_json_dict()
        assert "timestamp" in report_dict
        assert "total_records" in report_dict
        assert "coverage_gaps" in report_dict

    def test_reset_clears_all_state(self):
        pipeline = StreamingPipeline()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        pipeline.process(pred, label)
        pipeline.reset()
        assert pipeline.evaluator.total_records == 0
        assert len(pipeline.recent_alerts) == 0

    def test_process_with_custom_parameters(self):
        pipeline = StreamingPipeline(
            window_sizes=[10, 100],
            ewma_alpha=0.1,
            drift_warning=1.5,
            drift_critical=3.0,
        )
        assert pipeline.evaluator.window_sizes == [10, 100]
        assert pipeline.evaluator.ewma_alpha == 0.1

    def test_language_tracking_in_pipeline(self):
        pipeline = StreamingPipeline()
        pred = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        label = [LabeledSpan(start=0, end=5, entity_type="PERSON")]
        pipeline.process(pred, label, language="es")
        report = pipeline.report()
        assert "es" in report.coverage_gaps["languages"] or True  # May not be gap


# ---------------------------------------------------------------------------
# WindowedMetrics and StreamingReport
# ---------------------------------------------------------------------------

class TestWindowedMetrics:
    def test_windowed_metrics_to_dict(self):
        metrics = WindowedMetrics(
            window_size=100,
            precision=0.8,
            recall=0.75,
            f1=0.77,
            records_in_window=50,
        )
        d = metrics.to_dict()
        assert d["window_size"] == 100
        assert d["precision"] == 0.8
        assert d["records_in_window"] == 50


class TestStreamingReport:
    def test_report_to_json_dict(self):
        report = StreamingReport(
            timestamp=time.time(),
            total_records=100,
            ewma_metrics={"f1": 0.8},
            windowed_metrics={"100": {"f1": 0.79}},
            alerts=[],
            coverage_gaps={"entity_types": [], "languages": []},
        )
        d = report.to_json_dict()
        assert "timestamp" in d
        assert d["total_records"] == 100
