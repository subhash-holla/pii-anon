"""Real-time streaming evaluation for PII anonymization systems.

Provides windowed metrics, drift detection, and coverage monitoring
for deployment-time performance tracking without requiring the full
benchmark dataset.

Evidence basis:
- Page (1954): Sequential analysis via Page-Hinkley test
- Bifet & Gavalda (2007): ADWIN adaptive windowing for concept drift
- Gama et al. (2014): Survey on concept drift adaptation
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import (
    LabeledSpan,
    MatchMode,
)
from .span_metrics import _aligned_prf


class AlertLevel(str, Enum):
    """Severity levels for streaming alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class StreamingAlert:
    """Alert generated during streaming evaluation.

    Attributes:
        timestamp: Unix timestamp when alert was generated
        level: AlertLevel indicating severity
        metric_name: Name of the metric that triggered the alert
        message: Human-readable alert message
        value: Current metric value
        threshold: Threshold that was exceeded
    """

    timestamp: float
    level: AlertLevel
    metric_name: str
    message: str
    value: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "metric_name": self.metric_name,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class WindowedMetrics:
    """Metrics computed over a rolling window.

    Attributes:
        window_size: Size of the rolling window in records
        precision: Precision over the window
        recall: Recall over the window
        f1: F1 score over the window
        records_in_window: Number of records currently in the window
        entity_type_distribution: Count of entity types seen in window
        language_distribution: Count of languages seen in window
    """

    window_size: int
    precision: float
    recall: float
    f1: float
    records_in_window: int
    entity_type_distribution: dict[str, int] = field(default_factory=dict)
    language_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "window_size": self.window_size,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "records_in_window": self.records_in_window,
            "entity_type_distribution": dict(self.entity_type_distribution),
            "language_distribution": dict(self.language_distribution),
        }


@dataclass
class StreamingReport:
    """Comprehensive snapshot of streaming evaluation state.

    Attributes:
        timestamp: Unix timestamp of the report
        total_records: Total records processed since start/reset
        ewma_metrics: Current EWMA values for precision, recall, f1
        windowed_metrics: WindowedMetrics for each window size
        alerts: Recent alerts generated
        coverage_gaps: Entity types and languages with insufficient coverage
    """

    timestamp: float
    total_records: int
    ewma_metrics: dict[str, float]
    windowed_metrics: dict[str, Any]
    alerts: list[dict[str, Any]]
    coverage_gaps: dict[str, list[str]]

    def to_json_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_records": self.total_records,
            "ewma_metrics": self.ewma_metrics,
            "windowed_metrics": self.windowed_metrics,
            "alerts": self.alerts,
            "coverage_gaps": self.coverage_gaps,
        }


class StreamingEvaluator:
    """Maintains rolling window metrics for real-time evaluation.

    Uses exponential weighted moving average (EWMA) for smooth online
    estimation of precision, recall, and F1 without storing the full
    dataset.

    Attributes:
        window_sizes: List of window sizes to maintain
        ewma_alpha: Smoothing parameter for EWMA (0-1, default 0.05)
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        ewma_alpha: float = 0.05,
    ) -> None:
        """Initialize streaming evaluator.

        Args:
            window_sizes: Window sizes to maintain rolling metrics for.
                         Defaults to [100, 1000, 10000].
            ewma_alpha: Smoothing parameter for EWMA. Lower values give
                       more weight to historical data.
        """
        self.window_sizes = window_sizes or [100, 1000, 10000]
        self.ewma_alpha = ewma_alpha

        # Rolling windows (deques) for each window size
        self.windows: dict[int, deque[tuple[float, float, float]]] = {
            ws: deque(maxlen=ws) for ws in self.window_sizes
        }

        # EWMA tracking: (count, precision_ema, recall_ema, f1_ema)
        self.ewma_count = 0
        self.ewma_precision = 0.0
        self.ewma_recall = 0.0
        self.ewma_f1 = 0.0

        # Coverage tracking
        self.entity_type_counts: dict[str, int] = {}
        self.language_counts: dict[str, int] = {}
        self.total_records = 0

    def update(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        record_id: str = "",
        language: str = "en",
    ) -> None:
        """Update metrics with new record evaluation.

        Computes precision, recall, and F1 for this record and updates
        all rolling windows and EWMA trackers in O(1) amortized time.

        Args:
            predictions: Predicted spans for this record
            labels: Ground truth spans for this record
            record_id: Optional identifier for this record
            language: Language code for this record (default "en")
        """
        # Compute metrics for this record using aligned PRF
        precision, recall, f1, _tp, _fp, _fn = _aligned_prf(
            predictions, labels, MatchMode.EXACT,
        )

        # Update EWMA values
        if self.ewma_count == 0:
            self.ewma_precision = precision
            self.ewma_recall = recall
            self.ewma_f1 = f1
        else:
            self.ewma_precision = (
                self.ewma_alpha * precision
                + (1 - self.ewma_alpha) * self.ewma_precision
            )
            self.ewma_recall = (
                self.ewma_alpha * recall + (1 - self.ewma_alpha) * self.ewma_recall
            )
            self.ewma_f1 = self.ewma_alpha * f1 + (1 - self.ewma_alpha) * self.ewma_f1

        self.ewma_count += 1

        # Update rolling windows
        prf_tuple = (precision, recall, f1)
        for window_size in self.window_sizes:
            self.windows[window_size].append(prf_tuple)

        # Track coverage
        for span in labels:
            entity_type = span.entity_type
            self.entity_type_counts[entity_type] = (
                self.entity_type_counts.get(entity_type, 0) + 1
            )

        self.language_counts[language] = self.language_counts.get(language, 0) + 1
        self.total_records += 1

    def get_metrics(self) -> dict[int, WindowedMetrics]:
        """Get metrics for each window size.

        Returns:
            Dictionary mapping window_size to WindowedMetrics
        """
        metrics = {}

        for window_size in self.window_sizes:
            window = self.windows[window_size]

            if not window:
                metrics[window_size] = WindowedMetrics(
                    window_size=window_size,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    records_in_window=0,
                    entity_type_distribution={},
                    language_distribution={},
                )
                continue

            # Compute mean metrics over window
            precisions = [p for p, r, f in window]
            recalls = [r for p, r, f in window]
            f1s = [f for p, r, f in window]

            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            avg_f1 = sum(f1s) / len(f1s)

            metrics[window_size] = WindowedMetrics(
                window_size=window_size,
                precision=avg_precision,
                recall=avg_recall,
                f1=avg_f1,
                records_in_window=len(window),
                entity_type_distribution=dict(self.entity_type_counts),
                language_distribution=dict(self.language_counts),
            )

        return metrics

    def get_ewma(self) -> dict[str, float]:
        """Get current EWMA values.

        Returns:
            Dictionary with keys 'precision', 'recall', 'f1'
        """
        return {
            "precision": self.ewma_precision,
            "recall": self.ewma_recall,
            "f1": self.ewma_f1,
        }

    def get_total_records(self) -> int:
        """Get total number of records processed."""
        return self.total_records

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        for window_size in self.window_sizes:
            self.windows[window_size].clear()

        self.ewma_count = 0
        self.ewma_precision = 0.0
        self.ewma_recall = 0.0
        self.ewma_f1 = 0.0

        self.entity_type_counts.clear()
        self.language_counts.clear()
        self.total_records = 0


class DriftDetector:
    """Detects distributional shift in metrics using Page-Hinkley test.

    The Page-Hinkley test is a sequential hypothesis test for detecting
    changes in the mean of a data stream. It maintains a cumulative sum
    of deviations from the running mean and flags when the difference
    between maximum and minimum cumulative sum exceeds a threshold.

    References:
        Page (1954): Continuous inspection schemes. Biometrika 41(1-2)
    """

    def __init__(
        self,
        warning_threshold: float = 1.0,
        critical_threshold: float = 2.0,
        min_samples: int = 30,
    ) -> None:
        """Initialize drift detector.

        Args:
            warning_threshold: Multiplier of std dev for WARNING alert
            critical_threshold: Multiplier of std dev for CRITICAL alert
            min_samples: Minimum samples before checking for drift
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples

        # Per-metric tracking: {metric_name: (values, sum_of_values, sum_of_squares)}
        self.metric_history: dict[str, list[float]] = {}
        self.metric_stats: dict[str, dict[str, Any]] = {}

    def update(self, metric_name: str, value: float) -> StreamingAlert | None:
        """Update drift detection with new metric value.

        Args:
            metric_name: Name of the metric being tracked
            value: New metric value

        Returns:
            StreamingAlert if drift is detected, None otherwise
        """
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
            self.metric_stats[metric_name] = {
                "mean": 0.0,
                "std": 0.0,
                "min_cumsum": 0.0,
                "max_cumsum": 0.0,
                "cumsum": 0.0,
                "samples": 0,
                "drift_detected": False,
            }

        history = self.metric_history[metric_name]
        stats = self.metric_stats[metric_name]

        history.append(value)
        stats["samples"] += 1

        # Compute running mean and std
        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n
        std = math.sqrt(variance) if variance > 0 else 0.0

        stats["mean"] = mean
        stats["std"] = std

        # Page-Hinkley test
        # Compute cumulative sum of deviations from running mean
        delta = 0.0  # drift magnitude parameter
        cumsum = 0.0
        min_cumsum = 0.0
        max_cumsum = 0.0

        for x in history:
            cumsum += x - mean - delta
            min_cumsum = min(min_cumsum, cumsum)
            max_cumsum = max(max_cumsum, cumsum)

        stats["cumsum"] = cumsum
        stats["min_cumsum"] = min_cumsum
        stats["max_cumsum"] = max_cumsum

        # Check for drift only after minimum samples
        if n < self.min_samples:
            return None

        # Drift magnitude is the difference between max and min cumsum
        drift_magnitude = max_cumsum - min_cumsum

        # Normalize by std to get standardized drift
        normalized_drift = drift_magnitude / std if std > 0 else 0.0

        # Check thresholds
        alert = None
        if normalized_drift > self.critical_threshold * std:
            stats["drift_detected"] = True
            alert = StreamingAlert(
                timestamp=time.time(),
                level=AlertLevel.CRITICAL,
                metric_name=metric_name,
                message=f"CRITICAL: Drift detected in {metric_name}. "
                f"Drift magnitude: {drift_magnitude:.4f}, "
                f"Mean: {mean:.4f}, Std: {std:.4f}",
                value=normalized_drift,
                threshold=self.critical_threshold * std,
            )
        elif normalized_drift > self.warning_threshold * std:
            alert = StreamingAlert(
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                metric_name=metric_name,
                message=f"WARNING: Potential drift in {metric_name}. "
                f"Drift magnitude: {drift_magnitude:.4f}, "
                f"Mean: {mean:.4f}, Std: {std:.4f}",
                value=normalized_drift,
                threshold=self.warning_threshold * std,
            )

        return alert

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get current drift detection status for all metrics.

        Returns:
            Dictionary mapping metric_name to status dict containing:
            - mean: Running mean
            - std: Standard deviation
            - samples: Number of samples seen
            - drift_detected: Whether critical drift was detected
        """
        return {
            name: {
                "mean": stats["mean"],
                "std": stats["std"],
                "samples": stats["samples"],
                "drift_detected": stats["drift_detected"],
            }
            for name, stats in self.metric_stats.items()
        }

    def reset(self) -> None:
        """Reset all drift detection state."""
        self.metric_history.clear()
        self.metric_stats.clear()


class CoverageMonitor:
    """Tracks entity type and language coverage during streaming.

    Detects when certain entity types or languages have insufficient
    sample coverage for reliable evaluation.
    """

    def __init__(self, min_samples_per_type: int = 10) -> None:
        """Initialize coverage monitor.

        Args:
            min_samples_per_type: Minimum samples required per type/language
        """
        self.min_samples_per_type = min_samples_per_type
        self.entity_type_counts: dict[str, int] = {}
        self.language_counts: dict[str, int] = {}

    def update(
        self,
        entity_types: list[str],
        language: str = "en",
    ) -> None:
        """Update coverage with new record data.

        Args:
            entity_types: List of entity types found in this record
            language: Language code for this record
        """
        for entity_type in entity_types:
            self.entity_type_counts[entity_type] = (
                self.entity_type_counts.get(entity_type, 0) + 1
            )

        self.language_counts[language] = self.language_counts.get(language, 0) + 1

    def get_coverage_gaps(self) -> dict[str, list[str]]:
        """Identify entity types and languages with insufficient coverage.

        Returns:
            Dictionary with keys 'entity_types' and 'languages', each
            containing list of types/languages below min_samples_per_type
        """
        entity_gaps = [
            entity_type
            for entity_type, count in self.entity_type_counts.items()
            if count < self.min_samples_per_type
        ]

        language_gaps = [
            language
            for language, count in self.language_counts.items()
            if count < self.min_samples_per_type
        ]

        return {
            "entity_types": sorted(entity_gaps),
            "languages": sorted(language_gaps),
        }

    def get_distribution(self) -> dict[str, dict[str, int]]:
        """Get current entity type and language distributions.

        Returns:
            Dictionary with keys 'entity_types' and 'languages',
            each containing count dict
        """
        return {
            "entity_types": dict(self.entity_type_counts),
            "languages": dict(self.language_counts),
        }

    def get_new_types_since(self, last_check: set[str]) -> set[str]:
        """Get entity types seen since last check.

        Args:
            last_check: Set of entity types from previous check

        Returns:
            Set of new entity types not in last_check
        """
        current_types = set(self.entity_type_counts.keys())
        return current_types - last_check

    def reset(self) -> None:
        """Reset all coverage tracking."""
        self.entity_type_counts.clear()
        self.language_counts.clear()


class StreamingPipeline:
    """Integrated pipeline for real-time streaming evaluation.

    Combines StreamingEvaluator, DriftDetector, and CoverageMonitor
    for comprehensive deployment-time monitoring.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        ewma_alpha: float = 0.05,
        drift_warning: float = 1.0,
        drift_critical: float = 2.0,
    ) -> None:
        """Initialize streaming pipeline.

        Args:
            window_sizes: Window sizes for rolling metrics. Defaults to
                         [100, 1000, 10000].
            ewma_alpha: Smoothing parameter for EWMA (0-1).
            drift_warning: Threshold multiplier for drift warnings.
            drift_critical: Threshold multiplier for critical drift alerts.
        """
        self.evaluator = StreamingEvaluator(
            window_sizes=window_sizes,
            ewma_alpha=ewma_alpha,
        )
        self.drift_detector = DriftDetector(
            warning_threshold=drift_warning,
            critical_threshold=drift_critical,
        )
        self.coverage_monitor = CoverageMonitor()

        self.recent_alerts: list[StreamingAlert] = []
        self.max_alerts_history = 100

    def process(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        record_id: str = "",
        language: str = "en",
    ) -> list[StreamingAlert]:
        """Process a single record and update all components.

        Args:
            predictions: Predicted spans
            labels: Ground truth spans
            record_id: Optional record identifier
            language: Language code for this record

        Returns:
            List of alerts generated during processing
        """
        alerts: list[StreamingAlert] = []

        # Update evaluator
        self.evaluator.update(
            predictions,
            labels,
            record_id=record_id,
            language=language,
        )

        # Check for drift
        ewma = self.evaluator.get_ewma()
        for metric_name, value in ewma.items():
            alert = self.drift_detector.update(metric_name, value)
            if alert:
                alerts.append(alert)

        # Update coverage monitor
        entity_types = list(set(span.entity_type for span in labels))
        self.coverage_monitor.update(entity_types, language)

        # Store alerts with size limit
        self.recent_alerts.extend(alerts)
        if len(self.recent_alerts) > self.max_alerts_history:
            self.recent_alerts = self.recent_alerts[-self.max_alerts_history :]

        return alerts

    def report(self) -> StreamingReport:
        """Generate comprehensive streaming evaluation report.

        Returns:
            StreamingReport with current state of all components
        """
        ewma = self.evaluator.get_ewma()
        metrics = self.evaluator.get_metrics()
        coverage_gaps = self.coverage_monitor.get_coverage_gaps()

        # Serialize windowed metrics
        windowed_metrics_dict = {
            str(window_size): metric.to_dict()
            for window_size, metric in metrics.items()
        }

        # Serialize alerts
        alert_dicts = [alert.to_dict() for alert in self.recent_alerts]

        return StreamingReport(
            timestamp=time.time(),
            total_records=self.evaluator.get_total_records(),
            ewma_metrics=ewma,
            windowed_metrics=windowed_metrics_dict,
            alerts=alert_dicts,
            coverage_gaps=coverage_gaps,
        )

    def reset(self) -> None:
        """Reset all components to initial state."""
        self.evaluator.reset()
        self.drift_detector.reset()
        self.coverage_monitor.reset()
        self.recent_alerts.clear()
