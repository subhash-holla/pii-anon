"""Confidence calibration metrics for probabilistic PII detection outputs.

Measures how well a system's confidence scores align with actual accuracy,
enabling trust-worthy deployment decisions.

Evidence basis:
- Naeini et al. (2015): Expected Calibration Error (ECE)
- Guo et al. (2017): On Calibration of Modern Neural Networks
- Platt (1999): Probabilistic outputs for support vector machines (temperature scaling)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .base import EvalMetricResult, EvaluationLevel, LabeledSpan, MatchMode, MultiLevelMetric, safe_div


@dataclass
class CalibrationBin:
    """Represents a single confidence bin in a calibration analysis."""

    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float = field(default=0.0)  # |accuracy - confidence|

    def __post_init__(self) -> None:
        """Compute the gap if not set."""
        if self.gap == 0.0 and (self.avg_confidence != 0.0 or self.avg_accuracy != 0.0):
            self.gap = abs(self.avg_accuracy - self.avg_confidence)


@dataclass
class CalibrationReport:
    """Complete calibration analysis report."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    num_bins: int
    bins: list[CalibrationBin]
    total_predictions: int
    overconfident_fraction: float  # fraction of bins where confidence > accuracy
    underconfident_fraction: float
    suggested_temperature: float | None  # None if well-calibrated


class ExpectedCalibrationError(MultiLevelMetric):
    """Computes Expected Calibration Error for confidence predictions.

    ECE measures the weighted average gap between predicted confidence and
    actual accuracy across confidence bins.
    """

    name = "expected_calibration_error"

    def __init__(self, num_bins: int = 15) -> None:
        """Initialize ECE metric.

        Args:
            num_bins: Number of equal-width confidence bins (default 15).
        """
        super().__init__()
        self.num_bins = num_bins
        self._supported_levels = [EvaluationLevel.ENTITY, EvaluationLevel.DOCUMENT]

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        """Evaluation levels this metric can operate at."""
        return self._supported_levels

    def compute(
        self,
        predictions: list[LabeledSpan],
        ground_truth: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        """Compute Expected Calibration Error.

        Args:
            predictions: Predicted entities.
            ground_truth: Ground truth entities.
            level: Evaluation level (ENTITY or DOCUMENT).
            match_mode: Matching mode for evaluation.
            context: Context dict containing:
                - prediction_confidences: list[float] - confidence scores
                - prediction_correct: list[bool] - correctness labels

        Returns:
            EvalMetricResult with ECE value and bin metadata.
        """
        if context is None:
            context = {}

        confidences = context.get("prediction_confidences", [])
        correct = context.get("prediction_correct", [])

        if not confidences or not correct:
            return EvalMetricResult(
                name=self.name,
                value=0.0,
                level=level,
                metadata={"error": "Missing prediction_confidences or prediction_correct in context"},
            )

        report = reliability_diagram_data(confidences, correct, num_bins=self.num_bins)

        return EvalMetricResult(
            name=self.name,
            value=report.ece,
            level=level,
            metadata={
                "ece": report.ece,
                "mce": report.mce,
                "num_bins": report.num_bins,
                "total_predictions": report.total_predictions,
                "overconfident_fraction": report.overconfident_fraction,
                "underconfident_fraction": report.underconfident_fraction,
                "suggested_temperature": report.suggested_temperature,
                "bins": [
                    {
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "avg_confidence": b.avg_confidence,
                        "avg_accuracy": b.avg_accuracy,
                        "count": b.count,
                        "gap": b.gap,
                    }
                    for b in report.bins
                ],
            },
        )


class MaximumCalibrationError(MultiLevelMetric):
    """Computes Maximum Calibration Error for confidence predictions.

    MCE is the largest gap between predicted confidence and actual accuracy
    across all confidence bins.
    """

    name = "maximum_calibration_error"

    def __init__(self, num_bins: int = 15) -> None:
        """Initialize MCE metric.

        Args:
            num_bins: Number of equal-width confidence bins (default 15).
        """
        super().__init__()
        self.num_bins = num_bins
        self._supported_levels = [EvaluationLevel.ENTITY, EvaluationLevel.DOCUMENT]

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        """Evaluation levels this metric can operate at."""
        return self._supported_levels

    def compute(
        self,
        predictions: list[LabeledSpan],
        ground_truth: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        """Compute Maximum Calibration Error.

        Args:
            predictions: Predicted entities.
            ground_truth: Ground truth entities.
            level: Evaluation level (ENTITY or DOCUMENT).
            match_mode: Matching mode for evaluation.
            context: Context dict containing:
                - prediction_confidences: list[float] - confidence scores
                - prediction_correct: list[bool] - correctness labels

        Returns:
            EvalMetricResult with MCE value and bin metadata.
        """
        if context is None:
            context = {}

        confidences = context.get("prediction_confidences", [])
        correct = context.get("prediction_correct", [])

        if not confidences or not correct:
            return EvalMetricResult(
                name=self.name,
                value=0.0,
                level=level,
                metadata={"error": "Missing prediction_confidences or prediction_correct in context"},
            )

        report = reliability_diagram_data(confidences, correct, num_bins=self.num_bins)

        return EvalMetricResult(
            name=self.name,
            value=report.mce,
            level=level,
            metadata={
                "mce": report.mce,
                "ece": report.ece,
                "num_bins": report.num_bins,
                "total_predictions": report.total_predictions,
            },
        )


def reliability_diagram_data(
    confidences: list[float], correct: list[bool], num_bins: int = 15
) -> CalibrationReport:
    """Compute reliability diagram data and calibration metrics.

    Args:
        confidences: List of predicted confidence scores (0-1).
        correct: List of boolean correctness labels.
        num_bins: Number of equal-width bins (default 15).

    Returns:
        CalibrationReport with bins, ECE, MCE, and calibration info.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have same length")

    if not confidences:
        return CalibrationReport(
            ece=0.0,
            mce=0.0,
            num_bins=num_bins,
            bins=[],
            total_predictions=0,
            overconfident_fraction=0.0,
            underconfident_fraction=0.0,
            suggested_temperature=None,
        )

    # Clamp confidences to [0, 1]
    confidences = [max(0.0, min(1.0, c)) for c in confidences]

    # Create bins
    bins: list[CalibrationBin] = []
    bin_width = 1.0 / num_bins

    for bin_idx in range(num_bins):
        bin_lower = bin_idx * bin_width
        bin_upper = (bin_idx + 1) * bin_width

        # Collect predictions in this bin
        bin_confidences = []
        bin_correct = []

        for conf, corr in zip(confidences, correct):
            # Include predictions where bin_lower <= conf < bin_upper
            # For the last bin, include the upper boundary
            if bin_idx == num_bins - 1:
                in_bin = bin_lower <= conf <= bin_upper
            else:
                in_bin = bin_lower <= conf < bin_upper

            if in_bin:
                bin_confidences.append(conf)
                bin_correct.append(corr)

        if bin_confidences:
            avg_confidence = sum(bin_confidences) / len(bin_confidences)
            avg_accuracy = sum(bin_correct) / len(bin_correct)
            count = len(bin_confidences)
            gap = abs(avg_accuracy - avg_confidence)

            bins.append(
                CalibrationBin(
                    bin_lower=bin_lower,
                    bin_upper=bin_upper,
                    avg_confidence=avg_confidence,
                    avg_accuracy=avg_accuracy,
                    count=count,
                    gap=gap,
                )
            )

    # Compute ECE (weighted average gap)
    total_count = sum(b.count for b in bins)
    ece = 0.0
    if total_count > 0:
        ece = sum(b.gap * b.count for b in bins) / total_count

    # Compute MCE (maximum gap)
    mce = max((b.gap for b in bins), default=0.0)

    # Compute over/underconfident fractions
    overconfident_count = sum(1 for b in bins if b.avg_confidence > b.avg_accuracy)
    underconfident_count = sum(1 for b in bins if b.avg_confidence < b.avg_accuracy)
    non_empty_bins = len(bins)

    overconfident_fraction = safe_div(overconfident_count, non_empty_bins)
    underconfident_fraction = safe_div(underconfident_count, non_empty_bins)

    # Suggest temperature if poorly calibrated
    suggested_temperature = None
    if ece > 0.05:
        if overconfident_fraction > underconfident_fraction:
            # System is overconfident, increase temperature
            suggested_temperature = 1.0 + (ece * 5.0)
        else:
            # System is underconfident, decrease temperature
            suggested_temperature = max(0.5, 1.0 - (ece * 5.0))

    return CalibrationReport(
        ece=ece,
        mce=mce,
        num_bins=num_bins,
        bins=bins,
        total_predictions=total_count,
        overconfident_fraction=overconfident_fraction,
        underconfident_fraction=underconfident_fraction,
        suggested_temperature=suggested_temperature,
    )


class TemperatureScaler:
    """Utility class for temperature scaling of confidence scores.

    Temperature scaling adjusts confidence scores to improve calibration
    by applying a learned temperature parameter.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize temperature scaler.

        Args:
            temperature: Temperature parameter (>0). Values > 1 reduce confidence,
                        values < 1 increase confidence.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    @staticmethod
    def _logit(p: float) -> float:
        """Compute logit (log odds) of a probability.

        Args:
            p: Probability (0-1).

        Returns:
            Logit value.
        """
        # Clamp to avoid log(0) and division by zero
        p = max(1e-7, min(1.0 - 1e-7, p))
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Compute sigmoid of a value.

        Args:
            x: Input value.

        Returns:
            Sigmoid value (0-1).
        """
        # Clamp to avoid overflow
        x = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def scale(self, confidences: list[float]) -> list[float]:
        """Apply temperature scaling to confidences.

        Args:
            confidences: List of confidence scores (0-1).

        Returns:
            Scaled confidence scores.
        """
        scaled = []
        for conf in confidences:
            # Clamp confidence to valid range
            conf = max(0.0, min(1.0, conf))

            # Apply: scaled = sigmoid(logit(conf) / temperature)
            logit_val = self._logit(conf)
            scaled_logit = logit_val / self.temperature
            scaled_conf = self._sigmoid(scaled_logit)

            scaled.append(scaled_conf)

        return scaled

    def find_optimal_temperature(
        self, confidences: list[float], correct: list[bool], *, num_bins: int = 15
    ) -> float:
        """Find optimal temperature via grid search.

        Args:
            confidences: List of confidence scores (0-1).
            correct: List of boolean correctness labels.
            num_bins: Number of bins for ECE calculation.

        Returns:
            Optimal temperature value.
        """
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must have same length")

        if not confidences:
            return 1.0

        # Grid search over temperatures
        temperatures = [0.5 + 0.1 * i for i in range(26)]  # 0.5 to 3.0
        best_temp = 1.0
        best_ece = float("inf")

        for temp in temperatures:
            scaler = TemperatureScaler(temperature=temp)
            scaled_confs = scaler.scale(confidences)
            report = reliability_diagram_data(scaled_confs, correct, num_bins=num_bins)

            if report.ece < best_ece:
                best_ece = report.ece
                best_temp = temp

        return best_temp

    def calibration_summary(
        self, confidences: list[float], correct: list[bool]
    ) -> dict[str, Any]:
        """Compare calibration before and after temperature scaling.

        Args:
            confidences: List of confidence scores (0-1).
            correct: List of boolean correctness labels.

        Returns:
            Dictionary with before/after ECE, MCE, and optimal temperature.
        """
        if len(confidences) != len(correct):
            raise ValueError("confidences and correct must have same length")

        # Original calibration
        original_report = reliability_diagram_data(confidences, correct)

        # Find optimal temperature
        optimal_temp = self.find_optimal_temperature(confidences, correct)

        # Apply optimal temperature
        scaler = TemperatureScaler(temperature=optimal_temp)
        scaled_confs = scaler.scale(confidences)
        scaled_report = reliability_diagram_data(scaled_confs, correct)

        return {
            "original_ece": original_report.ece,
            "original_mce": original_report.mce,
            "scaled_ece": scaled_report.ece,
            "scaled_mce": scaled_report.mce,
            "optimal_temperature": optimal_temp,
            "ece_improvement": original_report.ece - scaled_report.ece,
            "mce_improvement": original_report.mce - scaled_report.mce,
        }
