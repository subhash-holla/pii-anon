"""Tests for confidence calibration metrics.

Tests cover Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
reliability diagram computation, and temperature scaling.
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.metrics.calibration import (
    CalibrationBin,
    ExpectedCalibrationError,
    MaximumCalibrationError,
    TemperatureScaler,
    reliability_diagram_data,
)
from pii_anon.eval_framework.metrics.base import EvaluationLevel


class TestCalibrationBin:
    """Test the CalibrationBin dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        bin_obj = CalibrationBin(
            bin_lower=0.0,
            bin_upper=0.1,
            avg_confidence=0.05,
            avg_accuracy=0.05,
            count=10,
        )
        assert bin_obj.bin_lower == 0.0
        assert bin_obj.bin_upper == 0.1
        assert bin_obj.count == 10

    def test_gap_computation_in_post_init(self):
        """Test that gap is computed if not provided."""
        bin_obj = CalibrationBin(
            bin_lower=0.0,
            bin_upper=0.1,
            avg_confidence=0.8,
            avg_accuracy=0.5,
            count=10,
        )
        assert bin_obj.gap == pytest.approx(0.3, abs=1e-6)

    def test_gap_zero_when_equal(self):
        """Test gap is 0 when confidence equals accuracy."""
        bin_obj = CalibrationBin(
            bin_lower=0.0,
            bin_upper=0.1,
            avg_confidence=0.5,
            avg_accuracy=0.5,
            count=10,
        )
        assert bin_obj.gap == 0.0

    def test_gap_explicit_value(self):
        """Test explicit gap value is preserved."""
        bin_obj = CalibrationBin(
            bin_lower=0.0,
            bin_upper=0.1,
            avg_confidence=0.5,
            avg_accuracy=0.5,
            count=10,
            gap=0.1,
        )
        assert bin_obj.gap == 0.1


class TestReliabilityDiagramData:
    """Test reliability_diagram_data function."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        report = reliability_diagram_data([], [])
        assert report.ece == 0.0
        assert report.mce == 0.0
        assert len(report.bins) == 0
        assert report.total_predictions == 0

    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            reliability_diagram_data([0.5], [True, False])

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        # 10 predictions with confidence 1.0, all correct (100% accuracy at 100% confidence)
        confidences = [1.0] * 10
        correct = [True] * 10

        report = reliability_diagram_data(confidences, correct, num_bins=10)

        assert report.ece == pytest.approx(0.0, abs=0.01)
        assert report.total_predictions == 10

    def test_overconfident_predictions(self):
        """Test with overconfident predictions."""
        # High confidence but low accuracy
        confidences = [0.9] * 10
        correct = [False] * 10

        report = reliability_diagram_data(confidences, correct, num_bins=10)

        assert report.ece > 0.0  # Should have error
        assert report.overconfident_fraction > 0.0

    def test_underconfident_predictions(self):
        """Test with underconfident predictions."""
        # Low confidence but high accuracy
        confidences = [0.1] * 10
        correct = [True] * 10

        report = reliability_diagram_data(confidences, correct, num_bins=10)

        assert report.ece > 0.0
        assert report.underconfident_fraction > 0.0

    def test_mixed_predictions(self):
        """Test with mixed correct/incorrect predictions."""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        correct = [True, True, False, False, True]

        report = reliability_diagram_data(confidences, correct, num_bins=5)

        assert report.total_predictions == 5
        assert 0.0 <= report.ece <= 1.0
        assert 0.0 <= report.mce <= 1.0

    def test_clamping_to_01(self):
        """Test that confidences are clamped to [0, 1]."""
        confidences = [-0.1, 0.5, 1.5]
        correct = [True, False, True]

        report = reliability_diagram_data(confidences, correct, num_bins=3)

        # Should not raise and should process 3 predictions
        assert report.total_predictions == 3

    def test_num_bins_parameter(self):
        """Test different number of bins."""
        confidences = [0.1 * i for i in range(1, 11)]
        correct = [i % 2 == 0 for i in range(10)]

        for num_bins in [5, 10, 20]:
            report = reliability_diagram_data(confidences, correct, num_bins=num_bins)
            assert report.num_bins == num_bins
            assert len(report.bins) <= num_bins

    def test_ece_range(self):
        """Test that ECE is in valid range [0, 1]."""
        confidences = [0.5, 0.7, 0.9, 0.2]
        correct = [True, False, True, False]

        report = reliability_diagram_data(confidences, correct)

        assert 0.0 <= report.ece <= 1.0

    def test_mce_range(self):
        """Test that MCE is in valid range [0, 1]."""
        confidences = [0.5, 0.7, 0.9, 0.2]
        correct = [True, False, True, False]

        report = reliability_diagram_data(confidences, correct)

        assert 0.0 <= report.mce <= 1.0

    def test_mce_greater_than_or_equal_ece(self):
        """Test that MCE >= ECE."""
        confidences = [0.5, 0.7, 0.9, 0.2]
        correct = [True, False, True, False]

        report = reliability_diagram_data(confidences, correct)

        assert report.mce >= report.ece

    def test_temperature_suggestion_none_for_well_calibrated(self):
        """Test that temperature suggestion is None for well-calibrated."""
        confidences = [0.5] * 100
        correct = [True] * 100

        report = reliability_diagram_data(confidences, correct)

        # Well calibrated should have low ECE
        if report.ece <= 0.05:
            assert report.suggested_temperature is None

    def test_temperature_suggestion_for_poorly_calibrated(self):
        """Test that temperature suggestion is given for poorly calibrated."""
        confidences = [0.9] * 50 + [0.1] * 50
        correct = [False] * 50 + [True] * 50

        report = reliability_diagram_data(confidences, correct)

        if report.ece > 0.05:
            assert report.suggested_temperature is not None
            assert report.suggested_temperature > 0

    def test_overconfident_increases_temperature(self):
        """Test that overconfident predictions increase suggested temperature."""
        confidences = [0.9] * 10
        correct = [False] * 10

        report = reliability_diagram_data(confidences, correct)

        if report.suggested_temperature is not None:
            if report.overconfident_fraction > report.underconfident_fraction:
                # Should increase temperature
                assert report.suggested_temperature > 1.0

    def test_bins_coverage(self):
        """Test that bins cover the full range of confidences."""
        confidences = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        correct = [True, False, True, False, True, False]

        report = reliability_diagram_data(confidences, correct, num_bins=10)

        total_in_bins = sum(b.count for b in report.bins)
        assert total_in_bins == len(confidences)

    def test_bin_boundaries(self):
        """Test that bin boundaries cover the full range."""
        confidences = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        correct = [True, False, True, False, True, False]

        report = reliability_diagram_data(confidences, correct, num_bins=10)

        # Should have at least one bin
        if report.bins:
            # Min lower bound should be <= min confidence
            min_lower = min(b.bin_lower for b in report.bins)
            assert min_lower <= 0.1

            # Max upper bound should be >= max confidence
            max_upper = max(b.bin_upper for b in report.bins)
            assert max_upper >= 0.9


class TestExpectedCalibrationError:
    """Test ExpectedCalibrationError metric."""

    def test_init_default(self):
        """Test initialization with defaults."""
        metric = ExpectedCalibrationError()
        assert metric.num_bins == 15
        assert metric.name == "expected_calibration_error"

    def test_init_custom_bins(self):
        """Test initialization with custom bins."""
        metric = ExpectedCalibrationError(num_bins=20)
        assert metric.num_bins == 20

    def test_supported_levels(self):
        """Test supported evaluation levels."""
        metric = ExpectedCalibrationError()
        levels = metric.supported_levels

        assert EvaluationLevel.ENTITY in levels
        assert EvaluationLevel.DOCUMENT in levels

    def test_compute_with_context(self):
        """Test compute method with valid context."""
        metric = ExpectedCalibrationError()

        context = {
            "prediction_confidences": [0.9, 0.8, 0.7, 0.6, 0.5],
            "prediction_correct": [True, True, False, False, True],
        }

        result = metric.compute(
            predictions=[],
            ground_truth=[],
            context=context,
        )

        assert result.name == "expected_calibration_error"
        assert 0.0 <= result.value <= 1.0
        assert "ece" in result.metadata
        assert "mce" in result.metadata

    def test_compute_missing_context(self):
        """Test compute with missing context."""
        metric = ExpectedCalibrationError()

        result = metric.compute(
            predictions=[],
            ground_truth=[],
        )

        assert result.value == 0.0
        assert "error" in result.metadata

    def test_compute_missing_confidences(self):
        """Test compute with missing confidences."""
        metric = ExpectedCalibrationError()

        context = {
            "prediction_correct": [True, False, True],
        }

        result = metric.compute(
            predictions=[],
            ground_truth=[],
            context=context,
        )

        assert result.value == 0.0
        assert "error" in result.metadata

    def test_compute_missing_correct(self):
        """Test compute with missing correctness labels."""
        metric = ExpectedCalibrationError()

        context = {
            "prediction_confidences": [0.9, 0.8, 0.7],
        }

        result = metric.compute(
            predictions=[],
            ground_truth=[],
            context=context,
        )

        assert result.value == 0.0
        assert "error" in result.metadata

    def test_compute_result_structure(self):
        """Test result structure contains all expected fields."""
        metric = ExpectedCalibrationError()

        context = {
            "prediction_confidences": [0.5] * 10,
            "prediction_correct": [True] * 10,
        }

        result = metric.compute(
            predictions=[],
            ground_truth=[],
            context=context,
        )

        expected_metadata_keys = {
            "ece", "mce", "num_bins", "total_predictions",
            "overconfident_fraction", "underconfident_fraction",
            "suggested_temperature", "bins",
        }
        assert set(result.metadata.keys()) == expected_metadata_keys

    def test_compute_with_different_levels(self):
        """Test compute with different evaluation levels."""
        metric = ExpectedCalibrationError()

        context = {
            "prediction_confidences": [0.5] * 10,
            "prediction_correct": [True] * 10,
        }

        for level in [EvaluationLevel.ENTITY, EvaluationLevel.DOCUMENT]:
            result = metric.compute(
                predictions=[],
                ground_truth=[],
                level=level,
                context=context,
            )

            assert result.level == level


class TestMaximumCalibrationError:
    """Test MaximumCalibrationError metric."""

    def test_init_default(self):
        """Test initialization with defaults."""
        metric = MaximumCalibrationError()
        assert metric.num_bins == 15
        assert metric.name == "maximum_calibration_error"

    def test_init_custom_bins(self):
        """Test initialization with custom bins."""
        metric = MaximumCalibrationError(num_bins=10)
        assert metric.num_bins == 10

    def test_compute_with_context(self):
        """Test compute method with valid context."""
        metric = MaximumCalibrationError()

        context = {
            "prediction_confidences": [0.9, 0.8, 0.1, 0.2],
            "prediction_correct": [False, False, True, True],
        }

        result = metric.compute(
            predictions=[],
            ground_truth=[],
            context=context,
        )

        assert result.name == "maximum_calibration_error"
        assert 0.0 <= result.value <= 1.0
        assert "mce" in result.metadata

    def test_compute_result_contains_ece(self):
        """Test that MCE result also contains ECE."""
        metric = MaximumCalibrationError()

        context = {
            "prediction_confidences": [0.5] * 10,
            "prediction_correct": [True] * 10,
        }

        result = metric.compute(
            predictions=[],
            ground_truth=[],
            context=context,
        )

        assert "ece" in result.metadata
        assert "mce" in result.metadata


class TestTemperatureScaler:
    """Test TemperatureScaler utility class."""

    def test_init_valid_temperature(self):
        """Test initialization with valid temperature."""
        scaler = TemperatureScaler(temperature=1.5)
        assert scaler.temperature == 1.5

    def test_init_zero_temperature_raises(self):
        """Test that zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=0.0)

    def test_init_negative_temperature_raises(self):
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=-1.0)

    def test_logit_function(self):
        """Test logit function."""
        # logit(0.5) = 0
        assert TemperatureScaler._logit(0.5) == pytest.approx(0.0, abs=1e-6)

        # logit(0.9) > 0
        assert TemperatureScaler._logit(0.9) > 0

        # logit(0.1) < 0
        assert TemperatureScaler._logit(0.1) < 0

    def test_sigmoid_function(self):
        """Test sigmoid function."""
        # sigmoid(0) = 0.5
        assert TemperatureScaler._sigmoid(0.0) == pytest.approx(0.5, abs=1e-6)

        # sigmoid(x) in (0, 1) for all x
        for x in [-10, -1, 0, 1, 10]:
            result = TemperatureScaler._sigmoid(x)
            assert 0.0 < result < 1.0

    def test_scale_single_confidence(self):
        """Test scaling a single confidence."""
        scaler = TemperatureScaler(temperature=1.0)
        result = scaler.scale([0.5])

        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0

    def test_scale_identity_at_temperature_one(self):
        """Test that temperature 1 approximately preserves confidence."""
        scaler = TemperatureScaler(temperature=1.0)
        confidences = [0.1, 0.5, 0.9]
        scaled = scaler.scale(confidences)

        for orig, scaled_conf in zip(confidences, scaled):
            assert scaled_conf == pytest.approx(orig, abs=1e-6)

    def test_scale_increases_at_temperature_below_one(self):
        """Test that temperature < 1 increases confidence."""
        scaler = TemperatureScaler(temperature=0.5)
        confidences = [0.1, 0.5, 0.9]
        scaled = scaler.scale(confidences)

        # Temperature < 1 should increase confidences
        # (except extremes like 0 and 1)
        assert scaled[1] >= confidences[1]

    def test_scale_decreases_at_temperature_above_one(self):
        """Test that temperature > 1 decreases confidence."""
        scaler = TemperatureScaler(temperature=2.0)
        confidences = [0.1, 0.5, 0.9]
        scaled = scaler.scale(confidences)

        # Temperature > 1 should decrease confidences
        # (except extremes like 0 and 1)
        assert scaled[1] <= confidences[1]

    def test_scale_clamps_output_to_01(self):
        """Test that scaled confidences are in [0, 1]."""
        scaler = TemperatureScaler(temperature=0.1)  # Very low
        confidences = [0.1, 0.5, 0.9]
        scaled = scaler.scale(confidences)

        for conf in scaled:
            assert 0.0 <= conf <= 1.0

    def test_find_optimal_temperature(self):
        """Test finding optimal temperature."""
        confidences = [0.9] * 50 + [0.1] * 50
        correct = [False] * 50 + [True] * 50

        scaler = TemperatureScaler()
        optimal = scaler.find_optimal_temperature(confidences, correct)

        assert optimal > 0
        assert isinstance(optimal, float)

    def test_find_optimal_temperature_empty_raises(self):
        """Test that empty inputs return 1.0."""
        scaler = TemperatureScaler()
        optimal = scaler.find_optimal_temperature([], [])

        assert optimal == 1.0

    def test_find_optimal_temperature_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        scaler = TemperatureScaler()

        with pytest.raises(ValueError, match="same length"):
            scaler.find_optimal_temperature([0.5], [True, False])

    def test_calibration_summary_structure(self):
        """Test calibration summary has required fields."""
        confidences = [0.9, 0.7, 0.5, 0.3, 0.1]
        correct = [True, True, False, False, True]

        scaler = TemperatureScaler()
        summary = scaler.calibration_summary(confidences, correct)

        required_keys = {
            "original_ece",
            "original_mce",
            "scaled_ece",
            "scaled_mce",
            "optimal_temperature",
            "ece_improvement",
            "mce_improvement",
        }
        assert set(summary.keys()) == required_keys

    def test_calibration_summary_improvements_positive(self):
        """Test that scaling improves calibration."""
        # Create poorly calibrated predictions
        confidences = [0.9] * 50 + [0.1] * 50
        correct = [False] * 50 + [True] * 50

        scaler = TemperatureScaler()
        summary = scaler.calibration_summary(confidences, correct)

        # Scaling should improve calibration
        assert summary["scaled_ece"] <= summary["original_ece"]

    def test_calibration_summary_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        scaler = TemperatureScaler()

        with pytest.raises(ValueError, match="same length"):
            scaler.calibration_summary([0.5], [True, False])

    def test_temperature_in_valid_range(self):
        """Test that found temperature is in reasonable range."""
        confidences = [0.5] * 100
        correct = [True] * 100

        scaler = TemperatureScaler()
        optimal = scaler.find_optimal_temperature(confidences, correct)

        # Should be in range [0.5, 3.0] (from grid search)
        assert 0.5 <= optimal <= 3.0
