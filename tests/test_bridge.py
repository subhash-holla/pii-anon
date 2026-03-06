"""Tests for the evaluation framework bridge."""


from pii_anon.bridge import (
    ResultAdapter,
    EvaluationPipelineConfig,
    QuickBenchReport,
)
from pii_anon.eval_framework.metrics.base import LabeledSpan
from pii_anon.types import (
    EnsembleFinding,
    ProcessingProfileSpec,
    SegmentationPlan,
)


class TestResultAdapterFindingsToSpans:
    """Tests for ResultAdapter.findings_to_spans method."""

    def test_findings_to_spans_empty_list(self):
        """Test converting empty findings list."""
        spans = ResultAdapter.findings_to_spans([])

        assert spans == []

    def test_findings_to_spans_dict_input(self):
        """Test converting dict findings to spans."""
        findings = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "span_start": 0,
                "span_end": 10,
            },
            {
                "entity_type": "PERSON_NAME",
                "span_start": 20,
                "span_end": 30,
            },
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 2
        assert spans[0].entity_type == "EMAIL_ADDRESS"
        assert spans[0].start == 0
        assert spans[0].end == 10
        assert spans[1].entity_type == "PERSON_NAME"
        assert spans[1].start == 20
        assert spans[1].end == 30

    def test_findings_to_spans_ensemble_finding_input(self):
        """Test converting EnsembleFinding objects to spans."""
        findings = [
            EnsembleFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.95,
                engines=["engine1"],
                span_start=0,
                span_end=10,
            ),
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.90,
                engines=["engine1", "engine2"],
                span_start=20,
                span_end=30,
            ),
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 2
        assert spans[0].entity_type == "EMAIL_ADDRESS"
        assert spans[0].start == 0
        assert spans[0].end == 10
        assert spans[1].entity_type == "PERSON_NAME"
        assert spans[1].start == 20
        assert spans[1].end == 30

    def test_findings_to_spans_with_record_id(self):
        """Test that record_id is assigned to all spans."""
        findings = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "span_start": 0,
                "span_end": 10,
            }
        ]

        spans = ResultAdapter.findings_to_spans(findings, record_id="record_123")

        assert len(spans) == 1
        assert spans[0].record_id == "record_123"

    def test_findings_to_spans_skips_none_spans(self):
        """Test that findings with None span bounds are skipped."""
        findings = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "span_start": 0,
                "span_end": 10,
            },
            {
                "entity_type": "PERSON_NAME",
                "span_start": None,
                "span_end": 30,
            },
            {
                "entity_type": "PHONE_NUMBER",
                "span_start": 40,
                "span_end": None,
            },
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 1
        assert spans[0].entity_type == "EMAIL_ADDRESS"

    def test_findings_to_spans_skips_missing_span_keys(self):
        """Test that findings missing span keys are skipped."""
        findings = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "span_start": 0,
                "span_end": 10,
            },
            {
                "entity_type": "PERSON_NAME",
                # Missing span_start and span_end
            },
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 1
        assert spans[0].entity_type == "EMAIL_ADDRESS"

    def test_findings_to_spans_converts_to_int(self):
        """Test that span bounds are converted to int."""
        findings = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "span_start": "0",
                "span_end": "10",
            }
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 1
        assert isinstance(spans[0].start, int)
        assert isinstance(spans[0].end, int)

    def test_findings_to_spans_dict_missing_entity_type(self):
        """Test handling of dict with missing entity_type."""
        findings = [
            {
                "span_start": 0,
                "span_end": 10,
            }
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 1
        assert spans[0].entity_type == ""


class TestResultAdapterSpansToFindings:
    """Tests for ResultAdapter.spans_to_findings method."""

    def test_spans_to_findings_empty_list(self):
        """Test converting empty spans list."""
        findings = ResultAdapter.spans_to_findings([])

        assert findings == []

    def test_spans_to_findings_single_span(self):
        """Test converting a single span."""
        spans = [
            LabeledSpan(
                entity_type="EMAIL_ADDRESS",
                start=0,
                end=10,
                record_id="rec1",
            )
        ]

        findings = ResultAdapter.spans_to_findings(spans)

        assert len(findings) == 1
        finding = findings[0]
        assert finding.entity_type == "EMAIL_ADDRESS"
        assert finding.span_start == 0
        assert finding.span_end == 10
        assert finding.confidence == 1.0

    def test_spans_to_findings_multiple_spans(self):
        """Test converting multiple spans."""
        spans = [
            LabeledSpan(entity_type="EMAIL_ADDRESS", start=0, end=10),
            LabeledSpan(entity_type="PERSON_NAME", start=20, end=30),
            LabeledSpan(entity_type="PHONE_NUMBER", start=40, end=50),
        ]

        findings = ResultAdapter.spans_to_findings(spans)

        assert len(findings) == 3
        assert findings[0].entity_type == "EMAIL_ADDRESS"
        assert findings[1].entity_type == "PERSON_NAME"
        assert findings[2].entity_type == "PHONE_NUMBER"

    def test_spans_to_findings_custom_engine_id(self):
        """Test specifying custom engine_id."""
        spans = [LabeledSpan(entity_type="EMAIL_ADDRESS", start=0, end=10)]

        findings = ResultAdapter.spans_to_findings(spans, engine_id="custom_engine")

        assert len(findings) == 1
        assert findings[0].engines == ["custom_engine"]

    def test_spans_to_findings_custom_confidence(self):
        """Test specifying custom confidence."""
        spans = [LabeledSpan(entity_type="EMAIL_ADDRESS", start=0, end=10)]

        findings = ResultAdapter.spans_to_findings(spans, confidence=0.85)

        assert len(findings) == 1
        assert findings[0].confidence == 0.85

    def test_spans_to_findings_default_values(self):
        """Test default values for engine_id and confidence."""
        spans = [LabeledSpan(entity_type="EMAIL_ADDRESS", start=0, end=10)]

        findings = ResultAdapter.spans_to_findings(spans)

        assert findings[0].engines == ["external"]
        assert findings[0].confidence == 1.0


class TestResultAdapterLabelsFromRecord:
    """Tests for ResultAdapter.labels_from_record method."""

    def test_labels_from_record_with_list_labels(self):
        """Test extracting labels from record with list of dicts."""

        class MockRecord:
            def __init__(self):
                self.labels = [
                    {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 10},
                    {"entity_type": "PERSON_NAME", "start": 20, "end": 30},
                ]
                self.record_id = "rec_123"

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert len(spans) == 2
        assert spans[0].entity_type == "EMAIL_ADDRESS"
        assert spans[0].start == 0
        assert spans[0].end == 10
        assert spans[0].record_id == "rec_123"

    def test_labels_from_record_with_labeled_span_objects(self):
        """Test extracting labels that are LabeledSpan objects."""

        class MockRecord:
            def __init__(self):
                self.labels = [
                    LabeledSpan(entity_type="EMAIL_ADDRESS", start=0, end=10),
                    LabeledSpan(entity_type="PERSON_NAME", start=20, end=30),
                ]
                self.record_id = "rec_456"

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert len(spans) == 2
        assert spans[0].entity_type == "EMAIL_ADDRESS"
        assert spans[0].record_id == "rec_456"

    def test_labels_from_record_missing_labels(self):
        """Test record without labels attribute."""

        class MockRecord:
            pass

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert spans == []

    def test_labels_from_record_empty_labels(self):
        """Test record with empty labels list."""

        class MockRecord:
            def __init__(self):
                self.labels = []
                self.record_id = "rec_789"

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert spans == []

    def test_labels_from_record_missing_record_id(self):
        """Test record without record_id attribute."""

        class MockRecord:
            def __init__(self):
                self.labels = [
                    {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 10}
                ]

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert len(spans) == 1
        assert spans[0].record_id == ""

    def test_labels_from_record_converts_to_int(self):
        """Test that start/end are converted to int."""

        class MockRecord:
            def __init__(self):
                self.labels = [
                    {"entity_type": "EMAIL_ADDRESS", "start": "5", "end": "15"}
                ]
                self.record_id = "rec_123"

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert isinstance(spans[0].start, int)
        assert isinstance(spans[0].end, int)


class TestEvaluationPipelineConfig:
    """Tests for EvaluationPipelineConfig dataclass."""

    def test_default_values(self):
        """Test default values of EvaluationPipelineConfig."""
        config = EvaluationPipelineConfig()

        assert isinstance(config.profile, ProcessingProfileSpec)
        assert config.profile.profile_id == "eval_pipeline"
        assert isinstance(config.segmentation, SegmentationPlan)
        assert config.scope == "eval"
        assert config.token_version == 1
        assert config.include_transform_context is True

    def test_custom_values(self):
        """Test setting custom values."""
        profile = ProcessingProfileSpec(profile_id="custom_profile")
        segmentation = SegmentationPlan()

        config = EvaluationPipelineConfig(
            profile=profile,
            segmentation=segmentation,
            scope="custom_scope",
            token_version=2,
            include_transform_context=False,
        )

        assert config.profile.profile_id == "custom_profile"
        assert config.scope == "custom_scope"
        assert config.token_version == 2
        assert config.include_transform_context is False

    def test_profile_defaults_to_eval_pipeline(self):
        """Test that profile defaults to eval_pipeline."""
        config = EvaluationPipelineConfig()

        assert config.profile.profile_id == "eval_pipeline"

    def test_segmentation_defaults_to_new_instance(self):
        """Test that segmentation defaults to new SegmentationPlan."""
        config1 = EvaluationPipelineConfig()
        config2 = EvaluationPipelineConfig()

        # Should be different instances
        assert config1.segmentation is not config2.segmentation


class TestQuickBenchReport:
    """Tests for QuickBenchReport dataclass."""

    def test_default_values(self):
        """Test default values of QuickBenchReport."""
        report = QuickBenchReport()

        assert report.records_evaluated == 0
        assert report.micro_f1 == 0.0
        assert report.macro_f1 == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.duration_seconds == 0.0
        assert report.batch_report is None

    def test_custom_values(self):
        """Test setting custom values."""
        report = QuickBenchReport(
            records_evaluated=100,
            micro_f1=0.85,
            macro_f1=0.82,
            precision=0.87,
            recall=0.83,
            duration_seconds=45.3,
            batch_report={"some": "data"},
        )

        assert report.records_evaluated == 100
        assert report.micro_f1 == 0.85
        assert report.macro_f1 == 0.82
        assert report.precision == 0.87
        assert report.recall == 0.83
        assert report.duration_seconds == 45.3
        assert report.batch_report == {"some": "data"}

    def test_all_metrics_zero_by_default(self):
        """Test that all metrics are zero by default."""
        report = QuickBenchReport()

        assert report.records_evaluated == 0
        assert report.micro_f1 == 0.0
        assert report.macro_f1 == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.duration_seconds == 0.0

    def test_partial_initialization(self):
        """Test initializing with some values."""
        report = QuickBenchReport(
            records_evaluated=50,
            micro_f1=0.75,
        )

        assert report.records_evaluated == 50
        assert report.micro_f1 == 0.75
        assert report.macro_f1 == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0


class TestResultAdapterIntegration:
    """Integration tests for ResultAdapter."""

    def test_round_trip_spans_findings_spans(self):
        """Test converting spans -> findings -> spans."""
        original_spans = [
            LabeledSpan(entity_type="EMAIL_ADDRESS", start=0, end=10),
            LabeledSpan(entity_type="PERSON_NAME", start=20, end=30),
        ]

        findings = ResultAdapter.spans_to_findings(original_spans)
        converted_spans = ResultAdapter.findings_to_spans(findings)

        assert len(converted_spans) == len(original_spans)
        for orig, conv in zip(original_spans, converted_spans):
            assert orig.entity_type == conv.entity_type
            assert orig.start == conv.start
            assert orig.end == conv.end

    def test_findings_to_spans_mixed_input_types(self):
        """Test findings_to_spans with dict and EnsembleFinding mix."""
        findings = [
            {
                "entity_type": "EMAIL_ADDRESS",
                "span_start": 0,
                "span_end": 10,
            },
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.95,
                engines=["engine1"],
                span_start=20,
                span_end=30,
            ),
        ]

        spans = ResultAdapter.findings_to_spans(findings)

        assert len(spans) == 2
        assert spans[0].entity_type == "EMAIL_ADDRESS"
        assert spans[1].entity_type == "PERSON_NAME"

    def test_labels_from_record_with_mixed_label_types(self):
        """Test labels_from_record with mixed dict and object labels."""

        class MockRecord:
            def __init__(self):
                self.labels = [
                    {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 10},
                    LabeledSpan(entity_type="PERSON_NAME", start=20, end=30),
                ]
                self.record_id = "rec_mixed"

        record = MockRecord()
        spans = ResultAdapter.labels_from_record(record)

        assert len(spans) == 2
        assert spans[0].entity_type == "EMAIL_ADDRESS"
        assert spans[1].entity_type == "PERSON_NAME"
        assert all(s.record_id == "rec_mixed" for s in spans)
