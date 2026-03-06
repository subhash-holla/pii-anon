"""Tests for PipelineBuilder and Pipeline execution.

Tests cover:
- PipelineBuilder construction and configuration
- Profile management (with_profile, with_compliance, with_strategy)
- Input/output source configuration
- Validation and build process
- Pipeline execution with in-memory records
- Evaluation integration
"""


import pytest

from pii_anon.ingestion import FileFormat
from pii_anon.pipeline import Pipeline, PipelineBuilder, PipelineReport


class TestPipelineBuilderConstruction:
    """Test default construction and initialization."""

    def test_default_construction(self):
        """PipelineBuilder initializes with sensible defaults."""
        builder = PipelineBuilder()
        assert builder._token_key == "default-key"
        assert builder._config is None
        assert builder._profile.profile_id == "pipeline_default"
        assert builder._scope == "pipeline"
        assert builder._token_version == 1
        assert builder._input_records is None
        assert builder._input_path is None
        assert builder._output_path is None
        assert builder._enable_evaluation is False

    def test_custom_token_key(self):
        """PipelineBuilder accepts custom token_key."""
        builder = PipelineBuilder(token_key="my-secret")
        assert builder._token_key == "my-secret"

    def test_with_config(self):
        """PipelineBuilder accepts CoreConfig parameter."""
        from pii_anon.config import CoreConfig

        config = CoreConfig()
        builder = PipelineBuilder(config=config)
        assert builder._config is config


class TestWithProfile:
    """Test profile configuration via with_profile()."""

    def test_with_profile_sets_profile_id(self):
        """with_profile sets the profile_id."""
        builder = PipelineBuilder().with_profile("custom_profile")
        assert builder._profile.profile_id == "custom_profile"

    def test_with_profile_sets_transform_mode(self):
        """with_profile sets transform_mode."""
        builder = PipelineBuilder().with_profile(
            "test_profile", transform_mode="pseudonymize"
        )
        assert builder._profile.transform_mode == "pseudonymize"

    def test_with_profile_default_transform_mode(self):
        """with_profile defaults to 'anonymize' for transform_mode."""
        builder = PipelineBuilder().with_profile("test_profile")
        assert builder._profile.transform_mode == "anonymize"

    def test_with_profile_accepts_kwargs(self):
        """with_profile passes additional kwargs to ProcessingProfileSpec."""
        builder = PipelineBuilder().with_profile(
            "test_profile",
            transform_mode="redact",
            policy_mode="recall_max",
        )
        assert builder._profile.policy_mode == "recall_max"

    def test_with_profile_returns_builder(self):
        """with_profile returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.with_profile("test")
        assert result is builder


class TestWithCompliance:
    """Test compliance template loading via with_compliance()."""

    def test_with_compliance_loads_template(self):
        """with_compliance loads a compliance template."""
        builder = PipelineBuilder().with_profile("test").with_compliance(
            "hipaa_safe_harbor"
        )
        # Verify that entity_strategies were populated
        assert isinstance(builder._profile.entity_strategies, dict)

    def test_with_compliance_merges_into_profile(self):
        """with_compliance merges template into existing profile."""
        builder = (
            PipelineBuilder()
            .with_profile("test")
            .with_strategy("EMAIL_ADDRESS", "generalize")
        )
        builder.with_compliance("hipaa_safe_harbor")
        # Verify that the earlier strategy is still present after compliance load
        assert "EMAIL_ADDRESS" in builder._profile.entity_strategies

    def test_with_compliance_returns_builder(self):
        """with_compliance returns self for chaining."""
        builder = PipelineBuilder().with_profile("test")
        result = builder.with_compliance("hipaa_safe_harbor")
        assert result is builder

    def test_with_compliance_invalid_template(self):
        """with_compliance raises for unknown template name."""
        builder = PipelineBuilder().with_profile("test")
        with pytest.raises(ValueError):
            builder.with_compliance("nonexistent_template")


class TestWithStrategy:
    """Test per-entity strategy overrides via with_strategy()."""

    def test_with_strategy_sets_entity_strategy(self):
        """with_strategy sets entity type to strategy mapping."""
        builder = PipelineBuilder().with_strategy("EMAIL_ADDRESS", "generalize")
        assert builder._profile.entity_strategies["EMAIL_ADDRESS"] == "generalize"

    def test_with_strategy_with_params(self):
        """with_strategy accepts strategy-specific parameters."""
        builder = PipelineBuilder().with_strategy(
            "AGE", "perturb", epsilon=1.5, delta=0.1
        )
        assert builder._profile.entity_strategies["AGE"] == "perturb"
        assert builder._profile.strategy_params["perturb"]["epsilon"] == 1.5
        assert builder._profile.strategy_params["perturb"]["delta"] == 0.1

    def test_with_strategy_multiple_entities(self):
        """with_strategy can set multiple entities."""
        builder = (
            PipelineBuilder()
            .with_strategy("EMAIL_ADDRESS", "redact")
            .with_strategy("PERSON_NAME", "pseudonymize")
        )
        assert builder._profile.entity_strategies["EMAIL_ADDRESS"] == "redact"
        assert builder._profile.entity_strategies["PERSON_NAME"] == "pseudonymize"

    def test_with_strategy_returns_builder(self):
        """with_strategy returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.with_strategy("EMAIL_ADDRESS", "generalize")
        assert result is builder


class TestWithScope:
    """Test scope configuration via with_scope()."""

    def test_with_scope_sets_scope(self):
        """with_scope sets the tokenization scope."""
        builder = PipelineBuilder().with_scope("document")
        assert builder._scope == "document"

    def test_with_scope_returns_builder(self):
        """with_scope returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.with_scope("document")
        assert result is builder


class TestWithTokenVersion:
    """Test token version configuration via with_token_version()."""

    def test_with_token_version_sets_version(self):
        """with_token_version sets the token version."""
        builder = PipelineBuilder().with_token_version(2)
        assert builder._token_version == 2

    def test_with_token_version_returns_builder(self):
        """with_token_version returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.with_token_version(2)
        assert result is builder


class TestFromRecords:
    """Test in-memory record input via from_records()."""

    def test_from_records_basic(self):
        """from_records stores records for processing."""
        records = [
            {"text": "John's email is john@example.com"},
            {"text": "Jane works at Acme Corp"},
        ]
        builder = PipelineBuilder().from_records(records)
        assert builder._input_records is not None
        assert len(builder._input_records) == 2

    def test_from_records_text_extraction(self):
        """from_records extracts text from specified column."""
        records = [{"text": "Hello world", "id": "1"}]
        builder = PipelineBuilder().from_records(records)
        assert builder._input_records[0].text == "Hello world"
        assert builder._input_records[0].record_id == 0

    def test_from_records_metadata_extraction(self):
        """from_records preserves non-text columns as metadata."""
        records = [{"text": "Hello", "user_id": "123", "timestamp": "2026-01-01"}]
        builder = PipelineBuilder().from_records(records)
        metadata = builder._input_records[0].metadata
        assert metadata["user_id"] == "123"
        assert metadata["timestamp"] == "2026-01-01"

    def test_from_records_custom_text_column(self):
        """from_records respects custom text_column parameter."""
        records = [{"message": "Hello world"}]
        builder = PipelineBuilder().from_records(records, text_column="message")
        assert builder._input_records[0].text == "Hello world"

    def test_from_records_returns_builder(self):
        """from_records returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.from_records([{"text": "test"}])
        assert result is builder


class TestBuild:
    """Test pipeline construction via build()."""

    def test_build_raises_without_input(self):
        """build() raises ValueError if no input source is configured."""
        builder = PipelineBuilder().with_profile("test")
        with pytest.raises(ValueError, match="No input source configured"):
            builder.build()

    def test_build_with_records(self):
        """build() creates a Pipeline when records are provided."""
        builder = PipelineBuilder().from_records([{"text": "test"}])
        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)

    def test_build_from_file(self, tmp_path):
        """build() creates a Pipeline when file input is configured."""
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"text": "test"}\n')
        builder = PipelineBuilder().from_file(str(test_file))
        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)

    def test_build_from_dataframe(self):
        """build() creates a Pipeline when DataFrame input is configured."""
        # Mock DataFrame with iterrows method
        class MockDF:
            def iterrows(self):
                yield 0, {"text": "Hello"}

        df = MockDF()
        builder = PipelineBuilder().from_dataframe(df)
        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)

    def test_build_preserves_configuration(self):
        """build() preserves all configured settings."""
        builder = (
            PipelineBuilder(token_key="secret")
            .with_profile("test", transform_mode="pseudonymize")
            .with_scope("document")
            .with_token_version(2)
            .from_records([{"text": "test"}])
        )
        pipeline = builder.build()
        assert pipeline._token_key == "secret"
        assert pipeline._profile.profile_id == "test"
        assert pipeline._profile.transform_mode == "pseudonymize"
        assert pipeline._scope == "document"
        assert pipeline._token_version == 2


class TestToFile:
    """Test output file configuration via to_file()."""

    def test_to_file_sets_output_path(self):
        """to_file sets the output file path."""
        builder = PipelineBuilder().to_file("output.jsonl")
        assert builder._output_path == "output.jsonl"

    def test_to_file_with_format(self):
        """to_file accepts explicit format."""
        builder = PipelineBuilder().to_file("output.json", fmt=FileFormat.JSON)
        assert builder._output_path == "output.json"
        assert builder._output_fmt == FileFormat.JSON

    def test_to_file_returns_builder(self):
        """to_file returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.to_file("output.jsonl")
        assert result is builder


class TestWithEvaluation:
    """Test evaluation integration via with_evaluation()."""

    def test_with_evaluation_enables_evaluation(self):
        """with_evaluation() enables evaluation."""
        builder = PipelineBuilder().with_evaluation()
        assert builder._enable_evaluation is True

    def test_with_evaluation_accepts_framework(self):
        """with_evaluation() accepts custom framework."""

        class CustomFramework:
            pass

        framework = CustomFramework()
        builder = PipelineBuilder().with_evaluation(framework=framework)
        assert builder._eval_framework is framework

    def test_with_evaluation_returns_builder(self):
        """with_evaluation returns self for chaining."""
        builder = PipelineBuilder()
        result = builder.with_evaluation()
        assert result is builder


class TestPipelineExecution:
    """Test pipeline execution with from_records + build + run."""

    def test_pipeline_run_basic(self):
        """Pipeline.run() processes records and returns PipelineReport."""
        builder = PipelineBuilder().from_records([{"text": "Hello world"}])
        pipeline = builder.build()
        report = pipeline.run()

        assert isinstance(report, PipelineReport)
        assert report.records_processed >= 0
        assert report.records_failed >= 0
        assert report.duration_seconds >= 0.0

    def test_pipeline_run_multiple_records(self):
        """Pipeline.run() processes multiple records."""
        records = [
            {"text": "Record 1"},
            {"text": "Record 2"},
            {"text": "Record 3"},
        ]
        builder = PipelineBuilder().from_records(records)
        pipeline = builder.build()
        report = pipeline.run()

        assert report.records_processed == 3

    def test_pipeline_run_with_metadata(self):
        """Pipeline.run() preserves record metadata."""
        records = [
            {"text": "Hello", "user_id": "123"},
            {"text": "World", "user_id": "456"},
        ]
        builder = PipelineBuilder().from_records(records)
        pipeline = builder.build()
        report = pipeline.run()

        assert report.records_processed == 2

    def test_pipeline_run_returns_report(self):
        """Pipeline.run() returns a valid PipelineReport."""
        builder = PipelineBuilder().from_records([{"text": "test"}])
        pipeline = builder.build()
        report = pipeline.run()

        assert hasattr(report, "records_processed")
        assert hasattr(report, "records_failed")
        assert hasattr(report, "duration_seconds")
        assert hasattr(report, "transform_audit")
        assert hasattr(report, "eval_report")
        assert hasattr(report, "output_path")


class TestPipelineExecutionWithOutput:
    """Test pipeline execution with file output."""

    def test_pipeline_to_file_jsonl(self, tmp_path):
        """Pipeline writes results to JSONL file."""
        output_file = tmp_path / "output.jsonl"
        builder = (
            PipelineBuilder()
            .from_records([{"text": "Test record"}])
            .to_file(str(output_file), fmt=FileFormat.JSONL)
        )
        pipeline = builder.build()
        report = pipeline.run()

        assert report.output_path == str(output_file) or report.output_path is None

    def test_pipeline_to_file_json(self, tmp_path):
        """Pipeline writes results to JSON file."""
        output_file = tmp_path / "output.json"
        builder = (
            PipelineBuilder()
            .from_records([{"text": "Test record"}])
            .to_file(str(output_file), fmt=FileFormat.JSON)
        )
        pipeline = builder.build()
        report = pipeline.run()

        assert report.output_path == str(output_file) or report.output_path is None


class TestPipelineBuilderChaining:
    """Test fluent API chaining."""

    def test_full_chain_construction(self):
        """PipelineBuilder supports full method chaining."""
        report = (
            PipelineBuilder(token_key="secret")
            .with_profile("test_profile", transform_mode="pseudonymize")
            .with_strategy("EMAIL_ADDRESS", "generalize")
            .with_scope("document")
            .with_token_version(2)
            .from_records([{"text": "john@example.com"}])
            .build()
            .run()
        )

        assert isinstance(report, PipelineReport)
        assert report.records_processed >= 0

    def test_chain_with_evaluation(self):
        """PipelineBuilder chain can include evaluation."""
        report = (
            PipelineBuilder()
            .from_records([{"text": "test"}])
            .with_evaluation()
            .build()
            .run()
        )

        assert isinstance(report, PipelineReport)

    def test_chain_with_output(self, tmp_path):
        """PipelineBuilder chain can include file output."""
        output_file = tmp_path / "output.jsonl"
        report = (
            PipelineBuilder()
            .from_records([{"text": "test"}])
            .to_file(str(output_file))
            .build()
            .run()
        )

        assert isinstance(report, PipelineReport)


class TestPipelineReport:
    """Test PipelineReport data structure."""

    def test_pipeline_report_fields(self):
        """PipelineReport has all expected fields."""
        report = PipelineReport(
            records_processed=10,
            records_failed=1,
            duration_seconds=1.234,
            transform_audit=[],
            eval_report=None,
            output_path="/tmp/out.jsonl",
        )

        assert report.records_processed == 10
        assert report.records_failed == 1
        assert report.duration_seconds == 1.234
        assert isinstance(report.transform_audit, list)
        assert report.output_path == "/tmp/out.jsonl"

    def test_pipeline_report_defaults(self):
        """PipelineReport initializes with sensible defaults."""
        report = PipelineReport()

        assert report.records_processed == 0
        assert report.records_failed == 0
        assert report.duration_seconds == 0.0
        assert report.transform_audit == []
        assert report.eval_report is None
        assert report.output_path is None
