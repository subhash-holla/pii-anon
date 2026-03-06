"""Fluent pipeline builder for PII detection, transformation, and evaluation.

Provides a chainable API that simplifies common workflows into concise,
readable configurations.  Handles orchestrator construction, strategy
selection, compliance templates, ingestion, and optional evaluation.

Usage::

    from pii_anon.pipeline import PipelineBuilder

    report = (
        PipelineBuilder(token_key="secret")
        .with_profile("gdpr_audit", transform_mode="pseudonymize")
        .with_compliance("gdpr_pseudonymization")
        .with_evaluation()
        .from_file("input.jsonl")
        .to_file("output.jsonl")
        .build()
        .run()
    )
    print(f"Processed {report.records_processed} records in {report.duration_seconds}s")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pii_anon.config import CoreConfig
from pii_anon.ingestion import (
    FileFormat,
    IngestConfig,
    IngestRecord,
    read_file,
    write_results,
)
from pii_anon.ingestion.dataframe import read_dataframe
from pii_anon.transforms.policies import load_compliance_template
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@dataclass
class PipelineReport:
    """Results from a pipeline run.

    Attributes
    ----------
    records_processed : int
        Total records successfully processed.
    records_failed : int
        Records that encountered errors during processing.
    duration_seconds : float
        Total wall-clock time for the pipeline run.
    transform_audit : list[dict[str, Any]]
        Per-record transformation audit trail.
    eval_report : Any | None
        Evaluation report (if evaluation was enabled).
    output_path : str | None
        Path to the output file (if file output was configured).
    """

    records_processed: int = 0
    records_failed: int = 0
    duration_seconds: float = 0.0
    transform_audit: list[dict[str, Any]] = field(default_factory=list)
    eval_report: Any = None
    output_path: str | None = None


class Pipeline:
    """Configured pipeline ready for execution.

    Created by ``PipelineBuilder.build()``.  Call ``run()`` to execute.
    """

    def __init__(
        self,
        *,
        token_key: str,
        config: CoreConfig | None,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
        input_records: list[IngestRecord] | None,
        input_path: str | None,
        input_config: IngestConfig | None,
        input_dataframe: Any | None,
        df_text_column: str,
        output_path: str | None,
        output_fmt: FileFormat | None,
        collect_dataframe: bool,
        enable_evaluation: bool,
        eval_framework: Any | None,
    ) -> None:
        self._token_key = token_key
        self._config = config
        self._profile = profile
        self._segmentation = segmentation
        self._scope = scope
        self._token_version = token_version
        self._input_records = input_records
        self._input_path = input_path
        self._input_config = input_config
        self._input_dataframe = input_dataframe
        self._df_text_column = df_text_column
        self._output_path = output_path
        self._output_fmt = output_fmt
        self._collect_dataframe = collect_dataframe
        self._enable_evaluation = enable_evaluation
        self._eval_framework = eval_framework

    def _get_records(self) -> list[IngestRecord]:
        """Resolve input records from the configured source."""
        if self._input_records is not None:
            return self._input_records
        if self._input_path is not None:
            return list(read_file(self._input_path, self._input_config))
        if self._input_dataframe is not None:
            return list(read_dataframe(
                self._input_dataframe,
                text_column=self._df_text_column,
            ))
        return []

    def run(self) -> PipelineReport:
        """Execute the pipeline synchronously.

        Returns
        -------
        PipelineReport
            Summary of the pipeline execution.
        """
        from pii_anon.orchestrator import PIIOrchestrator

        start = time.monotonic()

        # Build orchestrator
        orch = PIIOrchestrator(
            token_key=self._token_key,
            config=self._config,
        )

        records = self._get_records()

        # Process records, accumulating only lightweight audit/eval data.
        # Results are yielded through a generator for streaming output.
        all_audit: list[dict[str, Any]] = []
        processed = 0
        failed = 0
        eval_predictions_count = 0

        # Lazy imports for evaluation (only when needed).
        _result_adapter = None
        if self._enable_evaluation:
            try:
                from pii_anon.bridge import ResultAdapter
                _result_adapter = ResultAdapter
            except ImportError:
                pass

        def _process() -> Any:
            """Generator that yields results for streaming output."""
            nonlocal processed, failed, eval_predictions_count
            for record in records:
                try:
                    payload: dict[str, Any] = {"text": record.text}
                    payload.update(record.metadata)
                    result = orch.run(
                        payload,
                        profile=self._profile,
                        segmentation=self._segmentation,
                        scope=self._scope,
                        token_version=self._token_version,
                    )
                    result["metadata"] = record.metadata
                    result["record_id"] = record.record_id
                    processed += 1

                    # Collect audit
                    audit_entries = result.get("transform_audit", [])
                    if isinstance(audit_entries, list):
                        all_audit.extend(audit_entries)

                    # Collect eval predictions inline to avoid second pass.
                    if _result_adapter is not None:
                        findings = result.get("ensemble_findings", [])
                        spans = _result_adapter.findings_to_spans(
                            findings,
                            record_id=str(result.get("record_id", "")),
                        )
                        eval_predictions_count += len(spans)

                    yield result
                except Exception:
                    failed += 1

        # Write output via streaming generator.
        output_written = None
        if self._output_path:
            write_results(
                _process(),
                self._output_path,
                fmt=self._output_fmt,
            )
            output_written = self._output_path
        else:
            # Consume the generator even when there's no file output.
            for _ in _process():
                pass

        eval_report = None
        if self._enable_evaluation and eval_predictions_count > 0:
            eval_report = {"predictions_count": eval_predictions_count}

        duration = time.monotonic() - start

        return PipelineReport(
            records_processed=processed,
            records_failed=failed,
            duration_seconds=round(duration, 3),
            transform_audit=all_audit,
            eval_report=eval_report,
            output_path=output_written,
        )


class PipelineBuilder:
    """Fluent builder for constructing and configuring PII pipelines.

    Example
    -------
    >>> report = (
    ...     PipelineBuilder(token_key="secret")
    ...     .with_profile("default")
    ...     .from_records([{"text": "John's email is john@test.com"}])
    ...     .build()
    ...     .run()
    ... )
    >>> report.records_processed
    1
    """

    def __init__(
        self,
        token_key: str = "default-key",
        *,
        config: CoreConfig | None = None,
    ) -> None:
        self._token_key = token_key
        self._config = config
        self._profile = ProcessingProfileSpec(profile_id="pipeline_default")
        self._segmentation = SegmentationPlan()
        self._scope = "pipeline"
        self._token_version = 1
        self._input_records: list[IngestRecord] | None = None
        self._input_path: str | None = None
        self._input_config: IngestConfig | None = None
        self._input_dataframe: Any = None
        self._df_text_column = "text"
        self._output_path: str | None = None
        self._output_fmt: FileFormat | None = None
        self._collect_dataframe = False
        self._enable_evaluation = False
        self._eval_framework: Any = None

    def with_profile(
        self,
        profile_id: str,
        *,
        transform_mode: str = "anonymize",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Configure the processing profile.

        Parameters
        ----------
        profile_id : str
            Profile identifier.
        transform_mode : str
            Transformation mode (e.g., ``"anonymize"``, ``"pseudonymize"``).
        **kwargs : Any
            Additional profile fields.
        """
        self._profile = ProcessingProfileSpec(
            profile_id=profile_id,
            transform_mode=transform_mode,
            **kwargs,
        )
        return self

    def with_compliance(self, template_name: str) -> "PipelineBuilder":
        """Load a compliance template and apply it to the profile.

        Parameters
        ----------
        template_name : str
            Template name (e.g., ``"hipaa_safe_harbor"``).
        """
        policy = load_compliance_template(template_name)
        entity_strategies, strategy_params = policy.to_profile_overrides()
        self._profile.entity_strategies.update(entity_strategies)
        self._profile.strategy_params.update(strategy_params)
        return self

    def with_strategy(
        self,
        entity_type: str,
        strategy: str,
        **params: Any,
    ) -> "PipelineBuilder":
        """Override the strategy for a specific entity type.

        Parameters
        ----------
        entity_type : str
            Entity type (e.g., ``"EMAIL_ADDRESS"``).
        strategy : str
            Strategy ID (e.g., ``"generalize"``).
        **params : Any
            Strategy-specific parameters.
        """
        self._profile.entity_strategies[entity_type] = strategy
        if params:
            existing = self._profile.strategy_params.get(strategy, {})
            existing.update(params)
            self._profile.strategy_params[strategy] = existing
        return self

    def with_scope(self, scope: str) -> "PipelineBuilder":
        """Set the tokenization scope."""
        self._scope = scope
        return self

    def with_token_version(self, version: int) -> "PipelineBuilder":
        """Set the token version."""
        self._token_version = version
        return self

    def with_segmentation(self, plan: SegmentationPlan) -> "PipelineBuilder":
        """Set the segmentation plan."""
        self._segmentation = plan
        return self

    def with_evaluation(self, framework: Any = None) -> "PipelineBuilder":
        """Enable evaluation after processing.

        Parameters
        ----------
        framework : EvaluationFramework | None
            Custom framework instance (uses default if None).
        """
        self._enable_evaluation = True
        self._eval_framework = framework
        return self

    def from_file(
        self,
        path: str | Path,
        *,
        fmt: FileFormat | None = None,
        text_column: str = "text",
        encoding: str = "utf-8",
    ) -> "PipelineBuilder":
        """Set file input source.

        Parameters
        ----------
        path : str | Path
            Input file path.
        fmt : FileFormat | None
            File format (auto-detected if None).
        text_column : str
            Column name containing text.
        encoding : str
            File encoding.
        """
        self._input_path = str(path)
        self._input_config = IngestConfig(
            format=fmt,
            text_column=text_column,
            encoding=encoding,
        )
        return self

    def from_records(
        self,
        records: list[dict[str, Any]],
        *,
        text_column: str = "text",
    ) -> "PipelineBuilder":
        """Set in-memory records as input.

        Parameters
        ----------
        records : list[dict]
            List of dicts with at least a text column.
        text_column : str
            Key containing the text to process.
        """
        self._input_records = [
            IngestRecord(
                record_id=idx,
                text=str(r.get(text_column, "")),
                metadata={k: v for k, v in r.items() if k != text_column},
            )
            for idx, r in enumerate(records)
        ]
        return self

    def from_dataframe(
        self,
        df: Any,
        *,
        text_column: str = "text",
    ) -> "PipelineBuilder":
        """Set a DataFrame as input.

        Parameters
        ----------
        df : DataFrame-like
            Any object with ``iterrows()`` or ``iter_rows()``.
        text_column : str
            Column name containing text.
        """
        self._input_dataframe = df
        self._df_text_column = text_column
        return self

    def to_file(
        self,
        path: str | Path,
        *,
        fmt: FileFormat | None = None,
    ) -> "PipelineBuilder":
        """Set file output destination.

        Parameters
        ----------
        path : str | Path
            Output file path.
        fmt : FileFormat | None
            Output format (auto-detected if None).
        """
        self._output_path = str(path)
        self._output_fmt = fmt
        return self

    def to_dataframe(self) -> "PipelineBuilder":
        """Collect results as a DataFrame (requires pandas)."""
        self._collect_dataframe = True
        return self

    def build(self) -> Pipeline:
        """Validate configuration and construct the pipeline.

        Returns
        -------
        Pipeline
            A configured pipeline ready for ``run()``.

        Raises
        ------
        ValueError
            If no input source is configured.
        """
        has_input = (
            self._input_records is not None
            or self._input_path is not None
            or self._input_dataframe is not None
        )
        if not has_input:
            raise ValueError(
                "No input source configured. Use from_file(), "
                "from_records(), or from_dataframe()."
            )

        return Pipeline(
            token_key=self._token_key,
            config=self._config,
            profile=self._profile,
            segmentation=self._segmentation,
            scope=self._scope,
            token_version=self._token_version,
            input_records=self._input_records,
            input_path=self._input_path,
            input_config=self._input_config,
            input_dataframe=self._input_dataframe,
            df_text_column=self._df_text_column,
            output_path=self._output_path,
            output_fmt=self._output_fmt,
            collect_dataframe=self._collect_dataframe,
            enable_evaluation=self._enable_evaluation,
            eval_framework=self._eval_framework,
        )
