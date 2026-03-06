"""Bridge between the anonymization orchestrator and the evaluation framework.

Provides adapters that convert orchestrator output (``EnsembleFinding``) to
evaluation framework input (``LabeledSpan``) and vice versa, plus high-level
pipeline classes that chain detection → transformation → evaluation.

Usage::

    from pii_anon.bridge import ResultAdapter, EvaluationPipeline, QuickBench

    # Convert orchestrator findings to eval-framework spans
    spans = ResultAdapter.findings_to_spans(result["ensemble_findings"])

    # Run a full detection+evaluation pipeline
    pipeline = EvaluationPipeline(orchestrator, framework, profile)
    report = pipeline.evaluate_record(text, ground_truth, language="en")

    # One-liner benchmark
    report = QuickBench.run(orchestrator, dataset="pii_anon_eval_v1")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pii_anon.eval_framework.metrics.base import LabeledSpan
from pii_anon.types import (
    EnsembleFinding,
    Payload,
    ProcessingProfileSpec,
    SegmentationPlan,
)


class ResultAdapter:
    """Convert between orchestrator and evaluation framework data structures."""

    @staticmethod
    def findings_to_spans(
        findings: list[dict[str, Any]] | list[EnsembleFinding],
        record_id: str = "",
    ) -> list[LabeledSpan]:
        """Convert orchestrator findings to evaluation ``LabeledSpan`` instances.

        Parameters
        ----------
        findings : list
            Either a list of ``EnsembleFinding`` objects or dicts from
            the orchestrator's JSON output.
        record_id : str
            Record identifier to assign to all spans.

        Returns
        -------
        list[LabeledSpan]
            Spans suitable for the evaluation framework.
        """
        spans: list[LabeledSpan] = []
        for finding in findings:
            if isinstance(finding, dict):
                entity_type = finding.get("entity_type", "")
                span_start = finding.get("span_start")
                span_end = finding.get("span_end")
            else:
                entity_type = finding.entity_type
                span_start = finding.span_start
                span_end = finding.span_end

            if span_start is not None and span_end is not None:
                spans.append(
                    LabeledSpan(
                        entity_type=entity_type,
                        start=int(span_start),
                        end=int(span_end),
                        record_id=record_id,
                    )
                )
        return spans

    @staticmethod
    def spans_to_findings(
        spans: list[LabeledSpan],
        engine_id: str = "external",
        confidence: float = 1.0,
    ) -> list[EnsembleFinding]:
        """Convert evaluation ``LabeledSpan`` instances to ``EnsembleFinding``.

        Useful for importing external annotations into the orchestrator's
        data model.

        Parameters
        ----------
        spans : list[LabeledSpan]
            Evaluation framework spans.
        engine_id : str
            Engine ID to assign to all findings.
        confidence : float
            Default confidence score.

        Returns
        -------
        list[EnsembleFinding]
            Orchestrator-compatible findings.
        """
        return [
            EnsembleFinding(
                entity_type=span.entity_type,
                confidence=confidence,
                engines=[engine_id],
                span_start=span.start,
                span_end=span.end,
            )
            for span in spans
        ]

    @staticmethod
    def labels_from_record(record: Any) -> list[LabeledSpan]:
        """Extract ground truth LabeledSpans from an EvalBenchmarkRecord.

        Parameters
        ----------
        record : EvalBenchmarkRecord
            A benchmark record with ``labels`` field.

        Returns
        -------
        list[LabeledSpan]
            Ground truth spans.
        """
        spans: list[LabeledSpan] = []
        labels = getattr(record, "labels", [])
        record_id = getattr(record, "record_id", "")
        for label in labels:
            if isinstance(label, dict):
                entity_type = label.get("entity_type", "")
                start = label.get("start", 0)
                end = label.get("end", 0)
            else:
                entity_type = getattr(label, "entity_type", "")
                start = getattr(label, "start", 0)
                end = getattr(label, "end", 0)
            spans.append(
                LabeledSpan(
                    entity_type=entity_type,
                    start=int(start),
                    end=int(end),
                    record_id=str(record_id),
                )
            )
        return spans


@dataclass
class EvaluationPipelineConfig:
    """Configuration for the evaluation pipeline.

    Attributes
    ----------
    profile : ProcessingProfileSpec
        Detection and transformation profile.
    segmentation : SegmentationPlan
        Segmentation settings.
    scope : str
        Tokenization scope.
    token_version : int
        Token version.
    include_transform_context : bool
        Whether to pass original/anonymized text to privacy metrics.
    """

    profile: ProcessingProfileSpec = field(
        default_factory=lambda: ProcessingProfileSpec(profile_id="eval_pipeline")
    )
    segmentation: SegmentationPlan = field(default_factory=SegmentationPlan)
    scope: str = "eval"
    token_version: int = 1
    include_transform_context: bool = True


class EvaluationPipeline:
    """High-level pipeline: detect → transform → evaluate.

    Bridges the orchestrator (detection engine) with the evaluation
    framework, handling all data format conversions automatically.

    Parameters
    ----------
    orchestrator : PIIOrchestrator | AsyncPIIOrchestrator
        The PII orchestrator instance.
    framework : EvaluationFramework
        The evaluation framework instance.
    config : EvaluationPipelineConfig | None
        Pipeline configuration.
    """

    def __init__(
        self,
        orchestrator: Any,
        framework: Any,
        config: EvaluationPipelineConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.framework = framework
        self.config = config or EvaluationPipelineConfig()

    def evaluate_record(
        self,
        text: str,
        ground_truth: list[LabeledSpan],
        *,
        language: str = "en",
        record_id: str = "",
    ) -> Any:
        """Detect PII in text and evaluate against ground truth.

        Parameters
        ----------
        text : str
            Input text to process.
        ground_truth : list[LabeledSpan]
            Ground truth annotations.
        language : str
            Language code.
        record_id : str
            Record identifier for reporting.

        Returns
        -------
        EvaluationReport
            Comprehensive evaluation report.
        """
        cfg = self.config
        payload: Payload = {"text": text}

        # Run orchestrator
        result = self.orchestrator.run(
            payload,
            profile=cfg.profile,
            segmentation=cfg.segmentation,
            scope=cfg.scope,
            token_version=cfg.token_version,
        )

        # Convert findings to LabeledSpans
        raw_findings = result.get("ensemble_findings", [])
        predictions = ResultAdapter.findings_to_spans(raw_findings, record_id=record_id)

        # Build context for privacy metrics
        context: dict[str, Any] | None = None
        if cfg.include_transform_context:
            transformed = result.get("transformed_payload", {})
            anonymized_text = transformed.get("text", "")
            context = {
                "original_text": text,
                "anonymized_text": anonymized_text,
            }

        # Evaluate
        return self.framework.evaluate(
            predictions=predictions,
            labels=ground_truth,
            language=language,
            context=context,
        )

    def evaluate_dataset(
        self,
        records: list[Any],
        *,
        language: str | None = None,
    ) -> Any:
        """Evaluate the orchestrator on a full dataset.

        Parameters
        ----------
        records : list[EvalBenchmarkRecord]
            Dataset records with text and labels.
        language : str | None
            Override language for all records (None = use per-record).

        Returns
        -------
        BatchEvaluationReport
            Aggregated results across all records.
        """
        def predict_fn(record: Any) -> list[LabeledSpan]:
            payload: Payload = {"text": record.text}
            cfg = self.config
            result = self.orchestrator.run(
                payload,
                profile=cfg.profile,
                segmentation=cfg.segmentation,
                scope=cfg.scope,
                token_version=cfg.token_version,
            )
            raw_findings = result.get("ensemble_findings", [])
            return ResultAdapter.findings_to_spans(
                raw_findings, record_id=str(getattr(record, "record_id", ""))
            )

        return self.framework.evaluate_batch(
            records=records,
            predict_fn=predict_fn,
        )


@dataclass
class QuickBenchReport:
    """Report from a QuickBench run.

    Attributes
    ----------
    records_evaluated : int
        Number of records processed.
    micro_f1 : float
        Micro-averaged F1 score.
    macro_f1 : float
        Macro-averaged F1 score.
    precision : float
        Micro-averaged precision.
    recall : float
        Micro-averaged recall.
    duration_seconds : float
        Total wall-clock time.
    batch_report : Any
        Full BatchEvaluationReport from the framework.
    """

    records_evaluated: int = 0
    micro_f1: float = 0.0
    macro_f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    duration_seconds: float = 0.0
    batch_report: Any = None


class QuickBench:
    """Convenience class for one-liner benchmarking.

    Usage::

        report = QuickBench.run(
            orchestrator=orch,
            dataset="pii_anon_eval_v1",
            language="en",
        )
        print(f"F1: {report.micro_f1:.3f}")
    """

    @staticmethod
    def run(
        orchestrator: Any,
        *,
        dataset: str = "pii_anon_eval_v1",
        language: str | None = None,
        profile: ProcessingProfileSpec | None = None,
        max_records: int | None = None,
    ) -> QuickBenchReport:
        """Run a quick benchmark of the orchestrator on a dataset.

        Parameters
        ----------
        orchestrator : PIIOrchestrator
            The orchestrator to benchmark.
        dataset : str
            Dataset name to load.
        language : str | None
            Filter dataset by language.
        profile : ProcessingProfileSpec | None
            Processing profile (uses default if None).
        max_records : int | None
            Limit number of records (None = all).

        Returns
        -------
        QuickBenchReport
            Summary of benchmark results.
        """
        from pii_anon.eval_framework import EvaluationFramework
        from pii_anon.eval_framework.datasets.schema import load_eval_dataset

        # Load dataset
        records = load_eval_dataset(dataset, language=language)
        if max_records is not None:
            records = records[:max_records]

        # Set up pipeline
        framework = EvaluationFramework()
        pipeline_config = EvaluationPipelineConfig(
            profile=profile or ProcessingProfileSpec(profile_id="quickbench"),
        )
        pipeline = EvaluationPipeline(orchestrator, framework, pipeline_config)

        # Run evaluation
        start = time.monotonic()
        batch_report = pipeline.evaluate_dataset(records, language=language)
        duration = time.monotonic() - start

        # Extract summary metrics
        micro = getattr(batch_report, "micro_averaged", {})

        return QuickBenchReport(
            records_evaluated=getattr(batch_report, "records_evaluated", len(records)),
            micro_f1=micro.get("f1", 0.0),
            macro_f1=getattr(batch_report, "macro_averaged", {}).get("f1", 0.0),
            precision=micro.get("precision", 0.0),
            recall=micro.get("recall", 0.0),
            duration_seconds=round(duration, 3),
            batch_report=batch_report,
        )
