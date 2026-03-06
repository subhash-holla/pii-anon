"""Core data structures for PII detection, fusion, and transformation.

This module defines the canonical data types used throughout pii-anon:

- **Detection**: ``EngineFinding`` (individual engine results), ``EnsembleFinding``
  (merged multi-engine results).
- **Audit trails**: ``FusionAuditRecord`` (fusion decisions), ``ConfidenceEnvelope``
  (aggregate confidence metrics).
- **Configuration**: ``ProcessingProfileSpec`` (detection parameters),
  ``SegmentationPlan`` (text chunking strategy).
- **Capabilities and guarantees**: ``EngineCapabilities``, ``GuaranteeProfile``,
  ``GuaranteeReport``.

Type Aliases
~~~~~~~~~~~~
- ``FusionMode``: selection of fusion strategy (union, consensus, etc.)
- ``PolicyMode``: trade-off strategy (recall vs precision)
- ``RiskLevel``: confidence-based risk classification (low/moderate/high)
- ``TransformMode``: output format (pseudonymization vs anonymization)
- ``Payload``: dict of field names to scalar values (text, numbers, bools)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ── Type Aliases ───────────────────────────────────────────────────────────

FusionMode = Literal[
    "union_high_recall",
    "weighted_consensus",
    "calibrated_majority",
    "intersection_consensus",
] | str
"""Fusion strategy selection.

- ``union_high_recall``: emit all findings (highest recall, no merging).
- ``weighted_consensus``: merge findings by location, weight by engine.
- ``calibrated_majority``: require N engines to agree (precision-focused).
- ``intersection_consensus``: strict multi-engine agreement.
"""

PolicyMode = Literal["recall_max", "balanced", "precision_guarded"]
"""Execution policy trade-off.

- ``recall_max``: prioritize detection over false positives.
- ``balanced``: balance recall and precision (default).
- ``precision_guarded``: accept missed findings to avoid false positives.
"""

RiskLevel = Literal["low", "moderate", "high"]
"""Confidence-based risk classification."""

TransformMode = Literal[
    "pseudonymize", "anonymize", "redact", "generalize", "synthetic", "perturb",
] | str
"""Output transformation mode.

- ``pseudonymize``: replace with reversible tokens.
- ``anonymize``: replace with non-reversible placeholders.
- ``redact``: mask characters (full, partial, length-preserving).
- ``generalize``: reduce precision (age→range, zip→prefix).
- ``synthetic``: generate realistic fake values.
- ``perturb``: add calibrated noise to numerical values.
"""

PayloadScalar = str | int | float | bool | None
"""Scalar types allowed in payloads."""

Payload = dict[str, PayloadScalar]
"""Input payload: field names to scalar values (typically text)."""


@dataclass(slots=True)
class EngineFinding:
    """Single PII detection from one engine.

    Represents the atomic result of a detection engine. Multiple engine findings
    are later merged by fusion strategies into ``EnsembleFinding`` instances.

    Attributes
    ----------
    entity_type : str
        Classification (e.g., "EMAIL", "PERSON_NAME", "CREDIT_CARD").
    confidence : float
        Detection confidence in [0, 1]. Higher = more confident.
    field_path : str | None
        Name of the field scanned (e.g., "email", "notes"). *None* for
        unstructured text.
    span_start : int | None
        Start character offset in the field value (0-indexed).
    span_end : int | None
        End character offset (exclusive).
    explanation : str | None
        Human-readable rationale (e.g., "regex email pattern (context-boosted)").
    engine_id : str
        Identifier of the engine that produced this finding (e.g., "regex-oss").
    language : str
        Language code (e.g., "en", "es", "fr").
    """
    entity_type: str
    confidence: float
    field_path: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    explanation: str | None = None
    engine_id: str = "unknown"
    language: str = "en"


@dataclass(slots=True)
class EnsembleFinding:
    """Merged PII detection from one or more engines.

    The output of a fusion strategy. Represents a deduplicated, consensus-based
    detection that may incorporate signals from multiple engines.

    Attributes
    ----------
    entity_type : str
        Classification (e.g., "EMAIL", "PERSON_NAME").
    confidence : float
        Merged confidence in [0, 1]. May be a weighted average, minimum,
        or custom aggregate depending on fusion strategy.
    engines : list[str]
        IDs of engines that contributed to this finding.
    field_path : str | None
        Name of the field where entity was found.
    span_start : int | None
        Start character offset in the field.
    span_end : int | None
        End character offset (exclusive).
    explanation : str | None
        Rationale (e.g., fusion strategy details).
    language : str
        Language code.
    """
    entity_type: str
    confidence: float
    engines: list[str]
    field_path: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    explanation: str | None = None
    language: str = "en"


@dataclass(slots=True)
class FusionAuditRecord:
    """Audit trail for a fusion decision.

    Documents how an ``EnsembleFinding`` was derived from raw engine outputs.
    Enables traceability and debugging of fusion choices.

    Attributes
    ----------
    strategy : str
        Fusion strategy used (e.g., "weighted_consensus").
    entity_type : str
        The entity type of the finding.
    field_path : str | None
        The field where the entity was found.
    span_start : int | None
        Start character offset.
    span_end : int | None
        End character offset.
    source_engines : list[str]
        Engines that contributed to this finding.
    source_count : int
        Number of raw findings merged to create this ensemble finding.
    fused_confidence : float
        The final merged confidence score.
    notes : list[str]
        Optional notes (e.g., "consensus threshold not met").
    """
    strategy: str
    entity_type: str
    field_path: str | None
    span_start: int | None
    span_end: int | None
    source_engines: list[str]
    source_count: int
    fused_confidence: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ConfidenceEnvelope:
    """Aggregate confidence metrics for a batch of findings.

    Summarizes the overall confidence and risk profile of detection results,
    useful for threshold-based decision making and audit reporting.

    Attributes
    ----------
    score : float
        Macro-average confidence in [0, 1] across all findings.
    risk_level : RiskLevel
        Classification ("low", "moderate", "high") based on confidence.
    contributors : list[str]
        Engines that contributed findings.
    notes : list[str]
        Contextual notes (e.g., "No findings detected").
    by_entity_type : dict[str, float]
        Per-entity-type average confidence (e.g., {"EMAIL": 0.92, ...}).
    """
    score: float
    risk_level: RiskLevel
    contributors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    by_entity_type: dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessingProfileSpec:
    """Configuration for detection and transformation behavior.

    A profile encapsulates all the parameters needed to run a detection job:
    which engines to use, how to merge results, how to transform output, etc.

    Attributes
    ----------
    profile_id : str
        Human-readable profile name.
    mode : FusionMode
        Fusion strategy (default "weighted_consensus").
    engine_weights : dict[str, float]
        Per-engine confidence weights. Default weight is 1.0.
    min_consensus : int
        Minimum number of engines required to agree (used by some fusion modes).
    policy_mode : PolicyMode
        Trade-off strategy ("recall_max", "balanced", "precision_guarded").
    language : str
        Language code (e.g., "en").
    use_case : str
        Use case label for traceability (e.g., "pii_redaction", "compliance_audit").
    objective : Literal["accuracy", "balanced", "speed"]
        Optimization objective. Affects engine selection and execution strategy.
    transform_mode : TransformMode
        Output format ("pseudonymize" or "anonymize").
    placeholder_template : str
        Format string for anonymization (e.g., "<{entity_type}:anon_{index}>").
    entity_tracking_enabled : bool
        Whether to track entity mentions and clusters for consistency.
    use_external_competitors : bool
        Whether to allow external/competitor engines (e.g., Spacy, Stanza).
    external_competitor_allowlist : list[str]
        Explicit allowlist of external engines (overrides policy).
    """
    profile_id: str
    mode: FusionMode = "weighted_consensus"
    engine_weights: dict[str, float] = field(default_factory=dict)
    min_consensus: int = 1
    policy_mode: PolicyMode = "balanced"
    language: str = "en"
    use_case: str = "default"
    objective: Literal["accuracy", "balanced", "speed"] = "balanced"
    transform_mode: TransformMode = "pseudonymize"
    placeholder_template: str = "<{entity_type}:anon_{index}>"
    entity_tracking_enabled: bool = True
    use_external_competitors: bool = True
    external_competitor_allowlist: list[str] = field(default_factory=list)
    entity_strategies: dict[str, str] = field(default_factory=dict)
    """Per-entity-type strategy override.

    Maps entity type names to strategy IDs, e.g.
    ``{"EMAIL_ADDRESS": "generalize", "AGE": "perturb"}``.
    Overrides the global ``transform_mode`` for specified types.
    """
    strategy_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-strategy configuration parameters.

    Maps strategy IDs to parameter dicts, e.g.
    ``{"perturb": {"epsilon": 1.0}, "redact": {"mode": "partial_start"}}``.
    """


@dataclass
class SegmentationPlan:
    """Configuration for text chunking (segmentation) strategy.

    When processing very long documents, text is split into overlapping
    segments to manage memory and execution time. Overlaps ensure findings
    at segment boundaries are not missed.

    Attributes
    ----------
    enabled : bool
        Whether segmentation is active (default False).
    max_tokens : int
        Maximum tokens per segment (approximate, default 4096).
    overlap_tokens : int
        Overlap size between segments to catch boundary findings (default 128).
    key_field : str
        Field to segment (typically "text").
    """
    enabled: bool = False
    max_tokens: int = 4096
    overlap_tokens: int = 128
    key_field: str = "text"


@dataclass
class BoundaryReconciliationTrace:
    """Audit trail for boundary reconciliation after segmentation.

    Records statistics about how findings at segment boundaries were merged
    and deduplicated.

    Attributes
    ----------
    segments_processed : int
        Number of text segments created.
    overlap_tokens : int
        Token count of overlap regions between segments.
    merged_spans : int
        Number of spans merged across segment boundaries.
    deduped_findings : int
        Number of duplicate findings removed.
    """
    segments_processed: int
    overlap_tokens: int
    merged_spans: int
    deduped_findings: int


@dataclass
class EngineCapabilities:
    """Capability declaration for a detection engine.

    Engines report what they can do: which languages they support, whether
    they can handle streaming, and if they accept runtime configuration.

    Attributes
    ----------
    adapter_id : str
        Engine identifier.
    native_dependency : str | None
        Name of the required Python package (e.g., "presidio_analyzer").
    dependency_available : bool
        Whether the dependency is installed.
    supports_languages : list[str]
        List of language codes the engine supports (default ["en"]).
    supports_streaming : bool
        Whether the engine can process streaming inputs (default False).
    supports_runtime_configuration : bool
        Whether the engine accepts runtime config dicts (default True).
    """
    adapter_id: str
    native_dependency: str | None
    dependency_available: bool
    supports_languages: list[str] = field(default_factory=lambda: ["en"])
    supports_streaming: bool = False
    supports_runtime_configuration: bool = True


@dataclass
class GuaranteeProfile:
    """Privacy guarantee specification for evaluation and compliance.

    Defines the acceptable privacy risk thresholds for a use case.
    Used by the guarantee system to validate that processing meets
    compliance requirements.

    Attributes
    ----------
    profile_id : str
        Guarantee profile name.
    taxonomy : list[str]
        Entity types covered by this guarantee (e.g., ["EMAIL", "PERSON_NAME"]).
    language : str
        Language code.
    max_leakage_rate : float
        Maximum fraction of entities that can be missed (default 0.02 = 2%).
    min_token_stability : float
        Minimum consistency of token assignments across invocations
        (default 0.98).
    max_llm_leakage : float
        Maximum fraction of PII that can be leaked to LLM outputs
        (default 0.01 = 1%).
    """
    profile_id: str
    taxonomy: list[str]
    language: str = "en"
    max_leakage_rate: float = 0.02
    min_token_stability: float = 0.98
    max_llm_leakage: float = 0.01


@dataclass
class GuaranteeReport:
    """Result of evaluating processing against a guarantee profile.

    Documents whether privacy guarantees were met and provides detailed
    metrics and confidence intervals for compliance audits.

    Attributes
    ----------
    evaluation_id : str
        Unique evaluation run identifier.
    profile_id : str
        The guarantee profile that was evaluated.
    assumptions : dict[str, Any]
        Assumptions made during evaluation (e.g., model, dataset).
    confidence_intervals : dict[str, tuple[float, float]]
        Per-metric 95% confidence intervals (lower, upper).
    metrics : dict[str, float]
        Measured metrics (e.g., "leakage_rate": 0.015).
    passed : bool
        Whether all thresholds were met.
    failure_buckets : list[dict[str, Any]]
        Categorized violations (e.g., entity types, documents failed).
    """
    evaluation_id: str
    profile_id: str
    assumptions: dict[str, Any]
    confidence_intervals: dict[str, tuple[float, float]]
    metrics: dict[str, float]
    passed: bool
    failure_buckets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StrategyComparisonResult:
    """Evaluation metrics for a fusion strategy on a dataset.

    Used for strategy benchmarking and selection. Provides F-beta scores,
    detection counts, and average confidence.

    Attributes
    ----------
    strategy : str
        Fusion strategy name.
    span_fbeta : float
        F-beta score on entity span prediction (default beta=1).
    findings_count : int
        Total number of findings merged by this strategy.
    avg_confidence : float
        Average confidence of all findings.
    """
    strategy: str
    span_fbeta: float
    findings_count: int
    avg_confidence: float
