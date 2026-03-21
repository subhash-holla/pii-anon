from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EngineRuntimeConfig(BaseModel):
    enabled: bool = True
    weight: float = 1.0
    timeout_ms: int = 1_000
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    entity_weights: dict[str, float] = Field(default_factory=dict)
    """Per-entity-type weight overrides for mixture-of-experts fusion.

    Maps entity type names to engine-specific weights.  During weighted
    consensus fusion the per-entity weight is used *instead of* the global
    ``weight`` for findings of that entity type.

    Example: ``{"PERSON_NAME": 1.8, "ORGANIZATION": 1.6}`` means this
    engine's votes count 1.8× for PERSON_NAME findings and 1.6× for
    ORGANIZATION findings, while all other entity types use ``weight``.
    """


class LoggingConfig(BaseModel):
    level: str = "INFO"
    structured: bool = True


class StreamConfig(BaseModel):
    enabled: bool = True
    max_chunk_tokens: int = 2048
    overlap_tokens: int = 128
    max_concurrency: int = 8
    large_text_threshold_tokens: int = 100_000


class ConfidenceConfig(BaseModel):
    """Tuning parameters for regex context-aware confidence adjustment."""
    context_boost: float = Field(default=0.10, description=(
        "Confidence increase when context keywords are found near the match."
    ))
    context_penalty: float = Field(default=0.15, description=(
        "Confidence decrease for high-false-positive entity types when "
        "context keywords are absent."
    ))
    context_window: int = Field(default=50, description=(
        "Number of characters (±) around the matched span to search for "
        "context keywords."
    ))
    confidence_cap: float = Field(default=0.99, description=(
        "Maximum confidence score after context boosting."
    ))
    confidence_floor: float = Field(default=0.40, description=(
        "Minimum confidence score after context penalty."
    ))


class FusionConfig(BaseModel):
    """Tuning parameters for multi-engine fusion strategies."""
    iou_threshold: float = Field(default=0.5, description=(
        "Minimum Intersection-over-Union for two entity spans to be "
        "considered overlapping and merged."
    ))
    min_gap_chars: int = Field(default=5, description=(
        "Character gap after which span overlap checking is skipped "
        "(performance optimization)."
    ))


class MoEConfig(BaseModel):
    """Configuration for Mixture-of-Experts ensemble."""
    top_k: int = Field(default=3, description=(
        "Number of experts to activate per entity type."
    ))
    performance_floor: bool = Field(default=True, description=(
        "Always include the best-performing expert for each entity type "
        "to guarantee ensemble >= best individual expert."
    ))
    min_expert_weight: float = Field(default=0.15, description=(
        "Minimum weight for the performance floor expert."
    ))
    iou_threshold: float = Field(default=0.5, description=(
        "IoU threshold for span overlap clustering."
    ))
    custom_experts: dict[str, dict[str, Any]] = Field(default_factory=dict, description=(
        "User-defined expert registrations."
    ))


class RiskConfig(BaseModel):
    """Thresholds for aggregate risk-level classification."""
    low_risk_threshold: float = Field(default=0.90, description=(
        "Confidence scores >= this are classified as 'low' risk."
    ))
    moderate_risk_threshold: float = Field(default=0.75, description=(
        "Confidence scores >= this (and < low_risk_threshold) are "
        "classified as 'moderate' risk. Below this is 'high' risk."
    ))


class RouterConfig(BaseModel):
    """Tuning parameters for the policy router's execution plans."""
    ensemble_confidence_threshold: float = Field(default=0.70, description=(
        "Low-confidence threshold for ensemble objective. Below this "
        "triggers escalation (if enabled)."
    ))
    accuracy_confidence_threshold: float = Field(default=0.88, description=(
        "Low-confidence threshold for accuracy objective."
    ))
    balanced_confidence_threshold: float = Field(default=0.80, description=(
        "Low-confidence threshold for balanced objective."
    ))
    ensemble_concurrency_cap: int = Field(default=8, description=(
        "Maximum number of engines to run concurrently in ensemble mode."
    ))
    accuracy_concurrency_cap: int = Field(default=4, description=(
        "Maximum concurrent engines for accuracy objective."
    ))
    balanced_concurrency_cap: int = Field(default=3, description=(
        "Maximum concurrent engines for balanced objective."
    ))
    segmentation_token_threshold: int = Field(default=2000, description=(
        "Minimum document token count to enable segmentation for long "
        "documents and multilingual mixes."
    ))


class BenchmarkConfig(BaseModel):
    regex_p50_ms: float = 100.0
    multi_engine_p50_ms: float = 500.0
    throughput_docs_per_hour: float = 10_000.0
    linear_scaling_r2: float = 0.95


class TransformConfig(BaseModel):
    default_mode: str = "pseudonymize"
    placeholder_template: str = "<{entity_type}:anon_{index}>"
    compliance_template: str | None = None
    entity_strategies: dict[str, str] = Field(default_factory=dict)
    strategy_params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TrackingConfig(BaseModel):
    enabled: bool = True
    min_link_score: float = 0.8
    allow_email_name_link: bool = True
    require_unique_short_name: bool = True


class DenyListConfig(BaseModel):
    enabled: bool = True
    lists: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "PERSON_NAME": [
                "new york", "san francisco", "los angeles", "united states",
                "north america", "south america", "east coast", "west coast",
                "great britain", "new zealand", "south africa", "north korea",
                "south korea", "united kingdom", "hong kong", "el salvador",
                "costa rica", "puerto rico", "sri lanka", "saudi arabia",
                "test user", "sample data", "john doe", "jane doe",
                # Organization names misdetected as person names
                "oscorp technologies", "massive dynamic", "pied piper inc",
                "prestige worldwide", "wayne enterprises", "weyland industries",
                "hooli technologies", "stark labs", "sterling cooper", "soylent inc",
                "globex industries", "vandelay industries", "cyberdyne systems",
                "dunder mifflin", "aperture science", "umbrella group",
                "tyrell corporation", "acme corp", "initech llc",
                # Address/location parts misdetected as person names
                "ash place", "birch court", "oak drive", "elm street",
                "maple avenue", "pine road", "cedar lane",
                # Common non-name phrases
                "investigation report", "current address", "bar license",
                "discharge summary", "primary subject", "audit period",
                "traumatic stress disorder",
            ],
        }
    )


class AllowListConfig(BaseModel):
    enabled: bool = False
    lists: dict[str, list[str]] = Field(default_factory=dict)


class CompetitorPolicyConfig(BaseModel):
    enabled: bool = True
    runtime_leverage_enabled: bool = True
    allowed_adapters: list[str] = Field(
        default_factory=lambda: ["spacy-ner-compatible", "stanza-ner-compatible"]
    )
    benchmark_adapters: list[str] = Field(
        default_factory=lambda: [
            "presidio",
            "scrubadub",
            "gliner",
        ]
    )


class CoreConfig(BaseModel):
    default_language: str = "en"
    auto_discover_engines: bool = False
    engines: dict[str, EngineRuntimeConfig] = Field(
        default_factory=lambda: {
            "regex-oss": EngineRuntimeConfig(enabled=True, weight=1.0),
            "presidio-compatible": EngineRuntimeConfig(enabled=False, weight=1.2),
            "llm-guard-compatible": EngineRuntimeConfig(enabled=False, weight=1.1),
            "scrubadub-compatible": EngineRuntimeConfig(enabled=False, weight=0.5),
            "spacy-ner-compatible": EngineRuntimeConfig(enabled=False, weight=0.5),
            "stanza-ner-compatible": EngineRuntimeConfig(enabled=False, weight=0.5),
            "gliner-compatible": EngineRuntimeConfig(enabled=False, weight=1.0),
        }
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    competitor_policy: CompetitorPolicyConfig = Field(default_factory=CompetitorPolicyConfig)
    deny_list: DenyListConfig = Field(default_factory=DenyListConfig)
    allow_list: AllowListConfig = Field(default_factory=AllowListConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    """Context-aware confidence adjustment parameters for the regex engine."""
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    """Multi-engine fusion strategy parameters."""
    moe: MoEConfig = Field(default_factory=MoEConfig)
    """Mixture-of-Experts ensemble parameters."""
    risk: RiskConfig = Field(default_factory=RiskConfig)
    """Risk-level classification thresholds."""
    router: RouterConfig = Field(default_factory=RouterConfig)
    """Policy router execution plan parameters."""

    @classmethod
    def default(cls) -> "CoreConfig":
        return cls()
