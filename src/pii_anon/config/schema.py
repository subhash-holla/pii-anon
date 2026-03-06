from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EngineRuntimeConfig(BaseModel):
    enabled: bool = True
    weight: float = 1.0
    timeout_ms: int = 1_000
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    structured: bool = True


class StreamConfig(BaseModel):
    enabled: bool = True
    max_chunk_tokens: int = 2048
    overlap_tokens: int = 128
    max_concurrency: int = 8
    large_text_threshold_tokens: int = 100_000


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
            "scrubadub-compatible": EngineRuntimeConfig(enabled=False, weight=0.9),
            "spacy-ner-compatible": EngineRuntimeConfig(enabled=False, weight=0.95),
            "stanza-ner-compatible": EngineRuntimeConfig(enabled=False, weight=0.95),
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

    @classmethod
    def default(cls) -> "CoreConfig":
        return cls()
