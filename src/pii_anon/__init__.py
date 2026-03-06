from .benchmarks import BenchmarkSummary, run_benchmark
from .config import ConfigManager, CoreConfig
from .errors import ConfigurationError, EngineExecutionError, FusionError, TokenizationError
from .evaluation import (
    CompetitorComparisonReport,
    FloorCheckResult,
    PipelineEvaluationReport,
    ProfileBenchmarkResult,
    StrategyEvaluationReport,
    StrategyEvaluator,
    SystemBenchmarkResult,
    compare_competitors,
    evaluate_pipeline,
)
from .ingestion import (  # noqa: F401  — public re-export
    FileFormat,
    FileIngestResult,
    IngestConfig,
    IngestRecord,
    read_file,
    read_dataframe,
    results_to_dataframe,
    write_results,
)
from .orchestrator import AsyncPIIOrchestrator, PIIOrchestrator
from .segmentation import StreamingChunker, estimate_token_count
from .types import (
    ConfidenceEnvelope,
    EngineCapabilities,
    EngineFinding,
    ProcessingProfileSpec,
    FusionAuditRecord,
    GuaranteeProfile,
    GuaranteeReport,
    SegmentationPlan,
    StrategyComparisonResult,
)

# Evaluation Framework — truly lazy imports.  The eval_framework sub-package
# is heavy (~150+ classes, taxonomy data, metrics code).  Only load it when
# one of its names is actually accessed.
_EVAL_FRAMEWORK_NAMES = {
    "EvaluationFramework",
    "EvaluationFrameworkConfig",
    "PII_TAXONOMY",
    "SUPPORTED_LANGUAGES",
    "EntityTypeRegistry",
    "ComplianceValidator",
    "CompositeConfig",
    "CompositeScore",
    "compute_composite",
    "PIIRateEloEngine",
    "EloRating",
    "Leaderboard",
}


def __getattr__(name: str) -> object:  # noqa: N807
    if name in _EVAL_FRAMEWORK_NAMES:
        from . import eval_framework as _ef  # noqa: F811
        return getattr(_ef, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# v1.0.0: Transformation strategies & policies
from .transforms import (  # noqa: E402, F401  — public re-export (after __getattr__)
    TransformStrategy,
    TransformContext,
    TransformResult,
    StrategyMetadata,
    StrategyRegistry,
    PlaceholderStrategy,
    TokenizationStrategy,
    RedactionStrategy,
    GeneralizationStrategy,
    SyntheticReplacementStrategy,
    PerturbationStrategy,
)
from .transforms.policies import (  # noqa: E402, F401  — public re-export
    EntityTransformRule,
    TransformPolicy,
    load_compliance_template,
    list_compliance_templates,
)

# v1.0.0: Eval framework bridge
from .bridge import (  # noqa: E402, F401  — public re-export
    ResultAdapter,
    EvaluationPipeline,
    EvaluationPipelineConfig,
    QuickBench,
    QuickBenchReport,
)

# v1.0.0: Enterprise pseudonymization
from .tokenization import (  # noqa: E402, F401  — public re-export
    KeyManager,
    KeyVersion,
    ReidentificationService,
    ReidentificationAuditEntry,
    TokenStore,
    TokenMapping,
    InMemoryTokenStore,
    SQLiteTokenStore,
)

# v1.0.0: Pipeline builder
from .pipeline import (  # noqa: E402, F401  — public re-export
    PipelineBuilder,
    Pipeline,
    PipelineReport,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # Core orchestration
    "AsyncPIIOrchestrator",
    "PIIOrchestrator",
    "ConfigManager",
    "CoreConfig",
    # Existing evaluation / benchmarks
    "StrategyEvaluator",
    "StrategyEvaluationReport",
    "PipelineEvaluationReport",
    "SystemBenchmarkResult",
    "FloorCheckResult",
    "ProfileBenchmarkResult",
    "CompetitorComparisonReport",
    "compare_competitors",
    "evaluate_pipeline",
    "run_benchmark",
    "BenchmarkSummary",
    # Types
    "ProcessingProfileSpec",
    "EngineFinding",
    "EngineCapabilities",
    "FusionAuditRecord",
    "StrategyComparisonResult",
    "ConfidenceEnvelope",
    "GuaranteeProfile",
    "GuaranteeReport",
    "SegmentationPlan",
    # Errors
    "ConfigurationError",
    "EngineExecutionError",
    "FusionError",
    "TokenizationError",
    # Evaluation Framework
    "EvaluationFramework",
    "EvaluationFrameworkConfig",
    "PII_TAXONOMY",
    "SUPPORTED_LANGUAGES",
    "EntityTypeRegistry",
    "ComplianceValidator",
    # File ingestion
    "FileFormat",
    "FileIngestResult",
    "IngestConfig",
    "IngestRecord",
    "read_file",
    "read_dataframe",
    "results_to_dataframe",
    "write_results",
    # Streaming chunker
    "StreamingChunker",
    "estimate_token_count",
    # v1.0.0: Composite metric & rating engine
    "CompositeConfig",
    "CompositeScore",
    "compute_composite",
    "PIIRateEloEngine",
    "EloRating",
    "Leaderboard",
    # v1.0.0: Transformation strategies
    "TransformStrategy",
    "TransformContext",
    "TransformResult",
    "StrategyMetadata",
    "StrategyRegistry",
    "PlaceholderStrategy",
    "TokenizationStrategy",
    "RedactionStrategy",
    "GeneralizationStrategy",
    "SyntheticReplacementStrategy",
    "PerturbationStrategy",
    # v1.0.0: Transformation policies
    "EntityTransformRule",
    "TransformPolicy",
    "load_compliance_template",
    "list_compliance_templates",
    # v1.0.0: Eval framework bridge
    "ResultAdapter",
    "EvaluationPipeline",
    "EvaluationPipelineConfig",
    "QuickBench",
    "QuickBenchReport",
    # v1.0.0: Enterprise pseudonymization
    "KeyManager",
    "KeyVersion",
    "ReidentificationService",
    "ReidentificationAuditEntry",
    "TokenStore",
    "TokenMapping",
    "InMemoryTokenStore",
    "SQLiteTokenStore",
    # v1.0.0: Pipeline builder
    "PipelineBuilder",
    "Pipeline",
    "PipelineReport",
]
