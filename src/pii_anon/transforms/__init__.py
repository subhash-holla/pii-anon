"""Pluggable transformation strategies for PII anonymization and pseudonymization.

This package provides a strategy pattern for transforming detected PII entities.
Each strategy implements a different anonymization/pseudonymization technique
(placeholder, tokenization, redaction, generalization, synthetic replacement,
perturbation) and can be configured per entity type.

Typical usage::

    from pii_anon.transforms import StrategyRegistry, RedactionStrategy

    registry = StrategyRegistry()
    registry.register(RedactionStrategy())
    result = registry.get("redact").transform("John Smith", "PERSON_NAME", context)
"""

from pii_anon.transforms.base import (
    StrategyMetadata,
    TransformContext,
    TransformResult,
    TransformStrategy,
)
from pii_anon.transforms.registry import StrategyRegistry
from pii_anon.transforms.strategies import (
    GeneralizationStrategy,
    PerturbationStrategy,
    PlaceholderStrategy,
    RedactionStrategy,
    SyntheticReplacementStrategy,
    TokenizationStrategy,
)

__all__ = [
    "GeneralizationStrategy",
    "PerturbationStrategy",
    "PlaceholderStrategy",
    "RedactionStrategy",
    "StrategyMetadata",
    "StrategyRegistry",
    "SyntheticReplacementStrategy",
    "TokenizationStrategy",
    "TransformContext",
    "TransformResult",
    "TransformStrategy",
]
