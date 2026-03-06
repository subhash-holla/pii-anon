"""Abstract base class and data structures for transformation strategies.

The ``TransformStrategy`` defines the contract that all PII transformation
strategies must implement.  Strategies are responsible for converting a
detected PII plaintext into a privacy-preserving replacement.

Strategy Lifecycle
------------------
1. **Registration**: strategies are registered in a ``StrategyRegistry``.
2. **Resolution**: the orchestrator resolves which strategy to use per entity
   type, using the processing profile's ``entity_strategies`` mapping.
3. **Transformation**: ``transform()`` receives plaintext, entity type, and
   full context, returning a ``TransformResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pii_anon.types import EnsembleFinding


@dataclass(frozen=True)
class TransformContext:
    """Context passed to a strategy during transformation.

    Contains all information a strategy might need to make decisions:
    the detected entity, its position, the surrounding text, language,
    and entity-tracking metadata.

    Attributes
    ----------
    entity_type : str
        Canonical entity type (e.g. ``"PERSON_NAME"``, ``"EMAIL_ADDRESS"``).
    plaintext : str
        The original PII value to transform.
    field_path : str | None
        Payload field where the entity was found.
    language : str
        ISO 639-1 language code.
    scope : str
        Tokenization scope (e.g. user ID).
    finding : EnsembleFinding
        The full ensemble detection result.
    cluster_id : str
        Identity cluster this mention belongs to.
    placeholder_index : int
        Monotonic index within the cluster (for placeholder numbering).
    is_first_mention : bool
        Whether this is the first mention of this cluster in the document.
    mention_index : int
        0-based position of this mention within the document.
    document_text : str
        The full document text (for strategies that need broader context).
    token_key : str
        Secret key for cryptographic strategies.
    token_version : int
        Token version for versioned pseudonymization.
    strategy_params : dict[str, Any]
        Strategy-specific parameters from the processing profile.
    """

    entity_type: str
    plaintext: str
    field_path: str | None = None
    language: str = "en"
    scope: str = "default"
    finding: EnsembleFinding | None = None
    cluster_id: str = ""
    placeholder_index: int = 0
    is_first_mention: bool = True
    mention_index: int = 0
    document_text: str = ""
    token_key: str = ""
    token_version: int = 1
    strategy_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformResult:
    """Result of a transformation operation.

    Attributes
    ----------
    replacement : str
        The privacy-preserving replacement text.
    strategy_id : str
        ID of the strategy that produced this result.
    is_reversible : bool
        Whether the transformation can be reversed.
    metadata : dict[str, Any]
        Strategy-specific audit information (e.g. noise budget spent,
        generalization level, synthetic source locale).
    """

    replacement: str
    strategy_id: str
    is_reversible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyMetadata:
    """Capability declaration for a transformation strategy.

    Attributes
    ----------
    strategy_id : str
        Unique identifier (e.g. ``"redact"``, ``"generalize"``).
    description : str
        Human-readable description of what the strategy does.
    reversible : bool
        Whether transformations can be undone.
    format_preserving : bool
        Whether output maintains the same format as input
        (e.g. email → email, phone → phone).
    supports_entity_types : list[str] | None
        Entity types this strategy supports. ``None`` means all types.
    """

    strategy_id: str
    description: str
    reversible: bool = False
    format_preserving: bool = False
    supports_entity_types: list[str] | None = None


class TransformStrategy(ABC):
    """Abstract base class for PII transformation strategies.

    Subclasses must implement ``transform()`` and set ``strategy_id``.
    The optional ``is_reversible()``, ``supports_entity_types()``, and
    ``metadata()`` methods provide capability declarations.

    Attributes
    ----------
    strategy_id : str
        Unique identifier for this strategy.
    """

    strategy_id: str = "unknown"

    @abstractmethod
    def transform(
        self,
        plaintext: str,
        entity_type: str,
        context: TransformContext,
    ) -> TransformResult:
        """Transform a PII plaintext into a privacy-preserving replacement.

        Parameters
        ----------
        plaintext : str
            The original PII value.
        entity_type : str
            Entity type classification.
        context : TransformContext
            Full transformation context.

        Returns
        -------
        TransformResult
            The replacement text and audit metadata.
        """
        raise NotImplementedError

    def is_reversible(self) -> bool:
        """Whether this strategy produces reversible transformations."""
        return False

    def supports_entity_types(self) -> list[str] | None:
        """Entity types supported by this strategy, or None for all."""
        return None

    def metadata(self) -> StrategyMetadata:
        """Return capability metadata for this strategy."""
        return StrategyMetadata(
            strategy_id=self.strategy_id,
            description=f"Transform strategy: {self.strategy_id}",
            reversible=self.is_reversible(),
            supports_entity_types=self.supports_entity_types(),
        )
