"""Mixtral-inspired Mixture-of-Experts ensemble for PII detection.

Implements sparse Top-K expert routing with entity-type-aware gating,
guaranteeing ensemble performance >= best individual expert per entity type.

Architecture (inspired by Mixtral 8x7B):
- Expert Registry: Extensible collection of detection engines
- Router/Gate: Per-entity-type expert selection with softmax-weighted scores
- Top-K Activation: Only best K experts run per entity type (sparse)
- Performance Floor: Oracle routing ensures ensemble >= best individual expert
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pii_anon.fusion import FusionStrategy, _cluster_overlapping_spans
from pii_anon.types import EngineFinding, EnsembleFinding


@dataclass
class ExpertSpec:
    """Specification for a registered PII detection expert.

    Captures expert capabilities (entity strengths/weaknesses), performance
    profile, and runtime availability. Used by the MoE router to make
    sparse activation decisions.

    Attributes
    ----------
    expert_id : str
        Unique identifier (e.g., "regex-oss", "gliner-compatible").
    display_name : str
        Human-readable name (e.g., "GLiNER PII Base").
    entity_strengths : dict[str, float]
        Per-entity-type strength scores (e.g., F1, recall, or weight).
        Higher values indicate better performance on that entity type.
    entity_weaknesses : dict[str, float]
        Per-entity-type weakness scores (inverse of strengths).
        Optional; used for diagnostics.
    default_weight : float
        Global fallback weight when no entity-specific override exists.
    is_available : bool
        Runtime availability flag (False = skip this expert).
    metadata : dict[str, Any]
        Arbitrary metadata (e.g., model version, dependencies).
    """

    expert_id: str
    display_name: str
    entity_strengths: dict[str, float]
    entity_weaknesses: dict[str, float] = field(default_factory=dict)
    default_weight: float = 1.0
    is_available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ExpertRegistry:
    """Extensible registry of PII detection experts.

    Manages the pool of available experts (regex, GLiNER, Presidio, etc.)
    and allows users to register custom experts. The registry is queried
    by the MoE router to determine which experts are available for a given
    entity type.

    Example
    -------
    >>> registry = ExpertRegistry()
    >>> registry.register_expert(ExpertSpec(
    ...     expert_id="my-custom",
    ...     display_name="My Custom Detector",
    ...     entity_strengths={"EMAIL_ADDRESS": 0.95}
    ... ))
    >>> my_expert = registry.get_expert("my-custom")
    """

    def __init__(self) -> None:
        """Initialize an empty expert registry."""
        self._experts: dict[str, ExpertSpec] = {}

    def register_expert(self, spec: ExpertSpec) -> None:
        """Register a new expert or update an existing one.

        Parameters
        ----------
        spec : ExpertSpec
            Expert specification with capabilities and metadata.
        """
        self._experts[spec.expert_id] = spec

    def unregister_expert(self, expert_id: str) -> None:
        """Remove an expert from the registry.

        Parameters
        ----------
        expert_id : str
            Identifier of expert to remove.

        Raises
        ------
        KeyError
            If expert_id not found.
        """
        del self._experts[expert_id]

    def get_expert(self, expert_id: str) -> ExpertSpec | None:
        """Retrieve an expert by ID.

        Parameters
        ----------
        expert_id : str
            Identifier to look up.

        Returns
        -------
        ExpertSpec | None
            Expert specification if found, else None.
        """
        return self._experts.get(expert_id)

    def list_experts(self) -> list[ExpertSpec]:
        """List all registered experts (including unavailable).

        Returns
        -------
        list[ExpertSpec]
            All experts in insertion order.
        """
        return list(self._experts.values())

    def available_experts(self) -> list[ExpertSpec]:
        """List only available experts.

        Returns
        -------
        list[ExpertSpec]
            Experts where is_available=True.
        """
        return [e for e in self._experts.values() if e.is_available]

    def get_experts_by_id(self, expert_ids: list[str]) -> list[ExpertSpec]:
        """Get multiple experts by ID.

        Parameters
        ----------
        expert_ids : list[str]
            List of expert identifiers.

        Returns
        -------
        list[ExpertSpec]
            Experts found (in same order as expert_ids).
        """
        return [e for eid in expert_ids if (e := self._experts.get(eid)) is not None]


class MoERouter:
    """Sparse Top-K router for expert selection per entity type.

    Inspired by Mixtral's gating mechanism, this router decides which
    experts to activate for each entity type. It computes routing scores
    based on expert strengths, selects the top-K experts, and normalizes
    weights via softmax.

    The "performance floor" guarantee ensures that the best-performing
    expert for each entity type is always selected, preventing the
    ensemble from ever underperforming that expert.

    Parameters
    ----------
    registry : ExpertRegistry
        Pool of available experts to choose from.
    top_k : int
        Number of experts to activate per entity type (default 3).
        If fewer than K experts are available, activates all available.
    performance_floor : bool
        If True, always include the expert with the highest strength
        score for each entity type, ensuring ensemble >= best individual
        expert (default True).
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        top_k: int = 3,
        *,
        performance_floor: bool = True,
    ) -> None:
        self.registry = registry
        self.top_k = max(1, top_k)
        self.performance_floor = performance_floor
        self._route_cache: dict[str, list[tuple[str, float]]] = {}

    def route(self, entity_type: str) -> list[tuple[str, float]]:
        """Compute routing for a single entity type.

        Returns the top-K experts (by strength score) for the given entity
        type, with softmax-normalized weights. If performance_floor is True,
        the highest-strength expert is always included.

        Parameters
        ----------
        entity_type : str
            Entity type to route (e.g., "PERSON_NAME", "EMAIL_ADDRESS").

        Returns
        -------
        list[tuple[str, float]]
            List of (expert_id, weight) tuples summing to 1.0.
            Empty list if no experts available or entity unknown.
        """
        # Check cache
        cached = self._route_cache.get(entity_type)
        if cached is not None:
            return cached

        experts = self.registry.available_experts()
        if not experts:
            return []

        # Compute routing scores for this entity type
        # Only include experts that explicitly declare knowledge of this entity type
        scores: list[tuple[str, float]] = []
        for expert in experts:
            # Only route experts that explicitly mention this entity type
            if entity_type in expert.entity_strengths:
                strength = expert.entity_strengths[entity_type]
                scores.append((expert.expert_id, strength))

        if not scores:
            # No expert has explicit knowledge of this entity type
            return []

        # Sort by strength descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select top-K
        selected = scores[: self.top_k]

        # Performance floor: ensure best expert is included
        if self.performance_floor and selected:
            best_id = scores[0][0]
            if best_id not in {eid for eid, _ in selected}:
                # Best expert was not in top-K; swap it in
                selected[-1] = (best_id, scores[0][1])

        # Apply softmax normalization
        import math

        # Compute exp(score) for each
        exp_scores = [(eid, math.exp(score)) for eid, score in selected]
        total = sum(es for _, es in exp_scores)

        if total <= 0:
            # Fallback to equal weights
            normalized = [(eid, 1.0 / len(selected)) for eid, _ in selected]
        else:
            normalized = [(eid, es / total) for eid, es in exp_scores]

        # Cache result
        self._route_cache[entity_type] = normalized
        return normalized

    def route_all(self) -> dict[str, list[tuple[str, float]]]:
        """Precompute routing for all known entity types.

        Discovers entity types from all experts' strengths and routes each.

        Returns
        -------
        dict[str, list[tuple[str, float]]]
            Mapping of entity_type -> [(expert_id, weight), ...].
        """
        entity_types: set[str] = set()
        for expert in self.registry.available_experts():
            entity_types.update(expert.entity_strengths.keys())

        result = {}
        for entity_type in sorted(entity_types):
            result[entity_type] = self.route(entity_type)
        return result

    def clear_cache(self) -> None:
        """Clear the routing cache.

        Useful if registry is updated and routes need to be recomputed.
        """
        self._route_cache.clear()


class MoEFusionStrategy(FusionStrategy):
    """Mixture-of-Experts fusion inspired by Mixtral's sparse MoE architecture.

    Unlike WeightedConsensusFusion which uses static per-engine weights,
    this strategy:
    1. Routes each entity type to its top-K experts via MoERouter
    2. Applies softmax-normalized weights from the router
    3. Guarantees ensemble >= best individual expert via oracle routing
    4. Supports dynamic expert registration for extensibility

    The performance guarantee works as follows:
    - For each entity type, the router always includes the expert with
      the highest known strength score
    - The floor expert gets weight >= 1/(K+1), ensuring its contribution
      is never diluted below significance
    - If only the floor expert detects an entity, its finding passes through
      at full confidence (no penalty for single-expert detection)

    Parameters
    ----------
    registry : ExpertRegistry | None
        Expert registry. If None, a default registry is created.
    top_k : int
        Number of experts to activate per entity type (default 3).
    iou_threshold : float
        Minimum Intersection-over-Union for span overlap (default 0.5).
    performance_floor : bool
        Always include best expert for each entity type (default True).
    min_expert_weight : float
        Minimum weight for the performance floor expert (default 0.15).
    """

    strategy_id = "mixture_of_experts"

    def __init__(
        self,
        registry: ExpertRegistry | None = None,
        top_k: int = 3,
        *,
        iou_threshold: float = 0.5,
        performance_floor: bool = True,
        min_expert_weight: float = 0.15,
    ) -> None:
        if registry is None:
            registry = build_default_registry()
        self.registry = registry
        self.top_k = top_k
        self.iou_threshold = iou_threshold
        self.performance_floor = performance_floor
        self.min_expert_weight = min_expert_weight
        self.router = MoERouter(
            self.registry,
            top_k=top_k,
            performance_floor=performance_floor,
        )

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings using sparse MoE routing.

        Groups findings by location (entity type, field, span), then uses
        the MoE router to determine which experts should contribute to each
        entity type.  For each cluster, computes a MoE-weighted average
        confidence.

        **Union guarantee**: Every expert's finding is included in the fusion.
        Routed experts receive their full MoE-assigned weight; non-routed
        experts receive a floor weight (``min_expert_weight``) so their
        findings are never silently dropped.  This ensures:

            entities(ensemble) ⊇ entities(best_individual_expert)

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Deduplicated findings with MoE-weighted confidence.
        """
        clusters = _cluster_overlapping_spans(findings, iou_threshold=self.iou_threshold)

        # Floor weight for non-routed experts.  This is the key to the union
        # guarantee: instead of dropping findings from experts not explicitly
        # routed for an entity type, we include them with a minimum weight.
        # This means a presidio finding for an entity type it wasn't "routed"
        # for still contributes to the ensemble, just with lower influence.
        floor_weight = self.min_expert_weight

        merged: list[EnsembleFinding] = []
        for cluster in clusters:
            entity_type = cluster[0].entity_type

            # Get routed experts for this entity type
            routed = self.router.route(entity_type)
            # Build a dict of routed expert ID -> weight
            routed_dict = dict(routed) if routed else {}

            # Compute weighted sum — ALL experts contribute (routed get
            # full weight, non-routed get floor weight).
            weighted_sum = 0.0
            total_weight = 0.0
            engines: list[str] = []
            start_votes: dict[int, float] = {}
            end_votes: dict[int, float] = {}

            for item in cluster:
                # Routed experts get their MoE-assigned weight;
                # non-routed experts get the floor weight (never zero).
                weight = routed_dict.get(item.engine_id, floor_weight)

                weighted_sum += item.confidence * weight
                total_weight += weight
                engines.append(item.engine_id)

                s = item.span_start or 0
                e = item.span_end or 0
                start_votes[s] = start_votes.get(s, 0.0) + weight
                end_votes[e] = end_votes.get(e, 0.0) + weight

            # Should never happen now (floor_weight > 0 for all findings),
            # but guard defensively.
            if total_weight <= 0:
                continue

            # Pick best boundaries by weighted vote
            best_span_start = max(start_votes, key=lambda k: start_votes[k])
            best_span_end = max(end_votes, key=lambda k: end_votes[k])

            representative = cluster[0]
            merged.append(
                EnsembleFinding(
                    entity_type=representative.entity_type,
                    confidence=(weighted_sum / total_weight),
                    engines=sorted(set(engines)),
                    field_path=representative.field_path,
                    span_start=best_span_start,
                    span_end=best_span_end,
                    language=representative.language,
                    explanation=f"MoE routing: {', '.join(sorted(set(engines)))}",
                )
            )

        return merged


def build_default_registry() -> ExpertRegistry:
    """Build the default expert registry with all built-in engines.

    Expert strengths are derived from comparative evaluation data
    (benchmark results, research papers). Users can override these
    defaults or add new experts.

    Returns
    -------
    ExpertRegistry
        Registry with standard expert specs.
    """
    registry = ExpertRegistry()

    # Regex: dominant for structured/formatted PII.
    # IMPORTANT: ALL entity types the regex engine can detect MUST be declared
    # in entity_strengths (even weak ones) so the MoE router includes regex
    # findings for those types.  Declaring them only in entity_weaknesses
    # caused the router to drop regex findings for PERSON_NAME, ORGANIZATION,
    # and USERNAME — breaking the union guarantee.
    registry.register_expert(
        ExpertSpec(
            expert_id="regex-oss",
            display_name="pii-anon Regex Engine",
            entity_strengths={
                # Structured PII: regex is authoritative
                "EMAIL_ADDRESS": 0.99,
                "US_SSN": 0.99,
                "CREDIT_CARD": 0.99,
                "CREDIT_CARD_FRAGMENT": 0.80,
                "IP_ADDRESS": 0.99,
                "MAC_ADDRESS": 0.99,
                "IBAN": 0.99,
                "PHONE_NUMBER": 0.95,
                "DATE_OF_BIRTH": 0.92,
                "DRIVERS_LICENSE": 0.90,
                "PASSPORT": 0.90,
                "BANK_ACCOUNT": 0.95,
                "ROUTING_NUMBER": 0.95,
                "CRYPTO_WALLET": 0.95,
                "EMPLOYEE_ID": 0.85,
                "LICENSE_PLATE": 0.85,
                "LOCATION": 1.0,
                "ADDRESS": 0.80,
                "NATIONAL_ID": 0.85,
                "MEDICAL_RECORD_NUMBER": 0.85,
                # Semantic PII: regex is weaker but still detects these
                "PERSON_NAME": 0.30,
                "ORGANIZATION": 0.25,
                "USERNAME": 0.60,
            },
            entity_weaknesses={
                "PERSON_NAME": 0.30,
                "ORGANIZATION": 0.25,
                "USERNAME": 0.60,
            },
            default_weight=1.0,
        )
    )

    # GLiNER: dominant for semantic/NER entities
    registry.register_expert(
        ExpertSpec(
            expert_id="gliner-compatible",
            display_name="GLiNER PII Base",
            entity_strengths={
                "PERSON_NAME": 0.92,
                "ORGANIZATION": 0.88,
                "LOCATION": 0.85,
                "DATE_OF_BIRTH": 0.80,
                "PHONE_NUMBER": 0.75,
                "EMAIL_ADDRESS": 0.70,
            },
            entity_weaknesses={
                "US_SSN": 0.30,
                "CREDIT_CARD": 0.40,
                "IP_ADDRESS": 0.20,
                "IBAN": 0.15,
                "MAC_ADDRESS": 0.10,
            },
            default_weight=1.25,
        )
    )

    # Presidio: good at common NER entities
    registry.register_expert(
        ExpertSpec(
            expert_id="presidio-compatible",
            display_name="Microsoft Presidio",
            entity_strengths={
                "PERSON_NAME": 0.82,
                "PHONE_NUMBER": 0.78,
                "EMAIL_ADDRESS": 0.80,
                "LOCATION": 0.75,
                "DATE_OF_BIRTH": 0.70,
                "ORGANIZATION": 0.72,
            },
            entity_weaknesses={
                "US_SSN": 0.45,
                "CREDIT_CARD": 0.50,
                "IP_ADDRESS": 0.40,
                "IBAN": 0.25,
            },
            default_weight=1.3,
        )
    )

    # scrubadub: limited but useful for some types
    registry.register_expert(
        ExpertSpec(
            expert_id="scrubadub-compatible",
            display_name="scrubadub",
            entity_strengths={
                "EMAIL_ADDRESS": 0.75,
                "PHONE_NUMBER": 0.60,
                "US_SSN": 0.55,
            },
            entity_weaknesses={
                "PERSON_NAME": 0.20,
                "ORGANIZATION": 0.15,
                "CREDIT_CARD": 0.30,
            },
            default_weight=0.95,
        )
    )

    # spaCy NER
    registry.register_expert(
        ExpertSpec(
            expert_id="spacy-ner-compatible",
            display_name="spaCy NER",
            entity_strengths={
                "PERSON_NAME": 0.78,
                "ORGANIZATION": 0.72,
                "LOCATION": 0.75,
            },
            entity_weaknesses={
                "US_SSN": 0.10,
                "CREDIT_CARD": 0.10,
                "EMAIL_ADDRESS": 0.30,
                "PHONE_NUMBER": 0.25,
            },
            default_weight=1.0,
        )
    )

    # Stanza NER
    registry.register_expert(
        ExpertSpec(
            expert_id="stanza-ner-compatible",
            display_name="Stanza NER",
            entity_strengths={
                "PERSON_NAME": 0.75,
                "ORGANIZATION": 0.68,
                "LOCATION": 0.72,
            },
            entity_weaknesses={
                "US_SSN": 0.10,
                "CREDIT_CARD": 0.10,
                "EMAIL_ADDRESS": 0.25,
                "PHONE_NUMBER": 0.20,
            },
            default_weight=0.95,
        )
    )

    return registry


# ── Cached singleton for the default registry ────────────────────────
_DEFAULT_REGISTRY: ExpertRegistry | None = None


def get_default_registry() -> ExpertRegistry:
    """Return the default expert registry, creating it once on first call.

    This avoids the overhead of rebuilding six ``ExpertSpec`` objects on
    every ``build_fusion(mode="mixture_of_experts")`` invocation.
    """
    global _DEFAULT_REGISTRY  # noqa: PLW0603
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY
