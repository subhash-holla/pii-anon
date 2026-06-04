"""Fusion strategies for merging PII findings from multiple engines.

When multiple detection engines run on the same payload, fusion strategies
merge their findings to produce consensus results. Different strategies
prioritize different goals:

- **Union (High Recall)**: emit all findings (no deduplication).
- **Weighted Consensus**: merge by location, weight by engine reliability.
- **Calibrated Majority**: require N engines to agree (precision-focused).
- **Intersection Consensus**: strict agreement (highest precision).

Custom strategies can be registered via ``register_fusion_strategy()``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pii_anon.errors import FusionError
from pii_anon.types import EngineFinding, EnsembleFinding, FusionAuditRecord, FusionMode

if TYPE_CHECKING:
    # Annotation-only import (guarded to avoid the fusion<->moe import cycle —
    # moe imports FROM fusion). `from __future__ import annotations` keeps the
    # SLABias reference a string at runtime, so no cycle is created.
    from pii_anon.moe import SLABias

FindingKey = tuple[str, str | None, int | None, int | None, str]
AuditKey = tuple[str, str | None, int | None, int | None]


class FusionStrategy:
    """Abstract base for fusion strategies.

    Subclasses implement ``merge()`` to combine raw engine findings into
    deduplicated ensemble findings.
    """
    strategy_id = "base"

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge raw engine findings into ensemble findings.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Deduplicated ensemble findings.
        """
        raise NotImplementedError


class UnionHighRecallFusion(FusionStrategy):
    """Union-based fusion: emit all findings without merging.

    Maximizes recall by including every engine finding, even if only one
    engine detected it. Useful when any detection is valuable and false
    positives can be filtered downstream.

    **Trade-off**: High recall, potentially low precision.
    """
    strategy_id = "union_high_recall"

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Wrap each raw finding as an ensemble finding.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            One ensemble finding per raw finding (no deduplication).
        """
        out: list[EnsembleFinding] = []
        for finding in findings:
            out.append(
                EnsembleFinding(
                    entity_type=finding.entity_type,
                    confidence=finding.confidence,
                    engines=[finding.engine_id],
                    field_path=finding.field_path,
                    span_start=finding.span_start,
                    span_end=finding.span_end,
                    explanation=finding.explanation,
                    language=finding.language,
                )
            )
        return out


def _overlap_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """Return Intersection-over-Union of two span ranges."""
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 0 else 0.0


# Key for grouping overlapping spans: (entity_type, field_path, language)
_OverlapGroupKey = tuple[str, str | None, str]


def _cluster_overlapping_spans(
    findings: list[EngineFinding],
    *,
    iou_threshold: float = 0.5,
) -> list[list[EngineFinding]]:
    """Cluster findings whose spans overlap above a threshold.

    Findings of the same entity type, field, and language are clustered when
    their span IoU exceeds ``iou_threshold``.  Non-overlapping findings each
    form their own single-element cluster.  Within each cluster the span used
    for the merged result is the one with the highest weighted confidence so
    that the best engine's boundaries win.
    """
    by_group: dict[_OverlapGroupKey, list[EngineFinding]] = defaultdict(list)
    for f in findings:
        key = (f.entity_type, f.field_path, f.language)
        by_group[key].append(f)

    clusters: list[list[EngineFinding]] = []
    for group in by_group.values():
        n = len(group)
        if n == 1:
            # Common fast-path: single-engine finding — no sorting needed.
            clusters.append(group)
            continue
        # Sort by span_start for sweep-line clustering
        group.sort(key=lambda f: (f.span_start or 0, f.span_end or 0))
        used = bytearray(n)  # bytearray is faster than list[bool]
        for i in range(n):
            if used[i]:
                continue
            anchor = group[i]
            cluster = [anchor]
            used[i] = 1
            a_start = anchor.span_start or 0
            a_end = anchor.span_end or 0
            for j in range(i + 1, n):
                if used[j]:
                    continue
                b = group[j]
                b_start = b.span_start or 0
                b_end = b.span_end or 0
                if b_start >= a_end + 5:
                    # Past the anchor span — no more overlaps possible
                    break
                if _overlap_iou(a_start, a_end, b_start, b_end) >= iou_threshold:
                    cluster.append(b)
                    used[j] = 1
                    # Expand anchor envelope to union of spans
                    if b_start < a_start:
                        a_start = b_start
                    if b_end > a_end:
                        a_end = b_end
            clusters.append(cluster)
    return clusters


class WeightedConsensusFusion(FusionStrategy):
    """Confidence-weighted consensus: merge findings from same location.

    Groups findings by location (entity type, field, span), then computes
    a weighted average confidence. Engine weights reflect relative reliability.

    Supports per-entity-type ``entity_weights``: weight overrides that let
    each engine contribute more (or less) for entity types it specialises in.
    For example, GLiNER may get weight 1.8 for ``PERSON_NAME`` while regex
    gets 0.5, but regex gets weight 2.0 for ``US_SSN`` while GLiNER gets 0.6.

    .. note::
       For full mixture-of-experts routing, use ``MoEFusionStrategy`` from
       ``pii_anon.moe`` (selected via ``mode="mixture_of_experts"`` in
       ``build_fusion()``).

    When multiple engines detect the same entity at slightly different
    boundaries (e.g. regex finds "123-45-6789" at (15, 26) while Presidio
    finds it at (15, 27)), an overlap-aware pre-clustering step merges them
    into a single finding using the best engine's span boundaries.

    **Trade-off**: Balanced precision and recall. Recovers high-confidence
    matches even if only one trusted engine found them.

    Parameters
    ----------
    weights : dict[str, float] | None
        Per-engine global weights. Default weight is 1.0. Higher weight = more
        influential engine.
    entity_weights : dict[str, dict[str, float]] | None
        Per-entity-type per-engine weight overrides.  Outer key is engine ID,
        inner key is entity type.  When present, overrides the global weight
        for that engine+entity combination.
        Example: ``{"gliner-compatible": {"PERSON_NAME": 1.8}}``
    iou_threshold : float
        Minimum Intersection-over-Union for two spans to be considered
        overlapping (default 0.5).
    """
    strategy_id = "weighted_consensus"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        entity_weights: dict[str, dict[str, float]] | None = None,
        iou_threshold: float = 0.5,
    ) -> None:
        self.weights = weights or {}
        self.entity_weights = entity_weights or {}
        self.iou_threshold = iou_threshold
        self._weight_cache: dict[tuple[str, str], float] = {}

    def _get_weight(self, engine_id: str, entity_type: str) -> float:
        """Resolve weight for a specific engine + entity type combination.

        Returns the per-entity override if one exists, otherwise the global
        engine weight, otherwise 1.0.  Results are cached in ``_weight_cache``
        for O(1) lookups on subsequent calls with the same arguments.
        """
        cache_key = (engine_id, entity_type)
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached
        engine_overrides = self.entity_weights.get(engine_id)
        if engine_overrides:
            override = engine_overrides.get(entity_type)
            if override is not None:
                self._weight_cache[cache_key] = override
                return override
        w = self.weights.get(engine_id, 1.0)
        self._weight_cache[cache_key] = w
        return w

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings by location, weighted by engine reliability.

        Overlapping spans of the same entity type are clustered first,
        then each cluster is reduced to a single ensemble finding using
        weighted-average confidence and the highest-confidence engine's
        span boundaries.

        Uses per-entity expert weights when available (mixture-of-experts),
        falling back to global engine weights.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Findings deduplicated by location with weighted-average confidence.
        """
        clusters = _cluster_overlapping_spans(findings, iou_threshold=self.iou_threshold)

        merged: list[EnsembleFinding] = []
        for cluster in clusters:
            weighted_sum = 0.0
            total_weight = 0.0
            engines: set[str] = set()
            entity_type = cluster[0].entity_type
            # Collect all span boundaries for majority voting.
            # When engines disagree on exact boundaries, use the most common
            # start/end pair rather than the single highest-confidence engine's
            # boundaries.  This avoids the problem where a high-weight engine
            # with slightly-off boundaries overrides a correct detection from
            # another engine, turning a true positive into a false negative in
            # exact-match evaluation.
            start_votes: dict[int, float] = {}
            end_votes: dict[int, float] = {}
            for item in cluster:
                weight = self._get_weight(item.engine_id, entity_type)
                weighted_sum += item.confidence * weight
                total_weight += weight
                engines.add(item.engine_id)
                s = item.span_start or 0
                e = item.span_end or 0
                start_votes[s] = start_votes.get(s, 0.0) + weight
                end_votes[e] = end_votes.get(e, 0.0) + weight

            # Pick the boundary with the most weighted votes
            best_span_start = max(start_votes, key=lambda k: start_votes[k])
            best_span_end = max(end_votes, key=lambda k: end_votes[k])

            representative = cluster[0]
            merged.append(
                EnsembleFinding(
                    entity_type=representative.entity_type,
                    confidence=(weighted_sum / total_weight) if total_weight else 0.0,
                    engines=sorted(engines),
                    field_path=representative.field_path,
                    span_start=best_span_start,
                    span_end=best_span_end,
                    language=representative.language,
                )
            )
        return merged


class CalibratedMajorityFusion(FusionStrategy):
    """Consensus-threshold fusion: require N engines to agree.

    Builds on weighted consensus but filters results to only those found by
    at least N engines. Increases precision at potential cost of recall.

    **Trade-off**: Precision-focused. Good when multiple independent engines
    provide stronger signal than single high-confidence detection.

    Parameters
    ----------
    min_consensus : int
        Minimum number of engines required (default 2).
    """
    strategy_id = "calibrated_majority"

    def __init__(self, min_consensus: int = 2) -> None:
        self.min_consensus = min_consensus
        self._weighted_fusion = WeightedConsensusFusion()

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings by location and filter by consensus threshold.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Findings found by at least ``min_consensus`` engines, with
            weighted-average confidence.
        """
        weighted = self._weighted_fusion.merge(findings)
        return [finding for finding in weighted if len(finding.engines) >= self.min_consensus]


class IntersectionConsensusFusion(FusionStrategy):
    """Strict intersection: only emit findings from all engines.

    Highest precision. Only reports entities that **all** configured engines
    independently detected. Excludes findings from subset of engines.

    **Trade-off**: Highest precision, lowest recall. Suitable when false
    positives are costly and missed detections are acceptable.

    Parameters
    ----------
    min_consensus : int
        Minimum number of engines required (default/min 2).
    """
    strategy_id = "intersection_consensus"

    def __init__(self, min_consensus: int = 2) -> None:
        self.min_consensus = max(2, min_consensus)

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings by location and require strict intersection.

        Only returns findings detected by at least ``min_consensus`` engines.
        Uses the minimum confidence across all engines for conservative scoring.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Findings found by at least ``min_consensus`` engines, with
            minimum confidence across contributors.
        """
        grouped: dict[FindingKey, list[EngineFinding]] = defaultdict(list)
        for finding in findings:
            key = (
                finding.entity_type,
                finding.field_path,
                finding.span_start,
                finding.span_end,
                finding.language,
            )
            grouped[key].append(finding)

        merged: list[EnsembleFinding] = []
        for (entity_type, field_path, span_start, span_end, language), bucket in grouped.items():
            engines = sorted({item.engine_id for item in bucket})
            if len(engines) < self.min_consensus:
                continue
            merged.append(
                EnsembleFinding(
                    entity_type=entity_type,
                    confidence=min(item.confidence for item in bucket),
                    engines=engines,
                    field_path=field_path,
                    span_start=span_start,
                    span_end=span_end,
                    explanation="intersection consensus",
                    language=language,
                )
            )
        return merged


CustomFusionFactory = Callable[[dict[str, float], int], FusionStrategy]
_CUSTOM_FACTORIES: dict[str, CustomFusionFactory] = {}


def register_fusion_strategy(mode: str, factory: CustomFusionFactory) -> None:
    """Register a custom fusion strategy.

    Allows users to implement domain-specific fusion logic and make it
    available to the orchestrator.

    Parameters
    ----------
    mode : str
        Strategy name (e.g., "my_custom_fusion").
    factory : CustomFusionFactory
        Callable that accepts ``(weights: dict[str, float], min_consensus: int)``
        and returns a ``FusionStrategy`` instance.

    Notes
    -----
    Custom strategies are discovered by ``available_fusion_modes()`` and
    can be selected in ``ProcessingProfileSpec.mode``.

    Example
    -------
    >>> def my_factory(weights, min_consensus):
    ...     return MyCustomFusion(weights, min_consensus)
    >>> register_fusion_strategy("my_custom", my_factory)
    """
    _CUSTOM_FACTORIES[mode] = factory


# --------------------------------------------------------------------------- #
# v2 construction seam (S2-01 — ADDITIVE; the frozen v1 above is UNCHANGED).
#
# The v1 ``CustomFusionFactory = Callable[[dict, int], FusionStrategy]`` signature
# is a real public API (``available_fusion_modes()`` discovers it; selectable via
# ``ProcessingProfileSpec.mode``) and cannot carry the richer config a
# feature-conditioned MoE / a verify-on-load gate needs (a gate path, a key-ring,
# a top-k, per-entity weights). The v2 seam adds that capacity ALONGSIDE v1 — a
# single :class:`FusionBuildSpec` carrier + a factory taking it — leaving v1 100%
# byte-identical. This is the path S2-02 registers its gate-configured MoE mode
# through.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FusionBuildSpec:
    """Immutable, richer construction spec for a v2 fusion factory.

    Carries everything :func:`build_fusion` plus a feature-conditioned MoE need,
    including the S2-05 verify-on-load gate path / key-ring. Frozen so the spec a
    factory receives is a deterministic, tamper-evident value (NFR-005). All
    fields are assembled by :func:`build_fusion` from its own kwargs — a v2
    factory never has to re-derive them.

    Attributes
    ----------
    mode : str
        The selected fusion mode (the registration key).
    weights : dict[str, float]
        Per-engine global weights (the v1 ``weights`` arg).
    min_consensus : int
        Consensus threshold (the v1 ``min_consensus`` arg).
    entity_weights : dict[str, dict[str, float]] | None
        Per-engine per-entity-type weight overrides.
    iou_threshold : float
        Minimum Intersection-over-Union for span overlap.
    gate_path : str | Path | None
        Optional ``gate_v1.json`` path for a learned gate (S2-05 verify-on-load).
    key_ring : Any | None
        Optional trust root (a ``KeyRing``) for verifying ``gate_path``. Typed
        loosely to avoid an import cycle; a v2 factory narrows it as needed.
    top_k : int
        Number of experts to activate per entity type (MoE seam).
    sla_bias : SLABias | None
        Optional aux-loss-free SLA selection-bias (DC-03) for the inner MoE
        router. Carried like ``gate_path`` — the spec transports it, but
        :func:`build_fusion`'s PUBLIC signature is intentionally NOT changed to
        source it (the v2 seam carries ``sla_bias=None``); a caller that
        hand-builds a :class:`FusionBuildSpec` with a real ``SLABias`` gets the
        wired latency-biased path. ``None`` (the default) is inert (NFR-026).
    """

    mode: str
    weights: dict[str, float]
    min_consensus: int
    entity_weights: dict[str, dict[str, float]] | None
    iou_threshold: float
    gate_path: str | Path | None
    key_ring: Any | None
    top_k: int
    sla_bias: SLABias | None = None


CustomFusionFactoryV2 = Callable[[FusionBuildSpec], FusionStrategy]
_CUSTOM_FACTORIES_V2: dict[str, CustomFusionFactoryV2] = {}


def register_fusion_strategy_v2(mode: str, factory: CustomFusionFactoryV2) -> None:
    """Register a richer (v2) custom fusion strategy.

    Unlike the frozen v1 :func:`register_fusion_strategy` (whose factory takes
    only ``(weights, min_consensus)``), a v2 factory receives a populated
    :class:`FusionBuildSpec` carrying the full construction context (gate path,
    key-ring, top-k, per-entity weights, iou threshold). Use this to register a
    mode that needs more than the v1 two-argument signature can express.

    Parameters
    ----------
    mode : str
        Strategy name (the dispatch key for :func:`build_fusion`).
    factory : CustomFusionFactoryV2
        Callable ``(spec: FusionBuildSpec) -> FusionStrategy``.

    Notes
    -----
    v2 factories are consulted by :func:`build_fusion` ADDITIVELY — after the
    builtin dispatch and ahead of the v1 fall-through — so registering a v2 mode
    never shadows a builtin and never disturbs the frozen v1 path.
    """
    _CUSTOM_FACTORIES_V2[mode] = factory


#: Single source of truth for the BUILTIN fusion modes (S2-01 drift guard).
#: ``available_fusion_modes()`` reads this and :func:`build_fusion` dispatches
#: exactly these branches, so a mode added to one place but not the other is
#: caught by the drift-guard test. Must stay in lockstep with the ``if mode ==``
#: branches in :func:`build_fusion`.
_BUILTIN_FUSION_MODES: tuple[str, ...] = (
    "union_high_recall",
    "weighted_consensus",
    "calibrated_majority",
    "intersection_consensus",
    "mixture_of_experts",
    "swarm",
)


def available_fusion_modes() -> list[str]:
    """List all available fusion strategies (built-in and registered).

    Returns
    -------
    list[str]
        Sorted list of strategy names.
    """
    return sorted(
        set(_BUILTIN_FUSION_MODES)
        | set(_CUSTOM_FACTORIES.keys())
        | set(_CUSTOM_FACTORIES_V2.keys())
    )


def build_fusion(
    mode: FusionMode,
    *,
    weights: dict[str, float],
    min_consensus: int,
    entity_weights: dict[str, dict[str, float]] | None = None,
    iou_threshold: float = 0.5,
) -> FusionStrategy:
    """Instantiate a fusion strategy by name.

    Selects and configures a fusion strategy based on the mode string.
    Applies engine weights and consensus thresholds as appropriate.

    Parameters
    ----------
    mode : FusionMode
        Strategy name (e.g., "weighted_consensus", "calibrated_majority").
    weights : dict[str, float]
        Per-engine global weights (passed to strategies that support it).
    min_consensus : int
        Consensus threshold (passed to strategies that support it).
    entity_weights : dict[str, dict[str, float]] | None
        Per-entity-type per-engine weight overrides for mixture-of-experts
        fusion.  Outer key is engine ID, inner key is entity type.

    Returns
    -------
    FusionStrategy
        Configured strategy instance.

    Raises
    ------
    FusionError
        If mode is not recognized.
    """
    if mode == "union_high_recall":
        return UnionHighRecallFusion()
    if mode == "calibrated_majority":
        return CalibratedMajorityFusion(min_consensus=min_consensus)
    if mode == "intersection_consensus":
        return IntersectionConsensusFusion(min_consensus=min_consensus)
    if mode == "weighted_consensus":
        return WeightedConsensusFusion(
            weights=weights, entity_weights=entity_weights, iou_threshold=iou_threshold,
        )
    if mode == "mixture_of_experts":
        from pii_anon.moe import MoEFusionStrategy, get_default_registry

        # Lazy import (mirrors the moe import above) to avoid the fusion<->routing
        # import cycle. Wrap with the recall-floor so a shared-layer (regex-oss)
        # span the MoE expert-weight floor drops is re-injected (FR-016/AX-003).
        from pii_anon.routing.floor_fusion import FloorProjectingFusion
        return FloorProjectingFusion(MoEFusionStrategy(
            registry=get_default_registry(),
            top_k=3,
            iou_threshold=iou_threshold,
            performance_floor=True,
            min_expert_weight=0.15,
        ))
    if mode == "swarm":
        from pii_anon.swarm import SwarmFusionStrategy

        # Wrap with the recall-floor so a shared-layer (regex-oss) span the swarm
        # Layer-4 emission/corroboration gate drops is re-injected (FR-016/AX-003).
        from pii_anon.routing.floor_fusion import FloorProjectingFusion
        return FloorProjectingFusion(SwarmFusionStrategy())
    # v2 construction seam (S2-01) — consulted ADDITIVELY after the builtin
    # dispatch and ahead of the frozen v1 fall-through, so a v2 mode never
    # shadows a builtin and the v1 path stays byte-identical. The richer
    # FusionBuildSpec is assembled from this function's kwargs (gate_path /
    # key_ring default to None, top_k to the builtin default — they are carried,
    # not yet sourced from build_fusion's public signature, keeping it stable).
    if mode in _CUSTOM_FACTORIES_V2:
        spec = FusionBuildSpec(
            mode=mode,
            weights=weights,
            min_consensus=min_consensus,
            entity_weights=entity_weights,
            iou_threshold=iou_threshold,
            gate_path=None,
            key_ring=None,
            top_k=3,
            sla_bias=None,
        )
        return _CUSTOM_FACTORIES_V2[mode](spec)
    if mode in _CUSTOM_FACTORIES:
        return _CUSTOM_FACTORIES[mode](weights, min_consensus)
    raise FusionError(f"Unknown fusion mode `{mode}`")


def build_fusion_audit(
    strategy: FusionStrategy,
    merged: list[EnsembleFinding],
    source_findings: list[EngineFinding],
) -> list[FusionAuditRecord]:
    """Generate audit records documenting fusion decisions.

    Creates one audit record per ensemble finding, linking it to the raw
    findings that contributed to it. Enables traceability for compliance
    and debugging.

    Parameters
    ----------
    strategy : FusionStrategy
        The strategy used to perform the fusion.
    merged : list[EnsembleFinding]
        The ensemble findings produced by fusion.
    source_findings : list[EngineFinding]
        The raw engine findings that were merged.

    Returns
    -------
    list[FusionAuditRecord]
        Audit records with lineage information.
    """
    by_key: dict[AuditKey, list[EngineFinding]] = defaultdict(list)
    for item in source_findings:
        key = (item.entity_type, item.field_path, item.span_start, item.span_end)
        by_key[key].append(item)

    audits: list[FusionAuditRecord] = []
    for finding in merged:
        key = (finding.entity_type, finding.field_path, finding.span_start, finding.span_end)
        source = by_key.get(key, [])
        engines = sorted({entry.engine_id for entry in source}) or finding.engines
        audits.append(
            FusionAuditRecord(
                strategy=strategy.strategy_id,
                entity_type=finding.entity_type,
                field_path=finding.field_path,
                span_start=finding.span_start,
                span_end=finding.span_end,
                source_engines=engines,
                source_count=len(source),
                fused_confidence=finding.confidence,
                notes=["derived_from_source_findings"],
            )
        )
    return audits


# --------------------------------------------------------------------------- #
# S2-02 — the wired "distilled_moe" control-path mode (ADDITIVE, v2 seam only).
#
# Registers a v2 fusion factory that builds the recall-floor-wrapped MoE strategy
# whose inner MoERouter carries a verify-on-load DistilledTopKGate loaded from the
# spec's gate_path / key_ring (or no gate when gate_path is None ⇒ static softmax,
# NFR-026). This keeps the gate wiring behind the v2 registry + a factory closure;
# build_fusion's PUBLIC signature is intentionally NOT changed to source gate_path
# (the v2 seam in build_fusion currently carries gate_path=None — threading the
# public signature is DEFERRED to a follow-up, per the S2-02 non-goal). A caller
# that hand-builds a FusionBuildSpec with a real gate_path/key_ring gets the wired
# gated path today; the floor wrap is preserved so the gate stays advisory (AX-003).
# --------------------------------------------------------------------------- #
DISTILLED_MOE_MODE = "distilled_moe"


def _build_distilled_moe(spec: FusionBuildSpec) -> FusionStrategy:
    """v2 factory for the gate-configured, floor-wrapped MoE strategy."""
    from pii_anon.moe import MoEFusionStrategy, MoERouter, get_default_registry
    from pii_anon.routing.distilled_gate import load_distilled_gate
    from pii_anon.routing.floor_fusion import FloorProjectingFusion

    registry = get_default_registry()
    inner = MoEFusionStrategy(
        registry=registry,
        top_k=spec.top_k,
        iou_threshold=spec.iou_threshold,
        performance_floor=True,
        min_expert_weight=0.15,
        sla_bias=spec.sla_bias,
    )
    # Load the verified gate ONLY through the fail-closed verify-on-load seam.
    # gate_path is None ⇒ no gate ⇒ the inner router's static softmax (NFR-026).
    gate = None
    if spec.gate_path is not None:
        key_ring: Any = spec.key_ring
        if key_ring is None:
            # A gate_path with no key_ring cannot be verified — fail closed rather
            # than silently skip verification (mirrors MoERouter's construction seam).
            raise ValueError("distilled_moe gate_path requires a key_ring for verify-on-load")
        gate = load_distilled_gate(spec.gate_path, key_ring=key_ring)
    # Replace the inner strategy's default (no-gate) router with a gate-equipped
    # one — the advisory-weighting layer. The floor wrap below keeps the gate from
    # ever dropping a shared span (AX-003). The optional SLA bias (DC-03) is
    # threaded too so the latency-biased path is wired when a caller supplies it.
    inner.router = MoERouter(
        registry,
        top_k=spec.top_k,
        performance_floor=True,
        gate=gate,
        sla_bias=spec.sla_bias,
    )
    return FloorProjectingFusion(inner)


register_fusion_strategy_v2(DISTILLED_MOE_MODE, _build_distilled_moe)
