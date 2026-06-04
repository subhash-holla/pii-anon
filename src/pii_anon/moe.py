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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pii_anon.fusion import FusionStrategy, _cluster_overlapping_spans
from pii_anon.moe_gate_signing import KeyRing, verify
from pii_anon.types import EngineFinding, EnsembleFinding


def load_verified_gate(
    path: str | Path,
    *,
    key_ring: KeyRing,
    verify_on_load: bool = True,
) -> dict[str, Any] | None:
    """Load a ``gate_v1.json`` artifact through the verify-on-load seam (S2-05).

    This is the ONLY sanctioned way a learned gate enters the control path. It
    is fail-closed by default:

    * **Absence is graceful** — if ``path`` does not exist, returns ``None`` so the
      caller keeps the static ``entity_strengths`` softmax (NFR-026). Absence of
      the artifact is fine.
    * **Present-but-unverifiable is a hard error** — a present file that fails
      signature verification (a tampered byte, a missing/empty/malformed
      signature, an unknown/retired key id, or malformed JSON) raises
      :class:`~pii_anon.errors.GateSignatureError`. It does NOT degrade to the
      static fallback (present-and-broken != absent) and NEVER returns the
      unsigned content.

    The gate file is parsed as JSON only — it is never routed through any unsafe
    object-deserialization path.

    Parameters
    ----------
    path : str | Path
        Filesystem path to the ``gate_v1.json`` envelope.
    key_ring : KeyRing
        The trust root (current + grace-window previous keys). The signature's
        ``key_id`` must resolve here or verification fails loud.
    verify_on_load : bool
        Defaults to ``True`` (fail-closed). The ``False`` escape hatch returns the
        raw payload WITHOUT verification and exists ONLY for explicit offline
        tooling; it is never the default and never reachable from the production
        ``MoERouter`` construction seam (which passes ``verify_on_load=True``).

    Returns
    -------
    dict | None
        The verified gate payload, or ``None`` if the artifact is absent.

    Raises
    ------
    GateSignatureError
        If a present artifact fails verification (when ``verify_on_load`` is True).
    """
    gate_path = Path(path)
    if not gate_path.exists():
        return None

    # JSON only — the gate file is never deserialized through any unsafe path.
    from pii_anon.errors import GateSignatureError

    try:
        envelope = json.loads(gate_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        if verify_on_load:
            raise GateSignatureError("malformed gate envelope: file is not valid JSON") from exc
        return None

    if not verify_on_load:
        # Explicit offline-tooling escape hatch ONLY: return the raw payload
        # without verification. Never the default; never the production path.
        if not isinstance(envelope, dict):
            return None
        inner = envelope.get("payload")
        if isinstance(inner, dict):
            return dict(inner)
        return dict(envelope)

    verified: dict[str, Any] = verify(envelope, key_set=key_ring)
    return verified


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


@dataclass(frozen=True)
class RouteContext:
    """Feature-conditioned routing signal threaded into :meth:`MoERouter.route`.

    Carries the per-decision features a learned gate (S2-02 ``DistilledTopKGate``)
    reads — everything *beyond* the ``entity_type`` positional, which stays the
    first argument so existing call sites are untouched. The context is purely
    additive: with no gate it is **inert** (the static-softmax ``base`` is
    returned unchanged), and it is the distinct cache key that lets two different
    contexts for the same entity type coexist without collision.

    Frozen + hashable by construction (AX-002 / NFR-005): ``features`` is a
    tuple-of-pairs (not a dict) so the whole context is hashable for cache keying
    and immutable for determinism. It deliberately does NOT carry ``entity_type``.

    Attributes
    ----------
    language : str | None
        Language code of the field under routing (e.g. ``"es"``). ``None`` when
        unknown / not yet detected.
    field_path : str | None
        Name of the field being routed (e.g. ``"notes"``). ``None`` for
        unstructured text.
    checksum_gated : bool | None
        Whether the shared-layer span at this location was checksum/keyword-gated
        — the S2-03 rules-first early-exit signal. ``None`` when not applicable.
    features : tuple[tuple[str, float], ...]
        An extensible, order-stable feature vector (name -> value) the distilled
        gate reads. Tuple-of-pairs keeps the context hashable; empty by default.
    """

    language: str | None = None
    field_path: str | None = None
    checksum_gated: bool | None = None
    features: tuple[tuple[str, float], ...] = ()


#: Metadata key on :class:`ExpertSpec` carrying a per-expert latency cost (ms).
#: The DC-03 SLA selection-bias reads ONLY this key — never ``default_weight``
#: (the NFR-004 power signal) and never ``entity_strengths``.
LATENCY_COST_KEY = "latency_cost_ms"


@dataclass(frozen=True)
class SLABias:
    """Aux-loss-free SLA selection-bias config (DeepSeek-V3 loss-free balancing).

    The SLA bias adds a per-expert latency penalty to :meth:`MoERouter.route`'s
    **selection logits only** — never :class:`~pii_anon.types.EnsembleFinding`
    ``confidence`` and never ``Shared``-layer membership — so that under a latency
    SLA the router biases *selection* toward cheaper experts, with **no auxiliary
    loss term** (a bias on the routing logits, not on a gradient/objective).

    The penalty for an expert carrying a latency cost ``cost_e`` (from
    ``ExpertSpec.metadata[LATENCY_COST_KEY]``) is::

        biased_logit(e) = strength_e − strength · (cost_e / reference_ms)

    Frozen + hashable (AX-002 / NFR-005), mirroring :class:`RouteContext`. Kept
    DISTINCT from ``RouteContext`` (the gate's per-call feature channel): the SLA
    is router-construction state, not a per-call gate feature.

    Default-off: ``strength == 0.0`` makes the bias term identically 0, so a bare
    ``SLABias()`` (and ``sla_bias=None``) is fully inert — ``route()`` is then
    byte-identical to the static-softmax baseline (NFR-026).

    Attributes
    ----------
    strength : float
        The bias magnitude β ≥ 0. ``0.0`` (the default) ⇒ the bias is inert.
    reference_ms : float
        A logit-scale latency normalizer (ms) so the penalty is unit-decoupled.
        Defaults to ``1.0``.
    """

    strength: float = 0.0
    reference_ms: float = 1.0


def _is_finite_number(value: object) -> bool:
    """True iff *value* is a real, finite, non-``bool`` number.

    A Python ``bool`` is an ``int`` subclass, so it is rejected explicitly — a
    ``latency_cost_ms`` of ``True``/``False`` is "no cost", not 1.0/0.0.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _latency_cost(spec: ExpertSpec) -> float | None:
    """Return the expert's latency cost (ms), or ``None`` when it has no opinion.

    Reads ``spec.metadata[LATENCY_COST_KEY]`` and returns it ONLY when it is a
    real, finite, non-negative, non-``bool`` number. A missing / ``NaN`` / ``±inf``
    / negative / ``bool`` / non-numeric cost is coerced to ``None`` ("no cost") so
    the SLA bias never produces a NaN or negative logit and never raises — an
    expert with no usable cost is simply not penalized.
    """
    raw = spec.metadata.get(LATENCY_COST_KEY)
    if not _is_finite_number(raw):
        return None
    cost = float(raw)  # type: ignore[arg-type]  # _is_finite_number narrows to a real number
    if cost < 0.0:
        return None
    return cost


@runtime_checkable
class RoutingGate(Protocol):
    """Advisory re-weighting hook for :class:`MoERouter` (the S2-02 plug point).

    A gate sees the static-softmax ``base`` weights for an entity type plus the
    optional :class:`RouteContext`, and returns advisory re-weighted / re-ordered
    ``(expert_id, weight)`` pairs. The router **re-normalizes and validates** the
    output (finite, ≥0, Σ=1.0) before use, and falls back to ``base`` on an
    invalid return — a misbehaving gate can never emit NaN/negative weights
    downstream (AX-002 / NFR-005).

    The gate is **advisory on weighting/selection only**: it can re-order or
    re-weight experts but it can **never cause a shared-layer span to be
    dropped** — the recall-floor no-drop invariant (FR-016 / AX-003) is enforced
    downstream by :class:`~pii_anon.routing.floor_fusion.FloorProjectingFusion`
    /:class:`~pii_anon.routing.shared_layer.SharedLayerProjector`, which re-inject
    a floored span post-fusion regardless of how the gate re-weighted it.

    This story (S2-01) ships only the Protocol + the plug; the first concrete
    implementation is S2-02's ``DistilledTopKGate``.
    """

    def adjust(
        self,
        entity_type: str,
        context: RouteContext | None,
        base: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Return advisory re-weighted ``(expert_id, weight)`` pairs.

        Parameters
        ----------
        entity_type : str
            The entity type being routed.
        context : RouteContext | None
            The feature-conditioned routing signal (``None`` for the bare path).
        base : list[tuple[str, float]]
            The static-softmax weights the router computed (sums to 1.0).
        """
        ...


def _normalize_weights(
    pairs: list[tuple[str, float]],
) -> list[tuple[str, float]] | None:
    """Normalize ``(id, weight)`` pairs to a valid distribution, or ``None``.

    Returns a finite, non-negative, sum-to-1.0 distribution over the SAME ids in
    the SAME order. Returns ``None`` (signalling "invalid — fall back to base")
    when the input cannot be coerced to a valid distribution: a non-finite weight
    (NaN / ±inf), or a non-positive total mass after clamping negatives to zero.

    This is the single weight-normalizer shared by the static-softmax path and
    the gate-output validator (one definition, no drift). Empty input returns an
    empty list (a degenerate-but-valid distribution).
    """
    if not pairs:
        return []
    clamped: list[tuple[str, float]] = []
    for eid, w in pairs:
        if not math.isfinite(w):
            return None
        clamped.append((eid, w if w > 0.0 else 0.0))
    total = sum(w for _, w in clamped)
    if total <= 0.0:
        return None
    return [(eid, w / total) for eid, w in clamped]


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
    gate_path : str | Path | None
        Optional path to a learned-gate artifact (``gate_v1.json``). When given
        together with ``key_ring``, the gate is loaded through the fail-closed
        verify-on-load seam (S2-05): the signature is verified before the payload
        is accepted. Absence of the file is graceful (``None`` -> static softmax,
        NFR-026); a present-but-unverifiable file raises ``GateSignatureError``.
        ``None`` (the default) keeps the no-gate static-softmax behaviour
        unchanged — this parameter is strictly additive.
    key_ring : KeyRing | None
        Trust root for verifying ``gate_path``. Required when ``gate_path`` is
        provided. ``None`` (the default) preserves the no-gate path.
    sla_bias : SLABias | None
        Optional aux-loss-free SLA selection-bias (DC-03). When given with
        ``strength != 0.0``, a per-expert latency penalty is added to the
        selection logits in :meth:`route` so the router biases *selection* toward
        cheaper experts (from ``ExpertSpec.metadata[LATENCY_COST_KEY]``). ``None``
        (the default) or ``SLABias(strength=0.0)`` is fully inert — ``route()`` is
        then byte-identical to the static softmax (NFR-026). The bias is advisory
        and selection-only: it never mutates ``EnsembleFinding.confidence`` and the
        recall-floor (``FloorProjectingFusion``) remains the no-drop authority
        (AX-003).
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        top_k: int = 3,
        *,
        performance_floor: bool = True,
        gate_path: str | Path | None = None,
        key_ring: KeyRing | None = None,
        gate: RoutingGate | None = None,
        sla_bias: SLABias | None = None,
    ) -> None:
        self.registry = registry
        self.top_k = max(1, top_k)
        self.performance_floor = performance_floor
        # Optional aux-loss-free SLA selection-bias (DC-03). Default None ⇒ the
        # bias branch in route() is never taken (byte-identical static softmax,
        # NFR-026). The per-expert latency cost is read lazily from
        # metadata[LATENCY_COST_KEY] into _latency_cost_cache (invalidated by
        # clear_cache()); it is NEVER default_weight (the POWER-TIER TRAP) and
        # never entity_strengths.
        self._sla_bias = sla_bias
        self._latency_cost_cache: dict[str, float] | None = None
        # Optional advisory re-weighting hook (S2-02 DistilledTopKGate). Default
        # None ⇒ route() returns the static-softmax base unchanged (NFR-026). The
        # gate is advisory on weighting only; the recall-floor (FR-016/AX-003) is
        # enforced downstream by FloorProjectingFusion and cannot be defeated by it.
        self.gate = gate
        # Cache widened from `entity_type` to `(entity_type, context)`. A frozen
        # RouteContext is hashable; the `None`-context key reproduces the
        # pre-S2-01 cache EXACTLY (byte-identical on every existing call site).
        self._route_cache: dict[
            tuple[str, RouteContext | None], list[tuple[str, float]]
        ] = {}
        # Verified learned-gate payload (None on the no-gate static-softmax path,
        # NFR-026). Loaded ONLY through the fail-closed verify-on-load seam — the
        # production construction path always passes verify_on_load=True so a
        # tampered/unsigned gate can never reach route().
        self.gate_payload: dict[str, Any] | None = None
        if gate_path is not None:
            if key_ring is None:
                raise ValueError("gate_path requires a key_ring for verify-on-load")
            self.gate_payload = load_verified_gate(
                gate_path, key_ring=key_ring, verify_on_load=True
            )

    def route(
        self,
        entity_type: str,
        *,
        context: RouteContext | None = None,
    ) -> list[tuple[str, float]]:
        """Compute routing for a single entity type.

        Returns the top-K experts (by strength score) for the given entity
        type, with softmax-normalized weights. If performance_floor is True,
        the highest-strength expert is always included.

        The optional ``context`` carries feature-conditioned routing signal
        (language / field / checksum-gated / a feature vector) for an advisory
        :class:`RoutingGate`. **The widening is additive and inert**: with no
        ``context`` and no ``gate`` the computation, the result, and the cache
        are byte-identical to the pre-S2-01 static softmax. When a ``gate`` is
        configured, the static ``base`` is computed exactly as before and then
        handed to ``gate.adjust(...)``; the gate's output is re-normalized and
        validated (finite, ≥0, Σ=1.0) before use, falling back to ``base`` on an
        invalid return so a misbehaving gate can never propagate NaN/negative
        weights (AX-002 / NFR-005). The gate is advisory on weighting only — it
        cannot drop a floored shared span (FR-016 / AX-003, enforced downstream).

        Parameters
        ----------
        entity_type : str
            Entity type to route (e.g., "PERSON_NAME", "EMAIL_ADDRESS").
        context : RouteContext | None
            Feature-conditioned routing signal for the advisory gate. ``None``
            (the default, every pre-S2-01 caller) reproduces today's behaviour
            and today's cache exactly.

        Returns
        -------
        list[tuple[str, float]]
            List of (expert_id, weight) tuples summing to 1.0.
            Empty list if no experts available or entity unknown.
        """
        # Check cache (keyed on (entity_type, context); None-context == today).
        cache_key = (entity_type, context)
        cached = self._route_cache.get(cache_key)
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

        # Aux-loss-free SLA selection-bias (DC-03). Applied to the SELECTION
        # LOGITS only, BEFORE the sort, ONLY when an SLA is active. Default-off
        # (sla_bias=None or strength==0.0) ⇒ `_sla_active` is False, the bias is a
        # no-op, AND the sort stays the pre-S2-04 stable single-key sort, so the
        # result is byte-identical to the static softmax / the S2-01 oracle
        # (NFR-026 — equal-strength experts keep insertion order). The bias
        # reshapes `scores`, affecting both top-k membership AND the softmax
        # weights, but never touches confidence or Shared-layer membership (AX-003).
        sla_active = self._sla_bias is not None and self._sla_bias.strength != 0.0
        if sla_active:
            scores = self._apply_sla_bias(scores)
            # Total-order (logit, expert_id) tie-break: equal BIASED logits resolve
            # deterministically by expert_id, so the order is stable across
            # registry-insertion permutation (AX-002 / NFR-005). Gated behind the
            # SLA path so the default-off sort is unchanged (byte-identical, A7).
            scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        else:
            # Pre-S2-04 stable single-key sort: preserves insertion order for
            # equal-strength experts ⇒ byte-identical to the recorded S2-01 oracle.
            scores.sort(key=lambda x: x[1], reverse=True)

        # Select top-K
        selected = scores[: self.top_k]

        # Performance floor: ensure best expert is included
        if self.performance_floor and selected:
            best_id = scores[0][0]
            if best_id not in {eid for eid, _ in selected}:
                # Best expert was not in top-K; swap it in
                selected[-1] = (best_id, scores[0][1])

        # Apply softmax normalization — compute exp(score) for each
        exp_scores = [(eid, math.exp(score)) for eid, score in selected]
        total = sum(es for _, es in exp_scores)

        if total <= 0:
            # Fallback to equal weights
            base = [(eid, 1.0 / len(selected)) for eid, _ in selected]
        else:
            base = [(eid, es / total) for eid, es in exp_scores]

        # Advisory gate (S2-02). Default gate=None ⇒ `base` is returned UNCHANGED
        # (byte-identical to the pre-S2-01 static softmax). When present, the gate
        # may re-weight/re-order; its output is re-normalized + validated, and any
        # invalid return (non-finite / all-zero) falls back to `base` so a
        # misbehaving gate can never emit NaN/negative weights downstream.
        normalized = base
        if self.gate is not None:
            adjusted = self.gate.adjust(entity_type, context, list(base))
            validated = _normalize_weights(adjusted)
            normalized = validated if validated is not None else base

        # Cache result
        self._route_cache[cache_key] = normalized
        return normalized

    def _latency_cost_by_id(self) -> dict[str, float]:
        """Return ``{expert_id: latency_cost_ms}`` over the available experts.

        Built lazily on first use from ``metadata[LATENCY_COST_KEY]`` (only experts
        carrying a real, finite, non-negative cost appear — see :func:`_latency_cost`)
        and cached so a repeated ``route()`` adds no per-call recompute. The cache
        is invalidated by :meth:`clear_cache`. It is read ONLY by the SLA-bias step
        — never by the static-softmax path.
        """
        if self._latency_cost_cache is None:
            self._latency_cost_cache = {
                e.expert_id: cost
                for e in self.registry.available_experts()
                if (cost := _latency_cost(e)) is not None
            }
        return self._latency_cost_cache

    def _apply_sla_bias(
        self, scores: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Add the per-expert SLA latency penalty to the selection logits.

        Returns ``scores`` UNCHANGED when no SLA is active (``sla_bias is None`` or
        ``strength == 0.0``) — the default-off fast path (NFR-026). Otherwise
        subtracts ``strength · (cost_e / reference_ms)`` from each expert that
        carries a latency cost (others keep their raw logit). Pure-stdlib scalar
        ``math``; never produces a NaN/negative-via-bad-cost logit because
        :func:`_latency_cost` already coerced any unusable cost to "no cost".
        """
        sla = self._sla_bias
        if sla is None or sla.strength == 0.0:
            return scores
        costs = self._latency_cost_by_id()
        if not costs:
            return scores
        beta = sla.strength
        reference = sla.reference_ms
        return [
            (
                eid,
                logit - beta * (costs[eid] / reference) if eid in costs else logit,
            )
            for eid, logit in scores
        ]

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

        Useful if registry is updated and routes need to be recomputed. Also
        invalidates the lazily-built per-expert latency-cost cache (DC-03) so a
        registry change re-reads ``metadata[LATENCY_COST_KEY]`` on the next route.
        """
        self._route_cache.clear()
        self._latency_cost_cache = None


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
        sla_bias: SLABias | None = None,
    ) -> None:
        if registry is None:
            registry = build_default_registry()
        self.registry = registry
        self.top_k = top_k
        self.iou_threshold = iou_threshold
        self.performance_floor = performance_floor
        self.min_expert_weight = min_expert_weight
        # Optional aux-loss-free SLA selection-bias (DC-03), threaded into the
        # inner router. Default None ⇒ inert (NFR-026). Selection-only: the bias
        # cannot move merge()'s confidence math (single-expert clusters stay
        # byte-identical) and the recall-floor wrap remains the no-drop authority.
        self.sla_bias = sla_bias
        self.router = MoERouter(
            self.registry,
            top_k=top_k,
            performance_floor=performance_floor,
            sla_bias=sla_bias,
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
        routed_cache: dict[str, dict[str, float]] = {}
        for cluster in clusters:
            entity_type = cluster[0].entity_type

            # Get routed experts for this entity type (cached per entity type).
            if entity_type not in routed_cache:
                routed = self.router.route(entity_type)
                routed_cache[entity_type] = dict(routed) if routed else {}
            routed_dict = routed_cache[entity_type]

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


def reset_default_registry() -> None:
    """Invalidate the cached default registry singleton.

    Call after applying offline calibration so the next
    ``get_default_registry()`` reflects calibrated strengths.
    """
    global _DEFAULT_REGISTRY  # noqa: PLW0603
    _DEFAULT_REGISTRY = None
