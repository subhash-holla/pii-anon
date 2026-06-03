"""S2-01 — `MoERouter.route()` feature-conditioned widening + advisory `RoutingGate`.

Pins the ADDITIVE, INERT-by-default, byte-for-byte backward-compatible seam the
learned `DistilledTopKGate` (S2-02) plugs into:

* **A1** `route(entity_type, *, context=None)` is byte-identical to today's
  static-softmax when no `context`/`gate` is given; accepts a `RouteContext`.
* **A2** no-regression on the two bare-`entity_type` call sites (`route_all`,
  `MoEFusionStrategy.merge`) — recomputed against an INDEPENDENT softmax oracle.
* **A3** the `@runtime_checkable RoutingGate` Protocol + the advisory plug.
* **A4** the gate is advisory-only — a hostile gate that zeroes the regex-oss
  expert STILL cannot drop a floored shared span (the floor re-injects it).
* **A5** gate-output validation — non-finite / negative / non-normalized weights
  never propagate; the router re-normalizes or falls back to the static base.
* **A8** NFR-026 graceful-degradation default (no gate ⇒ static softmax).
* **A9** determinism — same `(entity_type, context)` ⇒ equal results; the
  `RouteContext` is frozen + hashable.

Traces: FR-018 (learned-routing seam) · NFR-026 (optional-dependency graceful
degradation) · NFR-005 (run determinism) · AX-002 (determinism) · AX-003 (the
recall-floor no-drop authority, NOT weakened by the advisory gate). Synthetic
fixtures only (AX-001) — no real PII.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from pii_anon.fusion import FusionStrategy
from pii_anon.moe import (
    ExpertRegistry,
    ExpertSpec,
    MoEFusionStrategy,
    MoERouter,
    RouteContext,
    RoutingGate,
    _normalize_weights,
    build_default_registry,
)
from pii_anon.routing.floor_fusion import FloorProjectingFusion
from pii_anon.types import EngineFinding

# --------------------------------------------------------------------------- #
# Synthetic fixtures (AX-001 — no real PII).
# --------------------------------------------------------------------------- #


def _synthetic_registry() -> ExpertRegistry:
    """A small, fully-controlled registry: two experts over PERSON_NAME."""
    reg = ExpertRegistry()
    reg.register_expert(
        ExpertSpec(
            expert_id="expert-alpha",
            display_name="Alpha",
            entity_strengths={"PERSON_NAME": 0.9, "EMAIL_ADDRESS": 0.4},
        )
    )
    reg.register_expert(
        ExpertSpec(
            expert_id="expert-beta",
            display_name="Beta",
            entity_strengths={"PERSON_NAME": 0.6, "EMAIL_ADDRESS": 0.95},
        )
    )
    return reg


def _independent_static_softmax(
    registry: ExpertRegistry, entity_type: str, top_k: int = 3,
    *, performance_floor: bool = True,
) -> list[tuple[str, float]]:
    """A SECOND, independent implementation of the pre-story static softmax.

    Deliberately re-derived from the documented spec (NOT importing the
    production `route`) so A2's no-regression check has real teeth: it fails if
    the production path drifts from the mathematical definition.
    """
    scores = [
        (e.expert_id, e.entity_strengths[entity_type])
        for e in registry.available_experts()
        if entity_type in e.entity_strengths
    ]
    if not scores:
        return []
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = scores[:top_k]
    if performance_floor and selected:
        best_id = scores[0][0]
        if best_id not in {eid for eid, _ in selected}:
            selected[-1] = (best_id, scores[0][1])
    exp = [(eid, math.exp(s)) for eid, s in selected]
    total = sum(v for _, v in exp)
    if total <= 0:
        return [(eid, 1.0 / len(selected)) for eid, _ in selected]
    return [(eid, v / total) for eid, v in exp]


# Representative entity set whose pre-story static-softmax output is RECORDED
# from the live (pre-S2-01) code as the no-regression reference (A1/A2).
_REPRESENTATIVE_ENTITIES = (
    "PERSON_NAME", "EMAIL_ADDRESS", "US_SSN", "ORGANIZATION",
    "PHONE_NUMBER", "LOCATION", "USERNAME", "CREDIT_CARD",
    "DATE_OF_BIRTH", "IP_ADDRESS",
)


class _ReweightingGate:
    """Synthetic advisory gate: structurally a `RoutingGate` (duck-typed)."""

    def __init__(self, boost_expert: str, factor: float = 4.0) -> None:
        self.boost_expert = boost_expert
        self.factor = factor
        self.calls: list[tuple[str, RouteContext | None, list[tuple[str, float]]]] = []

    def adjust(
        self,
        entity_type: str,
        context: RouteContext | None,
        base: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        self.calls.append((entity_type, context, base))
        return [
            (eid, w * (self.factor if eid == self.boost_expert else 1.0))
            for eid, w in base
        ]


class _HostileZeroingGate:
    """Advisory gate that ZEROES one expert — the A4 adversary."""

    def __init__(self, victim_expert: str) -> None:
        self.victim_expert = victim_expert

    def adjust(
        self,
        entity_type: str,
        context: RouteContext | None,
        base: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        return [(eid, 0.0 if eid == self.victim_expert else w) for eid, w in base]


class _NaNGate:
    """Misbehaving gate emitting non-finite / negative weights — A5 adversary."""

    def adjust(
        self,
        entity_type: str,
        context: RouteContext | None,
        base: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        return [(eid, float("nan")) for eid, _ in base]


class _NegativeGate:
    def adjust(
        self,
        entity_type: str,
        context: RouteContext | None,
        base: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        return [(eid, -1.0 * (i + 1)) for i, (eid, _) in enumerate(base)]


# --------------------------------------------------------------------------- #
# A1 — route() widened, context-aware, backward-compatible  [CONTRACT-TEST]
# --------------------------------------------------------------------------- #


def test_fr018_route_no_context_is_byte_identical_to_static_base() -> None:
    """A1: route("PERSON_NAME") (no context/gate) == pre-story static softmax."""
    router = MoERouter(_synthetic_registry())
    got = router.route("PERSON_NAME")
    expected = _independent_static_softmax(_synthetic_registry(), "PERSON_NAME")
    assert got == expected
    assert math.isclose(sum(w for _, w in got), 1.0, abs_tol=1e-9)


def test_fr018_route_accepts_routecontext_kwarg_no_typeerror() -> None:
    """A1: route(entity_type, context=RouteContext(...)) is accepted (no gate)."""
    router = MoERouter(_synthetic_registry())
    ctx = RouteContext(language="es")
    got = router.route("PERSON_NAME", context=ctx)
    # Context is INERT without a gate ⇒ identical to the static base.
    assert got == router.route("PERSON_NAME")


def test_fr018_context_keyed_cache_does_not_collide() -> None:
    """A1: distinct contexts for the same entity type get distinct cache keys."""
    router = MoERouter(_synthetic_registry())
    base = router.route("PERSON_NAME")  # None-context key (today's cache)
    es = router.route("PERSON_NAME", context=RouteContext(language="es"))
    fr = router.route("PERSON_NAME", context=RouteContext(language="fr"))
    # No gate ⇒ all equal in VALUE, but the cache must hold distinct keys
    # (None, es, fr) without the es result overwriting the None result.
    assert base == es == fr
    assert router.route("PERSON_NAME") == base  # None key still intact


def test_nfr026_route_on_default_registry_matches_recorded_reference() -> None:
    """A1/A2: every representative entity's no-context route == recorded ref."""
    router = MoERouter(build_default_registry())
    for et in _REPRESENTATIVE_ENTITIES:
        got = router.route(et)
        expected = _independent_static_softmax(build_default_registry(), et)
        assert got == expected, f"static-softmax drift for {et}"


# --------------------------------------------------------------------------- #
# A2 — no-regression on existing call sites  [PROPERTY-TEST]
# --------------------------------------------------------------------------- #


def test_fr018_route_all_identical_to_independent_oracle() -> None:
    """A2: route_all() over the default registry matches the softmax oracle."""
    reg = build_default_registry()
    router = MoERouter(reg)
    routed = router.route_all()
    assert routed  # non-empty
    for et, weights in routed.items():
        expected = _independent_static_softmax(build_default_registry(), et)
        assert weights == expected, f"route_all drift for {et}"
        if weights:
            assert math.isclose(sum(w for _, w in weights), 1.0, abs_tol=1e-9)


def test_nfr005_moefusion_merge_unaffected_by_widening() -> None:
    """A2: MoEFusionStrategy.merge (a bare-entity_type call site) still merges."""
    strategy = MoEFusionStrategy(registry=build_default_registry())
    findings = [
        EngineFinding(
            entity_type="PERSON_NAME", confidence=0.9, field_path="notes",
            span_start=0, span_end=8, engine_id="gliner-compatible", language="en",
        ),
        EngineFinding(
            entity_type="PERSON_NAME", confidence=0.7, field_path="notes",
            span_start=0, span_end=8, engine_id="presidio-compatible", language="en",
        ),
    ]
    merged = strategy.merge(findings)
    assert len(merged) == 1
    assert merged[0].entity_type == "PERSON_NAME"
    assert 0.0 <= merged[0].confidence <= 1.0


# --------------------------------------------------------------------------- #
# A3 — RoutingGate Protocol + advisory plug  [UNIT-TEST]
# --------------------------------------------------------------------------- #


def test_fr018_routinggate_is_runtime_checkable() -> None:
    """A3: a duck-typed adjust(...) object isinstance-passes RoutingGate."""
    gate = _ReweightingGate(boost_expert="expert-alpha")
    assert isinstance(gate, RoutingGate)
    assert not isinstance(object(), RoutingGate)


def test_fr018_gate_adjust_invoked_with_static_base() -> None:
    """A3: route() calls gate.adjust(entity_type, context, base); uses result."""
    gate = _ReweightingGate(boost_expert="expert-alpha", factor=4.0)
    router = MoERouter(_synthetic_registry(), gate=gate)
    ctx = RouteContext(language="en", field_path="notes")
    out = router.route("PERSON_NAME", context=ctx)
    # adjust was called once with the entity_type, the context, and the base.
    assert len(gate.calls) == 1
    called_et, called_ctx, called_base = gate.calls[0]
    assert called_et == "PERSON_NAME"
    assert called_ctx == ctx
    assert called_base == _independent_static_softmax(
        _synthetic_registry(), "PERSON_NAME"
    )
    # boosting alpha shifts weight toward it vs the static base, re-normalized.
    assert math.isclose(sum(w for _, w in out), 1.0, abs_tol=1e-9)
    base_alpha = dict(_independent_static_softmax(_synthetic_registry(), "PERSON_NAME"))[
        "expert-alpha"
    ]
    assert dict(out)["expert-alpha"] > base_alpha


def test_nfr026_gate_none_is_never_consulted() -> None:
    """A3/A8: gate=None ⇒ base returned unchanged (gate never invoked)."""
    router = MoERouter(_synthetic_registry(), gate=None)
    assert router.route("PERSON_NAME") == _independent_static_softmax(
        _synthetic_registry(), "PERSON_NAME"
    )


# --------------------------------------------------------------------------- #
# A4 — advisory scope is bounded: a gate cannot defeat the floor
#       [CONTRACT-TEST] [AUDIT]
# --------------------------------------------------------------------------- #


def test_ax003_hostile_gate_cannot_drop_floored_shared_span() -> None:
    """A4: a gate zeroing the regex-oss expert STILL cannot drop a shared span.

    Runs the REAL FloorProjectingFusion(MoEFusionStrategy(... gate=hostile ...))
    path: the shared-layer (regex-oss) span survives because the floor projector
    re-injects it post-fusion (FR-016/AX-003), regardless of the advisory gate.
    """
    registry = build_default_registry()
    hostile = _HostileZeroingGate(victim_expert="regex-oss")
    inner = MoEFusionStrategy(registry=registry)
    # Inject the hostile gate into the inner router (advisory-weighting layer).
    inner.router = MoERouter(
        registry, top_k=inner.top_k,
        performance_floor=inner.performance_floor, gate=hostile,
    )
    pipeline = FloorProjectingFusion(inner)

    # A checksum-gated shared-layer (regex-oss) span on a synthetic doc.
    shared_span = EngineFinding(
        entity_type="US_SSN", confidence=0.99, field_path="ssn",
        span_start=0, span_end=11, engine_id="regex-oss", language="en",
        explanation="regex checksum-gated",
    )
    out = pipeline.merge([shared_span])

    survivors = [
        f for f in out
        if f.entity_type == "US_SSN" and f.span_start == 0 and f.span_end == 11
    ]
    assert survivors, "hostile advisory gate dropped a floored shared span"
    # Tightened (axiom-compliance OBS-2): pin the re-injection MECHANISM, not mere
    # survival. The bare inner MoE merge drops this span (the hostile gate zeroes
    # regex-oss ⇒ total_weight==0 ⇒ cluster skipped), so the ONLY way it appears in
    # the full pipeline output is the SharedLayerProjector re-injecting it with the
    # `shared_floor` provenance marker (FR-016/AX-003). Asserting that marker
    # specifically (dropping the loose `regex-oss in engines` fallback, which a
    # normally-merged span would also satisfy) proves the floor did the saving.
    assert any("shared_floor" in (f.explanation or "") for f in survivors), (
        "floored span survived but NOT via shared_floor re-injection — the AX-003 "
        "floor mechanism is not what saved it "
        f"(explanations={[f.explanation for f in survivors]})"
    )


# --------------------------------------------------------------------------- #
# A5 — gate-output validation (determinism / safety)  [UNIT-TEST]
# --------------------------------------------------------------------------- #


def test_nfr005_nan_gate_output_never_propagates() -> None:
    """A5: a NaN-emitting gate ⇒ valid distribution (no NaN downstream)."""
    router = MoERouter(_synthetic_registry(), gate=_NaNGate())
    out = router.route("PERSON_NAME")
    assert all(math.isfinite(w) for _, w in out)
    assert all(w >= 0.0 for _, w in out)
    assert math.isclose(sum(w for _, w in out), 1.0, abs_tol=1e-9)


def test_nfr005_negative_gate_output_never_propagates() -> None:
    """A5: a negative-weight gate ⇒ valid distribution (≥0, Σ=1.0)."""
    router = MoERouter(_synthetic_registry(), gate=_NegativeGate())
    out = router.route("PERSON_NAME")
    assert all(math.isfinite(w) and w >= 0.0 for _, w in out)
    assert math.isclose(sum(w for _, w in out), 1.0, abs_tol=1e-9)


def test_nfr005_unnormalized_gate_output_is_renormalized() -> None:
    """A5: an un-normalized (but valid) gate output is re-normalized to Σ=1.0."""
    gate = _ReweightingGate(boost_expert="expert-alpha", factor=10.0)
    router = MoERouter(_synthetic_registry(), gate=gate)
    out = router.route("PERSON_NAME")
    assert math.isclose(sum(w for _, w in out), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# A8 — NFR-026 graceful-degradation default  [UNIT-TEST]
# --------------------------------------------------------------------------- #


def test_nfr026_no_gate_runs_pure_static_softmax_no_exceptions() -> None:
    """A8: no gate artifact, no gate= ⇒ static softmax, 0 unhandled exceptions."""
    router = MoERouter(build_default_registry())
    assert router.gate_payload is None
    for et in _REPRESENTATIVE_ENTITIES:
        weights = router.route(et)
        if weights:
            assert math.isclose(sum(w for _, w in weights), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# A9 — determinism  [PROPERTY-TEST]
# --------------------------------------------------------------------------- #


def test_ax002_route_deterministic_for_same_entity_and_context() -> None:
    """A9: two route() calls on the same (entity_type, context) are equal."""
    gate = _ReweightingGate(boost_expert="expert-alpha")
    router = MoERouter(_synthetic_registry(), gate=gate)
    ctx = RouteContext(language="es", field_path="x", checksum_gated=True,
                       features=(("len", 3.0), ("digit_ratio", 0.5)))
    first = router.route("PERSON_NAME", context=ctx)
    second = router.route("PERSON_NAME", context=ctx)
    assert first == second


def test_ax002_routecontext_is_frozen_and_hashable() -> None:
    """A9: RouteContext is frozen (mutation raises) and hashable (cache key)."""
    ctx = RouteContext(language="en", features=(("a", 1.0),))
    # hashable ⇒ usable as a dict key (the widened cache key needs this).
    _ = {(("PERSON_NAME", ctx)): 1}
    assert hash(ctx) == hash(RouteContext(language="en", features=(("a", 1.0),)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.language = "fr"  # type: ignore[misc]


def test_ax002_routecontext_default_is_empty_features() -> None:
    """A9: a bare RouteContext() carries no features and equals another bare."""
    assert RouteContext() == RouteContext()
    assert RouteContext().features == ()


def test_fr018_routing_gate_protocol_is_a_fusionstrategy_independent_symbol() -> None:
    """A3 guard: RoutingGate is a Protocol, not accidentally a FusionStrategy."""
    assert not issubclass(_ReweightingGate, FusionStrategy)


# --------------------------------------------------------------------------- #
# REFACTOR edge tests for the shared `_normalize_weights` helper (additive).
# Pins the contract of the single normalizer the static path and the gate-output
# validator both rely on — no behaviour change, additive coverage only.
# --------------------------------------------------------------------------- #


def test_nfr005_normalize_weights_empty_is_empty_distribution() -> None:
    """REFACTOR: empty input ⇒ empty list (a degenerate-but-valid distribution)."""
    assert _normalize_weights([]) == []


def test_nfr005_normalize_weights_posinf_is_invalid_returns_none() -> None:
    """REFACTOR: a +inf weight is non-finite ⇒ None (signals fall-back-to-base)."""
    assert _normalize_weights([("a", float("inf")), ("b", 0.5)]) is None


def test_nfr005_normalize_weights_neginf_is_invalid_returns_none() -> None:
    """REFACTOR: a -inf weight is non-finite ⇒ None (fall-back-to-base)."""
    assert _normalize_weights([("a", float("-inf"))]) is None


def test_nfr005_normalize_weights_all_nonpositive_returns_none() -> None:
    """REFACTOR: all-zero/negative mass cannot normalize ⇒ None."""
    assert _normalize_weights([("a", 0.0), ("b", -3.0)]) is None


def test_nfr005_normalize_weights_clamps_negatives_and_renormalizes() -> None:
    """REFACTOR: a single negative is clamped to 0; the rest re-normalize to 1.0."""
    out = _normalize_weights([("a", -1.0), ("b", 3.0), ("c", 1.0)])
    assert out is not None
    assert dict(out)["a"] == 0.0
    assert math.isclose(sum(w for _, w in out), 1.0, abs_tol=1e-9)
    assert [eid for eid, _ in out] == ["a", "b", "c"]  # order preserved
