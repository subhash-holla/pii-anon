"""S2-04 — aux-loss-free SLA selection-bias (DC-03): a per-expert latency bias on
**selection logits only**.

Pins the DeepSeek-V3 loss-free-load-balancing mechanism added to
:meth:`MoERouter.route`: a per-expert latency penalty added to the **selection
logits only** (never :class:`~pii_anon.types.EnsembleFinding` ``confidence``, never
``Shared``-layer membership) so that, under a latency SLA, the router biases
*selection* toward cheaper experts — with **no auxiliary loss term**. Per-expert
latency cost enters via ``ExpertSpec.metadata["latency_cost_ms"]`` (optional; a
missing cost ⇒ bias 0). Pure-stdlib ``math``. **Default-off** (``sla_bias=None`` or
``strength=0.0``) ⇒ byte-identical static softmax (NFR-026). The bias is
**advisory**: the recall-floor (``FloorProjectingFusion``) remains the no-drop
authority (AX-003).

* **A1** the SLA bias shifts selection mass toward the cheaper expert (NFR-009).
* **A2** the bias changes top-k membership under truncation (NFR-010).
* **A3** ``EnsembleFinding.confidence`` byte-identical with vs without the bias.
* **A4** the recall-floored output is byte-identical under an aggressive β (AX-003).
* **A5** the ``Shared``-layer membership is invariant under the aggressive bias.
* **A6** determinism + the total-order ``(logit, expert_id)`` tie-break (AX-002).
* **A7** default-off byte-identical to the independent static-softmax oracle.
* **A8** a missing / NaN / inf / negative / bool / non-numeric cost ⇒ not penalized.
* **A9** ``default_weight``/``entity_strengths`` orthogonality — THE POWER-TIER TRAP.
* **A10** existing callers (``route_all`` / ``MoEFusionStrategy.merge``) intact.
* **A11** ``SLABias`` is frozen + hashable (mirrors ``RouteContext``).

Traces: NFR-009 (full-swarm latency — the SLA selection-bias MECHANISM) · NFR-010
(throughput floor) · NFR-026 (graceful degradation) · AX-002 (determinism — the
total-order tie-break; pure-stdlib, no RNG/time) · AX-003 (recall-floor advisory —
the bias re-orders/re-weights *selection* only and can never drop a ``Shared``
span) · AX-001 (synthetic fixtures only — no real PII).

Test-type tags: ``[UNIT-TEST]`` ``[PROPERTY-TEST]`` ``[AUDIT]``.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from pii_anon.moe import (
    LATENCY_COST_KEY,
    ExpertRegistry,
    ExpertSpec,
    MoEFusionStrategy,
    MoERouter,
    SLABias,
    _latency_cost,
    build_default_registry,
)
from pii_anon.routing.floor_fusion import FloorProjectingFusion, _shared_subset
from pii_anon.routing.shared_layer import is_shared_floor
from pii_anon.types import EngineFinding

# --------------------------------------------------------------------------- #
# Synthetic fixtures (AX-001 — no real PII).                                   #
# --------------------------------------------------------------------------- #

#: An aggressive bias that would (in routing) deprioritize a costly expert hard.
_AGGRESSIVE = SLABias(strength=100.0, reference_ms=1.0)


def _two_experts_inverted() -> ExpertRegistry:
    """Two experts over PERSON_NAME with INVERTED strength/latency.

    ``expert-fast`` is slightly weaker (0.60) but cheap (1.0 ms); ``expert-slow``
    is slightly stronger (0.65) but expensive (50.0 ms). Under an SLA bias the
    cheaper expert's selection mass should rise.
    """
    reg = ExpertRegistry()
    reg.register_expert(
        ExpertSpec(
            expert_id="expert-fast",
            display_name="Fast",
            entity_strengths={"PERSON_NAME": 0.60},
            metadata={"latency_cost_ms": 1.0},
        )
    )
    reg.register_expert(
        ExpertSpec(
            expert_id="expert-slow",
            display_name="Slow",
            entity_strengths={"PERSON_NAME": 0.65},
            metadata={"latency_cost_ms": 50.0},
        )
    )
    return reg


#: The engine_id of the always-on shared layer (the recall-floor source). The
#: hostile registry below makes THIS expert expensive so the aggressive bias
#: genuinely EVICTS it from the routed top-k (eviction-and-recovery for A4/A5).
_SHARED_ENGINE_ID = "regex-oss"


def _regex_oss_evicting_registry() -> ExpertRegistry:
    """Synthetic registry where the shared ``regex-oss`` expert is EXPENSIVE.

    Unlike ``build_default_registry()`` (whose ``regex-oss`` carries NO
    ``latency_cost_ms``, so the aggressive bias is an inert no-op — the original
    A4/A5 fixture gap the axiom reviewer flagged), here ``regex-oss`` is
    authoritative on US_SSN (strength 0.99) but carries a LARGE ``latency_cost_ms``
    (5000.0). Two cheap competitors (strengths 0.70 / 0.65, cost 1.0) outrank it
    once the aggressive ``SLABias(strength=100)`` demotes it. With ``top_k=2`` +
    ``performance_floor=False`` the bias GENUINELY EVICTS ``regex-oss`` from the
    routed set — so A4/A5 exercise the dynamic eviction-and-recovery path (the
    recall-floor still yields the shared span), not an inert-bias no-op.
    """
    reg = ExpertRegistry()
    reg.register_expert(
        ExpertSpec(
            expert_id=_SHARED_ENGINE_ID,  # the shared / recall-floor expert
            display_name="Regex (expensive)",
            entity_strengths={"US_SSN": 0.99},
            metadata={"latency_cost_ms": 5000.0},  # so the bias can evict it
        )
    )
    reg.register_expert(
        ExpertSpec(
            expert_id="gliner-cheap",
            display_name="GLiNER (cheap)",
            entity_strengths={"US_SSN": 0.70},
            metadata={"latency_cost_ms": 1.0},
        )
    )
    reg.register_expert(
        ExpertSpec(
            expert_id="presidio-cheap",
            display_name="Presidio (cheap)",
            entity_strengths={"US_SSN": 0.65},
            metadata={"latency_cost_ms": 1.0},
        )
    )
    return reg


def _us_ssn_shared_span() -> EngineFinding:
    """A checksum-gated US_SSN shared span emitted by the ``regex-oss`` engine."""
    return EngineFinding(
        entity_type="US_SSN",
        confidence=0.99,
        field_path="ssn",
        span_start=0,
        span_end=11,
        engine_id=_SHARED_ENGINE_ID,
        language="en",
        explanation="regex checksum-gated",
    )


def _routed_ids(sla_bias: SLABias | None) -> set[str]:
    """The routed (selected) expert-id set for US_SSN on the evicting registry.

    ``top_k=2`` + ``performance_floor=False`` so the floor-swap does NOT re-add
    ``regex-oss`` to *routing* — letting the aggressive bias genuinely evict it.
    """
    router = MoERouter(
        _regex_oss_evicting_registry(),
        top_k=2,
        performance_floor=False,
        sla_bias=sla_bias,
    )
    return {eid for eid, _ in router.route("US_SSN")}


def _independent_static_softmax(
    registry: ExpertRegistry,
    entity_type: str,
    top_k: int = 3,
    *,
    performance_floor: bool = True,
) -> list[tuple[str, float]]:
    """A SECOND, independent implementation of the pre-bias static softmax.

    Re-derived from the documented spec (NOT importing the production ``route``),
    mirroring the S2-01 oracle in ``test_moe_router_seam.py`` so the default-off
    byte-identity (A7) has real teeth: it fails if the production path drifts.
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


#: Representative entity set whose static-softmax output is the default-off ref.
_REPRESENTATIVE_ENTITIES = (
    "PERSON_NAME",
    "EMAIL_ADDRESS",
    "US_SSN",
    "ORGANIZATION",
    "PHONE_NUMBER",
    "LOCATION",
    "USERNAME",
    "CREDIT_CARD",
    "DATE_OF_BIRTH",
    "IP_ADDRESS",
)


# --------------------------------------------------------------------------- #
# A1 — SLA bias shifts selection toward the cheaper expert  [UNIT-TEST]        #
# --------------------------------------------------------------------------- #


def test_nfr009_bias_shifts_selection_mass_toward_cheaper_expert() -> None:
    """A1: with SLABias(β>0), the cheap expert's weight strictly rises.

    expert-fast (strength 0.60 / cost 1.0) vs expert-slow (strength 0.65 / cost
    50.0): the bias lifts the cheaper expert's softmax mass vs the strength=0
    baseline.
    """
    reg = _two_experts_inverted()
    baseline = dict(MoERouter(reg).route("PERSON_NAME"))
    biased = dict(
        MoERouter(reg, sla_bias=SLABias(strength=0.5, reference_ms=10.0)).route(
            "PERSON_NAME"
        )
    )
    assert biased["expert-fast"] > baseline["expert-fast"], (
        "SLA bias did not lift the cheaper expert's selection mass"
    )
    assert math.isclose(sum(biased.values()), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# A2 — bias changes top-k membership under truncation  [UNIT-TEST]            #
# --------------------------------------------------------------------------- #


def test_nfr010_bias_promotes_cheap_expert_into_truncated_topk() -> None:
    """A2: a cheap expert just below the top-k cutoff is promoted INTO the set.

    Four experts, ``top_k=2``, ``performance_floor=False`` so the floor swap does
    not mask the bias. ``e-cheap`` (strength 0.70, cost 1.0) sits just below the
    top-2 strength cutoff (``e-mid1``/``e-mid2`` at 0.80/0.75); a large β promotes
    it into the selected set and demotes a costly one — proving the bias changes
    the *selection set*, not only the weights.
    """
    reg = ExpertRegistry()
    reg.register_expert(
        ExpertSpec("e-top", "Top", {"PERSON_NAME": 0.90}, metadata={"latency_cost_ms": 80.0})
    )
    reg.register_expert(
        ExpertSpec("e-mid1", "Mid1", {"PERSON_NAME": 0.80}, metadata={"latency_cost_ms": 70.0})
    )
    reg.register_expert(
        ExpertSpec("e-mid2", "Mid2", {"PERSON_NAME": 0.75}, metadata={"latency_cost_ms": 60.0})
    )
    reg.register_expert(
        ExpertSpec("e-cheap", "Cheap", {"PERSON_NAME": 0.70}, metadata={"latency_cost_ms": 1.0})
    )

    baseline_ids = {
        eid for eid, _ in MoERouter(reg, top_k=2, performance_floor=False).route("PERSON_NAME")
    }
    biased_ids = {
        eid
        for eid, _ in MoERouter(
            reg, top_k=2, performance_floor=False, sla_bias=_AGGRESSIVE
        ).route("PERSON_NAME")
    }
    assert "e-cheap" not in baseline_ids, "fixture broken: cheap expert already in baseline top-k"
    assert "e-cheap" in biased_ids, "SLA bias failed to promote the cheap expert into the top-k set"
    assert len(biased_ids) == 2, "top-k cardinality changed"


# --------------------------------------------------------------------------- #
# A3 — EnsembleFinding.confidence byte-identical  [UNIT-TEST]                  #
# --------------------------------------------------------------------------- #


def test_dc03_confidence_byte_identical_with_and_without_bias() -> None:
    """A3: a single-expert-per-type cluster ⇒ confidence is byte-identical.

    With one expert per type the weighted average cannot move with the weights, so
    the bias (selection-only) must leave every ``EnsembleFinding.confidence``
    byte-identical. Pins the literal DC-03 criterion: the bias changes the
    selection set but NOT a stored confidence.
    """
    findings = [
        EngineFinding(
            entity_type="PERSON_NAME",
            confidence=0.83,
            field_path="notes",
            span_start=0,
            span_end=8,
            engine_id="gliner-compatible",
            language="en",
        ),
        EngineFinding(
            entity_type="EMAIL_ADDRESS",
            confidence=0.77,
            field_path="email",
            span_start=0,
            span_end=15,
            engine_id="regex-oss",
            language="en",
        ),
    ]
    plain = MoEFusionStrategy(registry=build_default_registry())
    biased = MoEFusionStrategy(registry=build_default_registry(), sla_bias=_AGGRESSIVE)

    conf_plain = {(f.entity_type, f.span_start): f.confidence for f in plain.merge(findings)}
    conf_biased = {(f.entity_type, f.span_start): f.confidence for f in biased.merge(findings)}
    assert conf_plain == conf_biased, "SLA bias perturbed a stored EnsembleFinding.confidence"


# --------------------------------------------------------------------------- #
# A4 — recall-floor byte-identical under aggressive bias  [UNIT-TEST] [AUDIT]  #
# --------------------------------------------------------------------------- #


def test_ax003_floored_output_byte_identical_under_aggressive_bias() -> None:
    """A4: floored output byte-identical even when the bias EVICTS regex-oss.

    Eviction-and-recovery (NOT an inert-bias no-op): on the
    ``_regex_oss_evicting_registry`` the aggressive ``SLABias(strength=100)``
    GENUINELY demotes the expensive ``regex-oss`` out of the routed top-k (with
    ``top_k=2`` + ``performance_floor=False`` so the floor-swap doesn't re-add it
    to *routing*) — yet the REAL ``FloorProjectingFusion(MoEFusionStrategy(...))``
    output is BYTE-IDENTICAL bias-on vs bias-off, because the recall-floor still
    yields the checksum-gated shared span (via ``min_expert_weight`` in
    ``merge()`` and the projector backstop, FR-016 / AX-003). The bias changed the
    routed SET but not the floored OUTPUT.
    """
    # Anti-no-op guard: assert the bias actually perturbed the routed set, so this
    # test fails LOUDLY if the fixture ever silently reverts to an inert bias.
    off_routed = _routed_ids(None)
    on_routed = _routed_ids(_AGGRESSIVE)
    assert _SHARED_ENGINE_ID in off_routed, "fixture broken: regex-oss not routed bias-off"
    assert _SHARED_ENGINE_ID not in on_routed, (
        "fixture broken: the aggressive bias did NOT evict regex-oss from routing "
        "(A4 would be an inert-bias no-op, not eviction-and-recovery)"
    )
    assert off_routed != on_routed, "the aggressive bias did not change the routed set"

    shared_span = _us_ssn_shared_span()

    def _floored(
        sla_bias: SLABias | None,
    ) -> list[tuple[str, str | None, int | None, int | None, float]]:
        inner = MoEFusionStrategy(
            registry=_regex_oss_evicting_registry(),
            top_k=2,
            performance_floor=False,
            sla_bias=sla_bias,
        )
        out = FloorProjectingFusion(inner).merge([shared_span])
        # type-carrying key + confidence + the floor provenance, sorted for stability
        return sorted(
            (f.entity_type, f.field_path, f.span_start, f.span_end, f.confidence) for f in out
        )

    assert _floored(None) == _floored(_AGGRESSIVE), (
        "an aggressive SLA bias that EVICTED regex-oss from routing changed the "
        "recall-floored output set (AX-003 violated)"
    )


# --------------------------------------------------------------------------- #
# A5 — Shared membership invariant  [UNIT-TEST]                                #
# --------------------------------------------------------------------------- #


def test_ax003_shared_membership_invariant_under_aggressive_bias() -> None:
    """A5: the regex-oss shared input set + the floored Shared-membership are
    bias-invariant.

    Membership invariance (distinct from A4's output-value invariance). Two
    membership sets must be identical bias-on vs bias-off:

    1. the shared INPUT set the floor consumes (the regex-oss subset of findings);
    2. the SHARED span membership in the floored output (the US_SSN regex-oss span
       must be present either way — whether via the normal MoE merge or via the
       floor re-injection — and the set of ``shared_floor`` re-injected keys must
       match exactly).

    This pins that the selection-only SLA bias cannot add/drop a Shared span from
    the floor's input OR from the final membership EVEN WHEN the bias genuinely
    evicts ``regex-oss`` from the routed set: the bias re-orders *selection* but
    the recall-floor (FR-016 / AX-003) keeps the shared set membership fixed.
    """
    # Anti-no-op guard (shared with A4): the aggressive bias must actually evict
    # regex-oss from routing, so A5 exercises eviction-and-recovery membership
    # invariance — not an inert-bias no-op. Fails LOUDLY on a reverted fixture.
    assert _SHARED_ENGINE_ID in _routed_ids(None), "fixture broken: regex-oss not routed bias-off"
    assert _SHARED_ENGINE_ID not in _routed_ids(_AGGRESSIVE), (
        "fixture broken: the aggressive bias did NOT evict regex-oss from routing"
    )

    shared_span = _us_ssn_shared_span()
    findings = [shared_span]
    key = ("US_SSN", 0, 11)

    # (1) The shared INPUT set the projector consumes is the regex-oss subset of
    # the same findings — derived independently of the router/bias.
    shared_input = {
        (f.entity_type, f.span_start, f.span_end)
        for f in _shared_subset(findings, _SHARED_ENGINE_ID)
    }
    assert shared_input == {key}

    def _membership(
        sla_bias: SLABias | None,
    ) -> tuple[set[tuple[str, int | None, int | None]], set[tuple[str, int | None, int | None]]]:
        inner = MoEFusionStrategy(
            registry=_regex_oss_evicting_registry(),
            top_k=2,
            performance_floor=False,
            sla_bias=sla_bias,
        )
        out = FloorProjectingFusion(inner).merge(findings)
        present = {(f.entity_type, f.span_start, f.span_end) for f in out}
        reinjected = {
            (f.entity_type, f.span_start, f.span_end) for f in out if is_shared_floor(f)
        }
        return present, reinjected

    off_present, off_reinjected = _membership(None)
    on_present, on_reinjected = _membership(_AGGRESSIVE)

    # The shared span is present in the floored output BOTH ways (never dropped),
    # even though the bias evicted regex-oss from the routed set bias-on.
    assert key in off_present and key in on_present, "a Shared span left the floored output"
    # Final shared-membership + the re-injected-key set are byte-identical on/off.
    assert off_present == on_present, "Shared-span membership shifted under the SLA bias (AX-003)"
    assert off_reinjected == on_reinjected, "the floor re-injected a DIFFERENT key set under bias"


# --------------------------------------------------------------------------- #
# A6 — determinism + total-order tie-break  [PROPERTY-TEST]                    #
# --------------------------------------------------------------------------- #


def test_ax002_bias_is_replay_deterministic() -> None:
    """A6: two route()/route_all() runs with the same SLABias are equal."""
    reg = _two_experts_inverted()
    bias = SLABias(strength=0.7, reference_ms=5.0)
    r1 = MoERouter(reg, sla_bias=bias)
    r2 = MoERouter(reg, sla_bias=bias)
    assert r1.route("PERSON_NAME") == r2.route("PERSON_NAME")
    assert r1.route_all() == r2.route_all()


def test_ax002_total_order_tiebreak_stable_across_insertion_permutation() -> None:
    """A6: equal biased logits resolve by (logit, expert_id) — permutation-stable.

    Two experts crafted so their BIASED logits are exactly equal
    (strength − β·cost/ref: 0.50 − 1·(10/10) = −0.50 for both). The
    ``(logit, expert_id)`` secondary key makes the sort a total order, so the
    routed order is identical regardless of registry-insertion order.
    """
    bias = SLABias(strength=1.0, reference_ms=10.0)

    def _reg(order: tuple[str, ...]) -> ExpertRegistry:
        specs = {
            "expert-aaa": ExpertSpec(
                "expert-aaa", "A", {"PERSON_NAME": 0.50}, metadata={"latency_cost_ms": 10.0}
            ),
            "expert-zzz": ExpertSpec(
                "expert-zzz", "Z", {"PERSON_NAME": 0.50}, metadata={"latency_cost_ms": 10.0}
            ),
        }
        reg = ExpertRegistry()
        for eid in order:
            reg.register_expert(specs[eid])
        return reg

    forward = MoERouter(_reg(("expert-aaa", "expert-zzz")), sla_bias=bias).route("PERSON_NAME")
    reverse = MoERouter(_reg(("expert-zzz", "expert-aaa")), sla_bias=bias).route("PERSON_NAME")
    assert [eid for eid, _ in forward] == [eid for eid, _ in reverse], (
        "tie-break order is NOT stable across registry-insertion permutation"
    )
    # And the order is the deterministic (logit asc already equal ⇒ expert_id desc
    # under reverse=True) total order: 'expert-zzz' before 'expert-aaa'.
    assert [eid for eid, _ in forward] == ["expert-zzz", "expert-aaa"]


# --------------------------------------------------------------------------- #
# A7 — default-off byte-identical to static softmax  [UNIT-TEST] [PROPERTY]    #
# --------------------------------------------------------------------------- #


def test_nfr026_sla_bias_none_byte_identical_to_static_softmax_oracle() -> None:
    """A7: sla_bias=None ⇒ route() == the independent static-softmax oracle."""
    router = MoERouter(build_default_registry(), sla_bias=None)
    for et in _REPRESENTATIVE_ENTITIES:
        assert router.route(et) == _independent_static_softmax(build_default_registry(), et), (
            f"sla_bias=None drifted from static softmax for {et}"
        )


def test_nfr026_sla_bias_strength_zero_byte_identical_to_static_softmax_oracle() -> None:
    """A7: SLABias(strength=0.0) ⇒ identically inert (the bias term is 0)."""
    router = MoERouter(build_default_registry(), sla_bias=SLABias(strength=0.0))
    for et in _REPRESENTATIVE_ENTITIES:
        assert router.route(et) == _independent_static_softmax(build_default_registry(), et), (
            f"SLABias(strength=0.0) is not inert for {et}"
        )


# --------------------------------------------------------------------------- #
# A8 — missing / invalid metadata expert not penalized  [UNIT-TEST]           #
# --------------------------------------------------------------------------- #


def test_nfr026_missing_latency_metadata_expert_not_penalized() -> None:
    """A8: an expert with no latency_cost_ms keeps its raw strength (bias 0).

    Two experts at equal strength; only ``e-costed`` carries a cost. Under the
    bias, ``e-free`` (no metadata) is unbiased while ``e-costed`` is penalized ⇒
    e-free's selection mass strictly exceeds e-costed's.
    """
    reg = ExpertRegistry()
    reg.register_expert(ExpertSpec("e-free", "Free", {"PERSON_NAME": 0.50}))  # no metadata
    reg.register_expert(
        ExpertSpec("e-costed", "Costed", {"PERSON_NAME": 0.50}, metadata={"latency_cost_ms": 40.0})
    )
    out = dict(MoERouter(reg, sla_bias=_AGGRESSIVE).route("PERSON_NAME"))
    assert out["e-free"] > out["e-costed"], "an expert with NO latency_cost_ms was penalized"


@pytest.mark.parametrize(
    "bad_cost",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        True,
        "slow",
        None,
        [1.0],
        # An UNBOUNDED Python int wider than the C-double range: math.isfinite
        # raises OverflowError on it, which (pre-remediation) propagated unhandled
        # through _latency_cost -> route(). The one numeric type that can exceed
        # the double range while staying isinstance(int). (security-S2-04-01)
        10**400,
        -(10**400),
    ],
)
def test_nfr026_invalid_latency_cost_treated_as_no_cost(bad_cost: object) -> None:
    """A8: a NaN/inf/negative/bool/non-numeric/unbounded-int cost ⇒ "no cost".

    The expert carrying an INVALID cost is treated identically to one carrying no
    cost: its biased logit equals its raw strength, and routing never raises (the
    unbounded-int rows pin that the ``math.isfinite`` ``OverflowError`` is caught
    so ``route()`` is TOTAL over hostile numeric metadata). Pin against a clean
    equal-strength expert with NO metadata as the unbiased oracle, and assert the
    distribution stays finite, non-negative, and sums to 1.0.
    """
    reg = ExpertRegistry()
    reg.register_expert(ExpertSpec("e-clean", "Clean", {"PERSON_NAME": 0.50}))  # no metadata
    reg.register_expert(
        ExpertSpec("e-bad", "Bad", {"PERSON_NAME": 0.50}, metadata={"latency_cost_ms": bad_cost})
    )
    out = dict(MoERouter(reg, sla_bias=_AGGRESSIVE).route("PERSON_NAME"))
    assert math.isclose(out["e-clean"], out["e-bad"], abs_tol=1e-12), (
        f"invalid latency_cost_ms {bad_cost!r} was not coerced to 'no cost'"
    )
    assert all(math.isfinite(w) and w >= 0.0 for w in out.values())
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9), (
        f"distribution did not sum to 1.0 under invalid cost {bad_cost!r}"
    )


# --------------------------------------------------------------------------- #
# A9 — default_weight orthogonality (THE POWER-TIER TRAP)  [UNIT-TEST] [AUDIT] #
# --------------------------------------------------------------------------- #


def test_nfr009_bias_ignores_default_weight_power_signal() -> None:
    """A9: mutating default_weight changes NOTHING under the bias.

    THE POWER-TIER TRAP: the bias reads ONLY metadata["latency_cost_ms"], never
    ``default_weight`` (the NFR-004 power/reliability signal, which route() does
    not read today). Two registries differing ONLY in default_weight must route
    identically under the same SLA bias.
    """
    def _reg(power: float) -> ExpertRegistry:
        reg = ExpertRegistry()
        reg.register_expert(
            ExpertSpec(
                "expert-fast",
                "Fast",
                {"PERSON_NAME": 0.60},
                default_weight=power,
                metadata={"latency_cost_ms": 1.0},
            )
        )
        reg.register_expert(
            ExpertSpec(
                "expert-slow",
                "Slow",
                {"PERSON_NAME": 0.65},
                default_weight=power,
                metadata={"latency_cost_ms": 50.0},
            )
        )
        return reg

    bias = SLABias(strength=0.5, reference_ms=10.0)
    low_power = MoERouter(_reg(0.10), sla_bias=bias).route("PERSON_NAME")
    high_power = MoERouter(_reg(99.0), sla_bias=bias).route("PERSON_NAME")
    assert low_power == high_power, "the SLA bias READ default_weight — POWER-TIER TRAP violated"


def test_nfr009_bias_responds_only_to_latency_cost() -> None:
    """A9: mutating latency_cost_ms (NOT default_weight) is what shifts routing."""
    def _reg(cost: float) -> ExpertRegistry:
        reg = ExpertRegistry()
        reg.register_expert(
            ExpertSpec("expert-a", "A", {"PERSON_NAME": 0.60}, metadata={"latency_cost_ms": 1.0})
        )
        reg.register_expert(
            ExpertSpec("expert-b", "B", {"PERSON_NAME": 0.60}, metadata={"latency_cost_ms": cost})
        )
        return reg

    bias = SLABias(strength=0.5, reference_ms=10.0)
    cheap_b = dict(MoERouter(_reg(1.0), sla_bias=bias).route("PERSON_NAME"))
    costly_b = dict(MoERouter(_reg(50.0), sla_bias=bias).route("PERSON_NAME"))
    assert cheap_b["expert-b"] > costly_b["expert-b"], (
        "mutating latency_cost_ms did NOT shift routing — the bias is not reading it"
    )


# --------------------------------------------------------------------------- #
# A10 — existing callers intact  [UNIT-TEST]                                   #
# --------------------------------------------------------------------------- #


def test_nfr010_route_all_and_merge_produce_valid_outputs_under_bias() -> None:
    """A10: with sla_bias set, route_all() Σ=1.0 + merge yields valid findings."""
    strategy = MoEFusionStrategy(registry=build_default_registry(), sla_bias=_AGGRESSIVE)
    routed = strategy.router.route_all()
    assert routed
    for weights in routed.values():
        if weights:
            assert math.isclose(sum(w for _, w in weights), 1.0, abs_tol=1e-9)
    findings = [
        EngineFinding(
            entity_type="PERSON_NAME",
            confidence=0.9,
            field_path="notes",
            span_start=0,
            span_end=8,
            engine_id="gliner-compatible",
            language="en",
        ),
        EngineFinding(
            entity_type="PERSON_NAME",
            confidence=0.7,
            field_path="notes",
            span_start=0,
            span_end=8,
            engine_id="presidio-compatible",
            language="en",
        ),
    ]
    merged = strategy.merge(findings)
    assert len(merged) == 1
    assert 0.0 <= merged[0].confidence <= 1.0


def test_nfr026_sla_bias_none_matches_s2_01_reference_on_bare_call_sites() -> None:
    """A10: sla_bias=None ⇒ route_all() matches the S2-01 static-softmax oracle."""
    reg = build_default_registry()
    routed = MoERouter(reg, sla_bias=None).route_all()
    for et, weights in routed.items():
        assert weights == _independent_static_softmax(build_default_registry(), et), (
            f"sla_bias=None regressed the bare route_all() call site for {et}"
        )


# --------------------------------------------------------------------------- #
# A11 — SLABias frozen + hashable  [UNIT-TEST]                                 #
# --------------------------------------------------------------------------- #


def test_ax002_slabias_is_frozen_and_hashable() -> None:
    """A11: SLABias is immutable (mutation raises) and hashable (dict key/set)."""
    bias = SLABias(strength=0.5, reference_ms=10.0)
    # hashable ⇒ usable as a dict key / in a set
    _ = {bias: 1}
    assert bias in {SLABias(strength=0.5, reference_ms=10.0)}
    assert hash(bias) == hash(SLABias(strength=0.5, reference_ms=10.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        bias.strength = 1.0  # type: ignore[misc]


def test_ax002_slabias_defaults_are_inert() -> None:
    """A11: a bare SLABias() defaults to strength=0.0 / reference_ms=1.0 (inert)."""
    assert SLABias() == SLABias(strength=0.0, reference_ms=1.0)
    assert SLABias().strength == 0.0


# --------------------------------------------------------------------------- #
# A11+ — SLABias rejects a hostile/degenerate config at construction           #
# (review remediation: code-quality-S2-04-01 + security-S2-04-02, MAJOR).      #
# The SLA-bias ingress must be TOTAL — a reference_ms<=0 ZeroDivisionError or a #
# NaN strength/reference_ms distribution-poison must fail-fast, never reach     #
# route() (NFR-026 "0 unhandled exceptions"; _apply_sla_bias "never raises").   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_ms": 0.0},  # ZeroDivisionError at cost/reference (code-quality-01)
        {"reference_ms": -1.0},  # negative normalizer
        {"reference_ms": float("nan")},  # NaN normalizer ⇒ poisoned distribution
        {"reference_ms": float("inf")},  # non-finite normalizer
        {"strength": -1.0},  # β must be >= 0 (silent bias-inversion otherwise)
        {"strength": float("nan")},  # NaN strength ⇒ poisoned distribution
        {"strength": float("inf")},  # non-finite strength
    ],
)
def test_nfr026_slabias_rejects_hostile_config_at_construction(
    kwargs: dict[str, float],
) -> None:
    """A11+: a non-finite/degenerate SLABias raises ValueError at construction.

    Fail-fast closes the production-path holes the story-gate flagged: a
    ``reference_ms=0.0`` would ``ZeroDivisionError`` at the ``cost / reference``
    division in ``_apply_sla_bias``; a ``NaN`` strength/reference_ms would poison
    the static-softmax distribution (which the bare no-gate path does NOT route
    through ``_normalize_weights``). Validating at the boundary makes the bias
    ingress TOTAL over its construction state — it can never raise downstream.
    """
    with pytest.raises(ValueError):
        SLABias(**kwargs)


def test_nfr026_slabias_valid_configs_construct_fine() -> None:
    """A11+: the default + a normal positive config still construct (no over-strict).

    The validator must NOT reject the legitimate surface: ``SLABias()`` (the
    inert default, strength=0.0 / reference_ms=1.0) and a normal active config
    must both build. ``strength=0.0`` is explicitly allowed (β >= 0, the inert
    boundary); only ``strength < 0`` / non-finite is rejected.
    """
    assert SLABias() == SLABias(strength=0.0, reference_ms=1.0)  # default inert
    assert SLABias(strength=0.0).reference_ms == 1.0  # strength==0 boundary allowed
    bias = SLABias(strength=2.0, reference_ms=3.0)  # normal active config
    assert bias.strength == 2.0
    assert bias.reference_ms == 3.0


# --------------------------------------------------------------------------- #
# REFACTOR — additive edge tests (no behaviour change). Harden the helper       #
# contracts + the cache-invalidation seam + full branch coverage.              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1.0, 1.0),
        (0.0, 0.0),
        (42, 42.0),  # int accepted, coerced to float
        (None, None),  # missing
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (-0.5, None),  # negative
        (True, None),  # bool is an int subclass — rejected
        (False, None),
        ("3.0", None),  # non-numeric
        ([1.0], None),
        # Unbounded ints wider than the C-double range: math.isfinite raises
        # OverflowError on coercion; the guarded _is_finite_number returns False
        # ⇒ "no cost", never raising (security-S2-04-01 remediation).
        (10**400, None),
        (-(10**400), None),
    ],
)
def test_latency_cost_helper_coercion(raw: object, expected: float | None) -> None:
    """REFACTOR: _latency_cost coerces exactly the documented set to None."""
    spec = ExpertSpec("e", "E", {"X": 0.5}, metadata={LATENCY_COST_KEY: raw})
    assert _latency_cost(spec) == expected


def test_latency_cost_helper_missing_key_is_none() -> None:
    """REFACTOR: an ExpertSpec with NO metadata key ⇒ _latency_cost is None."""
    assert _latency_cost(ExpertSpec("e", "E", {"X": 0.5})) is None


def test_latency_cost_cache_invalidated_by_clear_cache() -> None:
    """REFACTOR: clear_cache() re-reads latency costs after a registry mutation.

    The cost cache is lazy + invalidated by clear_cache(); after mutating an
    expert's latency_cost_ms and clearing, the next route reflects the new cost.
    """
    reg = ExpertRegistry()
    cheap = ExpertSpec("e-a", "A", {"PERSON_NAME": 0.60}, metadata={LATENCY_COST_KEY: 1.0})
    other = ExpertSpec("e-b", "B", {"PERSON_NAME": 0.60}, metadata={LATENCY_COST_KEY: 1.0})
    reg.register_expert(cheap)
    reg.register_expert(other)
    router = MoERouter(reg, sla_bias=SLABias(strength=0.5, reference_ms=10.0))

    before = dict(router.route("PERSON_NAME"))
    assert math.isclose(before["e-a"], before["e-b"], abs_tol=1e-12)  # equal cost ⇒ equal

    # Make e-a expensive, clear both the route cache AND the latency-cost cache.
    cheap.metadata[LATENCY_COST_KEY] = 80.0
    router.clear_cache()
    after = dict(router.route("PERSON_NAME"))
    assert after["e-b"] > after["e-a"], "clear_cache() did not invalidate the latency-cost cache"


def test_reference_ms_scales_the_penalty() -> None:
    """REFACTOR: a larger reference_ms shrinks the penalty (unit-decoupling).

    Same β + same costs, two normalizers: the smaller reference_ms applies a
    HARDER penalty to the costly expert, so the cheap expert's mass is higher.
    """
    reg = ExpertRegistry()
    reg.register_expert(ExpertSpec("e-fast", "F", {"PERSON_NAME": 0.60}, metadata={LATENCY_COST_KEY: 1.0}))
    reg.register_expert(ExpertSpec("e-slow", "S", {"PERSON_NAME": 0.60}, metadata={LATENCY_COST_KEY: 50.0}))
    hard = dict(MoERouter(reg, sla_bias=SLABias(strength=1.0, reference_ms=1.0)).route("PERSON_NAME"))
    soft = dict(MoERouter(reg, sla_bias=SLABias(strength=1.0, reference_ms=100.0)).route("PERSON_NAME"))
    assert hard["e-fast"] > soft["e-fast"], "reference_ms did not scale the penalty"


def test_apply_sla_bias_inert_when_no_costs_in_registry() -> None:
    """REFACTOR: bias active but NO expert carries a cost ⇒ output unchanged.

    Exercises the `if not costs: return scores` fast path in _apply_sla_bias — an
    SLA is configured but the registry has no latency metadata, so routing equals
    the unbiased baseline.
    """
    reg = ExpertRegistry()
    reg.register_expert(ExpertSpec("e-a", "A", {"PERSON_NAME": 0.60}))  # no metadata
    reg.register_expert(ExpertSpec("e-b", "B", {"PERSON_NAME": 0.40}))  # no metadata
    baseline = MoERouter(reg).route("PERSON_NAME")
    biased = MoERouter(reg, sla_bias=_AGGRESSIVE).route("PERSON_NAME")
    assert biased == baseline, "bias perturbed routing despite NO latency costs in the registry"


def test_apply_sla_bias_direct_guard_returns_scores_unchanged() -> None:
    """REFACTOR: _apply_sla_bias is self-safe — inert guard returns input as-is.

    Calls the helper directly (defensive coverage): with sla_bias=None and with
    SLABias(strength=0.0) it returns the scores list unchanged, independent of
    route()'s own sla_active gate.
    """
    reg = ExpertRegistry()
    reg.register_expert(ExpertSpec("e-a", "A", {"PERSON_NAME": 0.60}, metadata={LATENCY_COST_KEY: 5.0}))
    scores = [("e-a", 0.60)]
    assert MoERouter(reg, sla_bias=None)._apply_sla_bias(list(scores)) == scores
    assert MoERouter(reg, sla_bias=SLABias(strength=0.0))._apply_sla_bias(list(scores)) == scores
