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
    ExpertRegistry,
    ExpertSpec,
    MoEFusionStrategy,
    MoERouter,
    SLABias,
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
    """A4: the REAL FloorProjectingFusion output is byte-identical bias-on/off.

    An aggressive SLABias that would deprioritize regex-oss in routing cannot
    change the floored span set: ``FloorProjectingFusion(MoEFusionStrategy(...))``
    re-injects the checksum-gated shared (regex-oss) span post-fusion (FR-016 /
    AX-003) regardless of the selection bias.
    """
    shared_span = EngineFinding(
        entity_type="US_SSN",
        confidence=0.99,
        field_path="ssn",
        span_start=0,
        span_end=11,
        engine_id="regex-oss",
        language="en",
        explanation="regex checksum-gated",
    )

    def _floored(sla_bias: SLABias | None) -> list[tuple[str, str | None, int | None, int | None, float]]:
        inner = MoEFusionStrategy(registry=build_default_registry(), sla_bias=sla_bias)
        out = FloorProjectingFusion(inner).merge([shared_span])
        # type-carrying key + confidence + the floor provenance, sorted for stability
        return sorted(
            (f.entity_type, f.field_path, f.span_start, f.span_end, f.confidence) for f in out
        )

    assert _floored(None) == _floored(_AGGRESSIVE), (
        "an aggressive SLA bias changed the recall-floored output set (AX-003 violated)"
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
    the floor's input OR from the final membership: the bias re-orders *selection*
    but the recall-floor (FR-016 / AX-003) keeps the shared set membership fixed.
    """
    shared_span = EngineFinding(
        entity_type="US_SSN",
        confidence=0.99,
        field_path="ssn",
        span_start=0,
        span_end=11,
        engine_id="regex-oss",
        language="en",
        explanation="regex checksum-gated",
    )
    findings = [shared_span]
    key = ("US_SSN", 0, 11)

    # (1) The shared INPUT set the projector consumes is the regex-oss subset of
    # the same findings — derived independently of the router/bias.
    shared_input = {
        (f.entity_type, f.span_start, f.span_end) for f in _shared_subset(findings, "regex-oss")
    }
    assert shared_input == {key}

    def _membership(
        sla_bias: SLABias | None,
    ) -> tuple[set[tuple[str, int | None, int | None]], set[tuple[str, int | None, int | None]]]:
        inner = MoEFusionStrategy(registry=build_default_registry(), sla_bias=sla_bias)
        out = FloorProjectingFusion(inner).merge(findings)
        present = {(f.entity_type, f.span_start, f.span_end) for f in out}
        reinjected = {
            (f.entity_type, f.span_start, f.span_end) for f in out if is_shared_floor(f)
        }
        return present, reinjected

    off_present, off_reinjected = _membership(None)
    on_present, on_reinjected = _membership(_AGGRESSIVE)

    # The shared span is present in the floored output BOTH ways (never dropped).
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
    [float("nan"), float("inf"), float("-inf"), -1.0, True, "slow", None, [1.0]],
)
def test_nfr026_invalid_latency_cost_treated_as_no_cost(bad_cost: object) -> None:
    """A8: a NaN/inf/negative/bool/non-numeric cost ⇒ "no cost" (never penalized).

    The expert carrying an INVALID cost is treated identically to one carrying no
    cost: its biased logit equals its raw strength, and routing never raises. Pin
    against a clean equal-strength expert with NO metadata as the unbiased oracle.
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
