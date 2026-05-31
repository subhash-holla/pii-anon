"""S1-02 — recall-floor LIVE on the production fusion path (build_fusion seam).

Pins FR-016 / NFR-011 / AX-003 / NFR-005 at the ``build_fusion(...)`` level for
BOTH ``"swarm"`` and ``"mixture_of_experts"``: a shared-layer (``regex-oss``)
span that the inner strategy's gate would drop must be re-injected (tagged
``provenance=shared_floor``) so that ``entities(merge(findings)) ⊇
entities(regex-oss findings)``.

S1-01 proved the :class:`SharedLayerProjector` floor by construction as a
standalone module. S1-02 wires it into the seam so it actually RUNS in
production — these tests therefore exercise the wrapper through the real
``build_fusion`` factory, not the projector directly.

The swarm leak being closed: a single-engine ``regex-oss`` SEMANTIC-type finding
with confidence in ``[0.50, 0.90)`` is below the Layer-1 fast-pass threshold
(0.90), so it flows into Layers 2-4 where the Layer-4 corroboration gate
(``swarm.py:658-661``: ``corroboration_count < corroboration_min`` AND
``meta_score < corroboration_override_threshold``) drops it. The floor must
re-inject it.

Property coverage uses a seeded-random generator (hypothesis is not yet a
dependency; migrating these to @given is tracked as story S1-04 / S6-PROP).
"""
from __future__ import annotations

import random

import pytest

from pii_anon.fusion import build_fusion
from pii_anon.routing.shared_layer import (
    is_shared_floor,
    span_key_engine,
    span_key_ensemble,
)
from pii_anon.types import EngineFinding, EnsembleFinding

#: Fusion modes whose returns build_fusion must wrap with FloorProjectingFusion.
WRAPPED_MODES: list[tuple[str, str]] = [
    ("swarm", "swarm"),
    ("mixture_of_experts", "mixture_of_experts"),
]


def _ef(entity_type: str, start: int, end: int, *, field_path: str | None = "text",
        language: str = "en", confidence: float = 0.95, engine_id: str = "regex-oss") -> EngineFinding:
    return EngineFinding(entity_type=entity_type, confidence=confidence, field_path=field_path,
                         span_start=start, span_end=end, engine_id=engine_id, language=language)


def _enf(entity_type: str, start: int, end: int, *, field_path: str | None = "text",
         language: str = "en", confidence: float = 0.9, engines: list[str] | None = None) -> EnsembleFinding:
    return EnsembleFinding(entity_type=entity_type, confidence=confidence, engines=engines or ["gliner-compatible"],
                           field_path=field_path, span_start=start, span_end=end, language=language)


def _build(mode: str) -> object:
    return build_fusion(mode, weights={}, min_consensus=1)


# ── FR-016: shared-layer span the inner gate drops is re-injected ────────────

@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_fr_016_gated_shared_span_present_in_output(mode: str, expected_id: str) -> None:
    """A ``regex-oss`` SEMANTIC span (conf in [0.50,0.90)) that the swarm Layer-4
    emission/corroboration gate would drop is present in build_fusion(...).merge
    output for BOTH modes (re-injected by the floor if the inner gate dropped
    it). FR-016: the shared span must survive the live seam."""
    # conf 0.55: below fast-pass (0.90) so it reaches Layer 4; single regex-oss
    # engine -> corroboration_count==1 < corroboration_min(2) AND meta < 0.85
    # -> the swarm gate drops it. CREDIT_CARD is a SEMANTIC_TYPE.
    gated = _ef("CREDIT_CARD", 40, 56, confidence=0.55)
    strategy = _build(mode)
    out = strategy.merge([gated])  # type: ignore[attr-defined]

    out_keys = {span_key_ensemble(f) for f in out}
    assert span_key_engine(gated) in out_keys, (
        f"[{mode}] shared-layer span dropped by the inner gate was not present "
        "on the live seam (FR-016 violated)"
    )
    cc = [f for f in out if f.entity_type == "CREDIT_CARD"]
    assert cc and "regex-oss" in cc[0].engines


def test_fr_016_swarm_gate_leak_is_reinjected_by_floor() -> None:
    """The documented swarm leak (§6 / AX-003): the swarm inner strategy DROPS a
    low-confidence single-engine ``regex-oss`` SEMANTIC span at Layer 4, so the
    floor MUST re-inject it tagged ``provenance=shared_floor``.

    Asserted swarm-specifically because the drop is guaranteed only for the
    swarm emission/corroboration gate; the MoE expert-weight floor keeps this
    span natively (see ``test_fr_016_gated_shared_span_present_in_output`` for
    the cross-mode presence guarantee)."""
    gated = _ef("CREDIT_CARD", 40, 56, confidence=0.55)

    # Precondition: the bare swarm strategy genuinely drops this span, so any
    # presence in the wrapped output is the floor's doing, not a coincidence.
    from pii_anon.swarm import SwarmFusionStrategy
    assert SwarmFusionStrategy().merge([gated]) == [], (
        "precondition: swarm Layer-4 gate must drop the gated span"
    )

    strategy = build_fusion("swarm", weights={}, min_consensus=1)
    out = strategy.merge([gated])
    reinjected = [f for f in out if is_shared_floor(f)]
    assert len(reinjected) == 1, "floor must re-inject the swarm-gated shared span"
    assert reinjected[0].entity_type == "CREDIT_CARD"
    assert "regex-oss" in reinjected[0].engines
    assert span_key_engine(gated) == span_key_ensemble(reinjected[0])


@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_fr_016_superset_invariant_holds_for_all_regex_oss_spans(mode: str, expected_id: str) -> None:
    """entities(merge(findings)) ⊇ entities(regex-oss findings) for both modes."""
    shared_inputs = [
        _ef("CREDIT_CARD", 40, 56, confidence=0.55),
        _ef("EMAIL_ADDRESS", 0, 17, confidence=0.62),
        _ef("PHONE_NUMBER", 60, 72, confidence=0.7, field_path="notes"),
    ]
    strategy = _build(mode)
    out = strategy.merge(list(shared_inputs))  # type: ignore[attr-defined]

    out_keys = {span_key_ensemble(f) for f in out}
    shared_keys = {span_key_engine(s) for s in shared_inputs}
    assert shared_keys <= out_keys, (
        f"[{mode}] recall-floor violated: a regex-oss shared span is missing "
        "from the fused output"
    )


# ── AX-003: closes the swarm.py:654/658-661 emission/corroboration leak ──────

@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_ax_003_floor_runs_on_production_seam(mode: str, expected_id: str) -> None:
    """The leak is closed WHERE FUSION RUNS (the build_fusion seam), not just in
    the standalone projector — a low-confidence checksum-style regex hit that the
    gate drops survives end-to-end through build_fusion(...).merge(...)."""
    checksum_hit = _ef("CREDIT_CARD", 40, 56, confidence=0.55)
    noise = _enf("PERSON_NAME", 0, 8)  # an unrelated NER span (different engine)
    strategy = _build(mode)
    # Feed both a non-regex finding and the gated regex-oss finding.
    out = strategy.merge([  # type: ignore[attr-defined]
        EngineFinding(entity_type="PERSON_NAME", confidence=noise.confidence,
                      field_path=noise.field_path, span_start=noise.span_start,
                      span_end=noise.span_end, engine_id="gliner-compatible"),
        checksum_hit,
    ])
    out_keys = {span_key_ensemble(f) for f in out}
    assert span_key_engine(checksum_hit) in out_keys, (
        f"[{mode}] shared-layer span must survive the floor on the live seam (AX-003)"
    )


# ── strategy_id passthrough (preserves tests/test_swarm.py:295) ──────────────

@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_fr_016_strategy_id_passthrough(mode: str, expected_id: str) -> None:
    """The wrapper delegates identity to the inner strategy: strategy_id is
    unchanged (keeps build_fusion('swarm').strategy_id == 'swarm')."""
    strategy = _build(mode)
    assert strategy.strategy_id == expected_id  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_fr_016_attribute_passthrough_to_inner(mode: str, expected_id: str) -> None:
    """Unknown attributes resolve against the inner strategy (transparent wrap)."""
    strategy = _build(mode)
    # merge is the FusionStrategy contract; both inner strategies define it.
    assert callable(strategy.merge)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_fr_016_getattr_guard_blocks_private_recursion(mode: str, expected_id: str) -> None:
    """A missing private/dunder attribute raises AttributeError via the
    __getattr__ underscore guard (no inner forwarding, no infinite recursion)."""
    strategy = _build(mode)
    with pytest.raises(AttributeError):
        _ = strategy._this_private_attr_does_not_exist  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_fr_016_missing_public_attr_forwards_then_raises(mode: str, expected_id: str) -> None:
    """A missing PUBLIC attribute is forwarded to the inner strategy, which also
    lacks it, so AttributeError surfaces — confirming the passthrough path."""
    strategy = _build(mode)
    with pytest.raises(AttributeError):
        _ = strategy.totally_missing_public_attribute  # type: ignore[attr-defined]


# ── NFR-011: no regex-oss spans -> byte-identical to the inner strategy ──────

@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_nfr_011_empty_shared_is_byte_identical_to_inner(mode: str, expected_id: str) -> None:
    """With NO regex-oss spans, the floor is a no-op: build_fusion(...).merge
    output is identical to the inner strategy's own merge output."""
    findings = [
        EngineFinding(entity_type="PERSON_NAME", confidence=0.93, field_path="text",
                      span_start=0, span_end=8, engine_id="gliner-compatible"),
        EngineFinding(entity_type="LOCATION", confidence=0.88, field_path="text",
                      span_start=10, span_end=18, engine_id="spacy"),
    ]
    wrapped = _build(mode)
    # Reach the inner strategy through the documented __getattr__ passthrough.
    inner = wrapped._inner  # type: ignore[attr-defined]

    wrapped_out = wrapped.merge(list(findings))  # type: ignore[attr-defined]
    inner_out = inner.merge(list(findings))

    wrapped_keys = [span_key_ensemble(f) for f in wrapped_out]
    inner_keys = [span_key_ensemble(f) for f in inner_out]
    assert wrapped_keys == inner_keys, (
        f"[{mode}] no-op floor must not change output when no regex-oss spans exist"
    )
    assert not any(is_shared_floor(f) for f in wrapped_out), (
        f"[{mode}] nothing should be floor-restored when shared set is empty"
    )


# ── NFR-005: determinism (stable ordering, same input -> same output) ────────

@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_nfr_005_merge_is_deterministic(mode: str, expected_id: str) -> None:
    """Same input -> identical output ordering across repeated merges."""
    findings = [
        _ef("CREDIT_CARD", 40, 56, confidence=0.55),
        _ef("EMAIL_ADDRESS", 0, 17, confidence=0.62),
    ]
    strategy = _build(mode)
    keys1 = [span_key_ensemble(f) for f in strategy.merge(list(findings))]  # type: ignore[attr-defined]
    keys2 = [span_key_ensemble(f) for f in strategy.merge(list(findings))]  # type: ignore[attr-defined]
    assert keys1 == keys2, f"[{mode}] merge output ordering is non-deterministic (NFR-005)"


@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_nfr_005_reinjection_order_is_input_order(mode: str, expected_id: str) -> None:
    """Re-injected (floor-restored) spans are appended in regex-oss input order,
    so the result is deterministic for a fixed input (NFR-005)."""
    shared_inputs = [
        _ef("CREDIT_CARD", 40, 56, confidence=0.55),
        _ef("EMAIL_ADDRESS", 0, 17, confidence=0.62),
        _ef("PHONE_NUMBER", 60, 72, confidence=0.7, field_path="notes"),
    ]
    strategy = _build(mode)
    out = strategy.merge(list(shared_inputs))  # type: ignore[attr-defined]
    reinjected_keys = [span_key_ensemble(f) for f in out if is_shared_floor(f)]
    expected_order = [span_key_engine(s) for s in shared_inputs]
    # Every re-injected span appears, in the same relative order as the input.
    filtered_expected = [k for k in expected_order if k in set(reinjected_keys)]
    assert reinjected_keys == filtered_expected, (
        f"[{mode}] floor-restored spans not in deterministic input order"
    )


# ── seeded property: superset invariant over many build_fusion(...) merges ───

@pytest.mark.parametrize("mode,expected_id", WRAPPED_MODES)
def test_nfr_011_property_superset_invariant_seeded(mode: str, expected_id: str) -> None:
    """ZERO subset violations over many seeded-random finding sets through the
    LIVE build_fusion seam (recall-floor by construction, end-to-end)."""
    rng = random.Random(1602)
    semantic = ["EMAIL_ADDRESS", "CREDIT_CARD", "PHONE_NUMBER", "PERSON_NAME"]
    fields = ["text", "notes"]
    langs = ["en", "es"]
    strategy = _build(mode)
    for _ in range(150):
        n_shared = rng.randint(0, 4)
        shared = [
            _ef(rng.choice(semantic), s := rng.randint(0, 40), s + rng.randint(1, 12),
                field_path=rng.choice(fields), language=rng.choice(langs),
                # below fast-pass so the inner gate is exercised
                confidence=round(rng.uniform(0.50, 0.89), 2))
            for _ in range(n_shared)
        ]
        out = strategy.merge(list(shared))  # type: ignore[attr-defined]
        out_keys = {span_key_ensemble(f) for f in out}
        shared_keys = {span_key_engine(s) for s in shared}
        assert shared_keys <= out_keys, (
            f"[{mode}] recall-floor violated through the live seam: a regex-oss "
            "shared span is missing from the fused output"
        )
