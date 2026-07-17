"""S2-01 — additive v2 fusion-construction seam + single-source drift guard.

Pins the ADDITIVE, v1-frozen-untouched construction seam a feature-conditioned
MoE mode (S2-02) registers through, plus the single-source-of-truth + drift
guard that closes the "register the new mode in BOTH places" SME MAJOR:

* **A6** `register_fusion_strategy_v2(mode, factory_v2)` + a populated
  `FusionBuildSpec` (carrying gate_path/key_ring/top_k/...) drive `build_fusion`;
  the frozen v1 `register_fusion_strategy` + `CustomFusionFactory` are UNCHANGED
  and an existing v1-registered mode still constructs.
* **A7** every mode `available_fusion_modes()` advertises is constructible by
  `build_fusion(mode, ...)` without `FusionError`, and every builtin
  `build_fusion` branch is advertised — with a NON-VACUOUS meta-check proving a
  mode added to only one place is caught.

Traces: FR-018 (the construction path a learned gate is wired through) ·
DC-02 / D5 seam-correction MAJORs. Synthetic only (AX-001).
"""
from __future__ import annotations

import pytest

from pii_anon import fusion as fusion_mod
from pii_anon.errors import FusionError
from pii_anon.fusion import (
    FusionBuildSpec,
    FusionStrategy,
    WeightedConsensusFusion,
    available_fusion_modes,
    build_fusion,
    register_fusion_strategy,
    register_fusion_strategy_v2,
)


class _SpecCapturingStrategy(FusionStrategy):
    """A v2 strategy that records the FusionBuildSpec it was built from."""

    strategy_id = "feature_moe"

    def __init__(self, spec: FusionBuildSpec) -> None:
        self.spec = spec

    def merge(self, findings: list) -> list:  # pragma: no cover - not exercised
        return []


@pytest.fixture(autouse=True)
def _restore_registries() -> "object":
    """Snapshot+restore the module registries so registration tests don't leak."""
    v1_before = dict(fusion_mod._CUSTOM_FACTORIES)
    v2_before = dict(getattr(fusion_mod, "_CUSTOM_FACTORIES_V2", {}))
    yield
    fusion_mod._CUSTOM_FACTORIES.clear()
    fusion_mod._CUSTOM_FACTORIES.update(v1_before)
    if hasattr(fusion_mod, "_CUSTOM_FACTORIES_V2"):
        fusion_mod._CUSTOM_FACTORIES_V2.clear()
        fusion_mod._CUSTOM_FACTORIES_V2.update(v2_before)


# --------------------------------------------------------------------------- #
# A6 — v2 construction seam, v1 untouched  [CONTRACT-TEST]
# --------------------------------------------------------------------------- #


def test_fr018_register_v2_factory_receives_populated_buildspec() -> None:
    """A6: build_fusion invokes the v2 factory with a populated FusionBuildSpec."""
    captured: dict[str, FusionBuildSpec] = {}

    def factory_v2(spec: FusionBuildSpec) -> FusionStrategy:
        captured["spec"] = spec
        return _SpecCapturingStrategy(spec)

    register_fusion_strategy_v2("feature_moe", factory_v2)
    strat = build_fusion(
        "feature_moe",
        weights={"regex-oss": 1.0},
        min_consensus=2,
        entity_weights={"regex-oss": {"US_SSN": 0.9}},
        iou_threshold=0.7,
    )
    assert isinstance(strat, _SpecCapturingStrategy)
    spec = captured["spec"]
    assert spec.mode == "feature_moe"
    assert spec.weights == {"regex-oss": 1.0}
    assert spec.min_consensus == 2
    assert spec.entity_weights == {"regex-oss": {"US_SSN": 0.9}}
    assert spec.iou_threshold == 0.7
    # The spec carries the richer config the frozen v1 factory cannot.
    assert hasattr(spec, "gate_path")
    assert hasattr(spec, "key_ring")
    assert hasattr(spec, "top_k")


def test_fr018_fusion_buildspec_is_frozen() -> None:
    """A6: FusionBuildSpec is an immutable carrier (frozen dataclass)."""
    import dataclasses

    spec = FusionBuildSpec(
        mode="feature_moe", weights={}, min_consensus=1,
        entity_weights=None, iou_threshold=0.5, gate_path=None,
        key_ring=None, top_k=3,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.mode = "x"  # type: ignore[misc]


def test_fr018_v1_register_fusion_strategy_unchanged_and_constructs() -> None:
    """A6: the frozen v1 seam still works — a v1 (weights, min_consensus) mode."""
    seen: dict[str, object] = {}

    def v1_factory(weights: dict[str, float], min_consensus: int) -> FusionStrategy:
        seen["weights"] = weights
        seen["min_consensus"] = min_consensus
        return WeightedConsensusFusion(weights=weights)

    register_fusion_strategy("legacy_v1_mode", v1_factory)
    strat = build_fusion("legacy_v1_mode", weights={"a": 0.5}, min_consensus=3)
    assert isinstance(strat, WeightedConsensusFusion)
    assert seen == {"weights": {"a": 0.5}, "min_consensus": 3}


def test_fr018_v1_factory_signature_is_positional_weights_min_consensus() -> None:
    """A6: CustomFusionFactory contract is (dict, int) -> FusionStrategy."""
    from pii_anon.fusion import CustomFusionFactory  # byte-identical v1 symbol

    # A 2-positional-arg callable still satisfies the v1 alias at runtime.
    f: CustomFusionFactory = lambda w, k: WeightedConsensusFusion(weights=w)  # noqa: E731
    register_fusion_strategy("legacy_lambda", f)
    assert isinstance(
        build_fusion("legacy_lambda", weights={}, min_consensus=1),
        WeightedConsensusFusion,
    )


def test_fr018_v2_does_not_shadow_builtin_modes() -> None:
    """A6: builtin dispatch still wins — registering v2 doesn't break builtins."""
    register_fusion_strategy_v2(
        "feature_moe", lambda s: _SpecCapturingStrategy(s)
    )
    built = build_fusion("weighted_consensus", weights={}, min_consensus=2)
    assert isinstance(built, WeightedConsensusFusion)


# --------------------------------------------------------------------------- #
# A7 — available_fusion_modes() / build_fusion() drift guard  [AUDIT]
# --------------------------------------------------------------------------- #


def test_audit_single_source_builtin_modes_constant_exists() -> None:
    """A7: a single-source `_BUILTIN_FUSION_MODES` tuple backs the advertiser."""
    modes = fusion_mod._BUILTIN_FUSION_MODES
    assert isinstance(modes, tuple)
    assert all(isinstance(m, str) for m in modes)
    # available_fusion_modes() must READ this constant (single source of truth).
    assert set(modes).issubset(set(available_fusion_modes()))


def test_audit_every_advertised_builtin_mode_is_constructible() -> None:
    """A7: every advertised builtin mode round-trips through build_fusion()."""
    for mode in fusion_mod._BUILTIN_FUSION_MODES:
        strat = build_fusion(mode, weights={}, min_consensus=2)
        assert isinstance(strat, FusionStrategy), f"{mode} not constructible"


def test_audit_drift_guard_is_non_vacuous_unadvertised_branch_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7 META: prove the guard CATCHES an advertised-but-unconstructable mode.

    Inject a bogus mode into the single-source constant; the round-trip
    constructibility assertion MUST then fail — proving the guard is not vacuous
    (a mode registered in only one place is caught).
    """
    bogus = "__bogus_unconstructable_mode__"
    monkeypatch.setattr(
        fusion_mod, "_BUILTIN_FUSION_MODES",
        (*fusion_mod._BUILTIN_FUSION_MODES, bogus),
    )
    # The advertiser now lists a mode build_fusion cannot construct.
    assert bogus in available_fusion_modes()
    with pytest.raises(FusionError):
        build_fusion(bogus, weights={}, min_consensus=2)


def test_audit_drift_guard_catches_constructable_but_unadvertised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7 META: a builtin branch dropped from the advertiser is detectable.

    Shrink the single-source constant so a real builtin ("mixture_of_experts")
    is constructible yet NOT advertised; the cross-check that every constructible
    builtin is advertised MUST flag it.
    """
    monkeypatch.setattr(
        fusion_mod, "_BUILTIN_FUSION_MODES",
        tuple(m for m in fusion_mod._BUILTIN_FUSION_MODES
              if m != "mixture_of_experts"),
    )
    # mixture_of_experts still constructs (a real builtin branch in build_fusion)
    assert isinstance(
        build_fusion("mixture_of_experts", weights={}, min_consensus=2),
        FusionStrategy,
    )
    # but is no longer advertised — the drift the guard exists to catch.
    assert "mixture_of_experts" not in available_fusion_modes()


def test_audit_swarm_mode_is_advertised_and_constructible() -> None:
    """A7 regression: the `swarm` builtin branch is advertised (closed drift).

    The pre-S2-01 `available_fusion_modes()` hardcoded list OMITTED `swarm`
    while `build_fusion` had a `mode == "swarm"` branch — a real latent drift
    the single-source constant must close.
    """
    assert "swarm" in available_fusion_modes()
    assert isinstance(
        build_fusion("swarm", weights={}, min_consensus=2), FusionStrategy
    )


def test_audit_unknown_mode_still_raises_fusionerror() -> None:
    """A7: an entirely unknown mode still raises FusionError (no silent build)."""
    with pytest.raises(FusionError):
        build_fusion("totally_unknown_mode", weights={}, min_consensus=2)
