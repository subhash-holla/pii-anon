"""pii_anon.routing — MoE-router redesign + recall-floor + agentic interception.

Houses the Design-stage DC-01/02/03/13 components behind the existing fusion/
orchestrator seams. The first shipped component is the SharedLayerProjector
(DC-01), which enforces the recall-floor invariant ``entities(output) ⊇
entities(shared)`` BY CONSTRUCTION (FR-016 / NFR-011 / AX-003).
"""
from __future__ import annotations

from pii_anon.routing.shared_layer import (
    SHARED_FLOOR_PROVENANCE,
    ProjectionResult,
    SharedLayerProjector,
    is_shared_floor,
    span_key_engine,
    span_key_ensemble,
)

__all__ = [
    "SHARED_FLOOR_PROVENANCE",
    "DistilledGatePayload",
    "DistilledTopKGate",
    "FloorProjectingFusion",
    "ProjectionResult",
    "SharedLayerProjector",
    "SurvivalOracle",
    "distill_topk_gate",
    "emit_signed_gate",
    "is_shared_floor",
    "load_distilled_gate",
    "span_key_engine",
    "span_key_ensemble",
]


def __getattr__(name: str) -> object:
    """Lazily expose the cycle-prone routing members without importing eagerly.

    ``floor_fusion`` / ``distilled_gate`` / ``gate_distillation`` import from
    ``pii_anon.fusion`` / ``pii_anon.moe`` / ``pii_anon.swarm_learner``, which in
    turn import this package; a top-level import here would create a cycle.
    Resolving the name on first access keeps ``from pii_anon.routing import X``
    working while preserving the lazy-import discipline used at the build_fusion
    seam (S2-02 adds the distilled-gate runtime + its offline producer).
    """
    if name == "FloorProjectingFusion":
        from pii_anon.routing.floor_fusion import FloorProjectingFusion

        return FloorProjectingFusion
    if name in ("DistilledGatePayload", "DistilledTopKGate", "load_distilled_gate"):
        from pii_anon.routing import distilled_gate

        return getattr(distilled_gate, name)
    if name in ("SurvivalOracle", "distill_topk_gate", "emit_signed_gate"):
        from pii_anon.routing import gate_distillation

        return getattr(gate_distillation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
