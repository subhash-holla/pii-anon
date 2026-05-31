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
    "FloorProjectingFusion",
    "ProjectionResult",
    "SharedLayerProjector",
    "is_shared_floor",
    "span_key_engine",
    "span_key_ensemble",
]


def __getattr__(name: str) -> object:
    """Lazily expose ``FloorProjectingFusion`` without importing it eagerly.

    ``floor_fusion`` imports from ``pii_anon.fusion``, which imports this
    package; a top-level import here would create a cycle. Resolving the name on
    first access keeps ``from pii_anon.routing import FloorProjectingFusion``
    working while preserving the lazy-import discipline used at the build_fusion
    seam.
    """
    if name == "FloorProjectingFusion":
        from pii_anon.routing.floor_fusion import FloorProjectingFusion

        return FloorProjectingFusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
