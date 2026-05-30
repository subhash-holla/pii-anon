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
    "ProjectionResult",
    "SharedLayerProjector",
    "is_shared_floor",
    "span_key_engine",
    "span_key_ensemble",
]
