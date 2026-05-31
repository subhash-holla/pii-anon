"""DC-01 — recall-floor LIVE on the fusion path (FR-016 / NFR-011 / AX-003).

S1-01 proved the :class:`~pii_anon.routing.shared_layer.SharedLayerProjector`
floor BY CONSTRUCTION as a standalone module. S1-02 wires it into the
``build_fusion`` seam so it actually RUNS in production:
:class:`FloorProjectingFusion` is a transparent wrapper around any inner
:class:`~pii_anon.fusion.FusionStrategy` that, after the inner strategy's
``merge``, delegates to the projector to re-inject any shared-layer
(``regex-oss``) span the inner gate dropped.

    merge(findings) = projector.project(
        inner.merge(findings),
        [f for f in findings if f.engine_id == shared_engine_id],
    ).findings

Wrapping is applied to ``swarm`` and ``mixture_of_experts`` in
``build_fusion`` (the two strategies that historically gate spans away — the
swarm Layer-4 emission/corroboration gate and the MoE expert-weight floor).
The wrapper is intentionally identity-preserving:

* ``strategy_id`` shadows the class attribute with the inner strategy's id, so
  ``build_fusion("swarm").strategy_id == "swarm"`` still holds;
* ``__getattr__`` forwards any unknown attribute to the inner strategy, so the
  wrapper is a drop-in for callers that reach for inner-strategy internals.

**Shared-source guarantee** (NFR-005 / speed): the shared set is derived from
the ``regex-oss`` subset of the *same* ``findings`` already passed to ``merge``
— regex is NEVER run a second time. Combined with the projector's stable
ordering, this keeps ``merge`` deterministic for a fixed input.

**Graceful degradation**: a profile that disables ``regex-oss`` yields an empty
shared set, so the floor is a no-op and the output is byte-identical to the
inner strategy's ``merge``.
"""
from __future__ import annotations

from typing import Any

from pii_anon.fusion import FusionStrategy
from pii_anon.routing.shared_layer import SharedLayerProjector
from pii_anon.types import EngineFinding, EnsembleFinding

#: Engine id of the always-on shared layer whose spans define the recall floor.
SHARED_ENGINE_ID = "regex-oss"


def _shared_subset(
    findings: list[EngineFinding], shared_engine_id: str
) -> list[EngineFinding]:
    """Return the shared-layer (``shared_engine_id``) subset of *findings*.

    Single source of truth for the floor: the shared set is the ``regex-oss``
    slice of the very findings already being fused — regex is never re-run, so
    the floor adds no nondeterminism and no extra detection cost (NFR-005).
    """
    return [f for f in findings if f.engine_id == shared_engine_id]


class FloorProjectingFusion(FusionStrategy):
    """Wrap an inner :class:`FusionStrategy` with the recall-floor projection.

    Parameters
    ----------
    inner:
        The fusion strategy whose ``merge`` output is floor-protected.
    projector:
        The :class:`SharedLayerProjector` enforcing
        ``entities(output) ⊇ entities(shared)``. Defaults to a fresh projector
        keyed on ``shared_engine_id``.
    shared_engine_id:
        Engine id whose spans constitute the recall floor (default
        ``"regex-oss"``).
    """

    def __init__(
        self,
        inner: FusionStrategy,
        projector: SharedLayerProjector | None = None,
        shared_engine_id: str = SHARED_ENGINE_ID,
    ) -> None:
        self._inner = inner
        self._shared_engine_id = shared_engine_id
        self._projector = projector or SharedLayerProjector(
            shared_engine_id=shared_engine_id
        )
        # Instance attribute shadows the FusionStrategy class attr so that
        # identity passes through to the inner strategy (keeps strategy_id
        # contracts intact, e.g. build_fusion("swarm").strategy_id == "swarm").
        self.strategy_id = inner.strategy_id

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Floor-protected merge.

        Runs the inner strategy, then re-injects any ``shared_engine_id`` span
        the inner gate dropped. Deterministic for a fixed input.
        """
        shared = _shared_subset(findings, self._shared_engine_id)
        merged = self._inner.merge(findings)
        return self._projector.project(merged, shared).findings

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to the wrapped inner strategy.

        Makes :class:`FloorProjectingFusion` a transparent stand-in for *inner*.
        Only reached for attributes not found on the wrapper itself; the leading
        underscore guard avoids masking ``__init__``-time lookups of private
        attributes (e.g. during unpickling) with confusing recursion.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)
