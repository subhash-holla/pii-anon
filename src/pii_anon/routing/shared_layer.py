"""Recall-floor by construction (DC-01 — FR-016 / NFR-011 / AX-003).

``SharedLayerProjector`` is the SINGLE post-fusion chokepoint that guarantees

    entities(output) ⊇ entities(shared)

by re-injecting any shared-layer span that a downstream emission / corroboration
gate dropped. Both ``MoEFusionStrategy`` and ``SwarmFusionStrategy`` delegate to
it, unifying the two historically-divergent floor mechanisms — the MoE
floor-weight (``moe.py``) and the swarm Layer-4 emission/corroboration gate
(``swarm.py`` ~lines 654/658-660) — into one *by-construction* invariant.

The floor is a PROJECTION, decoupled from any learned router: it holds for any
present or future routing/early-exit policy. The match key is *type-carrying*
``(field_path, span_start, span_end, entity_type, language)`` so that an NER
relabel of the same offsets to a different type does NOT count as covering a
shared-layer span (AX-003's type-carrying intent).

Graceful degradation: an empty shared set makes ``project`` a no-op, so callers
that never configured a shared layer see byte-identical behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pii_anon.types import EngineFinding, EnsembleFinding

#: Provenance marker stamped on re-injected (floor-restored) findings.
SHARED_FLOOR_PROVENANCE = "shared_floor"
_FLOOR_MARKER = f"provenance={SHARED_FLOOR_PROVENANCE}"

#: Type-carrying span identity: (field_path, span_start, span_end, entity_type, language).
SpanKey = tuple[str | None, int | None, int | None, str, str]


def span_key_engine(finding: EngineFinding) -> SpanKey:
    """Type-carrying identity key for a shared-layer (engine) finding."""
    return (finding.field_path, finding.span_start, finding.span_end,
            finding.entity_type, finding.language)


def span_key_ensemble(finding: EnsembleFinding) -> SpanKey:
    """Type-carrying identity key for a fused (ensemble) finding."""
    return (finding.field_path, finding.span_start, finding.span_end,
            finding.entity_type, finding.language)


def is_shared_floor(finding: EnsembleFinding) -> bool:
    """True iff *finding* was re-injected by the projector (floor-restored)."""
    return finding.explanation is not None and _FLOOR_MARKER in finding.explanation


@dataclass(slots=True)
class ProjectionResult:
    """Output of :meth:`SharedLayerProjector.project`.

    ``findings`` is the floor-guaranteed result; ``reinjected`` lists the span
    keys that a downstream gate had dropped and the projector restored — an
    audit channel for FR-008 (canonical-run) / compliance (graft from the
    Design Proposal-B audit record).
    """

    findings: list[EnsembleFinding]
    reinjected: list[SpanKey] = field(default_factory=list)

    @property
    def violations_blocked(self) -> int:
        """Number of recall-floor violations the projector blocked."""
        return len(self.reinjected)


@dataclass(slots=True)
class SharedLayerProjector:
    """Enforce ``entities(output) ⊇ entities(shared)`` by re-injection.

    Parameters
    ----------
    shared_engine_id:
        Fallback engine id recorded on a re-injected finding when the source
        shared finding has none (defaults to the always-on ``regex-oss``).
    """

    shared_engine_id: str = "regex-oss"

    def project(
        self,
        output: list[EnsembleFinding],
        shared: list[EngineFinding],
    ) -> ProjectionResult:
        """Return *output* augmented so its span set is a superset of *shared*.

        Any shared span absent from *output* (by type-carrying key) is appended
        as an :class:`EnsembleFinding` tagged with the ``shared_floor``
        provenance marker, at the shared layer's own confidence. Ordering is
        stable: original output first, then re-injected spans in input order —
        making the result deterministic for a fixed input (NFR-005).
        """
        present: set[SpanKey] = {span_key_ensemble(f) for f in output}
        result: list[EnsembleFinding] = list(output)
        reinjected: list[SpanKey] = []
        seen: set[SpanKey] = set()
        for src in shared:
            key = span_key_engine(src)
            if key in present or key in seen:
                continue
            seen.add(key)
            reinjected.append(key)
            result.append(
                EnsembleFinding(
                    entity_type=src.entity_type,
                    confidence=src.confidence,
                    engines=[src.engine_id or self.shared_engine_id],
                    field_path=src.field_path,
                    span_start=src.span_start,
                    span_end=src.span_end,
                    explanation=(
                        f"{_FLOOR_MARKER}: re-injected by SharedLayerProjector "
                        "(downstream gate dropped a shared-layer span)"
                    ),
                    language=src.language,
                )
            )
        return ProjectionResult(findings=result, reinjected=reinjected)
