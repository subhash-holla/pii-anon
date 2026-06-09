"""The committed NFR-009 latency-ceiling registry (S7-04).

NFR-009 — *full-swarm latency per profile*: ``p50 ≤ declared per-profile budget
(committed numeric ceiling + p99; NOT sub-0.24ms parity)``. This module IS that
declared budget: a small, frozen, committed registry the SDO gate's G5 latency
half (``_g5_audit_latency``) and the canonical-run producer both read. The
committed numbers are part of the program's completion contract — they are
pinned by ``tests/test_latency_ceilings.py`` exactly as the SDO threshold
literals are pinned, so a silent loosening fails a test.

Profile axis
------------
Ceilings are keyed by the benchmark **objective** (``speed`` / ``balanced`` /
``accuracy`` / ``ensemble``) — the engine-selection axis that determines the
latency class (NFR-007's "speed profiles / accuracy profiles" reading). The
FULL SWARM — NFR-009's subject — runs as ``objective="ensemble"`` (the
``pii-anon-swarm`` benchmark system), so the ``ensemble`` row is the one a
canonical run is measured against.

Grounding (2026-06-09)
----------------------
Census full-swarm p50 = 91.5 ms (148,994 records); fresh in-env per-record
measurement p50/p95/p99 = 80.5/112/133 ms (n=12, post-warmup). The committed
ensemble budget (250/500/1000 ms) carries ~3-7.5x headroom — a REAL
sub-second-p99 commitment, deliberately NOT sub-0.24ms regex parity (the
NFR-009 carve-out) and deliberately not vacuous.

Pass-2 seam
-----------
The R10 PERSONA-CONDITIONAL re-scope asks for *per-detector-class* p50/p95/p99.
``LatencyCeiling.detector_class`` (default ``"full-swarm"``) is that seam: a
future per-engine-class commitment adds registry rows without a schema change
(BUILD-AND-FLAG; flagged on the Pass-2 list).

Import discipline
-----------------
Stdlib-only (pinned by a test): this module joins the SDO gate's import graph
and the import-boundary AST scan set, so it must never widen the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "LatencyCeiling",
    "COMMITTED_LATENCY_CEILINGS",
    "ceiling_for",
]


@dataclass(frozen=True)
class LatencyCeiling:
    """One committed per-profile latency budget (milliseconds).

    Attributes
    ----------
    profile:
        The benchmark objective this budget governs (``speed`` / ``balanced`` /
        ``accuracy`` / ``ensemble``).
    p50_ms / p95_ms / p99_ms:
        The committed ceilings — a measured percentile must be ``≤`` its
        ceiling for the G5 latency half to pass. Ordered by construction
        (``p50_ms ≤ p95_ms ≤ p99_ms``; pinned by a test).
    detector_class:
        The detector class the budget covers. ``"full-swarm"`` today; the
        per-detector-class refinement (R10) extends the registry with further
        rows — the Pass-2 seam.
    """

    profile: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    detector_class: str = "full-swarm"


# The committed NFR-009 budgets — the declared per-profile numeric ceilings.
# These literals ARE the contract (pinned by tests/test_latency_ceilings.py).
COMMITTED_LATENCY_CEILINGS: Mapping[str, LatencyCeiling] = {
    # NFR-007 literal: speed profiles p50 ≤ 1 ms (the regex path measures ~0.46 ms).
    "speed": LatencyCeiling(profile="speed", p50_ms=1.0, p95_ms=5.0, p99_ms=10.0),
    # Declared budgets for the non-ensemble objectives (regex-backed today; the
    # budget leaves room for richer engine configs without ever nearing the swarm's).
    "balanced": LatencyCeiling(
        profile="balanced", p50_ms=50.0, p95_ms=150.0, p99_ms=300.0
    ),
    "accuracy": LatencyCeiling(
        profile="accuracy", p50_ms=250.0, p95_ms=500.0, p99_ms=1000.0
    ),
    # NFR-009's subject — the FULL SWARM (objective="ensemble", the
    # pii-anon-swarm system): a real sub-second-p99 commitment with measured
    # headroom (see the module docstring grounding).
    "ensemble": LatencyCeiling(
        profile="ensemble", p50_ms=250.0, p95_ms=500.0, p99_ms=1000.0
    ),
}


def ceiling_for(profile: object) -> LatencyCeiling | None:
    """The committed :class:`LatencyCeiling` for ``profile``, or ``None``.

    Fail-closed lookup shared by the gate and the producer: the profile comes
    off an ARTIFACT, so a hostile non-str value (a list / dict — unhashable —
    or an int / bool) must return ``None`` rather than reach the dict lookup
    and crash. An unknown profile name is ``None`` — a measurement that names
    no committed budget can never be certified against one.
    """
    if not isinstance(profile, str):
        return None
    return COMMITTED_LATENCY_CEILINGS.get(profile)
