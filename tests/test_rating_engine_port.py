"""Contract tests for S3-01: RatingEnginePort + RatingEngineRegistry.

Implements:
    FR-003  — rating-engine abstraction (structural Protocol port).
    NFR-026 — graceful degradation / entry-point registry discovery.

Design trace: D-EVAL DECISION 2 (RatingEnginePort 3-tier ladder; the
``glicko-legacy`` facade is the verbatim fallback tier registered via the
``pii_anon.rating_engines`` entry-point group).

These tests pin the MINIMAL port surface the 4 production callers use:
``run_round_robin`` and ``get_rating`` (plus a no-arg-callable ``__init__``).
The richer engine API stays on the concrete class and is intentionally NOT
asserted here so the port is not over-constrained.
"""

from __future__ import annotations

from pii_anon.eval_framework.rating import (
    RatingEnginePort,
    RatingEngineRegistry,
)
from pii_anon.eval_framework.rating.elo import (
    EloRating,
    PIIRateEloEngine,
    RatingUpdate,
)

RATING_GROUP = "pii_anon.rating_engines"


# ---------------------------------------------------------------------------
# fr_003: PIIRateEloEngine satisfies RatingEnginePort structurally
# ---------------------------------------------------------------------------

def test_fr_003_elo_engine_is_a_rating_engine_port_runtime_checkable() -> None:
    """Given the runtime-checkable Protocol, when an Elo engine instance is
    tested with isinstance, then it satisfies RatingEnginePort with ZERO
    inheritance (structural) — proving zero call-site changes are required."""
    engine = PIIRateEloEngine()
    assert isinstance(engine, RatingEnginePort)


def test_fr_003_port_exposes_exactly_the_minimal_caller_surface() -> None:
    """The port must expose ONLY the methods the production callers use:
    run_round_robin + get_rating. Over-constraining the port (e.g. adding
    tournament/governance methods) would break the structural contract for
    future tiers, so we assert the callable surface is present and minimal."""
    assert hasattr(RatingEnginePort, "run_round_robin")
    assert hasattr(RatingEnginePort, "get_rating")
    # Richer engine-only API must NOT be part of the port contract.
    for engine_only in (
        "run_reidentification_tournament",
        "tournament_summary",
        "evaluate_governance",
        "update_from_match",
        "get_leaderboard",
    ):
        assert not hasattr(RatingEnginePort, engine_only), (
            f"{engine_only} must stay on the concrete class, not the port"
        )


def test_fr_003_port_methods_are_behaviourally_usable_via_elo() -> None:
    """An object typed as the port can be exercised through the port surface:
    run_round_robin returns RatingUpdate records and get_rating returns an
    EloRating (or None) — confirming the signatures align with the engine."""
    port: RatingEnginePort = PIIRateEloEngine()
    updates = port.run_round_robin({"sys_a": 0.9, "sys_b": 0.4})
    assert isinstance(updates, list)
    assert all(isinstance(u, RatingUpdate) for u in updates)

    rating = port.get_rating("sys_a")
    assert isinstance(rating, EloRating)
    assert port.get_rating("does-not-exist") is None


# ---------------------------------------------------------------------------
# nfr_026: RatingEngineRegistry discovery + graceful degradation
# ---------------------------------------------------------------------------

def test_nfr_026_discover_entrypoint_engines_finds_glicko_legacy() -> None:
    """Given the entry-point group, when discover_entrypoint_engines runs,
    then 'glicko-legacy' is discovered and resolves to a PIIRateEloEngine."""
    registry = RatingEngineRegistry()
    discovered = registry.discover_entrypoint_engines(RATING_GROUP)
    assert isinstance(discovered, list)
    assert "glicko-legacy" in discovered

    engine = registry.get("glicko-legacy")
    assert isinstance(engine, PIIRateEloEngine)
    # The discovered engine satisfies the port too.
    assert isinstance(engine, RatingEnginePort)


def test_nfr_026_discovery_degrades_gracefully_on_absent_group() -> None:
    """Given an absent / unknown entry-point group, when discovery runs,
    then it returns an empty list rather than raising (NFR-026 graceful
    degradation when optional extras are not installed)."""
    registry = RatingEngineRegistry()
    discovered = registry.discover_entrypoint_engines(
        "pii_anon.rating_engines_does_not_exist"
    )
    assert discovered == []


def test_nfr_026_registry_register_get_round_trip() -> None:
    """The registry supports a register/get round-trip keyed by name."""
    registry = RatingEngineRegistry()
    engine = PIIRateEloEngine()
    registry.register("glicko-legacy", engine)
    assert registry.get("glicko-legacy") is engine
    assert registry.get("unregistered") is None


def test_nfr_026_registry_get_unknown_returns_none() -> None:
    """get() on an empty registry returns None (no KeyError)."""
    assert RatingEngineRegistry().get("anything") is None
