"""S7-04 — the committed NFR-009 latency-ceiling registry (RED suite).

The ceilings ARE part of the SDO completion contract (the G5 latency half reads
them), so the numeric literals are PINNED here exactly as the SDO threshold
literals are pinned in ``test_competitive_supremacy.py`` — a silent loosening of
a committed budget must fail a test.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from pii_anon.eval_framework.evaluation.latency_ceilings import (
    COMMITTED_LATENCY_CEILINGS,
    LatencyCeiling,
    ceiling_for,
)

# The four benchmark objectives (the engine-selection axis that determines the
# latency class — NFR-007's "speed profiles / accuracy profiles" reading).
_OBJECTIVES = ("speed", "balanced", "accuracy", "ensemble")


def test_registry_commits_a_ceiling_for_every_objective() -> None:
    """[CONTRACT-TEST] NFR-009: a committed NUMERIC ceiling exists per profile."""
    assert set(COMMITTED_LATENCY_CEILINGS) == set(_OBJECTIVES)
    for profile, ceiling in COMMITTED_LATENCY_CEILINGS.items():
        assert isinstance(ceiling, LatencyCeiling)
        assert ceiling.profile == profile


def test_ceiling_is_frozen() -> None:
    """[CONTRACT-TEST] A committed budget cannot be mutated at runtime."""
    ceiling = COMMITTED_LATENCY_CEILINGS["ensemble"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ceiling.p50_ms = 10_000.0  # type: ignore[misc]


def test_every_ceiling_is_finite_positive_and_percentile_ordered() -> None:
    """[CONTRACT-TEST] p50 ≤ p95 ≤ p99, all finite and > 0 (a budget of 0/-1/inf
    would be non-physical — either vacuously unpassable or vacuously loose)."""
    for ceiling in COMMITTED_LATENCY_CEILINGS.values():
        for value in (ceiling.p50_ms, ceiling.p95_ms, ceiling.p99_ms):
            assert isinstance(value, float)
            assert value > 0.0
            assert value != float("inf")
        assert ceiling.p50_ms <= ceiling.p95_ms <= ceiling.p99_ms


def test_speed_p50_is_the_nfr007_literal() -> None:
    """[CONTRACT-TEST] NFR-007: speed profiles p50 ≤ 1 ms — the committed speed
    budget IS that literal."""
    assert COMMITTED_LATENCY_CEILINGS["speed"].p50_ms == 1.0


def test_ensemble_full_swarm_budget_literals_are_pinned() -> None:
    """[CONTRACT-TEST] NFR-009: the FULL-SWARM (objective="ensemble") committed
    budget — p50 ≤ 500 ms, p95 ≤ 1000 ms, p99 ≤ 2000 ms. Grounded in
    measurement (census swarm p50 91.5 ms; fresh quiet-env p50/p95/p99 =
    80.5/112/133 ms at n=12; a SATURATED host tripled the single-shot p50 to
    ~275 ms, hence the min-of-3 producer estimator + contention headroom here):
    a REAL sub-2s-p99 commitment, NOT sub-0.24ms parity and NOT vacuous.
    Changing these is changing the program's completion contract — do it
    consciously."""
    ensemble = COMMITTED_LATENCY_CEILINGS["ensemble"]
    assert ensemble.p50_ms == 500.0
    assert ensemble.p95_ms == 1000.0
    assert ensemble.p99_ms == 2000.0


def test_detector_class_defaults_to_full_swarm_pass2_seam() -> None:
    """[CONTRACT-TEST] The R10 "per-detector-class p50/p95/p99" refinement is a
    Pass-2 seam: every committed ceiling today is the full-swarm class; a future
    per-engine-class commitment extends the registry without a schema change."""
    for ceiling in COMMITTED_LATENCY_CEILINGS.values():
        assert ceiling.detector_class == "full-swarm"


def test_ceiling_for_returns_committed_row_or_none() -> None:
    """[UNIT-TEST] ``ceiling_for`` is the shared gate/producer lookup: a known
    profile returns its committed row; an unknown / non-str / unhashable profile
    returns ``None`` — fail-closed, NEVER a crash (the artifact supplies the
    profile string, so a hostile list/dict must not reach a dict lookup)."""
    assert ceiling_for("ensemble") is COMMITTED_LATENCY_CEILINGS["ensemble"]
    assert ceiling_for("speed") is COMMITTED_LATENCY_CEILINGS["speed"]
    assert ceiling_for("bogus-profile") is None
    assert ceiling_for("") is None
    assert ceiling_for(None) is None  # type: ignore[arg-type]
    assert ceiling_for(123) is None  # type: ignore[arg-type]
    assert ceiling_for(["ensemble"]) is None  # type: ignore[arg-type]
    assert ceiling_for({"profile": "ensemble"}) is None  # type: ignore[arg-type]
    assert ceiling_for(True) is None  # type: ignore[arg-type]


def test_module_imports_stdlib_only() -> None:
    """[SECURITY-TEST] The ceiling registry is a leaf config module: it imports
    NOTHING outside the standard library (no swarm/moe/fusion/policy, no numpy,
    no pii_anon sibling) — so adding it to the SDO gate's import graph can never
    widen the import boundary."""
    import pii_anon.eval_framework.evaluation.latency_ceilings as lc

    source = Path(lc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level == 0:
                imported.add(node.module.split(".")[0])
            else:  # a relative import inside the package is a pii_anon import
                imported.add("pii_anon")
    allowed = {"dataclasses", "typing", "__future__", "collections"}
    assert imported <= allowed, f"unexpected imports: {sorted(imported - allowed)}"
