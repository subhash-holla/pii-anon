"""sp7 panel #9/#4 — per-(engine,type) single-engine acceptance.

A single per-type bar cannot fit both engines: GLiNER's raw confidence caps
~0.87 for semantic types, so ORGANIZATION@0.90 is structurally inert for GLiNER
while right for Presidio. The per-(engine,type) overlay opens the GLiNER
semantic single-engine channel; it is strictly ADDITIVE (bars <= the per-type
fallback) and fail-loud on a mistyped map.
"""
from __future__ import annotations

import pytest

from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy


def test_default_has_gliner_overlay() -> None:
    cfg = SwarmConfig()
    gliner = cfg.single_engine_min_confidence_by_engine.get("gliner-compatible", {})
    assert gliner.get("ORGANIZATION") == 0.82
    assert "LOCATION" in gliner and "DATE_TIME" in gliner
    # every gliner overlay bar is <= its per-type fallback (or new) => additive
    for etype, bar in gliner.items():
        fallback = cfg.single_engine_min_confidence.get(etype)
        if fallback is not None:
            assert bar <= fallback, f"{etype} overlay {bar} > per-type {fallback} (not additive)"


def test_fail_loud_on_mistyped_overlay() -> None:
    with pytest.raises(ValueError, match="single_engine_min_confidence_by_engine"):
        SwarmConfig(single_engine_min_confidence_by_engine={"gliner-compatible": 0.9})  # type: ignore[dict-item]


class _F:
    def __init__(self, conf: float) -> None:
        self.confidence = conf


class _Cand:
    """Minimal SpanCandidate stand-in for the acceptance-conf unit."""

    def __init__(self, engine_id: str, etype: str, raw: float) -> None:
        self.engine_findings = {engine_id: _F(raw)}
        self.entity_type = etype
        self.raw_confidences = {engine_id: raw}


def test_per_engine_bar_admits_gliner_semantic_singleton() -> None:
    cfg = SwarmConfig()
    # a gliner ORGANIZATION singleton at 0.84: rejected by the per-type 0.90
    # bar, ADMITTED by the gliner overlay 0.82.
    cand = _Cand("gliner-compatible", "ORGANIZATION", 0.84)
    assert SwarmFusionStrategy._single_engine_acceptance_conf(cand, cfg) == 0.84  # type: ignore[arg-type]


def test_per_engine_overlay_does_not_open_presidio() -> None:
    cfg = SwarmConfig()
    # presidio has no overlay, so it still uses the per-type ORGANIZATION 0.90
    # bar — a 0.84 presidio ORG singleton stays rejected.
    cand = _Cand("presidio-compatible", "ORGANIZATION", 0.84)
    assert SwarmFusionStrategy._single_engine_acceptance_conf(cand, cfg) is None  # type: ignore[arg-type]
