"""Pair-emission regression pins + dead acceptance entries (lever #5, 2026-07-17).

The 2026-07-16 investigation claimed a CORROBORATION INVERSION (agreeing
pairs killed while singletons emit). Direct merge-level measurement with the
production calibration artifacts REFUTED the general claim: agreeing pairs
emit today. The merge-level tests below PIN that behavior — they are the
regression net for any future DS/temperature artifact refresh (the exact
inversion class an artifact swap could introduce).

What WAS genuinely dead, and is made live here:
- gliner had no plain "date" label, so the shipped by_engine DATE_TIME 0.82
  bar had no emitter at all;
- the anonymization profile's JOB_TITLE bar (per-type 0.90) sat above
  gliner's ~0.87 confidence cap — inert even in the profile it was for.
"""
from __future__ import annotations

from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
from pii_anon.types import EngineFinding


def _finding(engine_id: str, etype: str, conf: float) -> EngineFinding:
    # REAL production adapter ids (sp6 close: a non-production id is scaled
    # as identity by the temperature scaler and masks channel inertness).
    return EngineFinding(
        entity_type=etype, confidence=conf, field_path="text",
        span_start=100, span_end=115, engine_id=engine_id,
        explanation=f"{engine_id} native ner", language="en",
    )


class TestCorroboratedAcceptanceMergeLevel:
    def test_agreeing_presidio_does_not_kill_gliner_person(self) -> None:
        """Pair-emission pin: gliner PERSON_NAME @0.95 emits alone; with an
        agreeing presidio finding it must KEEP emitting. Measured green on
        2026-07-17 production artifacts — pinned against any future
        DS/temperature refresh introducing a corroboration inversion."""
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([
            _finding("gliner-compatible", "PERSON_NAME", 0.95),
            _finding("presidio-compatible", "PERSON_NAME", 0.85),
        ])
        spans = [(f.span_start, f.span_end, str(f.entity_type)) for f in merged]
        assert (100, 115, "PERSON_NAME") in spans, f"got {spans}"

    def test_agreement_cannot_lower_the_bar(self) -> None:
        """Two sub-bar engines agreeing is still not enough: gliner 0.85 +
        presidio 0.85 PERSON_NAME (bar 0.92 for both) stays gated."""
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([
            _finding("gliner-compatible", "PERSON_NAME", 0.85),
            _finding("presidio-compatible", "PERSON_NAME", 0.85),
        ])
        assert not any(
            f.span_start == 100 and str(f.entity_type) == "PERSON_NAME" for f in merged
        )

    def test_singleton_emission_is_unchanged(self) -> None:
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([_finding("gliner-compatible", "PERSON_NAME", 0.95)])
        spans = [(f.span_start, f.span_end, str(f.entity_type)) for f in merged]
        assert (100, 115, "PERSON_NAME") in spans


class TestDeadEntriesLive:
    def test_gliner_has_a_plain_date_label(self) -> None:
        """The shipped by_engine DATE_TIME 0.82 bar had NO emitter: gliner's
        label set carried 'date of birth' but no plain 'date'."""
        from pii_anon.engines.gliner_adapter import GLiNERAdapter
        assert "date" in GLiNERAdapter._PII_LABELS
        assert GLiNERAdapter._LABEL_MAP.get("date") == "DATE_TIME"

    def test_profile_job_title_bar_is_reachable_by_gliner(self) -> None:
        """gliner confidence caps ~0.87; the profile's JOB_TITLE bar must have
        a gliner overlay at or under the other gliner semantic bars."""
        cfg = SwarmConfig.anonymization_profile()
        gliner = cfg.single_engine_min_confidence_by_engine.get("gliner-compatible", {})
        assert gliner.get("JOB_TITLE") is not None and gliner["JOB_TITLE"] <= 0.87
