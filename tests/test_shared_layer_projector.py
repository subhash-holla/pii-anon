"""DC-01 SharedLayerProjector — recall-floor BY CONSTRUCTION.

Pins FR-016 / NFR-011 / AX-003: entities(output) ⊇ entities(shared), ZERO
violations. Also pins the swarm.py:654/658-660 leak fix (a sub-fast-pass
shared-layer regex span that the Layer-4 emission/corroboration gate would drop
is re-injected by the projector).

Property coverage uses a seeded-random generator (hypothesis is not yet a
dependency; migrating these to @given is tracked as story S6-PROP).
"""
from __future__ import annotations

import random

from pii_anon.types import EngineFinding, EnsembleFinding
from pii_anon.routing.shared_layer import (
    SHARED_FLOOR_PROVENANCE,
    ProjectionResult,
    SharedLayerProjector,
    is_shared_floor,
    span_key_engine,
    span_key_ensemble,
)


def _ef(entity_type: str, start: int, end: int, *, field_path: str | None = "text",
        language: str = "en", confidence: float = 0.95, engine_id: str = "regex-oss") -> EngineFinding:
    return EngineFinding(entity_type=entity_type, confidence=confidence, field_path=field_path,
                         span_start=start, span_end=end, engine_id=engine_id, language=language)


def _enf(entity_type: str, start: int, end: int, *, field_path: str | None = "text",
         language: str = "en", confidence: float = 0.9, engines: list[str] | None = None) -> EnsembleFinding:
    return EnsembleFinding(entity_type=entity_type, confidence=confidence, engines=engines or ["gliner-compatible"],
                           field_path=field_path, span_start=start, span_end=end, language=language)


def test_fr_016_reinjects_dropped_shared_span() -> None:
    """A shared span absent from the fused output is re-injected, tagged shared_floor."""
    shared = [_ef("CREDIT_CARD", 10, 26)]
    output: list[EnsembleFinding] = []  # downstream gate dropped everything
    result = SharedLayerProjector().project(output, shared)
    assert isinstance(result, ProjectionResult)
    keys = {span_key_ensemble(f) for f in result.findings}
    assert span_key_engine(shared[0]) in keys
    reinjected = [f for f in result.findings if is_shared_floor(f)]
    assert len(reinjected) == 1
    assert SHARED_FLOOR_PROVENANCE in (reinjected[0].explanation or "")
    assert result.violations_blocked == 1


def test_fr_016_present_shared_span_not_duplicated() -> None:
    """If the output already covers the shared span (same offset+type+field+lang), no duplicate."""
    shared = [_ef("EMAIL_ADDRESS", 0, 17)]
    output = [_enf("EMAIL_ADDRESS", 0, 17)]
    result = SharedLayerProjector().project(output, shared)
    assert len(result.findings) == 1
    assert result.violations_blocked == 0


def test_nfr_011_empty_shared_is_noop() -> None:
    output = [_enf("PERSON_NAME", 5, 12)]
    result = SharedLayerProjector().project(output, [])
    assert [span_key_ensemble(f) for f in result.findings] == [span_key_ensemble(output[0])]
    assert result.violations_blocked == 0


def test_ax_003_closes_swarm_layer4_emission_leak() -> None:
    """Reproduce the swarm.py:654 leak: a low-confidence shared regex hit that the
    emission gate would drop is preserved by the floor."""
    # shared layer found a checksum-validated card; fusion's meta_score fell below
    # emission_threshold so Layer-4 dropped it -> output omits it.
    shared = [_ef("CREDIT_CARD", 40, 56, confidence=0.55)]
    output = [_enf("PERSON_NAME", 0, 8)]  # only an unrelated span survived
    result = SharedLayerProjector().project(output, shared)
    keys = {span_key_ensemble(f) for f in result.findings}
    assert span_key_engine(shared[0]) in keys, "shared-layer span must survive the floor (AX-003)"


def test_type_carrying_relabel_does_not_cover_shared() -> None:
    """An NER relabel to a different type at the SAME offset does not 'cover' the
    shared span — the type-carrying superset re-injects the shared type (AX-003 intent)."""
    shared = [_ef("US_DRIVER_LICENSE", 3, 12)]
    output = [_enf("PERSON_NAME", 3, 12)]  # same offsets, different type
    result = SharedLayerProjector().project(output, shared)
    keys = {span_key_ensemble(f) for f in result.findings}
    assert span_key_engine(shared[0]) in keys
    assert result.violations_blocked == 1


def test_nfr_011_property_superset_invariant_seeded() -> None:
    """ZERO subset violations over many seeded-random (output, shared) pairs."""
    rng = random.Random(42)
    types = ["EMAIL_ADDRESS", "CREDIT_CARD", "PERSON_NAME", "PHONE_NUMBER", "US_SSN", "IBAN"]
    fields = ["text", "notes", None]
    langs = ["en", "es", "zh"]
    projector = SharedLayerProjector()
    for _ in range(2000):
        n_shared = rng.randint(0, 6)
        n_out = rng.randint(0, 6)
        shared = [_ef(rng.choice(types), s := rng.randint(0, 40), s + rng.randint(1, 12),
                      field_path=rng.choice(fields), language=rng.choice(langs)) for _ in range(n_shared)]
        output = [_enf(rng.choice(types), s := rng.randint(0, 40), s + rng.randint(1, 12),
                       field_path=rng.choice(fields), language=rng.choice(langs)) for _ in range(n_out)]
        result = projector.project(output, shared)
        out_keys = {span_key_ensemble(f) for f in result.findings}
        shared_keys = {span_key_engine(s) for s in shared}
        assert shared_keys <= out_keys, "recall-floor violated: a shared span was not in the output"
        # re-injected spans must not duplicate keys already present
        reinjected_keys = [span_key_ensemble(f) for f in result.findings if is_shared_floor(f)]
        assert len(reinjected_keys) == len(set(reinjected_keys))


def test_determinism_repeatable() -> None:
    """NFR-005: same inputs -> identical projected output (stable ordering)."""
    shared = [_ef("EMAIL_ADDRESS", 0, 10), _ef("CREDIT_CARD", 20, 36)]
    output = [_enf("PERSON_NAME", 5, 9)]
    r1 = SharedLayerProjector().project(output, shared)
    r2 = SharedLayerProjector().project(output, shared)
    assert [span_key_ensemble(f) for f in r1.findings] == [span_key_ensemble(f) for f in r2.findings]
