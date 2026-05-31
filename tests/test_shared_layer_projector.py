"""DC-01 SharedLayerProjector — recall-floor BY CONSTRUCTION.

Pins FR-016 / NFR-011 / AX-003: entities(output) ⊇ entities(shared), ZERO
violations. Also pins the swarm.py:654/658-660 leak fix (a sub-fast-pass
shared-layer regex span that the Layer-4 emission/corroboration gate would drop
is re-injected by the projector).

Property coverage uses a seeded-random generator (hypothesis is not yet a
dependency; migrating these to @given is tracked as story S6-PROP).
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

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


# --- hypothesis strategy for property-based superset-invariant coverage (S1-04) ---
_PROP_TYPES = ["EMAIL_ADDRESS", "CREDIT_CARD", "PERSON_NAME", "PHONE_NUMBER", "US_SSN", "IBAN"]
_PROP_FIELDS: list[str | None] = ["text", "notes", None]
_PROP_LANGS = ["en", "es", "zh"]


@st.composite
def _span_specs(draw: st.DrawFn) -> tuple[str, str | None, str, int, int]:
    """One generated span: (entity_type, field_path, language, start, end).

    Mirrors the offset domain of the retired seeded generator: start in [0, 40],
    length in [1, 12] (so span_end = start + length is always > span_start).
    """
    entity_type = draw(st.sampled_from(_PROP_TYPES))
    field_path = draw(st.sampled_from(_PROP_FIELDS))
    language = draw(st.sampled_from(_PROP_LANGS))
    start = draw(st.integers(min_value=0, max_value=40))
    length = draw(st.integers(min_value=1, max_value=12))
    return (entity_type, field_path, language, start, start + length)


@settings(max_examples=400, derandomize=True)
@given(
    shared_specs=st.lists(_span_specs(), max_size=6),
    output_specs=st.lists(_span_specs(), max_size=6),
)
def test_nfr_011_property_superset_invariant(
    shared_specs: list[tuple[str, str | None, str, int, int]],
    output_specs: list[tuple[str, str | None, str, int, int]],
) -> None:
    """NFR-011: ZERO subset violations over hypothesis-generated (output, shared) pairs.

    Property: for every generated pair, shared_keys ⊆ out_keys (the recall floor
    holds by construction) AND the re-injected shared-floor spans never duplicate
    a key already present in the projected output.
    """
    shared = [_ef(t, start, end, field_path=fp, language=lang)
              for (t, fp, lang, start, end) in shared_specs]
    output = [_enf(t, start, end, field_path=fp, language=lang)
              for (t, fp, lang, start, end) in output_specs]
    result = SharedLayerProjector().project(output, shared)
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
