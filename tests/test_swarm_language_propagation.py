"""S1-05 — Swarm language propagation (Sprint-1 gate remediation).

Pins FR-016 / NFR-011 (correctness of the LIVE recall-floor), NFR-024/025
(per-language fairness correctness) and AX-002 at the ``build_fusion("swarm")``
seam AND at the raw ``SwarmFusionStrategy`` emission level.

The bug (verified at HEAD, workflow ``wftzms2fs`` MAJOR): ``SwarmFusionStrategy``
emits ``EnsembleFinding`` objects WITHOUT propagating ``language``, so they fall
back to the ``"en"`` default (``types.py:124/162``). ``MoEFusionStrategy`` does it
correctly (``moe.py:431 language=representative.language``). Because the live
``SharedLayerProjector`` key is language-carrying
``(field_path, start, end, entity_type, language)``, a non-English ``regex-oss``
span that the swarm KEEPS is emitted mislabeled ``"en"`` and therefore does not
match its true-language shared key — so the projector re-injects a DUPLICATE
tagged ``shared_floor``. The identical ``en`` input emits exactly one span.

These tests assert the corrected behaviour: every swarm-emitted finding carries
the source/representative finding's language (mirror ``moe.py:431``), so the
non-English path emits exactly ONE span (``language=="es"``) with no
``shared_floor`` duplicate, while the ``en`` path is unchanged.
"""
from __future__ import annotations

from pii_anon.fusion import build_fusion
from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
from pii_anon.types import EngineFinding, EnsembleFinding


def _es_email_regex_span(language: str = "es") -> EngineFinding:
    """A high-confidence ``regex-oss`` EMAIL span the swarm fast-pass KEEPS.

    Confidence 0.97 is at/above the Layer-1 fast-pass threshold, so the span is
    emitted by the Layer-1 fast-pass emission site (``swarm.py:613-621``).
    """
    return EngineFinding(
        entity_type="EMAIL_ADDRESS",
        confidence=0.97,
        field_path="correo",
        span_start=0,
        span_end=20,
        engine_id="regex-oss",
        language=language,
    )


def _is_shared_floor(finding: EnsembleFinding) -> bool:
    """Heuristic mirroring the projector's ``shared_floor`` provenance tag."""
    return "shared_floor" in (finding.explanation or "")


# ── (i) build_fusion("swarm") es regex span → exactly ONE span, language es ──

def test_fr_016_swarm_es_kept_regex_span_emits_single_es_span_no_floor_dup() -> None:
    """FR-016 / NFR-011: a kept ``regex-oss`` es span yields ONE es span, no dup.

    The reproduction from the Sprint-1-close gate: at HEAD this returns TWO spans
    ``[(...,'en',floor=False), (...,'es',floor=True)]`` because the swarm emits
    the kept fast-pass span mislabeled ``'en'`` and the language-keyed projector
    re-injects the true-``'es'`` shared span as a ``shared_floor`` duplicate.
    After the fix the swarm propagates ``language='es'`` so the projector sees a
    matching key and re-injects nothing.
    """
    out = build_fusion("swarm", weights={}, min_consensus=1).merge(
        [_es_email_regex_span("es")]
    )

    assert len(out) == 1, (
        "expected exactly ONE span for a kept es regex span, got "
        f"{[(f.span_start, f.span_end, f.language, _is_shared_floor(f)) for f in out]}"
    )
    only = out[0]
    assert only.language == "es", (
        f"swarm-emitted span must carry source language 'es', got {only.language!r}"
    )
    assert only.span_start == 0 and only.span_end == 20
    assert not any(_is_shared_floor(f) for f in out), (
        "no shared_floor duplicate must be re-injected once languages agree"
    )


# ── (ii) raw swarm fast-pass AND Layer-4 emissions carry source language ──────

def test_nfr_024_swarm_fast_pass_emission_propagates_source_language() -> None:
    """NFR-024/025: the Layer-1 fast-pass emission (``swarm.py:613-621``) must
    carry the SOURCE finding's language, not the ``'en'`` default.

    A conf>=fast_pass_threshold ``regex-oss`` span is emitted by the fast-pass
    site; on the raw strategy (no floor wrapper) it is the only output.
    """
    strategy = SwarmFusionStrategy(SwarmConfig())
    out = strategy.merge([_es_email_regex_span("es")])

    assert len(out) == 1
    assert "fast_pass" in (out[0].explanation or ""), (
        "expected the fast-pass emission site to produce this span"
    )
    assert out[0].language == "es", (
        f"fast-pass emission must propagate source language 'es', got {out[0].language!r}"
    )


def test_nfr_024_swarm_layer4_emission_propagates_representative_language() -> None:
    """NFR-024/025: the Layer-4 emission (``swarm.py:663-675``) must carry the
    representative/source finding's language, not the ``'en'`` default.

    ``IBAN`` is in ``STRUCTURED_TYPES`` but NOT ``SEMANTIC_TYPES`` (mod-97
    checksum is authoritative), so a single below-fast-pass IBAN span skips the
    Layer-4 corroboration gate and IS emitted by the Layer-4 emission site
    (``swarm:ds=...`` explanation). This is the same proven fixture pattern as
    ``test_swarm.py::test_structured_type_single_engine_accepted``. Its emitted
    language must mirror the representative (source) finding's language.
    """
    cfg = SwarmConfig(fast_pass_threshold=0.95, emission_threshold=0.3, corroboration_min=2)
    strategy = SwarmFusionStrategy(cfg)
    findings = [
        EngineFinding(
            entity_type="IBAN", confidence=0.85, field_path="iban",
            span_start=10, span_end=30, engine_id="regex-oss", language="es",
        ),
    ]
    out = strategy.merge(findings)

    layer4 = [f for f in out if "swarm:ds=" in (f.explanation or "")]
    assert layer4, (
        "expected at least one Layer-4 emitted finding; got explanations "
        f"{[f.explanation for f in out]}"
    )
    assert all(f.language == "es" for f in layer4), (
        "Layer-4 emission must propagate representative language 'es', got "
        f"{[(f.entity_type, f.language) for f in layer4]}"
    )


def test_ax_002_swarm_non_en_languages_round_trip_unchanged_offsets() -> None:
    """AX-002: for any non-en language the swarm-kept span preserves its source
    language end-to-end through the live floor (exactly one span per language)."""
    for lang in ("es", "zh", "fr", "de"):
        out = build_fusion("swarm", weights={}, min_consensus=1).merge(
            [_es_email_regex_span(lang)]
        )
        assert len(out) == 1, (
            f"[{lang}] expected exactly ONE span, got "
            f"{[(f.span_start, f.span_end, f.language) for f in out]}"
        )
        assert out[0].language == lang, (
            f"[{lang}] swarm-emitted span must carry source language {lang!r}, "
            f"got {out[0].language!r}"
        )
        assert not any(_is_shared_floor(f) for f in out)


# ── (iii) the en input is unchanged (still ONE span, language en) ─────────────

def test_ax_002_swarm_en_input_unchanged_single_en_span() -> None:
    """AX-002 regression guard: the ``en`` input behaviour is unchanged — still
    exactly ONE span carrying ``language=='en'`` with no shared_floor duplicate.
    """
    out = build_fusion("swarm", weights={}, min_consensus=1).merge(
        [_es_email_regex_span("en")]
    )

    assert len(out) == 1, (
        "en input must still emit exactly ONE span, got "
        f"{[(f.span_start, f.span_end, f.language, _is_shared_floor(f)) for f in out]}"
    )
    assert out[0].language == "en"
    assert out[0].span_start == 0 and out[0].span_end == 20
    assert not any(_is_shared_floor(f) for f in out)
