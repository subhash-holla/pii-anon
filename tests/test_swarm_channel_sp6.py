"""sp6 Wave-1 — open the swarm's NER channel (evidence-driven, cross-dataset).

Phase-A mining (6 datasets, `_enhancements/sp6-general-capabilities/_evidence/`)
measured the same loss on every dataset: the pool engines FIND gold that the
pipeline then discards. TAB/ECHR counterfactual: regex∪gliner∪presidio lifts
relaxed recall 0.091→0.582, yet presidio's contribution surviving fusion was
ZERO and half of gliner's gold-matching PERSON findings were dropped. Three
mechanisms, three fixes:

1. **Presidio label mismatch** — the adapter emitted raw presidio labels
   (``PERSON``, ``NRP``) that never type-vote with the pool vocabulary
   (``PERSON_NAME``, …), so its clusters split and died. → normalize in the
   adapter.
2. **GLiNER blind spots + window artifacts** — no organization/location/
   occupation labels at all (594 gretel + ~950 TAB org-gold had NO ML channel);
   window STARTS could fall mid-word (the end is whitespace-aligned, the
   overlap re-entry was not), producing mid-word spans ("Col⟨leen Redding⟩").
   → extend labels; word-align window starts; emission hygiene in the adapter
   (it owns the text): boundary-snapped spans only.
3. **The singleton gate** — for SEMANTIC_TYPES the corroboration override
   (0.85) is unreachable for single-engine candidates (fallback meta caps at
   sigmoid(0.5)=0.62, engine-own confidence not even an input). → an ADDITIVE
   Layer-4 acceptance branch: a single-engine semantic candidate whose
   engine-own confidence clears a per-type bar is emitted. Additive-only ⇒
   AX-003/leak-direction safe by construction.
"""
from __future__ import annotations

import pytest

from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
from pii_anon.types import EngineFinding


# ---------------------------------------------------------------------------
# 1. Presidio label normalization
# ---------------------------------------------------------------------------


class TestPresidioLabelNormalization:
    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("presidio_analyzer")
        from pii_anon.engines.presidio_adapter import PresidioAdapter

        return PresidioAdapter(enabled=True)

    def test_person_normalizes_to_pool_vocabulary(self, adapter) -> None:
        found = adapter.detect(
            {"text": "John Smith was born on 14 March 2001 in Istanbul."},
            {"language": "en"},
        )
        labels = {str(f.entity_type) for f in found}
        assert "PERSON_NAME" in labels, f"raw labels leaked: {labels}"
        assert "PERSON" not in labels

    def test_location_and_datetime_normalize(self, adapter) -> None:
        found = adapter.detect(
            {"text": "The hearing was held in Strasbourg on 14 March 2001."},
            {"language": "en"},
        )
        labels = {str(f.entity_type) for f in found}
        assert "LOCATION" in labels
        assert "DATE_TIME" in labels


# ---------------------------------------------------------------------------
# 2. GLiNER label extension + window hygiene
# ---------------------------------------------------------------------------


class TestGlinerExtension:
    @pytest.fixture(scope="class")
    def adapter(self):
        pytest.importorskip("gliner")
        from pii_anon.engines.gliner_adapter import GLiNERAdapter

        return GLiNERAdapter(enabled=True)

    def test_organization_detected(self, adapter) -> None:
        found = adapter.detect(
            {"text": "She works for the International Committee of the Red Cross in Geneva."},
            {"language": "en"},
        )
        labels = {str(f.entity_type) for f in found}
        assert "ORGANIZATION" in labels, f"got {labels}"

    def test_location_detected(self, adapter) -> None:
        found = adapter.detect(
            {"text": "He moved from Istanbul to a small town near Lyon last year."},
            {"language": "en"},
        )
        labels = {str(f.entity_type) for f in found}
        assert "LOCATION" in labels, f"got {labels}"

    def test_window_starts_are_word_aligned(self, adapter) -> None:
        """Window re-entry (end − overlap) must snap forward to a word
        boundary — a mid-word start makes the model report mid-word spans."""
        text = ("Colleen Redding attended. " + "The proceedings continued at length. " * 40) * 3
        for offset, _chunk in adapter._windows(text):
            if offset == 0:
                continue
            assert text[offset - 1].isspace(), (
                f"window starts mid-word at offset {offset}: "
                f"...{text[offset-12:offset]}|{text[offset:offset+12]}..."
            )

    def test_no_mid_word_spans_emitted(self, adapter) -> None:
        """Adapter-level emission hygiene: spans snap to word boundaries."""
        filler = "The court considered the procedural history at some length. "
        text = filler * 30 + "The witness Colleen Redding testified clearly. " + filler * 30
        for f in adapter.detect({"text": text}, {"language": "en"}):
            s, e = f.span_start, f.span_end
            assert s == 0 or not (text[s - 1].isalnum() and text[s].isalnum()), (
                f"mid-word span start: {text[max(0,s-8):e]!r} ({f.entity_type})"
            )
            assert e == len(text) or not (text[e - 1].isalnum() and text[e].isalnum()), (
                f"mid-word span end: {text[s:min(len(text),e+8)]!r} ({f.entity_type})"
            )


# ---------------------------------------------------------------------------
# 3. Fusion: single-engine acceptance (the channel opener)
# ---------------------------------------------------------------------------


def _gliner_person(conf: float) -> EngineFinding:
    # The REAL production adapter id (sp6 close: tests originally used a
    # non-production id 'gliner' that the temperature scaler treated as
    # identity, masking the channel being INERT under the real calibration).
    return EngineFinding(
        entity_type="PERSON_NAME", confidence=conf, field_path="text",
        span_start=100, span_end=115, engine_id="gliner-compatible",
        explanation="gliner native ner", language="en",
    )


class TestSingleEngineAcceptance:
    def test_high_confidence_singleton_person_is_emitted(self) -> None:
        """A gliner-only PERSON_NAME at 0.95 was structurally unemittable
        (fallback meta cap 0.62 < corroboration override 0.85). The additive
        acceptance branch must emit it."""
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([_gliner_person(0.95)])
        spans = [(f.span_start, f.span_end, str(f.entity_type)) for f in merged]
        assert (100, 115, "PERSON_NAME") in spans, f"got {spans}"

    def test_low_confidence_singleton_is_still_gated(self) -> None:
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([_gliner_person(0.55)])
        assert not any(
            f.span_start == 100 and str(f.entity_type) == "PERSON_NAME" for f in merged
        )

    def test_acceptance_is_per_type_config(self) -> None:
        """Disabling the per-type entry restores the old gate exactly."""
        cfg = SwarmConfig(single_engine_min_confidence={})
        strategy = SwarmFusionStrategy(config=cfg)
        merged = strategy.merge([_gliner_person(0.99)])
        assert not any(
            f.span_start == 100 and str(f.entity_type) == "PERSON_NAME" for f in merged
        )

    def test_untyped_singleton_not_accepted(self) -> None:
        """A type with no configured bar keeps the corroboration gate."""
        strategy = SwarmFusionStrategy()
        finding = EngineFinding(
            entity_type="CREDIT_CARD", confidence=0.99, field_path="text",
            span_start=5, span_end=21, engine_id="gliner",
            explanation="gliner native ner", language="en",
        )
        merged = strategy.merge([finding])
        assert not any(
            f.span_start == 5 and str(f.entity_type) == "CREDIT_CARD" for f in merged
        )

    def test_accepted_singleton_confidence_is_engine_own(self) -> None:
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([_gliner_person(0.93)])
        hit = [f for f in merged if f.span_start == 100][0]
        assert hit.confidence == pytest.approx(0.93, abs=1e-6)

    def test_acceptance_reads_raw_confidence_under_temperature_scaling(self) -> None:
        """CLOSE MAJOR pin: the bar must compare the engine's RAW confidence.
        Layer 3 replaces engine_findings with temperature-scaled copies
        (production temperature.json: gliner T=2.474 → raw 0.98 becomes
        0.828 < every bar), which made the channel INERT in production while
        the docstring promised engine-own semantics."""
        from pii_anon.swarm import TemperatureScaler

        class _Harsh(TemperatureScaler):  # type: ignore[misc]
            def scale(self, engine_id: str, confidence: float) -> float:
                return confidence * 0.5  # any calibration squash

        strategy = SwarmFusionStrategy(temperature_scaler=_Harsh())
        merged = strategy.merge([_gliner_person(0.95)])
        spans = [(f.span_start, f.span_end, str(f.entity_type)) for f in merged]
        assert (100, 115, "PERSON_NAME") in spans, (
            f"acceptance must key on RAW 0.95, not the scaled 0.475; got {spans}"
        )

    def test_nan_or_invalid_bar_rejects_instead_of_accepting_everything(self) -> None:
        """CLOSE low pin: a NaN/negative/bool bar silently DISABLED the gate
        (conf < NaN is False → accept-everything). Invalid bars must reject."""
        for bad_bar in (float("nan"), -1.0, 0.0, False, 2.0):
            cfg = SwarmConfig(single_engine_min_confidence={"PERSON_NAME": bad_bar})
            strategy = SwarmFusionStrategy(config=cfg)
            merged = strategy.merge([_gliner_person(0.30)])
            assert not any(
                f.span_start == 100 and str(f.entity_type) == "PERSON_NAME"
                for f in merged
            ), f"bar {bad_bar!r} must reject, never accept-everything"

    def test_wrong_typed_acceptance_map_fails_loud_at_load(self) -> None:
        """CLOSE low pin: a null/str map from from_json loaded fine and
        crashed the FIRST merge() — a deferred crash on the masking path."""
        with pytest.raises(ValueError, match="single_engine_min_confidence"):
            SwarmConfig(single_engine_min_confidence=None)  # type: ignore[arg-type]

    def test_huge_integer_bar_rejects_without_overflow(self) -> None:
        """ROUND-2 close MINOR pin: a 400-digit JSON-integer bar passed load
        checks, then float(bar) raised OverflowError in merge() — the exact
        10**400 denial class from the program's history. Must reject, not
        crash."""
        cfg = SwarmConfig(single_engine_min_confidence={"PERSON_NAME": 10**400})
        strategy = SwarmFusionStrategy(config=cfg)
        merged = strategy.merge([_gliner_person(0.99)])  # must not raise
        assert not any(
            f.span_start == 100 and str(f.entity_type) == "PERSON_NAME"
            for f in merged
        )

    def test_presidio_medical_license_stays_maskable(self) -> None:
        """ROUND-2 close MAJOR pin: remapping MEDICAL_LICENSE->NPI_NUMBER
        inverted the leak direction — the raw label IS in the orchestrator's
        SUPPORTED_ENTITY_TYPES while NPI_NUMBER is not, so a conf-1.0
        medical-license emission stopped being maskable on the
        weighted_consensus/union_high_recall paths. The adapter must leave
        MEDICAL_LICENSE untouched."""
        pytest.importorskip("presidio_analyzer")
        from pii_anon.engines.presidio_adapter import PresidioAdapter
        from pii_anon.orchestrator import SUPPORTED_ENTITY_TYPES

        assert "MEDICAL_LICENSE" not in PresidioAdapter._LABEL_MAP
        assert "MEDICAL_LICENSE" in SUPPORTED_ENTITY_TYPES
        # And no remap may ever map a supported raw label to an unsupported
        # target (the general inversion class, pinned for every entry).
        for raw, mapped in PresidioAdapter._LABEL_MAP.items():
            if raw in SUPPORTED_ENTITY_TYPES:
                assert mapped in SUPPORTED_ENTITY_TYPES, (
                    f"leak inversion: supported raw {raw!r} remapped to "
                    f"unsupported {mapped!r}"
                )

    def test_regex_fast_pass_and_floor_behavior_unchanged(self) -> None:
        """AX-003 sanity: a high-confidence regex finding still emits exactly
        as before; the acceptance branch is ADDITIVE."""
        regex_finding = EngineFinding(
            entity_type="EMAIL_ADDRESS", confidence=0.95, field_path="text",
            span_start=10, span_end=30, engine_id="regex-oss",
            explanation="regex email pattern", language="en",
        )
        strategy = SwarmFusionStrategy()
        merged = strategy.merge([regex_finding])
        assert any(
            f.span_start == 10 and str(f.entity_type) == "EMAIL_ADDRESS" for f in merged
        )


# ---------------------------------------------------------------------------
# 4. Vanilla: GPS plausibility (mining candidate 7, the date-fragment class)
# ---------------------------------------------------------------------------


class TestGpsPlausibility:
    """The date-fragment FP class ('15/09'; Nemotron coordinate P=0.072, home
    P=0.157) is dropped at EVAL ONLY. The sp6 close proved that narrowing the
    PATTERN was a production LEAK: regex-oss is the AX-003 floor source, so a
    previously-masked pair like '41, -87' reached production UNMASKED with no
    downstream restore — the sp2 showstopper class. The masking path keeps
    the permissive pattern; ``eval_cross_type_arbitration`` gates the drop."""

    @pytest.fixture(scope="class")
    def prod_engine(self):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        return RegexEngineAdapter(enabled=True)  # the PRODUCTION masking path

    @pytest.fixture(scope="class")
    def eval_engine(self):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        return RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)

    def _gps_spans(self, engine, text: str) -> list[str]:
        return [
            text[f.span_start:f.span_end]
            for f in engine.detect({"text": text}, {"language": "en"})
            if str(f.entity_type) == "GPS_COORDINATES"
        ]

    def test_real_decimal_coordinates_detected_on_both_paths(
        self, prod_engine, eval_engine
    ) -> None:
        text = "Site located at -88.7183, -124.1441 per survey."
        assert self._gps_spans(prod_engine, text) == ["-88.7183, -124.1441"]
        assert self._gps_spans(eval_engine, text) == ["-88.7183, -124.1441"]

    def test_production_keeps_masking_integer_pairs(self, prod_engine) -> None:
        """LEAK-DIRECTION pin: these pairs were masked pre-sp6 and must stay
        masked — over-masking a date fragment is the safe direction."""
        assert self._gps_spans(
            prod_engine, "Vehicle last seen near 41, -87 heading north."
        ) == ["41, -87"]
        assert self._gps_spans(
            prod_engine, "Vehicle last seen near 40.7, -74 heading north."
        ) == ["40.7, -74"]

    def test_eval_drops_undecimaled_date_fragments(self, eval_engine) -> None:
        assert self._gps_spans(eval_engine, "Invoice dated 15/09 was archived.") == []
        assert self._gps_spans(eval_engine, "Items 12, 34 were returned.") == []

    def test_eval_keeps_half_decimal_coordinates(self, eval_engine) -> None:
        """A pair with a decimal in EITHER half is coordinate-shaped, not a
        date fragment — the eval drop must not touch it."""
        assert self._gps_spans(
            eval_engine, "Anchored at 19, -155.5 off the coast."
        ) == ["19, -155.5"]


# ---------------------------------------------------------------------------
# 5. The anonymization workload profile
# ---------------------------------------------------------------------------


class TestAnonymizationProfile:
    def test_profile_widens_quasi_identifier_acceptance(self) -> None:
        cfg = SwarmConfig.anonymization_profile()
        for qtype in ("LOCATION", "DATE_TIME", "NATIONALITY", "JOB_TITLE"):
            assert qtype in cfg.single_engine_min_confidence
        # The default map stays home-safe: no quasi-identifier singletons.
        default = SwarmConfig()
        for qtype in ("LOCATION", "DATE_TIME", "NATIONALITY", "JOB_TITLE"):
            assert qtype not in default.single_engine_min_confidence

    def test_profile_accepts_location_singleton(self) -> None:
        cfg = SwarmConfig.anonymization_profile()
        strategy = SwarmFusionStrategy(config=cfg)
        finding = EngineFinding(
            entity_type="LOCATION", confidence=0.9, field_path="text",
            span_start=40, span_end=50, engine_id="gliner",
            explanation="gliner native ner", language="en",
        )
        merged = strategy.merge([finding])
        assert any(
            f.span_start == 40 and str(f.entity_type) == "LOCATION" for f in merged
        )

    def test_default_config_drops_location_singleton(self) -> None:
        strategy = SwarmFusionStrategy()
        finding = EngineFinding(
            entity_type="LOCATION", confidence=0.9, field_path="text",
            span_start=40, span_end=50, engine_id="gliner",
            explanation="gliner native ner", language="en",
        )
        merged = strategy.merge([finding])
        assert not any(
            f.span_start == 40 and str(f.entity_type) == "LOCATION" for f in merged
        )
