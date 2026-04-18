"""Tests for the swarm fusion pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from pii_anon.types import EngineFinding


# ── SwarmConfig ──────────────────────────────────────────────────────────


class TestSwarmConfig:
    def test_default_values(self) -> None:
        from pii_anon.swarm import SwarmConfig
        cfg = SwarmConfig()
        assert cfg.fast_pass_threshold == 0.90
        assert cfg.iou_threshold == 0.3
        assert cfg.corroboration_min == 2
        assert cfg.emission_threshold == 0.50

    def test_from_json(self, tmp_path: Path) -> None:
        from pii_anon.swarm import SwarmConfig
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"fast_pass_threshold": 0.80, "iou_threshold": 0.4}))
        cfg = SwarmConfig.from_json(p)
        assert cfg.fast_pass_threshold == 0.80
        assert cfg.iou_threshold == 0.4
        assert cfg.corroboration_min == 2  # default preserved


# ── TemperatureScaler ──────────────────────────────────────────────────────


class TestTemperatureScaler:
    def test_identity_at_t1(self) -> None:
        from pii_anon.swarm import TemperatureScaler
        scaler = TemperatureScaler(temperatures={"engine-a": 1.0})
        assert abs(scaler.scale("engine-a", 0.8) - 0.8) < 0.01

    def test_higher_temp_softens(self) -> None:
        from pii_anon.swarm import TemperatureScaler
        scaler = TemperatureScaler(temperatures={"engine-a": 2.0})
        # Higher temperature should push confidence toward 0.5.
        scaled = scaler.scale("engine-a", 0.9)
        assert 0.5 < scaled < 0.9

    def test_lower_temp_sharpens(self) -> None:
        from pii_anon.swarm import TemperatureScaler
        scaler = TemperatureScaler(temperatures={"engine-a": 0.5})
        scaled = scaler.scale("engine-a", 0.8)
        assert scaled > 0.8  # sharpened toward 1.0

    def test_unknown_engine_defaults_to_t1(self) -> None:
        from pii_anon.swarm import TemperatureScaler
        scaler = TemperatureScaler(temperatures={})
        assert abs(scaler.scale("unknown", 0.7) - 0.7) < 0.01

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from pii_anon.swarm import TemperatureScaler
        scaler = TemperatureScaler(temperatures={"a": 1.5, "b": 0.8})
        p = tmp_path / "temp.json"
        scaler.save(p)
        loaded = TemperatureScaler.load(p)
        assert abs(loaded.scale("a", 0.7) - scaler.scale("a", 0.7)) < 1e-6


# ── InformativenessScorer ──────────────────────────────────────────────────


class TestInformativenessScorer:
    def test_fixed_confidence_gets_low_score(self) -> None:
        from pii_anon.swarm import InformativenessScorer
        scorer = InformativenessScorer.from_engine_findings({
            "regex": [0.55, 0.80, 0.99, 0.45, 0.92],
            "spacy": [0.82, 0.82, 0.82, 0.82, 0.82],
        })
        assert scorer.score("regex") > scorer.score("spacy")
        assert scorer.score("spacy") <= 0.2  # near-zero variance

    def test_unknown_engine_returns_default(self) -> None:
        from pii_anon.swarm import InformativenessScorer
        scorer = InformativenessScorer()
        assert scorer.score("unknown") == 0.5

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from pii_anon.swarm import InformativenessScorer
        scorer = InformativenessScorer(scores={"a": 0.9, "b": 0.1})
        p = tmp_path / "info.json"
        scorer.save(p)
        loaded = InformativenessScorer.load(p)
        assert loaded.score("a") == 0.9


# ── DawidSkeneAggregator ──────────────────────────────────────────────────


class TestDawidSkeneAggregator:
    def test_untrained_falls_back_to_majority_vote(self) -> None:
        from pii_anon.swarm import DawidSkeneAggregator
        ds = DawidSkeneAggregator()
        assert not ds.is_trained
        label, conf = ds.infer({"a": "EMAIL", "b": "EMAIL", "c": "PERSON"})
        assert label == "EMAIL"
        assert conf > 0.5

    def test_train_em_basic(self) -> None:
        from pii_anon.swarm import DawidSkeneAggregator
        # Simple scenario: 2 engines, engine_a is always correct, engine_b is random.
        annotations = []
        for i in range(20):
            true_label = "EMAIL" if i < 10 else "PERSON"
            engine_a_label = true_label  # engine_a is perfect
            engine_b_label = "PERSON"  # engine_b always says PERSON (biased)
            annotations.append({"engine_a": engine_a_label, "engine_b": engine_b_label})

        ds = DawidSkeneAggregator.train_em(annotations, max_iter=50)
        assert ds.is_trained
        # DS should learn engine_a is reliable; both agree on PERSON → PERSON.
        label, conf = ds.infer({"engine_a": "PERSON", "engine_b": "PERSON"})
        assert label == "PERSON"
        assert conf > 0.5

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        from pii_anon.swarm import DawidSkeneAggregator
        ds = DawidSkeneAggregator(
            confusion_matrices={"eng1": {"EMAIL": {"EMAIL": 0.9, "O": 0.1}}},
            class_priors={"EMAIL": 0.3, "O": 0.7},
        )
        p = tmp_path / "ds.json"
        ds.save(p)
        loaded = DawidSkeneAggregator.load(p)
        assert loaded.is_trained


# ── Engine Pruning ─────────────────────────────────────────────────────────


class TestEnginePruning:
    def test_regex_always_kept(self) -> None:
        from pii_anon.swarm import _prune_redundant_findings
        findings = [
            EngineFinding("EMAIL", 0.99, "text", 0, 20, None, "regex-oss", "en"),
            EngineFinding("EMAIL", 0.82, "text", 0, 20, None, "spacy", "en"),
            EngineFinding("EMAIL", 0.80, "text", 0, 20, None, "stanza", "en"),
        ]
        pruned = _prune_redundant_findings(findings, similarity_threshold=0.85, max_engines=2)
        engine_ids = {f.engine_id for f in pruned}
        assert "regex-oss" in engine_ids

    def test_redundant_pair_pruned(self) -> None:
        from pii_anon.swarm import _prune_redundant_findings
        # spacy and stanza detect same type → one should be pruned.
        findings = [
            EngineFinding("PERSON_NAME", 0.82, "text", 0, 10, None, "spacy", "en"),
            EngineFinding("PERSON_NAME", 0.80, "text", 0, 10, None, "stanza", "en"),
            EngineFinding("EMAIL", 0.99, "text", 20, 40, None, "regex-oss", "en"),
        ]
        pruned = _prune_redundant_findings(findings, similarity_threshold=0.85, max_engines=4)
        engine_ids = {f.engine_id for f in pruned}
        # Both or one of spacy/stanza, but not both if they overlap >= threshold.
        assert "regex-oss" in engine_ids
        assert len(engine_ids) <= 3

    def test_single_engine_unchanged(self) -> None:
        from pii_anon.swarm import _prune_redundant_findings
        findings = [
            EngineFinding("EMAIL", 0.99, "text", 0, 20, None, "regex-oss", "en"),
        ]
        pruned = _prune_redundant_findings(findings)
        assert len(pruned) == 1


# ── Logistic Fallback ──────────────────────────────────────────────────────


class TestLogisticFallback:
    def test_high_confidence_regex_structured(self) -> None:
        from pii_anon.swarm import _logistic_fallback_score
        score = _logistic_fallback_score(
            ds_confidence=0.95, corroboration_count=3,
            regex_detected=True, is_structured=True,
        )
        assert score > 0.90

    def test_low_confidence_single_engine_semantic(self) -> None:
        from pii_anon.swarm import _logistic_fallback_score
        score = _logistic_fallback_score(
            ds_confidence=0.4, corroboration_count=1,
            regex_detected=False, is_structured=False,
        )
        assert score < 0.50


# ── SwarmFusionStrategy (Integration) ────────────────────────────────────


class TestSwarmFusionStrategy:
    def test_empty_findings(self) -> None:
        from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
        strategy = SwarmFusionStrategy(config=SwarmConfig())
        result = strategy.merge([])
        assert result == []

    def test_fast_pass_high_confidence_regex(self) -> None:
        from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
        cfg = SwarmConfig(fast_pass_threshold=0.90)
        strategy = SwarmFusionStrategy(config=cfg)
        findings = [
            EngineFinding("EMAIL_ADDRESS", 0.99, "text", 10, 30, None, "regex-oss", "en"),
        ]
        result = strategy.merge(findings)
        assert len(result) == 1
        assert result[0].entity_type == "EMAIL_ADDRESS"
        assert result[0].confidence == 0.99
        assert "fast_pass" in (result[0].explanation or "")

    def test_low_confidence_regex_goes_to_layer3(self) -> None:
        from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
        cfg = SwarmConfig(fast_pass_threshold=0.90, emission_threshold=0.3, corroboration_min=1)
        strategy = SwarmFusionStrategy(config=cfg)
        findings = [
            EngineFinding("PERSON_NAME", 0.55, "text", 0, 10, None, "regex-oss", "en"),
            EngineFinding("EMAIL_ADDRESS", 0.80, "text", 20, 40, None, "gliner-compatible", "en"),
        ]
        result = strategy.merge(findings)
        # Should go through aggregation layers, not fast-pass.
        assert len(result) >= 1
        assert all("fast_pass" not in (r.explanation or "") for r in result)

    def test_corroboration_filter_suppresses_single_engine_semantic(self) -> None:
        from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
        cfg = SwarmConfig(
            fast_pass_threshold=0.95,
            corroboration_min=2,
            corroboration_override_threshold=0.99,
            emission_threshold=0.3,
        )
        strategy = SwarmFusionStrategy(config=cfg)
        findings = [
            EngineFinding("PERSON_NAME", 0.60, "text", 0, 10, None, "gliner-compatible", "en"),
        ]
        result = strategy.merge(findings)
        # Single engine PERSON_NAME should be suppressed by corroboration filter.
        assert len(result) == 0

    def test_structured_type_single_engine_accepted(self) -> None:
        from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
        cfg = SwarmConfig(fast_pass_threshold=0.95, emission_threshold=0.3, corroboration_min=2)
        strategy = SwarmFusionStrategy(config=cfg)
        findings = [
            EngineFinding("EMAIL_ADDRESS", 0.85, "text", 10, 30, None, "regex-oss", "en"),
        ]
        result = strategy.merge(findings)
        # EMAIL_ADDRESS is structured → no corroboration required.
        assert len(result) == 1

    def test_multi_engine_agreement_accepted(self) -> None:
        from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
        cfg = SwarmConfig(
            fast_pass_threshold=0.95,
            corroboration_min=2,
            emission_threshold=0.3,
            iou_threshold=0.3,
            similarity_threshold=0.99,  # high threshold so no engines are pruned
        )
        strategy = SwarmFusionStrategy(config=cfg)
        # Give engines different entity type sets so they aren't pruned as redundant.
        findings = [
            EngineFinding("PERSON_NAME", 0.70, "text", 0, 10, None, "regex-oss", "en"),
            EngineFinding("EMAIL_ADDRESS", 0.85, "text", 20, 40, None, "regex-oss", "en"),
            EngineFinding("PERSON_NAME", 0.82, "text", 0, 10, None, "gliner-compatible", "en"),
            EngineFinding("PERSON_NAME", 0.80, "text", 0, 11, None, "presidio-compatible", "en"),
            EngineFinding("PHONE_NUMBER", 0.75, "text", 50, 65, None, "presidio-compatible", "en"),
        ]
        result = strategy.merge(findings)
        # Look for the PERSON_NAME finding specifically.
        person_findings = [r for r in result if r.entity_type == "PERSON_NAME"]
        assert len(person_findings) >= 1
        assert person_findings[0].entity_type == "PERSON_NAME"

    def test_build_fusion_factory(self) -> None:
        from pii_anon.fusion import build_fusion
        strategy = build_fusion("swarm", weights={}, min_consensus=1)
        assert strategy.strategy_id == "swarm"


# ── Feature Extraction ─────────────────────────────────────────────────────


class TestFeatureExtraction:
    def test_extract_features_returns_20(self) -> None:
        from pii_anon.swarm import SpanCandidate
        from pii_anon.swarm_learner import extract_features
        candidate = SpanCandidate(
            entity_type="EMAIL_ADDRESS",
            span_start=10,
            span_end=30,
            field_path="text",
            engine_findings={
                "regex-oss": EngineFinding("EMAIL_ADDRESS", 0.99, "text", 10, 30, None, "regex-oss", "en"),
                "gliner": EngineFinding("EMAIL_ADDRESS", 0.75, "text", 10, 29, None, "gliner", "en"),
            },
            ds_confidence=0.95,
            corroboration_count=2,
        )
        features = extract_features(candidate, total_engines=6)
        # Feature count bumped from 20 → 21 in FEATURE_VERSION=2 when
        # the multilingual context-keyword feature was added to give
        # the meta-learner non-English signal coverage.  Feature 21
        # (``context_has_multilang_keywords``) defaults to 0.0 when the
        # surrounding text is empty, as it is in this fixture.
        from pii_anon.swarm_learner import FEATURE_VERSION
        assert FEATURE_VERSION == 2
        assert len(features) == 21
        assert all(isinstance(f, float) for f in features)
        assert features[0] == 0.95  # ds_confidence
        assert features[1] == 2.0   # corroboration_count
        assert features[7] == 1.0   # regex_detected
        assert features[20] == 0.0  # context_has_multilang_keywords (no text supplied)


# ── Taxonomy Mapping ───────────────────────────────────────────────────────


class TestTaxonomyMapping:
    def test_conll_per_maps_to_person_name(self) -> None:
        from pii_anon.swarm_datasets import map_entity_type
        assert map_entity_type("conll2003", "PER") == "PERSON_NAME"

    def test_conll_misc_maps_to_ignore(self) -> None:
        from pii_anon.swarm_datasets import map_entity_type
        assert map_entity_type("conll2003", "MISC") == "_IGNORE"

    def test_ai4privacy_firstname(self) -> None:
        from pii_anon.swarm_datasets import map_entity_type
        assert map_entity_type("ai4privacy", "firstname") == "PERSON_NAME"

    def test_unknown_type_passes_through(self) -> None:
        from pii_anon.swarm_datasets import map_entity_type
        assert map_entity_type("conll2003", "UNKNOWN_TYPE") == "UNKNOWN_TYPE"

    def test_i2b2_patient_maps_to_person(self) -> None:
        from pii_anon.swarm_datasets import map_entity_type
        assert map_entity_type("i2b2", "PATIENT") == "PERSON_NAME"
