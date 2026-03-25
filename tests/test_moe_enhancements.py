"""Tests for MoE enhancements: calibration, manifests, similarity guard, sync bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pii_anon.calibration.store import CalibrationResult, CalibrationStore
from pii_anon.calibration.online import OnlineCalibrationConfig, OnlineCalibrator
from pii_anon.calibration.dominance import DominanceReport, DominanceViolation
from pii_anon.engines.manifest import ExpertProfileData, ManifestLoader
from pii_anon.errors import CalibrationError, ExpertManifestError
from pii_anon.moe import ExpertRegistry, ExpertSpec
from pii_anon.moe_similarity import ExpertSimilarityGuard
from pii_anon.moe_sync import MoeSyncBridge, create_default_bridge


# ═══════════════════════════════════════════════════════════════════════════
# CalibrationResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibrationResult:
    def test_round_trip(self) -> None:
        result = CalibrationResult(
            calibrated_at="2026-01-01T00:00:00Z",
            dataset="test_ds",
            engine_entity_f1={"engine1": {"EMAIL_ADDRESS": 0.95}},
            sample_counts={"engine1": {"EMAIL_ADDRESS": 100}},
        )
        data = result.to_dict()
        restored = CalibrationResult.from_dict(data)
        assert restored.calibrated_at == "2026-01-01T00:00:00Z"
        assert restored.engine_entity_f1["engine1"]["EMAIL_ADDRESS"] == 0.95

    def test_from_dict_unsupported_version(self) -> None:
        with pytest.raises(CalibrationError, match="Unsupported"):
            CalibrationResult.from_dict({"schema_version": "2.0"})

    def test_from_dict_minor_version_ok(self) -> None:
        result = CalibrationResult.from_dict({"schema_version": "1.5"})
        assert result.schema_version == "1.5"

    def test_defaults(self) -> None:
        result = CalibrationResult()
        assert result.schema_version == "1.0"
        assert result.engine_entity_f1 == {}
        assert result.skipped_engines == []


# ═══════════════════════════════════════════════════════════════════════════
# CalibrationStore Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibrationStore:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = CalibrationStore(path=tmp_path / "cal.json")
        result = CalibrationResult(
            calibrated_at="2026-01-01T00:00:00Z",
            dataset="bench",
            engine_entity_f1={"e1": {"PERSON_NAME": 0.88}},
            sample_counts={"e1": {"PERSON_NAME": 50}},
        )
        store.save(result)
        loaded = store.load()
        assert loaded is not None
        assert loaded.engine_entity_f1["e1"]["PERSON_NAME"] == 0.88

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = CalibrationStore(path=tmp_path / "missing.json")
        assert store.load() is None

    def test_env_var_override(self, tmp_path: Path) -> None:
        env_path = str(tmp_path / "env_cal.json")
        with patch.dict(os.environ, {"PII_ANON_CALIBRATION_PATH": env_path}):
            store = CalibrationStore()
            assert str(store.path) == env_path

    def test_explicit_path_overrides_env(self, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit.json"
        with patch.dict(os.environ, {"PII_ANON_CALIBRATION_PATH": "/env/path"}):
            store = CalibrationStore(path=explicit)
            assert store.path == explicit

    def test_apply_to_registry(self, tmp_path: Path) -> None:
        registry = ExpertRegistry()
        spec = ExpertSpec(
            expert_id="e1",
            display_name="E1",
            entity_strengths={"EMAIL_ADDRESS": 0.50},
        )
        registry.register_expert(spec)

        store = CalibrationStore(path=tmp_path / "cal.json")
        result = CalibrationResult(
            engine_entity_f1={"e1": {"EMAIL_ADDRESS": 0.92}},
            sample_counts={"e1": {"EMAIL_ADDRESS": 20}},
        )
        store.save(result)

        mock_router = MagicMock()
        updated = store.apply_to_registry(registry, mock_router, min_samples=10)
        assert "e1" in updated
        assert "EMAIL_ADDRESS" in updated["e1"]
        assert spec.entity_strengths["EMAIL_ADDRESS"] == 0.92
        mock_router.clear_cache.assert_called_once()

    def test_apply_skips_low_samples(self, tmp_path: Path) -> None:
        registry = ExpertRegistry()
        spec = ExpertSpec(
            expert_id="e1",
            display_name="E1",
            entity_strengths={"EMAIL_ADDRESS": 0.50},
        )
        registry.register_expert(spec)

        store = CalibrationStore(path=tmp_path / "cal.json")
        result = CalibrationResult(
            engine_entity_f1={"e1": {"EMAIL_ADDRESS": 0.92}},
            sample_counts={"e1": {"EMAIL_ADDRESS": 3}},
        )
        store.save(result)

        updated = store.apply_to_registry(registry, None, min_samples=10)
        assert updated == {}
        assert spec.entity_strengths["EMAIL_ADDRESS"] == 0.50  # Unchanged

    def test_atomic_write(self, tmp_path: Path) -> None:
        store = CalibrationStore(path=tmp_path / "sub" / "cal.json")
        result = CalibrationResult(dataset="test")
        store.save(result)
        assert store.path.exists()
        # No temp file left behind
        assert not store.path.with_suffix(".json.tmp").exists()


# ═══════════════════════════════════════════════════════════════════════════
# OnlineCalibrator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOnlineCalibrator:
    def _make_calibrator(
        self, enabled: bool = True, min_obs: int = 2
    ) -> tuple[OnlineCalibrator, ExpertRegistry, MagicMock]:
        registry = ExpertRegistry()
        registry.register_expert(
            ExpertSpec(
                expert_id="eng1",
                display_name="Engine 1",
                entity_strengths={"EMAIL_ADDRESS": 0.80},
            )
        )
        router = MagicMock()
        config = OnlineCalibrationConfig(enabled=enabled, min_observations=min_obs, ema_alpha=0.1)
        calibrator = OnlineCalibrator(registry, router, config)
        return calibrator, registry, router

    def test_disabled_returns_none(self) -> None:
        cal, _, _ = self._make_calibrator(enabled=False)
        result = cal.observe("eng1", "EMAIL_ADDRESS", {(0, 5)}, {(0, 5)})
        assert result is None

    def test_observe_below_threshold(self) -> None:
        cal, _, _ = self._make_calibrator(min_obs=3)
        assert cal.observe("eng1", "EMAIL_ADDRESS", {(0, 5)}, {(0, 5)}) is None
        assert cal.observe("eng1", "EMAIL_ADDRESS", {(0, 5)}, {(0, 5)}) is None

    def test_observe_updates_strength(self) -> None:
        cal, registry, router = self._make_calibrator(min_obs=2)
        # Two perfect observations
        cal.observe("eng1", "EMAIL_ADDRESS", {(0, 5)}, {(0, 5)})
        new_val = cal.observe("eng1", "EMAIL_ADDRESS", {(0, 5)}, {(0, 5)})
        assert new_val is not None
        spec = registry.get_expert("eng1")
        assert spec is not None
        # EMA: alpha=0.1, observed_f1=1.0, current=0.80 -> 0.1*1.0 + 0.9*0.80 = 0.82
        assert abs(spec.entity_strengths["EMAIL_ADDRESS"] - 0.82) < 0.01
        router.clear_cache.assert_called()

    def test_floor_strength(self) -> None:
        registry = ExpertRegistry()
        registry.register_expert(
            ExpertSpec(
                expert_id="eng1",
                display_name="Engine 1",
                entity_strengths={"EMAIL_ADDRESS": 0.06},
            )
        )
        router = MagicMock()
        config = OnlineCalibrationConfig(enabled=True, min_observations=1, ema_alpha=0.9, floor_strength=0.05)
        cal = OnlineCalibrator(registry, router, config)
        # Observe with zero F1 (all misses)
        new_val = cal.observe("eng1", "EMAIL_ADDRESS", set(), {(0, 5)})
        assert new_val is not None
        assert new_val >= 0.05  # Floor enforced

    def test_reset_restores_original(self) -> None:
        cal, registry, _ = self._make_calibrator(min_obs=1)
        cal.observe("eng1", "EMAIL_ADDRESS", {(0, 5)}, {(0, 5)})
        spec = registry.get_expert("eng1")
        assert spec is not None
        assert spec.entity_strengths["EMAIL_ADDRESS"] != 0.80  # Changed
        cal.reset()
        assert spec.entity_strengths["EMAIL_ADDRESS"] == 0.80  # Restored

    def test_snapshot(self) -> None:
        cal, _, _ = self._make_calibrator()
        snap = cal.snapshot()
        assert "eng1" in snap
        assert "EMAIL_ADDRESS" in snap["eng1"]

    def test_enabled_property(self) -> None:
        cal_on, _, _ = self._make_calibrator(enabled=True)
        cal_off, _, _ = self._make_calibrator(enabled=False)
        assert cal_on.enabled is True
        assert cal_off.enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# DominanceReport / DominanceViolation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDominanceTypes:
    def test_violation_fields(self) -> None:
        v = DominanceViolation(
            entity_type="PERSON_NAME",
            best_expert_id="eng1",
            best_expert_f1=0.95,
            ensemble_f1=0.90,
            gap=0.05,
        )
        assert v.entity_type == "PERSON_NAME"
        assert v.gap == 0.05

    def test_report_passed(self) -> None:
        report = DominanceReport(passed=True, violations=[], entity_type_results={})
        assert report.passed is True

    def test_report_failed(self) -> None:
        v = DominanceViolation("EMAIL", "eng1", 0.95, 0.90, 0.05)
        report = DominanceReport(passed=False, violations=[v])
        assert report.passed is False
        assert len(report.violations) == 1


# ═══════════════════════════════════════════════════════════════════════════
# ManifestLoader Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestManifestLoader:
    def test_load_json_manifest(self, tmp_path: Path) -> None:
        data = {
            "schema_version": "1.0",
            "expert_id": "test-engine",
            "display_name": "Test Engine",
            "entity_strengths": {"EMAIL_ADDRESS": 0.9, "PERSON_NAME": 0.7},
            "default_weight": 1.0,
        }
        path = tmp_path / "expert_manifest.json"
        path.write_text(json.dumps(data))

        loader = ManifestLoader()
        profile = loader.load_from_path(path)
        assert profile["expert_id"] == "test-engine"
        assert profile["entity_strengths"]["EMAIL_ADDRESS"] == 0.9

    def test_load_yaml_manifest(self, tmp_path: Path) -> None:
        yaml_content = """
schema_version: "1.0"
expert_id: "yaml-engine"
display_name: "YAML Engine"
entity_strengths:
  EMAIL_ADDRESS: 0.85
default_weight: 1.2
"""
        path = tmp_path / "expert_manifest.yaml"
        path.write_text(yaml_content)

        loader = ManifestLoader()
        profile = loader.load_from_path(path)
        assert profile["expert_id"] == "yaml-engine"
        assert profile["entity_strengths"]["EMAIL_ADDRESS"] == 0.85

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="not found"):
            loader.load_from_path(tmp_path / "nope.json")

    def test_validate_missing_expert_id(self) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="expert_id"):
            loader.validate({"display_name": "No ID"})

    def test_validate_missing_display_name(self) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="display_name"):
            loader.validate({"expert_id": "x"})

    def test_validate_strength_out_of_range(self) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="\\[0.0, 1.0\\]"):
            loader.validate(
                {
                    "expert_id": "x",
                    "display_name": "X",
                    "entity_strengths": {"EMAIL_ADDRESS": 1.5},
                }
            )

    def test_validate_negative_strength(self) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="\\[0.0, 1.0\\]"):
            loader.validate(
                {
                    "expert_id": "x",
                    "display_name": "X",
                    "entity_strengths": {"EMAIL_ADDRESS": -0.1},
                }
            )

    def test_validate_bad_default_weight(self) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="default_weight"):
            loader.validate(
                {
                    "expert_id": "x",
                    "display_name": "X",
                    "entity_strengths": {},
                    "default_weight": 0,
                }
            )

    def test_validate_unsupported_schema(self) -> None:
        loader = ManifestLoader()
        with pytest.raises(ExpertManifestError, match="schema version"):
            loader.validate(
                {
                    "schema_version": "2.0",
                    "expert_id": "x",
                    "display_name": "X",
                }
            )

    def test_validate_valid_manifest(self) -> None:
        loader = ManifestLoader()
        result = loader.validate(
            {
                "expert_id": "test",
                "display_name": "Test",
                "entity_strengths": {"EMAIL_ADDRESS": 0.9},
                "entity_weaknesses": {"PERSON_NAME": 0.2},
                "default_weight": 1.3,
                "metadata": {"type": "rule-based"},
            }
        )
        assert result["expert_id"] == "test"
        assert result["default_weight"] == 1.3
        assert result["metadata"]["type"] == "rule-based"

    def test_load_for_adapter_with_manifest(self, tmp_path: Path) -> None:
        """Test manifest file takes priority over expert_profile()."""
        # Create a manifest file in the adapter's directory
        data = {
            "expert_id": "manifest-engine",
            "display_name": "From Manifest",
            "entity_strengths": {"EMAIL_ADDRESS": 0.99},
        }
        manifest = tmp_path / "expert_manifest.json"
        manifest.write_text(json.dumps(data))

        # Create mock adapter whose source file is in tmp_path
        adapter = MagicMock()
        adapter.adapter_id = "manifest-engine"
        adapter.expert_profile.return_value = {
            "expert_id": "manifest-engine",
            "display_name": "From Profile",
            "entity_strengths": {"EMAIL_ADDRESS": 0.50},
        }

        loader = ManifestLoader()
        with patch("inspect.getfile", return_value=str(tmp_path / "adapter.py")):
            profile = loader.load_for_adapter(adapter)

        assert profile is not None
        assert profile["display_name"] == "From Manifest"

    def test_load_for_adapter_with_expert_profile(self) -> None:
        """Test fallback to expert_profile() when no manifest exists."""
        adapter = MagicMock()
        adapter.adapter_id = "profile-engine"
        adapter.expert_profile.return_value = {
            "expert_id": "profile-engine",
            "display_name": "From Profile",
            "entity_strengths": {"PERSON_NAME": 0.88},
        }

        loader = ManifestLoader()
        with patch("inspect.getfile", side_effect=TypeError):
            profile = loader.load_for_adapter(adapter)

        assert profile is not None
        assert profile["display_name"] == "From Profile"

    def test_load_for_adapter_no_profile(self) -> None:
        """Test None when adapter has neither manifest nor profile."""
        adapter = MagicMock()
        adapter.adapter_id = "bare-engine"
        adapter.expert_profile.return_value = None

        loader = ManifestLoader()
        with patch("inspect.getfile", side_effect=TypeError):
            profile = loader.load_for_adapter(adapter)

        assert profile is None

    def test_load_real_manifests(self) -> None:
        """Test that existing engine manifest files in the repo can be loaded."""
        engines_dir = Path(__file__).parent.parent / "src" / "pii_anon" / "engines"
        loader = ManifestLoader()
        loaded_count = 0
        for p in engines_dir.glob("expert_manifest_*.yaml"):
            profile = loader.load_from_path(p)
            assert profile["expert_id"]
            assert profile["display_name"]
            assert len(profile["entity_strengths"]) > 0
            loaded_count += 1
        assert loaded_count >= 4  # At least regex, presidio, scrubadub, spacy


# ═══════════════════════════════════════════════════════════════════════════
# ExpertSimilarityGuard Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExpertSimilarityGuard:
    def _spec(self, expert_id: str, strengths: dict[str, float]) -> ExpertSpec:
        return ExpertSpec(
            expert_id=expert_id,
            display_name=expert_id,
            entity_strengths=strengths,
        )

    def test_no_existing_experts(self) -> None:
        guard = ExpertSimilarityGuard(threshold=0.95)
        candidate = self._spec("new", {"EMAIL_ADDRESS": 0.9})
        result = guard.check(candidate, [])
        assert not result.is_similar
        assert result.action_taken == "allowed"

    def test_identical_strengths_triggers(self) -> None:
        guard = ExpertSimilarityGuard(threshold=0.95)
        existing = self._spec("old", {"EMAIL_ADDRESS": 0.9, "PERSON_NAME": 0.8})
        candidate = self._spec("new", {"EMAIL_ADDRESS": 0.9, "PERSON_NAME": 0.8})
        result = guard.check(candidate, [existing])
        assert result.is_similar
        assert result.similarity_score >= 0.99
        assert result.most_similar_id == "old"
        assert result.action_taken == "warned"

    def test_different_strengths_allowed(self) -> None:
        guard = ExpertSimilarityGuard(threshold=0.95)
        existing = self._spec("regex", {"EMAIL_ADDRESS": 0.99, "US_SSN": 0.98})
        candidate = self._spec("ner", {"PERSON_NAME": 0.90, "ORGANIZATION": 0.85})
        result = guard.check(candidate, [existing])
        assert not result.is_similar
        assert result.action_taken == "allowed"

    def test_reject_action_raises(self) -> None:
        guard = ExpertSimilarityGuard(threshold=0.95, action="reject")
        existing = self._spec("old", {"EMAIL_ADDRESS": 0.9})
        candidate = self._spec("new", {"EMAIL_ADDRESS": 0.9})
        with pytest.raises(ExpertManifestError, match="too similar"):
            guard.check(candidate, [existing])

    def test_custom_threshold(self) -> None:
        guard = ExpertSimilarityGuard(threshold=0.50)
        # Both have EMAIL_ADDRESS but different values -> cosine may be > 0.5
        existing = self._spec("old", {"EMAIL_ADDRESS": 0.9, "PERSON_NAME": 0.1})
        candidate = self._spec("new", {"EMAIL_ADDRESS": 0.8, "PERSON_NAME": 0.2})
        result = guard.check(candidate, [existing])
        assert result.is_similar  # Very similar vectors

    def test_cosine_zero_vector(self) -> None:
        score = ExpertSimilarityGuard._cosine({}, {"EMAIL_ADDRESS": 0.9})
        assert score == 0.0

    def test_cosine_orthogonal(self) -> None:
        score = ExpertSimilarityGuard._cosine(
            {"EMAIL_ADDRESS": 1.0},
            {"PERSON_NAME": 1.0},
        )
        assert abs(score) < 0.01

    def test_cosine_identical(self) -> None:
        strengths = {"EMAIL_ADDRESS": 0.9, "PERSON_NAME": 0.8}
        score = ExpertSimilarityGuard._cosine(strengths, strengths)
        assert abs(score - 1.0) < 0.001

    def test_skips_self_comparison(self) -> None:
        guard = ExpertSimilarityGuard(threshold=0.5)
        spec = self._spec("same", {"EMAIL_ADDRESS": 0.9})
        result = guard.check(spec, [spec])
        assert not result.is_similar  # Should skip self


# ═══════════════════════════════════════════════════════════════════════════
# MoeSyncBridge Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMoeSyncBridge:
    def _make_engine(self, adapter_id: str, entity_types: list[str] | None = None) -> MagicMock:
        engine = MagicMock()
        engine.adapter_id = adapter_id
        engine.expert_profile.return_value = None
        caps = MagicMock()
        caps.supported_entity_types = entity_types
        engine.capabilities.return_value = caps
        return engine

    def test_register_via_capabilities_fallback(self) -> None:
        expert_registry = ExpertRegistry()
        bridge = MoeSyncBridge(expert_registry=expert_registry, default_strength=0.60)

        engine = self._make_engine("eng1", entity_types=["EMAIL_ADDRESS", "PERSON_NAME"])
        # Patch manifest loader to find nothing
        with patch.object(bridge._manifest_loader, "load_for_adapter", return_value=None):
            spec = bridge.on_engine_registered(engine)

        assert spec is not None
        assert spec.expert_id == "eng1"
        assert spec.entity_strengths["EMAIL_ADDRESS"] == 0.60
        assert expert_registry.get_expert("eng1") is not None

    def test_register_via_manifest(self) -> None:
        expert_registry = ExpertRegistry()
        bridge = MoeSyncBridge(expert_registry=expert_registry)

        engine = self._make_engine("eng1")
        profile = ExpertProfileData(
            expert_id="eng1",
            display_name="Engine 1",
            entity_strengths={"EMAIL_ADDRESS": 0.92},
        )
        with patch.object(bridge._manifest_loader, "load_for_adapter", return_value=profile):
            spec = bridge.on_engine_registered(engine)

        assert spec is not None
        assert spec.entity_strengths["EMAIL_ADDRESS"] == 0.92

    def test_skip_already_registered(self) -> None:
        expert_registry = ExpertRegistry()
        existing = ExpertSpec(
            expert_id="eng1",
            display_name="Existing",
            entity_strengths={"EMAIL_ADDRESS": 0.99},
        )
        expert_registry.register_expert(existing)
        bridge = MoeSyncBridge(expert_registry=expert_registry)

        engine = self._make_engine("eng1")
        spec = bridge.on_engine_registered(engine)
        assert spec is existing  # Returned existing, not re-registered

    def test_skip_no_profile_no_capabilities(self) -> None:
        expert_registry = ExpertRegistry()
        bridge = MoeSyncBridge(expert_registry=expert_registry)

        engine = self._make_engine("eng1", entity_types=None)
        with patch.object(bridge._manifest_loader, "load_for_adapter", return_value=None):
            spec = bridge.on_engine_registered(engine)

        assert spec is None
        assert expert_registry.get_expert("eng1") is None

    def test_unregister(self) -> None:
        expert_registry = ExpertRegistry()
        expert_registry.register_expert(
            ExpertSpec(
                expert_id="eng1",
                display_name="E1",
                entity_strengths={"EMAIL_ADDRESS": 0.9},
            )
        )
        router = MagicMock()
        bridge = MoeSyncBridge(expert_registry=expert_registry, router=router)

        bridge.on_engine_unregistered("eng1")
        assert expert_registry.get_expert("eng1") is None
        router.clear_cache.assert_called()

    def test_unregister_nonexistent_is_safe(self) -> None:
        expert_registry = ExpertRegistry()
        bridge = MoeSyncBridge(expert_registry=expert_registry)
        bridge.on_engine_unregistered("nonexistent")  # Should not raise

    def test_similarity_rejection_blocks_registration(self) -> None:
        expert_registry = ExpertRegistry()
        guard = ExpertSimilarityGuard(threshold=0.5, action="reject")
        expert_registry.register_expert(
            ExpertSpec(
                expert_id="existing",
                display_name="Existing",
                entity_strengths={"EMAIL_ADDRESS": 0.9},
            )
        )
        bridge = MoeSyncBridge(
            expert_registry=expert_registry,
            similarity_guard=guard,
        )

        engine = self._make_engine("new-eng")
        profile = ExpertProfileData(
            expert_id="new-eng",
            display_name="New",
            entity_strengths={"EMAIL_ADDRESS": 0.91},  # Very similar to existing
        )
        with patch.object(bridge._manifest_loader, "load_for_adapter", return_value=profile):
            spec = bridge.on_engine_registered(engine)

        assert spec is None  # Blocked by similarity guard

    def test_router_cache_cleared_on_register(self) -> None:
        expert_registry = ExpertRegistry()
        router = MagicMock()
        bridge = MoeSyncBridge(expert_registry=expert_registry, router=router)

        engine = self._make_engine("eng1", entity_types=["EMAIL_ADDRESS"])
        with patch.object(bridge._manifest_loader, "load_for_adapter", return_value=None):
            bridge.on_engine_registered(engine)

        router.clear_cache.assert_called()


class TestCreateDefaultBridge:
    def test_creates_bridge_with_defaults(self) -> None:
        registry = ExpertRegistry()
        bridge = create_default_bridge(registry)
        assert isinstance(bridge, MoeSyncBridge)

    def test_custom_similarity_config(self) -> None:
        registry = ExpertRegistry()
        bridge = create_default_bridge(
            registry,
            similarity_threshold=0.80,
            similarity_action="reject",
        )
        assert bridge._similarity_guard is not None
        assert bridge._similarity_guard.threshold == 0.80
        assert bridge._similarity_guard.action == "reject"


# ═══════════════════════════════════════════════════════════════════════════
# CLI Command Tests (smoke tests for new commands)
# ═══════════════════════════════════════════════════════════════════════════


class TestMoECLICommands:
    def test_calibrate_offline_registered(self) -> None:
        """Verify the calibrate-offline command is registered in the CLI app."""
        from pii_anon.cli import create_app

        app = create_app()
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "calibrate-offline" in command_names

    def test_verify_dominance_registered(self) -> None:
        """Verify the verify-dominance command is registered in the CLI app."""
        from pii_anon.cli import create_app

        app = create_app()
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "verify-dominance" in command_names


# ═══════════════════════════════════════════════════════════════════════════
# Error Types Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorTypes:
    def test_calibration_error(self) -> None:
        from pii_anon.errors import CalibrationError, PiiAnonError

        err = CalibrationError("bad calibration")
        assert isinstance(err, PiiAnonError)
        assert "bad calibration" in str(err)

    def test_expert_manifest_error(self) -> None:
        from pii_anon.errors import ExpertManifestError, PiiAnonError

        err = ExpertManifestError("bad manifest")
        assert isinstance(err, PiiAnonError)
        assert "bad manifest" in str(err)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: MoE reset_default_registry
# ═══════════════════════════════════════════════════════════════════════════


class TestMoEReset:
    def test_reset_default_registry(self) -> None:
        from pii_anon.moe import reset_default_registry

        # Just verify it doesn't crash
        reset_default_registry()
