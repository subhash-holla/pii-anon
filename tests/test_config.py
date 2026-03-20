from pathlib import Path

from pii_anon.config import ConfigManager
from pii_anon.config.schema import (
    ConfidenceConfig,
    CoreConfig,
    EngineRuntimeConfig,
    FusionConfig,
    RiskConfig,
    RouterConfig,
)


def test_json_config_load(tmp_path: Path) -> None:
    config_path = tmp_path / "core.json"
    config_path.write_text('{"engines": {"presidio-compatible": {"enabled": true}}}', encoding="utf-8")

    config = ConfigManager().load(config_path)
    assert config.engines["presidio-compatible"].enabled is True


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PII_CORE__LOGGING__LEVEL", "DEBUG")
    monkeypatch.setenv("PII_CORE__ENGINES__PRESIDIO_COMPATIBLE__ENABLED", "true")
    config = ConfigManager().load()
    assert config.logging.level == "DEBUG"
    assert config.engines["presidio-compatible"].enabled is True


# ═══════════════════════════════════════════════════════════════════════════
# ConfidenceConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_confidence_config_defaults() -> None:
    """Test ConfidenceConfig with default values."""
    cfg = ConfidenceConfig()
    assert cfg.context_boost == 0.10
    assert cfg.context_penalty == 0.15
    assert cfg.context_window == 50
    assert cfg.confidence_cap == 0.99
    assert cfg.confidence_floor == 0.40


def test_confidence_config_custom_values() -> None:
    """Test ConfidenceConfig with custom values."""
    cfg = ConfidenceConfig(
        context_boost=0.20,
        context_penalty=0.25,
        context_window=100,
        confidence_cap=0.95,
        confidence_floor=0.30,
    )
    assert cfg.context_boost == 0.20
    assert cfg.context_penalty == 0.25
    assert cfg.context_window == 100
    assert cfg.confidence_cap == 0.95
    assert cfg.confidence_floor == 0.30


# ═══════════════════════════════════════════════════════════════════════════
# FusionConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_fusion_config_defaults() -> None:
    """Test FusionConfig with default values."""
    cfg = FusionConfig()
    assert cfg.iou_threshold == 0.5
    assert cfg.min_gap_chars == 5


def test_fusion_config_custom_values() -> None:
    """Test FusionConfig with custom values."""
    cfg = FusionConfig(iou_threshold=0.7, min_gap_chars=10)
    assert cfg.iou_threshold == 0.7
    assert cfg.min_gap_chars == 10


# ═══════════════════════════════════════════════════════════════════════════
# RiskConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_risk_config_defaults() -> None:
    """Test RiskConfig with default values."""
    cfg = RiskConfig()
    assert cfg.low_risk_threshold == 0.90
    assert cfg.moderate_risk_threshold == 0.75


def test_risk_config_custom_values() -> None:
    """Test RiskConfig with custom values."""
    cfg = RiskConfig(low_risk_threshold=0.85, moderate_risk_threshold=0.70)
    assert cfg.low_risk_threshold == 0.85
    assert cfg.moderate_risk_threshold == 0.70


# ═══════════════════════════════════════════════════════════════════════════
# RouterConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_router_config_defaults() -> None:
    """Test RouterConfig with default values."""
    cfg = RouterConfig()
    assert cfg.ensemble_confidence_threshold == 0.70
    assert cfg.accuracy_confidence_threshold == 0.88
    assert cfg.balanced_confidence_threshold == 0.80
    assert cfg.ensemble_concurrency_cap == 8
    assert cfg.accuracy_concurrency_cap == 4
    assert cfg.balanced_concurrency_cap == 3
    assert cfg.segmentation_token_threshold == 2000


def test_router_config_custom_values() -> None:
    """Test RouterConfig with custom values."""
    cfg = RouterConfig(
        ensemble_confidence_threshold=0.65,
        accuracy_confidence_threshold=0.85,
        balanced_confidence_threshold=0.75,
        ensemble_concurrency_cap=10,
        accuracy_concurrency_cap=5,
        balanced_concurrency_cap=2,
        segmentation_token_threshold=3000,
    )
    assert cfg.ensemble_confidence_threshold == 0.65
    assert cfg.accuracy_confidence_threshold == 0.85
    assert cfg.balanced_confidence_threshold == 0.75
    assert cfg.ensemble_concurrency_cap == 10
    assert cfg.accuracy_concurrency_cap == 5
    assert cfg.balanced_concurrency_cap == 2
    assert cfg.segmentation_token_threshold == 3000


# ═══════════════════════════════════════════════════════════════════════════
# EngineRuntimeConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_engine_runtime_config_defaults() -> None:
    """Test EngineRuntimeConfig with default values."""
    cfg = EngineRuntimeConfig()
    assert cfg.enabled is True
    assert cfg.weight == 1.0
    assert cfg.timeout_ms == 1_000
    assert cfg.params == {}
    assert cfg.entity_weights == {}


def test_engine_runtime_config_with_entity_weights() -> None:
    """Test EngineRuntimeConfig with entity_weights field."""
    cfg = EngineRuntimeConfig(
        enabled=True,
        weight=1.2,
        timeout_ms=2000,
        entity_weights={"PERSON_NAME": 1.8, "ORGANIZATION": 1.6}
    )
    assert cfg.weight == 1.2
    assert cfg.timeout_ms == 2000
    assert cfg.entity_weights == {"PERSON_NAME": 1.8, "ORGANIZATION": 1.6}


def test_engine_runtime_config_with_params() -> None:
    """Test EngineRuntimeConfig with params field."""
    cfg = EngineRuntimeConfig(
        params={"key1": "value1", "key2": 42, "key3": 3.14, "key4": True}
    )
    assert cfg.params["key1"] == "value1"
    assert cfg.params["key2"] == 42
    assert cfg.params["key3"] == 3.14
    assert cfg.params["key4"] is True


# ═══════════════════════════════════════════════════════════════════════════
# CoreConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_core_config_default() -> None:
    """Test CoreConfig with default values."""
    cfg = CoreConfig.default()
    assert cfg.default_language == "en"
    assert cfg.auto_discover_engines is False
    assert "regex-oss" in cfg.engines
    assert cfg.engines["regex-oss"].enabled is True
    assert cfg.engines["regex-oss"].weight == 1.0


def test_core_config_has_new_sub_configs() -> None:
    """Test CoreConfig has all new sub-config fields."""
    cfg = CoreConfig()
    assert isinstance(cfg.confidence, ConfidenceConfig)
    assert isinstance(cfg.fusion, FusionConfig)
    assert isinstance(cfg.risk, RiskConfig)
    assert isinstance(cfg.router, RouterConfig)


def test_core_config_confidence_sub_config() -> None:
    """Test CoreConfig.confidence sub-config."""
    cfg = CoreConfig()
    assert cfg.confidence.context_boost == 0.10
    assert cfg.confidence.context_penalty == 0.15


def test_core_config_fusion_sub_config() -> None:
    """Test CoreConfig.fusion sub-config."""
    cfg = CoreConfig()
    assert cfg.fusion.iou_threshold == 0.5
    assert cfg.fusion.min_gap_chars == 5


def test_core_config_risk_sub_config() -> None:
    """Test CoreConfig.risk sub-config."""
    cfg = CoreConfig()
    assert cfg.risk.low_risk_threshold == 0.90
    assert cfg.risk.moderate_risk_threshold == 0.75


def test_core_config_router_sub_config() -> None:
    """Test CoreConfig.router sub-config."""
    cfg = CoreConfig()
    assert cfg.router.ensemble_confidence_threshold == 0.70
    assert cfg.router.accuracy_confidence_threshold == 0.88


def test_core_config_with_json_file(tmp_path: Path) -> None:
    """Test loading CoreConfig with new sub-configs from JSON."""
    config_path = tmp_path / "core.json"
    config_path.write_text(
        '''{
            "confidence": {"context_boost": 0.20},
            "fusion": {"iou_threshold": 0.7},
            "risk": {"low_risk_threshold": 0.85},
            "router": {"ensemble_concurrency_cap": 12}
        }''',
        encoding="utf-8"
    )
    config = ConfigManager().load(config_path)
    assert config.confidence.context_boost == 0.20
    assert config.fusion.iou_threshold == 0.7
    assert config.risk.low_risk_threshold == 0.85
    assert config.router.ensemble_concurrency_cap == 12


def test_engine_runtime_config_entity_weights_empty_dict() -> None:
    """Test EngineRuntimeConfig with empty entity_weights."""
    cfg = EngineRuntimeConfig(entity_weights={})
    assert cfg.entity_weights == {}


def test_core_config_engines_with_entity_weights() -> None:
    """Test CoreConfig engines can have entity_weights."""
    cfg = CoreConfig(
        engines={
            "regex-oss": EngineRuntimeConfig(
                enabled=True,
                weight=1.0,
                entity_weights={"PERSON_NAME": 1.5, "US_SSN": 2.0}
            )
        }
    )
    assert cfg.engines["regex-oss"].entity_weights["PERSON_NAME"] == 1.5
    assert cfg.engines["regex-oss"].entity_weights["US_SSN"] == 2.0
