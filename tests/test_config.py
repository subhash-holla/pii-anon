from pathlib import Path

from pii_anon.config import ConfigManager


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
