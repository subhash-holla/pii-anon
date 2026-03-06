from __future__ import annotations

from pii_anon.config import ConfigManager


def test_config_manager_prefers_rebranded_env_prefix(monkeypatch) -> None:
    monkeypatch.setenv("PII_ANON__DEFAULT_LANGUAGE", "fr")
    cfg = ConfigManager().load()
    assert cfg.default_language == "fr"


def test_config_manager_accepts_legacy_env_prefix(monkeypatch) -> None:
    monkeypatch.setenv("PII_CORE__TRACKING__MIN_LINK_SCORE", "0.9")
    cfg = ConfigManager().load()
    assert cfg.tracking.min_link_score == 0.9
