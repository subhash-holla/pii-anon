from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from pii_anon.config import ConfigManager
from pii_anon.errors import ConfigurationError


def test_config_file_errors(tmp_path: Path) -> None:
    manager = ConfigManager()
    with pytest.raises(ConfigurationError, match="Config file not found"):
        manager.load(tmp_path / "missing.json")

    bad = tmp_path / "core.txt"
    bad.write_text("k=v", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported config file format"):
        manager.load(bad)


def test_invalid_json_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "core.json"
    path.write_text('["not-a-mapping"]', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Config root must be a mapping object"):
        ConfigManager().load(path)


def test_yaml_loader_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "core.yaml"
    path.write_text("k: v", encoding="utf-8")

    class YamlNone:
        @staticmethod
        def safe_load(_: str) -> None:
            return None

    class YamlList:
        @staticmethod
        def safe_load(_: str) -> list[str]:
            return ["x"]

    monkeypatch.setitem(sys.modules, "yaml", YamlNone)
    assert ConfigManager._load_yaml(path) == {}

    monkeypatch.setitem(sys.modules, "yaml", YamlList)
    with pytest.raises(ConfigurationError, match="Config root must be a mapping object"):
        ConfigManager._load_yaml(path)


def test_yaml_loader_import_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "core.yaml"
    path.write_text("k: v", encoding="utf-8")

    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "yaml":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ConfigurationError, match="YAML config requires PyYAML"):
        ConfigManager._load_yaml(path)


def test_coerce_value_variants() -> None:
    assert ConfigManager._coerce_value("true") is True
    assert ConfigManager._coerce_value("none") is None
    assert ConfigManager._coerce_value("12") == 12
    assert ConfigManager._coerce_value("12.5") == 12.5
    assert ConfigManager._coerce_value('{"a":1}') == {"a": 1}
    assert ConfigManager._coerce_value("{bad}") == "{bad}"


def test_invalid_config_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "core.json"
    path.write_text('{"stream": {"max_concurrency": "bad"}}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        ConfigManager().load(path)
