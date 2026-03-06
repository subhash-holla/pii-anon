from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from pii_anon.config.schema import CoreConfig
from pii_anon.errors import ConfigurationError


class ConfigManager:
    def __init__(self, env_prefix: str = "PII_ANON__", legacy_env_prefixes: tuple[str, ...] = ("PII_CORE__",)) -> None:
        self.env_prefix = env_prefix
        self.legacy_env_prefixes = legacy_env_prefixes

    def load(self, config_path: str | Path | None = None) -> CoreConfig:
        data: dict[str, Any] = {}
        if config_path is not None:
            data = self._load_from_file(Path(config_path))

        env_data = self._load_from_env()
        merged = self._deep_merge(data, env_data)

        try:
            return CoreConfig.model_validate(merged)
        except Exception as exc:  # pydantic raises validation specific exceptions
            raise ConfigurationError(f"Invalid configuration: {exc}") from exc

    def _load_from_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ConfigurationError("Config root must be a mapping object.")
            return cast(dict[str, Any], loaded)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return self._load_yaml(path)
        raise ConfigurationError("Unsupported config file format. Use JSON or YAML.")

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigurationError(
                "YAML config requires PyYAML. Install with `pip install pyyaml` or use JSON config."
            ) from exc

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ConfigurationError("Config root must be a mapping object.")
        return cast(dict[str, Any], loaded)

    def _load_from_env(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        prefixes = (self.env_prefix, *self.legacy_env_prefixes)
        for key, value in os.environ.items():
            matched_prefix = next((prefix for prefix in prefixes if key.startswith(prefix)), None)
            if matched_prefix is None:
                continue

            path = [chunk for chunk in key[len(matched_prefix) :].split("__") if chunk]
            lowered = [chunk.lower() for chunk in path]
            if lowered and lowered[0] == "engines" and len(lowered) >= 2:
                # Shell environment variables cannot contain hyphens, so allow
                # adapter identifiers to be provided with underscores.
                lowered[1] = lowered[1].replace("_", "-")
            if not lowered:
                continue
            self._deep_set(out, lowered, self._coerce_value(value))
        return out

    def _deep_set(self, root: dict[str, Any], path: list[str], value: Any) -> None:
        current = root
        for key in path[:-1]:
            nxt = current.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                current[key] = nxt
            current = nxt
        current[path[-1]] = value

    def _deep_merge(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        out = dict(left)
        for key, right_value in right.items():
            left_value = out.get(key)
            if isinstance(left_value, dict) and isinstance(right_value, dict):
                out[key] = self._deep_merge(left_value, right_value)
            else:
                out[key] = right_value
        return out

    @staticmethod
    def _coerce_value(raw: str) -> Any:
        stripped = raw.strip()
        lowered = stripped.lower()

        if lowered in {"true", "false"}:
            return lowered == "true"

        if lowered in {"null", "none"}:
            return None

        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except ValueError:
            pass

        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return raw

        return raw
