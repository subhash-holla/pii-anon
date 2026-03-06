from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
from threading import Lock
from typing import Any

from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineCapabilities


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, EngineAdapter] = {}
        self._lock = Lock()

    def register(self, engine: EngineAdapter) -> None:
        with self._lock:
            self._engines[engine.adapter_id] = engine

    def unregister(self, adapter_id: str) -> None:
        with self._lock:
            if adapter_id in self._engines:
                self._engines[adapter_id].shutdown()
                del self._engines[adapter_id]

    def get(self, adapter_id: str) -> EngineAdapter | None:
        with self._lock:
            return self._engines.get(adapter_id)

    def list_engines(self, *, include_disabled: bool = False) -> list[EngineAdapter]:
        with self._lock:
            values = list(self._engines.values())
        if include_disabled:
            return values
        return [engine for engine in values if engine.enabled]

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._engines.keys())

    def initialize(self, config_by_engine: dict[str, Any]) -> None:
        with self._lock:
            engines = list(self._engines.values())
        for engine in engines:
            engine.initialize(config_by_engine.get(engine.adapter_id, {}))

    def discover_entrypoint_engines(self, group: str = "pii_anon.engines") -> list[str]:
        discovered: list[str] = []
        try:
            entry_points = metadata.entry_points()
            selected: Iterable[metadata.EntryPoint]
            if hasattr(entry_points, "select"):
                selected = entry_points.select(group=group)
            else:
                selected = entry_points.get(group) or ()
        except Exception:
            return discovered

        for ep in selected:
            try:
                loaded = ep.load()
                if isinstance(loaded, type) and issubclass(loaded, EngineAdapter):
                    instance = loaded()
                elif isinstance(loaded, EngineAdapter):
                    instance = loaded
                else:
                    continue
            except Exception:
                continue

            self.register(instance)
            discovered.append(instance.adapter_id)
        return discovered

    def health_report(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            engines = list(self._engines.values())
        return {engine.adapter_id: engine.health_check() for engine in engines}

    def capabilities_report(self) -> dict[str, EngineCapabilities]:
        with self._lock:
            engines = list(self._engines.values())
        return {engine.adapter_id: engine.capabilities() for engine in engines}
