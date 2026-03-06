from importlib import metadata

from pii_anon.engines import EngineAdapter, EngineRegistry
from pii_anon.types import EngineFinding, Payload


class DummyEngine(EngineAdapter):
    adapter_id = "dummy-ep"

    def detect(self, payload: Payload, context: dict[str, object]) -> list[EngineFinding]:
        return []


class DummyEntryPoint:
    def load(self):
        return DummyEngine


class DummyEntryPoints:
    def select(self, *, group: str):
        if group == "pii_anon.engines":
            return [DummyEntryPoint()]
        return []


def test_engine_entrypoint_discovery(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())
    registry = EngineRegistry()
    discovered = registry.discover_entrypoint_engines()
    assert "dummy-ep" in discovered
