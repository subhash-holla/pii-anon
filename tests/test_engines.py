from pii_anon.engines import EngineRegistry, RegexEngineAdapter


def test_engine_registry_register_unregister() -> None:
    registry = EngineRegistry()
    engine = RegexEngineAdapter()
    registry.register(engine)
    assert "regex-oss" in registry.ids()

    registry.unregister("regex-oss")
    assert "regex-oss" not in registry.ids()


def test_engine_health_and_capabilities_report() -> None:
    registry = EngineRegistry()
    registry.register(RegexEngineAdapter(enabled=True))
    health = registry.health_report()
    caps = registry.capabilities_report()

    assert health["regex-oss"]["healthy"] is True
    assert caps["regex-oss"].supports_streaming is True
