from pii_anon.errors import ConfigurationError, EngineExecutionError, FusionError, TokenizationError


def test_error_hierarchy_instantiation() -> None:
    errors = [
        ConfigurationError("bad config"),
        EngineExecutionError("engine failure"),
        FusionError("fusion failure"),
        TokenizationError("token failure"),
    ]
    for item in errors:
        assert isinstance(item, Exception)
