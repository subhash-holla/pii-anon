# Engine Plugin Guide

## Implement adapter

```python
from pii_anon.engines import EngineAdapter
from pii_anon.types import EngineFinding, Payload

class CustomAdapter(EngineAdapter):
    adapter_id = "custom-adapter"

    def detect(self, payload: Payload, context: dict[str, object]) -> list[EngineFinding]:
        return []
```

## Optional lifecycle

- `initialize(config)`
- `health_check()`
- `shutdown()`
- `capabilities()`

## Register at runtime

```python
from pii_anon import PIIOrchestrator

orch = PIIOrchestrator(token_key="change-me")
orch.register_engine(CustomAdapter(enabled=True))
print(orch.capabilities())
```

## Entry-point auto-discovery

Declare entry points under group `pii_anon.engines` in package metadata.
