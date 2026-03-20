# Engine Plugin Guide

pii-anon's detection pipeline is built on pluggable engine adapters. You can add your own detection engine to either offering:

- **pii-anon** uses the built-in regex engine only (for sub-millisecond speed)
- **pii-anon-ensemble** fuses multiple engines through Mixture-of-Experts routing

This guide shows how to build a custom engine, register it with the ensemble, and tune its MoE weights.

---

## Step 1: Implement the Adapter

Every engine must subclass `EngineAdapter` and implement `detect()`:

```python
from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineFinding, Payload

class MyNERAdapter(EngineAdapter):
    adapter_id = "my-ner-engine"            # Unique ID (used in config and MoE registry)
    native_dependency = "my_ner_package"     # Optional: pip package name for availability checks

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._model = None

    def initialize(self, config: dict | None = None) -> None:
        """Load model / warm up resources. Called once at startup."""
        import my_ner_package
        self._model = my_ner_package.load("en")

    def detect(self, payload: Payload, context: dict) -> list[EngineFinding]:
        """Core detection method. Must return a list of EngineFinding objects."""
        text = payload.get("text", "")
        language = context.get("language", "en")
        findings = []

        for entity in self._model.predict(text):
            findings.append(EngineFinding(
                entity_type=entity.label,           # e.g., "PERSON_NAME", "ORGANIZATION"
                confidence=entity.score,             # float in [0, 1]
                span_start=entity.start,             # character offset (inclusive)
                span_end=entity.end,                 # character offset (exclusive)
                engine_id=self.adapter_id,           # must match adapter_id
                language=language,
                explanation=f"my-ner: {entity.label} ({entity.score:.2f})",
            ))

        return findings

    def health_check(self) -> dict:
        """Optional: return diagnostic info."""
        return {"status": "ok", "model_loaded": self._model is not None}

    def shutdown(self) -> None:
        """Optional: release resources."""
        self._model = None
```

### Required fields on `EngineFinding`

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_type` | str | Yes | PII category (e.g., `"EMAIL_ADDRESS"`, `"PERSON_NAME"`) |
| `confidence` | float | Yes | Detection confidence in [0, 1] |
| `span_start` | int | Yes | Start character offset in the text |
| `span_end` | int | Yes | End character offset (exclusive) |
| `engine_id` | str | Yes | Must match your `adapter_id` |
| `language` | str | No | ISO 639-1 code (default: `"en"`) |
| `explanation` | str | No | Human-readable rationale |
| `field_path` | str | No | Field name if payload has multiple fields |

---

## Step 2: Register the Engine

### Option A: Runtime registration (simplest)

```python
from pii_anon import PIIOrchestrator

orch = PIIOrchestrator(token_key="your-secret-key")
orch.register_engine(MyNERAdapter(enabled=True))

# Verify it's registered
print(orch.list_engines())  # [..., 'my-ner-engine']
```

### Option B: Entry-point auto-discovery (for published packages)

Add to your package's `pyproject.toml`:

```toml
[project.entry-points."pii_anon.engines"]
my_ner = "my_package.adapter:MyNERAdapter"
```

Then enable auto-discovery in pii-anon config:

```yaml
auto_discover_engines: true
```

Or via environment variable:

```bash
export PII_ANON__AUTO_DISCOVER_ENGINES=true
```

### Option C: Configuration file

Enable/disable and configure engines via YAML:

```yaml
engines:
  my-ner-engine:
    enabled: true
    weight: 1.2                          # Global weight in consensus fusion
    timeout_ms: 2000                     # Per-record timeout
    entity_weights:                      # Per-entity-type weight overrides
      PERSON_NAME: 1.5
      ORGANIZATION: 1.3
    params:                              # Engine-specific parameters
      model_name: "en_large"
      batch_size: 32
```

---

## Step 3: Register with the MoE Expert Registry

For your engine to participate in the Mixture-of-Experts routing (used by pii-anon-ensemble), register an `ExpertSpec` that declares which entity types your engine is strong at:

```python
from pii_anon.moe import ExpertRegistry, ExpertSpec, get_default_registry

# Get the default registry (pre-populated with built-in experts)
registry = get_default_registry()

# Register your engine's expertise profile
registry.register_expert(ExpertSpec(
    expert_id="my-ner-engine",           # Must match adapter_id
    display_name="My Custom NER",
    entity_strengths={                    # Scores in [0, 1] — how good is this engine?
        "PERSON_NAME": 0.90,             # Excellent at names
        "ORGANIZATION": 0.85,            # Good at orgs
        "LOCATION": 0.80,               # Good at locations
        "DATE_OF_BIRTH": 0.60,          # Moderate at dates
    },
    default_weight=1.2,                  # Fallback weight for unlisted entity types
))
```

The MoE router uses `entity_strengths` to decide which experts to activate for each entity type (top-K selection), how much weight each expert's findings receive during fusion, and whether to apply the performance floor guarantee.

**Tip:** Set `entity_strengths` conservatively at first. Run the benchmark and tune based on per-entity F1 scores.

---

## Step 4: Configure MoE Fusion Parameters

Fine-tune how the ensemble fuses findings from all engines:

```yaml
moe:
  top_k: 3                              # How many experts to activate per entity type
  performance_floor: true                # Guarantee: ensemble >= best individual expert
  min_expert_weight: 0.15               # Floor weight for non-routed experts
  iou_threshold: 0.5                    # Span overlap threshold for clustering
```

| Parameter | Default | Description |
|---|---|---|
| `top_k` | 3 | Number of top-scoring experts activated per entity type |
| `performance_floor` | true | When true, all experts contribute (with floor weight) |
| `min_expert_weight` | 0.15 | Minimum weight for non-routed experts |
| `iou_threshold` | 0.5 | IoU threshold for clustering overlapping spans |

---

## Example: Adding a Hugging Face NER Model

```python
from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineFinding, Payload

class HuggingFaceNERAdapter(EngineAdapter):
    adapter_id = "hf-ner"
    native_dependency = "transformers"

    def initialize(self, config=None):
        from transformers import pipeline
        model = (config or {}).get("model", "dslim/bert-base-NER")
        self._pipe = pipeline("ner", model=model, aggregation_strategy="simple")

    def detect(self, payload: Payload, context: dict) -> list[EngineFinding]:
        text = payload.get("text", "")
        results = self._pipe(text)
        TYPE_MAP = {"PER": "PERSON_NAME", "ORG": "ORGANIZATION", "LOC": "LOCATION"}
        findings = []
        for ent in results:
            mapped = TYPE_MAP.get(ent["entity_group"])
            if not mapped:
                continue
            findings.append(EngineFinding(
                entity_type=mapped,
                confidence=ent["score"],
                span_start=ent["start"],
                span_end=ent["end"],
                engine_id=self.adapter_id,
                language=context.get("language", "en"),
            ))
        return findings
```

Register and use:

```python
from pii_anon import PIIOrchestrator
from pii_anon.moe import get_default_registry, ExpertSpec

adapter = HuggingFaceNERAdapter(enabled=True)
adapter.initialize({"model": "dslim/bert-base-NER"})

orch = PIIOrchestrator(token_key="secret")
orch.register_engine(adapter)

# Register MoE expertise
registry = get_default_registry()
registry.register_expert(ExpertSpec(
    expert_id="hf-ner",
    display_name="HuggingFace BERT NER",
    entity_strengths={"PERSON_NAME": 0.88, "ORGANIZATION": 0.82, "LOCATION": 0.80},
))
```

---

## Built-in Engine Adapters (Reference)

| Adapter ID | Package | Key Strengths |
|---|---|---|
| `regex-oss` | (built-in) | EMAIL (0.99), SSN (0.99), CC (0.99), PHONE (0.95) |
| `presidio-compatible` | `presidio-analyzer` | PERSON_NAME (0.82), EMAIL (0.80), PHONE (0.78) |
| `scrubadub-compatible` | `scrubadub` | EMAIL (0.75), PHONE (0.60), SSN (0.55) |
| `gliner-compatible` | `gliner` | PERSON_NAME (0.92), ORG (0.88), LOCATION (0.85) |
| `spacy-ner-compatible` | `spacy` | PERSON_NAME (0.78), ORG (0.72), LOCATION (0.75) |
| `stanza-ner-compatible` | `stanza` | PERSON_NAME (0.75), ORG (0.68), LOCATION (0.72) |

---

## Lifecycle Methods

| Method | When Called | Required |
|---|---|---|
| `__init__(enabled)` | Construction | Yes |
| `initialize(config)` | After registration, before first `detect()` | No |
| `detect(payload, context)` | Every record | Yes |
| `health_check()` | On demand (diagnostics) | No |
| `capabilities()` | On demand (introspection) | No |
| `shutdown()` | Application teardown | No |
| `dependency_available()` | Pre-flight checks | No |
