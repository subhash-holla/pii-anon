# Configuration Reference

pii-anon is configurable via YAML/JSON files, environment variables, or Python code. All settings have sensible defaults — zero configuration is required to get started.

---

## Loading Configuration

```python
from pii_anon.config import ConfigManager

# From a YAML or JSON file
config = ConfigManager().load("pii-anon.yaml")

# From environment variables (automatic)
# All PII_ANON__* env vars are picked up automatically
config = ConfigManager().load()
```

---

## Environment Variables

Prefix all variables with `PII_ANON__`. Use double underscores for nested keys. Hyphens in adapter IDs become underscores.

```bash
export PII_ANON__DEFAULT_LANGUAGE=fr
export PII_ANON__ENGINES__REGEX_OSS__ENABLED=true
export PII_ANON__ENGINES__PRESIDIO_COMPATIBLE__WEIGHT=1.5
export PII_ANON__MOE__TOP_K=4
export PII_ANON__TRANSFORM__DEFAULT_MODE=anonymize
export PII_ANON__TRACKING__MIN_LINK_SCORE=0.85
```

Values are auto-coerced: `true`/`false` become booleans, numbers become int/float, JSON strings become dicts/lists.

Legacy prefix `PII_CORE__` is also supported for backward compatibility.

---

## Full Configuration Example

```yaml
# ─── General ────────────────────────────────────────────────
default_language: en
auto_discover_engines: true              # Auto-discover entry-point plugins

# ─── Engine Configuration ───────────────────────────────────
# Each engine can be enabled/disabled and weighted independently.
# pii-anon (speed offering) uses only regex-oss.
# pii-anon-swarm uses all enabled engines fused through MoE.
engines:
  regex-oss:
    enabled: true
    weight: 1.0
    timeout_ms: 500
    params:
      context_boost: 0.10
    entity_weights:                      # Per-entity-type weight overrides
      EMAIL_ADDRESS: 2.0
      PERSON_NAME: 0.6
  presidio-compatible:
    enabled: true
    weight: 1.3
    timeout_ms: 2000
    params:
      entities: ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON_NAME"]
  scrubadub-compatible:
    enabled: true
    weight: 0.95
  gliner-compatible:
    enabled: false                       # Requires model download from HuggingFace
    weight: 1.25
  spacy-ner-compatible:
    enabled: false
    weight: 1.0
  stanza-ner-compatible:
    enabled: false
    weight: 0.95

# ─── Mixture-of-Experts (Ensemble Fusion) ───────────────────
moe:
  top_k: 3                              # Experts activated per entity type
  performance_floor: true                # Guarantee: ensemble >= best expert
  min_expert_weight: 0.15               # Floor weight for non-routed experts
  iou_threshold: 0.5                    # Span overlap threshold for clustering

# ─── Confidence Scoring (Regex Engine) ──────────────────────
confidence:
  context_boost: 0.10                    # Confidence increase with context keywords
  context_penalty: 0.15                  # Confidence decrease without context
  context_window: 50                     # Characters (±) to search for context
  confidence_cap: 0.99                   # Maximum confidence after boosting
  confidence_floor: 0.40                 # Minimum confidence after penalty

# ─── Deny-List / Allow-List ─────────────────────────────────
deny_list:
  enabled: true
  lists:
    PERSON_NAME:
      - "new york"
      - "test user"
      - "john doe"
    ORGANIZATION:
      - "inc"
      - "llc"
allow_list:
  enabled: false
  lists:
    EMAIL_ADDRESS:
      - "admin@company.com"

# ─── Transformation ─────────────────────────────────────────
transform:
  default_mode: pseudonymize             # pseudonymize | anonymize | redact | generalize | synthetic | perturb
  placeholder_template: "<{entity_type}:anon_{index}>"
  entity_strategies:                     # Per-entity-type overrides
    EMAIL_ADDRESS: redact
    CREDIT_CARD: pseudonymize
  strategy_params:
    pseudonymize:
      key_rotation_enabled: true

# ─── Entity Tracking (Long-Context Linking) ─────────────────
tracking:
  enabled: true
  min_link_score: 0.8                    # Similarity threshold for linking entities
  allow_email_name_link: true            # Link email prefixes to person names
  require_unique_short_name: true        # Require unique short names per identity

# ─── Streaming / Segmentation ───────────────────────────────
stream:
  enabled: true
  max_chunk_tokens: 2048                 # Segment size
  overlap_tokens: 128                    # Overlap between segments
  max_concurrency: 8                     # Max parallel segment processing

# ─── Fusion Strategy ────────────────────────────────────────
fusion:
  iou_threshold: 0.5                     # IoU for span overlap detection
  min_gap_chars: 5                       # Character gap for performance optimization

# ─── Risk Classification ────────────────────────────────────
risk:
  low_risk_threshold: 0.90               # Confidence >= 0.90 → low risk
  moderate_risk_threshold: 0.75          # Confidence >= 0.75 → moderate risk

# ─── Router / Execution Plan ────────────────────────────────
router:
  ensemble_confidence_threshold: 0.70
  accuracy_confidence_threshold: 0.88
  balanced_confidence_threshold: 0.80
  ensemble_concurrency_cap: 8
  accuracy_concurrency_cap: 4
  balanced_concurrency_cap: 3

# ─── Logging ────────────────────────────────────────────────
logging:
  level: INFO                            # DEBUG | INFO | WARNING | ERROR
  structured: true                       # JSON-structured log output
```

---

## Parameter Reference

### Engine Configuration (`engines.<adapter-id>`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | true | Enable/disable this engine |
| `weight` | float | 1.0 | Global weight in consensus fusion |
| `timeout_ms` | int | 1000 | Per-record execution timeout (ms) |
| `entity_weights` | dict | {} | Per-entity-type weight overrides |
| `params` | dict | {} | Engine-specific parameters |

### MoE Configuration (`moe`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_k` | int | 3 | Experts activated per entity type |
| `performance_floor` | bool | true | Include all experts with floor weight |
| `min_expert_weight` | float | 0.15 | Minimum weight for non-routed experts |
| `iou_threshold` | float | 0.5 | Span overlap threshold for clustering |

### Confidence Configuration (`confidence`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `context_boost` | float | 0.10 | Confidence boost when context keywords found |
| `context_penalty` | float | 0.15 | Confidence penalty when no context |
| `context_window` | int | 50 | Characters (±) to search for context |
| `confidence_cap` | float | 0.99 | Maximum confidence |
| `confidence_floor` | float | 0.40 | Minimum confidence |

### Transform Configuration (`transform`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `default_mode` | str | "pseudonymize" | Default transformation strategy |
| `placeholder_template` | str | `<{entity_type}:anon_{index}>` | Output placeholder format |
| `entity_strategies` | dict | {} | Per-entity-type mode overrides |
| `strategy_params` | dict | {} | Per-strategy parameters |

### Tracking Configuration (`tracking`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | true | Enable entity continuity tracking |
| `min_link_score` | float | 0.8 | Minimum similarity for linking |
| `allow_email_name_link` | bool | true | Link email prefixes to names |

### Stream Configuration (`stream`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | true | Enable streaming segmentation |
| `max_chunk_tokens` | int | 2048 | Segment size |
| `overlap_tokens` | int | 128 | Overlap between segments |
| `max_concurrency` | int | 8 | Max parallel segments |

---

## Processing Profiles

Profiles configure the detection pipeline at runtime:

```python
from pii_anon.types import ProcessingProfileSpec

profile = ProcessingProfileSpec(
    profile_id="high_recall",
    mode="union_high_recall",            # Fusion strategy
    policy_mode="recall_max",            # Confidence trade-off
    objective="ensemble",                # Use MoE ensemble
    language="en",
    transform_mode="anonymize",
    entity_tracking_enabled=True,
)
```

### Fusion Modes

| Mode | Description |
|---|---|
| `union_high_recall` | Emit all findings from all engines (maximum recall) |
| `weighted_consensus` | Weighted merge — higher-weighted engines dominate |
| `calibrated_majority` | Require N engines to agree |
| `intersection_consensus` | Strict multi-engine agreement only |
| `mixture_of_experts` | MoE routing with per-entity-type expert selection |

### Policy Modes

| Mode | Description |
|---|---|
| `recall_max` | Maximize detection at the cost of some false positives |
| `balanced` | Balance precision and recall (default) |
| `precision_guarded` | Accept missed detections to minimize false positives |

### Transform Modes

| Mode | Description |
|---|---|
| `pseudonymize` | Reversible, deterministic tokens |
| `anonymize` | Non-reversible placeholders |
| `redact` | Character masking (e.g., `****`) |
| `generalize` | Reduce precision (e.g., full date → year only) |
| `synthetic` | Replace with realistic fake values |
| `perturb` | Add calibrated noise (differential privacy) |
