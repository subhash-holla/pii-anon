# Configuration

`ConfigManager` supports JSON/YAML config files and environment overrides.

- Preferred prefix: `PII_ANON__`
- Backward-compatible prefix: `PII_CORE__`

## Config example

```yaml
default_language: en
auto_discover_engines: true
engines:
  regex-oss:
    enabled: true
    weight: 1.0
  presidio-compatible:
    enabled: true
    weight: 1.2
    params:
      entities: ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON_NAME"]
  llm-guard-compatible:
    enabled: false
    weight: 1.1
  scrubadub-compatible:
    enabled: false
  spacy-ner-compatible:
    enabled: false
    weight: 0.95
  stanza-ner-compatible:
    enabled: false
    weight: 0.95
  gliner-compatible:
    enabled: false
    weight: 1.0
stream:
  enabled: true
  max_chunk_tokens: 2048
  overlap_tokens: 128
  max_concurrency: 8
benchmark:
  regex_p50_ms: 100
  multi_engine_p50_ms: 500
  throughput_docs_per_hour: 10000
  linear_scaling_r2: 0.95
transform:
  default_mode: pseudonymize
  placeholder_template: "<{entity_type}:anon_{index}>"
tracking:
  enabled: true
  min_link_score: 0.8
  allow_email_name_link: true
  require_unique_short_name: true
competitor_policy:
  enabled: true
  runtime_leverage_enabled: true
  allowed_adapters: ["spacy-ner-compatible", "stanza-ner-compatible"]
  benchmark_adapters: ["presidio", "scrubadub", "gliner"]
logging:
  level: INFO
  structured: true
```

## Environment overrides

```bash
export PII_ANON__DEFAULT_LANGUAGE=fr
export PII_ANON__ENGINES__REGEX_OSS__ENABLED=true
export PII_ANON__STREAM__MAX_CONCURRENCY=16
export PII_ANON__TRACKING__MIN_LINK_SCORE=0.85
export PII_ANON__TRANSFORM__DEFAULT_MODE=anonymize
export PII_ANON__COMPETITOR_POLICY__RUNTIME_LEVERAGE_ENABLED=true
```

## Loading

```python
from pii_anon.config import ConfigManager

config = ConfigManager().load("pii-anon.yaml")
```
