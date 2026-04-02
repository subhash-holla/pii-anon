# Design Proposal A: Simplicity-Focused Architecture

## 1. Design Philosophy

Minimize new code, files, and dependencies by extending the existing FusionStrategy / CalibrationStore / ExpertRegistry infrastructure rather than replacing it. Every new file must justify its existence; every new dependency must be optional. The four-layer pipeline (regex fast-pass, Dawid-Skene aggregation, meta-learner, post-processing) is implemented as a single `SwarmFusionStrategy.merge()` call so callers see no API change.

## 2. Architecture Overview

### High-Level System Diagram

```
                         ┌──────────────────────────────────────────────────┐
                         │              PIIOrchestrator                     │
                         │  (unchanged -- calls fusion.merge() as before)  │
                         └──────────────┬───────────────────────────────────┘
                                        │ list[EngineFinding]
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        SwarmFusionStrategy.merge()                          │
│                                                                               │
│  ┌─────────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌────────┐│
│  │  Layer 1:       │   │  Layer 2:        │   │  Layer 3:     │   │Layer 4:││
│  │  Regex          │──▶│  Dawid-Skene     │──▶│  Meta-learner │──▶│Post-   ││
│  │  Fast-Pass      │   │  Aggregation     │   │  (XGBoost)    │   │process ││
│  │                 │   │  + Span Reconcile│   │               │   │        ││
│  │ accept high-conf│   │  EM loop (pure   │   │ trained       │   │checksum││
│  │ regex, tag rest │   │  Python) + IoU   │   │ classifier on │   │validate││
│  │ for NER engines │   │  clustering      │   │ DS + engine   │   │corrob. ││
│  │                 │   │                  │   │ features      │   │filter  ││
│  └─────────────────┘   └──────────────────┘   └───────────────┘   └────────┘│
│                                                                               │
│  Calibration: TemperatureScaler (per-engine) applied before Layer 2          │
│  Artifacts:   ~/.pii_anon/swarm/ (xgb model, DS priors, taxonomy map)    │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Component Inventory

| Component | Technology | Purpose | Justification |
|-----------|-----------|---------|---------------|
| SwarmFusionStrategy | Python (FusionStrategy subclass) | Orchestrate the 4-layer pipeline | Reuses existing FusionStrategy ABC; single entry point |
| DawidSkeneAggregator | Pure Python (no deps) | Bayesian label aggregation via EM | Replaces weighted voting; ~120 lines of EM code, no crowd-kit dep |
| SwarmMetaLearner | XGBoost (optional dep) | Trained classifier on span features | Lighter than CRF; works at span level not sequence level |
| SwarmTrainer | Python script | Offline training pipeline | Single script: load data, extract features, train XGBoost, serialize |
| TemperatureScaler | Pure Python | Per-engine confidence calibration | Extends CalibrationStore with T params; no new deps |
| TaxonomyMapper | Python (JSON config) | Map dataset labels to canonical types | Static JSON mapping file; no runtime dependency |

### New Files (5 total, plus 1 config)

| File | Lines (est.) | Responsibility |
|------|-------------|----------------|
| `src/pii_anon/swarm.py` | ~400 | SwarmFusionStrategy, DawidSkeneAggregator, TemperatureScaler |
| `src/pii_anon/swarm_learner.py` | ~250 | SwarmMetaLearner (XGBoost wrapper), feature engineering |
| `src/pii_anon/swarm_train.py` | ~300 | Training pipeline script (CLI entry point) |
| `src/pii_anon/swarm_taxonomy.py` | ~80 | TaxonomyMapper class + canonical type definitions |
| `tests/test_swarm.py` | ~350 | Unit tests for all components |
| `src/pii_anon/swarm_taxonomy.json` | ~100 | Dataset-to-canonical label mapping config |

## 3. Backend Design

### 3.1 Data Models

#### Entity: SpanCandidate (internal, not persisted)

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| entity_type | str | non-empty | Canonical PII type after taxonomy mapping |
| span_start | int | >= 0 | Start character offset |
| span_end | int | > span_start | End character offset (exclusive) |
| field_path | str or None | - | Source field name |
| language | str | default "en" | Language code |
| engine_votes | dict[str, float] | engine_id -> confidence | Raw confidence from each engine that detected this span |
| ds_label | str or None | - | Dawid-Skene estimated true label (set by Layer 2) |
| ds_confidence | float | [0, 1] | Dawid-Skene posterior probability |
| meta_score | float | [0, 1] | Meta-learner output probability |
| checksum_valid | bool or None | - | Checksum/format validation result (None = not applicable) |
| corroboration_count | int | >= 1 | Number of distinct engines that detected this span |

#### Entity: DawidSkeneParams (serialized to JSON)

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| error_rates | dict[str, dict[str, dict[str, float]]] | engine -> true_label -> observed_label -> prob | Learned confusion matrices per engine |
| class_priors | dict[str, float] | label -> prior prob | Estimated prior probability of each entity type |
| n_iterations | int | > 0 | Number of EM iterations used |
| converged | bool | - | Whether EM converged within tolerance |

#### Entity: SwarmConfig (serialized to JSON)

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| schema_version | str | "2.0" | Config schema version |
| regex_fast_pass_threshold | float | [0, 1], default 0.95 | Confidence threshold for regex bypass |
| regex_fast_pass_types | list[str] | - | Entity types eligible for fast-pass (structured PII) |
| ds_max_iterations | int | default 50 | Max EM iterations |
| ds_convergence_tol | float | default 1e-4 | EM convergence tolerance |
| corroboration_required_types | list[str] | - | Semantic types requiring 2+ engines |
| corroboration_min_engines | int | default 2 | Min engines for corroboration filter |
| meta_learner_path | str or None | - | Path to serialized XGBoost model |
| temperature_params | dict[str, float] | engine_id -> T | Per-engine temperature scaling parameters |

#### Extension to CalibrationResult (existing schema)

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| temperature_params | dict[str, float] | engine_id -> T value | Per-engine temperature scaling (added to metadata dict) |
| ds_priors | dict[str, float] | entity_type -> prior | Dawid-Skene class priors (added to metadata dict) |

**Relationships:** SwarmFusionStrategy loads SwarmConfig + CalibrationResult at init. SpanCandidate is ephemeral (created and consumed within a single `merge()` call). DawidSkeneParams is serialized during training and loaded at inference.

### 3.2 API Design

No new HTTP APIs. The system is a library. The public API surface is:

#### Configuration (ProcessingProfileSpec extension)

| Interface | Change | Description |
|-----------|--------|-------------|
| `ProcessingProfileSpec.mode` | Add `"swarm"` literal | New fusion mode selection |
| `build_fusion()` in fusion.py | Add `"swarm"` case | Factory instantiation |

#### Python API

```python
# Usage is identical to existing modes -- no API change for callers
profile = ProcessingProfileSpec(
    profile_id="swarm_default",
    mode="swarm",             # <-- new mode
    objective="accuracy",
)
result = orchestrator.run(payload, profile=profile, ...)
```

#### Training CLI

```bash
# Offline training (produces serialized artifacts)
python -m pii_anon.swarm_train \
    --datasets ai4privacy,conll2003,tab,bigcode,i2b2,pii_anon_eval \
    --output-dir ~/.pii_anon/swarm/ \
    --max-samples 50000
```

### 3.3 Service Architecture

There are no new services. The entire swarm pipeline runs in-process as a FusionStrategy. The dependency graph is:

```
orchestrator.py
    └── fusion.py::build_fusion(mode="swarm")
            └── swarm.py::SwarmFusionStrategy
                    ├── swarm.py::DawidSkeneAggregator  (pure Python)
                    ├── swarm.py::TemperatureScaler      (pure Python)
                    ├── swarm_learner.py::SwarmMetaLearner (optional: xgboost)
                    ├── swarm_taxonomy.py::TaxonomyMapper
                    ├── calibration/store.py::CalibrationStore (existing)
                    └── engines/regex/validators.py           (existing, for Layer 4)
```

**Error Handling Strategy:**
- If XGBoost is not installed, SwarmMetaLearner falls back to a simple logistic function on DS confidence (no ML dep required).
- If no trained model artifact exists at the configured path, the meta-learner layer is skipped (DS confidence used directly).
- If Dawid-Skene fails to converge, the last-iteration estimates are used (always produces output).
- Each layer is independently bypassable via SwarmConfig flags.

### 3.4 Database Schema

No database. All persistence is JSON files:

```
~/.pii_anon/
├── calibration.json              # existing (extended with temperature_params in metadata)
└── swarm/
    ├── config.json               # SwarmConfig
    ├── ds_priors.json            # DawidSkeneParams (from training)
    ├── meta_learner.xgb          # Serialized XGBoost model (binary)
    ├── feature_schema.json       # Feature names/order for meta-learner
    └── taxonomy_map.json         # Dataset label -> canonical type mapping
```

## 4. Middleware and Integration Layer

### 4.1 Authentication and Authorization

Not applicable (library, not a service).

### 4.2 Caching Strategy

| What is Cached | Where | TTL / Invalidation |
|----------------|-------|-------------------|
| MoE routing decisions | MoERouter._route_cache (existing) | Cleared on registry update |
| Dawid-Skene priors | SwarmFusionStrategy._ds_priors (in-memory) | Loaded once at init from JSON |
| XGBoost model | SwarmMetaLearner._model (in-memory) | Loaded once at init from .xgb file |
| Temperature params | TemperatureScaler._temps (in-memory) | Loaded once from CalibrationStore |
| Taxonomy mappings | TaxonomyMapper._map (in-memory) | Loaded once from JSON |
| Regex fast-pass types | SwarmFusionStrategy._fast_pass_types (set) | Loaded once from config |

No TTL-based expiration. All caches are populated at construction time and are immutable for the lifetime of the strategy instance. To pick up new training artifacts, construct a new SwarmFusionStrategy (or call `reload()`).

### 4.3 Error Handling

```
Layer failures → graceful degradation:

Layer 1 (regex fast-pass): Cannot fail. Simple confidence threshold check.
Layer 2 (Dawid-Skene):     If EM diverges → use last iteration estimates + log warning.
Layer 3 (meta-learner):    If model missing/corrupt → skip layer, use DS confidence.
Layer 4 (post-process):    If validator throws → treat as checksum_valid=None (neutral).

All errors are logged via pii_anon.observability.get_logger().
No errors propagate to callers unless ALL layers fail simultaneously (impossible by design).
```

### 4.4 Third-Party Integrations

| Integration | Purpose | Required? | Install |
|-------------|---------|-----------|---------|
| xgboost >= 2.0 | Meta-learner model | Optional | `pip install pii-anon[swarm-ml]` |
| scikit-learn >= 1.3 | Feature preprocessing, metrics | Optional (training only) | `pip install pii-anon[swarm-ml]` |
| numpy >= 1.24 | Array ops for DS and features | Optional | `pip install pii-anon[swarm-ml]` |

Runtime without ML deps: Layers 1, 2, and 4 work. Layer 3 uses logistic fallback.

## 5. Frontend Design

Not applicable -- pii-anon is a backend library. No UI components.

## 6. Technology Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Core library | Python | >= 3.10 | Existing requirement; dataclasses, slots, match statements |
| Fusion strategy | Pure Python | - | No deps for Layers 1, 2, 4 |
| Meta-learner | XGBoost | >= 2.0.0 | Lightweight, fast inference, no GPU needed, serializes to single file |
| Training metrics | scikit-learn | >= 1.3.0 | F1, bootstrap CI, calibration curves |
| Array operations | numpy | >= 1.24.0 | Feature vectors for meta-learner |
| Serialization | JSON + XGBoost binary | - | JSON for config/priors, native XGBoost format for model |
| Testing | pytest | >= 7.0 | Existing test infrastructure |

## 7. Detailed Layer Design

### 7.1 Layer 1: Regex Fast-Pass

**Purpose:** Accept high-confidence regex detections immediately for structured PII types, bypassing NER engines to reduce latency.

**Algorithm:**
```python
def _regex_fast_pass(self, findings: list[EngineFinding]) -> tuple[list[EnsembleFinding], list[EngineFinding]]:
    """Partition findings into fast-pass accepts and candidates for further processing."""
    accepted = []
    remaining = []
    for f in findings:
        if (f.engine_id == "regex-oss"
            and f.entity_type in self._fast_pass_types
            and f.confidence >= self._config.regex_fast_pass_threshold):
            # Promote directly to EnsembleFinding
            accepted.append(EnsembleFinding(
                entity_type=f.entity_type,
                confidence=f.confidence,
                engines=["regex-oss"],
                field_path=f.field_path,
                span_start=f.span_start,
                span_end=f.span_end,
                explanation="regex fast-pass (high confidence structured PII)",
                language=f.language,
            ))
        else:
            remaining.append(f)
    return accepted, remaining
```

**Fast-pass eligible types** (structured, regex-authoritative):
`EMAIL_ADDRESS`, `US_SSN`, `CREDIT_CARD`, `IP_ADDRESS`, `MAC_ADDRESS`, `IBAN`, `PHONE_NUMBER`, `BANK_ACCOUNT`, `ROUTING_NUMBER`, `CRYPTO_WALLET`, `DRIVERS_LICENSE`, `PASSPORT`, `NATIONAL_ID`, `MEDICAL_RECORD_NUMBER`

**Selective engine activation (FR-002):** The fast-pass implicitly implements selective activation. Findings that pass Layer 1 are removed from the candidate pool, so downstream layers process fewer spans. The orchestrator already skips unavailable engines; this extends the principle to skip redundant work per-span.

### 7.2 Layer 2: Dawid-Skene Bayesian Aggregation

**Purpose:** Replace weighted voting with principled Bayesian estimation of true labels, accounting for per-engine error patterns.

**Implementation (pure Python EM loop):**

```python
class DawidSkeneAggregator:
    """Expectation-Maximization for multi-annotator label aggregation.

    At inference time, uses pre-trained error rates (from offline training).
    At training time, learns error rates from labeled + unlabeled data.
    """

    def __init__(self, priors: dict[str, float], error_rates: dict[str, dict[str, dict[str, float]]]):
        self._priors = priors          # P(true_label)
        self._error_rates = error_rates # P(observed | true, engine)

    def aggregate(self, span_candidates: list[SpanCandidate]) -> list[SpanCandidate]:
        """For each span, compute posterior P(true_label | observations)."""
        for span in span_candidates:
            posteriors = {}
            for label in self._priors:
                log_prob = math.log(self._priors[label] + 1e-10)
                for engine_id, observed_conf in span.engine_votes.items():
                    if engine_id in self._error_rates:
                        # P(engine says "label" | true label is "label")
                        rates = self._error_rates[engine_id]
                        if label in rates and span.entity_type in rates[label]:
                            log_prob += math.log(rates[label][span.entity_type] + 1e-10)
                posteriors[label] = log_prob

            # Normalize (log-sum-exp)
            max_lp = max(posteriors.values())
            denom = math.log(sum(math.exp(lp - max_lp) for lp in posteriors.values())) + max_lp

            best_label = max(posteriors, key=posteriors.get)
            span.ds_label = best_label
            span.ds_confidence = math.exp(posteriors[best_label] - denom)

        return span_candidates
```

**Training-time EM loop** (in `swarm_train.py`):

```
Initialize:
  class_priors = uniform over entity types
  error_rates = identity matrices (diagonal = 0.8, off-diagonal = uniform remainder)

Repeat until convergence or max_iterations:
  E-step: For each item, compute P(true_label | observations, error_rates, priors)
  M-step: Update error_rates = count(engine, true, observed) / count(engine, true)
          Update priors = mean(posterior probabilities across items)
  Check: max |priors_new - priors_old| < convergence_tol
```

**Span boundary reconciliation (FR-005):** Integrated into this layer. Before running DS aggregation, overlapping spans are clustered using the existing `_cluster_overlapping_spans()` function from `fusion.py`. Each cluster is merged into one SpanCandidate with combined engine_votes. Boundary selection uses weighted voting (same as existing WeightedConsensusFusion).

### 7.3 Layer 3: Meta-Learner (XGBoost)

**Purpose:** Trained classifier that takes DS output + engine features and predicts whether a span is a true positive.

**Feature Engineering (per SpanCandidate):**

| Feature | Type | Description |
|---------|------|-------------|
| ds_confidence | float | Dawid-Skene posterior probability |
| corroboration_count | int | Number of engines that detected this span |
| max_engine_confidence | float | Highest raw engine confidence |
| min_engine_confidence | float | Lowest raw engine confidence |
| mean_engine_confidence | float | Mean raw engine confidence |
| std_engine_confidence | float | Std dev of engine confidences |
| span_length | int | Character length of detected span |
| entity_type_id | int | Ordinal encoding of canonical entity type |
| has_regex | bool (0/1) | Whether regex engine detected this span |
| has_ner | bool (0/1) | Whether any NER engine (gliner/presidio/spacy/stanza) detected |
| regex_confidence | float | Regex engine confidence (0 if not detected) |
| best_ner_confidence | float | Best NER engine confidence (0 if none) |
| engine_agreement | float | Fraction of engines agreeing on entity type |
| calibrated_temp_{engine} | float | Temperature-scaled confidence per engine (6 features, one per engine) |

**Total: 20 features**

**Model:**
- XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
- Binary classification: 1 = true positive, 0 = false positive
- Output: predict_proba()[:, 1] used as meta_score
- Threshold: 0.5 default, tunable via config

**Fallback (no XGBoost installed):**
```python
def _logistic_fallback(self, ds_confidence: float, corroboration: int) -> float:
    """Simple logistic function when XGBoost is unavailable."""
    x = 2.0 * ds_confidence + 0.5 * min(corroboration, 4) - 2.0
    return 1.0 / (1.0 + math.exp(-x))
```

### 7.4 Layer 4: Post-Processing

**Checksum/format validation (FR-008):**
Reuse existing `engines/regex/validators.py` functions:

| Entity Type | Validator | Source |
|-------------|-----------|--------|
| CREDIT_CARD | `luhn_checksum()` | validators.py |
| US_SSN | `is_valid_ssn()` | validators.py |
| IBAN | `is_valid_iban()` | validators.py |
| ROUTING_NUMBER | `is_valid_aba_routing()` | validators.py |
| IP_ADDRESS | `is_valid_ipv4()` | validators.py |

Validation result adjusts the final confidence:
- `checksum_valid=True`: confidence *= 1.1 (capped at 1.0)
- `checksum_valid=False`: confidence *= 0.3 (strong penalty)
- `checksum_valid=None`: no adjustment

**Corroboration filtering (FR-007):**
For semantic entity types (`PERSON_NAME`, `ORGANIZATION`, `LOCATION`, `USERNAME`), require `corroboration_count >= 2`. Single-engine detections of these types are dropped unless the meta_score exceeds a high threshold (0.85).

## 8. Dawid-Skene Implementation Detail

The Dawid-Skene model treats each engine as an imperfect annotator with a per-class confusion matrix. Given N items (spans) and K annotators (engines), the model estimates:

1. **True label distribution** pi[j] = P(true label of item i = j)
2. **Error rates** theta[k][j][l] = P(engine k says l | true label is j)

**Inference-time simplification:** At inference, we do NOT run EM. We use the pre-trained error_rates and priors from the training phase. For each span, we compute the posterior directly:

```
P(true=j | observations) proportional to pi[j] * product_k theta[k][j][observed_k]
```

This is a single forward pass -- O(spans * engines * labels). No iteration needed at inference time. This is critical for meeting the 200ms latency target.

**Training-time EM:** Full EM runs offline during `swarm_train.py`. Convergence typically occurs in 10-20 iterations. The trained parameters (error_rates, priors) are serialized to `ds_priors.json`.

**Comparison with crowd-kit:** The crowd-kit library implements the same algorithm but adds dependencies (pandas, numpy) and object overhead. Our pure-Python implementation uses only `math.log`/`math.exp` and dicts. For the inference path (no EM), this is ~120 lines of code and zero dependencies.

## 9. Training Pipeline Design

### 9.1 Pipeline Overview

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Load        │   │  Run each    │   │  Extract     │   │  Train       │
│  datasets    │──▶│  engine on   │──▶│  features    │──▶│  models      │
│  + taxonomy  │   │  each sample │   │  per span    │   │              │
│  mapping     │   │              │   │              │   │ - DS EM      │
│              │   │              │   │              │   │ - XGBoost    │
│              │   │              │   │              │   │ - Temp scale │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                 │
                                                        ┌────────▼───────┐
                                                        │  Serialize     │
                                                        │  artifacts to  │
                                                        │  ~/.pii_anon/  │
                                                        │  swarm/     │
                                                        └────────────────┘
```

### 9.2 Dataset Integration (FR-011, FR-012)

| Dataset | Records | Format | Key Entity Types |
|---------|---------|--------|-----------------|
| AI4Privacy 209K | 209,000 | HuggingFace dataset, BIO tags | PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, ADDRESS, DOB |
| CoNLL-2003 | 22,137 | CoNLL BIO format | PER, ORG, LOC, MISC |
| TAB (Text Anonymization Benchmark) | ~3,000 | Standoff annotations | PERSON, ORGANIZATION, LOCATION, DATE, ID |
| BigCode PII | ~10,000 | JSON spans | EMAIL, IP_ADDRESS, KEY, USERNAME, PASSWORD |
| i2b2 2014 | ~1,300 | XML annotations | NAME, DATE, ID, CONTACT, AGE, LOCATION |
| pii-anon-eval-data 117K | 117,000 | Internal JSON format | All canonical types |

**Taxonomy Mapping (FR-012):**

```json
{
  "ai4privacy": {
    "FIRSTNAME": "PERSON_NAME",
    "LASTNAME": "PERSON_NAME",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONENUMBER": "PHONE_NUMBER",
    "SOCIALNUM": "US_SSN",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "STREETADDRESS": "ADDRESS",
    "DATE": "DATE_OF_BIRTH",
    "CITY": "LOCATION",
    "ZIPCODE": "LOCATION"
  },
  "conll2003": {
    "PER": "PERSON_NAME",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "MISC": null
  },
  "tab": {
    "PERSON": "PERSON_NAME",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION": "LOCATION",
    "DATE": "DATE_OF_BIRTH",
    "ID": "NATIONAL_ID"
  },
  "bigcode": {
    "EMAIL": "EMAIL_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "KEY": "CRYPTO_WALLET",
    "USERNAME": "USERNAME",
    "PASSWORD": "USERNAME"
  },
  "i2b2": {
    "NAME": "PERSON_NAME",
    "DATE": "DATE_OF_BIRTH",
    "ID": "MEDICAL_RECORD_NUMBER",
    "CONTACT": "PHONE_NUMBER",
    "AGE": "DATE_OF_BIRTH",
    "LOCATION": "LOCATION"
  },
  "pii_anon_eval": {
    "_identity": true
  }
}
```

Entries mapping to `null` are excluded from training. The `"_identity": true` flag means the dataset already uses canonical types.

### 9.3 Training Steps

1. **Load & normalize:** For each dataset, load records, apply taxonomy mapping, produce list of `(text, list[AnnotatedSpan])`.
2. **Run engines:** For each text, run all 6 engines. Collect `list[EngineFinding]` per engine per text.
3. **Align spans:** Match engine findings to gold annotations using IoU >= 0.5. Produce labeled training examples: `(span_features, is_true_positive)`.
4. **Train DS priors:** Run EM on the engine output labels (ignoring gold initially) to learn error_rates and priors. Then validate against gold labels.
5. **Train temperature scaling:** For each engine, fit a single temperature parameter T such that calibrated_confidence = sigmoid(logit(raw_confidence) / T) minimizes cross-entropy against gold labels.
6. **Extract features:** For each span, compute the 20 features described in Section 7.3.
7. **Train XGBoost:** XGBClassifier on (features, is_true_positive). 80/20 train/val split stratified by entity type.
8. **Evaluate:** Compute per-entity F1 with bootstrap 95% CIs. Verify dominance (swarm >= best engine per type).
9. **Serialize:** Write ds_priors.json, meta_learner.xgb, feature_schema.json, taxonomy_map.json.

### 9.4 Temperature Scaling (FR-009, FR-010)

Per-engine temperature parameter T_k calibrates raw confidence:

```
calibrated_conf = sigmoid(logit(raw_conf) / T_k)

where logit(p) = log(p / (1-p))
      sigmoid(x) = 1 / (1 + exp(-x))
```

- T > 1: softens overconfident engines (makes probabilities closer to 0.5)
- T < 1: sharpens underconfident engines (pushes probabilities toward 0 or 1)
- T = 1: no change

Fitted by minimizing negative log-likelihood on validation data per engine using scipy.optimize.minimize_scalar (or grid search if scipy unavailable).

**Calibration-aware fusion (FR-010):** Temperature-scaled confidences are used as inputs to both Dawid-Skene and the meta-learner. The DS error_rates are estimated on calibrated confidences (from training), so the entire pipeline is calibration-aware.

## 10. Configuration Schema

### ProcessingProfileSpec Extension

```python
# In types.py, the existing FusionMode type alias already allows arbitrary strings:
# FusionMode = Literal["union_high_recall", "weighted_consensus", ...] | str
# So mode="swarm" works without modifying the type.
```

### SwarmConfig (new, JSON)

```json
{
  "schema_version": "2.0",
  "regex_fast_pass": {
    "enabled": true,
    "threshold": 0.95,
    "types": [
      "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "IP_ADDRESS",
      "MAC_ADDRESS", "IBAN", "PHONE_NUMBER", "BANK_ACCOUNT",
      "ROUTING_NUMBER", "CRYPTO_WALLET", "DRIVERS_LICENSE",
      "PASSPORT", "NATIONAL_ID", "MEDICAL_RECORD_NUMBER"
    ]
  },
  "dawid_skene": {
    "max_iterations": 50,
    "convergence_tol": 1e-4,
    "priors_path": "~/.pii_anon/swarm/ds_priors.json"
  },
  "meta_learner": {
    "enabled": true,
    "model_path": "~/.pii_anon/swarm/meta_learner.xgb",
    "feature_schema_path": "~/.pii_anon/swarm/feature_schema.json",
    "threshold": 0.5,
    "fallback_enabled": true
  },
  "post_processing": {
    "checksum_validation": true,
    "corroboration_filter": true,
    "corroboration_types": ["PERSON_NAME", "ORGANIZATION", "LOCATION", "USERNAME"],
    "corroboration_min_engines": 2,
    "corroboration_override_threshold": 0.85
  },
  "temperature_scaling": {
    "enabled": true
  }
}
```

### build_fusion() Extension (in fusion.py)

```python
# Add to build_fusion():
if mode == "swarm":
    from pii_anon.swarm import SwarmFusionStrategy
    return SwarmFusionStrategy(iou_threshold=iou_threshold)
```

## 11. Data Flow Through the Four Layers

```
Input: list[EngineFinding] from orchestrator (all 6 engines)
       │
       ▼
┌─ Layer 1: Regex Fast-Pass ─────────────────────────────────┐
│  For each finding where engine="regex-oss"                  │
│    AND entity_type in fast_pass_types                       │
│    AND confidence >= 0.95:                                  │
│      → Move to accepted[] as EnsembleFinding                │
│  Remaining findings → candidates[]                          │
└─────────────────────────────────┬──────────────────────────┘
                                  │ candidates[]
                                  ▼
┌─ Temperature Scaling ──────────────────────────────────────┐
│  For each candidate:                                        │
│    confidence = sigmoid(logit(confidence) / T[engine_id])   │
└─────────────────────────────────┬──────────────────────────┘
                                  │
                                  ▼
┌─ Layer 2: Dawid-Skene + Span Reconciliation ───────────────┐
│  1. Cluster overlapping spans (reuse _cluster_overlapping)  │
│  2. For each cluster, build SpanCandidate with engine_votes │
│  3. Reconcile boundaries (weighted vote on start/end)       │
│  4. Compute posterior P(true_label | votes) using priors    │
│     and pre-trained error_rates                             │
│  Output: list[SpanCandidate] with ds_label, ds_confidence   │
└─────────────────────────────────┬──────────────────────────┘
                                  │ list[SpanCandidate]
                                  ▼
┌─ Layer 3: Meta-Learner ────────────────────────────────────┐
│  For each SpanCandidate:                                    │
│    features = extract_features(span)    # 20 features       │
│    span.meta_score = xgb_model.predict_proba(features)[1]   │
│  Filter: drop spans where meta_score < threshold            │
│  (If no model: use logistic fallback on ds_confidence)      │
└─────────────────────────────────┬──────────────────────────┘
                                  │ list[SpanCandidate]
                                  ▼
┌─ Layer 4: Post-Processing ─────────────────────────────────┐
│  A. Checksum validation:                                    │
│     For applicable types, run validators.py functions       │
│     Adjust confidence: valid → *1.1, invalid → *0.3        │
│                                                             │
│  B. Corroboration filter:                                   │
│     For semantic types (PERSON_NAME, ORG, LOC, USERNAME):   │
│     Drop if corroboration_count < 2                         │
│       UNLESS meta_score > 0.85 (high-confidence override)   │
│                                                             │
│  C. Convert SpanCandidate → EnsembleFinding                 │
└─────────────────────────────────┬──────────────────────────┘
                                  │
                                  ▼
Output: accepted[] + layer4_output[] → list[EnsembleFinding]
        (sorted by span_start for deterministic output)
```

## 12. Requirement Coverage Matrix

| Req ID | Requirement | How This Design Addresses It | Design Element |
|--------|------------|------------------------------|----------------|
| FR-001 | Regex fast-pass | Layer 1 accepts high-conf regex findings (>= 0.95) for structured types, bypassing NER | `SwarmFusionStrategy._regex_fast_pass()` |
| FR-002 | Selective engine activation | Fast-pass removes regex-resolved spans from candidate pool; remaining spans only processed by engines that declared strength for that type (via existing MoERouter) | Layer 1 partitioning + existing ExpertRegistry |
| FR-004 | Dawid-Skene Bayesian aggregation | Pure-Python EM implementation; inference uses pre-trained error matrices for O(1) posterior computation per span | `DawidSkeneAggregator` in swarm.py |
| FR-005 | Span boundary reconciliation | Reuses existing `_cluster_overlapping_spans()` with IoU clustering; weighted vote on start/end boundaries within each cluster | Layer 2 span reconciliation step |
| FR-006 | Trained meta-learner | XGBoost binary classifier on 20 span features; optional dep with logistic fallback | `SwarmMetaLearner` in swarm_learner.py |
| FR-007 | Corroboration filtering | Layer 4 requires 2+ engines for semantic types (PERSON_NAME, ORG, LOC, USERNAME); high-confidence override at 0.85 | Layer 4 corroboration filter |
| FR-008 | Checksum/format validation | Layer 4 reuses existing validators.py (Luhn, IBAN mod-97, SSN area rules, etc.); adjusts confidence based on validation result | Layer 4 checksum validation |
| FR-009 | Per-engine temperature scaling | TemperatureScaler applies sigmoid(logit(conf)/T) per engine; T fitted offline per engine | `TemperatureScaler` in swarm.py |
| FR-010 | Calibration-aware fusion | Temperature-scaled confidences feed into DS aggregation and meta-learner; entire pipeline operates on calibrated values | Temperature scaling applied before Layer 2 |
| FR-011 | Multi-dataset training | Training pipeline loads 6 datasets via configurable loaders; each produces normalized (text, spans) pairs | `swarm_train.py` dataset loading |
| FR-012 | Taxonomy mapping | JSON config maps each dataset's labels to canonical pii-anon types; TaxonomyMapper class | `swarm_taxonomy.py` + `swarm_taxonomy.json` |
| FR-013 | Offline training with serialized artifacts | Single CLI script produces ds_priors.json, meta_learner.xgb, feature_schema.json | `python -m pii_anon.swarm_train` |
| FR-014 | Per-entity F1 with bootstrap CIs | Training script computes per-type F1, 95% bootstrap CIs (1000 resamples), McNemar significance tests | Training pipeline evaluation step |
| FR-015 | Dominance verification | Training script compares swarm F1 vs best individual engine per type; fails with diagnostic if any type regresses | Training pipeline dominance check |
| FR-016 | mode="swarm" in ProcessingProfileSpec | Add "swarm" case to build_fusion(); FusionMode type already accepts arbitrary strings | fusion.py extension |

**Coverage Summary:**
- High-priority requirements covered: 16/16 (100%)
- Medium-priority requirements covered: N/A (all listed are high)
- Low-priority requirements noted: 0 (none specified)

## 13. Deployment Architecture

### Artifact Deployment

```
Production deployment:
  1. pip install pii-anon[swarm-ml]           # includes xgboost, sklearn, numpy
  2. python -m pii_anon.swarm_train ...    # offline, takes ~30min on 6 datasets
  3. ls ~/.pii_anon/swarm/                 # verify artifacts exist
  4. Use mode="swarm" in profiles          # runtime

Lightweight deployment (no ML deps):
  1. pip install pii-anon                     # core only
  2. Copy pre-built swarm/ artifacts       # from CI or shared storage
  3. Use mode="swarm"                      # meta-learner uses logistic fallback
```

### CI/CD Integration

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  pytest      │───▶│  Train on    │───▶│  Evaluate    │───▶│  Publish     │
│  unit tests  │    │  eval-data   │    │  dominance   │    │  artifacts   │
│              │    │  subset      │    │  check       │    │  to PyPI     │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

Dominance check gates release: if swarm F1 < best engine F1 for ANY entity type, the CI pipeline fails.

## 14. Risks and Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| XGBoost optional dep causes inconsistent behavior across installs | Users without xgboost get worse accuracy from logistic fallback | Document clearly; make swarm-ml extra the recommended install; log warning when falling back |
| Dawid-Skene priors stale after engine updates | Degraded aggregation quality | Include ds_priors.json version check; warn if priors older than 90 days; retrain script is easy to run |
| 200ms latency target with 4 layers | Layers add overhead vs single-pass fusion | Layer 1 reduces work for subsequent layers; DS inference is O(spans*engines*types) ~microseconds; XGBoost predict is ~1ms for 100 spans; total pipeline overhead ~5-10ms |
| Training data quality varies across datasets | Noisy labels degrade meta-learner | Weight datasets by quality (pii-anon-eval-data gets 2x weight); cross-validate per dataset |
| Corroboration filter drops valid single-engine detections | Recall loss for rare entities detected by only one engine | Override threshold (0.85 meta_score) allows high-confidence single-engine detections through; tunable per deployment |
| Temperature scaling assumes logistic calibration curve | May not fit all engines well | Validate calibration curves during training; fall back to T=1.0 for engines where temperature scaling increases loss |

### Key Trade-offs (Simplicity Focus)

1. **Single file for core logic (swarm.py) vs separate modules**: We pack DawidSkeneAggregator, TemperatureScaler, and SwarmFusionStrategy into one file. This reduces import complexity and makes the dependency graph trivial, at the cost of a larger single file (~400 lines). This is acceptable because the three classes are tightly coupled and always used together.

2. **Pure-Python DS vs crowd-kit**: We sacrifice the optimized numpy vectorization of crowd-kit for zero dependencies. At inference time (no EM), the performance difference is negligible (microseconds). At training time, EM on ~300K items may take 2-3 minutes in pure Python vs ~30 seconds with numpy. This is acceptable for an offline batch process.

3. **XGBoost vs CRF**: XGBoost operates on independent spans (no sequence modeling). A CRF could capture label dependencies (e.g., PERSON_NAME often follows "Dear"). We choose XGBoost because: (a) span-level features are sufficient given DS aggregation upstream, (b) XGBoost has simpler training and inference, (c) no sklearn-crfsuite dependency. The DS layer already captures inter-engine dependencies.

4. **JSON config vs YAML/TOML**: JSON is the only format the existing CalibrationStore uses. Adding YAML would require a new dependency. We stay with JSON for consistency.

5. **No microservice / no message queue**: The entire pipeline runs synchronously within `merge()`. This is the simplest possible deployment model. If latency becomes an issue in the future, the layers could be parallelized (Layer 1 is embarrassingly parallel), but we avoid premature optimization.
