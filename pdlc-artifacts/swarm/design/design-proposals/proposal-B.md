# Design Proposal B: Research-Aligned Scalable Architecture

## 1. Design Philosophy

This proposal treats PII detection as a **crowdsourcing problem** where engines are noisy annotators with learnable error patterns. Rather than hand-tuned heuristics for fusion, we apply proven statistical methods from the computational social science and NLP communities: Dawid-Skene for annotator disagreement, CRF for sequence-level consistency, and Platt scaling for confidence calibration. Every design decision favors statistical rigor and research alignment over implementation simplicity, accepting additional dependencies (crowd-kit, sklearn-crfsuite, scikit-learn) in exchange for principled, auditable aggregation that can be retrained as engines evolve.

## 2. Architecture Overview

### High-Level System Diagram

```
                         ┌──────────────────────────────────┐
                         │       PIIOrchestrator.run()       │
                         │   mode="swarm" dispatches to   │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │     SwarmPipeline (new)         │
                         │  Coordinates all 4 layers         │
                         └──────────────┬───────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
    ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌─────────▼─────────┐
    │   LAYER 1            │  │   LAYER 2            │  │   LAYER 3          │
    │   RegexFastPass      │  │   SelectiveActivation│  │   LearnedAggregation│
    │                      │  │                      │  │                    │
    │ - Confidence gate    │  │ - Coverage gap        │  │ - BIO tokenizer   │
    │ - Checksum validate  │  │   analysis           │  │ - Dawid-Skene EM  │
    │ - Accept/route       │  │ - Engine selector    │  │ - HMM correction  │
    │                      │  │ - Async parallel run │  │ - CRF meta-learner│
    └──────────┬──────────┘  └──────────┬──────────┘  │ - Corroboration   │
               │                        │              │   filter           │
               │ accepted               │ findings     └─────────┬─────────┘
               │ findings               │                        │
               │                        │              aggregated│
               │                        │              findings  │
               │     ┌──────────────────┼────────────────────────┘
               │     │                  │
               ▼     ▼                  ▼
    ┌──────────────────────────────────────────────┐
    │   LAYER 4: ValidationPostProcessor            │
    │                                               │
    │ - Checksum validators (Luhn, IBAN, ABA, VIN) │
    │ - Format validators                           │
    │ - Confidence adjustment / suppression          │
    │ - Span boundary reconciliation                 │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  list[EnsembleFinding]  │
              │  + FusionAuditRecords   │
              └────────────────────────┘


    ┌──────────────────────────────────────────────────────────────────┐
    │                    OFFLINE TRAINING SUBSYSTEM                     │
    │                                                                  │
    │  ┌─────────────┐  ┌───────────────┐  ┌────────────────────────┐│
    │  │ DatasetReg   │  │ TaxonomyMapper│  │ TrainingOrchestrator   ││
    │  │              │──▶│              │──▶│                        ││
    │  │ - fetch      │  │ - canonical   │  │ - feature extraction   ││
    │  │ - cache      │  │   20 types    │  │ - DS param estimation  ││
    │  │ - validate   │  │ - per-dataset │  │ - CRF training         ││
    │  │              │  │   mappings    │  │ - Platt calibration    ││
    │  └─────────────┘  └───────────────┘  │ - serialize artifacts  ││
    │                                       └────────────────────────┘│
    │                                                                  │
    │  Artifacts: models/swarm/                                     │
    │    ds_confusion_matrices.json                                    │
    │    crf_model.joblib                                              │
    │    platt_params.json                                             │
    │    taxonomy_map.json                                             │
    │    training_manifest.json                                        │
    └──────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │                    CALIBRATION SUBSYSTEM                         │
    │                                                                  │
    │  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
    │  │ PlattCalibrator   │  │ CalibrationStore │  │ Informativeness│ │
    │  │                   │  │ (extended)       │  │ Scorer         │ │
    │  │ - per-engine T    │  │ - platt params   │  │                │ │
    │  │ - sigmoid mapping │  │ - ECE metrics    │  │ - variance     │ │
    │  │ - ECE evaluation  │  │ - per-entity     │  │ - entropy      │ │
    │  └──────────────────┘  └─────────────────┘  │ - fusion weight│ │
    │                                              └────────────────┘ │
    └──────────────────────────────────────────────────────────────────┘
```

### Component Inventory

| Component | Technology | Purpose | Justification |
|-----------|-----------|---------|---------------|
| SwarmPipeline | Python class | Four-layer pipeline coordinator | Central orchestration for the new mode |
| RegexFastPass | Python class | Layer 1: confidence-gated regex bypass | Eliminates unnecessary NER calls for high-confidence structured PII |
| SelectiveActivator | Python class | Layer 2: coverage-gap engine selection | Avoids redundant engine execution based on MoE registry strengths |
| BIOTokenizer | Python class | Span-to-BIO conversion | Dawid-Skene operates at token level; need span-to-BIO and BIO-to-span conversion |
| DawidSkeneAggregator | crowd-kit 1.4+ | Layer 3: Bayesian label aggregation | Battle-tested DS implementation; learns per-engine confusion matrices |
| HMMCorrector | Python class (hmmlearn 0.3+) | BIO transition enforcement | Prevents invalid transitions (I-PER after B-LOC) that DS can produce |
| CRFMetaLearner | sklearn-crfsuite 0.3.6+ | Sequence-aware meta-learner | CRF naturally enforces valid entity sequences and captures context features |
| CorroborationFilter | Python class | Multi-engine agreement check for semantic types | Reduces false positives on ambiguous entities |
| ValidationPostProcessor | Python class | Layer 4: checksum/format validation | Catches remaining FPs via structural validation |
| PlattCalibrator | scikit-learn 1.4+ | Per-engine temperature scaling | Platt scaling (logistic regression on logits) maps raw scores to calibrated probabilities |
| InformativenessScorer | Python class | Calibration-aware fusion weighting | Downweights engines with fixed/uninformative confidence distributions |
| DatasetRegistry | Python class | Multi-dataset fetch, cache, iterate | Unified interface for 6+ heterogeneous datasets |
| TaxonomyMapper | Python class + JSON config | Cross-dataset entity type normalization | Maps 54+ source types to 20 canonical types |
| TrainingOrchestrator | Python class | Offline training pipeline | Coordinates data loading, feature extraction, model training, serialization |
| SwarmFusionStrategy | FusionStrategy subclass | Integration with build_fusion() | Backward-compatible entry point via mode="swarm" |
| SwarmConfig | dataclass | Configuration for swarm parameters | Typed, validatable configuration separate from ProcessingProfileSpec |
| BenchmarkEvaluator | Python class | Per-entity F1 + bootstrap CIs + significance tests | Statistical evaluation meeting research standards |
| DominanceVerifier | Python class | Swarm >= best engine per entity type | Automated dominance guarantee checking |

## 3. Backend Design

### 3.1 Data Models

#### Entity: BIOTag
Represents a single token's annotation from one engine in BIO format.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| token_idx | int | >= 0 | Token position in text |
| tag | str | BIO format: "O", "B-{TYPE}", "I-{TYPE}" | BIO label |
| engine_id | str | not empty | Source engine |
| confidence | float | [0.0, 1.0] | Raw or calibrated confidence |

#### Entity: TokenizedText
Text converted to token-level representation for BIO processing.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| text | str | not empty | Original text |
| tokens | list[str] | non-empty | Whitespace-tokenized tokens |
| offsets | list[tuple[int, int]] | len == len(tokens) | Character offset (start, end) per token |
| field_path | str or None | - | Source field name |

#### Entity: EngineAnnotation
Per-engine token-level annotations for a single text.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| engine_id | str | not empty | Engine identifier |
| tags | list[str] | len == num_tokens | BIO tag sequence |
| confidences | list[float] | len == num_tokens | Per-token confidence |

#### Entity: AggregatedToken
Output of Dawid-Skene aggregation for a single token.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| token_idx | int | >= 0 | Token position |
| label | str | BIO tag | Consensus label |
| posterior | float | [0.0, 1.0] | Posterior probability of label |
| engine_votes | dict[str, str] | - | Per-engine label at this position |

#### Entity: SwarmConfig
Configuration for swarm pipeline behavior.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| fast_pass_threshold | float | [0.5, 1.0], default 0.94 | Confidence above which regex bypasses NER |
| corroboration_min | int | >= 1, default 2 | Minimum engines for semantic types |
| corroboration_exempt_types | frozenset[str] | default {"EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "IBAN", "IP_ADDRESS", "MAC_ADDRESS"} | Types exempt from corroboration |
| meta_learner_path | Path or None | default None (use bundled) | Path to serialized CRF model |
| calibration_path | Path or None | default None (use bundled) | Path to calibration parameters |
| ds_max_iterations | int | >= 1, default 20 | Dawid-Skene EM iteration limit |
| hmm_correction | bool | default True | Enable HMM BIO correction |
| coverage_similarity_threshold | float | [0.0, 1.0], default 0.85 | Jaccard threshold for engine redundancy |
| validation_suppress_threshold | float | [0.0, 1.0], default 0.50 | Confidence floor below which findings are suppressed |
| max_engines | int | >= 1, default 4 | Maximum engines to activate in Layer 2 |

**Relationships:** SwarmConfig is stored as an optional field on ProcessingProfileSpec and loaded from a dedicated JSON file or inline configuration.

#### Entity: TrainingManifest
Metadata about a trained swarm model.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| schema_version | str | "2.0" | Artifact format version |
| trained_at | str | ISO-8601 | Training timestamp |
| datasets_used | list[str] | non-empty | Dataset identifiers used |
| total_samples | int | > 0 | Total training records |
| entity_types | list[str] | non-empty | Canonical types covered |
| crf_features | list[str] | non-empty | Feature names used by CRF |
| ds_engines | list[str] | non-empty | Engines in confusion matrices |
| metrics | dict[str, float] | - | Held-out evaluation metrics |
| python_version | str | - | Python version used for training |
| package_versions | dict[str, str] | - | crowd-kit, sklearn-crfsuite versions |

#### Entity: PlattParams
Per-engine calibration parameters.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| engine_id | str | not empty | Engine identifier |
| a | float | - | Platt sigmoid parameter A |
| b | float | - | Platt sigmoid parameter B |
| ece_before | float | [0.0, 1.0] | ECE before calibration |
| ece_after | float | [0.0, 1.0] | ECE after calibration |
| n_samples | int | > 0 | Validation samples used |
| informativeness | float | [0.0, 1.0] | Confidence distribution entropy ratio |

### 3.2 API Design

The swarm pipeline integrates through the existing PIIOrchestrator.run() API. No new public API endpoints are added; the entry point is `ProcessingProfileSpec(mode="swarm")`.

#### Internal Pipeline API (SwarmPipeline)

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config: SwarmConfig, registry: ExpertRegistry, engine_registry: EngineRegistry)` | Initialize with config and registries |
| `run` | `(findings_by_engine: dict[str, list[EngineFinding]], text: str, field_path: str) -> list[EnsembleFinding]` | Execute four-layer pipeline |
| `layer1_fast_pass` | `(regex_findings: list[EngineFinding]) -> tuple[list[EnsembleFinding], list[EngineFinding]]` | Returns (accepted, routed_to_layer2) |
| `layer2_activate` | `(routed: list[EngineFinding], text: str) -> dict[str, list[EngineFinding]]` | Determine and run needed engines |
| `layer3_aggregate` | `(all_findings: dict[str, list[EngineFinding]], text: str) -> list[EnsembleFinding]` | DS + HMM + CRF + corroboration |
| `layer4_validate` | `(findings: list[EnsembleFinding], text: str) -> list[EnsembleFinding]` | Checksum/format validation |

#### Training Pipeline API

| Method | Signature | Description |
|--------|-----------|-------------|
| `TrainingOrchestrator.run` | `(datasets: list[str], output_dir: Path, seed: int) -> TrainingManifest` | Full training pipeline |
| `DatasetRegistry.load` | `(dataset_id: str, split: str, max_samples: int) -> Iterator[TrainingRecord]` | Load and iterate dataset |
| `TaxonomyMapper.map` | `(source_type: str, dataset_id: str) -> str` | Map to canonical type |
| `BenchmarkEvaluator.evaluate` | `(predictions: list, ground_truth: list) -> EvalReport` | Per-entity F1 + CIs |
| `DominanceVerifier.verify` | `(swarm_metrics: dict, engine_metrics: dict) -> DominanceReport` | Check dominance property |

#### CLI Commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `pii-anon train-swarm` | `--datasets <list> --output <dir> --seed <int>` | Train meta-learner + calibration |
| `pii-anon calibrate-swarm` | `--dataset <name> --output <path>` | Run Platt calibration only |
| `pii-anon verify-dominance` | `--dataset <name> --tolerance <float>` | Check dominance guarantee |

### 3.3 Service Architecture

The architecture is a **monolithic in-process pipeline** -- no microservices, no message queues. This is deliberate: PII detection is latency-sensitive (200ms budget) and the pipeline is CPU-bound, not I/O-bound. All components live in the same Python process and communicate via function calls and shared data structures.

**Dependency Graph:**

```
SwarmPipeline
├── RegexFastPass
│   └── (uses regex_findings already computed by orchestrator)
├── SelectiveActivator
│   ├── ExpertRegistry (read strengths)
│   └── EngineRegistry (invoke engines)
├── LearnedAggregator
│   ├── BIOTokenizer
│   ├── PlattCalibrator
│   │   └── PlattParams (loaded from disk)
│   ├── InformativenessScorer
│   ├── DawidSkeneAggregator
│   │   └── crowd-kit DawidSkene
│   ├── HMMCorrector
│   │   └── Transition matrix (loaded from disk)
│   ├── CRFMetaLearner
│   │   └── sklearn-crfsuite CRF (loaded from disk)
│   └── CorroborationFilter
└── ValidationPostProcessor
    └── validators (Luhn, IBAN, ABA, VIN -- existing)
```

**Error Handling Strategy:**

1. **Graceful degradation**: If crowd-kit or sklearn-crfsuite is not installed, fall back to weighted consensus fusion with a warning log. This preserves NFR-011 (optional ML deps).
2. **Engine failures**: If an engine times out or crashes in Layer 2, proceed with available engines. Log the failure. If fewer than `corroboration_min` engines succeed for a semantic type, relax corroboration for that specific invocation.
3. **Model loading failures**: If serialized artifacts are missing or corrupt, fall back to the existing MoE fusion (mode="mixture_of_experts") with a warning. Never crash the pipeline due to missing models.
4. **Dawid-Skene non-convergence**: Cap EM iterations at `ds_max_iterations`. If the algorithm has not converged, use the current parameter estimates and log a warning.

### 3.4 Database Schema

No persistent database is used. All state is file-based:

**Calibration Store (extended)** -- `~/.pii_anon/calibration_v2.json`:

```json
{
  "schema_version": "2.0",
  "calibrated_at": "2026-03-27T12:00:00Z",
  "platt_params": {
    "regex-oss": {"a": -1.2, "b": 0.3, "ece_before": 0.22, "ece_after": 0.08, "informativeness": 0.85},
    "gliner-compatible": {"a": -0.8, "b": 0.1, "ece_before": 0.18, "ece_after": 0.06, "informativeness": 0.92}
  },
  "ds_confusion_matrices": {
    "regex-oss": {"PERSON_NAME": {"PERSON_NAME": 0.3, "O": 0.7}, "EMAIL_ADDRESS": {"EMAIL_ADDRESS": 0.99, "O": 0.01}},
    "gliner-compatible": {"PERSON_NAME": {"PERSON_NAME": 0.92, "O": 0.08}}
  }
}
```

**Training Artifacts** -- `models/swarm/`:

```
models/swarm/
├── training_manifest.json      # TrainingManifest
├── crf_model.joblib            # Serialized CRF (joblib -- safer than raw serialization)
├── ds_confusion_matrices.json  # Pre-computed DS error rates
├── hmm_transitions.json        # BIO transition probabilities
├── platt_params.json           # Per-engine Platt parameters
├── taxonomy_map.json           # Entity type mappings
└── feature_config.json         # CRF feature template spec
```

**Note on CRF serialization:** The CRF model is serialized via `joblib` (scikit-learn's recommended serializer). The training manifest records exact package versions so that load-time version checks can warn about potential incompatibilities. JSON is used for all other artifacts to ensure maximum portability and inspectability.

## 4. Middleware and Integration Layer

### 4.1 Authentication and Authorization

No changes. PII detection is a library, not a service. Authentication is the caller's responsibility.

### 4.2 Caching Strategy

| Cache | What | TTL | Invalidation |
|-------|------|-----|-------------|
| ExpertRegistry strengths | Engine capability scores | Session lifetime | `reset_default_registry()` |
| MoERouter route cache | Per-entity-type routing decisions | Session lifetime | `clear_cache()` on registry change |
| Platt parameters | Loaded sigmoid params | Process lifetime | Re-load on `calibrate-swarm` |
| CRF model | Loaded CRF weights | Process lifetime | Re-load on `train-swarm` |
| DS confusion matrices | Pre-trained engine error rates | Process lifetime | Re-load on `train-swarm` |
| BIO tokenizer offsets | Token boundaries for current text | Per-invocation | Recomputed per text |
| Dataset cache | Downloaded dataset files | Persistent on disk | Manual `--force-download` |

### 4.3 Error Handling

**Error Hierarchy:**

```
SwarmError (new base)
├── SwarmModelNotFoundError      # Artifacts missing
├── SwarmDependencyError         # crowd-kit / crfsuite not installed
├── SwarmCalibrationError        # Calibration params invalid
├── SwarmTrainingError           # Training pipeline failure
├── SwarmConvergenceWarning      # DS did not converge (warning, not fatal)
└── SwarmDominanceViolation      # Dominance check failed (eval-time only)
```

**Global Strategy:**
- All errors inherit from existing `PiiAnonError` base
- Pipeline errors trigger graceful degradation, not crashes
- Training errors are fatal (fail-fast) since offline
- All errors include structured metadata (engine_id, entity_type, layer)

### 4.4 Third-Party Integrations

| Library | Version | Purpose | Optional? |
|---------|---------|---------|-----------|
| crowd-kit | >= 1.4.0 | Dawid-Skene implementation | Yes (swarm-ml extra) |
| sklearn-crfsuite | >= 0.3.6 | CRF meta-learner | Yes (swarm-ml extra) |
| scikit-learn | >= 1.4.0 | Platt scaling (LogisticRegression), bootstrap CI | Yes (swarm-ml extra) |
| hmmlearn | >= 0.3.0 | HMM BIO transition correction | Yes (swarm-ml extra) |
| numpy | >= 1.24.0 | Array operations for DS / calibration | Yes (swarm-ml extra) |
| datasets (HuggingFace) | >= 2.16.0 | Dataset fetching for training | Yes (swarm-train extra) |

**pip extras:**
```
[project.optional-dependencies]
swarm-ml = ["crowd-kit>=1.4.0", "sklearn-crfsuite>=0.3.6", "scikit-learn>=1.4.0", "hmmlearn>=0.3.0", "numpy>=1.24.0"]
swarm-train = ["pii-anon[swarm-ml]", "datasets>=2.16.0"]
```

## 5. Frontend Design

Not applicable. pii-anon is a Python library, not a web application. The "frontend" is the Python API and CLI.

### 5.1 CLI Interface Design

```
pii-anon train-swarm
├── --datasets ai4privacy,conll2003,pii-anon-eval   (required)
├── --output models/swarm/                         (default)
├── --seed 42                                         (default)
├── --max-samples 500000                              (optional)
├── --validation-split 0.15                           (default)
└── --force-download                                  (re-fetch datasets)

pii-anon calibrate-swarm
├── --dataset pii-anon-eval                           (required)
├── --output ~/.pii_anon/calibration_v2.json          (default)
└── --max-samples 10000                               (optional)

pii-anon verify-dominance
├── --dataset pii-anon-eval                           (required)
├── --tolerance 0.02                                  (default)
└── --report-path dominance_report.json               (optional)
```

### 5.2 Python API Surface

```python
from pii_anon import PIIOrchestrator, ProcessingProfileSpec

# Existing API -- unchanged
orchestrator = PIIOrchestrator(token_key="secret")

# New: activate swarm via mode string
profile = ProcessingProfileSpec(
    profile_id="swarm",
    mode="swarm",
)

# Optional: fine-tune swarm behavior
from pii_anon.swarm import SwarmConfig

profile = ProcessingProfileSpec(
    profile_id="swarm_custom",
    mode="swarm",
    # SwarmConfig passed via strategy_params
    strategy_params={
        "swarm": {
            "fast_pass_threshold": 0.90,
            "corroboration_min": 3,
            "meta_learner_path": "models/swarm/crf_model.joblib",
        }
    },
)

result = orchestrator.run(payload, profile=profile, ...)
# Returns same format as always: list[EnsembleFinding]
```

## 6. Technology Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Core Pipeline | Python | 3.10+ | Existing project requirement |
| Bayesian Aggregation | crowd-kit (Toloka) | >= 1.4.0 | Production-grade DS/MACE; used in Toloka platform at scale |
| Meta-Learner | sklearn-crfsuite | >= 0.3.6 | CRF is the standard for sequence labeling; enforces BIO validity |
| Calibration | scikit-learn (LogisticRegression) | >= 1.4.0 | Platt scaling is 2-parameter logistic regression; sklearn is ubiquitous |
| BIO Correction | hmmlearn | >= 0.3.0 | Lightweight HMM library; enforces valid BIO transitions |
| Training Data | HuggingFace datasets | >= 2.16.0 | Standard dataset loading; handles caching, streaming, splits |
| Serialization | JSON + joblib (CRF only) | stdlib + joblib | JSON for config/params; joblib for CRF model (sklearn convention, safer than raw alternatives) |
| Evaluation | scipy.stats (bootstrap) | >= 1.10.0 | Bootstrap CI and paired tests; already a transitive dep of scikit-learn |

## 7. Detailed Layer Design

### 7.1 Layer 1: Regex Fast-Pass (RegexFastPass)

**New file:** `src/pii_anon/swarm/fast_pass.py`

```
class RegexFastPass:
    """Confidence-gated bypass for high-confidence regex findings."""

    def __init__(self, threshold: float = 0.94,
                 checksum_types: frozenset[str] = CHECKSUM_VALIDATED_TYPES):
        self.threshold = threshold
        self.checksum_types = checksum_types

    def process(self, regex_findings: list[EngineFinding])
        -> tuple[list[EnsembleFinding], list[EngineFinding]]:
        """Split regex findings into accepted (bypass NER) and routed (need NER).

        Acceptance criteria:
        1. confidence >= threshold, OR
        2. entity_type in checksum_types AND confidence >= 0.90
           (checksum-validated types get a lower threshold)

        Returns (accepted_as_ensemble, needs_further_processing)
        """
```

**Logic:**
- Iterate regex findings
- If confidence >= threshold OR (entity is checksum-validated type AND confidence >= 0.90): wrap as EnsembleFinding with `engines=["regex-oss"]`, `explanation="fast-pass: conf={conf:.2f}"`
- Otherwise: return as-is for Layer 2 routing
- Latency target: < 0.1ms (pure Python iteration, no model inference)

### 7.2 Layer 2: Selective Engine Activation (SelectiveActivator)

**New file:** `src/pii_anon/swarm/activation.py`

```
class SelectiveActivator:
    """Activate only engines with non-redundant coverage for uncovered types."""

    def __init__(self, expert_registry: ExpertRegistry,
                 similarity_threshold: float = 0.85,
                 max_engines: int = 4):
        self.expert_registry = expert_registry
        self.similarity_threshold = similarity_threshold
        self.max_engines = max_engines

    def select_engines(self, uncovered_types: set[str])
        -> list[str]:
        """Return engine IDs to activate for the given uncovered entity types.

        Algorithm:
        1. For each available engine, compute coverage = entity_strengths
           keys intersected with uncovered_types
        2. Greedily select the engine that covers the most uncovered types
        3. Remove covered types from uncovered set
        4. Repeat until all types covered or max_engines reached
        5. Prune: if two selected engines have Jaccard similarity > threshold
           on their coverage sets, drop the weaker one
        """
```

**Integration with orchestrator:** The `SwarmPipeline.layer2_activate()` method:
1. Computes `uncovered_types` = entity types where regex found low-confidence matches OR no match at all
2. Calls `SelectiveActivator.select_engines(uncovered_types)` to get engine list
3. Invokes only those engines via the existing `AsyncPIIOrchestrator` async engine execution
4. Returns `dict[engine_id, list[EngineFinding]]`

### 7.3 Layer 3: Learned Aggregation (LearnedAggregator)

**New files:**
- `src/pii_anon/swarm/bio.py` -- BIO tokenization and conversion
- `src/pii_anon/swarm/aggregation.py` -- Dawid-Skene + HMM + CRF
- `src/pii_anon/swarm/corroboration.py` -- Corroboration filter

#### 7.3.1 BIO Tokenization (BIOTokenizer)

The critical insight: Dawid-Skene operates on **discrete labeling tasks** (one label per item). For span-level NER, each **token** is an item, and the label is its BIO tag.

```
class BIOTokenizer:
    """Convert between span-level EngineFinding and token-level BIO tags."""

    def tokenize(self, text: str) -> TokenizedText:
        """Whitespace tokenization with character offsets.

        Uses simple whitespace splitting (not subword tokenization) because:
        1. All engines produce character-level spans, not subword spans
        2. Whitespace tokens align with NER output from all 6 engines
        3. Fast: no model loading, no dependency
        """

    def spans_to_bio(self, findings: list[EngineFinding],
                     tokenized: TokenizedText) -> list[str]:
        """Convert character-level spans to BIO tag sequence.

        For each token, find the finding whose span contains the token's
        midpoint. Assign B-{TYPE} to the first token of a span, I-{TYPE}
        to subsequent tokens, O to unmatched tokens.
        """

    def bio_to_spans(self, tags: list[str], confidences: list[float],
                     tokenized: TokenizedText) -> list[EnsembleFinding]:
        """Convert BIO tag sequence back to character-level spans.

        Merge consecutive B-X / I-X tokens into a single span.
        Confidence = mean of token-level confidences within the span.
        """
```

#### 7.3.2 Dawid-Skene Aggregation (DawidSkeneAggregator)

```
class DawidSkeneAggregator:
    """Token-level Bayesian aggregation using Dawid-Skene EM.

    Uses crowd-kit's DawidSkene implementation which:
    1. Initializes confusion matrices from majority vote
    2. E-step: compute posterior label distributions given current matrices
    3. M-step: re-estimate confusion matrices from posteriors
    4. Repeat until convergence or max_iterations
    """

    def __init__(self, max_iterations: int = 20,
                 precomputed_matrices: dict | None = None):
        self.max_iterations = max_iterations
        self.precomputed_matrices = precomputed_matrices

    def aggregate(self, engine_annotations: dict[str, EngineAnnotation],
                  tokenized: TokenizedText)
        -> list[AggregatedToken]:
        """Run Dawid-Skene on token-level annotations.

        Input format for crowd-kit:
        - task = token index
        - worker = engine_id
        - label = BIO tag at that position

        If precomputed_matrices are provided (from training), use them as
        initialization for faster convergence (warm-start).

        Returns per-token consensus labels with posterior probabilities.
        """
```

**crowd-kit integration detail:**

```python
import pandas as pd
from crowdkit.aggregation import DawidSkene

# Build annotation DataFrame
rows = []
for engine_id, annotation in engine_annotations.items():
    for token_idx, tag in enumerate(annotation.tags):
        rows.append({
            "task": token_idx,
            "worker": engine_id,
            "label": tag,
        })
df = pd.DataFrame(rows)

# Run Dawid-Skene
ds = DawidSkene(n_iter=self.max_iterations)
result = ds.fit_predict(df)
# result: pd.Series mapping task -> consensus label
```

#### 7.3.3 HMM BIO Correction (HMMCorrector)

Dawid-Skene treats each token independently -- it has no knowledge of BIO sequencing constraints. This means it can produce invalid sequences like `O B-PER I-LOC O` (I-LOC cannot follow B-PER).

```
class HMMCorrector:
    """Enforce valid BIO transitions via Viterbi decoding.

    Transition matrix learned from training data:
    - P(B-X | O) = learned from data
    - P(I-X | B-X) = learned
    - P(I-X | I-X) = learned
    - P(I-X | B-Y) = 0 (invalid: entity type mismatch)
    - P(I-X | O) = 0 (invalid: I without B)
    - P(B-X | I-Y) = learned (entity boundary)

    Emission matrix: DS posterior probabilities serve as emissions.
    """

    def __init__(self, transition_matrix: dict | None = None):
        self.transition_matrix = transition_matrix

    def correct(self, ds_output: list[AggregatedToken],
                tag_set: list[str]) -> list[AggregatedToken]:
        """Viterbi decoding to find most likely valid BIO sequence.

        If transition_matrix is None (no training), use hard constraints:
        - Block all invalid transitions (I after wrong B/I type)
        - Allow all valid transitions equally
        """
```

#### 7.3.4 CRF Meta-Learner (CRFMetaLearner)

The CRF is a **second-stage** model that takes the DS+HMM output plus rich features and produces the final prediction. This is the "stacking" approach from the research.

```
class CRFMetaLearner:
    """Sequence-aware meta-learner using Conditional Random Field.

    Features per token:
    1. DS consensus label (one-hot)
    2. DS posterior probability
    3. Per-engine raw predictions (one-hot per engine)
    4. Per-engine calibrated confidence
    5. Engine agreement count
    6. Token surface features (is_upper, is_digit, has_at, length)
    7. Context features (prev/next token surface)
    8. Entity type from regex (if any)
    9. Whether token is inside a regex-validated span

    The CRF learns:
    - Which engine combinations are most reliable per entity type
    - Context patterns that disambiguate entities
    - Valid BIO transitions (inherently, through CRF transition weights)
    """

    def __init__(self, model_path: Path | None = None):
        self.model = None
        if model_path and model_path.exists():
            self._load(model_path)

    def predict(self, features: list[list[dict]]) -> list[list[str]]:
        """Predict BIO tags for a sequence of tokens.

        Input: list of token feature dicts (one dict per token).
        Output: list of BIO tags.

        If model is not loaded, returns DS consensus labels as fallback.
        """

    def train(self, X_train: list[list[dict]], y_train: list[list[str]],
              X_val: list[list[dict]], y_val: list[list[str]]) -> dict:
        """Train CRF on labeled data with engine predictions as features.

        Uses L-BFGS optimization with L1+L2 regularization.
        Returns validation metrics.
        """

    def save(self, path: Path) -> None:
        """Serialize trained CRF to disk (joblib)."""

    def _load(self, path: Path) -> None:
        """Load serialized CRF from disk."""
```

**Feature extraction detail:**

```python
def extract_token_features(
    token: str,
    token_idx: int,
    tokens: list[str],
    ds_label: str,
    ds_posterior: float,
    engine_predictions: dict[str, str],      # engine_id -> BIO tag
    engine_confidences: dict[str, float],     # engine_id -> calibrated conf
    regex_validated: bool,
) -> dict[str, str | float]:
    features = {
        "ds_label": ds_label,
        "ds_posterior": ds_posterior,
        "agreement_count": sum(1 for v in engine_predictions.values() if v == ds_label),
        "n_engines": len(engine_predictions),
        "token.lower": token.lower(),
        "token.is_upper": token.isupper(),
        "token.is_title": token.istitle(),
        "token.is_digit": token.isdigit(),
        "token.has_at": "@" in token,
        "token.has_dot": "." in token,
        "token.has_dash": "-" in token,
        "token.length": min(len(token), 20),
        "regex_validated": regex_validated,
    }
    # Per-engine features
    for engine_id in sorted(engine_predictions.keys()):
        features[f"eng.{engine_id}.tag"] = engine_predictions[engine_id]
        features[f"eng.{engine_id}.conf"] = engine_confidences.get(engine_id, 0.0)
    # Context window
    if token_idx > 0:
        features["prev.lower"] = tokens[token_idx - 1].lower()
        features["prev.is_title"] = tokens[token_idx - 1].istitle()
    if token_idx < len(tokens) - 1:
        features["next.lower"] = tokens[token_idx + 1].lower()
        features["next.is_title"] = tokens[token_idx + 1].istitle()
    return features
```

#### 7.3.5 Corroboration Filter

```
class CorroborationFilter:
    """Require multiple engines to agree on ambiguous entity types."""

    def __init__(self, min_engines: int = 2,
                 exempt_types: frozenset[str] = CHECKSUM_VALIDATED_TYPES):
        self.min_engines = min_engines
        self.exempt_types = exempt_types

    def filter(self, findings: list[EnsembleFinding],
               engine_votes: dict[tuple[int,int], set[str]])
        -> list[EnsembleFinding]:
        """Remove findings without sufficient engine corroboration.

        - Exempt types (EMAIL, SSN, CC, etc.) pass through unconditionally
        - Semantic types (PERSON_NAME, ORG, LOCATION, etc.) require
          min_engines distinct engines to have predicted the same type
          at overlapping positions
        """
```

### 7.4 Layer 4: Validation Post-Processing

**New file:** `src/pii_anon/swarm/validation.py`

```
class ValidationPostProcessor:
    """Checksum and format validation on aggregated findings."""

    VALIDATORS: dict[str, Callable[[str], bool]] = {
        "CREDIT_CARD": validators.luhn_checksum,
        "IBAN": validators.iban_mod97,
        "ROUTING_NUMBER": validators.aba_routing,
        "VIN": validators.vin_check_digit,
        "US_SSN": validators.ssn_format,
        "EMAIL_ADDRESS": validators.email_format,
        "IP_ADDRESS": validators.ip_format,
        "MAC_ADDRESS": validators.mac_format,
    }

    def __init__(self, suppress_threshold: float = 0.50):
        self.suppress_threshold = suppress_threshold

    def validate(self, findings: list[EnsembleFinding], text: str)
        -> list[EnsembleFinding]:
        """Run validators on each finding's span text.

        If validator exists for entity_type:
        - Extract text[span_start:span_end]
        - Run validator
        - If FAIL: reduce confidence by 50%
        - If confidence drops below suppress_threshold: remove finding
        - If PASS: boost confidence by 5% (cap at 0.99)
        """
```

## 8. Calibration Subsystem

### 8.1 Platt Scaling (PlattCalibrator)

**New file:** `src/pii_anon/swarm/calibration.py`

```
class PlattCalibrator:
    """Per-engine temperature scaling via Platt's method.

    For each engine, learns parameters A, B such that:
        calibrated_conf = sigmoid(A * raw_conf + B)

    Trained by minimizing log-loss on a held-out validation set
    using sklearn's LogisticRegression.
    """

    def __init__(self, params: dict[str, PlattParams] | None = None):
        self.params = params or {}

    def calibrate(self, engine_id: str, raw_confidence: float) -> float:
        """Apply Platt scaling to a raw confidence score."""
        if engine_id not in self.params:
            return raw_confidence  # uncalibrated fallback
        p = self.params[engine_id]
        import math
        return 1.0 / (1.0 + math.exp(p.a * raw_confidence + p.b))

    def train(self, engine_id: str,
              raw_scores: list[float],
              true_labels: list[int]) -> PlattParams:
        """Fit Platt scaling parameters on validation data.

        raw_scores: engine confidence for each example
        true_labels: 1 if engine prediction was correct, 0 otherwise
        """

    def compute_ece(self, predicted_confs: list[float],
                    true_labels: list[int],
                    n_bins: int = 10) -> float:
        """Expected Calibration Error (Guo et al. 2017)."""
```

### 8.2 Informativeness Scorer

```
class InformativenessScorer:
    """Score engines by how informative their confidence distributions are.

    An engine that always outputs confidence=0.82 provides no discriminative
    signal. An engine with variable, calibrated confidence is more useful.

    Score = normalized entropy of confidence distribution on validation set.
    Score of 0.0 = constant confidence (useless for fusion)
    Score of 1.0 = maximum entropy (perfectly spread confidence values)
    """

    def score(self, confidences: list[float], n_bins: int = 20) -> float:
        """Compute informativeness as normalized entropy of binned confidences."""

    def compute_fusion_weights(self, platt_params: dict[str, PlattParams])
        -> dict[str, float]:
        """Derive fusion weights from informativeness scores.

        weight_i = informativeness_i / sum(informativeness_j for j in engines)

        Engines with informativeness < 0.1 get weight capped at 0.05
        to avoid division by zero while still including their findings.
        """
```

## 9. Training Pipeline

### 9.1 Dataset Registry

**New file:** `src/pii_anon/swarm/datasets.py`

```
class DatasetSpec:
    """Specification for a training dataset."""
    dataset_id: str         # e.g., "ai4privacy"
    hf_path: str           # HuggingFace path, e.g., "ai4privacy/pii-masking-200k"
    split_map: dict        # {"train": "train", "test": "test"}
    text_field: str        # Field name containing text
    label_field: str       # Field name containing labels
    label_format: str      # "bio", "spans", "token_labels"
    taxonomy_id: str       # Which taxonomy mapping to use

DATASET_SPECS: dict[str, DatasetSpec] = {
    "ai4privacy": DatasetSpec(
        dataset_id="ai4privacy",
        hf_path="ai4privacy/pii-masking-200k",
        split_map={"train": "train", "test": "validation"},
        text_field="source_text",
        label_field="privacy_mask",
        label_format="spans",
        taxonomy_id="ai4privacy_v1",
    ),
    "conll2003": DatasetSpec(
        dataset_id="conll2003",
        hf_path="eriktks/conll2003",
        split_map={"train": "train", "test": "test", "val": "validation"},
        text_field="tokens",
        label_field="ner_tags",
        label_format="bio",
        taxonomy_id="conll2003_v1",
    ),
    "pii-anon-eval": DatasetSpec(
        dataset_id="pii-anon-eval",
        hf_path=None,  # Local dataset
        split_map={"train": "train", "test": "test"},
        text_field="text",
        label_field="entities",
        label_format="spans",
        taxonomy_id="pii_anon_v1",
    ),
    "tab": DatasetSpec(
        dataset_id="tab",
        hf_path="ecthr_cases",  # TAB subset
        split_map={"train": "train", "test": "test"},
        text_field="text",
        label_field="entities",
        label_format="spans",
        taxonomy_id="tab_v1",
    ),
    "bigcode": DatasetSpec(
        dataset_id="bigcode",
        hf_path="bigcode/pii-dataset",
        split_map={"train": "train"},
        text_field="content",
        label_field="entities",
        label_format="spans",
        taxonomy_id="bigcode_v1",
    ),
    "i2b2": DatasetSpec(
        dataset_id="i2b2",
        hf_path=None,  # Requires DUA; load from local path
        split_map={"train": "train", "test": "test"},
        text_field="text",
        label_field="entities",
        label_format="spans",
        taxonomy_id="i2b2_v1",
    ),
}


class DatasetRegistry:
    """Fetch, cache, and iterate training datasets."""

    def __init__(self, cache_dir: Path = Path.home() / ".pii_anon" / "datasets"):
        self.cache_dir = cache_dir

    def load(self, dataset_id: str, split: str = "train",
             max_samples: int | None = None) -> Iterator[TrainingRecord]:
        """Load dataset, normalize to TrainingRecord format.

        Uses HuggingFace datasets library for remote datasets.
        Falls back to local file loading for datasets without hf_path.
        Gracefully skips unavailable datasets with a warning.
        """

    def available_datasets(self) -> list[str]:
        """Return dataset IDs that are loadable (HF accessible or local)."""
```

### 9.2 Taxonomy Mapping

**New file:** `src/pii_anon/swarm/taxonomy.py`

**Config file:** `src/pii_anon/swarm/taxonomy_maps/`

```json
// taxonomy_maps/ai4privacy_v1.json
{
  "first_name": "PERSON_NAME",
  "last_name": "PERSON_NAME",
  "name": "PERSON_NAME",
  "email": "EMAIL_ADDRESS",
  "phone_number": "PHONE_NUMBER",
  "street_address": "ADDRESS",
  "city": "LOCATION",
  "state": "LOCATION",
  "zip_code": "ADDRESS",
  "ssn": "US_SSN",
  "credit_card_number": "CREDIT_CARD",
  "date_of_birth": "DATE_OF_BIRTH",
  "company_name": "ORGANIZATION",
  "job_title": "_IGNORE",
  "ip_address": "IP_ADDRESS",
  "iban": "IBAN",
  "password": "_IGNORE",
  "username": "USERNAME",
  "url": "_IGNORE"
}
```

```json
// taxonomy_maps/conll2003_v1.json
{
  "PER": "PERSON_NAME",
  "ORG": "ORGANIZATION",
  "LOC": "LOCATION",
  "MISC": "_IGNORE"
}
```

```
class TaxonomyMapper:
    """Map entity labels between dataset taxonomies and pii-anon canonical types."""

    CANONICAL_TYPES: frozenset[str] = frozenset({
        "ADDRESS", "BANK_ACCOUNT", "CREDIT_CARD", "CRYPTO_WALLET",
        "DATE_OF_BIRTH", "DRIVERS_LICENSE", "EMAIL_ADDRESS", "EMPLOYEE_ID",
        "IBAN", "IP_ADDRESS", "LICENSE_PLATE", "LOCATION", "MAC_ADDRESS",
        "MEDICAL_RECORD_NUMBER", "NATIONAL_ID", "ORGANIZATION", "PASSPORT",
        "PERSON_NAME", "PHONE_NUMBER", "ROUTING_NUMBER", "USERNAME", "US_SSN",
    })

    def __init__(self, maps_dir: Path):
        self._maps: dict[str, dict[str, str]] = {}
        self._load_maps(maps_dir)

    def map(self, source_type: str, taxonomy_id: str) -> str:
        """Map a source label to canonical type. Returns '_IGNORE' for unmapped."""
```

### 9.3 Training Orchestrator

**New file:** `src/pii_anon/swarm/training.py`

```
class TrainingOrchestrator:
    """End-to-end offline training pipeline.

    Pipeline stages:
    1. Load and normalize datasets
    2. Run all engines on training texts (generate synthetic annotations)
    3. Extract features (engine predictions, confidences, token features)
    4. Train Dawid-Skene: estimate confusion matrices
    5. Train HMM: learn BIO transition probabilities
    6. Train CRF: fit meta-learner on engine features + DS output
    7. Train Platt scaling: fit calibration per engine
    8. Evaluate on held-out test split
    9. Serialize all artifacts
    """

    def __init__(self, engine_registry: EngineRegistry,
                 expert_registry: ExpertRegistry,
                 dataset_registry: DatasetRegistry,
                 taxonomy_mapper: TaxonomyMapper):
        ...

    def run(self, datasets: list[str], output_dir: Path,
            seed: int = 42, max_samples: int = 500_000,
            validation_split: float = 0.15) -> TrainingManifest:
        """Execute full training pipeline.

        Steps:
        1. Load combined dataset with taxonomy mapping
        2. Split into train/val/test (70/15/15)
        3. Generate engine predictions on train+val sets
           (run each engine on each record)
        4. Tokenize all texts to BIO format
        5. Train DS on train set -> confusion matrices
        6. Train HMM on gold BIO sequences -> transition matrix
        7. Run DS on val set -> DS predictions
        8. Extract CRF features on val set
        9. Train CRF on train set, tune on val set
        10. Train Platt scaling on val set
        11. Evaluate on test set
        12. Serialize everything to output_dir
        """
```

## 10. Data Flow Through the Four Layers

### Complete Flow for a Single Record

```
Input: payload = {"text": "Contact John Smith at john@acme.com or 555-123-4567"}

═══════════════════════════════════════════════════════════════
STEP 0: Orchestrator dispatches to SwarmPipeline
═══════════════════════════════════════════════════════════════

Orchestrator detects mode="swarm", runs regex engine first,
then passes results to SwarmPipeline.run().

═══════════════════════════════════════════════════════════════
LAYER 1: Regex Fast-Pass
═══════════════════════════════════════════════════════════════

Regex findings:
  - EMAIL_ADDRESS "john@acme.com" [24,37] conf=0.97  -> ACCEPT (>0.94)
  - PHONE_NUMBER "555-123-4567"  [41,53] conf=0.95   -> ACCEPT (>0.94)
  - PERSON_NAME "John Smith"     [8,18]  conf=0.55   -> ROUTE (< 0.94)

Accepted (bypass NER): [EMAIL_ADDRESS, PHONE_NUMBER]
Routed to Layer 2:     [PERSON_NAME needs further analysis]
Uncovered types:        {PERSON_NAME, ORGANIZATION, LOCATION, ...}

═══════════════════════════════════════════════════════════════
LAYER 2: Selective Engine Activation
═══════════════════════════════════════════════════════════════

SelectiveActivator analyzes:
  - PERSON_NAME uncovered -> need NER engines
  - GLiNER: strength=0.92 for PERSON_NAME (best) -> ACTIVATE
  - Presidio: strength=0.82 for PERSON_NAME      -> ACTIVATE
  - spaCy: strength=0.78 for PERSON_NAME          -> SKIP (Jaccard
      similarity with GLiNER > 0.85 threshold, weaker)
  - Stanza: strength=0.75                         -> SKIP (redundant)

Activated: [gliner-compatible, presidio-compatible]

Engine results:
  gliner:   PERSON_NAME "John Smith" [8,18] conf=0.89
  presidio: PERSON_NAME "John Smith" [8,18] conf=0.76
  (regex low-conf finding also passed through)

═══════════════════════════════════════════════════════════════
LAYER 3: Learned Aggregation
═══════════════════════════════════════════════════════════════

Step 3a: BIO Tokenization
  Text: "Contact John Smith at john@acme.com or 555-123-4567"
  Tokens: ["Contact", "John", "Smith", "at", "john@acme.com", "or", "555-123-4567"]
  Offsets: [(0,7), (8,12), (13,18), (19,21), (22,36), (37,39), (40,52)]

  Engine BIO annotations (for PERSON_NAME region only):
    regex:    [O, B-PERSON_NAME, I-PERSON_NAME, O, ...]  conf=[0,0.55,0.55,0,...]
    gliner:   [O, B-PERSON_NAME, I-PERSON_NAME, O, ...]  conf=[0,0.89,0.89,0,...]
    presidio: [O, B-PERSON_NAME, I-PERSON_NAME, O, ...]  conf=[0,0.76,0.76,0,...]

Step 3b: Platt Calibration
  regex PERSON_NAME conf: 0.55 -> calibrated: 0.41
  gliner PERSON_NAME conf: 0.89 -> calibrated: 0.87
  presidio PERSON_NAME conf: 0.76 -> calibrated: 0.73

Step 3c: Dawid-Skene Aggregation
  Token 1 ("John"):  All 3 engines say B-PERSON_NAME
    DS posterior: P(B-PERSON_NAME) = 0.96
  Token 2 ("Smith"): All 3 engines say I-PERSON_NAME
    DS posterior: P(I-PERSON_NAME) = 0.95

Step 3d: HMM Correction
  Sequence [O, B-PERSON_NAME, I-PERSON_NAME, O, ...] is valid.
  No correction needed.

Step 3e: CRF Meta-Learner
  Features: DS output + per-engine predictions + token features
  CRF predicts: [O, B-PERSON_NAME, I-PERSON_NAME, O, ...]
  CRF marginal confidence: [0.99, 0.94, 0.93, 0.99, ...]

Step 3f: BIO-to-Span Conversion
  B-PERSON_NAME at token 1 + I-PERSON_NAME at token 2
  -> PERSON_NAME span [8, 18] confidence = mean(0.94, 0.93) = 0.935

Step 3g: Corroboration Filter
  PERSON_NAME found by 3 engines (regex, gliner, presidio)
  corroboration_min = 2 -> PASS

═══════════════════════════════════════════════════════════════
LAYER 4: Validation Post-Processing
═══════════════════════════════════════════════════════════════

  PERSON_NAME: no checksum validator -> pass through
  EMAIL_ADDRESS (from fast-pass): email format check -> PASS (boost +5%)
  PHONE_NUMBER (from fast-pass): format check -> PASS (boost +5%)

═══════════════════════════════════════════════════════════════
FINAL OUTPUT
═══════════════════════════════════════════════════════════════

EnsembleFinding[]:
  1. EMAIL_ADDRESS "john@acme.com" [24,37] conf=0.99
     engines=["regex-oss"], explanation="fast-pass: conf=0.97, validated"
  2. PHONE_NUMBER "555-123-4567" [41,53] conf=0.99
     engines=["regex-oss"], explanation="fast-pass: conf=0.95, validated"
  3. PERSON_NAME "John Smith" [8,18] conf=0.935
     engines=["regex-oss","gliner-compatible","presidio-compatible"],
     explanation="swarm: DS+CRF, 3-engine corroboration"
```

## 11. New File Structure

```
src/pii_anon/
├── swarm/                          # NEW PACKAGE
│   ├── __init__.py                 # SwarmPipeline, SwarmConfig exports
│   ├── pipeline.py                 # SwarmPipeline class
│   ├── fast_pass.py                # RegexFastPass (Layer 1)
│   ├── activation.py               # SelectiveActivator (Layer 2)
│   ├── bio.py                      # BIOTokenizer, TokenizedText
│   ├── aggregation.py              # DawidSkeneAggregator, HMMCorrector
│   ├── meta_learner.py             # CRFMetaLearner
│   ├── corroboration.py            # CorroborationFilter (Layer 3)
│   ├── validation.py               # ValidationPostProcessor (Layer 4)
│   ├── calibration.py              # PlattCalibrator, InformativenessScorer
│   ├── config.py                   # SwarmConfig dataclass
│   ├── errors.py                   # SwarmError hierarchy
│   ├── datasets.py                 # DatasetRegistry, DatasetSpec
│   ├── taxonomy.py                 # TaxonomyMapper
│   ├── taxonomy_maps/              # JSON mapping files
│   │   ├── ai4privacy_v1.json
│   │   ├── conll2003_v1.json
│   │   ├── pii_anon_v1.json
│   │   ├── tab_v1.json
│   │   ├── bigcode_v1.json
│   │   └── i2b2_v1.json
│   ├── training.py                 # TrainingOrchestrator
│   ├── evaluation.py               # BenchmarkEvaluator, DominanceVerifier
│   └── fusion_adapter.py           # SwarmFusionStrategy (FusionStrategy subclass)
├── fusion.py                       # MODIFIED: add "swarm" to build_fusion()
├── types.py                        # MODIFIED: add "swarm" to FusionMode
└── orchestrator.py                 # MODIFIED: swarm dispatch path
```

## 12. Configuration Schema

### SwarmConfig (complete)

```python
@dataclass
class SwarmConfig:
    """Full configuration for the swarm pipeline."""

    # Layer 1: Fast-Pass
    fast_pass_threshold: float = 0.94
    fast_pass_checksum_threshold: float = 0.90

    # Layer 2: Selective Activation
    max_engines: int = 4
    coverage_similarity_threshold: float = 0.85
    engine_timeout_ms: int = 150

    # Layer 3: Aggregation
    ds_max_iterations: int = 20
    ds_warm_start: bool = True
    hmm_correction: bool = True
    use_crf_meta_learner: bool = True
    corroboration_min: int = 2
    corroboration_exempt_types: frozenset[str] = frozenset({
        "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "CREDIT_CARD_FRAGMENT",
        "IBAN", "IP_ADDRESS", "MAC_ADDRESS", "ROUTING_NUMBER",
        "CRYPTO_WALLET", "PASSPORT", "DRIVERS_LICENSE",
    })

    # Layer 4: Validation
    validation_enabled: bool = True
    validation_suppress_threshold: float = 0.50
    validation_boost: float = 0.05

    # Calibration
    calibration_enabled: bool = True
    calibration_path: Path | None = None

    # Model artifacts
    model_dir: Path | None = None  # default: ~/.pii_anon/models/swarm/

    # Fallback
    fallback_mode: str = "mixture_of_experts"  # if swarm deps unavailable
```

### Integration with ProcessingProfileSpec

```python
# In types.py, extend FusionMode:
FusionMode = (
    Literal[
        "union_high_recall",
        "weighted_consensus",
        "calibrated_majority",
        "intersection_consensus",
        "swarm",           # NEW
    ]
    | str
)

# In fusion.py, extend build_fusion():
def build_fusion(mode, *, weights, min_consensus, entity_weights, iou_threshold):
    ...
    if mode == "swarm":
        from pii_anon.swarm.fusion_adapter import SwarmFusionStrategy
        return SwarmFusionStrategy(
            strategy_params=...,  # from ProcessingProfileSpec
        )
    ...
```

### SwarmFusionStrategy

```python
class SwarmFusionStrategy(FusionStrategy):
    """Adapter that wraps SwarmPipeline as a FusionStrategy.

    The existing orchestrator calls strategy.merge(findings).
    This adapter:
    1. Intercepts the merge() call
    2. Delegates to SwarmPipeline.run() with the full context
    3. Returns list[EnsembleFinding]

    NOTE: Unlike other FusionStrategies, this one needs access to the
    original text (for BIO tokenization) and the engine registry (for
    selective activation). These are injected at construction time by
    the orchestrator.
    """
    strategy_id = "swarm"

    def __init__(self, config: SwarmConfig | None = None,
                 text: str = "",
                 field_path: str | None = None,
                 engine_registry: Any = None,
                 expert_registry: Any = None):
        self.pipeline = SwarmPipeline(
            config=config or SwarmConfig(),
            expert_registry=expert_registry,
            engine_registry=engine_registry,
        )
        self._text = text
        self._field_path = field_path

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Delegate to SwarmPipeline."""
        # Group findings by engine
        by_engine: dict[str, list[EngineFinding]] = {}
        for f in findings:
            by_engine.setdefault(f.engine_id, []).append(f)

        return self.pipeline.run(
            findings_by_engine=by_engine,
            text=self._text,
            field_path=self._field_path,
        )
```

**Orchestrator modification:** In `AsyncPIIOrchestrator._fuse()`, when `mode="swarm"`, pass the text and registries to the fusion strategy:

```python
# In orchestrator.py, modify the fusion construction for swarm
if profile.mode == "swarm":
    strategy = SwarmFusionStrategy(
        config=SwarmConfig(**profile.strategy_params.get("swarm", {})),
        text=text_value,
        field_path=field_path,
        engine_registry=self.registry,
        expert_registry=get_default_registry(),
    )
```

## 13. Requirement Coverage Matrix

| Req ID | Requirement | How This Design Addresses It | Design Element |
|--------|------------|------------------------------|----------------|
| FR-001 | Regex fast-pass | RegexFastPass class with configurable threshold (default 0.94); checksum-validated types get lower threshold (0.90) | `swarm/fast_pass.py::RegexFastPass` |
| FR-002 | Selective engine activation | SelectiveActivator uses greedy set-cover with Jaccard redundancy pruning; only activates engines covering uncovered types | `swarm/activation.py::SelectiveActivator` |
| FR-004 | Dawid-Skene Bayesian aggregation | crowd-kit DawidSkene on token-level BIO annotations; learns per-engine confusion matrices | `swarm/aggregation.py::DawidSkeneAggregator` |
| FR-005 | Span boundary reconciliation | BIO tokenization converts spans to tokens; DS aggregates at token level; BIO-to-span conversion produces clean boundaries; HMM enforces valid transitions | `swarm/bio.py::BIOTokenizer` + `swarm/aggregation.py::HMMCorrector` |
| FR-006 | Trained meta-learner | CRF meta-learner using sklearn-crfsuite; features include DS output, per-engine predictions, token features, context | `swarm/meta_learner.py::CRFMetaLearner` |
| FR-007 | Corroboration filtering | CorroborationFilter requires `corroboration_min` engines for semantic types; checksum-validated types exempt | `swarm/corroboration.py::CorroborationFilter` |
| FR-008 | Checksum/format validation | ValidationPostProcessor runs existing Luhn/IBAN/ABA/VIN validators on span text; adjusts confidence or suppresses | `swarm/validation.py::ValidationPostProcessor` |
| FR-009 | Per-engine temperature scaling | PlattCalibrator implements Platt sigmoid with sklearn LogisticRegression; per-engine A,B parameters | `swarm/calibration.py::PlattCalibrator` |
| FR-010 | Calibration-aware fusion weighting | InformativenessScorer computes entropy of confidence distributions; low-entropy engines get reduced weight | `swarm/calibration.py::InformativenessScorer` |
| FR-011 | Multi-dataset training | DatasetRegistry with 6 dataset specs; HuggingFace datasets for remote loading; local fallback for DUA-restricted data | `swarm/datasets.py::DatasetRegistry` |
| FR-012 | Taxonomy mapping | TaxonomyMapper with per-dataset JSON mapping files; 54+ source types mapped to 20 canonical types + _IGNORE | `swarm/taxonomy.py::TaxonomyMapper` + `swarm/taxonomy_maps/` |
| FR-013 | Offline training with serialized artifacts | TrainingOrchestrator produces `models/swarm/` directory with JSON configs + joblib CRF; inference loads without training deps | `swarm/training.py::TrainingOrchestrator` |
| FR-014 | Per-entity F1 with bootstrap CIs | BenchmarkEvaluator using scipy.stats bootstrap; paired-bootstrap significance tests between systems | `swarm/evaluation.py::BenchmarkEvaluator` |
| FR-015 | Dominance verification | DominanceVerifier checks swarm F1 >= best-engine F1 per entity type with configurable tolerance | `swarm/evaluation.py::DominanceVerifier` |
| FR-016 | mode="swarm" in ProcessingProfileSpec | SwarmFusionStrategy registered in build_fusion(); dispatched via mode string; all existing modes unchanged | `swarm/fusion_adapter.py` + `fusion.py` modification |
| NFR-001 | Latency <= 200ms | Fast-pass < 1ms; selective activation reduces engine count; CRF inference < 5ms; DS warm-start reduces iterations | Pipeline architecture + SwarmConfig.engine_timeout_ms |
| NFR-004 | F1 >= 0.85 | DS + CRF two-stage aggregation targets research-reported improvements; trained on 6 datasets | Full Layer 3 pipeline |
| NFR-005 | Precision >= 0.80 | Corroboration filter + validation layer + calibration-aware weighting eliminate FP explosion | Layers 3+4 |
| NFR-006 | Recall >= 0.83 | Fast-pass preserves high-confidence regex recall; union guarantee via DS (all engine votes considered) | Layers 1+3 |
| NFR-011 | Optional ML deps | All ML imports guarded by try/except; `swarm-ml` and `swarm-train` pip extras | `pyproject.toml` extras |
| NFR-013 | Backward compatibility | Existing modes untouched; new mode is additive; no API signature changes | `fusion.py::build_fusion()` extension |
| NFR-014 | API stability | Swarm activated via existing ProcessingProfileSpec.mode; run() signature unchanged | `orchestrator.py` modification |

**Coverage Summary:**
- High-priority requirements covered: 16/16 (100%)
- Medium-priority requirements covered: 3/3 (FR-003 via SelectiveActivator diversity, FR-017 via SwarmConfig, FR-018 via CLI)
- Non-functional requirements covered: 15/15 (100%)

## 14. Deployment Architecture

### Packaging

```toml
# pyproject.toml additions
[project.optional-dependencies]
swarm-ml = [
    "crowd-kit>=1.4.0",
    "sklearn-crfsuite>=0.3.6",
    "scikit-learn>=1.4.0",
    "hmmlearn>=0.3.0",
    "numpy>=1.24.0",
]
swarm-train = [
    "pii-anon[swarm-ml]",
    "datasets>=2.16.0",
    "scipy>=1.10.0",
]
```

### Model Distribution

Pre-trained swarm artifacts are distributed via:

1. **Bundled defaults**: Minimal pre-trained models shipped in `src/pii_anon/swarm/models/` (DS confusion matrices only, ~50KB)
2. **CLI download**: `pii-anon download-models --swarm` fetches full CRF model from package registry (~5MB)
3. **Custom training**: Users train their own models via `pii-anon train-swarm`

### CI/CD

```
Existing CI pipeline (unchanged)
└── + New jobs:
    ├── test-swarm-unit          # Unit tests for swarm/ package (no ML deps)
    ├── test-swarm-integration   # Integration tests with ML deps
    ├── test-swarm-benchmark     # F1 evaluation on pii-anon-eval (nightly)
    └── test-swarm-dominance     # Dominance verification (nightly)
```

## 15. Risks and Trade-offs

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| crowd-kit DawidSkene is slow on large texts | Latency exceeds 200ms for long documents | Medium | Cap text length for DS; use warm-start from pre-trained matrices; segment long texts |
| CRF model overfits to training dataset distribution | Poor generalization on unseen domains (medical, legal) | Medium | Multi-dataset training; L1+L2 regularization; cross-domain validation split |
| Whitespace tokenization misaligns with engine spans | BIO conversion produces incorrect tags | Low | Use midpoint-based token assignment; add alignment validation in tests |
| joblib serialization breaks across Python versions | Model artifacts not portable | Low | Use JSON for everything except CRF; document supported Python versions in manifest; test cross-version loading |
| crowd-kit API changes in future versions | Import errors | Low | Pin to compatible version range; wrap crowd-kit calls in adapter class |
| Dawid-Skene does not converge within iteration budget | Suboptimal aggregation | Low | Warm-start from pre-trained matrices; increase iteration budget; fall back to weighted consensus |
| HuggingFace datasets unavailable (network, DUA restrictions) | Training pipeline fails | Medium | Graceful handling of missing datasets (NFR-010); local dataset loading fallback; clear error messages |
| Six new dependencies increase install complexity | User friction | Medium | All optional (swarm-ml extra); base package unchanged; clear ImportError messages with installation instructions |

### Key Trade-offs

1. **Statistical rigor vs. latency**: Dawid-Skene EM and CRF inference add ~10-20ms per record. We accept this for significantly better aggregation quality. The 200ms budget is achievable with warm-started DS (3-5 iterations vs. 20) and optimized CRF feature extraction.

2. **Dependency count vs. correctness**: Six new optional dependencies is substantial. We justify this because each dependency replaces hand-rolled implementations that would be less tested, less correct, and harder to maintain. crowd-kit's DS implementation handles edge cases (degenerate matrices, empty annotations) that a naive implementation would miss.

3. **Token-level aggregation vs. span-level**: Operating at BIO token level is more complex than the existing span-level IoU clustering. But it is the only correct way to handle boundary disagreements: when Engine A says [10,18] and Engine B says [10,20], token-level aggregation naturally resolves this by voting per-token rather than choosing one span wholesale.

4. **Two-stage meta-learner (DS then CRF) vs. single model**: The two-stage approach is more complex but mirrors the research recommendation. DS handles the "which engines to trust" question; CRF handles the "given trust estimates, what is the most likely entity sequence" question. A single XGBoost model could do both but would not enforce BIO sequence validity.

5. **CRF over XGBoost**: We chose CRF despite XGBoost potentially achieving higher token-level accuracy, because CRF natively models sequence transitions (B-PER must be followed by I-PER or O, never I-LOC). XGBoost would require the HMM correction layer to be more aggressive, adding latency and complexity. The CRF subsumes the HMM's role while also learning from rich features.
