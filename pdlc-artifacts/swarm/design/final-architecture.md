# Final Architecture: pii-anon-swarm

**Date**: 2026-03-27
**Decision**: Hybrid — Proposal A (simplicity) as base, with selective activation, informativeness scoring, training manifest, and dominance verification from Proposal B.

---

## 1. System Architecture

```
                         pii-anon-swarm Pipeline
                         ══════════════════════════

 Input: text + ProcessingProfileSpec(mode="swarm")
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │  LAYER 1: Regex Fast-Pass                            │
 │  ─────────────────────                               │
 │  Run regex-oss engine on full text                   │
 │  For each finding with confidence >= fast_pass_threshold (0.90): │
 │    → Emit directly as EnsembleFinding (bypass Layers 2-3)       │
 │  Remaining low-confidence regex findings → Layer 2              │
 └──────────────────────┬───────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │  LAYER 2: Heterogeneous NER Detection                │
 │  ────────────────────────────────                    │
 │  _prune_redundant_engines():                         │
 │    Compute pairwise Jaccard on entity_strengths      │
 │    Greedy set-cover: select engines maximizing        │
 │    coverage with Jaccard < similarity_threshold (0.85)│
 │  Run selected engines on text                        │
 │  Combine: low-conf regex findings + NER findings     │
 └──────────────────────┬───────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │  LAYER 3: Learned Aggregation                        │
 │  ────────────────────────                            │
 │  Step 3a: Cluster overlapping spans (IoU >= 0.3)     │
 │  Step 3b: Temperature-scale per-engine confidences   │
 │  Step 3c: Dawid-Skene aggregation per cluster        │
 │           (pre-trained confusion matrices, no EM)    │
 │  Step 3d: Informativeness-weighted fusion             │
 │           (downweight fixed-confidence engines)       │
 │  Step 3e: XGBoost meta-learner prediction            │
 │           (20 features per candidate span)            │
 │           Fallback: logistic scoring function         │
 └──────────────────────┬───────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────┐
 │  LAYER 4: Validation & Post-Processing               │
 │  ─────────────────────────────────                   │
 │  Corroboration filter:                               │
 │    Semantic types (PERSON_NAME, ORG, LOCATION)       │
 │    require corroboration_min engines (default 2)     │
 │    unless meta_score > override_threshold (0.85)     │
 │  Checksum validation:                                │
 │    Run Luhn, IBAN mod-97, ABA, etc. on span text     │
 │    Suppress findings that fail validation             │
 │  Emit: list[EnsembleFinding] with calibrated conf    │
 └──────────────────────────────────────────────────────┘
```

## 2. New Files

| File | Responsibility | ~LOC |
|------|---------------|------|
| `src/pii_anon/swarm.py` | Core pipeline: SwarmFusionStrategy, DawidSkeneAggregator, TemperatureScaler, InformativenessScorer, finding pruning | ~500 |
| `src/pii_anon/swarm_learner.py` | XGBoostMetaLearner: feature extraction (20 features), predict, load/save, logistic fallback | ~300 |
| `src/pii_anon/swarm_train.py` | Training CLI: dataset loading, DS EM training, temperature fitting, XGBoost training, dominance verification | ~400 |
| `src/pii_anon/swarm_datasets.py` | Dataset loaders and taxonomy mapping for 6 industry datasets | ~250 |
| `src/pii_anon/swarm_config.json` | Default configuration (thresholds, paths, dataset specs) | ~50 |
| `tests/test_swarm.py` | Unit and integration tests for the swarm pipeline | ~500 |
| `tests/test_swarm_train.py` | Training pipeline tests | ~200 |

**Total**: ~2,200 LOC across 7 files (5 source + 2 test)

## 3. Modified Files

| File | Change |
|------|--------|
| `src/pii_anon/fusion.py` | Add `"swarm"` case to `build_fusion()` factory |
| `src/pii_anon/types.py` | Add `"swarm"` to `FusionMode` literal type |
| `pyproject.toml` | Add `[swarm-ml]` optional extra: xgboost, scikit-learn, numpy |

## 4. Key Data Structures

### SpanCandidate (internal to swarm.py)
```python
@dataclass
class SpanCandidate:
    entity_type: str
    span_start: int
    span_end: int
    field_path: str | None
    engine_findings: dict[str, EngineFinding]  # engine_id → finding
    ds_confidence: float  # Dawid-Skene consensus confidence
    meta_score: float     # XGBoost or logistic score
    corroboration_count: int  # number of engines that detected this
```

### SwarmConfig (loaded from JSON)
```python
@dataclass
class SwarmConfig:
    fast_pass_threshold: float = 0.90
    iou_threshold: float = 0.3  # lower than current 0.5 to merge more aggressively
    corroboration_min: int = 2
    corroboration_override_threshold: float = 0.85
    similarity_threshold: float = 0.85  # Jaccard threshold for engine pruning
    max_engines: int = 4
    meta_learner_path: str | None = None
    ds_params_path: str | None = None
    calibration_path: str | None = None
    emission_threshold: float = 0.50
```

### TrainingManifest (serialized with artifacts)
```python
@dataclass
class TrainingManifest:
    schema_version: str
    trained_at: str  # ISO-8601
    pii_anon_version: str
    python_version: str
    datasets_used: list[str]
    total_records: int
    per_entity_f1: dict[str, float]
    dominance_violations: list[str]  # empty if all pass
```

## 5. Dawid-Skene Implementation

Pure-Python, ~120 lines. No crowd-kit dependency.

**Training (EM algorithm)**:
```
Input: N labeled records, E engines, T entity types
For each record:
  Run all engines → per-engine per-span predictions
  Align predictions to gold labels
  Build annotator label matrix: annotations[engine][item] = predicted_type_or_O

EM Loop (max 50 iterations, convergence threshold 1e-4):
  E-step: For each item i, compute P(true_label=t | annotations) using Bayes
  M-step: Update confusion matrices π[e][t][t'] = P(engine e says t' | true label is t)
  Check convergence: max change in π < threshold

Output: confusion_matrices dict[engine_id → 2D array], class_priors dict[type → float]
```

**Inference** (pre-trained parameters, no EM):
```
For each SpanCandidate:
  For each engine in candidate.engine_findings:
    Look up P(engine says observed_type | true_type=t) from confusion_matrices
  Combine via Bayes: P(true_type=t | all observations) ∝ prior[t] × Π P(obs_e | t)
  ds_confidence = max_t P(true_type=t | all observations)
```

## 6. XGBoost Meta-Learner Features (20 features per SpanCandidate)

| # | Feature | Description |
|---|---------|-------------|
| 1 | ds_confidence | Dawid-Skene consensus probability |
| 2 | corroboration_count | Number of engines detecting this span |
| 3 | corroboration_ratio | corroboration_count / total_active_engines |
| 4 | max_engine_confidence | Highest calibrated confidence across engines |
| 5 | min_engine_confidence | Lowest calibrated confidence across engines |
| 6 | mean_engine_confidence | Mean calibrated confidence |
| 7 | std_engine_confidence | Std dev of calibrated confidences |
| 8 | regex_detected | Binary: did regex-oss detect this span? |
| 9 | regex_confidence | Regex confidence (0 if not detected) |
| 10 | span_length_chars | Character length of span |
| 11 | span_length_tokens | Approximate token count (split on whitespace) |
| 12 | entity_type_encoded | Integer encoding of entity type |
| 13 | is_structured_type | Binary: EMAIL, SSN, CC, IBAN, IP, MAC, etc. |
| 14 | has_checksum_validation | Binary: passed Luhn/IBAN/ABA validation |
| 15 | informativeness_score | Mean informativeness of contributing engines |
| 16 | boundary_agreement | IoU between engines' span boundaries |
| 17 | context_has_keywords | Binary: PII-related keywords within 50 chars |
| 18 | position_in_text | Normalized position (0=start, 1=end) |
| 19 | surrounding_entity_density | Count of other findings within 100 chars |
| 20 | engine_diversity_score | Mean pairwise Jaccard dissimilarity of contributing engines |

## 7. Training Pipeline

```
pii-anon train-swarm --datasets ai4privacy,conll2003,pii-anon-eval --output ./swarm-artifacts/
```

**Steps**:
1. Load datasets via DatasetSpec registry (graceful skip if unavailable)
2. Map entity types to canonical taxonomy via swarm_datasets.py
3. Split: 70% train, 15% calibration, 15% test
4. Run all available engines on train split → raw predictions
5. Align predictions to gold labels → annotator label matrix
6. Train Dawid-Skene confusion matrices (EM, 50 iterations)
7. Fit temperature scaling parameters per engine (minimize CE on calibration split)
8. Compute informativeness scores per engine (confidence variance on calibration split)
9. Extract 20 features per SpanCandidate on train split
10. Train XGBoost binary classifier (is_correct_detection: 0/1)
11. Evaluate on test split: per-entity-type F1 with bootstrap CIs
12. Run dominance verification: swarm F1 >= best-engine F1 per type
13. Serialize artifacts: ds_params.json, temperature.json, xgboost_model.ubj, manifest.json

## 8. Dataset Integration

| Dataset | Records | Types | Format | Loader |
|---------|---------|-------|--------|--------|
| pii-anon-eval-data | 117K | 20 | JSONL via package | `pii_anon_datasets.load_dataset()` |
| AI4Privacy pii-masking-200k | 209K | 54 | HuggingFace | `datasets.load_dataset("ai4privacy/pii-masking-200k")` |
| CoNLL-2003 | 20K | 4 (PER,LOC,ORG,MISC) | HuggingFace | `datasets.load_dataset("conll2003")` |
| TAB (Text Anonymization) | ~1K | 12 | Download + JSONL | Custom loader |
| BigCode PII | ~10K | 6 | HuggingFace | `datasets.load_dataset("bigcode/pii-dataset")` |
| i2b2 de-identification | ~1K | 18 | Requires DUA | Custom loader (skip if unavailable) |

**Taxonomy mapping** (single JSON file):
```json
{
  "ai4privacy": {"firstname": "PERSON_NAME", "lastname": "PERSON_NAME", "email": "EMAIL_ADDRESS", ...},
  "conll2003": {"PER": "PERSON_NAME", "LOC": "LOCATION", "ORG": "ORGANIZATION", "MISC": "_IGNORE"},
  "tab": {"PERSON": "PERSON_NAME", "LOCATION": "LOCATION", ...},
  "bigcode": {"EMAIL": "EMAIL_ADDRESS", "IP_ADDRESS": "IP_ADDRESS", ...},
  "i2b2": {"PATIENT": "PERSON_NAME", "DOCTOR": "PERSON_NAME", "HOSPITAL": "ORGANIZATION", ...}
}
```

## 9. Integration Points

### build_fusion() factory (fusion.py)
```python
def build_fusion(mode, ...):
    ...
    elif mode == "swarm":
        from pii_anon.swarm import SwarmFusionStrategy
        return SwarmFusionStrategy(config_path=swarm_config_path)
```

### FusionMode type (types.py)
```python
FusionMode = Literal[
    "union_high_recall",
    "weighted_consensus",
    "calibrated_majority",
    "intersection_consensus",
    "mixture_of_experts",
    "swarm",  # NEW
]
```

### Optional dependencies (pyproject.toml)
```toml
[project.optional-dependencies]
swarm-ml = ["xgboost>=2.0", "scikit-learn>=1.4", "numpy>=1.24"]
swarm-train = ["xgboost>=2.0", "scikit-learn>=1.4", "numpy>=1.24", "datasets>=2.14", "tqdm>=4.66"]
```

## 10. Risk Mitigation

| Risk | Mitigation | Trigger |
|------|-----------|---------|
| Span-level DS insufficient for boundary precision | Track boundary-level precision during training; if < 0.80, add boundary refinement heuristic | Week 1 prototype |
| XGBoost unavailable at inference | Logistic fallback function achieves F1 >= 0.80 | Validated during training |
| Stale training artifacts | Version check on load; warn if > 90 days old or different library version | Every inference |
| Dataset unavailable (i2b2 DUA) | Graceful skip with warning; pipeline trains on available datasets | Training time |
| F1 < 0.85 after full implementation | Escalation path: add token-level BIO from Proposal B as a refinement layer | Post Week 2 eval |

---

*Design phase complete. Hybrid architecture selected (Proposal A base + targeted B elements). 7 new files, ~2,200 LOC. Ready for Development.*
