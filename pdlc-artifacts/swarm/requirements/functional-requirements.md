# Functional Requirements: pii-anon-swarm

**Date**: 2026-03-27
**Source**: Discovery report + Research report + User direction for fundamental redesign

---

## Layer 1: Regex Fast-Pass

### FR-001: Confidence-gated regex fast-pass
**Source**: O-5, Research (dynamic confidence cascade)
**Priority**: High

Ability to accept regex-only detections immediately when the engine reports confidence above a configurable threshold, bypassing all downstream NER engines and aggregation layers.

**Verification**: Given a text with a Luhn-validated credit card number (regex confidence >= 0.94), the swarm produces an EnsembleFinding within 1ms without invoking any NER engine. Given a PERSON_NAME match with regex confidence 0.55, the swarm routes to Layer 2.

---

## Layer 2: Heterogeneous Parallel Detection

### FR-002: Selective engine activation based on entity-type coverage gaps
**Source**: P-2, G-4, Research (heterogeneous ensemble > homogeneous swarm)
**Priority**: High

Ability to activate only engines that provide non-redundant coverage for entity types where the regex fast-pass was not confident. The system must not run engines whose detection capabilities are fully subsumed by already-activated engines.

**Verification**: Given that regex detected all structured PII with high confidence but PERSON_NAME with low confidence, only engines with demonstrated PERSON_NAME recall (e.g., GLiNER, Presidio) are activated. spaCy and Stanza are not both activated if their detection overlap exceeds a configurable similarity threshold (default 0.85 Jaccard).

### FR-003: Engine diversity enforcement
**Source**: P-2, Research (LLM-TOPLA diversity-maximizing selection)
**Priority**: Medium

Ability to enforce a minimum diversity threshold among activated engines, measured by pairwise Jaccard dissimilarity on their detection output. The system must prefer engine combinations that maximize complementary error patterns.

**Verification**: Given 5 available NER engines, the system selects the subset with the lowest pairwise Jaccard similarity (highest diversity) that still covers all required entity types.

---

## Layer 3: Learned Aggregation

### FR-004: Dawid-Skene Bayesian aggregation for span-level label fusion
**Source**: Research (Dawid-Skene > majority voting), P-1
**Priority**: High

Ability to aggregate entity detections from multiple engines using the Dawid-Skene EM algorithm, which learns per-engine confusion matrices (accuracy patterns per entity type) and infers consensus labels that account for each engine's systematic biases.

**Verification**: Given 3 engines where Engine A consistently misses PHONE_NUMBER but excels at PERSON_NAME, the aggregator learns A's confusion matrix and downweights A's PHONE_NUMBER predictions while trusting its PERSON_NAME predictions. The aggregated F1 exceeds the best individual engine's F1.

### FR-005: Span boundary reconciliation with learned preferences
**Source**: P-1, G-1, Research (HMM sequence correction)
**Priority**: High

Ability to resolve conflicting span boundaries from multiple engines by learning which engine produces the most accurate boundaries per entity type, rather than using fixed IoU thresholds or majority voting.

**Verification**: Given Engine A detecting "john@example.com" at [10,28] and Engine B detecting "john@example.c" at [10,26], the system merges these into a single finding with the boundary from the engine that historically produces more accurate email boundaries. No duplicate findings are emitted.

### FR-006: Trained meta-learner for final prediction
**Source**: Research (CRF/XGBoost stacking meta-learner)
**Priority**: High

Ability to produce final entity predictions through a trained meta-learner model that takes as input: per-engine predictions, calibrated confidence scores, entity types, span positions, contextual features (surrounding tokens), and engine agreement patterns. The meta-learner must be trainable offline on labeled data and loadable at inference time without retraining.

**Verification**: A trained meta-learner loaded from a serialized artifact produces ensemble F1 >= 0.85 on the pii-anon benchmark, surpassing both the best individual engine and the Dawid-Skene aggregation alone.

### FR-007: Corroboration filtering for low-precision entity types
**Source**: G-2, O-2, Research (corroboration filtering)
**Priority**: High

Ability to require a configurable minimum number of corroborating engines before accepting detections of entity types known to have high false-positive rates (e.g., PERSON_NAME, ORGANIZATION, LOCATION). Structured PII types (EMAIL, SSN, CREDIT_CARD) detected by regex with checksum validation must be exempt from corroboration requirements.

**Verification**: A PERSON_NAME detected by only one NER engine is suppressed. A PERSON_NAME detected by 2+ engines is accepted. An EMAIL_ADDRESS detected by regex alone with Luhn/format validation is accepted without corroboration.

---

## Layer 4: Validation and Post-Processing

### FR-008: Checksum and format validation on aggregated spans
**Source**: Research (Layer 4 validation), existing regex validators
**Priority**: High

Ability to run format and checksum validators (Luhn, IBAN mod-97, ABA routing, VIN check-digit) on the text spans of aggregated findings, and to adjust confidence or suppress findings that fail validation.

**Verification**: An aggregated finding of entity_type=CREDIT_CARD whose span text fails Luhn checksum is either suppressed or confidence-reduced below the emission threshold.

---

## Confidence Calibration

### FR-009: Per-engine temperature scaling calibration
**Source**: P-3, G-3, Research (temperature scaling, Guo et al. 2017)
**Priority**: High

Ability to learn and apply a per-engine temperature scaling parameter that maps raw engine confidence scores to calibrated probabilities, trained on a held-out validation set by minimizing cross-entropy. The calibration parameters must be persistable and loadable separately from the meta-learner.

**Verification**: After calibration, each engine's Expected Calibration Error (ECE) on a held-out test set is <= 0.15. Reliability diagrams show predicted confidence vs. actual precision within 5 percentage points per bin.

### FR-010: Calibration-aware fusion weighting
**Source**: O-3, P-3
**Priority**: High

Ability to weight each engine's contribution to the aggregation based on the informativeness of its confidence scores. Engines with fixed/uninformative confidence (zero variance) must receive reduced influence compared to engines with calibrated, variable confidence.

**Verification**: An engine that reports fixed confidence=0.82 for all findings receives <= 50% of the aggregation weight of an engine with calibrated variable confidence (ECE <= 0.15), all else being equal.

---

## Training Data Integration

### FR-011: Multi-dataset training pipeline for meta-learner
**Source**: User requirement, Research (AI4Privacy, CoNLL-2003, TAB, BigCode, i2b2)
**Priority**: High

Ability to train the meta-learner and calibration parameters on a combined corpus drawn from multiple industry-standard datasets, with a unified schema mapping. The training pipeline must support:
- AI4Privacy pii-masking-200k (209K samples, 54 PII types)
- CoNLL-2003 (standard NER)
- TAB — Text Anonymization Benchmark (court cases)
- BigCode PII Dataset (source code)
- i2b2 de-identification (clinical text)
- pii-anon-eval-data (117K records)

**Verification**: The training pipeline successfully loads, normalizes, and trains on records from at least 3 of the 6 datasets. The trained model's per-entity-type F1 on a held-out test split is reported for each source dataset independently.

### FR-012: Entity type taxonomy mapping across datasets
**Source**: FR-011, Research (taxonomy misalignment)
**Priority**: High

Ability to map entity type labels from each source dataset's taxonomy to the pii-anon canonical taxonomy (20 entity types). The mapping must handle: different naming conventions (PERSON vs PER vs PERSON_NAME), granularity differences (first_name+last_name vs full name), and dataset-specific types that have no pii-anon equivalent (mapped to _IGNORE).

**Verification**: Given a CoNLL-2003 record with label "PER" and an AI4Privacy record with label "first_name", both map to the canonical "PERSON_NAME" type. Dataset-specific types like CoNLL's "MISC" map to "_IGNORE" and are excluded from training.

### FR-013: Offline training with serialized model artifacts
**Source**: User requirement (training separate from inference)
**Priority**: High

Ability to run the complete training pipeline (data loading, meta-learner training, calibration, evaluation) as an offline batch process that produces serialized model artifacts. Inference must load these artifacts without access to training data or training dependencies.

**Verification**: Running `pii-anon train-swarm --datasets ai4privacy,conll2003,pii-anon-eval` produces a serialized model directory. A fresh Python environment with only inference dependencies can load the artifacts and run swarm detection.

---

## Evaluation and Benchmarking

### FR-014: Per-entity-type F1 reporting with statistical significance
**Source**: Research (nervaluate, SemEval 2013 schemas)
**Priority**: High

Ability to report entity-level F1, precision, and recall per entity type, with bootstrap 95% confidence intervals and pairwise paired-bootstrap significance tests between systems (swarm vs. regex, swarm vs. individual engines).

**Verification**: The benchmark report includes per-entity-type F1 with 95% CIs for all 20 entity types, and pairwise significance tests showing whether the swarm improvement over the best individual engine is statistically significant (p < 0.05).

### FR-015: Dominance guarantee verification
**Source**: Research (ensemble must beat best individual)
**Priority**: High

Ability to verify that the swarm's per-entity-type F1 meets or exceeds the best individual engine's F1 for every entity type. Violations must be flagged with the entity type, the gap, and the engine that outperforms the swarm.

**Verification**: Running `pii-anon verify-dominance` on the benchmark dataset produces a pass/fail report. Any entity type where swarm F1 < best-engine F1 is listed as a violation with the magnitude of the gap.

---

## Configuration and API

### FR-016: Swarm mode as a ProcessingProfileSpec option
**Source**: Backward compatibility constraint
**Priority**: High

Ability to activate the new swarm architecture through the existing `ProcessingProfileSpec(mode="swarm")` configuration, without breaking existing modes ("weighted_consensus", "mixture_of_experts", etc.).

**Verification**: `PIIOrchestrator.run(profile=ProcessingProfileSpec(mode="swarm"))` uses the new four-layer pipeline. `mode="mixture_of_experts"` continues to use the existing MoE pipeline unchanged.

### FR-017: Configurable swarm pipeline parameters
**Source**: Personas (Library Developer, Compliance Engineer)
**Priority**: Medium

Ability to configure swarm behavior through a dedicated configuration object, including: fast-pass confidence threshold, corroboration minimum, engine activation list, meta-learner model path, and calibration parameters path.

**Verification**: Setting `fast_pass_threshold=0.85` causes the regex fast-pass to accept entities with confidence >= 0.85 directly. Setting `corroboration_min=3` requires 3 engines to agree on semantic entity types.

### FR-018: CLI commands for training and evaluation
**Source**: FR-013, FR-015
**Priority**: Medium

Ability to run swarm training, calibration, and dominance verification from the pii-anon CLI:
- `pii-anon train-swarm --datasets <list> --output <dir>`
- `pii-anon calibrate-swarm --dataset <name> --output <path>`
- `pii-anon verify-dominance --dataset <name>`

**Verification**: Each CLI command runs to completion and produces the expected output artifacts.
