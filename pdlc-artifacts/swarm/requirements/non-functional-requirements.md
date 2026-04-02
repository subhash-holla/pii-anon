# Non-Functional Requirements: pii-anon-swarm

**Date**: 2026-03-27

---

## Performance

### NFR-001: Swarm detection latency
**Source**: Discovery constraint (currently ~90ms/record)
**Priority**: High

The system must complete swarm detection in <= 200ms per record (p50) when all four layers are active. When the regex fast-pass accepts directly (Layer 1 only), latency must be <= 2ms per record.

**Measurement**: Benchmark 1000 records, report p50 and p99 latency. Fast-pass path measured separately.

### NFR-002: Swarm throughput
**Source**: Persona (Library Developer — data pipeline use case)
**Priority**: Medium

The system must sustain >= 10,000 records/hour in full swarm mode on a single CPU core. In fast-pass-only mode (high-confidence regex), throughput must be >= 500,000 records/hour.

**Measurement**: Timed benchmark on 10,000 records, report records/hour for each mode.

### NFR-003: Meta-learner inference overhead
**Source**: FR-006 (trained meta-learner)
**Priority**: High

The meta-learner inference step must add <= 5ms per record to the detection pipeline. Model loading at initialization must complete in <= 3 seconds.

**Measurement**: Profile meta-learner.predict() across 1000 records, report p50 and p99 latency. Time model loading separately.

---

## Accuracy

### NFR-004: Swarm F1 target
**Source**: Discovery (current F1=0.648), Research (target 97-98%)
**Priority**: High

The system must achieve entity-level F1 >= 0.85 on the pii-anon-eval-data benchmark (117K records) with strict span matching. This represents a +20 point improvement over the current swarm (0.648).

**Measurement**: Run full benchmark, report micro-averaged F1 with bootstrap 95% CI.

### NFR-005: Swarm precision floor
**Source**: P-1 (precision=0.528 currently)
**Priority**: High

The system must achieve entity-level precision >= 0.80, eliminating the false-positive explosion that currently tanks swarm quality.

**Measurement**: Report precision on full benchmark. Per-entity-type precision must be >= 0.60 for all 20 entity types.

### NFR-006: Swarm recall preservation
**Source**: Discovery (current recall=0.839, best among all systems)
**Priority**: High

The system must maintain entity-level recall >= 0.83, preserving the swarm's recall advantage over individual engines.

**Measurement**: Report recall on full benchmark. Per-entity-type recall must not decrease by more than 5 percentage points for any entity type vs. the current swarm.

### NFR-007: Dominance guarantee
**Source**: FR-015, Research (ensemble >= best individual)
**Priority**: High

The system must satisfy the dominance property: for every entity type, swarm F1 >= max(individual engine F1) - 0.02 (allowing 2-point tolerance for statistical noise).

**Measurement**: Run dominance verification on full benchmark. Zero violations allowed beyond the 2-point tolerance.

---

## Training

### NFR-008: Training pipeline reproducibility
**Source**: FR-013
**Priority**: Medium

The training pipeline must produce deterministic results given the same input data and random seed. Two training runs with the same configuration must produce models whose F1 scores differ by <= 0.005.

**Measurement**: Run training twice with same seed, compare F1 on held-out test set.

### NFR-009: Training time budget
**Source**: FR-011 (multi-dataset training)
**Priority**: Medium

The complete training pipeline (data loading, feature extraction, meta-learner training, calibration) must complete in <= 2 hours on a machine with 8 CPU cores and 32GB RAM, using up to 500K combined training records.

**Measurement**: Time the full training pipeline end-to-end.

### NFR-010: Training data compatibility
**Source**: FR-011, FR-012
**Priority**: High

The training pipeline must gracefully handle missing datasets. If only 2 of 6 datasets are available, training must succeed on the available data with appropriate warnings. The system must not require all datasets to be present.

**Measurement**: Run training with only pii-anon-eval-data and CoNLL-2003, verify model trains successfully and F1 >= 0.80 on pii-anon benchmark.

---

## Dependencies and Packaging

### NFR-011: Optional ML dependencies
**Source**: Scope constraint (new deps as optional extras)
**Priority**: High

All ML/DL dependencies required for the meta-learner (e.g., scikit-learn, xgboost, crowd-kit) must be declared as optional extras (e.g., `pip install pii-anon[swarm-ml]`). The base pii-anon package must remain installable with only pydantic as a required dependency.

**Measurement**: `pip install pii-anon` succeeds without ML dependencies. Attempting to use `mode="swarm"` without ML dependencies installed raises a clear ImportError with installation instructions.

### NFR-012: Serialized model portability
**Source**: FR-013
**Priority**: Medium

Trained model artifacts must be portable across Python 3.10-3.13 and across operating systems (Linux, macOS, Windows). Artifacts must not contain absolute paths or platform-specific binaries.

**Measurement**: Train on macOS, load and run inference on Linux. Verify identical F1 results.

---

## Compatibility

### NFR-013: Backward compatibility with existing modes
**Source**: Discovery constraint
**Priority**: High

All existing fusion modes ("union_high_recall", "weighted_consensus", "calibrated_majority", "intersection_consensus", "mixture_of_experts") must continue to function identically. No existing test may break.

**Measurement**: Full test suite (2273 tests) passes with zero regressions.

### NFR-014: API stability
**Source**: Persona (Library Developer)
**Priority**: High

The `PIIOrchestrator.run()` API signature and return format must remain unchanged. The new swarm mode must be activated solely through `ProcessingProfileSpec` configuration, not through new API methods.

**Measurement**: Existing integration tests pass without modification. The only new API surface is the `mode="swarm"` value and optional `SwarmConfig` parameter.

---

## Observability

### NFR-015: Swarm audit trail
**Source**: Existing audit infrastructure
**Priority**: Medium

The system must produce `FusionAuditRecord` entries for the new swarm pipeline that include: which layer produced each finding (fast-pass, NER, meta-learner), which engines contributed, the pre-calibration and post-calibration confidence scores, and whether corroboration was required/met.

**Measurement**: Running with `audit_enabled=True` produces audit records for every emitted finding with all specified fields populated.
