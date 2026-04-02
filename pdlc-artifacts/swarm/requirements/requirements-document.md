# Requirements Document: pii-anon-swarm

**Date**: 2026-03-27
**Status**: COMPLETE

---

## 1. Executive Summary

18 functional requirements and 15 non-functional requirements define the fundamental redesign of pii-anon-swarm. The new architecture replaces naive weighted voting with a four-layer pipeline: regex fast-pass, heterogeneous NER, learned aggregation (Dawid-Skene + trained meta-learner), and validation. Training uses industry-standard datasets alongside pii-anon-eval-data.

**Priority Breakdown**: 14 High, 5 Medium

---

## 2. Traceability Matrix

| Discovery Finding | Requirements |
|-------------------|-------------|
| P-1: Span duplication (+381 FPs) | FR-005, FR-006, FR-007, NFR-005 |
| P-2: Engine redundancy (5/6 zero marginal value) | FR-002, FR-003, NFR-002 |
| P-3: Fixed confidence (3/6 engines) | FR-009, FR-010, NFR-004 |
| G-1: No span deduplication | FR-005 |
| G-2: No corroboration requirement | FR-007 |
| G-3: No confidence calibration layer | FR-009, FR-010 |
| G-4: No engine pruning | FR-002, FR-003 |
| O-1: Aggressive span merging | FR-005, FR-006 |
| O-2: Corroboration filter | FR-007 |
| O-3: Calibration-aware weighting | FR-009, FR-010 |
| O-4: Drop redundant engines | FR-002, FR-003 |
| O-5: Confidence cascade | FR-001, NFR-001 |
| Research: Dawid-Skene aggregation | FR-004 |
| Research: Trained meta-learner | FR-006, FR-013, NFR-003 |
| Research: Four-layer architecture | FR-001, FR-002, FR-006, FR-008 |
| Research: Multi-dataset training | FR-011, FR-012, NFR-009, NFR-010 |
| Research: Statistical evaluation | FR-014, FR-015 |
| User: Industry datasets | FR-011, FR-012, FR-013 |

---

## 3. Requirements Summary

### Functional Requirements (18)

| ID | Title | Priority | Source |
|----|-------|----------|--------|
| FR-001 | Confidence-gated regex fast-pass | High | O-5, Research |
| FR-002 | Selective engine activation | High | P-2, G-4 |
| FR-003 | Engine diversity enforcement | Medium | P-2, Research |
| FR-004 | Dawid-Skene Bayesian aggregation | High | Research |
| FR-005 | Span boundary reconciliation with learned preferences | High | P-1, G-1 |
| FR-006 | Trained meta-learner for final prediction | High | Research |
| FR-007 | Corroboration filtering | High | G-2, O-2 |
| FR-008 | Checksum/format validation on aggregated spans | High | Research |
| FR-009 | Per-engine temperature scaling calibration | High | P-3, G-3 |
| FR-010 | Calibration-aware fusion weighting | High | O-3, P-3 |
| FR-011 | Multi-dataset training pipeline | High | User, Research |
| FR-012 | Entity type taxonomy mapping | High | FR-011 |
| FR-013 | Offline training with serialized artifacts | High | User |
| FR-014 | Per-entity-type F1 with statistical significance | High | Research |
| FR-015 | Dominance guarantee verification | High | Research |
| FR-016 | Swarm mode as ProcessingProfileSpec option | High | Compat |
| FR-017 | Configurable swarm parameters | Medium | Personas |
| FR-018 | CLI commands for training/evaluation | Medium | FR-013 |

### Non-Functional Requirements (15)

| ID | Title | Priority | Target |
|----|-------|----------|--------|
| NFR-001 | Swarm detection latency | High | <= 200ms/record (p50) |
| NFR-002 | Swarm throughput | Medium | >= 10K records/hour |
| NFR-003 | Meta-learner inference overhead | High | <= 5ms/record |
| NFR-004 | Swarm F1 target | High | >= 0.85 |
| NFR-005 | Swarm precision floor | High | >= 0.80 |
| NFR-006 | Swarm recall preservation | High | >= 0.83 |
| NFR-007 | Dominance guarantee | High | Swarm >= best engine - 0.02 |
| NFR-008 | Training reproducibility | Medium | Delta <= 0.005 F1 |
| NFR-009 | Training time budget | Medium | <= 2 hours |
| NFR-010 | Training data compatibility | High | Graceful with partial data |
| NFR-011 | Optional ML dependencies | High | pii-anon[swarm-ml] |
| NFR-012 | Model portability | Medium | Cross-platform, Py 3.10-3.13 |
| NFR-013 | Backward compatibility | High | 2273 tests pass |
| NFR-014 | API stability | High | No breaking changes |
| NFR-015 | Swarm audit trail | Medium | Full audit records |

---

## 4. Scope Boundaries

**In Scope**:
- Four-layer swarm pipeline (regex → NER → meta-learner → validation)
- Dawid-Skene aggregation and trained meta-learner
- Multi-dataset training pipeline (6 datasets)
- Per-engine confidence calibration
- CLI training/evaluation commands
- New `mode="swarm"` in ProcessingProfileSpec

**Out of Scope (Future Work)**:
- Fine-tuned DeBERTa-v3 with LoRA adapters (requires GPU training infrastructure)
- LLM-based detection agents (beyond current engine adapters)
- Real-time online learning (EMA calibration exists; meta-learner retraining is offline only)
- Multilingual expansion beyond current en/es/fr support

---

*Requirements phase complete. 18 FR + 15 NFR = 33 total requirements. Priority: 14 High, 5 Medium. Ready for Design.*
