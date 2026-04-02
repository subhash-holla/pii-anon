# PDLC MANIFEST — Swarm: Research-Driven Architecture Overhaul

## Goal
Transform pii-anon-swarm from naive weighted voting (F1=0.648) to a research-backed aggregation architecture targeting F1 >= 0.85, closing the gap with pii-anon regex (F1=0.792) and surpassing GLiNER (F1=0.763).

## Stage Status

| Stage | Status | Started | Completed |
|-------|--------|---------|-----------|
| Discovery | COMPLETE | 2026-03-27 | 2026-03-27 |
| Requirements | COMPLETE | 2026-03-27 | 2026-03-27 |
| Design | COMPLETE | 2026-03-27 | 2026-03-27 |
| Development | COMPLETE | 2026-03-27 | 2026-03-27 |
| Testing | COMPLETE | 2026-03-27 | 2026-03-27 |
| Management | NOT STARTED | — | — |

## Input
- Research report: "Making PII anonymization swarms actually work" (2026-03)
- Current benchmark: pii-anon-swarm F1=0.648, P=0.528, R=0.839
- Codebase: pii-anon with MoE fusion, 6 engine adapters, calibration system

## Discovery Artifacts
- `discovery/discovery-report.md` — Full discovery report with root causes and recommendations
- `discovery/precision-diagnosis.md` — Per-entity-type FP analysis (100 records)
- `discovery/precision-diagnosis.json` — Raw diagnostic data
- `discovery/engine-correlation.md` — Pairwise engine redundancy analysis (50 records)
- `discovery/confidence-analysis.md` — Per-engine confidence calibration assessment

## Key Decisions
1. Primary root cause is span duplication (+381 FPs/100 records), not aggregation algorithm
2. 5/6 engines have zero marginal value — regex catches everything they catch
3. 3/6 engines use fixed confidence scores — fusion is corrupted by uncalibrated inputs
4. Quick wins (span dedup + corroboration + calibration-aware weighting) should reach F1 >= 0.85
5. Dawid-Skene/HMM from research is lower priority than fixing the span merging problem

## Requirements Artifacts
- `requirements/requirements-document.md` — Complete requirements with traceability matrix
- `requirements/functional-requirements.md` — 18 FRs with acceptance criteria
- `requirements/non-functional-requirements.md` — 15 NFRs with measurement criteria

## Requirements Summary (33 total: 14 High, 5 Medium)
- FR-001..FR-008: Four-layer pipeline (fast-pass, NER, meta-learner, validation)
- FR-009..FR-010: Per-engine calibration and calibration-aware fusion
- FR-011..FR-013: Multi-dataset training pipeline (6 industry datasets)
- FR-014..FR-015: Evaluation with statistical significance and dominance verification
- FR-016..FR-018: Configuration, API, CLI
- NFR-001..NFR-003: Performance (<=200ms latency, >=10K rec/hr, <=5ms meta-learner)
- NFR-004..NFR-007: Accuracy (F1>=0.85, P>=0.80, R>=0.83, dominance)
- NFR-008..NFR-012: Training (reproducibility, time budget, portability)
- NFR-013..NFR-015: Compatibility and observability

## Key Findings (12 total: 3 pains, 4 gaps, 5 opportunities)
- P-1: Span duplication creates +381 FPs/100 records (Critical)
- P-2: 5/6 engines fully redundant (High)
- P-3: 3/6 engines use fixed confidence (High)
- O-1: Aggressive span merging could eliminate most FPs (Critical)
- O-2: Corroboration filter for semantic types (High)
- O-3: Calibration-aware weighting (High)
