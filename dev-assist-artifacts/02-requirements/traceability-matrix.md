# Traceability Matrix — pii-anon Requirements

> Chain: **PGO → UC → FR/NFR → (DC: Design) → (Story: Dev) → (Test: Testing)**. `provisional_status` = AGENT_SIMULATED for all rows (Stage 2; Pass-2 follow-up). `external_refs` carries cross-repo edges (`DATA:` = pii-anon-eval-data, `PAPER:` = pii-anon-research-paper) per the program traceability convention (`../../PROGRAM-MANIFEST.md`). DC/Story/Test columns fill in Stages 3–5.

## PGOs (Persona-Goal-Outcome triples; R0 bridge)
| PGO | Persona | Goal | Outcome | UCs |
|---|---|---|---|---|
| PGO-1 | P-07 evaluator | benchmark any pipeline credibly | reproducible BYO scorecard w/ CIs | UC-01, UC-06 |
| PGO-2 | P-04 researcher | publishable, defensible comparison | convergent ratings + coherent significance + calibration | UC-02, UC-03, UC-04 |
| PGO-3 | P-03/P-04 | prove + certify privacy end-state | pseudonymization-integrity + anon-vs-pseudo distinct + Tier-3 re-id | UC-05, UC-07–12 |
| PGO-4 | P-01/P-02 | detect+transform without breaking utility | recall-floor-guaranteed swarm + reversible pseudonyms | UC-13–18 |
| PGO-5 | P-05/P-02 | guard live agents | multi-channel masking + leakage measurement | UC-19–23 |
| PGO-6 | P-01/P-02 | run anywhere on any data | multimodal readers + stream/batch/offline + OS parity | UC-24–28 |
| PGO-7 | P-06 | trust + extend the library | orchestration + reproducible claims + contribution path | UC-18, UC-01 |

**Orphan scan:** 0 orphan UCs (all 28 → a PGO); 0 orphan FRs/NFRs (all → a UC; see requirements-document.md). ✅

## UC → FR/NFR → cross-repo edges (sparse boundary-crossing edges)
| UC | FRs | NFRs | external_refs |
|---|---|---|---|
| UC-01 BYO scorecard | FR-001, FR-002 | NFR-003 | `DATA:scorer-IO-contract + reference-adapters` ; `PAPER:P1 resource` |
| UC-02 Bradley-Terry | FR-003 | NFR-001 | `DATA:stats/bradley_terry.py (S6)` ; `PAPER:P1 §ratings (headline)` |
| UC-03 coherent significance | FR-004 | NFR-002, NFR-003 | `DATA:paired-bootstrap over per_record_f1` ; `PAPER:P1 §methodology` |
| UC-04 calibration | FR-005 | NFR-017, NFR-018, NFR-019, NFR-021 | `DATA:stats/calibration.py (S4 done)` |
| UC-05/07/08 pseudonymization-integrity + distinct | FR-006, FR-009, FR-010 | NFR-014, NFR-015, NFR-016 | `DATA:scoring/pseudonymization.py + anonymization.py (done)` ; `PAPER:P2 headline novelty` |
| UC-06 CI-gate + canonical-run | FR-007, FR-008 | NFR-006, NFR-004 | `DATA:per-language recall inputs + power ladder` |
| UC-09 Tier-3 LLM-adversary | FR-011, FR-012 | NFR-012, NFR-016 | `DATA:scoring/adversary/* + assemble_paired_set (S6 — blocking dep)` ; `PAPER:P1 §5 Tier-3` |
| UC-10 MIA | FR-013 | NFR-013 | `DATA:canary/membership splits (S6)` ; `PAPER:P1 §Tier-2` |
| UC-11/12 key-compromise + Pareto | FR-014, FR-015 | NFR-015 | `DATA:reidentification.py + ParetoFrontierAnalyzer` |
| UC-13/14 recall-floor | FR-016, FR-017 | NFR-011 | `AX-003` ; `DATA:detection scorer per-language` |
| UC-15 MoE-router | FR-018 | NFR-007, NFR-008, NFR-009, NFR-010 | — (CODE-local; swarm prior-art `03-design/_inputs/`) |
| UC-16/17/18 reversible pseudo + transforms + orchestration | FR-019, FR-020, FR-021, FR-022 | NFR-014, NFR-015 | `DATA:pseudonymization integrity scorer` |
| UC-19 query-aware masking | FR-023, FR-024 | — | `DATA:query-aware scorer (FR-014 eval-data, S5)` |
| UC-20/21 interception + session pseudonyms | FR-025, FR-026, FR-027, FR-030 | NFR-020 | `AX-006` |
| UC-22/23 agentic leakage + injection | FR-028, FR-029 | — | `DATA:agentic-oracle UC-08/FR-017-020 (S7)` ; `PAPER:agentic eval` |
| UC-24 multimodal readers | FR-031, FR-032, FR-033 | NFR-022, NFR-026 | — (CODE `ingestion/`) |
| UC-25 per-modality benchmark | FR-034, FR-035 | NFR-011 | `DATA:multimodal slices (future; WebPII deferred)` |
| UC-26/27 parity + portability | FR-036, FR-037 | NFR-022, NFR-023 | — |
| UC-28 multilingual | FR-038, FR-039 | NFR-025 | `DATA:multilingual slices + fairness` |
| (cross-cutting) | — | NFR-005, NFR-024 | `AX-001` (no-real-PII) |

## Cross-repo edge summary (program trace index — feeds PROGRAM-MANIFEST)
~14 boundary-crossing threads: **DATA-blocking** = `assemble_paired_set` (S6, for FR-011/Tier-3) + canary splits (S6, FR-013) + query-aware scorer (S5, FR-023) + agentic oracle (S7, FR-028/029). **DATA-reuse (exists)** = stats/calibration, pseudonymization/anonymization scorers, power ladder. **DATA-to-build** = `stats/bradley_terry.py` (S6). **PAPER-substantiates** = FR-003/004/008/009/011/013 → Paper 1; FR-006/009/010 + the benchmark → Paper 2.

## R9 — Verification-criteria strengthening (applied)
Every FR carries a boolean-testable Given/When/Then + a negative test; every NFR carries a quantified threshold + measurement method (see requirements-document.md). Strengthened during authoring: FR-008 canonical-run (added "unstamped number not emittable"); FR-010 no-merge (added "schema validation fails on merged field"); NFR-011 recall-floor (split property + ε gate per R10).
