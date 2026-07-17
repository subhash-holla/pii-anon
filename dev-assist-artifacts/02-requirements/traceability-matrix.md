# Traceability Matrix — pii-anon Requirements

> Chain: **PGO → UC → FR/NFR → (DC: Design) → (Story: Dev) → (Test: Testing)**. `provisional_status` = AGENT_SIMULATED for all rows (Stage 2; Pass-2 follow-up). `external_refs` carries cross-repo edges (`DATA:` = pii-anon-eval-data, `PAPER:` = pii-anon-research-paper) per the program traceability convention (`../../PROGRAM-MANIFEST.md`). DC/Story/Test columns: see **§FR/NFR → Story → Test → DC backfill** below (release-gate reconciliation, 2026-06-09 — the per-story fill every story gate deferred and no sprint gate closed); the DC definitions live in `../03-design/06-synthesis/D-implementation-ready-design.md` §D1.

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

## FR/NFR → Story → Test → DC backfill (release-gate reconciliation, 2026-06-09)

> Source of truth: the 30 **DONE** story files under `../04-development/02-stories/sprint-{1..7}/` (each declares `Implements` + an Evidence section naming its test files) + the SO-07..SO-23 sign-off ledger (`../_signoffs/`). Every cited test file verified present in `tests/` at reconciliation time. **HONESTY rule:** a row gets a story + pin ONLY where a DONE story claims the ID (or, where flagged, carries its substance under a different label); everything else is recorded `— (Pass-2 / unclaimed)` — no invented pins. Story states are all DONE; SO-x = the covering sign-off. Release-gate finding: **TRACE MAJOR-1 / TRACE-RC** (run `wf_3d7f1f9f-8d8`).

### Functional requirements

| ID | DC | Implementing story (state) | Test pin (file::test or file) | Notes |
|---|---|---|---|---|
| FR-001 | DC-12 | S6-04 (DONE, SO-20) | `tests/test_byo_pipeline.py` (A1–A13) | BYO-pipeline adapter contract; zero harness-core edits. |
| FR-002 | DC-12 | S6-04 (DONE, SO-20) | `tests/test_byo_pipeline.py` (A1–A13) | Identical scoring path for incumbents. |
| FR-003 | DC-06 | S3-01 (DONE 2026-05-31) + S3-02 + S3-03 (DONE, SO-08) | `tests/test_rating_engine_port.py`; `tests/test_bradley_terry_mle.py`; `tests/test_bayes_bt.py` | 3-tier ladder: glicko-legacy → MLE-BT → bayes-bt (claim-grade). |
| FR-004 | DC-07 | S3-04 (DONE, SO-09) | `tests/test_coherent_significance.py` (+ `tests/test_paired_set.py`) | Coherence by construction from one joint posterior. |
| FR-005 | DC-10 | S4-03 (DONE, SO-11) | `tests/test_selective_risk.py` | Per-class ECE/MCE/Brier/AURC + reliability + abstention. |
| FR-006 | DC-08 | S4-01 (DONE, SO-11) | `tests/test_deid_families.py` | Structurally distinct pseudonymization-integrity family. |
| FR-007 | DC-11 | S1-03 (DONE, SO-07) + S4-CS-01 (DONE, SO-09) | `tests/test_recall_floor_per_language_gate.py`; `tests/test_competitive_supremacy.py` | Per-language ε CI gate + composite ship/no-ship (SDO G1). |
| FR-008 | DC-11 | S4-CS-01 (DONE, SO-09) + S7-02 producer (DONE, SO-15) | `tests/test_canonical_run.py`; `tests/test_competitive_supremacy.py` | Refuses claim-grade without `canonical_claim_run is True` + non-blank provenance (G7). |
| FR-009 | DC-08 | S4-01 (DONE, SO-11) | `tests/test_deid_families.py` | 5-axis family (reversal/collision/referential/Art-4(5)). |
| FR-010 | DC-08 | S4-01 (DONE, SO-11; RRS separation S3-04; docs surfacing S7-05, SO-23) | `tests/test_deid_families.py` (no-merge CI guard) | AX-004 headline — no merge path/field. |
| FR-011 | DC-09 | S5-02 (DONE, SO-17; protocol S5-01, sandbox S5-04) | `tests/test_attack_reid_tier3.py` | REPRESENTATIVE in-tree Tier-3 adversary, de-circularized; real LLM-call path lazy/optional → Pass-2. |
| FR-012 | DC-09 | S5-02 (DONE, SO-17) | `tests/test_attack_reid_tier3.py` | Margin-commit + no-gold-link-consulted de-circularization invariant. |
| FR-013 | DC-09 | S5-03 (DONE, SO-18; protocol S5-01, sandbox S5-04) | `tests/test_attack_mia.py` | REPRESENTATIVE LiRA/Secret-Sharer; real ≥128-shadow training + DATA canary splits → Pass-2. |
| FR-014 | DC-09 | — (Pass-2 / unclaimed) | — | SHOULD. Key-compromise blast radius — DATA-owned (`DATA:reidentification.py`); no CODE story claimed it. |
| FR-015 | DC-09 | — (Pass-2 / unclaimed) | — | SHOULD. Residual-risk-vs-utility Pareto — DATA-owned (`ParetoFrontierAnalyzer`); never merged per AX-004. |
| FR-016 | DC-01 | S1-01 + S1-02 (DONE, SO-07) | `tests/test_shared_layer_projector.py`; `tests/test_floor_fusion_wiring.py` | Floor by construction, LIVE at the `build_fusion` seam. |
| FR-017 | DC-01/DC-11 | S1-03 (DONE, SO-07) — claimed under the FR-007 + NFR-011(ε) labels | `tests/test_recall_floor_per_language_gate.py` (`@requires_dataset`-gated) | Label-vs-substance: S1-03's Given/When/Then IS the FR-017 per-language `ensemble ≥ shared − ε` gate; recorded here so FR-017 is not orphaned. |
| FR-018 | DC-02 | S2-01 + S2-02 (DONE, SO-12) + S2-05 (DONE, SO-10) — **PARTIAL** | `tests/test_moe_router_seam.py`; `tests/test_distilled_gate.py`; `tests/test_gate_distillation.py`; `tests/test_moe_gate_verify_on_load.py` | Learned routing seam + distilled gate body + verify-on-load LIVE; rules-first early-exit / selective activation = S2-03 **BLOCKED** (orchestrator user-WIP) → Pass-2. |
| FR-019 | DC-04 | S6-03 (DONE, SO-10) | `tests/test_encrypted_store.py` | AEAD at rest + auditable key rotation + Art-4(5) separation. |
| FR-020 | DC-05 | — (Pass-2 / unclaimed) | — | SHOULD. Legal-regime mapping never storied this pass (baseline transform strategies pre-exist). |
| FR-021 | DC-05 | — (Pass-2 / unclaimed) | — | SHOULD. Orchestrate-incumbents-behind-one-floored-interface never storied this pass. |
| FR-022 | DC-05/DC-08 | — (Pass-2 / unclaimed) | — | SHOULD. Swarm-side distinct PI emission unclaimed; the eval-side distinct families are S4-01 (FR-006/009/010). |
| FR-023 | DC-13 | S6-01 (DONE, SO-19) | `tests/test_query_aware_masking.py` | Standalone pure primitive, default-to-mask; orchestrator router-pre-filter wire-in = `# SWITCH-POINT(ORCH)` Pass-2. |
| FR-024 | DC-13 | S6-01 (DONE, SO-19) — representative | `tests/test_query_aware_masking.py` | `score_query_aware_bound` vs mask-all baseline; the real DATA scorer = `# SWITCH-POINT(DATA)` Pass-2. |
| FR-025 | DC-13 | S6-02 (DONE, SO-14) | `tests/test_agentic_interception.py` | 4-channel least-privilege interception (AX-006). |
| FR-026 | DC-13 | S6-02 (DONE, SO-14) + S6-03 (DONE, SO-10) | `tests/test_agentic_interception.py`; `tests/test_encrypted_store.py` | No-raw-PII-persist; keyed HMAC-SHA256 surrogate (S6-02 iter-2 security fix). |
| FR-027 | DC-04/DC-13 | **Deferred (Pass-2; was bundled in S6-03 scope, descoped at release-gate reconciliation)** | — (no test pin exists) | TRACE-RC correction: FR-027 moved OUT of S6-03's Implements — `tests/test_encrypted_store.py` carries no session-pseudonym pin. |
| FR-028 | DC-13 | S6-05 (DONE, SO-14) | `tests/test_agentic_leakage_sankey.py` | 4-source/6-node Sankey per binding DC-13; the FR-028 "6 source channels" wording = tracked spec-reconciliation (PO / Pass-2). |
| FR-029 | DC-13 | S6-05 (DONE, SO-14) | `tests/test_agentic_leakage_sankey.py` | Injection ASR vs benign-task-success. |
| FR-030 | DC-13 | — (Pass-2 / unclaimed — not in any story's Implements) | representative: `tests/test_agentic_interception.py::test_ax002_a9_deterministic_replay` | SHOULD. S6-02's fixed-`surrogate_key` path + S7-04's seed-derived key give byte-identical replay evidence; the formal FR-030 claim is unowned → Pass-2. |
| FR-031 | DC-14 | S7-01 (DONE, SO-21) | `tests/test_native_readers.py` | Uniform `Iterator[IngestRecord]` across PDF/image/screenshot/DICOM/audio. |
| FR-032 | DC-14 | S7-01 (DONE, SO-21) — partial-honest | `tests/test_native_readers.py` | Text-format round-trip REAL + regression-pinned; native readers honestly report `supports_reconstruction=False` → Pass-2. |
| FR-033 | DC-14 | S7-01 (DONE, SO-21) — representative | `tests/test_native_readers.py` | Page-granular `source_coords` + in-range offset assertion. |
| FR-034 | DC-14 | S7-01 (DONE, SO-21) — representative | `tests/test_native_readers.py` | In-tree per-modality harness; corpus scale = DATA Pass-2. |
| FR-035 | DC-14 | S7-01 (DONE, SO-21) | `tests/test_native_readers.py` | Teeth-proven pytest regression gate on reader recall. |
| FR-036 | DC-14 | — (no owning story; upheld by an S7-01 guard) | `tests/test_native_readers.py::test_a12_stream_vs_batch_parity_and_text_round_trip` | MUST. Representative stream/batch parity guard only; full stream/batch/offline parity claim = Pass-2. |
| FR-037 | DC-14 | — (Pass-2 / unclaimed) | — | SHOULD. OS-matrix portability — CI-matrix concern, never storied. |
| FR-038 | DC-15 | S7-03 (DONE, SO-22) | `tests/test_multilingual_fairness.py` | Non-EN context feature active in detection. |
| FR-039 | DC-15 | S7-03 (DONE, SO-22) | `tests/test_multilingual_fairness.py` | Per-language fairness gap bounded + gated (fail-closed power semantics). |
| FR-040 | DC-01 | sp3-v220-rebaseline enhancement (DONE, 2026-07-10) | `tests/test_coverage_tranche_sp3.py` | GDPR Art-9 special-category detection (label-gated + intrinsic gene/rs-ID); census 63→66; 100% recall / 0 FP on train-en gold. Additive, leak-safe, no generator-filler anchor. |

### Non-functional requirements

| ID | DC | Implementing story (state) | Test pin (file::test or file) | Notes |
|---|---|---|---|---|
| NFR-001 | DC-06 | S3-03 (DONE, SO-08) | `tests/test_convergence_gate.py` (+ `tests/test_bayes_bt.py`) | Hard convergence gate (R̂/ESS/divergences) refuses claim-grade emission; numpyro/jax tiers env-gated in CI; real-corpus `PairedComparisonSet` ⛓ eval-data. |
| NFR-002 | DC-07 | S3-04 (DONE, SO-09) | `tests/test_coherent_significance.py` | BY CONSTRUCTION from one joint posterior. |
| NFR-003 | DC-06 | S3-02 (DONE, SO-08) | `tests/test_bradley_terry_mle.py::test_nfr_003_*` | Empirical-coverage study at corpus scale ⛓ eval-data S5 (gate finding TRACE-S304-MINOR-1). |
| NFR-004 | — | — (Pass-2 / unclaimed — DATA-owned power ladder) | — | Tier thresholds CONSUMED in-tree (S7-03 powered-group long-tail floor 200); corpus-scale slice counts are DATA-owned. |
| NFR-005 | DC-11 | S7-02 run-level (DONE, SO-15); component determinism S2-01/S2-02 (SO-12) + S2-05 (SO-10) | `tests/test_canonical_run.py` (byte-identical modulo timestamp); `tests/test_moe_gate_signing.py` | Wall-clock latency is the SANCTIONED non-reproducible field (per S7-04). |
| NFR-006 | DC-11 | S7-02 (DONE, SO-15) + S4-CS-01 (DONE, SO-09) + S2-05 gate envelope (SO-10) | `tests/test_canonical_run.py`; `tests/test_competitive_supremacy.py` | Provenance stamp + G7 strict-`is True` / non-blank-string hardening (keystone closes). |
| NFR-007 | DC-02 | — (measurement Pass-2; ceiling literal committed by S7-04, SO-16) | `tests/test_latency_ceilings.py::test_speed_p50_is_the_nfr007_literal` | The committed registry pins the literal; the shared/regex-path p50 MEASUREMENT is full-census Pass-2. |
| NFR-008 | DC-02 | — (Pass-2 / blocked) | — | Early-exit chunk latency depends on the rules-first Depth-1 early-exit = S2-03 **BLOCKED** (orchestrator user-WIP). |
| NFR-009 | DC-03/DC-11 | S7-04 (DONE, SO-16; selection-bias mechanism S2-04, SO-13) | `tests/test_latency_ceilings.py` | Committed p50/p95/p99 ceilings (`latency_ceilings.py`) + gate `_g5_audit_latency` + producer measured-latency LIVE; full-census latency = Pass-2. |
| NFR-010 | DC-03 | S2-04 mechanism (DONE, SO-13) — measurement Pass-2 | `tests/test_moe_sla_bias.py` | Lightweight-path selection bias in-tree; the ≥5,000 rec/s 8-core measurement not run in-tree. |
| NFR-011 | DC-01/DC-11 | S1-01/S1-02 by construction + S1-03 ε-gate + S1-04/S1-05 (all DONE, SO-07) | `tests/test_shared_layer_projector.py` (property, 2,000 cases / 0 violations); `tests/test_recall_floor_per_language_gate.py` | BOTH halves landed (superset-by-construction + per-language ε ≤ 0.005). |
| NFR-012 | DC-09 | S5-02 (DONE, SO-17) — representative | `tests/test_attack_reid_tier3.py` | Wilson CIs + 2-rung power ladder (≥385 REID_LOW / ≥897) in-tree; real paired-persona cohort ⛓ eval-data S6 → Pass-2. |
| NFR-013 | DC-09 | S5-03 (DONE, SO-18) — representative | `tests/test_attack_mia.py` | TPR@FPR∈{1e-3,1e-2} + Secret-Sharer exposure math; real ≥128-shadow LiRA training → Pass-2. |
| NFR-014 | DC-04/DC-08 | S4-01 (DONE, SO-11) + S6-03 (DONE, SO-10) | `tests/test_deid_families.py`; `tests/test_encrypted_store.py` | Unauthorized-reversal = 0; referential integrity = 100%. |
| NFR-015 | DC-04/DC-08 | S4-01 (SO-11) + S6-03 (SO-10) + S2-05 (SO-10) | `tests/test_deid_families.py`; `tests/test_encrypted_store.py`; `tests/test_moe_gate_signing.py` | Key/state separation across scorer + AEAD store + signed gate envelope; artifact-alone re-join FAILs. |
| NFR-016 | DC-09 | S5-01 (DONE, SO-14) | `tests/test_attack_reid_protocol.py` | Non-strippable anti-anonymity caveat; carried on every S5-02/S5-03 emitted report. |
| NFR-017 | DC-10 | S4-03 (DONE, SO-11) | `tests/test_selective_risk.py` | Post-temp-scaling ECE bars + the G4 tighten-only artifact-threshold clamp. |
| NFR-018 | DC-10 | S4-03 (DONE, SO-11) | `tests/test_selective_risk.py` | Brier + decomposition per powered class (COULD). |
| NFR-019 | DC-10 | S4-03 (DONE, SO-11) | `tests/test_selective_risk.py` | AURC + monotone risk-coverage curve (NaN-row fabrication closed at keystone close-7). |
| NFR-020 | DC-10 | S4-03 (DONE, SO-11) | `tests/test_selective_risk.py` | Calibrated-confidence coverage — the lone EXACT-100% MUST; bare-logit teeth. |
| NFR-021 | DC-10 | S4-03 (DONE, SO-11) | `tests/test_selective_risk.py` | ≥3-point abstention operating-point table at {1%, 2%, 5%}. |
| NFR-022 | DC-14 | — (Pass-2 / unclaimed) | — | OS-matrix CI never storied; the Stage-5 snapshot's successor pointer (S7-02) was superseded — the landed S7-02 is the canonical-run producer. |
| NFR-023 | DC-14 | — (representative guard only, via S7-01) | `tests/test_native_readers.py::test_a12_stream_vs_batch_parity_and_text_round_trip` | Stream/batch leg guarded; offline leg + the full divergence=0 claim = Pass-2. |
| NFR-024 | DC-15 | Cross-cutting (every story gate runs security-sast; sandbox S5-04, SO-10) | `tests/test_attack_sandbox.py` (no-egress/no-persist) + per-story security-sast gate YAMLs | AX-001; synthetic-shaped fixtures only; the FULL-repo scan remains the Pass-2 `S-sec` item. |
| NFR-025 | DC-15 | S7-03 (DONE, SO-22) | `tests/test_multilingual_fairness.py` | Worst-group recall gap ≤ 0.10 across POWERED groups; fail-closed power semantics. |
| NFR-026 | DC-15 | S2-02 (SO-12) + S3-01 (DONE 2026-05-31) + S7-01 (SO-21) | `tests/test_distilled_gate.py` (absent/unverifiable-gate fallback); `tests/test_rating_engine_port.py` (registry degrade); `tests/test_native_readers.py` (loud-not-silent optional deps) | Graceful degradation verified at the gate, rating, and reader seams. |

**Backfill tally:** 39 FR + 26 NFR = 65 rows. 30 FR rows + 20 NFR rows carry a DONE story + verified in-tree test pin (FR-017 via the S1-03 label-vs-substance note; FR-018 PARTIAL on the S2-03 block); 1 FR explicitly **Deferred-descoped** (FR-027); 8 FR (FR-014/015/020/021/022/030/036/037 — FR-030/036 with representative evidence cited) + 5 NFR (NFR-004/007/008/022/023 — NFR-007/023 with representative pins cited) + NFR-024 cross-cutting remain unclaimed / Pass-2 / partial-evidence. No pin fabricated; every cited file exists in `tests/` (spot-verified at reconciliation).

## Status Change Log

| Date | Action | Authority |
|---|---|---|
| 2026-06-09 | **Backfill applied:** appended §"FR/NFR → Story → Test → DC backfill" filling the Story/Test/DC columns for all 39 FRs + 26 NFRs from the 30 DONE story files + SO-07..SO-23 sign-offs — the column-fill every story gate deferred-to-sprint-gate and no sprint gate ever closed; header line 3's stale "DC/Story/Test columns fill in Stages 3–5" claim replaced with a pointer to the new section. | Release-gate finding **TRACE MAJOR-1 / TRACE-RC** (run `wf_3d7f1f9f-8d8`) |
| 2026-06-09 | **FR-027 descoped:** FR-027 (stable session pseudonyms) moved OUT of S6-03's Implements — no test pin exists in `tests/test_encrypted_store.py`; recorded as *Deferred (Pass-2; was bundled in S6-03 scope, descoped at release-gate reconciliation)*. | TRACE-RC (run `wf_3d7f1f9f-8d8`) |
| 2026-06-09 | **NFR-matrix supersession noted:** `../05-testing/03-nfr-verification/nfr-verification-matrix.md` (Stage-5 foundation snapshot, 2026-05-30) bannered as superseded; per-NFR rows whose successor stories landed flipped to VERIFIED-in-tree with test pins; genuinely-deferred (DATA-owned / corpus-scale / blocked) rows left DEFERRED. | TRACE-RC (run `wf_3d7f1f9f-8d8`) |
