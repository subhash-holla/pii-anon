# doc-source-index — pii-anon Stage 6 Documentation (D1 Harvest)

> **Wave D1 output.** Compiled 2026-06-10. Read-only compilation of all prior-stage artifacts.
> Every downstream documentation agent (D2 architect, D3 author, D4 verifier, D5 synthesis)
> MUST read this index rather than re-walking the artifact tree.

---

## Source Signal vs Gaps

**Mode: BROWNFIELD — all stages have canonical authored artifacts.** This is a mature v1.4.0
library that completed a full PDLC pass (Discovery → Requirements → Design → Development →
Testing) under the `pdlc/sota-program` branch. Stage 6 arrives with a rich, multi-sprint
development corpus (S1–S7, ~30 stories), 23 sign-offs, a live user-docs tree, and a
detailed SDO verdict.

| Signal category | Count / Status |
|---|---|
| Prior-stage canonical artifacts | 6 stages all present (brownfield + discovery + requirements + design + development + testing) |
| Sign-offs (SO-01..SO-23) | 23 — chronological narrative spine is COMPLETE |
| Story files (S1–S7, all sprints) | 30 story files; 25 unique DONE evidenced stories (sprints 1–7) |
| Review gate YAMLs | 130+ review YAML files across all stories |
| User-docs tree (docs/) | 20 .md files; 18 authored docs + 2 user-WIP/generated (see §User docs) |
| Code surface (src/pii_anon/) | ~90 Python files; 14 headline new modules from the SOTA program |
| doc-seed files (all 5 stages) | ALL ABSENT (0/5) — see OBSERVATION O-1 |
| examples-and-tests-catalog.md | ABSENT — see OBSERVATION O-2 |
| 00-validation/ | EMPTY (only .gitkeep) — see OBSERVATION O-3 |
| developer-assistant.yaml documentation: block | NOT PRESENT (no explicit deliverables list) — see OBSERVATION O-4 |

**Inferred-vs-explicit ratio:** The program's SDO completion-criterion and the full MUST/SHOULD
requirement set are explicit and fully traced. All 23 SO sign-offs carry explicit scope lines.
The user-docs tree is live and real. What is *inferred* (not authored as doc-seed prose):
the plain-language stage narrative that would have appeared in the five absent doc-seeds. The
MANIFEST.md `### S*-DONE` sections and the sign-off YAML `scope:` fields are the de-facto
authored narrative spine and partially compensate.

---

## 1. Canonical Artifacts Present

| Stage | File | Last-section anchor | ID families carried |
|---|---|---|---|
| 00-Brownfield | `/dev-assist-artifacts/00-brownfield-assessment/assessment-2026-05-30.md` | `§2 Per-Stage Signal Extraction`, `§3 Aggregate per-stage rating`, `§4 Findings` | (no FR/NFR/DC/UC — pre-requirements; 12 MAJORs + 11 MINORs + 8 OBSERVATIONs) |
| 00-Brownfield | `/dev-assist-artifacts/00-brownfield-assessment/artifact-inventory.md` | inventory table | (none) |
| 00-Brownfield | `/dev-assist-artifacts/00-brownfield-assessment/migration-log.md` | migration table | (none) |
| 00-Axioms | `/dev-assist-artifacts/00-axioms/project-axioms.yaml` | AX-001..AX-006 | AX-001..006 |
| 01-Discovery | `/dev-assist-artifacts/01-discovery/discovery-report.md` | `§1 Refined POV`, `§2 Why now`, `§3 Personas`, `§4 Market`, `§5 Use cases`, `§6 Concept value`, `§7 Top open items` | UC-01..28 (implicit — the UC set is enumerated in `04-use-cases.md`) |
| 01-Discovery | `/dev-assist-artifacts/01-discovery/04-use-cases.md` | 28 UC rows | UC-01..UC-28 |
| 01-Discovery | `/dev-assist-artifacts/01-discovery/personas.md` | P-01..P-07 | (personas, not FR/NFR) |
| 02-Requirements | `/dev-assist-artifacts/02-requirements/requirements-document.md` | `##FR table`, `##NFR table`, `§R7 MoSCoW`, `§R10 NFR threshold validation` | FR-001..039, NFR-001..026 (all 65 requirements; see §3 Requirement inventory) |
| 02-Requirements | `/dev-assist-artifacts/02-requirements/traceability-matrix.md` | `##PGOs`, `##UC→FR/NFR cross-repo edges`, `##R9 strengthening`, `##cross-repo edge summary` | FR-001..039, NFR-001..026, UC-01..28, PGO-1..7 |
| 03-Design | `/dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md` | `§D0 baseline`, `§D1 Design Cases (15 DCs)`, `§D2/D3 Workflow/UI`, `§D4/D5 3 headline decisions`, `§Multimodal DC-14`, `§SME findings`, `§Switch-points`, `§Cross-repo`, `§Verification` | DC-01..DC-15, FR-001..039 (partial), NFR-001..026 (partial) |
| 03-Design | `/dev-assist-artifacts/03-design/moe-architecture-and-guarantee.md` | MoE guarantee doc | DC-02, DC-03 |
| 04-Development | `/dev-assist-artifacts/04-development/development-log.md` | `§W1 Preflight`, `§W2 Planning`, `§W3 Quality`, `§W4 Testing`, `§W5 Stories`, `§W6 Execution` | FR/NFR referenced throughout |
| 04-Development | `/dev-assist-artifacts/MANIFEST.md` | `§Stage Status`, `§Discovery..Testing Phase Progress`, `§Sign-offs`, `§Handoff Signals (S*-DONE sections)`, `§Pivots Log`, `§Agent Deployment Ledger` | FR-001..039, NFR-001..026 (cited in DONE sections), SO-01..23 |
| 04-Development | `/dev-assist-artifacts/PDLC-JOURNEY.md` | `§Traceability spine`, `§Per-stage summary`, `§What shipped`, `§What's deferred`, `§Defensibility` | FR-016, NFR-011 (journey doc, focused on the first pass) |
| 05-Testing | `/dev-assist-artifacts/05-testing/release-readiness-report.md` | `##Verdict`, `##Evidence`, `##Caveats`, `##Pass-2 commitments`, `##Recommendation`, `##End-of-PDLC handoff` | FR-016, NFR-011, AX-003 |
| 05-Testing | `/dev-assist-artifacts/05-testing/03-nfr-verification/nfr-verification-matrix.md` | `##Pyramid`, `##Cross-cutting discipline`, `##Gaps` | NFR-001..026 (2 VERIFIED, 2 PARTIAL, 22 DEFERRED, 0 FAIL) |
| 05-Testing | `/dev-assist-artifacts/05-testing/02-architecture/test-architecture.md` | `##Pyramid`, `##Cross-cutting discipline`, `##Gaps` | (none explicit) |
| 05-Testing | `/dev-assist-artifacts/05-testing/_diagnostics/f2-gap-attribution.md` | F2 gap attribution | G6, NFR-011 |
| 05-Testing | `/dev-assist-artifacts/05-testing/benchmark-evidence/legacy-benchmark-evidence.md` | legacy benchmark evidence | (none) |

**Note on diamond D-decision.md files:** No `D-decision.md` files are present in
`dev-assist-artifacts/03-design/*/`. The design diamond's decisions are consolidated in
`D-implementation-ready-design.md:§D4/D5` (the three headline decisions as `###DECISION 1/2/3`
sections) and the `§D1 Design Cases` table. There are NO separate ADR diamond files; the
decision inventory (§4 below) is sourced entirely from the design synthesis doc.

---

## 2. Sign-off Ledger

All 23 sign-offs present at `/dev-assist-artifacts/_signoffs/`. Each is a YAML file.

| SO ID | File | Scope (one line) |
|---|---|---|
| SO-01 | `SO-01-m1.yaml` | Milestone M1 — brownfield assessment + migration DONE; 12 MAJORs carried forward; 6 axioms; eval-integrity steering decision. Stage transition: `00 → 01-Discovery` |
| SO-02 | `SO-02-discovery.yaml` | Discovery stage DONE — POV pivot (measurement-first); 7 personas; 28 UCs; SME panel MAJORs → Requirements. Stage transition: `01 → 02-Requirements` |
| SO-03 | `SO-03-requirements.yaml` | Requirements stage DONE — 39 FR + 26 NFR; traceability + cross-repo external_refs; R10 0-DIVERGED. Stage transition: `02 → 03-Design` |
| SO-04 | `SO-04-design.yaml` | Design stage DONE — 15 DCs; 3 Pugh decisions; SME heuristic eval (1 CATASTROPHIC resolved). Stage transition: `03 → 04-Development` |
| SO-05 | `SO-05-development.yaml` | Development stage started — W1–W5 planning complete; Sprint-1 in progress. Stage transition: `04 → partial-exec` |
| SO-06 | `SO-06-testing.yaml` | Testing stage DONE — release-readiness verdict: SHIP-WITH-CAVEATS (foundation) / DEFER (full redesign). Stage transition: `05 → next` |
| SO-07 | `SO-07-sprint1.yaml` | Sprint-1 COMPLETE — S1-01..05 DONE; recall-floor LIVE; sprint gate APPROVE. |
| SO-08 | `SO-08-s3-eval-integrity.yaml` | S3 eval-rating ladder COMPLETE (S3-01 RatingEnginePort + S3-02 MLE-BT + S3-03 bayes-bt + S3-04 coherent significance); SDO gate LIVE via S4-CS-01. |
| SO-09 | `SO-09-sdo-gate-live.yaml` | Eval-integrity → SOTA-Dominance-Objective arc CLOSED; S4-CS-01 CompetitiveSupremacyGate LIVE; G1/G3/G6/G7 + verdict machine + J-fallback. |
| SO-10 | `SO-10-security-musts.yaml` | Three D6 SME Security-MUST stories CLOSED + adversarially hardened: S2-05 (sign+verify gate_v1.json), S6-03 (encrypt token store AEAD), S5-04 (sandbox attacks/). |
| SO-11 | `SO-11-g2-g4-guarantees.yaml` | G2 (pseudonymization integrity) + G4 (calibration selective-risk) SDO guarantees code-computable + fabrication-hardened; S4-01 + S4-03 DONE. |
| SO-12 | `SO-12-s2-moe-router-core.yaml` | S2 MoE-router CORE LIVE — S2-01 widened MoERouter.route() + v2 fusion-construction seam; S2-02 DistilledTopKGate runtime + offline trainer; control-path adversarial close RECLOSE_PASS (0 upheld). |
| SO-13 | `SO-13-s2-04-sla-bias.yaml` | S2-04 (aux-loss-free SLA selection-bias, DC-03, SHOULD) LIVE; S2 work-stream CLOSED; Phase B (S5 attacks + S6 agentic) begins. |
| SO-14 | `SO-14-phase-b-core-g5-audit.yaml` | Phase-B core — three SDO G5 audit inputs LIVE: S6-02 (4-channel least-privilege interception), S5-01 (ReidAttack/MiaAttack seam), S6-05 (leakage-Sankey + injection resistance). |
| SO-15 | `SO-15-keystone-close.yaml` | S7-02 canonical-run keystone DONE — round-8 adversarial close RECLOSE_PASS (0 upheld / 517 probes); honest verdict NOT_YET / G6 FAIL; G1/G2/G3/G4/G7 all PASS. |
| SO-16 | `SO-16-s7-04-latency-g5.yaml` | S7-04 DONE — G5 (LAST placeholder guarantee) computed; latency-ceilings registry; 2-round adversarial close CLOSE_PASS (0 upheld / 764 probes); all G1–G7 computed, G6 binds. |
| SO-17 | `SO-17-s5-02-tier3-adversary.yaml` | S5-02 DONE — Tier-3 representative re-identification adversary + NFR-012 RRS statistical-power model; 5/5 story gate. |
| SO-18 | `SO-18-s5-03-mia-representative.yaml` | S5-03 DONE — representative membership-inference adversary (LiRA-shape) + Secret-Sharer + TPR@low-FPR; 5/5 story gate. |
| SO-19 | `SO-19-s6-01-query-aware-masking.yaml` | S6-01 DONE — query-aware masking gate (FR-023/UC-19) + representative FR-024 bound; default-to-mask subtractive policy; 5/5 story gate. |
| SO-20 | `SO-20-s6-04-byo-pipeline-sdk.yaml` | S6-04 DONE — BYO-pipeline SDK adapter + identical-incumbent scoring (FR-001 MUST / FR-002 SHOULD); 5/5 story gate. |
| SO-21 | `SO-21-s7-01-native-readers.yaml` | S7-01 DONE — native-format readers (DC-14/UC-24/25): PDF/image/screenshot/DICOM/audio behind Iterator[IngestRecord]; iter-1 REQUEST_CHANGES (zip-bomb fix) → iter-2 5/5 APPROVE. |
| SO-22 | `SO-22-s7-03-multilingual-fairness.yaml` | S7-03 DONE — multilingual context activation + powered worst-group fairness gate (DC-15/UC-28); FR-038/039; 5/5 story gate; feature surface COMPLETE. |
| SO-23 | `SO-23-s7-05-docs-discoverability.yaml` | S7-05 DONE [DOCS MUST] — docs discoverability; D6 SME Docs MAJORs closed; ALL FEATURE STORIES COMPLETE; docs teeth-proven via test_docs_discoverability.py. |

---

## 3. Requirement Inventory

All 39 FRs and 26 NFRs. Source: `/dev-assist-artifacts/02-requirements/requirements-document.md`.
MUST qualifier = appears in §R7 MUST list. File counts = number of dev-assist-artifacts/ files
mentioning each ID (indicates downstream trace density).

### 3a. Functional Requirements

| FR | Title (abbreviated) | MoSCoW | UC | Pillar | File-count |
|---|---|---|---|---|---|
| FR-001 | BYO-pipeline adapter contract | MUST | UC-01 | eval | 25 |
| FR-002 | Identical scoring path for incumbents | SHOULD | UC-01 | eval | 17 |
| FR-003 | Bayesian Bradley-Terry rating engine | MUST | UC-02 | eval | 16 |
| FR-004 | Coherent significance | MUST | UC-03 | eval | 27 |
| FR-005 | Calibration & selective-risk scorecard | SHOULD | UC-04 | eval | 45 |
| FR-006 | Pseudonymization integrity distinct family | MUST | UC-05 | eval | 28 |
| FR-007 | CI ship/no-ship gate + per-language recall floor | MUST | UC-06 | eval | 17 |
| FR-008 | Canonical-run / provenance gate | MUST | UC-06 | eval | 14 |
| FR-009 | Pseudonymization integrity 5-axis family | MUST | UC-07 | eval | 31 |
| FR-010 | Enforce anon vs pseudo as distinct families | MUST | UC-08 | eval | 27 |
| FR-011 | Real Tier-3 LLM-adversary re-id | MUST | UC-09 | eval | 38 |
| FR-012 | Control Tier-3 circularity | SHOULD | UC-09 | eval | 23 |
| FR-013 | Full-power MIA (LiRA@128 + Secret-Sharer) | MUST | UC-10 | eval | 24 |
| FR-014 | Key-compromise blast radius + rotation resilience | SHOULD | UC-11 | eval | 16 |
| FR-015 | Anonymization residual-risk-vs-utility Pareto | SHOULD | UC-12 | eval | 21 |
| FR-016 | Recall-floor by construction (ensemble ⊇ shared) | MUST | UC-13 | swarm | 44 |
| FR-017 | Per-language recall-floor CI gate | MUST | UC-14 | swarm | 13 |
| FR-018 | MoE-router: learned routing + span-dedup + selective activation + early-exit | MUST | UC-15 | swarm | 28 |
| FR-019 | Reversible pseudonymization + auditable key rotation | MUST | UC-16 | swarm | 24 |
| FR-020 | Six transform strategies with legal-regime mapping | SHOULD | UC-17 | swarm | 13 |
| FR-021 | Orchestrate incumbent detectors behind recall-floored interface | SHOULD | UC-18 | swarm | 9 |
| FR-022 | Pseudonymization-integrity emitted distinct from anonymization | SHOULD | UC-16/17 | both | 9 |
| FR-023 | Query-aware masking gate | SHOULD | UC-19 | swarm | 16 |
| FR-024 | Bound query-aware over-redaction + false-retention | SHOULD | UC-19 | eval | 26 |
| FR-025 | Intercept all four agent channels, least-privilege | MUST | UC-20 | swarm | 26 |
| FR-026 | Persist no raw PII to any channel after masking (AX-006) | MUST | UC-20 | swarm | 69 |
| FR-027 | Stable session pseudonyms, authorized-only reversal | SHOULD | UC-21 | swarm | 14 |
| FR-028 | Per-channel agentic leakage counts (leakage-Sankey) | MUST | UC-22 | eval | 18 |
| FR-029 | Prompt-injection exfiltration resistance | MUST | UC-23 | eval | 9 |
| FR-030 | Byte-identical agentic masking decisions given seed/key/scope | SHOULD | UC-20/21 | swarm | 11 |
| FR-031 | Native-format readers emit Iterator[IngestRecord] | MUST | UC-24 | swarm | 10 |
| FR-032 | Round-trip reconstruction preserves non-PII payload | MUST | UC-24 | swarm | 9 |
| FR-033 | Extraction-fidelity assertion per modality | SHOULD | UC-24 | swarm | 6 |
| FR-034 | Per-modality recall benchmark, scored separately | MUST | UC-25 | eval | 8 |
| FR-035 | CI gate on multimodal reader recall regression | MUST | UC-25 | eval | 6 |
| FR-036 | Identical scrub decisions across stream/batch/offline | MUST | UC-26 | swarm | 5 |
| FR-037 | OS-matrix portability | SHOULD | UC-27 | swarm | 7 |
| FR-038 | Multilingual non-EN context feature active | SHOULD | UC-28 | swarm | 11 |
| FR-039 | Per-language fairness gap bounded + gated | SHOULD | UC-28 | both | 7 |

**Candidate documentation orphans (low downstream trace density):** FR-033 (6 files, no
standalone design section — in DC-14 table only), FR-035 (6), FR-036 (5), FR-037 (7).
All four trace to DC-14 in the design table. The thinness is due to the S7-01 story carrying
all four together. These are DOCUMENTED but may deserve explicit callouts in the
API-reference and release notes.

### 3b. Non-Functional Requirements

| NFR | Title (abbreviated) | MoSCoW | Threshold | Pillar | File-count |
|---|---|---|---|---|---|
| NFR-001 | Bradley-Terry MCMC convergence | MUST | split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param; 0 divergences | eval | 19 |
| NFR-002 | Significance coherence | MUST | ZERO incoherent deltas | eval | 11 |
| NFR-003 | Bootstrap CI empirical coverage | SHOULD | 93–97% over ≥1,000 sim replicates | eval | 8 |
| NFR-004 | Statistical-power consumption (risk-tiered) | MUST | positives/slice ≥ 1,522 / 753 / 200 | both | 19 |
| NFR-005 | Scoring-run determinism | MUST | byte-identical across N=5 replays | eval | 41 |
| NFR-006 | Canonical-run provenance | MUST | 100% claim-grade numbers carry canonical_claim_run==True + provenance stamp | eval | 19 |
| NFR-007 | Shared/regex-path latency | SHOULD | p50 ≤ 1 ms | swarm | 11 |
| NFR-008 | Early-exit chunk latency | SHOULD | p50 ≤ 1 ms ∧ p95 ≤ 2 ms | swarm | 3 |
| NFR-009 | Full-swarm latency per profile | SHOULD | p50 ≤ declared per-profile budget + p99 | swarm | 23 |
| NFR-010 | Throughput floor (lightweight path) | SHOULD | ≥ 5,000 rec/sec on 8-core | swarm | 6 |
| NFR-011 | Router-on recall floor | MUST | entities(ensemble) ⊇ entities(shared) ZERO violations; ε ≤ 0.005 | both | 22 |
| NFR-012 | Tier-3 RRS power | MUST | ≥ 385 paired personas/cell; 2-rung ≥385/≥897; Wilson CIs | eval | 21 |
| NFR-013 | MIA power | MUST | ≥ 128 shadow models + Secret-Sharer; TPR@FPR∈{1e-3,1e-2} | eval | 18 |
| NFR-014 | Pseudonymization integrity | MUST | unauthorized-reversal = 0; referential integrity = 100% | both | 16 |
| NFR-015 | Key/state separation (Art 4(5) proxy) | SHOULD | artifact-alone re-join = FAIL flagged | both | 21 |
| NFR-016 | Non-strippable re-id caveat | MUST | 100% exported privacy artifacts carry anti-anonymity caveat | eval | 24 |
| NFR-017 | Post-temp-scaling ECE | SHOULD | ≤ 0.05 high-resource / ≤ 0.08 long-tail; ECE_post ≤ ECE_pre | both | 13 |
| NFR-018 | Brier + decomposition | COULD | reported per powered entity class | eval | 7 |
| NFR-019 | Selective-risk AURC | SHOULD | AURC + risk-coverage curve; monotone non-increasing | both | 8 |
| NFR-020 | Calibrated confidence on every finding | MUST | 100% findings carry calibrated confidence + provenance; 0 bare-logit | both | 12 |
| NFR-021 | Abstention coverage-risk operating point | SHOULD | ≥3-point table at {1%,2%,5%} selective risk | both | 8 |
| NFR-022 | Cross-OS / cross-cloud parity | SHOULD | divergence = 0 | swarm | 9 |
| NFR-023 | Stream/batch/offline parity | MUST | divergence = 0 incl. chunk-boundary | swarm | 6 |
| NFR-024 | No real PII in repo/fixtures/logs (AX-001) | MUST | 0 SHOWSTOPPER/CATASTROPHIC findings | both | 19 |
| NFR-025 | Multilingual worst-group fairness gap | SHOULD | worst-group recall gap ≤ 0.10 | both | 12 |
| NFR-026 | Optional-dependency graceful degradation | SHOULD | runs on shared layer, 0 unhandled exceptions, no silent recall loss | swarm | 49 |

**Sparse NFR (file-count ≤ 3 in dev-assist-artifacts/):** NFR-008 (3 files — only in the
requirements table, traceability matrix UC-15 row, and one S2-04 traceability review). NFR-008
is a SHOULD; DC-02 covers it implicitly. Not a gap but note it for the API-reference latency
section — the committed latency-ceiling registry in `latency_ceilings.py` is the concrete
realization.

---

## 4. Decision Inventory

Design decisions sourced from
`/dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md`.
No separate `D-decision.md` ADR diamond files exist; all decisions are in the synthesis doc.

| DC | Title | Implements (FR / NFR) | Decision type | Program status |
|---|---|---|---|---|
| DC-01 | SharedLayerProjector — recall-floor by construction | FR-016, NFR-011, AX-003 | Architecture / invariant | LIVE (S1-01, SO-07) |
| DC-02 | MoE-router: DistilledTopKGate + rules-first Depth-1 early-exit | FR-018, NFR-007/008/009 | ML routing / latency | LIVE (S2-01/02, SO-12); early-exit blocked by S2-03 |
| DC-03 | Aux-loss-free SLA selection-bias | NFR-009/010 | ML training / fairness | LIVE (S2-04, SO-13) |
| DC-04 | Reversible pseudonymization + auditable key rotation | FR-019, NFR-014/015 | Crypto / privacy | IN-TREE (token store encrypted S6-03, SO-10) |
| DC-05 | 6 transforms + legal-regime mapping + orchestrate incumbents | FR-020/021/022 | Transform surface | PARTIAL (existing strategies; orchestration SHOULD) |
| DC-06 | RatingEnginePort + RatingEngineRegistry (3-tier ladder) | FR-003, NFR-001/026 | Architecture / ports-adapters | LIVE (S3-01/02/03, SO-08) |
| DC-07 | Coherent significance (joint posterior) + Davidson ties | FR-004, NFR-002/003 | Statistical design | LIVE (S3-04, SO-08) |
| DC-08 | Distinct anon-vs-pseudo scoring families (no-merge, CI guard) | FR-006/009/010, NFR-014/015 | Data model / invariant | LIVE (S4-01, SO-11) |
| DC-09 | attacks/ package: real Tier-3 LLM-adversary + LiRA@128 MIA | FR-011/012/013, NFR-012/013 | Security / eval | LIVE (S5-01/02/03/04, SO-14/17/18) |
| DC-10 | Calibration & selective-risk reporter | FR-005, NFR-017/018/019/020/021 | Eval / calibration | LIVE (S4-03, SO-11) |
| DC-11 | CanonicalRunGate + provenance + CI ship/no-ship + RecallFloorVerdictGuard | FR-007/008, NFR-006/011(ε) | Gate / provenance | LIVE (S7-02, SO-15/16) |
| DC-12 | BYO-pipeline SDK adapter + identical-incumbent scoring | FR-001/002 | SDK / eval | LIVE (S6-04, SO-20) |
| DC-13 | Agentic interception: router pre-filter + query-aware + 4-channel | FR-023..030, AX-006 | Agentic / security | LIVE (S6-01/02/05, SO-14/19); orchestrator wire-in pass-2 |
| DC-14 | Multimodal readers + per-modality benchmark + parity | FR-031..037, NFR-022/023 | Ingestion / portability | LIVE (S7-01, SO-21); OCR/DICOM/audio extraction pass-2 |
| DC-15 | Multilingual context + fairness gate + no-real-PII | FR-038/039, NFR-024/025/026, AX-001 | Cross-cutting | LIVE (S7-03, SO-22) |

**DECISION 1 (D4/D5 headline):** SharedLayerProjector + DistilledTopKGate + Rules-first depth-1
early-exit. Source: `D-implementation-ready-design.md:§DECISION 1`. Pugh winner (8.4).

**DECISION 2 (D4/D5 headline):** Bayesian-BT spine for rating engine; MLE-BT smoke/fallback
only. Resolved CATASTROPHIC eval-01. Source: `D-implementation-ready-design.md:§DECISION 2`.
Pugh winner (8.6).

**DECISION 3 (D4/D5 headline):** Agentic interception via router pre-filter + unified floor.
Source: `D-implementation-ready-design.md:§DECISION 3`. Pugh winner (Option A).

---

## 5. Evidence Inventory

### 5a. Release Verdict
Source: `/dev-assist-artifacts/05-testing/release-readiness-report.md:##Verdict`

| Dimension | Status |
|---|---|
| Overall SDO verdict (current HEAD) | NOT_YET — `canonical_claim_run=True`; binding G6 FAIL (F2 0.7214 vs 0.75 threshold; coverage 0.824) |
| Stage-5 testing verdict | SHIP-WITH-CAVEATS (foundation / DC-01 SharedLayerProjector) / DEFER (full redesign) |
| G1 Recall floor | PASS |
| G2 Pseudonymization integrity | PASS |
| G3 Recall dominance | PASS |
| G4 Calibration selective risk | PASS |
| G5 Latency + audit | PASS (since SO-16, S7-04) |
| G6 Raw non-inferiority (F2) | FAIL — honest NOT_YET; F2 0.7214 vs GLiNER2 ~0.74; this is a methodology gap not a regression |
| G7 Certified run provenance | PASS |
| NFR matrix | 2 VERIFIED + 2 PARTIAL + 22 DEFERRED + 0 FAIL |
| Suite | ~3,685 passed / 16 skipped / 0 failed; coverage 88.66% |
| Caveats | benchmark numbers PROVISIONAL (smoke run until canonical run regen + significance repair) |

### 5b. Examples / Benchmarks Evidence
Source: artifact tree (no `examples-and-tests-catalog.md` — see O-2)

| Evidence item | Location | Notes |
|---|---|---|
| benchmark-raw.csv | `/artifacts/benchmarks/benchmark-raw.csv` | Raw competitor benchmarks |
| benchmark-results.json | `/artifacts/benchmarks/benchmark-results.json` | Structured results |
| floor-baseline.json | `/artifacts/benchmarks/floor-baseline.json` | Recall floor baseline artifact |
| floor-gate-report.md | `/artifacts/benchmarks/floor-gate-report.md` | Floor gate report |
| benchmark-diagnostics.json | `benchmark-diagnostics.json` (root) | Diagnostics |
| benchmark-summary.md | `docs/benchmark-summary.md` | GENERATED/user-WIP — auto-rewritten by the competitor benchmark script; treat as volatile |
| gate_v1.json | (runtime artifact; signed + verified via S2-05) | MoE gate artifact |
| test suite (3,685 tests) | `tests/` | Comprehensive; xdist `.venv/bin/python -m pytest -n auto` (~7min) |

---

## 6. Doc-Seeds (per-stage authored narrative)

| Stage | Expected path | Status | Consequence |
|---|---|---|---|
| 01-Discovery | `/dev-assist-artifacts/01-discovery/_doc/doc-seed.md` | ABSENT | No authored plain-language discovery narrative; the `discovery-report.md` POV/use-case sections are the substitute source |
| 02-Requirements | `/dev-assist-artifacts/02-requirements/_doc/doc-seed.md` | ABSENT | No authored requirements plain-language narrative; the `requirements-document.md` §R7/R10 sections and `traceability-matrix.md` §PGOs serve as substitute |
| 03-Design | `/dev-assist-artifacts/03-design/_doc/doc-seed.md` | ABSENT | No authored design narrative; the three DECISION sections in `D-implementation-ready-design.md` and SME findings section serve as substitute |
| 04-Development | `/dev-assist-artifacts/04-development/_doc/doc-seed.md` | ABSENT | No authored development narrative; the MANIFEST.md `### S*-DONE` sections + SO sign-off `scope:` fields partially compensate |
| 05-Testing | `/dev-assist-artifacts/05-testing/_doc/doc-seed.md` | ABSENT | No authored testing narrative; `release-readiness-report.md` and the NFR matrix are the substitute sources |

**Note:** All five doc-seeds absent. This is an OBSERVATION (O-1), not a blocker. The MANIFEST.md
`### S*-DONE` narrative sections (SO-07 through SO-23) are the de-facto authored spine and
are substantive — they carry the journey beats including the adversarial-close drama.

---

## 7. User Docs Tree

Source: `docs/` directory. 20 files.

| File | Category | Sections | Notes |
|---|---|---|---|
| `docs/README.md` | Index | top-level navigation | Doc index for the docs/ tree |
| `docs/quickstart.md` | Tutorial | `##Install`, `##Detect with explicit transform mode`, `##Stream processing`, `##CLI quickstart`, `##Evaluate your own pipeline` | Primary user onboarding doc |
| `docs/api-reference.md` | Reference | `##Primary APIs`, `##Profile fields`, `##Config schema additions`, `##Output additions`, `##Evaluation and benchmarks`, `##pii-rate-elo evaluation framework`, `##CLI integration`, `##PDLC SOTA program surfaces` | UPDATED by S7-05: `##PDLC SOTA program surfaces` section lists all new API families |
| `docs/anonymization-vs-pseudonymization.md` | Conceptual | `##Anonymization vs Pseudonymization` (table: question/reversibility/scorer/key-axes) | NEW (S7-05/A3 — FR-010 MUST). The no-merge invariant + vanilla-vs-swarm positioning. Load-bearing for D3 |
| `docs/evaluate-your-pipeline.md` | How-to | `##60-second version`, `##Installation`, `##Predictor contract`, `##Programmatic API`, `##CLI workflow`, `##Package as SDK plugin`, `##Incumbents scored on identical path`, `##Certify a run`, `##Reading results`, `##Statistical significance`, `##CI gating`, `##Tier 3 evaluation`, `##Troubleshooting` | EXTENDED by S6-04 (BYO/identical-path sections) + S7-05 (canonical-run + supremacy CLI section) |
| `docs/pii-rate-elo.md` | Conceptual | (pii-rate-elo framework explanation) | Core eval framework explanation |
| `docs/pii-rate-elo-value.md` | Conceptual | `##Why pii-rate-elo over plain F1?`, `##Where the rankings diverge` | USER-WIP — explicitly untracked from docs gate; do not treat as canonical |
| `docs/benchmark-summary.md` | Reference | `##Accuracy Objective`, `##Speed Objective`, `##Statistical Significance` | GENERATED — auto-rewritten by `compare_competitors()` benchmark script; treat as volatile user-WIP |
| `docs/recall-floor.md` | Reference | `##What it guarantees`, `##Usage`, `##Guarantees & verification`, `##Roadmap` | SharedLayerProjector user-facing doc; shipped in first PDLC pass |
| `docs/configuration.md` | Reference | (config schema) | Configuration reference |
| `docs/dependencies-and-platforms.md` | Reference | (dependencies + platforms) | Dependency matrix |
| `docs/engine-plugin-guide.md` | How-to | (plugin authoring) | Engine plugin guide |
| `docs/extend-swarm.md` | How-to | (swarm extension) | Swarm extension guide |
| `docs/swarm-architecture.md` | Architecture | (4-layer swarm architecture) | Swarm architecture explanation |
| `docs/complex-mode-example.md` | Tutorial | (complex mode example) | Advanced usage example |
| `docs/tutorial-llm-pipeline.md` | Tutorial | (LLM pipeline tutorial) | LLM pipeline integration tutorial |
| `docs/long-context-entity-tracking.md` | Reference | (entity tracking) | Long-context entity tracking |
| `docs/autoresearch-integration.md` | How-to | (autoresearch integration) | Autoresearch integration |
| `docs/evidence-ledger.md` | Reference | `##Claim: README benchmark`, `##Claim: Competitor comparison`, `##Claim: Fusion strategies`, `##Claim: Stream payloads`, `##Claim: Performance gates`, `##Claim: Optional engines` | Evidence ledger for documented claims |
| `docs/release-guide.md` | Operations | (release guide) | Release guide |

**User-WIP / generated (unstable):** `docs/pii-rate-elo-value.md` (user-WIP, excluded from
docs gate), `docs/benchmark-summary.md` (auto-generated by competitor benchmark script — see
`benchmark-harness-side-effects` memory note: the SCRIPT auto-rewrites README.md and clobbers
CWD artifacts).

---

## 8. Code Surface (API-Reference Verification)

### 8a. Top-level packages (src/pii_anon/)

| Package/module | Description | Headline public symbols |
|---|---|---|
| `pii_anon/__init__.py` | Package root | top-level exports |
| `pii_anon/orchestrator.py` | Sync/async `PIIOrchestrator`; the primary API | `PIIOrchestrator` (run/run_async/detect_only/run_stream/capabilities/discover_engines) |
| `pii_anon/pipeline.py` | Pipeline primitives | `evaluate_pipeline`, `run_benchmark`, `compare_competitors` |
| `pii_anon/cli.py` | Typer CLI (17+ commands) | `pii-anon canonical-run`, `pii-anon supremacy`, `pii-anon benchmark-publish-suite`, `pii-anon compare-competitors` |
| `pii_anon/bridge.py` | Cross-pillar bridge | bridge utilities |
| `pii_anon/moe.py` | MoE fusion strategy | `MoEFusionStrategy` |
| `pii_anon/swarm.py` | Swarm fusion strategy | `SwarmFusionStrategy` |
| `pii_anon/fusion.py` | Fusion builder | `build_fusion`, `register_fusion_strategy` |
| `pii_anon/types.py` | Core types | `EngineFinding`, `EnsembleFinding`, `LabeledSpan`, `ScoredFinding`, `EngineCapabilities` |
| `pii_anon/errors.py` | Error hierarchy | `PiiAnonError`, `MissingOptionalDependencyError` |

### 8b. New program modules (SOTA program surface)

| Module | FR/DC | Headline public symbols |
|---|---|---|
| `pii_anon/routing/shared_layer.py` | FR-016, DC-01 | `SharedLayerProjector`, `ProjectionResult`, `span_key_engine`, `is_shared_floor` |
| `pii_anon/routing/floor_fusion.py` | FR-016, DC-01 | `FloorProjectingFusion` |
| `pii_anon/routing/distilled_gate.py` | FR-018, DC-02 | `DistilledTopKGate` |
| `pii_anon/routing/gate_distillation.py` | FR-018, DC-02 | offline gate trainer |
| `pii_anon/policy/query_aware.py` | FR-023/024, DC-13 | `QueryAwareMaskingGate`, `MaskCandidate`, `QueryAwareDecision`, `QueryAwareBoundReport`, `score_query_aware_bound` |
| `pii_anon/policy/router.py` | DC-13 | `PolicyRouter` |
| `pii_anon/agentic/interception.py` | FR-025/026/027, DC-13 | `FourChannelGuard`, `InterceptionLedger`, `InterceptionRecord`, `AgentChannel`, `ChannelMasker`, `NoRawPIIPersistError` |
| `pii_anon/agentic/leakage_sankey.py` | FR-028/029, DC-13 | `LeakageSankey`, `SankeyEdge`, `build_leakage_sankey`, `score_injection_resistance`, `InjectionResistanceReport` |
| `pii_anon/tokenization/encrypted_store.py` | FR-019, NFR-014/015 | `EncryptedSQLiteTokenStore`, `KeyEnvelope`, `EnvelopeKeyProvider`, `StaticTestKeyProvider` |
| `pii_anon/ingestion/native.py` | FR-031..036, DC-14 | `NativeReaderRegistry`, `NativeReader` (Protocol), `ReaderCapabilities`, `ImageOcrReader`, `DicomReader`, `AudioReader`, `default_reader_registry`, `reader_capabilities` |
| `pii_anon/ingestion/native_pdf.py` | FR-031..036, DC-14 | `PdfTextReader` (stdlib PDF extraction with bounded FlateDecode) |
| `pii_anon/eval_framework/metrics/deid_families.py` | FR-006/009/010, DC-08 | `AnonymizationScorer`, `AnonymizationScore`, `PseudonymizationIntegrityScorer`, `PseudonymizationIntegrityScore`, `DeidFamilyScores` |
| `pii_anon/eval_framework/metrics/fairness_gate.py` | FR-039, DC-15 | `evaluate_language_fairness`, `FairnessGateReport`, `LanguageGroupSlice` |
| `pii_anon/eval_framework/byo_pipeline.py` | FR-001/002, DC-12 | `BYOPipelineRegistry`, `engine_predictor`, `incumbent_predictor`, `evaluate_incumbent`, `build_identical_path_leaderboard`, `gliner_predictor`, `presidio_predictor`, `INCUMBENT_SYSTEMS` |
| `pii_anon/eval_framework/attacks/spec.py` | FR-011..013, DC-09 | `AttackSpec`, `ResourceBudget`, `SandboxViolation`, `load_attack_spec` |
| `pii_anon/eval_framework/attacks/reid.py` | FR-011/012, DC-09 | `ReidAttack` (Protocol), `ReidPersona`, `ReidTarget`, `ReidGuess`, `ReidSuccessMetrics`, `BaselineDeterministicReidAttack`, `score_reid_attack` |
| `pii_anon/eval_framework/attacks/reid_tier3.py` | FR-011/012, DC-09 | Tier-3 representative adversary |
| `pii_anon/eval_framework/attacks/mia.py` | FR-013, DC-09 | `MiaAttack` (Protocol), `MiaRecord`, `RepresentativeMiaAttack`, `MiaSuccessReport`, `SecretSharerReport`, `MiaPowerReport`, `score_mia_attack`, `assess_mia_power`, `mia_attack_runner`, `canary_exposure` |
| `pii_anon/eval_framework/attacks/sandbox.py` | FR-029, DC-09 | sandboxed runner |
| `pii_anon/evaluation/canonical_run.py` | FR-007/008, DC-11 | `CanonicalRunGate`, `_assemble_base_payload`, `_attach_g2_deid_families`, `_attach_g4_calibration`, `_attach_g5_fields` |
| `pii_anon/eval_framework/evaluation/competitive_supremacy.py` | DC-11 (SDO gate) | `CompetitiveSupremacyGate`, `SupremacyVerdict`, `GuaranteeResult`, `Verdict`, `recall_floor_breachers`, `_g1_recall_floor`, `_g2_pseudonymization_integrity`, `_g3_recall_dominance`, `_g4_calibration_selective_risk`, `_g5_audit_latency`, `_g6_raw_noninferiority`, `_g7_certified_run`, `_finite_unit_score`, `_is_finite_number`, `_is_nonblank_str` |
| `pii_anon/eval_framework/evaluation/latency_ceilings.py` | NFR-009, DC-02 | committed per-profile numeric latency ceilings (NFR-009 registry) |
| `pii_anon/eval_framework/rating/` | FR-003, DC-06 | `RatingEnginePort`, `RatingEngineRegistry`, `PIIRateEloEngine` (glicko-legacy), `BradleyTerryMLEEngine`, `BayesBTEngine`, `convergence.py` (NFR-001 gate) |

### 8c. Existing baseline modules (relevant to api-reference)

| Module | Description |
|---|---|
| `pii_anon/engines/` | EngineAdapter, EngineRegistry, gliner/presidio/spacy/stanza/scrubadub/regex/llm-guard adapters |
| `pii_anon/transforms/` | TransformStrategy, StrategyRegistry, 6 strategies |
| `pii_anon/calibration/` | dominance, offline, online, store |
| `pii_anon/tracking/` | identity_ledger, linker |
| `pii_anon/segmentation/` | chunker, reconciler, segmenter |
| `pii_anon/tokenization/` | store, key_manager, providers, reidentification (+ new encrypted_store) |
| `pii_anon/eval_framework/metrics/composite.py` | CompositeScore, CompositeConfig, compute_composite, DeploymentProfile, FloorGateConfig |
| `pii_anon/eval_framework/rating/` | full rating ladder |
| `pii_anon/moe_gate_signing.py` | gate artifact signing (S2-05) |

---

## 9. Identifier Census

Total cross-artifacts occurrence counts (dev-assist-artifacts/ tree only):

| ID family | Total unique IDs | Max file count (highest) | Min file count (lowest) |
|---|---|---|---|
| FR-NNN | 39 (FR-001..039) | 69 (FR-026) | 5 (FR-036) |
| NFR-NNN | 26 (NFR-001..026) | 49 (NFR-026) | 3 (NFR-008) |
| DC-NN | 15 (DC-01..15) | 116 (DC-13) | 1 (DC-05) |
| UC-NN | 28 (UC-01..28) | (in traceability-matrix/requirements) | — |
| SO-NN | 23 (SO-01..23) | — | — |

All 39 FRs appear in at least 5 artifact files. All 26 NFRs appear in at least 3 files.
No FR or NFR has zero downstream mention — there are NO true orphans in the definition
sense. Low-count candidates for D4 coverage attention are noted below.

**D4 coverage watch-list (SHOULD → MUST documentation mapping check):**
- `FR-036` (5 files): stream/batch/offline parity MUST — thin trace outside stories; in DC-14
- `NFR-008` (3 files): early-exit chunk latency SHOULD — only in requirements table + 2 trace refs
- `DC-05` (1 file): 6 transforms + legal-regime mapping — only mentioned in DC-01..15 table; FR-020/021/022 are SHOULD and partially deferred
- `FR-033` / `FR-035` (6 files each): extraction-fidelity + CI regression gate — both addressed in S7-01 but deserve explicit API-reference callouts

---

## 10. Per-Deliverable Source Pre-Map

No explicit `documentation.deliverables` list in `developer-assistant.yaml`. The following
pre-map is inferred from the D6 SME Docs MAJORs (three named requirements in
`D-implementation-ready-design.md:§SME findings`), the S7-05 story implementation, and
standard Stage-6 deliverable categories. D2 should confirm or extend this list.

| Inferred deliverable | Candidate sources (file:section) |
|---|---|
| **API reference** (expand `docs/api-reference.md`) | `docs/api-reference.md:##PDLC SOTA program surfaces`; `docs/anonymization-vs-pseudonymization.md` (full); `D-implementation-ready-design.md:§D1 DC table`; all 14 new module public symbols in §8b; `docs/evaluate-your-pipeline.md:##Package as SDK plugin + ##Certify a run + ##Incumbents scored on identical path` |
| **Anonymization vs Pseudonymization conceptual doc** | `docs/anonymization-vs-pseudonymization.md` (complete new doc, FR-010); `eval_framework/metrics/deid_families.py` (scorers + no-merge invariant); `D-implementation-ready-design.md:§DECISION 2, §DC-08` |
| **Quickstart / getting-started** | `docs/quickstart.md` (primary); `docs/evaluate-your-pipeline.md:##60-second version`; `docs/api-reference.md:##Primary APIs` |
| **Evaluation / BYO-pipeline guide** | `docs/evaluate-your-pipeline.md` (full); `eval_framework/byo_pipeline.py:BYOPipelineRegistry`; `SO-20` scope; S6-04 story `§Implements` |
| **Architecture / design decisions** | `D-implementation-ready-design.md:§DECISION 1/2/3 + §D1 DCs`; `docs/swarm-architecture.md`; `docs/recall-floor.md`; `moe-architecture-and-guarantee.md`; `SO-12/SO-13` scopes |
| **Release notes / changelog** | `MANIFEST.md:§S*-DONE sections` (SO-07..SO-23 narrative beats); `development-log.md:§W6 Execution`; `release-readiness-report.md:##Caveats` + `##Verdict` |
| **Developer journey doc** | `MANIFEST.md:§Handoff Signals` (all `### S*-DONE` sections); `PDLC-JOURNEY.md:##Traceability spine + ##What shipped + ##What's deferred`; `_signoffs/SO-01..SO-23` scopes |
| **Recall-floor / swarm guarantees** | `docs/recall-floor.md` (existing); `routing/shared_layer.py:SharedLayerProjector`; `D-implementation-ready-design.md:§DECISION 1`; `SO-07`, `SO-12` |
| **SDO certification guide** | `docs/evaluate-your-pipeline.md:##Certify a run`; `evaluation/canonical_run.py:CanonicalRunGate`; `eval_framework/evaluation/competitive_supremacy.py` (G1–G7 guarantees); `SO-15`, `SO-16`; `release-readiness-report.md:##Verdict` |
| **Multilingual / fairness** | `eval_framework/metrics/fairness_gate.py`; S7-03 story `§Implements`; `SO-22` scope; `D-implementation-ready-design.md:§DC-15` |
| **Native-format readers guide** | `ingestion/native.py:NativeReaderRegistry + reader_capabilities()`; `ingestion/native_pdf.py:PdfTextReader`; S7-01 story `§Implements + §DONE sections`; `SO-21` scope |
| **Security / attacks / MIA** | `eval_framework/attacks/` (spec/reid/mia/sandbox); `SO-10` (security MUSTs); `SO-17`/`SO-18`; `D-implementation-ready-design.md:§DC-09` |
| **Agentic / query-aware masking** | `policy/query_aware.py`; `agentic/interception.py`; `agentic/leakage_sankey.py`; `SO-14`, `SO-19`; `docs/api-reference.md:##PDLC SOTA program surfaces:Agentic privacy` |
| **Glossary** | `01-discovery/discovery-report.md:§1 POV + §4 Market`; `anonymization-vs-pseudonymization.md`; `docs/pii-rate-elo.md`; `D-implementation-ready-design.md:§D0` axioms |

---

## 11. Missing-But-Expected (OBSERVATIONs)

| ID | Missing item | Expected path | Consequence |
|---|---|---|---|
| O-1 | All five per-stage doc-seeds | `01-discovery/_doc/doc-seed.md` through `05-testing/_doc/doc-seed.md` | The journey/glossary deliverables lose the staged plain-language authored narrative. Substitute sources are available (discovery-report.md POV sections, MANIFEST S*-DONE sections, SO scope lines). D3 authors can pull from substitutes; D5 synthesis should note this as an explicit caveat. The absence does NOT block; it is a quality-of-source signal. **Recommendation:** add the five doc-seeds on a follow-up pass to convert inferred narrative into explicit authored narrative. |
| O-2 | examples-and-tests-catalog.md | `05-testing/examples-and-tests-catalog.md` | The evidence inventory and "proof beats" in the changelog / journey doc cannot cite a curated example catalog. Substitute: the test suite (3,685 tests) is the living catalog; the `_reviews/story/*/synthesis.md` files carry per-story acceptance-test narratives. The `docs/evidence-ledger.md` partially covers documented claims. D3 authors should pull from story synthesis.md files and release-readiness-report.md caveats for proof beats. |
| O-3 | 00-validation/ contents | `dev-assist-artifacts/00-validation/` | Empty (only `.gitkeep`). Validation artifacts were not generated in this PDLC pass. D2 should note that any deliverable claiming "validated against user research" must be caveated as AGENT_SIMULATED with Pass-2 follow-up documented. |
| O-4 | `documentation:` block in developer-assistant.yaml | `developer-assistant.yaml:documentation:` | No explicit deliverables list or configuration block. D2 must author the authoritative deliverable set from scratch (inferred pre-map in §10 above is a starting point). The absence of a config block means D4 cannot assert "every configured deliverable is covered" — D2 defines scope, D4 verifies against D2's definition. |
| O-5 | Diamond D-decision.md ADR files | `03-design/*/D-decision.md` | No separate per-decision ADR files. Architecture-and-ADR deliverable must source from `D-implementation-ready-design.md:§DECISION 1/2/3` and `§D1 DC table` only. All decision context is present; it is not structured as individual ADR files. Not a blocker — structure is consolidated, not dispersed. |
| O-6 | NFR-008 defining implementation story | (no dedicated story for early-exit chunk latency SHOULD) | NFR-008 appears in only 3 artifact files. DC-02 covers it via the DistilledTopKGate + rules-first depth-1 early-exit, and `latency_ceilings.py` is the committed registry, but there is no dedicated story with formal acceptance tests specifically for the p50 ≤ 1 ms ∧ p95 ≤ 2 ms early-exit bound. D4 should flag this as a SHOULD gap: documented but not gated by a named acceptance test. |
| O-7 | `pii-rate-elo-value.md` in stable docs | `docs/pii-rate-elo-value.md` | Explicitly user-WIP and excluded from the docs gate test (`test_docs_discoverability.py`). D3 authors should NOT include it in any canonical documentation deliverable without user confirmation that it is stable. |

---

## D2 Handoff Notes

**What the architect should weigh when defining the deliverable set and information architecture:**

1. **The sign-off narrative spine is the strongest single source.** SO-01 through SO-23
   carry explicit `scope:` multi-line fields that tell the program's story chronologically,
   including the adversarial-close drama (SO-15's 7-round keystone), the CATASTROPHIC eval-01
   resolution (SO-08), the SDO gate fabrication-hardening arc (SO-09 through SO-16), and the
   feature surface completion (SO-17 through SO-23). A journey document authored from these
   23 scope lines alone would be substantive and accurate. D3 should treat these as the
   authoritative narrative anchor.

2. **The MANIFEST.md `### S*-DONE` sections are rich, dense narrative material.** The
   approximately 12 `### S*-DONE` subsections in MANIFEST.md (Sprint-1 COMPLETE through
   S7-05 DONE) carry per-story technical narratives, commit hashes, gate outcomes, and
   explicit links to the SDO guarantee axis each story closes. These are the best source for
   a technical release-notes style changelog.

3. **The absent test catalog (O-2) is a real gap for "proof beats."** The journey doc and
   API reference will want to cite evidence. The three synthesis.md files per story
   (e.g., `_reviews/story/S7-02/synthesis.md`) and the `release-readiness-report.md:##Evidence`
   section are the best available substitutes. D2 should decide whether to direct D3 to pull
   from these or to recommend authoring a lightweight examples-and-tests catalog first.

4. **The live user-docs tree is a standing commitment with teeth.** `test_docs_discoverability.py`
   (shipped by S7-05) enforces that all headline public symbols appear in the api-reference,
   all intra-docs links resolve, and the index covers every non-WIP doc file. Any new
   deliverable added to `docs/` must satisfy this gate. D2 should factor this constraint into
   the deliverable architecture — new docs added in D3 must not break the existing gate.

5. **The `docs/anonymization-vs-pseudonymization.md` is a new anchoring deliverable.** It
   closes the FR-010 MUST (distinct anon-vs-pseudo families) and contains the no-merge
   invariant verbatim. It should be a primary source for any conceptual/architecture
   documentation section on the de-identification scoring surface.

6. **`developer-assistant.yaml` has no `documentation:` block** (O-4). D2 must author the
   canonical deliverable set definition. The inferred pre-map in §10 is a good starting
   point but is not authoritative — D2's output IS the authoritative source for D3/D4/D5.

7. **The SDO verdict is honest NOT_YET — the docs must say so.** The release-readiness-report
   SHIP-WITH-CAVEATS verdict and the NOT_YET SDO verdict (G6 FAIL on F2) must be stated
   accurately in any changelog or release-notes deliverable. The f2-gap-attribution.md
   (`05-testing/_diagnostics/`) explains why this is a methodology gap, not a regression —
   that distinction is important narrative context for any "current limitations" section.

8. **The Pass-2 (real-user) commitment is documented and must be carried forward.** All
   requirements are AGENT_SIMULATED. The release-readiness-report, the brownfield assessment,
   and several SO scope lines explicitly name the Pass-2 follow-up items (latency thresholds,
   Tier-3 realism, OCR/DICOM extraction at real strength, OS-matrix portability certification).
   Any deliverable covering roadmap or caveats should reference these.

9. **The `docs/benchmark-summary.md` is generated/volatile.** It is auto-rewritten by the
   competitor benchmark script. D2 should exclude it from any documentation deliverable list
   or explicitly flag it as generated content with a maintenance note.

10. **Cross-repo edges are real and load-bearing.** The traceability-matrix `external_refs`
    columns carry `DATA:` (pii-anon-eval-data S5–S7) and `PAPER:` dependencies for FR-002,
    FR-003, FR-011, FR-013, UC-02, UC-09, UC-10. Any deliverable covering the eval framework
    or Tier-3 / MIA claims must cite these cross-repo boundaries and note what is CODE-local
    vs DATA-track.
