# Requirements — pii-anon (canonical, R0–R10)

> **Stage 2, AGENT_SIMULATED, representative scale** (22 agents, workflow `wvsqmzw7j`: 5 FR + 5 NFR authors via dev-assist fr/nfr-author + 6 prioritization respondents + 6 threshold validators). Brownfield. All requirements start `provisional_status: AGENT_SIMULATED` (Pass-2 real-user follow-up documented). **39 FRs · 26 NFRs.** Evaluation-led, co-equal pillars. Every FR/NFR traces to a UC→PGO (see `traceability-matrix.md`); cross-repo deps tagged `DATA:`/`PAPER:`.

## Functional Requirements (39)
Each FR has a boolean-testable Given/When/Then + a negative test in the workflow transcript; the headline ones are quoted in `traceability-matrix.md`.

| FR | Title | UC | Pillar | MoSCoW |
|---|---|---|---|---|
| **Eval core** ||||||
| FR-001 | BYO-pipeline adapter contract (score any system, zero harness-core edits) | UC-01 | eval | MUST |
| FR-002 | Identical scoring path for incumbents (Presidio/GLiNER2/OpenAI-Filter) | UC-01 | eval | SHOULD |
| FR-003 | Bayesian Bradley-Terry rating engine + credible intervals + rank distribution | UC-02 | eval | MUST |
| FR-004 | Coherent significance (CIs bracket estimates; effect sign ↔ verdict) | UC-03 | eval | MUST |
| FR-005 | Calibration & selective-risk scorecard (ECE/MCE/Brier/AURC + reliability) | UC-04 | eval | SHOULD |
| FR-006 | Pseudonymization integrity reported as a structurally distinct family | UC-05 | eval | MUST |
| FR-007 | CI ship/no-ship gate on composite + per-language recall floor | UC-06 | eval | MUST |
| FR-008 | Canonical-run / provenance gate (refuse claim-grade verdicts without it) | UC-06 | eval | MUST |
| **Privacy eval (headline quadrant)** ||||||
| FR-009 | Pseudonymization integrity as a 5-axis family (reversal/collision/referential/Art-4(5)) | UC-07 | eval | MUST |
| FR-010 | Enforce anon vs pseudo as distinct families (no merge path/field) | UC-08 | eval | MUST |
| FR-011 | Real Tier-3 LLM-adversary re-id (RRS/QIC/BSL), de-circularized | UC-09 | eval | MUST |
| FR-012 | Control Tier-3 circularity/contamination/non-stationarity | UC-09 | eval | SHOULD |
| FR-013 | Full-power MIA (LiRA@128 + Secret-Sharer), TPR@low-FPR | UC-10 | eval | MUST |
| FR-014 | Key-compromise blast radius + key-rotation resilience | UC-11 | eval | SHOULD |
| FR-015 | Anonymization residual-risk-vs-utility Pareto (never merged) | UC-12 | eval | SHOULD |
| **Swarm + transforms** ||||||
| FR-016 | Recall-floor by construction (ensemble spans ⊇ shared-layer spans) | UC-13 | swarm | MUST |
| FR-017 | Per-language recall-floor CI gate (router-on ≥ router-off − ε) | UC-14 | swarm | MUST |
| FR-018 | MoE-router: learned routing + span-dedup + selective activation + early-exit | UC-15 | swarm | MUST |
| FR-019 | Reversible pseudonymization: deterministic surrogates + auditable key rotation + Art-4(5) separation | UC-16 | swarm | MUST |
| FR-020 | Six transform strategies with legal-regime mapping, preserving joins | UC-17 | swarm | SHOULD |
| FR-021 | Orchestrate incumbent detectors behind one recall-floored interface | UC-18 | swarm | SHOULD |
| FR-022 | Pseudonymization-integrity emitted distinct from anonymization (swarm side) | UC-16/17 | both | SHOULD |
| **Agentic & contextual** ||||||
| FR-023 | Query-aware masking gate (retain query-relevant PII, mask the rest) | UC-19 | swarm | SHOULD |
| FR-024 | Bound query-aware over-redaction + false-retention vs baseline | UC-19 | eval | SHOULD |
| FR-025 | Intercept all four agent channels (prompt/memory/tool-I/O/trace) least-privilege | UC-20 | swarm | MUST |
| FR-026 | Persist no raw PII to any channel after masking (AX-006) | UC-20 | swarm | MUST |
| FR-027 | Stable session pseudonyms, authorized-only reversal | UC-21 | swarm | SHOULD |
| FR-028 | Per-channel agentic leakage counts (leakage-Sankey, 6 channels) | UC-22 | eval | MUST |
| FR-029 | Prompt-injection exfiltration resistance (ASR vs benign-task-success) | UC-23 | eval | MUST |
| FR-030 | Byte-identical agentic masking decisions given seed/key/scope | UC-20/21 | swarm | SHOULD |
| **Multimodal & portability** ||||||
| FR-031 | Native-format readers emit a uniform `Iterator[IngestRecord]` (PDF/image/screenshot/DICOM/audio) | UC-24 | swarm | MUST |
| FR-032 | Round-trip reconstruction preserves non-PII payload byte-for-byte | UC-24 | swarm | MUST |
| FR-033 | Extraction-fidelity assertion per modality (offsets→source coords) | UC-24 | swarm | SHOULD |
| FR-034 | Per-modality recall benchmark, scored separately | UC-25 | eval | MUST |
| FR-035 | CI gate on multimodal reader recall regression | UC-25 | eval | MUST |
| FR-036 | Identical scrub decisions across stream/batch/offline (parity) | UC-26 | swarm | MUST |
| FR-037 | OS-matrix portability (byte-identical across OS × local/cloud) | UC-27 | swarm | SHOULD |
| FR-038 | Multilingual non-EN context feature active in detection | UC-28 | swarm | SHOULD |
| FR-039 | Per-language fairness gap bounded + gated | UC-28 | both | SHOULD |
| FR-040 | Detect GDPR Article-9 special-category PII (sexual orientation, trade-union membership, genetic data) via specific-field-label / intrinsic-structure recognition — never generator-filler anchors | UC-13 | swarm | SHOULD |

> **FR-040 (grafted 2026-07-10, `/dev-assist-enhance sp3-v220-rebaseline`).** The eval substrate
> `pii-anon-eval-data` expanded 63→66 canonical types, adding the 3 GDPR Art-9 special categories
> (each powered ≥400 spans). This FR records the detection-coverage capability closing that gap:
> label-gated recognition (SEXUAL_ORIENTATION lexicon; TRADE_UNION_MEMBERSHIP / GENETIC_DATA
> labeled-value) plus intrinsic gene-symbol / dbSNP-rs-ID structure for GENETIC_DATA. Bound by the
> eval-integrity axiom (no "Record shows X" generator-filler anchor) and the leak-direction
> invariant (additive detections only; over-mask is the safe direction). Verified 100% recall / 0 FP
> on the train-en Art-9 gold + 600 negatives; census-reachable so it earns internal + external
> credit. Impl: `engines/regex/patterns.py`; tests: `tests/test_coverage_tranche_sp3.py`.

## Non-Functional Requirements (26, quantified)

| NFR | Title | Threshold (quantified) | Pillar | MoSCoW |
|---|---|---|---|---|
| **Eval statistical rigor** ||||||
| NFR-001 | Bradley-Terry MCMC convergence + identifiability | split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param; anchored/regularized; 0 divergences | eval | MUST |
| NFR-002 | Significance coherence | ZERO incoherent deltas (point∈CI ∧ sign↔verdict ∧ significant iff CI excludes 0) | eval | MUST |
| NFR-003 | Bootstrap CI empirical coverage | nominal-95% achieves 93–97% coverage over ≥1,000 sim replicates | eval | SHOULD |
| NFR-004 | Statistical-power consumption (risk-tiered) | positives/slice ≥ 1,522 (≈0.99±0.5pp) / 753 (≈0.98±1pp) / 200 long-tail | both | MUST |
| NFR-005 | Scoring-run determinism | byte-identical artifacts across N=5 replays given (seed,key,scope) | eval | MUST |
| NFR-006 | Canonical-run provenance | 100% claim-grade numbers carry `canonical_claim_run==True` + provenance stamp | eval | MUST |
| **Detection performance** ||||||
| NFR-007 | Shared/regex-path latency (honest) | speed profiles p50 ≤ 1 ms; accuracy profiles p50 ≤ budget | swarm | SHOULD |
| NFR-008 | Early-exit chunk latency | p50 ≤ 1 ms ∧ p95 ≤ 2 ms; no NER/Dawid-Skene on fast-pass chunks | swarm | SHOULD |
| NFR-009 | Full-swarm latency per profile | p50 ≤ declared per-profile budget (committed numeric ceiling + p99; NOT sub-0.24ms parity) | swarm | SHOULD |
| NFR-010 | Throughput floor (lightweight path) | ≥ 5,000 rec/sec on 8-core; per-class transparency | swarm | SHOULD |
| NFR-011 | Router-on recall floor | entities(ensemble) ⊇ entities(shared) ZERO violations; per-language ε ≤ 0.005 | both | MUST |
| **Privacy / re-identification** ||||||
| NFR-012 | Tier-3 RRS power | ≥ 385 paired personas/cell (REID_LOW); 2-rung ladder ≥385/≥897; Wilson CIs | eval | MUST |
| NFR-013 | MIA power | ≥ 128 shadow models + Secret-Sharer; report TPR@FPR∈{1e-3,1e-2} | eval | MUST |
| NFR-014 | Pseudonymization integrity | unauthorized-reversal = 0 (vs stated model); referential integrity = 100% | both | MUST |
| NFR-015 | Key/state separation (Art 4(5) proxy) | artifact-alone re-join = FAIL flagged; external secret required to reverse | both | SHOULD |
| NFR-016 | Non-strippable re-id caveat | 100% exported privacy artifacts carry the anti-anonymity caveat | eval | MUST |
| **Calibration & abstention** ||||||
| NFR-017 | Post-temp-scaling ECE | ECE ≤ 0.05 (high-resource classes) / ≤ 0.08 (long-tail, **REVISE-LOOSER**); ECE_post ≤ ECE_pre always | both | SHOULD |
| NFR-018 | Brier + decomposition | reported per powered entity class | eval | COULD |
| NFR-019 | Selective-risk AURC | AURC + risk-coverage curve; monotone non-increasing | both | SHOULD |
| NFR-020 | Calibrated confidence on every finding (AX-005) | 100% findings carry calibrated confidence + provenance; 0 bare-logit | both | MUST |
| NFR-021 | Abstention coverage-risk operating point | ≥3-point table at selective risk {1%,2%,5%} | both | SHOULD |
| **Portability / security / multilingual** ||||||
| NFR-022 | Cross-OS / cross-cloud parity | divergence = 0 (span set ∧ transformed output byte-identical) | swarm | SHOULD |
| NFR-023 | Stream/batch/offline parity | divergence = 0 incl. chunk-boundary cases | swarm | MUST |
| NFR-024 | No real PII in repo/fixtures/logs (AX-001) | 0 SHOWSTOPPER/CATASTROPHIC findings | both | MUST |
| NFR-025 | Multilingual worst-group fairness gap | worst-group recall gap ≤ 0.10 across powered language groups | both | SHOULD |
| NFR-026 | Optional-dependency graceful degradation | runs on shared layer, 0 unhandled exceptions, no silent recall loss | swarm | SHOULD |

## R7 — Prioritization (MoSCoW; reuses Discovery Kano + 6-respondent survey)
- **MUST (~22):** the eval-integrity foundation (FR-003/004/008, NFR-001/002/006), pseudonymization-integrity + distinct families (FR-006/009/010, NFR-014), real Tier-3 + MIA (FR-011/013, NFR-012/013), recall-floor (FR-016/017, NFR-011), agentic core (FR-025/026/028/029), multimodal core (FR-031/032/034/035/036), calibrated-confidence (NFR-020), parity/no-real-PII (NFR-023/024), power (NFR-004), determinism (NFR-005).
- **SHOULD (~15):** incumbent-identical scoring (FR-002), calibration scorecard (FR-005), Tier-3 controls (FR-012), key-compromise (FR-014), anonymization Pareto (FR-015), 6-transforms/orchestration (FR-020/021/022), query-aware (FR-023/024), session pseudonyms (FR-027), latency/throughput (NFR-007/008/009/010), AURC/abstention (NFR-019/021), portability/fairness/graceful-deg (NFR-022/025/026/FR-037/038/039).
- **COULD:** Brier decomposition (NFR-018), left-field eval (IRT/adaptive/meta-eval — backlog).
- **Distribution:** ~56% MUST / ~38% SHOULD / ~6% COULD across 65 requirements — MUSTs concentrate on the eval-integrity + headline-novelty critical path, exactly per the POV.

## R10 — NFR Threshold Validation (6 stress-tested; 0 DIVERGED)
| NFR threshold | Verdict | Refinement |
|---|---|---|
| Latency (re-scoped tiered) | PERSONA-CONDITIONAL | commit a numeric swarm ceiling + add p99 + per-detector-class p50/p95/p99 |
| Recall floor (ε + ensemble⊇shared) | PERSONA-CONDITIONAL | split into the property (ZERO violations) + the per-language ε gate |
| Statistical power (1522/753/200) | PERSONA-CONDITIONAL | KEEP tiers LOCKED (NIST-derived); assert at committed-lattice cell granularity |
| Calibration ECE ≤ 0.05 | **REVISE-LOOSER** | per-class: ≤0.05 high-resource, ≤0.08 long-tail; ECE_post ≤ ECE_pre always holds |
| Tier-3 RRS power (≥385) | PERSONA-CONDITIONAL | keep ≥385 REID_LOW + 2-rung ladder ≥385/≥897 |
| Determinism (byte-identical) | PERSONA-CONDITIONAL | scope to "within a key epoch"; seeded reproducible scoring |

**0 DIVERGED, 0 INSUFFICIENT-EVIDENCE** — all conditionally accepted with documented refinements (folded into the NFR thresholds above).

## Methodology & epistemic honesty
Representative-scale cohorts (single-session limit) — AGENT_SIMULATED, not real users; Pass-2 follow-up for load-bearing MUSTs + the latency/Tier-3 thresholds. Cross-repo: the eval-integrity FRs/NFRs depend on `pii-anon-eval-data` stats/scorers/`assemble_paired_set` (S5–S7) — see `external_refs` in `traceability-matrix.md`. SME caveats carried: EDPB Art 4(5) framed as an engineering proxy (NFR-015), HIPAA Expert Determination as the legitimizing hook, Tier-3 circularity controlled (FR-012), Bradley-Terry identifiability anchored (NFR-001).
