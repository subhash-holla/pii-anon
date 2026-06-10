# Changelog — pii-anon

> **Release:** `1.5.0rc1` (release candidate; feature-complete; `pdlc/sota-program` branch)
> **Date:** 2026-06-10 (as of Stage-6 documentation; sourced from sign-off ledger, not the system clock)
> **SDO verdict:** NOT_YET — see [Honest Status](#honest-status) before citing any benchmark number.

This changelog covers the full PDLC `pdlc/sota-program` sprint arc (Sprint 1 through Sprint 7, SO-07 through SO-23).
Each entry states: the story ID, its user-facing change, the FR/NFR IDs it closes, headline commits from the story gate
evidence, and the gate outcome. Entries are organized by work-stream, then presented chronologically within each stream.

---

## Sprint 1 — Recall-Floor Foundation (SO-07 · 2026-05-31)

**Theme:** guarantee `entities(output) ⊇ entities(shared)` by construction at the `build_fusion` seam,
closing the brownfield MAJOR (AX-003 violated in the `swarm.py` path) that caused silent recall loss.

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S1-01** | `SharedLayerProjector` — recall-floor by construction; re-injects shared-layer spans post-merge; `violations_blocked` counter; 2,000-case property test (0 violations). New `routing/shared_layer.py`. | FR-016, NFR-011, AX-003 | RED `ef85166` → GREEN `548f576` | 5/5 APPROVE |
| **S1-02** | `FloorProjectingFusion` — wires the projector into both `SwarmFusionStrategy` and `MoEFusionStrategy` at the `build_fusion` seam; the floor is now **live on the production path**, not standalone. New `routing/floor_fusion.py`. | FR-016, NFR-011 | RED → GREEN (SO-07 span) | Sprint gate APPROVE |
| **S1-03** | Per-language recall-floor ε-gate (ε ≤ 0.005); regression-guard test in `tests/test_recall_floor_per_language_gate.py` with proven "teeth" (deliberately broken guard ⇒ FAIL). | FR-017, NFR-011 | SO-07 span | 5/5 APPROVE |
| **S1-04** | Hypothesis `@given` property migration — dev-dep `hypothesis>=6.0`; upgrades recall-floor property suite to `@given` ergonomics. | NFR-011 | SO-07 span | APPROVE |
| **S1-05** | Swarm language propagation — `swarm.py` now propagates the `language` field on emission, fixing a pre-existing multilingual mislabel the floor exposed (duplicate spans on non-EN documents). | FR-016, FR-017 | SO-07 span (remediation) | APPROVE (remediated REQUEST_CHANGES from sprint gate) |

**Sprint-1 gate outcome:** Workflow `wftzms2fs` (11 agents) — REQUEST_CHANGES (1 MAJOR; remediated by S1-05) → APPROVE; 0/5 adversarial refutations upheld. Full suite: 2,690 passed / 12 skipped / 0 failed; coverage 86.22%; ruff + mypy --strict clean.

---

## Sprint 3 — Eval-Rating Ladder (SO-08 · 2026-05-31)

**Theme:** a 3-tier `RatingEnginePort` ladder replacing the internally-incoherent Glicko path; resolves the SME-panel CATASTROPHIC eval-01 (NFR-001 MCMC convergence). DC-06, DC-07.

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S3-01** | `RatingEnginePort` (`@runtime_checkable Protocol`: `run_round_robin` + `get_rating`); `RatingEngineRegistry` (entry-point group `pii_anon.rating_engines`); `PIIRateEloEngine` registered as `glicko-legacy`; AST import-boundary CI test (rating imports nothing from swarm/moe/fusion/policy). | FR-003 | RED `e5a554e` → GREEN `d5cf633` | 5/5 APPROVE |
| **S3-02** | `BradleyTerryMLEEngine` — pure-stdlib MM/Hunter-2004 + paired-bootstrap CIs + observable non-convergence (explicit diagnostic). Smoke / fallback tier behind the port. | FR-003, NFR-003 | SO-08 span | 5/5 APPROVE (1 MINOR: observable non-convergence; remediated in-loop) |
| **S3-03** | `BayesBTEngine` (NumPyro NUTS, claim-grade tier) + `convergence.py` (the hard NFR-001 gate: split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param ∧ 0 divergences; fails loud, names the binding constraint). **Resolves SME CATASTROPHIC eval-01.** Env-honest: lazy NUTS import; no silent fallback; `bayes-eval` optional extra added. | FR-003, NFR-001 | RED `33f389f` → GREEN `2b2110b` → REFACTOR `69cd45c` | 5/5 APPROVE (1 MINOR: divergence-arm fail-loud; remediated) |
| **S3-04** | Coherent significance by construction — one joint posterior → point ∈ CI ∧ sign ↔ verdict ∧ significant-iff-CI-excludes-0 cannot disagree; `rank_one_probability` J primitive; Davidson ties; FR-010/AX-004 separation. | FR-004, NFR-002, NFR-003 | SO-08/SO-09 span | 5/5 APPROVE (2 MINOR: traceability hygiene; remediated) |

**Sign-off SO-08** (2026-05-31): eval-integrity FOUNDATION complete. Full suite exit 0 (~2,757 passed / 0 failed); 86.03% coverage; `elo.py` + all 7 callers byte-identical throughout.

---

## Sprint 4 — SDO Gate + Deid Guarantees (SO-09 / SO-11 · 2026-06-01..02)

**Theme:** the SOTA-Dominance Objective gate (`CompetitiveSupremacyGate`) goes live and the G2/G4 guarantees become code-computable + fabrication-hardened. DC-08, DC-10, DC-11 (foundation).

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S4-CS-01** | `CompetitiveSupremacyGate` (`pii-anon supremacy` CLI): emits exactly one verdict `{CLAIM_GRADE_SOTA | PROVISIONAL_SOTA | NOT_YET}` + the single binding constraint per run; G1/G3/G6/G7 code-computable; MLE-bootstrap J-fallback; `RecallFloorVerdictGuard` (a recall-floor-breaching system can never be crowned); Tier-R/Tier-C registry with UNRUN honesty boundary. | FR-007, NFR-005, NFR-006 | RED `c407f52` → GREEN `4a04d0f` | 6/5 APPROVE @ iter-2 (iter-1 MAJOR: RecallFloorVerdictGuard absent; implemented + property-tested in-loop) |
| **S4-01** | `AnonymizationScorer` + `PseudonymizationIntegrityScorer` in `eval_framework/metrics/deid_families.py` — **distinct scoring families, never merged** (AX-004 no-merge CI guard); makes G2 code-computable. | FR-006, FR-009, FR-010, NFR-014, NFR-015 | RED `f385911` → GREEN `6aabeb2` → REFACTOR `97134ba` | 5/5 APPROVE + G2/G4 adversarial close `wi1yj97h9` (0 fabrication-possible) |
| **S4-03** | `SelectiveRiskReporter` in `eval_framework/metrics/selective_risk.py` — per-class ECE / Brier / AURC / ≥3-point abstention table / NFR-020 calibrated-confidence-coverage; makes G4 code-computable. | FR-005, NFR-017, NFR-018, NFR-019, NFR-020, NFR-021 | Same close arc as S4-01 | 5/5 APPROVE + same close |

**Adversarial close (SO-11, `wwty6wq9v` → `wi1yj97h9`):** caught phantom-0.0 missing-competitor "dominance" (G2) + NaN-ECE / coverage>1.0 / unclamped-threshold (G4) — MAJOR findings the story gates missed. Hardening: `_finite_unit_score` TypeGuard, `_is_finite_number`, `_g4_class_bar` tighten-only clamp. 0 fabrication-possible in re-close. Full suite: 3,186 passed / 0 failed; 87.05% coverage; ruff + both-mypy clean.

---

## Sprint 2 — MoE-Router (SO-12 / SO-13 · 2026-06-03)

**Theme:** learned routing at the `build_fusion` seam; signed `gate_v1.json` artifact; aux-loss-free SLA bias. DC-02, DC-03. Note: S2-03 (orchestrator early-exit) remains DEFERRED-blocked on the user-WIP `orchestrator.py` path (a `# SWITCH-POINT(ORCH)` Pass-2).

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S2-01** | `MoERouter.route()` widened to feature-conditioned routing + additive v2 fusion-construction seam + single-source drift guard; backward-compatible; fixes a latent `swarm`-mode advertise-drift. | FR-018, NFR-007, NFR-009 | SO-12 span | 5/5 APPROVE |
| **S2-02** | `DistilledTopKGate` — runtime gate entered ONLY via the S2-05 fail-closed verify-on-load boundary (AX-006); advisory, never drops a floored span (AX-003); offline trainer distils XGBoost survival oracle → signed `gate_v1.json`. | FR-018, DC-02 | SO-12 span | 5/5 APPROVE + control-path adversarial close RECLOSE_PASS (`wf_4664a0cd-f3a`: 35+ forge probes / 19 hostile floor variants / 400-trial fuzz; 0 upheld) |
| **S2-04** | `SLABias` — aux-loss-free latency bias on `route()` selection logits via `ExpertSpec.metadata["latency_cost_ms"]`; default-off (byte-identical static softmax, NFR-026); advisory + recall-floor-safe (AX-003). | NFR-009, NFR-010, DC-03 | RED `250312b` → GREEN `1f76bca` → REFACTOR `ff0c689` → remediation `3e9fbe7` | 6/5 APPROVE @ iter-2 (iter-1: 2 MAJOR — `reference_ms=0.0` ZeroDivisionError + `10**400` OverflowError; remediated) |
| **S2-05** | `pii_anon/moe_gate_signing.py` — HMAC-SHA256 detached signature + fail-closed verify-on-load for `gate_v1.json`; key rotation support. (Security MUST story, batched with SO-10.) | NFR-005, NFR-006, AX-006 | SO-10 span | 5/5 APPROVE (see Security hardening section) |

**Sign-off SO-12** (2026-06-03): full suite 3,291 passed / 15 skipped / 0 failed; 87.34% coverage; ruff + both-mypy clean / 131 files. SDO: NOT_YET (unchanged — a routing feature flips no guarantee).

---

## Sprint 5 — Privacy-Attack Protocols (SO-10 / SO-14 / SO-17 / SO-18 · 2026-06-01..09)

**Theme:** a real `attacks/` package behind sandboxed Protocols; Tier-3 representative re-id adversary; LiRA-shaped MIA. DC-09. Real-scale runs (≥385 personas; ≥128 shadow models) are Pass-2 DATA-track items.

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S5-04** | `eval_framework/attacks/sandbox.py` — capability + resource isolation for the attack harness; path-traversal allow-list (`..` bypass prevented). (Security MUST story.) | FR-029, NFR-024, AX-001 | RED `e4cf199` → GREEN `6deacbe` (path-traversal fix) | 5/5 APPROVE + adversarial re-close `wy3urfrhc` (0 still-broken) |
| **S5-01** | `ReidAttack` Protocol + `MiaAttack` Protocol (structural, zero call-site changes); `BaselineDeterministicReidAttack`; non-strippable NFR-016 re-id caveat stamped on every attack output; attacks import-boundary CI test. | FR-011, FR-013, NFR-016 | SO-14 span | 5/5 APPROVE |
| **S5-02** | `reid_tier3.py` — representative Tier-3 LLM-adversary (RRS/QIC/BSL de-circularized via the S5-01 Protocol seam; sandboxed); NFR-012 RRS statistical-power model (Wilson CIs; ≥385 paired personas/cell exact-rate-anchored). Real Tier-3 cohort (≥385) = DATA Pass-2. | FR-011, FR-012, NFR-012 | SO-17 span | 5/5 APPROVE |
| **S5-03** | `mia.py` — representative MIA family (LiRA-shaped + `SecretSharerReport`; NFR-013 TPR@low-FPR power model; ≥128 shadow models + canary-exposure). Real LiRA@128 + canary splits = Pass-2. | FR-013, NFR-013 | SO-18 span | 5/5 APPROVE. Suite: 3,739 passed / 16 skipped / 0 failed |

---

## Sprint 6 — Agentic Interception + BYO-SDK (SO-14 / SO-19 / SO-20 · 2026-06-04..09)

**Theme:** four-channel least-privilege guard; query-aware masking primitive; BYO-pipeline SDK adapter. DC-12, DC-13.

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S6-02** | `FourChannelGuard` (`agentic/interception.py`) — intercepts all 4 agent channels (prompt / tool-call / tool-return / assistant-turn); `NoRawPIIPersistError` fail-loud; keyed HMAC-SHA256 surrogates (iter-1 keyless-hash re-id MAJOR → fix). | FR-025, FR-026, FR-027, AX-006 | SO-14 span (iter-2 remediation) | 5/5 APPROVE |
| **S6-03** | `EncryptedSQLiteTokenStore` (`tokenization/encrypted_store.py`) — AEAD encryption at rest; AAD-bound rows; envelope-wrapped DEK; `EnvelopeKeyProvider` / `StaticTestKeyProvider`. (Security MUST story.) | FR-019, NFR-014, NFR-015 | RED `d949fab` → GREEN `de98c68` | 5/5 APPROVE + adversarial re-close `wy3urfrhc` (0 still-broken) |
| **S6-05** | `LeakageSankey` + `score_injection_resistance` (`agentic/leakage_sankey.py`) — per-channel agentic leakage counts + prompt-injection exfiltration resistance report. | FR-028, FR-029 | SO-14 span | 5/5 APPROVE |
| **S6-01** | `QueryAwareMaskingGate` (`policy/query_aware.py`) — subtractive-on-mask, default-to-mask policy (false-retention cannot occur by default); `score_query_aware_bound` (FR-024 over-redaction bound vs mask-all). Orchestrator wire-in (`# SWITCH-POINT(ORCH)`) is Pass-2 (S2-03 block). | FR-023, FR-024, DC-13 | SO-19 span | 5/5 APPROVE (iter-1 code-quality REQUEST_CHANGES → 5 MINOR remediated → APPROVE) |
| **S6-04** | `BYOPipelineRegistry` (`eval_framework/byo_pipeline.py`) — `pii_anon.byo_pipelines` entry-point discovery (never constructs engines at enumeration); `engine_predictor` bridge (bool-is-int span drops); 5 incumbent predictors (`gliner_predictor`, `presidio_predictor`, etc.); `evaluate_incumbent` (single delegation to `evaluate_external_system` — identical path BY CONSTRUCTION); `build_identical_path_leaderboard`. | FR-001, FR-002, DC-12 | SO-20 span | 5/5 APPROVE (first-pass; 0 MAJOR; 2 MINOR remediated `bbb8def`). Suite: 3,764 passed / 16 skipped / 0 failed @ 88.76% |

---

## Sprint 7 — Canonical Run, Multimodal, Multilingual, Docs (SO-15 / SO-16 / SO-21 / SO-22 / SO-23 · 2026-06-09)

**Theme:** the canonical-run keystone (flips G7; G1/G2/G4 go from PENDING to computed); latency ceilings (G5 — the last placeholder); native-format readers; multilingual fairness gate; docs discoverability with standing teeth.

| Story | Change | FR / NFR | Headline commits | Gate |
|---|---|---|---|---|
| **S7-02** | `canonical_run.py` (`evaluation/` package) — the canonical-run producer: runs `compare_competitors` + attaches G1 per-language ε / G2 deid-family fields / G4 calibration block / provenance stamp; `CanonicalRunGate` (fail-closed: `canonical_claim_run=True` only when all required fields pass validators). Thin `pii-anon canonical-run` CLI. | FR-008, NFR-005, NFR-006, DC-11 | RED `b1f61d5` → GREEN `890cb52` → REFACTOR `396da5a`, `fe45528` | 5/5 APPROVE + **7-round mandatory adversarial close** (close-4..close-9 + round-8 confirmatory). Round-8 `wf_3239f1fa-0c4` = RECLOSE_PASS 0 upheld / 517 probes. Full suite ~3,582 passed / 15 skipped / 0 failed @ 88.56%. Gate md5 walk → `1f327dd7dfad55551c87a0b9c8dfe188`. |
| **S7-04** | `latency_ceilings.py` (`eval_framework/evaluation/`) — the committed NFR-009 per-profile latency ceiling registry; `_g5_audit_latency` / `_g5_audit_pass` wired in `competitive_supremacy.py` (the LAST gate placeholder goes computed; all G1–G7 now compute on a certified run). | NFR-009, DC-02, DC-11 | Gate md5 → `3b842e81c3f03eafd11f9c655c1789a0`; producer → `d8f0f80e…` | 5/5 APPROVE + **2-round mandatory close**: round-1 `wf_2e3f36d5-afb` caught a **6th fabrication** (G5 breach-bury MAJOR — absent half short-circuited PENDING past a present-half breach); fix `3b842e8`; round-2 `wf_4c4df480-634` = CLOSE_PASS 0 upheld / 764 probes. Suite: 3,685 passed / 16 skipped / 0 failed @ 88.66%. |
| **S7-01** | `NativeReaderRegistry` + `NativeReader` Protocol + `ImageOcrReader` / `DicomReader` / `AudioReader` + `PdfTextReader` (`ingestion/native.py` + `native_pdf.py`); `pii_anon.readers` entry-point discovery; Iterator[IngestRecord] contract; LOUD on missing backends (never silent empty text). **Iter-1 caught and fixed an UNBOUNDED FLATEDECODE ZIP-BOMB** (memory-DoS ~1000x amplification) — 64 MiB per-stream chunked-inflate ceiling added (`258d3ec`). | FR-031, FR-032, FR-033, FR-034, FR-035, DC-14 | Iter-1 REQUEST_CHANGES (zip-bomb) → fix `258d3ec` → iter-2 5/5 APPROVE. Suite: 3,782 passed / 16 skipped / 0 failed @ 88.82%. | See note on FR-033/FR-035 below |
| **S7-03** | `evaluate_language_fairness` + `FairnessGateReport` (`eval_framework/metrics/fairness_gate.py`) — fail-closed powered worst-group recall-gap gate; INSUFFICIENT_POWER on <2 powered groups; dyadic-exact boundary; 100% module coverage. **Fixes provably-DEAD multilingual context keywords** (CJK/Hangul/Arabic CONTEXT_WORDS now fire via containment pass; Latin behavior byte-identical). | FR-038, FR-039, NFR-025, DC-15 | SO-22 span; Literal-type MINOR remediation `0e35431` | 5/5 APPROVE (first-pass; security-sast ZERO; AX-003 upheld BY CONSTRUCTION). Suite: 3,793 passed / 16 skipped / 0 failed @ 88.87%. |
| **S7-05** | `docs/anonymization-vs-pseudonymization.md` (NEW — the two scorer families + AX-004 no-merge invariant + vanilla-vs-swarm positioning); `docs/evaluate-your-pipeline.md` extended (SDK entry-point group + certify-a-run section with corrected artifact filename); `docs/recall-floor.md` verified live with `FloorProjectingFusion`; `docs/api-reference.md` PDLC-surfaces section (9 headline symbols); `tests/test_docs_discoverability.py` — standing docs-discoverability gate. Pre-existing broken `make docs-smoke` fixed. | FR-010, FR-001, FR-016, DC-11, DC-08 | Iter-1 REQUEST_CHANGES (phantom artifact filename in certify-a-run example) → fix; iter-2 5/5 APPROVE. Suite: 3,800 passed / 16 skipped / 0 failed @ 88.87%. | **[DOCS MUST]** |

**Notes on thin-trace FRs (D4 watch-list):**
- FR-033 (extraction-fidelity assertion per modality) and FR-035 (CI gate on multimodal reader recall regression) are SHOULD requirements carried by S7-01. They are in-tree via `NativeReaderRegistry` capability introspection but have no standalone acceptance tests beyond S7-01's coverage. Mark as DOCUMENTED-NOT-INDEPENDENTLY-GATED; Pass-2 to add per-modality CI regression fixture.
- NFR-008 (early-exit chunk latency, p50 ≤ 1 ms ∧ p95 ≤ 2 ms) — SHOULD; appears in the committed `latency_ceilings.py` registry (NFR-009 profile family) but has no dedicated named acceptance test for the early-exit path specifically. This is the O-6 SHOULD gap documented in D2.
- FR-036 (stream/batch/offline parity) — MUST; in DC-14 table; no standalone story. Carried as a Pass-2 commitment.

---

## Security Hardening Rollup

Three D6 SME Security-MUSTs were closed + adversarially hardened in SO-10 (2026-06-01). All three 5-reviewer APPROVE; a between-work-streams adversarial close (`wqhzndsp3`) caught a MAJOR the gates missed (S5-04 `..`-path-traversal allow-list bypass) and completeness gaps in S6-03 (scope_index/expires_at not in AAD; nonce test parity) → remediated in-loop → re-closed (`wy3urfrhc`, 0 still-broken).

| Security item | Story | Change | Close result |
|---|---|---|---|
| Sign + verify `gate_v1.json` | S2-05 | HMAC-SHA256 detached signature; fail-closed verify-on-load at the routing seam | Re-close `wy3urfrhc` (0 broken) |
| Encrypt token store at rest | S6-03 | `EncryptedSQLiteTokenStore` AEAD; AAD-bound rows; envelope-wrapped DEK | Re-close `wy3urfrhc` (0 broken) |
| Sandbox the attack harness | S5-04 | Path-traversal allow-list (`..` bypass blocked `e4cf199`→`6deacbe`); capability + resource isolation | Re-close `wy3urfrhc` (0 broken) |

**SDO-gate fabrication hardening (control-path closes, SO-09 through SO-16):**
Across the program arc the `competitive_supremacy.py` SDO gate and its `canonical_run.py` producer underwent 11 mandatory adversarial close rounds, discovering **11 holes and 6 fabrications** (including 1 CATASTROPHIC and 2 SHOWSTOPPERs):

- **Phantom-0.0 G2 win** (missing-competitor false-PASS) — `_finite_unit_score` + no-real-comparator PENDING guard
- **NaN-ECE / coverage>1.0 / unclamped ECE-threshold** (G4 false-PASS) — `_g4_class_bar` tighten-only clamp; per-class ECE non-negativity enforcement
- **Negative ECE fabrication** (G4, −1.0 "within bar") — per-class ECE `< 0.0` now counted as BREACH (excluded from `finite_eces`)
- **Whitespace / non-str provenance fail-open** (G7) — `_is_nonblank_str` (provenance is present only as `isinstance(str) ∧ value.strip()`)
- **`canonical_claim_run="false"` coercion** (string `"false"` coerced to bool `True`) — strict `is True` at both read sites
- **CATASTROPHIC NaN in `risk_coverage_curve` row** (NaN coverage/risk corrupted monotonicity sort → G4 PASS → CLAIM_GRADE_SOTA) — `_risk_coverage_is_monotone` validates every row via `_is_finite_number`
- **G5 breach-bury** (absent half short-circuited PENDING past a present-half breach) — `_G5Half(ok=None)` + breach-outranks-missing rule (S7-04 close-6)
- **G6 entity-coverage nested bool/+inf/0.0 mask** (SHOWSTOPPER, S7-02 close-9) — `_detected_entity_names` (valid [0,1] score > 0 required)
- **G1 recall-floor MASK via nested bool/+inf/0.0** (SHOWSTOPPER, S7-02 close-9) — same `_detected_entity_names` fix
- **CLI `IsADirectoryError`** — `path.is_file()` guard + `OSError` in parse-except
- **`10**5000` int→str digit-limit crash** — `_safe_repr`/`_safe_names` in detail f-strings

Final gate md5: `3b842e81c3f03eafd11f9c655c1789a0`. Canonical-run producer md5: `d8f0f80e113c3b5d59c06d0b5fd36fac`. Honest-input verdicts are byte-identical before and after all hardening passes.

---

## Honest Status

### SDO Verdict: NOT_YET

The current HEAD produces the following verdict on a certified canonical run:

| Guarantee | Status | Notes |
|---|---|---|
| G1 Recall floor | PASS | entities(ensemble) ⊇ entities(shared); ε ≤ 0.005 |
| G2 Pseudonymization integrity | PASS | distinct anon/pseudo families; competitors carry honest 0.0 |
| G3 Recall dominance | PASS | pii-anon composite rank-1 |
| G4 Calibration / selective-risk | PASS | per-class ECE within bar; monotone risk-coverage curve |
| G5 Latency + audit | PASS | `latency_ceilings.py` registry; no breach on committed profiles |
| G6 Raw non-inferiority (F2) | **FAIL** | F2 0.7214 vs GLiNER2 ~0.74 (threshold 0.75); **binding** |
| G7 Certified-run provenance | PASS | `canonical_claim_run=True`; full provenance stamp |
| **Overall** | **NOT_YET** | G6 binds; J=0.0 (MLE-bootstrap) |

**G6 is not a regression.** The `f2-gap-attribution.md` diagnostic (`05-testing/_diagnostics/`) confirms that the old code at `2761a27` is byte-identical to current HEAD at `use_case=default`. The raw-F2 gap vs GLiNER2 is a dataset-draw / evaluation-methodology gap (use_case=default raw-F2 vs pii-anon's census-matrix + composite), not a new code failure. See `dev-assist-artifacts/05-testing/_diagnostics/f2-gap-attribution.md` and MEMORY note `f2-gap-no-regression`.

### What 1.5.0rc1 IS and IS NOT claiming

**What it IS:**
- A feature-complete implementation of the full PDLC sprint arc (S1–S7; 25 evidenced DONE stories; 3,800 green tests; 88.87% coverage)
- A recall floor that is provably live by construction on every production fusion path
- A 3-tier rating ladder with a hard MCMC convergence gate (NFR-001)
- A fabrication-hardened SDO gate (11 holes closed; 0 fabrications possible on honest input)
- Three security MUSTs shipped and adversarially re-closed (sign/verify, AEAD token store, sandbox)
- An honest G6 FAIL verdict — it will not fabricate a SOTA claim it has not earned

**What it IS NOT:**
- CLAIM_GRADE_SOTA or PROVISIONAL_SOTA (G6 binds; see above)
- Validated against real users (all requirements are AGENT_SIMULATED; real-user Pass-2 is a documented follow-up — see release-readiness-report.md §Pass-2 commitments)
- Carrying certified benchmark numbers (numbers are PROVISIONAL: a smoke run until a full-census canonical regen + the significance pipeline repair land in Pass-2)
- Complete on the S2-03 early-exit orchestrator wire-in (blocked on user-WIP `orchestrator.py`; `# SWITCH-POINT(ORCH)` Pass-2)
- Complete on real Tier-C cloud-API runs (OpenAI Privacy Filter / Azure AI Language / AWS Comprehend) — Pass-2, no keys available in-tree

### Pass-2 Roadmap (AGENT_SIMULATED follow-ups; user-prioritized)

- Real-data canonical run + significance repair → G6 re-evaluation (the single binding constraint)
- Tier-C cloud-API evaluation (OpenAI Privacy Filter / Azure AI Language / AWS Comprehend) — required for CLAIM_GRADE
- Real NUTS J (numpyro/jax not in `.venv`; MLE-bootstrap J currently used)
- Latency threshold real-data validation: NFR-008 early-exit p50/p95; OS-matrix portability certification (NFR-022)
- Tier-3 re-id realism at ≥385 real personas / real LiRA@128 + canary splits (DATA-track; `pii-anon-eval-data`)
- OCR/DICOM/audio extraction at real backend strength (S7-01 capability-honest stubs; real backends Pass-2)
- S2-03 orchestrator early-exit wire-in (blocked on `orchestrator.py` WIP clearing)
- DC-13 orchestrator query-aware router pre-filter (`# SWITCH-POINT(ORCH)` Pass-2)
- DC-05 orchestration SHOULD: `orchestrate_incumbent_detectors` behind the recall-floored interface (FR-021)
- Five authored per-stage doc-seeds (currently absent; O-1 per D1 harvest)

---

## Sources

| Source file : section | Trace IDs supplied |
|---|---|
| `dev-assist-artifacts/MANIFEST.md:§Sign-offs (SO-07..SO-23)` | SO-07..SO-23; FR-016, FR-017, FR-018, FR-003, FR-004, FR-005, FR-006, FR-007, FR-008, FR-009, FR-010, FR-011, FR-012, FR-013, FR-019, FR-023, FR-024, FR-025, FR-026, FR-027, FR-028, FR-029, FR-031..FR-039; NFR-001..NFR-026 (partial); DC-01..DC-15 (partial) |
| `dev-assist-artifacts/MANIFEST.md:§S*-DONE sections (S1-01..S7-05 narrative)` | All story commit hashes, gate verdicts, adversarial-close workflow IDs, suite pass/fail/coverage figures |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§2 Sign-off Ledger` | SO-01..SO-23 scope one-liners |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§3 Requirement Inventory` | FR-001..FR-039 MoSCoW table; NFR-001..NFR-026 threshold table |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§4 Decision Inventory` | DC-01..DC-15 program-status column |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§5a Release Verdict` | G1–G7 verdict table; F2 0.7214; coverage 0.824; suite 3,685/3,800 counts |
| `dev-assist-artifacts/06-documentation/02-architecture/doc-architecture.md:§D-2 source mapping + §4 Honesty constraints` | D-2 caveats; honesty-constraints 1–4; O-1/O-6 gap notes |
| `dev-assist-artifacts/05-testing/release-readiness-report.md:##Verdict + ##Evidence + ##Caveats + ##Pass-2 commitments` | SHIP-WITH-CAVEATS foundation verdict; NFR matrix 2 VERIFIED + 2 PARTIAL + 22 DEFERRED + 0 FAIL; Pass-2 commitments |
| `dev-assist-artifacts/04-development/02-stories/sprint-1/S1-01-shared-layer-projector.md` | FR-016, NFR-011, AX-003; commit RED `ef85166` → GREEN `548f576`; gate APPROVE |
| `dev-assist-artifacts/04-development/02-stories/sprint-7/S7-02-canonical-run-producer.md` | FR-008, NFR-005, NFR-006, DC-11; close arc close-4..close-9; 7-round close; gate md5 `1f327dd7…` |
| `dev-assist-artifacts/04-development/development-log.md:§W6 Execution` | Sprint execution narrative; Pass-2 boundary; AGENT_SIMULATED provisional_status |
| `dev-assist-artifacts/PDLC-JOURNEY.md:§What's deferred` | Pass-2 deferred items list |

---

## Methodology

This changelog was authored directly from the canonical sprint-arc artifacts listed in the `## Sources` table above. No content was invented or inferred from imagination.

**Sourced directly (artifact-to-text transcription):**
- Story gate outcomes, commit hashes (RED/GREEN/REFACTOR), adversarial-close workflow IDs, and suite pass/fail/coverage counts were copied verbatim from `MANIFEST.md:§S*-DONE sections` and the SO YAML sign-off `scope:` fields (sourced via `doc-source-index.md:§2 Sign-off Ledger`).
- The G1–G7 verdict table is reproduced from `doc-source-index.md:§5a Release Verdict`, which itself consolidates `release-readiness-report.md:##Verdict`.
- FR/NFR/DC trace IDs were taken from `doc-source-index.md:§3 Requirement Inventory` and `§4 Decision Inventory`, cross-checked against the individual story files cited.

**Agent-inferred narrative (clearly identified):**
- The work-stream "Theme:" lines that introduce each sprint section are author-synthesized summaries of the DC scope — they are accurately derived from the DC-01..DC-15 table and the MANIFEST narrative beats, but they are not verbatim quotes from the artifacts.
- The "What 1.5.0rc1 IS and IS NOT" section is a structured synthesis of the `release-readiness-report.md:##Caveats + ##Pass-2 commitments` and the memory-bank `f2-gap-no-regression` and `g5-latency-audit-wiring` notes; it does not claim to be real-user-validated — it reflects AGENT_SIMULATED status.

**Absent doc-seeds (O-1):**
All five per-stage doc-seed narrative files are absent. The plain-language sprint narrative in this changelog is reconstructed from the SO sign-off `scope:` fields and `MANIFEST.md:§S*-DONE sections`, which are the de-facto authored spine, as noted in `doc-architecture.md:§4 honesty-constraint 2`. This is accurately described here so D4 can verify the provenance is not an invisible gap.

**Honesty constraints applied (per `doc-architecture.md:§4`):**
1. No statement claims "validated against real users" — every requirement is AGENT_SIMULATED; the Pass-2 section names the real-data follow-ups explicitly.
2. `docs/benchmark-summary.md` is NOT cited anywhere (auto-generated / volatile). Benchmark figures are sourced from `artifacts/benchmarks/benchmark-results.json` via the D1 harvest index.
3. The SDO verdict is stated accurately as NOT_YET / binding G6 FAIL (F2 0.7214 vs 0.75 threshold) in the "Honest Status" section. The f2-gap is cited as a methodology gap, not a code regression, per `f2-gap-attribution.md`.
4. `docs/pii-rate-elo-value.md` (user-WIP, O-7) is not cited anywhere in this document.

**NFR-008 / FR-033 / FR-035 SHOULD gaps (D2 O-6 + D2 §4 caveats):**
These three items are explicitly flagged in the "Sprint 7 Notes on thin-trace FRs" paragraph as DOCUMENTED-NOT-INDEPENDENTLY-GATED, consistent with the D2 architecture's instruction to mark them as such rather than omit or misrepresent their status.
