# Implementation-Ready Design — pii-anon (D6 Synthesis)

> **Stage 3, AGENT_SIMULATED** (17 agents, workflow `wfrj580jp`: 9 design-explorer proposals → 3 design-critic Pugh → 5 SME heuristic evaluators). Preserves the STRONG baseline (modular + ports-adapters + registries); all new work lands behind existing seams. Carries R10 thresholds as constraints. Cross-repo division of labor with `pii-anon-eval-data` (S5–S7). Documented alternates + switch-points retained.

## D0 Prep — preserved baseline + axioms
Baseline architecture (assessment-rated STRONG) is **stable**: `EngineAdapter`/`EngineRegistry`, `TransformStrategy`/`StrategyRegistry`, `FusionStrategy`/`build_fusion`, `ExpertRegistry`, sync/async `Orchestrator`, `ingestion/`. Axioms enforced: AX-001 no-real-PII, AX-002 determinism, **AX-003 recall-floor (the load-bearing T1 invariant)**, AX-004 anon≠pseudo, AX-005 calibrated-abstention, AX-006 least-privilege interception.

## D1 — Design Cases (15; DC↔FR/NFR traced, 0 orphan)
| DC | Title | Implements | Pillar |
|---|---|---|---|
| DC-01 | **SharedLayerProjector** — recall-floor by construction (single post-fusion chokepoint) | FR-016, NFR-011, AX-003 | swarm |
| DC-02 | MoE-router: DistilledTopKGate + rules-first Depth-1 early-exit (orchestrator hook) | FR-018, NFR-007/008/009 | swarm |
| DC-03 | Aux-loss-free SLA selection-bias (load balancer, selection-logits only) | NFR-009/010 | swarm |
| DC-04 | Reversible pseudonymization + auditable key rotation + key/state separation | FR-019, NFR-014/015 | swarm |
| DC-05 | 6 transforms + legal-regime mapping + orchestrate incumbents (recall-floored) | FR-020/021/022 | both |
| DC-06 | **RatingEnginePort + RatingEngineRegistry** (3-tier: glicko-legacy → MLE-BT → Bayesian-BT) | FR-003, NFR-001/026 | eval |
| DC-07 | Coherent significance (one joint posterior) + record-level paired outcomes + Davidson ties | FR-004, NFR-002/003 | eval |
| DC-08 | **Distinct anon-vs-pseudo scoring families** (no-merge, CI guard test) | FR-006/009/010, NFR-014/015 | eval |
| DC-09 | `attacks/` package: real Tier-3 LLM-adversary (de-circularized) + LiRA@128 MIA | FR-011/012/013, NFR-012/013 | eval |
| DC-10 | Calibration & selective-risk reporter (per-class ECE/Brier/AURC + abstention) | FR-005, NFR-017/018/019/020/021 | eval |
| DC-11 | CanonicalRunGate + provenance + CI ship/no-ship + per-language ε-gate + RecallFloorVerdictGuard | FR-007/008, NFR-006/011(ε) | eval |
| DC-12 | BYO-pipeline SDK adapter contract + identical-incumbent scoring | FR-001/002 | eval |
| DC-13 | Agentic interception: router pre-filter + query-aware gate + 4-channel least-privilege + no-raw-PII-persist | FR-023..030, AX-006 | both |
| DC-14 | Multimodal readers (`Iterator[IngestRecord]`) + per-modality benchmark + stream/batch/offline + OS parity | FR-031..037, NFR-022/023 | both |
| DC-15 | Multilingual context + fairness gate + no-real-PII + optional-dep degradation (cross-cutting) | FR-038/039, NFR-024/025/026, AX-001 | both |

## D2 Workflow / D3 UI (preserved + extended)
Workflow: linear sync + async-streaming orchestrator (unchanged); add an **orchestrator early-exit hook** at `_detect_on_text_field_async` (the ONLY place engine-skip is achievable — engines run before `merge()`). UI: Python API + Typer CLI + Make targets (unchanged); add CLI surface for BYO-pipeline scoring + canonical-run + the distinct anon/pseudo families (docs discoverability — SME MAJOR).

## D4 System / D5 Architecture — the 3 headline decisions

### DECISION 1 — Swarm MoE-router (Pugh winner: Proposal A spine + grafts; 8.4 vs 7.7 vs 7.4)
- **`SharedLayerProjector` (DC-01, the AX-003 fix):** ONE chokepoint `project(output, shared)` that both `MoEFusionStrategy` and `SwarmFusionStrategy` delegate to post-fusion. `Shared(input)` = always-on `RegexEngineAdapter` (checksum/keyword-gated, ~0.7ms/rec, 20 types) computed deterministically per chunk BEFORE any gate/threshold. Projector re-injects (offset+type+field+lang keyed) any Shared span a downstream gate dropped, tagged `provenance='shared_floor'` → `spans(output) ⊇ Shared(input)` **by construction**, closing the verified swarm.py:654/661 leak. Floor is a *projection decoupled from the router* → holds for any future router. (Graft from B: emit a `floor_violation_blocked` audit record — observable for FR-008/compliance.)
- **`DistilledTopKGate` (DC-02):** Switch/Mistral-style gate distilled offline from the *frozen XGBoost meta-learner as survival oracle* (BCE/KL over per-(entity_type,expert) survival labels; reuse RRS up-weighting + `select_f2_threshold`). Artifact `gate_v1.json` carries oracle-hash + `gate_feature_version`; **absent → static `entity_strengths` softmax (NFR-026)**. ADVISORY only (full-vs-floor weighting + early-exit selection); can never drop a Shared span.
- **Rules-first Depth-1 early-exit (DC-02):** at the orchestrator hook — chunks whose Shared spans are ALL checksum/keyword-gated exit before heavy NER (structurally provable correctness, deterministic); learned gate consulted only for UNCERTAIN chunks. (Sidesteps the pre-fusion-feature fidelity gap for the easy majority.)
- **Aux-loss-free SLA bias (DC-03, SHOULD):** DeepSeek-V3 bias on *selection logits only* (never fused confidence, never Shared membership), nudged per key-epoch toward the LOCKED 1522/753/200 power-tier latency budgets. Unit test: bias changes the selection set but NOT `EnsembleFinding.confidence` and NOT Shared membership.
- **REJECTED (alternates):** B's always-on TinyNER in the default Shared set (never-gated precision risk); C's full speculative-verification machine (v1 — alternate, switch-point if latency budgets unmet).
- **Seam corrections (verified):** `merge(list[EngineFinding])` has no text + runs after engines → engine-skip MUST be the orchestrator hook, not the fusion strategy; `register_fusion_strategy` signature `Callable[[dict,int],FusionStrategy]` is frozen (SME MAJOR) → add a richer construction seam / widen `MoERouter.route()` beyond a bare `entity_type` string; fix `available_fusion_modes()` to register the new mode in BOTH places.

### DECISION 2 — Eval rating/scoring engine (Pugh winner: Bayesian-BT spine, 8.6 vs 7.4 vs 7.1)
- **`RatingEnginePort` + `RatingEngineRegistry` (DC-06):** mirror the `engines/registry.py` entry-point pattern (group `pii_anon.rating_engines`); `PIIRateEloEngine` becomes a thin facade (preserves ~7 callers + `SystemScorecard.elo_rating/elo_rd` + the checked-in baseline leaderboard). **3-tier graceful-degrade ladder:** `glicko-legacy` (verbatim fallback, instant rollback) → `bradley-terry-mle` (pure-stdlib MM + paired bootstrap; fast PR-CI/smoke tier) → **`bayes-bt` (NumPyro Davidson/BT, NUTS — claim-grade default)**.
- **CATASTROPHIC resolved (eval-01):** NFR-001 names MCMC diagnostics literally (split-R̂≤1.01 ∧ bulk-ESS≥400/param ∧ 0 divergences). Frequentist tiers satisfy it only by-substitution → **only `bayes-bt` is claim-grade**; a hard convergence gate refuses claim-grade leaderboard emission on failure (fails loud). MLE/glicko are smoke/fallback only.
- **DC-07:** NFR-002 significance coherence holds **by construction** from one joint posterior (point∈CI, sign↔verdict, significant-iff-CI-excludes-0 cannot disagree) — eliminates the verified `elo.py:243/542/561` fabricated-outcome/fake-CI/decoupled-significance defects. Record-level paired outcomes (N·C(K,2) from `per_record_f1`) + **Davidson tie term** (SME eval-02). Separate Davidson sub-model for Tier-3 RRS (never merged — FR-010). Anchored identifiability (sum-to-zero + HalfNormal hierarchical prior).
- **HARD scope correction (critic):** the detector-side superset invariant is NOT in the rating engine. **`eval_framework.rating` imports nothing from swarm/moe/fusion/policy — pinned by a CI import-boundary property test.** The rating engine's only relationship to the floor: (1) structural non-interference; (2) it makes NFR-011's per-language ε≤0.005 a *powered posterior quantity*; (3) `RecallFloorVerdictGuard` (fail-closed) bars a floor-breaching system from claim-grade top-rank.
- **DC-11:** `CanonicalRunGate` refuses `canonical_claim_run==True` without the provenance stamp (seed/key/scope/dataset-hash/power-cell counts) — FR-008/NFR-006.
- **Deps:** `bayes-eval` optional extra (`numpyro/jax/arviz`); hard dep stays pydantic-only. **Cross-repo:** consumes eval-data's frozen `PairedComparisonSet`; the `stats/bradley_terry.py` primitive is **VERIFIED ABSENT in eval-data today → S6 dependency** (switch-point: ship a temporary local MM impl behind the port until S6 lands).

### DECISION 3 — Agentic interception (Pugh winner: Option A router pre-filter + unified floor)
- **DC-13:** router pre-filter surface (intercept + mask in `policy/router` before engines/prompt assembly) + subtractive-on-mask **query-aware gate**. Floor wrapped at the **`build_fusion` seam** (not a registered decorator — verified a decorator doesn't compose over built-in swarm/moe at fusion.py:500-512). 4-channel least-privilege interception (prompt/memory/tool-I/O/trace, AX-006); **no raw PII persisted post-masking** (FR-026). C's widened-shared-set = CI-gated WATCH item.

## Multimodal data-contract extension (DC-14)
Extend `ingestion/` readers to PDF/image-OCR/screenshot/DICOM/audio behind the SAME `Iterator[IngestRecord]` contract + lazy optional-deps; offsets map back to source coordinates (extraction-fidelity assertion). `Payload` stays text-keyed at the core; modality adapters normalize to text+coords. WebPII-style multimodal *eval* deferred (a separate follow-up).

## SME findings → MUST-address in Development (5 REQUEST_CHANGES; 1 CATASTROPHIC, ~15 MAJOR)
- **CATASTROPHIC (eval-01):** Bayesian-BT is claim-grade default; MLE not claim-grade. ✅ resolved in DECISION 2.
- **Security MAJORs (must-do stories):** (1) **sign + verify the `gate_v1.json` artifact** (auto-discovered learned gate on the control path = privilege-escalation risk); (2) **encrypt the token store at rest** (`SQLiteTokenStore` stores raw plaintext PII — tension with AX-001/key-separation); (3) **sandbox the attack harness** (ingests adversarial material).
- **Perf MAJORs:** commit numeric **p50/p95/p99 ceilings per detector class** (R10 mandate); bound the floor set-difference cost; **stable expert-id tie-break** for top-k (determinism NFR-005).
- **Eval MAJORs:** Davidson ties (eval-02); long-tail power-200 handling (eval-03); Tier-3 de-circularization lives in `attacks/` (DC-09); per-class ECE reporter (eval-06/DC-10).
- **API MAJORs:** new construction seam for feature-conditioned routing (frozen `register_fusion_strategy`); widen `MoERouter.route()`; `run_reidentification_tournament` consumes paired-comparison input.
- **Docs MAJORs (Documentation stage):** surface the distinct anon-vs-pseudo APIs (FR-010 headline); update `docs/evaluate-your-pipeline.md`; fix divergent recall-floor docs.

## Switch-points / documented alternates
Swarm: TinyNER→Shared if proven net-precision-positive; rules-first depth if distilled gate AUC weak; full speculative verify if latency budgets unmet; retire MoE floor-weight once projection-alone test is green. Eval: ship MLE-BT straight if a reviewer accepts frequentist convergence; demote bayes-bt to CI-only if dep footprint rejected. Agentic: B gateway facade if sole-writer-to-memory mandated; B reveal-only overlay if utility must reveal masked spans.

## Cross-repo division of labor
CODE owns: `routing/` (SharedLayerProjector, gate, depth, budget), `eval_framework/rating/` (port + 3 engines), `eval_framework/attacks/`, calibration/selective-risk reporter, `scoring_bridge`, BYO-SDK, agentic interception. DATA (eval-data S5–S7) owns: `stats/bradley_terry.py` (S6 — blocking for bayes/MLE), `assemble_paired_set` (S6 — Tier-3), canary splits (S6 — MIA), query-aware scorer (S5), agentic oracle (S7), the scorers/power primitives. PAPER consumes the results.

## Verification (Design → Testing)
Property test: `spans(output) ⊇ Shared` ZERO violations (all modes × gate-on/off × chunk-boundary). Per-language recall-floor ε-gate (≤0.005) wired to FR-007. CI import-boundary test (rating imports nothing from detection). N=5 determinism replay within key epoch. NFR-001 convergence gate. Distinct-family CI guard (no merged de-id field). Gate-artifact signature verification test. Token-store-encrypted-at-rest test.
