# Development Log — pii-anon (Stage 4)

> **Brownfield Mode — Source Signal vs Gaps.** Real v1.4.0 library (37K LOC, 2,548 existing tests, strict CI). STRICT TDD (RED→GREEN→REFACTOR). AUTONOMOUS pass, AGENT_SIMULATED gates. **Honest status: W1–W5 planning COMPLETE; W6 execution = S1-01 flagship DONE (real code, 7/7 green); S1-02…S7 TODO (scaffolded).** Several eval stories block on `pii-anon-eval-data` S6 (verified absent today).

## W1 Preflight
- **Tech stack:** Python 3.10–3.13 (dev 3.12), `.venv`, pytest 9 + ruff + mypy --strict + coverage (`--cov-fail-under=84`), GitHub Actions CI, setuptools. Hard dep pydantic-only; optional extras for engines/bayes/attacks.
- **Legacy inventory:** migrated prior-art in `03-design/_inputs/` + `05-testing/benchmark-evidence/` (cut-line = first canonical story commit; pre-cut code not held to retroactive RED-before-GREEN).
- **DIVERGED NFRs:** 0 (R10 returned 0 DIVERGED). No DIVERGED cascade.
- **Pass-2 commitments:** all FR/NFR `provisional_status: AGENT_SIMULATED`; the user is the Pass-2 cohort at Stage 5; latency + Tier-3 thresholds flagged for real-data Pass-2.

## W2 Planning — 7-sprint plan (15 DCs → ~30 stories)
Critical path (eval-integrity + recall-floor first, per the POV): **S1-01 → S1-02 → S1-03** (recall-floor) and **S3-01 → S3-03/04** (rating engine).

| Sprint | Theme | Stories | Status |
|---|---|---|---|
| **S1** | Recall-floor foundation (DC-01) | S1-01 SharedLayerProjector · S1-02 wire projector into MoE+Swarm fusion (post-merge delegate) · S1-03 per-language recall-floor CI ε-gate · S1-04 property-test infra (hypothesis migration) | **S1-01 DONE**; S1-02/03/04 TODO |
| **S2** | MoE-router (DC-02/03) | S2-01 widen `MoERouter.route()` + construction seam · S2-02 DistilledTopKGate + offline distillation trainer · S2-03 rules-first Depth-1 early-exit (orchestrator hook) · S2-04 aux-loss-free SLA bias · **S2-05 sign+verify `gate_v1.json` [SECURITY MUST]** | TODO |
| **S3** | Eval rating engine (DC-06/07) | S3-01 `RatingEnginePort`+`RatingEngineRegistry`+glicko-legacy facade · S3-02 bradley-terry-mle [⛓ DATA S6 `bradley_terry.py`; temp local MM until then] · S3-03 bayes-bt (NumPyro) claim-grade + convergence gate · S3-04 coherent significance + record-level paired outcomes + Davidson ties · S3-05 CI import-boundary test (rating ⊄ detection) | TODO |
| **S4** | Privacy-eval families (DC-08/10/11) | S4-01 distinct anon/pseudo scoring APIs + no-merge CI guard · S4-02 CanonicalRunGate + provenance + RecallFloorVerdictGuard · S4-03 calibration & selective-risk reporter (per-class ECE/Brier/AURC) | TODO |
| **S5** | Attacks (DC-09) | S5-01 `attacks/` skeleton + `ReidAttack` protocol · S5-02 real Tier-3 LLM-adversary (RRS/QIC/BSL, de-circularized) [⛓ DATA S6 `assemble_paired_set`] · S5-03 full-power MIA LiRA@128 + Secret-Sharer [⛓ DATA S6 canary] · **S5-04 sandbox attack harness [SECURITY MUST]** | TODO |
| **S6** | Agentic + BYO-SDK (DC-12/13) | S6-01 router pre-filter + query-aware masking gate · S6-02 4-channel least-privilege interception + no-raw-PII-persist (AX-006) · **S6-03 session pseudonyms + token-store encryption-at-rest [SECURITY MUST]** · S6-04 BYO-pipeline SDK adapter + identical-incumbent scoring · S6-05 agentic leakage-Sankey + prompt-injection resistance | TODO |
| **S7** | Multimodal + portability + release (DC-14/15) | S7-01 native-format readers (PDF/image/screenshot/DICOM/audio) · S7-02 per-modality benchmark + stream/batch/offline parity + OS-matrix · S7-03 multilingual context feature + fairness gate · S7-04 commit numeric latency ceilings (p50/p95/p99 per class) · **S7-05 docs discoverability [DOCS MUST]** · release gate | TODO |

**Cross-repo blockers (⛓):** S3-02, S5-02, S5-03 depend on `pii-anon-eval-data` S6 (`stats/bradley_terry.py`, `assemble_paired_set`, canary splits) — VERIFIED ABSENT today. Mitigation: ship temp local impls behind the ports until S6 lands (switch-points in design).

## W3 Quality
- **Reviewers:** the 8-specialist set; **5-gate cascade** (story/epic/sprint/release; contributor-readiness S8 deferred — OSS launch). Story-gate default = code-quality + axiom-compliance + traceability + (conditional security/performance per touched paths/tags).
- **TDD:** strict, RED-before-GREEN (git-evidenced). **NFR-verification matrix:** the 26 NFRs → Stage-5 verification (latency/recall-floor/power/calibration/parity).
- **Security MUST stories** (from SME REQUEST_CHANGES): S2-05 (gate signature), S5-04 (harness sandbox), S6-03 (token encryption-at-rest).

## W4 Testing setup
Extend the existing pytest suite (don't rebuild). Add `hypothesis` to the dev extra (S1-04) to migrate the seeded-random property tests to `@given`. Reuse `make benchmark*` for NFR verification; add the recall-floor property suite + per-language ε-gate to CI (S1-03). Per-modality + agentic-leakage harnesses scaffolded for S5/S7.

## W6 Execution — completed this pass
### ✅ S1-01 SharedLayerProjector (DC-01 / FR-016 / NFR-011 / AX-003) — DONE
- **RED** `ef85166` → **GREEN** `548f576`. `src/pii_anon/routing/shared_layer.py` + `tests/test_shared_layer_projector.py` (7 tests). Enforces `entities(output) ⊇ entities(shared)` by type-carrying re-injection; closes the swarm.py:654/658-660 leak; decoupled from any router; empty-shared no-op; deterministic.
- **Verification:** 7/7 new tests green (incl. seeded-random superset-invariant property test, 2,000 cases, ZERO violations; the AX-003 leak-closure test; determinism test). ruff clean; **mypy --strict** clean. swarm/fusion/moe suites (78 tests) unaffected. No public-API change.
- **Story-gate review:** `_reviews/story/S1-01-gate.yaml` → **APPROVE**.

### ⏭ Remaining (S1-02 … S7): TODO — scaffolded above, ready for `/dev-assist-story-claim`.
The flagship discharges the single load-bearing MUST (FR-016/NFR-011/AX-003) this redesign is gated on. The next-highest-leverage stories are S1-02 (wire the projector into both fusion strategies — makes the floor live in production) and S3-01/03 (the rating-engine port + Bayesian claim-grade engine — the eval-integrity foundation).

## Epistemic honesty
This pass delivered the recall-floor foundation as **real, tested, type-checked production code** and a complete sprint plan. It did NOT implement the full 4-theme redesign (MoE-router ML, Bayesian MCMC engine, attacks/, agentic, multimodal) — that is genuinely multi-sprint and partly blocked on eval-data S6. Status is reported honestly per-story; nothing is marked DONE that isn't green in-tree.
