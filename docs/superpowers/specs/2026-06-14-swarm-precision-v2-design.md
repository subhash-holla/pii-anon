# Swarm Best-in-Class — Resolution + Approach B (train the dormant meta-learner)

- **Status:** Draft for SME review — 2026-06-14. **Supersedes Approach A (v1 `2026-06-14-swarm-precision-approach-a-design.md` and the earlier v2 Approach-A content): Approach A targeted a non-problem and is dropped.**
- **Priority target (unchanged):** privacy-first F2 > balanced F1 > latency.

## 1. Resolution — there is no precision problem

Two SME panels + a corrected measurement dissolved the original premise:

- The benchmark's "`pii-anon-swarm` F1 0.610 / precision 0.486 / ~3.06M FP / 3rd-of-5" was the benchmark scoring **MoEFusionStrategy** (`competitor_compare._ensemble_detector`), not the canonical swarm.
- On the **production wiring** (canonical `build_fusion("swarm")` = `FloorProjectingFusion(SwarmFusionStrategy)`, strict-v1, full 30,995-record test split — the authoritative `../pii-anon-eval-data/results/baselines/sp2-tier1-en-12` + `sp2-test-swarm` runs):

  | System | P | R | F1 | F2 |
  |---|---|---|---|---|
  | pii_anon (core) | 0.869 | 0.888 | 0.878 | 0.884 |
  | **pii_anon_swarm** | 0.859 | 0.892 | **0.875** | **0.885** |
  | gliner | 0.812 | 0.718 | 0.762 | 0.735 |
  | presidio | 0.419 | 0.563 | 0.480 | 0.527 |

  The swarm is **co-leader, far above competitors**. Per-type, sensitive recall is healthy and ≈ core (CREDIT_CARD R0.74/P0.69, DATE_OF_BIRTH R0.81/P1.00, BANK_ACCOUNT R0.79/P0.98, SSN R0.98/P0.92; PERSON_NAME R0.92 — swarm beats core). The round-2 "sensitive-recall leak BLOCKER" and the FP attribution were **artifacts of a flawed throwaway harness** (arbitration-off + flat-0.85 confidence + a 21%-test-contaminated sample), now deleted. The right measurement instrument is the **eval-data baselines harness**, not a bespoke script.

**Therefore:** Approach A (authoritative-regex + gate-tightening to "fix precision") is moot — drop it. The real opportunity is below.

## 2. The genuine best-in-class gap

The swarm only **ties** core (F1 0.875 vs 0.878) because it runs the **hand-tuned logistic fallback** — there is no trained meta-learner (`~/.pii_anon/swarm/` has ds_params/temperature/informativeness but **no `xgboost_model.ubj`**; `scripts/train_swarm.py:594-598` never produces one). The 21-feature XGBoost meta-learner that is supposed to be the swarm's brain is dormant. **Training it is the untapped lever to push the swarm clearly above core and competitors.**

## 3. Workstream 1 — benchmark-wiring fix: SWAP (user decision: the MoE wiring is a bug)

- **Change:** repoint `competitor_compare`'s `pii-anon-swarm` routing (`_core_system_worker` at competitor_compare.py:1588 → `_ensemble_detector`/`MoEFusionStrategy`) to the **canonical production swarm** — `build_fusion("swarm")` via `first_party._swarm_predictor_from_engines(_swarm_pool())` (arbitration on, native confidences). **Retire the MoE wiring for `pii-anon-swarm`.** The benchmark's swarm number then reflects the true ~0.875, not the 0.610 artifact.
- **Decided:** the MoE wiring is a bug (user). This is a measurement-correctness swap, not an add.
- **Blast radius to handle (per architecture reviewer):** swapping changes what `pii-anon-swarm` means in every downstream surface (README "pii-rate-elo" section, `benchmark-summary.md`, `benchmark-diagnostics.json`, floor-gate). Re-baselining will regenerate those — and the benchmark *script* auto-rewrites README (user-WIP) + the dataset regenerates non-deterministically, so coordinate the re-baseline and review the regenerated docs (the same caveat that produced the CI/README issues earlier).
- **Isolation:** `competitive_supremacy.py` (SDO gate) does not import `competitor_compare` (RISK-6); `canonical_run` uses `build_fusion` directly, not the benchmark — so the swap is isolated from the SDO control path. Confirm `_ensemble_detector`/`MoEFusionStrategy` is not relied on by any OTHER benchmarked system before retiring it.
- **TDD:** a test asserting `pii-anon-swarm` routes through `build_fusion("swarm")` (imports `SwarmFusionStrategy`, not `MoEFusionStrategy`); a re-baseline pass confirming the swarm's per-type recall (esp. sensitive types) does not regress vs the canonical tournament numbers.

## 4. Workstream 2 — Approach B: train the meta-learner

- **Make `train_swarm.py` actually produce `xgboost_model.ubj`** from swarm pipeline output (candidate feature vectors labeled TP/FP against gold), persisted to the auto-discovered artifacts dir.
- **Cost-sensitive objective:** the current objective is symmetric `binary:logistic`. Since recall is privacy-first but the swarm must beat core, train with FP-penalizing class weights / `scale_pos_weight`, then select the operating point by **F2 with a precision floor**.
- **Deploy `select_f2_threshold`** (implemented at `swarm_learner.py:433` but never applied) to set `emission_threshold` from a **dev-split sweep** — the meta-learner owns the emission threshold.
- **Forward-compat (panel lesson):** keep the recall floor (`FloorProjectingFusion`) a **structural always-on safety net** that the meta-learner does NOT override; do not conflate it with the emission threshold. No floor-scoping (it breaks the pinned FR-016/AX-003 superset invariant and touches the SDO G1 path).
- **Training data:** generate candidates by running the production swarm pool (**arbitration on, native confidences** — the lesson from the flawed harness) on the **dev split** (record-ids provably disjoint from the 30,995 test set); use pii-anon-eval-data v1.3.0+ (RRS/persona fields for sample weights).

## 5. Methodology (baked-in lessons from both panels)

- **Right instrument:** measure with the **eval-data baselines harness** (the source of the canonical numbers), NOT a bespoke script. Assert the swarm path under test is `build_fusion("swarm")` with the `_swarm_pool` (arbitration on, native confidences).
- **No tuning on test:** all training/threshold selection on the **dev split**; one frozen test pass for reporting; assert dev∩test record-ids = ∅.
- **Per-type + sensitive recall gate:** post-change per-type recall ≥ pre-change − ε (ε=0 hard for {US_SSN, MEDICAL_RECORD_NUMBER, NATIONAL_ID, PASSPORT, DRIVERS_LICENSE, BANK_ACCOUNT, CREDIT_CARD, IBAN, DATE_OF_BIRTH}); these excluded from any precision-for-recall trade.
- **Significance:** F2 reported with a paired-bootstrap CI + Berg-Kirkpatrick p-value (these helpers must be **built** for per-record F2 / per-type recall — the existing `compute_micro_f1_confidence_interval` is micro-F1 only).
- **Determinism + per-language:** seed training; report es/fr/de/zh/ja breakdown (the dead-CJK lesson); ≥3 pool passes mean±sd or a deterministic pool for any pool-dependent number.

## 6. Success criterion

Swarm (with trained meta-learner) **clearly exceeds core** on the reported test pass — F2 and F1 above core's 0.884/0.878 with the delta **significant** (paired bootstrap) — AND no sensitive-type recall regression vs the current swarm. Pre-register the exact baseline (current swarm F2 0.885) and the decision rule before the test pass.

## 7. Risks

- **Overfit** to dev → strict dev/test disjointness + significance.
- **Recall regression** from a precision-tilted objective → the §5 per-type/sensitive gate is a hard precondition (the swarm's value is its recall edge; do not trade it away).
- **Non-determinism** in the pool (GLiNER) → seed + multi-pass.
- **The trained model could fight the recall floor** → §4 forward-compat (floor structural, model owns emission threshold).
- **Data access / compute** for training (pii-anon-eval-data v1.3.0+).

## 8. Next

SME-review this brief (the user's standing rigor preference) — focused on the Approach-B training design and the Workstream-1 wiring fix — before any implementation. Then TDD: Workstream 1 (wiring add) first, then Workstream 2 (meta-learner training).
