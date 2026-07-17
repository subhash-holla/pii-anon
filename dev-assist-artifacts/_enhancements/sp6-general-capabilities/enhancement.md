# Enhancement Amendment: sp6-general-capabilities

> Managed by /dev-assist-enhance. Opened 2026-07-10. Status: **IN_PROGRESS**.
> User steer: "use the external datasets to help improve our swarm and vanilla offerings…
> use these datasets and our capstone one (pii-anon-eval-data) to come up with GENERAL
> enhancements… I am open to retraining the swarm offering if needed."

## Classification

`new-capability` + `defect-fix` (multi-class). The generality bar is BINDING: an enhancement
qualifies only if the failure class it addresses appears across ≥2 datasets (or is a
domain-general capability gap), never a single-benchmark convention. Retraining the swarm
meta-learner is now AUTHORIZED — which changes the Approach-B calculus: external TRAIN splits
give leakage-free training data that teaches generalization (the old blockers were
home-corpus train-on-test overlap + home-floor-lock).

## Phases

| Phase | What | Discipline |
|---|---|---|
| A | Cross-dataset FN/FP failure-class mining: both detectors × 6 datasets (5 external + home dev) → ranked general-enhancement candidates with evidence | evidence-before-design; train/dev splits only |
| B | The swarm NER-channel decision: (1) train the meta-learner on external+home TRAIN splits, (2) confidence-aware NER-singleton pass-through, (3) emission-gate recalibration — DIVERGE/CONVERGE, then implement the winner | AX-003 floor by construction; leak-direction (a fusion change that DROPS differently is the dangerous direction); mandatory adversarial close on the fusion/production-path change |
| C | Vanilla general pattern fixes (GPS-on-date-fragments, two-cap-word-on-headers/labels, US-only ZIP, + Phase-A findings) | leak-safe additive or FP-guard-with-recall-proof; tuned on external TRAIN/dev only; home-dev regression check per change |
| D | Full re-measurement (home dev + 5 externals × both detectors), tuned rows added to `docs/external-validity-report.md` as SEPARATE labeled rows (zero-shot rows immutable), commits + sign-off | zero-shot-first reporting discipline |

## Known infrastructure gaps if Phase B converges on retraining (from the Approach-B panel)

train_swarm loads the full corpus (home test-id overlap → must swap to disjoint TRAIN splits);
no candidate-labeling API (merge() discards SpanCandidates); degenerate Dawid-Skene priors;
model artifact not shipped; `build_fusion("swarm")` passes no config so a swept
emission_threshold never reaches inference (persistence gap). Any trained gate artifact follows
the S2-05 sign+verify-on-load discipline (control-path artifact → close).

## Verified Phase-B seam facts (2026-07-10, read from source — not recollection)

- The NER-suppression mechanism, exactly: `swarm.py` Layer-4 emits a candidate iff
  `meta_score ≥ emission_threshold (0.50)`, AND for `SEMANTIC_TYPES` (PERSON_NAME,
  ORGANIZATION, LOCATION, DATE_OF_BIRTH, ADDRESS, USERNAME, PHONE_NUMBER, EMAIL_ADDRESS,
  CREDIT_CARD) additionally `corroboration_count ≥ corroboration_min` OR
  `meta_score ≥ corroboration_override_threshold (0.85)`. With no `xgboost_model.ubj`,
  `meta_score` comes from `_logistic_fallback_score` = sigmoid(2·ds_conf + 0.5·min(corr,4)
  + 0.8·regex + 0.3·structured − 2.0); a single-engine non-regex semantic candidate caps at
  sigmoid(2·1.0 + 0.5 − 2.0) = **0.62 < 0.85** — structurally unreachable, for ANY engine
  confidence. The engine's own confidence is not even an input to the fallback.
- The floor (`FloorProjectingFusion(SwarmFusionStrategy())` at `fusion.py:632`) re-injects
  regex-oss spans only — indifferent to ADDITIVE emission changes (AX-003 holds by
  construction for any change that only adds emissions).
- Training entry points exist: `swarm.py:328 train_em` (Dawid-Skene EM) +
  `swarm_learner.py:276 train` (XGBoost, `FEATURE_VERSION=3`); model discovery looks for
  `artifacts/xgboost_model.ubj`. `build_fusion("swarm")` constructs `SwarmFusionStrategy()`
  with NO config → any tuned `SwarmConfig` (thresholds) currently never reaches production
  inference (the persistence gap is REAL and must be closed by whichever option wins).

## Delta table

| # | Delta | Status |
|---|---|---|
| A | Phase-A mining (`wf_6aa8792e`, 6 miners + synthesis; piibench miner lost to a server error — its zero-shot data still informs via sp4): **13 ranked candidates, all hitting 4–5 datasets**; 24-item dropped-as-non-general list; evidence at `_evidence/phase-a-mining-synthesis.json` + miners journal. Headline: the pool engines already FIND the missing gold — TAB union counterfactual (regex∪gliner∪presidio) relaxed R 0.091→0.582 / F2 ≈0.10→≈0.55; presidio solo TAB R=0.530 with ZERO surviving fusion | DONE |
| B1 | Presidio label normalization (candidate 4a): raw PERSON/NRP/… → pool vocabulary in the adapter — presidio findings now type-vote with the pool and survive downstream maps | DONE (TDD) |
| B2 | GLiNER label extension (candidate 4b): +organization/location/city/occupation → ORGANIZATION/LOCATION/JOB_TITLE (594 gretel + ~950 TAB + 294 nemotron + 95 home ORG gold previously had NO ML channel) | DONE (TDD) |
| B3 | GLiNER window-start word-alignment + boundary-snap emission hygiene (the mining caught mid-word spans "Col⟨leen Redding⟩" — my sp4 windowing aligned ends but not overlap re-entries; snap-outward-never-inward = over-masking-safe) | DONE (TDD) |
| B4 | **Fusion single-engine acceptance** (candidate 2): `SwarmConfig.single_engine_min_confidence` per-type bars (0.80–0.90; checksum types deliberately absent) + an ADDITIVE Layer-4 branch emitting a non-regex singleton when its engine-own confidence clears the bar (the old gates were structurally unreachable for ML singletons: fallback meta caps at 0.62 < 0.85 override). Ordinary-gate candidates emit byte-identically; acceptance-emitted findings carry the engine-own confidence + an explanation marker | DONE (TDD) |
| C7 | GPS decimal-requirement (candidate 7 slice): both lat/lon halves require decimals — kills the date-fragment FP class ("15/09"; Nemotron coordinate P=0.072, home P=0.157) with zero home recall cost (77/77 home dev golds carry decimals) | DONE (TDD) |
| M | **The home-gate tuning loop (3 iterations, all preserved):** loose config FAILED (F2 −1.7pp, +14.8k FPs; `sp6-dev-swarm-channel`) → tightened bars + trimmed map −0.43pp with damage isolated to presidio-normalization-induced NER junk-pair corroboration (+2,713 PERSON FPs / −16 TPs; `sp6-dev-swarm-tight`) → shared `passes_ner_span_hygiene` at BOTH NER adapters (field-label veto + single-token bar, mining candidate 1) → **GATE PASS: F2 0.8925 vs 0.8928 baseline (−0.0003), R flat, +191 FPs/+3 TPs** (`sp6-dev-swarm-final`) | DONE |
| M2 | `SwarmConfig.anonymization_profile()` shipped: the quasi-identifier singleton map (LOCATION/DATE_TIME/NATIONALITY/JOB_TITLE) for document-anonymization workloads — the loose-config evidence (TAB 0.101→0.150 relaxed F2, +50%) belongs to this profile, reported as a separate labeled row | DONE |
| M3 | ~~Final external re-runs~~ SUPERSEDED — the close proved those runs measured an INERT channel; re-measure after remediation | KILLED |
| CL | **★ MANDATORY close round 1 (`wf_70092469`, 1,004 probes): CLOSE_FAIL — 2 MAJOR + 2 low, ALL REMEDIATED.** (1) **MAJOR/LEAK: my GPS narrowing was the sp2 showstopper class** — the differential probe proved 6/6 previously-masked coordinate pairs ("41, -87", "40.7, -74") reached production UNMASKED via the floor engine (nothing downstream restores a dropped regex span). Fix: pattern REVERTED to permissive on the masking path; the date-fragment precision drop moved to `_drop_undecimaled_gps` under `eval_cross_type_arbitration` (eval-only, exactly the sp2 discipline). (2) **MAJOR/INERT: the acceptance channel never fired in production** — Layer 3 replaces findings with TEMPERATURE-SCALED copies before Layer 4 (this machine's certified temperature.json: gliner raw 0.98→0.828 < every bar) and my tests used a non-production engine id ('gliner' ≠ 'gliner-compatible'). **The earlier "home gate PASS at zero cost" was the channel being OFF — claim retracted.** Fix: `SpanCandidate.raw_confidences` captured pre-scaling; the bar reads engine-OWN raw confidence (the documented semantics); tests pinned with the real adapter id + a harsh-scaler parity test. (3) low: NaN/negative/bool bars silently disabled the gate (accept-everything) → invalid bars now REJECT. (4) low: wrong-typed acceptance map from from_json crashed the first merge() → fail-loud at load (`__post_init__`). 22 sp6 tests + 249-test regression surface green | DONE |
| M4 | **★ Home-dev gate with the channel GENUINELY live: BOTH detectors IMPROVED** — vanilla F2 0.8916→0.8927 (+0.0011), swarm 0.8928→**0.8952** (+0.0023; +350 TPs / +266 FPs — the raw-confidence acceptance nets positive at home). `sp6-dev-live-channel/` | DONE — GATE PASS |
| CL2 | **Close round 2 (`wf_f218a51a`, 9,678 probes): R1–R4 all CONFIRMED** (floor 0/299 hostile + 250-fuzz; differential 410 pre-change emissions, ZERO coverage losses) **but 2 NEW upheld: (MAJOR) presidio MEDICAL_LICENSE→NPI_NUMBER remap = a SECOND leak-direction inversion** on the weighted_consensus/union_high_recall orchestrator paths (supported→unsupported label; conf-1.0 medical-license emission stopped being maskable) — invisible to the swarm-path differential; **fixed** (remap removed; mechanical whitelist audit: every other remap is a masking GAIN; the inversion CLASS pinned for all entries) + (MINOR) 400-digit JSON-int bar → OverflowError in merge() (`float(bar)` before isfinite — the historic 10**400 class) — **fixed** (reject fail-closed) | DONE |
| CL3 | **★ Close round 3 (`wf_2a986ebf`, 223 probes): CLOSE_PASS — 0 upheld, all anchors pass.** Zero inversions across all 32 presidio + 17 gliner labels on ALL FOUR fusion modes (parity sweeps lost zero supported spans; 6 masking GAINS confirmed); 26-shape bar-poisoning matrix all reject fail-closed; the channel proven NON-VACUOUSLY live in production (raw 0.95 accepted where scaled 0.767 < bar). One PRE-EXISTING residual (huge-int engine confidence crashes `_build_candidate` at HEAD too — outside sp6) flagged as background task `task_935f23f3`. **CLOSE STATE: certified clean after 3 rounds / 10,905 total probes.** | DONE — CLOSE_PASS |
| M5 | **★ Live-channel externals (default config): EVERY dataset improved** — ai4privacy 0.237→0.267 / gretel 0.389→0.439 / nemotron 0.324→0.335 / piibench 0.189→0.196 / TAB 0.101→0.138 (relaxed F2). **TAB anonymization-profile row: 0.491 relaxed / 0.432 strict — ~4.9× over pre-sp6, ~90% of the union counterfactual.** Broadcast report updated with labeled rows (zero-shot rows immutable). Caveat: the external run carried the pre-CL2-fix presidio remap in-process (NPI not in any external PRED_MAP → no effect on these numbers) | DONE |
| — | Remaining vanilla candidates (1 Title-Case FP stack, 3 labeled-field bridge, 5 date grammar, 6 name hygiene, 8 ORG grammar, 10 address, 11 structure pre-pass, 12 eval-harness, 13 retrain-features) — next tranche, evidence preserved | TODO |

## Invariants

Leak-direction; AX-003/FR-016 recall floor BY CONSTRUCTION (floor tests + close); SDO gate +
canonical producer untouched without the mandatory close; test splits NEVER mined or tuned on;
every reported tuned number labeled as tuned with its training provenance.
