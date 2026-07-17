# S4-CS-01 — CompetitiveSupremacyGate (the SOTA-Dominance Objective; program completion criterion)

> **Cold-pickup invariant**: executable cold. This gate IS the program's optimization function + the definition of "enhancement run complete." The guarantee-check + verdict + binding-constraint logic is **pure-Python over the benchmark JSON + a posterior summary** → fully testable in-tree with synthetic fixtures. Tier-C real-API adapters + the real bayes posterior are `importorskip`/Pass-2-gated (the honesty boundary).

| Field | Value |
|---|---|
| Epic | **E-CS Competitive-Supremacy** (new; DC-11 CanonicalRunGate / RecallFloorVerdictGuard family) |
| State | **DONE** (claimer=dev-assist-development-executor; claimed_at=2026-06-01; review_at=2026-06-01; gate=APPROVE 6/6 @ iter-2 [1 MAJOR remediated]; done_at=2026-06-01; scaffold — phased; G1/G3/G6/G7 + verdict machine + binding reporter + RecallFloorVerdictGuard LIVE; G2←S4-01/G4←S4-03/G5←S5-S6 tracked successors) |
| provisional_status | AGENT_SIMULATED (CLAIM_GRADE verdict is itself gated on a regenerated canonical run — see G7) |
| Implements | FR-007/FR-008 (canonical-run gate + provenance), NFR-006 (canonical run), the SDO objective J; consumes FR-016/NFR-011 (G1 floor), FR-009/010 (G2), FR-005/NFR-017 (G4), latency NFRs (G5), FR-003/004/NFR-001/002 (G7) |
| Traces | Program AMENDMENT — "SOTA-Dominance Objective (SDO)"; Design DC-11. POV: dominate-where-claimed (moat axes) + non-inferior on raw F1. |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[AUDIT]` `[INTEGRATION-TEST]` |
| Files owned | `src/pii_anon/eval_framework/evaluation/competitive_supremacy.py` (new — the gate), `src/pii_anon/eval_framework/evaluation/competitor_tiers.py` (new — Tier-R/Tier-C registry incl. gliner2 + Tier-C cited metadata + citations), CLI surface (a `pii-anon sdo`/`supremacy` report target in `src/pii_anon/cli.py`), an additive `bradley_terry.paired_bootstrap_draws` for the MLE-bootstrap J-fallback, `tests/test_competitive_supremacy.py` (new), `tests/test_competitor_tiers.py` (new). **`evaluation/competitor_compare.py` is NOT owned/modified** — see §2a RISK-6. |
| Depends on | S3-03 DONE (joint posterior for J), S3-04 DONE (`rank_one_probability` = J; coherent significance for G7); G2 needs S4-01 (anon/pseudo scorers); G4 needs S4-03 (calibration/selective-risk reporter). |
| Blocks | program completion (M4/M7) — this gate's `CLAIM_GRADE_SOTA` verdict IS "done." |
| Size | L (phased — ship skeleton + computable guarantees now; complete J + G2/G4 as deps land) |

## 1. Intent
Make the program optimize ONE explicit objective: **pii-anon is provably better than every well-known competitor pipeline, with MINIMUM GUARANTEES on the axes where we claim an advantage** — strict dominance on moat axes (reversibility, recall, calibration/selective-risk, audit), non-inferiority on raw detection F1 (the explicit OpenAI-≈0.96 honesty carve-out). Implement it as a first-class, code-checkable gate (`CompetitiveSupremacyGate`) that consumes `artifacts/benchmarks/benchmark-results.json` + the bayes-bt joint posterior, emits exactly one verdict `{CLAIM_GRADE_SOTA | PROVISIONAL_SOTA | NOT_YET}`, and ALWAYS prints the single **binding constraint** (which guarantee failed, or how far J is from the bar) so the program always knows the next thing to fix. This gate IS the definition of "enhancement run complete."

## 2. Competitor scope — tiered, honesty-gated (`competitor_tiers.py`)
- **Tier-R (run now):** `presidio`, `scrubadub`, `gliner`, **+ a new `gliner2` (GLiNER2-PII) adapter**. SDO is computed vs Tier-R for **PROVISIONAL** status.
- **Tier-C (cited, MUST-run-before-claim):** `openai-privacy-filter`, `azure-ai-language`, `aws-comprehend` — registered in `competitor_tiers.py` (NOT in `_COMPETITOR_META` — RISK-6, §2a). Until each is run (or explicitly **waived with a documented reason recorded in the gate**), `CLAIM_GRADE` is BLOCKED. Carry the unrun set as a visible **"honesty boundary"** in the gate output.
- Registry shape: `{name: {tier: R|C, package|api, citation, run_status: RUN|UNRUN|WAIVED, waiver_reason?}}`. Tier-C adapters need real APIs/keys → their runs are **Pass-2** (never agent-simulated as real — methodology invariant). The gate reads `run_status` from the benchmark JSON / a tier-status sidecar; it never fabricates a Tier-C result.

## 2a. Pre-claim de-risk (verified against live code + artifacts 2026-06-01)
- **RISK-6 (CRITICAL — `_COMPETITOR_META` is OFF-LIMITS):** the legacy file is `src/pii_anon/evaluation/competitor_compare.py` (the **sibling `evaluation/` package**, distinct from the gate's new home `eval_framework/evaluation/`). `_COMPETITOR_META.keys()` IS the run/expected-competitor source — `detector_factories_keys = list(_COMPETITOR_META.keys())` (`competitor_compare.py:2147`) and `expected = expected_competitors or list(_COMPETITOR_META.keys())` (`:3029`, `:3348`). Adding gliner2/Tier-C entries there would pull un-runnable competitors into the run/expected set → real behaviour change + failed runs. **So the Tier-R/Tier-C registry lives ENTIRELY in the new `competitor_tiers.py`; `competitor_compare.py` stays byte-identical.** Derive `run_status` from the benchmark JSON's existing `available_competitors`/`unavailable_competitors`/`expected_competitors` + per-system `available` flag — never fabricate.
- **Benchmark JSON read-paths (confirmed):** `data["run_metadata"]["canonical_claim_run"]` (== `False` today → G7 fails → verdict NOT_YET, the #1 binding constraint). G7 provenance: `run_metadata.{git_sha,dataset_sha256,matrix_sha256,timestamp_utc}` (all present). Per-system: `data["systems"][i].{system,recall,precision,f1,per_entity_recall,composite_score,elo_rating,qualification_status,latency_p50_ms,dominance_pass_by_profile,available,citation_url}`. Also `data.{floor_pass,profile_results,statistical_tests,expected_competitors,available_competitors,unavailable_competitors}`. Systems today: gliner, pii-anon, pii-anon-swarm, presidio, scrubadub.
- **Expected verdict on today's artifact (pin in the one real-artifact test, value-independent):** G3 recall-dominance PASSES (pii-anon-swarm 0.818 ≥ max competitor gliner 0.658); G6 F2 non-inferiority PASSES (core F2 ≈0.78 ≥ best Tier-R ≈0.70 − ε_F); G1 structural; **G7 FAILS on `canonical_claim_run=False`** → verdict **NOT_YET**, `binding_constraint` = `canonical_claim_run=False`.
- **J dependency LIVE:** `significance.rank_one_probability` (S3-04 DONE) is J. The in-tree MLE-bootstrap fallback needs an additive `bradley_terry.paired_bootstrap_draws(records, b, *, seed) -> NDArray[(b, n_systems)]` (the existing `paired_bootstrap` returns CIs, not raw θ rows) feeding `rank_one_probability`; label `j_source: bayes|mle-bootstrap|unavailable`.
- **Three-valued Gk logic:** PENDING (`None`) for G2←S4-01, G4←S4-03, G5 (latency/interception). PENDING never blocks PROVISIONAL but ALWAYS blocks CLAIM_GRADE — never `all(g.passed ...)` over `None`.
- **Gate placement + boundary:** gate lives in `eval_framework/evaluation/` (clean of swarm/moe/fusion/policy). Scope the new import-boundary assertion to the 2 new gate modules — do NOT broaden to all of `evaluation/` (`competitor_compare.py:909` legitimately imports `moe`).

## 3. Hard guarantees G (ALL must pass; STRICT dominance on moat axes, NON-INFERIORITY elsewhere)
Each guarantee is a pure function `(benchmark_json, posterior?, config) -> GuaranteeResult{passed, axis, observed, bar, binding_detail}`.
- **G1 Recall-floor by construction** (DONE/S1): `entities(ensemble) ⊇ entities(shared)` ∧ per-language ε ≤ 0.005. Wire to the S1 floor invariant + the S1-03 ε-gate artifact (`floor-gate-report.md`). **computable now.**
- **G2 Pseudonymization-integrity / reversibility**: pii-anon STRICTLY dominates every benchmarked system (distinct anon-vs-pseudo families, FR-009/010 — the empty quadrant; incumbents ≈ 0). **needs S4-01 scorers.**
- **G3 Recall dominance**: swarm recall ≥ max(competitor recall). **computable now** (benchmark JSON `recall`).
- **G4 Calibration / selective-risk**: per-class ECE ≤ R10 target ∧ AURC/selective-risk ≥ best competitor (calibrated abstention, AX-005). **needs S4-03 reporter.**
- **G5 Audit + orchestration**: p50/p95/p99 within the committed power-tier latency budgets (NFR latency tiers) ∧ auditable key rotation + 4-channel least-privilege interception present. **latency computable now**; interception presence flag from S6.
- **G6 Non-inferiority on raw detection**: pii-anon(core) F2 ≥ best Tier-R competitor F2 − ε_F (ε_F default 0.01) ∧ entity coverage ≥ 0.80. NOT required to beat OpenAI's cited ≈0.96 raw F1 — explicit honesty carve-out, **recorded in the gate output**. **computable now** (need F2 = (1+4)·P·R/(4P+R) from benchmark per-record or P/R).
- **G7 Certified run**: `canonical_claim_run == True` (full provenance stamp: seed/key/scope/dataset-hash/power counts) ∧ coherent significance BY CONSTRUCTION (bayes-bt; NFR-001 convergence) ∧ **RecallFloorVerdictGuard** (a floor-breaching system can never top-rank). **provenance computable now**; coherent-sig ← S3-04; convergence ← S3-03 (DONE).

## 4. Maximization target J + steering margins
- **J = P( rank(pii-anon) = 1 on the composite | bayes-bt joint posterior )** = `significance.rank_one_probability(posterior, "pii-anon")` (S3-04). The real posterior needs numpyro (`bayes-eval`) → the **claim-grade J is importorskip/Pass-2**; an in-tree **J-fallback** computes rank-1 probability from the **MLE-BT paired bootstrap** (S3-02 `paired_bootstrap`) so J is always reportable (labeled `j_source: bayes|mle-bootstrap`). 
- **Secondary steering margins (logged, NOT the gate):** composite margin Δ = composite(pii-anon) − max(competitor); rating/elo margin; `dominance_pass_by_profile` count (already in `SystemBenchmarkResult`); `entity_type_wins`.

## 5. Completion predicate (the verdict state machine)
- **CLAIM_GRADE_SOTA** ⟺ `canonical_claim_run` ∧ (G1..G7 all pass) ∧ J ≥ 0.95 ∧ (Tier-R ∪ Tier-C all RUN or WAIVED-with-reason).
- **PROVISIONAL_SOTA** ⟺ same but Tier-C not yet run (Tier-R only).
- **NOT_YET** ⟺ otherwise — the gate reports the **binding constraint(s)** + current J + per-axis gaps as the next-action target.
Always emit `binding_constraint` (single most important failing item, priority: canonical-run → any failed G → J gap → unrun Tier-C) so the program always knows the next thing to fix.

## 6. Phased build (honest dependencies — this story ships the skeleton)
- **NOW (this story):** `competitive_supremacy.py` skeleton + `competitor_tiers.py` registry + **G1/G3/G6/G7** (computable from the current benchmark JSON + glicko/MLE) + the verdict state machine + the binding-constraint reporter + J-fallback (MLE-bootstrap). Wire into CI as a **NON-BLOCKING report** immediately (BLOCKING only for canonical-claim emission). gliner2 Tier-R adapter + Tier-C metadata adapters (UNRUN, honesty boundary).
- **J upgrade:** swap J to the bayes posterior once a `bayes-eval` run exists (S3-03 engine is DONE; the run is Pass-2).
- **G2 / G4:** complete at S4-01 (anon/pseudo scorers) / S4-03 (calibration/selective-risk reporter).

## 7. Invariants to preserve
- POV measurement-first: Pillar-1 (pii-rate-elo) is the co-equal headline + the measurement instrument; the SDO ENCODES dominate-where-claimed + non-inferior-on-raw-F1.
- AX-003 recall-floor by construction; AX-004 anon vs pseudo NEVER merged; rating engine import-isolated from detection (the gate lives in `eval_framework/evaluation`, may import `rating` + read JSON, but NOT `swarm/moe/fusion/policy` — boundary test).
- ALL current README/benchmark numbers are PROVISIONAL (50-sample smoke, `canonical_claim_run=False`) → the gate's CLAIM_GRADE is itself gated on a regenerated canonical run (G7).

## 9. Test-type tags
`[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[AUDIT]` `[INTEGRATION-TEST]` — reviewers: code-quality + axiom-compliance + traceability + requirements-coverage (always for this gate); security-sast (CLI/load path); performance (latency-budget G5 reads).

## 12. Definition of Done (this phase)
- [x] **RED**: `tests/test_competitive_supremacy.py` — verdict state machine (synthetic benchmark JSON + synthetic posterior → CLAIM_GRADE/PROVISIONAL/NOT_YET per the predicate); each Gk pass/fail with a synthetic fixture; binding-constraint priority; J-fallback rank-prob; **the honesty carve-out** (G6 does NOT fail when a Tier-C raw-F1 exceeds pii-anon, only on Tier-R non-inferiority); Tier-C-unrun ⟹ CLAIM_GRADE BLOCKED ⟹ at most PROVISIONAL. `tests/test_competitor_tiers.py` — registry tiers, run_status, waiver-with-reason. Written first & failing (RED `512311b`).
- [x] **GREEN**: gate + registry + G1/G3/G6/G7 + verdict machine + binding-constraint reporter + J-fallback; gliner2 adapter metadata (graceful if pkg absent); Tier-C metadata (UNRUN). Non-blocking CI report wired (`pii-anon supremacy` prints verdict + binding constraint; exit 0 unless `--canonical-claim`). (GREEN `b8c9d9b`.)
- [x] **G2/G4/J-bayes** left as explicit tracked successors (named in the gate output `axes_pending`: `G2←S4-01`, `G4←S4-03`, `G5←S5/S6`; J-bayes path importorskip / Pass-2).
- [x] **Quality**: full suite green (2864 passed, 15 skipped — Tier-C/real-API/numpyro SKIP; 9 deselected = performance); ruff + mypy --strict clean; import-boundary GREEN (gate ⊄ detection internals); coverage 86.13% ≥ 84%.
- [x] **Honesty**: the gate output carries the visible Tier-C honesty boundary (`unrun_tier_c`) + the OpenAI raw-F1 carve-out (`carve_out_note`, always emitted) + `canonical_claim_run=False` banner. Marks nothing CLAIM_GRADE without a canonical run.
- [x] **Untouched**: `competitor_compare.py` **byte-identical** (md5 `7cae16c89f4c97136e1a12394dae2025` unchanged). The only additive edit to an existing rating file is `bradley_terry.paired_bootstrap_draws` (new method, existing methods untouched). user-WIP md5 unchanged (`orchestrator.py`, `test_moe_enhancements.py`, `benchmark-diagnostics.json`, `README.md`, `docs/*`, and the READ-ONLY `artifacts/benchmarks/*` — never written).
- [ ] **Story-gate APPROVE** (`_reviews/story/S4-CS-01-gate.yaml`) — pending orchestrator dispatch.

## Evidence (filled on completion)

**State**: REVIEW (in_progress→review 2026-06-01). *AGENT_SIMULATED execution: full suite + ruff + mypy ran on the dev `.venv` (numpy 2.0.2; numpyro/jax ABSENT → bayes-J path SKIPs). CLAIM_GRADE is itself gated on a regenerated canonical run (G7, S7) + the Tier-C Pass-2 API runs — neither is agent-simulated as real (methodology invariant).*

---

### Iteration 2 — in-loop gate remediation (REQUEST_CHANGES: 1 MAJOR axiom + minors)

**Iteration-2 commit SHAs (RED precedes GREEN precedes REFACTOR):**
- RED   `c407f52` — `test: S4-CS-01 RED (iter-2) — pin RecallFloorVerdictGuard (G7 AXIOM-S4CS01-001) + Tier-R∪Tier-C gating + strengthen J-unavailable assertion + correct boundary rationale`
- GREEN `4a04d0f` — `feat: S4-CS-01 GREEN (iter-2) — RecallFloorVerdictGuard (G7 + ranking coupling) + Tier-R∪Tier-C CLAIM_GRADE gating`
- REFACTOR `719df4b` — `refactor: S4-CS-01 (iter-2) — make _guarded_rank1 crown tie-break deterministic-consistent`

**FIX 1 — MAJOR (AXIOM-S4CS01-001) RecallFloorVerdictGuard — IMPLEMENTED (not deferred):**
- `recall_floor_breachers(benchmark, systems) -> frozenset[str]` (new, pure, exported) — a **recall-specific** breach predicate: a system breaches if its `qualification_status` is non-qualifying (∉ `{core, qualified}`, missing ⇒ non-qualifying/fail-loud) OR the run breaches (`_run_breaches_recall_floor`: per-language ε > `EPS_RECALL_PER_LANG` when the artifact is present — absent ⇒ PENDING-not-fabricated — OR a failing `recall`/`f1` profile floor-check). **Latency/throughput floor-checks are carved out** (the conflated ensemble `floor_pass` is deliberately NOT consulted; the real artifact's `floor_pass=False` is latency-driven, not a recall breach).
- **Ranking/J coupling** (`_guarded_rank1`): the rank-1 argmax is computed over the floor-**compliant** columns only → a breacher can never be J-argmax; if pii-anon itself breaches, its J is forced to 0.0. New `j_rank1_system` field surfaces the crowned (compliant) system.
- **G7 coupling** (`_g7_certified_run`): G7 FAILS if the top-composite system breaches OR the pii-anon claimant breaches — guard `binding_detail` names the floor breach.
- **[PROPERTY-TEST] the teeth** — `test_floor_breacher_with_highest_composite_never_top_ranks` **PASSED**: a floor-breaching system with the STRICTLY highest composite (0.99 > pii-anon 0.80) is NOT J-argmax (`j_rank1_system != "rogue-sota"`), is in `recall_floor_breachers`, AND the verdict is not CLAIM_GRADE with `G7.passed is False`. Companion `test_floor_compliant_top_system_passes_the_recall_floor_guard` **PASSED** (clean top system → guard does not demote; crowned = `pii-anon`).
- **CRITICAL invariant preserved** — `test_real_artifact_recall_floor_guard_does_not_perturb_headline` **PASSED**: on today's REAL artifact pii-anon is NOT a recall-floor breacher (latency-only floor failure correctly carved out), and the headline stays `NOT_YET` / `binding_constraint = canonical_claim_run=False (G7 certified-run gate)` (canonical = binding-priority #1, ahead of the guard sub-condition). Existing real-artifact test still green.

**FIX 2 — substantive (requirements-coverage OBS) §5 Tier-R ∪ Tier-C gating — FIXED:**
- `_decide` now gates `CLAIM_GRADE` on **(Tier-R ∪ Tier-C) all RUN-or-WAIVED** (was Tier-C only). New `unrun_tier_r` registry helper + honesty field + `from_artifacts(unrun_tier_r_waivers=...)` param (shares the reason-mandatory `waive` path). Binding-constraint message surfaces unrun Tier-C then unrun Tier-R.
- `test_unrun_tier_r_gliner2_blocks_claim_grade` **PASSED** (gliner2 UNRUN, no waiver, all else satisfied ⟹ PROVISIONAL, `"gliner2" in binding_constraint`); `test_waived_tier_r_gliner2_unblocks_claim_grade` **PASSED** (gliner2 WAIVED-with-reason ⟹ CLAIM_GRADE, `binding_constraint == ""`). 3 pre-existing CLAIM_GRADE tests + 1 PROVISIONAL test updated to waive gliner2 (they encoded the Tier-C-only defect).

**FIX 3 — code-quality MINOR — strengthened:** `test_j_unavailable_cannot_be_claim_grade` now asserts `verdict is Verdict.NOT_YET` (+ `"J" in binding_constraint`) with an inline §5 citation (J ≥ 0.95 required for CLAIM_GRADE *and* PROVISIONAL). **PASSED.**

**FIX 4 — axiom MINOR-2 — boundary-test rationale corrected:** `tests/test_rating_import_boundary.py` docstring + section comment now state the real reason for the per-module scope (forward-proof the 2 gate modules). Corrected the mis-location: `competitor_compare.py`'s `from pii_anon.moe import …` (line 909) lives in the **sibling** `src/pii_anon/evaluation/` package — NOT under the scanned `eval_framework/evaluation/`, which the glob never reaches (verified: `find src -name competitor_compare.py` ⇒ `src/pii_anon/evaluation/competitor_compare.py` only).

**Iteration-2 test counts:** +12 new tests (`test_competitive_supremacy.py` 37→47, +10: 6 RecallFloorVerdictGuard + 3 Tier-R-gate + 1 real-artifact guard invariant; `test_competitor_tiers.py` 19→21, +2 `unrun_tier_r` helper) + 1 strengthened (J-unavailable). Owned-file tests: **72 passed** (`test_competitive_supremacy.py` 47 + `test_competitor_tiers.py` 21 + `test_rating_import_boundary.py` 4).

**Iteration-2 quality gates:**
- Full suite: **2876 passed, 15 skipped, 9 deselected, 0 failed** (`python3 -m pytest`, 17m53s, exit 0). (+12 from 2864.)
- Coverage: **86.17%** (`--cov-fail-under=84` reached).
- ruff: **clean** (`All checks passed!`). mypy --strict: **clean** (`Success: no issues found in 122 source files`, `mypy src/pii_anon`).
- `supremacy` CLI re-run on the REAL (read-only) artifact: `NOT_YET` / `binding canonical_claim_run=False` / `J=1.0 mle-bootstrap`, **exit 0**; `--canonical-claim` **exit 1**.
- `evaluation/competitor_compare.py` **byte-identical** (md5 `7cae16c89f4c97136e1a12394dae2025` unchanged — RISK-6). No new deps (gate stays stdlib + numpy; bayes-J still importorskip). All user-WIP md5 unchanged (`orchestrator.py` `0afc6dee…`, `test_moe_enhancements.py` `910e9cd6…`, `benchmark-diagnostics.json` `47f9b116…`, `README.md` `8a0f1000…`); artifacts/benchmarks/* READ-ONLY (never written).

---

#### Iteration-1 evidence (original build)

**Commit SHAs (RED precedes GREEN precedes REFACTOR):**
- RED   `512311b` — `test: S4-CS-01 RED — pin SDO CompetitiveSupremacyGate verdict machine + Tier-R/Tier-C registry`
- GREEN `b8c9d9b` — `feat: S4-CS-01 GREEN — CompetitiveSupremacyGate (SDO) + Tier-R/Tier-C registry`
- REFACTOR `27ce89a` — `refactor: S4-CS-01 — drop unused RunStatus import + dead _binding_constraint accessor`

**Verdict on today's (provisional) benchmark JSON** (`artifacts/benchmarks/benchmark-results.json`, read-only):
- `verdict = NOT_YET`
- `binding_constraint = "canonical_claim_run=False (G7 certified-run gate)"` (value-independent; the #1 gate)
- `canonical_claim_run = False`

**J (the SDO objective):** `J = 1.0`, `j_source = mle-bootstrap` (pii-anon holds the top composite 0.7846; bayes path SKIP-gated — numpyro absent). J always reportable via the new `bradley_terry.paired_bootstrap_draws` → `significance.rank_one_probability`.

**Per-guarantee table (three-valued):**

| G | Axis | Verdict | Observed vs bar |
|---|---|---|---|
| G1 | Recall-floor by construction | **PENDING** | structural superset holds; per-language ε artifact ABSENT → never fabricated |
| G2 | Pseudonymization-integrity / reversibility | **PENDING** | ←S4-01 anon/pseudo scorers |
| G3 | Recall dominance | **PASS** | pii-anon-swarm recall 0.818 ≥ best competitor (gliner) 0.658 |
| G4 | Calibration / selective-risk | **PENDING** | ←S4-03 reporter |
| G5 | Audit + orchestration latency / interception | **PENDING** | ←S5/S6 |
| G6 | Non-inferiority on raw F2 | **PASS** | core F2 0.7793 ≥ best Tier-R F2 0.6967 − ε_F(0.01); OpenAI raw-F1 carve-out recorded |
| G7 | Certified run | **FAIL** | canonical_claim_run=False (provenance stamp present, but run is a 50-sample smoke) |

**Tier-R / Tier-C run/unrun honesty boundary:**
- Tier-R RUN (from benchmark `available_competitors` + per-system `available`): `gliner`, `presidio`, `scrubadub`.
- Tier-R UNRUN: `gliner2` (new adapter — not yet in the run).
- Tier-C UNRUN (Pass-2 real-API; the CLAIM_GRADE blocker until run-or-waived): `openai-privacy-filter`, `azure-ai-language`, `aws-comprehend`.
- Carve-out note + `canonical_claim_run=False` banner ALWAYS emitted.

**Threshold literals pinned:** `J_BAR=0.95`, `EPS_F2=0.01`, `ENTITY_COVERAGE_MIN=0.80`, `EPS_RECALL_PER_LANG=0.005`.

**Quality gates:**
- RED test count: 57 (`test_competitor_tiers.py` 23 + `test_competitive_supremacy.py` 30 incl. 1 real-artifact + boundary 4); GREEN owned-test pass count: **92** (the 57 + 4 `paired_bootstrap_draws` + 3 `supremacy` CLI + the bradley_terry pin set under the new file scope).
- Full suite: **2864 passed, 15 skipped, 9 deselected, 0 failed** (`python3 -m pytest`, 18m11s).
- Coverage: **86.13%** (`--cov-fail-under=84` reached).
- ruff: **clean** (`All checks passed!`). mypy --strict: **clean** (`Success: no issues found in 122 source files`, `mypy src/pii_anon`).
- Import-boundary: GREEN (the 2 new gate modules import only `eval_framework.rating` + read JSON; no swarm/moe/fusion/policy).
- `competitor_compare.py` byte-identical; all user-WIP md5 unchanged.

## History Log
- 2026-06-01 — CLAIMED → IN_PROGRESS on RED `512311b` (claimer=dev-assist-development-executor).
- 2026-06-01 — GREEN `b8c9d9b`: gate + tier registry + `paired_bootstrap_draws` + `supremacy` CLI; 92 owned tests pass; real-artifact verdict NOT_YET / binding canonical_claim_run=False / J=1.0(mle-bootstrap).
- 2026-06-01 — REFACTOR `27ce89a`: drop dead code (behaviour-identical); ruff + mypy --strict clean.
- 2026-06-01 — Full suite 2864 passed / 15 skipped / 0 failed, coverage 86.13%. IN_PROGRESS → REVIEW. Awaiting story-gate (code-quality + axiom-compliance + traceability + requirements-coverage; security-sast on CLI/load; performance on G5 reads).
- 2026-06-01 — **Story-gate iteration-1 = REQUEST_CHANGES** (6 reviewers): 1 MAJOR axiom-compliance (AXIOM-S4CS01-001: RecallFloorVerdictGuard absent) + minors (requirements-coverage §5 Tier-R∪Tier-C; code-quality weak J-unavailable assertion; axiom boundary-rationale mis-location).
- 2026-06-01 — **Iteration-2 in-loop remediation** (TDD RED→GREEN→REFACTOR): RED `c407f52` → GREEN `4a04d0f` → REFACTOR `719df4b`. Implemented the RecallFloorVerdictGuard (recall-specific breach predicate + J/ranking exclusion + G7 sub-condition); fixed §5 to gate CLAIM_GRADE on (Tier-R ∪ Tier-C) all RUN-or-WAIVED (unrun gliner2 now blocks); strengthened the J-unavailable assertion to NOT_YET; corrected the boundary-test rationale (competitor_compare.py is in the sibling `evaluation/`, not the scanned `eval_framework/evaluation/`). +12 new tests; the property-test teeth (floor-breacher with top composite never crowned + cannot be CLAIM_GRADE) PASS; real-artifact headline UNCHANGED (NOT_YET / canonical_claim_run=False / J=1.0). Full suite 2876 passed / 15 skipped / 0 failed, coverage 86.17%; ruff + mypy --strict clean; competitor_compare.py byte-identical; user-WIP md5 unchanged. State remains REVIEW — gate re-runs.
