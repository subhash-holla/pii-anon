# S4-CS-01 — CompetitiveSupremacyGate (the SOTA-Dominance Objective; program completion criterion)

> **Cold-pickup invariant**: executable cold. This gate IS the program's optimization function + the definition of "enhancement run complete." The guarantee-check + verdict + binding-constraint logic is **pure-Python over the benchmark JSON + a posterior summary** → fully testable in-tree with synthetic fixtures. Tier-C real-API adapters + the real bayes posterior are `importorskip`/Pass-2-gated (the honesty boundary).

| Field | Value |
|---|---|
| Epic | **E-CS Competitive-Supremacy** (new; DC-11 CanonicalRunGate / RecallFloorVerdictGuard family) |
| State | **TODO** (scaffold — phased; this story = skeleton + Tier-R/Tier-C registry + G1/G3/G6/G7 + verdict machine + binding-constraint reporter + CI non-blocking wire) |
| provisional_status | AGENT_SIMULATED (CLAIM_GRADE verdict is itself gated on a regenerated canonical run — see G7) |
| Implements | FR-007/FR-008 (canonical-run gate + provenance), NFR-006 (canonical run), the SDO objective J; consumes FR-016/NFR-011 (G1 floor), FR-009/010 (G2), FR-005/NFR-017 (G4), latency NFRs (G5), FR-003/004/NFR-001/002 (G7) |
| Traces | Program AMENDMENT — "SOTA-Dominance Objective (SDO)"; Design DC-11. POV: dominate-where-claimed (moat axes) + non-inferior on raw F1. |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[AUDIT]` `[INTEGRATION-TEST]` |
| Files owned | `src/pii_anon/eval_framework/evaluation/competitive_supremacy.py` (new — the gate), `src/pii_anon/eval_framework/evaluation/competitor_tiers.py` (new — Tier-R/Tier-C registry), `src/pii_anon/evaluation/competitor_compare.py` (extend `_COMPETITOR_META` ONLY — add gliner2 + Tier-C cited adapters as metadata; NO behavior change to existing paths), CLI surface (a `pii-anon sdo`/`supremacy` report target), `tests/test_competitive_supremacy.py` (new), `tests/test_competitor_tiers.py` (new) |
| Depends on | S3-03 DONE (joint posterior for J), S3-04 DONE (`rank_one_probability` = J; coherent significance for G7); G2 needs S4-01 (anon/pseudo scorers); G4 needs S4-03 (calibration/selective-risk reporter). |
| Blocks | program completion (M4/M7) — this gate's `CLAIM_GRADE_SOTA` verdict IS "done." |
| Size | L (phased — ship skeleton + computable guarantees now; complete J + G2/G4 as deps land) |

## 1. Intent
Make the program optimize ONE explicit objective: **pii-anon is provably better than every well-known competitor pipeline, with MINIMUM GUARANTEES on the axes where we claim an advantage** — strict dominance on moat axes (reversibility, recall, calibration/selective-risk, audit), non-inferiority on raw detection F1 (the explicit OpenAI-≈0.96 honesty carve-out). Implement it as a first-class, code-checkable gate (`CompetitiveSupremacyGate`) that consumes `artifacts/benchmarks/benchmark-results.json` + the bayes-bt joint posterior, emits exactly one verdict `{CLAIM_GRADE_SOTA | PROVISIONAL_SOTA | NOT_YET}`, and ALWAYS prints the single **binding constraint** (which guarantee failed, or how far J is from the bar) so the program always knows the next thing to fix. This gate IS the definition of "enhancement run complete."

## 2. Competitor scope — tiered, honesty-gated (`competitor_tiers.py`)
- **Tier-R (run now):** `presidio`, `scrubadub`, `gliner`, **+ a new `gliner2` (GLiNER2-PII) adapter**. SDO is computed vs Tier-R for **PROVISIONAL** status.
- **Tier-C (cited, MUST-run-before-claim):** `openai-privacy-filter`, `azure-ai-language`, `aws-comprehend` — add adapters behind the existing competitor matrix (`_COMPETITOR_META`). Until each is run (or explicitly **waived with a documented reason recorded in the gate**), `CLAIM_GRADE` is BLOCKED. Carry the unrun set as a visible **"honesty boundary"** in the gate output.
- Registry shape: `{name: {tier: R|C, package|api, citation, run_status: RUN|UNRUN|WAIVED, waiver_reason?}}`. Tier-C adapters need real APIs/keys → their runs are **Pass-2** (never agent-simulated as real — methodology invariant). The gate reads `run_status` from the benchmark JSON / a tier-status sidecar; it never fabricates a Tier-C result.

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
- [ ] **RED**: `tests/test_competitive_supremacy.py` — verdict state machine (synthetic benchmark JSON + synthetic posterior → CLAIM_GRADE/PROVISIONAL/NOT_YET per the predicate); each Gk pass/fail with a synthetic fixture; binding-constraint priority; J-fallback rank-prob; **the honesty carve-out** (G6 does NOT fail when a Tier-C raw-F1 exceeds pii-anon, only on Tier-R non-inferiority); Tier-C-unrun ⟹ CLAIM_GRADE BLOCKED ⟹ at most PROVISIONAL. `tests/test_competitor_tiers.py` — registry tiers, run_status, waiver-with-reason. Written first & failing.
- [ ] **GREEN**: gate + registry + G1/G3/G6/G7 + verdict machine + binding-constraint reporter + J-fallback; gliner2 adapter (graceful if pkg absent); Tier-C metadata (UNRUN). Non-blocking CI report wired (prints verdict + binding constraint; exit 0 unless `--canonical-claim`).
- [ ] **G2/G4/J-bayes** left as explicit tracked successors (named in the gate output as "axes pending: G2←S4-01, G4←S4-03, J-bayes←bayes-eval run").
- [ ] **Quality**: full suite green (Tier-C/real-API/numpyro tests SKIP); ruff + mypy --strict clean; import-boundary GREEN (gate ⊄ detection internals); coverage ≥ 84%.
- [ ] **Honesty**: the gate output carries the visible Tier-C honesty boundary + the OpenAI raw-F1 carve-out + `canonical_claim_run=False` PROVISIONAL banner. Marks nothing CLAIM_GRADE without a canonical run.
- [ ] **Untouched**: existing `competitor_compare.py` behavior unchanged (only `_COMPETITOR_META` extended additively); user WIP md5 unchanged (esp. `artifacts/benchmarks/*` + `benchmark-diagnostics.json` are READ-ONLY inputs — never written).
- [ ] **Story-gate APPROVE** (`_reviews/story/S4-CS-01-gate.yaml`).

## Evidence (filled on completion)
- RED/GREEN/REFACTOR SHAs · current verdict on the (provisional) benchmark JSON + the binding constraint · J value + j_source · per-guarantee table (G1..G7 pass/pending) · Tier-R/Tier-C run/unrun honesty boundary · ruff/mypy/suite/coverage · *AGENT_SIMULATED; CLAIM_GRADE gated on a regenerated canonical run (S7) + Tier-C Pass-2 runs.*
