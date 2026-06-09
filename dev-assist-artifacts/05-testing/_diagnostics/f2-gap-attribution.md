# F2-gap attribution — why the keystone's fresh core F2 (~0.65) sits below the census (~0.78)

> **Diagnostic for the S7-02 keystone's honest `G6 FAIL` finding.** User decision A (this session) = "diagnose the F2 gap first" — isolate **code-regression vs scale vs dataset vs config** before building more RC surface. Read-only controlled benchmarks via `compare_competitors()` (direct library calls — no README/CWD side-effects; all outputs to `/tmp`). Provisional status: AGENT_SIMULATED measurements on the in-tree `pii_anon_benchmark` corpus at representative scale on the current code; a full-census run is the documented Pass-2.

## Question
The keystone canonical run (current HEAD, `use_case="default"`, ~5000 records) measures **pii-anon core F2 ≈ 0.65–0.66 < gliner ≈ 0.74** ⇒ SDO guarantee **G6 (raw-F2 non-inferiority) FAILS** ⇒ verdict NOT_YET. The documented census shows **pii-anon core F2 ≈ 0.779 > gliner 0.697** (composite 0.7846, rank-1) at 148,994 records on git_sha `2761a27`, dataset_source `auto`. Is the drop a **code regression** (fixable before RC), a **scale** artifact, a **dataset** difference, or an **evaluation-config/methodology** effect?

## Method
Controlled `compare_competitors()` runs holding all-but-one variable fixed. Dataset held constant (the bundled `packages/pii_anon_datasets/.../pii_anon_benchmark_v1.jsonl`, read by both code versions — `src/pii_anon/benchmarks/` + `evaluation/competitor_compare.py` are **byte-identical across `2761a27..HEAD`**). Old code obtained by `git archive 2761a27 src` → run against the current venv's installed deps (the harness is byte-identical; only the detection code — orchestrator/swarm/fusion + the new `routing/` package — differs).

## Evidence

| # | Code | use_case / config | N | pii-anon **core** P / R / **F2** | gliner F2 | G6 |
|---|---|---|---|---|---|---|
| 1 | **current HEAD** | default · objective=balanced | 100 | 0.6236 / 0.6512 / **0.6455** | 0.7448 | FAIL |
| 2 | **current HEAD** | default · objective=accuracy | 100 | 0.6236 / 0.6512 / **0.6455** | 0.7448 | FAIL |
| 3 | **old `2761a27`** | default · objective=balanced | 100 | 0.6236 / 0.6512 / **0.6455** | 0.7448 | FAIL |
| 4 | current HEAD (keystone) | default · balanced | 5000 | ≈ — / — / **≈0.66** | ≈0.74 | FAIL |
| 5 | old `2761a27` (census) | **matrix** (6 profiles) · 044fec59 | 148,994 | 0.7197 / 0.7958 / **0.7793** | 0.6966 | (PASS) |

Co-measured at `use_case=default` (rows 1–3): **pii-anon-swarm** R=0.7419 ≥ gliner R=0.7140 ⇒ **G3 (recall dominance) PASS**; pii-anon-swarm P=0.4557 (low-precision floored ensemble). G6 reads pii-anon **core** (a different system from -swarm).

## Findings — what the gap is NOT
- **NOT a code regression.** Rows 1 vs 3: old code (`2761a27`, no `routing/`) and current HEAD (+1048 lines of S1-02 floor + S2 routing) produce **byte-identical** pii-anon detection at `use_case=default` (core 0.6236/0.6512/0.6455; swarm 0.4557/0.7419). The detection-path additions do not change what `compare_competitors` measures at the default config. *(Decision-rule cell: "C′ ≈ A′ ≈ 0.66 → no regression".)*
- **NOT scale.** Row 1 (N=100) ≈ row 4 (N=5000) ≈ 0.65 — pii-anon core F2 at `default` is flat across scale; small-sample noise is not the driver.
- **NOT the objective.** Rows 1 vs 2: `objective=accuracy` and `objective=balanced` give **identical** core P/R/F2 at `use_case=default` (objective changes only the composite weighting + `_build_core_config` engine selection, which is inert for this dataset's default path). The README's accuracy-vs-default gap was the **document profile** (`long_document` curated subset), not the objective.

## Finding — what the gap IS: evaluation methodology (config + scoring + dataset draw)
The census's pii-anon-core F2≈0.78 (row 5) comes from the **6-profile use-case matrix** (curated subsets like `long_document`, `structured_form_accuracy`, `multilingual_mix`) scored into a **composite** that folds in latency/throughput/entity-coverage — on a **different non-deterministic dataset draw** (sha `044fec59`; the current bundled corpus regenerates to `abfe651d`). The keystone measures **raw core F2 on the full, hard, mixed `use_case=default` dataset** (row 4) — strictly harder than any curated profile. So:
- pii-anon core's **raw-F2 dominance is profile/composite-dependent**, not present on the full mixed dataset.
- The keystone is **honest** to read NOT_YET / G6 FAIL at `use_case=default`.
- The **moat is intact at every config**: G3 recall dominance (swarm 0.7419 ≥ 0.7140), plus G1/G2/G4 PASS per the keystone's own run.

## Verdict (decision A)
**The F2 gap is an evaluation-methodology effect, NOT a fixable code regression and NOT a scale artifact.** → **Branch 3b: continue to RC accepting the honest NOT_YET.** No detection bug to fix before building more. The G6 raw-F2-on-full-dataset FAIL is real and defensible (the SDO's moat is reversibility / recall / calibration / audit + non-inferiority, not raw-F2 supremacy on the hardest mixed slice).

## Implications for the RC + Pass-2
- The headline **Pass-2** is a **full-census run on the current code** (and/or the matrix-profiled run): the census methodology (matrix + composite) is where pii-anon reaches rank-1. *(Matrix recovery check on current code @ N=300 across the 6 profiles: see "Matrix recovery" below — pending.)*
- The RC ships honestly: **NOT_YET at representative `use_case=default`**, with pii-anon's documented composite/profile strength carried as the `pass2_full_census_reference`.
- **No regression work is warranted.** If a future goal is to *also* win raw F2 on the full mixed dataset, that is a **detection-quality enhancement** (raise core precision/recall on hard short/noisy/multilingual text), not a regression fix — a Pass-2 product item, not an RC blocker.

## Matrix recovery (current code @ N=300, 6 profiles) — does pii-anon recover to census-level?
Run: `compare_competitors(matrix_path=use_case_matrix.json, max_samples=300)` on current HEAD.

| Profile | objective | pii-anon core F2 (P / R) | gliner F2 | raw-F2 winner |
|---|---|---|---|---|
| short_chat | speed | 0.6531 (0.617 / 0.663) | 0.7540 | gliner |
| long_document | accuracy | 0.6675 (0.691 / 0.662) | 0.7540 | gliner |
| structured_form_accuracy | accuracy | 0.6675 (0.691 / 0.662) | 0.7540 | gliner |
| structured_form_latency | speed | 0.6531 (0.617 / 0.663) | 0.7540 | gliner |
| log_lines | speed | 0.6531 (0.617 / 0.663) | 0.7540 | gliner |
| multilingual_mix | accuracy | 0.6675 (0.691 / 0.662) | 0.7540 | gliner |

**Top-level aggregate (ranked by composite):** **pii-anon comp=0.7035 (RANK-1)** · gliner 0.6643 · scrubadub 0.5140 · presidio 0.4988 · pii-anon-swarm 0.4760.

**Reading:** On the current dataset draw, current-code pii-anon **wins rank-1 on the `pii-rate-elo` COMPOSITE** (0.7035 > gliner 0.6643 — its operational latency/throughput + entity-coverage moat) but **loses raw F2 to gliner in EVERY profile** (0.65–0.67 < 0.754). Even the curated accuracy profiles do NOT reproduce the census's pii-anon raw F2≈0.78 (long_document here = P 0.691, not the README-committed 0.834) — and since old==current detection is byte-identical, that census→now shift (pii-anon F2 0.779→0.65; gliner 0.697→0.754) is the **dataset draw** (`044fec59`→`abfe651d`, non-deterministic regeneration), not code.

## Refined conclusion (supersedes the optimistic Pass-2 framing)
- **G6 (raw-F2 non-inferiority) is DATASET-DRAW-SENSITIVE.** It PASSED on the census draw (pii-anon 0.779 > gliner 0.697) and FAILS on the current draw (pii-anon 0.65 < gliner 0.754) — a ~0.10 gap, far beyond ε. pii-anon and gliner trade raw-F2 leadership between corpus draws.
- **pii-anon's robust win is the COMPOSITE (the project's headline `pii-rate-elo` metric), not raw F2.** Composite rank-1 holds on the current draw (0.7035 > 0.6643).
- **The SDO requires G6 PASS for PROVISIONAL_SOTA.** So a full-census run on the current dataset is NOT guaranteed to reach PROVISIONAL_SOTA — G6 depends on the draw. The honest RC endpoint is **NOT_YET**, with the composite rank-1 carried as the documented moat strength. *(This is stricter/more honest than the prior "full-census Pass-2 → PROVISIONAL_SOTA" assumption.)*
- **Still NO regression work warranted.** Raising pii-anon's raw F2 to robustly clear G6 on every draw is a **detection-quality enhancement** (Pass-2 product item: lift core precision/recall on hard short/noisy/multilingual text), not an RC blocker or a regression fix. Whether G6 (strict raw-F2 vs the single strongest cloud NER) is even the right bar for a composite/operational-moat product is a **requirements question for the PO**, deliberately NOT changed here (changing a gate axis to force a PASS is exactly what the no-fabrication invariant forbids).
