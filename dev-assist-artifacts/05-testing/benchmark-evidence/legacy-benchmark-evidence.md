# Legacy Benchmark Evidence (NFR-verification provenance)

> **Source Signal** (per `dev-assist-brownfield-assessment` Step 2). Wraps/merges and cites the raw evaluation dumps from the earlier swarm PDLC run. **All originals preserved untouched** in `pdlc-artifacts/development/`. Migrated 2026-05-30. These are historical "before/after" detection-accuracy snapshots, useful as Theme-1 baselines and as Stage-5 NFR-verification provenance.
>
> ⚠ **Cross-dataset / sample-cap caveats apply** (carried from the originals): some rounds ran on a synthetic fallback dataset because `pii-anon-datasets` was not installed; R1-vs-R3 is a cross-dataset comparison and not directly meaningful. Sample sizes are small (100–1000 records). Treat as directional, not canonical.

## Round 1 — baseline ("before")
**Source:** `pdlc-artifacts/development/round1-eval.txt` (`sha256:4d8b8d1d34cb268c…`)
Baseline per-entity-type P/R/F1/TP/FP/FN for regex-core + ensemble on `pii_anon_benchmark_v1` (200 records, 1636 labels). The "before" snapshot referenced by all later rounds.

## Round 3 — post-bug-fix ("after") + MoE union-guarantee verification (MERGE G2)
**Sources:** `pdlc-artifacts/development/round3-eval.txt` (`sha256:f8290e941649ab4a…`) + narrative companion `ROUND3_EVAL_SUMMARY.md` (`sha256:7e8a742cff230ca5…`).
Post-fix re-eval (200 records, synthetic fallback, 142 labels): per-entity metrics, R1-vs-R3 comparison, and **"MOE UNION GUARANTEE TEST = PASSED"** — the empirical confirmation of the superset theorem (see `../../03-design/moe-architecture-and-guarantee.md`). Bug-fixes-verified list + how-to-run included.

## Round 2 — per-segment profile evaluation (MERGE G3)
**Sources:** summary `pdlc-artifacts/development/profile-eval-round2.txt` (`sha256:fb41a216340a8354…`) + raw dataset `profile-eval.json` (`sha256:7b3f970817c731fe…`, 162KB, 7272 lines — verified field-for-field match with the summary).
1000-record per-segment results (difficulty × scenario × language × entity-type): 5 regex fixes; **P 0.758→0.873, R 0.840→0.910, F1 0.797→0.891, FP −50.6%**; per-difficulty and per-scenario deltas; tests 2224 passed. The `.json` is the machine-readable dataset behind the summary; both preserved in place.

→ *Feeds:* the Stage-5 NFR-verification matrix as historical baselines, and the Theme-1 redesign's regression baselines. **Superseded for publication** by the (pending) certified canonical benchmark run — these are NOT the numbers to publish (see the steering decision in the canonical MANIFEST).
