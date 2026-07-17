# Ensemble-v2 & Speed-Boost Prior-Art — Design Inputs (Theme 1 / Theme 4)

> **Source Signal** (per `dev-assist-brownfield-assessment` Step 2). Wraps and cites prior-art from earlier swarm sub-investigations; **all originals preserved untouched** in `pdlc-artifacts/ensemble-v2/` and `pdlc-artifacts/speed-boost/`. Migrated 2026-05-30.

## Ensemble-v2 — "Beat GLiNER" (entity-type normalization + corroboration)

Provenance index: `pdlc-artifacts/ensemble-v2/MANIFEST.md` (`sha256:cbee9f2f7a74f75d…`, KEEP_AS_IS) — before/after: **swarm F1 0.6265 → 0.8622** (beats GLiNER by +8.8 pts); 3 changes (entity-type normalization map, corroboration filter, regex pattern fixes).

**FP-source diagnosis** — `pdlc-artifacts/ensemble-v2/discovery/discovery-report.md` (`sha256:ef78a0f7a8951ef0…`)
500-record analysis: **9× FP explosion (5,079 vs 566)**, attributed by entity type and source engine, split into:
- **Category 1 — missing entity-type normalization: 65% of FPs** (IN_PAN / URL / US_DRIVER_LICENSE / US_BANK_NUMBER taxonomy mismatches).
- **Category 2 — genuine over-detection: 35%.**
→ *Feeds:* the redesign's **entity-type normalization layer** + corroboration design (a distinct, high-value insight: most "FPs" were taxonomy-mapping artifacts, not real over-detection).

**Validated outcome** — `pdlc-artifacts/ensemble-v2/testing/evaluation-results.md` (`sha256:1e3f2ba491de03f5…`)
1000-record P/R/F1 (**swarm 0.8622**), ~4,100 FPs eliminated by the two fixes, per-segment results **including the `edge_cases` regression (−5.7 pts)**, tests 2253/0.
→ *Feeds:* evidence that normalization + corroboration work — and the documented `edge_cases` caveat to carry forward.

## Speed-Boost — speed-profile detector fix (Theme 4 / portability)

**Source:** `pdlc-artifacts/speed-boost/MANIFEST.md` (`sha256:da2d4fdca53fae9a…`, KEEP_AS_IS — self-contained record)
Problem: the speed-profile detector scanned only EMAIL/PHONE (= 24% of labels), so speed-profile **F1 = 0.407**. Root cause: hardcoded 2-pattern stub in `_core_detector(objective='speed')`. Fix: swap in the full `RegexEngineAdapter` (~0.7ms/record, ~120× faster than GLiNER, all 20 types) → **F1 0.407 → 0.889**. Notes the informational floor-gate trade-off vs scrubadub.
→ *Feeds:* the Theme-4 portability/latency design + the swarm's always-on shared regex expert (the "shared-expert isolation" pattern); confirms a full-regex shared layer is both fast and high-coverage.
