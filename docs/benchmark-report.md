# pii-anon Benchmark Report — v1.7.0rc1

**Date:** 2026-07-13 · **Library:** pii-anon 1.7.0rc1 · **Eval data:** pii-anon-eval-data 2.2.0
· **Scoring:** strict-v1 exact `(entity_type, start, end)` multiset (home); relaxed = span IoU ≥ 0.5, same label (external)

This report is **reproducible** — every table below lists the exact command that regenerates it.
Numbers were measured on the **current v1.7.0rc1 code** (a fresh run, not sprint-accumulated).

> **Honesty notes, up front.** (1) The home headline is **English-only**; the full multilingual
> figure is materially lower and is reported alongside it. (2) The competitor leaderboard mixes
> vintages: the 11 external detectors carry their last full-run scores (2026-06, `sp5-test-13player`);
> only the two first-party systems are freshly re-scored on current code — because AWS Comprehend
> and Flair cannot run in every environment (`boto3` + credentials / `flair` absent here), a fully
> same-vintage competitor run requires the benchmark environment (see *Regenerating the full field*).
> (3) None of this is an external-validity claim; see [external-validity-report.md](external-validity-report.md).

---

## 1. Home — the pii-anon-eval-data benchmark

### 1a. Competitor-comparable slice (EN test, 66-type)

The corpus the competitor leaderboard is scored on: v2.2.0 **test split, English, 31,048 records, 201,880 gold spans**.

| System | Precision | Recall | **strict F2** | Coverage |
|---|---:|---:|---:|---:|
| `pii_anon_swarm` | 0.853 | 0.923 | **0.908** | 66/66 |
| `pii_anon` (vanilla) | 0.867 | 0.915 | **0.905** | 66/66 |

*(prior published: swarm 0.893 / vanilla 0.892 — the sp4–sp7 tranches lifted both.)*

```bash
# from the pii-anon-eval-data repo, with pii-anon-code/src on PYTHONPATH:
PYTHONPATH="src:../pii-anon-code/src" python -m pii_anon_datasets.cli baselines \
  --detectors pii_anon,pii_anon_swarm --split test --languages en --out out/home-en
```

### 1b. Entirety (full dataset, ALL languages)

The **complete** dataset — 808,084 records across all splits — scored first-party (rule-based /
zero-shot-ML: no train/test leakage, so the train split is a valid evaluation surface).

| Split | Records | Gold spans | `pii_anon` strict F2 |
|---|---:|---:|---:|
| test (all languages) | 157,045 | 623,250 | **0.820** |
| train | 547,586 | 2,174,429 | **0.820** |
| dev | 78,046 | 309,561 | **0.820** |

**The English-only 0.905 does not hold across all languages — the full multilingual figure is
~0.820** (consistent across train/test/dev, confirming no overfitting). `pii_anon_swarm` on the
full multilingual test measures **strict F2 0.823** (P 0.869 / R 0.812, on a seeded 12,000-record
representative multilingual sample — the 60 languages are interleaved, so the head sample tracks the
full distribution; the full 157k GLiNER run is ~6.5h and was not executed). So on multilingual text
**swarm ≈ vanilla ≈ 0.82** — the GLiNER NER channel contributes far less off-English than it does on
the 0.908 EN slice, which is the honest multilingual picture.

```bash
PYTHONPATH="src:../pii-anon-code/src" python -m pii_anon_datasets.cli baselines \
  --detectors pii_anon --split train --languages all --out out/home-train   # 547k
# (repeat with --split test / --split dev for the other rows)
```

---

## 2. pii-rate-elo — 13-player tournament

`pii-rate-elo` rates every detector in a merged assessment artifact by per-entity-type Glicko Elo
(66 gold-supported types × 13 players = 5,148 matches), on the EN test corpus above.

| # | System | Elo | 95% CI | F2 (micro) | Precision | Recall | Coverage |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | **pii_anon_swarm** | 1868.7 | [1810, 1928] | 0.908 | 0.853 | 0.923 | 66/66 |
| 2 | **pii_anon** | 1867.8 | [1809, 1927] | 0.905 | 0.867 | 0.915 | 66/66 |
| 3 | aws | 1541.4 | [1483, 1600] | 0.736 | 0.769 | 0.728 | 24/66 |
| 4 | gliner | 1489.5 | [1431, 1548] | 0.734 | 0.813 | 0.716 | 23/66 |
| 5 | gcp | 1448.5 | [1390, 1507] | 0.704 | 0.722 | 0.700 | 18/66 |
| 6 | azure | 1444.7 | [1386, 1503] | 0.696 | 0.730 | 0.688 | 17/66 |
| 7 | piiranha | 1437.4 | [1379, 1496] | 0.345 | 0.441 | 0.327 | 16/66 |
| 8 | scrubadub | 1431.1 | [1372, 1490] | 0.201 | 0.818 | 0.169 | 12/66 |
| 9 | regex | 1414.6 | [1356, 1473] | 0.396 | 0.857 | 0.349 | 9/66 |
| 10 | presidio | 1393.8 | [1335, 1453] | 0.526 | 0.419 | 0.562 | 20/66 |
| 11 | flair | 1385.7 | [1327, 1445] | 0.326 | 0.565 | 0.295 | 3/66 |
| 12 | spacy | 1385.6 | [1327, 1444] | 0.317 | 0.464 | 0.294 | 3/66 |
| 13 | stanza | 1385.2 | [1326, 1444] | 0.340 | 0.583 | 0.308 | 3/66 |

Both first-party systems sit ~**326 Elo above** the best competitor (AWS), statistically
distinguishable from every external at >2σ. **Mixed-vintage:** first-party fresh (v1.7.0rc1);
competitors from the last full run (`sp5-test-13player`, 2026-06). Committed artifacts:
[`artifacts/ratings/v1.7.0rc1-13player/`](../artifacts/ratings/v1.7.0rc1-13player/) (leaderboard.md
+ tournament.json + the two fresh first-party `baseline_results.json`).

```bash
# rate ANY merged assessment artifact (yours, ours, a vendor's) — this is the
# framework that produced the table above:
pii-anon rate-elo-assessment -a <merged_baseline_results.json> \
  --output markdown --artifact-dir out/rate-elo
```

### Regenerating the full field (same-vintage, incl. AWS/Flair)

The 11 competitors need their own environment (AWS Comprehend → `boto3` + credentials; Flair →
`pip install flair`; the cloud DLPs → `--cloud` + budget). Regenerate the whole field, then rate it:

```bash
# in a benchmark environment with all detectors installed + credentials set:
PYTHONPATH="src:../pii-anon-code/src" python -m pii_anon_datasets.cli baselines \
  --detectors pii_anon,pii_anon_swarm,presidio,gliner,scrubadub,spacy,stanza,piiranha,flair \
  --cloud --split test --languages en --out out/full-field           # add aws,azure,gcp via --cloud
pii-anon rate-elo-assessment -a out/full-field/baseline_results.json --artifact-dir out/rate-elo
```

---

## 3. External datasets (zero-shot, non-home)

Five public PII benchmarks, nothing tuned; seeded samples per the standing external-validity report.
Relaxed F2 (label + IoU ≥ 0.5). Full disclosure, per-type tables, and the transfer-gap discussion
live in [external-validity-report.md](external-validity-report.md).

| Dataset (cap) | `pii_anon` | `pii_anon_swarm` | prior (sp4/sp6) |
|---|---:|---:|---:|
| TAB — real ECHR court docs (200) | 0.494 | **0.505** | 0.100 (~5×) |
| Gretel finance (3000) | 0.459 | **0.488** | 0.379 |
| Nemotron-PII (5000) | 0.380 | **0.390** | 0.324 |
| ai4privacy-400k (5000) | 0.239 | **0.290** | 0.213 |
| PIIBench (5000) | 0.216 | **0.236** | 0.184 |

Every external improved vs the prior published numbers, and `pii_anon_swarm ≥ pii_anon` on all
five. These remain a **disclosed transfer gap** (0.24–0.51 external vs 0.82–0.91 home), never a
general best-in-class claim.

```bash
# from pii-anon-eval-data/external_eval (each ds_*.py knows its split + seeded cap):
PYTHONPATH="src:../pii-anon-code/src" python run_swarm_all.py     # pii_anon_swarm, all 5
```

---

## Reproducibility & honesty summary

- **Fresh:** all first-party numbers measured on v1.7.0rc1 HEAD.
- **Same corpus for the head-to-head:** first-party and competitors scored on the identical EN test
  gold (the rate-elo shared-gold invariant is enforced at ingest — it fails loud on a mismatch).
- **Mixed-vintage disclosed:** competitors carry sp5 (2026-06) scores; regenerate them same-vintage
  in a full-field environment via the commands above.
- **English vs multilingual disclosed:** 0.905 EN / ~0.820 all-languages.
- **SDO verdict:** competitive-supremacy is honestly **NOT_YET** (binding J ≈ 0.28); this report is
  performance disclosure, not a supremacy claim.
