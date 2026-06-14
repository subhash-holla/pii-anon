# pii-rate-elo across 12 detectors — the comprehensive comparison

`pii-rate-elo` turns a raw assessment leaderboard into ranked, **statistically
qualified** indicators. This page shows it run over all 12 detectors commonly
considered in the space — 3 cloud DLP services, 7 OSS detectors, and the two
first-party pii-anon systems — on the `pii-anon-eval-data` English **test**
split (30,995 records / 201,701 gold spans, strict `(start, end, entity_type)`
matching over the 63-type canonical taxonomy).

Reproduce it (cloud results replay from a stored artifact — **no API spend**):

```bash
# in pii-anon-eval-data: score the first-party systems, merge with the stored run
python -m pii_anon_datasets.cli baselines \
  --detectors pii_anon,pii_anon_swarm --split test --languages en --out results/baselines/first-party
python -m pii_anon_datasets.cli baselines \
  --merge results/baselines/first-party/*/baseline_results.json \
          results/baselines/_partial-all-10/baseline_results.json \
  --out results/baselines/tier1-en-12

# in pii-anon-code: rate every player
pii-anon rate-elo-assessment \
  --assessment-results ../pii-anon-eval-data/results/baselines/tier1-en-12/baseline_results.json
```

> **Eval integrity.** Every detection improvement was tuned on the **dev**
> split only; the **test** split here was untouched until this reported run.
> We deliberately do **not** key any recognizer on the dataset generator's
> `"Record shows X"` filler phrase — that would memorize the template, not the
> entity — so the open-vocabulary demographic types (nationality, ethnicity,
> political opinion) score near-zero recall here. That low recall is honest.

## F2 leaderboard (the ranking metric — recall-weighted)

| Rank | Detector | Type | Precision | Recall | F1 | F2 | Coverage |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **pii_anon_swarm** | local (first-party) | 0.859 | 0.892 | 0.875 | **0.885** | **63/63** |
| 2 | **pii_anon** | local (first-party) | 0.869 | 0.888 | 0.878 | **0.884** | **63/63** |
| 3 | aws | cloud | 0.769 | 0.729 | 0.748 | 0.737 | 24/63 |
| 4 | gliner | local | 0.812 | 0.718 | 0.762 | 0.735 | 23/63 |
| 5 | gcp | cloud | 0.722 | 0.701 | 0.712 | 0.705 | 18/63 |
| 6 | azure | cloud | 0.730 | 0.688 | 0.709 | 0.696 | 17/63 |
| 7 | presidio | local | 0.419 | 0.563 | 0.480 | 0.527 | 20/63 |
| 8 | regex | local | 0.856 | 0.348 | 0.495 | 0.395 | 9/63 |
| 9 | piiranha | local | 0.444 | 0.329 | 0.378 | 0.347 | 16/63 |
| 10 | stanza | local | 0.581 | 0.308 | 0.403 | 0.340 | 3/63 |
| 11 | spacy | local | 0.463 | 0.294 | 0.360 | 0.317 | 3/63 |
| 12 | scrubadub | local | 0.817 | 0.168 | 0.278 | 0.199 | 12/63 |

Both first-party systems lead the field — best **overall** (cloud + OSS), not
only best OSS — by **~0.15 F2** over the strongest incumbent (aws 0.737), while
covering **all 63** entity types versus the field-best 24/63. Swarm edges
vanilla on recall (0.892 vs 0.888); vanilla edges swarm on precision (0.869 vs
0.859) at ~200× the throughput, so the right default depends on the deployment.

## Elo ratings + pairwise significance (the qualified view)

`rate-elo-assessment` plays every gold-supported entity type as a match field
through the `PIIRateEloEngine` (63 types × all 66 player pairs = 4,158 matches),
producing ratings with rating-deviations, 95% CIs, and a distinguishability test.

| # | System | Elo | ±RD | 95% CI |
|---:|---|---:|---:|---|
| 1 | pii_anon | 1824.9 | 30.0 | [1766, 1884] |
| 2 | pii_anon_swarm | 1823.8 | 30.0 | [1765, 1883] |
| 3 | aws | 1541.9 | 30.0 | [1483, 1601] |
| 4 | gliner | 1494.9 | 30.0 | [1436, 1554] |
| … | … | … | … | … |

The pairwise-significance matrix reports both pii-anon systems as
**statistically distinguishable** (gap > 2·√(RDᵢ²+RDⱼ²)) from **every** one of
the ten competitors, and **indistinguishable from each other** — the honest
reading of a 0.001-F2 gap. aws and gliner are tied with each other but clearly
below the pii-anon pair.

## Per-system strengths and blind spots

The report surfaces each system's strongest and weakest entity types. pii-anon's
top types are the structured identifiers it validates by construction
(BAR_NUMBER, DEA_NUMBER, DEVICE_IDENTIFIER at F2 1.00); its honest blind spots
are open-vocabulary semantic types it does not yet recognize without a field
label (AGE, AUTHENTICATION_TOKEN, ETHNICITY at 0.00). Every competitor bottoms
out on the same long tail of rare structured and semantic types — which is
exactly why coverage (63/63 vs ≤24/63) separates the leaders.

## What this leaderboard does and does not measure

- **Measured here:** detection quality (precision / recall / F1 / F2, micro +
  macro, per entity type) and entity-type coverage, for all 12 systems, on the
  same strict-span scorer.
- **Not measured here:** latency / throughput and Tier-3 re-identification
  resistance are **not** carried by this artifact, so the tournament assigns no
  rating credit or penalty for them — the report discloses this explicitly
  rather than inventing axes for systems that lack them. The latency-aware
  **composite** metric (`metrics/composite.py`, which folds in p50 latency,
  throughput, and Tier-3 RRS) is the complementary instrument for deployment
  decisions; see [pii-rate-elo.md](pii-rate-elo.md) for the full algorithm.
