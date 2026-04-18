# pii-anon-swarm — Architecture, Training, and Retrain Procedure

`pii-anon-swarm` is the recall-optimised detection offering. It runs
alongside the basic `pii-anon` regex engine and shares the same public
surface (`PIIOrchestrator`), but routes detection through a four-layer
fusion pipeline tuned for compliance / batch-ETL use cases where a
missed entity is worse than a false positive.

This document covers the pipeline architecture, the training data flow,
the retrain procedure, and how Tier 3 signals from the v1.3.0 benchmark
dataset feed into the meta-learner. For the detection-side quickstart,
see [quickstart.md](quickstart.md). For the evaluation framework, see
[pii-rate-elo.md](pii-rate-elo.md).

---

## Four-layer pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         pii-anon-swarm                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Layer 1 — Regex fast-pass (structured PII)                         │
│     ├ Luhn / IBAN mod-97 / ABA checksum validators                   │
│     └ confidence ≥ fast_pass_threshold (default 0.90) → emit direct  │
│                               │                                      │
│                               ▼                                      │
│   Layer 2 — Heterogeneous NER engines                                │
│     ├ GLiNER (zero-shot transformer)                                 │
│     ├ Presidio (spaCy-backed)                                        │
│     └ Stanza / regex tail                                            │
│     └ redundancy pruning via IoU overlap (default 0.3)               │
│                               │                                      │
│                               ▼                                      │
│   Layer 3 — Dawid-Skene + XGBoost meta-learner                       │
│     ├ Bayesian aggregation of engine votes (EM-trained confusion)    │
│     ├ Temperature calibration per engine                             │
│     ├ Informativeness scoring per engine                             │
│     └ 21-feature vector → XGBoost → meta_score ∈ [0, 1]              │
│                               │                                      │
│                               ▼                                      │
│   Layer 4 — Validation & corroboration                               │
│     ├ Emission gate: meta_score ≥ emission_threshold (default 0.50)  │
│     ├ Corroboration gate for SEMANTIC_TYPES:                         │
│     │   require ≥ corroboration_min engines OR meta_score ≥ 0.85     │
│     └ Deduplicate against Layer 1 fast-pass results                  │
│                               │                                      │
│                               ▼                                      │
│                    EnsembleFinding (emitted)                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Source: [src/pii_anon/swarm.py](../src/pii_anon/swarm.py),
[src/pii_anon/swarm_learner.py](../src/pii_anon/swarm_learner.py).

### SEMANTIC_TYPES — the corroboration gate

Entity types with permissive regex surfaces (e.g. emails look like almost
anything with `@` in it, credit-card digits match partial-card fragments)
benefit from multi-engine corroboration. The set is pinned in
[swarm.py](../src/pii_anon/swarm.py) as:

```python
SEMANTIC_TYPES = frozenset({
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "DATE_OF_BIRTH",
    "ADDRESS", "USERNAME", "PHONE_NUMBER",
    "EMAIL_ADDRESS", "CREDIT_CARD",   # added 2026-Q2 for precision lift
})
```

Types *not* in the set skip the gate — that is correct for deterministic
structured formats (IBAN, US_SSN, BANK_ROUTING — the Luhn / mod-97 / ABA
checksums are stronger than multi-engine votes) but a precision hazard
for anything NER engines fire loosely on. `EMAIL_ADDRESS` and
`CREDIT_CARD` were added after the v10 benchmark showed their swarm
precision at 0.46 and 0.48 respectively; requiring multi-engine
agreement clamps false positives without hurting common-case recall.

---

## Training data flow

```
┌────────────────────────────────────────────────────────────────────┐
│  swarm_datasets.load_training_data(datasets, max_records)          │
│    ├ load_pii_anon_data()         — v1.3.0 canonical (~160K recs)  │
│    ├ load_ai4privacy()            — industry labels                │
│    └ load_conll2003()             — NER baseline                   │
│                         │                                          │
│                         ▼                                          │
│  list[TrainingRecord]  (Tier 3 fields: behavioral_signal_density,  │
│                         re_identification_resistance_score,       │
│                         persona_id, is_paired_profile)             │
│                         │                                          │
│                         ▼                                          │
│  Run heterogeneous engines on each record's text                   │
│                         │                                          │
│                         ▼                                          │
│  Dawid-Skene EM training → confusion matrices per engine           │
│  Temperature calibration → per-engine sigmoid                      │
│  Informativeness scoring                                           │
│                         │                                          │
│                         ▼                                          │
│  Feature extraction (21-dim) + TP/FP labels + sample_weights       │
│                         │                                          │
│                         ▼                                          │
│  XGBoostMetaLearner.train(features, labels, sample_weights=...)    │
│                         │                                          │
│                         ▼                                          │
│  F2 threshold sweep (select_f2_threshold)                          │
│    → pick emission_threshold maximising F2 on held-out slice       │
│                         │                                          │
│                         ▼                                          │
│  Artifacts → ~/.pii_anon/swarm/                                    │
│    ├ ds_params.json         (Dawid-Skene confusion + priors)       │
│    ├ temperature.json       (per-engine sigmoid params)            │
│    ├ informativeness.json   (per-engine weights)                   │
│    ├ xgboost_model.ubj      (21-feature XGBoost booster)           │
│    └ manifest.json          (fold metrics + feature version)       │
└────────────────────────────────────────────────────────────────────┘
```

### Tier 3 integration

Dataset v1.3.0 ships per-record Tier 3 annotations that the training
pipeline consumes in two places:

1. **`TrainingRecord` fields** — `load_pii_anon_data` in
   [swarm_datasets.py](../src/pii_anon/swarm_datasets.py) now populates
   `behavioral_signal_density`, `re_identification_resistance_score`,
   `persona_id`, and `is_paired_profile` from the v1.3.0 schema.

2. **Sample weighting** —
   [`compute_sample_weights_from_records`](../src/pii_anon/swarm_learner.py)
   turns RRS into a per-example loss multiplier. Records with low RRS
   (hard to de-identify — the failures the paper v10 adversary most
   wants to exploit) get up to `2.0×` weight; paired-profile records
   get an additional `1.5×`. The meta-learner sharpens its decision
   boundary on those hard cases without changing the inference-time
   feature vector shape.

**Why not feed Tier 3 signals as features?** Because at inference time
the caller doesn't know RRS — it's a dataset-side annotation. Features
the caller can't supply at prediction time are dead weight. Sample
weighting achieves the same "focus on hard cases" goal while keeping
the 21-feature input honest.

### F2 threshold selection

After training, the default `emission_threshold=0.50` is replaced by the
value that maximises F2 (β=2, privacy-first) on a held-out split.
[`select_f2_threshold`](../src/pii_anon/swarm_learner.py) sweeps
`[0.30, 0.70]` in `0.02` steps, computing TP/FP/FN at each threshold
and returning `(threshold, f_beta)`. The resulting value is written to
`manifest.json` and used by `SwarmConfig.__post_init__` at load time.

This matches the paper v10 §4.1 recommendation: F1 treats false
positives and false negatives as equally costly, but the GDPR / HIPAA /
CCPA cost model assigns `~4×` more weight to a missed entity than an
over-redaction. F2 preserves that asymmetry.

### Feature vector (21 dims)

| # | Feature | Source |
|---|---|---|
| 1 | `ds_confidence` | Dawid-Skene output |
| 2 | `corroboration_count` | # engines voting for this span |
| 3 | `corroboration_ratio` | count / total engines |
| 4-7 | min/max/mean/std engine confidence | calibrated |
| 8-9 | regex detected / regex confidence | Layer 1 signal |
| 10-11 | span length (chars, tokens) | |
| 12 | entity type encoded | categorical → int |
| 13 | is_structured_type | Luhn/mod-97 checksum types |
| 14 | has_checksum_validation | high-conf regex |
| 15 | informativeness score | per-engine weight |
| 16 | boundary agreement | IoU across engines |
| 17 | `context_has_keywords` (EN) | ±50 char window |
| 18 | position in text | 0.0 start → 1.0 end |
| 19 | surrounding entity density | count within 100 char |
| 20 | engine diversity | Jaccard across votes |
| **21** | `context_has_keywords` (ES/FR/DE/ZH/JA) | **added 2026-Q2** |

Feature 21 covers the ~56K non-English records in v1.3.0 where the
English-only feature 17 produces zero signal. The feature version
constant (`FEATURE_VERSION=2`) is persisted to `manifest.json` so the
loader can detect training/inference mismatches.

---

## Retraining

The retrain procedure uses a single Make target and honours environment
variable overrides for all tunables.

```bash
# Fast iteration — ~5 min on M1 macbook
make train-swarm SWARM_MAX_RECORDS=10000 SWARM_KFOLD=3

# Full production training — all 159,891 v1.3.0 records, 5-fold CV
make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy,conll2003 \
                 SWARM_MAX_RECORDS=0 SWARM_KFOLD=5
```

Artifacts land in `~/.pii_anon/swarm/` and are automatically picked up
on the next `SwarmConfig()` instantiation. Verify:

```bash
cat ~/.pii_anon/swarm/manifest.json | python -m json.tool
```

Look for:
- `"feature_version": 2`
- `"emission_threshold"` close to `0.5` (higher = stricter, lower = more recall)
- `"mean_f2"` ≥ 0.65 (the paper v10 industry-leadership bar)
- `"fold_f1"` / `"fold_f2"` standard deviation — low σ means the model is stable

After training, rerun the benchmark to refresh the leaderboard:

```bash
make benchmark-full
```

---

## Customisation hooks

Most swarm behaviour can be tuned without retraining. The knobs that
matter most:

| Knob | Default | Effect |
|---|---|---|
| `SwarmConfig.fast_pass_threshold` | 0.90 | Lower → more Layer 1 fast-pass emissions (skip fusion for high-conf regex) |
| `SwarmConfig.emission_threshold` | 0.50 (or F2-selected) | Raise to shed FPs, lower to catch more TPs |
| `SwarmConfig.corroboration_min` | 2 | Higher → stricter SEMANTIC_TYPES gate |
| `SwarmConfig.corroboration_override_threshold` | 0.85 | Lets a single high-conf engine override the corroboration gate |
| `SwarmConfig.iou_threshold` | 0.3 | How aggressively to deduplicate overlapping engine votes |

For hyperparameter search across these knobs, see
[autoresearch-integration.md](autoresearch-integration.md) — the
autoresearch library wraps the whole pipeline as an experiment loop
that iteratively improves the configuration.

---

## When to retrain

Retrain when any of the following changes:

1. **Dataset version** — a new `pii-anon-datasets` release
   (e.g. v1.3.0 → v1.4.0) with new entity types, languages, or Tier 3
   annotations.
2. **Engine adapter updates** — GLiNER model upgrade, new engine
   adapter added to the NER pool (Layer 2 distribution shifts).
3. **Feature vector change** — `FEATURE_VERSION` bump in
   [swarm_learner.py](../src/pii_anon/swarm_learner.py).
4. **Deployment profile shift** — moving from `standard` to
   `high_security` operationally may warrant retuning
   `emission_threshold` via the F2 sweep against a security-weighted
   metric.

The retrained artifacts are backward-compatible with older swarm
callers as long as `FEATURE_VERSION` matches; the loader rejects
artifacts from an incompatible version rather than silently feeding
mis-shaped vectors to XGBoost.

---

## References

- Paper v10: `../pii-anon-research-paper/Paper1-PII-Rate-Elo-Framework-v10.md`
- Dataset v1.3.0: `../pii-anon-eval-data/CHANGELOG.md`
- Dawid, A.P. & Skene, A.M. (1979). "Maximum Likelihood Estimation of Observer Error-Rates"
- Lermen et al. (2026). "Large-scale online deanonymization with LLMs" — motivates sample weighting on low-RRS records
- TAB (2022) — justifies F2 threshold selection over F1
