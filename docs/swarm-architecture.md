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

### The `regex-oss` baseline is `pii-anon` itself

The `pii-anon` standalone offering — the fast regex + checksum engine
that's exported as the default detector via `PIIOrchestrator` — ships
inside the swarm under the engine ID **`regex-oss`**.  It is a
**precision-maximising, recall-sacrificing package of heuristics** that
covers vanilla PII use cases consistently with high precision and
sub-millisecond latency.  Users who just want the job done quickly
with decent accuracy can rely on it standalone; the swarm layers
mixture-of-experts NER on top to recover the edge cases the baseline
intentionally doesn't chase.

It is not a separate optional component: it's always active, always
enabled (no third-party dependency), and has three privileged positions
in the pipeline:

1. **Layer 1 fast-pass** — any regex-oss finding above
   `SwarmConfig.fast_pass_threshold` (default 0.90) is emitted
   directly without entering fusion.  Luhn/mod-97/ABA checksums are
   stronger evidence than multi-engine votes for structured PII, so
   the swarm short-circuits on them.
2. **Pinned past the Layer 2 Jaccard pruner** — `regex-oss` always
   survives, regardless of overlap with other engines
   (see [swarm.py's `_prune_redundant_findings`](../src/pii_anon/swarm.py)).
3. **Feature slots in the XGBoost meta-learner** — features 8
   (`regex_detected`), 9 (`regex_confidence`), and 14
   (`has_checksum_validation`) are all derived exclusively from
   `regex-oss` output.

`scripts/train_swarm.py` asserts `"regex-oss"` is in the active engine
pool before training starts and fails loud if it isn't — the baseline
is a hard contract, not an optional add-on.  In the run log it appears
as ``regex-oss (pii-anon baseline)`` so its role is unambiguous.

### The baseline → swarm handoff contract

Every baseline emission routes through one of three paths inside the
swarm.  Which path depends on a single number: the regex's
`base_confidence` relative to `SwarmConfig.fast_pass_threshold`
(default 0.90) and whether the type is in `STRUCTURED_TYPES`.

```
                regex-oss fires
                      │
                      ▼
        confidence ≥ 0.90 AND in STRUCTURED_TYPES?
                      │
               ┌──────┴──────┐
              YES            NO
               │              │
               ▼              ▼
     Layer 1 fast-pass   Layer 3 fusion
     (emit directly)     (Dawid-Skene + meta-learner)
                              │
                              ▼
                    entity in SEMANTIC_TYPES?
                              │
                       ┌──────┴──────┐
                      YES            NO
                       │              │
                       ▼              ▼
              Layer 4 corroboration   Emit
              (need ≥2 engines)
```

**Fast-pass eligibility matrix** (what routes through which path):

| Category | Example types | Path | Why |
|---|---|---|---|
| Checksum-validated | CREDIT_CARD, IBAN, US_SSN, VIN, ROUTING_NUMBER | Layer 1 fast-pass | Luhn / mod-97 / check-digit math is stronger than any ensemble vote |
| Phase 3 keyword-gated | CVV, PIN, PASSWORD, COURT_CASE_NUMBER, DOCKET_NUMBER, BAR_NUMBER, INVOICE_NUMBER, INSURANCE_POLICY_NUMBER, SALARY | **Layer 1 fast-pass** | Regex requires the keyword adjacent to the captured group — structural guarantee on par with a checksum |
| Semantic / ambiguous | PERSON_NAME, ORGANIZATION, LOCATION, DATE_OF_BIRTH, ADDRESS | Layer 3 fusion → Layer 4 corroboration gate | Open-vocabulary, regex hit alone is not authoritative |
| Structural but permissive | EMAIL_ADDRESS, CREDIT_CARD (fragment cases) | Layer 3 fusion → Layer 4 corroboration gate | Regex surface matches too much without context; NER votes clamp precision |

This matrix is the **integration contract** locked in by
[`tests/test_swarm_baseline_integration.py`](../tests/test_swarm_baseline_integration.py):

- Every Phase 3 type's weakest `PatternSpec.base_confidence` is ≥ 0.90.
- Every Phase 3 type is in `STRUCTURED_TYPES`.
- Every Phase 3 type has a distinct index in `ENTITY_TYPE_ENCODING`.
- The `is_structured` feature (slot 13) fires for Phase 3 findings.
- The `regex_detected` feature (slot 8) fires when regex-oss is the
  sole contributor.

A future refactor that widens a pattern and drops its confidence below
0.90 will fail these tests before landing.

### Baseline coverage map — paper v11 §5.6

Every entity type the baseline emits has a context gate, a validator,
or both.  The emission decision tree:

- **Structural shape + checksum** → high confidence (0.93–0.99). Luhn
  on credit cards, mod-97 on IBAN, weighted sum on ABA routing,
  Verhoeff on Aadhaar, NHTSA check digit on VIN, HMRC rules on UK NI.
  These types survive the swarm's Layer 1 fast-pass at threshold 0.90.
- **Structural shape + required adjacent keyword** → medium-high
  confidence (0.85–0.92). The new Phase 3 gap-closure types — CVV, PIN,
  PASSWORD, COURT_CASE_NUMBER, DOCKET_NUMBER, BAR_NUMBER,
  INVOICE_NUMBER, INSURANCE_POLICY_NUMBER, SALARY — require a keyword
  in the immediate regex neighbourhood (not just the ±50 char context
  window).  This is what lets us emit CVV=123 without misfiring on
  every 3-digit number on the internet.
- **Shape with context boost** → confidence adjusted via ±50-char
  window keyword check. Types in `HIGH_FP_TYPES` (US_SSN, PERSON_NAME,
  LOCATION, ORGANIZATION, ADDRESS, PHONE_NUMBER, EMPLOYEE_ID,
  IP_ADDRESS, EMAIL_ADDRESS, CREDIT_CARD_FRAGMENT, BANK_ACCOUNT, VIN,
  LICENSE_PLATE, NATIONAL_ID) lose confidence when the context is
  silent. Non-HIGH_FP types float at base confidence unchanged.

**What the baseline doesn't chase** — and why the swarm exists:

| Category | Example types | Why the baseline skips | Swarm layer that picks it up |
|---|---|---|---|
| Free-form names | PERSON_NAME in novel phrasing, aliased or abbreviated | No structural signature | GLiNER / Presidio NER |
| Domain vocabulary | MEDICATION_NAME, HEALTH_CONDITION, JOB_TITLE | Open-vocabulary — no pattern can enumerate terms | Presidio with medical recognizer, GLiNER zero-shot |
| Multi-paragraph inference | Entity tracking ("the patient", "Mr. X" referring back) | Rules see only local context | Dawid-Skene on engine-by-engine judgement + corroboration |
| Behavioural quasi-identifiers | Writing style, interest signals (Tier 3) | Not in the rule surface at all | NER ensemble + Tier 3 scoring post-detection |

This is the paper v11 §5.6 recommendation materialised: use rules for
what rules are best at (structural, checksum-validated, keyword-gated)
and let the expert ensemble handle the open-vocabulary tail.

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

### Available datasets

Every entry is opt-in via `SWARM_DATASETS` (comma-separated list on the
Make / CLI). The first three ship out-of-the-box; the remaining three
mirror the mix used by the pii-rate-elo paper-submission evaluation, so
training data stays aligned with the evaluation reference — they require
`pip install 'pii-anon[swarm-train]'` for the HuggingFace `datasets`
dependency but are otherwise zero-config.

| Dataset | Source | Records | Languages | Best for |
|---|---|---:|---:|---|
| `pii_anon_eval` | Our canonical benchmark (v1.3.0) | ~160K | 60 | General coverage + Tier 3 behavioral-signal annotations |
| `ai4privacy` | AI4Privacy PII-Masking-200k | 200K | 8 | Industry-standard PII labels |
| `conll2003` | CoNLL-2003 English NER | 22K | 1 | NER baseline (PER / LOC / ORG) |
| `ai4privacy_400k` | AI4Privacy PII-Masking-400k (2024) | 400K | 17 | Broadest language coverage + newer entity types (passport, VIN, crypto wallets) |
| `tab` | Text Anonymization Benchmark (Pilan 2022) | 1,268 | en | Real-world legal text, peer-reviewed manual annotations |
| `meddocan` | MEDDOCAN (Marimon 2019) | 1,000 | es | Spanish clinical PHI — adds non-English clinical coverage |

The **default mix** (what you get with no flags) is the first three
rows above: `pii_anon_eval`, `ai4privacy_400k`, `tab`. That's our
canonical corpus plus two industry leaders — the minimum mix the
research paper evaluates against. `meddocan` is opt-in when you need
Spanish clinical coverage.

```bash
# Default: pii-anon eval + AI4Privacy-400K + TAB
make train-swarm SWARM_MAX_RECORDS=0 SWARM_KFOLD=5

# Add Spanish clinical coverage
make train-swarm \
    SWARM_DATASETS=pii_anon_eval,ai4privacy_400k,tab,meddocan \
    SWARM_MAX_RECORDS=0 SWARM_KFOLD=5
```

### Stratified sampling

When `SWARM_MAX_RECORDS > 0`, each dataset is sampled **stratified by
language** before training kicks off. Without stratification, a
`SWARM_MAX_RECORDS=10000` run against a 60-language corpus collapses
to English because the source ordering is English-first. With it, the
10K-record cap preserves the language distribution of the full pool.

The default strata key is `language`. Override via `--stratify-by`:

```bash
# Balance across dataset sources as well as language
python scripts/train_swarm.py \
    --datasets pii_anon_eval,ai4privacy_400k,tab \
    --max-records 10000 \
    --stratify-by language,source_dataset
```

The implementation is
[`swarm_datasets.stratified_sample(records, n, strata_keys, seed)`](../src/pii_anon/swarm_datasets.py):
proportional allocation with a floor of 1 record per represented
stratum, deterministic under a seed, no oversampling when the pool is
already smaller than the target. The training script prints a pool
summary (per-dataset / per-language / per-entity-type counts) before
engine passes so you can spot-check balance before committing to a
multi-hour run.

`SWARM_MAX_RECORDS=0` (unlimited) skips stratification entirely — the
full pool is used as-is. For the paper-aligned mix that's ~560K records
across 60+ languages, mixing synthetic + peer-reviewed + real-world
legal + (optionally) clinical Spanish — the same training distribution
the pii-rate-elo research paper uses for its reference evaluation. Plan
for a multi-hour K-fold CV run at that scale; drop `SWARM_KFOLD=3` or
cap via `SWARM_MAX_RECORDS` if you need a faster iteration cycle.

### Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│  swarm_datasets.load_training_data(datasets, max_records)          │
│    ├ load_pii_anon_data()         — v1.3.0 canonical (~160K recs)  │
│    ├ load_ai4privacy()            — AI4Privacy 200k                │
│    ├ load_ai4privacy_400k()       — AI4Privacy 400k (2024, 17 lg) │
│    ├ load_tab()                   — Text Anonymization Benchmark   │
│    ├ load_meddocan()              — Spanish clinical PHI           │
│    └ load_conll2003()             — CoNLL-2003 NER                 │
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
