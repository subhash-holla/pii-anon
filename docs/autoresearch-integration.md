# Iterating on pii-anon with the autoresearch library

`pii-anon-autoresearch` is the autonomous research loop that proposes,
runs, and records configuration experiments against `pii-anon` and
`pii-anon-swarm`. This document shows how the two repositories fit
together, what autoresearch can tune today, and the recommended
workflow for a production retrain driven by its results.

For the detection architecture itself, see
[swarm-architecture.md](swarm-architecture.md). For the evaluation
framework, see [pii-rate-elo.md](pii-rate-elo.md).

---

## The two repos at a glance

| Repo | Role |
|---|---|
| `pii-anon-code` (this repo) | The library — detection engines, swarm fusion, evaluation framework, benchmark artifacts. **Evaluation code is sacred — autoresearch never touches it.** |
| `pii-anon-autoresearch` (sibling) | The experiment loop — config-only tuning, results.tsv ledger, promote-to-library script. **Modifies `config.py` only.** |

This split is intentional. Separating configuration from evaluation
means every experiment is scored against the exact same metrics, so
the trajectory of `results.tsv` is a genuine quality curve rather than
an artefact of evaluator drift.

---

## The loop

```
┌──────────────────────────────────────────────────────────────────┐
│  program.md — hypothesis statement                                │
│    "Lowering CONTEXT_PENALTY from 0.15 to 0.10 for HIGH_FP_TYPES  │
│     should lift recall without crashing precision."               │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  autoresearch/config.py — modify one tunable                      │
│    CONFIDENCE_TUNING["CONTEXT_PENALTY"] = 0.10                    │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  python experiment.py --description "lower penalty"               │
│    ├ imports from pii_anon.* (immutable evaluate.py loads config) │
│    ├ runs PIIOrchestrator + swarm against pii_anon_benchmark      │
│    ├ computes composite + Elo vs. competitor baselines            │
│    └ appends row to results.tsv + prints grep-friendly metrics    │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Decision gate                                                    │
│    composite improved + no regression → keep config.py            │
│    regression                        → git checkout config.py     │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Every N accepted experiments:                                    │
│  python promote.py --apply                                        │
│    → writes patches back into pii-anon-code                       │
│    → library now ships the improved defaults                      │
└──────────────────────────────────────────────────────────────────┘
```

`autoresearch/program.md` describes this loop formally and enumerates
the tunable categories. The two most useful files to read first:

- [`autoresearch/config.py`](../../pii-anon-autoresearch/config.py) — every knob
- [`autoresearch/evaluate.py`](../../pii-anon-autoresearch/evaluate.py) — the immutable scorer (do not edit)

---

## What can be tuned today

| Category | Examples | Source |
|---|---|---|
| Confidence tuning | `CONTEXT_BOOST`, `CONTEXT_PENALTY`, `CONFIDENCE_CAP`, `CONFIDENCE_FLOOR` | injected into `pii_anon.engines.regex.confidence` at runtime |
| Context keywords | `EXTRA_CONTEXT_WORDS[entity_type]` | merged into `CONTEXT_WORDS` |
| Engine weights | per-engine weight in `WeightedConsensusFusion` | `FusionConfig.engine_weights` |
| Deny / allow lists | global + per-entity-type denylists | `DenyListConfig`, `AllowListConfig` |
| Pattern overrides | new `PatternSpec` entries appended | `PATTERN_REGISTRY` |
| Swarm knobs | `SWARM_EMISSION_THRESHOLD`, `SWARM_CORROBORATION_MIN`, fusion weights | `SwarmConfig` |

All changes happen **via the configuration layer only** — the library's
internal code is not touched until you run `promote.py`.

---

## Wiring the v1.3.0 Tier 3 signals into the loop

The library changes made in this release give autoresearch three new
things to optimise against:

### 1. Multilingual context coverage

`CONTEXT_WORDS` in
[engines/regex/confidence.py](../src/pii_anon/engines/regex/confidence.py)
now ships Spanish, French, German, Chinese, and Japanese synonyms for
high-loss entity types (PERSON_NAME, EMAIL, PHONE, ADDRESS,
LICENSE_PLATE, etc.). Autoresearch can extend this further by pushing
additional translations through `EXTRA_CONTEXT_WORDS` in `config.py`
and observing per-language recall in the results.tsv ledger.

### 2. Tier 3 hard-case weighting

`swarm_learner.compute_sample_weights_from_records` up-weights
low-RRS records during XGBoost training. An autoresearch experiment
can sweep `rrs_boost ∈ [1.0, 3.0]` and `paired_profile_boost ∈ [1.0, 2.0]`
to find the weighting that maximises F2 without collapsing precision
on the common (high-RRS) case.

### 3. F2 threshold selection

`swarm_learner.select_f2_threshold` picks `emission_threshold`
post-training. Autoresearch can override `beta` in the sweep (e.g.
β=1.5 for a less recall-heavy stance) and record the resulting
composite / F1 / F2 trajectory.

---

## Workflow for a production retrain

```bash
# 1. Freeze the baseline — a single experiment that captures current swarm.
cd ~/projects/pii-anon/pii-anon-autoresearch
python experiment.py --baseline --description "pre-v10 baseline" | tee -a runs.log

# 2. Iterate in autoresearch until composite plateaus on results.tsv.
#    Each accepted experiment modifies config.py with one tunable change.
python experiment.py --description "try: swarm emission 0.45"
python experiment.py --description "try: ES/FR/DE context words for EMAIL"
python analysis.py --export improvements.csv   # review trajectory

# 3. Promote the winning config back into the library.
python promote.py --dry-run    # inspect proposed patches
python promote.py --apply      # write patches into pii-anon-code

# 4. Retrain the swarm with the new config baked in.
cd ~/projects/pii-anon/pii-anon-code
make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy,conll2003 \
                 SWARM_MAX_RECORDS=0 SWARM_KFOLD=5

# 5. Regenerate the benchmark leaderboard.
make benchmark-full

# 6. Validate the industry-leadership bar.
python -c "
from pii_anon.eval_framework import (
    FloorGateConfig, GovernanceThresholds, load_baseline_leaderboard,
)
lb = load_baseline_leaderboard()
print('Leadership thresholds:', FloorGateConfig.industry_leadership())
print('Elo/RD thresholds:  ', GovernanceThresholds.industry_leadership())
for sc in lb.scorecards:
    print(f'  {sc.system_name}: composite={sc.composite_score:.3f}, '
          f'F1={sc.f1:.3f}, elo={sc.elo_rating:.0f}, RD={sc.elo_rd:.0f}')
"
```

---

## The before/after leaderboard

The `results.tsv` ledger in autoresearch is the before/after record of
every configuration change. Combined with the committed library
baseline at
[`artifacts/benchmarks/benchmark-results.json`](../artifacts/benchmarks/benchmark-results.json),
there are two comparable surfaces:

- **autoresearch/results.tsv** — per-experiment row with composite, F1,
  recall, precision, latency, Elo, `status` (keep / discard / crash /
  baseline), and description. The column order is stable so
  `analysis.py --export` produces a clean CSV for dashboards.
- **pii-anon/artifacts/benchmarks/benchmark-results.json** — the
  post-retrain snapshot consumed by `load_baseline_leaderboard()`.
  Update this by running `make benchmark-full` after a promote+retrain
  cycle.

The typical narrative is: "results.tsv shows a 0.03 composite lift
across 12 experiments in autoresearch; the promote + retrain cycle
landed that lift in the library; `benchmark-results.json` now reflects
it; `pii-anon rate-elo` users immediately see a stronger baseline."

---

## What autoresearch will not touch

Per `autoresearch/program.md`:

- `autoresearch/evaluate.py` is immutable — scoring code changes would
  invalidate the entire history of results.tsv.
- `pii-anon-code` internals — the library changes only via
  `promote.py`, which generates file patches from the config diff.
- Test size / dataset filtering — experiments must run against the
  full evaluation slice so metrics stay comparable.

This discipline is what makes the loop trustworthy. Any deviation is a
red flag and should be rejected in code review.

---

## Extending autoresearch

Two places are worth looking at when you need to add a new tunable:

1. **`autoresearch/config.py`** — add a constant with a default and a
   docstring describing the expected range. Import it in
   `autoresearch/evaluate.py` within the `_load_config()` section.
2. **Corresponding library seam** — the configuration surface
   (`CONTEXT_WORDS`, `FusionConfig`, `SwarmConfig`, `PATTERN_REGISTRY`,
   `EXTRA_CONTEXT_WORDS`) must accept the value without a code change.
   If it doesn't, land the seam in the library first and only then add
   the tunable.

The `promote.py` script knows about a fixed set of promotable
parameters (confidence tuning, context words, pattern overrides). Adding
a new category means extending `promote.py`'s patch generator too —
look for `_PROMOTABLE_CATEGORIES` in that file.
