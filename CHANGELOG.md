# Changelog

All notable changes to `pii-anon` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.4.0] — 2026-04-18

Major additive release focused on **evaluation as a first-class offering**
and **extension workflows**. Three tools now share equal footing:
`pii-anon` (regex engine), `pii-anon-swarm` (fusion pipeline), and
`pii-rate-elo` (evaluation framework). Fully backward-compatible with
v1.3.0 — every public API added has a default and every modified
function accepts the existing call shape.

### Added — pii-rate-elo evaluation framework

- **`evaluate_external_system(predictor, ...)`** in
  `pii_anon.eval_framework.external_evaluator` — one-call API that scores
  a user-supplied PII detector against the `pii-anon` benchmark and
  returns a full `ExternalEvaluationResult` with composite score,
  per-record F1 (for bootstrap CI), and latency distribution.
- **`load_baseline_leaderboard()`** — reads the checked-in `artifacts/benchmarks/benchmark-results.json`
  (now vendored in the wheel at `eval_framework/baselines/benchmark-results.json`)
  so users can compare against the published baselines (`pii-anon`,
  `pii-anon-swarm`, Presidio, GLiNER, Scrubadub) without installing any
  competitor packages.
- **`BaselineLeaderboard.with_scorecard(sc, replace=True)`** — splices a
  user scorecard into the baselines and runs the Elo tournament.
- **`Leaderboard.from_benchmark_scorecard(bench, sort_by="elo")`** —
  classmethod that runs the tournament if the engine hasn't already,
  and returns a sorted leaderboard. Takes defensive copies of input
  scorecards so repeated calls produce deterministic results.
- **`resolve_predictor_path("module:callable")`** — resolves a Python
  import path to a predictor for CLI integration.
- **New CLI: `pii-anon rate-elo`** — takes `--predictor module:callable`
  and emits a markdown / JSON / CSV leaderboard comparing the user
  system against baselines. Writes `scorecard.json` and
  `leaderboard.{json,md,csv}` to `--artifact-dir`.

### Added — Tier 3 evaluation (LLM re-identification resistance)

- **`normalize_reidentification_resistance(recall, precision)`** — RRS
  metric per Lermen et al. 2026 `RRS = 1 − (recall × precision)`.
- **`normalize_quasi_identifier_coverage(removed, total, weights=None)`** —
  QIC metric for quasi-identifier removal rate.
- **`normalize_behavioral_signal_leakage(cosine_similarity)`** — BSL
  metric for stylometric leakage through de-identification.
- **`CompositeConfig.for_deployment(profile)`** — preset weight mixes
  for `"standard"`, `"high_security"`, `"high_throughput"` (re-ID
  resistance weight 0.30 / 0.60 / 0.20 respectively).
- **`CompositeConfig.f2_privacy_first()`** — β=2 F-score preset that
  doubles recall weight per the TAB 2022 cost model.
- **Tier 3 dataset fields on `EvalBenchmarkRecord`**:
  `behavioral_signal_density`, `reidentification_contribution`,
  `behavioral_signals`, `re_identification_resistance_score`,
  `estimated_reid_recall`, `tier3_risk_level`, `is_paired_profile`,
  `persona_id`, `linked_profile_id`, `profile_type`,
  `esrc_attack_target`, `expected_reidentification_difficulty`,
  `behavioral_signal_removal_attempted`, `context_preservation`.
- **`pii-anon-datasets` v1.3.0+ support** — loader reads
  `annotations` (v1.1+ canonical) alongside the legacy `labels` field;
  canonical dataset name `"pii_anon"` now the default with legacy
  fallbacks.

### Added — Industry-leadership bar (paper v10)

- **`FloorGateConfig.industry_leadership()`** — F1 ≥ 0.60, F2 ≥ 0.65,
  privacy ≥ 0.70, fairness ≥ 0.50, entity coverage ≥ 0.80.
- **`GovernanceThresholds.industry_leadership()`** — Elo ≥ 1600,
  RD ≤ 80, matches ≥ 10.
- **`evaluate_floor_gates(..., f2=...)`** — the floor gate evaluator
  now enforces `min_f2` when a threshold and score are both supplied.

### Added — Swarm extension workflows

- **`SwarmConfig.force_include_engines: tuple[str, ...]`** — pin a
  custom engine past the Layer 2 Jaccard pruner. Pinned engines
  bypass both the similarity check and the `max_engines` cap.
- **`SEMANTIC_TYPES`** gained `EMAIL_ADDRESS` and `CREDIT_CARD` — these
  had swarm precision of 0.46 and 0.48 on the benchmark because they
  bypassed the Layer 4 corroboration gate.
- **`swarm_datasets.load_jsonl(path, taxonomy_name=...)`** — generic
  JSONL loader for bring-your-own-data training. Supports `.jsonl.gz`,
  the `annotations` / `labels` alias, malformed-span rejection.
- **`swarm_datasets.register_taxonomy(name, mapping)`** — register a
  private entity-type vocabulary at runtime.
- **`swarm_datasets.register_dataset_loader(name, loader)`** — register
  a custom dataset loader addressable from the CLI.
- **`swarm_datasets.load_training_data([...])`** now auto-dispatches
  file-path-like entries (containing `/`, ending in `.jsonl` / `.jsonl.gz`
  / `.json`) to `load_jsonl`.
- **`swarm_learner.compute_sample_weights_from_records(records, rrs_boost, paired_profile_boost)`**
  — converts Tier 3 RRS annotations into XGBoost sample weights so the
  meta-learner sharpens on hard cases.
- **`swarm_learner.select_f2_threshold(scores, labels, beta=2.0)`** —
  F2-optimal emission-threshold sweep per paper v10.
- **`XGBoostMetaLearner.train(sample_weights=...)`** — now accepts
  per-example loss weights; `early_stopping` is plumbed through.
- **21-dim feature vector, `FEATURE_VERSION = 2`** — added
  `context_has_multilang_keywords` for non-English records
  (Spanish / French / German / Chinese / Japanese).
- **`TrainingRecord`** gained `behavioral_signal_density`,
  `re_identification_resistance_score`, `persona_id`,
  `is_paired_profile`.

### Added — Multilingual regex context coverage

- **`engines/regex/confidence.CONTEXT_WORDS`** now carries Spanish,
  French, German, Chinese, Japanese, Korean, Arabic, and Portuguese
  synonyms for the top-loss entity types (`PERSON_NAME`,
  `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `ADDRESS`,
  `LICENSE_PLATE`). Addresses the 56K+ non-English records in
  `pii-anon-datasets` v1.3.0 where English-only context boosting
  produced zero signal.

### Changed

- **Default dataset resolution** prefers the v1.1+ canonical
  `data/pii_anon.jsonl.gz` layout; legacy `eval_framework/data/pii_anon_eval_v1.jsonl.gz`
  is auto-detected as a fallback.
- **`_prune_redundant_findings`** processes pinned engines before the
  `max_engines` cap so they always survive.
- **`_aggregate_candidate`** now returns copies of caller-owned
  `EngineFinding` objects via `dataclasses.replace` rather than
  mutating them — double-scaling on retry is no longer possible.
- **`compute_composite(config=None)`** uses a cached module-level
  `_DEFAULT_CONFIG` sentinel for a ~25% speedup on the hot path.
  Defensive copies on `CompositeScore.config` guard against mutation
  leaking back into the singleton.
- **`SpanCandidate`** is now `slots=True` — ~40% memory reduction on
  the hot path.
- **`DawidSkeneAggregator`** caches a frozenset of prior keys at init
  time, eliminating per-`infer()` dict-key set rebuild.
- **`Leaderboard`** gained `from_benchmark_scorecard` classmethod; its
  `to_markdown` / `to_csv` / `to_json` surfaces remain unchanged.

### Fixed

- **XGBoost early-stopping** — `XGBoostMetaLearner.train(early_stopping=N)`
  now actually configures `xgb.train(early_stopping_rounds=N)`. Prior
  versions silently ignored the parameter.
- **Latency p50 calculation** — `external_evaluator` uses
  `statistics.median` for an unbiased p50 on even-length sample lists.
  Warmup records' latencies are excluded from the measured distribution.
- **F2 threshold sweep fallback** — returns `(0.5, 0.0)` cleanly when
  no threshold yields a positive F_beta (avoids divide-by-zero).

### Documentation

- **New**: `docs/pii-rate-elo.md` — algorithm reference (Tier 1/2/3,
  F2, RRS, QIC, BSL, Elo/Glicko, floor gates, deployment profiles,
  industry-leadership bar).
- **New**: `docs/evaluate-your-pipeline.md` — end-to-end guide for
  scoring your own detector (programmatic API + CLI).
- **New**: `docs/swarm-architecture.md` — 4-layer pipeline, 21-feature
  vector, retrain procedure, Tier 3 sample weighting.
- **New**: `docs/extend-swarm.md` — unified bring-your-own-engine +
  bring-your-own-data walkthrough.
- **New**: `docs/autoresearch-integration.md` — iterate on the library
  with the `pii-anon-autoresearch` experiment loop.
- **Updated**: `docs/api-reference.md`, `docs/quickstart.md`,
  `docs/engine-plugin-guide.md`, `docs/release-guide.md`, `docs/README.md`.

### Packaging / platform

- `[tool.setuptools.package-data]` — added
  `eval_framework/baselines/*.json` so the vendored baseline leaderboard
  ships with the wheel.
- `pyproject.toml` dependency `pydantic` now capped at `<3` to guard
  against the next breaking release.
- Added classifiers: `Operating System :: OS Independent`, `Typing :: Typed`,
  `Intended Audience :: Information Technology / Science/Research`.
- **CI**: `cross-platform-smoke` job now runs the core test suite +
  CLI smoke on macOS-latest + Windows-latest (Python 3.12) alongside
  the full Linux matrix.

### Migration notes

No breaking changes. To adopt the new APIs:

```python
# Old — still works
from pii_anon.eval_framework import compute_composite, PIIRateEloEngine

# New — score your own pipeline against baselines in one call
from pii_anon.eval_framework import (
    evaluate_external_system, load_baseline_leaderboard,
)

result = evaluate_external_system(my_detector, max_records=2_000)
print(load_baseline_leaderboard().with_scorecard(result.scorecard).to_markdown())
```

```bash
# New CLI
pii-anon rate-elo --predictor my_pkg:predict --max-records 2000
```

---

## [1.3.0] — 2026-03-27

- Added the swarm pipeline (Dawid-Skene + XGBoost meta-learner + F2
  threshold selection).
- Renamed `pii-anon-ensemble` → `pii-anon-swarm`.
- Updated benchmark to 151K records.

## [1.2.1] — 2026-03-21

- MoE (mixture-of-experts) swarm architecture.
- Removed hardcoded version references.

## [1.1.0] — 2026-03-15

- `pii-anon-eval-data` v1.1.0 compatibility.
- Expanded benchmark dataset coverage.

## [1.0.0] — 2026-02-23

- Initial PyPI release.
- Regex + checksum detection engine, orchestrator, and basic evaluation
  framework.
