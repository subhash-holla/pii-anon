# Changelog

All notable changes to `pii-anon` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — sp2: external assessment supremacy + 12-player pii-rate-elo

### Added
- **First-party BYO predictors** — `first_party_predictor("pii_anon" |
  "pii_anon_swarm")` in `eval_framework/byo_pipeline.py`: pii-anon's own two
  detection surfaces expressed as ordinary `Predictor` callables emitting
  NATIVE labels, built on the engine seam (no orchestrator dependency —
  resolves the SWITCH-POINT(ORCH)). The swarm variant pools regex +
  GLiNER/Presidio/Stanza (when importable) through `build_fusion("swarm")`.
- **`pii-anon rate-elo-assessment`** — rates EVERY detector in a merged
  `pii-anon-baseline-results/v1` assessment artifact (the pii-anon-eval-data
  `baselines` leaderboard output) via per-entity-type F2 matches through the
  `PIIRateEloEngine`. Fail-loud no-fabrication ingestion validation; report
  carries Elo±RD with 95% CIs, pairwise-significance matrix, per-system
  strongest/weakest entity types, and an explicit axis-disclosure block.
- **External-coverage tranche** — 21 new native detection labels grounded in
  sampled eval-data gold shapes (TAX_ID, JOB_TITLE, HEALTH_CONDITION,
  MEDICATION_NAME, HEALTH_INSURANCE_ID, CREDIT_CARD_FRAGMENT, VISA_NUMBER,
  PRESCRIPTION_NUMBER, DEVICE_IDENTIFIER, SOCIAL_MEDIA_HANDLE,
  EDUCATION_LEVEL, GENDER, NATIONALITY, ETHNICITY, POLITICAL_OPINION,
  RELIGIOUS_BELIEF, MARITAL_STATUS, HOUSEHOLD_SIZE, VEHICLE_MODEL,
  PROCEDURE_NAME, BIOMETRIC_ID), with zero-width-character-tolerant value
  classes for the corpus's adversarial obfuscation; ISO-8601 datetime
  pattern; corpus-form additions for SWIFT/DL/INVOICE/COURT_CASE/DOCKET/
  SALARY/API_KEY.

### Fixed
- **Strict-extent detection hygiene** (dev-split-driven, eval-integrity
  discipline — tuned on dev, reported on test): PERSON_NAME role-word
  absorption, next-field-label absorption, title-prefix extent conventions
  (name-only for title+full-name, title-kept for title+surname), dialogue-
  speaker form; ORGANIZATION sentence/newline crossing and scoped-case
  context captures; dotted-date IP-fragment false positives; nested/duplicate
  same-type emission dedup; cross-type arbitration (specific-type spans
  shadow generic PERSON/DATE matches). Dev-split (en, n=4000) micro-F2
  0.767 → 0.890 at 63/63 type coverage; internal census guard n=10000
  F2 0.853 → 0.889 with p50 0.37 ms (speed ceiling green).

---

## [1.5.0-rc.1] — 2026-06-09

**The PDLC SOTA program release candidate** (branch `pdlc/sota-program`; LOCAL-ONLY
tag — not published). The full program changelog with per-story trace IDs lives at
`dev-assist-artifacts/06-documentation/03-authoring/changelog.md`; the program
narrative at `.../project-journey.md`.

**Honest status:** the SDO (state-of-the-art dominance) verdict is **NOT_YET** —
binding constraint G6 (raw-detection F2 non-inferiority, draw-sensitive; attributed
to evaluation methodology, not a code regression); G1/G2/G3/G4/G5/G7 all PASS on a
certified run. All program cohort research is AGENT_SIMULATED with a tracked Pass-2
roadmap. This RC claims honest machinery, not the crown.

### Added — evaluation integrity & the SDO gate
- Rating-engine ladder behind `RatingEnginePort` (+ `pii_anon.rating_engines` entry
  points): glicko-legacy, `bradley-terry-mle`, claim-grade `bayes-bt` with an
  NFR-001 convergence gate; coherent significance + Davidson ties (S3-01..04).
- The `CompetitiveSupremacyGate` (`pii-anon supremacy`) — G1–G7 guarantee verdicts
  with a single binding constraint; fabrication-hardened across 9 adversarial
  closes (11 holes / 6 fabrications found and closed; final closes 0-upheld).
- The certified-run producer (`pii-anon canonical-run`) with the fail-closed
  `CanonicalRunGate`; NFR-009 latency ceilings registry (S7-02/S7-04).
- Distinct de-identification scorer families — `AnonymizationScorer` vs
  `PseudonymizationIntegrityScorer`, never merged (AX-004; S4-01) — and the
  per-class calibration/selective-risk reporter (S4-03).

### Added — privacy attack surface (representative; sandboxed)
- `eval_framework/attacks/`: `ReidAttack`/`MiaAttack` protocols, the resource-
  sandboxed runner, the Tier-3 LLM-adversary representative (RRS/QIC/BSL,
  de-circularized) and the LiRA-shaped + Secret-Sharer MIA family (S5-01..04).

### Added — swarm routing & the recall floor
- `SharedLayerProjector` + `FloorProjectingFusion`: recall-floor by construction
  (`entities(output) ⊇ entities(shared)`), per-language ε-gate CI teeth (S1).
- MoE learned routing core: feature-conditioned `route()`, the signed
  `gate_v1.json` verify-on-load boundary, `DistilledTopKGate`, aux-loss-free SLA
  bias (S2-01/02/04/05).

### Added — agentic privacy
- `QueryAwareMaskingGate` — subtractive-on-mask, default-to-mask (S6-01).
- 4-channel least-privilege interception + leakage-Sankey audit (S6-02/05).
- `EncryptedSQLiteTokenStore` — AEAD at rest, AAD-bound rows, envelope-wrapped
  DEK, fail-loud (S6-03, adversarially closed).

### Added — extensibility & multimodal
- BYO-pipeline SDK: `pii_anon.byo_pipelines` entry points, `BYOPipelineRegistry`,
  `evaluate_incumbent` / `build_identical_path_leaderboard` — incumbents and BYO
  systems scored by the literal same evaluator (S6-04).
- Native-format readers behind `Iterator[IngestRecord]`: a pure-stdlib PDF text
  reader (bounded FlateDecode inflate — zip-bomb hardened), capability-honest
  OCR/DICOM/audio seams + `ocr`/`dicom` extras, `pii_anon.readers` entry points
  (S7-01).
- Multilingual context activation (CJK/Hangul/Arabic keywords now fire) + the
  fail-closed powered worst-group fairness gate `evaluate_language_fairness`
  (S7-03).

### Documentation
- Docs discoverability with standing teeth (`tests/test_docs_discoverability.py`),
  the anonymization-vs-pseudonymization guide, the certify-a-run guide, the
  program-surfaces API reference; `make docs-smoke` fixed (S7-05). Stage-6
  documentation set compiled (verdict: DOCUMENTED).

### Changed
- Trove classifier → `4 - Beta` for the RC (revert at final 1.5.0).

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
