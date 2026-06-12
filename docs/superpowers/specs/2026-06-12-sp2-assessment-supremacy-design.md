# sp2 — External Assessment Supremacy + 12-Player pii-rate-elo (Design)

**Date:** 2026-06-12 · **Status:** APPROVED (user, via design gate)
**Repos touched:** CODE (`pii-anon-core/pii-anon-code`), DATA (`pii-anon-core/pii-anon-eval-data`)

## Goal

1. Vanilla pii-anon and pii-anon-swarm run inside the `pii-anon baselines`
   assessment in the DATA repo exactly like the other 10 detectors, so every
   future run of "the same evaluation" includes them, on the English split and
   later the entire dataset.
2. At least one of (vanilla, swarm) ranks #1 by micro-F2 on the English test
   split — the bar is aws 0.737 / gliner 0.735 (OSS-best and overall-best are
   0.002 apart, effectively one bar: **F2-micro ≥ 0.74**) — and both offerings
   post best-in-table entity-type coverage (~40+/63 vs aws 24/63).
3. pii-rate-elo ingests the merged assessment artifact and produces a
   12-player rating report with clear per-player indicators.

## Ground truth about the benchmark (drives all decisions)

- Artifact of record: `results/baselines/_partial-all-10/baseline_results.json`
  (DATA repo) — full English test split, 30,995 records, 201,701 gold spans,
  matches the user's table exactly. Cloud results merge from here; **no cloud
  API re-spend**.
- Scoring: `strict-v1` exact `(start, end, entity_type)` matching, multiset
  semantics, whitespace-trimmed predictions; F2-micro is the ranking metric.
- Gold distribution: PERSON_NAME 32.4%, EMAIL_ADDRESS 14.6%, PHONE_NUMBER
  8.7% — strict-span recall on these three decides micro-F2.
- Coverage = label-map reachability over the 63-type canonical taxonomy
  (`src/pii_anon_datasets/taxonomy.py`).
- Tuning surface: **dev split only** (en dev = 15,484 records / 102,071 gold).
  Test split is reserved for the final reported run.

## Part 1 — DATA repo: two native adapters (Option A, approved)

- `baselines/pii_anon_baseline.py` (name `pii_anon`, vanilla
  `mode=weighted_consensus`) and `baselines/pii_anon_swarm_baseline.py`
  (name `pii_anon_swarm`, `mode=swarm`), both conforming to
  `DetectorAdapter` (`src/pii_anon_datasets/baselines/contract.py`):
  `available()` via `find_spec("pii_anon")` (no heavy import), lazy `build()`,
  `detect()` returning `AdapterSpan`s, explicit `label_map` from pii-anon
  native labels → 63-type canonical (renames like US_SSN→SOCIAL_SECURITY_NUMBER,
  VIN→VEHICLE_IDENTIFICATION_NUMBER, GPS_COORDINATES→LATITUDE_LONGITUDE, …;
  deliberate drops map to `None`).
- Registry entries in `baselines/registry.py`; pyproject optional group
  (e.g. `pii-anon = ["pii-anon"]`) documented as editable-install of the
  sibling repo. Note: both repos install a `pii-anon` console script — document
  `python -m pii_anon_datasets.cli` as the collision-safe invocation.
- Tests in DATA repo: contract conformance, label-map validity against the
  taxonomy (DX-02 fail-loud), smoke detect, registry resolution.
- Commits: scoped to the new files on `feat/v2-scoring-harness`; user WIP
  (`gcp_dlp_baseline.py`, `tests/test_baselines_adapters.py`) untouched.

## Part 2 — CODE repo: detection iteration (dev-split discipline)

Measure → analyze per-entity FN/FP → fix → re-measure, in priority order:

1. **Strict-span hygiene** on PERSON_NAME / EMAIL_ADDRESS / PHONE_NUMBER
   (boundary conventions: trailing punctuation, titles, multi-token extents).
2. **Free coverage** via label-map renames (no behavior change).
3. **Missing structured types**: VISA_NUMBER, PRESCRIPTION_NUMBER,
   HEALTH_INSURANCE_ID, CREDIT_CARD_FRAGMENT, DEVICE_IDENTIFIER,
   SOCIAL_MEDIA_HANDLE, URL, TIMESTAMP (+ POSTAL_CODE et al. as renames).
4. **Lexicon/context types**: GENDER, MARITAL_STATUS, EDUCATION_LEVEL,
   NATIONALITY, JOB_TITLE, HEALTH_CONDITION, MEDICATION_NAME, PROCEDURE_NAME.
5. **Swarm**: same loop; retune emission/corroboration gates and recalibrate
   Dawid-Skene / XGBoost / temperature on the dev split if precision drags.

Constraints (all hard):
- General-purpose recognizers only — no template-specific hacks
  (eval-integrity axiom).
- NFR-009 ceilings hold: vanilla p50 ≤ 1 ms, swarm p50 ≤ 500 ms; sp1 perf
  gate stays green.
- `competitive_supremacy.py` + canonical-run producer byte-identical
  (md5 `3b842e81…` / `d8f0f80e…`) → no SDO adversarial close triggered.
- CODE suite + ruff + both-mypy green; DATA suite green.
- User-WIP in CODE (`orchestrator.py` calibration wiring + tests, README,
  benchmark artifacts) left intact and uncommitted.

## Part 3 — CODE repo: pii-rate-elo assessment ingestion (12 players)

- New module under `src/pii_anon/eval_framework/rating/` (e.g.
  `assessment_ingest.py`): parse `pii-anon-baseline-results/v1` merged JSON →
  per-player scorecards (micro/macro P/R/F1/F2, coverage, per-entity table).
  Validation discipline mirrors the no-fabrication rules: every consumed value
  routed through finite/[0,1]/non-blank-str checks; malformed players are
  excluded loudly, never defaulted.
- **Elo**: matches are per-entity-type F2 comparisons (63 fields × 66 pairs ×
  12 players) through the existing `PIIRateEloEngine` — granular, fully
  comparable across cloud and local players. Tournament summary feeds the
  report (ratings, RD, pairwise significance, min distinguishable diff).
- **Composite with axis disclosure**: detection axes from the artifact for
  everyone; latency/throughput only where measured (ours, locally measured;
  cloud marked UNMEASURED and reweighted — never silently zeroed); Tier-3 only
  where available. Per-player "axes evaluated" is printed in the report.
- CLI: `pii-anon rate-elo --assessment-results <path>` (new mode), writing
  leaderboard.md/json + significance matrix + per-entity strength/weakness
  indicators.
- Tests with exact reference-value anchors (program lesson: never bounds).

## Final deliverable run (approved)

1. Final test-split run of both adapters (swarm ≈ 45–90 min).
2. Merge with `_partial-all-10` → 12-player leaderboard.
3. `rate-elo --assessment-results` over the merged artifact → rating report.
4. Update `docs/pii-rate-elo-value.md` narrative to the 12-player story.
5. Honest reporting of final standings, whatever they are.

## Out of scope (Pass-2 candidates)

- Multilingual ("entire dataset") runs — the adapters are language-agnostic
  by construction, so extension is a run, not a build; non-English detection
  quality work is its own iteration.
- Generic BYO entry-point bridge in the DATA harness (Option C).
- Re-running cloud detectors.
