# Changelog

All notable changes to `pii-anon` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.7.0rc1] — 2026-07-13 — sp4-sp7: external validity, rate-elo hardening, swarm NER channel, world-leading quality program

> Release candidate, LOCAL-ONLY tag — not published. Publication remains gated on the
> external-validity disclosure (FR-027, `docs/external-validity-report.md`): measured
> zero-shot performance against non-home datasets must be disclosed before any
> "best-in-class" broadcast. The SDO competitive-supremacy verdict is honestly
> **NOT_YET** (binding J ≈ 0.28) — a "not yet #1", never a SOTA claim; the
> `competitive_supremacy.py` / `canonical_run.py` control paths are byte-identical
> (md5 `3b842e81`) across every close in this release.
>
> Home-benchmark results are synthetic and home-tuned and are **NOT** an
> external-validity claim. Headline movement this release: **home vanilla strict-F2
> 0.8916 → 0.9114** (cumulative sp7, home substrate); **TAB relaxed F2 0.10 → ~0.49**
> zero-shot on real court documents (~5×; still a disclosed transfer gap, not a
> general-performance claim); external zero-shot relaxed F2 spans **0.10 (TAB) to
> 0.39–0.46 (Gretel finance)** vs 0.89 at home. The LLM-reconstruction-resistance
> report now returns a *measured* bound against a disclosed adversary class (never
> "impossible"): **verbatim leakage ~5.9%, masked Tier-3 re-identification ~6.1% vs a
> 98.0% unmasked baseline** (n=1500, live-BSL adversary, 0 FR-036 parity violations).

### Added (sp7 — world-leading quality program, largest tranche)

- **Natural-language & locale date grammar with DOB-cue promotion** — detects prose
  dates ("27 May 1994", "1st day of January, 2023", "May 1994", "3rd of April 1980")
  and space-form datetimes, and re-types a date to `DATE_OF_BIRTH` when a
  word-bounded birth cue sits within 30 chars (leak-safe, cue-gated relabel). The
  single biggest external lever: home vanilla strict-F2 **+0.0037** (precision
  +0.0167, recall flat), **TAB relaxed F2 0.101 → 0.397 (+0.295, ~4×)**, nemotron
  +0.0116, gretel +0.0073.
- **Labeled-field value bridge** — a shared cue→value extractor (`SSN:`,
  `Date of Birth:`, `Account Number:`, …) tolerant of markdown/quote wrappers that
  extracts the value span (never the label word) and types it from the label; runs
  on both detection and masking paths (over-masking a labeled value is safe) and every
  cue maps only to a supported type. Home vanilla strict-F2 **+0.0016**; nemotron
  **+0.0227 (0.335 → 0.358)**, gretel +0.0113, ai4privacy +0.0083.
- **`CUSTOMER_ID` labeled-field type** — genuine new masking coverage of a
  previously 0%-recall quasi-identifier; nemotron relaxed F2 +0.0064, home neutral.
- **Organization / institution detection** — new patterns for institutions
  (Ministry/Tribunal/Board/…), courts (descriptor-gated so "Sinop Assize Court"
  matches but the "Birch Court" street suffix does not), "Ministry of Justice" forms,
  and firms ("Harper & Associates"). TAB 0.397 → 0.435, nemotron +0.0024.
- **Name grammar & span hygiene (safe subset)** — honorific + Unicode full names
  ("Osman Çağlayan"), honorific + initials ("Mr S. Esmer" → "S. Esmer"),
  section-word-guarded First + mid-initial + Surname, and in-place diacritic
  widening; names captured without the honorific to match home convention. Home
  strict-F2 +0.0003 (precision up), TAB 0.444 → 0.489, gretel +0.010.
- **Surname mention-propagation** — an additive detection post-pass that propagates
  each detected multi-token person's surname to bare standalone occurrences the
  grammar missed (a partial-mention leak the coreference masker cannot reach, since it
  needs the full verbatim value). Home strict-F2 **0.9084 → 0.9114 (+0.0030)**,
  `PERSON_NAME` recall 0.8004 → 0.8160; O(n) occupancy scan.
- **Public-domain geo gazetteer (`LOCATION`)** — curated countries + US states with
  no external gazetteer dependency; ambiguous single tokens (Georgia/Jordan/Reading)
  fire only with a location cue, and locations nested inside a detected address are
  dropped (leak-safe — the address still masks). Home neutral, TAB +0.0071. A
  `geo_subtype()` advisory `STATE`/`COUNTRY` signal is attached to the finding's
  explanation only — the scored `entity_type` stays `LOCATION`, so home scoring is
  byte-identical (scoring the taxonomy split measured home −0.0021 and was rejected).
- **Unicode-normalization detection pre-pass** — strips zero-width/format characters
  (ZWSP/ZWNJ/ZWJ/BOM/soft-hyphen) and NFKC-folds fullwidth/compat forms for
  detection, remapping spans back to original offsets so the mask still covers the
  obfuscation characters. ASCII fast-path keeps home byte-identical; strictly additive
  (spans only expand), closing the zero-width/fullwidth evasion class.
- **Multilingual detection hygiene** — same-span `SSN`/`NATIONAL_ID` dedup plus an
  honorific-`ORGANIZATION` veto. Home overall strict-F2 **0.9033 → 0.9062 (+0.0029)**;
  Arabic +0.032, Chinese +0.031, Hindi +0.026, Korean +0.024, Japanese +0.007.
- **Value-consistent (coreference) masking — production and report** — after the
  primary transform, sweeps the original field text for every remaining verbatim
  occurrence of a detected sticky-type value (13 types, cross-field) and redacts it
  with the same replacement; default ON, no-coreference payloads stay byte-identical,
  and the runtime masker shares its type/length rules with the report masker so they
  stay in lock-step. Verbatim-leakage bound **8.35% → 5.93% (−29% rel)**, `PERSON_NAME`
  22.5% → 15.4%; detection byte-identical.
- **Per-type verbatim-leakage breakdown** in the assurance report — localises
  residual leakage (`PERSON_NAME` ≈91% of it; structured EMAIL/PHONE/ADDRESS/SSN all
  measure 0.00%).
- **LLM-reconstruction-resistance report (`reconstruction_resistance_report`)** — a
  reproducible, shareable JSON + HTML report that claims a *measured bound against a
  disclosed adversary class, never "impossible"*. Three axes, each with a Wilson 95%
  CI and a non-strippable caveat: verbatim leakage (direct-reconstruction bound),
  Tier-3 re-identification (RRS/QIC/BSL, masked vs unmasked baseline), and membership
  inference (LiRA-shape, TPR@low-FPR); it also asserts FR-036 stream/batch
  scrub-decision parity. Simulated adversaries are labelled `AGENT_SIMULATED` and a
  real-LLM/full-power run is a separately-labelled switch-point. Flagship run (n=1500,
  live-BSL adversary): **verbatim leak 5.93%, masked re-id 6.13% vs a 98.0% unmasked
  baseline (~88pp protection delta), 0 parity violations**.
- **Per-(engine, type) single-engine acceptance for the swarm** — opens the GLiNER
  semantic channel that per-type confidence bars had left structurally unreachable
  (GLiNER confidence caps ~0.87). All default GLiNER bars sit at or below the per-type
  fallback, so the change is strictly additive and leak-safe. Home swarm strict-F2
  **0.8989 → 0.8996 (+0.0007)**; nemotron swarm relaxed F2 **0.3830 → 0.3914
  (+0.0084)**; ships as default. No retrain (bars derived from measured GLiNER
  confidence ranges + the home-floor gate). SDO verdict re-certified **NOT_YET**
  (J ≈ 0.28) after regeneration.

### Changed (sp7)

- **Address grammar rework (masking-path)** — dropped the global case-insensitive
  match that was slurping lowercase prose, added a first-token function-word guard,
  promoted the full USPS unambiguous-suffix set (additive), gated common-noun suffixes
  behind a unit/postcode lookahead, and stopped addresses from crossing line breaks.
  Home strict-F2 neutral (0.9035 → 0.9035); gretel 0.422 → 0.456 (+0.034), TAB +0.0044,
  nemotron +0.0014, ai4privacy −0.0014.
- **Eval-only Title-Case-noise person/org suppression** — suppresses
  markdown-header / determiner-led / all-header-word false positives on the scoring
  path only; production masking still over-masks headings (the eval-emission is a
  proven subset of production-emission, so no masking-convention change). Home vanilla
  F2 0.8916 → 0.8932 (+0.0016) (precision +0.0054, zero recall loss); nemotron +0.011,
  gretel +0.020.
- **Numeric-identifier scoring guards (scoring-only) plus additive hemisphere GPS**
  — drops money/rating-shaped GPS and cue-less bare-9 SSNs on the scoring path
  (masking path untouched) and adds a `"40.7234 N, 123.1235 W"` GPS pattern
  additively. Home neutral; nemotron +0.0012, gretel +0.0019. An IBAN mod-97 guard was
  cut, not shipped, once the gate caught it removing 225 checksum-invalid home gold
  IBANs.
- **Live background-knowledge (BSL) adversary is now the report default** — an empty
  `auxiliary_knowledge` had made the BSL channel identically 0, biasing the re-id bound
  optimistically; with the background channel live the report is more honest (unmasked
  baseline 98.0%, masked 6.13% at n=1500). Added `max_token_store_size` /
  `max_ledger_scopes` pass-throughs (defaults unchanged).
- **`run_file` error surfacing** — failures now produce typed error entries with
  per-failure and summary warnings and an `on_error="raise"` option, instead of failing
  silently.
- **Deny-list canonical folding** — lowercase + whitespace-collapse +
  edge-punctuation-strip applied symmetrically at load and lookup, so `"John Doe"`,
  `" john  doe "`, and `"John Doe."` all match one entry. Conservative (does not fold
  confusables/zero-width — the deny-list only drops). Home neutral.

### Fixed (sp7)

- **Dotted-format phone numbers were rejected by the validator** — the version-number
  false-positive rule discarded every dotted phone (e.g. `765.340.8856`); the 3-3-4
  shape is now exempted while NANP rules still apply. Home neutral, real-world recall
  win.
- **International phone formats now pass the NANP validator** — positional NANP /
  invalid-area-code rules are gated to weakly-formatted candidates, so `+CC`,
  `00`-international, trunk-`0`, and strong E.164 shapes are no longer rejected. Home
  delta 0.
- **Provenance scope fabrication-vector** — the canonical producer stamped
  `scope=data-v2.0.0` while the substrate was v2.2.0; the resolver and validator now
  stamp and accept `data-v<resolved-version>`, and the benchmark artifacts the
  docs-link test reads were restored. Detection numbers unchanged (label only).

### Security (sp7)

- **Silent streaming PII leak in the segmenter (FR-036 parity)** — a multi-token
  entity straddling a segment boundary could be detected by neither adjacent window at
  small overlap (batch found the phone `415 555 1234`; segmented at overlap 0 dropped
  it entirely). A `MIN_SAFE_OVERLAP_TOKENS=24` floor (plus a step ≥ 1 guarantee for
  degenerate configs) now guarantees any entity up to 24 tokens is wholly contained in
  at least one window regardless of the caller's overlap; raising overlap only adds
  de-duplicated re-detection, so segmented detection is a proven superset of batch. The
  default configuration is byte-identical. Documented residual: entities longer than 24
  tokens.
- **Numeric payload fields silently leaked PII** — both the detection and transform
  loops skipped every non-string field, so a value like `{'ssn_number': 573337773}`
  passed through completely unmasked; numeric scalars (int/float, never bool) are now
  coerced for detection and a numeric field with findings is masked, while benign
  numeric fields keep their original value and type. Strictly additive.

### Added (sp6 — swarm NER channel opened)

- **Swarm NER channel opened** — Presidio and GLiNER findings now flow through the
  swarm fusion pipeline instead of being discarded before they can vote. Presidio
  labels are normalized into the pool vocabulary (previously its findings never
  type-voted — e.g. Presidio-solo on TAB reached recall 0.530 with *zero* findings
  surviving fusion), and GLiNER gains `organization`/`location`/`occupation` labels
  mapped to ORGANIZATION/LOCATION/JOB_TITLE (~1,900 ORG gold spans previously had no ML
  channel). Label remaps are whitelist-audited so every mapping is a masking *gain*;
  MEDICAL_LICENSE is deliberately left unmapped so a supported, maskable label does not
  become unsupported. Home dev (default config): swarm strict-F2 **0.8928 → 0.8952
  (+350 TP / +266 FP)**, vanilla 0.8916 → 0.8927. Across the five non-home externals
  (zero-shot, default, relaxed F2): ai4privacy 0.237 → 0.267, gretel 0.389 → 0.439,
  nemotron 0.324 → 0.335, piibench 0.189 → 0.196, TAB 0.101 → 0.138.
- **Single-engine acceptance** (`SwarmConfig.single_engine_min_confidence`) — a
  per-type confidence bar lets a high-confidence NER finding be emitted without
  cross-engine corroboration, via an additive Layer-4 branch that reads each engine's
  *raw* (pre-temperature-scaling) confidence. The branch is additive-only — ordinary
  corroborated findings still emit byte-identically — so the recall floor
  (AX-003 / FR-016) holds by construction. Invalid bars reject fail-closed and
  acceptance-emitted findings carry the engine-own confidence plus an explanation
  marker. (Extended per-(engine, type) in sp7, above.)
- **Document-anonymization profile** (`SwarmConfig.anonymization_profile()`) — a
  quasi-identifier singleton map (LOCATION / DATE_TIME / NATIONALITY / JOB_TITLE) tuned
  for document-anonymization workloads. On TAB this profile reaches relaxed F2 0.491 /
  strict 0.432 — roughly **4.9×** the pre-sp6 zero-shot baseline and ~90% of the
  engine-union upper-bound counterfactual. Reported as a separate, explicitly labelled
  tuned row; external-validity remains a disclosed gate, not a general-performance
  claim.

### Fixed (sp6)

- **NER span hygiene at the adapter boundary** — a shared `passes_ner_span_hygiene`
  check now runs at both NER adapters, vetoing field-label-position matches and barring
  single-token PERSON spans, so Title-Case junk dies engine-side and junk pairs cannot
  corroborate one another. Without it, Presidio normalization alone added +2,713 home
  PERSON false positives.
- **GLiNER long-window boundary alignment** — window-start word-alignment plus
  outward-only boundary-snap fixes the mid-word spans (e.g. "Col⟨leen Redding⟩")
  introduced by the sp4 long-document windowing; snapping only outward keeps the change
  over-masking-safe.
- **Eval-time GPS date-fragment false positives** — undecimaled coordinate
  look-alikes ("15/09") are now dropped under eval-only cross-type arbitration
  (`eval_cross_type_arbitration`). The production masking path keeps the permissive GPS
  pattern unchanged: narrowing it was proven to leak floor-sourced coordinate pairs
  ("41, -87") to production unmasked, so this precision gain is eval-only and never runs
  on the masking path.
- **Fail-closed config validation** — invalid single-engine confidence bars
  (NaN / negative / boolean, and the 400-digit-integer overflow class) now reject
  instead of silently disabling the gate into accept-everything, and wrong-typed
  acceptance maps fail loudly at load rather than crashing on first use.

### Added (sp5 — rate-elo rating-stack hardening)

- **Claim-grade Bayesian rating (`bayes-bt`) is now runnable.** Installing the
  `bayes-eval` extras (numpyro / jax / arviz, CPU) activates the previously-deferred
  claim-grade path for the first time: a Davidson tie-aware joint posterior over the
  13-player per-entity-F2 design. The first real-NUTS run passes the NFR-001
  convergence gate on real chains — split-R̂ 1.0011 (≤ 1.01), bulk-ESS 2734 (≥ 400),
  0 divergences.
- **13-player merged assessment + Glicko tournament.** pii_anon and swarm rate
  **1866.40 / 1865.57 Elo**, both statistically distinguishable from all 11 external
  detectors (next-best AWS 1542); leaderboard and tournament export included. At
  assessment/rating scope the first-party family holds 1.000 of the claim-grade rank-1
  mass (swarm 0.658 / core 0.342), while core-vs-swarm is **not** statistically
  significant (θ CI [−0.65, +0.43]) — a within-family coin-flip, not an external
  threat. This is assessment-scope evidence only; it does not move the SDO binding J,
  which stays NOT_YET (J = 0.2775).

### Fixed (sp5)

- **F1 (CATASTROPHIC): the NFR-001 convergence gate was NaN/inf-blind.** An all-NaN
  posterior passed `claim_grade=True` (both `NaN > RHAT_MAX` and `NaN < ESS_MIN`
  evaluate False) and fabricated a downstream J = 1.0. Non-finite draws and NaN
  diagnostics (finite draws ≥ ~1e153 overflow variance to NaN) are now first-class
  binding constraints; subnormal unmixed constant chains read R̂ = inf and negative
  divergence counts are refused.
- **F2 (MAJOR, J-consumer): exact-tie rank-1 mass was awarded entirely to the
  alphabetically-first player** — a relabeling-variant bias toward `pii-anon` in the
  binding-J race. Tie mass now splits fractionally, and the shared validator refuses
  non-finite θ draws. The mandatory close caught that the tie-split fix alone would
  have re-introduced a NaN-vacates-the-J-bar fabrication, so both are landed together.
- **F3 (MAJOR): the claim-grade NUTS model funneled on tied / near-tied designs**
  (80–158 divergences at default config — the claim-grade tier was structurally
  unavailable on the core-vs-swarm shape). A non-centered reparameterization
  (θ = σ·z) with target_accept 0.99 now fits these designs claim-grade.
- **F4/F5 (MAJOR): assessment ingestion now enforces the shared-gold invariant.**
  Contradictory cross-player gold counts, phantom single-player types, empty scored
  players, counts-impossible per-type F2, n_gold total mismatches, and
  non-integer / blank gold keys now fail loud instead of silently skewing ratings; the
  checks re-run at tournament entry.

### Changed (sp5)

- **Editable-install metadata drift fixed** — the `pii_anon_datasets` dist-info
  reported 1.3.0 while the module was 2.2.0, the cause of the CLI `--merge` refusal and
  a stale version stamp (the sp3 report's stale stamp is corrected as ours, not the
  dataset's).
- **Rating-stack changes are eval-framework / assessment only.**
  `competitive_supremacy.py` and `canonical_run.py` were untouched (md5 `3b842e81`
  verified across all three close rounds), and the honest SDO anchors are byte-identical
  throughout: NOT_YET / J = 0.2775 / all G PASS. The five fixes were certified by a
  3-round adversarial close (77 + 116 + 44 probes) including a 571-artifact
  zero-regression sweep.

### Added (sp4 — external validity + GLiNER long-doc windowing)

- **External-validity disclosure report** (`docs/external-validity-report.md`) — the
  standing zero-shot evaluation of `pii_anon` and `pii_anon_swarm` against five
  non-home public PII benchmarks (ai4privacy-400k, Nemotron-PII, Gretel finance,
  TAB/ECHR real court documents, PIIBench). It gates any best-in-class broadcast:
  measured relaxed F2 ranges **0.10 (TAB, real court documents) to 0.39 (Gretel
  finance)** zero-shot versus **0.89 at home** — a large, honestly disclosed transfer
  gap (a measured bound on out-of-domain performance, not a claimed impossibility). On
  PIIBench, the one external set with a published multi-system baseline table, our
  zero-shot **strict F1 0.13 lands inside the published 8-baseline family** (all
  < 0.14), with far higher per-type scores where an honest type counterpart exists
  (IBAN 0.76, IP 0.48, EMAIL 0.46). The report also documents a flagged follow-up:
  even after the windowing fix below, the swarm fusion structurally suppresses NER-only
  findings out-of-domain (0/5 GLiNER findings survive on a real judgment; the dormant
  meta-learner caps single-engine spans at ~0.62, below the 0.85 emission bar), so the
  swarm's generalization channel is discarded externally and `swarm ≈ vanilla` off-home
  — the follow-up the sp6/sp7 channel work addresses.

### Fixed (sp4)

- **GLiNER long-document detection collapse** — the GLiNER adapter previously fed
  whole documents to the model unwindowed, so NER detection degraded with input length
  (**3 findings at 500 chars to 0 at ≥ 2,000 chars** on a real court judgment) and long
  documents lost their entire NER contribution. The adapter now windows long inputs
  (**400-char whitespace-aligned windows + 100-char overlap**, size swept on the TAB
  dev split: **gold-PERSON overlap 18/56 to 50/56** vs a 1,200-char window), re-bases
  span offsets, and dedupes overlap echoes. This is a genuine production detection fix
  that applies to any long input — not benchmark tuning; the external label-mappings and
  scoring live only in the eval harness and never touch the production masking path.
  Home-dev regression check confirmed no regression: swarm F2 0.8924 → 0.8928 with
  precision flat. Covered by `tests/test_gliner_windowing_sp4.py`.

---

## [1.6.0rc1] — 2026-07-10 — sp2 + sp3: external-assessment supremacy, Art-9 coverage, v2.2.0 re-baseline

> Release candidate, LOCAL-ONLY tag. Publication remains gated on the external-validity
> program (FR-027): performance against non-home datasets must be measured and disclosed
> before any "best-in-class" broadcast. Home-benchmark result (pii-anon-eval-data v2.2.0,
> 66-type strict-v1 test split, 31,048 records): pii_anon_swarm F2 0.893 / pii_anon F2
> 0.892 — #1/#2 of 13 detectors at full 66/66 coverage (best external: aws 0.736 at
> 24/66). Synthetic-data, home-team-tuned; NOT an external-validity claim.

### Added (sp3 — v2.2.0 re-baseline, 2026-07-10)
- **GDPR Article-9 special-category detection (FR-040)** — `SEXUAL_ORIENTATION`
  (label-gated closed lexicon), `TRADE_UNION_MEMBERSHIP` (label-gated
  proper-noun value), `GENETIC_DATA` (label-gated value + intrinsic
  gene-symbol / dbSNP rs-ID structure, Greek-block-aware). pii-anon is the
  first non-LLM detector on the eval-data harness to reach the Art-9
  categories (previously 0/3 for every off-the-shelf detector and pii-anon
  itself). Never anchors on generator filler (eval-integrity axiom).
- **Value-class recall recovery** — CVV / PIN / PASSWORD /
  INSURANCE_POLICY_NUMBER / AUTHENTICATION_TOKEN now reach the v2.2.0
  corpus's obfuscated secret forms: base64 values, zero-width-embedded
  keywords, OCR variants (`8earer`, `P0L`), quoted code/config/JSON
  passwords, truncated-JWT placeholders. AUTHENTICATION_TOKEN went from a
  complete miss (recall 0.00) to 1.00 on the dev split; CVV/PIN 0.40→1.00;
  INSURANCE 0.52→1.00 with precision UP.
- **Census re-derivation 63→66** — `DATA_V2_CORPUS_ENTITY_TYPES` + version
  pin re-derived for `pii_anon_datasets` 2.2.0; the standing
  pattern-label-alignment gate keeps every registry label honest.

### Added (sp2 — external assessment)
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
  shadow generic PERSON/DATE matches, **eval-only** — production over-masks).
- **Adversarial-review remediation** (12 confirmed findings, multi-agent
  close): a SHOWSTOPPER production PII-leak channel closed (cross-type
  arbitration made eval-only so the masking path never drops a maskable
  PERSON_NAME for a non-masked type); the `"Record shows"` generator-filler
  anchor removed as benchmark gaming; over-capturing HEALTH_CONDITION /
  MEDICATION / EDUCATION / ORG-CamelCase patterns tightened; ISO-8601
  timezone-extent + rate-elo CLI DoV hardening; merge cross-split guard.

### Results — external assessment (`pii-anon-eval-data`, en test split, strict-span)
- **pii_anon_swarm F2 0.885 (#1)** and **pii_anon F2 0.884 (#2)** sweep the top
  of the 12-detector leaderboard — best overall (cloud + OSS), ~0.15 F2 above
  the strongest incumbent (aws 0.737, gliner 0.735), at **63/63** entity-type
  coverage vs the field-best 24/63. The 12-player `rate-elo-assessment` rates
  both first-party systems statistically distinguishable from every competitor.
  Internal census guard n=10000: F2 0.884, p50 0.32 ms (speed ceiling green).

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
