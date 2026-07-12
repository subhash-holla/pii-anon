# Enhancement Amendment: sp7-world-leading

> Managed by /dev-assist-enhance. Opened 2026-07-11. Status: **IN_PROGRESS** (Phase 0 scoping).
> User steer: "build both vanilla + swarm into the world-leading options for PII detection and
> anonymization — maintain home performance while performing to a similar level on industry-leading
> datasets (retraining the swarm is authorized; run an upper limit on vanilla). AND build out
> streaming / LLM-fed operation with SCIENTIFIC, reproducible, shareable evidence that LLMs cannot
> reconstruct PII after it passes through the library."

## ★ Honest scoping of the "LLMs can't reconstruct PII" guarantee (READ FIRST)

The universal claim "no LLM can reconstruct any PII" is **unprovable** — it is a negative over an
unbounded adversary class, and any report asserting it would be scientifically dishonest. What sp7
CAN deliver, and what the report will claim, is a **measured resistance bound against SPECIFIED,
disclosed adversary classes**, reproducibly:

- **Tier-3 LLM re-identification** (RRS/QIC/BSL) via the existing `eval_framework/attacks/reid_tier3.py`
  — success rate with Wilson 95% CIs at declared power (NFR-012: ≥385 paired personas/cell).
- **Membership-inference** (LiRA-shaped + Secret-Sharer) via `attacks/mia.py` — TPR@FPR∈{1e-3,1e-2}
  at declared shadow-model power (NFR-013).
- **Agentic / streaming leakage** via `agentic/interception.py` + `leakage_sankey.py` — per-channel
  verbatim-survival counts (a leak = caller-supplied ground-truth PII surviving VERBATIM post-mask).

The report's headline is a BOUND ("under adversary class X at power P, re-identification success ≤ R
with 95% CI [lo, hi]"), never "impossible". Every number carries its seed, its power, and a
NON-STRIPPABLE caveat (the NFR-016 anti-anonymity caveat pattern). Agent-simulated adversaries are
labelled AGENT_SIMULATED; a real-LLM-API adversary run is a budget-gated, separately-labelled row.

## Classification (closed 6-class, multi-class)

| Phase | Class | Anchor |
|---|---|---|
| A — vanilla upper-limit | `defect-fix` + `new-capability` | mining candidates 1/3/5-11 (sp6 `_evidence/`); FR-036-family detection |
| B — swarm meta-learner retrain | `new-capability` | `swarm_learner.py` train + `xgboost_model.ubj`; the sp6-proven live channel |
| C — streaming + LLM-boundary guarantees | `new-capability` | FR-011/013/025/026/028/029/036; the attacks + agentic + assurance instruments |

## Phase plan (sequential; measure → decide → build → close per phase)

- **Phase 0 (this step):** per-dataset ACHIEVABLE CEILINGS for both detectors (oracle-perfect +
  engine-union + gold-noise floor from the sp6 dropped-list), so "similar level" gets a NUMBER per
  dataset, not an aspiration. + an audit of the streaming (FR-036 parity) and LLM-boundary
  (attacks/agentic/assurance) surfaces → the concrete Phase-C build plan.
- **Phase A:** vanilla tranche (regex Title-Case suppression, labeled-field bridge, date/name/ORG/
  address grammars), external-TRAIN-tuned, home-gate + external-remeasure per batch.
- **Phase B:** swarm retrain — candidate-labeling API, disjoint external+home TRAIN, signed
  `xgboost_model.ubj` (S2-05 verify-on-load), floor-invariant + mandatory close.
- **Phase C:** the streaming/LLM-reconstruction evidence report — wire the Tier-3/MIA/leakage
  instruments into a reproducible, shareable report (JSON + HTML + one-page), with the honest bounds
  above; FR-036 stream/batch parity assertion.

## Phase 0 RESULTS (measured, `_evidence/phase0-scoping.json`)

**★ "Similar level to home 0.89" is honestly UNATTAINABLE zero-shot on foreign datasets** — even a
PERFECT detector is capped by taxonomy mismatch (oracle-perfect F2 0.65–0.96) and gold noise. The
right definition of "similar level" is **within a stated margin of THAT dataset's no-new-model
perfect-fusion ceiling** — and the measured bottleneck is FUSION discarding complementary channels,
not detection quality (the engines already find the recall: union reach R up to 0.60).

Honest targets (relaxed F2, swarm; each ≥76% of its binding no-new-model ceiling):

| Dataset | vanilla now | swarm now | **target** | binding ceiling | do-not-chase (taxonomy/convention) |
|---|---|---|---|---|---|
| ai4privacy-400k | 0.213 | 0.267 | **0.40** | 0.523 | oracle 0.871 (SURNAME/BUILDINGNUM split 15.6%) |
| Nemotron-PII | 0.324 | 0.335 | **0.48** | 0.507 | oracle 0.789 (first/last split, generic-NER 25%) |
| Gretel finance | 0.379 | 0.439 | **0.50** | ~0.50 | oracle 0.959 (partial-annotation-by-design) |
| TAB (real docs) | 0.100 | 0.138 | **0.55** | 0.594 | oracle 0.949 (NO_MASK/QUASI, document-level) |
| PIIBench | 0.184 | 0.196 | **0.30** | 0.270 | oracle 0.648 (generic-NER-as-PII gold noise) |

## Sequenced execution (each masking-path change → MANDATORY adversarial close)

- **STEP 1 — FR-036 parity FOUNDATION (FIRST):** ★ the streaming audit found **GAP-1, a real silent
  PII leak** — a multi-token entity straddling a segment boundary is dropped (batch detects
  `415 555 1234`; segmented at overlap=0 → EMPTY). Fix boundary-safety + context-window halo + a
  golden stream==batch==offline parity test + determinism pin. Precondition for ALL downstream
  measurement (every number must sit on a parity-identical masking path).
- **STEP 2 — Phase A vanilla (A1→A6):** A1 Title-Case FP suppression (default-to-mask ONLY — the sp2
  showstopper) → A2 labeled-field bridge → A3 date grammar → A4 phone/postcode/address → A5 TAB
  docket+DEM → A6 scrubadub label plumbing (eval-side). Home-gate + external-remeasure per item.
- **STEP 3 — Phase B fusion RECALIBRATION (config-first, NO retrain yet):** per-(engine,type)
  acceptance + emission recalibration + post-fusion precision/boundary gates (sp6 cand-2, extended).
  A learned meta-learner retrain is HELD IN RESERVE behind measured residual gap — the home-substrate
  retrain is floor-locked (memory), and the foreign-taxonomy channel-discard is config-addressable, so
  retraining is funded only against a proven residual, not speculatively.
- **STEP 4 — Phase C evidence report:** `reconstruction_resistance_report(corpus, masking, *, seed,
  surrogate_key)` — the ONE load-bearing gap is the corpus→attack-substrate adapter
  (`# SWITCH-POINT(DATA)` `assemble_paired_set`, verified absent); every instrument (Tier-3 re-id +
  MIA + leakage Sankey + assurance) already ships and was exercised end-to-end. Emits measured BOUNDS
  (Wilson CI, TPR@low-FPR, per-channel leakage) + seeds + non-strippable caveats + a real-LLM row.
  Gated by the STEP-1 parity assertion.

## Delta table

| # | Delta | Status |
|---|---|---|
| P0 | Phase-0 scoping (`wf_1ed2ad19`, 8 agents): measured ceilings + surface audits → honest targets + sequenced plan (`_evidence/phase0-scoping.json`) | DONE |
| S1 | **FR-036 parity foundation — GAP-1 silent-leak FIXED.** Reproduced (batch detects `415 555 1234`; segmented at overlap 0 → EMPTY across all 3 segments). Fix: `Segmenter.MIN_SAFE_OVERLAP_TOKENS=24` floor (any entity ≤24 tokens wholly contained in ≥1 window regardless of caller overlap; raising overlap is leak-SAFE — over-detect + dedupe, never drop) + step≥1 guarantee for degenerate configs. Golden parity tests (`test_segmentation_parity_sp7.py`, 33 cases: no-boundary-leak + segmented-⊇-batch + no-zero-step). Default path byte-identical (short text = single segment). 92 tests green, lint/mypy clean | DONE (TDD) — CLOSE PENDING |
| — | STEP-1 mandatory adversarial close (masking-path change: segmenter feeds detection→masking) | TODO |
| A1 | **Title-Case-noise PERSON/ORG FP suppression — EVAL-ONLY (leak-safe by construction).** `_drop_titlecase_noise_person` under `eval_cross_type_arbitration` (markdown-wrapped + determiner-led + all-header-words, with a given-name override). sp2 discipline: production masking (arbitration OFF) STILL over-masks headings — the test proves eval-emission ⊆ production-emission. **Home vanilla F2 0.8916→0.8932 (+0.0016, precision +0.0054, recall +0.0006 — IMPROVES home, zero recall loss)**; externals nemotron 0.324→0.335, gretel 0.379→0.399, ai4privacy 0.213→0.217. SDO byte-identical. Conservative curated-word version; a wordfreq-based widening (higher home-recall risk) is a follow-up | DONE (TDD) |
| A2 | **Labeled-field value bridge (sp6 candidate #3) — new capability, additive + leak-safe.** `engines/regex/labeled_fields.py`: a shared cue→value bridge (`SSN:`, `Date of Birth:`, `Account Number:`…) that tolerates markdown/quote wrappers, extracts the VALUE span (cue-capture hygiene — never the label word), and types it FROM THE LABEL. Every cue maps ONLY to `SUPPORTED_ENTITY_TYPES`; runs on BOTH paths (over-masking a labeled value is safe). `_apply_label_wins` = (1) defer to same-type pattern coverage (keeps validators/checksums authoritative — fixed the invalid-ABA / VIN-length / SPDX-licence over-fires) + (2) leak-safe different-type relabel (span stays covered). **Home vanilla strict F2 +0.0016 (P +0.0015, R +0.0016 — improves BOTH, no regression); externals nemotron +0.0227 (0.335→0.358), gretel +0.0113, ai4privacy +0.0083, precision & recall moving together.** Mandatory close CLOSE_PASS (112 hostile prod+eval probes, 0 coverage-shrink leaks, 0 unsupported-type emissions). Committed SDO artifact byte-identical; the canonical/SDO detection path DOES change, so the next canonical regeneration must re-certify. v1 defers the FP-prone copula bridge (`is\|was\|of`) + runaway ADDRESS capture | DONE (TDD + close) |
| A3 | **Natural-language / locale date grammar + DOB-cue promotion (sp6 candidate #5) — ★ the biggest lever.** `patterns.py` `_DATE_PROSE` (day-first "27 May 1994", legal "1st day of January, 2023", Month-year "May 1994", ordinal "3rd of April 1980") + `_DATETIME_SPACE` (space datetime), emitting DATE_TIME (benchmark-ignored on home scoring, masked in prod, scored on externals). `_promote_dob_by_cue` re-types DATE_TIME/DATE_ISO→DATE_OF_BIRTH when a word-bounded birth cue is within 30 chars (leak-safe relabel, all paths, cue-gated). Boundary hygiene: span starts at the date token. **Home vanilla strict F2 +0.0037 (precision +0.0167 — DOB promotion recovers home DOB gold; recall flat, no regression). TAB relaxed F2 0.101→0.397 (+0.295, ~4×; R 0.091→0.378) — one lever, 72% of TAB's 0.55 target; nemotron +0.0116, gretel +0.0073, ai4privacy +0.0088.** Close CLOSE_PASS (60 coverage probes 0 leaks, 0 spurious promotions, 0 ReDoS). SDO committed-artifact byte-identical | DONE (TDD + close) |
| #8 | **Organization/institution capability (sp6 candidate #8) — additive, leak-safe.** `patterns.py` 4 new ORGANIZATION patterns: `_ORGANIZATION_INSTITUTION` (tail keyword Ministry/Tribunal/Government/Board/… + diacritic-aware ATOM run + leading-stopword veto), `_ORGANIZATION_COURT` (descriptor-gated: "Sinop Assize Court" ✓ but "Birch Court" address ✗), `_ORGANIZATION_INSTITUTION_OF` ("Ministry of Justice"; Department/Office excluded), `_ORGANIZATION_FIRM` ("Harper & Associates"). ORGANIZATION is supported + person-shadowing, so the existing eval-only person-drop cleans the TAB institution-as-person FP face for free. **The close's home-gate FAIL (−0.0008) diagnosed to ONE cause — "Birch Court" ×161 (a residential street suffix, not an institution) — fixed by the court-descriptor gate → home F2 delta +0.0000 (neutral). TAB 0.397→0.435 (+0.038; P 0.492→0.518 R 0.378→0.418), nemotron +0.0024, gretel +0.0019.** Close CLOSE_PASS (56 coverage probes 0 leaks, 0 ReDoS). SDO byte-identical | DONE (TDD + close) |
| #7 | **Numeric-identifier guards (sp6 candidate #7) — scoring-only suppressors + additive hemisphere GPS.** `_drop_nongeo_gps` (drops money/rating-shaped GPS, `$1,125.00`/`4.5/5`) + `_drop_bare9_ssn_noncontext` (drops sequential/delimiter-glued bare-9 SSN with no SSN cue) in the arbitration seam (SCORING-ONLY — masking path untouched, sp6 lesson); `_GPS_HEMISPHERE` pattern ("40.7234 N, 123.1235 W", ADDITIVE so GPS coverage never shrinks). **The gate CAUGHT a home recall FAIL (−0.0053) and diagnosed it to the IBAN mod-97 drop removing 225 home gold IBANs (home synthetic IBANs are checksum-invalid) — the mining's "home-floor sign-off" flag was correct, so the IBAN guard was DROPPED, not shipped.** Final: home strict F2 +0.0000 (neutral); externals nemotron +0.0012, gretel +0.0019, ai4privacy +0.0009, tab +0.0000 — modest, as the spec honestly predicted (the sp6 GPS drop already removed the bulk). Close CLOSE_PASS (PROD masking-path coverage invariant + retention, 0 leaks) | DONE (TDD + close) |
| A4 | **Address grammar rework (sp6 candidate #10) — masking-path rewrite, two-tier evidence-gated.** `patterns.py` `_ADDRESS` rebuilt: global `re.IGNORECASE` DROPPED (it slurped lowercase prose), `(?i:)` on the suffix only; first-token function-word guard; full-USPS unambiguous suffix set (tier-1, ADDITIVE — the gretel FN mass); `_ADDRESS_AMBIGUOUS` common-noun suffixes gated on unit/postcode LOOKAHEAD (tier-2, no span extension); `\s+`→`[ \t]+` (an address never crosses a line break). **Three gate-caught fixes: (1) 117 "Pl" FPs from `9902\nLicense pl` → the newline fix; (2) span over-extension from independent unit/zip captures → all-or-nothing city/state tail (home strict-span match); (3) 312 lost home "… Way, City, ST" addresses → "Way" mis-tiered to tier-2, moved back to tier-1.** Final: **home strict F2 0.9035→0.9035 (neutral, P −0.0001 noise); gretel 0.422→0.456 (+0.034; R 0.532→0.582); TAB +0.0044 (P 0.518→0.551 via the func-guard); nemotron +0.0014; ai4privacy −0.0014.** Close CLOSE_PASS (0 lost real-address coverage on the preserve set, 0 ReDoS). SDO byte-identical | DONE (TDD + close) |
| #6 | **Name grammar & span hygiene, SAFE subset (sp6 candidate #6) — additive + eval-only.** `patterns.py` P1 `_PERSON_TITLE_FULL_U` (honorific + Unicode full name → "Osman Çağlayan"), P2 `_PERSON_TITLE_INITIALS` ("Mr S. Esmer" → "S. Esmer"), P3 untitled First+mid-initial+Surname (section-word guarded), P3C ALL-CAPS variant (mid-initial anchor REQUIRED — cannot re-open the A1 flood), P4 in-place diacritic widening of `_PERSON_FULL_NAME` (guards left ASCII). Eval-only: `_trim_salutation_led_person` ("Ciao Nalda"→"Nalda") + multilingual DE/FR/IT/ES article extension of `_NAME_LEADING_STOPWORDS`. Names captured SANS honorific = home convention (no masking-convention change). **EXCLUDED per user flags: mention-propagation (eval-data-owner sign-off) + bare ALL-CAPS two-token names.** **Home strict F2 0.9035→0.9038 (+0.0003, precision +0.0014 — IMPROVES home); TAB 0.444→0.489 (+0.044; R 0.425→0.472); gretel +0.010; ai4privacy +0.0009.** Close CLOSE_PASS (additive coverage invariant, 0 ReDoS). SDO byte-identical | DONE (TDD + close) |
| — | **★ Phase A (vanilla detection) COMPLETE** — cumulative: home 0.8916→0.9038, TAB 0.10→0.489 (~5×), gretel 0.42→0.474, nemotron 0.335→0.374, ai4privacy 0.213→0.234. Deferred (need user sign-off): #6 mention-propagation · #9 geo taxonomy-split · #6 TAB honorific with-title variant | — |
| #9 | **Geo gazetteer, LOCATION-only home-safe subset (sp6 candidate #9).** `engines/regex/geo_lexicon.py` — curated PUBLIC-DOMAIN facts (countries + US states, no external gazetteer dependency / licensing). `extract_locations` emits LOCATION additively (both paths); ambiguous single tokens (Georgia/Jordan/Reading) fire ONLY with a location cue ("in/at/from/near <Place>") via a COLLIDE veto (the sp3 label-gating lesson); `_drop_location_nested_in_address` (leak-safe — ADDRESS/GPS still masks). **Home strict F2 0.9030→0.9030 (perfectly neutral); TAB +0.0071 (R 0.472→0.485); ai4privacy −0.0006, nemotron −0.0004 (noise).** Marginal, as honestly predicted — the high-value nemotron 712 state/county/country spans are UNREACHABLE without the DEFERRED taxonomy split (needs user sign-off to expand SUPPORTED_ENTITY_TYPES + the label census). Close CLOSE_PASS. SDO byte-identical | DONE (TDD + close) |
| — | **★★ Phase A (all 8 vanilla levers A2·A3·#8·#7·A4·#6·#9 + A1) COMPLETE** | — |
| C | **★ Phase C — LLM-reconstruction-resistance report (the headline deliverable).** `assurance/reconstruction_resistance.py` (pure) + `reconstruction_resistance_cli.py` (driver). Emits a reproducible, shareable JSON + HTML report claiming a MEASURED BOUND vs a disclosed adversary class (never "impossible"). Three axes, each with Wilson 95% CI + non-strippable NFR-016 caveat: (1) **verbatim leakage** — fraction of gold PII surviving verbatim in the mask (direct reconstruction bound); (2) **Tier-3 re-id** (RRS/QIC/BSL via `RepresentativeTier3ReidAttack`) masked vs unmasked baseline (the protection delta); (3) **MIA** (LiRA-shape via `RepresentativeMiaAttack`) TPR@low-FPR. The load-bearing `assemble_paired_set` corpus→attack-substrate adapter (reidx-01: signals RE-EXTRACTED from masked text, never gold). **FR-036 stream/batch parity assertion** (streamed redaction ⊇ batch — over-mask safe). Adversaries labelled AGENT_SIMULATED; real-LLM/LiRA@128 are SWITCH-POINT(DATA). **First real run (home n=1500, regex masker): verbatim leak 8.35% [Wilson], masked re-id recall 0.73% vs unmasked baseline, 0 parity violations.** 6 tests, lint/mypy clean, SDO byte-identical | DONE (TDD) |
| PANEL | **★ Expert-panel library audit (`wf_28487925`, 54 agents, 30 verified survivors) + top-pick execution.** 10 expert lenses → adversarial verify → ranked roadmap (`scratchpad/panel_roadmap.json`). Executed the 3 leak-safe, measured, no-sign-off-needed picks: **#1 CUSTOMER_ID labeled-field type** (`1e7c051` + eval-data `35c03e1`) — genuine NEW masking coverage of a 0%-recall quasi-identifier; nemotron relaxed F2 +0.0064, home neutral, close CLOSE_PASS. **#3 per-type verbatim-leakage breakdown** (`c038c1e`) — localises the Phase-C leak: PERSON_NAME 22.5% = 91% of leaks, structured PII (EMAIL/PHONE/ADDRESS/SSN) all 0.00%. **#2 value-consistent (coreference) masking** (`8bc3b4d`) — redacts every verbatim occurrence of a detected sticky-type value; overall verbatim leak **8.35%→5.93% (−29% rel), PERSON_NAME 22.5%→15.4%**, detection byte-identical. **Panel corrections baked in** (the verify pass fixed over-claims): geo STATE/COUNTRY split is EVAL-ONLY (library split drops home LOCATION) = benchmark-alignment NOT genuine masking gain; the "ORGANIZATION channel inert" premise was FALSE; the name-split lever craters precision (+7767 FP). **Needs user sign-off (flagged, not executed):** #4 GLiNER acceptance profile (+4pp nemotron, medium home-risk, SDO re-cert) · #9 per-(engine,type) map (Phase B) · geo eval-only relabel (external repo) | 3/roadmap DONE · sign-off items flagged |
| B | **Phase B — swarm fusion recalibration: foundation VERIFIED, per-(engine,type) refinement BLOCKED on TRAIN data.** Measured (swarm vs vanilla, cap 600): **floor invariant HOLDS — swarm ≥ vanilla on both (TAB 0.4957→0.4970, gretel 0.4822→0.5015)**, so `FloorProjectingFusion` propagates ALL Phase A gains to the swarm by construction, and the sp6 `single_engine_min_confidence` per-type acceptance map adds modest corroboration (TAB +0.0013, gretel +0.0193). **Swarm now MEETS/NEARS its targets: gretel 0.502 ≥ 0.50 target; TAB 0.497 near 0.55.** The mining candidate #2 per-(ENGINE,type) refinement (admit more single-engine gliner above TRAIN-derived alone-precision bars) is DEFERRED: (a) it needs nemotron TRAIN alone-precision but "nemotron TRAIN must be downloaded — only the test parquet is local" (a download-authorization / user-data boundary); (b) it carries the mining's explicit precision-crater risk, demanding a full per-(engine,type) swarm measurement campaign + mandatory close — a focused follow-up, not a rushed tail-of-session change. NO unverified masking-path swarm change shipped (discipline held) | FOUNDATION DONE · refinement BLOCKED/DEFERRED |

## User decisions (2026-07-11, locked)

1. **Success bar = approach each dataset's measured ceiling** (targets table above), NOT the
   home-tuned 0.89. Report honestly, no asterisks.
2. **Retraining held in reserve** — fusion RECALIBRATION first (config, no learned weights); fund a
   learned meta-learner retrain only against a proven per-dataset residual gap.
3. **Sequence = detection first, then the guarantee** — Phase A vanilla → Phase B recalibration →
   Phase C evidence report (so the guarantee measures the improved masking).

## Invariants

Leak-direction; AX-003/FR-016 floor BY CONSTRUCTION; SDO gate + canonical producer untouched without
the mandatory close; test splits NEVER tuned/mined; every tuned/retrained number labelled with its
training provenance; the guarantee report claims measured BOUNDS, never impossibility; adapter label
changes audited vs SUPPORTED_ENTITY_TYPES on ALL fusion modes (the sp6 inversion-class lesson).
