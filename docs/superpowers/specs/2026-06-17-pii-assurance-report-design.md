# PII Handling Assurance Report — Design Spec

| | |
|---|---|
| **Status** | **PHASES 1 & 2 IMPLEMENTED & CERTIFIED** (2026-06-18) — all four dimension areas (detection · leakage · re-identification · utility/fairness/compliance) + all four outputs (JSON+bundle · Markdown · HTML executive · one-page summary). Both adversarial closes passed 0-upheld. |
| **Date** | 2026-06-17 (design) · 2026-06-18 (Phase 1 close PASS) |
| **Author** | Subhash Holla (with Claude Code) |
| **Increment** | New feature on the closed v1.5.0rc1 program (focused dev-assist increment) |
| **Grounded by** | Workflow `wf_61a2957c-b8a` — 7/8 reuse areas verified against live source + 3 adversarial reviewers (21 SHOWSTOPPER/MAJOR findings incorporated below) |

## Phase 1 implementation status (2026-06-18)

Package `src/pii_anon/assurance/` (+ public `src/pii_anon/eval_framework/validation/no_fabrication.py`). 222 dedicated tests, ruff + mypy(strict) clean. **`HARNESS_CLOSE_CERTIFIED=True`** — flipped only after a **0-upheld adversarial close on the certified (`MEASURED`) path** (18 rounds; 11 attacking the live `MEASURED` path; final confirmatory round 0-upheld). The close upheld and we fixed, in total, **~30 blocking findings** (incl. multiple SHOWSTOPPER-class forges: phantom-0 wins, silver self-grading, bool/NaN/range/surrogate fabrication, BaseException PII leak, period-toggle & cross-process non-determinism, invisible-char leakage laundering, type-vocabulary phantom-loss). The external verification mode the design promised is **implemented** (`python -m pii_anon.assurance verify`). Residual non-blocking OBSERVATIONS (documented, non-reachable / by-design): user-supplied label scrubbing is best-effort (§10.2); `char_kappa`/zero-width-CI defense-in-depth parity deferred to Phase 2.

**PII-egress gate hardening (2026-06-18, close rounds 19–22, all blocking findings fixed then 0-upheld).** A realistic sample report surfaced a `PiiEgressError` the 18-round close had not exercised, and the follow-on hardening caught three real PII-leak classes — each found by the close, not by audit:
- **Round 19 (CLOSE_PASS):** Layer-2 detector-surface containment false-positived on a bare 4-digit invoice number that coincidentally substring-matched inside a sha256 digest in `report.json`. Fixed with a **distinctiveness floor** (`_is_distinctive_surface`) + **token-boundary containment** (`_token_contained`), aligning Layer 2 with Layer 1's existing <5-char/4-digit exclusion. (The fix only narrows the false-positive class; every distinctive raw-PII class still fails the gate closed — confirmed by a forge test.)
- **Rounds 20–21 (both caught a real MAJOR):** a quality fix added a dotted-numeric **version-label exemption** so a legitimate pipeline version (`2.3.1`) renders readable instead of `[redacted]`. The close caught a **year-first dotted date** (`1990.05.21`, a birthdate / date-named dataset) leaking verbatim via labels (collides with CalVer; round 20), then a **decorated 2-digit-year date** (`v90.05.21`, `90.05.21rc1`) leaking because the `v`/`rc` decoration blinded the detector (round 21).
- **Round 22 (CLOSE_PASS):** the structural fix — the version exemption **defers to the PII detector as the date oracle**: strip the `v`/pre-release decoration to the bare dotted core and re-consult `detector(bare)`; if it tags anything, the token is not a version. A brute-force over ~2.9M version-shaped tokens found no leak; real semver/CalVer cores stay readable; a 4-digit-leading form is redacted as date-ambiguous (intended).

Lessons recorded: for an ambiguous-shape safety classifier, **defer to the detector** rather than hand-enumerating shapes; a **label-only** value (absent from every record) is guarded **solely** by `scrub_label`, since `scan_output` only matches raw-corpus tokens. The byte-level egress gate (`scan_output`/`assert_safe`) was unchanged throughout — only the Layer-2 detector path (round 19) and the label scrubber (rounds 20–22) changed.

Produced-report verdicts are HONEST by construction: a number is `MEASURED` only with gold labels + a passing power gate (support + clusters + CI-halfwidth, tighten-only) + provenance gate + certified harness; silver mode never crowns a winner; real re-id/MIA attacks are `NOT_ASSESSABLE`.

## Phase 2 implementation status (2026-06-18)

The remaining dimensions + polished outputs are implemented and certified by the mandatory **second adversarial close (6 rounds; final round-6 0-upheld-blocking AND 0-upheld-minor)**. New code: `assessors/reidentification.py`, `assessors/utility_fairness.py` (the user-facing `utility_fairness_compliance` dimension is expanded by the runner into three honestly-separated axes — `utility` / `fairness` / `compliance` — each with its own claim strength), `render/html.py` (self-contained inline-SVG executive, no JS, `html.escape`-hardened), `render/summary.py` (one-page certificate), `stats.mean_bootstrap_ci`. 288 dedicated tests, ruff + mypy(strict, 174 files) clean, full library suite green, **zero existing source modified**.

The close caught **8 real honesty bugs that the test suite and manual audits all missed** (the close is the only reliable certifier — same lesson as Phase 1):
- **Re-identification — two DIRECTION INVERSIONS (the leak-direction class).** A word-token / Jaccard stand-in is fundamentally obfuscation-fragile: a **cross-record identifier permutation** (round 2) and **character-dotting** `A.l.p.h.o.n.s.e` (round 5) both left PII human-readable yet defeated token matching, so a strictly-weaker transform reported strictly-lower re-id risk. Fixed structurally (not whack-a-mole) by replacing it with an **obfuscation-robust compacted-substring linkability** — `_compact` (NFKC + casefold + strip every non-alphanumeric), then linkability = fraction of outputs in which some real distinctive (≥5-char) identifier (own or another record's) survives as a substring. **Monotone-safe by construction** (more readable PII surviving ⇒ ≥ risk, never an inversion); a fail-closed LOWER BOUND; ADVISORY-only, never MEASURED, never a head-to-head winner; the non-strippable `ANTI_ANONYMITY_CAVEAT` is surfaced verbatim. Documented residual (non-upheld): homoglyph / alnum-noise insertion garbles readability and evades — covered by the lower-bound framing + caveat.
- **Utility — phantom-perfect 1.0 (×2, the leak-direction class).** `SemanticPreservationMetric` returns 1.0 on a degenerate input, so a zero-anonymization / content-destroying transform scored a publishable MEASURED 1.0 that out-ranked a genuinely-preserving transform — via an **empty output** (round 1) and, separately, an **empty non-PII skeleton** for a wholly-PII record (round 3). Fixed by marking such records UNSCOREABLE (mirrors the certified leakage degeneracy guard); too few scoreable ⇒ `NOT_ASSESSABLE`. Lesson: a degeneracy guard must cover *every* degenerate input to the metric.
- **Fairness — phantom-0.** A detector that matches zero gold rendered a 0 recall-gap "perfectly fair" PASS; fixed to `NOT_ASSESSABLE` when no powered group has any recall (a 0 gap among all-zero recalls is "equally undetected", not "fair").
- **Numeric no-fabrication gate (`report.py`) — three escapes hardened fail-closed.** A forged **bool** under any new score key serialized as `true` (the bool rule was inverted to reject bool unless a known `_FLAG_KEYS` flag); an out-of-`[0,1]` **int**-typed rate bypassed the float-only range check (ints now range-checked symmetric with floats except an explicit `_COUNT_KEYS` allow-list); every **count** (`support`, `n`, agreement counts, …) now routes through `_count`/`_finite_int` so a negative / huge-int (the int→str digit-limit DoV) raises `NoFabricationError` instead of serializing or crashing `json.dumps`.
- **Renderer — the `ANTI_ANONYMITY_CAVEAT` was stripped from the one-page marketing certificate** (`summary.html`, the highest-stakes advertisable surface); fixed so every dimension's caveats are surfaced. Both HTML renderers are `html.escape`-hardened (verified against `<script>` / `onerror` / attribute-breaking payloads) and emit no JavaScript.

---

## 1. Goal

Let a user generate a **single, audit-grade, advertisable "PII Handling Assurance Report"** that runs **their own PII pipeline** *and* **pii-anon's own offering** on **their own dataset**, across the full PII lifecycle, and is methodologically defensible enough that the user can publicly advertise their ecosystem's handling of PII on it.

The report's load-bearing property is **honesty under adversarial reading**: every number is either provably defensible or visibly hedged, and the report *structurally cannot* overstate either the user's posture or pii-anon's. This mirrors the discipline of the existing `competitive_supremacy.py` SDO gate, extended to a user-facing artifact.

## 2. Locked decisions (from brainstorming)

1. **Dual measurement mode**, claim strengths never blended: rigorous **labeled-gold** (publishable) + advisory **unlabeled-silver**.
2. **All four dimensions** in scope: detection, residual leakage, re-identification resistance, utility/fairness/compliance.
3. **Configurable pipeline interface**: `detect(text)->spans` and/or `transform(text)->text`; a dimension self-marks `NOT_ASSESSABLE` when its required input is absent.
4. **Four outputs**: JSON + reproducibility bundle, technical Markdown, polished self-contained HTML executive report, one-page assurance summary.
5. **Audit-grade bar**: reproducible, CIs + significance, methodology + limitations, claim-strength labels, no-fabrication gate on every emitted value.
6. **Process**: focused dev-assist increment (light Requirements + Design + story-gated Development + **mandatory adversarial close**).
7. **Architecture**: A+B blend — thin runner over a paired/differential substrate; dimensions as in-tree assessor objects behind a port; reuse the no-fabrication validators; no premature plugin platform.
8. **Re-id dimension kept, honestly labeled** (not descoped, not headline): structural risk + explicitly ADVISORY synthetic-adversary stand-in + `NOT_ASSESSABLE` for trained-shadow attacks unless preconditions hold.
9. **Build sequencing: safety/honesty core first** (Phase 1), then the remaining dimensions + polished formats (Phase 2).

## 3. Non-goals / YAGNI (v1)

- **No entry-point plugin platform** for assessors or renderers (in-tree registration only; revisit when a real third party needs it).
- **No raw-PII snippets in any artifact** (see §10) — structural illustration only.
- **No dependency on the user-WIP orchestrator** (`orchestrator.py`, the S2-03 block). pii-anon's transform path is composed from the verified `transforms/` primitives directly.
- **No new learned models** (the query-aware learned relevance model, perturbation strategy completion, etc. stay Pass-2).
- **No retroactive regeneration** of the committed benchmark artifact.

## 4. The honesty model (the heart of "defendable")

### 4.1 Three-valued claim strength
Every emitted number carries exactly one `ClaimStrength`:

| Label | Meaning | When |
|---|---|---|
| `MEASURED` | Publishable / advertisable | Computed against user-supplied **gold labels**, **and** the power gate (§11.3) passes, **and** the provenance gate (§9) passes, **and** the adversarial close (§18) is green. |
| `ADVISORY` | Honest but explicitly hedged; not for headline claims | Computed against a **silver/adjudicated** reference, or via a deterministic stand-in adversary, or under-powered. |
| `NOT_ASSESSABLE` | Shown with the reason; never rendered as 0 or as a pass | Required input absent, preconditions unmet, or only one system can supply the input (no phantom win — mirrors G2). |

**Rules (enforced, tested):**
- A `MEASURED` value **must** trace to gold labels + a passing power cell. Mislabeling is a fabrication-class failure.
- Silver mode **never** produces a `MEASURED` head-to-head winner (§7.2).
- A dimension only one system can supply → the other system is `NOT_ASSESSABLE`; **no "win" is rendered** (§6.3).
- `MEASURED` and `ADVISORY` numbers are **never** combined into one headline.

### 4.2 Well-formedness ≠ honesty (red-team SHOWSTOPPER)
The no-fabrication validators check that a *number is well-formed* (finite, in `[0,1]`, non-blank). They **cannot** tell whether a number is *scientifically honest* or whether a string *contains PII*. The spec therefore separates three distinct concerns, each its own gate:
- **Numeric integrity** → the no-fabrication gate (§8.1).
- **Claim honesty** → the claim-strength engine + power/provenance gates (§4.1, §9, §11.3) producing the label at the *read site*, not as a post-hoc scrub.
- **PII safety** → the PII-egress gate (§8.2), a *separate* fail-closed byte-level gate.

## 5. Architecture

New top-level package **`src/pii_anon/assurance/`** (sibling to `eval_framework`; it is an application/orchestration layer composing eval_framework + engines + transforms + attacks). Every heavy import happens **inside** functions (mirrors `first_party.py`).

```
assurance/
  __init__.py          # public API: run_assurance_report, AssuranceConfig, AssuranceReport, PipelineAdapter, ClaimStrength
  config.py            # AssuranceConfig (dimensions, mode, sample size, seed, dataset ref, output formats, gate thresholds)
  adapters.py          # PORT 1: PipelineAdapter (detect?/transform?/Capabilities); pii-anon adapter via first_party + transforms
  dataset.py           # dual-mode load (LabeledDataset|RawDataset), dataset sha256 fingerprint, in-memory-only guarantees
  adjudication.py      # Approach-B paired substrate; silver-reference adjudication; agreement/kappa/disagreement
  claim_strength.py    # the three-valued engine; per-metric read-site validators returning MEASURED/ADVISORY/NOT_ASSESSABLE
  stats.py             # cluster bootstrap, paired bootstrap, Holm-Bonferroni multiplicity, power gate, effect size
  provenance.py        # fail-closed provenance gate + repro bundle (hash, versions, seed, env, user-pipeline descriptor)
  pii_egress.py        # PORT-adjacent SERVICE: fail-closed byte-level PII-egress gate (containment scan + detector pass)
  report.py            # AssuranceReport dataclass (raw-text-eliding __repr__); methodology + limitations
  runner.py            # orchestrator: config -> dataset -> adapters -> substrate -> assessors -> report -> gates -> render
  assessors/           # PORT 2: one Assessor per dimension; each declares required inputs, self-marks NOT_ASSESSABLE
    base.py            #   Assessor protocol + DimensionResult (value, CI, significance, claim_strength, reasons)
    detection.py
    leakage.py
    reidentification.py
    utility_fairness.py
    compliance.py
  render/              # PORT 3: Renderer per format; all raw-text-free by construction
    base.py
    json_bundle.py
    markdown.py
    html.py            # self-contained, inline-SVG, no JS, injection-hardened
    summary.py         # one-page assurance certificate
  synthetic_mirror.py  # generate a no-real-PII corpus matching the dataset's type/length/language distribution
  __main__.py          # CLI: python -m pii_anon.assurance ...
```

Plus one shared extraction in the existing tree:
```
eval_framework/validation/no_fabrication.py   # PUBLIC home for the validators (see §8.1 / §12)
```

### 5.1 Data flow
`config` → load + **fingerprint** dataset (auto labeled vs raw) → build user + pii-anon **adapters** → **paired substrate** (+ silver reference if unlabeled) → run selected **assessors** (each emits value, CI, significance-vs-other, claim-strength, not-assessable reasons) → assemble `AssuranceReport` → **numeric gate** validates every value (fail-closed) → **render** each format → **PII-egress gate** on the final bytes of each artifact (fail-closed) → write. Methodology + limitations are generated from the actual run, not boilerplate.

## 6. Measurement methodology per dimension

Each assessor declares `required_inputs` and returns a `DimensionResult` per system with `claim_strength` + `not_assessable_reason`.

### 6.1 Detection (the one genuinely MEASURED head-to-head dimension)
- Reuse `EntityLevelF1Metric.compute(predictions, labels, level=ENTITY, match_mode=STRICT)` → `EvalMetricResult.per_entity_breakdown` (P/R/F1/support per type). Also report `MatchMode.PARTIAL` (IoU≥0.5) as a secondary view.
- Composite via `compute_composite(...)` / `CompositeConfig` — but the composite is **weight-config-dependent and floor-gate-capped (non-monotone)**; it is reported as a *labeled, configured* index, never as "the" quality score, and the exact `CompositeConfig` is serialized into the bundle. Bound it with `_finite_unit_score`, not `_is_finite_number` (MINOR finding).
- Labeled mode → `MEASURED` (if power gate passes). Silver mode → `ADVISORY` agreement stats only (§7.2).

### 6.2 Residual leakage (fail-closed LOWER BOUND)
- **Labeled mode:** for each gold PII span, check whether its **surface text survives verbatim** in the pipeline's transformed output (independent of the pipeline's own detector recall). Leakage = surviving gold PII / total gold PII. This is a **lower bound** ("leakage of *referenced* entities") — stated verbatim in the report.
- **Fail closed toward MORE leakage:** empty / too-short / unparseable / exception / detector-abstain output ⇒ `NOT_ASSESSABLE`, **never 0**. A no-op transform must score ~100% leakage; a transform returning `""` scores `NOT_ASSESSABLE`.
- **Min-specificity guard:** entropy/length/dictionary filter so common-substring collisions aren't counted as leaks; characterize its FPR on a synthetic negative control; cap substring-search complexity.
- Requires `transform`. Pipeline without `transform` → `NOT_ASSESSABLE` (no phantom 0).

### 6.3 Re-identification resistance (kept, honestly labeled)
- **Default `NOT_ASSESSABLE`.** The real attack API needs a candidate persona pool + gold persona-target links + auxiliary signals; the MIA scorer needs per-record loss + gold membership + ≥128 shadow models. User datasets/pipelines lack these.
- **MIA `auc_approx` is provably degenerate (always 0.5)** → excluded entirely; it never contributes to composite.
- Where computable, report **structural linkage / quasi-identifier coverage** (genuinely computable from spans/types).
- A deterministic **synthetic-adversary linkage self-consistency** stand-in may be reported **ADVISORY only**, labeled "representative adversary, not a trained-shadow attack", with the **non-strippable `ANTI_ANONYMITY_CAVEAT` (reid.py) surfaced verbatim**. Closed-world `|C|` stated. Never contributes to a `MEASURED` claim, never built from a silver reference.
- Real LiRA@128 / Secret-Sharer attacks, when preconditions are met, are `MEASURED` but reported with a CI over R independent attack seeds (pinned), "reproducible-in-distribution, not bit-identical."

### 6.4 Utility, fairness & compliance
- **Utility:** format preservation / information loss via `utility_metrics`. Requires `transform`.
- **Fairness:** per-language / per-group gaps via `fairness_metrics` + `fairness_gate`. Requires group/language metadata; else `NOT_ASSESSABLE` for the missing strata.
- **Drop/relabel attribute-inference** (keyword-presence heuristic; returns ~1.0 on any pronoun) → `NOT_ASSESSABLE` or ADVISORY "presence of demographic-correlated surface tokens" with a stated PII-free-control baseline.
- **Compliance:** `ComplianceValidator.validate_all(detected_types)` → per-standard coverage (NIST/GDPR/HIPAA/ISO-27701/CCPA) + explicit gaps. This is a **capability/coverage** axis (not data-dependent accuracy); reported as coverage %, not as a head-to-head "win" unless both systems expose comparable type sets.
- **Pseudonymization/reversibility** is a **pii-anon capability axis**: if the user's pipeline doesn't emit a pseudonym map, the comparator is `NOT_ASSESSABLE` and **no superlative is rendered** (G2 phantom-0 rule, verbatim).

## 7. Dual-mode measurement

### 7.1 Labeled-gold mode (publishable)
Dataset carries gold `LabeledSpan`s → detection + residual-leakage are `MEASURED` subject to the power/provenance/close gates. The publishable path.

### 7.2 Unlabeled-silver mode (advisory, no winner)
- pii-anon is one of the two systems that build the silver reference it would be graded against → **circular / self-favoring**. Therefore silver mode **never** crowns a `MEASURED` head-to-head winner.
- It reports: **agreement rate + Cohen's κ**, a **disagreement breakdown** (only-pii-anon / only-user / both), and a **sensitivity swing** (recompute metrics with *each* system as sole reference; if F1 ordering flips between union-gold and intersection-gold, label the comparison non-robust). All `ADVISORY`.
- Optional stronger silver: adjudicate disagreement spans with a **third, independent** detector (not under comparison) or a sampled human/LLM pass; report κ rather than P/R/F1.
- The silver reference is a **raw-derived intermediate** → in-memory only, never persisted (§10).

## 8. The two gates (explicitly separate)

### 8.1 Numeric no-fabrication gate
Runs every numeric value in the assembled report through the validators (`_finite_unit_score` for `[0,1]` scores incl. composite; `_is_finite_number`; `_is_nonblank_str` for provenance/identifier strings; nested values — list elements, dict values, curve rows — audited too, per the SDO close lesson). Rejects bool-as-int, NaN/±inf, out-of-range. **Fail-closed**: a value that can't be validated is not emitted. Source of these validators: §12 (public module).

### 8.2 PII-egress gate (NEW, distinct, fail-closed)
Runs on the **final serialized bytes** of *every* artifact (JSON, MD, HTML, one-pager, bundle entries), as the **last step before any write**:
1. **Containment scan**: no substring of length ≥ *k* from any in-memory raw record may appear in the output bytes.
2. **Detector pass** (defense in depth): run a detector over the output itself; any residual PII → fail.
On any positive, **refuse to emit that artifact** (raise). Kept conceptually and in-code separate from §8.1 so neither is mistaken for the other.

## 9. Provenance & reproducibility (fail-closed gate)

A canonical-run-style gate refuses the `MEASURED`/publishable label unless **all** present and non-blank (strict `is True` for the publishable flag):
`dataset_sha256` + `seed` + full **env capture** + pinned **pii-anon version** + pinned dep versions + a **user-pipeline descriptor** (version string, or — better — a content hash of its outputs over the dataset) + recorded **user-pipeline determinism**. A networked/non-deterministic user pipeline caps the comparison at `ADVISORY`.

**Reproducibility claim, stated precisely:** *"recomputable by a holder of the original dataset whose sha256 matches X; not independently reproducible without the data (AX-001)."*

**Synthetic-mirror corpus (in the bundle):** generate a no-real-PII corpus matching the dataset's type/length/language distribution + ship the full pipeline, so an external auditor can re-run the *methodology* end-to-end on synthetic data even though headline numbers stay data-bound. The synthetic-mirror run output goes in the bundle.

## 10. PII-safety invariants (fail-closed, fatal)

The user's dataset has **real PII**, but AX-001 forbids real PII in the library/tests. Invariants (each tested):
1. **Raw PII in memory only.** The bundle stores the **hash**, never the data. The silver reference and adjudication substrate are raw-derived → in-memory only, never persisted.
2. **No raw-derived snippets in any artifact.** Show entity **type + position + length + a masked template** (run of block chars of length N). Illustrative examples drawn **only from synthetic/planted** records. **Do not self-redact with pii-anon** (the SUT — circular).
3. **Scrub all user-text-derived strings before serialization** — especially predictor **exception messages** and the `errors` list (the existing reuse path leaks `exc` text into JSON). Capture only `type(exc).__name__` + `record_id` + a redacted/omitted message.
4. **`__repr__` on every raw-bearing dataclass elides text fields.** A process-wide **excepthook** in the runner scrubs tracebacks of locals before any write. **Post-run disk scan** asserts no temp/cache/bundle file contains record bytes. A **crash-mid-run test** asserts no PII reached disk.
5. **Tests are 100% synthetic.** A test injects a predictor that raises `ValueError(record.text)` and asserts no planted token appears in any emitted byte.

## 11. Statistics

### 11.1 Confidence intervals — cluster (per-record) bootstrap
Resample **records** (the cluster unit) with replacement at the pinned seed; recompute **pooled** tp/fp/fn → P/R/F1/F2 → `compute_composite` on each resample; take percentile CIs of the **end** metric. **Never** bootstrap per-record-F1 for a micro headline; **never** bootstrap per-entity. Report resampling unit + B in the methodology. (No CI API is currently exported — this is built in `stats.py`.)

### 11.2 A-vs-B significance — paired bootstrap + multiplicity
Use the **paired bootstrap** over the per-record delta vector for the head-to-head. Apply **Holm-Bonferroni** across the full reported comparison family; state the family size; **pre-register one primary comparison** (user-pipeline vs pii-anon on the primary detection metric); all others secondary/exploratory. Report **effect size** alongside p (at 100k+ records, trivial deltas become "significant"). (Multiplicity correction is **not present** — built in `stats.py`; reuse `pairwise_significance` / Davidson where applicable.)

### 11.3 Power gate (gates the MEASURED label)
`MEASURED`-publishable requires: gold label **and** per-cell support ≥ a committed floor **and** bootstrap CI half-width ≤ a committed bound **and** significance vs the comparator established. Otherwise demote to `ADVISORY` with the reason (e.g. "insufficient power: n=12, CI ±0.31"). Carry `sample_size` + per-entity support into the bundle.

## 12. Reuse map (verified) + what must be built

**Reused (verified signatures):**
- Detection: `eval_framework.metrics.span_metrics` (`EntityLevelF1Metric`, `LabeledSpan`, `EvalMetricResult`, `MatchMode`), `metrics.composite` (`compute_composite`, `CompositeConfig`, `CompositeScore`).
- Transform (non-orchestrator): `transforms.base` (`TransformStrategy`, `TransformResult`, `TransformContext`), `transforms.strategies` (`Redaction/Placeholder/Tokenization/Generalization/SyntheticReplacement`), `transforms.policies` (`TransformPolicy`, `load_compliance_template`), `transforms.registry.StrategyRegistry`.
- Compliance: `eval_framework.standards.compliance` (`ComplianceStandard`, `ComplianceValidator.validate/validate_all`), `taxonomy`.
- Predictors/adapters: `eval_framework.external_evaluator` (`Predictor`, `evaluate_external_system`), `eval_framework.first_party` (`pii_anon_predictor`, `pii_anon_swarm_predictor`), `eval_framework.byo_pipeline`.
- Attacks: `eval_framework.attacks` (reid + mia; preconditions per §6.3), `harness.attack`.
- Metrics: `metrics.privacy_metrics`, `metrics.utility_metrics`, `metrics.fairness_metrics`, `metrics.fairness_gate`, `metrics.selective_risk`.
- Significance: `rating.significance` (`pairwise_significance`, `PairwiseVerdict`, Davidson), `rating.paired_set` (paired bootstrap / effect size — confirm exact symbols at impl).
- Reporting: `eval_framework.evaluation.reporting.ReportGenerator` (extend, not replace).
- Provenance pattern: `eval_framework.evaluation.canonical_run`.

**Validators to make public (§8.1):** create `eval_framework/validation/no_fabrication.py` housing `is_finite_number`, `finite_unit_score`, `is_nonblank_str`, `detected_entity_names`, `safe_repr` (currently private at `competitive_supremacy.py:530/736/772/795/1762`). **Decision:** v1 creates the public module and `assurance` imports it; `competitive_supremacy.py` is **left untouched** to avoid disturbing the sacred gate, with a **contract test pinning behavioral equivalence** so the two copies cannot drift. A later DRY-refactor of `competitive_supremacy.py` to import the public module (which would itself trigger the mandatory close) is deferred.

**Must build:** the assurance package (§5), `stats.py` (cluster/paired bootstrap, Holm-Bonferroni, power gate, CI — none exported today), the PII-egress gate, the claim-strength engine, the silver-reference adjudication, the synthetic-mirror generator, the HTML executive + one-page renderers.

## 13. Outputs

| Artifact | Audience | Content |
|---|---|---|
| `report.json` + `repro-bundle/` | machine / auditor | full structured results, claim-strength per value, provenance, env, dataset hash, `CompositeConfig`, synthetic-mirror run, methodology, limitations |
| `report.md` | security engineer / CI | every dimension, per-entity tables, CIs, methodology, limitations, claim-strength column |
| `report.html` | exec / auditor | headline scorecard, side-by-side comparison, inline-SVG charts, claim-strength badges, methodology appendix; self-contained, no JS, injection-hardened |
| `summary.html` | trust-center / marketing | one-page posture + key scores + claim-strength badge + dataset/version fingerprint |

All four are produced by raw-text-free renderers and pass §8.2 before write.

## 14. Error handling

Broken/missing user pipeline → reported as adapter-error (scrubbed), never a crash (NFR-026 pattern). Missing labels → auto-fallback to silver mode with a prominent banner. Assessors fail-closed to `NOT_ASSESSABLE` with reason. Bounded dataset size + malformed-input guards (DoV pattern). Renderer crash-surface hardening (int→str digit limits, unhashable elements, `__repr__`, HTML/SVG injection).

## 15. Testing strategy (TDD, synthetic only)

Per-assessor exact-metric tests; adapter contract tests (spans-only / transform-only / both / broken); claim-strength tests (labeled→MEASURED, raw→ADVISORY, missing→NOT_ASSESSABLE, gate rejects mislabeled MEASURED, under-powered→ADVISORY); numeric-gate fabrication-vector tests (NaN/inf/>1/neg/bool/blank, nested values → fail-closed); **PII-egress tests** (planted synthetic PII incl. via exception messages never appears in any output/bundle; crash-mid-run leaves no PII on disk); silver-mode tests (no MEASURED winner; κ + sensitivity swing); leakage tests (no-op transform ~100%; `""`→NOT_ASSESSABLE; min-specificity FPR on negative control); stats tests (cluster-bootstrap CI contains the point estimate; Holm-Bonferroni family control; power gate demotion); renderer golden tests (deterministic given seed); reproducibility test (same dataset+seed → byte-identical JSON modulo timestamp); end-to-end (synthetic weaker "user pipeline" vs pii-anon → all formats, both gates PASS).

## 16. Public API + CLI

```python
from pii_anon.assurance import run_assurance_report, AssuranceConfig, PipelineAdapter

report = run_assurance_report(AssuranceConfig(
    dataset="my_data.jsonl",                       # labeled if gold spans present, else silver mode
    pipeline=PipelineAdapter(detect=my_detect, transform=my_transform, name="acme-dlp",
                             version="1.4.2", deterministic=True),
    dimensions=["detection", "leakage", "reidentification", "utility_fairness_compliance"],
    sample_size=2000, seed=20260617,
    outputs=["json", "markdown", "html", "summary"], out_dir="./assurance-out",
    power_min_support=50, power_max_ci_halfwidth=0.05, egress_min_k=8))
```
CLI: `python -m pii_anon.assurance --dataset … --pipeline-entrypoint acme.dlp:predict --transform-entrypoint acme.dlp:redact --out ./assurance-out`

## 17. Build sequencing

**Phase 1 — Safety + honesty core + the two genuinely-MEASURED dimensions.** Public `no_fabrication` module + contract test; `claim_strength` engine; `stats.py` (cluster/paired bootstrap, Holm-Bonferroni, power gate); `provenance` gate + repro bundle + synthetic-mirror; `pii_egress` gate; `dataset` (dual-mode, fingerprint, in-memory invariants); `adapters`; `adjudication` (paired substrate + silver κ/sensitivity); `detection` + `leakage` assessors; `runner`; `json` + `markdown` renderers. **Drive Phase 1 end-to-end through the mandatory adversarial close.** No `MEASURED` label ships until the close is 0-upheld.

**Phase 2 — Remaining dimensions + polished formats.** `reidentification` (honestly labeled), `utility_fairness`, `compliance` assessors; HTML executive + one-page renderers; CLI polish. Second adversarial close.

## 18. Adversarial close as a HARD release gate

Because pii-anon is **grading its own homework** and `competitive_supremacy.py` (a simpler artifact) needed a **7-round** close to stop forging its own verdict, a **0-upheld adversarial close is a hard release gate** for this package. Budget multiple rounds; run a confirmatory round after any hardening. Until the close passes, **no value carries the `MEASURED`/publishable label**. Provide an **external/third-party verification mode** (re-run from the repro bundle by someone other than pii-anon). The **self-grading conflict is disclosed in the methodology**, not hidden behind "audit-grade."

## 19. Traceability (increment IDs — to be folded into the dev-assist matrix)

- **FR-A01** BYO pipeline adapter (detect and/or transform) with capability detection.
- **FR-A02** Dual-mode dataset load + fingerprint; auto labeled-vs-silver.
- **FR-A03** Paired/differential substrate + silver adjudication (κ, sensitivity, no winner).
- **FR-A04** Detection assessor (MEASURED in labeled mode).
- **FR-A05** Residual-leakage assessor (fail-closed lower bound).
- **FR-A06** Re-id assessor (honestly labeled; NOT_ASSESSABLE default).
- **FR-A07** Utility/fairness/compliance assessor.
- **FR-A08** Claim-strength engine (three-valued, read-site).
- **FR-A09** Numeric no-fabrication gate (public validators module).
- **FR-A10** PII-egress gate (byte-level, fail-closed).
- **FR-A11** Provenance gate + repro bundle + synthetic-mirror.
- **FR-A12** Four renderers (JSON/MD/HTML/summary).
- **FR-A13** CLI + public API.
- **NFR-A01** PII-safety invariants (§10) — fail-closed, tested.
- **NFR-A02** Reproducibility (§9) — byte-identical given dataset+seed.
- **NFR-A03** Statistical rigor (§11) — cluster/paired bootstrap + multiplicity + power gate.
- **NFR-A04** Defendability (§18) — 0-upheld close before any MEASURED label.
- **NFR-A05** No-orchestrator dependency (composed from `transforms/` primitives).

## 20. Open risks / deferred

- Exact symbols for paired bootstrap / effect size in `rating/paired_set.py` to confirm at implementation.
- The DRY-refactor of `competitive_supremacy.py` onto the public validators module (deferred; would trigger its own mandatory close).
- Learned relevance model, perturbation-strategy completion, full LiRA harness — Pass-2.
- Real-user validation of the report's *usefulness* (vs agent-simulated) — a dev-assist Pass-2 item.
