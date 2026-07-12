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
| — | STEP 2 Phase A remainder (A2 labeled-field bridge → A6) · STEP 3 Phase B fusion recalibration · STEP 4 Phase C evidence report | TODO |

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
