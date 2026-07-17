# SP1 "Honest G6" — Design Spec

Date: 2026-06-10
Status: approved by PO (section-by-section), pending written-spec review
Branch context: `pdlc/sota-program`, post v1.5.0rc1 RC close (SO-24)

## 1. Why this iteration exists

The v1.5.0rc1 program closed with an honest SDO verdict of **NOT_YET**: G1–G5 and G7
all PASS with zero pending axes, and **G6 (raw-detection F2 non-inferiority) is the
single binding FAIL** — core F2 0.7214 vs best Tier-R (gliner) 0.75, ε_F = 0.01.

The 2026-06-10 canonical artifact decomposes that loss precisely:

| Fact | pii-anon | gliner |
|---|---:|---:|
| Census precision (n=8 records) | 0.592 | 0.964 |
| Census recall | 0.763 | 0.711 |
| Census F2 | 0.7214 | 0.75 |
| Zero-recall entity types | DEA_NUMBER, DRIVERS_LICENSE, LICENSE_PLATE (+DOB 0.5) | 8 types at 0.0 |
| Benchmark-scale (149K) precision | 0.83 (quality profiles) | 0.91 |
| Benchmark-scale F1 | 0.814 (quality) / 0.756 (speed) | 0.764 |

Two structural findings drive the design:

1. **The census is 8 records** (`in-tree-fresh-8-records`; the CLI default
   `--max-samples 8` is documented as "kept tight so the run stays fast"). At
   benchmark scale pii-anon's measured precision is 0.83, not 0.59 — at those
   rates pii-anon's F2 ≈ 0.81 vs gliner ≈ 0.70. A statistically powered census
   likely flips G6 on its own; the 8-record draw is too small to represent the
   system's own measured behaviour.
2. **At least one entity-label mismatch double-penalizes detection.** The DEA
   pattern + checksum validator exist (`engines/regex/patterns.py:506`,
   `validators.py:329`) but emit `MEDICAL_LICENSE`; census ground truth says
   `DEA_NUMBER`. Every such mismatch scores one false positive AND one false
   negative.

## 2. Locked decisions (PO answers, 2026-06-10)

- **Directions:** all four selected — G6 detection gap, orchestrator wire-ins,
  Pass-2 evidence, evaluation-as-product. Decomposed into SP1–SP4 (§3).
- **Orchestrator WIP:** stays the PO's; `src/pii_anon/orchestrator.py` and
  `tests/test_moe_enhancements.py` are READ-ONLY this iteration. Wire-ins become
  mount-ready seams (SP2), not live wiring.
- **Exit bar:** G6 PASS **on a powered census** — detection improvements plus a
  statistically sized canonical draw. A PASS on the 8-record draw is explicitly
  NOT the bar (coin-flip win).
- **Tier-C:** no cloud runs, no spend, no adapters this iteration. Target verdict
  is therefore **PROVISIONAL_SOTA** (CLAIM_GRADE requires Tier-C run-or-waived).
- **Process:** hybrid — superpowers flow (spec → plan → TDD → verification),
  code-review at milestones, and the MANDATORY adversarial SDO close for any
  control-path change (`competitive_supremacy.py`, `canonical_run.py`,
  gate artifacts, or the canonical CLI surface). No story files / signoff YAMLs /
  5-reviewer gates this iteration.

## 3. Program decomposition

| Sub-project | Scope | Depends on |
|---|---|---|
| **SP1 — Honest G6** (this spec) | Track A detection quality + Track B powered census → certified PROVISIONAL_SOTA | — |
| SP2 — Mount-ready ORCH seams | Adapters + contract tests for MoE gate v2 route, query-aware router pre-filter, pii-anon-itself predictor, reader capabilities; final mount = few-line orchestrator change once PO WIP clears | SP1 (sequencing only) |
| SP3 — Pass-2 local-real evidence | Corpus fairness canonical wire-in (control-path ⇒ mandates SDO close), real extraction fixtures; LLM-adversary/LiRA/Tier-C stay deferred (spend) | SP1 |
| SP4 — Evaluation as a product | `pii-rate-elo` reproducible harness + methodology docs, new files only | SP1 numbers |

Each sub-project gets its own spec → plan → implementation cycle. This spec covers SP1 only.

## 4. SP1 exit criteria

A certified canonical artifact, produced at powered scale, that the **unchanged**
SDO gate (`competitive_supremacy.py` md5 `3b842e81c3f03eafd11f9c655c1789a0`)
reads as **G1–G5, G7 PASS + G6 PASS → PROVISIONAL_SOTA**, with:

- census size N sized so the F2 MDE ≤ ε_F = 0.01 at α = 0.05 / power 0.80
  (expected N = 5,000–10,000; exact N computed in B1 from span density);
- seed pinned; two same-seed runs byte-identical modulo timestamp;
- full suite, ruff, mypy (both modes), `make perf` green;
- no gate or threshold weakened anywhere.

## 5. Track A — Detection quality (in-tree; no control path)

Work lands in `src/pii_anon/engines/regex/{patterns,validators,confidence}.py`
and `engines/regex_adapter.py`. Components in leverage order:

- **A1 — Entity-label alignment audit (first).** Reconcile all 45 registered
  `PatternSpec.entity_type` values against the eval taxonomy
  (`eval_framework/taxonomy.py`) and the dataset `TAXONOMY_MAP`
  (`swarm_datasets.py`). Fix mismatches (known: DEA → `MEDICAL_LICENSE`). Add a
  STANDING contract test: every registered pattern label maps into the taxonomy;
  unmapped labels fail the suite.
- **A2 — Zero-recall entity fixes.** Diagnose `DRIVERS_LICENSE` and
  `LICENSE_PLATE` against actual census surface forms (patterns exist; suspects
  are the context-gated regexes and the `HIGH_FP_TYPES` −0.15 penalty +
  0.50 emit floor interaction). Then DATE_OF_BIRTH (0.5), and EMAIL/PERSON_NAME
  (0.875) only if cheap.
- **A3 — Precision program.** Target known FP factories measured-first:
  PERSON_NAME over-capture (base 0.68; extend negative-lookahead exclusions),
  generic NATIONAL_ID/AADHAAR shapes (wire existing Verhoeff/VIN validators if
  the PatternSpecs don't reference them), and threshold strategy (consider a
  higher emit floor for `HIGH_FP_TYPES` only). Every change ships with its
  measured per-entity P/R/F2 delta; nothing lands on intuition.
- **A4 — Measurement discipline.** All measurement via `compare_competitors()`
  directly (NEVER the README-rewriting benchmark script), fixed seed, one frozen
  draw for the whole track. Per-change report: per-entity P/R/F2 **and latency**.
  Hard guardrails per change: G1 recall floor green; entity coverage ≥ 0.8;
  NFR-009 latency ceilings respected (`make perf` green).

Target, measured on the powered draw (B1's N): precision ≥ 0.75 with recall
≥ 0.79 → F2 ≥ 0.79, real margin over gliner (~0.70–0.75), robust to draw
variance.

## 6. Track B — Powered census (parallel to Track A)

- **B1 — Power analysis, written down.** New doc
  `docs/powered-census-protocol.md`: N computed from dataset span density for
  F2 MDE ≤ 0.01 (α = 0.05, power 0.80), arithmetic shown, exact command + seed
  recorded for reproduction.
- **B2 — Draw qualification.** Confirm sampler dataset-source resolution at N
  (external `pii_anon_datasets` v2 vs in-tree fixture — artifact must say
  which); entity-type mix keeps the coverage denominator honest; same-seed
  byte-identity verified at N (NFR-005 at scale).
- **B3 — Runtime qualification.** One timed dry run at N across all five Tier-R
  systems; budget gate ~1.5h total (estimate ~15 min for the slowest system at
  83ms/doc).
- **B4 — No code changes expected.** The track uses the public
  `--max-samples`/`--seed` CLI surface only. Any forced edit to
  `canonical_run.py` / `competitive_supremacy.py` / CLI defaults ⇒ STOP, run the
  adversarial SDO close first (budget 2+ rounds + confirmatory round).

## 7. Certification & merge

Order: Track A deltas land (measured) → Track B dry-run qualifies the draw →
certified powered run (`pii-anon canonical-run --seed <pinned> --max-samples <N>`)
→ supremacy verdict read → full suite + gates green → milestone code-review.

- The PO's 8-record `artifacts/canonical/canonical-run.json` (2026-06-10) is
  archived as `canonical-run-8rec-2026-06-10.json`, not destroyed — the
  before/after pair is methodology evidence.
- **Honest-failure path:** if the powered census still reads G6 FAIL after
  Track A, nothing is weakened. Publish the honest number, decompose the
  per-entity gap on the powered draw, iterate Track A against the new yardstick.
  The exit bar is reached only by real detection wins.
- NOT in SP1: version bump, RC ceremony, publishing.

## 8. Testing & verification

- **TDD per detection change:** failing fixture test first (surface form →
  expected type/span/label), then the fix. Representative-metric tests anchor at
  EXACT reference values, not bounds (standing program lesson).
- **Standing contract test** from A1 (label alignment) prevents silent
  recurrence.
- **Latency co-equal:** per-change latency reported next to P/R/F2; G5 is live
  and the speed moat is the composite story — we do not trade it away.
- **Suite discipline:** full suite via `.venv` xdist (~7 min); ruff + mypy both
  modes; slow census-at-N tests env-gated. Code-review milestone at Track A
  completion and before certification.

## 9. Risks & guardrails

- **Protected (read-only) all iteration:** `src/pii_anon/orchestrator.py`,
  `tests/test_moe_enhancements.py`, `README.md`, `docs/benchmark-summary.md`,
  `docs/pii-rate-elo-value.md`, `artifacts/benchmarks/*`.
- **Draw stability:** benchmark dataset can regenerate non-deterministically —
  pin the seed, freeze one draw for Track A deltas, never regenerate mid-track.
- **Precision/recall coupling:** recall floor + coverage checked per change, not
  at the end.
- **Scale surprises:** powered census may surface new weak entities the 8-record
  draw never sampled — honest signal, feeds iteration; not a reason to resize.
- **Close discipline:** any control-path touch ⇒ adversarial SDO close before
  proceeding; confirmatory round after any hardening.

## 10. After SP1

SP2 (mount-ready seams) → SP3 (fairness wire-in, rides its own SDO close) →
SP4 (pii-rate-elo as a product). Each re-enters at spec stage with SP1's
powered numbers as ground truth.
