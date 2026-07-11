# External-Validity Report — pii-anon on five non-home PII benchmarks

**Date:** 2026-07-10 · **Library:** pii-anon 1.6.0rc1 · **Program record:** `dev-assist-artifacts/_enhancements/sp4-external-validity/`

This report discloses how pii-anon's two first-party detectors perform on the five most relevant
public PII datasets **other than** the library's own benchmark (`pii-anon-eval-data`). It exists
because the home-benchmark result is not an external-validity claim, and no "best-in-class"
statement should be broadcast without exactly this disclosure.

## TL;DR — the honest picture

- **Home benchmark** (pii-anon-eval-data v2.2.0, 66-type strict-v1, 31,048 test records):
  pii_anon_swarm **F2 0.893** / pii_anon **F2 0.892** — #1/#2 of 13 detectors at full 66/66
  coverage (best external: AWS Comprehend 0.736 at 24/66).
- **Zero-shot on five external datasets:** relaxed-F2 between **0.10 (real court documents)
  and 0.39 (finance formats)** — a large transfer gap. The rule-based engine's patterns encode
  the home corpus's label conventions; they do not transfer for free.
- **Context:** on PIIBench, the one external dataset with a published multi-system baseline
  table, our zero-shot strict F1 (0.130) lands **inside the published 8-baseline family**
  (Presidio, spaCy, Piiranha, … — all < 0.14), with far higher per-type scores where an honest
  type counterpart exists (IBAN 0.76, IP 0.48, EMAIL 0.46).
- **Conclusion we are entitled to:** pii-anon is the strongest detector *on its own benchmark by
  a wide, statistically-significant margin* (13-player Elo tournament: both first-party systems
  above every external at >2σ), with best-in-field type coverage — and it behaves like every
  other rule-based system zero-shot on foreign label conventions. It is NOT entitled to a
  general "best PII detector" claim, and this report is the standing reason why.

## The five datasets (selection rationale in the sp4 enhancement record)

| Dataset | Register | Split scored | Records | Gold spans |
|---|---|---|---|---:|
| [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | synthetic prose (community standard) | validation/1en (seeded 5,000 of 17,046) | 5,000 | 5,838 |
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | multi-industry documents (2026) | test, locale=us (seeded 5,000 of 50,000) | 5,000 | 43,117 |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | finance formats (SWIFT/EDI/logs) | English_test (full) | 2,891 | 12,853 |
| [TAB — Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) | **REAL** ECHR court judgments | echr_test (full) | 127 docs | DIRECT+QUASI mentions |
| [PIIBench](https://huggingface.co/datasets/pritesh-2711/pii-bench) | unified 10-source benchmark | test (seeded 5,000 of 100,002) | 5,000 | BIO-derived spans |

Excluded with reasons: i2b2/n2c2 (DUA-controlled — remains the pre-registered FR-027
correlation study), Kaggle PII-DD (credentialed download), REDACT/SPY (too new/small),
beki/privy (register covered by Gretel).

## Zero-shot results (nothing tuned; first reported numbers)

Scoring: one shared core (`external_eval/common.py` in the eval-data repo) — **strict** exact
(start, end, mapped-label) and **relaxed** (same label, span IoU ≥ 0.5); each over ALL gold and
over **reachable** gold only (types our native labels can map onto — the label-map ceiling,
disclosed per dataset). Seed 20260710. `pii_anon` = the vanilla regex engine; `pii_anon_swarm` =
the ensemble (regex + GLiNER + Presidio + Scrubadub through the swarm fusion).

| Dataset | detector | strict F2 | relaxed F2 | relaxed F2 (reachable) | gold reachable |
|---|---|---:|---:|---:|---:|
| ai4privacy-400k | pii_anon | 0.174 | 0.213 | 0.245 | 84% |
| | pii_anon_swarm | 0.204 | 0.237 | 0.272 | |
| Nemotron-PII | pii_anon | 0.296 | 0.324 | 0.403 | 75% |
| | pii_anon_swarm | 0.296 | 0.324 | 0.403 | |
| Gretel finance | pii_anon | 0.255 | 0.379 | 0.390 | 95% |
| | pii_anon_swarm | 0.302 | 0.389 | 0.401 | |
| TAB (real documents) | pii_anon | 0.050 | 0.100 | 0.106 | 94% |
| | pii_anon_swarm | 0.056 | 0.101 | 0.107 | |
| PIIBench (F1) | pii_anon | 0.124 | 0.184 | — | — |
| | pii_anon_swarm | 0.130 | 0.189 | 0.282 (reach) | |

Per-dataset mapping decisions, gold-quirk notes (e.g. ai4privacy's scrambled placeholder values;
PIIBench gold where only 81.7% of EMAIL spans contain "@"; Nemotron's split first/last-name
convention absorbing 75% of strict FPs), and full per-type tables live in the result JSONs:
`pii-anon-eval-data/external_eval/results/*.json`.

## What the external run found and what was fixed

1. **GLiNER long-document collapse (FIXED, shipped in this tranche).** The NER engine's
   detection collapses with input length (3 findings on the first 500 chars of a real judgment,
   0 at ≥2,000 chars); the adapter fed whole documents unwindowed, so every long document lost
   ALL NER contribution. The adapter now windows long inputs (400-char whitespace-aligned
   windows + 100-char overlap, swept on TAB **dev**, offsets re-based, overlap-deduped).
   `tests/test_gliner_windowing_sp4.py`.
2. **Swarm fusion suppresses NER-only findings (FLAGGED — the headline follow-up).** Even with
   windowing, 0/5 gliner findings on a real judgment survive the swarm fusion: under the dormant
   meta-learner fallback, single-engine spans cap at ~0.62 confidence, below the 0.85 emission
   bar — the swarm's only generalization channel is structurally discarded, which is why
   swarm ≈ vanilla on every external dataset. Fixing this is a production-masking-path change
   (and feeds canonical G1), so it needs its own design + mandatory adversarial close.
3. **Generalizable pattern defects (FLAGGED):** the GPS-coordinates pattern false-fires on date
   fragments ("15/09"); the capitalized-two-word name pattern fires on markdown headers and
   field labels; the ZIP grammar is US-only. These are honest external findings, to be fixed as
   general improvements (not benchmark-specific tuning) with home-benchmark regression checks.

## sp6 update (2026-07-11): the NER channel opened — measured gains, separately labeled

The sp6 cross-dataset mining found the pool engines were already detecting most of the missing
gold and the pipeline was discarding it (three mechanisms: un-normalized presidio labels, missing
GLiNER labels + window artifacts, and a corroboration gate structurally unreachable for
single-engine ML findings). The fixes went through a 3-round mandatory adversarial close (which
caught, and we fixed, two leak-direction inversions the eval numbers alone would never have shown —
a GPS pattern narrowing and a presidio label remap that each stopped previously-masked spans from
being masked). Post-fix, HOME improved too (dev swarm F2 0.8928 → 0.8952; vanilla 0.8916 → 0.8927).

Swarm relaxed-F2, zero-shot rows above vs the sp6 DEFAULT config (all label-mapping and sampling
identical; the zero-shot rows remain the pre-sp6 record):

| Dataset | pre-sp6 swarm | sp6 default | sp6 anonymization profile |
|---|---:|---:|---:|
| ai4privacy-400k | 0.237 | **0.267** | — |
| Gretel finance | 0.389 | **0.439** | — |
| Nemotron-PII | 0.324 | **0.335** | — |
| PIIBench | 0.189 | **0.196** | — |
| TAB (real documents) | 0.101 | **0.138** | **0.491** (strict 0.432) |

The **anonymization profile** (`SwarmConfig.anonymization_profile()`) additionally accepts
single-engine quasi-identifier findings (LOCATION / DATE_TIME / NATIONALITY / JOB_TITLE) — the
types document-anonymization tasks treat as maskable gold and short-record corpora do not
annotate. It is a documented workload profile, reported as its own row, never the default and
never conflated with the zero-shot record. On TAB — real court judgments, the academic
anonymization benchmark — it recovers ~90% of the engine-union counterfactual headroom
(0.101 → 0.491 relaxed F2).

## Methodology honesty

- **Zero-shot first:** every number above was measured before any tuning against the dataset;
  any future tuned run must use that dataset's TRAIN split only and be reported as a separate,
  labeled row (none exists yet).
- **Label mapping:** our native labels were mapped into each dataset's own taxonomy only where
  semantics genuinely correspond; every judgment call is recorded in the result JSONs. A gold
  type with no honest counterpart counts against recall and is disclosed as unreachable.
- **The home-benchmark caveat cuts both ways:** external detectors on OUR leaderboard face the
  mirror problem (their labels projected onto our 66 types); coverage columns disclose it there
  identically. Cross-benchmark comparisons should always be read with the hosting benchmark's
  home-team advantage in mind.
- Sampling seeded (20260710) and disclosed; TAB/Gretel scored in full.
