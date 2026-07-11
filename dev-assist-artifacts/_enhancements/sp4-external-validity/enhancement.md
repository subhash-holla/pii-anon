# Enhancement Amendment: sp4-external-validity

> Managed by /dev-assist-enhance. Opened 2026-07-10. Status: **IN_PROGRESS**.
> User steer: "somewhat unconvinced the library is good enough to be broadcast as the best
> solution out there — pick the top 5 datasets other than pii-anon-eval-data, work with them,
> ensure the library does good, enhance where possible; at bare minimum broadcast performance
> against all other datasets."

## Classification

`new-capability` (external-dataset evaluation harness) + the FR-027 external-validity spirit
(the pre-registered i2b2/TAB correlation stays DUA-gated Pass-2; this is the PUBLIC-dataset
sibling). Deliverables: (1) `external_eval/` harness in the DATA repo; (2) zero-shot-first
cross-dataset results for pii_anon + pii_anon_swarm; (3) the broadcast disclosure document;
(4) targeted enhancements using external TRAIN splits only, re-reported separately.

## The five datasets (evidence-based selection, 2026-07-10)

| # | Dataset | Why | Split used | Gold format |
|---|---|---|---|---|
| 1 | ai4privacy/pii-masking-400k | most-adopted open PII benchmark | validation/1en.jsonl (17,046) | char spans (`privacy_mask`) |
| 2 | nvidia/Nemotron-PII | 2026 multi-industry synthetic PII | test parquet (100k, locale=us) | char spans (`spans`) |
| 3 | gretelai/synthetic_pii_finance_multilingual | finance-domain documents | English_test parquet (2,891) | char spans (`pii_spans` JSON) |
| 4 | TAB (NorskRegnesentral text-anonymization-benchmark) | REAL ECHR court documents — the academic anonymization standard; attacks the synthetic-only ceiling | echr_test.json (127 docs) | `entity_mentions` w/ identifier_type; gold = DIRECT+QUASI |
| 5 | PIIBench (pritesh-2711/pii-bench) | unified 10-source benchmark w/ PUBLISHED 8-system baselines (all span-F1 < 0.14) | test.jsonl (100k, BIO tokens) | BIO → char spans |

Excluded with reasons: i2b2/n2c2 (DUA-controlled — remains the FR-027 pre-registered Pass-2),
Kaggle PII-DD (credentialed download), REDACT/SPY (too new/small), beki/privy (register covered
by Gretel finance/structured).

## Discipline

- **Zero-shot FIRST**: the first reported number per dataset precedes ANY tuning. Tuning (if
  any) uses that dataset's TRAIN split only and is re-reported as a separate, labeled row.
- **One scoring core** (`external_eval/common.py`): strict exact-span + relaxed IoU≥0.5, over
  all gold AND reachable-only gold (label-map ceiling disclosed per dataset).
- **Honesty**: every native→dataset label-mapping decision documented; unreachable gold types
  disclosed; sampling seeded (20260710) + disclosed; the home-benchmark home-team caveat
  carries into the broadcast doc.
- Leak-direction + AX-003 invariants as in sp3; production masking path untouched by any
  eval-side mapping.

## Delta table (final, 2026-07-10)

| # | Delta | Status |
|---|---|---|
| 1 | `external_eval/` harness in the DATA repo: shared scoring core (`common.py`: strict + relaxed IoU≥0.5, all-gold + reachable-only, seeded sampling) + 5 dataset adapters (5-agent workflow `wf_06f84a5f`, every mapping decision documented) | DONE |
| 2 | Zero-shot vanilla runs (5/5): relaxed F2 — ai4privacy 0.213 · Nemotron 0.324 · Gretel 0.379 · TAB 0.100 · PIIBench 0.184 (strict F1 0.130 = inside the published 8-baseline family, all <0.14) | DONE |
| 3 | **GLiNER long-document collapse FOUND + FIXED**: detection collapses with input length (3 findings @500 chars → 0 @≥2,000 on a real judgment); adapter now windows (400-char whitespace-aligned + 100 overlap, swept on TAB DEV — 50/56 gold-PERSON overlap vs 18/56 @1200; offsets re-based, overlap-deduped). `tests/test_gliner_windowing_sp4.py` (4 tests) | DONE (TDD) |
| 4 | Windowed swarm re-run (5/5): aggregate Δ ≈ +0.001 — because **the swarm fusion suppresses NER-only findings** (verified: 0/5 gliner findings on a real judgment survive fusion; dormant-meta-learner singleton cap ~0.62 < 0.85 emission bar). The swarm's generalization channel is structurally discarded → swarm ≈ vanilla externally. **FLAGGED as the headline follow-up** (production masking path + feeds canonical G1 → needs own design + mandatory close) | DONE (finding) |
| 5 | Broadcast deliverable: `docs/external-validity-report.md` (5-dataset tables, per-dataset quirks, methodology honesty, the entitled-vs-not-entitled conclusion) + wired into the docs index (the A1 discoverability teeth caught the missing link) | DONE |
| 6 | Generalizable pattern defects flagged for a general-fix pass (GPS-on-date-fragments, two-cap-word name pattern on markdown headers, US-only ZIP) — NOT fixed as benchmark tuning | TODO (follow-up) |
| 7 | Home-dev swarm regression check for the windowing change (home records 400–2,000 chars are newly windowed) | RUNNING |
