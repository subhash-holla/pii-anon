# Enhancement Amendment: sp5-rate-elo

> Managed by /dev-assist-enhance. Opened 2026-07-10. Status: **IN_PROGRESS**.
> User steer: "investigate and enhance the pii-rate-elo as much as we can."

## Classification

Multi-class: `new-capability` (claim-grade bayes path activated; 13-player tournament) +
`defect-fix` (whatever the adversarial hunt confirms). Control-path discipline: `canonical_run.py`
(owned by the user's parallel session `task_dc3b46b5`) + `competitive_supremacy.py` are OFF-LIMITS;
rating-stack changes that alter J semantics would require the mandatory close — flagged, not made.

## Delta table

| # | Delta | Status |
|---|---|---|
| 1 | `bayes-eval` extras installed (numpyro 0.21 / jax 0.10.2 / arviz 1.2 — CPU): the claim-grade `bayes-bt` engine is RUNNABLE for the first time (was the deferred Pass-2) | DONE |
| 2 | Editable-install metadata drift fixed: pip dist-info for `pii_anon_datasets` was 1.3.0 while the module was 2.2.0 — the cause of the CLI `--merge` refusal + the sp3 run's stale version stamp (sp3 report §4 corrected — the stale stamp was OURS, not tier1-en-all's) | DONE |
| 3 | 13-player merged assessment artifact (`sp5-test-13player/baseline_results.json`, composed, provenance-noted) + glicko tournament via `rate-elo-assessment` CLI: pii_anon 1866 / swarm 1866 Elo, both statistically distinguishable from ALL 11 externals (aws 1542 next); leaderboard.md + tournament.json | DONE |
| 4 | ★ FIRST real-NUTS claim-grade rating run (`run_bayes_claim_grade.py`): Davidson tie-aware joint posterior over the 13-player per-entity-F2 design; **NFR-001 gate PASS on real chains** (split-R̂ 1.0011 ≤ 1.01, bulk-ESS 2734 ≥ 400, 0 divergences); claim-grade J (assessment scope): first-party family = 1.000 of rank-1 mass (swarm .658/core .342, every external .000); core-vs-swarm NOT significant (θ CI [−0.65,+0.43]) — the SDO binding J<0.95 is quantified as a within-family coin-flip, not an external threat | DONE |
| 5 | 4-lens adversarial defect-hunt + enhancement scout (`wf_ea1ec403-90f`): **5 CONFIRMED (1 CATASTROPHIC) / 3 downgraded / 0 refuted** + 13 minors + 24 enhancement ideas; 5 verifications lost to session limits (status: PLAUSIBLE-unverified — incl. the RD=30.00 glicko-RD question) | DONE |
| 6 | **F1 fix (CATASTROPHIC)**: NFR-001 gate NaN/inf-blind → non-finite draws are now a first-class binding constraint (`convergence.py`; pre-fix an all-NaN posterior passed claim_grade AND fabricated J=1.0 downstream) | DONE (TDD) |
| 7 | **F3 fix (MAJOR)**: bayes-bt centered-parameterization funnel → non-centered (θ=σ·z) + target_accept 0.99; tied/near-tied/Davidson-tie-heavy designs now fit claim-grade (were 80–158 divergences = the claim-grade tier structurally unavailable on the core-vs-swarm shape) | DONE (TDD) |
| 8 | **F4/F5 fix (MAJOR)**: `_reconcile_shared_gold` in assessment ingestion — contradictory per-type gold counts, phantom single-player types, and empty-`by_entity_type` scored players now fail loud; real 13-player artifact still loads | DONE (TDD) |
| 9 | **F2 fix (MAJOR, J-consumer)**: `rank_one_distribution` tie mass split fractionally (was: all tie mass to the alphabetically-first column — relabeling-variant, biased toward 'pii-anon' in the J race). Canonical SDO verified **byte-identical** post-fix (NOT_YET / J=0.2775 / all G PASS) | DONE (TDD) |
| 9a | **★ MANDATORY close round 1 (`wf_f5c661f6`, 77 probes): CLOSE_FAIL — caught 2 MAJORs incl. a fabrication vector MY tie-split fix introduced** (one NaN draw → all-NaN shares → `NaN<J_BAR` False → forged CLAIM_GRADE with j_value=nan; pre-fix argmax had tolerated it) + an overflow-to-NaN gate evasion (finite draws ≥1e153 → NaN diagnostics → both comparisons False → claim-grade certified) + 4 minors + counts-impossible-F2 (assessment surface). ALL REMEDIATED: `_validate_samples` refuses non-finite draws; NaN diagnostics = binding constraint; subnormal unmixed chains → rhat=inf; negative divergences refused; n_params normalized; counts-consistency (reported per-type f2 must equal what its own tp/fp/fn imply — verified exact on all real rows) + n_gold totals cross-check + tournament-entry re-reconciliation | DONE (TDD, 9 pins) |
| 9b | **Round 2 confirmatory (`wf_d97fcccc`, 116 probes): 1 upheld MINOR** (hand-built-report seam: phantom `per_entity_f2` key / NaN f2 value passed the gold-only entry check; +73 Elo demonstrated) — **fixed** (f2/gold key-set equality + finite-unit f2 in `_reconcile_shared_gold`) + pinned. Convergence/bayes/ingest refuters 0-upheld incl. a **571-artifact regression sweep (zero regressions)**; non-centered model verified same-law vs the centered snapshot. 2 agents lost to session limits (j-fab re-probe + sweep) → folded into round 3 | DONE |
| 9c | **Round 3 targeted confirmatory (`wf_0d6b9581`, 44 probes): §B J-fabrication FULLY CLOSED (0 upheld — NaN/inf/all-NaN raise loud via EVERY entry to the J path; honest path emits finite J; anchors all pass, 72/72 tests).** 1 same-class in-process MINOR upheld (gold=0 phantom colluded into all players dodged the totals check) → **fixed** (per_entity_gold values must be non-bool int ≥1, keys non-blank) + pinned; the refuter's own probe script re-run against the fix — every actionable probe now rejects (the bool-gold "FAIL" was a vacuous no-op probe, verified directly; the positive-gold-redistribution pass is the refuter's own documented RESIDUAL BOUNDARY: a fully self-consistent forgery is observationally indistinguishable at any local validator). **CLOSE STATE: 0 actionable upheld across rounds 1–3; J path certified 0-upheld.** | DONE |
| 10 | `test_bayes_bt.py` import-safety test strengthened: precondition "numpyro absent" inverted into a spurious failure once the extra was installed → now proves LAZINESS directly (`numpyro/jax not in sys.modules` post-construction) in any environment | DONE |
| 11 | Deferred (flagged, not fixed): 13 minors/observations + the `--engine` CLI gap + paired-counts export (top API enhancement); 5 PLAUSIBLE-unverified findings for a follow-up verify pass; `render_assessment_report` markdown injection via detector names (round-2 in-passing note, pre-existing, report-render only) | TODO |

## Invariants

Same as sp3/sp4 (leak-direction n/a here; no-fabrication local twins in assessment_ingest preserved;
AX-002 determinism for any new rating surface; control-path files untouched).
