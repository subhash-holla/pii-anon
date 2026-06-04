# Story Gate Synthesis — ★ S7-02 (canonical-run producer + CanonicalRunGate — the KEYSTONE)

**Aggregate verdict: REQUEST_CHANGES** (iteration 1). 4 of 6 reviewers REQUEST_CHANGES; 2 MAJOR (+ MINORs). 0 SHOWSTOPPER / 0 CATASTROPHIC. **This is the gate doing exactly its job on the program's completion-criterion artifact.**

## Reviewer set + verdicts

| Reviewer | Verdict | Key finding |
|---|---|---|
| **requirements-coverage** | **REQUEST_CHANGES** | **MAJOR** — census-profile scope-laundering (NFR-006); proved the J-crown FLIPS to gliner on the fresh numbers at the stamped scope |
| **security-sast** (PRIMARY) | **REQUEST_CHANGES** | **MAJOR** #1 census-profile no-fabrication hole; **MAJOR #2 vector #11 CONFIRMED** (gate OverflowError on huge-int); 2 MINOR |
| **axiom-compliance** | **REQUEST_CHANGES** | **MAJOR** — census-profile = epistemic/provenance violation, OUTSIDE the AX-001 sanction; G1/G2/G4/AX-002/003/004 UPHELD |
| **traceability** | **REQUEST_CHANGES** | **MAJOR** — the gate-read headline metrics don't trace to the run the provenance describes (add `detection_metrics_provenance`) |
| code-quality | APPROVE | 3 MINOR (comment order; `_matrix_sha256` cwd-relative path; `_aligned_prf` private import) |
| performance-benchmark | APPROVE | 1 MINOR — `latency_ms` profiles not census-traceable + docstring overstates "MEASURED"; CONFIRMED pii-anon stays rank-1 with *real* latency |

## The central MAJOR (4 independent reviewers) — census-profile scope-laundering

The gate-read G3/G6/J detection metrics (`recall`/`precision`/`composite_score`/`per_entity_recall`) are HARDCODED `_CENSUS_PROFILES` (canonical_run.py:192-259), **byte-identical** to the prior `benchmark-results.json` run — which carries `canonical_claim_run=False`, `git_sha=2761a27`, `dataset_source=auto`, `2026-04-27`. The producer re-stamps them under FRESH provenance (`scope=data-v2.0.0`, `git_sha=4ddcae6`, fresh `dataset_sha256`/`timestamp`), with **no in-band disclosure** of the census origin. The fresh detection run (max_samples=8) is fenced into a side block the gate doesn't read.

**Materiality (empirically proven by requirements-coverage + axiom):** on the artifact's OWN freshly-measured numbers, the composite crown flips — pii-anon 0.7166 < gliner 0.7224. **PROVISIONAL_SOTA on the composite axis is true only at the undisclosed 148,994-record census, not at the stamped representative scope.** The CanonicalRunGate cannot self-catch this (it validates field *presence*, never detection-vs-provenance coupling).

This falls OUTSIDE the AX-001 keystone-teeth sanction: that sanction permits a synthetic *input* through a real scorer (G1/G2/G4 — verified honest); it does NOT permit transcribing a prior run's actual *output* numbers under fresh provenance.

## The honest endpoint shift (load-bearing for the program)

The program ASSUMED a representative-scale canonical run → PROVISIONAL_SOTA. The reviewers establish that is OPTIMISTIC: pii-anon's raw-detection **composite/J dominance genuinely requires full-census scale** (a small sample is razor-thin/flips to gliner). G3 (recall dominance) IS robust at fresh scale; G1/G2/G4 PASS freshly; but J/G6 (the composite/F2 race) need the full census. AND the census numbers are from OLD code (git_sha=2761a27) — they may not reflect the current HEAD's detection. So the HONEST autonomous endpoint is likely **NOT_YET at representative scale** (binding: J<0.95 — the composite race needs a fresh full-census run on the current code), with PROVISIONAL_SOTA being a documented **Pass-2** (the user's full-census re-run) and CLAIM_GRADE a further Pass-2 (Tier-C + real-NUTS J).

## Remediation (iteration 2) — integrity-first

1. **Make the gate-read G3/G6/J FRESH-measured** (current code), remove `_CENSUS_PROFILES` from the gate-read path; provenance honestly describes the fresh run. **Report whatever verdict the fresh run honestly yields** — do NOT engineer PROVISIONAL_SOTA.
2. **Demote the documented census to a TRANSPARENT `pass2_full_census_reference` block** (source `git_sha=2761a27`, `dataset=auto`, 148994 records, the rank-1 composite) — the documented Pass-2 basis, explicitly NOT this run's measurement.
3. **Harden vector #11** in the SDO gate (`_is_finite_number` → wrap `math.isfinite` in `try/except (OverflowError, ValueError): return False`) — pre-authorized by the story; a gate change requiring the close to re-verify.
4. **MINORs:** G1 ε NaN-guard (`_finite_unit_score`); the `representative_in_tree_detection` scope mislabel; comment order; `_matrix_sha256` `__file__`-anchored; the latency "MEASURED" docstring honesty.
5. **Tests** assert the MACHINERY (canonical_claim_run=True + G1/G2/G4 compute + G7 PASS + fail-closed gate + SO-11) + the HONEST verdict behavior — NOT a hardcoded PROVISIONAL_SOTA.
6. **THEN the MANDATORY adversarial close** (11 vectors) on the honest producer.

## What verified GREEN (keep)

G1 (floored-fusion ε=0 by construction), G2 (real S4-01 scorers, SO-11 comparator, AX-004 separate keys), G4 (real S4-03 reporter over synthetic-calibrated input) — all within AX-001. The fail-closed gate (every producer-emittable vector), determinism (byte-identical modulo timestamp), the output-path sandbox (`..`-traversal-safe), import isolation, off-limits byte-identity (SDO gate `fa575f4f`, competitor_compare `7cae16c8`), nothing under `artifacts/benchmarks/*`, the SO-11 contract (both sides). The underlying SOTA claim (pii-anon rank-1 at census, even with real latency) is NOT disputed — only the provenance honesty of presenting it as a fresh representative run.
