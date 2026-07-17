# Enhancement Amendment: sp3-v220-rebaseline

> Managed by /dev-assist-enhance. Opened + closed 2026-07-10. Status: **CLOSED**.
> Close verdict **SHIP-WITH-CAVEATS** (`05-regression/regression-readiness-report.md`); sign-off
> `_signoffs/2026-07-10-enhancement-sp3-v220-rebaseline.yaml`.

## Change request (verbatim scope)

Re-baseline and improve pii-anon + pii-anon-swarm against the **pii-anon-eval-data v2.2.0**
substrate (66-type strict-v1, 31,048 en test records / 201,880 gold spans, frozen leaderboard,
Zenodo-archived). The eval-data repo moved 63→66 types (3 GDPR Art-9 special-category types:
`GENETIC_DATA`, `SEXUAL_ORIENTATION`, `TRADE_UNION_MEMBERSHIP`, each powered ≥400 spans) and
**rebuilt the corpus** (782,677 records, 17 powered langs / 11 scripts, Thai native-script
migration, blind-realism rows) — so the sp2-era first-party numbers (vanilla F2 ~0.886 / swarm
~0.885 on the 63-type cut) are stale on two axes, and `results/tier-a/art9_coverage_v22dev.md`
explicitly records **pii_anon = 63/66, Art-9 reachable: none** (recall 0 by construction).

## Classification (closed 6-class set — multi-class request)

| Punch item | Class | Rationale |
|---|---|---|
| PL-1 Art-9 detection capability (3 types) + adapter LABEL_MAP 63→66 | `new-capability` | New detection surface grafted onto the FR-036-family detection-coverage requirements; downstream via first-party seam (S6-04/sp2 stories) |
| PL-2 v2.2.0 substrate-drift gap closure (dev-measured per-type deltas) | `defect-fix` | Narrow deltas driven by measured mismatches on the rebuilt corpus; regression-scoped Stage-5 is the main event |
| PL-3 Test-split certification + 13-detector leaderboard merge + SDO recompute | (close evidence — not a behavior class) | Feeds `05-regression/regression-readiness-report.md` at `--close` |

Docs-only: no. Un-classifiable: no.

## Invariants (MUST-not-touch list)

- **Leak-direction (sp2 SHOWSTOPPER lesson):** an eval-time precision optimization that DROPS a
  detection must never run in the production masking path; over-masking is the safe direction.
  All deltas here are additive detections or eval-side label-map entries.
- **AX-003 recall floor:** `FloorProjectingFusion` / `SharedLayerProjector` untouched.
- **SDO gate:** `evaluation/competitive_supremacy.py` (md5 `3b842e81…`) untouched — any change
  triggers the MANDATORY adversarial close. Same for `evaluation/canonical_run.py` unless the
  close is run.
- **Tuning discipline:** dev split ONLY (15,510 en records); test reserved for the PL-3 reported
  runs.
- **User WIP preserved:** untracked `src/pii_anon/assurance/`, `src/pii_anon/eval_framework/validation/`,
  `tests/assurance/`, the assurance design spec, and the deleted-in-worktree
  `artifacts/benchmarks/*` files stay exactly as found; narrow explicit `git add` only.
- **Eval-integrity:** no generator-filler anchors (the "Record shows" lesson) — patterns must
  detect the entity, not the template.

## Delta table (updated per wave)

| # | Delta | Item | Status |
|---|---|---|---|
| 1 | Dev-split baseline sp3-dev-iter0 (measurement) — vanilla F2 0.8840 / swarm 0.8849; Δ −0.006 substrate drift; coverage 63/66 | PL-2 | DONE |
| 2 | Art-9 detection: 4 patterns (`_SEXUAL_ORIENTATION`, `_TRADE_UNION`, `_GENETIC_LABELED`, `_GENETIC_INTRINSIC`) in `patterns.py` | PL-1 | DONE |
| 3 | DATA LABEL_MAP 63→66 (+ AUTHENTICATION_TOKEN identity) in `pii_anon_baseline.py` (swarm shares it) | PL-1/PL-2 | DONE |
| 4 | Census constant + version pin 2.0.0/63 → 2.2.0/66 in `test_pattern_label_alignment.py` (Art-9 census-reachable → no allowlist) | PL-1 | DONE |
| 5 | PL-2 value-class recovery: 8 patterns (CVV/PIN/PASSWORD/INSURANCE_POLICY/AUTHENTICATION_TOKEN base64/alnum/`_ZW`/OCR-P0L) | PL-2 | DONE |
| 6 | RED→GREEN test `tests/test_coverage_tranche_sp3.py` (23 detect + 4 FP-guard cases) | PL-1/PL-2 | DONE |
| 7 | FR-040 graft + traceability-matrix row (append-only) | PL-1 | DONE |
| 8 | Dev-split re-measure sp3-dev-iter1 — vanilla 0.8916 / swarm 0.8924, precision flat, 66/66 | PL-2 | DONE |
| 9 | AUTH_TOKEN polish (OCR-`8earer` + zero-width + JWT-period) → dev-en R/P/F2 = 1.000 | PL-2 | DONE |
| 10 | Test-split certification (31,048 rec): swarm F2 0.893 / vanilla 0.892, #1/#2 of 13 | PL-3 | DONE |
| 11 | Full suite (89.49%, 2 pre-existing REDs), ruff+mypy clean, ReDoS-checked, validator-drift clean | close | DONE |
| 12 | Regression-readiness report (forced-full) + SDO recompute (NOT_YET, gate byte-identical) | close | DONE |

## References

- `impact.md` — pre-hoc + post-hoc impact-sets + escalate verdict + forced-full decision
- `punch-list.md` — the delta loop ledger
- Prior art: sp2 assessment spec `docs/superpowers/specs/2026-06-12-sp2-assessment-supremacy-design.md`;
  swarm fusion Workstream-1 (commits `8ef2d20`/`e455021`/`d7be99e`)
