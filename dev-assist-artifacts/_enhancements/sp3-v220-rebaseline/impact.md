# Impact Set — sp3-v220-rebaseline

> **Agent-derived, advisory — never claims complete traversal.** Depth cap 5 / impact_bound 10
> (defaults; no `knowledge:` block in `developer-assistant.yaml`). Walk executed agent-side.

## Pre-hoc impact-set (deduplicated union, three-source anchoring)

| # | Artifact / ID | Source | Why reached |
|---|---|---|---|
| 1 | FR-001 (BYO-pipeline adapter contract, MUST) | Reverse Matrix | The first-party seam (`eval_framework/first_party.py` → DATA adapters) implements it; LABEL_MAP extension lands on its implementing surface (DATA repo), FR criteria unchanged |
| 2 | FR-002 (identical incumbent scoring, SHOULD) | Reverse Matrix | Leaderboard comparability depends on the single delegation path |
| 3 | UC-01 (external assessment) | matrix UC column | The v2.2.0 external assessment is UC-01's scenario |
| 4 | DC-12 (BYO-SDK surface) | matrix DC column | Adapter/predictor seam design case |
| 5 | **FR-040 (NEW, grafted)** — Art-9 special-category detection (SEXUAL_ORIENTATION / TRADE_UNION_MEMBERSHIP / GENETIC_DATA) | new-capability graft | The 63/66→66/66 coverage delta; append to requirements-document.md + matrix |
| 6 | `tests/test_pattern_label_alignment.py` | cataloger/test surface | Pins ALLOWED_NON_CORPUS_LABELS ↔ DATA label map alignment; will change with new labels |
| 7 | `tests/test_swarm_baseline_integration.py` | cataloger/test surface | Pins swarm fast-pass confidence floor (≥0.90 structured-ID) |
| 8 | first-party predictor tests (`tests/**/test_first_party*`) | cataloger/test surface | Native-label emission surface |
| 9 | traceability-matrix.md | R8 canonical | FR-040 row + Story/Test backfill |

Code surfaces (not PDLC artifacts, listed for the delta plan): `src/pii_anon/engines/regex/patterns.py`
(+ possibly `confidence.py` context words), `src/pii_anon/eval_framework/first_party.py`;
DATA repo: `baselines/pii_anon_baseline.py` + `pii_anon_swarm_baseline.py` (LABEL_MAP).

## Escalate-to-pivot check

- **MUST-touch:** NO existing MUST requirement's criteria are mutated. FR-001/FR-016 are
  *reached* (their implementing surfaces are adjacent) but their artifacts + contracts are
  untouched. FR-040 is a NEW row (classified SHOULD — external-coverage capability, not a
  release-blocking MUST).
- **Over-bound:** deduped artifact set = 9 ≤ impact_bound 10.
- **Un-anchorable:** no — all items anchor via Reverse Matrix + cataloger/test surfaces.

**Verdict: proceed-as-delta (no human checkpoint tripped).**

## Forced-full trigger evaluation (mandatory mechanical, persisted for --close)

| Trigger | State |
|---|---|
| Anchors to none of the three sources | not tripped |
| `knowledge.enabled: off` / substrate absent for a concept-dependent class | **TRIPPED** — no `da_links` store exists; PL-1 is `new-capability` (concept-dependent) |
| Un-classifiable request | not tripped |
| No prior full release-readiness report | not tripped (`05-testing/release-readiness-report.md` exists, 2026-05-30) |

**FORCED-FULL VERDICT: YES — at `--close`, run a FULL Stage-5 (not regression-scoped).**
Note: the canonical Stage-5 report is 2026-05-30-stale vs ~40 landed stories, so a full re-run
is independently overdue; the trigger and the value align.

## Post-hoc re-anchor (2026-07-10, after the delta landed)

Artifacts actually mutated (diffed vs `_pre-enhancement/` snapshots):

| Mutated | Anticipated pre-hoc? | Notes |
|---|---|---|
| `src/pii_anon/engines/regex/patterns.py` (+12 patterns) | yes (impl surface) | PL-1 Art-9 (4) + PL-2 value-class (8) |
| `tests/test_coverage_tranche_sp3.py` (NEW) | yes (test surface) | 30 cases, RED→GREEN |
| `tests/test_pattern_label_alignment.py` (census 2.2.0/66 + AUTHENTICATION_TOKEN allowlist) | yes (item 6) | standing gate kept honest |
| DATA `baselines/pii_anon_baseline.py` (LABEL_MAP 63→66) | yes (item 1) | swarm adapter shares it |
| `requirements-document.md` (FR-040) | yes (item 5) | append-only graft |
| `traceability-matrix.md` (FR-040 row) | yes (item 9) | append-only |

**No UNANTICIPATED MUST was touched.** FR-040 is the anticipated new SHOULD graft; no existing
MUST criteria were mutated. Post-hoc set == pre-hoc set → no §3 escalate re-trip.

**FR-contract-drift check (defect-fix PL-2, code↔FR boundary).** PL-2 widened detection value
classes (additive recall recovery). Checked against the pinned contracts: FR-016/NFR-011/AX-003
recall-floor (`floor_fusion.py` / `shared_layer.py` **byte-identical**, verified); FR-036 stream/
batch/offline parity (no path-specific logic added — patterns apply uniformly); the SDO gate
`competitive_supremacy.py` (md5 `3b842e81…`) + `canonical_run.py` producer **byte-identical**.
No FR contract is violated — the deltas EXTEND coverage without altering any pinned Given/When/Then.

**Pre-existing failures surfaced (NOT introduced by this delta — verified via `git stash` of the
tracked code changes):** `test_docs_discoverability.py::test_a2_all_relative_doc_links_resolve`
(broken link to `artifacts/benchmarks/benchmark-results.json`, which is **deleted in the user's
working tree** — user WIP) and `test_canonical_run.py::test_provenance_scope_matches_actual_sampler_used`
(the canonical-run producer stamps `data-v2.0.0` scope while the installed dataset is 2.2.0 —
control-path drift, flagged as a tracked follow-up `task_dc3b46b5`). Both are outside this delta's
impact set and were RED at session start.
