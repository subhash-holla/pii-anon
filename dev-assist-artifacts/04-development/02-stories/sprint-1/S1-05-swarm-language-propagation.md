# S1-05 — Swarm language propagation (Sprint-1 gate remediation: kills the floor over-injection)

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01) — gate remediation |
| State | **DONE** (gate APPROVE 2026-05-31; `_reviews/story/S1-05-gate.yaml`) |
| Implements | FR-016/NFR-011 (correctness of the LIVE floor), NFR-024/025 (per-language fairness correctness), AX-002 |
| Traces | Sprint-1-close gate `wftzms2fs` MAJOR (axiom-compliance) + MINOR (traceability); Design D-SWARM DECISION 1 |
| Test-type tags | `[UNIT-TEST]` `[INTEGRATION-TEST]` `[PROPERTY-TEST]` |
| Files owned | `src/pii_anon/swarm.py` (emission only), `tests/test_swarm_language_propagation.py` (new), `tests/test_floor_fusion_wiring.py` (docstring + seam property migration) |
| Depends on | S1-02 (the live floor that exposed the bug) |
| Size | S |

## 1. Intent
The Sprint-1-close verification (workflow `wftzms2fs`) found a **MAJOR** correctness regression on the now-live floored path: `SwarmFusionStrategy` emits `EnsembleFinding`s **without** propagating `language`, so they default to `'en'` (`types.py:162`). `MoEFusionStrategy` does it correctly (`moe.py:431 language=representative.language`). Because `SharedLayerProjector`'s key is language-carrying `(field_path, start, end, entity_type, language)`, for ANY non-English document a `regex-oss` span the swarm **keeps** (emitted mislabeled `'en'`) does not match the true-language (`'es'`/`'zh'`/…) shared key, so the projector **re-injects a duplicate** tagged `shared_floor`.

**Reproduced** (verified at HEAD): `build_fusion("swarm", weights={}, min_consensus=1).merge([EngineFinding("EMAIL_ADDRESS",0.97,"correo",0,20,engine_id="regex-oss",language="es")])` → TWO spans `[(...,'en',floor=False), (...,'es',floor=True)]`; the identical `en` input → ONE. Duplicates flow into anonymization and corrupt `FusionAuditRecord` lineage.

Root-cause fix (recommended by the gate): make swarm propagate language onto emitted findings. This ALSO fixes the broader multilingual mislabeling that corrupts per-language fairness metrics (NFR-024/025) and that forced the S1-03 `field_path` workaround. Does NOT breach AX-003 (superset always held), but is a precision/correctness defect that MUST be fixed before Sprint-1 sign-off.

## 2. Given/When/Then (acceptance)
- **Given** `build_fusion("swarm").merge([regex-oss span, language="es"])` that the swarm KEEPS, **then** the output contains exactly ONE span at those offsets, carrying `language="es"`, with NO `shared_floor` duplicate.
- **Given** a swarm-emitted `EnsembleFinding` for a non-English source finding (both fast-pass `swarm.py:613-621` and Layer-4 `swarm.py:663-675` paths), **then** its `.language` equals the representative/source finding's language (mirror `moe.py:431`), NOT the `'en'` default.
- **Given** the `en` input, **then** behavior is unchanged (still ONE span, `language="en"`).
- **Regression**: the ~78 swarm/fusion/moe tests + the full suite stay green (update any test that asserted the old buggy `'en'` default — that is correcting an assertion of a bug).

## 3. Approach (VALIDATED — gate-recommended fix (a))
- In `SwarmFusionStrategy.merge`, set `language=<source/representative finding language>` on EVERY emitted `EnsembleFinding` — both the Layer-1 fast-pass emission (`swarm.py:613-621`) and the Layer-4 emission (`swarm.py:663-675`). Use the cluster's representative finding's `.language` exactly as `moe.py:431` does (`language=representative.language`). Read 600-680 to locate the representative/source object available at each emission site.
- Do NOT change the projector key (language-carrying is correct once producers agree on language); do NOT change MoE.
- This is emission-only surgery; do not alter clustering/Dawid-Skene/meta-learner logic.

## 4. MINOR remediation (same diff, gate traceability finding)
- Fix the stale docstring in `tests/test_floor_fusion_wiring.py:21-22` ("hypothesis is not yet a dependency") — it IS a dep since S1-04.
- Migrate the seam-level seeded property test `tests/test_floor_fusion_wiring.py::test_nfr_011_property_superset_invariant_seeded` (uses `random.Random(1602)`) to `hypothesis @given` (same pattern S1-04 used in `test_shared_layer_projector.py`). Keep determinism (`@settings(..., derandomize=True)`).

## 5. RED → GREEN → REFACTOR
- **RED**: `tests/test_swarm_language_propagation.py` — assert the single-span/correct-language acceptance above; confirm it FAILS at HEAD (duplicate present, language `'en'`). Commit RED.
- **GREEN**: swarm emission language propagation. Confirm the duplicate is gone + 78 swarm/moe/fusion green. Commit GREEN.
- **CLEANUP**: §4 MINOR (docstring + seam property @given migration). Commit.
- Full suite green; ruff + mypy --strict clean.

## ⛔ Scope / safety
- Files owned: ONLY `src/pii_anon/swarm.py`, `tests/test_swarm_language_propagation.py`, `tests/test_floor_fusion_wiring.py`, this story `.md`.
- DO NOT touch `src/pii_anon/orchestrator.py` or `tests/test_moe_enhancements.py` (user WIP). Narrow explicit git staging only.

### Evidence (commit hashes — branch `pdlc/sota-program`)
- **Dependency note**: S1-02 (the live floor that exposed this bug) is implementation-complete with all DoD boxes checked EXCEPT its own story-gate APPROVE, which is blocked by the very Sprint-1-close MAJOR (`wftzms2fs`) that THIS story remediates. Proceeding per the explicit gate-recommended root-cause fix (the alternative is a deadlock). S1-02 evidence (line 53) already documents that the swarm KEEPS the `regex-oss` span natively — that kept-but-mislabeled-`en` span is exactly what S1-05 fixes.
- **Reproduction (verified at HEAD `a5eb44f`)**: `build_fusion("swarm", weights={}, min_consensus=1).merge([EngineFinding("EMAIL_ADDRESS",0.97,"correo",0,20,engine_id="regex-oss",language="es")])` → `'es': count=2 -> [(0,20,'en',shared_floor=False), (0,20,'es',shared_floor=True)]`; identical `en` input → `'en': count=1 -> [(0,20,'en',shared_floor=False)]`.
- **RED** `6fe5660` — `test: S1-05 RED — pin FR-016/NFR-011/NFR-024/AX-002 swarm language propagation` (2 files: this story + `tests/test_swarm_language_propagation.py` (new)). 5 tests collected; 4 FAIL at HEAD for the correct reason (fast-pass emits `'en'` not source `'es'`; Layer-4 IBAN emits `'en'` not `'es'`; the live floor re-injects a `shared_floor` `'es'` duplicate so the `es` path returns 2 spans). The `en` regression-guard PASSES at HEAD (the buggy `'en'` default happens to match) → RED gate satisfied.
- **GREEN** `6dbb37b` — `feat: S1-05 GREEN — swarm propagates language at both emission sites` (2 files: `src/pii_anon/swarm.py` + this story). Emission-ONLY surgery: (1) Layer-1 fast-pass site now sets `language=f.language` (source `EngineFinding`); (2) Layer-4 site now sets `language=_representative_language(candidate)` — a new deterministic module helper that mirrors `moe.py:431`'s `representative.language` by reading the cluster's source `engine_findings` (prefers the shared `regex-oss` finding, else first-in-insertion-order; `SpanCandidate` carries no language of its own). Clustering / Dawid-Skene / temperature / informativeness / meta-learner logic UNCHANGED; projector + MoE UNCHANGED.
  - **AFTER**: `'es' -> count=1 -> [(0,20,'es',shared_floor=False)]`; `'en' -> count=1 -> [(0,20,'en',shared_floor=False)]` (duplicate gone, `en` unchanged).
  - All 5 new tests pass. Targeted suite (`test_swarm_language_propagation` + `test_swarm` + `test_fusion` + `test_moe` + `test_floor_fusion_wiring` + `test_shared_layer_projector` + `test_recall_floor_per_language_gate`): 115 passed, 1 skipped. Broader `-k "swarm or fusion or moe"`: 341 passed, 0 failed (incl. all `test_moe_enhancements.py` — STOP-guard never triggered). **NO existing test asserted the old buggy `'en'`** → no other test file required editing.
- **CLEANUP** `f940df2` — `refactor: S1-05 — migrate seam superset property to hypothesis @given + fix stale docstring` (2 files: `tests/test_floor_fusion_wiring.py` + this story). §4 MINOR closed: (1) stale module docstring lines 21-22 ("hypothesis is not yet a dependency") corrected — hypothesis has been a dep since S1-04; (2) `test_nfr_011_property_superset_invariant_seeded` (`random.Random(1602)`) migrated to `test_nfr_011_property_superset_invariant` via a `@st.composite` span-spec strategy + `@given`, `@settings(max_examples=400, derandomize=True)` (same pattern as `test_shared_layer_projector.py`); `import random` removed. Migrated file: 23 passed. Determinism preserved via `derandomize=True`.
- **Full suite** (`pytest -m "not performance"`, `--cov-fail-under=84`): exit **0** — **2702 collected (2690 passed / 12 skipped / 0 failed / 0 errors)**; total coverage **86.22%** (≥84 gate; was 86.15% at S1-02, delta from +5 net new tests). No regressions vs. the 2685-passed baseline.
- **Lint / types**: `ruff check src tests` → "All checks passed!"; `mypy src/pii_anon` (--strict) → "Success: no issues found in 113 source files".
- **Swarm emission sites changed (the ONLY production change)**: (1) Layer-1 fast-pass `EnsembleFinding(...)` — added `language=f.language`; (2) Layer-4 `EnsembleFinding(...)` — added `language=_representative_language(candidate)`; (3) new module-level helper `_representative_language(candidate)` (deterministic, reads `candidate.engine_findings`). No other `swarm.py` logic touched.
- **Existing tests updated to a corrected language assertion**: NONE — no pre-existing test asserted the buggy `'en'` default; the full `-k "swarm or fusion or moe"` set (341) and the whole suite stayed green without editing any other test file.
- **User WIP untouched (verified by md5, identical before/after)**: `src/pii_anon/orchestrator.py` = `0afc6deed62bbd0653ae1051b723bace`; `tests/test_moe_enhancements.py` = `910e9cd66ad6e38c7bb64a9c51ecb1cb`. Never read-modified, staged, or committed. All commits used narrow explicit `git add <paths>` (never `-A`/`.`/`-u`); benchmarks, README, docs, and other story files were never staged.

## 12. Definition of Done
- [x] RED commit precedes GREEN; duplicate reproduced then eliminated (git-evidenced `6fe5660` → `6dbb37b`; `es` 2 spans → 1 span)
- [x] swarm propagates `language` on both emission paths; `en` behavior unchanged; floor over-injection gone
- [x] MINOR closed: docstring fixed + seam property test migrated to `@given` (`f940df2`)
- [x] ~78 swarm/fusion/moe green (115 targeted + 341 broader); full suite green (2690 passed/12 skipped/0 failed @ 86.22% cov); ruff + mypy --strict clean; user WIP untouched (md5-verified)
- [ ] Story-gate review APPROVE (`_reviews/story/S1-05-gate.yaml`) — awaiting orchestrator reviewer dispatch
