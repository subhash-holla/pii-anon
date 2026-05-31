# S1-03 — Per-language recall-floor CI ε-gate (ε ≤ 0.005)

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01 / DC-11) |
| State | **DONE** (gate APPROVE 2026-05-31; `_reviews/story/S1-03-gate.yaml`) |
| Implements | FR-007 (canonical-run floor gate), NFR-011 (per-language ε ≤ 0.005) |
| Traces | Design DC-11 (RecallFloorVerdictGuard / per-language ε-gate) |
| Test-type tags | `[CI-GATE]` `[INTEGRATION-TEST]` |
| Files owned | `tests/test_recall_floor_per_language_gate.py` (new); OPTIONAL additive extend `src/pii_anon/eval_framework/metrics/composite.py` (FloorGateConfig) |
| Depends on | S1-02 (floor must be live for the gate to be meaningful) |
| Size | S |

## 1. Intent
A CI gate asserting, per language L: `recall_ensemble[L] ≥ recall_shared[L] − ε` with `ε ≤ 0.005`. Because S1-02 makes `spans(ensemble) ⊇ spans(shared)` by construction, per-language `recall(ensemble) ≥ recall(shared)` holds with ε=0 in the ideal; this gate is a **REGRESSION GUARD** with a 0.005 tolerance for scoring-boundary edge cases (e.g., span-offset rounding).

## 2. Given/When/Then (acceptance)
- **Given** a labeled multilingual eval set, **when** ensemble + shared-only (`regex-oss`) detection run, **then** for every language L: `recall_ensemble[L] ≥ recall_shared[L] − 0.005`; CI fails loud otherwise.
- **Given** the eval dataset is absent, **then** the test SKIPS via `@requires_dataset` (no false CI green/red).

## 3. Approach
- Reuse per-language recall machinery in `eval_framework/metrics/fairness_metrics.py:172-228` (already groups by language; computes recall per lang).
- Define module constant `RECALL_FLOOR_EPSILON = 0.005`.
- New pytest test runs in the DEFAULT `pytest -m "not performance"` CI step (`.github/workflows/ci.yml:32`) — **no new workflow or Make target needed**; the test auto-collects.
- OPTIONAL (additive only): extend `FloorGateConfig` / `evaluate_floor_gates` (`composite.py:643-690 / 818-922`) with a `min_recall_per_language_epsilon` field for the benchmark runner. Must NOT change existing global-gate behavior.
- Guard behind `@requires_dataset` (conftest marker, `tests/conftest.py:38-41`).

## 4. Notes
- Keep the gate cheap (small labeled fixture acceptable when the full dataset is absent — but prefer real `@requires_dataset` data when present).
- The shared-only recall is computed by running ONLY the `regex-oss` engine; the ensemble recall by the floored fusion path (post-S1-02).

## 12. Definition of Done
- [x] RED: failing gate test — git-evidenced `7d637fd`; the per-language floor assertion FAILS against the BARE inner `SwarmFusionStrategy` (`recall_ensemble[es]=0.0 < recall_shared[es]=1.0 − ε`)
- [x] `RECALL_FLOOR_EPSILON = 0.005` (module-local — composite.py untouched); per-language assertion over `en`/`es`/`zh`; `@requires_dataset` integration test skips gracefully
- [x] runs in CI default `pytest -m "not performance"` step (auto-collected; no new workflow/Make target); ruff clean (`src tests`); mypy --strict clean (`src/pii_anon`, 113 files); full suite green
- [ ] Story-gate review APPROVE (`_reviews/story/S1-03-gate.yaml`) — awaiting orchestrator reviewer dispatch

### Evidence (commit hashes — branch `pdlc/sota-program`)
- **RED** `7d637fd` — `test: S1-03 RED — pin FR-007/NFR-011 per-language recall-floor ε-gate (ε≤0.005)` (2 files: this story + `tests/test_recall_floor_per_language_gate.py`). Ensemble computed from the BARE `SwarmFusionStrategy` (pre-S1-02 state): per-language recalls `{'en':1.0,'es':0.0,'zh':1.0}` → the `es` floor assertion FAILS (`0.0 ≥ 1.0 − 0.005` is false). Teeth assertion passes, `@requires_dataset` test skips → RED gate satisfied.
- **GREEN** `f1638a5` — `feat: S1-03 GREEN — per-language recall-floor ε-gate green via LIVE floored seam` (1 file: test). Flip the ensemble computation to the LIVE `build_fusion("swarm", weights={}, min_consensus=1)` path (`FloorProjectingFusion`, S1-02 commit `a14888e`). The floor re-injects the swarm-gated `es` span → for every L, `recall_ensemble[L] ≥ recall_shared[L] − 0.005` holds. Targeted: **2 passed, 1 skipped**.
- **REFACTOR**: none required — GREEN code already at refactor quality (factored helpers `_ef`/`_lang_of`/`_to_labeled`/`_recall_per_language`/`_synthetic_multilingual_case`; no dead code; ruff + mypy clean on first pass).
- **What the test asserts**: per-language recall floor for languages `en`/`es`/`zh`. (a) `recall_shared` = per-language recall of the shared layer (`regex-oss` findings); (b) `recall_ensemble` = per-language recall of the floored `build_fusion("swarm").merge(findings)` output. Per-language recall reuses the eval-framework `_aligned_prf` (STRICT) over `LabeledSpan` — the same primitive `fairness_metrics` groups-by-language with — keyed by language recovered from `field_path` (robust to the swarm dropping `.language` on natively-emitted findings). Assertion: ∀L `recall_ensemble[L] ≥ recall_shared[L] − RECALL_FLOOR_EPSILON`.
- **Teeth (regression guard has bite)**: a sibling test proves that against the BARE `SwarmFusionStrategy().merge(...)` (no floor), `recall[es]` drops to 0.0 < floor — documenting the gate would FAIL if S1-02 were reverted. Precondition explicitly asserts `SwarmFusionStrategy().merge([es_span]) == []` (the swarm Layer-4 emission/corroboration gate `swarm.py:654/658-661` drops the uncorroborated `es` SEMANTIC span at conf 0.55, below fast-pass 0.90). Synthetic-shaped values per AX-001 — NO real PII; only span offsets/types feed the metrics.
- **Self-contained vs skip**: the gate test + teeth test ALWAYS run (no external dataset). The OPTIONAL `@requires_dataset` real-multilingual-dataset test SKIPS gracefully when `pii-anon-datasets` is absent (currently skips; marked placeholder for the dataset-runner follow-on).
- **composite.py**: NOT touched. `RECALL_FLOOR_EPSILON` kept module-local in the test per the story's stated preference (avoids any risk to the global `FloorGateConfig` / `evaluate_floor_gates` behaviour).
- **Full suite** (`pytest -m "not performance"`, enforces `--cov-fail-under=84`): exit 0 — **2685 passed, 12 skipped, 0 failed, 0 errors** (delta vs ~2683/11 baseline = +2 passed, +1 skipped, all from S1-03). Total coverage **86.21%** (≥84 gate).
- **Quality gates**: ruff `check src tests` → All checks passed; mypy `--strict` on `src/pii_anon` → Success, no issues (113 files). (Direct `mypy tests/<file>` shows only the project-standard `import-untyped` notes also present on the approved S1-02 sibling test `test_floor_fusion_wiring.py`; tests are out of the configured `packages=["pii_anon"]` mypy scope by design.)
- **Safety**: `src/pii_anon/orchestrator.py` and `tests/test_moe_enhancements.py` (user WIP) were NEVER read-modified, staged, committed, or reverted. Narrow explicit staging only; no `git add -A/./-u`. No benchmark artifacts, README, docs, or other story files touched.
