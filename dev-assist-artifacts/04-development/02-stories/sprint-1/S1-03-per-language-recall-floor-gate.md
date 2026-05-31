# S1-03 — Per-language recall-floor CI ε-gate (ε ≤ 0.005)

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01 / DC-11) |
| State | **IN_PROGRESS** (owner: dev-assist-development-executor; claimed_at 2026-05-30; started_at 2026-05-30) |
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
- [ ] RED: failing gate test (no ε constant / no per-language assertion yet)
- [ ] `RECALL_FLOOR_EPSILON = 0.005`; per-language assertion; `@requires_dataset` skip
- [ ] runs in CI default pytest step; ruff + mypy --strict clean; full suite green
- [ ] Story-gate review APPROVE (`_reviews/story/S1-03-gate.yaml`)
