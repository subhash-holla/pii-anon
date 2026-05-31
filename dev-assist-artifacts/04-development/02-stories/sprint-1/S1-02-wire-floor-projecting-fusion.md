# S1-02 — Wire SharedLayerProjector live (FloorProjectingFusion at the build_fusion seam)

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01) |
| State | **DONE** (gate APPROVE 2026-05-31; `_reviews/story/S1-02-gate.yaml`; swarm-language MAJOR remediated by S1-05) |
| Implements | FR-016, NFR-011, AX-003 (now **LIVE** on the fusion path); touches NFR-005 (determinism), FR-008 (audit) |
| Traces | UC-13 → PGO-4; Design D-SWARM DECISION 1 + D-AGENTIC DECISION 3 (floor wrapped at the `build_fusion` seam) |
| Test-type tags | `[UNIT-TEST]` `[INTEGRATION-TEST]` |
| Files owned | `src/pii_anon/routing/floor_fusion.py` (new), `src/pii_anon/routing/__init__.py`, `src/pii_anon/fusion.py` (build_fusion seam), `tests/test_floor_fusion_wiring.py` (new) |
| Depends on | S1-01 (DONE) |
| Size | M |

## 1. Intent
Make the recall-floor **LIVE on the production fusion path**: both `MoEFusionStrategy` and `SwarmFusionStrategy` (as built by `build_fusion`) must delegate to `SharedLayerProjector.project()` post-`merge()`, so a shared-layer (`regex-oss`) span that the swarm Layer-4 emission/corroboration gate (`swarm.py:654/658-661`) or MoE weighting would drop is re-injected. S1-01 proved the floor by-construction as a standalone module; S1-02 wires it into the seam so it actually runs in production.

## 2. Given/When/Then (acceptance)
- **Given** `build_fusion("swarm", ...)`, **when** `.merge(findings)` runs on findings containing a `regex-oss` span that the swarm emission gate would drop, **then** that span is present in the output tagged `provenance=shared_floor`.
- **Given** `build_fusion("mixture_of_experts", ...)`, **when** `.merge(findings)` runs, **then** every `regex-oss` shared span is present in the output (re-injected if the inner strategy dropped it).
- **Given** any wrapped mode, **then** `strategy.strategy_id` is unchanged (delegates to the inner strategy) — preserves `tests/test_swarm.py:295`.
- **Given** findings with NO `regex-oss` spans, **then** output is byte-identical to the inner strategy's merge (no-op floor).
- **Given** fixed input, **then** merge output is deterministic (stable ordering, NFR-005).

## 3. Design ref / approach (VALIDATED by Plan agent)
- **Seam = Option A**: a `FloorProjectingFusion(FusionStrategy)` wrapper holding `inner: FusionStrategy`, a `SharedLayerProjector`, and `shared_engine_id="regex-oss"`.
  `merge(findings)` = `self._projector.project(self._inner.merge(findings), [f for f in findings if f.engine_id == shared_engine_id]).findings`.
- Wrap ONLY `swarm` + `mixture_of_experts` returns in `build_fusion` (`fusion.py:500-510`) via a **LAZY import** (`from pii_anon.routing.floor_fusion import FloorProjectingFusion`) to avoid the fusion↔routing import cycle (mirror the existing lazy `from pii_anon.moe import MoEFusionStrategy` at fusion.py:501).
- Wrapper MUST: subclass `FusionStrategy`; set `self.strategy_id = inner.strategy_id` (instance attr shadows class attr); add `__getattr__(self, name) -> Any` passthrough to `inner` (annotate `-> Any` for mypy --strict); preserve determinism.
- **Shared-source guarantee**: derive `shared` from the `regex-oss` subset of the SAME `findings` passed to `merge` — DO NOT run regex twice (determinism + speed).

## 4. Risks / non-gaps (documented, NOT to "fix" here)
- A profile that DISABLES `regex-oss` makes `shared==[]` → floor no-ops silently. Default path always enables `regex-oss` (`orchestrator.py:242`; hard fallback `policy/router.py:154`). Registry-level always-on enforcement = follow-on, not S1-02.
- The single-engine speed fast-path (`orchestrator.py:692-706`) bypasses fusion; it does NO gating so it cannot violate the floor (the single engine's spans pass straight through). Floor applies wherever fusion runs. Document, don't change.
- FR-008 audit: re-injected findings already carry `provenance=shared_floor` + `source_count=0` via `build_fusion_audit`; an explicit `floor_violation_blocked` audit note is a clean follow-on (not S1-02).

## 5. At-risk tests to keep green
- `tests/test_swarm.py:295` (`build_fusion("swarm").strategy_id == "swarm"`) — fixed by `strategy_id` passthrough. **Only known breaker.**
- `tests/test_fusion.py:172` isinstance on `weighted_consensus` — unaffected (only swarm/moe wrapped).
- `tests/test_moe.py:659-660` — direct constructor, unaffected.
- Full suite (~2,555) + the 78 swarm/fusion/moe must stay green.

## 6. RED → GREEN → REFACTOR
- **RED**: `tests/test_floor_fusion_wiring.py` — build_fusion-level tests for BOTH modes: a `regex-oss` `EngineFinding` (semantic type, conf in [0.50,0.90)) that the swarm gate drops must appear in `.merge(...)` output as `is_shared_floor`; `strategy_id` passthrough; empty-shared no-op equals inner.merge; determinism. Fails: `FloorProjectingFusion` absent + build_fusion returns bare strategies.
- **GREEN**: add `floor_fusion.py`; wrap swarm+moe in build_fusion (lazy import); re-export from `routing/__init__.py`.
- **REFACTOR**: extract the `regex-oss` shared-extraction helper; mypy/ruff clean.

### Evidence (commit hashes — branch `pdlc/sota-program`)
- **RED** `4760657` — `test: S1-02 RED — pin FR-016/NFR-011/AX-003/NFR-005 floor LIVE at build_fusion seam` (2 files: story + `tests/test_floor_fusion_wiring.py`). 18 tests collected, 8 failed (FR-016/AX-003 superset + `_inner` absent) / 10 passed → RED gate satisfied (the load-bearing floor invariants fail because `FloorProjectingFusion` is absent and `build_fusion` returns bare strategies).
- **GREEN** `a14888e` — `feat: S1-02 GREEN — FloorProjectingFusion wraps swarm+moe at build_fusion seam` (4 files: `routing/floor_fusion.py` (new), `fusion.py`, `routing/__init__.py`, test). All wiring tests pass; 104 targeted tests green; ruff + mypy --strict clean.
- **REFACTOR** `fa36891` — `refactor: S1-02 — cover FloorProjectingFusion.__getattr__ guard branch (100%)` (1 file, test-only). `_shared_subset` helper already factored out in GREEN; added 2 tests exercising the `__getattr__` underscore-guard + missing-public-attr forward paths → `floor_fusion.py` at 100% line+branch coverage. 108 targeted tests green.
- **Full suite** (`pytest -m "not performance"`, enforces `--cov-fail-under=84`): exit 0 — **2679 passed, 11 skipped, 0 failed, 0 errors**; total coverage **86.15%** (≥84 gate), `floor_fusion.py` 100%, `shared_layer.py` 100%, `fusion.py` 97%. No new failures vs. the ~2671-test baseline (delta = +23 new wiring tests). (Run twice: first run 86.12% pre-REFACTOR, final 86.15% with the 2 guard-branch tests.)
- **At-risk tests verified GREEN**: `test_swarm.py::test_build_fusion_factory` (`strategy_id == "swarm"`), `test_fusion.py` weighted_consensus isinstance, `test_moe.py::test_moe_strategy_id`.
- **Implementation notes**: shared set derived from the `regex-oss` subset of the SAME `findings` (regex never re-run → deterministic, NFR-005). `routing/__init__.py` re-exports `FloorProjectingFusion` via PEP-562 module `__getattr__` (lazy) to keep the fusion↔routing cycle broken. Swarm inner provably DROPS the gated span (verified `SwarmFusionStrategy().merge([gated]) == []`); MoE keeps it natively → floor no-ops for MoE on that input but the superset guarantee still holds for both modes.
- §4 risks (regex-oss-disabled profile no-op; single-engine fast-path bypass; explicit `floor_violation_blocked` audit note) DOCUMENTED as follow-ons, NOT changed in S1-02 per story scope.

## 12. Definition of Done
- [x] RED commit precedes GREEN (git-evidenced `4760657` → `a14888e`); failing build_fusion-level tests for both modes
- [x] `FloorProjectingFusion` implemented; build_fusion wraps swarm+moe; lazy import (no cycle — verified `import pii_anon.fusion; import pii_anon.routing` succeeds)
- [x] `strategy_id` passthrough; both modes floor-enforced; empty-shared no-op; deterministic
- [x] ruff + mypy --strict clean; no public-API change; full suite green (esp. the 78 swarm/fusion/moe; targeted run 108 green, full suite exit 0 @ 86.12% cov)
- [ ] Story-gate review APPROVE (`_reviews/story/S1-02-gate.yaml`) — awaiting orchestrator reviewer dispatch
