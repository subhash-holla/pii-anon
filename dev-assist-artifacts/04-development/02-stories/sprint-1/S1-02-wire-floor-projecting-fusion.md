# S1-02 — Wire SharedLayerProjector live (FloorProjectingFusion at the build_fusion seam)

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01) |
| State | **IN_PROGRESS** (owner: dev-assist-development-executor; claimed_at 2026-05-30; started_at 2026-05-30) |
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

## 12. Definition of Done
- [ ] RED commit precedes GREEN (git-evidenced); failing build_fusion-level tests for both modes
- [ ] `FloorProjectingFusion` implemented; build_fusion wraps swarm+moe; lazy import (no cycle)
- [ ] `strategy_id` passthrough; both modes floor-enforced; empty-shared no-op; deterministic
- [ ] ruff + mypy --strict clean; no public-API change; full suite green (esp. the 78)
- [ ] Story-gate review APPROVE (`_reviews/story/S1-02-gate.yaml`)
