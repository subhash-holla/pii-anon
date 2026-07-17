# S1-01 — SharedLayerProjector (recall-floor by construction)

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01) |
| State | **DONE** |
| Implements | FR-016, NFR-011, AX-003 |
| Traces | UC-13 → PGO-4; Design D-SWARM (Proposal A spine) |
| Test-type tags | `[UNIT-TEST]` `[PROPERTY-TEST]` |
| Files owned | `src/pii_anon/routing/{__init__,shared_layer}.py`, `tests/test_shared_layer_projector.py` |
| Size | S |

## 1. Intent
Guarantee `entities(output) ⊇ entities(shared)` **by construction** so no downstream gate (emission threshold, semantic-type corroboration) can drop a shared-layer span — closing the verified `swarm.py:654/658-660` leak and unifying the two divergent floor mechanisms.

## 2. Given/When/Then (acceptance)
- **Given** a fused output missing a shared-layer span, **when** `SharedLayerProjector.project(output, shared)` runs, **then** the span is re-injected (type-carrying key, `provenance=shared_floor`) and `violations_blocked` counts it.
- **Given** an empty shared set, **then** project is a no-op (graceful degradation).
- **Given** an NER relabel at the same offsets, **then** the shared type is still re-injected (type-carrying superset).

## 3. Design ref
`03-design/06-synthesis/D-implementation-ready-design.md` §DECISION 1 (DC-01). Floor is a projection decoupled from any router; MoE floor-weight kept as documented defense-in-depth.

## 4–11. (TDD cycle, evidence)
- **RED** `ef85166`: 7 failing tests (`ModuleNotFoundError`).
- **GREEN** `548f576`: `shared_layer.py`; 7/7 green; ruff clean; mypy --strict clean; swarm/fusion/moe (78 tests) unaffected.
- **REFACTOR**: none required (module is minimal + pure).

## 12. Definition of Done ✅
- [x] FR-016/NFR-011/AX-003 pinned by tests (incl. property test, 2000 cases, 0 violations)
- [x] RED commit precedes GREEN commit (git-evidenced)
- [x] ruff + mypy --strict clean; no public-API change; existing suites green
- [x] Story-gate review APPROVE (`_reviews/story/S1-01-gate.yaml`)
- [ ] **Successor S1-02**: wire the projector into `MoEFusionStrategy` + `SwarmFusionStrategy` post-`merge()` so the floor is live in production (TODO — tracked).
