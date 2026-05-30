# Recall-floor by construction (`SharedLayerProjector`)

> Status: **shipped (foundation)**. New in the PDLC SOTA program (branch `pdlc/sota-program`). Module: `pii_anon.routing.shared_layer`.

## What it guarantees

The swarm's value is **recall** — catching sensitive spans. `SharedLayerProjector` makes that a *structural* guarantee, not a statistical hope:

```
entities(output) ⊇ entities(shared)
```

The **shared layer** is the always-on, high-precision, checksum/keyword-gated regex pass (`regex-oss`). Any span it finds is guaranteed to survive into the final output — even if a downstream emission threshold or semantic-type corroboration gate would otherwise have dropped it. This closes a real leak where a sub-fast-pass regex hit entering fusion could be suppressed by the Layer-4 gate.

The floor is a **projection**, decoupled from any routing/early-exit policy, so it holds for any present or future router.

## Usage

```python
from pii_anon.routing.shared_layer import SharedLayerProjector

projector = SharedLayerProjector()              # shared_engine_id="regex-oss"
result = projector.project(fused_findings, shared_findings)
result.findings            # floor-guaranteed superset of shared spans
result.violations_blocked  # how many dropped shared spans were re-injected (audit signal)
```

- Re-injected findings are tagged `provenance=shared_floor` in `explanation`; detect with `is_shared_floor(finding)`.
- The match key is **type-carrying** — `(field_path, span_start, span_end, entity_type, language)` — so an NER relabel of the same offsets to a different type does **not** count as covering a shared-layer span.
- **Graceful degradation:** an empty `shared` list makes `project()` a no-op (byte-identical to not using it).
- **Deterministic:** stable ordering (original output first, then re-injected in input order).

## Guarantees & verification

| Property | How |
|---|---|
| `output ⊇ shared` (0 violations) | property test, 2,000 seeded cases (`tests/test_shared_layer_projector.py`) |
| Layer-4 emission/corroboration leak closed | `test_ax_003_closes_swarm_layer4_emission_leak` |
| Type-carrying superset | `test_type_carrying_relabel_does_not_cover_shared` |
| Determinism | `test_determinism_repeatable` |

Type-checked (`mypy --strict`), lint-clean (`ruff`), no public-API change, existing swarm/fusion/moe suites unaffected.

## Roadmap (this is the foundation)

The projector ships as a standalone, tested module. **Production wiring** — delegating to it post-`merge()` from both `MoEFusionStrategy` and `SwarmFusionStrategy`, plus the per-language recall-floor CI ε-gate — is the immediate next step (story S1-02/S1-03). It is the load-bearing piece of the broader MoE-router redesign (learned routing, early-exit, SLA balancing) tracked in `dev-assist-artifacts/04-development/development-log.md`.

> **Benchmark caveat:** the published headline numbers in the README remain a 50-sample smoke run (`canonical_claim_run=False`) and are **provisional** pending a regenerated canonical run + the significance-pipeline repair. Don't cite them as certified.
