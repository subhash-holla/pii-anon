# S2-01 Story Gate — Synthesis

| Field | Value |
|---|---|
| Gate | story |
| Scope | **S2-01** — widen `MoERouter.route()` to feature-conditioned routing + additive richer fusion-construction seam (DC-02 foundation) |
| Iteration | 1 |
| Date | 2026-06-03 |
| Reviewer set | code-quality · security-sast · requirements-coverage · traceability · axiom-compliance (per `select-reviewer-set.sh story` — 4 config default_set + axiom-compliance via gate_overrides) |
| **Aggregate verdict** | **APPROVE** |

## Reviewer verdicts

| Reviewer | Verdict | Findings |
|---|---|---|
| code-quality | APPROVE | 2 MINOR (test-name typo `routingate`; `test_audit_*` omit `fr018_` prefix) |
| security-sast | APPROVE | 0 (1 forward note: S2-02 must route any real `gate_path` back through `load_verified_gate(verify_on_load=True)`) |
| requirements-coverage | APPROVE | 0 blocking · 2 OBSERVATION (NFR-026 SHOULD label; successor stories S2-02/03/04 not yet authored — epic-gate concern) |
| traceability | APPROVE | 1 MINOR (matrix backfill — DEFERRED-to-sprint-gate per S2-05/S4-01 convention) |
| axiom-compliance | APPROVE | 0 blocking · 2 OBSERVATION (v2 `gate_path`/`key_ring` hardwired None — correct for seam-only, forward note for S2-02; A4 survivor assertion loose `or`) |

**Aggregation:** 0 SHOWSTOPPER · 0 CATASTROPHIC · 0 MAJOR · 4 MINOR · 6 OBSERVATION → all 5 verdicts APPROVE ⇒ **APPROVE**.

## Cross-reviewer joint signals
- **Honest seam-scoping of FR-018 (raised by requirements-coverage + traceability + the story itself):** FR-018 is a MUST bundling learned-routing + dedup + selective-activation + early-exit. This story ships ONLY the seam and explicitly defers the gate body→S2-02, early-exit→S2-03, SLA-bias→S2-04. Both reviewers independently confirmed the partial coverage is honestly scoped (no over-claim; the traceability-matrix carries no "done" marker for FR-018).
- **The v2 seam's `gate_path`/`key_ring` are inert-by-design (raised by security-sast + axiom-compliance):** `build_fusion` assembles the `FusionBuildSpec` with `gate_path=None`/`key_ring=None`; the v2 path never loads an unsigned gate. Both flagged the same forward-obligation: when S2-02 wires a real `DistilledTopKGate`, its gate review MUST confirm any real `gate_path` is routed back through the S2-05 fail-closed `load_verified_gate(verify_on_load=True)` boundary. **Recorded as an S2-02 entry gate.**
- **A4 (AX-003 floor-survival) is non-vacuous (axiom-compliance empirically falsified the alternative):** without the floor wrapper the hostile gate that zeroes `regex-oss` drops the lone shared span (0 survivors on the bare inner merge); only `FloorProjectingFusion` re-injection restores it.

## Remediation (in-loop, before DONE)
Per the discipline (remediate substantive MINOR + ALL MAJOR in-loop; defer batch items per established convention):

| Finding | Severity | Disposition |
|---|---|---|
| code-quality MINOR-1 — `routingate` test-name typo | MINOR | **REMEDIATED** (commit `620c73d`): renamed `test_fr018_routingate_*` → `test_fr018_routing_gate_*`. |
| axiom-compliance OBS-2 — A4 survivor assertion loose `or` | OBSERVATION | **REMEDIATED** (commit `620c73d`): dropped the loose `regex-oss in engines` fallback; A4 now asserts the surviving span carries the `shared_floor` provenance marker specifically — pins the SharedLayerProjector re-injection MECHANISM (AX-003 is the program's load-bearing T1 invariant, worth pinning exactly). |
| code-quality MINOR-2 — `test_audit_*` omit `fr018_` prefix | MINOR | **DEFERRED-to-S2-sprint-gate** (the recurring test-naming batch-cleanup item, accepted at the S2-05/S4-01/S5-04 story gates as a faithful AUDIT-keyed convention). |
| traceability MINOR-1 — traceability-matrix Story/Test backfill | MINOR | **DEFERRED-to-S2-sprint-gate** (matrix defers DC/Story/Test to Stages 3-5; mirrors S2-05/S4-01/S5-04/S4-CS-01). |
| requirements-coverage OBS (NFR-026 SHOULD label; successors unauthored) · axiom-compliance OBS-1 (gate_path None) · security forward note | OBSERVATION | **ACCEPTED** — non-blocking; the forward notes become S2-02's entry gate. |

Post-remediation: 34/34 owned tests green; ruff clean.

## Outcome
**S2-01 → DONE.** The DC-02 MoE-router seam is LIVE: the feature-conditioned `route(entity_type, *, context=RouteContext|None)` + the advisory `@runtime_checkable RoutingGate` Protocol + the additive v2 `register_fusion_strategy_v2`/`FusionBuildSpec` construction seam + the single-source `_BUILTIN_FUSION_MODES` drift guard — all additive, inert-by-default, byte-for-byte backward-compatible (v1 `register_fusion_strategy` untouched; every existing `route()`/`build_fusion()` result identical pre/post). Closed a real latent `swarm`-mode advertise-drift en route. The seam unblocks **S2-02** (`DistilledTopKGate` implements `RoutingGate`, wired via the v2 seam — with the S2-05 verify-on-load entry-gate obligation) and **S2-04** (SLA bias on the selection logits this `route()` exposes).
