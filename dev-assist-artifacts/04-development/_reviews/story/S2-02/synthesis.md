# S2-02 Story Gate — Synthesis

| Field | Value |
|---|---|
| Gate | story (+ control-path adversarial close) |
| Scope | **S2-02** — `DistilledTopKGate` (runtime `RoutingGate`) + offline distillation trainer emitting the signed `gate_v1.json` (DC-02 core) |
| Iteration | 1 |
| Date | 2026-06-03 |
| Reviewer set | security-sast (PRIMARY) · code-quality · requirements-coverage · traceability · axiom-compliance |
| **Aggregate verdict** | **APPROVE** (5/5) **+ adversarial close RECLOSE_PASS (0 upheld)** |

## Reviewer verdicts

| Reviewer | Verdict | Findings |
|---|---|---|
| security-sast (PRIMARY) | APPROVE | 0 blocking · 2 OBSERVATION (generic ValueError for fail-closed misconfig; in-process-only direct constructor, out-of-threat-model) |
| code-quality | APPROVE | 0 blocking · 2 OBSERVATION (comment-precision nit; documented `Any` key_ring annotation) |
| requirements-coverage | APPROVE | 0 blocking · 2 OBSERVATION (NFR-026 SHOULD label; successor story files unauthored — epic-gate concern) |
| traceability | APPROVE | 1 MINOR (matrix backfill — DEFERRED-to-sprint-gate) |
| axiom-compliance | APPROVE | 0 blocking · 1 OBSERVATION (future RouteContext-conditioning must re-pin A6 determinism) |

**Aggregation:** 0 SHOWSTOPPER · 0 CATASTROPHIC · 0 MAJOR · 1 MINOR (deferred) · 7 OBSERVATION → all 5 APPROVE ⇒ **APPROVE**. No in-loop remediation required.

## Cross-reviewer joint signals
- **Control-path artifact fail-closed, verified independently (security-sast + axiom-compliance + traceability):** the ONLY file-reading construction path is `load_distilled_gate` → `load_verified_gate(verify_on_load=True)`; the production v2 wiring `_build_distilled_moe` verifies + fails closed on gate_path-without-key_ring; the AST no-unverified-construction guard is non-vacuous (security-sast ran an independent mutation probe — catches `=False`, non-True, omitted-kwarg). The S2-01 entry-gate obligation is discharged.
- **A8 real-XGBoost arm RAN (code-quality + requirements-coverage):** the `importorskip("xgboost")` arm executed against xgboost 3.2.0 (real `train()` + distill + sign + load + re-weight), not skipped — the frozen meta-learner plugs the `SurvivalOracle` Protocol.

## Control-path adversarial close (Workflow `wf_4664a0cd-f3a`)
`gate_v1.json` is a control-path/privilege-escalation artifact (the reason S2-05 exists). Per the SO-10/SO-11 standing catch-net methodology (a 5/5 gate is necessary-but-not-sufficient for security-sensitive control-path work), a between-work-streams close ran 3 independent break-probe refuters + an integration/SDO sweep. **Result: RECLOSE_PASS — 0 upheld refutations.**

| Refuter | Attack | Result |
|---|---|---|
| verify-bypass | obtain a usable production gate from a file without a real HMAC verify | **HOLDS** — 1 construction path, both `load_verified_gate` sites verify, no `verify_on_load=False` in production; file-read primitive unconditionally HMAC-gated |
| forge/fabricate | make a malformed/forged gate be accepted, or leak a bad weight / invented expert | **HOLDS** — 35+ malformed-but-signed payloads rejected by `from_payload` (proved the validator, not the signature, is the catch-net); phantom expert ignored; every route output finite/≥0/Σ=1.0/ids⊆base; v2 seam raises at construction on a NaN gate |
| floor-defeat (AX-003) | make an advisory gate drop a floored shared span, or break determinism | **HOLDS** — 19 hostile variants + 400-trial fuzz on the REAL `FloorProjectingFusion`; floored span survived every time; V16 proof: even with the inner floor removed + gate zeroing regex-oss, `SharedLayerProjector` re-injects as sole-survivor authority |

Sweep: suite **3291 passed / 0 failed**; SDO **NOT_YET** / `canonical_claim_run=False`; the 5 consumed primitives byte-identical; ruff+mypy clean; 0 new issues. The 2 refuter-self-identified "false alarms" were correctly adjudicated as non-defects.

**Methodology:** a clean close (0 upheld) does not mean it was unnecessary — it independently confirmed the verify-on-load entry-gate, the no-fabrication validator, and the advisory-floor bound under live attack. The catch-net stays standing for any future control-path-artifact change.

## Deferred (per established convention, non-blocking)
- traceability MINOR-1 — traceability-matrix S2-02 Story/Test backfill → S2 sprint gate (mirrors S2-05/S4-01/S2-01).

## Outcome
**S2-02 → DONE.** The DC-02 MoE-router **core is LIVE**: the runtime `DistilledTopKGate` (the first concrete S2-01 `RoutingGate`) enters the control path ONLY through the S2-05 fail-closed verify-on-load boundary (AX-006); advisory per-(entity_type,expert) re-weighting that can never drop a floored shared span (AX-003, adversarially confirmed); absent/unverifiable/unknown ⇒ static `entity_strengths` softmax (NFR-026). The offline producer distills the frozen XGBoost survival oracle (via a `SurvivalOracle` Protocol) into a compact gate using RRS up-weighting + the closed-form BCE/KL-optimal compact student + temperature-softmax, stamps `oracle_hash` + `gate_feature_version`, and signs `gate_v1.json` via the S2-05 `KeyRing`. Deterministic (NFR-005); xgboost optional/lazy. Suite 3291 pass / 0 fail, cov 87.34%, ruff + BOTH-mypy clean. SDO verdict unchanged (NOT_YET).
