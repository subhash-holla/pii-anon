# NFR Verification Matrix — pii-anon (Stage 5, current state)

> Honest current-state verification. **VERIFIED** = passing test/evidence in-tree now. **DEFERRED** = code not yet implemented (successor story named). **0 FAIL.** All thresholds AGENT_SIMULATED (Pass-2 for latency/Tier-3). Benchmark numbers PROVISIONAL (50-sample smoke; canonical run pending S7).

| NFR | Threshold | Status | Evidence / successor |
|---|---|---|---|
| NFR-011 | recall-floor: entities(ensemble) ⊇ entities(shared), 0 violations; per-lang ε≤0.005 | **VERIFIED (by-construction half)** | `routing/shared_layer.py` + property test (2,000 cases, 0 violations), commits ef85166→548f576, 7/7 green; per-language ε-gate → S1-03 |
| NFR-024 | no real PII in repo/fixtures/logs (AX-001) | **VERIFIED** | new fixtures synthetic-shaped only; security-sast story gate APPROVE; full repo scan → S-sec |
| NFR-005 | byte-identical given seed/key/scope | **PARTIAL** | projector determinism test passes; full-pipeline determinism → S2/S3 |
| NFR-026 | optional-dep graceful degradation | **PARTIAL** | projector empty-shared no-op; gate-absent fallback designed → S2-02 |
| NFR-001 | Bradley-Terry R̂≤1.01 ∧ ESS≥400 ∧ 0 div | DEFERRED | S3-03 (bayes-bt); ⛓ eval-data S6 |
| NFR-002 | significance coherence (CIs bracket; sign↔verdict) | DEFERRED | S3-04 (by-construction from joint posterior) |
| NFR-003 | bootstrap CI coverage 93–97% | DEFERRED | S3-02 (MLE tier) |
| NFR-004 | power tiers 1522/753/200 | DEFERRED | S1-03 + eval-data power ladder (exists) |
| NFR-006 | canonical-run provenance 100% | DEFERRED | S4-02 (CanonicalRunGate) |
| NFR-007/008/009/010 | latency (honest tiers) + throughput | NFR-009 VERIFIED-in-tree (S7-04: committed p50/p95/p99 ceilings `latency_ceilings.py` pinned by `test_latency_ceilings.py`; gate `_g5_audit_latency` + producer measured-latency LIVE; full-census latency = Pass-2); NFR-007/008/010 DEFERRED | S2/S7; per-detector-class p50/p95/p99 = Pass-2 (the `detector_class` seam) |
| NFR-012 | Tier-3 RRS powered ≥385 | DEFERRED | S5-02; ⛓ eval-data S6 `assemble_paired_set` |
| NFR-013 | MIA LiRA@128 + TPR@low-FPR | DEFERRED | S5-03; ⛓ eval-data S6 canary |
| NFR-014/015 | pseudonymization integrity + key/state separation | DEFERRED | S4-01/S6-03 |
| NFR-016 | non-strippable re-id caveat 100% | DEFERRED | S4-01 |
| NFR-017/018/019/020/021 | calibration ECE/Brier/AURC + abstention | DEFERRED | S4-03 |
| NFR-022/023 | OS-matrix + stream/batch/offline parity (divergence=0) | DEFERRED | S7-02 |
| NFR-025 | multilingual worst-group fairness gap ≤0.10 | DEFERRED | S7-03 |

**Accessibility audit (T4): N/A** — product surface is a Python API + Typer/Rich CLI; no web/GUI. CLI output-format + exit-code correctness is the analog (existing `tests/test_cli*.py` + CI smoke).

**Summary:** 2 VERIFIED + 2 PARTIAL + 22 DEFERRED + 0 FAIL. The single load-bearing MUST this redesign is gated on (NFR-011 recall-floor) is verified by construction.
