# S7-04 — NFR-009 committed latency ceilings + the SDO gate `_g5_audit_latency` wiring + producer G5 emission

| Field | Value |
|---|---|
| Story | S7-04 |
| Sprint | 7 |
| State | **TODO** (authored 2026-06-09; SO-15 `next.immediate`) |
| Size | M |
| Implements | **NFR-009** (full-swarm latency per profile — p50 ≤ declared per-profile budget, committed NUMERIC ceiling + p99; NOT sub-0.24ms parity; the R10 PERSONA-CONDITIONAL re-scope: "commit a numeric swarm ceiling + add p99 + per-detector-class p50/p95/p99"). Makes **G5** (audit + orchestration latency / interception) code-compute a REAL value on a certified run — the LAST `_pending_guarantee` placeholder, modeled EXACTLY on how G2 (S4-01) and G4 (S4-03) went placeholder → computed. Consumes **FR-025** (4-channel least-privilege interception — owned/verified by S6-02; re-asserted here by the gate's 4-channel-completeness check), **FR-026** (no-raw-PII-persist), **FR-028** (leakage-Sankey; the "6 channels" wording is reconciled at the S6-05 source-of-record as a 6-NODE graph = 4 source channels + 2 sinks — the gate keys on the 4 source channels), **FR-029** (injection ASR vs benign-task-success) as the G5 audit half. Upholds **AX-001** (synthetic-only), **AX-002** (determinism — the audit half is keyed-deterministic; wall-clock latency is the SANCTIONED non-reproducible measurement per NFR-005), **AX-006** (least-privilege interception). |
| Traces | Design **DC-11** (the canonical-run / gate seam this completes). The S4-CS-01 SDO story §3 G5 definition: "p50/p95/p99 within the committed latency budgets ∧ auditable 4-channel least-privilege interception present". The SO-14 G5-audit features (S6-02 `InterceptionLedger`, S6-05 `LeakageSankey` + `score_injection_resistance`) — **consumed read-only**; this story wires their outputs into the artifact + the gate. SO-15 `findings_waived_or_deferred.g5-code-computable` + `next.immediate`. |
| Files owned | `src/pii_anon/eval_framework/evaluation/latency_ceilings.py` (**new** — the committed NFR-009 ceiling registry; stdlib-only imports). `tests/test_latency_ceilings.py` (**new**). `src/pii_anon/eval_framework/evaluation/competitive_supremacy.py` (**CONTROL-PATH** — replace the G5 `_pending_guarantee` placeholder with `_g5_audit_latency(systems, run_metadata)`; ANY change here ⇒ the MANDATORY adversarial close). `src/pii_anon/evaluation/canonical_run.py` (**CONTROL-PATH producer** — emit the G5 latency + audit blocks; extend `CanonicalRunGate.validate`). **Additive** to `tests/test_competitive_supremacy.py` + `tests/test_canonical_run.py` + `tests/test_rating_import_boundary.py` (extend the AST-scanned gate-module set with `latency_ceilings.py`). |
| Depends on | S7-02 (the canonical-run producer — DONE, SO-15); S6-02 (`agentic/interception.py` — DONE); S6-05 (`agentic/leakage_sankey.py` — DONE); S4-CS-01 G5 definition. CONSUMES read-only: `src/pii_anon/evaluation/competitor_compare.py` (the RISK-6 SIBLING `evaluation/` pkg, NOT `eval_framework/`; `_ensemble_detector` — the full-swarm detect path; **OFF-LIMITS byte-identical** md5 `7cae16c89f4c97136e1a12394dae2025`), `src/pii_anon/eval_framework/evaluation/competitor_tiers.py` (byte-identical `d9202479d84c519d5866b7d9d515c903`). |

## 1. Intent

Close the last PENDING-placeholder guarantee. G5 = **audit-pass ∧ latency-within-ceiling**, two halves:

* **Latency half (NFR-009):** there is NO latency-ceiling infrastructure in src today (grep-clean). Commit a small numeric ceiling registry (decision **(a)** below), measure the REAL full-swarm per-record p50/p95/p99 in the producer, and gate measured ≤ committed.
* **Audit half (FR-026/028/029):** the S5/S6 audit surface is LIVE but writes no benchmark artifact. The producer runs the REAL `FourChannelGuard` + `build_leakage_sankey` + `score_injection_resistance` (synthetic inputs, keyed-deterministic) and emits their summaries; the gate validates 4-channel completeness, no-raw-PII-persist, zero leaks, and the ASR/benign bars.

**This will NOT flip the verdict** — G6 honestly FAILS and binds (decision A). G5 is completeness, not a flip. The smoke artifact (no G5 fields) stays honestly PENDING.

## 2. Approach / scope — the two carried DESIGN decisions

### (a) The latency-ceiling SCHEMA (NFR-009)

New module `eval_framework/evaluation/latency_ceilings.py` (stdlib-only; added to the import-boundary AST scan set):

* `LatencyCeiling` — frozen dataclass `{profile, p50_ms, p95_ms, p99_ms, detector_class="full-swarm"}`. The `detector_class` field is the **Pass-2 seam** for the R10 "per-detector-class p50/p95/p99" refinement: a future per-engine-class commitment extends the registry without schema change (BUILD-AND-FLAG).
* `COMMITTED_LATENCY_CEILINGS` — keyed by the benchmark **objective** (the engine-selection axis that determines the latency class: `speed` / `balanced` / `accuracy` / `ensemble`). NFR-007's "speed profiles / accuracy profiles" maps to this axis; the **full swarm** (NFR-009's subject) runs as `objective="ensemble"` (`competitor_compare.py:2175`). Committed numbers, grounded in measurement (census swarm p50 = 91.5 ms; this-env fresh quiet measure p50 80.5 / p95 112 / p99 133 ms at n=12; ★ a SATURATED host — 10-worker xdist — tripled a single-shot p50 to ~275 ms, which drove BOTH the min-of-3 producer estimator and the contention headroom below):
  * `speed`: p50 ≤ **1.0**, p95 ≤ 5.0, p99 ≤ 10.0 ms (the NFR-007 literal; core regex path measures ~0.46 ms)
  * `balanced`: p50 ≤ 50.0, p95 ≤ 150.0, p99 ≤ 300.0 ms (declared budget; non-ensemble objectives are regex-backed today)
  * `accuracy`: p50 ≤ 250.0, p95 ≤ 500.0, p99 ≤ 1000.0 ms (declared budget for heavier configs)
  * `ensemble` (the FULL SWARM — NFR-009's row): p50 ≤ **500.0**, p95 ≤ **1000.0**, p99 ≤ **2000.0** ms — ~6x quiet headroom + contention headroom; a REAL sub-2s-p99 commitment, NOT sub-0.24ms parity and NOT vacuous. The producer measures **min-of-3 per record** (timeit-style — a single-shot timing on a loaded host measures the scheduler, not the swarm).
* `ceiling_for(profile) -> LatencyCeiling | None` — the lookup the gate + producer share.

### (b) The G5 audit-field CONTRACT (what the gate reads)

Emitted by the producer under `run_metadata` (absent on the smoke artifact ⇒ G5 PENDING, honest):

```
run_metadata.latency_summary = {
  system: "pii-anon-swarm",          # MUST be a ladder member (a non-SUT measure cannot certify)
  profile: "ensemble",               # MUST be in the committed registry
  p50_ms / p95_ms / p99_ms: float,   # measured; finite, ≥ 0, ordered p50 ≤ p95 ≤ p99
  n_records: int,                    # the timing sample size (≥ 1; non-bool int)
  measurement: <transparency label>
}
run_metadata.latency_ceiling_ms = {p50_ms, p95_ms, p99_ms}   # OPTIONAL echo; TIGHTEN-ONLY
run_metadata.audit_summary = {
  interception: { counts_by_channel: {PROMPT, MEMORY, TOOL_IO, TRACE: int ≥ 1 each},
                  no_raw_pii_persist: true,      # strict `is True`
                  records_total: int },
  leakage_sankey: { blocked: int ≥ 0, leaked: int == 0 },
  injection_resistance: { attack_success_rate: ≤ 0.0,        # the committed ASR bar
                          benign_task_success_rate: ≥ 0.95,  # anti-degenerate (FR-029 pairing)
                          n_payloads: int ≥ 1 }
}
```

**Gate** (`_g5_audit_latency(systems, run_metadata)`, replacing `_pending_guarantee("G5", ...)` at the seam; the `overrides["G5"]` successor-override is preserved EXACTLY like g2/g4):

* Three-valued. **MISSING-SHAPE** (block absent / non-dict / a required key absent) ⇒ **PENDING** — never fabricated (G2's "a half-populated artifact cannot fabricate the missing half" precedent; the smoke artifact path). **PRESENT-BUT-INVALID value** ⇒ **FAIL** (a corrupt present measurement cannot certify — the G1-corrupt-ε / G4-corrupt-ECE precedent).
* EVERY artifact value routes through a validator: latency ms via `_is_finite_number` ∧ ≥ 0 ∧ percentile-ordered (a p50 > p95 is non-physical ⇒ corrupt); rates via `_finite_unit_score`; counts via non-bool-int guards; `profile`/`system`/`measurement` stamps via `_is_nonblank_str` + membership; `no_raw_pii_persist` via strict `is True` (the canonical_claim_run coercion lesson). NESTED values are first-class (the close-9 lesson — the audit blocks are nested dicts).
* The effective latency bar per percentile = `min(committed, artifact-supplied)` — the artifact ceiling may only TIGHTEN (`_g4_class_bar` pattern); an invalid artifact ceiling value is REJECTED (committed stands).
* The committed ASR bar (0.0) + benign bar (0.95) are gate-owned constants — NOT artifact-overridable.
* PASS ⟺ latency-within-ceiling ∧ audit-clean. Detail strings name the failing axis + worst slack. Never a crash on hostile shapes.

**Producer:** a dedicated per-record timing loop over the SAME sampled records via the off-limits-imported `_ensemble_detector` (the exact full-swarm path `compare_competitors` benchmarks; 2-record warmup; nearest-rank percentiles; n stamped honestly) + the REAL audit surface run with a seed-derived fixed surrogate key (deterministic given seed — FR-030). Two separate guards (interception/Sankey vs injection-scoring) so channel counts stay semantically clean. Writes ONLY `artifacts/canonical/`.

**`CanonicalRunGate.validate` extension (step 7):** a certified run REQUIRES the G5 fields present + shape-valid + **audit-integrity bars** (persist `is True`, leaked == 0, ASR ≤ 0.0, benign ≥ 0.95 — an audit breach means the artifact itself is untrustworthy, like the step-3 ε bound). The **latency ceiling comparison stays the SDO gate's job** — over-budget latency is an HONEST measured outcome on a still-certified run (like G6's F2), NOT a certification defect. This line is deliberate and documented.

### NFR-005 determinism note (sanctioned change)

`timestamp_utc` was "the LONE non-reproducible field". Wall-clock latency is non-reproducible BY CONSTRUCTION and NFR-005 already excludes wall-clock speed from determinism — the measured `latency_summary` values join `timestamp_utc` in the determinism-comparison exclusion (the test helper pops both). The audit half stays byte-deterministic (keyed HMAC under a seed-derived key). The gate-read composite is UNTOUCHED (still the fixed reference speed — composite/J byte-identical).

## 3. Given / When / Then (acceptance — named)

Latency registry: ceilings frozen + ordered (p50 ≤ p95 ≤ p99) + all four objectives present + speed.p50 == 1.0 (NFR-007 literal) + stdlib-only imports + boundary-scan extended.
Gate PENDING: both-blocks-absent ⇒ PENDING (smoke unchanged); one-half-absent ⇒ PENDING naming the absent half; PENDING never blocks PROVISIONAL, always blocks CLAIM_GRADE; `overrides["G5"]` (True/False/None) wins over computation.
Gate PASS: a valid in-ceiling artifact ⇒ G5 PASS naming the margins.
Gate FAIL (honest): p99 over committed ceiling; leaked == 1; ASR == 0.25; benign == 0.5; persist False; a missing channel in counts_by_channel.
Gate fail-closed (adversarial): NaN/+inf/-1.0/bool/10**400 in each latency percentile ⇒ FAIL never crash/PASS; p50 > p95 inversion ⇒ FAIL; bool/huge-int channel counts ⇒ FAIL; non-str profile / unknown profile ⇒ FAIL; non-ladder system ⇒ FAIL; loosening artifact ceiling (1e9) clamped to committed; tighten honored; string "true" persist ⇒ FAIL; nested container garbage (lists/dicts where scalars expected) ⇒ FAIL/PENDING never crash.
Producer: produced artifact carries both blocks; latency values real + ordered + n_records == min(sampled count, the 200-record timing cap); audit deterministic across same-seed runs (byte-identical modulo timestamp + latency_summary); G5 computes (not PENDING) on the produced artifact; validate refuses a payload missing G5 fields / breaching audit bars; verdict still NOT_YET binding G6 (G5 PASS does NOT flip — completeness assertion).

## 5. Notes / non-goals

* **Pass-2 flags:** per-detector-class p50/p95/p99 (the schema seam exists via `detector_class`); full-census latency measurement; a live-agent runtime audit (the S6 honesty boundary); real key-rotation audit evidence.
* **Do NOT touch:** `orchestrator.py` (user-WIP), `competitor_compare.py` / `competitor_tiers.py` (off-limits byte-identical), `artifacts/benchmarks/*`. The G6/G3/J read-paths and the composite are UNCHANGED.
* The smoke artifact's G5 PENDING detail string changes shape (the G2/G4 "benchmark lacks the … fields" precedent); the `_PENDING_SUCCESSORS["G5"]` successor string is kept byte-identical (axes_pending stability). Honest-input VERDICTS (verdict + binding constraint) byte-identical on both standing artifacts.

## 9. Test-type tags + reviewer set

`[UNIT-TEST]` `[CONTRACT-TEST]` `[SECURITY-TEST]` `[PROPERTY-TEST]` `[AUDIT]` — reviewers (SDO-critical set): **security-sast PRIMARY** + axiom-compliance + requirements-coverage + traceability + code-quality. **§12: a `competitive_supremacy.py` change ⇒ the MANDATORY adversarial close (bar = 0 upheld, fabrication refuters MUST complete; round-8 template with split flat-scalar + nested-value refuters + the StructuredOutput mandate; confirmatory round after any hardening).**

## 12. Definition of Done

- [ ] RED→GREEN per commit (registry / gate / producer); ruff + BOTH `mypy src/pii_anon` AND `--strict` clean; full xdist suite green, coverage ≥ 84%.
- [ ] Honest-input verdicts byte-identical: smoke ⇒ NOT_YET / canonical_claim_run=False (G7); pre-S7-04 produced shape (no G5 fields) ⇒ G5 PENDING.
- [ ] Fresh produced artifact: G5 computes REAL (expected PASS in-ceiling), verdict NOT_YET binding G6 (completeness, not a flip).
- [ ] Story gate 5/5 APPROVE (MAJORs + substantive MINORs in-loop) THEN the adversarial close 0-upheld with fabrication refuters completed.
- [ ] User-WIP + off-limits md5 unchanged; producer writes only `artifacts/canonical/`; import boundary green.

## Evidence (filled on completion)

_(pending)_
