# Release Gate Synthesis — release — v1.5.0-rc1 (iteration 2)

> Canonical gate record per `templates/development/gate-checklist.md.tmpl`.
> Persisted at `dev-assist-artifacts/04-development/_reviews/release/v1.5.0-rc1/synthesis.md`.
> Supersedes the iteration-1 synthesis (REQUEST_CHANGES, 2026-06-10) at this same path; the
> iteration-1 record is preserved in git history and summarized in the Refinement Iteration
> Record below.

**Gate type**: release
**Scope ID**: v1.5.0-rc1
**Date**: 2026-06-10
**Refinement iteration**: 2 of 3 (re-dispatch of the SAME 6-reviewer set per the iteration-1
REQUEST_CHANGES branch; remediation commit under review: `fc31ee6` at HEAD)
**Gate integrity**: **complete** — 6/6 dispatched reviewers reported canonical YAML; missing
reviewers: none.

## Reviewer Set

(Same set as iteration 1, re-dispatched unchanged per the REQUEST_CHANGES rule; release-gate
set = 6 reviewers. Accessibility was not selected for this scope; contributor-experience is
S8-only.)

| Reviewer | Verdict | S/C/MAJ/MIN/OBS |
|---|---|---|
| `dev-assist-development-code-quality` | APPROVE | 0/0/0/0/0 |
| `dev-assist-development-security-sast` | APPROVE | 0/0/0/0/2 |
| `dev-assist-development-requirements-coverage` | APPROVE | 0/0/0/0/0 |
| `dev-assist-development-traceability` | APPROVE | 0/0/0/0/0 |
| `dev-assist-development-performance-benchmark` | APPROVE | 0/0/0/0/3 |
| `dev-assist-development-axiom-compliance` | APPROVE | 0/0/0/0/1 |

**Merged totals**: 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 MAJOR / 0 MINOR / 6 OBSERVATION —
6 findings, all informational.

Iteration-over-iteration: 0/0/1/6/14 (iteration 1) → 0/0/0/0/6 (iteration 2). The iteration-1
MAJOR (TRACE-RC-01, the never-backfilled traceability matrix) is cleared — the traceability
reviewer that carried it now reports APPROVE with zero findings, and the backfilled matrix is
independently referenced as evidence this iteration (AX-RC-03 cites
`traceability-matrix.md` rows FR-011/012/013/017/024 + NFR-011/012/013). All six iteration-1
MINORs are likewise cleared or verified closed (see PERF-RC-06 for the explicit PERF-RC-01
closure verification).

## Aggregation

Per the `dev-assist-development` skill's aggregation rule:

```
if any(severity == SHOWSTOPPER): HALT_GATE
if any(verdict == REQUEST_CHANGES): REQUEST_CHANGES (merged findings)
return APPROVE
```

- Any SHOWSTOPPER? **No** (0 across all six reviewers) → not HALT_GATE.
- Any reviewer verdict REQUEST_CHANGES? **No** — 6/6 APPROVE → fall through.

**Aggregate verdict: APPROVE** (6 APPROVE / 0 REQUEST_CHANGES / 0 HALT_GATE).

## Merged Findings (ordered by severity)

### SHOWSTOPPER (0)

None.

### CATASTROPHIC (0)

None.

### MAJOR (0)

None. (The iteration-1 MAJOR TRACE-RC-01 was remediated by the executor amendment and did not
recur on re-review.)

### MINOR (0)

None. (All six iteration-1 MINORs cleared on re-review; PERF-RC-06 below records the explicit
closure verification for PERF-RC-01.)

### OBSERVATION (6)

#### PERF-RC-04 — uncommitted working-tree benchmark churn is the known harness side-effect; rc-HEAD verdicts byte-stable — `dev-assist-development-performance-benchmark`

- **Location**: `artifacts/benchmarks/*`; `benchmark-diagnostics.json`;
  `docs/benchmark-summary.md` (uncommitted working tree)
- **Finding**: Uncommitted working-tree modifications exist to six benchmark artifacts
  (artifacts/benchmarks/benchmark-results.json, floor-baseline.json, benchmark-raw.csv,
  floor-gate-report.md; benchmark-diagnostics.json; docs/benchmark-summary.md). These are the
  KNOWN, expected non-deterministic side-effects of the benchmark SCRIPT / dataset regeneration
  (record count 140,855 → 148,994; sub-millisecond latency jitter, e.g. pii-anon speed p50
  0.395 → 0.448ms, scrubadub 0.243 → 0.237ms) — NOT a code regression and NOT part of the rc
  candidate (`fc31ee6` touches none of them; the last commit to touch them was `a35b40f` /
  v1.3.0). DECISIVE STABILITY CHECK: all six profile floor_pass verdicts are byte-identical
  between the committed rc HEAD and the working-tree re-run (short_chat=False,
  long_document=True, structured_form_accuracy=True, structured_form_latency=False,
  log_lines=False, multilingual_mix=True). The three speed-profile floor failures are an
  EXISTING, intentional state — the floor-gate compares pii-anon against scrubadub's
  sub-0.25ms regex-only path, which is the explicit NFR-009 carve-out (latency_ceilings.py
  line 30: "deliberately NOT sub-0.24ms regex parity"); they do not breach any committed
  NFR-009 ceiling and are unchanged from iteration 1.
- **Suggested resolution**: Do NOT stage these working-tree benchmark mods into the rc tag —
  the committed rc artifacts at HEAD are the canonical, verdict-stable set. The
  non-deterministic dataset regeneration remains a long-standing harness-reproducibility note
  (informational, no RC action), carried forward from iteration 1.

#### AX-RC-03 — AX-003/AX-005 fully saturated in-tree; corpus-scale verification remains DATA-pillar Pass-2 — `dev-assist-development-axiom-compliance`

- **Location**: `dev-assist-artifacts/02-requirements/traceability-matrix.md` (rows
  FR-011/012/013/017/024, NFR-011/012/013); `tests/test_recall_floor_per_language_gate.py`
  (@requires_dataset); `tests/test_selective_risk.py`
- **Finding**: AX-pii-anon-003 (recall-floor) and AX-pii-anon-005 (calibrated abstention) are
  FULLY saturated in-tree at the unit/property level — the superset-by-construction property
  test, shared-floor re-injection, the per-language epsilon ≤ 0.005 regression gate, per-class
  ECE/Brier/AURC, the 0-bare-logit NFR-020 coverage audit, and the ≥ 3-point abstention
  operating-point table all pass on HEAD `fc31ee6`. The axiom SPIRIT is upheld; the only
  form-lag is corpus-SCALE verification: the per-language epsilon gate is
  @requires_dataset-gated and several adjacent FR-011/012/013 (Tier-3 re-id / MIA) and FR-024
  (query-aware bound) rows are labeled REPRESENTATIVE in-tree with the real LLM-call /
  ≥ 128-shadow / DATA-scorer path explicitly deferred to Pass-2 (DATA-owned). This is correctly
  and honestly disclosed in the backfilled matrix and the unchanged caveat set, and is
  acceptable for a LOCAL-ONLY rc, but the at-scale axiom verification remains open work for
  the pii-anon-eval-data pillar.
- **Suggested resolution**: No action required for this LOCAL-ONLY rc. Track the corpus-scale
  recall-floor and calibration runs as the DATA-pillar Pass-2 items already recorded in the
  matrix; re-confirm AX-003/AX-005 at scale before any claim-grade (non-local) publication or
  the eventual GA tag.

#### PERF-RC-05 — no new performance regression in remediation commit fc31ee6; control-path md5s match attestation — `dev-assist-development-performance-benchmark`

- **Location**: git `fc31ee6` (HEAD); `src/pii_anon/eval_framework/attacks/reid.py`;
  `latency_ceilings.py`
- **Finding**: No new performance regression introduced by the remediation commit `fc31ee6`.
  The only .py file in the commit is src/pii_anon/eval_framework/attacks/reid.py with 3 pure
  docstring additions (ReidAttack.attack, MiaAttack.membership_scores protocol stubs,
  BaselineDeterministicReidAttack.attack) — no logic, no hot-path or latency-relevant code
  change. The NFR-009 ceilings registry latency_ceilings.py (md5
  `848f21bef3731ad56bae3b0b5aa86fee`) is byte-identical and untouched. The 3 control-path md5s
  match attestation exactly: competitive_supremacy `3b842e81c3f03eafd11f9c655c1789a0`,
  canonical_run `d8f0f80e113c3b5d59c06d0b5fd36fac`, competitor_compare
  `7cae16c89f4c97136e1a12394dae2025`. No benchmark/perf artifact is in the commit
  (`git show fc31ee6 --stat`). Committed latency budgets carry ~6x quiet-env margin (census
  full-swarm p50 91.5ms; quiet-env p50/p95/p99 = 80.5/112/133ms vs the 500/1000/2000ms
  ensemble ceiling) plus documented host-contention headroom.
- **Suggested resolution**: None — no regression in this dimension.

#### PERF-RC-06 — iteration-1 PERF-RC-01 (dual NFR-001 numbering) verified CLOSED — `dev-assist-development-performance-benchmark`

- **Location**: `pdlc-artifacts/swarm/requirements/non-functional-requirements.md:15`
- **Finding**: PERF-RC-01 (iteration-1 MINOR, performance dimension) is genuinely CLOSED. The
  dual-NFR-001 numbering supersession note is present at
  pdlc-artifacts/swarm/requirements/non-functional-requirements.md:15, correctly recording
  that the Discovery-era 200ms p50 NFR-001 is superseded by the R10-rescoped, in-tree-verified
  NFR-009 ensemble budget (p50 ≤ 500ms / p95 ≤ 1000ms / p99 ≤ 2000ms, committed in
  src/pii_anon/eval_framework/evaluation/latency_ceilings.py per S7-04/SO-16), and
  disambiguates the eval-pillar scheme (where NFR-001 = Bradley-Terry convergence).
  `git show fc31ee6` confirms the one-line note is the only change to that file. This was a
  Pass-2-allowed item the team chose to address proactively this gate.
- **Suggested resolution**: None required — remediation verified complete.

#### SEC-RC-01 (carried forward) — OCR/DICOM extras are floored (unpinned) version specs — `dev-assist-development-security-sast`

- **Location**: `pyproject.toml:140-141`
- **Finding**: The OCR/DICOM optional extras are floored (lower-bound) version specs, not
  pins: pyproject.toml:140-141 declares ocr=['pytesseract>=0.3','Pillow>=10'] and
  dicom=['pydicom>=2.4']. Independently confirmed on disk. The iteration-1 pip-audit-clean
  evidence reflects the publish-time-style resolution (pillow 12.2.0 / pydicom 3.0.2 /
  pytesseract 0.3.13) in a throwaway venv; because the floors are unpinned, a future fresh
  install can resolve different transitive versions whose CVE posture is not frozen by this
  RC. Acceptable for a LOCAL-ONLY tag+build (not published), and the CVE-surface packages are
  correctly isolated to optional extras (the core install path carries only pydantic>=2.8,<3,
  properly upper-bounded; no VCS/git direct-ref deps). No current breach. Carried forward from
  iteration 1 unchanged.
- **Suggested resolution**: No action for this local-only RC. At actual publish time (a
  separate future gate), re-run pip-audit against the publish-time resolved closure and
  consider an upper bound or lockfile for the heavyweight extras if reproducible CVE posture
  becomes a release requirement.

#### SEC-RC-02 (carried forward) — committed non-production key material correctly scoped and labeled — `dev-assist-development-security-sast`

- **Location**: `src/pii_anon/moe_gate_signing.py:78-102`;
  `src/pii_anon/tokenization/encrypted_store.py:164-212`
- **Finding**: Committed non-production key material is correctly scoped, env-overridable, and
  clearly labeled — but it is genuine bytes in source. Independently verified:
  moe_gate_signing.py:83 `_TEST_HMAC_KEY = b'S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD'`
  (a self-labeling non-production string, overridable via the PII_ANON_GATE_SIGNING_KEY env
  var through the correct key-provider seam), and tokenization/encrypted_store.py:164
  StaticTestKeyProvider documented as a Pass-2 drop-in behind the EnvelopeKeyProvider protocol
  (no real KMS/HSM/keyring custody claimed; performs no networking/subprocess). The only
  credential-shaped strings in src are synthetic PII-detector training/eval data
  (taxonomy.py:567 example_patterns — a synthetic OpenAI-style key placeholder and a synthetic
  AWS-style access-key-ID placeholder, literal prefixes not reproduced here, neither a
  valid-length key; generator.py:189 `_fake_api_key` produces a seeded random
  OpenAI-prefix-style string) — these are detector training data; the product itself detects
  credential-shaped PII. A git-tracked real-credential regex scan of src/ is clean. No real
  secret is committed anywhere in the diff or tree. This is the honest AGENT_SIMULATED
  key-custody caveat, flagged only because secret-shaped strings exist in source and a naive
  scanner would surface them. Carried forward from iteration 1 unchanged.
- **Suggested resolution**: No action. The env-overridable provider seams are the correct
  pattern; keep the Pass-2 real-custody (KMS/HSM/keyring) migration tracked. The synthetic
  taxonomy/dataset credential shapes must remain (they are detector test data) — exclude them
  from any future secret-scanner allow-list rather than removing them.

## Cross-Reviewer Pattern Detection

- **Aggregator output: none.** `cross_reviewer_patterns` is empty — no formal multi-reviewer
  pattern was recorded by the aggregation pass.
- **Scribe note (non-binding; alters no severity and no verdict):** the iteration-1
  convergence cluster — *canonical traceability/NFR artifacts lagging the SO-08..SO-23
  sign-off ledger*, independently touched by three reviewers in iteration 1 — does NOT
  reproduce this iteration. The traceability reviewer that carried the cluster's MAJOR now
  reports zero findings; the backfilled matrix is cited as live evidence by axiom-compliance
  (AX-RC-03), and the NFR-numbering supersession note is independently verified closed by
  performance-benchmark (PERF-RC-06). The remediation held under re-review by the full set.
- The only residual alignment is intra-dimension: the two security observations
  (SEC-RC-01/SEC-RC-02) are explicitly marked carried-forward-unchanged accepted states of a
  LOCAL-ONLY rc, both with future-gate routing (publish-time pip-audit; Pass-2 key custody) —
  informational, not a cross-reviewer pattern.

## Refinement Iteration Record

| Iteration | Date | Findings status pre-iteration (S/C/MAJ/MIN/OBS) | Refinement applied | Findings status post-iteration |
|---|---|---|---|---|
| 1 | 2026-06-10 | 0/0/1/6/14 | Executor amendment per the iteration-1 Next Action: traceability-matrix backfill (TRACE-RC-01) + RC-mechanics MINORs + reid.py docstrings + the NFR-001 supersession note, landing as commit `fc31ee6` (docs/build-metadata/docstrings only; protected md5s byte-identical) | Resolved → re-dispatched same 6-reviewer set |
| 2 | 2026-06-10 | 0/0/0/0/6 | None required — all reviewers APPROVE on re-review; remaining findings are informational OBSERVATIONs with feed-forward routing | **Gate closed: APPROVE at iteration 2 of 3** |

(3-strike escalation per `five-gate-cascade.md#3-strike-escalation-rule` not reached.)

## Pass 2 Commitment Check (DIVERGED stories only)

**Not binding at this gate — zero MUST FR/NFR carries `provisional_status: DIVERGED`**
(established at iteration 1 by requirements-coverage REQ-RC-01:
`requirements-document.md:107` records "0 DIVERGED, 0 INSUFFICIENT-EVIDENCE"; NFR-017 is
REVISE-LOOSER on a SHOULD, not DIVERGED; unchanged at iteration 2 — requirements-coverage
re-approves with zero findings).

- [ ] Pass 2 commitment scheduled in DoD — N/A (no DIVERGED story/epic/sprint in scope)
- [ ] Pass 2 outcome logged in `traceability-matrix.md` Status Change Log — N/A (the matrix
  backfill owed by iteration-1 TRACE-RC-01 has landed; see AX-RC-03's row citations)
- [ ] `provisional_status` updated on Pass 2 outcome — N/A

The program-wide `AGENT_SIMULATED` → real-user Pass-2 roadmap remains an honest release
caveat, not a verification blocker for a local-only RC. AX-RC-03 adds the corpus-scale
AX-003/AX-005 re-confirmation to the DATA-pillar Pass-2 ledger (binding before any
claim-grade publication or the GA tag, not before this rc).

## Verdict + Next Action

**Verdict: APPROVE** (iteration 2 of 3; 6/6 reviewers APPROVE; 0 SHOWSTOPPER / 0 CATASTROPHIC /
0 MAJOR / 0 MINOR / 6 OBSERVATION)

**Next action (APPROVE branch)**: close the release scope `v1.5.0-rc1`; transition the gate
status; update `MANIFEST.md` (release-gate row → APPROVE, iteration 2, 2026-06-10). The RC
mechanics may proceed (LOCAL-ONLY annotated tag + sdist/wheel build per the RC spec — the tag
must NOT be pushed; see the iteration-1 CQ-RC-01 guard).

Execution notes for the close:

1. **Tag hygiene (from PERF-RC-04)**: do NOT stage the six uncommitted working-tree benchmark
   artifact mods into the rc tag — tag at the committed HEAD `fc31ee6`, whose benchmark
   artifact set is the canonical, verdict-stable record (all six profile floor_pass verdicts
   byte-identical between HEAD and the re-run).
2. **Protected control paths verified this iteration (PERF-RC-05)**: competitive_supremacy.py
   `3b842e81c3f03eafd11f9c655c1789a0`, canonical_run.py `d8f0f80e113c3b5d59c06d0b5fd36fac`,
   competitor_compare.py `7cae16c89f4c97136e1a12394dae2025`, latency_ceilings.py
   `848f21bef3731ad56bae3b0b5aa86fee` — all byte-identical to attestation. No
   `competitive_supremacy.py` change occurred in remediation, so no SDO adversarial close was
   triggered by this gate. Tag/build steps must not alter these files.
3. **Feed-forward routing of the 6 OBSERVATIONs (no gate action owed)**:
   - PERF-RC-04 → informational; harness-reproducibility note stays on the backlog.
   - PERF-RC-05 / PERF-RC-06 → none (clean verification records).
   - SEC-RC-01 → re-run pip-audit against the resolved closure at the future PUBLISH gate;
     consider upper bounds/lockfile for heavyweight extras if reproducible CVE posture becomes
     a release requirement.
   - SEC-RC-02 → keep the Pass-2 real-custody (KMS/HSM/keyring) migration tracked; route the
     synthetic detector-data credential shapes to a secret-scanner allow-list, never delete.
   - AX-RC-03 → DATA-pillar Pass-2: corpus-scale recall-floor + calibration runs; re-confirm
     AX-003/AX-005 at scale before claim-grade (non-local) publication or the GA tag.
4. **Standing caveats carried into the release record (unchanged)**: the SDO verdict remains
   the honest NOT_YET with binding G6 — an honest outcome, not a gate defect; do not "fix" F2
   or weaken G6. Record the honest caveat set verbatim; FR-032-native and FR-034-corpus are
   not marked fully-verified at this gate (per the iteration-1 REQ-RC-03 directive, which the
   requirements-coverage re-approval leaves standing).

Branch reference (for the record):

- **APPROVE → close scope; transition status; update MANIFEST.** ← this gate
- REQUEST_CHANGES → executor amends; re-dispatch the same reviewer set; iteration count
  incremented.
- HALT_GATE → orchestrator stops; surface the SHOWSTOPPER to the user; must be resolved before
  proceeding.
- ESCALATE → 3-strike escalation; user adjudication required.

---

**Template version**: 1.0 (developer-assistant v0.4.0) — scribe synthesis, release gate
v1.5.0-rc1, iteration 2, 2026-06-10.
