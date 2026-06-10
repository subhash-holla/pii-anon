# Story Gate Synthesis — S7-05 (iteration 2)

- **Gate type:** story
- **Scope:** S7-05 (docs discoverability)
- **Iteration:** 2 (re-dispatch of the same 5-reviewer set after the iteration-1 REQUEST_CHANGES amendment)
- **Date:** 2026-06-10
- **Gate integrity:** complete — 5/5 reviewers reported; no missing reviewers.

**Iteration history:** iteration 1 (2026-06-10) aggregated **REQUEST_CHANGES** on a single MAJOR (requirements-coverage MAJ-1: the certify-a-run example pointed at `./certified/benchmark-results.json`, a file the producer never writes) plus 4 observations. The executor remediated in `8c9cec3..5f825b6` (docs/evaluate-your-pipeline.md, docs/recall-floor.md, tests/test_docs_discoverability.py only) and the same reviewer set was re-dispatched per the gate rule. This iteration confirms the MAJOR closed with teeth.

## Reviewer Set

| Reviewer | Verdict | SHOWSTOPPER | CATASTROPHIC | MAJOR | MINOR | OBSERVATION |
|---|---|---|---|---|---|---|
| dev-assist-development-code-quality | APPROVE | 0 | 0 | 0 | 0 | 0 |
| dev-assist-development-security-sast | APPROVE | 0 | 0 | 0 | 0 | 3 |
| dev-assist-development-requirements-coverage | APPROVE | 0 | 0 | 0 | 0 | 0 |
| dev-assist-development-traceability | APPROVE | 0 | 0 | 0 | 0 | 5 |
| dev-assist-development-axiom-compliance | APPROVE | 0 | 0 | 0 | 0 | 0 |
| **Totals** | — | **0** | **0** | **0** | **0** | **8** |

## Aggregation Rule + Aggregate Verdict

**Rule:** any SHOWSTOPPER ⇒ HALT_GATE; else any CATASTROPHIC or MAJOR (equivalently, any reviewer verdict of REQUEST_CHANGES) ⇒ REQUEST_CHANGES; else (MINOR/OBSERVATION only, all reviewers APPROVE) ⇒ APPROVE. A gate with missing reviewers cannot APPROVE.

**Applied:** 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 MAJOR / 0 MINOR / **8 OBSERVATION**, with all five reviewers returning APPROVE and gate integrity complete.

### Aggregate verdict: **APPROVE**

All eight findings are observations — five of them are positive closure/verification confirmations (the iteration-1 MAJ-1 remediation, the carried orchestrator-md5 OBS, the A5 teeth counterfactual, program-invariant byte-identity, and the FR-010 trace chain). Nothing blocks.

## Merged Findings

### SHOWSTOPPER (0)

None.

### CATASTROPHIC (0)

None.

### MAJOR (0)

None. (Iteration-1 MAJ-1 is confirmed closed — see OBS-2, OBS-3, OBS-6.)

### MINOR (0)

None.

### OBSERVATION (8)

**OBS-1 — [traceability] No S7-05 row in traceability-matrix.md — established matrix convention, not a regression.**

- **Location:** `dev-assist-artifacts/02-requirements/traceability-matrix.md:25`.
- **Description:** The traceability-matrix.md carries no S7-05 row, but this is NOT a regression introduced by this story and NOT a finding-#4 violation. The matrix is FR/NFR → use-case → component grained and tracks zero per-story rows program-wide (grep for any S[n]-[nn] story ID returns 0). FR-010 itself is already traced as implemented/"done" in the matrix (line 25: FR-010 → DATA:scoring/pseudonymization.py + anonymization.py). S7-05 is a documentation-surfacing story for an already-traced FR, pinned by teeth tests; the story spec accurately documents the authority chain ("No numbered FR exists for docs discoverability itself ... the dev-log [DOCS MUST] tag + SME MAJORs are the authority chain").
- **Suggested resolution:** No action for this story. Optional program-level note: the matrix's FR→component granularity (no per-story rows) is the established convention; if per-story tracing is ever desired it would be a cross-cutting matrix-format change, not an S7-05 obligation.

**OBS-2 — [traceability] Iteration-1 MAJOR-1 genuinely closed: certify-a-run example now reads the file the producer actually writes.**

- **Location:** `docs/evaluate-your-pipeline.md:347-357`; `src/pii_anon/evaluation/canonical_run.py:1275`; `src/pii_anon/cli.py:908`.
- **Description:** The certify-a-run example MAJOR-1 from iteration 1 is genuinely closed. The remediated example reads `./certified/canonical-run.json` (docs/evaluate-your-pipeline.md:354) with a comment naming the emitted file (line 350). Verified against the producer: canonical_run.py:1275 writes `out_dir / "canonical-run.json"` and cli.py:908 reports the same path; the `--output-dir ./certified` flag flows directly into out_dir, so the example is copy-paste-runnable. The prior phantom filename (benchmark-results.json, never written by the canonical-run producer) is gone.
- **Suggested resolution:** None required — closed.

**OBS-3 — [security-sast] Doc security claims verified accurate; cli.py change is help-text-only; no new dependency/secret/egress surface; MAJ-1 remediation confirmed with teeth.**

- **Location:** `docs/recall-floor.md`, `src/pii_anon/cli.py`, `docs/evaluate-your-pipeline.md`, `tests/test_docs_discoverability.py`.
- **Description:** Doc security claims verified accurate within scope: (a) recall-floor.md now names FloorProjectingFusion (pii_anon.routing.floor_fusion) as the fusion-path applier wrapping any inner FusionStrategy — matches the iteration-1 OBS remediation intent; (b) the cli.py epilog change is help/epilog string text only — no new @app.command, no behavior change, no new capability grant, no egress (confirmed by diff grep), satisfying the story's "security-sast not triggered unless cli.py edits exceed help text" gate condition. No dependency manifests (pyproject.toml/lockfiles) were touched by the 3-commit diff, so there is no new CVE/supply-chain/denied-license surface. No secrets, credentials, API keys, or private keys were introduced, and no dangerous-API primitives (unsafe deserializers, dynamic code-running calls, shell-spawning calls) entered the diff. The MAJOR-1 remediation (example now reads ./certified/canonical-run.json, the file the producer actually writes per canonical_run.py:1275 + cli.py:908) is genuinely closed, and the A5 teeth bite (counterfactual: the old phantom benchmark-results.json fails the `.endswith('canonical-run.json')` assertion). All program-invariant + user-WIP files have zero diff lines.
- **Suggested resolution:** None — informational.

**OBS-4 — [traceability] Remediation diff is exactly as scoped; FloorProjectingFusion mention accurate; program invariants untouched.**

- **Location:** `git diff 8c9cec3..5f825b6`; `src/pii_anon/routing/floor_fusion.py:59`.
- **Description:** No new traceability issue introduced by the remediation diff. The remediation (8c9cec3..5f825b6) touches only docs/evaluate-your-pipeline.md, docs/recall-floor.md, and tests/test_docs_discoverability.py as scoped. The FloorProjectingFusion mention added to recall-floor.md is accurate (the class exists at routing/floor_fusion.py:59 as a FusionStrategy wrapper over an inner strategy that re-injects shared-layer spans). Program invariants verified untouched across the full e573243^..5f825b6 range: orchestrator.py, competitive_supremacy.py, canonical_run.py, README.md, docs/pii-rate-elo-value.md, docs/benchmark-summary.md all produce empty diffs. Owned test suite 7 passed.
- **Suggested resolution:** None required.

**OBS-5 — [security-sast] Iteration-1 orchestrator.py md5 OBS resolved (carried): no drift — a measurement-frame artifact, not a code change.**

- **Location:** `src/pii_anon/orchestrator.py`.
- **Description:** ITERATION-1 OBS RESOLVED (carried): orchestrator.py md5 invariant confirmed NOT drifted. Working-tree md5 = `0afc6deed62bbd0653ae1051b723bace` (the dispatch-pinned user-WIP value, unchanged) and committed-blob md5 = `4a837c52ccdb27925d1f7885e71667d0` (git show HEAD, unchanged). `git diff e573243^..5f825b6 -- src/pii_anon/orchestrator.py` is empty (0 diff lines). This was a measurement-frame artifact (working-tree vs committed blob), not a code change. No action required.
- **Suggested resolution:** None — informational.

**OBS-6 — [traceability] A5 path teeth bite as claimed: counterfactual phantom path assert-FAILs.**

- **Location:** `tests/test_docs_discoverability.py:124-131`.
- **Description:** The A5 path teeth bite as claimed. The regex `supremacy\s+--artifact\s+(\S+)` matches exactly one example; the current path `./certified/canonical-run.json` passes the `.endswith("canonical-run.json")` assertion, while the counterfactual old phantom path `./certified/benchmark-results.json` would assert-FAIL (confirmed by running the counterfactual directly). Future path drift in any certified-dir supremacy example is now caught by the test, not a reviewer.
- **Suggested resolution:** None required.

**OBS-7 — [security-sast] Test A7 child-process use is hermetic and injection-free; A2 URL literals are skip-prefixes, not egress.**

- **Location:** `tests/test_docs_discoverability.py:76,175-183`.
- **Description:** Test A7 runs the CLI as a child process with a hard-coded argv list (`[sys.executable, '-m', 'pii_anon.cli', '--help']`) — no shell-mode invocation (the command is an argv vector, never routed through a shell), no user-input interpolation, env constrained to `PATH=/usr/bin:/bin` with no inherited environment. This is a hermetic test-only `--help` invocation, not a network egress and not an injection surface. The `http://`, `https://`, `mailto:` literals in test A2 (line 76) are prefix checks that SKIP external links during the link-resolution sweep — no network call is made (the opposite of egress). Both patterns are benign within the security-sast dimension. No action required.
- **Suggested resolution:** None — informational.

**OBS-8 — [traceability] FR-010 → Story → Test → DC chain holds; A3 pins the no-merge invariant against real scorer symbols.**

- **Location:** `tests/test_docs_discoverability.py:88-98`; `docs/anonymization-vs-pseudonymization.md`; `dev-assist-artifacts/02-requirements/requirements-document.md:21`.
- **Description:** FR-010 → Story → Test → DC chain holds. FR-010 exists in requirements-document.md:21 (MUST: "Enforce anon vs pseudo as distinct families (no merge path/field)"). The A3 test (tests/test_docs_discoverability.py:88-98) pins the no-merge invariant against real scorer symbols AnonymizationScorer / PseudonymizationIntegrityScorer (located at eval_framework/metrics/deid_families.py:172 and :238) and the doc states the invariant verbatim ("never merged into a single de-id score", no combined/deid_score/privacy_score field). Test body is semantically aligned with the FR text. All other FR/NFR IDs referenced in the test anchors (FR-001/002/023/031/039, NFR-025) exist in requirements-document.md (1 row each).
- **Suggested resolution:** None required.

## Cross-Reviewer Pattern Detection

None detected by the aggregator (`cross_reviewer_patterns: []`). No convergent or divergent finding clusters require action. Worth noting (informational, not a pattern flag): security-sast (OBS-3) and traceability (OBS-2, OBS-6) **independently** confirmed the iteration-1 MAJ-1 closure — both verified the example path against the producer's actual emission (canonical_run.py:1275 + cli.py:908) and both confirmed the A5 counterfactual bite (the old phantom path fails the new `.endswith("canonical-run.json")` assertion). That is the desired remediation-verification redundancy, and it lands the iteration-1 lesson: doc examples are now anchored to producer-emitted filenames by teeth, the docs analogue of the program's recurring exact-rate-anchor lesson.

## Verdict + Next Action

| Aggregate verdict | Action |
|---|---|
| **APPROVE** (selected) | **Close scope; update MANIFEST.** |
| REQUEST_CHANGES | Executor amends the findings, then re-dispatch the SAME reviewer set. |
| HALT_GATE | Stop; surface SHOWSTOPPER to user. |

**Next action:** Close S7-05 and update MANIFEST (story → DONE; record the 2-iteration gate trail: iteration 1 REQUEST_CHANGES on MAJ-1 → remediation `8c9cec3..5f825b6` → iteration 2 APPROVE, 0 blocking findings / 8 observations).

Notes for the close:

1. All 8 observations are no-action-required (five are positive closure confirmations; OBS-1 carries an optional program-level note about matrix granularity that is explicitly not an S7-05 obligation).
2. This is a docs-only story: the e573243^..5f825b6 range leaves `competitive_supremacy.py`, `canonical_run.py`, `orchestrator.py`, and all user-WIP files byte-identical (verified by OBS-3/OBS-4/OBS-5), so **no SDO adversarial close is triggered** by this gate.
