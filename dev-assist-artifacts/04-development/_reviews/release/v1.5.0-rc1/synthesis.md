# Release Gate Synthesis — release — v1.5.0-rc1 (iteration 1)

> Canonical gate record per `templates/development/gate-checklist.md.tmpl`.
> Persisted at `dev-assist-artifacts/04-development/_reviews/release/v1.5.0-rc1/synthesis.md`.

**Gate type**: release
**Scope ID**: v1.5.0-rc1
**Date**: 2026-06-10
**Refinement iteration**: 1 of 3
**Gate integrity**: **complete** — 6/6 dispatched reviewers reported canonical YAML; missing reviewers: none.

## Reviewer Set

(Computed by the orchestrator per the `five-gate-cascade.md` selection algorithm; release-gate set = 6
reviewers. Accessibility was not selected for this scope; contributor-experience is S8-only.)

| Reviewer | Verdict | S/C/MAJ/MIN/OBS |
|---|---|---|
| `dev-assist-development-code-quality` | APPROVE | 0/0/0/3/3 |
| `dev-assist-development-security-sast` | APPROVE | 0/0/0/0/2 |
| `dev-assist-development-requirements-coverage` | APPROVE | 0/0/0/0/3 |
| `dev-assist-development-traceability` | **REQUEST_CHANGES** | 0/0/**1**/2/2 |
| `dev-assist-development-performance-benchmark` | APPROVE | 0/0/0/1/2 |
| `dev-assist-development-axiom-compliance` | APPROVE | 0/0/0/0/2 |

**Merged totals**: 0 SHOWSTOPPER / 0 CATASTROPHIC / 1 MAJOR / 6 MINOR / 14 OBSERVATION — 21 findings.

## Aggregation

Per the `dev-assist-development` skill's aggregation rule:

```
if any(severity == SHOWSTOPPER): HALT_GATE
if any(verdict == REQUEST_CHANGES): REQUEST_CHANGES (merged findings)
return APPROVE
```

- Any SHOWSTOPPER? **No** (0 across all six reviewers) → not HALT_GATE.
- Any reviewer verdict REQUEST_CHANGES? **Yes** — traceability (carries the single MAJOR) → REQUEST_CHANGES.

**Aggregate verdict: REQUEST_CHANGES** (5 APPROVE / 1 REQUEST_CHANGES).

## Merged Findings (ordered by severity)

### SHOWSTOPPER (0)

None.

### CATASTROPHIC (0)

None.

### MAJOR (1)

#### TRACE-RC-01 — canonical traceability matrix never backfilled (program-wide) — `dev-assist-development-traceability`

- **Location**: `dev-assist-artifacts/02-requirements/traceability-matrix.md:3`
- **Finding**: The canonical traceability-matrix.md was never backfilled with the Story/Test/DC
  columns — program-wide. Its header still reads "DC/Story/Test columns fill in Stages 3-5"
  (line 3) and the body remains UC-organized only (PGO/UC/FR-NFR/external_refs); there are zero
  Story rows, zero Test rows, and no Status Change Log. Every story gate flagged this exact MINOR
  and deferred it "to the S2/S4/S5/S6 sprint gate" (S2-05-gate.yaml:101-102,
  S4-03-gate.yaml:131/192, S5-04-gate.yaml:163, plus the S6-02 deferred TRACE-S6-02-MAJOR-1), but
  only the S1 sprint gate ever ran (`_reviews/sprint/` contains S1-gate.yaml alone) — sprints
  S2-S7 were closed via work-stream sign-offs (SO-08..SO-23), so those deferred matrix-backfill
  items were never closed. At a RELEASE gate the canonical matrix should reflect the final
  FR/NFR → Story → Test → DC chain; it does not. Mitigating: the chain IS fully reconstructable
  (every story's Implements field declares its IDs and every MUST has an in-tree test pin), so
  this is reconciliation debt, not an irreparable break.
- **Suggested resolution**: Backfill traceability-matrix.md with the Story and Test columns for
  all 30 DONE stories (FR/NFR → claiming story → the in-tree test file/function that pins it →
  DC), and add a Status Change Log section. This is a docs-only change (no code touched, no
  protected md5 affected) and closes the chain of deferred matrix-backfill MINORs accumulated
  across S2-S7. It can be done before tagging since it does not alter the package.

### MINOR (6)

#### CQ-RC-01 — no local-only guard on rc tags (accidental push → TestPyPI publish) — `dev-assist-development-code-quality`

- **Location**: `.github/workflows/release.yml:7,50` + `Makefile` (no rc-push guard)
- **Finding**: The release.yml tag trigger `v*-rc*` (line 7) routes any pushed rc tag to TestPyPI
  (line 50: `contains(github.ref_name, '-rc')`). The RC spec mandates LOCAL-ONLY — the annotated
  tag v1.5.0-rc1 must NEVER be pushed. However, the Makefile has no target that creates the
  annotated tag with an explicit "do not push" warning, and no CI guard blocks a push of a local
  rc tag. A single accidental `git push --tags` or `git push origin v1.5.0-rc1` would trigger a
  TestPyPI publish silently.
- **Suggested resolution**: Add a Makefile `rc-tag` target that creates the local annotated tag
  and prints a prominent warning: "LOCAL-ONLY: do NOT push this tag — pushing triggers TestPyPI
  publish via release.yml". Optionally add a pre-push hook or a CI branch-protection rule that
  rejects rc-tag pushes from the pdlc/ branch.

#### TRACE-RC-02 — FR-027 listed in S6-03 Implements without a test pin — `dev-assist-development-traceability`

- **Location**: `dev-assist-artifacts/04-development/02-stories/sprint-6/S6-03-encrypt-token-store.md:10`
- **Finding**: S6-03 lists FR-027 (stable session pseudonyms, SHOULD) in its "Implements" field as
  "the bundled SHOULD sub-scope per the sprint plan" (line 10) while its own Non-goal section
  (line 55) defers the session-pseudonym sub-feature, and there is no test in the suite pinning
  FR-027 (confirmed: zero references to fr_027/session-pseudonym anywhere in tests/). A SHOULD
  listed under Implements without a test pin is a forward-traceability accuracy issue
  (finding-pattern #1, SHOULD-scoped). The tension is openly disclosed in the same story, so it
  is not hidden.
- **Suggested resolution**: Move FR-027 out of S6-03's Implements field into an explicit
  "Deferred (Pass-2)" line (matching how S6-02/S6-05 phrase it), so the matrix backfill does not
  record FR-027 as delivered-by-S6-03 without evidence.

#### TRACE-RC-03 — Stage-5 NFR verification matrix stale relative to HEAD — `dev-assist-development-traceability`

- **Location**: `dev-assist-artifacts/05-testing/03-nfr-verification/nfr-verification-matrix.md:11`
- **Finding**: The Stage-5 nfr-verification-matrix.md is stale relative to HEAD: 22 of 26 NFRs
  still read "DEFERRED → successor story" (e.g. NFR-001→S3-03, NFR-006→S4-02,
  NFR-014/015→S4-01/S6-03, NFR-017..021→S4-03, NFR-012→S5-02, NFR-013→S5-03, NFR-022/023→S7-02,
  NFR-025→S7-03) although those successor stories have ALL since landed (SO-08..SO-23) with
  in-tree test pins. Only NFR-009 was updated in place (line 16, the S7-04 edit). The
  release-gate caveat set explicitly says the stale Stage-5 testing report should not be
  re-litigated, so this is advisory — but the artifact reads as the NFR→verification index and
  now under-reports the actual coverage.
- **Suggested resolution**: Either refresh the NFR statuses to in-tree-VERIFIED with their
  landed-story + test-file evidence, or add a one-line banner noting this is the Stage-5
  foundation snapshot superseded by the SO ledger (the S3-03 gate already recommended this flip
  at line 102). Pairs naturally with the TRACE-RC-01 matrix backfill.

#### PERF-RC-01 — two NFR numbering schemes both define an "NFR-001" (stale 200ms vs binding 500ms) — `dev-assist-development-performance-benchmark`

- **Location**: `pdlc-artifacts/swarm/requirements/non-functional-requirements.md:9-13` vs `dev-assist-artifacts/05-testing/03-nfr-verification/nfr-verification-matrix.md:16`
- **Finding**: Two distinct NFR numbering schemes coexist and both define a "NFR-001" with
  different meaning, which could mislead a future reviewer about which latency commitment binds.
  pdlc-artifacts/swarm/requirements/non-functional-requirements.md:9-13 defines NFR-001 = swarm
  detection latency p50<=200ms (a Discovery-era constraint, "~90ms currently"), while
  dev-assist-artifacts/05-testing/03-nfr-verification/nfr-verification-matrix.md uses the
  eval-pillar scheme where NFR-001 = Bradley-Terry Rhat<=1.01. The OPERATIVE, in-tree-verified
  latency commitment is NFR-009 (full-swarm ensemble p50<=500/p95<=1000/p99<=2000ms), correctly
  recorded in the matrix line 16 ("NFR-009 VERIFIED-in-tree") and grounded in S7-04. The 200ms
  discovery figure is effectively superseded by the R10-rescoped 500ms ensemble budget
  (documented headroom rationale in S7-04 grounding + latency_ceilings.py docstring), and the
  perf test `test_multi_engine_latency_p50_under_500ms` asserts against the committed 500ms, not
  the stale 200ms. This is a documentation-traceability nit, NOT a budget violation — the gate's
  actual ceiling is met live with ~2.5x headroom. No action required for the RC; flag for a
  Pass-2 NFR-doc consolidation.
- **Suggested resolution**: Pass-2: add a one-line cross-reference in the swarm NFR doc noting
  NFR-001's 200ms p50 is superseded by the R10-rescoped NFR-009 ensemble budget (p50<=500ms) per
  S7-04, so a future reader does not read the stale 200ms as the binding ceiling.

#### CQ-RC-02 — trove classifier claims Production/Stable on an RC build — `dev-assist-development-code-quality`

- **Location**: `pyproject.toml:32` (Classifier: Development Status :: 5 - Production/Stable)
- **Finding**: The "Development Status :: 5 - Production/Stable" trove classifier is set in
  pyproject.toml at HEAD. For an RC release (v1.5.0-rc1) the correct classifier is "4 - Beta".
  When the RC mechanics run `make build`, the resulting wheel will carry the Production/Stable
  claim despite being a candidate release. This is misleading to PyPI tooling and downstream
  consumers that inspect trove classifiers to filter for stable releases.
- **Suggested resolution**: As part of the RC mechanics version-bump commit (1.4.0 → 1.5.0rc1),
  change the classifier line to "Development Status :: 4 - Beta". When the final 1.5.0 release is
  tagged and the -rc suffix is removed, revert to "5 - Production/Stable".

#### CQ-RC-03 — missing per-method docstrings on reid attack methods (ruff D102) — `dev-assist-development-code-quality`

- **Location**: `src/pii_anon/eval_framework/attacks/reid.py:191,271`
- **Finding**: Public concrete method `BaselineDeterministicReidAttack.attack()` and the Protocol
  stub method `ReidAttack.attack()` are missing per-method docstrings (ruff D102). The Protocol
  class itself is fully documented; the missing docs are on the method signatures only. The two
  `MiaAttack.membership_scores()` and `ReidAttack.attack()` Protocol stubs at lines 191 and 213
  follow the standard Protocol-stub pattern (`...` body, class-level contract), but the concrete
  implementation at line 271 has only inline comments rather than a docstring.
- **Suggested resolution**: Add a one-sentence docstring to
  `BaselineDeterministicReidAttack.attack()` (line 271) referencing the Protocol contract. For
  Protocol stubs, adding a brief redirect docstring ("See ReidAttack protocol docstring.") is
  sufficient and silences D102.

### OBSERVATION (14)

#### TRACE-RC-04 — FR-028 channel-taxonomy wording divergence (4 sources vs "6 channels"), tracked — `dev-assist-development-traceability`

- **Location**: `dev-assist-artifacts/02-requirements/requirements-document.md:41`
- **Finding**: Spec-vs-implementation wording divergence on the agentic channel taxonomy: FR-028
  requirement text says "leakage-Sankey, 6 channels" (and UC-22 names 6 source channels) but the
  implementation covers 4 source channels rendered as a 6-NODE graph (4 sources + blocked/leaked
  sinks). This is properly TRACKED as a PO/Pass-2 decision (SO-14 channel-taxonomy-4-vs-6, lines
  55-57) and reconciled in S7-04's Implements note — disclosed, not a hidden defect. Flagging
  only so the matrix backfill records the 4-source coverage against the 6-channel requirement
  text with the tracked-gap annotation.
- **Suggested resolution**: No action required for the RC. At the matrix backfill, annotate
  FR-028 as "covered: 4 source channels; widening to 6 = tracked PO/Pass-2 per SO-14".

#### REQ-RC-01 — DIVERGED + Pass-2 cross-check satisfied: zero DIVERGED MUSTs — `dev-assist-development-requirements-coverage`

- **Location**: `dev-assist-artifacts/02-requirements/requirements-document.md:97-107`
- **Finding**: DIVERGED + Pass-2 cross-check (my mandatory release-gate obligation): ZERO MUST
  FR/NFR carry provisional_status DIVERGED. The requirements doc's R10 threshold-validation
  produced "0 DIVERGED, 0 INSUFFICIENT-EVIDENCE" (requirements-document.md:107); the only
  "DIVERGED" string tokens in the corpus are in that "0 DIVERGED" statement
  (requirements-document.md + SO-03-requirements.yaml). NFR-017 is REVISE-LOOSER (a loosened
  threshold: ECE <= 0.05 high-resource / <= 0.08 long-tail), NOT DIVERGED, and is a SHOULD (out
  of the MUST set). All FR/NFR carry provisional_status=AGENT_SIMULATED (representative scale)
  with the program-wide Pass-2 roadmap — which is honest-caveat #2, distinct from a DIVERGED
  provisional-threshold needing a scheduled Pass-2 confirmation. Therefore NO MUST owes a
  DIVERGED-style Pass-2 commitment at this gate; the cross-check is satisfied.
- **Suggested resolution**: No action. No DIVERGED MUST exists, so the "Pass-2 must complete
  before release" rule does not bind any MUST. The AGENT_SIMULATED → real-user Pass-2 roadmap is
  correctly carried as a release caveat, not a verification blocker for a local-only RC.

#### AX-RC-01 — D6 design-axioms-checklist appendix absent (axiom saturation still fully traceable) — `dev-assist-development-axiom-compliance`

- **Location**: `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md`
- **Finding**: The D6 design-axioms-checklist appendix file (_appendix/design-axioms-checklist.md,
  dispatched as design_axioms_checklist_path) is absent — D6 produced
  06-synthesis/D-implementation-ready-design.md but not a standalone per-layer saturation
  checklist. This did NOT impede the review: the authoritative axiom source
  (dev-assist-artifacts/00-axioms/project-axioms.yaml) exists and is complete, the D6 synthesis
  enumerates all 6 axioms and maps each to its enforcing module/layer
  (D-implementation-ready-design.md:6,11,33-35), and every DONE story carries a per-axiom status
  review under _reviews/story/<S>/axiom-compliance.yaml. Per-layer axiom saturation is therefore
  fully traceable. Likewise docs/architecture/axioms.md (the contributor-readiness S8 gate's
  artifact) is absent, but S8 is not in this RC's program scope (the RC close is docs → release
  gate → tag → build); axiom commentary lives in the registry + D6 synthesis + dedicated docs
  (recall-floor.md, anonymization-vs-pseudonymization.md).
- **Suggested resolution**: Non-blocking for this local-only RC. If/when the program reaches the
  S8 contributor-readiness gate, author docs/architecture/axioms.md (or regenerate the D6
  _appendix/design-axioms-checklist.md) consolidating the per-layer axiom saturation already
  evidenced across the synthesis + story reviews.

#### REQ-RC-02 — FR-017 substance fully verified but not labeled by its own ID — `dev-assist-development-requirements-coverage`

- **Location**: `dev-assist-artifacts/04-development/02-stories/sprint-1/S1-03-per-language-recall-floor-gate.md:10` ; `tests/test_recall_floor_per_language_gate.py:1-220` ; `dev-assist-artifacts/02-requirements/traceability-matrix.md:30`
- **Finding**: FR-017 (MUST — per-language recall-floor CI gate, router-on >= router-off -
  epsilon) has its SUBSTANCE fully verified by named, passing, teeth-proven tests
  (`test_nfr_011_per_language_recall_floor_holds_on_floored_fusion` +
  `test_nfr_011_floor_has_teeth_bare_swarm_drops_non_english_below_floor` in
  tests/test_recall_floor_per_language_gate.py) and is covered in docs (CHG + ADR per
  doc-coverage-audit.md:47,61) — but its own ID FR-017 is absent from any story "Implements" row
  (S1-03, the per-language-recall-floor-gate story, lists "FR-007, NFR-011" only) and from the
  test docstring (which cites "FR-007 / NFR-011"). FR-017 and NFR-011 are the FR-framing and the
  quantified-threshold of the SAME gate (traceability-matrix.md:30 maps UC-13/14 recall-floor →
  FR-016, FR-017 + NFR-011). This is a traceability-label nuance, NOT a silent coverage gap: the
  MUST's content is verified with named passing tests and is grep-confirmed covered (0 orphans)
  by the independent Stage-6 doc-coverage-audit. No coverage violation; recorded for awareness.
  (The own-ID labeling question is properly the traceability reviewer's dimension; logged as
  OBSERVATION to avoid cross-role policing.)
- **Suggested resolution**: Optional (non-blocking): add FR-017 to S1-03's "Implements" row and
  to the test-file docstring header for explicit own-ID traceability, so the MUST is
  verified-by-its-own-ID end to end. No code or test change required — the substance is already
  verified.

#### PERF-RC-02 — first dedicated performance-benchmark sign-off on the G5 latency surface (clean) — `dev-assist-development-performance-benchmark`

- **Location**: `dev-assist-artifacts/04-development/_reviews/story/S7-04/`
- **Finding**: Per the conditional story-gate dispatch model, performance-benchmark did NOT
  participate in the S7-04 story gate (the 5 reviewers on record under
  dev-assist-artifacts/04-development/_reviews/story/S7-04/ are axiom-compliance, code-quality,
  requirements-coverage, security-sast, traceability). This is correct behavior — the latency NFR
  wiring was instead exercised by the MANDATORY SDO adversarial close (2 rounds; round-1 caught
  the G5 breach-bury MAJOR, round-2 confirmatory CLOSE_PASS 0-upheld/764 probes per the
  manifest), which is the appropriate catch-net for a competitive_supremacy.py change. This
  release gate is therefore the first dedicated performance-benchmark sign-off on the G5 latency
  surface, and it is clean. Noted for audit-trail completeness, not as a gap.
- **Suggested resolution**: None. The SDO close adequately covered the latency-wiring adversarial
  surface; this release gate confirms the budget is met at release scale.

#### REQ-RC-03 — honest-caveat-set verification: all five caveats stated and tracked — `dev-assist-development-requirements-coverage`

- **Location**: `dev-assist-artifacts/_signoffs/SO-15-keystone-close.yaml` ; `dev-assist-artifacts/04-development/_reviews/story/S7-01/requirements-coverage.yaml:22-31` ; `dev-assist-artifacts/04-development/_reviews/story/S7-02/dev-assist-development-requirements-coverage.yaml:10-40`
- **Finding**: HONEST-CAVEAT-SET VERIFICATION (the gate's purpose — confirm the caveats are
  STATED, not hidden): all five honest caveats are present and tracked. (1) SDO NOT_YET / binding
  G6 raw-F2 non-inferiority FAIL is an honest VERDICT OUTCOME, not an unverified MUST — the
  FR-007/FR-008/NFR-006 gate EXISTS and computes honestly (G1/G2/G3/G4/G5/G7 ALL PASS, zero
  pending; protected md5s byte-identical: competitive_supremacy.py 3b842e81...,
  canonical_run.py d8f0f80e..., competitor_compare.py 7cae16c8...), and the S7-02 keystone
  caught+closed an iter-1 scope-laundering finding so the NOT_YET holds on fresh numbers (the
  composite crown flips to gliner on the artifact's own fresh draw). (2) all cohort research
  AGENT_SIMULATED with Pass-2 roadmap — confirmed S5-02/S5-03/S7-03 coverage reviews.
  (3) representative-vs-Pass-2 splits tracked per story for
  FR-011/FR-013/NFR-012/NFR-013/NFR-004/NFR-025 via "# SWITCH-POINT(DATA)" + story sections, NOT
  silently claimed complete. (4) FR-032-native + FR-034-corpus remain OPEN MUST-board rows
  flagged representative-this-sprint — the S7-01 coverage review (requirements-coverage-story-02)
  carries the explicit directive "do not mark either fully-verified at the release gate on S7-01
  alone". (5) the Stage-5 stale NFR matrix (2 VERIFIED + 2 PARTIAL + 22 DEFERRED, dated
  2026-05-30) is superseded — its "DEFERRED → [story]" rows (NFR-001→S3-03, NFR-002→S3-04,
  NFR-006→S7-02, NFR-012→S5-02, NFR-013→S5-03, NFR-014/016→S4-01, etc.) are all now discharged by
  completed S2-S7 DONE stories per the SO-08..SO-23 ledger.
- **Suggested resolution**: No action required for the coverage dimension — all caveats are
  stated and tracked. The release SHIP-WITH-CAVEATS verdict is the honest disposition; the gate
  should record the caveat set verbatim and NOT mark FR-032-native or FR-034-corpus
  fully-verified.

#### AX-RC-02 — recall-floor.md Roadmap section stale (axiom itself fully upheld in code) — `dev-assist-development-axiom-compliance`

- **Location**: `docs/recall-floor.md:3,46`
- **Finding**: docs/recall-floor.md (the AX-003 headline doc) carries a stale Roadmap line: line
  46 says production wiring "is the immediate next step (story S1-02/S1-03)" and line 3 stamps
  "Status: shipped (foundation)", but S1-02 (FloorProjectingFusion wired into build_fusion) and
  S1-03 (per-language ε-gate) are BOTH DONE and live in-tree (confirmed:
  src/pii_anon/routing/floor_fusion.py + tests/test_recall_floor_per_language_gate.py, both
  green). The AXIOM ITSELF is fully upheld in code — this is purely a doc-currency lag in the
  forward-looking framing, not an axiom violation. Documentation-currency is Stage-6's dimension
  (verdict DOCUMENTED); flagged here only as an axiom-adjacent note. The doc's guarantee framing
  is otherwise correct and honest (states entities(output) superset entities(shared)
  structurally, carries an explicit benchmark caveat at line 48).
- **Suggested resolution**: Non-blocking for this RC. In a future docs pass, update the
  recall-floor.md Roadmap section to reflect that S1-02/S1-03 production wiring + the
  per-language ε-gate are shipped (change "is the immediate next step" to past tense / "shipped
  in S1-02/S1-03").

#### CQ-RC-04 — ruff D (pydocstyle) rules not in the CI select set (~797 latent violations) — `dev-assist-development-code-quality`

- **Location**: `pyproject.toml` (ruff `[tool.ruff.lint]` — no `select` key; D rules absent from CI)
- **Finding**: The ruff lint configuration has no explicit `select` list, defaulting to the E/F/W
  rule sets. Pydocstyle (D) rules are not enabled. Running `ruff check src --select D` finds 797
  violations across the entire src tree. The new program-added modules (query_aware.py,
  byo_pipeline.py, native.py, native_pdf.py, fairness_gate.py) are individually clean on
  D100-D103. The violations are concentrated in pre-existing modules (swarm.py, cli.py,
  bridge.py, metrics/core.py, pipeline.py). This is not a blocking issue for the release gate
  but is the pre-condition for passing a future contributor-readiness gate.
- **Suggested resolution**: At contributor-readiness gate: add `select = ["D"]` with a
  per-file-ignores list for legacy modules, or enable D rules on new modules only via a
  per-file-ignores allow-list pattern. Resolve the ~797 violations incrementally before enabling
  the full deny.

#### SEC-RC-01 — OCR/DICOM extras are floored (unpinned) version specs — `dev-assist-development-security-sast`

- **Location**: `pyproject.toml:140-141`
- **Finding**: The OCR/DICOM extras are floored (lower-bound) version specs, not pins:
  pyproject.toml declares ocr=['pytesseract>=0.3','Pillow>=10'] and dicom=['pydicom>=2.4']. The
  pip-audit clean evidence (2026-06-09: zero advisories) reflects the CURRENT resolution (pillow
  12.2.0 / pydicom 3.0.2 / pytesseract 0.3.13), which satisfies the floors and is internally
  consistent. Because the floors are unpinned, a future fresh install can resolve different
  transitive versions whose CVE posture is not frozen by this RC. This is acceptable for a
  LOCAL-ONLY tag+build (not published) and matches the documented S7-01/S7-05 carry-forward
  closure. No breach; noted so the published-RC gate re-runs pip-audit against whatever the
  publish-time resolver picks.
- **Suggested resolution**: No action required for this local-only RC. At actual publish time (a
  separate future gate), re-run pip-audit against the publish-time resolved closure and consider
  an upper bound or lockfile for the heavyweight extras if reproducible CVE posture becomes a
  release requirement.

#### PERF-RC-03 — live host-contention sensitivity confirmed, within the documented envelope — `dev-assist-development-performance-benchmark`

- **Location**: `src/pii_anon/evaluation/canonical_run.py:778-830` (min-of-3 estimator) + `latency_ceilings.py:21-30`
- **Finding**: Live host-contention sensitivity confirmed and well within design. A
  freshly-produced artifact measured ensemble p50 between 201.6ms (lightly loaded) and 458.7ms
  (heavy concurrent load from running the perf suite + multiple analyses in parallel) — both
  under the 500ms p50 ceiling but the latter is within ~8% of it. This is exactly the
  wall-clock-load-sensitivity the registry docstring (latency_ceilings.py:21-30) and S7-04
  grounding anticipate (a saturated 10-worker host tripled a single-shot p50 to ~275ms; hence the
  min-of-3 estimator + headroom). On a quiet release host the p50 is ~80-90ms. No action: the
  budget holds even under adversarial local contention. Worth noting only so a future maintainer
  who sees a ~460ms p50 in a noisy CI run does not mistake it for a regression — it is the
  documented contention envelope, not a breach.
- **Suggested resolution**: None required. Optionally, for the eventual real RC certification
  run, capture the latency_summary on a quiet/dedicated host so the recorded artifact reflects
  the ~80-90ms quiet-env p50 rather than a contention-inflated value.

#### CQ-RC-05 — metrics/core.py plugin compute() stubs missing docstrings — `dev-assist-development-code-quality`

- **Location**: `src/pii_anon/metrics/core.py:16,24,39,51,65,79,91` (7 plugin compute() stubs)
- **Finding**: Seven MetricPlugin subclasses (SpanFBetaMetric, LeakageAtTMetric,
  BoundaryLossMetric, TokenStabilityMetric, LLMLeakageMetric, FairnessGapMetric plus the base
  MetricPlugin.compute() abstract) are missing per-method docstrings. These classes lack
  module-level class docstrings. This pre-existing code is not new to the program's sprint work
  but the module is in the public src tree. Not blocking at release scale given ruff D rules are
  not in the enforced CI select set.
- **Suggested resolution**: Track as a contributor-readiness gate item. When enabling
  `ruff --select D` in CI, add a batch of one-line class and method docstrings to
  src/pii_anon/metrics/core.py.

#### SEC-RC-02 — committed non-production key material correctly scoped and labeled — `dev-assist-development-security-sast`

- **Location**: `src/pii_anon/moe_gate_signing.py:78-102`; `src/pii_anon/tokenization/encrypted_store.py:164-212`
- **Finding**: The committed non-production signing/test key material is correctly scoped and
  clearly labeled, but it is genuine bytes in source: moe_gate_signing.py:83 _TEST_HMAC_KEY =
  b'S2-05-NON-PRODUCTION-TEST-KEY-DO-NOT-USE-IN-PROD' (env-overridable via
  PII_ANON_GATE_SIGNING_KEY) and the StaticTestKeyProvider (encrypted_store.py:164) which takes
  raw KEK bytes in-process. Both are explicitly AGENT_SIMULATED stand-ins with a Pass-2
  real-custody (KMS/HSM/keyring) drop-in seam behind the EnvelopeKeyProvider/key-provider
  protocol, perform no networking/subprocess, and are never claimed as real custody. The PII
  taxonomy/dataset fixtures (taxonomy.py:567 — a synthetic OpenAI-style key sample and a
  synthetic AWS-style access-key-ID sample, literal prefixes not reproduced here;
  generator.py:189 _fake_api_key) are deterministic synthetic detector training/eval data, not
  live credentials. No real secret is committed anywhere in the diff. This is the honest
  AGENT_SIMULATED key-custody caveat, consistent with the documented Pass-2 roadmap; flagged
  only because secret-shaped strings exist in source and a naive secret scanner would surface
  them.
- **Suggested resolution**: No action. The env-overridable provider seams are the correct
  pattern; keep the Pass-2 real-custody migration tracked. The synthetic taxonomy/dataset
  credential shapes must remain (they are detector test data) — exclude them from any future
  secret-scanner allow-list rather than removing them.

#### CQ-RC-06 — `Any` annotations on optional-dep adapters lack justifying comments — `dev-assist-development-code-quality`

- **Location**: `src/pii_anon/pipeline.py:68,90,96` + `bridge.py:200-201` (Any without justifying comment on optional-dep adapters)
- **Finding**: Several production-path parameters use `Any` annotation for optional-dependency
  adapter objects (DataFrame, eval_framework) without a co-located justifying comment. The
  moe_gate_signing.py Any usage is justified by the lazy-import crypto surface. The pipeline.py
  and bridge.py usages appear to be structural (optional external objects that have no shared
  base class across optional deps), which is a valid reason, but the intent is not stated in a
  comment.
- **Suggested resolution**: Add a brief "# Any: optional external object (pandas/eval_framework)
  — no shared Protocol at this layer" comment at each Any-annotated parameter. Does not block
  release; track for a follow-up cleanup commit.

#### TRACE-RC-05 — mixed test-naming convention (ID-in-name vs docstring-mapped) — `dev-assist-development-traceability`

- **Location**: `tests/test_native_readers.py:3`
- **Finding**: Mixed test-naming convention across the program. Earlier sprints embed the ID in
  the function name (e.g. test_fr008_*, test_nfr006_*, test_fr028_*); later sprints (S4-03
  selective-risk, S6-01 query-aware, S6-04 BYO, S7-01 native-readers, S7-03
  multilingual-fairness, S5-02 tier3, S5-03 MIA, S7-04 latency) use acceptance-criterion function
  names (test_a1..test_aN) with the FR/NFR mapped in the module docstring (lines 3-9) +
  per-criterion comments instead. The forward chain is fully traceable in every file (verified:
  each declared ID appears in its test file's docstring/comments), so this does not break
  traceability — but a grep keyed only on "def test_fr_NNN" under-counts coverage and a future
  reviewer/script should scan file content, not just function names.
- **Suggested resolution**: Optional: adopt one convention going forward (ID-in-name is the most
  script-friendly), or document in the test-architecture that docstring+comment ID-mapping is the
  accepted alternative. No change needed for this RC.

## Cross-Reviewer Pattern Detection

- **Aggregator output: none.** `cross_reviewer_patterns` is empty — no formal multi-reviewer
  pattern was recorded by the aggregation pass.
- **Scribe note (non-binding; alters no severity and no verdict):** a convergence is nonetheless
  visible in the merged findings on one root cause — *the canonical traceability/NFR artifacts
  lag the SO-08..SO-23 sign-off ledger*. Three reviewers touch it independently: traceability
  TRACE-RC-01 (MAJOR, Stage-2 matrix never backfilled) and TRACE-RC-03 (Stage-5 NFR matrix
  stale); performance-benchmark PERF-RC-01 (dual "NFR-001" numbering against the same stale
  matrix); requirements-coverage REQ-RC-03 caveat (5) (stale Stage-5 matrix superseded by the SO
  ledger). One docs-only backfill/refresh pass — TRACE-RC-01 + TRACE-RC-03, folding in the
  TRACE-RC-04 (FR-028 4-vs-6 annotation), TRACE-RC-02 (FR-027 deferred line), and REQ-RC-02
  (FR-017 own-ID) annotations — closes the cluster together.
- Single-reviewer cluster, for the backlog (not a cross-reviewer pattern): code-quality
  docstring/D-rule coverage (CQ-RC-03 / CQ-RC-04 / CQ-RC-05) fronts the future S8
  contributor-readiness gate.

## Refinement Iteration Record

| Iteration | Date | Findings status pre-iteration (S/C/MAJ/MIN/OBS) | Refinement applied | Findings status post-iteration |
|---|---|---|---|---|
| 1 | 2026-06-10 | 0/0/1/6/14 | pending — executor amendment per Next Action below | to be recorded at the iteration-2 re-dispatch |

(After 3 unresolved iterations → 3-strike escalation per
`five-gate-cascade.md#3-strike-escalation-rule`.)

## Pass 2 Commitment Check (DIVERGED stories only)

**Not binding at this gate — zero MUST FR/NFR carries `provisional_status: DIVERGED`** (verified
by requirements-coverage REQ-RC-01: `requirements-document.md:107` records "0 DIVERGED, 0
INSUFFICIENT-EVIDENCE"; NFR-017 is REVISE-LOOSER on a SHOULD, not DIVERGED).

- [ ] Pass 2 commitment scheduled in DoD — N/A (no DIVERGED story/epic/sprint in scope)
- [ ] Pass 2 outcome logged in `traceability-matrix.md` Status Change Log — N/A (note: the Status
  Change Log section itself is owed by TRACE-RC-01)
- [ ] `provisional_status` updated on Pass 2 outcome — N/A

The program-wide `AGENT_SIMULATED` → real-user Pass-2 roadmap is carried as an honest release
caveat (caveat #2 in REQ-RC-03), not as a verification blocker for a local-only RC.

## Verdict + Next Action

**Verdict: REQUEST_CHANGES** (iteration 1 of 3)

**Next action (REQUEST_CHANGES branch)**: executor amends → re-dispatch the SAME 6-reviewer set →
iteration count increments to 2 of 3.

Amendment scope for the executor:

1. **Gate-clearing (the sole MAJOR)** — TRACE-RC-01: backfill `traceability-matrix.md` with
   Story/Test columns for all 30 DONE stories (FR/NFR → claiming story → in-tree test pin → DC)
   + add a Status Change Log. Docs-only; no code touched; can land before tagging.
2. **Fold into the same amendment pass (RC-mechanics-coupled MINORs)**:
   - CQ-RC-01 — add the `make rc-tag` local-only target + push-guard warning (an accidentally
     pushed rc tag silently publishes to TestPyPI via release.yml).
   - CQ-RC-02 — flip the trove classifier to "Development Status :: 4 - Beta" in the
     1.4.0 → 1.5.0rc1 version-bump commit (revert at final 1.5.0).
   - TRACE-RC-02 — move FR-027 from S6-03 `Implements` to an explicit "Deferred (Pass-2)" line
     BEFORE the matrix backfill records it.
   - TRACE-RC-03 — refresh the Stage-5 NFR matrix statuses or add the
     superseded-by-the-SO-ledger banner (pairs with TRACE-RC-01).
   - CQ-RC-03 — one-line docstrings on the reid.py attack methods (cheap; fold in
     opportunistically).
3. **Explicitly deferred by the flagging reviewer (no RC action)**: PERF-RC-01 (Pass-2 NFR-doc
   cross-reference for the superseded 200ms figure).
4. **OBSERVATIONs (14)**: no action required at this gate. Feed-forward routing: TRACE-RC-04 +
   REQ-RC-02 annotations land with the TRACE-RC-01 backfill; CQ-RC-04 / CQ-RC-05 / CQ-RC-06 +
   AX-RC-01 + TRACE-RC-05 → the S8 contributor-readiness backlog; AX-RC-02 → a future docs pass;
   SEC-RC-01 → re-run pip-audit at the future publish gate; SEC-RC-02 → keep the Pass-2
   real-custody migration tracked; PERF-RC-03 → capture the certification latency_summary on a
   quiet host.

Guardrails for the amendment pass (carried forward from the reviewer record):

- All remediation is docs/build-metadata/Makefile only — the protected gate md5s must remain
  byte-identical (`competitive_supremacy.py` `3b842e81…`, `canonical_run.py` `d8f0f80e…`,
  `competitor_compare.py` `7cae16c8…`). No `competitive_supremacy.py` change is in scope, so no
  SDO adversarial close is triggered by this remediation.
- Per REQ-RC-03: record the honest caveat set verbatim in the release record and do NOT mark
  FR-032-native or FR-034-corpus fully-verified at this gate.
- The SDO NOT_YET / binding-G6 verdict is an honest outcome, not a gate defect — remediation must
  not "fix" F2 or weaken G6.

Branch reference (for the record):

- APPROVE → close scope; transition status; update MANIFEST.
- **REQUEST_CHANGES → executor amends; re-dispatch the same reviewer set; iteration count
  incremented.** ← this gate
- HALT_GATE → orchestrator stops; surface the SHOWSTOPPER to the user; must be resolved before
  proceeding.
- ESCALATE → 3-strike escalation; user adjudication required.

---

**Template version**: 1.0 (developer-assistant v0.4.0) — scribe synthesis, release gate
v1.5.0-rc1, 2026-06-10.
