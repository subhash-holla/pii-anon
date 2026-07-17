# Story Gate Synthesis — S7-03 (multilingual fairness)

- **Gate type:** story
- **Scope:** S7-03
- **Iteration:** 1
- **Date:** 2026-06-10
- **Gate integrity:** **complete** — 5/5 reviewers reported; no missing reviewers.

## Reviewer Set

| Reviewer | Verdict | SHOWSTOPPER | CATASTROPHIC | MAJOR | MINOR | OBSERVATION |
|---|---|---:|---:|---:|---:|---:|
| dev-assist-development-code-quality | APPROVE | 0 | 0 | 0 | 1 | 1 |
| dev-assist-development-security-sast | APPROVE | 0 | 0 | 0 | 0 | 0 |
| dev-assist-development-requirements-coverage | APPROVE | 0 | 0 | 0 | 0 | 1 |
| dev-assist-development-traceability | APPROVE | 0 | 0 | 0 | 0 | 2 |
| dev-assist-development-axiom-compliance | APPROVE | 0 | 0 | 0 | 0 | 5 |
| **Totals** | — | **0** | **0** | **0** | **1** | **9** |

## Aggregation Rule + Aggregate Verdict

**Rule:** any SHOWSTOPPER or CATASTROPHIC from any reviewer ⇒ HALT_GATE; otherwise any
MAJOR (equivalently, any reviewer verdict of REQUEST_CHANGES) ⇒ REQUEST_CHANGES;
otherwise (MINOR/OBSERVATION only) ⇒ APPROVE.

**Aggregate verdict: APPROVE** — 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 MAJOR / 1 MINOR /
9 OBSERVATION. All five reviewers return APPROVE on iteration 1 — a first-pass clean
gate. The single MINOR is a typing-strictness advisory (a `Literal` annotation), and
five of the nine OBSERVATIONs are axiom-compliance *upheld* confirmations (positive
evidence records, not defects). Security-sast reports zero findings of any severity.

## Merged Findings

### MINOR (1)

#### MIN-1 — code-quality — `FairnessGateReport.verdict` typed as bare `str` rather than `Literal`

- **Location:** `src/pii_anon/eval_framework/metrics/fairness_gate.py:67`
- **Description:** The `verdict` field on `FairnessGateReport` is typed as bare `str`
  with the allowed values documented only as an inline comment
  (`# "PASS" | "FAIL" | "INSUFFICIENT_POWER"`). Using
  `Literal["PASS", "FAIL", "INSUFFICIENT_POWER"]` would give mypy exhaustiveness
  guarantees at call sites.
- **Suggested resolution:** Import `Literal` from `typing` and change the annotation to
  `verdict: Literal["PASS", "FAIL", "INSUFFICIENT_POWER"]`. The inline comment can then
  be removed. Mypy is currently clean in both modes so this is advisory — the bare-str
  pattern is consistent with sibling modules in the codebase.

### OBSERVATION (9)

#### OBS-1 — traceability — Stale S1 summary-table row in development-log.md (per-story entries are authoritative)

- **Location:** `dev-assist-artifacts/04-development/development-log.md:16`
- **Description:** Doc-hygiene only (outside the traceability dimension; recorded as a
  note): development-log.md line 16 (the S1 summary-table row) still reads "S1-01 DONE;
  S1-02/03/04 TODO", which is stale — the authoritative per-story log entries at
  lines 42 (S1-03 GREEN f1638a5) and 44 (S1-05 GREEN 6dbb37b) record both S7-03
  dependencies as DONE. The dependency-DONE precondition for S7-03 is satisfied; only
  the older summary cell is out of date.
- **Suggested resolution:** Refresh the S1 summary-table status cell to match the
  per-story DONE entries (lines 42/44). Does not block S7-03.

#### OBS-2 — requirements-coverage — NFR-025 matrix row could distinguish "primitive shipped" from "corpus-scale verified"

- **Location:** `dev-assist-artifacts/05-testing/03-nfr-verification/nfr-verification-matrix.md:23`
- **Description:** The NFR verification matrix row for NFR-025 still reads
  "DEFERRED | S7-03". That is correct (the corpus-scale worst-group fairness number
  remains DATA Pass-2 via the eval-data S6 blocker, so NFR-025 is not yet fully
  VERIFIED), but after this story lands the matrix could optionally be annotated to
  record that the GATE PRIMITIVE is now in-tree and tested (A6–A10) — distinguishing
  "primitive shipped" from "corpus-scale verified", matching how NFR-009 is annotated
  "VERIFIED-in-tree ... full-census = Pass-2". Purely a snapshot-precision note; no
  coverage gap — the deferral is valid and tracked, and updating the traceability
  matrix is out of this story's owned-files scope.
- **Suggested resolution:** Optional at epic/sprint gate: annotate the NFR-025 matrix
  row to note the gate primitive is in-tree (A6–A10) with corpus-scale deferred to
  DATA Pass-2 (eval-data S6), mirroring the NFR-009 "VERIFIED-in-tree / full-census
  Pass-2" phrasing. Not required for story APPROVE.

#### OBS-3 — axiom-compliance — AX-pii-anon-002 / AX-002 (deterministic-reproducible) UPHELD; exact-value anchors honored

- **Location:** `src/pii_anon/engines/regex/confidence.py:325-340` (precomputed map);
  `src/pii_anon/eval_framework/metrics/fairness_gate.py:142-143,160-167`; tests A5/A7
- **Description:** AX-pii-anon-002 / AX-002 (deterministic-reproducible) UPHELD. The
  containment pass iterates a precomputed, import-time-frozen, sorted tuple map
  (`_NON_LATIN_CONTEXT_KW`) — no unseeded random/uuid/time, allocation-light, pure.
  `evaluate_language_fairness` sorts powered/unpowered/violating groups
  deterministically and computes `worst_group_recall_gap` from integer-count-derived
  recalls; A7 uses dyadic-rational (8ths) fixtures so the boundary float arithmetic
  (1.0 vs 7/8 at gap_threshold 0.125) is EXACT — the recurring exact-value-anchor
  discipline (RISK-6) is honored, not bounds. A5 pins 5 byte-identical replays of
  `has_context_words`/`adjust_confidence`/the gate report (passing). The doc note that
  runtime mutation of CONTEXT_WORDS won't refresh the import-time snapshot is correctly
  disclosed (not a supported surface).
- **Suggested resolution:** None — determinism axiom upheld.

#### OBS-4 — axiom-compliance — AX-pii-anon-003 (ensemble-recall-floor) UPHELD by construction; IBAN suppression gate byte-identical

- **Location:** `src/pii_anon/engines/regex/confidence.py:387-401` (has_context_words
  containment fallthrough) + `:432-433` (adjust_confidence monotonic boost);
  `src/pii_anon/engines/regex_adapter.py:674` (IBAN gate unaffected)
- **Description:** AX-pii-anon-003 (ensemble-recall-floor-guarantee) UPHELD by
  construction. The FR-038 containment pass in `has_context_words` is purely additive
  and monotonic: it executes only AFTER the existing token-set intersection misses and
  can flip the result False->True but NEVER True->False (it only adds substring matches
  for non-Latin keywords). The sole non-inverted consumer is `adjust_confidence`
  (confidence.py:432), where True -> min(CONFIDENCE_CAP, base + CONTEXT_BOOST) is a
  monotonic non-decreasing confidence adjustment (CONTEXT_BOOST=+0.10). The one
  inverted-polarity caller (regex_adapter.py:674, where `has_context_words('IBAN', ...)`
  True KEEPS a SWIFT/BIC span) is provably unaffected: IBAN carries only Latin keywords
  (bank/wire/iban/transfer/international/bic) so IBAN is absent from
  `_NON_LATIN_CONTEXT_KW` and the new fallthrough never adds a True there — that
  suppression gate is byte-identical. Net effect: no detection path can be suppressed
  and no early-exit is introduced; the shared regex layer's context recall on
  CJK/Hangul/Arabic strengthens. entities(ensemble) superset-of entities(shared) holds.
- **Suggested resolution:** None — no action required; recall-floor axiom upheld.

#### OBS-5 — axiom-compliance — Fail-closed fairness semantics: INSUFFICIENT_POWER (never PASS) on <2 powered groups

- **Location:** `src/pii_anon/eval_framework/metrics/fairness_gate.py:108-128` (input
  validation), `:145-158` (fail-closed <2 powered), `:22-24` (observational unpowered);
  tests A8/A9
- **Description:** Fail-closed fairness semantics uphold the spirit of the program's
  no-fabrication invariant at the gate boundary: `evaluate_language_fairness` returns
  INSUFFICIENT_POWER (`worst_group_recall_gap=None`, never PASS) on <2 powered groups —
  A9 pins BOTH zero powered and exactly-one powered to refuse PASS — and raises
  domain-named ValueError on corrupt input (empty slices, gap_threshold outside [0,1],
  power_floor<1, duplicate language). An unpowered cohort can therefore never
  manufacture a fairness PASS without >=2 powered groups of evidence; unpowered groups
  are carried observationally in `unpowered_groups` + `per_language_recall` (visible,
  never silently dropped). This is reasoning-visibility-adjacent (AX-pii-anon-005
  flavor): the verdict, the powered/unpowered partition, the per-language recalls, and
  the named violators are all surfaced on `FairnessGateReport`.
- **Suggested resolution:** None — fail-closed-with-evidence is the correct posture for
  a fairness gate.

#### OBS-6 — axiom-compliance — One-recall-definition consistency confirmed; stale "base.py" prose reference in story text

- **Location:** `src/pii_anon/eval_framework/metrics/fairness_gate.py:40` (import) +
  `:134`; test A10 (`tests/test_multilingual_fairness.py:254-266`)
- **Description:** One-recall-definition consistency confirmed: fairness_gate.py
  imports `_aligned_prf` from `.span_metrics` — the IDENTICAL primitive used by
  fairness_metrics.py, streaming.py, and span_metrics.py's own scorers (and the S1-03
  per-language recall-floor gate). A10 contract-pins exact per-language recall equality
  between the gate and a direct `_aligned_prf(MatchMode.STRICT)` call. Minor
  prose-vs-code drift OUTSIDE the axiom dimension (flagged only as an observation, no
  cross-role policing): the story text (line 11) and the dispatch say `_aligned_prf`
  lives in `metrics/base.py`, but it actually lives in `metrics/span_metrics.py` and
  base.py does not re-export it — the production import and the A10 test both correctly
  use `.span_metrics`, so there is no code defect, only a stale doc reference for
  code-quality/traceability to note if they wish.
- **Suggested resolution:** Optionally correct the story prose "base.py" ->
  "span_metrics.py" (code-quality/traceability dimension; not blocking).

#### OBS-7 — traceability — Anchor-only test names (A1–A11): FR/NFR -> Test trace not greppable by ID at function-name level

- **Location:** `tests/test_multilingual_fairness.py` (def test_a1.._a11)
- **Description:** S7-03 test functions use anchor-only names (`test_a1_...` through
  `test_a11_...`) with FR/NFR IDs carried only in the module docstring, section
  comments (e.g. `# A1 / A2 — FR-038`,
  `# A6 / A7 / A8 / A9 — the fairness gate (FR-039 / NFR-025 / NFR-004)`), and
  per-anchor docstrings — not in any function name. A grep over test function NAMES for
  `fr_038`/`fr038`/`nfr_025` therefore returns nothing, so the FR/NFR -> Test trace is
  not greppable by ID at the function-name level. This is NOT a violation: the program
  uses both conventions extensively (240 test fns embed FR/NFR/AX IDs in-name; 249 use
  the test_aN_ anchor-only style across 14 files), and the anchor-only-with-docstring-IDs
  form is the established current pattern for feature stories — the same-sprint sibling
  S7-01 (test_native_readers.py, test_a1_.../test_a2_... with 17 FR references in
  docstrings) and S6-01 (test_query_aware_masking.py) use it identically. The dependency
  stories S1-03/S1-05 happen to use the greppable test_fr_NNN/test_nfr_NNN form, which
  is the only reason this is worth noting.
- **Suggested resolution:** Optional (no change required for this gate): for
  cross-story greppability, consider the test_fr_038_a1_... / test_nfr_025_a6_... naming
  form used by S1-03/S1-05 in any future multilingual/fairness tests, or maintain the
  per-story FR->anchor map in the story file (already present via the
  Implements/Traces/A1–A11 mapping).

#### OBS-8 — axiom-compliance — AX-pii-anon-001 / AX-001 (synthetic-only-no-real-PII) UPHELD for the new fixtures

- **Location:** `tests/test_multilingual_fairness.py:44-64` (_spans/_slice offset-only
  fixtures), `:77`, `:109`, `:139`
- **Description:** AX-pii-anon-001 / AX-001 (synthetic-only-no-real-PII) UPHELD for the
  new test fixtures. The fairness fixtures are offset-only synthetic LabeledSpans
  (entity_type + integer start/end, no values). The only numeric literals are the
  placeholder phone '000-0000' and SSN-context '=000'; pattern scan found zero
  Luhn-valid card numbers, zero structurally-valid SSNs, and zero real email shapes.
  Non-Latin literals in the keyword assertions (电话/メール/전화/هاتف/信用卡 etc.) are
  dictionary context terms, not personal data. (Note: this axiom's primary
  reviewer_hook is security-sast, severity_default CATASTROPHIC — flagged here only as
  an in-scope OBSERVATION confirming nothing in this diff trips it.)
- **Suggested resolution:** None — synthetic-only axiom upheld; defer the authoritative
  real-PII scan to security-sast.

#### OBS-9 — code-quality — Ax-anchor test naming is convention-consistent; FR/NFR linkage carried by the story's acceptance table

- **Location:** `tests/test_multilingual_fairness.py:71-291`
- **Description:** Test function names use acceptance-criterion anchors (e.g.,
  `test_a1_zh_phone_context_fires_and_boosts_exactly`) rather than the reviewer-pattern
  convention of embedding FR/NFR IDs in the function name (e.g.,
  `test_fr038_zh_phone_context_fires`). The names are readable and map unambiguously to
  the story's Ax table, which itself maps to FR-038/FR-039/NFR-025. No action required;
  the story's acceptance table provides the FR/NFR linkage.
- **Suggested resolution:** No change required. The Ax-anchor naming is consistent with
  this project's story-gate test convention and each test's docstring labels the FR/NFR
  clearly.

## Cross-Reviewer Pattern Detection

The aggregator flagged no formal cross-reviewer patterns (`cross_reviewer_patterns: []`),
and at 1 MINOR / 9 OBSERVATION there is no convergent defect signal. Scribe notes for
the record:

- **P1 — Independent convergence on the anchor-only naming convention (benign;
  code-quality OBS-9 × traceability OBS-7).** Two reviewers independently observed
  that the A1–A11 anchor-only test names are not greppable by FR/NFR ID at the
  function-name level, and BOTH independently concluded it is the established,
  convention-consistent pattern for feature stories (same form as the same-sprint
  sibling S7-01 and S6-01). Convergent confirmation of a style fact, not a defect; the
  optional greppability suggestion stands for future fairness tests.
- **P2 — Doc-hygiene bundle (3 items, 3 reviewers; none blocking, none code).**
  (a) stale S1 summary-table cell in development-log.md (traceability OBS-1);
  (b) optional NFR-025 matrix annotation distinguishing "primitive in-tree (A6–A10)"
  from "corpus-scale verified — DATA Pass-2" at the epic/sprint gate
  (requirements-coverage OBS-2); (c) story-prose "base.py" -> "span_metrics.py"
  correction (axiom-compliance OBS-6). Carry all three as story-close /
  epic-gate hygiene actions.
- **P3 — Positive-evidence density (axiom-compliance, 5/5 OBS are UPHELD records).**
  All four program axioms touched by this diff are confirmed upheld with cited
  mechanism (frozen keyword map determinism; additive-monotonic containment preserving
  the recall floor with the IBAN inverted-polarity caller proven byte-identical;
  fail-closed INSUFFICIENT_POWER semantics; offset-only synthetic fixtures). Notably,
  the recurring exact-value-anchor lesson (RISK-6; bit S5-02/S5-03/S6-01) is honored
  this time by design — A7's dyadic-rational (8ths) fixtures make the gap-threshold
  boundary arithmetic exact rather than bound-asserted.

## Verdict + Next Action

**Aggregate verdict: APPROVE** (iteration 1, first pass).

Per the gate protocol:

- **APPROVE → close scope / update MANIFEST — TAKEN.**
- REQUEST_CHANGES → executor amends + re-dispatch the same reviewer set — *not taken*.
- HALT_GATE → stop, surface SHOWSTOPPER to user — *not applicable* (0 SHOWSTOPPER /
  0 CATASTROPHIC).

**Close-out actions:**

1. Mark S7-03 DONE; update `dev-assist-artifacts/MANIFEST.md` with the story-gate
   result (5/5 APPROVE, iteration 1, 0 MAJ / 1 MIN / 9 OBS).
2. **MIN-1 (advisory, non-blocking):** adopt
   `verdict: Literal["PASS", "FAIL", "INSUFFICIENT_POWER"]` in
   `fairness_gate.py` at story close or on next touch of the module.
3. Carry the P2 doc-hygiene bundle: refresh the development-log S1 summary cell
   (OBS-1); at the epic/sprint gate optionally annotate the NFR-025 matrix row
   "primitive in-tree (A6–A10) / corpus-scale DATA Pass-2" (OBS-2); correct the story
   prose `_aligned_prf` home to `metrics/span_metrics.py` (OBS-6).
4. NFR-025 remains DEFERRED (corpus-scale worst-group fairness is DATA Pass-2 via the
   eval-data S6 blocker) — the deferral is valid, tracked, and unchanged by this gate.

**Scope note:** S7-03 touches the shared regex confidence layer (an additive,
monotonic containment pass), a new standalone fairness-gate metric module, and its
tests — no change to `competitive_supremacy.py`, the canonical-run producer, or any
`gate_v1.json` control-path artifact — so the story gate closes without a mandatory
SDO adversarial close (consistent with the standing catch-net rule: feature-surface
stories that do not touch the gate need no SDO close).
