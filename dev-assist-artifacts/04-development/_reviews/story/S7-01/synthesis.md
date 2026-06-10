# Story Gate Synthesis — S7-01 (native-format readers)

- **Gate type:** story
- **Scope:** S7-01
- **Iteration:** 2
- **Date:** 2026-06-10
- **Gate integrity:** **complete** — 5/5 reviewers reported; no missing reviewers.

## Reviewer Set

| Reviewer | Verdict | SHOWSTOPPER | CATASTROPHIC | MAJOR | MINOR | OBSERVATION |
|---|---|---:|---:|---:|---:|---:|
| dev-assist-development-code-quality | APPROVE | 0 | 0 | 0 | 0 | 0 |
| dev-assist-development-security-sast | APPROVE | 0 | 0 | 0 | 0 | 1 |
| dev-assist-development-requirements-coverage | APPROVE | 0 | 0 | 0 | 0 | 2 |
| dev-assist-development-traceability | APPROVE | 0 | 0 | 0 | 1 | 1 |
| dev-assist-development-axiom-compliance | APPROVE | 0 | 0 | 0 | 0 | 0 |
| **Totals** | — | **0** | **0** | **0** | **1** | **4** |

## Aggregation Rule + Aggregate Verdict

**Rule:** any SHOWSTOPPER or CATASTROPHIC from any reviewer ⇒ HALT_GATE; otherwise any
MAJOR (equivalently, any reviewer verdict of REQUEST_CHANGES) ⇒ REQUEST_CHANGES;
otherwise (MINOR/OBSERVATION only) ⇒ APPROVE.

**Aggregate verdict: APPROVE** — 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 MAJOR / 1 MINOR /
4 OBSERVATION. All five reviewers return APPROVE. The iteration-1 blocking set is
cleared: MAJ-1 (unbounded FlateDecode inflate / zip-bomb ceiling) and MAJ-2
(new-module coverage below the §12 ≥90% DoD bar) were remediated in commit `258d3ec`,
which added the gate-remediation test anchors A14 (zip-bomb FlateDecode ceiling),
A15/A16/A16b (parser + decoder edge paths), and A17 (registry fail-closed); the
iteration-1 MIN-1 docstring gap no longer appears in any reviewer output. Code-quality
and security-sast — the two iteration-1 REQUEST_CHANGES lanes — now report zero MAJOR.

## Merged Findings

### MINOR (1)

#### MIN-1 — traceability — Story-doc anchor list lags the remediation test anchors (A14–A17)

- **Location:** `dev-assist-artifacts/04-development/02-stories/sprint-7/S7-01-native-format-readers.md`
  (sections 3 and 12) vs `tests/test_native_readers.py`
- **Description:** Story-doc anchor list lags the test file. The remediation commit
  `258d3ec` added test anchors A14 (zip-bomb FlateDecode ceiling), A15/A16/A16b
  (parser+decoder edge paths) and A17 (registry fail-closed) to
  `tests/test_native_readers.py`, but the story's section-3 Given/When/Then acceptance
  list and the section-12 Definition of Done still enumerate only A1–A13 ("all A1–A13
  green"). The new anchors trace cleanly to iteration-1 findings (A14 → security-sast
  MAJOR-1; A15/A16/A16b/A17 → code-quality MAJOR-2 coverage floor) rather than to any
  new FR/NFR, and the full FR/NFR → anchor mapping for the declared IDs
  (FR-031..036, NFR-023/026, AX-001/002) is intact, so this is a
  doc-freshness/cross-link advisory only — the same anchor-name-convention
  carry-forward already logged as an iteration-1 OBSERVATION. No FR/NFR claim is left
  untested and no new traceability row is owed (these are remediation tests, not
  requirements).
- **Suggested resolution:** On story close, refresh section-3/section-12 to read
  "A1–A17" (or add a one-line "A14–A17 = gate-remediation anchors" note) so the
  story's documented acceptance set matches the shipped test anchors. Non-blocking for
  this gate.

### OBSERVATION (4)

#### OBS-1 — traceability — AX-001/AX-002 resolve to project axioms, not FR/NFR rows (convention-consistent)

- **Location:** `dev-assist-artifacts/00-axioms/project-axioms.yaml` (referenced via
  `MANIFEST.md:144`); `tests/test_native_readers.py` A5
- **Description:** AX-001 and AX-002 resolve to project axioms in
  `00-axioms/project-axioms.yaml` (synthetic-only-no-real-pii;
  deterministic-pseudonymization), per `dev-assist-artifacts/MANIFEST.md`, NOT to
  FR/NFR rows in requirements-document.md (AX-002 has 0 literal hits there). This is
  the established convention used across SO-08/SO-13/SO-19/SO-20 sign-offs, so the
  AX-002 absence from requirements-document.md is correct — recorded only so a future
  reviewer does not mistake it for a non-existent-ID violation (would otherwise be
  SHOWSTOPPER pattern-2). The story's AX-001 (synthetic fixtures) and AX-002 (A5
  determinism) claims are both backed by tests.
- **Suggested resolution:** No action — convention-consistent. Optional: cross-link
  the axioms file from the story's Traces row for one-hop verifiability.

#### OBS-2 — requirements-coverage — FR-037/NFR-022 (SHOULD) deferral properly tracked; carry into sprint snapshot

- **Location:** `dev-assist-artifacts/04-development/02-stories/sprint-7/S7-01-native-format-readers.md`
  (provisional_status; section 5 non-goals)
- **Description:** Carry-forward (no action; sprint-snapshot tracking). The
  SHOULD-level FR-037 / NFR-022 (full OS-matrix portability certification) remains
  explicitly deferred to Pass-2. The deferral is properly tracked: it appears in the
  story provisional_status, in non-goals (section 5), and the benchmark-layer
  machinery (`make benchmark-portable-*`) is noted as already existing. No MUST is
  gated on it, so this is not a coverage gap — recording it so the epic/sprint
  coverage snapshot carries the open SHOULD with its successor seam.
- **Suggested resolution:** No action this iteration. Ensure the FR-037/NFR-022
  portability-certification successor is listed in the Sprint-7 Tier-coverage
  snapshot as deferred-with-successor.

#### OBS-3 — security-sast — OCR/DICOM extras not CVE-scannable at story scope; pip-audit owed at the epic gate

- **Location:** `pyproject.toml:140-141` (`[project.optional-dependencies]` ocr/dicom)
- **Description:** The OCR/DICOM extras (pytesseract>=0.3, Pillow>=10, pydicom>=2.4)
  are not installed in this env and so cannot be CVE-scanned at this story gate. They
  are optional, lazy-imported, loud-on-absent extras (default-install attack surface
  unchanged) and all version-range pinned with permissive licenses, so there is no
  current breach. Pillow in particular has a historically active CVE stream; the
  floor `>=10` with no upper pin means a transitive-resolved Pillow should be
  asserted clean. This is the iteration-1 carry-forward OBSERVATION re-affirmed: run
  pip-audit on the ocr/dicom extras at the EPIC gate (where the extras are
  installable) before any release. No action required at the story gate.
- **Suggested resolution:** At the epic gate, install the ocr+dicom extras and run
  pip-audit; assert no HIGH/CRITICAL CVE in pytesseract/Pillow/pydicom or their
  transitive closure before release.

#### OBS-4 — requirements-coverage — FR-032/FR-034 verified at the representative bar only; keep MUST-board successors open

- **Location:** `src/pii_anon/ingestion/native.py:74-76` (supports_reconstruction
  default False); `tests/test_native_readers.py` A10/A12
- **Description:** Carry-forward (no action; documents the verified split). The MUST
  FR-032 (byte-for-byte round-trip) and MUST FR-034 (per-modality recall) are
  satisfied at this story's planned bar in the REAL/representative slice only:
  FR-032 round-trip is REAL+pinned for text formats (A12) while native-format
  reconstruction ships honestly as `supports_reconstruction=False` (A2) deferred via
  `# SWITCH-POINT`; FR-034 is a representative in-tree harness (A10) with corpus
  scale deferred via `# SWITCH-POINT(DATA)`. Both deferrals carry tracked
  successors. This is the planned representative bar for S7-01, not a silent gap,
  but the native-reconstruction (FR-032-full) and corpus-recall (FR-034-full)
  successors must remain open MUST-board entries until their Pass-2 stories close.
- **Suggested resolution:** No action this iteration. Keep FR-032 (native
  reconstruction) and FR-034 (corpus-scale recall) as open MUST-board rows flagged
  representative-this-sprint with named Pass-2 successors; do not mark either MUST
  fully-verified at the release gate on the strength of S7-01 alone.

## Cross-Reviewer Pattern Detection

The aggregator flagged no formal cross-reviewer patterns (`cross_reviewer_patterns: []`),
and at 1 MINOR / 4 OBSERVATION there is no convergent defect signal. Scribe notes for
the record:

- **P1 — Remediation traced, not silently absorbed (traceability MIN-1 × iteration-1
  MAJ-1/MAJ-2).** The iteration-1 blocking findings resolve to named, attributable
  test anchors (A14 ← security-sast MAJ-1; A15/A16/A16b/A17 ← code-quality MAJ-2),
  so the remediation is independently auditable. The only residue is the
  story-document's stale A1–A13 enumeration — a doc-freshness fix at story close,
  not a code or coverage gap.
- **P2 — Epic-gate carry-forward bundle (3 items, 2 reviewers).** Forward-looking,
  none blocking at story scope: (a) pip-audit of the ocr/dicom extras in an
  extras-installed environment (security-sast OBS-3); (b) FR-037/NFR-022
  portability-certification successor in the Sprint-7 Tier-coverage snapshot
  (requirements-coverage OBS-2); (c) FR-032-full / FR-034-full MUST-board rows kept
  open with named Pass-2 successors (requirements-coverage OBS-4). Carry all three
  into the sprint/epic-gate snapshot so each is claimed before its bar.

## Verdict + Next Action

**Aggregate verdict: APPROVE** (iteration 2).

Per the gate protocol:

- **APPROVE → close scope / update MANIFEST — TAKEN.**
- REQUEST_CHANGES → executor amends + re-dispatch the same reviewer set — *not taken*.
- HALT_GATE → stop, surface SHOWSTOPPER to user — *not applicable* (0 SHOWSTOPPER /
  0 CATASTROPHIC).

**Close-out actions:**

1. Mark S7-01 DONE; update `dev-assist-artifacts/MANIFEST.md` with the story-gate
   result (5/5 APPROVE, iteration 2, 0 MAJ / 1 MIN / 4 OBS).
2. **MIN-1 (at story close, non-blocking):** refresh story sections 3 and 12 to
   enumerate A1–A17 (or add the one-line "A14–A17 = gate-remediation anchors" note).
3. Carry the P2 bundle into the Sprint-7 / epic-gate coverage snapshot:
   ocr/dicom-extras pip-audit; FR-037/NFR-022 deferred-with-successor; FR-032-full
   and FR-034-full open MUST-board rows.

**Scope note:** S7-01 touches ingestion readers only — no change to
`competitive_supremacy.py`, the canonical-run producer, or any `gate_v1.json`
control-path artifact — so the story gate closes without a mandatory SDO adversarial
close (consistent with the standing catch-net rule: feature-surface stories that do
not touch the gate need no SDO close).
