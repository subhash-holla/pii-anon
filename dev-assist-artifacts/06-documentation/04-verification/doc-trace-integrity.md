# doc-trace-integrity — pii-anon Stage 6 Documentation (D4 Verification, Axis 2 + honesty + a11y)

> **Wave D4 output.** Read-only verification of internal cross-link resolution, `## Sources`
> conformance against the D2 `source_mapping`, trace-ID resolution (no phantom FR/NFR/DC/SO/AX),
> the epistemic-honesty constraints (§4 of `doc-architecture.md`), and the a11y disposition.
> Coverage (Axis 1) + accuracy-against-code (Axis 3) are in the sibling `doc-coverage-audit.md`.
> **Read-only on all sources; fixes happen via D3 re-author loopback, not here.**
>
> Abbreviations: JRN=project-journey · CHG=changelog · ADR=architecture-and-adr · API=api-reference ·
> USR=user-operator-guide · CON=contributor-handbook.

---

## 0. Verdict (Axis 2 + honesty)

| Check | Verdict | Basis |
|---|---|---|
| **Internal cross-link resolution** | **PASS** | Every `docs/*.md` target (13), every sibling deliverable (6), every `dev-assist-artifacts/*` Sources path resolves to a real file. No dangling link. |
| **Trace-ID resolution (no phantom IDs)** | **PASS** | All cited FR/NFR/DC ≤ census maxima (FR-039/NFR-026/DC-15); SO cited ≤ SO-23 (23 real files in `_signoffs/`); AX cited ≤ AX-006. Zero phantom IDs. |
| **`## Sources` ↔ D2 `source_mapping` conformance** | **PASS-WITH-ONE-DEVIATION** | All six carry `## Methodology` + `## Sources`. One out-of-mapping source (CON cites MEMORY.md) — finding T-03. No fabricated *artifact* citation. |
| **Epistemic-honesty (§4 constraints 1–4)** | **PASS** | No real-user-validation claim; NOT_YET / G6-FAIL stated accurately everywhere; benchmark-summary.md + pii-rate-elo-value.md NOT cited as canonical anywhere. |
| **A11y (rendered doc-site WCAG)** | **N/A-WITH-REASON** | Markdown-only artifact set; no rendered doc-site ships in this RC (the sibling `pii-anon-doc` repo owns the render). D6 render deferred. The reused `dev-assist-testing-a11y-auditor` has no rendered target to audit. See §4. |

No SHOWSTOPPER/CATASTROPHIC finding originates on the trace/honesty axes. (The single CATASTROPHIC
driving the D3 loopback is F-01 in `doc-coverage-audit.md §3b` — a code-accuracy fabrication, not a trace
defect. T-03 below is its likely *root cause*: the handbook leaned on a non-canonical source for the very
section that fabricated the contract.)

---

## 1. Internal cross-link + Sources-path resolution

### 1a. `docs/*.md` targets referenced across the six deliverables — ALL RESOLVE

`quickstart.md` · `evaluate-your-pipeline.md` · `configuration.md` · `anonymization-vs-pseudonymization.md`
· `recall-floor.md` · `dependencies-and-platforms.md` · `api-reference.md` · `swarm-architecture.md` ·
`engine-plugin-guide.md` · `extend-swarm.md` · `release-guide.md` — **11/11 exist** under `docs/`.
`benchmark-summary.md` and `pii-rate-elo-value.md` also exist but are correctly **NOT cited as canonical**
(O-7 — see §3). No dangling `docs/` link.

### 1b. Sibling-deliverable cross-links — ALL RESOLVE

JRN/USR/ADR/API/CON cross-reference each other ("see `contributor-handbook.md`", "see
`architecture-and-adr.md`", API "Cross-reference: `docs/api-reference.md`, …"). All six target files exist in
`03-authoring/`. **6/6 resolve.** The `documentation-set.md` index they will all link to is a D5 build (not
yet present) — not a D4 dangling-link finding (it is correctly out of D3 scope per D2 §2).

### 1c. `dev-assist-artifacts/*` Sources-block paths — ALL RESOLVE

Spot-verified every distinct artifact path cited in the six `## Sources` blocks:
`MANIFEST.md` (top-level — cited correctly as `dev-assist-artifacts/MANIFEST.md`, NOT the wrong
`04-development/MANIFEST.md`), `_signoffs/SO-01..SO-23.yaml` (23/23 present), `development-log.md`,
`PDLC-JOURNEY.md`, `release-readiness-report.md`, `05-testing/_diagnostics/f2-gap-attribution.md`,
`03-design/06-synthesis/D-implementation-ready-design.md`, `03-design/moe-architecture-and-guarantee.md`,
`00-axioms/project-axioms.yaml`, `00-brownfield-assessment/assessment-2026-05-30.md`,
`artifacts/benchmarks/benchmark-results.json`. **All exist.** No broken Sources path.

### 1d. Section-anchor spot-checks — RESOLVE (one cosmetic drift)

- `release-readiness-report.md`: cited `##Verdict`, `##Caveats`, `##Pass-2 commitments`, `##Evidence`,
  `##Recommendation`, `##End-of-PDLC handoff` — all present (actual Caveats header is
  "## Caveats / known-state (explicit)" and Pass-2 is "## Pass-2 (real-user) commitments"; the cited short
  forms resolve to these — cosmetic drift, no finding).
- `f2-gap-attribution.md`: cited `##Refined conclusion` and `##Findings`/`##Refined conclusion` — present
  ("## Refined conclusion (supersedes…)", "## Findings — what the gap is NOT"). Resolve.
- `D-implementation-ready-design.md`: §DECISION 1/2/3, §D1 Design Cases, §SME findings, §Switch-points,
  §Cross-repo — all are real anchors per D1 §1 + the synthesis doc's own structure. Resolve.

---

## 2. Trace-ID resolution (no phantom FR/NFR/DC/SO/AX)

| ID family | Max cited in deliverables | Real census max | Phantom? |
|---|---|---|---|
| FR-NNN | FR-039 | FR-039 | none |
| NFR-NNN | NFR-026 | NFR-026 | none |
| DC-NN | DC-15 | DC-15 | none |
| SO-NN | SO-23 | SO-23 (23 files in `_signoffs/`) | none |
| AX-NNN | AX-006 | AX-006 | none |

Every forward ID claim resolves to a real ID in the D1 census / source-of-truth artifact. **No deliverable
cites a non-existent ID.** (The journey's §11 mention of "the FR-019 erratum" in the SO-20 row is a narrative
characterization, not a phantom ID — FR-019 is real; the changelog/journey correctly attribute the BYO-SDK to
FR-001/002 and note FR-019 is reversible-pseudonymization, not BYO.)

---

## 3. `## Sources` ↔ D2 `source_mapping` conformance + epistemic honesty

### 3a. Methodology + Sources blocks present on all six — PASS

Every deliverable carries both required blocks (§4 of `doc-architecture.md`). Each labels authored-narrative
vs reverse-compiled-from-artifact, and each carries the O-1/O-2/O-3 caveats where they apply. No deliverable
is missing `## Methodology` or `## Sources` → no MAJOR on that axis.

### 3b. Per-deliverable conformance

- **JRN (D-1):** Sources match the D2 D-1 mapping (SO-01..23, MANIFEST §Handoff Signals, PDLC-JOURNEY,
  brownfield assessment, release-readiness-report, f2-gap-attribution). No undeclared source; no mapped
  authoritative source left uncited. **Conformant.**
- **CHG (D-2):** Sources match the D-2 mapping (MANIFEST §S*-DONE, development-log §W6, SO ledger,
  release-readiness-report, the 30 story files via D1 index). Cites the D1 index + D2 architecture as
  provenance (acceptable — they are the harvest/architecture inputs every author reads). **Conformant.**
- **ADR (D-4):** Sources match the D-4 mapping (D-implementation-ready-design §DECISION 1/2/3 + §D1 DCs,
  moe-architecture-and-guarantee, swarm-architecture, project-axioms). **Conformant.**
- **API (D-5):** Sources match the D-5 mapping (docs/api-reference §PDLC surfaces + the 14 §8b module source
  files read directly + D1 §8 + D2 §5). Correctly cites live `src/` files as the verification ground truth.
  **Conformant** (the accuracy *defects* F-05/F-06 are in `doc-coverage-audit.md §3b`, not conformance).
- **USR (D-6):** Sources match the D-6 mapping (quickstart, evaluate-your-pipeline, configuration,
  anonymization-vs-pseudonymization, recall-floor, dependencies-and-platforms, cli.py, pyproject,
  native.py, fairness_gate.py, release-readiness-report). USR's Methodology **honestly discloses** that
  `docs/release-guide.md` (a mapped ops-subsection source) was **NOT read** and the ops-release subsection is
  therefore omitted → finding **T-01 (MINOR)**: a mapped authoritative source left uncited + a planned ops
  subsection absent. It is disclosed, not hidden. **Conformant-with-one-uncited-mapped-source.**
- **CON (D-7):** Sources list the D-7 mapping (extend-swarm, engine-plugin-guide, pyproject entry-points,
  the Protocol contracts, development-log §W3/W4/W6, project-axioms, assessment, Makefile) **plus MEMORY.md**
  — an **out-of-mapping, non-canonical source** → finding **T-03 (OBSERVATION→MINOR)**. See §3d.

### 3c. Epistemic-honesty constraints (§4 constraints 1–4) — PASS on all four

1. **AGENT_SIMULATED never presented as real-user-validated (O-3):** PASS. JRN §1/§2 honesty boundaries,
   CHG "What it IS NOT" ("Validated against real users — NO"), USR Methodology O-3 note, CON O-3 caveat, ADR
   Methodology ("SME panel is agent-simulated, not human SME review"). No deliverable claims real-user validation.
2. **NOT_YET / binding-G6-FAIL stated accurately (constraint 3):** PASS. JRN §12 dual-verdict table
   (smoke G7-bound vs produced G6-bound), CHG "Honest Status" G6 FAIL, ADR §1.4, USR header + §5c, API §10b.
   All cite f2-gap-attribution as the methodology-gap-not-regression explanation. **No deliverable states the
   verdict as PASS/SOTA.** (Minor cross-deliverable wording variance T-02, below — not a falsity.)
3. **Generated/WIP sources not cited as canonical (constraint 4 / O-7):** PASS. `docs/benchmark-summary.md`
   and `docs/pii-rate-elo-value.md` are explicitly excluded in JRN ("Deliberately NOT cited"), CHG Methodology
   items 2+4, USR Methodology, ADR Methodology. Numbers trace to `release-readiness-report.md` /
   `benchmark-results.json` instead. Confirmed by grep: neither file appears in any `## Sources` block.
4. **NFR-008 / FR-033 / FR-035 / DC-05 thin-trace marked documented-not-gated (§4 add'l caveats):** PASS.
   CHG "Notes on thin-trace FRs", ADR §3 notes, API §11 NFR-008 callout + §8 FR-033/035/036 callouts. Each is
   marked DOCUMENTED-NOT-(INDEPENDENTLY-)GATED, a SHOULD gap, exactly per the D2 mandate (→ OBSERVATION, not failure).

### 3d. Source-mapping deviation detail (T-03)

CON §"Methodology" + §"Sources" cite **`MEMORY.md`** (the developer's `.claude/` auto-memory) for §3b
(anchor-tests lesson) and §4c (adversarial-close history / fabrication examples). MEMORY.md is **not in the
D2 D-7 `source_mapping`** and is **not a canonical artifact** (it is a personal running log). The handbook
honestly flags it ("the MEMORY notes are not a formal source mapping but provide context-verified
corroboration"). The cited facts (S5-02/S5-03/S6-01 anchor lessons; the CATASTROPHIC NaN-curve / G7
fail-open fabrications) **do corroborate** against `development-log.md` + the SO ledger, so this is not a
fabricated citation. But it is an **out-of-mapping source**, and — notably — §2b's fabricated `RatingEnginePort`
contract (CATASTROPHIC F-01) sits in the same deliverable that substituted memory/prose for the literal
`rating/port.py` its Sources row *claims* it "verified." **Disposition: MINOR (out-of-mapping source), with a
note to D5 that T-03 and F-01 share a root cause — the handbook authored the contract from recollection, not
from the file.** The D3 re-author for F-01 should re-derive §2b/§2d strictly from the live Protocol files.

---

## 4. A11y disposition — N/A-WITH-REASON (D2-planned)

Per `doc-architecture.md §8.6` and the dispatch contract: the Stage-6 artifact-tree deliverables are
**markdown-only** documents under `dev-assist-artifacts/06-documentation/03-authoring/`. **No rendered
doc-site (HTML) ships in this RC** — the sibling `pii-anon-doc` repo owns the rendered site, and the D6
render is **deferred** (recorded here). There is no new a11y agent in Stage 6; WCAG audit is performed by
**reusing `dev-assist-testing-a11y-auditor`** *only when an HTML render exists*. Because no rendered target
exists in this wave, the reused a11y-auditor has nothing to audit and **`doc-accessibility-audit.md` is not
produced** in this directory. This is the D2-planned disposition, recorded so D5 frames the verdict correctly:
a11y is **N/A-WITH-REASON (deferred to the D6 render in pii-anon-doc)**, NOT a gap or a finding. This verifier
invents no WCAG findings (it is not the a11y authority).

---

## 5. Findings index (this file)

| ID | Severity | Location | One-line |
|---|---|---|---|
| T-01 | MINOR | USR Methodology / ops subsection | mapped authoritative source `docs/release-guide.md` not read → release-operations subsection omitted (disclosed, not hidden) |
| T-02 | OBSERVATION | JRN/CHG vs USR/API verdict wording | cross-deliverable variance: JRN/CHG foreground the dual (smoke-G7 vs produced-G6) verdicts; USR/API state only the produced-G6 verdict — all accurate, none false; D5 may want one canonical phrasing |
| T-03 | MINOR | CON Methodology + Sources | out-of-mapping non-canonical source (`MEMORY.md`) cited for §3b/§4c; shares a root cause with F-01 (contract authored from recollection, not from `rating/port.py`) |
| (a11y) | N/A-WITH-REASON | §4 | markdown-only; no rendered doc-site this RC; D6 render (pii-anon-doc) deferred; reused a11y-auditor has no target — `doc-accessibility-audit.md` intentionally not produced |

> Cross-link integrity, trace-ID resolution, and the four epistemic-honesty constraints all **PASS**. The
> D3 loopback is driven solely by the CATASTROPHIC code-accuracy fabrication F-01 (see
> `doc-coverage-audit.md §3b`), with the MAJOR/MINOR example-accuracy findings F-02/F-03/F-04 ride-along for
> the same re-author pass. T-01/T-02/T-03 are non-blocking and resolvable in the same loop.
