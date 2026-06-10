# Documentation Set — pii-anon SOTA Program (canonical root index)

> **Wave D5 synthesis output (the canonical index AND a trace surface).** This is the front
> door to the Stage-6 documentation set: one row per deliverable, linking each authored doc
> under `03-authoring/` back to the primary sources it compiles (so a reader can walk from any
> deliverable to its FR/NFR/DC/SO origins) and forward to its D4 verification status. The
> companion verdict lives at `05-synthesis/documentation-readiness-report.md`.
>
> **Build provenance.** Assembled read-only from `02-architecture/doc-architecture.md` (the
> resolved deliverable set §1 + the IA §2), the six authored `## Sources` blocks under
> `03-authoring/`, the two D4 verification reports under `04-verification/`, and the D1 index
> `01-harvest/doc-source-index.md`. No deliverable was authored or edited here. Date carried
> as 2026-06-10 (from the artifacts; not the system clock — D6 close may re-stamp).

---

## How to read this set (audience modes)

pii-anon is an **EXTERNAL-PRODUCT** OSS library/SDK (PyPI entry-point groups, public README +
LICENSE, a versioned CLI, a BYO-pipeline SDK adapter), so `user-operator-guide.md` is a *user
guide* with operations folded in — not an internal runbook (`doc-architecture.md §3`). The set
is ordered **lifecycle-then-audience hybrid** (`doc-architecture.md §9`): read the journey
first, then the slice that matches your role.

| If you are a… | Start with | Then |
|---|---|---|
| **Maintainer** (evolving the library) | `project-journey.md` → `architecture-and-adr.md` | `changelog.md`, `contributor-handbook.md`, `api-reference.md` |
| **Operator** (run / release / CI-gate) | `user-operator-guide.md` (§5 certify-a-run, §8g release) | `changelog.md`, `api-reference.md` |
| **Integrator** (embed / BYO-SDK) | `api-reference.md` → `user-operator-guide.md` (§4 evaluate-your-pipeline) | `architecture-and-adr.md` |
| **Contributor** (engines / readers / raters) | `contributor-handbook.md` (§2 plugin seams) | `architecture-and-adr.md`, `api-reference.md` |
| **Auditor** (privacy / compliance / eval) | `project-journey.md` (defensibility) → `changelog.md` (Honest Status) | `architecture-and-adr.md` (guarantees), `api-reference.md` (SDO surface) |

**One honesty note up front** (it is the spine of the whole set, `doc-architecture.md §4`): the
program's own SDO completion gate returns an **honest NOT_YET / binding-G6-FAIL** verdict, and
**all requirements are AGENT_SIMULATED** (the `00-validation/` dir is empty). No deliverable
claims real-user validation or a SOTA pass; benchmark numbers are PROVISIONAL. Every deliverable
carries `## Methodology` + `## Sources`, and the generated/WIP docs (`benchmark-summary.md`,
`pii-rate-elo-value.md`) are never cited as canonical.

---

## The deliverable index (per-entry: audience → purpose → primary sources → D4 status)

All authored deliverables live under
`dev-assist-artifacts/06-documentation/03-authoring/`. This index (D-3) lives at the
`06-documentation/` top level and is built in D5, not D3 (`doc-architecture.md §2`).

| # | Deliverable | Audience(s) | One-line purpose | Primary sources it compiles (`file:section` + IDs) | D4 status |
|---|---|---|---|---|---|
| **D-1** | `03-authoring/project-journey.md` | Maintainer, Auditor (all, as the spine) | The chronological PDLC story: brownfield → feature-complete v1.4.0 / honest NOT_YET, along the 23-SO spine | `~/_signoffs/SO-01..SO-23.yaml:scope` (spine) · `~/MANIFEST.md:§S*-DONE` · `~/PDLC-JOURNEY.md` · `~/00-brownfield-assessment/assessment-2026-05-30.md:§1-3` · `~/05-testing/release-readiness-report.md:##Verdict/##Caveats` · `~/05-testing/_diagnostics/f2-gap-attribution.md`; weaves DC-01→DC-15, SO-01..23 | **PASS** — coverage clean; Sources conformant (`doc-trace-integrity.md §3b`); honesty PASS |
| **D-2** | `03-authoring/changelog.md` | Maintainer, Operator, Auditor | Per-sprint/work-stream technical record + honest caveats + Pass-2 roadmap | `~/MANIFEST.md:§S*-DONE (S1-01..S7-05)` · `~/development-log.md:§W6` · `~/_signoffs/SO-01..23` · `~/05-testing/release-readiness-report.md:##Verdict/##Caveats/##Pass-2` · 30 story files `~/04-development/02-stories/sprint-*/`; tags FR/NFR/DC + SO per entry | **PASS** — coverage clean; Sources conformant; honesty PASS |
| **D-4** | `03-authoring/architecture-and-adr.md` | Maintainer, Contributor, Auditor | 3 headline Pugh decisions (as ADR-001/002/003) + the DC-01..15 module map, consolidated | `~/03-design/06-synthesis/D-implementation-ready-design.md:§DECISION 1/2/3 + §D1 DC table` · `~/03-design/moe-architecture-and-guarantee.md` · `~/00-axioms/project-axioms.yaml` (AX-001..006) · `docs/swarm-architecture.md`; covers DC-01..15 + FR/NFR | **PASS** — coverage clean; Sources conformant; DC-05 PARTIAL + thin-trace caveats carried |
| **D-5** | `03-authoring/api-reference.md` | Integrator, Maintainer, Operator, Auditor | Artifact-tree API compilation over the 14 SOTA modules; signatures verified against live code | D1 §8b (14 modules) + §8a/§8c · live `src/pii_anon/*` (38 verification points) · `docs/api-reference.md:##PDLC SOTA program surfaces` · `~/03-design/…:§D1 DC table`; covers every FR/NFR realized by a §8b module + thin-trace callouts | **PASS (Axis 3)** — all >18 symbols present in code; F-05/F-06 **CLOSED at retry-1** (`doc-coverage-audit.md` Retry-1 addendum) |
| **D-6** | `03-authoring/user-operator-guide.md` | Operator, Integrator, User | Install → use → evaluate-your-pipeline → certify-a-run → CI-gate → release (user guide + ops subsection) | `docs/quickstart.md` · `docs/evaluate-your-pipeline.md` · `docs/configuration.md` · `docs/anonymization-vs-pseudonymization.md` · `docs/recall-floor.md` · `docs/dependencies-and-platforms.md` · `docs/release-guide.md` · `src/pii_anon/cli.py` (19 cmds) · `pyproject.toml`; covers FR-001/002/007/008, NFR-005/006/009, DC-11/12/14/15 | **PASS** — coverage clean; F-03/F-04 + T-01 **CLOSED at retry-1** (Retry-1 addendum rows 4/5/6) |
| **D-7** | `03-authoring/contributor-handbook.md` | Contributor, Maintainer | The 4 plugin seams (engines/raters/BYO/readers) + the TDD + 5-reviewer-gate + adversarial-close discipline | `docs/extend-swarm.md` · `docs/engine-plugin-guide.md` · `pyproject.toml:[entry-points]` (lines 61/70/78/87) · the Protocol files (`rating/port.py`, `ingestion/native.py`, `byo_pipeline.py`, `engines/base.py`) · `~/development-log.md:§W3/W4/W6` · `~/00-axioms/project-axioms.yaml`; covers FR-001/002/003/016/018/021/031/032, DC-01/02/06/08/09/12/13/14 | **PASS (after loopback)** — F-01 CATASTROPHIC + F-02 MAJOR + T-03 **CLOSED at retry-1** (addendum rows 1/2/3); fabrication sweep CLEAN |
| **D-3** | `documentation-set.md` (this file) | All (index / trace surface) | The canonical root index linking every deliverable to its sources + D4 status | the resolved set (`doc-architecture.md §1`) + the six `## Sources` blocks + the live `docs/` tree (D1 §7) | **N/A** — D5 build; trace surface verified self-consistent here |

**Coverage cross-walk (for the auditor):** the per-deliverable ID sets above roll up to **0
documentation orphans** across 23 MUST FR + 13 MUST NFR + 15 DC — the full coverage matrix is in
`04-verification/doc-coverage-audit.md §1/§2` (most MUSTs land in 4–6 deliverables).

---

## D3-loopback history (the honest record — the CATASTROPHIC was caught and closed)

D4 round-0 ran one re-author loop (iteration 1 of the cap-3 budget). It found a CATASTROPHIC
fabrication and seven other findings; the D3 re-author closed all eight, confirmed by the
**Retry-1 re-verification addendum** in `04-verification/doc-coverage-audit.md` (each fix
re-checked against live source 2026-06-10, plus a fabrication sweep).

| Finding | Sev (round-0) | Deliverable | What was wrong | What closed it (cite) |
|---|---|---|---|---|
| **F-01** | **CATASTROPHIC** | CON §2b | `RatingEnginePort` contract **fabricated** as `fit`/`ratings`/`rank_one_probability`; the live Protocol (`rating/port.py:26`) is `run_round_robin` + `get_rating`. A plugin built to the doc fails `isinstance`. (Root cause shared with T-03: authored from recollection, not the file.) | Re-authored to the real 2-method contract; `grep` for the fabricated names over CON → NONE — `doc-coverage-audit.md` Retry-1 row 1; `doc-trace-integrity.md §3d` |
| **F-02** | MAJOR | CON §2d | `ReaderCapabilities(formats=/languages=/extraction_fidelity=)` fabricated; `capabilities` mis-typed as `@property` | Re-authored to the six real fields + `capabilities()` method — Retry-1 row 2 |
| **F-03** | MAJOR | USR §8b | non-runnable `EncryptedSQLiteTokenStore(key_envelope=KeyEnvelope.from_env(...))`; `from_env` absent | Re-authored to `key_provider=StaticTestKeyProvider(...)` — Retry-1 row 4 |
| **F-04** | MINOR | USR §6b | `reader_capabilities()` iterated as a dict (`.items()`); it returns a `list` | Re-authored to list iteration — Retry-1 row 5 |
| **F-05** | MINOR | API §10a | `CanonicalRunGate.validate` documented `-> None`/"raises"; real `-> tuple[bool, list[str]]`/"never raises" | Corrected to the real signature — Retry-1 row 7 |
| **F-06** | MINOR | API Methodology | stale line-number citations (PdfTextReader, StaticTestKeyProvider, EncryptedSQLiteTokenStore) | Anchors refreshed — Retry-1 row 8 |
| **T-01** | MINOR | USR ops | `docs/release-guide.md` unread → release-ops subsection omitted (disclosed) | §8g added, sourced from release-guide — Retry-1 row 6 |
| **T-03** | MINOR | CON Methodology/Sources | out-of-mapping non-canonical `MEMORY.md` cited (shared root cause with F-01) | MEMORY.md citations removed; `grep` → NONE — Retry-1 row 3 |

**Disposition:** all 8 CLOSED; fabrication sweep CLEAN (no new `ClassName(kwarg` fabrication);
**Axis-3 = PASS**; no further loopback warranted (`doc-coverage-audit.md` Retry-1 disposition).

---

## A11y disposition — N/A-WITH-REASON

The Stage-6 deliverables are **markdown-only** documents under `03-authoring/`; **no rendered
doc-site (HTML) ships in this RC** — the sibling `pii-anon-doc` repo owns the rendered site
(`user-operator-guide.md §8g` repo table), and the D6 render is deferred. There is no new a11y
agent in Stage 6; WCAG audit is performed by **reusing `dev-assist-testing-a11y-auditor`** only
when an HTML render exists. Because no rendered target exists, the reused auditor has nothing to
audit and **no `doc-accessibility-audit.md` is produced**. This is the D2-planned disposition
(`doc-architecture.md §8.6`), recorded in `doc-trace-integrity.md §4`. If D6 later renders these
to HTML, that render binds the reused a11y-auditor and the docs-discoverability gate
(`test_docs_discoverability.py`) if any doc is promoted into `docs/`.

---

## Open-observations ledger (non-blocking — bring-forward backlog)

All items below are MINOR/OBSERVATION; none blocks the DOCUMENTED verdict. The brownfield D1
observations (O-1..O-7) are **quality-of-source signals with named substitutes**, not authoring
failures (`doc-architecture.md §9`, brownfield note). They surface as a Pass-2 backlog.

| ID | Source | Item | Disposition |
|---|---|---|---|
| **O-1** | D1 §11 | All 5 per-stage doc-seeds absent; journey/changelog narrative reconstructed from the SO ledger + MANIFEST `§S*-DONE` | OBSERVATION — substitute spine named in JRN/CHG Methodology; author 5 doc-seeds Pass-2 (CON §6 item 2) |
| **O-2** | D1 §11 | No `examples-and-tests-catalog.md`; the 3,685-test suite is the living catalog | OBSERVATION — proof beats pulled from story `synthesis.md` + release-readiness `##Evidence`; curated catalog Pass-2 (CON §6 item 5) |
| **O-3** | D1 §11 | `00-validation/` empty (`.gitkeep` only) — all requirements AGENT_SIMULATED | OBSERVATION — carried verbatim in every deliverable; real-user validation is Pass-2 (honesty constraint 1, PASS per `doc-trace-integrity.md §3c`) |
| **O-4** | D1 §11 | No `documentation:` block in `developer-assistant.yaml` | OBSERVATION — D2's output IS the authoritative scope; operator may revise at D6 sign-off |
| **O-5** | D1 §11 | No `D-decision.md` diamond ADR files | OBSERVATION — ADRs consolidated from `D-implementation-ready-design.md §DECISION 1/2/3`; stated in ADR Methodology |
| **O-6 / NFR-008** | D1 §11, D4 O-02 | NFR-008 early-exit chunk latency documented but **not gated by a named acceptance test** (a SHOULD gap) | OBSERVATION — marked DOCUMENTED-NOT-GATED in API §11, ADR ADR-001/§3, CHG Sprint-7 notes (`doc-trace-integrity.md §3c` constraint 4 PASS) |
| **O-7** | D1 §11 | `pii-rate-elo-value.md` (user-WIP) + `benchmark-summary.md` (generated) must not be cited as canonical | OBSERVATION — confirmed by grep NOT cited anywhere (`doc-trace-integrity.md §3c` constraint 3 PASS) |
| **T-02** | D4 `doc-trace-integrity.md §5` | Verdict-wording variance: JRN/CHG foreground the dual (smoke-G7 vs produced-G6) verdicts; USR/API state only the produced-G6 verdict | OBSERVATION — all accurate, none false; D5 notes the canonical phrasing is the **produced-artifact NOT_YET / G6-FAIL** (see readiness report) |
| **O-01 / NFR-023** | D4 `doc-coverage-audit.md §2a/§5` | NFR-023 (MUST parity) covered as a string token in USR §8f (thinnest MUST-NFR row); concept also in ADR §3 | MINOR-flavored OBSERVATION — covered, not orphan; FR-036/NFR-023 carried as a Pass-2 commitment in CHG |
| **O-02 / thin-but-covered** | D4 `doc-coverage-audit.md §1/§5` | FR-017 (2 docs), FR-004 (3), FR-034 (2), DC-05 (2, PARTIAL) | OBSERVATION — each correctly scoped; DC-05 PARTIAL caveat carried in ADR §3 |
| **T-01 / T-03 / F-01..F-06** | D4 (round-0) | the D3-loopback findings | **CLOSED** — see the D3-loopback table above; retained here for the honest record, not as open items |

**Standing-gate note:** the D3 artifact-tree deliverables live under `dev-assist-artifacts/`,
NOT under `docs/`, so they do not trip `tests/test_docs_discoverability.py`. If D6 promotes any
into `docs/`, that gate binds (`doc-architecture.md §2` standing constraint, §8.5).

---

## Sources (this index)

Assembled read-only from:
- `dev-assist-artifacts/06-documentation/02-architecture/doc-architecture.md` (§1 resolved set, §2 IA, §3 audiences, §4 honesty, §8 verification plan, §9 decision record) — the contract.
- The six authored `## Sources` blocks under `dev-assist-artifacts/06-documentation/03-authoring/` (project-journey, changelog, architecture-and-adr, api-reference, user-operator-guide, contributor-handbook).
- `dev-assist-artifacts/06-documentation/04-verification/doc-coverage-audit.md` (Axis 1 + Axis 3 + the Retry-1 re-verification addendum).
- `dev-assist-artifacts/06-documentation/04-verification/doc-trace-integrity.md` (Axis 2 + honesty + §4 a11y disposition).
- `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md` (§7 user-docs tree, §8 code surface, §11 observations O-1..O-7).
