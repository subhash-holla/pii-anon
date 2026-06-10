# doc-architecture — pii-anon Stage 6 Documentation (D2 Architecture)

> **Wave D2 output.** Architecture-only: this file decides *what* to document, *for whom*, *in
> what shape*, *sourced from where*. It authors no deliverable and renders nothing. The D3
> authors write only what is mapped here and cite exactly the sources listed; the D4 verifier
> checks each authored `## Sources` block against the per-deliverable `source_mapping` below.
> Single primary input: `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md`.

---

## 0. Authority Note — config absence (D1 O-4) and the resolution rule

`developer-assistant.yaml` carries **no `documentation:` block** (confirmed: O-4; `grep documentation:`
returns nothing). Per the dispatch contract, **defaults apply**:

- `depth: standard`
- `deliverables: [project-journey, architecture-and-adr, api-reference, user-operator-guide, contributor-handbook, changelog]` (+ the always-on `documentation-set`)
- `architecture_diverge: false` → this `doc-architecture.md` is written **DIRECTLY**; no DIVERGE/CONVERGE
  sub-tree is produced (no `A-frame-*/B-pugh-scorecard.md/C-validation.md/D-decision.md` under this dir).
- `audiences`: not enumerated → the standard five (maintainer / operator / integrator / contributor / auditor) apply.

**Because there is no config block, D2's output IS the authoritative scope.** D4 cannot assert
"every *configured* deliverable is covered" (there is no config to check against); D4 instead verifies
against the resolved set in §1 below. **The operator may override any of this at the D6 sign-off**
(add/drop a deliverable, change depth) — any such override must be recorded with rationale.

**Mode framing (D1 §Source Signal):** This is a **mature-brownfield** v1.4.0 library that completed a
full PDLC pass under `pdlc/sota-program`. Unlike a thin brownfield input, the source corpus is *rich*
(6 stages of canonical artifacts, 23 sign-offs, ~30 stories, a live 20-file user-docs tree, a detailed
SDO verdict). The conditional triggers therefore fire on **real authored signal**, not reverse-extracted
guesses — so the full default deliverable set resolves `IN`. The single brownfield *thinness* is the
absent doc-seed prose (O-1); substitute narrative sources are abundant and named per-deliverable below.

---

## 1. Resolved Deliverable Set

`resolved = (config.deliverables = defaults) ∩ (project-type triggers)`, then `operator_overrides`
(none at D2 — config absent; operator acts at D6). The three ALWAYS deliverables are `IN`
unconditionally. All six conditional/default deliverables resolve `IN` because every trigger fired
against confirmed signal.

| # | Deliverable | Verdict | Deciding rule (config ∩ trigger) | Trigger evidence (D1) |
|---|---|---|---|---|
| D-1 | `project-journey.md` | **IN** | **ALWAYS** (signature deliverable; never dropped) | 23 SO sign-offs + MANIFEST `### S*-DONE` spine + `PDLC-JOURNEY.md` |
| D-2 | `changelog.md` | **IN** | **ALWAYS** | 30 story files + SO ledger + release-readiness verdict |
| D-3 | `documentation-set.md` | **IN** | **ALWAYS** (canonical index / trace surface; **not authored in D3** — D5 builds it) | the resolved set itself + the live `docs/` tree |
| D-4 | `architecture-and-adr.md` | **IN** | default ∩ **Design ran ✓** | `D-implementation-ready-design.md:§DECISION 1/2/3` + `§D1 DC table` (15 DCs); `moe-architecture-and-guarantee.md`. **Caveat: NO `D-decision.md` diamond ADR files exist (O-5)** — decisions are consolidated in the synthesis doc, not dispersed as per-decision ADRs |
| D-5 | `api-reference.md` | **IN** | default ∩ **code surface ✓** | ~90 Python files; 14 headline SOTA modules (D1 §8b); orchestrator public API verified present (`run/run_async/detect_only/run_stream/capabilities/discover_engines`). **This is the ARTIFACT-TREE compilation, distinct from the existing user-docs `docs/api-reference.md`** (which S7-05 already gated) |
| D-6 | `user-operator-guide.md` | **IN** | default ∩ **CLI/ops surface ✓** | Typer CLI 17+ commands (`canonical-run`, `supremacy`, `benchmark-publish-suite`, `compare-competitors`); `docs/quickstart.md`, `configuration.md`, `evaluate-your-pipeline.md`, `release-guide.md`. **Mode = EXTERNAL PRODUCT → "user guide" (see §3)** |
| D-7 | `contributor-handbook.md` | **IN** | default ∩ **OSS / multi-contributor ✓** | LICENSE present; README present; 4 entry-point groups in `pyproject.toml` (`engines`/`rating_engines`/`byo_pipelines`/`readers`); `docs/extend-swarm.md` + `engine-plugin-guide.md`; the TDD+5-reviewer-gate machinery. **Caveat: NO top-level `CONTRIBUTING.md` exists** — author leans on entry-point groups + existing extension docs + development-log gate discipline |

**OUT:** none. **DEFERRED:** none. No trigger failed; no operator override is in scope at D2.

**Why nothing is OUT (contrast with thin-brownfield):** the dispatch contract drops a conditional only
when its trigger does not fire (e.g. *no* Design artifacts → `architecture-and-adr` OUT). Here Design ran,
code exists, a CLI/ops surface exists, and the repo is OSS — all four conditionals are honestly triggered.

---

## 2. Information Architecture (the `documentation-set.md` index tree)

D3 authors all deliverables **except D-3** (the index) under
`dev-assist-artifacts/06-documentation/03-authoring/`. D5 then assembles `documentation-set.md` (the
index) at `06-documentation/` top level (or `05-synthesis/`, per the synthesis stage's own convention).
The index is itself a **trace surface**: every entry links back to its sources (the `## Sources` block of
each authored deliverable) and forward to the FR/NFR/DC IDs it covers.

**Authoring locations (D3 outputs):**

```
06-documentation/
├── 01-harvest/doc-source-index.md            (D1 — DONE)
├── 02-architecture/doc-architecture.md        (D2 — THIS FILE)
├── 03-authoring/
│   ├── project-journey.md                     (D-1, Opus)
│   ├── changelog.md                           (D-2, Sonnet)
│   ├── architecture-and-adr.md                (D-4, Sonnet)
│   ├── api-reference.md                        (D-5, Sonnet)   [artifact-tree compilation]
│   ├── user-operator-guide.md                 (D-6, Sonnet)
│   └── contributor-handbook.md                (D-7, Sonnet)
└── (D5) documentation-set.md                   (D-3 — index, NOT authored in D3)
```

**Reading order in the index (lifecycle-then-audience hybrid, single-frame — see §6):**

1. **`project-journey.md`** — the narrative front door. How the program got from brownfield assessment
   to a feature-complete v1.4.0 with an honest NOT_YET SDO verdict. Links onward to every other doc.
2. **`architecture-and-adr.md`** — the three headline decisions + 15 DCs. The "why it is shaped this way."
3. **`api-reference.md`** — the "what you call" surface. Cross-links to the live `docs/api-reference.md`.
4. **`user-operator-guide.md`** — the "how you run / operate / certify it" surface (install → detect →
   stream → evaluate-your-pipeline → certify a run → CI-gate). Cross-links to `docs/quickstart.md` et al.
5. **`contributor-handbook.md`** — the "how you extend / contribute" surface (entry-point plugins +
   TDD/gate discipline). Cross-links to `docs/extend-swarm.md` + `engine-plugin-guide.md`.
6. **`changelog.md`** — the per-sprint / per-work-stream record + the honest caveats and Pass-2 roadmap.

**Cross-link rules (D5 enforces in the index; D3 should seed inline links):**
- Every deliverable links to `project-journey.md` (the spine) and to `changelog.md` (the current state).
- `api-reference.md` (D-5, artifact-tree) explicitly cross-references the **existing** user-docs
  `docs/api-reference.md` and must NOT contradict it — the user-docs file is gated by
  `tests/test_docs_discoverability.py` (confirmed present). D-5 is a *compilation/trace* layer on top,
  not a replacement.
- `architecture-and-adr.md` links each decision to the DCs it realizes and the FRs/NFRs those DCs implement.
- The index entry for each deliverable carries its covered-ID set (for D4's coverage audit, §6 below).

**Standing constraint (D1 handoff note 4):** any new doc *added to `docs/`* must satisfy
`test_docs_discoverability.py` (headline symbols present in api-reference, intra-docs links resolve, index
covers every non-WIP file). The D3 artifact-tree deliverables live under `dev-assist-artifacts/`, NOT under
`docs/`, so they do **not** trip this gate — but if D6 later promotes any of them into `docs/`, the gate binds.

---

## 3. Audience Map

**External-product vs internal-tool resolution (per `references/audience-modes.md`):**
pii-anon is a **packaged, installable, OSS library/SDK** — it ships on PyPI-style entry-point groups,
has a public README + LICENSE, a versioned CLI, a BYO-pipeline SDK adapter (FR-001 MUST), and a
discoverability-gated user-docs tree. It is consumed by *external integrators and operators*, not run as
an internal-only service. **→ This project is EXTERNAL PRODUCT mode.** Consequence: `user-operator-guide.md`
is authored as a **user guide** (install / use / evaluate / certify), with operator-runbook concerns
(release, CI-gating, provenance) folded in as an *operations* subsection — NOT as an internal operator runbook.

**The five audiences × the deliverables that serve each:**

| Audience | Who they are | Primary deliverables | Secondary |
|---|---|---|---|
| **Maintainer** | core devs evolving the library | `project-journey.md`, `architecture-and-adr.md`, `changelog.md` | `contributor-handbook.md`, `api-reference.md` |
| **Operator** | runs / releases / CI-gates pii-anon in a pipeline | `user-operator-guide.md` (ops subsection: certify-a-run, CI ship/no-ship, release), `changelog.md` | `api-reference.md` |
| **Integrator** | embeds pii-anon / BYO-pipeline SDK into their own product | `api-reference.md`, `user-operator-guide.md` (evaluate-your-pipeline, BYO-SDK) | `architecture-and-adr.md` |
| **Contributor** | external OSS contributors adding engines/readers/raters | `contributor-handbook.md`, `architecture-and-adr.md` | `api-reference.md`, `project-journey.md` |
| **Auditor** | privacy/compliance/eval reviewers checking claims | `project-journey.md` (defensibility), `changelog.md` (honest caveats), `architecture-and-adr.md` (guarantees) | `api-reference.md` (SDO/canonical-run surface) |

**The auditor is a first-class audience here** because the program's whole spine is an evidence/guarantee
arc (SDO gate G1–G7, NFR thresholds, the anti-anonymity caveat NFR-016, the honest NOT_YET verdict). Every
deliverable that states a guarantee or a benchmark number serves the auditor and therefore MUST carry the
epistemic-honesty blocks in §4.

---

## 4. Epistemic-Honesty Requirements (apply to EVERY deliverable)

These are non-negotiable and are checked by D4. Each authored deliverable MUST carry:

- **`## Methodology`** — how the content was derived: which artifacts were read, what was *authored
  narrative* vs *reverse-compiled from code/sign-offs*, and which observations (O-1..O-7) qualify it.
- **`## Sources`** — the exact file:section + ID list the author actually cited, matched against this
  doc's `source_mapping` for that deliverable (§5). D4 flags any `## Sources` entry not in the mapping,
  and any mapped authoritative source the author failed to cite.

**The four standing honesty constraints (carry verbatim where they apply):**

1. **AGENT_SIMULATED is NEVER presented as real-user-validated (O-3).** `00-validation/` is empty
   (confirmed: only `.gitkeep`). No deliverable may claim "validated against user research / real users."
   All requirements are AGENT_SIMULATED. Any sentence implying real-user validation must instead say
   "agent-simulated; real-user validation is a documented Pass-2 follow-up." The journey and changelog
   authors especially must caveat this.

2. **The doc-seed narrative is inferred, not authored (O-1).** All five per-stage doc-seeds are absent
   (confirmed). The plain-language stage narrative is *reconstructed* from substitute sources (the
   `discovery-report.md` POV sections, the MANIFEST `### S*-DONE` sections, and the SO `scope:` lines).
   The journey author must state this in `## Methodology`: the narrative spine is the SO ledger + MANIFEST
   S*-DONE sections, NOT authored doc-seeds. Recommend the five doc-seeds as a follow-up pass.

3. **The SDO verdict is honest NOT_YET — say so (D1 handoff note 7).** Current HEAD is
   `NOT_YET / canonical_claim_run=True / binding G6 FAIL` (F2 0.7214 vs 0.75 threshold; coverage 0.824).
   This is a **methodology gap, not a regression** (`f2-gap-attribution.md`; old code ≡ current at
   `use_case=default`). The changelog and journey "current state / limitations" sections MUST state the
   NOT_YET verdict accurately and cite the f2-gap attribution as the explanation. G1/G2/G3/G4/G5/G7 PASS;
   G6 binds. Benchmark numbers are PROVISIONAL (smoke run until canonical regen + significance repair).

4. **Generated/WIP docs are excluded (O-7, D1 §7).** `docs/pii-rate-elo-value.md` (user-WIP, excluded from
   the docs gate) and `docs/benchmark-summary.md` (auto-rewritten by the competitor benchmark script —
   volatile) MUST NOT be cited as canonical sources by any deliverable. If a number is needed, cite the
   stable artifact (`/artifacts/benchmarks/*.json`, `release-readiness-report.md:##Evidence`) instead.

**Additional per-deliverable caveats (the SHOULD/thin-trace flags D4 watches):**
- **O-6 / NFR-008 SHOULD gap:** early-exit chunk latency (p50 ≤ 1 ms ∧ p95 ≤ 2 ms) appears in only 3 files
  and has **no dedicated story with formal acceptance tests**. DC-02 covers it implicitly; `latency_ceilings.py`
  is the committed registry. The api-reference latency section and the changelog MUST mark NFR-008 as
  *documented but not gated by a named acceptance test* (a SHOULD gap, not a failure).
- **DC-05 thin trace (1 file):** 6 transforms + legal-regime mapping; FR-020/021/022 are SHOULD and
  partially deferred. Architecture-and-ADR must note DC-05 is the least-elaborated decision and the
  orchestration SHOULD is partial.
- **FR-033 / FR-035 / FR-036 thin trace (5–6 files each):** extraction-fidelity assertion, multimodal-recall
  CI-regression gate, and stream/batch/offline parity all ride the single S7-01 (FR-033/035) / DC-14 (FR-036)
  story. The api-reference must give them explicit callouts so they are not lost (D1 §3a, §9 watch-list).
- **Cross-repo edges (D1 handoff note 10):** the eval-framework / Tier-3 / MIA surfaces have `DATA:` and
  `PAPER:` external_refs (pii-anon-eval-data S5–S7) for FR-002/003/011/013, UC-02/09/10. The api-reference
  and architecture-and-ADR must mark what is **CODE-local vs DATA-track** so an integrator does not assume
  a DATA-track dependency ships in this repo.

---

## 5. Per-Deliverable Source Mapping (the D3 author contract)

For each `IN` deliverable: the **authoritative** `source_mapping` (exact `file:section` + the trace-ID set
the author must cover and cite), the **substitute** sources where doc-seeds are absent, the **caveats**
the author must carry, and the **out-of-scope** exclusions. D3 authors read ONLY their mapped sources and
cite EXACTLY this list in `## Sources`. Paths are repo-relative; `dev-assist-artifacts/` is abbreviated `~/`.

---

### D-1 · `project-journey.md` — model: **Opus** (signature, narrative-heavy)

**Purpose:** the chronological story of the PDLC pass, brownfield → feature-complete v1.4.0 / honest NOT_YET.

**Authoritative sources (cite all):**
- `~/_signoffs/SO-01.yaml` … `SO-23.yaml` — `scope:` fields = the chronological spine (23 sign-offs; confirmed count). **This is the strongest single source** (D1 handoff note 1).
- `~/MANIFEST.md:§Handoff Signals` — all `### S*-DONE` sections (Sprint-1 COMPLETE … S7-05 DONE; confirmed present at lines 153+); per-story technical beats incl. the adversarial-close drama.
- `~/PDLC-JOURNEY.md:§Traceability spine + §Per-stage summary + §What shipped + §What's deferred + §Defensibility`.
- `~/00-brownfield-assessment/assessment-2026-05-30.md:§4 Findings` — the starting condition (12 MAJORs + 8 OBSERVATIONs).
- `~/05-testing/release-readiness-report.md:##Verdict + ##Caveats + ##End-of-PDLC handoff` — the landing condition.

**Trace IDs to weave (not exhaustively — narratively):** SO-01..23 (spine); the headline arc DC-01 (recall floor) → DC-02/03 (MoE) → DC-06/07 (rating) → DC-08/10 (deid families + calibration) → DC-09 (attacks) → DC-11 (canonical-run/SDO gate) → DC-12 (BYO-SDK) → DC-13 (agentic) → DC-14 (multimodal) → DC-15 (multilingual/fairness).

**Substitute-for-absent-doc-seed (O-1):** the SO `scope:` lines + MANIFEST `### S*-DONE` sections ARE the de-facto authored narrative. State in `## Methodology` that no authored doc-seeds existed; narrative is reconstructed from these.

**Caveats to carry:** O-1 (inferred narrative); O-3 (AGENT_SIMULATED — the journey describes an agent-run PDLC, not real-user validation); honesty-constraint 3 (the landing verdict is NOT_YET / G6 FAIL, and that is a methodology gap not a regression — cite `~/05-testing/_diagnostics/f2-gap-attribution.md`); honesty-constraint 4 (do not cite benchmark-summary.md / pii-rate-elo-value.md).

**Out of scope:** API signatures, CLI invocation detail, plugin-authoring how-to (those live in D-5/D-6/D-7).

---

### D-2 · `changelog.md` — model: **Sonnet**

**Purpose:** per-sprint / per-work-stream technical record + honest caveats + Pass-2 roadmap.

**Authoritative sources (cite all):**
- `~/MANIFEST.md:§S*-DONE sections` (SO-07..SO-23 narrative beats; commit hashes, gate outcomes, the SDO axis each story closes — D1 handoff note 2: best source for technical release-notes).
- `~/development-log.md:§W6 Execution` (and `§W5 Stories` for the planning frame).
- `~/_signoffs/SO-01.yaml..SO-23.yaml` — `scope:` one-liners as the changelog entry headers.
- `~/05-testing/release-readiness-report.md:##Verdict + ##Caveats + ##Pass-2 commitments + ##Recommendation`.
- The 30 story files under `~/04-development/02-stories/sprint-*/` for per-story detail (S1–S7).

**Organize by:** sprint/work-stream (Sprint-1 recall floor → S2 MoE-router → S3 rating ladder → S4 deid+calibration+SDO-gate → S5 attacks → S6 agentic/BYO → S7 canonical-run/readers/multilingual/docs). Tag each entry with the FR/NFR/DC it closes and the SO that signed it off.

**Substitute-for-absent-doc-seed (O-1):** MANIFEST S*-DONE sections substitute for an authored 04/05 testing narrative.

**Caveats to carry:** honesty-constraint 3 (lead the "current state" with the honest NOT_YET / binding-G6-FAIL verdict; benchmark numbers PROVISIONAL); honesty-constraint 4 (benchmark-summary.md is generated — do not cite as a number source; cite `/artifacts/benchmarks/benchmark-results.json` + `release-readiness-report.md:##Evidence`); O-6 (NFR-008 documented-not-gated); honesty-constraint 1 (Pass-2 items are AGENT_SIMULATED follow-ups: latency thresholds, Tier-3 realism, OCR/DICOM extraction at real strength, OS-matrix certification — D1 handoff note 8).

**Out of scope:** prose narrative of *why* decisions were made (that is the journey + ADR).

---

### D-4 · `architecture-and-adr.md` — model: **Sonnet**

**Purpose:** the headline design decisions + the 15 Design Cases as consolidated ADRs.

**Authoritative sources (cite all):**
- `~/03-design/06-synthesis/D-implementation-ready-design.md:§DECISION 1` (SharedLayerProjector + DistilledTopKGate + rules-first depth-1 early-exit; Pugh winner 8.4), `:§DECISION 2` (Bayesian-BT spine, MLE-BT smoke/fallback; resolved CATASTROPHIC eval-01; Pugh winner 8.6), `:§DECISION 3` (agentic interception via router pre-filter + unified floor; Pugh winner Option A).
- `~/03-design/06-synthesis/D-implementation-ready-design.md:§D1 Design Cases` — the **DC-01..DC-15 table** (each DC → FR/NFR implemented, decision type, program status).
- `~/03-design/moe-architecture-and-guarantee.md` (full) — the MoE guarantee doc (DC-02, DC-03).
- `~/03-design/06-synthesis/D-implementation-ready-design.md:§SME findings` (the CATASTROPHIC eval-01 resolution), `:§Switch-points`, `:§D0 baseline`.
- Existing user docs for the user-facing architecture framing: `docs/swarm-architecture.md`, `docs/recall-floor.md`, `docs/anonymization-vs-pseudonymization.md` (the no-merge invariant verbatim — D1 handoff note 5, load-bearing for the deid scoring surface; FR-010 MUST).

**Trace IDs to cover (all 15 DCs + their FR/NFR):** DC-01 (FR-016/NFR-011/AX-003) … DC-15 (FR-038/039, NFR-024/025/026, AX-001). Per D1 §4 the program-status column (LIVE / PARTIAL / IN-TREE / pass-2 wire-ins) MUST be reflected — e.g. DC-02 early-exit "blocked by S2-03", DC-13 "orchestrator wire-in pass-2", DC-14 "OCR/DICOM/audio extraction pass-2".

**Caveat (O-5 — critical):** there are **NO `D-decision.md` diamond ADR files** (confirmed absent). All
decision context is consolidated in `D-implementation-ready-design.md:§DECISION 1/2/3` + `§D1 DC table`,
NOT dispersed as individual ADRs. State this in `## Methodology`: ADRs are *compiled from* the synthesis
doc, the structure is consolidated not dispersed. **Do not invent ADR numbers/files that do not exist.**

**Other caveats:** DC-05 thinness (1 file; FR-020/021/022 SHOULD + partial); cross-repo CODE-local-vs-DATA-track
marking on the eval/attacks DCs (DC-06/07/09); honesty-constraint 1 (no real-user validation of the design —
the SME panel is agent-simulated heuristic eval, not human SME review).

**Out of scope:** API call signatures (D-5); install/run steps (D-6).

---

### D-5 · `api-reference.md` (artifact-tree compilation) — model: **Sonnet**

**Purpose:** the trace/compilation API surface over the SOTA program — distinct from, and cross-linked to,
the existing gated user-docs `docs/api-reference.md`. **This is the contract D4's accuracy-against-code audit checks.**

**Authoritative sources (cite all):**
- **D1 §8b (14 new SOTA modules) — the headline public-symbol inventory**, each with its FR/DC. The author MUST verify every signature against the live code (D4 re-checks): e.g. `pii_anon/orchestrator.py` (`PIIOrchestrator.run/run_async/detect_only/run_stream/run_stream_async/capabilities/discover_engines` — confirmed present), `eval_framework/byo_pipeline.py:BYOPipelineRegistry`, `routing/shared_layer.py:SharedLayerProjector`, `policy/query_aware.py:QueryAwareMaskingGate`, `agentic/interception.py:FourChannelGuard`, `ingestion/native.py:NativeReaderRegistry`, `eval_framework/metrics/deid_families.py` scorers, `eval_framework/evaluation/competitive_supremacy.py` (`_g1`.._g7` — confirmed present), `evaluation/canonical_run.py:CanonicalRunGate`, `eval_framework/evaluation/latency_ceilings.py`.
- **D1 §8a (top-level packages)** + **§8c (baseline modules relevant to api-reference)** for the full surface (`pipeline.py`, `cli.py`, `types.py`, `errors.py`, engines/transforms/calibration/tracking/segmentation/tokenization).
- `docs/api-reference.md:##PDLC SOTA program surfaces` (the existing gated listing — D-5 compiles/traces on top, must NOT contradict it).
- `docs/anonymization-vs-pseudonymization.md` (full) for the deid family API framing (FR-010).
- `~/03-design/06-synthesis/D-implementation-ready-design.md:§D1 DC table` to map each symbol to its DC.

**Trace IDs to cover:** every FR/NFR realized by a §8b module. **Explicit callouts required** for the thin-trace
FRs (D1 §9 watch-list): FR-033, FR-035, FR-036 (multimodal extraction-fidelity / CI-regression / parity) and the
NFR-008 latency bound (point at `latency_ceilings.py` as the concrete realization). DC-05 transforms surface.

**Caveats:** O-2 (no `examples-and-tests-catalog.md` — the **test suite (3,685 tests) is the living catalog**;
pull proof beats from `~/_reviews/story/*/synthesis.md` + `release-readiness-report.md:##Evidence`); accuracy —
every signature is verified against code, NOT copied from prose; cross-repo marking (BYO/rating/attacks symbols
that depend on `DATA:`/`PAPER:` external_refs must say CODE-local-vs-DATA-track); honesty-constraint 4.

**Out of scope:** narrative/why (D-1/D-4); end-to-end tutorials (D-6).

---

### D-6 · `user-operator-guide.md` (EXTERNAL-PRODUCT "user guide" + ops subsection) — model: **Sonnet**

**Purpose:** install → use → evaluate-your-pipeline → certify-a-run → CI-gate → release. User-facing, with
operations folded in (NOT an internal operator runbook — §3 mode = external product).

**Authoritative sources (cite all):**
- `docs/quickstart.md:##Install + ##Detect with explicit transform mode + ##Stream processing + ##CLI quickstart + ##Evaluate your own pipeline` (primary onboarding).
- `docs/evaluate-your-pipeline.md` (full — `##60-second version`, `##Predictor contract`, `##Programmatic API`, `##CLI workflow`, `##Package as SDK plugin`, `##Incumbents scored on identical path`, `##Certify a run`, `##Reading results`, `##Statistical significance`, `##CI gating`, `##Tier 3 evaluation`, `##Troubleshooting`).
- `docs/configuration.md` (config schema) + `docs/dependencies-and-platforms.md` (dependency/platform matrix).
- **CLI census** — `pii_anon/cli.py` (17+ Typer commands): `pii-anon canonical-run`, `pii-anon supremacy`, `pii-anon benchmark-publish-suite`, `pii-anon compare-competitors` (verify command list against code).
- **Ops subsection sources:** `docs/release-guide.md`; `evaluation/canonical_run.py:CanonicalRunGate` + `eval_framework/evaluation/competitive_supremacy.py` (G1–G7) for the certify-a-run flow; `~/05-testing/release-readiness-report.md:##Verdict` for what a current certify run honestly returns.

**Trace IDs to cover:** FR-001/002 (BYO-SDK; SO-20), FR-007/008 (CI gate + canonical-run provenance; SO-15/16),
NFR-005/006 (determinism + provenance), the operator-facing NFRs (NFR-009 latency profiles via `latency_ceilings.py`).

**Caveats:** honesty-constraint 3 (a certify run currently returns NOT_YET / G6 FAIL — the guide must show the
honest output, not a fabricated PASS; this is the program's own discipline — the SDO gate refuses to fabricate);
honesty-constraint 4 (benchmark-summary.md is generated/volatile — do not embed its numbers as canonical);
cross-repo (Tier-3 evaluation section depends on the DATA track — mark it); O-7 (do not cite pii-rate-elo-value.md).

**Out of scope:** plugin authoring (D-7); internal design rationale (D-4).

---

### D-7 · `contributor-handbook.md` — model: **Sonnet**

**Purpose:** how an external contributor extends pii-anon (entry-point plugins) and the contribution discipline (TDD + 5-reviewer story gate + the adversarial close for control-path code).

**Authoritative sources (cite all):**
- `docs/extend-swarm.md` (full — swarm extension) + `docs/engine-plugin-guide.md` (full — plugin authoring).
- **Entry-point groups** — `pyproject.toml:[project.entry-points."pii_anon.engines"]` (line 61), `["pii_anon.rating_engines"]` (line 70), `["pii_anon.byo_pipelines"]` (line 78), `["pii_anon.readers"]` (line 87). All four confirmed present; the handbook documents how a third-party package advertises a plugin on each group.
- **The Protocol contracts** to implement: `pii_anon/engines/` (`EngineAdapter`), `eval_framework/rating/` (`RatingEnginePort`), `eval_framework/byo_pipeline.py` (`engine_predictor`/`incumbent_predictor`), `ingestion/native.py` (`NativeReader` Protocol + `ReaderCapabilities`).
- **The TDD + gate machinery** — `~/development-log.md:§W3 Quality + §W4 Testing + §W6 Execution` (the RED→GREEN→REFACTOR + 5-reviewer story-gate discipline; the adversarial close for any `competitive_supremacy.py` / control-path-artifact change).
- Verification gates: `make lint` (ruff src+tests), `make type` (mypy src/pii_anon strict), `make test` (PYTHONPATH=src pytest); xdist via `.venv/bin/python -m pytest -n auto`.

**Trace IDs to cover:** FR-021 (orchestrate incumbent detectors behind recall-floored interface), FR-001/002
(BYO-SDK contract), FR-031 (readers group), FR-003 (rating-engine port); DC-06 (ports-adapters architecture).

**Caveat (critical):** **NO top-level `CONTRIBUTING.md` exists** (confirmed absent — only LICENSE + README).
The handbook is *compiled from* the existing extension docs + entry-point groups + the development-log gate
discipline; state this in `## Methodology` and recommend authoring a `CONTRIBUTING.md` as a follow-up. The
README + LICENSE establish the OSS trigger. Honesty-constraint 1 (the gate discipline is agent-run; the
5-reviewer "panel" is agent-simulated, not human reviewers).

**Out of scope:** end-user run steps (D-6); API enumeration (D-5).

---

### D-3 · `documentation-set.md` — model: **N/A (D5 builds it, not D3)**

**Purpose:** the canonical index / trace surface. **Authored by D5 synthesis, not by a D3 author.** D2 specifies
its shape here (§2 above): the reading order, the per-entry covered-ID set, and the cross-link rules. Its sources
are the resolved set (§1) + the six authored deliverables' `## Sources` blocks + the live `docs/` tree (D1 §7).

---

## 6. Depth Profile (`config.depth = standard`)

`depth` is unset in config → **standard** applies. Standard ≠ lightweight (journey + changelog + index only-ish)
and ≠ full (every triggered deliverable at maximum exhaustive detail). At **standard**:

- **All six default deliverables + the index are authored** (none dropped to a stub) — justified because the
  source corpus is rich, not thin. A lightweight profile would have been wrong for a mature v1.4.0 with a live
  docs tree and a 23-sign-off spine.
- **Depth calibration per deliverable:**
  - `project-journey.md` — **full narrative** (Opus; the signature deliverable always gets full treatment).
  - `changelog.md` — **complete per-story** (all 30 stories / 23 SOs), but entry-level not essay-level.
  - `architecture-and-adr.md` — **all 15 DCs + 3 decisions**, decision-level depth (not re-deriving the Pugh math; cite the winners).
  - `api-reference.md` — **all 14 SOTA modules + top-level surface**, signature-level (not line-by-line internals); thin-trace FRs get explicit callouts but not exhaustive sub-API trees.
  - `user-operator-guide.md` — **the documented user+ops flows**, mapped to existing `docs/` (not re-authoring every `docs/` page; compiling + cross-linking).
  - `contributor-handbook.md` — **the four extension surfaces + the gate discipline**, how-to depth (not a full internals tour).
- **Not produced at standard** (would be a `full`-only add): per-FR exhaustive sub-pages, a separate glossary
  deliverable (the D1 §10 "Glossary" pre-map item is folded into the journey/ADR/api intros, not a standalone doc),
  per-module deep-dives. If the operator wants these, that is a D6 depth override.

---

## 7. D3 Dispatch Notes (one author per deliverable)

The orchestrator dispatches one **doc-author** per `IN`-and-D3-authored deliverable, each receiving the matching
`source_mapping` from §5. Model hints:

| Deliverable | Author model | Why |
|---|---|---|
| D-1 `project-journey.md` | **Opus** | signature, narrative-synthesis-heavy across 23 SOs + MANIFEST + journey + brownfield + verdict |
| D-2 `changelog.md` | **Sonnet** | structured per-sprint compilation; mechanical-but-precise |
| D-4 `architecture-and-adr.md` | **Sonnet** | decision-table compilation from a single synthesis doc |
| D-5 `api-reference.md` | **Sonnet** | signature compilation + code-verification; precision over prose |
| D-6 `user-operator-guide.md` | **Sonnet** | flow compilation + cross-link from existing `docs/` |
| D-7 `contributor-handbook.md` | **Sonnet** | extension-surface how-to from existing docs + entry-points |
| D-3 `documentation-set.md` | **(D5)** | index — built in synthesis, not dispatched as a D3 author |

**Every author MUST:** (a) read ONLY its mapped sources (§5) plus this `doc-architecture.md` and the D1 index;
(b) emit `## Methodology` + `## Sources` (§4); (c) carry the per-deliverable caveats verbatim; (d) NOT cite the
generated/WIP docs (O-7); (e) for D-5, verify every signature against live code, not prose.

---

## 8. D4 Verification Plan

The D4 verifier checks each authored deliverable against this architecture. Plan:

1. **Coverage audit vs the FR/NFR/DC census (D1 §3, §4, §9).** Every MUST FR (the §R7 list) and every DC
   (DC-01..15) must have a documentation home across the set. The thin-trace watch-list is the priority check:
   FR-033 / FR-035 / FR-036, NFR-008, DC-05 — each must be *explicitly* present (per the §5 callout requirements),
   not merely implied. SHOULD/COULD requirements may be summarized; MUSTs must be individually traceable.
2. **Trace-link integrity.** Each deliverable's `## Sources` block is checked against its §5 `source_mapping`:
   (a) no cited source outside the mapping; (b) no mapped *authoritative* source left uncited; (c) every
   forward ID claim (FR/NFR/DC) resolves to a real ID in the D1 census. The index (D-3) cross-links must all resolve.
3. **Accuracy-against-code (D-5 api-reference specifically).** Every documented public symbol + signature is
   re-verified against `src/pii_anon/` (the orchestrator API, the 14 §8b modules, the `_g1`.._g7` SDO functions,
   the CLI command list). A documented signature that does not match code is a D4 failure. (D2 already spot-verified
   the orchestrator methods, `test_docs_discoverability.py`, the `_g*` functions, and the entry-point groups exist.)
4. **Epistemic-honesty audit (§4).** D4 flags: any "validated against real users" claim (O-3 violation); any
   citation of `pii-rate-elo-value.md` or `benchmark-summary.md` as canonical (O-7); any statement of the SDO
   verdict as PASS/SOTA rather than the honest NOT_YET / G6 FAIL; any presentation of agent-simulated signal as
   human-validated. Each deliverable must carry `## Methodology` + `## Sources`.
5. **Standing-gate non-regression.** If any D3 deliverable is (or will be) placed under `docs/`, D4 confirms
   `tests/test_docs_discoverability.py` still passes. The D3 artifact-tree outputs live under
   `dev-assist-artifacts/` and do NOT trip this gate; this check is conditional on a D6 promotion decision.
6. **Accessibility (a11y).** **N/A with reason for the Stage-6 artifact-tree deliverables** — they are
   **markdown-only** documents under `dev-assist-artifacts/`, not a rendered doc-site. There is no new a11y agent
   in this stage. **IF** a later wave (D6) renders these to HTML (a doc-site), THEN that render is audited by
   **reusing `dev-assist-testing-a11y-auditor`** (per the dispatch contract — no new a11y agent). Until an HTML
   render exists, a11y is N/A-with-reason, recorded here so D5's verdict frames it correctly.

---

## 9. Decision Record (single-frame — no DIVERGE/CONVERGE)

`config.architecture_diverge = false` (default, config absent) → **single-frame architecture, decided directly.**
No `A-frame-*/B-pugh-scorecard.md/C-validation.md/D-decision.md` files are produced under this directory.

**The single frame chosen: lifecycle-then-audience hybrid (a Frame-B core with Frame-A indexing).** Rationale:

- A pure **audience-first** frame (Frame A) would fragment the rich, chronological program narrative — the
  program's value IS its story (brownfield → SDO arc → feature-complete), and the signature `project-journey.md`
  is inherently lifecycle-ordered. Audience-first would bury the spine.
- A pure **trace-first** frame (Frame C) is unnecessary as the *organizing* principle because the D1 index +
  the D4 coverage audit (§8.1) already guarantee every MUST FR + DC has a home; trace integrity is enforced as a
  *check*, not as the IA shape. Organizing the whole set around requirement coverage would over-fragment for a
  reader and duplicate the traceability-matrix that already exists.
- The chosen hybrid **orders the set by lifecycle** (journey → architecture → API → use/operate → contribute →
  changelog, §2) while the **index (D-3) and the audience map (§3) provide the audience-first entry points**, and
  the **D4 coverage audit provides trace-first rigor**. This captures the strengths of all three frames without a
  3-frame divergence — which a single rich-but-coherent corpus does not justify (and which `architecture_diverge:
  false` forbids by default).

**Brownfield note for D5's verdict:** the deliverable set was resolved against a *mature-brownfield* corpus with
real authored signal (not reverse-extracted guesses), so the only "gaps" are the absent doc-seeds (O-1), the absent
test catalog (O-2), the empty validation dir (O-3), and the absent CONTRIBUTING.md — all **brownfield/process-expected
quality-of-source signals with named substitutes**, NOT authoring failures. D5 should frame any residual gap as
DOCUMENTED-WITH-CAVEAT (the journey/changelog/api authors caveat them), and note that the config-absence (O-4) means
the operator may revise this scope at the D6 sign-off.

---

## Sources

This architecture doc was derived from (read-only):
- `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md` (D1 index — primary input; all sections).
- `references/audience-modes.md` (external-product vs internal-tool resolution → §3 = external product).
- Spot-checks against live repo (verification only, not re-walk): `src/pii_anon/orchestrator.py` (public methods),
  `tests/test_docs_discoverability.py` (gate exists), `src/pii_anon/eval_framework/evaluation/competitive_supremacy.py`
  (`_g1`.._g7`), `pyproject.toml` (4 entry-point groups, lines 61/70/78/87), `README.md` + `LICENSE` (OSS trigger;
  no `CONTRIBUTING.md`), and confirmation of O-1 (no doc-seed.md), O-3 (00-validation empty), O-4 (no documentation:
  block), O-5 (no D-decision.md).

## Methodology

D2 (architecture-only) decided the deliverable set, IA, audience map, per-deliverable source mappings, depth
profile, dispatch notes, and verification plan. It authored no deliverable, rendered nothing, and scored no
frames (architecture_diverge=false → no Pugh loop). Deliverable resolution = config defaults (config block absent,
O-4 → defaults applied) ∩ project-type triggers, all of which fired against confirmed signal → full set IN, none
OUT/DEFERRED, no operator override in scope at D2. The mature-brownfield framing means the source mappings point at
real authored artifacts with named substitutes for the four quality-of-source gaps (O-1/O-2/O-3 + absent
CONTRIBUTING.md). All facts trace to the D1 index; the handful of load-bearing existence/signature claims were
spot-verified against the live repo rather than assumed.
