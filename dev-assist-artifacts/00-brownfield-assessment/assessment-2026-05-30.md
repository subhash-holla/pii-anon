# Brownfield Assessment — pii-anon (pii-anon-code library)

**Date**: 2026-05-30
**Assessment by**: developer-assistant, `dev-assist-brownfield-assessment` skill, via a 7-agent Workflow fan-out (5 per-stage signal extractors + 2 legacy-inventory classifiers; read-only)
**Project root**: `/Users/subhashholla/Development/pii_anonymize_pseudonymize/pii-anon-core/pii-anon-code`
**Program**: library pillar of the 3-repo pii-anon SOTA program (`../PROGRAM-MANIFEST.md`), Milestone M1.

---

## 1. Project Profile

### Identity
| Field | Value |
|---|---|
| Name | pii-anon |
| Description | Dual-pillar privacy library: (1) `pii-rate-elo` evaluation framework, (2) detection + anonymization/pseudonymization (fast regex path + the 4-layer **swarm**) |
| Primary language | Python (3.10–3.13; 3.14 experimental) |
| Framework | none (standalone library) |
| Runtime category | library + Typer/Rich CLI (no web/GUI surface) |

### Scope & Maturity
| Field | Value |
|---|---|
| Source LOC | 37,011 (Python, `src/pii_anon/`) |
| Tests | **2,548** test functions, 111–112 files, 523 test classes |
| Git commits | 24 (large AI-assisted/squashed; since 2026-02-17, ~3.5 mo) |
| Contributors | 3 identities; effectively single-maintainer |
| Top-level dirs | `src/ tests/ docs/ scripts/ artifacts/ notebooks/ packages/ pdlc-artifacts/ dev-assist-artifacts/ …` |

### Tooling
| Capability | Status |
|---|---|
| Linter | ruff (default E/F rules only — no explicit `select`) |
| Type-checker | mypy `strict = true` |
| Test framework | pytest (markers: performance/asyncio/parametrize; branch coverage, `--cov-fail-under=84`) |
| CI | GitHub Actions — `ci.yml` (lint→type→test→perf→build→pkg-size→twine→CLI-smoke→notebook; Py 3.10–3.13 matrix + macOS/Windows smoke) + `release.yml` (OIDC trusted publish) |

**Read:** unusually mature for its age — production-grade CI, strict typing, ~2,500 tests. The maturity is in **code/CI**, not in **formal PDLC prose** (expected for a code-first OSS library) — which is exactly the brownfield profile where this program's value is formalizing the implicit and closing the four theme gaps.

---

## 2. Per-Stage Signal Extraction

| Stage | Dimension | Rating | Top evidence |
|---|---|---|---|
| **Discovery** | Personas | PARTIAL | `README.md:18-22` tool-matrix "When to use"; explicit persona cards exist but feature-scoped (`pdlc-artifacts/swarm/discovery/discovery-report.md:66-78`) |
| | Use cases | **STRONG** | 5 real-world jobs (`README.md:360-370`); 18 CLI commands (`cli.py:44-769`); predictor contract (`docs/evaluate-your-pipeline.md:53-108`) |
| | Workflow maps | **STRONG** | `docs/quickstart.md:18-110`; `docs/tutorial-llm-pipeline.md:7-49`; legacy flow diagrams |
| | Market context | **STRONG** | leaderboard `README.md:144-150`; "why pii-rate-elo over F1" `README.md:227-247`; significance tables `README.md:195-218` |
| | Concept value | WEAK | no testimonials/NPS/downloads; value asserted via benchmark + test-count only |
| **Requirements** | Functional reqs | **STRONG** | 18 FRs with Given/When/Then (`pdlc-artifacts/swarm/requirements/functional-requirements.md`); CHANGELOG/API/tests |
| | Quantified NFRs | **STRONG** | 15 NFRs + `FloorGateConfig` (`composite.py:662-690`); executable SLA tests (`tests/performance/test_perf_sla.py`) |
| | Traceability | **STRONG** | Discovery→Req matrix; in-code `EVIDENCE_REGISTRY` (`references.py:323-353`); `docs/evidence-ledger.md` |
| | Prioritization | PARTIAL | priority tiers in swarm req doc; no living repo-level roadmap/known-limitations |
| | Threshold validation | PARTIAL | floor-gate report exists **but** significance numbers incoherent; floors disabled in canonical run |
| **Design** | Design cases | **STRONG** | concern-aligned subpackages; documented in `final-architecture.md` |
| | Workflow shape | **STRONG** | sync/async orchestrators; stream/batch; `PolicyRouter`; escalation (`orchestrator.py:918-967`) |
| | UI metaphor | **STRONG** | 17-command typer tree; 40+ Make targets |
| | System archetype | **STRONG** | "three tools, one library"; 4-layer swarm (`docs/swarm-architecture.md:17-53`) |
| | Architecture pattern | **STRONG** | 4 registries + ports-adapters (`EngineAdapter`) + strategy hierarchies |
| | Cross-cutting axioms | PARTIAL | determinism/calibration strong; **recall-floor not by-construction in swarm path** |
| **Development** | TDD discipline | PARTIAL | strong outcome (2,548 tests, 84% gate) but no RED→GREEN in squashed history; one perf test relaxed to pass |
| | Story discipline | WEAK | no commit convention (1/24 Conventional); no FR/UC trace tokens |
| | Review gates | MISSING | no CODEOWNERS/CONTRIBUTING/PR template; 23/24 direct-to-main |
| | Reviewer specialization | MISSING | single author; no routing |
| | CI quality | **STRONG** | full gate chain on every push/PR; mypy strict; cross-version+OS matrix; perf is a hard gate |
| **Testing** | Test type coverage | **STRONG** | 2,548 tests; unit/integration/perf tiers; registered markers; **no property/fuzz (`@given`=0)** |
| | Benchmark harness | PARTIAL | excellent self-tested floor-gate machinery, but **never run in CI**; committed artifacts are a 50-sample smoke run |
| | Accessibility | MISSING (N/A) | Python API + CLI, no web UI — excluded, not penalized (CLI smoke covers the analog) |
| | Examples+tests | PARTIAL | README/summary byte-synced + quickstart notebook executed; ~15 doc snippets not machine-verified |

### Aggregate per-stage rating
| Stage | Rating | Strongest | Weakest |
|---|---|---|---|
| Discovery | **PARTIAL** | Use cases | Concept value |
| Requirements | **PARTIAL** | Traceability | Threshold validation |
| Design | **STRONG** | Architecture pattern | Cross-cutting axioms (recall-floor gap) |
| Development | **PARTIAL** | CI quality | Review gates / Reviewer specialization (MISSING) |
| Testing | **PARTIAL** | Test type coverage | Benchmark-harness CI-wiring + published-evidence integrity |

---

## 3. Findings (5-Severity Taxonomy)

| Severity | Count |
|---|---|
| SHOWSTOPPER | 0 |
| CATASTROPHIC | 0 |
| MAJOR | 12 |
| MINOR | 11 |
| OBSERVATION | 8 |

### CATASTROPHIC
(none — no PDLC stage is absent; signal is rich in code/CI.)

### MAJOR (12)

**Academic defensibility / benchmark integrity (Theme 2 — highest program priority):**
1. **Published benchmark numbers are from an uncertified 50-sample smoke run.** `benchmark-results.json` has `max_samples=50`, `canonical_claim_run=False`, `floor_pass=False`, `all_competitors_available=False`, yet `README.md:20` publishes "F1=0.76, 0.4 ms/record, 3M docs/hour" as a Production/Stable headline. The harness self-refuses to certify it. → Regenerate + commit a canonical run (`max_samples=0`, all competitors, strict runtime) and gate README numeric updates on `canonical_claim_run==True`; until then annotate provenance. *(Held at MAJOR not CATASTROPHIC: the latency floor failures are transparently disclosed in-README with a composite-Elo rebuttal.)*
2. **Statistical-significance reporting is internally incoherent.** Every pairwise comparison is "n.s." at p≈0.49–0.50 even where Cohen's d is "large" (up to +1.76), while 95% CIs are ~0.002 wide and several do **not** bracket their own point estimate (`pii-anon F1=0.756, CI [0.728,0.730]`; `presidio 0.491, CI [0.499,0.501]`). A paired bootstrap on ~149K samples cannot produce all three — this is a computation/labeling bug. `docs/benchmark-summary.md:57-80`. → Audit `scripts/render_benchmark_summary.py` + the bootstrap impl before any academic claim rests on these numbers.
3. **The competitor latency-floor gate fails and is not enforced.** `floor-gate-report.md:3` "Overall floor pass: False" (short_chat / structured_form_latency / log_lines); the canonical Make suite passes `--no-enforce-floors` (`Makefile:96`) and **CI never runs the benchmark** (`ci.yml` only runs in-process `@performance` SLA, which use loose absolute thresholds). A competitive regression passes every PR. → Wire a (nightly/release) floor-gate CI job + reconcile the Makefile.

**Swarm / Theme-1 (MoE redesign anchor):**
4. **The swarm fails its own quantified NFR thresholds, still marked COMPLETE.** NFR-004 F1≥0.85 vs shipped **0.610**; NFR-005 precision≥0.80 vs **0.486**; NFR-007 dominance violated (last by composite, 0.556). → Re-baseline the swarm NFR targets to measured/realistic values + add an actual-vs-target column; the MoE redesign should target a measured floor, not the aspirational 0.85.
5. **The recall-floor axiom (AX-003) is NOT guaranteed by construction across the pipeline.** The MoE path floor-weights non-routed experts (`moe.py:354-378`), but `SwarmFusionStrategy.merge()` Layer-4 emission gate + `SEMANTIC_TYPES` corroboration can drop a shared-layer (regex-oss) finding that fell below fast-pass (`swarm.py:651-661`). Two divergent floor mechanisms; only fast-pass regex hits are protected. → Unify into one "shared-layer span set" superset invariant + property test + per-language recall CI gate **before** the router redesign.
6. **The swarm is under-tested for regression and its retrain code is coverage-omitted.** `swarm_train.py`, `swarm_datasets.py`, `evaluation/pipeline.py` are in the coverage omit list (`pyproject.toml:194-198`); no gate prevents further swarm regression. → Freeze a golden swarm scorecard + add a regression gate (composite/F1 ≥ plain pii-anon on accuracy profiles, or gate on recall where it leads at 0.818).

**Correctness / Theme-3 / Testing:**
7. **No property-based/fuzz testing for crypto + checksum invariants.** A pseudonymization library with HMAC tokenization + reversible mapping + Luhn/IBAN/ABA validators has zero `@given`; reversibility/collision-freedom bugs are a data-leak class. → Add `hypothesis` for tokenize↔detokenize round-trip, checksum accept/reject, span round-trips.
8. **Theme-3 agentic interception (AX-006) has zero design realization.** Declared axiom, no module/port/workflow in `src/`. Biggest greenfield gap. → Full Design diamond for the interception surface (bounded scope, no-raw-PII-post-masking) before any Theme-3 code.
9. **Market-context claims are in tension with shipped numbers** (swarm "Recall=0.82, highest" headline vs composite last; "outperform Presidio/Scrubadub" vs all-pairwise-n.s.). `README.md:21,24`. → Qualify headline claims to the profiles/metrics where they hold and are significant.
10. **No real concept-value / adoption evidence** for a published v1.4.0 package (no testimonials/NPS/downloads/adopters). → Capture real PyPI/GitHub stats + 3–5 design-partner testimonials; AGENT_SIMULATED concept-value panel as interim.
11–12. *(Testing)* benchmark floor gate never in CI (#3 from the Testing lens) and the 50-sample committed artifacts (#1 from the Testing lens) — same root causes, recorded per-stage.

### MINOR (11)
- No library-level `personas.md` (signal implicit in README + feature-scoped legacy cards).
- Composite-weight doc/code drift (`docs/pii-rate-elo.md` F1=0.25/ref 100k dph vs `composite.py` 0.50/1M).
- `docs/evidence-ledger.md` stale (lists `llm_guard`, not the shipped `gliner`; only 6 claims).
- No living roadmap / KNOWN-LIMITATIONS doc.
- No canonical `03-design/` doc (rich legacy design unmigrated).
- MoE expert strengths are hardcoded magic numbers without provenance citation (`moe.py:439-605`).
- ruff has no explicit `[tool.ruff.lint] select` (no I/B/UP/**S**-security rules) for a security library.
- No human-review gate (CODEOWNERS/CONTRIBUTING/PR template) — mitigated by strong CI.
- No reviewer specialization — solo maintainer.
- No commit-message/branch convention (1/24 Conventional).
- README/docs usage snippets not executed (no doctest harness; only the quickstart notebook runs in CI).

### OBSERVATION (8)
- `tutorial-llm-pipeline.md` thin vs sibling docs (flagship LLM/RAG use case).
- Two named research briefs cited by MANIFEST not vendored in-repo → vendor into `01-discovery/` (they're the two landscape docs).
- Canonical Requirements/Design freshly bootstrapped (NOT_STARTED) while rich legacy signal exists → migrate.
- Architecture pattern is exemplary — preserve the registry/factory seams when the router lands.
- Legacy swarm Design shows real DIVERGE/CONVERGE rigor (Proposal A/B + 7-criterion Pugh) — reuse as the Theme-1 template.
- One perf test was weakened to pass; absolute perf thresholds don't imply published speed claims hold.
- Accessibility N/A (Python API + CLI) — formalize `NO_COLOR`/non-TTY plain-output tests instead.

---

## 4. Bring-Forward Plan (Impact × Effort)

### High impact / Low effort (do first)
1. **Re-baseline swarm NFR targets** to measured reality + add actual-vs-target column — anchors Theme-1. (S)
2. **Migrate the legacy `pdlc-artifacts/`** (esp. `swarm/discovery/precision-diagnosis`, `engine-correlation`, `confidence-analysis`, `design/*`, `moe-guarantee-analysis.md`) into `03-design/_inputs/` as cited Theme-1 design inputs — see `artifact-inventory.md`. (S–M, mostly mechanical)
3. **Annotate the published benchmark provenance** (50-sample smoke) in README/summary immediately; schedule a canonical run. (S)
4. **Vendor the two landscape briefs** into `01-discovery/` so Discovery traces to in-repo evidence. (S)

### High impact / High effort (plan into sprints)
1. **Fix the statistical-significance computation** (paired bootstrap CIs/p-values) + regenerate a **canonical** benchmark run — prerequisite for all Theme-2 paper claims. (M–L)
2. **Unify the recall-floor (AX-003) by construction** across MoE + swarm paths + property test + per-language recall CI gate — prerequisite for the Theme-1 router redesign. (M–L)
3. **Wire benchmark floor-gate + swarm regression gate into CI** (nightly/release on a committed fixture). (M)
4. **Author the agentic-interception design** (Theme-3 / AX-006) — full greenfield diamond. (L)
5. **Add property-based testing** (hypothesis) for tokenization + checksum invariants. (M)

### Low impact / Low effort (polish)
ruff explicit `select` (+S); fix composite-weight doc/code drift; refresh `evidence-ledger.md` to the 5-system baseline; add KNOWN-LIMITATIONS/roadmap; adopt Conventional Commits + commitlint; expand `tutorial-llm-pipeline.md`; add a doctest/exec harness for README snippets.

### Low impact / High effort (defer)
CODEOWNERS + human-review gates (until a 2nd maintainer); full 30-user concept-value research (until real adopters — AGENT_SIMULATED interim now).

---

## 5. Retroactive Artifact Proposals

| Stage | Canonical artifact | Extractable from | Gaps to fill |
|---|---|---|---|
| Discovery | `01-discovery/personas.md` | README tool-matrix + use cases + legacy persona cards | the v1.4.0 headline persona (3rd-party Pipeline Evaluator using pii-rate-elo); Theme-3/4 users; priority tiering |
| Discovery | `01-discovery/use-cases.md` | 5 README jobs + 18 CLI commands + `__all__` + quickstart | UC-IDs + Given/When/Then; agentic + multimodal jobs (net-new) |
| Discovery | `01-discovery/market-analysis.md` | leaderboard + rank-swap narrative + significance tables | Pugh/Kano framing; reconcile claims vs n.s./floor failures; add the 2 briefs |
| Requirements | `02-requirements/requirements-document.md` | legacy swarm FR/NFR docs (migrate-and-update) | re-baseline drifted NFRs; provisional_status column; extend to regex-core + pii-rate-elo; link NFR→SLA test |
| Requirements | threshold-validation report | floor-gate report + floor-baseline.json + MDE block + SLA tests | fix incoherent significance; floor-enforcement policy; declare measured-vs-smoke provenance |
| Design | `03-design/` whole-system design (5-Diamond) + `_inputs/` + ADRs | src layout + `swarm-architecture.md` + legacy proposals/critique/final-architecture | unified recall-floor section; agentic-interception design (fresh); whole-system synthesis tying the 3 offerings |
| Development | `CONTRIBUTING.md` + CODEOWNERS + per-story review evidence | Makefile `all` gate chain + `release-guide.md` + legacy eval rounds | commit convention; FR/NFR trace tokens; review-gate verdicts |
| Testing | test-architecture + NFR-verification matrix + examples-and-tests catalog | the suite taxonomy + floor-gate methodology + sync-tests | property-test + multimodal/agentic surface gaps; per-number provenance; doctest closure |

---

## 6. Pass-2 Recommendations (where real evidence unlocks most value)
| Recommendation | Reason | Protocol |
|---|---|---|
| Real adopter/persona validation | Discovery value rests on benchmark + test-count, no demand signal | 3–5 design-partner interviews + PyPI/GitHub telemetry |
| Canonical benchmark on declared hardware | Published numbers are a 50-sample smoke run | full `max_samples=0` run, all competitors, pinned env |
| Real-SME academic review | Theme-2 defensibility; the significance bug shows internal review gaps | 1–2 statistician/SME reviews of the eval methodology |

---

## 7. Methodology Block (Epistemic Honesty)
- **What this is:** an agent-conducted (7-agent fan-out), read-only, pattern-match of the repo's PDLC signal against the developer-assistant rigor rubric. Every STRONG/PARTIAL/WEAK cites file:line; MISSING cites the search that found nothing.
- **What it is NOT:** a code-correctness review (structural, not behavioral); a verdict on project quality; a substitute for maintainer context.
- **Context override applied:** mature code-first OSS library — "no formal PDLC prose doc" findings are down-severitied (MINOR/OBSERVATION) with rationale; genuinely missing rigor in a *shipped* library (failing/un-enforced floors, incoherent significance, swarm regression risk, smoke-run-as-published-evidence) stays MAJOR+.
- **What I might be missing:** maintainer intent behind the smoke-run artifacts; whether the speed-floor failures are an accepted trade-off; private adoption signal. The user should confirm these.

---

## 8. Next Steps (M1 continuation)
1. **`/dev-assist-migrate`** — walk the 24-item `artifact-inventory.md` (0 archives, 0 deletions; mostly WRAP into `03-design/_inputs/` + `05-testing/benchmark-evidence/`). The `swarm/` precision/correlation/calibration diagnoses and `moe-guarantee-analysis.md` are load-bearing Theme-1 design inputs.
2. **`/dev-assist-absorb`** the two landscape briefs into `01-discovery/`.
3. **`/dev-assist-discovery`** (brownfield mode) — pre-populated from this assessment's STRONG/PARTIAL signal; gaps (concept-value, agentic/multimodal personas+UCs) surfaced as open items.
4. Record the **M1 sign-off** (po-required) and update MANIFEST + PROGRAM-MANIFEST.

> The DATA track (`pii-anon-eval-data` S5–S7) runs in parallel and is unblocked.
