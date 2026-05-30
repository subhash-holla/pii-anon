---
dev_assist_config: ./developer-assistant.yaml
schema_version: 2
---

# pii-anon — PDLC Stage Manifest

> Single source of truth for stage progress, sub-phase status, agent deployments, sign-offs, and test mode. Updated by stage skills as work progresses; reconcile manually via `/dev-assist-manifest-update`.
>
> Per-project plugin configuration lives in `developer-assistant.yaml` (see `dev_assist_config` in frontmatter). This project is the **library** pillar of a coordinated 3-repo program — see `../PROGRAM-MANIFEST.md`.

## Stage Status

| Stage | Status | Started | Completed |
|-------|--------|---------|-----------|
| 00-Brownfield Assessment | COMPLETE | 2026-05-30 | 2026-05-30 |
| 01-Discovery | COMPLETE | 2026-05-30 | 2026-05-30 |
| 02-Requirements | COMPLETE | 2026-05-30 | 2026-05-30 |
| 03-Design | NOT_STARTED | — | — |
| 04-Development | NOT_STARTED | — | — |
| 05-Testing | NOT_STARTED | — | — |

Valid status values: `NOT_STARTED`, `IN_PROGRESS`, `COMPLETE`, `BLOCKED`, `DEFERRED`, `SUPERSEDED`.

## Project Metadata

| Field | Value |
|---|---|
| Project name | pii-anon (pii-anon-code library) |
| Created | 2026-05-30 |
| Plugin version | 0.1.0 |
| Plan reference (current stage) | `.claude/plans/cobalt-sentinel.md` (Discovery) · program master: `.claude/plans/users-subhashholla-downloads-pii-enterp-rippling-scott.md` |
| Language profile | python |
| Active domain packs | none (project-authored axioms instead — see `00-axioms/project-axioms.yaml`) |
| Test mode | tests-as-stage |
| Sign-off policy | po-required (sign-off required at every stage transition) |
| Scope | brownfield (mature v1.4.0 library; legacy `pdlc-artifacts/` present, to be migrated) |
| Run depth | full rigor (interviews 30 / surveys 60 / research 30 / SME 9) |
| Git | branch `pdlc/sota-program`; baseline tag `pre-pdlc-program` |

## Discovery Phase Progress

_To be populated as stage 01 progresses._

| Section | Status | Artifact | User Approved? |
|---------|--------|----------|----------------|
| 0. POV Stress Test | VALIDATED | 00-pov-stress-test.md | ✅ AGENT_SIMULATED (POV pivot: measurement-first; eval headline) |
| 1. Motivation & Background | VALIDATED | 01-motivation-background.md | ✅ |
| 2. Personas & Workflows | VALIDATED | personas.md | ✅ 7 personas (2 eval-dedicated) |
| 3. Market Research (Pugh + JTBD + Kano) | VALIDATED | 03-market-research.md | ✅ detectors + eval-benchmarks, live-verified 2026 |
| 4. Use Cases | VALIDATED | 04-use-cases.md | ✅ 28 UCs (15 eval / 12 swarm / 1 both) + 3-SME panel |
| 5. Concept Value Study | VALIDATED | 06-concept-value-study.md | ✅ 8-archetype cohort + SME findings |
| 6. Final Discovery Report | VALIDATED | discovery-report.md | ✅ canonical |

## Requirements Phase Progress

_To be populated as stage 02 progresses._

| Phase | Status | Artifact | Notes |
|-------|--------|----------|-------|
| R0. UC↔P/G/O Bridge | VALIDATED | traceability-matrix.md (PGOs + 0-orphan scan) | ✅ |
| R1. Low-Fidelity Requirements | FOLDED | (into R4) | representative scale |
| R2. Interview Guide | FOLDED | (reused Discovery concept-value/Kano) | representative scale |
| R3. Simulated Interviews | FOLDED | (reused Discovery §5 cohort) | representative scale |
| R4. High-Fidelity Requirements | VALIDATED | requirements-document.md (39 FR + 26 NFR) | ✅ |
| R5. Prioritization Survey | FOLDED | (6-respondent MoSCoW survey) | representative scale |
| R6. Simulated Survey Responses | VALIDATED | (6 simulated-survey-respondent) | ✅ |
| R7. Prioritization Analysis | VALIDATED | requirements-document.md §R7 (MoSCoW) | ✅ ~56/38/6 MUST/SHOULD/COULD |
| R8. Final Requirements Artifacts | VALIDATED | requirements-document.md + traceability-matrix.md | ✅ canonical |
| R9. Cleanup + Verification-Strengthening | VALIDATED | (boolean G/W/T + quantified thresholds) | ✅ |
| R10. NFR Threshold Validation | VALIDATED | requirements-document.md §R10 | ✅ 0 DIVERGED · 5 PERSONA-CONDITIONAL · 1 REVISE-LOOSER |

## Design Phase Progress

_To be populated as stage 03 progresses._

| Diamond | Status | Artifacts | Outcome |
|---------|--------|-----------|---------|
| Prep | NOT_STARTED | — | — |
| D1 Design Cases | NOT_STARTED | — | — |
| D2 Workflow | NOT_STARTED | — | — |
| D3 UI | NOT_STARTED | — | — |
| D4 System | NOT_STARTED | — | — |
| D5 Architecture | NOT_STARTED | — | — |
| D6 Synthesis | NOT_STARTED | — | — |

## Development Phase Progress

_To be populated as stage 04 progresses._

| Wave | Status | Artifacts | Notes |
|------|--------|-----------|-------|
| W1. Preflight | NOT_STARTED | — | — |
| W2. Planning | NOT_STARTED | — | — |
| W3. Quality | NOT_STARTED | — | — |
| W4. Testing setup | NOT_STARTED | — | — |
| W5. Stories | NOT_STARTED | — | — |
| W6. Execution | NOT_STARTED | — | — |

## Testing Phase Progress

_To be populated as stage 05 progresses._

| Section | Status | Artifact | Notes |
|---------|--------|----------|-------|
| Test architecture | NOT_STARTED | — | — |
| NFR verification matrix | NOT_STARTED | — | — |
| Accessibility test plan | NOT_STARTED | — | n/a — Python API + CLI, no interactive web UI |
| Benchmark harnesses | NOT_STARTED | — | MoE-floor / agentic / multimodal suites |
| Examples and tests catalog | NOT_STARTED | — | — |

## Sign-offs

_Populated by `/dev-assist-signoff` at stage / gate transitions (policy: po-required)._

| Sign-off ID | Date | Type | Scope | Signer | File |
|---|---|---|---|---|---|
| SO-01-m1 | 2026-05-30 | milestone-close | M1: brownfield assessment + legacy migration (24 items) + vendored briefs; gates → Discovery | AGENT_SIMULATED | `_signoffs/SO-01-m1.yaml` |
| SO-02-discovery | 2026-05-30 | stage-transition | Discovery COMPLETE: POV pivot (eval headline), 7 personas, 28 UCs, concept-value; 30 agents; gates → Requirements | AGENT_SIMULATED | `_signoffs/SO-02-discovery.yaml` |
| SO-03-requirements | 2026-05-30 | stage-transition | Requirements COMPLETE: 39 FR + 26 NFR, R7 MoSCoW, R10 (0 DIVERGED); 22 agents; gates → Design | AGENT_SIMULATED | `_signoffs/SO-03-requirements.yaml` |

## Methodology Notes

- **Program context**: library pillar of a 3-repo program (`../PROGRAM-MANIFEST.md`). Siblings: `pii-anon-eval-data` (active dev-assist project, mid-Development S5 — owns benchmark data + scorer/stats harness) and `pii-anon-research-paper` (thin; Paper 1 AsiaCCS/CODASPY, Paper 2 EMNLP). dev-assist owns within-stage dispatch; the Workflow tool fans out at between-options / between-repos / between-sprints seams.
- **Cadence**: user validates at each checkpoint (program milestones M0–M7); sign-off required at every stage transition (po-required).
- **Evidence**: two user research briefs (`pii_enterprise_landscape_may26.md`, `pii_eval_may26.md`) + existing repo docs (README, `docs/swarm-architecture.md`, `docs/pii-rate-elo.md`, benchmark artifacts) + sibling `pii-anon-eval-data` artifacts + own SOTA research (MoE architectures + PII benchmarks).
- **Active axioms (6)**: see `00-axioms/project-axioms.yaml` — synthetic-only-no-real-pii (AX-001), deterministic-pseudonymization (AX-002), ensemble-recall-floor-guarantee (AX-003, the load-bearing T1 invariant), anonymization-pseudonymization-separation (AX-004), calibrated-abstention (AX-005), least-privilege-agentic-interception (AX-006). To be refined in Stage 3 D0 prep.
- **Verified code findings driving design** (from the approved plan): F1 the swarm never executes sparsely (`policy/router.py:60` runs all engines serially via `default_compat`); F2 the Glicko RD cannot converge (`elo.py:198` is match-count-only) → replace with Bayesian Bradley-Terry; F3 most Tier-3/stats harness already exists in eval-data (build `assemble_paired_set` first); F4 live bug `swarm.py:745` (`predict_candidate` called without `text=`) → fix in a `FEATURE_VERSION` 3→4 retrain.
- **Epistemic honesty**: all simulated-cohort outputs are marked AGENT_SIMULATED and are not a substitute for real consumers; Pass-2 real-user research is a documented follow-up.
- **Steering decision (2026-05-30, user)**: the eval-integrity findings (50-sample smoke-run published numbers; internally-incoherent significance computation; results presentation) are NOT a pre-PDLC hotfix — they are folded into the overhaul as a first-class **redesign of how the evaluation pillar (pii-rate-elo) computes and presents results** (Theme 2 / Pillar 1). Carry them as Discovery→Requirements→Design inputs (NFRs on statistical rigor, a canonical-run/publication policy, and honest reporting). Treat all current README/benchmark numbers as **PROVISIONAL** until a certified canonical run + corrected significance pipeline land.
- **Co-equal pillars (2026-05-30, user)**: Pillar 1 — the `pii-rate-elo` **evaluation framework** — is a FIRST-CLASS product overhauled through ALL five stages (its own personas, use cases, FRs/NFRs, design diamonds, stories, and verification), co-equal with Pillar 2 (the swarm), AND it is the instrument that measures Pillar 2. Every stage must give the evaluation pillar equal weight. Candidate enhancements for both pillars live in `01-discovery/_inputs/enhancement-catalog.md` (validated in Discovery/Requirements — not locked).

## Handoff Signals

### Brownfield Assessment complete (2026-05-30)
> M1 assessment done via a 7-agent fan-out (5 per-stage + 2 legacy-inventory). Per-stage: **Discovery PARTIAL · Requirements PARTIAL · Design STRONG · Development PARTIAL · Testing PARTIAL**. Findings: 0 SHOWSTOPPER · 0 CATASTROPHIC · 12 MAJOR · 11 MINOR · 8 OBSERVATION. Headline MAJORs: (1) published benchmark numbers are a 50-sample smoke run (`canonical_claim_run=False`); (2) internally-incoherent significance computation; (3) recall-floor AX-003 not guaranteed by-construction in the swarm path; (4) swarm fails its own NFR targets (F1 0.85→0.610) / last by composite. Artifacts: `00-brownfield-assessment/assessment-2026-05-30.md` + `artifact-inventory.md` (24 legacy files; 0 deletions; mostly WRAP→`03-design/_inputs/`). **Next:** `/dev-assist-migrate` (24 items) + `/dev-assist-absorb` (2 landscape briefs) → `/dev-assist-discovery` (brownfield mode). M1 sign-off pending migrate+absorb.

### M1 complete → Discovery (2026-05-30)
> Migration done: 24 legacy items → 5 canonical citation artifacts (`03-design/_inputs/swarm-moe-prior-art.md`, `ensemble-v2-and-speed-prior-art.md`; `03-design/moe-architecture-and-guarantee.md`; `04-development/_provenance/legacy-pdlc-manifest-moe-guarantee.md`; `05-testing/benchmark-evidence/legacy-benchmark-evidence.md`); originals preserved untouched (`migration-log.md`). Two landscape briefs vendored to `01-discovery/_inputs/`. **M1 signed off** (`_signoffs/SO-01-m1.yaml`). 12 MAJOR findings carried forward as Discovery/Requirements/Design inputs — eval-integrity folded into the Pillar-1 overhaul; recall-floor AX-003 is the Theme-1 design mandate; current benchmark numbers PROVISIONAL. **Ready for Discovery.** Run: `/dev-assist-discovery` (brownfield mode). DATA track (`pii-anon-eval-data` S5–S7) runs in parallel.

### Discovery complete → Requirements (2026-05-30)
> Stage 1 COMPLETE (30 AGENT_SIMULATED agents, representative scale). **POV pivot:** measurement-first; Pillar-1 `pii-rate-elo` is the headline; pseudonymization-integrity is the defensible empty quadrant; swarm re-scoped OFF raw-F1 (OpenAI Privacy Filter F1≈0.96 + Presidio contest it) onto reversibility + recall-floor + audit + orchestration. 7 personas (2 eval-dedicated), 28 use cases (15 eval / 12 swarm / 1 both), concept-value (3 high-willingness anchor on eval+pseudonymization-integrity; medium = "prove it"). Signed off `_signoffs/SO-02-discovery.yaml`. **Top Requirements priorities:** (1) eval-integrity foundation [critical path]; (2) pseudonymization-integrity + distinct anon-vs-pseudo families; (3) recall-floor by construction (AX-003); (4) running Tier-3 + agentic + multimodal. Cross-repo: stats/scorers/`assemble_paired_set` in eval-data S5–S7. **Ready for Requirements.** Run: `/dev-assist-requirements`.

### Requirements complete → Design (2026-05-30)
> Stage 2 COMPLETE (22 AGENT_SIMULATED agents). **39 FRs + 26 NFRs**, evaluation-led (≈20 eval / 17 swarm / 2 both), MoSCoW ≈56/38/6. MUST critical path = eval-integrity foundation (FR-003 Bradley-Terry, FR-004 coherent-significance, FR-008 canonical-run, NFR-001/002/006) + headline novelty (FR-009/010 pseudonymization-integrity + distinct families, FR-011 Tier-3) + recall-floor-by-construction (FR-016/NFR-011, AX-003). R10: 0 DIVERGED (5 PERSONA-CONDITIONAL + 1 REVISE-LOOSER ECE). Cross-repo `external_refs` set (DATA-blocking: `assemble_paired_set`/canary/query-aware/agentic-oracle in eval-data S5–S7). Signed off `_signoffs/SO-03-requirements.yaml`. **Ready for Design.** Run: `/dev-assist-design` (5-Diamond Cascade).

## Pivots Log

| Date | Stage(s) affected | Source finding | Plan file |
|---|---|---|---|
| _(empty)_ | | | |

## Directory Structure

```
dev-assist-artifacts/
├── MANIFEST.md                           ← this file (status tracker)
├── 00-axioms/                            ← project-axioms.yaml (6 project-authored axioms)
├── 00-security/                          ← security-checklist + exceptions (no-real-PII invariant)
├── 00-brownfield-assessment/             ← Stage 0 gap analysis (created by /dev-assist-assess)
├── 00-validation/                        ← validator reports
├── _signoffs/                            ← sign-off records (po-required)
├── 01-discovery/                         ← Stage 1 outputs
├── 02-requirements/                      ← Stage 2 outputs
├── 03-design/                            ← Stage 3 outputs
├── 04-development/                       ← Stage 4 outputs
└── 05-testing/                           ← Stage 5 outputs
```

Sibling files:
- `developer-assistant.yaml` — plugin configuration.

## Agent Deployment Ledger

| Stage | Agents deployed | Running total |
|---|---|---|
| Brownfield (M1 assess) | 7 | 7 |
| Discovery | 30 | 37 |
| Requirements | 22 | 59 |
| Design | 0 | 59 |
| Development | 0 | 59 |
| Testing | 0 | 59 |
| **Total** | **59** | **59** |

---

_Scaffolded by developer-assistant `/dev-assist-start` for the library pillar of the 3-repo pii-anon SOTA program._
