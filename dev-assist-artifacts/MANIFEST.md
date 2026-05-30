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
| 01-Discovery | NOT_STARTED | — | — |
| 02-Requirements | NOT_STARTED | — | — |
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
| 0. POV Stress Test | NOT_STARTED | — | — |
| 1. Motivation & Background | NOT_STARTED | — | — |
| 2. Personas & Workflows | NOT_STARTED | — | — |
| 3. Market Research (Pugh + JTBD + Kano) | NOT_STARTED | — | — |
| 4. Use Cases | NOT_STARTED | — | — |
| 5. Concept Value Study | NOT_STARTED | — | — |
| 6. Final Discovery Report | NOT_STARTED | — | — |

## Requirements Phase Progress

_To be populated as stage 02 progresses._

| Phase | Status | Artifact | Notes |
|-------|--------|----------|-------|
| R0. UC↔P/G/O Bridge | NOT_STARTED | — | — |
| R1. Low-Fidelity Requirements | NOT_STARTED | — | — |
| R2. Interview Guide | NOT_STARTED | — | — |
| R3. Simulated Interviews | NOT_STARTED | — | — |
| R4. High-Fidelity Requirements | NOT_STARTED | — | — |
| R5. Prioritization Survey | NOT_STARTED | — | — |
| R6. Simulated Survey Responses | NOT_STARTED | — | — |
| R7. Prioritization Analysis | NOT_STARTED | — | — |
| R8. Final Requirements Artifacts | NOT_STARTED | — | — |
| R9. Cleanup + Verification-Strengthening | NOT_STARTED | — | — |
| R10. NFR Threshold Validation | NOT_STARTED | — | — |

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
| Discovery | 0 | 7 |
| Requirements | 0 | 7 |
| Design | 0 | 7 |
| Development | 0 | 7 |
| Testing | 0 | 7 |
| **Total** | **7** | **7** |

---

_Scaffolded by developer-assistant `/dev-assist-start` for the library pillar of the 3-repo pii-anon SOTA program._
