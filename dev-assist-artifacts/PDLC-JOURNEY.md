# PDLC Journey — pii-anon (autonomous pass, 2026-05-30)

> Consolidation of one end-to-end developer-assistant PDLC pass on the `pii-anon-code` library, run autonomously. AGENT_SIMULATED at representative scale; the user is the Pass-2 cohort. Full per-stage artifacts under `dev-assist-artifacts/`; program coordination in `../PROGRAM-MANIFEST.md`.

## Traceability spine (POV → code → verdict)
**POV** (measurement-first; eval headline; pseudonymization-integrity quadrant; swarm off the F1 arms race)
→ **7 personas** (2 eval-dedicated) → **28 UCs** (15 eval / 12 swarm / 1 both)
→ **39 FRs + 26 NFRs** (eval-led; R10 0-DIVERGED) → **15 Design Cases** + 3 Pugh-won decisions
→ **S1-01 `SharedLayerProjector`** (real code, RED `ef85166`→GREEN `548f576`, 7/7 green)
→ **Release verdict: SHIP-WITH-CAVEATS** (foundation) / **DEFER** (full redesign).

## Per-stage summary
| Stage | Outcome | Agents |
|---|---|---|
| M0 Bootstrap | 3-repo program scaffold; CODE dev-assist project; 6 axioms; tag `pre-pdlc-program` | — |
| M1 Brownfield assess + migrate | 12 MAJOR findings; 24 legacy files → 5 canonical citation artifacts (0 deletions) | 7 |
| Discovery | POV pivot (measurement-first); personas/market/UCs/concept-value | 30 |
| Requirements | 39 FR + 26 NFR; traceability + cross-repo `external_refs`; R10 0-DIVERGED | 22 |
| Design | 15 DCs; SharedLayerProjector / Bayesian-BT rating / agentic pre-filter; 5-SME (1 CAT resolved) | 17 |
| Development | W1–W5 plan (7 sprints, ~30 stories) + **S1-01 flagship DONE** (recall-floor) | direct TDD |
| Testing | release-readiness verdict; NFR matrix (0 FAIL) | direct |
| Documentation | `docs/recall-floor.md`; this journey; PROGRAM-MANIFEST | direct |
| **Total Workflow-orchestrated** | | **~106** |

## What shipped (real, in-tree, green)
- `src/pii_anon/routing/shared_layer.py` + `tests/test_shared_layer_projector.py` — the **recall-floor-by-construction** guarantee (FR-016/NFR-011/AX-003), the load-bearing T1 invariant. mypy --strict + ruff clean; no public-API change; 78 adjacent tests unaffected.
- `docs/recall-floor.md` — user-facing doc.
- A complete, traceable PDLC artifact set (Discovery→Testing) + 6 sign-offs.

## What's deferred (the roadmap — ~29 stories)
- **S1-02/03:** wire the projector into both fusion strategies + per-language recall ε-gate (makes the floor live in production).
- **Eval-integrity critical path (S3):** `RatingEnginePort` 3-tier ladder + Bayesian-BT (claim-grade) + coherent significance — **blocks on eval-data S6 `bradley_terry.py`** (verified absent).
- **S4:** distinct anon-vs-pseudo families + CanonicalRunGate + calibration reporter.
- **S5:** `attacks/` — real Tier-3 LLM-adversary + LiRA@128 MIA (**blocks on eval-data S6** `assemble_paired_set` + canary).
- **S6:** agentic interception + BYO-pipeline SDK + **token-store encryption** (security MUST).
- **S7:** multimodal readers + portability + **canonical benchmark run** + docs.
- **Security MUST:** gate-artifact signing (S2-05), attack-harness sandbox (S5-04), token encryption (S6-03).

## Defensibility (papers)
The pivot — pseudonymization-integrity scoring + Bayesian-BT ratings + real Tier-3 re-id, in the empty quadrant no public benchmark (RAT-Bench 2026, TAB, PIIBench, PrivaCI) occupies — is the publishable contribution. Feeds Paper 1 (PII-Rate-Elo) and Paper 2 (library/benchmark). The eval-integrity fixes (significance repair, canonical-run policy) are the precondition for claim-grade results.

## Program status
M0 ✅ · M1 ✅ · CODE Discovery/Requirements/Design ✅ (M2) · CODE Development partial (M4: S1-01) · DATA track (eval-data S5–S7) + Papers (M3/M5/M6) ongoing. Resume from S1-02 + the eval-integrity critical path, in lockstep with eval-data S6.
