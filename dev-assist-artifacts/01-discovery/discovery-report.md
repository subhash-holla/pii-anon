# Discovery Report — pii-anon (canonical)

> **Stage 1 Discovery, AGENT_SIMULATED, representative scale** (30 agents across 2 workflows: `wk5clrxov` foundational + `wdjmcpymh` UC/CVS/SME). Brownfield mode over a mature v1.4.0 library. Ready for Requirements. Pass-2 real-user research is a documented follow-up; all numeric claims about the current library are PROVISIONAL.

## 1. Refined POV (the headline pivot)
pii-anon is a **privacy measurement-and-engineering system whose defensible core is measurement, not detection.**
- **Pillar 1 — `pii-rate-elo` (lead product):** the first open, reproducible, multilingual framework that scores **pseudonymization integrity** and **LLM-era re-identification resistance** as *distinct, legally-grounded* families alongside calibrated detection — an empty quadrant no public benchmark (RAT-Bench 2026 / TAB / PIIBench / PrivaCI) occupies. **Bring-your-own-pipeline**: benchmarks *any* system (incl. Presidio, GLiNER2-PII, OpenAI Privacy Filter) with Bayesian Bradley-Terry ratings + bootstrap CIs + first-class calibration.
- **Pillar 2 — the swarm (re-scoped):** a **reversible-by-construction, recall-floor-guaranteed pseudonymizer with auditable key rotation** + an orchestration layer over best-of-breed detectors. Competes on **reversibility + recall-floor invariant + audit + orchestration**, NOT raw F1 (where OpenAI Privacy Filter ≈0.96 and Presidio already win; swarm real F1=0.610).
- **Honesty boundary (time-stamped):** as of 2026-05 no verified public benchmark scores pseudonymization integrity; "academic-grade defensibility" is a roadmap commitment gated on the canonical run + significance fix + running Tier-3 scorer.

## 2. Why now
PII protection is now a system-level control across an expanding LLM/agentic surface; the 2026 bar jumped (OpenAI Privacy Filter, GLiNER2-PII, RAT-Bench); law forces the anon-vs-pseudo distinction; and **measurement is the gap**. (See §1 motivation.)

## 3. Personas (7; 2 eval-dedicated, 5 dual-pillar)
P-01 Privacy engineer (H·both) · P-02 ML/platform engineer (H·both) · P-03 Compliance auditor (M·both) · **P-04 Academic researcher (H·eval)** · P-05 Agentic-security dev (H·both) · P-06 OSS maintainer/vendor (H·both) · **P-07 Third-party pipeline evaluator (H·eval)**. (See `personas.md`.)

## 4. Market (compete in TWO markets)
Detectors (Presidio/GLiNER2/OpenAI-Filter/Comprehend/Azure/Scrubadub) — pii-anon does NOT lead here. Eval frameworks (RAT-Bench #1 today, PIIBench, PII-Bench, AgentLeak, TAB, PrivLM/PrivaCI, AMBENCH, WebPII) — pii-anon is #2, gap = un-running Tier-3 + significance/canonical defects; **uncontested in the pseudonymization-integrity quadrant**. (See `03-market-research.md`.)

## 5. Use cases (28; 15 eval · 12 swarm · 1 both)
Full table in `04-use-cases.md`. Headline UCs: UC-05/07/08 (pseudonymization-integrity + anon-vs-pseudo distinct), UC-09 (real Tier-3 LLM-adversary), UC-01/02/03/06 (BYO-SDK + Bradley-Terry + coherent significance + canonical-gate), UC-13/14/15 (recall-floor by construction + per-language CI + MoE-router), UC-19/20/22 (query-aware masking + least-privilege interception + leakage-Sankey).

## 6. Concept value
3 high-willingness archetypes (privacy-eng, compliance, academic) anchor on the **eval + pseudonymization-integrity** value → validates the pivot. Medium archetypes converge on "**prove it**": fix smoke-run/significance, build the claimed Tier-3/agentic/recall-floor. (See `06-concept-value-study.md`.)

## 7. Top open items → Requirements
1. **Eval-integrity foundation is the critical path** (Bradley-Terry, significance repair, canonical-run policy) — gates both pillars' credibility.
2. **Pseudonymization-integrity + distinct anon-vs-pseudo families** = the headline novelty (UC-05/07/08).
3. **Recall-floor unification by construction** (AX-003) — the load-bearing swarm invariant before the MoE-router redesign.
4. **Build the claimed-but-absent surfaces:** running Tier-3 LLM-adversary scorer, agentic interception (AX-006), multimodal readers.
5. **Legal precision:** Art 4(5) as engineering proxy (not settled verdict); HIPAA Expert Determination hook; CCPA distinct contour.
6. **DATA-track coupling:** the stats primitives + scorers + `assemble_paired_set` live in the sibling `pii-anon-eval-data` (S5–S7) → cross-repo Requirements traceability.

## Handoff to Requirements
> Discovery COMPLETE (2026-05-30). 7 personas, 28 UCs (eval-led), refined POV (measurement-first; pseudonymization-integrity headline; swarm off the F1 arms race). 30 AGENT_SIMULATED agents; representative-scale cohorts (single-session limit, documented). **Top Requirements priorities:** (1) eval-integrity foundation [critical path]; (2) pseudonymization-integrity + distinct families; (3) recall-floor by construction; (4) running Tier-3 + agentic + multimodal. Cross-repo: stats/scorers in eval-data. Ready for Requirements. Run: `/dev-assist-requirements`.
