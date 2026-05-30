# Discovery §3 — Market Research (JTBD + Kano + Pugh)

> **Brownfield Mode.** Triangulated from jtbd-analyst + kano-analyst + pugh-comparator (workflow `wk5clrxov`, AGENT_SIMULATED, competitors/benchmarks live-verified 2026-05-30). Two markets — **detectors** AND **eval frameworks** — because pii-anon competes in both.

## JTBD (functional / emotional / social)
- **Detection jobs:** "strip PII before model/index/logs" (functional); "don't be the engineer who leaked PHI" (emotional); "pass the compliance audit" (social).
- **Evaluation jobs:** "credibly compare pipelines with stated power"; "prove safe-to-release"; "gate CI on regression"; "publish a defensible methodology."
- **Forces:** strong *push* (LLM/agentic privacy incidents, GDPR/HIPAA exposure) + *pull* (OpenAI Privacy Filter, RAT-Bench raise the bar); *anxiety* = trusting an uncertified benchmark; *habit* = Presidio incumbency.

## Competitive landscape (live-verified)
**Detectors / anonymizers:** Microsoft Presidio (OSS, incumbent, reversible anon + evaluator), **GLiNER2-PII / GLiNER-Guard** (arXiv 2605.05277, May 2026), **OpenAI Privacy Filter** (2026-04-22, open-weight, F1≈0.96, MoE-style), AWS Comprehend / Azure AI Language PII (managed), Scrubadub, Google Sensitive Data Protection (tokenization/FPE/KMS).
**Eval benchmarks/frameworks:** **RAT-Bench** (Imperial, arXiv 2602.12806, Feb 2026 — strongest competitor; detection+anon-by-re-id-risk, multilingual), **PIIBench** (arXiv 2604.15776, Apr 2026), **PII-Bench** (arXiv 2502.18545, query-aware masking), **AgentLeak** (arXiv 2602.11510, Feb 2026 — internal-channel leakage), TAB, SPY, PrivLM-Bench, PrivaCI-Bench, AMBENCH, WebPII, AgentDojo/InjecAgent.

## Pugh (subject = pii-anon; honest two-sided)
- **Detector matrix:** pii-anon-swarm loses on raw F1 (OpenAI Privacy Filter ≈0.96, GLiNER2, Presidio all ahead of the swarm's real 0.610) and on latency floors. **Wins on:** reversible pseudonymization + key rotation, recall-floor guarantee (once unified), orchestration of multiple engines, 60-language breadth.
- **Eval matrix:** pii-rate-elo is **#2 behind RAT-Bench today** (the gap is the un-running Tier-3 scorer + the significance/canonical-run defects). **Uncontested where it leads:** the composite (F1+latency+throughput+privacy+fairness+Tier-3), the floor-gate governance, and — the **empty quadrant** — pseudonymization-integrity scoring + distinct anon-vs-pseudo families + EDPB Art 4(5).

## Kano (candidate enhancements → satisfaction shape, across personas)
- **Must-have (table stakes):** P1-A1 Bradley-Terry, P1-A2 significance repair, P1-A4 canonical-run policy, P2 recall-floor unification, P1-A3 power. *(Their absence is the current dissatisfier.)*
- **Performance (more is better):** P1-B1 calibration, P1-C1 full-power MIA, P1-B4 richer span matching, P1-E2 new baselines, P1-E3 cross-dataset robustness.
- **Delighters:** **P1-C3 pseudonymization-integrity / anon-vs-pseudo split (the headline novelty)**, P1-C2 real Tier-3 LLM-adversary re-id, P1-D2 agentic leakage-Sankey, P1-B3 query-aware masking, P1-E1 BYO-pipeline SDK + uncertainty leaderboard, P1-F1 IRT leaderboard.
- **Indifferent / reverse (watch):** raw-F1 chasing on the swarm (reverse — don't compete there); broad "control-plane" breadth claims (indifferent until built).

## pii-anon market position (honest)
Maturity is in **code/CI**, not in published evidence. The win is **not** a better detector (OpenAI/Presidio/GLiNER2 contest that) but the **measurement layer + the pseudonymization-integrity quadrant + orchestration of incumbents**. Opportunities: (1) own pseudonymization-integrity eval; (2) become the neutral BYO-pipeline benchmark that scores *even the incumbents*; (3) turn OpenAI Privacy Filter / Presidio / GLiNER2 into first-class baselines (credibility); (4) the contribution flywheel via the OSS-maintainer persona.
