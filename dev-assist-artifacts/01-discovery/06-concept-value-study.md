# Discovery §5 — Concept Value Study

> **Brownfield Mode.** Representative cohort (8 archetypes) + 3 SME reviewers (workflow `wdjmcpymh`, AGENT_SIMULATED — NOT a substitute for real-user research; Pass-2 is a documented follow-up).

## Willingness (to adopt the refined concept)
| Archetype | Willingness | Signal |
|---|---|---|
| P-01 Privacy engineer | **high** | reversible pseudonymization + residual-risk evidence solves a real release-gate pain |
| P-03 Compliance auditor | **high** | anon-vs-pseudo distinction + Art 4(5) + calibration evidence is exactly what audits need |
| P-04 Academic researcher | **high** | Bradley-Terry + real Tier-3 + pseudonymization-integrity = publishable; the headline lands |
| P-02 ML/platform engineer | medium | wants proof: canonical run + latency floors before trusting (latency-paranoid, Presidio-incumbent) |
| P-05 Agentic-security dev | medium | values multi-channel leakage eval; wants the interception surface actually built (currently zero) |
| P-06 OSS maintainer/vendor | medium | will adopt the benchmark if claims are reproducible + their engine plugs in |
| P-07 Pipeline evaluator | medium | wants the BYO-SDK + uncertainty leaderboard; skeptical until canonical-run policy is real |
| Skeptical buyer (Presidio + eyeing OpenAI Filter) | medium | "why not just Presidio + OpenAI Filter?" → answer must be the eval + pseudonymization-integrity layer, not a better detector |

**Read:** the 3 *high* willingness archetypes all anchor on the **evaluation + pseudonymization-integrity** value (validating the POV pivot). The *medium* ones converge on one message: **prove it** — fix the smoke-run/significance defects and build what's claimed.

## Most-valued (across cohort)
Pseudonymization-integrity scoring (P1-C3) · real Tier-3 LLM-adversary re-id (P1-C2) · Bayesian Bradley-Terry + coherent significance (P1-A1/A2) · canonical-run policy (P1-A4) · calibration/abstention (P1-B1) · query-aware masking (P1-B3) · BYO-pipeline SDK (P1-E1) · recall-floor guarantee (P2).

## Must-fix (gating adoption)
1. **Replace smoke-run published numbers with a certified canonical run** + provenance stamps (every medium-willingness archetype named this).
2. **Fix the significance computation** (coherent CIs/p-values) — academic + evaluator both reject the methodology otherwise.
3. **Actually build** the Tier-3 running scorer, the agentic interception surface, and the recall-floor unification (claims currently outrun the code).
4. Make the latency story honest (the swarm fails speed floors) — re-scope or fix.

## New needs surfaced
- HIPAA **Expert Determination** workflow support (compliance) · per-jurisdiction legal crosswalk depth (GDPR/HIPAA/CCPA distinct contours) · cost/latency budget transparency · contribution path for external engines (OSS flywheel).

## SME findings (severity-tagged, folded into UCs + carried to Requirements)
- **MAJOR (legal):** don't over-claim EDPB Art 4(5)/Guidelines 01/2025 as settled — frame the separation test as a defensible engineering proxy; lean on HIPAA Expert Determination; reflect CCPA/CPRA's distinct contour in the compliance crosswalk.
- **MAJOR (academic):** Tier-3 LLM-adversary must address circularity/contamination/non-stationarity; Bradley-Terry needs identifiability (anchor/regularization); "convergence as a data property" must be stated precisely; calibration multiplicity + ECE bias controlled; power vs absolute-rate success criteria reconciled.
- **MAJOR (incumbent):** position as orchestrate-and-benchmark over Presidio/OpenAI-Filter, not replace; keep recall-floor invariant realistic (⊇ shared layer, not ⊇ all detectors); BYO-pipeline credibility requires scoring incumbents *identically*.
- **OBSERVATION:** anonymisation "out of scope" phrasing should invoke the motivated-intruder standard; cross-jurisdiction breadth vs depth trade-off.
