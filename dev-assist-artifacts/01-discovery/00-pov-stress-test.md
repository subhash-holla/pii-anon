# Discovery §0 — POV Stress Test

> **Brownfield Mode — Source Signal vs Gaps.** Synthesized from a 3-critic stress-test fan-out (Adjacent-Product Skeptic, Persona Realist, OSS/Monetization Strategist) — workflow `wk5clrxov`, AGENT_SIMULATED, live-verified 2026-05-30. All three returned **holds-with-changes**: the POV survives but its *headline* must pivot.

## Original POV (entering Discovery)
"An open-source, dual-pillar privacy **control plane**: a SOTA MoE-swarm detector/anonymizer (Pillar 2) + pii-rate-elo, a defensible evaluation framework (Pillar 1) — usable local↔cloud, stream↔batch↔offline, multimodal."

## What the critics broke (verified, live)
- **The "control plane / dual-pillar OSS detector + evaluator" category already exists** — Microsoft **Presidio** (+ presidio-evaluator, MIT) does detection + reversible anonymize/de-anonymize + ships an evaluator that even recommends F2. The swarm *wraps* Presidio/GLiNER at Layer 2 — it competes with a dependency. (Adjacent-Product Skeptic)
- **"SOTA detector" is contested on the metric that matters.** **OpenAI Privacy Filter** (2026-04-22) is open-weight, locally-runnable, **1.5B/50M-active (sparse/MoE-style)**, **F1≈0.96** — it pre-empts the swarm's MoE-router *novelty* and beats the swarm's real F1 (0.610; published 0.76 is an uncertified 50-sample smoke run). (Skeptic — CATASTROPHIC for the detector claim)
- **"Academic-grade defensibility" is presently falsified by the repo's own assessment** (smoke-run numbers, incoherent significance, Tier-3 scorer not yet running). The peer-preprinted **RAT-Bench** (Imperial, arXiv 2602.12806, Feb 2026) currently has more academic standing and out-ranks pii-anon in the sibling eval-data Pugh. (Skeptic + Persona Realist)

## The one quadrant that survives every probe (all 3 critics converge)
**Scoring PSEUDONYMIZATION INTEGRITY** — authorized-reversal success, collision rate, referential integrity, **EDPB Art 4(5) key/state separation** — unified with **LLM re-identification resistance** and **multilingual (60-language) breadth**, in **one reproducible CC0 artifact**. Verified that RAT-Bench, TAB, PIIBench, PrivaCI-Bench, and the agentic suites cover detection + anonymization-by-re-id-risk but **none score pseudonymization integrity**. It maps 1:1 to the GDPR anonymization-vs-pseudonymization legal distinction → technically uncontested **and** legally motivated. The OSS/Monetization critic adds: anchor a **contribution flywheel** on the privacy-tool maintainer persona.

## ✅ Refined POV (carried into Discovery §1–§6 and all stages)
> **pii-anon is a privacy *measurement-and-engineering* system whose defensible core is measurement, not detection.**
>
> **Headline (Pillar 1 — `pii-rate-elo`, the lead product):** the first open, reproducible, multilingual benchmark + framework that scores **pseudonymization integrity** and **LLM-era re-identification resistance** *as distinct, legally-grounded families* alongside calibrated detection — a quadrant no public benchmark (RAT-Bench 2026 / TAB / PIIBench / PrivaCI / agentic suites) occupies. It is **bring-your-own-pipeline**: it benchmarks *any* system, explicitly including best-of-breed incumbents (Presidio, GLiNER2-PII, **OpenAI Privacy Filter**), with Bayesian Bradley-Terry ratings + bootstrap CIs and first-class calibration.
>
> **Pillar 2 (the swarm), re-scoped:** from "SOTA detector" → a **reversible-by-construction, recall-floor-guaranteed pseudonymizer with auditable key rotation**, plus an **orchestration layer** over best-of-breed open detectors. It competes on **reversibility + the ensemble-superset (recall-floor) invariant + audit trail**, NOT on raw detection F1.
>
> **Honesty boundary (time-stamped, falsifiable):** *as of 2026-05, no verified public benchmark scores pseudonymization integrity.* "Academic-grade defensibility" is a **roadmap commitment** (gated on the canonical run + significance fix + running Tier-3 scorer), not a present-tense claim.

## Carry-forward consequences
- **Pillar 1 is the headline; Pillar 2 re-scopes off the F1 arms race** → directly shapes Requirements priorities + Design diamonds.
- **OpenAI Privacy Filter, Presidio, GLiNER2-PII become first-class *baselines* the eval framework benchmarks** (competitor → credibility hook).
- Closes the smoke-run/significance/Tier-3 gaps as the price of the defensibility claim (already the steering decision).
