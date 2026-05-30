# Discovery §1 — Motivation & Background

> **Brownfield Mode.** Synthesized from the motivation agent (workflow `wk5clrxov`, AGENT_SIMULATED) + the two landscape briefs.

## Problem
PII protection has shifted from a one-shot preprocessing step into a **system-level control** that must hold across an expanding surface — model training/fine-tuning, prompt assembly, retrieval indexes, tool calls, agent memory, logs/traces, and multimodal artifacts. Two failures compound: (a) **detection** of indirect/contextual/multilingual/multimodal PII is hard and generic LLMs over-redact or miss; (b) **evaluation** of whether a pipeline is actually safe is fragmented, single-score, and statistically weak — so teams can't credibly choose, gate, or release.

## Why now (four forces, evidence-grounded)
1. **The privacy boundary moved** — GenAI/agentic deployment means one false negative leaks into prompts, tools, memory, traces, and completions, not just a dataset.
2. **The bar jumped in 2026** — OpenAI Privacy Filter (open-weight, F1≈0.96), GLiNER2-PII, and RAT-Bench reset expectations for both detection and evaluation.
3. **Law forces the anon-vs-pseudo distinction** — GDPR/EDPB (Art 4(5)), HIPAA (Safe Harbor vs Expert Determination), CCPA — yet tooling conflates them; pseudonymized data is still personal data.
4. **Measurement is the gap** — no public benchmark scores pseudonymization integrity or runs LLM-adversary re-identification on real systems; calibration/abstention and agentic-channel leakage are under-measured.

## Pillar 1 — Evaluation (`pii-rate-elo`) motivation [headline]
Enterprises and researchers need a **statistically-defensible, calibration-aware, LLM/agentic-era evaluation framework** that benchmarks *any* pipeline (including incumbents), distinguishes anonymization from pseudonymization, and produces uncertainty-aware, reproducible verdicts. This is the defensible core and the publishable contribution.

## Pillar 2 — Swarm (detection + anon/pseudo) motivation
Where reversibility, recall-floor guarantees, audit trails, multilingual breadth, and orchestration of best-of-breed engines matter, a configurable swarm + reversible pseudonymizer is valuable — **not** as an F1 leader, but as a recall-floor-guaranteed, auditable, reversible transform layer.

## Background anchors
NIST SP 800-122 / GenAI profile; GDPR Art 4(5) + EDPB 2025 pseudonymisation guidance; HIPAA Safe Harbor / Expert Determination; the two landscape briefs (`_inputs/`); prior swarm investigations (`03-design/_inputs/`).
