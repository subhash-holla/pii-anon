# Candidate Enhancement Catalog — both pillars co-equal

> A **candidate backlog** (hypotheses, NOT a locked spec) for the SOTA overhaul. Discovery validates demand (concept-value + Kano), Requirements prioritizes (MoSCoW + Pugh), Design shapes, Development builds, Testing verifies. Grounded in the two landscape briefs (`pii_enterprise_landscape_may26.md`, `pii_eval_may26.md`), the brownfield assessment (`../../00-brownfield-assessment/assessment-2026-05-30.md`), and the swarm prior-art (`../../03-design/_inputs/`). Each item tags the stage it primarily lands in and the finding/theme it serves.
>
> **Both pillars are first-class through ALL five stages.** Pillar 1 (evaluation / `pii-rate-elo`) is overhauled as a product in its own right — its own personas, use cases, FRs/NFRs, design diamonds, stories, and verification — AND it is the instrument that measures Pillar 2. Pillar 2 (detection + anonymization/pseudonymization / the swarm) is the MoE-router redesign.

---

## PILLAR 1 — Evaluation framework (`pii-rate-elo`)

### P1-A. Statistical rigor (fixes assessment MAJORs + raises the bar)
- **P1-A1 Bayesian Bradley-Terry / Davidson rating** replacing heuristic Glicko (`elo.py:198` RD is match-count-only → can't converge). MLE + paired-record **bootstrap credible intervals**, rank-distribution, convergence as a data property. *(Design+Dev; fixes the "incoherent significance" + "non-convergence" findings.)*
- **P1-A2 Significance pipeline repair** — paired bootstrap over per-record outcomes (retain `per_record_f1`, currently computed-then-discarded); CIs that bracket their point estimates; p-values consistent with effect sizes. *(Dev; fixes the computation bug.)*
- **P1-A3 Pre-registered power + stratified enrichment** — NIST sample-size tiers (≥753 positives/slice ≈ recall 0.98 ±1pp; ≥1,522 ≈ 0.99 ±0.5pp; ≥200 long-tail) per crossed cell; reuse eval-data's `stats/power.py`. *(Requirements NFR + Testing.)*
- **P1-A4 Canonical-run + provenance policy** — never publish smoke-run numbers; gate README/claim updates on `canonical_claim_run==True`; stamp every number with sample-cap + env. *(Requirements NFR + Dev; fixes the 50-sample-published finding.)*

### P1-B. Detection eval (Tier 1) — make calibration & relevance first-class
- **P1-B1 Calibration suite** — ECE, MCE, Brier, reliability diagrams, **selective-risk / coverage-risk (AURC)** curves; temperature-scaling before/after. Operationalizes AX-005 (calibrated abstention) — the top recommendation in both briefs. *(Design+Dev.)*
- **P1-B2 Abstention/deferral evaluation** — measure the privacy/utility lift of routing uncertain spans to human review (review-workload, time-to-adjudication, HITL lift). *(Dev+Testing.)*
- **P1-B3 Query-aware masking eval (PII-Bench)** — does the system keep query-*relevant* PII while masking the rest? Relevance P/R + over-redaction rate + answer-quality delta. New metric family; ties to Theme 3. *(Design+Dev.)*
- **P1-B4 Richer span matching** — partial-overlap / boundary-tolerant scoring beyond SemEval 4-mode; per-class + fairness (worst-group gaps by language/script/entity/difficulty). *(Dev.)*

### P1-C. Privacy / re-identification eval (Tier 2/3) — the headline novelty
- **P1-C1 Full-power membership inference** — LiRA at 128 shadow models (vs current 5) + Secret-Sharer canary/extraction; report **TPR @ low FPR** (Carlini), not just AUC. New `eval_framework/attacks/`. *(Design+Dev; fixes Tier-2 underpowered finding.)*
- **P1-C2 Real Tier-3 re-identification** — an **LLM adversary** (Staab 2024 / Lermen 2026) computing RRS/QIC/BSL on ≥2 real systems; wire eval-data's adversary harness + the (to-build) `assemble_paired_set` over the 2,500 paired personas. De-circularize via re-extracted signals. *(Design+Dev; closes the "Tier-3 unevaluated / all-zero" finding — the strongest single paper result.)*
- **P1-C3 Anonymization vs pseudonymization as DISTINCT eval families** (AX-004) — anonymization → privacy-utility **Pareto** + residual re-id risk; pseudonymization → unauthorized-reversal rate, authorized-reversal success, collision rate, referential integrity, key-compromise blast radius, **EDPB Art 4(5) separation test**. Never merged into one "de-id score." *(Design+Dev; reuse eval-data scorers.)*

### P1-D. LLM/agentic-era eval (new frontier — co-developed with Theme 3)
- **P1-D1 Contextual-integrity eval (PrivaCI-Bench)** — regulation-aware, norm-based privacy reasoning (maps to the compliance crosswalk: GDPR/HIPAA-SH/HIPAA-ED/CCPA/PCI as legally-distinct columns). *(Design+Dev.)*
- **P1-D2 Agentic leakage eval (AgentDojo / InjecAgent / AgentLeak)** — measure PII leakage across **prompt, retrieval, tool-I/O, memory, output, trace** channels (a leakage-Sankey); prompt-injection robustness; internal-channel leaks output-only audits miss. *(Design+Dev.)*

### P1-E. Eval-as-a-product (industry adoption)
- **P1-E1 "Bring-your-own-pipeline" SDK hardening** — make `evaluate_external_system` a first-class benchmarking SDK with adapters (spaCy / REST / cloud APIs / prompted-LLM); CI-gating recipe; uncertainty-aware leaderboard (rank distributions, Chatbot-Arena style). *(Design+Dev.)*
- **P1-E2 New baselines + external-benchmark plug-ins** — AWS Comprehend, Azure PII, prompted-LLM detectors; harmonized-taxonomy adapters for TAB / AMBENCH / PII-Bench (drop-in) + PrivLM/PrivaCI (via attacks/compliance). License-gated so paid/cloud stay out of the MIT headline claim. *(Dev+Testing.)*
- **P1-E3 Cross-dataset robustness eval** — train-on-one / test-on-another; report the cross-domain delta (recent work shows large degradation across institutions). *(Testing.)*

### P1-F. Left-field / bleeding-edge (high-novelty, validate in Discovery)
- **P1-F1 Item-Response-Theory leaderboard** — model per-record *difficulty* + per-system *ability* (psychometrics / modern LLM-eval). More informative than aggregate F1; surfaces discriminating records. *(Design; research-paper gold.)*
- **P1-F2 Adaptive/active benchmarking** — select the most *informative* records to evaluate (computerized-adaptive-testing style) → more statistical power per dollar. *(Design+Dev.)*
- **P1-F3 Meta-evaluation / construct validity** — does the composite rank *predict downstream task outcomes*? Validate the ruler itself (correlation of composite vs real task success). *(Testing; defensibility.)*
- **P1-F4 Decision-theoretic, profile-specific cost scoring** — generalize F2's "FN 4× FP" into a full per-deployment-profile cost matrix (healthcare vs streaming vs release). *(Design.)*

---

## PILLAR 2 — Detection + anonymization/pseudonymization (the swarm)

The MoE-router redesign (detailed in `../../03-design/_inputs/swarm-moe-prior-art.md` + `moe-architecture-and-guarantee.md` + the master plan): learned sparse router (Mistral/Switch), shared-expert isolation (DeepSeekMoE/Qwen2-MoE), auxiliary-loss-free SLA load-balancing (DeepSeek-V3), Mixture-of-Depths early-exit, speculative span verification, calibrated abstention. **Hard invariant:** unify the recall-floor (AX-003) by construction across MoE + swarm paths (the live gap). Plus the 6 transform strategies and reversible-pseudonymization vault.

## CROSS-CUTTING

- **Theme 3 — Agentic & contextual privacy:** query-aware masking gate (`policy/router.py`), trace/memory/tool-leakage interception (`orchestrator.py` wrapper, AX-006 least-privilege), session pseudonyms. Co-developed with P1-D.
- **Theme 4 — Multimodal & portability:** PDF/image/screenshot/DICOM/audio readers (extend `ingestion/`); non-EN context features; stream/batch/offline × cloud/local × OS matrix. (WebPII multimodal eval = a deliberate follow-up, not v1.)

---

## How both pillars thread the stages (co-equal)

| Stage | Pillar 1 (evaluation) | Pillar 2 (swarm) |
|---|---|---|
| Discovery | personas: pipeline-evaluator, academic researcher, compliance auditor, CI-gating engineer; JTBD "credibly compare / prove-safe-to-release / gate-CI"; market = eval **benchmarks** (PIIBench/PII-Bench/TAB/PrivLM/PrivaCI/AgentDojo/RAT-Bench) | personas: library-integrating engineer, streaming/latency engineer; market = **detectors** (Presidio/GLiNER/Comprehend/Azure) |
| Requirements | FRs for rating/calibration/attacks/Tier-3/external-SDK/canonical-run; NFRs for power, determinism, significance-correctness, ECE bounds | FRs for routing/early-exit/shared-experts; NFRs re-baselined to measured floors + recall-floor guarantee |
| Design | rating-engine redesign, attacks/ package, calibration+selective-risk, benchmark-adapter layer, anon/pseudo distinct APIs, **reporting/presentation redesign** | learned-router architecture diamond (Pugh over Mistral/Qwen/DeepSeek/Llama framings) |
| Development | TDD stories ↔ eval-data S5–S7 (stats + scorers + `assemble_paired_set`); cross-repo trace | TDD stories in worktrees; recall-floor property test + per-language CI gate |
| Testing | verify eval NFRs (power/calibration/significance) **and use the fixed eval to measure the redesigned swarm**; canonical run as release gate; meta-eval | NFR verification: latency floors pass w/ no recall regression |
