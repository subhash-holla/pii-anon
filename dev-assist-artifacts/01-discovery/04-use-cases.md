# Discovery §4 — Use Cases

> **Brownfield Mode.** 28 UCs from 5 use-case-author agents + a 3-SME review panel (workflow `wdjmcpymh`, AGENT_SIMULATED). Each carries situation/intent/outcome in the workflow transcript; below is the canonical UC↔persona↔pillar↔acceptance table that feeds the Requirements R0 bridge. **Pillar balance: 15 evaluation UCs · 12 swarm · 1 both** (evaluation is the headline, per the refined POV).

| UC | Title | Persona | Pillar | Pri | Acceptance signal (boolean-testable) |
|---|---|---|---|---|---|
| **Pillar-1 Evaluation core** ||||||
| UC-01 | Score a BYO PII pipeline against a canonical dataset | P-07 | eval | H | `evaluate_external_system` returns composite + per-metric + bootstrap CI for any registered adapter (REST/spaCy/LLM), zero harness-core edits |
| UC-02 | Rank systems via Bayesian Bradley-Terry + credible intervals | P-04 | eval | H | ≥3 systems → BT strengths + 95% CI + rank-distribution + `separable_at_95` per pair |
| UC-03 | Coherent significance (CIs bracket estimates; p ↔ effect size) | P-04 | eval | H | every delta's point estimate ∈ its bootstrap CI AND sign(effect)↔significance verdict; automated check passes |
| UC-04 | Calibration & selective-risk first-class | P-04 | eval | M | Tier-1 run emits ECE/MCE/Brier/AURC + reliability diagram; post-temp-scaling ECE ≤ pre |
| UC-05 | Pseudonymization-integrity scored as a distinct family | P-04 | eval | H | integrity family reported separately from anonymization Pareto; **no merged "de-id score" field** |
| UC-06 | CI-gate ship/no-ship on composite + recall floor + canonical stamp | P-07 | eval | H | gate=PASS iff composite≥thr ∧ recall≥floor ∧ `canonical_claim_run==True`; refuses claim-grade verdict otherwise; provenance stamp on every number |
| **Pillar-1 Privacy eval (HEADLINE quadrant)** ||||||
| UC-07 | Score pseudonymization integrity (Art 4(5), reversal, collision, referential) | P-03 | eval | H | emits unauthorized-/authorized-reversal rate, collision rate, referential-integrity, key-separation pass/fail |
| UC-08 | Enforce anon vs pseudo as structurally distinct families | P-03 | eval | H | no code path merges the two families; output carries its end-state label |
| UC-09 | Real Tier-3 LLM-adversary re-id (RRS/QIC/BSL), de-circularized | P-04 | eval | H | RRS/QIC/BSL computed on ≥2 real systems via a running LLM adversary; observed signals re-extracted (not gold) |
| UC-10 | Full-power MIA (LiRA@128 + Secret-Sharer), TPR@low-FPR | P-04 | eval | H | AUC + TPR@FPR∈{1e-3,1e-2}; de-id'd AUC≈0.5 vs raw≫0.5 control |
| UC-11 | Audit who can re-identify — key-compromise blast radius + rotation | P-03 | eval | M | report blast-radius under key compromise + key-rotation resilience |
| UC-12 | Anonymization residual-risk vs utility Pareto under LLM adversary | P-04 | eval | M | publishable Pareto frontier; privacy + utility never merged |
| **Pillar-2 Swarm + transforms** ||||||
| UC-13 | Recall-floor guarantee by construction (MoE + swarm paths) | P-01 | swarm | H | property test: `entities(ensemble) ⊇ entities(shared)` across inputs/shuffles |
| UC-14 | Per-language recall-floor CI gate | P-02 | swarm | H | CI fails if router-on recall < router-off recall − ε for any language |
| UC-15 | MoE-router: span-dedup + selective expert activation at SLA | P-02 | swarm | H | latency-floor profiles pass with no recall regression vs baseline |
| UC-16 | Reversible pseudonymization with auditable key rotation (Art 4(5)) | P-01 | swarm | H | same (value,key,scope)→same token; rotation + reversal audited; key/state separable |
| UC-17 | Six transforms with legal-regime mapping, preserving joins | P-01 | swarm | M | each strategy maps to a legal regime; deterministic surrogates preserve joins |
| UC-18 | Orchestrate incumbent detectors behind one recall-floored interface | P-06 | swarm | M | Presidio/GLiNER2/OpenAI-Filter pluggable; ensemble ⊇ each |
| **Theme-3 Agentic & contextual privacy** ||||||
| UC-19 | Query-aware masking gate (keep only relevant PII) | P-05/P-02 | swarm | H | relevance P/R + over-redaction rate + answer-quality delta on RAG prompts |
| UC-20 | Least-privilege multi-channel interception (prompt/memory/tool/trace) | P-05/P-02 | swarm | H | no raw PII persisted post-masking to any channel (AX-006) |
| UC-21 | Stable session pseudonyms for multi-turn continuity | P-05 | swarm | M | same entity → stable surrogate across a session; reversal authorized-only |
| UC-22 | Agentic leakage measurement across channels (leakage-Sankey) | P-05/P-02 | eval | H | per-channel leak counts (prompt/retrieval/tool/memory/output/trace) |
| UC-23 | Prompt-injection exfiltration resistance | P-05 | eval | H | attack-success-rate vs benign-task-success on AgentDojo/InjecAgent-style suites |
| **Theme-4 Multimodal & portability** ||||||
| UC-24 | Scrub PII from native formats (PDF/image/screenshot/DICOM/audio) | P-01 | swarm | H | each format → `Iterator[IngestRecord]`; round-trip + extraction correct |
| UC-25 | Benchmark detector recall per-modality + gate readers (pii-rate-elo) | P-01 | eval | H | per-modality recall scored; multimodal readers CI-gated |
| UC-26 | Identical scrub decisions across stream/batch/offline (parity) | P-02 | swarm | H | same input → same spans/transforms across all 3 modes |
| UC-27 | Portable local/cloud + OS-matrix, no behavioral drift | P-02 | swarm | M | identical results on macOS/Linux/Windows × local/cloud |
| UC-28 | Multilingual non-EN context features + per-language fairness eval | P-01 | both | M | non-EN context feature active; worst-group fairness gap bounded |

**SME-panel adjustments folded in:** (1) treat EDPB Art 4(5) / Guidelines 01/2025 as *current-but-evolving* — frame the separation test as a defensible engineering proxy, not a settled legal verdict (legal SME); (2) lean on HIPAA **Expert Determination** as the legitimizing hook (under-leveraged); (3) keep the recall-floor invariant *realistic* — it guarantees ⊇ the shared layer, not ⊇ every possible detector (academic SME); (4) position vs Presidio as orchestrate-and-benchmark, not replace (incumbent SME).
