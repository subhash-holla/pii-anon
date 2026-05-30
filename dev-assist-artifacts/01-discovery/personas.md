# Discovery §2 — Personas & Workflows

> **Brownfield Mode.** 7 personas from a parallel persona-researcher fan-out (workflow `wk5clrxov`, AGENT_SIMULATED), priority-classified, extending the sibling eval-data personas (not re-derived). **Both pillars first-class:** 2 eval-dedicated + 5 dual-pillar. Full agent detail in the workflow transcript.

| # | Persona | Priority | Pillar | One-line |
|---|---|---|---|---|
| P-01 | **Enterprise Privacy / Data-Protection Engineer** | high | both | owns PII detect+anon controls in data pipelines (ingress, RAG, transcripts) |
| P-02 | **ML / Platform Engineer (LLMOps)** | high | both | embeds PII scrubbing as a guardrail in LLM/RAG/agent/streaming pipelines |
| P-03 | **Compliance Auditor / Privacy Assurance** | medium | both | certifies de-id/pseudonymization meets GDPR/HIPAA/CCPA; needs evidence |
| P-04 | **Academic De-id / Privacy Researcher** | high | **eval** | benchmarks + publishes de-id methodology; needs reproducible, defensible stats |
| P-05 | **Agentic-Platform / Security Developer** | high | both | runtime privacy control plane for live agents (prompt/memory/tool/trace) |
| P-06 | **OSS Maintainer / Privacy-Tool Vendor** | high | both | builds/extends detectors; evaluates + trusts the library + benchmark claims |
| P-07 | **Third-Party Pipeline Evaluator** | high | **eval** | "which detector do we ship / is it regressing?" — BYO-pipeline scorecard |

### P-01 Enterprise Privacy Engineer (high · both)
- **Goals:** low-latency masking that preserves business context; reversible pseudonyms for case continuity; provable residual-risk for release decisions.
- **Pains:** over-redaction breaks downstream analytics; uncertain recall → can't claim "anonymized"; key/vault management burden.
- **JTBD:** "scrub PII at ingress without breaking joins"; "prove this dataset is safe to share"; "reconnect a case to a person under control."

### P-02 ML / Platform Engineer (high · both)
- **Goals:** sub-ms pre-prompt scrubbing; CI-gateable privacy regression; stream + batch parity.
- **Pains:** latency budgets (the swarm fails speed floors); engines run on every chunk; no regression gate.
- **JTBD:** "strip PII before it reaches the model/index/logs"; "gate my pipeline in CI"; "process Kafka/Spark at SLA."

### P-03 Compliance Auditor (medium · both)
- **Goals:** legally-grounded evidence — residual re-id risk, calibration, audit logs; clear anon-vs-pseudo distinction.
- **Pains:** vendors conflate anonymization/pseudonymization; "de-id accuracy" hides residual risk; no calibrated confidence to set review thresholds.
- **JTBD:** "certify safe-to-release"; "map a transform to its legal regime (Safe Harbor / Art 4(5) / CCPA)"; "audit who can re-identify."

### P-04 Academic Researcher (high · **eval**) — Pillar-1 headline user
- **Goals:** reproducible, statistically-defensible benchmarks (CIs, power, significance) publishable at top venues; novel metrics (re-id resistance, calibration).
- **Pains:** fragmented domain benchmarks; vendor single-score claims; non-convergent/heuristic ratings; Tier-3 unevaluated.
- **JTBD:** "credibly compare systems with stated power"; "publish a defensible methodology"; "measure re-id resistance against an LLM adversary."

### P-05 Agentic-Platform / Security Developer (high · both) — Theme 3
- **Goals:** mask PII across prompt/memory/tool-I/O/trace channels; least-privilege interception; query-aware (keep relevant PII).
- **Pains:** traces/memory are an ungoverned PII store; output-only audits miss internal channels; prompt-injection exfiltration.
- **JTBD:** "wrap my agent so no raw PII leaks to traces/tools"; "mask only query-irrelevant PII"; "test leakage across all channels."

### P-06 OSS Maintainer / Privacy-Tool Vendor (high · both) — contribution flywheel
- **Goals:** trustworthy benchmark to prove their own tool; easy engine/transform/metric extension; honest claims.
- **Pains:** can't reproduce vendor numbers; taxonomy mismatch across tools; closed benchmarks.
- **JTBD:** "benchmark my detector credibly"; "plug my engine into the swarm"; "trust the leaderboard."

### P-07 Third-Party Pipeline Evaluator (high · **eval**) — v1.4.0 headline
- **Goals:** bring-your-own-pipeline scorecard with CIs; CI-gate on privacy/quality regression; compare candidates fairly.
- **Pains:** no neutral cross-pipeline benchmark; adapters are bespoke; significance unclear.
- **JTBD:** "score someone else's PII pipeline"; "gate ship/no-ship in CI"; "rank candidates with uncertainty."

## Workflow-map themes (per persona → theme)
P-01/P-02 → Theme 1 (swarm) + Theme 4 (portability); P-03 → Theme 2 (eval/compliance); **P-04/P-07 → Theme 2 (eval framework, headline)**; P-05 → Theme 3 (agentic); P-06 → cross-cutting (extension + trust).
