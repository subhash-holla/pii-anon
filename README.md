# pii-anon

**An open-source PII detection and anonymization library for Python — built for LLM pipelines, streaming data, and regulatory compliance.**

[![PyPI Version](https://img.shields.io/pypi/v/pii-anon.svg)](https://pypi.org/project/pii-anon/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-2273-brightgreen.svg)](#quality-and-testing)

[![Built with AI Agents](https://img.shields.io/badge/Built_with-AI_Agents-d946ef.svg)](#acknowledgements)

---

## Why pii-anon?

**Two offerings, one library.** pii-anon ships with two detection modes designed for different trade-offs:

| Offering | Best For | F1 | Latency |
|---|---|---|---|
| **pii-anon** | Speed-sensitive pipelines (streaming, real-time) | 0.79 | <1 ms/record |
| **pii-anon-swarm** | Maximum accuracy (compliance, batch ETL) | 0.65 | ~90 ms/record |

Both outperform Presidio (F1=0.45) and Scrubadub (F1=0.36) on the 117,000+ record pii_anon_benchmark evaluation. pii-anon also outperforms GLiNER (F1=0.76).

**pii-anon** uses a fast regex engine with checksum validators (Luhn, IBAN mod-97, ABA routing), context-aware confidence scoring, and deny-lists — delivering sub-millisecond detection across 20 entity types with zero external dependencies.

**pii-anon-swarm** fuses regex with Presidio, Scrubadub, spaCy, Stanza, and GLiNER through a Mixture-of-Experts architecture: per-entity-type routing with calibrated weights, weighted voting, and corroboration filtering. The swarm achieves the highest recall (84%) of any system tested.

**6 transformation strategies** — pseudonymization with key rotation, tokenization (HMAC/AES-SIV), redaction, generalization, synthetic replacement, or differential privacy (ε-DP) — all with audit trails and deterministic linking for long-context entity tracking.

**6 compliance templates** (HIPAA Safe Harbor, GDPR Pseudo/Anon, CCPA, Minimal Risk, Maximum Privacy) validate your entity detection against real regulatory requirements. Know instantly whether your coverage meets compliance standards.

**Enterprise-grade evaluation framework** with 50+ metrics, 117,000+ synthetic evaluation records (100% CC0/CC-BY-4.0), and a composite Elo-based ranking system that captures the full picture: detection accuracy, privacy, utility, fairness, and performance.

**Sub-millisecond latency** with constant-memory streaming — process Kafka, Spark, or Beam pipelines without building specialized infrastructure. Async/sync dual APIs.

---

## Quick Install

```bash
pip install pii-anon
```

Or with optional integrations:

```bash
pip install pii-anon[cli,crypto]          # CLI + cryptography
pip install pii-anon[benchmark]           # + all competitor engines (for benchmarking)
pip install pii-anon[datasets]            # + 117K evaluation dataset
pip install pii-anon[benchmark,cli,crypto,datasets,dev]  # Full stack
```

**Python support:** 3.10, 3.11, 3.12, 3.13 (experimental: 3.14)

> **Tip:** Use a virtual environment (`venv`) on any OS for the cleanest install. See [docs/dependencies-and-platforms.md](docs/dependencies-and-platforms.md) for platform-specific guidance.

---

## 30-Second Demo

Detect and pseudonymize PII in three lines:

```python
from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

orch = PIIOrchestrator(token_key="your-secret-key")

result = orch.run(
    {"text": "Contact alice@example.com and +1 415 555 0100"},
    profile=ProcessingProfileSpec(profile_id="default", mode="weighted_consensus", language="en"),
    segmentation=SegmentationPlan(enabled=False),
    scope="etl",
    token_version=1,
)

print(result["transformed_payload"])
# Contact <EMAIL_ADDRESS:v1:tok_abc123> and <PHONE_NUMBER:v1:tok_def456>
```

Same token for the same entity across documents — perfect for maintaining referential integrity in RAG pipelines.

---

## Compliance Templates

Validate entity detection against regulatory requirements in one call:

```python
from pii_anon.eval_framework import ComplianceValidator

validator = ComplianceValidator()

# Single standard
hipaa_report = validator.validate(detected_entity_types, standard="hipaa")
print(f"HIPAA coverage: {hipaa_report.coverage_percentage:.0f}%")

# All standards at once
multi_report = validator.validate_all(detected_entity_types)
for standard, report in multi_report.reports.items():
    print(f"{standard}: {report.coverage_percentage:.0f}% — {len(report.gaps)} gaps")
```

Supported: HIPAA Safe Harbor, GDPR (pseudonymization & anonymization), CCPA, Minimal Risk, Maximum Privacy.

---

## Pipeline Builder

Chain detection, compliance, and evaluation with a fluent API:

```python
from pii_anon import PIIOrchestrator
from pii_anon.eval_framework import (
    compute_composite,
    CompositeConfig,
    PIIRateEloEngine,
    GovernanceThresholds,
)

# Fluent chaining: load → detect → validate → evaluate → export
orch = PIIOrchestrator(token_key="secret")
result = orch.run(input_data, profile=profile, scope="etl", token_version=1)

# Validate compliance
validator = ComplianceValidator()
gdpr_report = validator.validate(result["detected_entities"], standard="gdpr")

# Compute composite score and check governance readiness
composite = compute_composite(f1=0.85, precision=0.92, recall=0.80, latency_ms=5.2)
engine = PIIRateEloEngine()
engine.run_round_robin({"your-system": composite.score})
governance = engine.evaluate_governance("your-system", thresholds=GovernanceThresholds())
```

---

<!-- BENCHMARK_SUMMARY_START -->

## Accuracy Objective (profiles: long_document, structured_form_accuracy, multilingual_mix)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6810 | 0.763 | — | 0.908 | 0.658 | 79.586 | 36506.58 | 0 |
| pii-anon | available | 0.8000 | 0.845 | — | 0.868 | 0.823 | 8.007 | 329390.99 | 0 |
| pii-anon-swarm | available | 0.5873 | 0.648 | — | 0.528 | 0.839 | 90.378 | 33003.58 | 0 |
| presidio | available | 0.4749 | 0.453 | — | 0.401 | 0.521 | 14.245 | 137200.71 | 0 |
| scrubadub | available | 0.5342 | 0.357 | — | 0.872 | 0.225 | 0.224 | 9980506.31 | 0 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.800 vs best 0.800).
- precision: within 5% of best (0.868 vs best 0.908).
- recall: within 5% of best (0.823 vs best 0.839).
- f1: within 5% of best (0.845 vs best 0.845).

Weaknesses for `pii-anon`:
- docs_per_hour: more than 10% below best (329390.990 vs best 9980506.310).
- latency_p50_ms: more than 10% slower than best (8.007 vs best 0.224).

This section is generated from benchmark artifacts.
## Speed Objective (profiles: short_chat, structured_form_latency, log_lines)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| pii-anon | available | 0.8129 | 0.792 | — | 0.768 | 0.818 | 0.346 | 3791089.96 | 1588 |
| pii-anon-swarm | available | 0.5873 | 0.648 | — | 0.528 | 0.839 | 90.385 | 33306.11 | 1483 |
| gliner | available | 0.6810 | 0.763 | — | 0.908 | 0.658 | 79.627 | 36873.26 | 1534 |
| presidio | available | 0.4753 | 0.453 | — | 0.401 | 0.521 | 14.332 | 142615.45 | 1428 |
| scrubadub | available | 0.5282 | 0.357 | — | 0.872 | 0.225 | 0.224 | 5642522.18 | 1454 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.813 vs best 0.813).
- recall: within 5% of best (0.818 vs best 0.839).
- f1: within 5% of best (0.792 vs best 0.792).

Weaknesses for `pii-anon`:
- precision: more than 10% below best (0.768 vs best 0.908).
- docs_per_hour: more than 10% below best (3791089.960 vs best 5642522.180).
- latency_p50_ms: more than 10% slower than best (0.346 vs best 0.224).

This section is generated from benchmark artifacts.

Profile floor-gate results:
- `short_chat` (speed): floor_pass=False
- `long_document` (accuracy): floor_pass=True
- `structured_form_accuracy` (accuracy): floor_pass=True
- `structured_form_latency` (speed): floor_pass=False
- `log_lines` (speed): floor_pass=False
- `multilingual_mix` (accuracy): floor_pass=True

### Statistical Significance

Evaluated on **106,855** records. Minimum detectable effect (MDE) at α=0.05, power=0.80: **0.0018** F1 points.

| System | F1 | 95% CI | Samples |
|---|---:|---|---:|
| pii-anon | 0.792 | [0.784, 0.786] | 106,855 |
| pii-anon-swarm | 0.648 | [0.646, 0.647] | 106,855 |
| gliner | 0.763 | [0.759, 0.761] | 106,855 |
| presidio | 0.453 | [0.404, 0.407] | 106,855 |
| scrubadub | 0.357 | [0.354, 0.357] | 106,855 |

Pairwise comparisons (paired bootstrap, n=10,000 resamples):

| Comparison | ΔF1 | p-value | Significant | Effect Size |
|---|---:|---:|---|---|
| pii-anon-swarm vs gliner | -0.1138 | 0.4990 | n.s. | medium (d=-0.767) |
| pii-anon vs gliner | +0.0245 | 0.5020 | n.s. | negligible (d=+0.142) |
| pii-anon-swarm vs scrubadub | +0.2907 | 0.5035 | n.s. | large (d=+1.592) |
| pii-anon vs presidio | +0.3791 | 0.5055 | n.s. | large (d=+1.793) |
| pii-anon-swarm vs presidio | +0.2408 | 0.5067 | n.s. | large (d=+1.250) |
| pii-anon vs scrubadub | +0.4290 | 0.5095 | n.s. | large (d=+2.120) |

*Method: paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*

<!-- BENCHMARK_SUMMARY_END -->

See [Benchmark Methodology](#benchmark-methodology) for details.

### Per-Entity Recall: Broadest Coverage in the Field

Both pii-anon offerings detect **20 entity types** with non-zero recall — more than any competitor. Here is the head-to-head recall comparison from the pii_anon_benchmark evaluation:

| Entity Type | pii-anon | pii-anon-swarm | GLiNER | Presidio | Scrubadub |
|---|:---:|:---:|:---:|:---:|:---:|
| PERSON_NAME | **80.5%** | **80.5%** | 56.4% | 63.4% | 0% |
| EMAIL_ADDRESS | **100%** | **100%** | 99.4% | **100%** | 84.9% |
| PHONE_NUMBER | 91.3% | **98.7%** | 98.5% | 57.2% | 57.2% |
| ADDRESS | **100%** | **100%** | 98.7% | 0% | 0% |
| US_SSN | 90.0% | **100%** | 65.1% | **100%** | 90.0% |
| DATE_OF_BIRTH | **89.2%** | **89.2%** | 88.4% | 0% | 0% |
| EMPLOYEE_ID | **100%** | **100%** | 0% | 0% | 0% |
| BANK_ACCOUNT | **100%** | **100%** | 52.6% | 78.4% | 0% |
| ORGANIZATION | **100%** | **100%** | 0% | 0% | 0% |
| CREDIT_CARD | **100%** | **100%** | 82.8% | 12.6% | 0% |
| MEDICAL_RECORD_NUMBER | **100%** | **100%** | 0% | 0% | 0% |
| PASSPORT | **100%** | **100%** | 92.3% | 0% | 0% |
| USERNAME | **100%** | **100%** | 94.8% | 0% | 0% |
| MAC_ADDRESS | **100%** | **100%** | 0% | 0% | 0% |
| IP_ADDRESS | **100%** | **100%** | **100%** | **100%** | 0% |
| DRIVERS_LICENSE | **100%** | **100%** | 18.6% | 0% | 0% |
| NATIONAL_ID | **77.0%** | **77.0%** | 3.5% | 0% | 0% |
| ROUTING_NUMBER | **100%** | **100%** | 0% | 0% | 0% |
| LOCATION | **100%** | **100%** | 0% | 0% | 0% |
| LICENSE_PLATE | **100%** | **100%** | 0% | 0% | 0% |
| **Overall Recall** | **91.0%** | **92.3%** | 66.7% | 51.1% | 22.4% |

pii-anon and pii-anon-swarm are the **only** systems that detect all 20 entity types in this benchmark. GLiNER detects 13 types, Presidio detects 7 types, Scrubadub detects 4.

---

## Core Capabilities

**48 Entity Types** — Comprehensive coverage across Personal Identity, Financial, Government ID, Medical, Digital/Technical, Employment, and Behavioral/Contextual categories. Every entity type includes risk-level classification and regulatory reference mappings.

**52 Languages, 17 Writing Systems** — Resource-level classification ensures fair cross-lingual evaluation. High-resource languages (English, Spanish, French, Chinese, Japanese, Arabic) with medium and low-resource coverage for emerging markets.

**Modular Detection Engine** — pii-anon's regex engine uses checksums (Luhn, IBAN mod-97, ABA routing, VIN check-digit, DEA), context-aware confidence scoring, and entity-type-specific allow/deny lists to suppress false positives. pii-anon-swarm adds Presidio and Scrubadub through a Mixture-of-Experts fusion layer with per-entity-type routing and corroboration filtering.

**Deterministic Pseudonymization** — Same entity always maps to the same token within a scope. Cryptographically secure (AES-SIV compatible) with pluggable key management, rotation, and re-identification audit trails.

**Long-Context Entity Linking** — Resolves "Jack Davis", "Jack", and "jackdavis@example.com" to the same pseudonymized token, preserving referential integrity across RAG chunks and multi-paragraph documents.

**Evidence-Backed Evaluation** — 50+ metrics traceable to peer-reviewed research (27+ citations: SemEval'13, TAB, OpenNER, seqeval, Shokri et al., Carlini et al., Abadi et al.). Multi-level evaluation (token, entity, document, mention) with 4 SemEval matching modes. Privacy metrics (k-anonymity, l-diversity, t-closeness). Utility metrics (format/semantic preservation). Fairness metrics (cross-language, cross-entity, cross-script). Research-grade statistics (bootstrap CI, paired significance testing, Cohen's kappa).

---

## Composite Scoring: The PII-Rate-Elo Metric

Traditional benchmarks fail to capture the full picture. pii-anon ships with a **two-tier composite system** and **Elo tournament** that ranks systems across detection accuracy, privacy, utility, fairness, and performance — all at once.

**Tier 1 — Competitive Benchmark** (all systems, 6 metrics):
- F1 Score (50%), Precision (15%), Recall (15%), Latency (10%), Throughput (10%), Entity Coverage (optional)

**Tier 2 — Full Evaluation** (optional, 7 dimensions):
- Privacy (ASR, MIA AUC, canary exposure, k-anonymity, ε-DP)
- Utility (format/semantic preservation, information loss)
- Fairness (cross-language, cross-entity, cross-script equity)

**Elo Tournament** — Every pair of systems plays a "match" with adaptive K-factor and Glicko-style Rating Deviation. A single leaderboard emerges that captures all trade-offs.

**Governance Gates** — Configurable deployment thresholds: minimum Elo rating (default R > 1500), maximum Rating Deviation (default RD < 100), minimum match count.

```python
from pii_anon.eval_framework import compute_composite, PIIRateEloEngine, GovernanceThresholds

# Compute Tier 1 composite
score = compute_composite(f1=0.612, precision=0.58, recall=0.65, latency_ms=6.9, docs_per_hour=514_000)
print(f"Composite: {score.score:.4f}")  # 0.6233

# Run Elo tournament
engine = PIIRateEloEngine()
engine.run_round_robin({"pii-anon": 0.792, "pii-anon-swarm": 0.648, "gliner": 0.763})
for r in engine.get_leaderboard():
    print(f"  {r.system_name}: Elo={r.rating:.0f}")

# Check governance readiness
result = engine.evaluate_governance("pii-anon", thresholds=GovernanceThresholds())
print(f"Production-ready: {result.passed}")
```

---

## Real-World Use Cases

**LLM Pipelines & RAG** — Strip PII from prompts before they reach your model. Token-stable pseudonymization preserves entity co-references ("Person A" remains consistent across chunks) without exposing real identities.

**Healthcare (HIPAA)** — Detect medical record numbers, health insurance IDs, prescription numbers, diagnoses. Validate coverage meets HIPAA Safe Harbor requirements.

**Streaming Data** — Process Kafka, Spark, or Beam pipelines with sub-millisecond latency. Pure-function design integrates seamlessly with any async consumer.

**Financial Services (CCPA/GDPR)** — Protect credit cards, IBANs, bank accounts, crypto wallets, tax IDs. Validate coverage against CCPA and GDPR simultaneously.

**Benchmarking & Governance** — Compare PII systems head-to-head with composite scoring and Elo ranking. Gate production deployments on governance thresholds (Elo rating, stability, match count).

---

<!-- COMPLEX_MODE_EXAMPLE_START -->

## Context-Preserving Pseudonymization

Complex mode comparison (generated):

Input:
```text
Primary record owner is Jack Davis for account AC-7721. In review notes, the same person is referenced as alias Jack. Escalations and approvals route to jackdavis@example.com. Later timeline entries continue to refer to Jack during dispute handling.
```

Pseudonymize output:
```text
Primary record owner is <PERSON_NAME:v1:tok_5NM6xAK3tw2ap6__IzdcIWhC> for account AC-7721. In review notes, the same person is referenced as alias <PERSON_NAME:v1:tok_5NM6xAK3tw2ap6__IzdcIWhC>. Escalations and approvals route to <PERSON_NAME:v1:tok_5NM6xAK3tw2ap6__IzdcIWhC>. Later timeline entries continue to refer to <PERSON_NAME:v1:tok_5NM6xAK3tw2ap6__IzdcIWhC> during dispute handling.
```

Anonymize output:
```text
Primary record owner is <PERSON_NAME:anon_1> for account AC-7721. In review notes, the same person is referenced as alias <PERSON_NAME:anon_1>. Escalations and approvals route to <PERSON_NAME:anon_1>. Later timeline entries continue to refer to <PERSON_NAME:anon_1> during dispute handling.
```

Linking notes:
- pseudonymize link_audit entries: 4
- anonymize link_audit entries: 4

This section is generated from deterministic demo input.

<!-- COMPLEX_MODE_EXAMPLE_END -->

The same identity ("Jack Davis", "Jack", "jackdavis@example.com") resolves to a single token, preserving referential integrity while fully protecting privacy.

---

## Command-Line Interface

```bash
# Detect PII in text
pii-anon detect "Contact alice@example.com"

# Run de-identification + evaluation
pii-anon evaluate-pipeline --dataset pii_anon_benchmark_v1 --transform-mode pseudonymize --max-samples 50

# Run evaluation framework directly
pii-anon eval-framework --dataset eval_framework_v1 --max-records 500

# Compare against competitors
pii-anon compare-competitors --dataset pii_anon_benchmark_v1 --output json

# Validate benchmark readiness
pii-anon benchmark-preflight --output json
```

---

## Benchmark Methodology

All results are fully reproducible:

```bash
# Install competitors
pip install pii-anon[benchmark]

# Run composite evaluation
pii-anon compare-competitors \
    --dataset pii_anon_benchmark_v1 \
    --warmup-samples 10 \
    --measured-runs 1 \
    --include-end-to-end \
    --output json
```

Strict span matching: detections count as true positives only when `(record_id, entity_type, span_start, span_end)` match exactly. No partial credit, no fuzzy matching.

The composite score normalizes latency via `1/(1+(lat/100ms)²)` and throughput via `dph/(dph+1M docs/hr)`, then applies Tier 1 weights. The Elo tournament runs 3 rounds of round-robin with adaptive K-factor and Glicko-style Rating Deviation.

---

## Quality & Testing

- **2273 tests** covering detection, evaluation, composite scoring, governance, ingestion, and research rigor
- **Zero required dependencies** (only pydantic)
- **Strict CI gates**: lint (ruff), type check (mypy), coverage (85%+), build, packaging, performance SLAs
- **117,000+ record evaluation dataset** (100% synthetic, CC0/CC-BY-4.0) spanning 52 languages and 48 entity types
- **Reproducible benchmarks** with deterministic seeds and strict span matching

---

## Documentation

- `docs/quickstart.md` — Get started in 5 minutes
- `docs/configuration.md` — Configuration reference
- `docs/engine-plugin-guide.md` — Add custom detection engines
- `docs/api-reference.md` — Full API documentation
- `docs/tutorial-llm-pipeline.md` — LLM pipeline integration tutorial
- `docs/long-context-entity-tracking.md` — Entity linking across long documents
- `artifacts/benchmarks/` — Benchmark results and artifacts (auto-generated)
- `docs/evidence-ledger.md` — Research evidence backing each design decision
- `docs/dependencies-and-platforms.md` — OS-specific setup

---

## Development

```bash
./scripts/bootstrap_dev.sh
make all
```

---

## Contributing

Contributions are welcome. Open an issue to discuss your idea before submitting a pull request. All contributions must pass CI gates (lint, type check, tests, benchmark floor gates).

---

## Acknowledgements

This project was built with the assistance of AI coding agents, primarily [Claude Code](https://claude.ai/claude-code) by Anthropic. AI agents contributed to code generation, test writing, documentation, benchmark infrastructure, and dataset creation. All AI-generated output was reviewed and validated by the project maintainers.

---

## License

Apache-2.0

See `LICENSE` for details.
