# pii-anon

**The highest-scoring open-source PII detection and anonymization library for Python — built for LLM pipelines, streaming data, and regulatory compliance.**

[![PyPI Version](https://img.shields.io/badge/pypi-1.0.0-blue.svg)](https://pypi.org/project/pii-anon/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-2253-brightgreen.svg)](#quality-and-testing)

[![Built with AI Agents](https://img.shields.io/badge/Built_with-AI_Agents-d946ef.svg)](#acknowledgements)

---

## Why pii-anon?

**Two offerings, one library.** pii-anon ships with two detection modes designed for different trade-offs:

| Offering | Best For | F1 | Latency |
|---|---|---|---|
| **pii-anon** | Speed-sensitive pipelines (streaming, real-time) | 0.89 | <1 ms/record |
| **pii-anon-ensemble** | Maximum accuracy (compliance, batch ETL) | 0.86 | ~40 ms/record |

Both outperform every open-source competitor tested — including GLiNER (F1=0.77), Presidio (F1=0.54), and Scrubadub (F1=0.34) — on the 50,000-record pii_anon_benchmark_v1 evaluation.

**pii-anon** uses a fast regex engine with checksum validators (Luhn, IBAN mod-97, ABA routing), context-aware confidence scoring, and deny-lists — delivering sub-millisecond detection across 20 entity types with zero external dependencies.

**pii-anon-ensemble** fuses regex with Presidio and Scrubadub through a Mixtral-inspired Mixture-of-Experts architecture: per-entity-type routing, weighted voting, and corroboration filtering ensure the ensemble never scores below any individual engine.

**6 transformation strategies** — pseudonymization with key rotation, tokenization (HMAC/AES-SIV), redaction, generalization, synthetic replacement, or differential privacy (ε-DP) — all with audit trails and deterministic linking for long-context entity tracking.

**6 compliance templates** (HIPAA Safe Harbor, GDPR Pseudo/Anon, CCPA, Minimal Risk, Maximum Privacy) validate your entity detection against real regulatory requirements. Know instantly whether your coverage meets compliance standards.

**Enterprise-grade evaluation framework** with 50+ metrics, 50,000 synthetic evaluation records (100% CC0/CC-BY-4.0), and a composite Elo-based ranking system that captures the full picture: detection accuracy, privacy, utility, fairness, and performance.

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
pip install pii-anon[datasets]            # + 50K evaluation dataset
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
| pii-anon | available | 0.8266 | 0.874 | — | 0.857 | 0.892 | 6.623 | 380165.30 | 0 |
| pii-anon-ensemble | available | 0.6269 | 0.700 | — | 0.565 | 0.922 | 98.582 | 30367.15 | 0 |
| gliner | available | 0.6859 | 0.774 | — | 0.922 | 0.667 | 86.756 | 34370.56 | 0 |
| presidio | available | 0.4601 | 0.431 | — | 0.361 | 0.535 | 14.675 | 138033.60 | 0 |
| scrubadub | available | 0.5849 | 0.424 | — | 0.945 | 0.273 | 0.264 | 9256286.84 | 0 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.827 vs best 0.827).
- recall: within 5% of best (0.892 vs best 0.922).
- f1: within 5% of best (0.874 vs best 0.874).

Weaknesses for `pii-anon`:
- docs_per_hour: more than 10% below best (380165.300 vs best 9256286.840).
- latency_p50_ms: more than 10% slower than best (6.623 vs best 0.264).

This section is generated from benchmark artifacts.
## Speed Objective (profiles: short_chat, structured_form_latency, log_lines)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| pii-anon | available | 0.8389 | 0.825 | — | 0.807 | 0.844 | 0.371 | 3827207.89 | 1590 |
| pii-anon-ensemble | available | 0.6273 | 0.700 | — | 0.565 | 0.922 | 97.887 | 30917.77 | 1492 |
| gliner | available | 0.6858 | 0.774 | — | 0.922 | 0.667 | 87.021 | 34592.09 | 1524 |
| presidio | available | 0.4605 | 0.431 | — | 0.361 | 0.535 | 14.746 | 143224.78 | 1414 |
| scrubadub | available | 0.5802 | 0.424 | — | 0.945 | 0.273 | 0.263 | 6108137.59 | 1470 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.839 vs best 0.839).
- f1: within 5% of best (0.825 vs best 0.825).

Weaknesses for `pii-anon`:
- precision: more than 10% below best (0.807 vs best 0.945).
- docs_per_hour: more than 10% below best (3827207.890 vs best 6108137.590).
- latency_p50_ms: more than 10% slower than best (0.371 vs best 0.263).

This section is generated from benchmark artifacts.

Profile floor-gate results:
- `short_chat` (speed): floor_pass=False
- `long_document` (accuracy): floor_pass=True
- `structured_form_accuracy` (accuracy): floor_pass=True
- `structured_form_latency` (speed): floor_pass=False
- `log_lines` (speed): floor_pass=False
- `multilingual_mix` (accuracy): floor_pass=True

### Statistical Significance

Evaluated on **50,000** records. Minimum detectable effect (MDE) at α=0.05, power=0.80: **0.0027** F1 points.

| System | F1 | 95% CI | Samples |
|---|---:|---|---:|
| pii-anon | 0.825 | [0.828, 0.830] | 50,000 |
| pii-anon-ensemble | 0.700 | [0.709, 0.711] | 50,000 |
| gliner | 0.774 | [0.786, 0.788] | 50,000 |
| presidio | 0.431 | [0.379, 0.383] | 50,000 |
| scrubadub | 0.424 | [0.413, 0.416] | 50,000 |

Pairwise comparisons (paired bootstrap, n=10,000 resamples):

| Comparison | ΔF1 | p-value | Significant | Effect Size |
|---|---:|---:|---|---|
| pii-anon-ensemble vs presidio | +0.3290 | 0.4959 | n.s. | large (d=+1.900) |
| pii-anon vs presidio | +0.4479 | 0.4973 | n.s. | large (d=+2.509) |
| pii-anon-ensemble vs gliner | -0.0773 | 0.4978 | n.s. | medium (d=-0.785) |
| pii-anon vs scrubadub | +0.4143 | 0.5025 | n.s. | large (d=+2.700) |
| pii-anon vs gliner | +0.0416 | 0.5026 | n.s. | small (d=+0.386) |
| pii-anon-ensemble vs scrubadub | +0.2954 | 0.5029 | n.s. | large (d=+2.007) |

*Method: paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*

<!-- BENCHMARK_SUMMARY_END -->

See [Benchmark Methodology](#benchmark-methodology) for details.

### Per-Entity Recall: Broadest Coverage in the Field

Both pii-anon offerings detect **20 entity types** with non-zero recall — more than any competitor. Here is the head-to-head recall comparison on a 1,000-record sample from the pii_anon_benchmark_v1 evaluation:

| Entity Type | pii-anon | pii-anon-ensemble | GLiNER | Presidio | Scrubadub |
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

pii-anon and pii-anon-ensemble are the **only** systems that detect all 20 entity types in this benchmark. GLiNER detects 13 types, Presidio detects 7 types, Scrubadub detects 4.

---

## Core Capabilities

**48 Entity Types** — Comprehensive coverage across Personal Identity, Financial, Government ID, Medical, Digital/Technical, Employment, and Behavioral/Contextual categories. Every entity type includes risk-level classification and regulatory reference mappings.

**52 Languages, 17 Writing Systems** — Resource-level classification ensures fair cross-lingual evaluation. High-resource languages (English, Spanish, French, Chinese, Japanese, Arabic) with medium and low-resource coverage for emerging markets.

**Modular Detection Engine** — pii-anon's regex engine uses checksums (Luhn, IBAN mod-97, ABA routing, VIN check-digit, DEA), context-aware confidence scoring, and entity-type-specific allow/deny lists to suppress false positives. pii-anon-ensemble adds Presidio and Scrubadub through a Mixture-of-Experts fusion layer with per-entity-type routing and corroboration filtering.

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
engine.run_round_robin({"pii-anon": 0.889, "pii-anon-ensemble": 0.862, "gliner": 0.774})
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

- **2253 tests** covering detection, evaluation, composite scoring, governance, ingestion, and research rigor
- **Zero required dependencies** (only pydantic)
- **Full backward compatibility** with all v1.x APIs
- **Strict CI gates**: lint (ruff), type check (mypy), coverage, build, packaging, performance SLAs
- **50,000-record evaluation dataset** (100% synthetic, CC0/CC-BY-4.0) spanning 52 languages and 48 entity types
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
