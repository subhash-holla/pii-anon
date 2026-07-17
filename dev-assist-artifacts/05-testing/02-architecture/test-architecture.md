# Test Architecture — pii-anon (Stage 5)

> **Brownfield Mode.** Extends the existing 2,548-test pytest suite (do not rebuild). AGENT_SIMULATED stage; the user is the Pass-2 cohort.

## Pyramid (current)
- **Unit** (bulk of 2,548): per-module (engines, transforms, tokenization, eval_framework, fusion, swarm, moe, ingestion).
- **Integration**: `tests/integration/`, `*_integration.py` (eval-framework, ingestion, swarm-baseline).
- **Property** (NEW): `tests/test_shared_layer_projector.py` — seeded-random superset-invariant (2,000 cases). Migration to `hypothesis @given` tracked as S1-04.
- **Performance/SLA**: `tests/performance/test_perf_sla.py` (absolute thresholds, CI-gated) + the comparator-relative floor gate (`make benchmark`, manual — to be CI-wired in S7).
- **Benchmark**: `scripts/run_competitor_benchmark.py` (provisional — published numbers are a 50-sample smoke run; canonical run pending S7).

## Cross-cutting discipline
strict mypy + ruff on every push; branch coverage `--cov-fail-under=84`; markers (`performance`); dataset-gated skips. New `routing/` module: ruff + mypy --strict clean.

## Gaps (→ stories)
property-based testing for crypto/checksum invariants (S1-04/hypothesis); per-modality + agentic-leakage harnesses (S5/S7); doctest/exec harness for README snippets (docs); canonical benchmark run + CI-wired floor gate (S7).
