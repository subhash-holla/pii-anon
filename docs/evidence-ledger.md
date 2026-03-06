# Evidence Ledger

This ledger links user-facing claims to reproducible evidence generation steps.

## Claim: README benchmark summary is generated from benchmark artifacts
- Command: `python scripts/render_benchmark_summary.py --input-json benchmark-results.json --output-markdown benchmark-summary.md --update-readme README.md`
- Validation: `python scripts/check_readme_benchmark.py --readme README.md --summary benchmark-summary.md`
- Artifact: `benchmark-summary.md` + README benchmark section
- Acceptance: README benchmark section matches generated summary exactly

## Claim: Competitor comparison includes top OSS baselines
- Command: `python scripts/run_competitor_benchmark.py --dataset pii_anon_benchmark_v1 --output-json benchmark-results.json --output-csv benchmark-raw.csv`
- Artifact: `benchmark-results.json`, `benchmark-raw.csv`
- Acceptance: Four systems are present (`pii-anon`, `presidio`, `scrubadub`, `llm_guard`) with explicit availability diagnostics

## Claim: Fusion strategies are comparable with reproducible metrics
- Command: `pii-anon evaluate --dataset pii_anon_benchmark_v1 --output json`
- Artifact: CI job output `evaluate` command logs
- Acceptance: Non-empty strategy table + winner strategy

## Claim: Library can process stream payloads for LLM pipelines
- Command: `pii-anon detect-stream ./sample_payloads.txt --output json`
- Artifact: CI smoke output with item count and per-item results
- Acceptance: Streaming command exits 0 and returns records

## Claim: Performance gates enforce latency/throughput thresholds
- Command: `pytest -m performance`
- Artifact: `tests/performance/test_perf_sla.py` outputs in CI logs
- Acceptance: NFR thresholds pass on CI hardware

## Claim: Optional engines degrade gracefully when dependencies are absent
- Command: `pii-anon capabilities --output json`
- Artifact: engine capability report with `dependency_available` flags
- Acceptance: command exits 0 and health/capability report remains well-formed
