## Accuracy Objective (profiles: long_document, structured_form_accuracy, multilingual_mix)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon-full | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon-minimal | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon-standard | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| piiranha | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| presidio | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| scrubadub | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |

Strengths for `pii-anon`:
- No benchmark data available.

Weaknesses for `pii-anon`:
- No benchmark data available.

This section is generated from benchmark artifacts.
## Speed Objective (profiles: short_chat, structured_form_latency, log_lines)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon | available | 0.5855 | 0.405 | — | 0.969 | 0.256 | 0.018 | 112882199.29 | 0 |
| pii-anon-full | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon-minimal | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| pii-anon-standard | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| piiranha | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| presidio | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| scrubadub | skipped (parallel worker process failed) | 0.0000 | 0.000 | — | 0.000 | 0.000 | 0.000 | 0.00 | 0 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.586 vs best 0.586).
- precision: within 5% of best (0.969 vs best 0.969).
- recall: within 5% of best (0.256 vs best 0.256).
- f1: within 5% of best (0.405 vs best 0.405).
- docs_per_hour: within 5% of best (112882199.290 vs best 112882199.290).
- latency_p50_ms: within 5% of best (0.018 vs best 0.018).

Weaknesses for `pii-anon`:
- No metric crossed the weakness threshold in this run.

This section is generated from benchmark artifacts.

Profile floor-gate results:
- `short_chat` (speed): floor_pass=False
- `long_document` (accuracy): floor_pass=False
- `structured_form_accuracy` (accuracy): floor_pass=False
- `structured_form_latency` (speed): floor_pass=False
- `log_lines` (speed): floor_pass=False
- `multilingual_mix` (accuracy): floor_pass=False
