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
| gliner | available | 0.6810 | 0.763 | — | 0.908 | 0.658 | 79.627 | 36873.26 | 1534 |
| pii-anon | available | 0.8129 | 0.792 | — | 0.768 | 0.818 | 0.346 | 3791089.96 | 1588 |
| pii-anon-swarm | available | 0.5873 | 0.648 | — | 0.528 | 0.839 | 90.385 | 33306.11 | 1483 |
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
| gliner | 0.763 | [0.759, 0.761] | 106,855 |
| pii-anon | 0.792 | [0.784, 0.786] | 106,855 |
| pii-anon-swarm | 0.648 | [0.646, 0.647] | 106,855 |
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
