## Accuracy Objective (profiles: long_document, structured_form_accuracy, multilingual_mix)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6875 | 0.774 | — | 0.922 | 0.667 | 92.396 | 38748.21 | 0 |
| pii-anon | available | 0.5981 | 0.702 | — | 0.579 | 0.894 | 183.360 | 15084.35 | 0 |
| pii-anon-full | available | 0.5576 | 0.646 | — | 0.506 | 0.895 | 192.220 | 14673.79 | 0 |
| pii-anon-minimal | available | 0.5979 | 0.702 | — | 0.579 | 0.894 | 184.238 | 15027.06 | 0 |
| pii-anon-standard | available | 0.5978 | 0.702 | — | 0.579 | 0.894 | 184.570 | 15028.43 | 0 |
| presidio | available | 0.6005 | 0.604 | — | 0.733 | 0.514 | 13.669 | 151828.62 | 0 |
| scrubadub | available | 0.5361 | 0.345 | — | 0.943 | 0.211 | 0.269 | 10224859.52 | 0 |

Strengths for `pii-anon`:
- recall: within 5% of best (0.894 vs best 0.895).

Weaknesses for `pii-anon`:
- composite_score: more than 10% below best (0.598 vs best 0.688).
- precision: more than 10% below best (0.579 vs best 0.943).
- docs_per_hour: more than 10% below best (15084.350 vs best 10224859.520).
- latency_p50_ms: more than 10% slower than best (183.360 vs best 0.269).

This section is generated from benchmark artifacts.
## Speed Objective (profiles: short_chat, structured_form_latency, log_lines)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6414 | 0.774 | — | 0.922 | 0.667 | 276.806 | 12601.87 | 1573 |
| pii-anon | available | 0.5869 | 0.407 | — | 0.973 | 0.257 | 0.011 | 104378530.11 | 1542 |
| pii-anon-full | available | 0.3580 | 0.430 | — | 0.587 | 0.339 | 525.509 | 5763.97 | 1394 |
| pii-anon-minimal | available | 0.5497 | 0.617 | — | 0.471 | 0.894 | 138.357 | 20045.20 | 1520 |
| pii-anon-standard | available | 0.4497 | 0.503 | — | 0.347 | 0.911 | 340.611 | 9192.39 | 1445 |
| presidio | available | 0.5332 | 0.604 | — | 0.733 | 0.514 | 119.039 | 22982.05 | 1511 |
| scrubadub | available | 0.5363 | 0.345 | — | 0.943 | 0.211 | 0.272 | 10445854.87 | 1505 |

Strengths for `pii-anon`:
- precision: within 5% of best (0.973 vs best 0.973).
- docs_per_hour: within 5% of best (104378530.110 vs best 104378530.110).
- latency_p50_ms: within 5% of best (0.011 vs best 0.011).

Weaknesses for `pii-anon`:
- recall: more than 10% below best (0.257 vs best 0.911).
- f1: more than 10% below best (0.407 vs best 0.774).

This section is generated from benchmark artifacts.

Profile floor-gate results:
- `short_chat` (speed): floor_pass=True
- `long_document` (accuracy): floor_pass=False
- `structured_form_accuracy` (accuracy): floor_pass=False
- `structured_form_latency` (speed): floor_pass=True
- `log_lines` (speed): floor_pass=True
- `multilingual_mix` (accuracy): floor_pass=False

### Statistical Significance

Evaluated on **50,000** records. Minimum detectable effect (MDE) at α=0.05, power=0.80: **0.0027** F1 points.

| System | F1 | 95% CI | Samples |
|---|---:|---|---:|
| gliner | 0.774 | [0.786, 0.788] | 50,000 |
| pii-anon | 0.407 | [0.403, 0.406] | 50,000 |
| pii-anon-full | 0.430 | [0.378, 0.381] | 50,000 |
| pii-anon-minimal | 0.617 | [0.621, 0.623] | 50,000 |
| pii-anon-standard | 0.503 | [0.503, 0.504] | 50,000 |
| presidio | 0.604 | [0.511, 0.516] | 50,000 |
| scrubadub | 0.345 | [0.339, 0.342] | 50,000 |

Pairwise comparisons (paired bootstrap, n=10,000 resamples):

| Comparison | ΔF1 | p-value | Significant | Effect Size |
|---|---:|---:|---|---|
| pii-anon-minimal vs gliner | -0.1653 | 0.4965 | n.s. | large (d=-1.641) |
| pii-anon vs gliner | -0.3830 | 0.4967 | n.s. | large (d=-3.094) |
| pii-anon-minimal vs scrubadub | +0.2814 | 0.4977 | n.s. | large (d=+2.192) |
| pii-anon-minimal vs presidio | +0.1082 | 0.4981 | n.s. | small (d=+0.478) |
| pii-anon-standard vs scrubadub | +0.1625 | 0.4984 | n.s. | large (d=+1.340) |
| pii-anon-full vs scrubadub | +0.0392 | 0.4986 | n.s. | small (d=+0.236) |
| pii-anon-full vs gliner | -0.4075 | 0.4993 | n.s. | large (d=-2.796) |
| pii-anon-standard vs presidio | -0.0107 | 0.5001 | n.s. | negligible (d=-0.048) |
| pii-anon vs presidio | -0.1095 | 0.5003 | n.s. | small (d=-0.461) |
| pii-anon-standard vs gliner | -0.2842 | 0.5022 | n.s. | large (d=-3.108) |
| pii-anon-full vs presidio | -0.1340 | 0.5028 | n.s. | medium (d=-0.537) |
| pii-anon vs scrubadub | +0.0637 | 0.5054 | n.s. | small (d=+0.433) |

*Method: paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*
