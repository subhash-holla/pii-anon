## Accuracy Objective (profiles: long_document, structured_form_accuracy, multilingual_mix)

Benchmark dataset: `pii_anon_benchmark_v1`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6859 | 0.774 | — | 0.922 | 0.667 | 86.756 | 34370.56 | 0 |
| pii-anon | available | 0.8266 | 0.874 | — | 0.857 | 0.892 | 6.623 | 380165.30 | 0 |
| pii-anon-ensemble | available | 0.6269 | 0.700 | — | 0.565 | 0.922 | 98.582 | 30367.15 | 0 |
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
| gliner | available | 0.6858 | 0.774 | — | 0.922 | 0.667 | 87.021 | 34592.09 | 1524 |
| pii-anon | available | 0.8389 | 0.825 | — | 0.807 | 0.844 | 0.371 | 3827207.89 | 1590 |
| pii-anon-ensemble | available | 0.6273 | 0.700 | — | 0.565 | 0.922 | 97.887 | 30917.77 | 1492 |
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
| gliner | 0.774 | [0.786, 0.788] | 50,000 |
| pii-anon | 0.825 | [0.828, 0.830] | 50,000 |
| pii-anon-ensemble | 0.700 | [0.709, 0.711] | 50,000 |
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
