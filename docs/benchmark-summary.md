## Accuracy Objective (profiles: long_document, structured_form_accuracy, multilingual_mix)

Benchmark dataset: `pii_anon_benchmark`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6793 | 0.764 | — | 0.910 | 0.658 | 83.301 | 34264.71 | 0 |
| pii-anon | available | 0.7713 | 0.814 | — | 0.828 | 0.799 | 9.963 | 270684.91 | 0 |
| pii-anon-swarm | available | 0.5562 | 0.610 | — | 0.486 | 0.818 | 94.925 | 30083.59 | 0 |
| presidio | available | 0.5083 | 0.491 | — | 0.402 | 0.631 | 15.372 | 112040.93 | 0 |
| scrubadub | available | 0.5158 | 0.333 | — | 0.857 | 0.206 | 0.239 | 9026785.40 | 0 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.771 vs best 0.771).
- recall: within 5% of best (0.799 vs best 0.818).
- f1: within 5% of best (0.814 vs best 0.814).

Weaknesses for `pii-anon`:
- docs_per_hour: more than 10% below best (270684.910 vs best 9026785.400).
- latency_p50_ms: more than 10% slower than best (9.963 vs best 0.239).

This section is generated from benchmark artifacts.
## Speed Objective (profiles: short_chat, structured_form_latency, log_lines)

Benchmark dataset: `pii_anon_benchmark`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6792 | 0.764 | — | 0.910 | 0.658 | 83.562 | 34643.42 | 1539 |
| pii-anon | available | 0.7819 | 0.756 | — | 0.720 | 0.796 | 0.448 | 3320714.12 | 1583 |
| pii-anon-swarm | available | 0.5565 | 0.610 | — | 0.486 | 0.818 | 94.583 | 30559.70 | 1471 |
| presidio | available | 0.5087 | 0.491 | — | 0.402 | 0.631 | 15.299 | 117237.20 | 1445 |
| scrubadub | available | 0.5116 | 0.333 | — | 0.857 | 0.206 | 0.238 | 6270289.94 | 1450 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.782 vs best 0.782).
- recall: within 5% of best (0.796 vs best 0.818).
- f1: within 5% of best (0.756 vs best 0.764).

Weaknesses for `pii-anon`:
- precision: more than 10% below best (0.720 vs best 0.910).
- docs_per_hour: more than 10% below best (3320714.120 vs best 6270289.940).
- latency_p50_ms: more than 10% slower than best (0.448 vs best 0.238).

This section is generated from benchmark artifacts.

Profile floor-gate results:
- `short_chat` (speed): floor_pass=False
- `long_document` (accuracy): floor_pass=True
- `structured_form_accuracy` (accuracy): floor_pass=True
- `structured_form_latency` (speed): floor_pass=False
- `log_lines` (speed): floor_pass=False
- `multilingual_mix` (accuracy): floor_pass=True

### Statistical Significance

Evaluated on **148,994** records. Minimum detectable effect (MDE) at α=0.05, power=0.80: **0.0015** F1 points.

| System | F1 | 95% CI | Samples |
|---|---:|---|---:|
| gliner | 0.764 | [0.736, 0.738] | 148,994 |
| pii-anon | 0.756 | [0.728, 0.730] | 148,994 |
| pii-anon-swarm | 0.610 | [0.590, 0.592] | 148,994 |
| presidio | 0.491 | [0.499, 0.501] | 148,994 |
| scrubadub | 0.333 | [0.326, 0.329] | 148,994 |

> **Known issue (this run only):** the **F1** column is the micro-averaged F1
> (pooled TP/FP/FN), but the **95% CI** in this table was bootstrapped over the
> *per-record (macro) F1* distribution — a different statistic — so a point
> estimate can sit outside its own interval. This is a reporting artifact of this
> run. The CI computation has since been corrected to a micro-F1 cluster
> bootstrap (`MetricAggregator.compute_micro_f1_confidence_interval`) and will be
> self-consistent from the next benchmark run onward. The pairwise comparisons
> below (paired per-record bootstrap) are unaffected.

Pairwise comparisons (paired bootstrap, n=10,000 resamples):

| Comparison | ΔF1 | p-value | Significant | Effect Size |
|---|---:|---:|---|---|
| pii-anon-swarm vs scrubadub | +0.2635 | 0.4921 | n.s. | large (d=+1.297) |
| pii-anon vs gliner | -0.0082 | 0.4959 | n.s. | negligible (d=-0.038) |
| pii-anon vs scrubadub | +0.4014 | 0.4966 | n.s. | large (d=+1.763) |
| pii-anon-swarm vs gliner | -0.1461 | 0.4998 | n.s. | medium (d=-0.764) |
| pii-anon-swarm vs presidio | +0.0911 | 0.5013 | n.s. | small (d=+0.467) |
| pii-anon vs presidio | +0.2290 | 0.5034 | n.s. | large (d=+1.039) |

*Method: paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*
