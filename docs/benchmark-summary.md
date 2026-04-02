## Accuracy Objective (profiles: long_document, structured_form_accuracy, multilingual_mix)

Benchmark dataset: `pii_anon_benchmark`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6797 | 0.766 | — | 0.912 | 0.661 | 86.212 | 33285.70 | 0 |
| pii-anon | available | 0.7734 | 0.816 | — | 0.834 | 0.799 | 10.252 | 270049.59 | 0 |
| pii-anon-swarm | available | 0.5560 | 0.611 | — | 0.488 | 0.818 | 96.540 | 29572.25 | 0 |
| presidio | available | 0.5122 | 0.496 | — | 0.407 | 0.635 | 15.278 | 111932.54 | 0 |
| scrubadub | available | 0.5166 | 0.333 | — | 0.860 | 0.207 | 0.241 | 8988393.13 | 0 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.773 vs best 0.773).
- recall: within 5% of best (0.799 vs best 0.818).
- f1: within 5% of best (0.816 vs best 0.816).

Weaknesses for `pii-anon`:
- docs_per_hour: more than 10% below best (270049.590 vs best 8988393.130).
- latency_p50_ms: more than 10% slower than best (10.252 vs best 0.241).

This section is generated from benchmark artifacts.
## Speed Objective (profiles: short_chat, structured_form_latency, log_lines)

Benchmark dataset: `pii_anon_benchmark`
Warm-up samples/system: `100`. Measured runs/system: `3`.

| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| gliner | available | 0.6797 | 0.766 | — | 0.912 | 0.661 | 86.244 | 33622.11 | 1539 |
| pii-anon | available | 0.7834 | 0.758 | — | 0.724 | 0.796 | 0.395 | 3228447.23 | 1583 |
| pii-anon-swarm | available | 0.5556 | 0.611 | — | 0.488 | 0.818 | 97.236 | 29711.85 | 1470 |
| presidio | available | 0.5126 | 0.496 | — | 0.407 | 0.635 | 15.100 | 117127.02 | 1448 |
| scrubadub | available | 0.5101 | 0.333 | — | 0.860 | 0.207 | 0.243 | 5088632.90 | 1448 |

Strengths for `pii-anon`:
- composite_score: within 5% of best (0.783 vs best 0.783).
- recall: within 5% of best (0.796 vs best 0.818).
- f1: within 5% of best (0.758 vs best 0.766).

Weaknesses for `pii-anon`:
- precision: more than 10% below best (0.724 vs best 0.912).
- docs_per_hour: more than 10% below best (3228447.230 vs best 5088632.900).
- latency_p50_ms: more than 10% slower than best (0.395 vs best 0.243).

This section is generated from benchmark artifacts.

Profile floor-gate results:
- `short_chat` (speed): floor_pass=False
- `long_document` (accuracy): floor_pass=True
- `structured_form_accuracy` (accuracy): floor_pass=True
- `structured_form_latency` (speed): floor_pass=False
- `log_lines` (speed): floor_pass=False
- `multilingual_mix` (accuracy): floor_pass=True

### Statistical Significance

Evaluated on **140,855** records. Minimum detectable effect (MDE) at α=0.05, power=0.80: **0.0016** F1 points.

| System | F1 | 95% CI | Samples |
|---|---:|---|---:|
| gliner | 0.766 | [0.767, 0.769] | 140,855 |
| pii-anon | 0.758 | [0.758, 0.760] | 140,855 |
| pii-anon-swarm | 0.611 | [0.614, 0.615] | 140,855 |
| presidio | 0.496 | [0.522, 0.524] | 140,855 |
| scrubadub | 0.333 | [0.340, 0.342] | 140,855 |

Pairwise comparisons (paired bootstrap, n=10,000 resamples):

| Comparison | ΔF1 | p-value | Significant | Effect Size |
|---|---:|---:|---|---|
| pii-anon-swarm vs scrubadub | +0.2735 | 0.4922 | n.s. | large (d=+1.482) |
| pii-anon vs gliner | -0.0090 | 0.4951 | n.s. | negligible (d=-0.055) |
| pii-anon-swarm vs presidio | +0.0910 | 0.4988 | n.s. | medium (d=+0.551) |
| pii-anon vs presidio | +0.2356 | 0.4995 | n.s. | large (d=+1.269) |
| pii-anon-swarm vs gliner | -0.1536 | 0.4996 | n.s. | large (d=-1.089) |
| pii-anon vs scrubadub | +0.4181 | 0.5004 | n.s. | large (d=+2.057) |

*Method: paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*
