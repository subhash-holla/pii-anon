# The pii-rate-elo Evaluation Framework

`pii-rate-elo` is the evaluation framework built into `pii-anon`. It scores
PII-detection / de-identification systems on a single, composite, pairwise-
comparable scale that combines detection quality, operational cost, privacy
guarantees, and resistance to LLM-era re-identification attacks.

This document is the algorithm reference. For the end-to-end "score my own
pipeline" workflow, see [evaluate-your-pipeline.md](./evaluate-your-pipeline.md).

---

## Why it exists

Reporting F1 alone paints a misleading picture. A system that catches 99% of
entities at 10-second-per-document latency and another that catches 90% at
5-millisecond latency are incomparable under a single F1 number. And neither
number tells you whether behavioral signals still leak enough information
for an LLM to re-identify the speaker (the Lermen et al. 2026 attack).

`pii-rate-elo` solves three problems:

| Problem | How pii-rate-elo addresses it |
|---|---|
| F1 over-weights precision when the cost of a missed entity is catastrophic | **F2** preset (β=2) double-weights recall, per TAB 2022 |
| Accuracy and speed are incommensurable | **Composite metric** normalizes latency / throughput alongside F1 so they can be traded off on one scale |
| Single-shot benchmark scores hide uncertainty | **Glicko-style Elo** with rating deviation (RD) reports pairwise significance + 95% CI |
| Entity-level de-id doesn't capture behavioral leakage | **Tier 3** — RRS + QIC + BSL — models LLM-era adversaries |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         pii-rate-elo framework                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Tier 1 (always on)     Tier 2 (opt-in)       Tier 3 (opt-in)        │
│   ───────────────        ──────────────        ──────────────         │
│   Detection: F1, F2,     Privacy score         RRS (re-id resistance) │
│   precision, recall      Utility score         QIC (quasi-ID coverage)│
│   Latency p50            Fairness score        BSL (behav. leakage)   │
│   Throughput                                                          │
│   Entity coverage                                                     │
│           │                     │                      │              │
│           └─── CompositeConfig.weighted_avg(...) ──────┘              │
│                                │                                      │
│                                ▼                                      │
│                  CompositeScore ∈ [0, 1]                              │
│                                │                                      │
│                                ▼                                      │
│                  PIIRateEloEngine.run_round_robin                     │
│                  (Glicko-style pairwise, margin-of-victory weighted)  │
│                                │                                      │
│                                ▼                                      │
│                  Leaderboard (ranked by Elo, with 95% CI)             │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

All symbols are exported from `pii_anon.eval_framework`.

---

## Tier 1 — Detection + efficiency

Tier 1 is what every system gets scored on, whether or not the evaluator has
privacy annotations. Components and their defaults (in
[composite.py](../src/pii_anon/eval_framework/metrics/composite.py)):

| Component | Weight (default) | Weight (`f2_privacy_first`) | What it measures |
|---|---:|---:|---|
| F1 | 0.25 | 0.10 | Harmonic mean of precision and recall |
| **F2** | 0.00 | **0.35** | β=2 F-score — double-weights recall (TAB 2022) |
| Precision | 0.075 | 0.05 | TP / (TP + FP) |
| Recall | 0.075 | 0.10 | TP / (TP + FN) |
| Latency | 0.10 | 0.10 | Normalized against `reference_latency_ms` (default 100 ms) |
| Throughput | 0.10 | 0.10 | Normalized against `reference_throughput_dph` (default 100,000 dph) |
| Entity coverage | 0.00 | 0.00 | Fraction of gold entity types with non-zero recall |

The default preset is tuned for a balanced "detection quality + operational
cost" message; `CompositeConfig.f2_privacy_first()` is the recommended
preset for any compliance-motivated evaluation because it matches the cost
model where a missed entity carries more regulatory weight than an extra
alert.

### The F2 formula

```
F_β = (1 + β²) · (P · R) / (β² · P + R)
```

At β=2: `F2 = 5·P·R / (4·P + R)`. Doubling recall weight is equivalent to
saying "a false negative is 4× worse than a false positive", which aligns
with the GDPR / HIPAA / CCPA cost model (data breach penalty > over-redaction
ergonomics).

### Latency and throughput normalization

Both are mapped to `[0, 1]` via a smooth reference-anchored curve:

```
norm_latency(t)    = exp( -ln(2) · t / reference_latency_ms )   # 0 at ∞, 0.5 at reference, 1 at 0
norm_throughput(d) = d / (d + reference_throughput_dph)         # standard saturation curve
```

This makes 100 ms → 0.5 and 1000 ms → ~0.06 by default, so slow systems
cannot compensate purely by being accurate.

---

## Tier 2 — Privacy, utility, fairness

Tier 2 layers in three `[0, 1]` scores computed elsewhere in the eval
framework:

- **Privacy**: a composition of k-anonymity, l-diversity, t-closeness, and
  re-identification risk (`metrics/privacy_metrics.py`).
- **Utility**: format preservation + semantic similarity between original
  and de-identified text (`metrics/utility_metrics.py`).
- **Fairness**: worst-group gap across language, script, entity type,
  and difficulty strata (`metrics/fairness_metrics.py`).

When enabled, the composite formula becomes:

```
C_tier2 = α · Privacy + (1 − α) · Utility    # α = cfg.alpha_privacy, default 0.6
C_total = β · C_tier1 + (1 − β) · (pu_weight · C_tier2 + fair_weight · Fairness)
```

where β is `cfg.beta_tier_balance` (default 0.5 — equal split between
detection/cost and privacy/utility/fairness).

---

## Tier 3 — LLM-era re-identification resistance

Added in composite v2 to align with Lermen et al. (2026) "Large-scale online
deanonymization with LLMs" (arXiv:2602.16800), which demonstrated that LLMs
re-identify 67% of users with 90% precision *after* direct PII entities are
removed. Tier 3 scores whether a system defends against that threat model.

| Metric | Formula | Source |
|---|---|---|
| **RRS** — Re-identification Resistance Score | `1 − (attack_recall × attack_precision)` | Lermen et al. 2026 ESRC pipeline output |
| **QIC** — Quasi-Identifier Coverage | `Σ(wᵢ · removed_i) / Σ(wᵢ · total_i)` | `behavioral_signals` block on each dataset record |
| **BSL** — Behavioral Signal Leakage | `1 − cosine_similarity(orig_embedding, sanitized_embedding)` | Stylometry-trained embedding model |

All three live in [composite.py](../src/pii_anon/eval_framework/metrics/composite.py):
`normalize_reidentification_resistance`, `normalize_quasi_identifier_coverage`,
`normalize_behavioral_signal_leakage`. When `cfg.tier3_weight > 0`, the
composite becomes a unified weighted average across all active components
(flat, not nested) — this keeps interpretation simple.

Dataset v1.3.0 ships populated `behavioral_signals`, `privacy_risk.re_identification_resistance_score`,
and the 4-variant `context_preservation` block (including `anonymized_llm_sanitized`)
so Tier 3 inputs are available out-of-the-box — see
[../pii-anon-eval-data/CHANGELOG.md](../../pii-anon-eval-data/CHANGELOG.md).

---

## Deployment profiles

Different use cases want different weight mixes. Rather than forcing every
user to hand-tune a `CompositeConfig`, three presets live in the library:

| Profile | Detection | Operational | Re-ID resistance | Use case |
|---|---:|---:|---:|---|
| `standard` | 0.50 | 0.20 | 0.30 | Default — balanced |
| `high_security` | 0.30 | 0.10 | **0.60** | Finance, healthcare, legal |
| `high_throughput` | 0.40 | **0.40** | 0.20 | Streaming, log redaction |

Select via:

```python
from pii_anon.eval_framework import CompositeConfig, DeploymentProfile

cfg = CompositeConfig.for_deployment("high_security")  # typed as DeploymentProfile
```

Or on the CLI:

```
pii-anon rate-elo --predictor my_pkg.det:predict --deployment-profile high_security
```

---

## Floor gates — the safety layer

Composite averaging can hide catastrophic weaknesses: a system with 0.95 F1
but 0.1 fairness might still score above 0.7. Floor gates catch that.

```python
from pii_anon.eval_framework import CompositeConfig, FloorGateConfig

cfg = CompositeConfig(
    floor_gates=FloorGateConfig(
        enabled=True,
        min_f1=0.60,
        min_privacy=0.70,
        min_fairness=0.50,
        min_entity_coverage=0.80,
        cap_score=0.40,
    ),
)
```

If any enabled gate fails, the composite is capped at `cap_score` (default
0.40) regardless of how well the other components scored. Result includes
a `FloorGateResult` with per-gate pass/fail and remediation suggestions.

### Industry-leadership bar (paper v10)

Paper v10 §4.1.5 formalises a minimum standard a system must clear to
qualify as a next-generation benchmark leader. Two presets encode it:

```python
from pii_anon.eval_framework import (
    CompositeConfig, FloorGateConfig, GovernanceThresholds,
)

cfg = CompositeConfig(floor_gates=FloorGateConfig.industry_leadership())
thresholds = GovernanceThresholds.industry_leadership()
```

The combined bar:

| Layer | Threshold | Why |
|---|---|---|
| **Metric (FloorGateConfig)** | F1 ≥ 0.60, **F2 ≥ 0.65**, entity coverage ≥ 0.80, fairness ≥ 0.50 | Catches catastrophic weakness in any dimension; F2 enforces the recall-heavy cost model |
| **Composite** | overall ≥ 0.75 | All tiers combined must land above the qualifying line |
| **Elo (GovernanceThresholds)** | rating ≥ 1600, RD ≤ 80, matches ≥ 10 | Ensures the score is statistically distinguishable from the baselines, not a lucky single run |

Pairing both layers gives a principled all-or-nothing gate: a release
candidate either clears every threshold or is not a leader. Use it as a
release gate in CI, or as a bar new systems must cross to be added to
the published baselines.

---

## Elo — pairwise rating with uncertainty

`PIIRateEloEngine` in [rating/elo.py](../src/pii_anon/eval_framework/rating/elo.py)
runs a Glicko-style round-robin across system composite scores:

1. **Margin-of-victory** — each pair's result is weighted by the composite
   delta, so a 0.85-vs-0.35 win moves ratings more than 0.85-vs-0.83.
2. **Rating deviation (RD)** — starts at 350, shrinks with each match,
   serves as the half-width of a 95% CI (`1.96 · RD`).
3. **Pairwise significance** — two systems are "distinguishable" when
   `|R_i − R_j| > 2 · √(RD_i² + RD_j²)`.
4. **Tournament summary** — `engine.tournament_summary()` returns ranked
   list + full pairwise significance matrix + minimum distinguishable
   difference.

This means a single benchmark run produces both *point ratings* and
*uncertainty bounds*, so downstream consumers can reason about whether
system A truly beats system B or is within noise.

### Governance thresholds

`GovernanceThresholds` sets minimum Elo, maximum RD, and minimum sample
counts. `engine.evaluate_governance(thresholds)` returns a structured
pass/fail per system — useful for CI gating.

```python
from pii_anon.eval_framework import GovernanceThresholds

result = engine.evaluate_governance(
    GovernanceThresholds(min_elo=1500, max_rd=200, min_matches=4)
)
```

---

## Scorecards and leaderboards

A `SystemScorecard` is the stable wire format for one system's result.
A `BenchmarkScorecard` groups them into a named run. A `Leaderboard` is
a sorted view plus rendering:

```python
from pii_anon.eval_framework import (
    BenchmarkScorecard, Leaderboard, SystemScorecard,
)

bench = BenchmarkScorecard(benchmark_name="my-run", dataset_name="pii_anon")
bench.add_system(SystemScorecard(system_name="A", f1=0.8, composite_score=0.75))
bench.add_system(SystemScorecard(system_name="B", f1=0.7, composite_score=0.65))

board = Leaderboard.from_benchmark_scorecard(bench)
print(board.to_markdown())   # also .to_json(), .to_csv()
```

The `Leaderboard.from_benchmark_scorecard(bench)` classmethod runs the
tournament for you if you haven't already — it's the one-call path.

---

## Reproducibility and evidence

| Artifact | Location |
|---|---|
| Algorithm source | [src/pii_anon/eval_framework/metrics/composite.py](../src/pii_anon/eval_framework/metrics/composite.py) |
| Elo engine source | [src/pii_anon/eval_framework/rating/elo.py](../src/pii_anon/eval_framework/rating/elo.py) |
| Reference benchmark data | [../pii-anon-eval-data](../../pii-anon-eval-data) (v1.3.0: 159,891 records) |
| Committed baseline leaderboard | [artifacts/benchmarks/benchmark-results.json](../artifacts/benchmarks/benchmark-results.json) |
| Research references | `pii_anon.eval_framework.research.references` (`all_references()`) |
| Design paper | `../pii-anon-research-paper/Paper1-PII-Rate-Elo-Framework-v10.md` |

Every component is backed by peer-reviewed research; the `EVIDENCE_REGISTRY`
maps each design decision to the citation that motivates it.

---

## Summary

`pii-rate-elo` is three layers: a **composite metric** that normalizes
accuracy and cost onto one scale, **Glicko-style Elo** that adds pairwise
significance and 95% CIs, and **floor gates / deployment profiles** that
add customization and safety. Tier 3 extends the framework to model
LLM-era re-identification attacks. All of it is exposed as a stable public
API that external systems can plug into via
[evaluate_external_system(...)](./evaluate-your-pipeline.md).
