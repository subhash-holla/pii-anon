# Evaluate Your Own PII Pipeline with pii-rate-elo

This guide shows how to score any PII detector — yours, a vendor's, a
research prototype — against the `pii-anon` benchmark and get back a
pii-rate-elo leaderboard that places it head-to-head with the published
baselines (`pii-anon`, `pii-anon-swarm`, Presidio, GLiNER, Scrubadub).

No competitor packages need to be installed. The baseline scorecards are
read from a checked-in artifact. You only need: Python ≥ 3.10, `pii-anon`,
and `pii-anon-datasets`.

For algorithm details, see [pii-rate-elo.md](./pii-rate-elo.md).

---

## The 60-second version

```python
from pii_anon.eval_framework import evaluate_external_system, load_baseline_leaderboard

def my_detector(text: str):
    # return iterable of (entity_type, start, end) tuples
    ...

result = evaluate_external_system(my_detector, system_name="my-detector")
leaderboard = load_baseline_leaderboard().with_scorecard(result.scorecard)
print(leaderboard.to_markdown())
```

That's it. You'll see your system ranked alongside `pii-anon`, `pii-anon-swarm`,
GLiNER, Presidio, and Scrubadub with composite score, F1, precision, recall,
latency, throughput, Elo rating, and rating deviation.

---

## Installation

```bash
pip install "pii-anon[cli,datasets]"
```

The `datasets` extra pulls in `pii-anon-datasets` (v1.3.0: 159,891 records,
63 entity types, 60 languages). If it can't be installed from PyPI yet,
clone the sibling repo and install locally:

```bash
git clone https://github.com/subhash-holla/pii-anon-eval-data.git
pip install -e ./pii-anon-eval-data
```

---

## The predictor contract

Your detector is any callable with this signature:

```python
Predictor = Callable[[str], Iterable[tuple[str, int, int]]]
```

- **Input**: a Unicode string (one evaluation record).
- **Output**: an iterable of `(entity_type, start, end)` tuples.
  - `start`/`end` are 0-indexed, half-open offsets into the input string.
  - `entity_type` is a non-empty string. Common names (`PERSON_NAME`,
    `EMAIL_ADDRESS`, `US_SSN`, etc.) match our taxonomy — see
    [TAXONOMY](../../pii-anon-eval-data/TAXONOMY.md).
- **Extra tuple elements** are ignored. A detector that returns
  `(type, start, end, confidence)` will work fine.
- **Malformed spans** — out-of-bounds offsets, empty entity types, reversed
  spans — are dropped silently (same policy as the reference evaluator).

### Adapter example — plain offsets

```python
def simple(text: str):
    return [
        ("EMAIL_ADDRESS", 8, 25),
        ("PHONE_NUMBER", 30, 45),
    ]
```

### Adapter example — wrapping a spaCy NER model

```python
import spacy
nlp = spacy.load("en_core_web_sm")

SPACY_TO_PII = {"PERSON": "PERSON_NAME", "ORG": "ORGANIZATION", "GPE": "LOCATION"}

def spacy_predictor(text: str):
    doc = nlp(text)
    for ent in doc.ents:
        pii_type = SPACY_TO_PII.get(ent.label_)
        if pii_type:
            yield pii_type, ent.start_char, ent.end_char
```

### Adapter example — wrapping a REST endpoint

```python
import requests

def remote_predictor(text: str):
    resp = requests.post("https://my-pii-service/detect", json={"text": text}, timeout=10)
    resp.raise_for_status()
    for item in resp.json()["entities"]:
        yield item["type"], item["start"], item["end"]
```

---

## Programmatic API

### Basic call

```python
from pii_anon.eval_framework import evaluate_external_system

result = evaluate_external_system(
    my_detector,
    system_name="my-detector",
    dataset="pii_anon",          # v1.1+ canonical dataset
    language="en",                # filter by BCP 47 code, or None for all
    max_records=2_000,            # cap for fast iteration; None = ~160K records
    warmup_records=20,            # exclude warmup from latency measurement
)

print(f"F1={result.scorecard.f1:.3f}")
print(f"Composite={result.scorecard.composite_score:.3f}")
print(f"Samples={result.records_evaluated}, skipped={result.skipped_records}")
```

`result` is an `ExternalEvaluationResult`:

| Field | Type | Notes |
|---|---|---|
| `scorecard` | `SystemScorecard` | Drop-in for `Leaderboard.with_scorecard(...)` |
| `composite` | `CompositeScore` | Full per-component breakdown (`.components` dict) |
| `per_record_f1` | `list[float]` | For bootstrap CI and paired significance |
| `latency_ms_samples` | `list[float]` | Raw latencies in milliseconds |
| `records_evaluated` | `int` | Predictor ran successfully on this many records |
| `skipped_records` | `int` | Predictor raised on this many (when `on_error="skip"`) |
| `errors` | `list[str]` | First 5 predictor errors (for quick diagnosis) |

### Compare against baselines

```python
from pii_anon.eval_framework import load_baseline_leaderboard

baseline = load_baseline_leaderboard()
print("Comparing against:", baseline.system_names())
# ['gliner', 'pii-anon', 'pii-anon-swarm', 'presidio', 'scrubadub']

leaderboard = baseline.with_scorecard(result.scorecard)

print(leaderboard.to_markdown())
# Also: .to_json(indent=2), .to_csv()
```

The baselines come from a committed artifact
([artifacts/benchmarks/benchmark-results.json](../artifacts/benchmarks/benchmark-results.json)),
so **you never need to install Presidio, GLiNER, or Scrubadub** to reproduce
the comparison.

### Custom weights via deployment profiles

```python
result = evaluate_external_system(
    my_detector,
    system_name="my-detector",
    deployment_profile="high_security",   # | "standard" | "high_throughput"
)
```

Profiles, in one line each:

- **`standard`** (default) — balanced: detection 50%, ops 20%, re-ID 30%.
- **`high_security`** — finance/health/legal: detection 30%, ops 10%,
  re-ID **60%**. Heavy penalty for behavioral leakage.
- **`high_throughput`** — streaming/log redaction: detection 40%, ops
  **40%**, re-ID 20%. Latency matters as much as F1.

See [pii-rate-elo.md](./pii-rate-elo.md) for the full weight tables.

### Fully custom config

If the presets don't match your cost model, build a `CompositeConfig`:

```python
from pii_anon.eval_framework import CompositeConfig, FloorGateConfig

cfg = CompositeConfig(
    weight_detection_f1=0.40,
    weight_detection_recall=0.20,
    weight_latency=0.10,
    weight_throughput=0.10,
    weight_privacy=0.20,
    reference_latency_ms=50.0,       # we want to reward sub-50ms systems
    floor_gates=FloorGateConfig(
        enabled=True, min_f1=0.7, min_privacy=0.8, cap_score=0.30,
    ),
)
cfg.validate()   # will raise if weights don't sum to 1.0

result = evaluate_external_system(my_detector, composite_config=cfg)
```

Full config surface lives in
[composite.py](../src/pii_anon/eval_framework/metrics/composite.py).

---

## CLI workflow

The `pii-anon rate-elo` command wraps the entire flow:

```bash
pii-anon rate-elo \
    --predictor my_package.detector:predict \
    --system-name "my-detector" \
    --language en \
    --max-records 2000 \
    --deployment-profile standard \
    --output markdown \
    --artifact-dir ./my-eval-results
```

Argument reference:

| Flag | Purpose |
|---|---|
| `--predictor / -p` | `module.submod:callable` import path to your predictor |
| `--system-name / -n` | Display name in the leaderboard (default: `external-system`) |
| `--dataset` | Dataset identifier — default `pii_anon` (v1.3.0 canonical) |
| `--language` | BCP 47 filter (e.g. `en`, `es`, `zh`) or omit for all |
| `--max-records` | Evaluation cap — `0` for the full ~160K records |
| `--warmup-records` | Warmup count excluded from latency measurement |
| `--deployment-profile` | `standard` \| `high_security` \| `high_throughput` |
| `--output` | `markdown` \| `json` \| `csv` for stdout rendering |
| `--artifact-dir` | If set, writes `scorecard.json` and `leaderboard.{json,md,csv}` |
| `--baseline-artifact` | Override path to `benchmark-results.json` |
| `--on-error` | `skip` (default) \| `raise` when predictor fails on a record |

The stdout output lets you paste the leaderboard straight into a PR or a
report; the `--artifact-dir` output gives you stable files for CI gating.

---

## Reading the results

Example output:

```
| Rank | System          | Composite | F1     | Precision | Recall | Latency | Throughput  | Elo  | RD  |
|------|-----------------|-----------|--------|-----------|--------|---------|-------------|------|-----|
| 1    | my-detector     | 0.791     | 0.812  | 0.805     | 0.819  |   4.1ms |     876,420 | 1580 | 247 |
| 2    | pii-anon        | 0.782     | 0.758  | 0.724     | 0.795  |   0.4ms |   3,064,895 | 1552 | 247 |
| 3    | gliner          | 0.680     | 0.766  | 0.912     | 0.661  |  86.2ms |      33,605 | 1533 | 247 |
```

How to read it:

- **Composite** — the headline score. 0.0–1.0, higher is better. It's a
  weighted mix of F1/F2, latency, throughput, and (if Tier 3 inputs are
  provided) re-identification resistance.
- **Elo** — pairwise rating. A 1580 vs 1552 gap *looks* close but the RD
  column is 247 for both — that means the 95% CI overlaps and the gap
  **is not statistically distinguishable**. To get tighter RDs, run more
  records.
- **RD** (Rating Deviation) — 350 is untrained, ~100 is a confident rating.
  Expected on a 2000-record run: ~240-260.
- **Latency** is p50 millisecond per record. **Throughput** is `3,600,000 / latency`
  (documents per hour at p50).

### Getting statistical significance

```python
from pii_anon.eval_framework import PIIRateEloEngine

engine = PIIRateEloEngine()
engine.run_round_robin({sc.system_name: sc.composite_score for sc in leaderboard.systems})

summary = engine.tournament_summary()
for pair, info in summary["pairwise_significance"].items():
    if not info["significant"]:
        print(f"{pair}: NOT distinguishable (Δ={info['rating_diff']:.0f} < threshold {info['significance_threshold']:.0f})")
```

If your system is "not distinguishable" from `pii-anon` on your first run,
it's either truly comparable or your sample size is too small — doubling
`max_records` typically shrinks RD enough to decide.

---

## CI gating example

Block merges that drop your composite score below a threshold:

```python
# tests/test_quality_gate.py
import pytest
from pii_anon.eval_framework import evaluate_external_system
from my_package.detector import predict

COMPOSITE_FLOOR = 0.70
F1_FLOOR = 0.75

def test_pii_rate_elo_floor():
    result = evaluate_external_system(
        predict,
        system_name="my-detector",
        max_records=2_000,
        deployment_profile="high_security",
    )
    assert result.scorecard.composite_score >= COMPOSITE_FLOOR, (
        f"composite dropped to {result.scorecard.composite_score:.3f} "
        f"(floor {COMPOSITE_FLOOR})"
    )
    assert result.scorecard.f1 >= F1_FLOOR
```

Run it in GitHub Actions with `pytest tests/test_quality_gate.py`.

---

## Tier 3 (LLM re-identification) evaluation

If you have an ESRC-style attack pipeline that reports recall/precision
against de-identified text, pass those as Tier 3 inputs:

```python
from pii_anon.eval_framework import compute_composite, CompositeConfig

# Already ran your detector — now attach the adversary's numbers
composite = compute_composite(
    f1=result.scorecard.f1,
    precision=result.scorecard.precision,
    recall=result.scorecard.recall,
    latency_ms=result.scorecard.latency_p50_ms,
    docs_per_hour=result.scorecard.docs_per_hour,
    reidentification_recall=0.38,          # attacker's recall on your output
    reidentification_precision=0.82,       # attacker's precision on your output
    quasi_identifiers_removed=45,          # QI signals you removed
    quasi_identifiers_total=82,            # QI signals in the original text
    behavioral_signal_similarity=0.24,     # stylometry embedding cosine
    config=CompositeConfig.for_deployment("high_security"),
)

print(f"Tier 3 sub-score: {composite.reidentification_sub:.3f}")
print(f"Adjusted composite: {composite.score:.3f}")
```

The dataset (`pii-anon-datasets` ≥ v1.3.0) ships per-record
`behavioral_signals`, `re_identification_resistance_score`, and a 4th
anonymized variant (`anonymized_llm_sanitized`) so you can drive this
evaluation locally. See the
[CHANGELOG](../../pii-anon-eval-data/CHANGELOG.md) for schema details.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Evaluation dataset 'pii_anon' not found` | `pip install pii-anon-datasets` or set `PII_ANON_DATASET_ROOT`. |
| `FileNotFoundError: benchmark artifact not found` | Run `make benchmark-full` once to generate `artifacts/benchmarks/benchmark-results.json`, or pass `baseline_artifact=` / `--baseline-artifact`. |
| All my `per_record_f1` values are 1.0 | You're probably returning zero predictions *and* the record has zero labels — the evaluator gives F1=1 for that degenerate case. Filter `language="en"` to avoid sparsely-labeled records, or increase `max_records`. |
| `predictor produced zero successful predictions` | Your callable raised on every record. Set `on_error="raise"` to see the exception. |
| Elo doesn't change no matter what I do | You're running against the full baseline — composite differences are small. Use a larger `max_records` or pass a custom `CompositeConfig` with sharper weights. |
| Leaderboard shows me last with throughput=12,328,743,903 | You returned zero predictions, so latency ≈ 0 and throughput saturates. This is a healthy sanity check — a predictor that does nothing ranks last. |

---

## Next steps

- **Understand the scoring** → [pii-rate-elo.md](./pii-rate-elo.md)
- **Contribute your system** → open a PR to add your system to the official
  baselines in `artifacts/benchmarks/benchmark-results.json`
- **Customize weights** → [composite.py](../src/pii_anon/eval_framework/metrics/composite.py)
  — all fields are dataclass attributes with docstrings.
- **Build a richer adapter** → use
  `EvaluationFramework.evaluate_batch(records, predict_fn=...)` directly
  for deep per-entity and per-language diagnostics.
