# Quickstart

## Install

```bash
pip install "pii-anon[cli]"
```

If you also want optional engine-based detection and benchmark comparison:

```bash
pip install "pii-anon[engines,llm-guard,datasets]"
```

For OS-specific environment setup and dependency details, see:
- `docs/dependencies-and-platforms.md`

## Detect with explicit transform mode

```python
from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

orch = PIIOrchestrator(token_key="change-me")
result = orch.run(
    {"text": "Primary owner Jack Davis can be reached at jackdavis@example.com"},
    profile=ProcessingProfileSpec(
        profile_id="quickstart",
        mode="weighted_consensus",
        language="en",
        transform_mode="pseudonymize",
        entity_tracking_enabled=True,
    ),
    segmentation=SegmentationPlan(enabled=False),
    scope="quickstart",
    token_version=1,
)

print(result["transformed_payload"]["text"])
print(result["link_audit"])
```

Switch to anonymization placeholders:

```python
result = orch.run(
    {"text": "Primary owner Jack Davis can be reached at jackdavis@example.com"},
    profile=ProcessingProfileSpec(
        profile_id="quickstart-anon",
        mode="weighted_consensus",
        language="en",
        transform_mode="anonymize",
        placeholder_template="<{entity_type}:anon_{index}>",
        entity_tracking_enabled=True,
    ),
    segmentation=SegmentationPlan(enabled=False),
    scope="quickstart",
    token_version=1,
)
```

## Stream processing

```python
payloads = [{"text": "alice@example.com"}, {"text": "123-45-6789"}]
for item in orch.run_stream(
    payloads,
    profile=ProcessingProfileSpec(profile_id="stream", mode="intersection_consensus"),
    segmentation=SegmentationPlan(enabled=False),
    scope="stream",
    token_version=1,
):
    print(item["confidence_envelope"])
```

## CLI quickstart

```bash
pii-anon detect "Primary owner Jack Davis" --transform-mode pseudonymize --output json
pii-anon detect "Primary owner Jack Davis" --transform-mode anonymize --output json
pii-anon capabilities --output json
pii-anon evaluate --output json
pii-anon evaluate-pipeline --dataset pii_anon_benchmark --transform-mode pseudonymize --max-samples 25 --output json
pii-anon eval-framework --dataset pii_anon_eval --max-records 250 --output json
pii-anon benchmark-preflight --output json
pii-anon compare-competitors --dataset pii_anon_benchmark --dataset-source package-only --require-all-competitors --require-native-competitors --output json
pii-anon benchmark-publish-suite --artifacts-dir artifacts/benchmarks --output json
```

## Evaluate your own pipeline

Score any PII detector against the benchmark and get a pii-rate-elo leaderboard alongside our baselines:

```python
from pii_anon.eval_framework import evaluate_external_system, load_baseline_leaderboard

def my_detector(text: str):
    return [("EMAIL_ADDRESS", 0, 17)]   # iterable of (entity_type, start, end)

result = evaluate_external_system(my_detector, system_name="my-detector", max_records=500)
print(load_baseline_leaderboard().with_scorecard(result.scorecard).to_markdown())
```

Or from the CLI (accepts `module:callable` paths):

```bash
pii-anon rate-elo --predictor my_pkg.detector:predict --max-records 2000 --deployment-profile high_security
```

Full guide: [evaluate-your-pipeline.md](evaluate-your-pipeline.md).
Algorithm reference: [pii-rate-elo.md](pii-rate-elo.md).

For extended documentation, PDLC artifacts, and composite metric methodology, see the [pii-anon-doc](https://github.com/subhash-holla/pii-anon-doc) repository.
