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
pii-anon evaluate-pipeline --dataset pii_anon_benchmark_v1 --transform-mode pseudonymize --max-samples 25 --output json
pii-anon eval-framework --dataset eval_framework_v1 --max-records 250 --output json
pii-anon benchmark-preflight --output json
pii-anon compare-competitors --dataset pii_anon_benchmark_v1 --dataset-source package-only --require-all-competitors --require-native-competitors --output json
pii-anon benchmark-publish-suite --artifacts-dir artifacts/benchmarks --output json
pii-anon benchmark-publish-suite --reuse-current-env --install-no-deps --output json
pii-anon benchmark-publish-suite --no-strict-runtime --no-require-all-competitors --no-require-native-competitors --no-include-end-to-end --no-allow-core-native-engines --no-enforce-publish-claims --no-validate-readme-sync --dataset-source auto --output json
```

For publish-grade metrics, run `benchmark-preflight` and `benchmark-publish-suite` on Linux or macOS.

For extended documentation, PDLC artifacts, and composite metric methodology, see the [pii-anon-doc](https://github.com/subhash-holla/pii-anon-doc) repository.
