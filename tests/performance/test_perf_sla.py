import time
import tracemalloc

import pytest

from pii_anon import PIIOrchestrator
from pii_anon.config import CoreConfig, EngineRuntimeConfig
from pii_anon.engines import RegexEngineAdapter
from pii_anon.tokenization.store import InMemoryTokenStore
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@pytest.mark.performance
def test_regex_latency_p50_under_300ms() -> None:
    engine = RegexEngineAdapter(enabled=True)
    text = " ".join(["alice@example.com"] * 1_000)
    payload = {"text": text}
    context = {"language": "en", "policy_mode": "balanced"}

    # Warm-up runs to reduce first-invocation noise in CI / shared runners.
    for _ in range(3):
        engine.detect(payload, context)

    latencies = []
    for _ in range(7):
        start = time.perf_counter()
        findings = engine.detect(payload, context)
        latencies.append((time.perf_counter() - start) * 1000.0)
        assert findings
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    # 300ms accommodates shared CI runners and Python 3.13 variance.
    assert p50 <= 300.0


@pytest.mark.performance
def test_multi_engine_latency_p50_under_500ms() -> None:
    config = CoreConfig(
        engines={
            "regex-oss": EngineRuntimeConfig(enabled=True, weight=1.0),
            "presidio-compatible": EngineRuntimeConfig(enabled=True, weight=1.2),
            "llm-guard-compatible": EngineRuntimeConfig(enabled=True, weight=1.1),
            "scrubadub-compatible": EngineRuntimeConfig(enabled=True, weight=0.9),
        }
    )
    orch = PIIOrchestrator(token_key="k", config=config)
    text = "alice@example.com 123-45-6789 Dr Smith"
    profile = ProcessingProfileSpec(profile_id="perf", mode="intersection_consensus", min_consensus=1)
    segmentation = SegmentationPlan(enabled=False)

    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        orch.run({"text": text}, profile=profile, segmentation=segmentation, scope="perf", token_version=1)
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    assert p50 <= 500.0


@pytest.mark.performance
def test_throughput_over_10k_docs_per_hour() -> None:
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(profile_id="perf", mode="weighted_consensus")
    segmentation = SegmentationPlan(enabled=False)
    docs = [{"text": "alice@example.com"}] * 200

    start = time.perf_counter()
    for item in docs:
        orch.run(item, profile=profile, segmentation=segmentation, scope="perf", token_version=1)
    elapsed = max(1e-6, time.perf_counter() - start)

    docs_per_hour = (len(docs) / elapsed) * 3600.0
    assert docs_per_hour >= 10_000.0


@pytest.mark.performance
def test_linear_scaling_proxy() -> None:
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(profile_id="perf", mode="weighted_consensus")
    segmentation = SegmentationPlan(enabled=False)

    sizes = [20, 40, 80, 160]
    times = []
    for size in sizes:
        docs = [{"text": "alice@example.com"}] * size
        start = time.perf_counter()
        for item in docs:
            orch.run(item, profile=profile, segmentation=segmentation, scope="perf", token_version=1)
        times.append(time.perf_counter() - start)

    # Linear trend validation (R^2 >= 0.95).
    x_mean = sum(sizes) / len(sizes)
    y_mean = sum(times) / len(times)
    ss_tot = sum((y - y_mean) ** 2 for y in times)
    ss_res = 0.0
    slope = sum((sx - x_mean) * (sy - y_mean) for sx, sy in zip(sizes, times)) / max(
        1e-9,
        sum((sx - x_mean) ** 2 for sx in sizes),
    )
    intercept = y_mean - (slope * x_mean)
    for x, y in zip(sizes, times):
        pred = (slope * x) + intercept
        ss_res += (y - pred) ** 2
    r2 = 1.0 - (ss_res / max(1e-9, ss_tot))
    assert r2 >= 0.95


@pytest.mark.performance
def test_large_document_latency_under_5s() -> None:
    """10K-word document with ~200 scattered PII entities processes in <5s."""
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(profile_id="perf", mode="weighted_consensus")
    segmentation = SegmentationPlan(enabled=False)

    filler = "The quick brown fox jumped over the lazy dog. "
    pii_entities = [
        "alice@example.com",
        "Bob Smith",
        "123-45-6789",
        "jane.doe@corp.net",
        "Dr. Johnson",
        "987-65-4321",
    ]
    words: list[str] = []
    for i in range(200):
        words.append(filler * 8)
        words.append(pii_entities[i % len(pii_entities)])
    words.append(filler * 50)
    text = " ".join(words)

    # Warm-up
    orch.run({"text": "alice@example.com"}, profile=profile, segmentation=segmentation, scope="perf", token_version=1)

    start = time.perf_counter()
    result = orch.run({"text": text}, profile=profile, segmentation=segmentation, scope="perf", token_version=1)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert len(result.get("ensemble_findings", [])) > 0


@pytest.mark.performance
def test_token_store_performance_10k() -> None:
    """10K put/get operations complete in <1s."""
    from pii_anon.tokenization.store import TokenMapping

    store = InMemoryTokenStore()

    start = time.perf_counter()
    for i in range(10_000):
        store.put(
            TokenMapping(
                scope="bench",
                token=f"<EMAIL:v1:tok_{i}>",
                plaintext=f"user{i}@example.com",
                entity_type="EMAIL_ADDRESS",
                version=1,
            )
        )
    for i in range(10_000):
        found = store.get(f"<EMAIL:v1:tok_{i}>", scope="bench")
        assert found is not None
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


@pytest.mark.performance
def test_linker_100_unique_entities() -> None:
    """100 unique person names + emails linked in <2s."""
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(
        profile_id="perf",
        mode="weighted_consensus",
        entity_tracking_enabled=True,
    )
    segmentation = SegmentationPlan(enabled=False)

    names = [f"Person{i} Lastname{i}" for i in range(100)]
    emails = [f"person{i}@company{i}.com" for i in range(100)]
    lines = [f"{names[i]} can be reached at {emails[i]}." for i in range(100)]
    text = " ".join(lines)

    start = time.perf_counter()
    result = orch.run({"text": text}, profile=profile, segmentation=segmentation, scope="perf", token_version=1)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
    assert len(result.get("ensemble_findings", [])) > 0


@pytest.mark.performance
def test_ensemble_multi_entity_throughput() -> None:
    """Realistic multi-entity docs achieve >5K docs/hour."""
    config = CoreConfig(
        engines={
            "regex-oss": EngineRuntimeConfig(enabled=True, weight=1.0),
            "presidio-compatible": EngineRuntimeConfig(enabled=True, weight=1.2),
        }
    )
    orch = PIIOrchestrator(token_key="k", config=config)
    profile = ProcessingProfileSpec(
        profile_id="perf",
        mode="weighted_consensus",
        audit_enabled=False,
    )
    segmentation = SegmentationPlan(enabled=False)
    doc = {"text": "Alice Smith alice@example.com 123-45-6789 Dr. Johnson 555-0123"}
    docs = [doc] * 100

    start = time.perf_counter()
    for item in docs:
        orch.run(item, profile=profile, segmentation=segmentation, scope="perf", token_version=1)
    elapsed = max(1e-6, time.perf_counter() - start)

    docs_per_hour = (len(docs) / elapsed) * 3600.0
    assert docs_per_hour >= 5_000.0


@pytest.mark.performance
def test_memory_bounded_batch() -> None:
    """1000 docs use <100MB peak RSS via tracemalloc."""
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(
        profile_id="perf",
        mode="weighted_consensus",
        audit_enabled=False,
    )
    segmentation = SegmentationPlan(enabled=False)

    tracemalloc.start()
    for i in range(1000):
        orch.run(
            {"text": f"user{i}@example.com called 555-{i:04d}"},
            profile=profile,
            segmentation=segmentation,
            scope="perf",
            token_version=1,
        )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 100.0
