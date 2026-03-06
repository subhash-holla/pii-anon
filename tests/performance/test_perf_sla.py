import time

import pytest

from pii_anon import PIIOrchestrator
from pii_anon.config import CoreConfig, EngineRuntimeConfig
from pii_anon.engines import RegexEngineAdapter
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@pytest.mark.performance
def test_regex_latency_p50_under_300ms() -> None:
    engine = RegexEngineAdapter(enabled=True)
    text = " ".join(["alice@example.com"] * 10_000)
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
