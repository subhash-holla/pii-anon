from __future__ import annotations

import time
from dataclasses import dataclass

from pii_anon.benchmarks.datasets import load_benchmark_dataset
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@dataclass
class BenchmarkSummary:
    dataset: str
    samples: int
    total_seconds: float
    docs_per_hour: float


def run_benchmark(
    orchestrator: PIIOrchestrator,
    *,
    dataset: str = "pii_anon_benchmark_v1",
    mode: str = "weighted_consensus",
    max_samples: int | None = None,
) -> BenchmarkSummary:
    records = load_benchmark_dataset(dataset)
    if max_samples is not None:
        records = records[: max(0, max_samples)]
    profile = ProcessingProfileSpec(profile_id="benchmark", mode=mode)
    segmentation = SegmentationPlan(enabled=False)

    start = time.perf_counter()
    for item in records:
        orchestrator.run(
            {"text": item.text},
            profile=profile,
            segmentation=segmentation,
            scope="benchmark",
            token_version=1,
        )
    elapsed = max(1e-6, time.perf_counter() - start)

    docs_per_hour = (len(records) / elapsed) * 3600.0
    return BenchmarkSummary(
        dataset=dataset,
        samples=len(records),
        total_seconds=elapsed,
        docs_per_hour=docs_per_hour,
    )
