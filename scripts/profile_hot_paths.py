#!/usr/bin/env python
"""Profile pii-anon hot paths to identify optimization opportunities."""
from __future__ import annotations
import cProfile
import pstats
import io
import time
import tracemalloc
import sys
sys.path.insert(0, "src")

from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan
from pii_anon.benchmarks.datasets import load_benchmark_dataset

# Load sample records
try:
    records = load_benchmark_dataset("pii_anon_benchmark")[:50]
    print(f"Loaded {len(records)} records for profiling")
except Exception as e:
    print(f"Dataset unavailable: {e}")
    sys.exit(1)

orch = PIIOrchestrator(token_key="profile-key")
seg = SegmentationPlan(enabled=False)

# Warm up JIT / caches
for rec in records[:5]:
    orch.run({"text": rec.text}, profile=ProcessingProfileSpec(profile_id="p", mode="weighted_consensus", language="en"),
             segmentation=seg, scope="warmup", token_version=1)

def profile_scenario(label: str, mode: str) -> None:
    profile_spec = ProcessingProfileSpec(profile_id="p", mode=mode, language="en")
    tracemalloc.start()
    pr = cProfile.Profile()
    start = time.perf_counter()
    pr.enable()
    for rec in records:
        orch.run({"text": rec.text}, profile=profile_spec, segmentation=seg, scope="profile", token_version=1)
    pr.disable()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Total: {elapsed*1000:.1f}ms | Per-record: {elapsed/len(records)*1000:.2f}ms")
    print(f"Peak memory: {peak/1024:.1f} KiB")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())

profile_scenario("Regex (weighted_consensus)", "weighted_consensus")
profile_scenario("Intersection consensus", "intersection_consensus")
profile_scenario("Union high recall", "union_high_recall")
print("\n=== Composite scoring performance ===")
from pii_anon.eval_framework import compute_composite, CompositeConfig

pr = cProfile.Profile()
pr.enable()
for _ in range(10000):
    compute_composite(f1=0.85, precision=0.87, recall=0.83, latency_ms=20, docs_per_hour=500000)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)
print(s.getvalue())

print("\n=== Tier 3 composite scoring performance ===")
cfg = CompositeConfig.for_deployment("high_security")
pr = cProfile.Profile()
pr.enable()
for _ in range(10000):
    compute_composite(
        f1=0.85, precision=0.87, recall=0.83, latency_ms=20, docs_per_hour=500000,
        reidentification_recall=0.3, reidentification_precision=0.8,
        quasi_identifiers_removed=5, quasi_identifiers_total=10,
        behavioral_signal_similarity=0.25,
        config=cfg,
    )
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(10)
print(s.getvalue())
