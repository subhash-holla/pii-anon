from pii_anon import PIIOrchestrator
from pii_anon.benchmarks import load_benchmark_dataset, run_benchmark


def test_benchmark_dataset_loads() -> None:
    data = load_benchmark_dataset("pii_anon_benchmark_v1")
    assert len(data) > 0


def test_benchmark_runner_returns_summary() -> None:
    orch = PIIOrchestrator(token_key="k")
    summary = run_benchmark(orch, dataset="pii_anon_benchmark_v1", max_samples=20)
    assert summary.samples > 0
    assert summary.docs_per_hour > 0
