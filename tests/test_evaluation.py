import pytest

from pii_anon import PIIOrchestrator
from pii_anon.benchmarks import BenchmarkRecord
from pii_anon.evaluation import StrategyEvaluator
from pii_anon.evaluation import compare as compare_module


def test_strategy_evaluator_returns_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [
        BenchmarkRecord(
            record_id="r1",
            text="Contact alice@example.com",
            labels=[{"entity_type": "EMAIL_ADDRESS", "start": 8, "end": 25}],
            language="en",
        ),
        BenchmarkRecord(
            record_id="r2",
            text="Call +1 415 555 0100",
            labels=[{"entity_type": "PHONE_NUMBER", "start": 5, "end": 20}],
            language="en",
        ),
    ]
    monkeypatch.setattr(compare_module, "load_benchmark_dataset", lambda _dataset: samples)

    orch = PIIOrchestrator(token_key="k")
    evaluator = StrategyEvaluator(orch)
    report = evaluator.compare_strategies(
        ["weighted_consensus", "union_high_recall"],
        dataset="pii_anon_benchmark_v1",
    )
    assert report.winner in {"weighted_consensus", "union_high_recall"}
    assert len(report.results) == 2
