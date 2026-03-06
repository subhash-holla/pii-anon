import pytest

from pii_anon.metrics import (
    BoundaryLossMetric,
    FairnessGapMetric,
    LLMLeakageMetric,
    MetricPlugin,
    SpanFBetaMetric,
)


def test_metric_plugin_base_raises() -> None:
    plugin = MetricPlugin()
    with pytest.raises(NotImplementedError):
        plugin.compute([], [], {})


def test_span_fbeta_metric_computes_expected_fields() -> None:
    metric = SpanFBetaMetric(beta=1.0)
    out = metric.compute(
        predictions=[{"entity_type": "EMAIL_ADDRESS", "start": 1, "end": 5}],
        labels=[{"entity_type": "EMAIL_ADDRESS", "start": 1, "end": 5}],
        context={},
    )
    assert out.name == "span_fbeta"
    assert out.value == 1.0
    assert out.metadata["precision"] == 1.0
    assert out.metadata["recall"] == 1.0


def test_boundary_loss_metric_paths() -> None:
    metric = BoundaryLossMetric()
    out = metric.compute(
        predictions=[{"id": "a"}],
        labels=[
            {"id": "a", "boundary_case": True},
            {"id": "b", "boundary_case": True},
            {"id": "c", "boundary_case": False},
        ],
        context={},
    )
    assert out.name == "boundary_loss"
    assert out.metadata["total"] == 2
    assert out.metadata["missed"] == 1


def test_llm_leakage_metric_and_fairness_gap_metric() -> None:
    llm_metric = LLMLeakageMetric()
    leakage = llm_metric.compute(
        predictions=[{"output": "... tok_x ..."}, {"output": "safe"}],
        labels=[{"token": "tok_x"}, {"token": "tok_y"}],
        context={},
    )
    assert leakage.name == "llm_leakage"
    assert leakage.metadata["leaked"] == 1

    fairness = FairnessGapMetric().compute(
        predictions=[{"id": "1"}],
        labels=[
            {"id": "1", "group": "A"},
            {"id": "2", "group": "A"},
            {"id": "3", "group": "B"},
        ],
        context={},
    )
    assert fairness.name == "fairness_gap"
    assert "A" in fairness.metadata["by_group"]
    assert "B" in fairness.metadata["by_group"]
