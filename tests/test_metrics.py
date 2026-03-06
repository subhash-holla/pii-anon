from pii_anon.metrics import LeakageAtTMetric, TokenStabilityMetric


def test_leakage_at_t_metric():
    metric = LeakageAtTMetric()
    out = metric.compute(predictions=[{"output": "safe"}], labels=[{"token": "email@example.com"}], context={"horizon_seconds": 30})
    assert out.name == "leakage_at_t"


def test_token_stability_metric():
    metric = TokenStabilityMetric()
    out = metric.compute(
        predictions=[
            {"entity_key": "u1", "token": "tok_a"},
            {"entity_key": "u1", "token": "tok_a"},
            {"entity_key": "u2", "token": "tok_b"},
        ],
        labels=[],
        context={},
    )
    assert out.value == 1.0
