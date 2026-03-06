from pii_anon.fusion import build_fusion, register_fusion_strategy
from pii_anon.types import EngineFinding


def test_intersection_fusion_mode() -> None:
    fusion = build_fusion("intersection_consensus", weights={}, min_consensus=2)
    findings = [
        EngineFinding("EMAIL_ADDRESS", 0.9, field_path="text", span_start=0, span_end=5, engine_id="a"),
        EngineFinding("EMAIL_ADDRESS", 0.8, field_path="text", span_start=0, span_end=5, engine_id="b"),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    assert merged[0].confidence == 0.8


def test_custom_fusion_registration() -> None:
    class ZeroFusion:
        strategy_id = "zero"

        def merge(self, findings):
            return []

    register_fusion_strategy("zero_mode", lambda _w, _m: ZeroFusion())
    fusion = build_fusion("zero_mode", weights={}, min_consensus=1)
    assert fusion.merge([]) == []
