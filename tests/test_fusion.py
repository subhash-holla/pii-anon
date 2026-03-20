from pii_anon.fusion import (
    WeightedConsensusFusion,
    build_fusion,
    register_fusion_strategy,
)
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


# ═══════════════════════════════════════════════════════════════════════════
# WeightedConsensusFusion._get_weight() Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_weighted_consensus_get_weight_with_global_weight() -> None:
    """Test _get_weight with global engine weights only."""
    fusion = WeightedConsensusFusion(weights={"engine1": 1.5, "engine2": 0.8})
    assert fusion._get_weight("engine1", "PERSON_NAME") == 1.5
    assert fusion._get_weight("engine2", "PERSON_NAME") == 0.8
    assert fusion._get_weight("engine3", "PERSON_NAME") == 1.0  # default


def test_weighted_consensus_get_weight_with_entity_weights() -> None:
    """Test _get_weight with entity-type overrides."""
    fusion = WeightedConsensusFusion(
        weights={"engine1": 1.0, "engine2": 1.0},
        entity_weights={
            "engine1": {"PERSON_NAME": 1.8, "ORGANIZATION": 1.6},
            "engine2": {"US_SSN": 2.0}
        }
    )
    # Entity-type specific weights should override global weights
    assert fusion._get_weight("engine1", "PERSON_NAME") == 1.8
    assert fusion._get_weight("engine1", "ORGANIZATION") == 1.6
    assert fusion._get_weight("engine1", "US_SSN") == 1.0  # falls back to global
    assert fusion._get_weight("engine2", "US_SSN") == 2.0
    assert fusion._get_weight("engine2", "PERSON_NAME") == 1.0  # falls back to global


def test_weighted_consensus_get_weight_cache() -> None:
    """Test that _get_weight correctly caches results."""
    fusion = WeightedConsensusFusion(
        weights={"engine1": 1.5},
        entity_weights={"engine1": {"PERSON_NAME": 1.8}}
    )
    # Call twice, second should use cache
    w1 = fusion._get_weight("engine1", "PERSON_NAME")
    w2 = fusion._get_weight("engine1", "PERSON_NAME")
    assert w1 == w2 == 1.8
    # Verify cache was populated
    assert ("engine1", "PERSON_NAME") in fusion._weight_cache
    assert fusion._weight_cache[("engine1", "PERSON_NAME")] == 1.8


def test_weighted_consensus_get_weight_cache_different_entities() -> None:
    """Test _weight_cache with different entity types."""
    fusion = WeightedConsensusFusion(
        weights={"engine1": 1.0},
        entity_weights={"engine1": {"PERSON_NAME": 1.8, "ORGANIZATION": 1.6}}
    )
    w1 = fusion._get_weight("engine1", "PERSON_NAME")
    w2 = fusion._get_weight("engine1", "ORGANIZATION")
    assert w1 == 1.8
    assert w2 == 1.6
    assert len(fusion._weight_cache) == 2


# ═══════════════════════════════════════════════════════════════════════════
# WeightedConsensusFusion Mixture-of-Experts Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_weighted_consensus_mixture_of_experts_simple() -> None:
    """Test weighted consensus with entity-specific weights."""
    fusion = WeightedConsensusFusion(
        weights={"gliner": 1.0, "regex": 1.0},
        entity_weights={
            "gliner": {"PERSON_NAME": 1.8},  # GLiNER is good at person names
            "regex": {"US_SSN": 2.0}  # Regex is good at SSNs
        }
    )
    findings = [
        # Two engines detecting same PERSON_NAME at same location
        EngineFinding("PERSON_NAME", 0.85, field_path="text", span_start=10, span_end=20, engine_id="gliner"),
        EngineFinding("PERSON_NAME", 0.80, field_path="text", span_start=10, span_end=20, engine_id="regex"),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    # gliner: 0.85 * 1.8 = 1.53
    # regex: 0.80 * 1.0 = 0.80
    # weighted_sum = 2.33, total_weight = 2.8, confidence ≈ 0.832
    assert 0.83 <= merged[0].confidence <= 0.84


def test_weighted_consensus_mixture_of_experts_different_entities() -> None:
    """Test entity-weights with different entity types detected."""
    fusion = WeightedConsensusFusion(
        weights={"e1": 1.0, "e2": 1.0},
        entity_weights={
            "e1": {"PERSON_NAME": 1.8},
            "e2": {"US_SSN": 2.0}
        }
    )
    findings = [
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=0, span_end=10, engine_id="e1"),
        EngineFinding("US_SSN", 0.95, field_path="text", span_start=20, span_end=31, engine_id="e2"),
    ]
    merged = fusion.merge(findings)
    # Should have two separate findings (different entity types and positions)
    assert len(merged) == 2
    person_finding = [f for f in merged if f.entity_type == "PERSON_NAME"][0]
    ssn_finding = [f for f in merged if f.entity_type == "US_SSN"][0]
    # PERSON_NAME: 0.9 * 1.8 = 1.62, total_weight = 1.8, confidence = 0.9
    assert person_finding.confidence == 0.9
    # US_SSN: 0.95 * 2.0 = 1.9, total_weight = 2.0, confidence = 0.95
    assert ssn_finding.confidence == 0.95


def test_weighted_consensus_entity_weights_with_boundary_disagreement() -> None:
    """Test entity_weights when engines disagree on boundaries."""
    fusion = WeightedConsensusFusion(
        weights={"e1": 1.0, "e2": 1.0},
        entity_weights={"e1": {"US_SSN": 2.0}, "e2": {"US_SSN": 1.0}}
    )
    findings = [
        EngineFinding("US_SSN", 0.9, field_path="text", span_start=15, span_end=26, engine_id="e1"),
        EngineFinding("US_SSN", 0.85, field_path="text", span_start=15, span_end=27, engine_id="e2"),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    # Should use weighted majority voting for boundaries
    # Both have same start (15), but end differs (26 vs 27)
    # e1: weight 2.0, e2: weight 1.0
    # start_votes: {15: 3.0}
    # end_votes: {26: 2.0, 27: 1.0}
    # Best end should be 26 (higher weight)
    assert merged[0].span_start == 15
    assert merged[0].span_end == 26


def test_weighted_consensus_build_fusion_with_entity_weights() -> None:
    """Test build_fusion function with entity_weights parameter."""
    fusion = build_fusion(
        "weighted_consensus",
        weights={"e1": 1.0},
        min_consensus=1,
        entity_weights={"e1": {"PERSON_NAME": 1.8}},
        iou_threshold=0.5
    )
    assert isinstance(fusion, WeightedConsensusFusion)
    assert fusion.entity_weights == {"e1": {"PERSON_NAME": 1.8}}
    assert fusion._get_weight("e1", "PERSON_NAME") == 1.8


def test_weighted_consensus_empty_entity_weights() -> None:
    """Test that empty entity_weights defaults to 1.0 weights."""
    fusion = WeightedConsensusFusion(
        weights={"e1": 1.5},
        entity_weights={}
    )
    # Should fall back to global weight
    assert fusion._get_weight("e1", "PERSON_NAME") == 1.5


def test_weighted_consensus_none_entity_weights() -> None:
    """Test that None entity_weights initializes to empty dict."""
    fusion = WeightedConsensusFusion(
        weights={"e1": 1.5},
        entity_weights=None
    )
    assert fusion.entity_weights == {}
    assert fusion._get_weight("e1", "PERSON_NAME") == 1.5


def test_weighted_consensus_overlapping_spans_with_entity_weights() -> None:
    """Test weighted consensus handles overlapping spans with entity weights."""
    fusion = WeightedConsensusFusion(
        weights={"e1": 1.0, "e2": 1.0},
        entity_weights={"e1": {"PERSON_NAME": 2.0}}
    )
    findings = [
        # Same entity type, overlapping spans
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=10, span_end=20, engine_id="e1"),
        EngineFinding("PERSON_NAME", 0.85, field_path="text", span_start=10, span_end=21, engine_id="e2"),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    # e1 weight for PERSON_NAME is 2.0: 0.9 * 2.0 = 1.8
    # e2 weight for PERSON_NAME is 1.0: 0.85 * 1.0 = 0.85
    # total = 2.65, weighted_sum / total_weight = 2.65 / 3.0 ≈ 0.883
    assert 0.88 <= merged[0].confidence <= 0.89


def test_calibrated_majority_with_entity_weights() -> None:
    """Test calibrated majority fusion (uses weighted consensus internally)."""
    fusion = build_fusion(
        "calibrated_majority",
        weights={"e1": 1.0},
        min_consensus=2,
        entity_weights={"e1": {"PERSON_NAME": 1.5}}
    )
    findings = [
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=0, span_end=10, engine_id="e1"),
        EngineFinding("PERSON_NAME", 0.85, field_path="text", span_start=0, span_end=10, engine_id="e2"),
    ]
    merged = fusion.merge(findings)
    # Both engines found it, so should be included despite min_consensus=2
    assert len(merged) == 1
    assert merged[0].engines == ["e1", "e2"]


def test_union_high_recall_preserves_all_findings() -> None:
    """Test UnionHighRecallFusion doesn't merge any findings."""
    from pii_anon.fusion import UnionHighRecallFusion
    fusion = UnionHighRecallFusion()
    findings = [
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=0, span_end=10, engine_id="e1"),
        EngineFinding("PERSON_NAME", 0.85, field_path="text", span_start=0, span_end=10, engine_id="e2"),
        EngineFinding("PERSON_NAME", 0.80, field_path="text", span_start=5, span_end=15, engine_id="e1"),
    ]
    merged = fusion.merge(findings)
    # Should have all 3 findings (no merging)
    assert len(merged) == 3
    assert all(isinstance(f.engines, list) and len(f.engines) == 1 for f in merged)


def test_union_high_recall_via_build_fusion() -> None:
    """Test building UnionHighRecallFusion through build_fusion."""
    fusion = build_fusion(
        "union_high_recall",
        weights={},
        min_consensus=1
    )
    findings = [
        EngineFinding("EMAIL_ADDRESS", 0.9, field_path="text", span_start=0, span_end=5, engine_id="a"),
        EngineFinding("PERSON_NAME", 0.8, field_path="text", span_start=10, span_end=20, engine_id="b"),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 2


def test_intersection_consensus_requires_min_consensus() -> None:
    """Test IntersectionConsensusFusion requires min_consensus engines."""
    from pii_anon.fusion import IntersectionConsensusFusion
    fusion = IntersectionConsensusFusion(min_consensus=2)
    findings = [
        # Only one engine found this
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=0, span_end=10, engine_id="e1"),
    ]
    merged = fusion.merge(findings)
    # Should be empty (only 1 engine, needs 2)
    assert len(merged) == 0


def test_build_fusion_unknown_mode_raises_error() -> None:
    """Test build_fusion raises error for unknown fusion mode."""
    from pii_anon.errors import FusionError
    import pytest
    with pytest.raises(FusionError, match="Unknown fusion mode"):
        build_fusion(
            "nonexistent_mode",
            weights={},
            min_consensus=1
        )


def test_weighted_consensus_multiple_overlapping_clusters() -> None:
    """Test weighted consensus with multiple separate clusters."""
    fusion = WeightedConsensusFusion(weights={"e1": 1.0, "e2": 1.0})
    findings = [
        # Cluster 1: positions 0-10
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=0, span_end=10, engine_id="e1"),
        EngineFinding("PERSON_NAME", 0.85, field_path="text", span_start=2, span_end=9, engine_id="e2"),
        # Cluster 2: positions 20-30 (far apart, no overlap)
        EngineFinding("PERSON_NAME", 0.95, field_path="text", span_start=20, span_end=30, engine_id="e1"),
    ]
    merged = fusion.merge(findings)
    # Should have two separate findings
    assert len(merged) == 2
    assert merged[0].engines == ["e1", "e2"]
    assert merged[1].engines == ["e1"]


def test_weighted_consensus_non_overlapping_different_languages() -> None:
    """Test that different languages don't get merged together."""
    fusion = WeightedConsensusFusion(weights={"e1": 1.0})
    findings = [
        EngineFinding("PERSON_NAME", 0.9, field_path="text", span_start=0, span_end=10,
                     engine_id="e1", language="en"),
        EngineFinding("PERSON_NAME", 0.85, field_path="text", span_start=0, span_end=10,
                     engine_id="e1", language="es"),
    ]
    merged = fusion.merge(findings)
    # Should have two findings (different languages)
    assert len(merged) == 2
