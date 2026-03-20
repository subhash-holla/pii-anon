"""Tests for Mixtral-inspired Mixture-of-Experts ensemble system."""

import pytest

from pii_anon.moe import (
    ExpertRegistry,
    ExpertSpec,
    MoEFusionStrategy,
    MoERouter,
    build_default_registry,
)
from pii_anon.types import EngineFinding


# ═══════════════════════════════════════════════════════════════════════════
# ExpertRegistry Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_registry_register_and_get() -> None:
    """Test registering and retrieving experts."""
    registry = ExpertRegistry()
    spec = ExpertSpec(
        expert_id="test-expert",
        display_name="Test Expert",
        entity_strengths={"EMAIL_ADDRESS": 0.9},
    )
    registry.register_expert(spec)
    assert registry.get_expert("test-expert") == spec
    assert registry.get_expert("nonexistent") is None


def test_registry_unregister() -> None:
    """Test unregistering an expert."""
    registry = ExpertRegistry()
    spec = ExpertSpec(
        expert_id="test-expert",
        display_name="Test Expert",
        entity_strengths={"EMAIL_ADDRESS": 0.9},
    )
    registry.register_expert(spec)
    assert registry.get_expert("test-expert") is not None
    registry.unregister_expert("test-expert")
    assert registry.get_expert("test-expert") is None


def test_registry_unregister_nonexistent_raises() -> None:
    """Test that unregistering a nonexistent expert raises KeyError."""
    registry = ExpertRegistry()
    with pytest.raises(KeyError):
        registry.unregister_expert("nonexistent")


def test_registry_list_experts() -> None:
    """Test listing all experts."""
    registry = ExpertRegistry()
    spec1 = ExpertSpec(
        expert_id="expert1",
        display_name="Expert 1",
        entity_strengths={"EMAIL_ADDRESS": 0.9},
    )
    spec2 = ExpertSpec(
        expert_id="expert2",
        display_name="Expert 2",
        entity_strengths={"PERSON_NAME": 0.8},
    )
    registry.register_expert(spec1)
    registry.register_expert(spec2)
    experts = registry.list_experts()
    assert len(experts) == 2
    assert spec1 in experts
    assert spec2 in experts


def test_registry_available_experts() -> None:
    """Test filtering only available experts."""
    registry = ExpertRegistry()
    spec1 = ExpertSpec(
        expert_id="available",
        display_name="Available",
        entity_strengths={"EMAIL_ADDRESS": 0.9},
        is_available=True,
    )
    spec2 = ExpertSpec(
        expert_id="unavailable",
        display_name="Unavailable",
        entity_strengths={"EMAIL_ADDRESS": 0.8},
        is_available=False,
    )
    registry.register_expert(spec1)
    registry.register_expert(spec2)
    available = registry.available_experts()
    assert len(available) == 1
    assert spec1 in available
    assert spec2 not in available


def test_registry_get_experts_by_id() -> None:
    """Test retrieving multiple experts by ID."""
    registry = ExpertRegistry()
    spec1 = ExpertSpec(
        expert_id="expert1",
        display_name="Expert 1",
        entity_strengths={"EMAIL_ADDRESS": 0.9},
    )
    spec2 = ExpertSpec(
        expert_id="expert2",
        display_name="Expert 2",
        entity_strengths={"PERSON_NAME": 0.8},
    )
    registry.register_expert(spec1)
    registry.register_expert(spec2)
    experts = registry.get_experts_by_id(["expert1", "expert2", "nonexistent"])
    assert len(experts) == 2
    assert spec1 in experts
    assert spec2 in experts


# ═══════════════════════════════════════════════════════════════════════════
# MoERouter Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_router_single_entity_type() -> None:
    """Test routing for a single entity type."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    router = MoERouter(registry, top_k=3)
    routed = router.route("EMAIL_ADDRESS")
    assert len(routed) == 1
    assert routed[0][0] == "expert1"
    # Weight should be 1.0 for single expert
    assert abs(routed[0][1] - 1.0) < 0.01


def test_router_top_k_selection() -> None:
    """Test that router selects only top-K experts."""
    registry = ExpertRegistry()
    for i in range(5):
        registry.register_expert(
            ExpertSpec(
                expert_id=f"expert{i}",
                display_name=f"Expert {i}",
                entity_strengths={"EMAIL_ADDRESS": 0.5 + i * 0.1},
            )
        )
    router = MoERouter(registry, top_k=3)
    routed = router.route("EMAIL_ADDRESS")
    assert len(routed) == 3
    # Top-3 should be expert4, expert3, expert2 (highest strengths)
    ids = [eid for eid, _ in routed]
    assert "expert4" in ids
    assert "expert3" in ids
    assert "expert2" in ids


def test_router_softmax_normalization() -> None:
    """Test that router applies softmax normalization."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 1.0},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="expert2",
            display_name="Expert 2",
            entity_strengths={"EMAIL_ADDRESS": 2.0},
        )
    )
    router = MoERouter(registry, top_k=10)
    routed = router.route("EMAIL_ADDRESS")
    total_weight = sum(w for _, w in routed)
    assert abs(total_weight - 1.0) < 0.01


def test_router_performance_floor() -> None:
    """Test that performance floor ensures best expert is included."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="best",
            display_name="Best",
            entity_strengths={"EMAIL_ADDRESS": 10.0},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="medium",
            display_name="Medium",
            entity_strengths={"EMAIL_ADDRESS": 5.0},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="weak",
            display_name="Weak",
            entity_strengths={"EMAIL_ADDRESS": 1.0},
        )
    )
    router = MoERouter(registry, top_k=2, performance_floor=True)
    routed = router.route("EMAIL_ADDRESS")
    ids = [eid for eid, _ in routed]
    assert "best" in ids


def test_router_unknown_entity_type() -> None:
    """Test routing for unknown entity type returns empty (not routed)."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
            default_weight=1.5,
        )
    )
    router = MoERouter(registry, top_k=3)
    routed = router.route("UNKNOWN_TYPE")
    # Unknown entity types that no expert explicitly claims are not routed
    assert len(routed) == 0


def test_router_caching() -> None:
    """Test that router caches routing results."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    router = MoERouter(registry, top_k=3)
    routed1 = router.route("EMAIL_ADDRESS")
    routed2 = router.route("EMAIL_ADDRESS")
    assert routed1 is routed2  # Same object from cache


def test_router_clear_cache() -> None:
    """Test clearing the router cache."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    router = MoERouter(registry, top_k=3)
    routed1 = router.route("EMAIL_ADDRESS")
    router.clear_cache()
    routed2 = router.route("EMAIL_ADDRESS")
    # After clearing, should get new list objects
    assert routed1 == routed2  # Same content
    assert routed1 is not routed2  # Different objects


def test_router_route_all() -> None:
    """Test routing for all entity types."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9, "PERSON_NAME": 0.8},
        )
    )
    router = MoERouter(registry, top_k=3)
    all_routes = router.route_all()
    assert "EMAIL_ADDRESS" in all_routes
    assert "PERSON_NAME" in all_routes
    assert len(all_routes) == 2


# ═══════════════════════════════════════════════════════════════════════════
# MoEFusionStrategy Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_moe_fusion_basic() -> None:
    """Test basic MoE fusion with overlapping findings."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="expert2",
            display_name="Expert 2",
            entity_strengths={"EMAIL_ADDRESS": 0.8},
        )
    )
    fusion = MoEFusionStrategy(registry=registry, top_k=3)
    findings = [
        EngineFinding(
            "EMAIL_ADDRESS",
            0.9,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="expert1",
        ),
        EngineFinding(
            "EMAIL_ADDRESS",
            0.8,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="expert2",
        ),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    assert merged[0].entity_type == "EMAIL_ADDRESS"
    assert len(merged[0].engines) == 2


def test_moe_fusion_per_entity_routing() -> None:
    """Test that MoE routes experts per entity type."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="regex",
            display_name="Regex",
            entity_strengths={"EMAIL_ADDRESS": 0.99, "PERSON_NAME": 0.3},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="ner",
            display_name="NER",
            entity_strengths={"EMAIL_ADDRESS": 0.5, "PERSON_NAME": 0.9},
        )
    )
    fusion = MoEFusionStrategy(registry=registry, top_k=2)

    # EMAIL_ADDRESS: should prefer regex
    email_findings = [
        EngineFinding(
            "EMAIL_ADDRESS",
            0.99,
            field_path="text",
            span_start=0,
            span_end=10,
            engine_id="regex",
        ),
        EngineFinding(
            "EMAIL_ADDRESS",
            0.5,
            field_path="text",
            span_start=0,
            span_end=10,
            engine_id="ner",
        ),
    ]
    merged = fusion.merge(email_findings)
    assert len(merged) == 1
    # Regex has higher strength, should have more weight

    # PERSON_NAME: should prefer ner
    name_findings = [
        EngineFinding(
            "PERSON_NAME",
            0.3,
            field_path="text",
            span_start=11,
            span_end=20,
            engine_id="regex",
        ),
        EngineFinding(
            "PERSON_NAME",
            0.9,
            field_path="text",
            span_start=11,
            span_end=20,
            engine_id="ner",
        ),
    ]
    merged = fusion.merge(name_findings)
    assert len(merged) == 1


def test_moe_fusion_includes_non_routed_with_floor_weight() -> None:
    """Test that MoE fusion includes non-routed experts with floor weight.

    The union guarantee requires that NO finding is ever dropped.  Non-routed
    experts receive the ``min_expert_weight`` floor weight instead of being
    skipped — ensuring ensemble output ⊇ any individual expert's output.
    """
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="expert2",
            display_name="Expert 2",
            entity_strengths={"PERSON_NAME": 0.8},
        )
    )
    fusion = MoEFusionStrategy(registry=registry, top_k=1)

    # expert2 is not routed for EMAIL_ADDRESS (not in its entity_strengths)
    # but should still contribute with a floor weight (union guarantee)
    findings = [
        EngineFinding(
            "EMAIL_ADDRESS",
            0.9,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="expert1",
        ),
        EngineFinding(
            "EMAIL_ADDRESS",
            0.8,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="expert2",
        ),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    # Both experts contribute (expert2 with floor weight)
    assert sorted(merged[0].engines) == ["expert1", "expert2"]
    # Confidence is dominated by expert1 (higher routed weight)
    assert merged[0].confidence >= 0.80


def test_moe_fusion_explanation() -> None:
    """Test that MoE fusion includes explanation."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    fusion = MoEFusionStrategy(registry=registry, top_k=3)
    findings = [
        EngineFinding(
            "EMAIL_ADDRESS",
            0.9,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="expert1",
        ),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    assert "MoE routing" in (merged[0].explanation or "")


def test_moe_fusion_empty_findings() -> None:
    """Test MoE fusion with empty findings."""
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="expert1",
            display_name="Expert 1",
            entity_strengths={"EMAIL_ADDRESS": 0.9},
        )
    )
    fusion = MoEFusionStrategy(registry=registry)
    merged = fusion.merge([])
    assert merged == []


def test_moe_fusion_with_default_registry() -> None:
    """Test MoE fusion using default registry."""
    fusion = MoEFusionStrategy()
    findings = [
        EngineFinding(
            "EMAIL_ADDRESS",
            0.99,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="regex-oss",
        ),
        EngineFinding(
            "EMAIL_ADDRESS",
            0.70,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="gliner-compatible",
        ),
    ]
    merged = fusion.merge(findings)
    # Both experts are in the default registry and both have EMAIL_ADDRESS strength
    assert len(merged) == 1
    assert merged[0].entity_type == "EMAIL_ADDRESS"
    # Weighted average: regex (0.99) has higher strength in default registry
    assert "regex-oss" in merged[0].engines


def test_moe_fusion_performance_guarantee() -> None:
    """Test that MoE guarantees ensemble >= best individual expert.

    When only the best expert finds something, its score should be preserved.
    """
    registry = ExpertRegistry()
    registry.register_expert(
        ExpertSpec(
            expert_id="best",
            display_name="Best",
            entity_strengths={"EMAIL_ADDRESS": 10.0},
        )
    )
    registry.register_expert(
        ExpertSpec(
            expert_id="weak",
            display_name="Weak",
            entity_strengths={"EMAIL_ADDRESS": 1.0},
        )
    )
    fusion = MoEFusionStrategy(registry=registry, top_k=2, performance_floor=True)

    # Only best expert finds the entity
    findings = [
        EngineFinding(
            "EMAIL_ADDRESS",
            0.95,
            field_path="text",
            span_start=0,
            span_end=5,
            engine_id="best",
        ),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 1
    # Confidence should be close to the best expert's confidence
    assert merged[0].confidence >= 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Default Registry Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_build_default_registry() -> None:
    """Test building the default registry."""
    registry = build_default_registry()
    experts = registry.available_experts()
    assert len(experts) > 0


def test_default_registry_has_standard_experts() -> None:
    """Test that default registry includes standard experts."""
    registry = build_default_registry()
    expert_ids = {e.expert_id for e in registry.available_experts()}
    expected = {
        "regex-oss",
        "gliner-compatible",
        "presidio-compatible",
        "scrubadub-compatible",
        "spacy-ner-compatible",
        "stanza-ner-compatible",
    }
    assert expected.issubset(expert_ids)


def test_default_registry_regex_strengths() -> None:
    """Test that regex expert has high strengths for structured PII."""
    registry = build_default_registry()
    regex = registry.get_expert("regex-oss")
    assert regex is not None
    assert regex.entity_strengths["EMAIL_ADDRESS"] >= 0.95
    assert regex.entity_strengths["US_SSN"] >= 0.95
    assert regex.entity_strengths["CREDIT_CARD"] >= 0.95


def test_default_registry_gliner_strengths() -> None:
    """Test that GLiNER expert has high strengths for semantic PII."""
    registry = build_default_registry()
    gliner = registry.get_expert("gliner-compatible")
    assert gliner is not None
    assert gliner.entity_strengths["PERSON_NAME"] >= 0.85
    assert gliner.entity_strengths["ORGANIZATION"] >= 0.85
    assert gliner.entity_strengths["LOCATION"] >= 0.80


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_moe_integration_multi_entity_types() -> None:
    """Test MoE with multiple entity types in one batch."""
    fusion = MoEFusionStrategy()
    findings = [
        # EMAIL findings (regex dominates)
        EngineFinding(
            "EMAIL_ADDRESS",
            0.99,
            field_path="text",
            span_start=0,
            span_end=15,
            engine_id="regex-oss",
        ),
        EngineFinding(
            "EMAIL_ADDRESS",
            0.70,
            field_path="text",
            span_start=0,
            span_end=15,
            engine_id="gliner-compatible",
        ),
        # NAME findings (GLiNER dominates)
        EngineFinding(
            "PERSON_NAME",
            0.30,
            field_path="text",
            span_start=20,
            span_end=30,
            engine_id="regex-oss",
        ),
        EngineFinding(
            "PERSON_NAME",
            0.92,
            field_path="text",
            span_start=20,
            span_end=30,
            engine_id="gliner-compatible",
        ),
    ]
    merged = fusion.merge(findings)
    assert len(merged) == 2
    # Check entity types are present
    entity_types = {f.entity_type for f in merged}
    assert "EMAIL_ADDRESS" in entity_types
    assert "PERSON_NAME" in entity_types


def test_moe_strategy_id() -> None:
    """Test that MoE fusion strategy has correct ID."""
    fusion = MoEFusionStrategy()
    assert fusion.strategy_id == "mixture_of_experts"
