from __future__ import annotations

from pii_anon.policy import PolicyRouter
from pii_anon.types import ProcessingProfileSpec


def _capabilities() -> dict[str, dict[str, object]]:
    return {
        "regex-oss": {
            "dependency_available": True,
            "supports_languages": ["en", "es", "fr"],
        },
        "presidio-compatible": {
            "dependency_available": True,
            "supports_languages": ["en", "es", "fr"],
        },
        "scrubadub-compatible": {
            "dependency_available": True,
            "supports_languages": ["en"],
        },
        "llm-guard-compatible": {
            "dependency_available": False,
            "supports_languages": ["en"],
        },
        "spacy-ner-compatible": {
            "dependency_available": True,
            "supports_languages": ["en"],
        },
        "stanza-ner-compatible": {
            "dependency_available": True,
            "supports_languages": ["en"],
        },
    }


def test_router_default_mode_preserves_compatibility() -> None:
    router = PolicyRouter()
    profile = ProcessingProfileSpec(profile_id="x")
    plan = router.select({"text": "hello"}, profile, _capabilities())
    assert plan.plan_id == "default_compat"
    assert "regex-oss" in plan.engine_ids
    assert plan.escalate_on_low_confidence is False


def test_router_accuracy_mode_prefers_accuracy_engines() -> None:
    router = PolicyRouter()
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="accuracy",
        language="fr",
    )
    plan = router.select({"text": "a " * 2500}, profile, _capabilities())
    assert plan.plan_id == "accuracy_guarded"
    assert "regex-oss" in plan.engine_ids
    assert "presidio-compatible" in plan.engine_ids
    assert plan.segmentation_enabled is True


def test_router_speed_mode_prefers_lightweight_path() -> None:
    router = PolicyRouter()
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="short_chat",
        objective="speed",
        language="en",
    )
    plan = router.select({"text": "Contact alice@example.com"}, profile, _capabilities())
    assert plan.plan_id == "speed_guarded"
    assert plan.engine_ids[0] == "stanza-ner-compatible"
    assert plan.fusion_mode == "union_high_recall"
    assert plan.escalate_on_low_confidence is False


def test_router_structured_form_split_profiles_map_to_accuracy_and_speed() -> None:
    router = PolicyRouter()
    accuracy = router.select(
        {"text": "Name: Jane Doe"},
        ProcessingProfileSpec(
            profile_id="x",
            use_case="structured_form_accuracy",
            objective="balanced",
            language="en",
        ),
        _capabilities(),
    )
    latency = router.select(
        {"text": "Name: Jane Doe"},
        ProcessingProfileSpec(
            profile_id="x",
            use_case="structured_form_latency",
            objective="balanced",
            language="en",
        ),
        _capabilities(),
    )

    assert accuracy.plan_id == "balanced_guarded"
    assert "presidio-compatible" in accuracy.engine_ids
    assert latency.plan_id == "balanced_guarded"
    assert latency.engine_ids == ["stanza-ner-compatible"]


# ═══════════════════════════════════════════════════════════════════════════
# PolicyRouter with Custom router_config Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_router_with_custom_ensemble_threshold() -> None:
    """Test PolicyRouter respects custom ensemble_confidence_threshold."""
    router = PolicyRouter(router_config={
        "ensemble_confidence_threshold": 0.65,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="ensemble",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    assert plan.low_confidence_threshold == 0.65


def test_router_with_custom_accuracy_threshold() -> None:
    """Test PolicyRouter respects custom accuracy_confidence_threshold."""
    router = PolicyRouter(router_config={
        "accuracy_confidence_threshold": 0.85,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="accuracy",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    assert plan.low_confidence_threshold == 0.85


def test_router_with_custom_balanced_threshold() -> None:
    """Test PolicyRouter respects custom balanced_confidence_threshold."""
    router = PolicyRouter(router_config={
        "balanced_confidence_threshold": 0.75,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="balanced",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    assert plan.low_confidence_threshold == 0.75


def test_router_with_custom_ensemble_concurrency() -> None:
    """Test PolicyRouter respects custom ensemble_concurrency_cap."""
    router = PolicyRouter(router_config={
        "ensemble_concurrency_cap": 12,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="ensemble",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    # Concurrency cap should be min(12, available engines)
    assert plan.concurrency_cap <= 12


def test_router_with_custom_accuracy_concurrency() -> None:
    """Test PolicyRouter respects custom accuracy_concurrency_cap."""
    router = PolicyRouter(router_config={
        "accuracy_concurrency_cap": 6,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="accuracy",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    assert plan.concurrency_cap <= 6


def test_router_with_custom_balanced_concurrency() -> None:
    """Test PolicyRouter respects custom balanced_concurrency_cap."""
    router = PolicyRouter(router_config={
        "balanced_concurrency_cap": 2,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="balanced",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    assert plan.concurrency_cap <= 2


def test_router_with_custom_segmentation_threshold() -> None:
    """Test PolicyRouter respects custom segmentation_token_threshold."""
    router = PolicyRouter(router_config={
        "segmentation_token_threshold": 500,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="long_document",
        objective="balanced",
        language="en",
    )
    # Document with 600 tokens (> 500 threshold), "a " * 600 = 600 words/tokens
    plan = router.select({"text": "a " * 600}, profile, _capabilities())
    assert plan.segmentation_enabled is True

    # Document with 400 tokens (< 500 threshold)
    plan2 = router.select({"text": "a " * 400}, profile, _capabilities())
    assert plan2.segmentation_enabled is False


def test_router_with_high_segmentation_threshold() -> None:
    """Test that high segmentation threshold prevents segmentation."""
    router = PolicyRouter(router_config={
        "segmentation_token_threshold": 5000,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="long_document",
        objective="balanced",
        language="en",
    )
    # Document with 600 tokens (< 5000 threshold)
    plan = router.select({"text": "a " * 600}, profile, _capabilities())
    assert plan.segmentation_enabled is False


def test_router_with_all_custom_config_values() -> None:
    """Test PolicyRouter with all router_config values customized."""
    router = PolicyRouter(router_config={
        "ensemble_confidence_threshold": 0.65,
        "accuracy_confidence_threshold": 0.85,
        "balanced_confidence_threshold": 0.75,
        "ensemble_concurrency_cap": 12,
        "accuracy_concurrency_cap": 5,
        "balanced_concurrency_cap": 2,
        "segmentation_token_threshold": 1500,
    })

    # Test ensemble objective
    plan = router.select(
        {"text": "a " * 100},
        ProcessingProfileSpec(
            profile_id="x",
            use_case="multilingual_mix",
            objective="ensemble",
            language="en",
        ),
        _capabilities(),
    )
    assert plan.low_confidence_threshold == 0.65
    assert plan.concurrency_cap <= 12

    # Test accuracy objective
    plan = router.select(
        {"text": "a " * 100},
        ProcessingProfileSpec(
            profile_id="x",
            use_case="multilingual_mix",
            objective="accuracy",
            language="en",
        ),
        _capabilities(),
    )
    assert plan.low_confidence_threshold == 0.85
    assert plan.concurrency_cap <= 5

    # Test balanced objective with long document
    plan = router.select(
        {"text": "a " * 2000},  # 2000 tokens, > 1500 threshold
        ProcessingProfileSpec(
            profile_id="x",
            use_case="long_document",
            objective="balanced",
            language="en",
        ),
        _capabilities(),
    )
    assert plan.low_confidence_threshold == 0.75
    assert plan.concurrency_cap <= 2
    assert plan.segmentation_enabled is True


def test_router_with_none_router_config() -> None:
    """Test PolicyRouter with None router_config uses defaults."""
    router = PolicyRouter(router_config=None)
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="ensemble",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    # Should use default ensemble_confidence_threshold of 0.70
    assert plan.low_confidence_threshold == 0.70


def test_router_with_empty_router_config() -> None:
    """Test PolicyRouter with empty router_config uses defaults."""
    router = PolicyRouter(router_config={})
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="balanced",
        language="en",
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    # Should use default balanced_confidence_threshold of 0.80
    assert plan.low_confidence_threshold == 0.80


def test_router_partial_config_override() -> None:
    """Test PolicyRouter with partial router_config keeps other defaults."""
    router = PolicyRouter(router_config={
        "ensemble_confidence_threshold": 0.60,
        # Other fields should use defaults
    })

    # Ensemble objective uses custom threshold
    plan1 = router.select(
        {"text": "a " * 100},
        ProcessingProfileSpec(
            profile_id="x",
            use_case="multilingual_mix",
            objective="ensemble",
            language="en",
        ),
        _capabilities(),
    )
    assert plan1.low_confidence_threshold == 0.60

    # Accuracy objective uses default threshold
    plan2 = router.select(
        {"text": "a " * 100},
        ProcessingProfileSpec(
            profile_id="x",
            use_case="multilingual_mix",
            objective="accuracy",
            language="en",
        ),
        _capabilities(),
    )
    assert plan2.low_confidence_threshold == 0.88


def test_router_default_language() -> None:
    """Test PolicyRouter defaults to 'en' when language is not specified."""
    router = PolicyRouter()
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="balanced",
        language=None,  # Not specified, should default to 'en'
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    # Should work with default language 'en'
    assert plan.plan_id == "balanced_guarded"


def test_router_multilingual_mix_enables_segmentation() -> None:
    """Test PolicyRouter enables segmentation for multilingual_mix with long text."""
    router = PolicyRouter(router_config={
        "segmentation_token_threshold": 500,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="accuracy",
        language="en",
    )
    # Document with 600 tokens (> 500 threshold)
    plan = router.select({"text": "a " * 600}, profile, _capabilities())
    assert plan.segmentation_enabled is True


def test_router_short_document_no_segmentation() -> None:
    """Test PolicyRouter disables segmentation for short documents."""
    router = PolicyRouter(router_config={
        "segmentation_token_threshold": 500,
    })
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="long_document",
        objective="balanced",
        language="en",
    )
    # Document with 200 tokens (< 500 threshold)
    plan = router.select({"text": "a " * 200}, profile, _capabilities())
    assert plan.segmentation_enabled is False


def test_router_unsupported_language_fallback() -> None:
    """Test PolicyRouter handles unsupported language gracefully."""
    router = PolicyRouter()
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="multilingual_mix",
        objective="balanced",
        language="xx",  # Unsupported language
    )
    plan = router.select({"text": "a " * 100}, profile, _capabilities())
    # Should still produce a valid plan
    assert plan.plan_id == "balanced_guarded"
    assert len(plan.engine_ids) > 0


def test_router_payload_without_text_key() -> None:
    """Test PolicyRouter handles payload with alternative string field."""
    router = PolicyRouter()
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="default",
        objective="balanced",
        language="en",
    )
    # Payload without "text" key, but with another string field
    plan = router.select({"content": "a " * 100}, profile, _capabilities())
    assert plan.plan_id == "default_compat"


def test_router_empty_payload() -> None:
    """Test PolicyRouter handles empty payload."""
    router = PolicyRouter()
    profile = ProcessingProfileSpec(
        profile_id="x",
        use_case="default",
        objective="balanced",
        language="en",
    )
    # Empty payload
    plan = router.select({}, profile, _capabilities())
    assert plan.plan_id == "default_compat"
