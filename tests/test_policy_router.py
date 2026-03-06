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
