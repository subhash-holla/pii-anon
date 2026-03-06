from __future__ import annotations

from pii_anon import PIIOrchestrator
from pii_anon.engines import EngineAdapter
from pii_anon.policy import ExecutionPlan
from pii_anon.types import EngineFinding, ProcessingProfileSpec, Payload, SegmentationPlan


class LowConfidenceEngine(EngineAdapter):
    adapter_id = "low-confidence"

    def detect(self, payload: Payload, context: dict[str, object]) -> list[EngineFinding]:
        text = str(payload.get("text", ""))
        if "alice@example.com" not in text:
            return []
        start = text.index("alice@example.com")
        return [
            EngineFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.2,
                field_path="text",
                span_start=start,
                span_end=start + len("alice@example.com"),
                engine_id=self.adapter_id,
            )
        ]


class HighConfidenceEngine(EngineAdapter):
    adapter_id = "high-confidence"

    def detect(self, payload: Payload, context: dict[str, object]) -> list[EngineFinding]:
        text = str(payload.get("text", ""))
        if "alice@example.com" not in text:
            return []
        start = text.index("alice@example.com")
        return [
            EngineFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.99,
                field_path="text",
                span_start=start,
                span_end=start + len("alice@example.com"),
                engine_id=self.adapter_id,
            )
        ]


def test_detect_only_returns_execution_plan() -> None:
    orch = PIIOrchestrator(token_key="k")
    out = orch.detect_only(
        {"text": "alice@example.com"},
        profile=ProcessingProfileSpec(profile_id="x", use_case="short_chat", objective="speed"),
        segmentation=SegmentationPlan(enabled=False),
        scope="bench",
        token_version=1,
    )
    assert "execution_plan" in out
    assert out["execution_plan"]["plan_id"] in {"speed_guarded", "balanced_guarded", "accuracy_guarded"}


def test_plan_driven_segmentation_path(monkeypatch) -> None:
    orch = PIIOrchestrator(token_key="k")

    monkeypatch.setattr(
        orch._async.router,
        "select",
        lambda payload, profile, capabilities: ExecutionPlan(
            plan_id="segmented",
            engine_ids=["regex-oss"],
            fusion_mode="weighted_consensus",
            segmentation_enabled=True,
            escalate_on_low_confidence=False,
        ),
    )
    text = " ".join(["alice@example.com"] * 200)
    out = orch.detect_only(
        {"text": text},
        profile=ProcessingProfileSpec(profile_id="x", use_case="long_document", objective="accuracy"),
        segmentation=SegmentationPlan(enabled=False, max_tokens=40, overlap_tokens=5),
        scope="bench",
        token_version=1,
    )
    assert out["boundary_trace"] is not None
    assert out["boundary_trace"]["segments_processed"] >= 2


def test_low_confidence_escalation_adds_secondary_engine(monkeypatch) -> None:
    orch = PIIOrchestrator(token_key="k")
    orch.register_engine(LowConfidenceEngine())
    orch.register_engine(HighConfidenceEngine())

    monkeypatch.setattr(
        orch._async.router,
        "select",
        lambda payload, profile, capabilities: ExecutionPlan(
            plan_id="forced",
            engine_ids=["low-confidence"],
            fusion_mode="weighted_consensus",
            segmentation_enabled=False,
            low_confidence_threshold=0.8,
            escalate_on_low_confidence=True,
            escalation_engine_ids=["high-confidence"],
        ),
    )

    out = orch.detect_only(
        {"text": "alice@example.com"},
        profile=ProcessingProfileSpec(profile_id="x", use_case="short_chat", objective="accuracy"),
        segmentation=SegmentationPlan(enabled=False),
        scope="bench",
        token_version=1,
    )

    assert out["confidence_envelope"]["score"] > 0.2
    assert "high-confidence" in out["confidence_envelope"]["contributors"]


def test_external_competitors_disabled_removes_external_capabilities(monkeypatch) -> None:
    orch = PIIOrchestrator(token_key="k")
    observed: dict[str, object] = {}

    def fake_select(payload, profile, capabilities):
        _ = payload, profile
        observed["capabilities"] = dict(capabilities)
        return ExecutionPlan(
            plan_id="forced",
            engine_ids=["regex-oss"],
            fusion_mode="weighted_consensus",
            segmentation_enabled=False,
            escalate_on_low_confidence=False,
        )

    monkeypatch.setattr(orch._async.router, "select", fake_select)

    orch.detect_only(
        {"text": "alice@example.com"},
        profile=ProcessingProfileSpec(
            profile_id="x",
            use_case="short_chat",
            objective="balanced",
            use_external_competitors=False,
        ),
        segmentation=SegmentationPlan(enabled=False),
        scope="bench",
        token_version=1,
    )

    capabilities = observed["capabilities"]
    assert isinstance(capabilities, dict)
    assert "spacy-ner-compatible" not in capabilities
    assert "stanza-ner-compatible" not in capabilities


def test_external_allowlist_limits_external_capabilities(monkeypatch) -> None:
    orch = PIIOrchestrator(token_key="k")
    observed: dict[str, object] = {}

    def fake_select(payload, profile, capabilities):
        _ = payload, profile
        observed["capabilities"] = dict(capabilities)
        return ExecutionPlan(
            plan_id="forced",
            engine_ids=["regex-oss"],
            fusion_mode="weighted_consensus",
            segmentation_enabled=False,
            escalate_on_low_confidence=False,
        )

    monkeypatch.setattr(orch._async.router, "select", fake_select)

    orch.detect_only(
        {"text": "alice@example.com"},
        profile=ProcessingProfileSpec(
            profile_id="x",
            use_case="short_chat",
            objective="balanced",
            use_external_competitors=True,
            external_competitor_allowlist=["spacy-ner-compatible"],
        ),
        segmentation=SegmentationPlan(enabled=False),
        scope="bench",
        token_version=1,
    )

    capabilities = observed["capabilities"]
    assert isinstance(capabilities, dict)
    assert "spacy-ner-compatible" in capabilities
    assert "stanza-ner-compatible" not in capabilities
