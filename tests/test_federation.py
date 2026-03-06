from pii_anon import PIIOrchestrator
from pii_anon.engines import EngineAdapter
from pii_anon.types import EngineFinding, ProcessingProfileSpec, Payload, SegmentationPlan


class AlwaysFailEngine(EngineAdapter):
    adapter_id = "always-fail"

    def detect(self, payload: Payload, context: dict[str, object]) -> list[EngineFinding]:
        raise RuntimeError("boom")


class NameEngine(EngineAdapter):
    adapter_id = "name-engine"

    def detect(self, payload: Payload, context: dict[str, object]) -> list[EngineFinding]:
        text = str(payload.get("text", ""))
        if "Alice" not in text:
            return []
        start = text.index("Alice")
        return [
            EngineFinding(
                entity_type="PERSON_NAME",
                confidence=0.8,
                field_path="text",
                span_start=start,
                span_end=start + 5,
                engine_id=self.adapter_id,
            )
        ]


def test_orchestrator_tokenizes_and_reports_envelope_and_audit() -> None:
    orch = PIIOrchestrator(token_key="k")
    result = orch.run(
        {"email": "alice@example.com"},
        profile=ProcessingProfileSpec(profile_id="p1", mode="weighted_consensus"),
        segmentation=SegmentationPlan(enabled=False),
        scope="claims",
        token_version=1,
    )
    # PERSON and EMAIL aliases are intentionally unified for continuity in v1.0.1.
    assert "<PERSON_NAME:v1:tok_" in result["transformed_payload"]["email"]
    assert "confidence_envelope" in result
    assert "fusion_audit" in result


def test_segmentation_generates_boundary_trace() -> None:
    orch = PIIOrchestrator(token_key="k")
    text = " ".join(["alice@example.com"] * 200)
    result = orch.run(
        {"text": text},
        profile=ProcessingProfileSpec(profile_id="p1", mode="union_high_recall"),
        segmentation=SegmentationPlan(enabled=True, max_tokens=50, overlap_tokens=10),
        scope="stream",
        token_version=2,
    )
    assert result["boundary_trace"] is not None
    assert result["boundary_trace"]["segments_processed"] >= 2


def test_runtime_engine_registration_and_isolation() -> None:
    orch = PIIOrchestrator(token_key="k")
    orch.register_engine(NameEngine())
    orch.register_engine(AlwaysFailEngine())

    result = orch.run(
        {"text": "Alice alice@example.com"},
        profile=ProcessingProfileSpec(profile_id="p2", mode="union_high_recall"),
        segmentation=SegmentationPlan(enabled=False),
        scope="test",
        token_version=1,
    )

    entity_types = [item["entity_type"] for item in result["ensemble_findings"]]
    assert "PERSON_NAME" in entity_types
    assert "EMAIL_ADDRESS" in entity_types


def test_stream_processing_returns_all_records() -> None:
    orch = PIIOrchestrator(token_key="k")
    payloads = [{"text": "alice@example.com"}, {"text": "123-45-6789"}]
    out = list(
        orch.run_stream(
            payloads,
            profile=ProcessingProfileSpec(profile_id="stream", mode="weighted_consensus"),
            segmentation=SegmentationPlan(enabled=False),
            scope="s",
            token_version=1,
        )
    )
    assert len(out) == 2


def test_detect_only_matches_run_findings_and_audit() -> None:
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(
        profile_id="detect-only",
        mode="weighted_consensus",
        use_case="short_chat",
        objective="balanced",
    )
    segmentation = SegmentationPlan(enabled=False)

    detect = orch.detect_only(
        {"text": "Dr Smith alice@example.com 123-45-6789"},
        profile=profile,
        segmentation=segmentation,
        scope="bench",
        token_version=1,
    )
    run = orch.run(
        {"text": "Dr Smith alice@example.com 123-45-6789"},
        profile=profile,
        segmentation=segmentation,
        scope="bench",
        token_version=1,
    )

    assert detect["ensemble_findings"] == run["ensemble_findings"]
    assert detect["fusion_audit"] == run["fusion_audit"]
    assert detect["confidence_envelope"] == run["confidence_envelope"]
