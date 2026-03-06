from pii_anon import PIIOrchestrator
from pii_anon.tokenization import SQLiteTokenStore
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


def test_end_to_end_detect_and_tokenize_with_persistence(tmp_path) -> None:
    store = SQLiteTokenStore(tmp_path / "tokens.db")
    orchestrator = PIIOrchestrator(token_key="key", token_store=store)

    result = orchestrator.run(
        {"text": "Reach me at alice@example.com"},
        profile=ProcessingProfileSpec(profile_id="int", mode="weighted_consensus"),
        segmentation=SegmentationPlan(enabled=False),
        scope="integration",
        token_version=1,
    )

    tokenized_text = result["transformed_payload"]["text"]
    assert "tok_" in tokenized_text
    assert len(result["fusion_audit"]) >= 1
