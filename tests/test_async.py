import pytest

from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@pytest.mark.asyncio
async def test_sync_async_equivalence_for_envelope() -> None:
    orch = PIIOrchestrator(token_key="k")
    payload = {"text": "Contact alice@example.com"}
    profile = ProcessingProfileSpec(profile_id="p1", mode="weighted_consensus")
    segmentation = SegmentationPlan(enabled=False)

    sync_result = orch.run(payload, profile=profile, segmentation=segmentation, scope="s", token_version=1)
    async_result = await orch.run_async(payload, profile=profile, segmentation=segmentation, scope="s", token_version=1)

    assert sync_result["confidence_envelope"]["risk_level"] == async_result["confidence_envelope"]["risk_level"]
    assert len(sync_result["ensemble_findings"]) == len(async_result["ensemble_findings"])
