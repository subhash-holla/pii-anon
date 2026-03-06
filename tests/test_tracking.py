from __future__ import annotations

from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


def _run(text: str, *, mode: str, scope: str) -> dict[str, object]:
    orch = PIIOrchestrator(token_key="k")
    profile = ProcessingProfileSpec(
        profile_id="tracking-test",
        mode="weighted_consensus",
        transform_mode=mode,  # type: ignore[arg-type]
        entity_tracking_enabled=True,
        language="en",
    )
    return orch.run(
        {"text": text},
        profile=profile,
        segmentation=SegmentationPlan(enabled=True, max_tokens=12, overlap_tokens=4),
        scope=scope,
        token_version=1,
    )


def test_pseudonymize_keeps_alias_chain_consistent() -> None:
    text = (
        "Primary owner is Jack Davis. "
        "Notes reference alias Jack. "
        "Escalation email is jackdavis@example.com. "
        "Later reviewers refer to Jack during processing."
    )
    result = _run(text, mode="pseudonymize", scope="doc-1")
    link_audit = result["link_audit"]

    replacements = {
        item["replacement"]
        for item in link_audit
        if item["mention_text"] in {"Jack Davis", "Jack", "jackdavis@example.com"}
    }
    assert len(replacements) == 1


def test_anonymize_keeps_alias_chain_consistent() -> None:
    text = (
        "Primary owner is Jack Davis. "
        "Notes reference alias Jack. "
        "Escalation email is jackdavis@example.com."
    )
    result = _run(text, mode="anonymize", scope="doc-2")
    link_audit = result["link_audit"]

    replacements = {
        item["replacement"]
        for item in link_audit
        if item["mention_text"] in {"Jack Davis", "Jack", "jackdavis@example.com"}
    }
    assert replacements == {"<PERSON_NAME:anon_1>"}


def test_short_name_ambiguity_does_not_overlink() -> None:
    text = (
        "Participants include Jack Davis and Jack Miller. "
        "A follow-up says alias Jack approved changes."
    )
    result = _run(text, mode="pseudonymize", scope="doc-3")
    link_audit = result["link_audit"]

    # Two distinct full-name clusters must remain distinct.
    full_name_clusters = {
        item["cluster_id"]
        for item in link_audit
        if item["mention_text"] in {"Jack Davis", "Jack Miller"}
    }
    assert len(full_name_clusters) == 2

    # Ambiguous short-name mention should not collapse both full names into one cluster.
    short_name_items = [item for item in link_audit if item["mention_text"] == "Jack"]
    assert short_name_items
    assert short_name_items[0]["cluster_id"] not in full_name_clusters
