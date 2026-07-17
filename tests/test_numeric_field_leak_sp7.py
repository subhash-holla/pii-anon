"""sp7 panel (API lens) — numeric payload fields silently leaked PII.

Both the detection and transform loops skipped every non-string field
(``if not isinstance(value, str): continue``), so an SSN or phone number
stored as an ``int``/``float`` passed through COMPLETELY UNMASKED with zero
signal to the caller. Fix: numeric scalars (int/float, not bool) are coerced
to their string form for detection, and a numeric field WITH findings is
masked (necessarily becoming a string — masking must alter the value); a
numeric field with no findings keeps its original value and type. Strictly
ADDITIVE (detects + masks more) => leak-safe.
"""
from __future__ import annotations

import pytest

from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

_PROFILE = ProcessingProfileSpec(profile_id="p1", mode="weighted_consensus")
_SEG = SegmentationPlan(enabled=False)


@pytest.fixture(scope="module")
def orch() -> PIIOrchestrator:
    return PIIOrchestrator(token_key="k")


def _run(orch: PIIOrchestrator, payload: dict) -> dict:
    return orch.run(
        payload, profile=_PROFILE, segmentation=_SEG, scope="s", token_version=1
    )


class TestNumericFieldLeak:
    def test_int_ssn_field_is_masked(self, orch: PIIOrchestrator) -> None:
        res = _run(orch, {"ssn_number": 573337773})
        out = res["transformed_payload"]["ssn_number"]
        assert out != 573337773, "int SSN field passed through unmasked (PII leak)"
        assert "573337773" not in str(out)

    def test_int_ssn_detected(self, orch: PIIOrchestrator) -> None:
        res = _run(orch, {"ssn_number": 573337773})
        fields = {f["field_path"] for f in res["ensemble_findings"]}
        assert "ssn_number" in fields, "no finding emitted for the numeric field"

    def test_benign_numeric_field_type_preserved(self, orch: PIIOrchestrator) -> None:
        # a numeric field with NO findings keeps its original value AND type.
        res = _run(orch, {"quantity": 42, "ratio": 0.75})
        assert res["transformed_payload"]["quantity"] == 42
        assert res["transformed_payload"]["ratio"] == 0.75

    def test_bool_field_untouched(self, orch: PIIOrchestrator) -> None:
        # bool is an int subclass but is never PII — must not be coerced.
        res = _run(orch, {"active": True})
        assert res["transformed_payload"]["active"] is True

    def test_string_fields_unaffected(self, orch: PIIOrchestrator) -> None:
        res = _run(orch, {"name": "John Smith"})
        assert "John Smith" not in str(res["transformed_payload"]["name"])
