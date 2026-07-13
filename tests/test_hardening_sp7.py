"""sp7 panel (API + code-quality lenses) — reliability & leak-branch hardening.

1. ``run_file`` error surfacing: a failed record silently VANISHED from the
   output. Now the error keeps the exception type, a warning is logged, and
   ``on_error="raise"`` re-raises for callers that need all-or-nothing.
2. Coverage for the leak-SENSITIVE branches that had none: the
   ``_COMMON_GIVEN_NAMES`` override (the ONLY guard stopping A1's Title-Case
   suppressor from dropping real names) and the eval-only drop EFFECT paths.
"""
from __future__ import annotations

import json

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

_PROFILE = ProcessingProfileSpec(profile_id="p1", mode="weighted_consensus")
_SEG = SegmentationPlan(enabled=False)


class TestRunFileErrorSurfacing:
    def _write_jsonl(self, tmp_path, rows):
        p = tmp_path / "in.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def _boom_on_marker(self, orch, monkeypatch):
        real = orch.run

        def flaky(payload, **kw):
            if any("BOOM" in str(v) for v in payload.values()):
                raise RuntimeError("engine exploded")
            return real(payload, **kw)

        monkeypatch.setattr(orch, "run", flaky)

    def test_skip_mode_records_typed_error(self, tmp_path, monkeypatch) -> None:
        orch = PIIOrchestrator(token_key="k")
        self._boom_on_marker(orch, monkeypatch)
        src = self._write_jsonl(tmp_path, [{"text": "fine"}, {"text": "BOOM"}])
        res = orch.run_file(
            src, profile=_PROFILE, segmentation=_SEG, scope="s", token_version=1
        )
        assert res.records_failed == 1
        assert res.records_processed == 1
        # the error message now carries the exception TYPE, not just str(exc)
        assert any("RuntimeError" in e for e in res.errors), res.errors

    def test_raise_mode_reraises_first_failure(self, tmp_path, monkeypatch) -> None:
        orch = PIIOrchestrator(token_key="k")
        self._boom_on_marker(orch, monkeypatch)
        src = self._write_jsonl(tmp_path, [{"text": "BOOM"}])
        with pytest.raises(RuntimeError, match="engine exploded"):
            orch.run_file(
                src, profile=_PROFILE, segmentation=_SEG, scope="s",
                token_version=1, on_error="raise",
            )


class TestLeakSensitiveBranches:
    """The eval-only drop EFFECT paths — each 'continue' branch is the exact
    code whose correctness IS the sp2 leak-direction contract."""

    _EVALP = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)
    _PRODP = RegexEngineAdapter(enabled=True)

    def _types_spans(self, engine, text):
        return {
            (str(f.entity_type), text[f.span_start:f.span_end])
            for f in engine.detect({"text": text}, {"language": "en"})
            if f.span_start is not None
        }

    def test_common_given_name_override_protects_real_name(self) -> None:
        # "Summer Johnson" — 'Summer' is a header-ish word AND a given name;
        # the override branch must KEEP the person on the scoring path.
        text = "Contact Summer Johnson for details."
        eval_persons = {s for t, s in self._types_spans(self._EVALP, text) if t == "PERSON_NAME"}
        assert any("Summer" in p for p in eval_persons), eval_persons

    def test_undecimaled_gps_drop_effect(self) -> None:
        # the GPS drop 'continue' branch actually fires on the scoring path...
        text = "grid ref 41, -87 area"
        eval_gps = {s for t, s in self._types_spans(self._EVALP, text) if t == "GPS_COORDINATES"}
        prod_gps = {s for t, s in self._types_spans(self._PRODP, text) if t == "GPS_COORDINATES"}
        assert "41, -87" not in eval_gps
        # ...while the masking path keeps the span (the sp6 floor invariant).
        assert "41, -87" in prod_gps

    def test_person_shadow_drop_effect(self) -> None:
        # a PERSON span inside an ORGANIZATION span is dropped at scoring.
        text = "The Sinop Assize Court adjourned."
        pairs = self._types_spans(self._EVALP, text)
        persons_in_org = {s for t, s in pairs if t == "PERSON_NAME" and s in "Sinop Assize Court"}
        assert not persons_in_org, pairs

    def test_dob_shadow_drop_effect(self) -> None:
        # one finding for a DOB-context date: DATE_OF_BIRTH wins, the generic
        # DATE_TIME duplicate on the same span is gone (both paths).
        text = "DOB: 12/28/1966 on file."
        for eng in (self._EVALP, self._PRODP):
            pairs = self._types_spans(eng, text)
            assert ("DATE_OF_BIRTH", "12/28/1966") in pairs
            assert ("DATE_TIME", "12/28/1966") not in pairs
