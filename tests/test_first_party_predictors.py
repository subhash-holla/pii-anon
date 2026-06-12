"""First-party predictor factories (sp2): pii-anon vanilla + swarm as BYO predictors.

The factories express pii-anon's own detection surfaces (the vanilla regex
engine and the swarm fusion pipeline) as ordinary BYO ``Predictor`` callables
emitting NATIVE entity labels — the seam external harnesses (e.g. the
pii-anon-eval-data baselines leaderboard) adapt from.
"""
from __future__ import annotations

from typing import Any

import pytest

from pii_anon.eval_framework.byo_pipeline import (
    FIRST_PARTY_SYSTEMS,
    _swarm_predictor_from_engines,
    first_party_predictor,
    pii_anon_predictor,
)
from pii_anon.engines.regex_adapter import RegexEngineAdapter


class TestFirstPartyRoster:
    def test_first_party_systems_lists_both(self) -> None:
        assert FIRST_PARTY_SYSTEMS == ("pii_anon", "pii_anon_swarm")

    def test_unknown_name_refused_fail_closed(self) -> None:
        # An incumbent name is NOT a first-party system; refusal is a
        # domain-named ValueError (mirrors incumbent_predictor).
        with pytest.raises(ValueError, match="first-party"):
            first_party_predictor("gliner")

    def test_factory_result_is_cached(self) -> None:
        assert first_party_predictor("pii_anon") is first_party_predictor("pii_anon")


class TestVanillaPredictor:
    def test_emits_native_email_span_with_exact_extent(self) -> None:
        text = "Contact alice@example.com today."
        spans = list(pii_anon_predictor(text))
        emails = [s for s in spans if s[0] == "EMAIL_ADDRESS"]
        assert emails, f"no EMAIL_ADDRESS in {spans!r}"
        _, start, end = emails[0]
        assert text[start:end] == "alice@example.com"

    def test_keeps_native_iban_label(self) -> None:
        # Native labels, NOT benchmark-canonicalized: IBAN must stay IBAN
        # (competitor_compare's normalizer folds it into BANK_ACCOUNT; the
        # first-party predictors must not).
        spans = list(pii_anon_predictor("IBAN: DE89370400440532013000"))
        assert any(s[0] == "IBAN" for s in spans), spans


class _ExplodingEngine:
    """A pool engine whose detect always fails (degraded-host stand-in)."""

    adapter_id = "exploding"

    def detect(self, payload: Any, context: Any) -> list[Any]:
        raise RuntimeError("boom")


class TestSwarmPredictor:
    def test_regex_only_pool_emits_email_via_recall_floor(self) -> None:
        # With a regex-only pool the swarm fusion still emits the regex span:
        # the FloorProjectingFusion wrapper re-injects any regex-oss span the
        # Layer-4 gate drops (FR-016/AX-003), so this is deterministic.
        predict = _swarm_predictor_from_engines([RegexEngineAdapter(enabled=True)])
        text = "Contact alice@example.com today."
        spans = list(predict(text))
        assert any(
            s[0] == "EMAIL_ADDRESS" and text[s[1] : s[2]] == "alice@example.com"
            for s in spans
        ), spans

    def test_failing_pool_engine_degrades_gracefully(self) -> None:
        # One broken engine must not take down the swarm — its findings are
        # simply absent (graceful degradation, mirroring _ensemble_detector).
        predict = _swarm_predictor_from_engines(
            [_ExplodingEngine(), RegexEngineAdapter(enabled=True)]  # type: ignore[list-item]
        )
        spans = list(predict("Contact alice@example.com today."))
        assert any(s[0] == "EMAIL_ADDRESS" for s in spans), spans

    def test_swarm_factory_registered(self) -> None:
        # Constructing the predictor must be cheap (no model loads): engine
        # constructors are lazy; heavy weights load on first call only.
        predict = first_party_predictor("pii_anon_swarm")
        assert callable(predict)
