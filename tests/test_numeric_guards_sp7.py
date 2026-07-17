"""sp7 Phase-A #7 — numeric-identifier guard bundle (sp6 mining candidate #7).

Precision guards for GPS/SSN/IBAN + one additive GPS recall pattern. Every
SUPPRESSOR is scoring-only (behind eval_cross_type_arbitration) so the
production masking path is UNCHANGED — the sp6 lesson (never narrow a
masking-path pattern; '41, -87' must still reach the mask). The hemisphere-GPS
pattern is ADDITIVE so GPS coverage never shrinks.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.patterns import _GPS_HEMISPHERE
from pii_anon.engines.regex_adapter import RegexEngineAdapter

_SCORE = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)
_PROD = RegexEngineAdapter(enabled=True)


def _spans(engine, text: str, etype: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == etype and f.span_start is not None
    ]


class TestGpsGuards:
    @pytest.mark.parametrize(
        "text,badspan",
        [
            ("| Subtotal | $1,125.00 | 15 |", "1,125.00"),
            ("Service Quality | 4.5/5 | reviewed", "4.5/5"),
        ],
    )
    def test_money_rating_gps_dropped_scoring_kept_prod(self, text: str, badspan: str) -> None:
        # scoring path drops the money/rating-shaped GPS FP...
        assert badspan not in _spans(_SCORE, text, "GPS_COORDINATES")
        # ...but the production masking path keeps it (leak-safe over-mask).
        assert badspan in _spans(_PROD, text, "GPS_COORDINATES")

    def test_real_coordinates_kept_on_both_paths(self) -> None:
        text = "coordinates 37.7749, -122.4194 downtown"
        assert "37.7749, -122.4194" in _spans(_SCORE, text, "GPS_COORDINATES")
        assert "37.7749, -122.4194" in _spans(_PROD, text, "GPS_COORDINATES")

    def test_hemisphere_gps_additive_both_paths(self) -> None:
        text = "The site is at 40.7234 N, 123.1235 W near the ridge."
        assert "40.7234 N, 123.1235 W" in _spans(_PROD, text, "GPS_COORDINATES")
        assert "40.7234 N, 123.1235 W" in _spans(_SCORE, text, "GPS_COORDINATES")

    def test_hemisphere_pattern_no_false_fire(self) -> None:
        # the additive hemisphere pattern must not fire without digit+hemisphere
        # structure ("24/7" is a separate base-_GPS behavior, not hemisphere).
        for t in [
            "support 24/7 for members",
            "see John N. Smith and Ed W. Poe",
            "vitamin D and E were low",
            "1 N, 2 items W left",
        ]:
            assert not _GPS_HEMISPHERE.search(t), f"hemisphere false-fire on {t!r}"


class TestSsnGuard:
    def test_glued_sequential_ssn_dropped_scoring_kept_prod(self) -> None:
        text = "Apt 59075&151=123456789&552=876543210 order"
        score_ssn = _spans(_SCORE, text, "US_SSN")
        assert "123456789" not in score_ssn and "876543210" not in score_ssn
        # production keeps them (leak-safe)
        prod_ssn = _spans(_PROD, text, "US_SSN")
        assert "123456789" in prod_ssn and "876543210" in prod_ssn

    def test_positive_context_ssn_kept_on_scoring(self) -> None:
        text = "social security number 546092103 belongs to the account"
        assert "546092103" in _spans(_SCORE, text, "US_SSN")


# NOTE: an IBAN mod-97 scoring-drop was prototyped and rejected — it regressed
# home recall (home gold IBANs are checksum-invalid synthetic data). See the
# note in regex_adapter.py.
