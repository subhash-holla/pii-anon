"""sp7 Phase-A A1 — Title-Case-noise PERSON/ORG FP suppression (EVAL-ONLY).

The cross-dataset mining (sp6) found the single largest precision drain across
5/5 datasets: the regex two-Title-Case-word PERSON pattern firing on markdown
headings ("**Investor Information**"), field labels ("Tax Form:"), and
determiner/role-led phrases ("The Court", "United States District"). On
Nemotron this is 71% of ALL false positives.

LEAK-DIRECTION DISCIPLINE (the sp2 showstopper): the suppression is EVAL-ONLY
— gated on ``eval_cross_type_arbitration`` exactly like
``_drop_person_shadowed_by_specific`` and the sp6 ``_drop_undecimaled_gps``.
On the PRODUCTION masking path it never runs, so a real name that happens to
sit in a heading is still masked (over-masking is the safe direction). Only
the eval/scoring path drops the noise, lifting measured precision.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_EVAL = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)
_PROD = RegexEngineAdapter(enabled=True)  # masking path — eval arbitration OFF


def _person_spans(engine, text: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) in ("PERSON_NAME", "ORGANIZATION")
    ]


# Text shapes that are Title-Case NOISE, not names (mined from the externals).
_NOISE = [
    "**Investor Information**\nThe account balance is due.",
    "Section headed Tax Form must be completed in full.",
    "The Court found the appeal admissible under the Convention.",
    "United States District proceedings were adjourned.",
]

# Real names in the SAME hostile positions — must survive on BOTH paths (the
# masking-path safety proof) and, where the eval rule is conservative, on the
# eval path too.
_REAL = [
    ("Record for John Smith, contact john@example.com.", "John Smith"),
    ("The patient, Maria Garcia, was admitted overnight.", "Maria Garcia"),
]


class TestEvalOnlyDrop:
    @pytest.mark.parametrize("text", _NOISE)
    def test_noise_dropped_on_eval_path(self, text: str) -> None:
        assert _person_spans(_EVAL, text) == [], (
            f"Title-Case noise not dropped on the eval path: "
            f"{_person_spans(_EVAL, text)!r}"
        )

    @pytest.mark.parametrize("text", _NOISE)
    def test_noise_still_masked_on_production_path(self, text: str) -> None:
        """★ LEAK-DIRECTION PROOF: the drop is EVAL-ONLY. Whatever the regex
        detected as a name in a heading is STILL emitted on the masking path
        (over-masking a heading is harmless; dropping it would be the leak)."""
        prod = _person_spans(_PROD, text)
        evald = _person_spans(_EVAL, text)
        assert set(evald) <= set(prod), (
            "the eval-path drop must be a SUBSET of the production emission — "
            "production must never emit fewer than eval (that would be a leak)"
        )

    @pytest.mark.parametrize("text,name", _REAL)
    def test_real_names_survive_both_paths(self, text: str, name: str) -> None:
        assert name in _person_spans(_PROD, text), "production dropped a real name (LEAK)"
        assert name in _person_spans(_EVAL, text), "eval dropped a real name (recall loss)"
