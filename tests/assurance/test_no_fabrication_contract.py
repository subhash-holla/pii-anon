"""Contract test: the public no-fabrication validators behave IDENTICALLY to the
private originals in ``competitive_supremacy.py``, so the two copies cannot drift.

Also pins the fabrication-rejection vectors directly (so the public module's
guarantees are tested in their own right, not only by transitive equivalence).

NOTE: the battery is iterated INSIDE each test rather than via ``parametrize`` —
pytest stringifies parametrize values to build test IDs, and ``str(10**5000)``
hits the CPython int→str digit limit (the exact crash class these validators
defend against). ``safe_repr`` is used for any failure label.
"""

from __future__ import annotations

from pii_anon.eval_framework.evaluation import competitive_supremacy as cs
from pii_anon.eval_framework.validation import (
    detected_entity_names,
    finite_unit_score,
    is_finite_number,
    is_nonblank_str,
    safe_names,
    safe_repr,
)

# A battery spanning every adversarial class the SDO close found.
_SCALARS = [
    True, False, 0, 1, -1, 2, 0.0, 1.0, 0.5, -0.0, -0.1, 1.1, 0.999999,
    float("nan"), float("inf"), -float("inf"),
    10**400, 10**5000, -(10**400), 1e308, 1e309,
    "x", "", " ", "  ", "\t\n", "  abc  ", "0.5", "0",
    None, [], {}, [1], {"a": 1}, object(),
]


def test_is_finite_number_matches_private() -> None:
    for v in _SCALARS:
        assert is_finite_number(v) == cs._is_finite_number(v), safe_repr(v)


def test_finite_unit_score_matches_private() -> None:
    for v in _SCALARS:
        assert finite_unit_score(v) == cs._finite_unit_score(v), safe_repr(v)


def test_is_nonblank_str_matches_private() -> None:
    for v in _SCALARS:
        assert is_nonblank_str(v) == cs._is_nonblank_str(v), safe_repr(v)


def test_safe_repr_matches_private() -> None:
    for v in _SCALARS:
        assert safe_repr(v) == cs._safe_repr(v)


def test_safe_names_matches_private() -> None:
    items = {"b", "a", "c"}
    assert safe_names(items) == cs._safe_names(items)
    big = {10**5000, "z", 1}
    assert safe_names(big) == cs._safe_names(big)


def test_detected_entity_names_matches_private() -> None:
    recall = {
        "EMAIL": 0.9, "PHONE": 0.0, "SSN": -0.1, "NAME": 1.0,
        "FAKE": True, "INF": float("inf"), "HUGE": 10**400, "OVER": 1.5,
        "NAN": float("nan"),
    }
    assert detected_entity_names(recall) == cs._detected_entity_names(recall)


# --- direct guarantees (not just transitive) -------------------------------


def test_finite_unit_score_rejects_fabrication_vectors() -> None:
    for bad in (True, False, float("nan"), float("inf"), -float("inf"), -0.1, 1.1, 10**400, "0.5", None):
        assert finite_unit_score(bad) is None
    for good in (0.0, 1.0, 0.5):
        assert finite_unit_score(good) == good


def test_is_finite_number_rejects_bool_and_nonfinite() -> None:
    assert is_finite_number(True) is False
    assert is_finite_number(False) is False
    assert is_finite_number(float("nan")) is False
    assert is_finite_number(10**5000) is False  # int too large to convert
    assert is_finite_number(3) is True
    assert is_finite_number(2.5) is True


def test_is_nonblank_str_rejects_truthy_nonstrings() -> None:
    assert is_nonblank_str("   ") is False
    assert is_nonblank_str("") is False
    assert is_nonblank_str(1) is False
    assert is_nonblank_str(True) is False
    assert is_nonblank_str(["a"]) is False
    assert is_nonblank_str("git-sha-abc") is True


def test_safe_repr_never_crashes_on_huge_int() -> None:
    # repr(10**5000) raises ValueError (int->str digit limit) -> the except branch.
    out = safe_repr(10**5000)
    assert isinstance(out, str) and "unrepresentable int" in out


def test_safe_repr_truncates_long_honest_repr() -> None:
    out = safe_repr("a" * 500, limit=120)
    assert out.endswith("…(truncated)") and len(out) <= 120 + len("…(truncated)")


def test_detected_entity_names_excludes_zero_negative_and_nonphysical() -> None:
    recall = {
        "real": 0.7,
        "perfect": 1.0,
        "zero": 0.0,
        "neg": -0.3,
        "bool": True,
        "inf": float("inf"),
        "over": 2.0,
    }
    assert detected_entity_names(recall) == {"real", "perfect"}
