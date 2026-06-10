"""S7-03 — multilingual context activation + powered worst-group fairness gate (DC-15).

Implements FR-038 (multilingual non-EN context feature ACTIVE in detection —
the CJK/Hangul/Arabic keywords in CONTEXT_WORDS are provably dead today:
the ``[A-Za-zÀ-ÿ]+`` tokenizer cannot produce a non-Latin token and the
containment fallthrough the tokenizer comment promises was never
implemented) and FR-039/NFR-025 (per-language fairness gap bounded + gated:
worst-group recall gap over POWERED language groups, fail-closed verdict).
Upholds NFR-004 (the power floor defines "powered"), NFR-024/AX-001
(synthetic fixtures only) and AX-002 (pure deterministic gate).

Anchors A1–A11 per the story contract
(dev-assist-artifacts/04-development/02-stories/sprint-7/
S7-03-multilingual-fairness.md). Exact-value anchors throughout — the A7
boundary uses dyadic rationals (8ths) so float arithmetic is EXACT.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pii_anon.engines.regex.confidence import (
    _NON_LATIN_CONTEXT_KW,
    _WORD_RE,
    CONTEXT_BOOST,
    CONTEXT_WORDS,
    adjust_confidence,
    has_context_words,
)
from pii_anon.eval_framework import LabeledSpan, MatchMode
from pii_anon.eval_framework.metrics.fairness_gate import (
    FairnessGateReport,
    LanguageGroupSlice,
    evaluate_language_fairness,
)
from pii_anon.eval_framework.metrics.span_metrics import _aligned_prf


# ---------------------------------------------------------------------------
# Synthetic fixtures (AX-001 — no real PII; synthetic-shaped values only)
# ---------------------------------------------------------------------------

def _spans(entity_type: str, count: int, *, width: int = 8) -> list[LabeledSpan]:
    """``count`` non-overlapping synthetic spans of ``entity_type``."""
    return [
        LabeledSpan(
            entity_type=entity_type,
            start=i * (width + 2),
            end=i * (width + 2) + width,
        )
        for i in range(count)
    ]


def _slice(
    language: str, gold_n: int, matched_n: int
) -> LanguageGroupSlice:
    """A language slice where exactly ``matched_n`` of ``gold_n`` gold spans
    are predicted (STRICT span identity) — recall is exactly matched/gold."""
    gold = _spans("PHONE_NUMBER", gold_n)
    return LanguageGroupSlice(
        language=language, gold=gold, predicted=gold[:matched_n]
    )


# ---------------------------------------------------------------------------
# A1 / A2 — FR-038: non-Latin context keywords actually fire
# ---------------------------------------------------------------------------

def test_a1_zh_phone_context_fires_and_boosts_exactly() -> None:
    """[UNIT-TEST] A1: a ZH phone-context window matches (provably False
    pre-fix — the tokenizer cannot produce a CJK token) and the confidence
    boost is exactly base + CONTEXT_BOOST."""
    assert has_context_words("PHONE_NUMBER", "请拨打电话联系") is True

    text = "请拨打电话联系 000-0000"
    start = text.index("000-0000")
    end = start + len("000-0000")
    boosted = adjust_confidence("PHONE_NUMBER", 0.5, text, start, end)
    assert boosted == min(0.99, 0.5 + CONTEXT_BOOST)


def test_a2_ja_ko_ar_zh_keywords_fire() -> None:
    """[UNIT-TEST] A2: JA/KO/AR/ZH keywords each containment-match inside a
    synthetic sentence — exact booleans."""
    assert has_context_words("EMAIL_ADDRESS", "メールを送ってください") is True
    assert has_context_words("PHONE_NUMBER", "전화 주세요") is True
    assert has_context_words("PHONE_NUMBER", "اتصل على هاتف العمل") is True
    assert has_context_words("CREDIT_CARD", "请用信用卡支付") is True
    # A non-matching CJK sentence stays False (no phone keyword present).
    assert has_context_words("PHONE_NUMBER", "今天天气很好") is False


# ---------------------------------------------------------------------------
# A3 / A4 — the Latin/ASCII token path is byte-identical; the partition is exact
# ---------------------------------------------------------------------------

def test_a3_latin_token_path_behavior_pins() -> None:
    """[UNIT-TEST] A3: exact prior-behavior pins for the token path —
    untouched by the containment pass."""
    assert has_context_words("PHONE_NUMBER", "call me") is True
    assert has_context_words("PHONE_NUMBER", "nothing here") is False
    # Accented Latin keyword still matches via the TOKEN path.
    assert has_context_words("PHONE_NUMBER", "marque el teléfono") is True
    # Unknown entity types have no keywords.
    assert has_context_words("NO_SUCH_TYPE", "call phone tel") is False
    # Underscored/log-style tokenization unchanged.
    assert has_context_words("US_SSN", "social_security_number=000") is True


def test_a4_non_latin_partition_is_exact() -> None:
    """[UNIT-TEST] A4: the containment map holds EXACTLY the keywords with
    zero Latin-alphabet runs — no Latin keyword is ever containment-matched
    (so dead-but-Latin aliases like 'e-mail' are NOT revived)."""
    expected_phone = tuple(
        sorted(["电话", "手机", "電話", "携帯", "전화", "هاتف", "جوال"])
    )
    assert _NON_LATIN_CONTEXT_KW["PHONE_NUMBER"] == expected_phone

    for entity_type, keywords in _NON_LATIN_CONTEXT_KW.items():
        for keyword in keywords:
            assert _WORD_RE.search(keyword) is None, (entity_type, keyword)

    # Completeness: every CONTEXT_WORDS keyword with no Latin run is mapped.
    for entity_type, keywords in CONTEXT_WORDS.items():
        non_latin = {kw for kw in keywords if _WORD_RE.search(kw) is None}
        mapped = set(_NON_LATIN_CONTEXT_KW.get(entity_type, ()))
        assert non_latin == mapped, entity_type


# ---------------------------------------------------------------------------
# A5 — determinism (AX-002)
# ---------------------------------------------------------------------------

def test_a5_context_and_gate_are_deterministic() -> None:
    """[UNIT-TEST] A5: 5 replays yield identical booleans, confidences and
    gate reports."""
    text = "请拨打电话联系 000-0000"
    start = text.index("000-0000")
    end = start + len("000-0000")
    slices = [_slice("en", 4, 4), _slice("es", 4, 3), _slice("zh", 2, 2)]

    results = [
        (
            has_context_words("PHONE_NUMBER", "请拨打电话联系"),
            adjust_confidence("PHONE_NUMBER", 0.5, text, start, end),
            evaluate_language_fairness(
                slices, gap_threshold=0.10, power_floor=2
            ),
        )
        for _ in range(5)
    ]
    assert all(r == results[0] for r in results[1:])


# ---------------------------------------------------------------------------
# A6 / A7 / A8 / A9 — the fairness gate (FR-039 / NFR-025 / NFR-004)
# ---------------------------------------------------------------------------

def test_a6_gate_fails_with_exact_gap_and_violators() -> None:
    """[UNIT-TEST] A6: en 4/4, es 3/4, zh 2/2 at power_floor=2 — gap exactly
    0.25, verdict FAIL, violators exactly ['es']."""
    report = evaluate_language_fairness(
        [_slice("en", 4, 4), _slice("es", 4, 3), _slice("zh", 2, 2)],
        gap_threshold=0.10,
        power_floor=2,
    )
    assert isinstance(report, FairnessGateReport)
    assert report.verdict == "FAIL"
    assert report.worst_group_recall_gap == 0.25
    assert report.per_language_recall == {"en": 1.0, "es": 0.75, "zh": 1.0}
    assert report.powered_groups == ["en", "es", "zh"]
    assert report.violating_groups == ["es"]
    assert report.n_powered == 3


def test_a7_pass_boundary_at_exact_threshold() -> None:
    """[UNIT-TEST] A7: dyadic boundary — recalls 1.0 vs 7/8 at threshold
    0.125 is a PASS (<= semantics, float-EXACT); 6/8 flips to FAIL."""
    at_boundary = evaluate_language_fairness(
        [_slice("en", 8, 8), _slice("es", 8, 7)],
        gap_threshold=0.125,
        power_floor=2,
    )
    assert at_boundary.worst_group_recall_gap == 0.125
    assert at_boundary.verdict == "PASS"
    assert at_boundary.violating_groups == []

    past_boundary = evaluate_language_fairness(
        [_slice("en", 8, 8), _slice("es", 8, 6)],
        gap_threshold=0.125,
        power_floor=2,
    )
    assert past_boundary.worst_group_recall_gap == 0.25
    assert past_boundary.verdict == "FAIL"
    assert past_boundary.violating_groups == ["es"]


def test_a8_power_filter_excludes_unpowered_groups_exactly() -> None:
    """[UNIT-TEST] A8: a 1-gold-positive language is excluded from the gap
    (NFR-004) yet reported observationally in unpowered_groups."""
    report = evaluate_language_fairness(
        [_slice("en", 4, 4), _slice("es", 4, 4), _slice("rare", 1, 0)],
        gap_threshold=0.10,
        power_floor=2,
    )
    assert report.verdict == "PASS"
    assert report.worst_group_recall_gap == 0.0
    assert report.powered_groups == ["en", "es"]
    assert report.unpowered_groups == ["rare"]
    assert report.per_language_recall["rare"] == 0.0
    assert report.n_powered == 2


def test_a9_fail_closed_insufficient_power_and_corrupt_input() -> None:
    """[SECURITY-TEST] A9: zero or one powered group can NEVER PASS —
    INSUFFICIENT_POWER; corrupt parameters are refused fail-loud."""
    zero_powered = evaluate_language_fairness(
        [_slice("en", 2, 2), _slice("es", 2, 1)],
        gap_threshold=0.10,
        power_floor=1000,
    )
    assert zero_powered.verdict == "INSUFFICIENT_POWER"
    assert zero_powered.worst_group_recall_gap is None
    assert zero_powered.powered_groups == []

    one_powered = evaluate_language_fairness(
        [_slice("en", 8, 8), _slice("rare", 1, 1)],
        gap_threshold=0.10,
        power_floor=2,
    )
    assert one_powered.verdict == "INSUFFICIENT_POWER"
    assert one_powered.powered_groups == ["en"]

    with pytest.raises(ValueError, match="power_floor"):
        evaluate_language_fairness([_slice("en", 4, 4)], power_floor=0)
    with pytest.raises(ValueError, match="gap_threshold"):
        evaluate_language_fairness(
            [_slice("en", 4, 4)], gap_threshold=1.5
        )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_language_fairness(
            [_slice("en", 4, 4), _slice("en", 4, 3)]
        )
    with pytest.raises(ValueError, match="slice"):
        evaluate_language_fairness([])


# ---------------------------------------------------------------------------
# A10 — consistency with the S1-03 recall definition
# ---------------------------------------------------------------------------

def test_a10_gate_recalls_equal_aligned_prf_recalls() -> None:
    """[CONTRACT-TEST] A10: the gate's per-language recalls EXACTLY equal
    the `_aligned_prf` STRICT recalls (one recall definition program-wide —
    the same primitive S1-03 and fairness_metrics use)."""
    slices = [_slice("en", 5, 4), _slice("es", 8, 6), _slice("zh", 3, 3)]
    report = evaluate_language_fairness(
        slices, gap_threshold=0.5, power_floor=2
    )
    for group in slices:
        _, recall, *_ = _aligned_prf(
            list(group.predicted), list(group.gold), MatchMode.STRICT
        )
        assert report.per_language_recall[group.language] == recall


# ---------------------------------------------------------------------------
# A11 — import-boundary audit
# ---------------------------------------------------------------------------

def test_a11_fairness_gate_imports_nothing_forbidden() -> None:
    """[AUDIT] A11: fairness_gate.py imports nothing from swarm/moe/fusion/
    orchestrator."""
    import ast

    import pii_anon.eval_framework.metrics.fairness_gate as gate_mod

    src = Path(gate_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"swarm", "moe", "fusion", "orchestrator"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")
            assert not (
                len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden
            ), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")
                assert not (
                    len(head) >= 2
                    and head[0] == "pii_anon"
                    and head[1] in forbidden
                ), alias.name
