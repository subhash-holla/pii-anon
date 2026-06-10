"""S6-01 — query-aware masking gate (RED).

Pins FR-023 (the query-aware masking gate — retain query-relevant PII, mask the
rest) + a representative FR-024 bound (over-redaction + false-retention vs the
mask-all baseline). The gate is a STANDALONE primitive (the orchestrator
router-pre-filter wire-in is Pass-2/blocked on the protected orchestrator.py).

Headline safety invariant: default-to-mask — a candidate is retained ONLY on a
positive, reason-stamped relevance signal (false-retention = a leak). All spans +
queries are SYNTHETIC (AX-001); the gate + bound are pure + deterministic (AX-002).
"""

from __future__ import annotations

import pytest

from pii_anon.policy.query_aware import (
    MaskCandidate,
    QueryAwareBoundReport,
    QueryAwareMaskingGate,
    score_query_aware_bound,
)

# --------------------------------------------------------------------------- #
# Synthetic candidates (AX-001) — detected PII spans the gate decides over.
# --------------------------------------------------------------------------- #
_EMAIL = MaskCandidate(
    entity_type="EMAIL_ADDRESS", span_start=10, span_end=30,
    surface="jane.roe@example.test", field_path="body",
)
_SSN = MaskCandidate(
    entity_type="US_SSN", span_start=40, span_end=51,
    surface="900-00-1234", field_path="body",
)
_PERSON = MaskCandidate(
    entity_type="PERSON_NAME", span_start=0, span_end=8,
    surface="Jane Roe", field_path="body",
)


def _decisions(gate: QueryAwareMaskingGate, candidates, query):
    return {d.candidate.entity_type: d for d in gate.decide(candidates, query=query)}


def test_a1_retains_query_named_entity_type() -> None:
    """[UNIT-TEST] A1: a query naming the entity type retains that candidate."""
    gate = QueryAwareMaskingGate()
    d = _decisions(gate, [_EMAIL, _SSN], query="what is the customer's email?")
    assert d["EMAIL_ADDRESS"].retain is True
    assert d["EMAIL_ADDRESS"].reason  # non-empty reason naming the entity-type match


def test_a2_masks_unrelated_entity_type() -> None:
    """[UNIT-TEST] A2: a candidate the query did not ask for is masked."""
    gate = QueryAwareMaskingGate()
    d = _decisions(gate, [_EMAIL, _SSN], query="what is the customer's email?")
    assert d["US_SSN"].retain is False


def test_a3_retains_surface_token_match() -> None:
    """[UNIT-TEST] A3: a query referencing the specific value retains it."""
    gate = QueryAwareMaskingGate()
    d = _decisions(gate, [_PERSON], query="please confirm the record for Jane Roe")
    assert d["PERSON_NAME"].retain is True


def test_a4_default_to_mask_on_blank_query_is_safe() -> None:
    """[SECURITY-TEST] A4: an empty / whitespace query masks EVERYTHING (no leak)."""
    gate = QueryAwareMaskingGate()
    for query in ("", "   ", "\t\n"):
        decisions = gate.decide([_EMAIL, _SSN, _PERSON], query=query)
        assert all(d.retain is False for d in decisions)


def test_a5_no_retention_without_a_positive_signal() -> None:
    """[SECURITY-TEST] A5: a query unrelated to any candidate retains nothing;
    every retained decision carries a non-empty reason."""
    gate = QueryAwareMaskingGate()
    unrelated = gate.decide([_EMAIL, _SSN, _PERSON], query="what time does the store open?")
    assert all(d.retain is False for d in unrelated)
    # And where retention IS justified, the reason is always stamped.
    related = gate.decide([_EMAIL], query="give me the email address")
    assert all((not d.retain) or d.reason for d in related)


def test_a6_subtractive_on_mask_property() -> None:
    """[PROPERTY-TEST] A6: the retained set ⊆ candidates with a relevance signal —
    no candidate is retained without an entity-type OR surface-token match."""
    gate = QueryAwareMaskingGate()
    candidates = [_EMAIL, _SSN, _PERSON]
    query = "confirm the email for Jane Roe"
    for d in gate.decide(candidates, query=query):
        if d.retain:
            et_match = d.candidate.entity_type.lower().replace("_", " ") in query.lower() \
                or any(tok in query.lower() for tok in d.candidate.entity_type.lower().split("_"))
            surface_match = any(
                tok and tok in query.lower() for tok in d.candidate.surface.lower().split()
            )
            assert et_match or surface_match


def test_a7_deterministic_and_order_independent() -> None:
    """[PROPERTY-TEST] A7 / AX-002: replay + candidate-permutation invariant."""
    gate = QueryAwareMaskingGate()
    q = "what is the email?"
    a = _decisions(gate, [_EMAIL, _SSN, _PERSON], q)
    b = _decisions(gate, [_PERSON, _SSN, _EMAIL], q)
    assert {k: v.retain for k, v in a.items()} == {k: v.retain for k, v in b.items()}


def test_a8_score_query_aware_bound_computes_rates() -> None:
    """[UNIT-TEST] A8: the bound report's integer-count rates match by hand; the
    mask-all baseline has false-retention 0."""
    gate = QueryAwareMaskingGate()
    candidates = [_EMAIL, _SSN, _PERSON]
    decisions = gate.decide(candidates, query="what is the email?")
    # gold: only EMAIL is query-relevant for "what is the email?".
    gold = [True, False, False]
    report = score_query_aware_bound(decisions, gold)
    assert isinstance(report, QueryAwareBoundReport)
    assert report.n_candidates == 3
    # EMAIL retained ∧ gold-relevant ⇒ no false-retention, no over-redaction here.
    assert report.false_retention_rate == pytest.approx(0.0)
    assert report.over_redaction_rate == pytest.approx(0.0)
    assert report.baseline["false_retention_rate"] == pytest.approx(0.0)
    assert report.baseline["over_redaction_rate"] > 0.0  # mask-all over-redacts the relevant span


def test_a9_false_retention_zero_when_gate_retains_only_relevant() -> None:
    """[UNIT-TEST] A9: when the gate retains only gold-relevant spans, the leak
    rate is 0 while it still beats mask-all on over-redaction."""
    gate = QueryAwareMaskingGate()
    candidates = [_EMAIL, _SSN]
    decisions = gate.decide(candidates, query="email please")
    gold = [True, False]
    report = score_query_aware_bound(decisions, gold)
    assert report.false_retention_rate == pytest.approx(0.0)
    assert report.over_redaction_rate < 1.0


def test_a10_score_bound_fails_loud_on_length_mismatch() -> None:
    """[SECURITY-TEST] A10: a decisions/gold length mismatch is refused with a
    domain-named ValueError (the S5-02/03 fail-loud discipline)."""
    gate = QueryAwareMaskingGate()
    decisions = gate.decide([_EMAIL], query="email")
    with pytest.raises(ValueError, match="length"):
        score_query_aware_bound(decisions, [True, False])


def test_a11_module_imports_nothing_forbidden() -> None:
    """[AUDIT] A11: query_aware.py imports nothing from swarm/moe/fusion/
    orchestrator (a standalone policy primitive)."""
    import ast
    from pathlib import Path

    import pii_anon.policy.query_aware as qa

    src = Path(qa.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"swarm", "moe", "fusion", "orchestrator"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")
            assert not (len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")
                assert not (len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden), alias.name
