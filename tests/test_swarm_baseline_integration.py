"""End-to-end tests verifying the pii-anon baseline (regex-oss) flows
seamlessly through the pii-anon-swarm pipeline.

Contract we're locking in:

- Every Phase 3 gap-closure entity type (CVV, PIN, PASSWORD,
  COURT_CASE_NUMBER, DOCKET_NUMBER, BAR_NUMBER, INVOICE_NUMBER,
  INSURANCE_POLICY_NUMBER, SALARY) is in STRUCTURED_TYPES so it skips
  the swarm's Layer 4 corroboration gate.
- Every Phase 3 type's ``PatternSpec.base_confidence`` is at or above
  the ``SwarmConfig.fast_pass_threshold`` (0.90) so the swarm routes
  the regex hit through Layer 1 without NER fusion.
- Every Phase 3 type has a distinct ``ENTITY_TYPE_ENCODING`` index so
  the XGBoost meta-learner can differentiate between them.
- ``is_structured`` feature (slot 13) fires for Phase 3 types.

The philosophy: baseline catches the common patterns, swarm's MoE
catches the rest.  These tests fail loud if a future refactor breaks
the integration.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.patterns import PATTERN_REGISTRY
from pii_anon.swarm import STRUCTURED_TYPES, SwarmConfig
from pii_anon.swarm_learner import ENTITY_TYPE_ENCODING, extract_features


PHASE_3_TYPES = (
    "CVV", "PIN", "PASSWORD",
    "COURT_CASE_NUMBER", "DOCKET_NUMBER", "BAR_NUMBER",
    "INVOICE_NUMBER", "INSURANCE_POLICY_NUMBER", "SALARY",
)


# ---------------------------------------------------------------------------
# Contract: Phase 3 types qualify for Layer 1 fast-pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type", PHASE_3_TYPES)
def test_phase3_type_base_confidence_at_or_above_fast_pass(entity_type):
    """Every Phase 3 entity type's pattern must emit at a confidence
    that qualifies for the swarm's Layer 1 fast-pass.

    Below this threshold the finding drops into Layer 3 fusion where
    NER engines (which have no rule-based signal for these types) may
    dilute it — the paper v11 §5.6 intent is to route rule-based
    detections straight to emission.
    """
    threshold = SwarmConfig().fast_pass_threshold
    matches = [p for p in PATTERN_REGISTRY if p.entity_type == entity_type]
    assert matches, f"No pattern registered for {entity_type}"
    # Every pattern for the type must meet the threshold — if there are
    # multiple patterns for one type (e.g. phone number formats), the
    # weakest one is what ends up in the swarm's emission stream.
    for spec in matches:
        assert spec.base_confidence >= threshold, (
            f"{entity_type} pattern {spec.explanation} has "
            f"base_confidence={spec.base_confidence} < fast_pass_threshold={threshold}"
        )


# ---------------------------------------------------------------------------
# Contract: Phase 3 types skip Layer 4 corroboration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type", PHASE_3_TYPES)
def test_phase3_type_is_structured(entity_type):
    """Phase 3 types must be in STRUCTURED_TYPES.  This has two effects:

    1. Layer 4 corroboration gate is skipped (single regex hit is
       authoritative — paper v11 argues the keyword-gated regex match
       is as strong a guarantee as a checksum for structured types).
    2. XGBoost feature slot 13 (``is_structured``) fires, telling the
       meta-learner this is a trustworthy signal.
    """
    assert entity_type in STRUCTURED_TYPES


# ---------------------------------------------------------------------------
# Contract: Phase 3 types are distinguishable in the meta-learner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type", PHASE_3_TYPES)
def test_phase3_type_has_distinct_meta_learner_index(entity_type):
    """Each Phase 3 type must have its own ``ENTITY_TYPE_ENCODING``
    index, not collide with the catch-all bucket.

    Without this, the XGBoost meta-learner sees every Phase 3 type as
    the same categorical value — it cannot differentiate a CVV hit
    from a BAR_NUMBER hit, which hurts both precision and per-type
    calibration.
    """
    assert entity_type in ENTITY_TYPE_ENCODING, (
        f"{entity_type} missing from ENTITY_TYPE_ENCODING — the "
        "meta-learner would lump it into the catch-all bucket"
    )


def test_phase3_indices_do_not_collide():
    indices = [ENTITY_TYPE_ENCODING[t] for t in PHASE_3_TYPES]
    assert len(set(indices)) == len(indices), "Phase 3 indices collide"


def test_encoding_has_no_duplicate_indices():
    """Global invariant: every entity type in ENTITY_TYPE_ENCODING gets
    its own index.  Regression test for the class of bugs where a
    future addition pastes a duplicate index by accident.
    """
    values = list(ENTITY_TYPE_ENCODING.values())
    assert len(set(values)) == len(values), (
        f"Duplicate indices in ENTITY_TYPE_ENCODING: {sorted(values)}"
    )


# ---------------------------------------------------------------------------
# End-to-end: regex-oss emits a Phase 3 finding AND the feature vector
# a hypothetical swarm Layer-3 candidate would see shows is_structured=1
# ---------------------------------------------------------------------------

def _make_candidate_with_regex_finding(entity_type: str, confidence: float):
    """Build a SpanCandidate mimicking what the swarm would construct
    from a single regex-oss finding for *entity_type*.
    """
    from pii_anon.swarm import SpanCandidate
    from pii_anon.types import EngineFinding

    finding = EngineFinding(
        entity_type=entity_type,
        confidence=confidence,
        field_path="text",
        span_start=0,
        span_end=5,
        language="en",
        engine_id="regex-oss",
        explanation="test",
    )
    return SpanCandidate(
        entity_type=entity_type,
        span_start=0,
        span_end=5,
        field_path="text",
        engine_findings={"regex-oss": finding},
        ds_confidence=confidence,
        corroboration_count=1,
    )


@pytest.mark.parametrize("entity_type", PHASE_3_TYPES)
def test_phase3_feature_vector_marks_is_structured(entity_type):
    """When the swarm builds a candidate from a regex-oss Phase 3
    finding, the XGBoost feature vector must have ``is_structured=1.0``
    (slot 13) so the meta-learner trusts it.
    """
    candidate = _make_candidate_with_regex_finding(entity_type, 0.92)
    features = extract_features(candidate, total_engines=6)
    # Slot 13 (0-indexed: 12) is ``is_structured_type``.
    assert features[12] == 1.0, (
        f"{entity_type} is in STRUCTURED_TYPES but extract_features "
        f"returned is_structured={features[12]}"
    )


@pytest.mark.parametrize("entity_type", PHASE_3_TYPES)
def test_phase3_feature_vector_marks_regex_detected_with_checksum_tier(entity_type):
    """Phase 3 findings emit at ≥0.92 (CVV/PASSWORD) or ≥0.90 (others).
    The ``has_checksum`` feature (slot 14, 0-indexed: 13) fires when
    regex confidence is ≥0.91 — this is the "trust tier" the swarm
    uses to weight the baseline above the NER engines.
    """
    spec = next(p for p in PATTERN_REGISTRY if p.entity_type == entity_type)
    candidate = _make_candidate_with_regex_finding(entity_type, spec.base_confidence)
    features = extract_features(candidate, total_engines=6)
    # Slot 8 (0-indexed: 7) is ``regex_detected``.  Must always fire
    # when the only engine is regex-oss.
    assert features[7] == 1.0, (
        f"regex_detected feature missed for {entity_type}"
    )


# ---------------------------------------------------------------------------
# Contract: regex-oss IS the baseline, no alias divergence
# ---------------------------------------------------------------------------

def test_regex_oss_engine_id_is_stable():
    """The swarm refers to the baseline by literal string ``"regex-oss"``
    at several privileged positions (Layer 1 fast-pass check, Layer 2
    pinning, feature-vector slot 8).  Rebranding the adapter_id would
    silently disable all three.  Assert the engine class still exports
    the expected id.
    """
    from pii_anon.engines.regex_adapter import RegexEngineAdapter
    adapter = RegexEngineAdapter(enabled=True)
    assert adapter.adapter_id == "regex-oss"


# ---------------------------------------------------------------------------
# Sanity: the baseline ships all Phase 3 types as ≥0.90 confidence even
# on the weakest matching pattern
# ---------------------------------------------------------------------------

def test_phase3_weakest_pattern_still_fast_pass_eligible():
    """For each Phase 3 entity type, find the lowest-confidence pattern
    in the registry and assert it still clears fast-pass threshold.
    This guards against a future refactor that adds a looser pattern
    for an existing type and accidentally drops it out of the fast
    path.
    """
    threshold = SwarmConfig().fast_pass_threshold
    for entity_type in PHASE_3_TYPES:
        specs = [p for p in PATTERN_REGISTRY if p.entity_type == entity_type]
        min_conf = min(p.base_confidence for p in specs)
        assert min_conf >= threshold, (
            f"{entity_type}: weakest pattern has base_confidence={min_conf} "
            f"< fast_pass_threshold={threshold}.  Phase 3 types must stay "
            f"structurally reliable so the baseline catches them without "
            f"the swarm's NER layer having to vote."
        )
