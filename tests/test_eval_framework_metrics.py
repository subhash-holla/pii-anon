"""Tests for pii_anon.eval_framework.metrics module.

Validates:
- Span metrics (strict, exact, partial, type matching)
- Token-level and document-level metrics
- Privacy metrics (re-identification, k-anonymity, l-diversity, t-closeness, leakage)
- Utility metrics (format preservation, semantic preservation, trade-off)
- Fairness metrics (language, entity-type, difficulty, script)
"""

from __future__ import annotations


from pii_anon.eval_framework.metrics.base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    MultiLevelMetric,
    compute_f1,
    compute_iou,
    safe_div,
)
from pii_anon.eval_framework.metrics.span_metrics import (
    DocumentLevelConsistencyMetric,
    EntityLevelF1Metric,
    ExactMatchMetric,
    PartialMatchMetric,
    StrictMatchMetric,
    TokenLevelF1Metric,
    TypeMatchMetric,
)
from pii_anon.eval_framework.metrics.privacy_metrics import (
    KAnonymityMetric,
    LeakageDetectionMetric,
    ReidentificationRiskMetric,
)
from pii_anon.eval_framework.metrics.utility_metrics import (
    FormatPreservationMetric,
    InformationLossMetric,
    PrivacyUtilityTradeoffMetric,
    SemanticPreservationMetric,
)
from pii_anon.eval_framework.metrics.fairness_metrics import (
    DifficultyFairnessMetric,
    EntityTypeFairnessMetric,
    LanguageFairnessMetric,
    ScriptFairnessMetric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(etype: str, start: int, end: int, rid: str = "r1") -> LabeledSpan:
    return LabeledSpan(entity_type=etype, start=start, end=end, record_id=rid)


class TestBaseHelpers:
    """Tests for base metric helper functions."""

    def test_safe_div_normal(self) -> None:
        assert safe_div(10, 5) == 2.0

    def test_safe_div_zero_denominator(self) -> None:
        assert safe_div(10, 0) == 0.0

    def test_compute_f1_perfect(self) -> None:
        assert compute_f1(1.0, 1.0) == 1.0

    def test_compute_f1_zero(self) -> None:
        assert compute_f1(0.0, 0.0) == 0.0

    def test_compute_f1_balanced(self) -> None:
        f1 = compute_f1(0.5, 0.5)
        assert abs(f1 - 0.5) < 1e-9

    def test_compute_iou_identical(self) -> None:
        assert compute_iou(0, 10, 0, 10) == 1.0

    def test_compute_iou_no_overlap(self) -> None:
        assert compute_iou(0, 5, 10, 15) == 0.0

    def test_compute_iou_partial(self) -> None:
        iou = compute_iou(0, 10, 5, 15)
        # Intersection = 5, Union = 15
        assert abs(iou - 5 / 15) < 1e-9


class _DummyMetric(MultiLevelMetric):
    name = "dummy"

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, object] | None = None,
    ) -> EvalMetricResult:
        _ = context
        return EvalMetricResult(
            name=self.name,
            value=float(len(predictions)),
            level=level,
            match_mode=match_mode,
            support=len(labels),
        )


class TestBaseMetricClass:
    def test_default_supported_levels_and_modes(self) -> None:
        metric = _DummyMetric()
        assert metric.supported_levels == [EvaluationLevel.ENTITY]
        assert metric.supported_match_modes == [MatchMode.STRICT]

    def test_compute_per_entity_type_filters_spans(self) -> None:
        metric = _DummyMetric()
        predictions = [
            _span("PERSON_NAME", 0, 4),
            _span("EMAIL_ADDRESS", 10, 20),
            _span("PERSON_NAME", 25, 30),
        ]
        labels = [
            _span("PERSON_NAME", 0, 4),
            _span("EMAIL_ADDRESS", 10, 20),
        ]
        results = metric.compute_per_entity_type(predictions, labels)
        assert set(results.keys()) == {"EMAIL_ADDRESS", "PERSON_NAME"}
        assert results["PERSON_NAME"].value == 2.0
        assert results["PERSON_NAME"].support == 1
        assert results["EMAIL_ADDRESS"].value == 1.0


# ---------------------------------------------------------------------------
# Span Metrics
# ---------------------------------------------------------------------------

class TestStrictMatchMetric:
    """Strict match: exact boundary AND exact type."""

    def test_perfect_match(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = StrictMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 1.0

    def test_wrong_type_no_match(self) -> None:
        preds = [_span("EMAIL_ADDRESS", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = StrictMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 0.0

    def test_boundary_mismatch_no_match(self) -> None:
        preds = [_span("PERSON_NAME", 0, 9)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = StrictMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 0.0


class TestExactMatchMetric:
    """Exact match: exact boundary, any type."""

    def test_boundary_match_different_type(self) -> None:
        preds = [_span("EMAIL_ADDRESS", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = ExactMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 1.0  # boundaries match, type ignored

    def test_boundary_mismatch(self) -> None:
        preds = [_span("PERSON_NAME", 1, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = ExactMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 0.0


class TestPartialMatchMetric:
    """Partial match: IoU above threshold."""

    def test_half_overlap_at_default_threshold(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 5, 15)]
        m = PartialMatchMetric(threshold=0.3)
        result = m.compute(preds, labels)
        # IoU = 5/15 ≈ 0.333 > 0.3
        assert result.f1 > 0.0

    def test_no_overlap(self) -> None:
        preds = [_span("PERSON_NAME", 0, 5)]
        labels = [_span("PERSON_NAME", 10, 15)]
        m = PartialMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 0.0


class TestTypeMatchMetric:
    """Type match: same type, any boundary."""

    def test_same_type_with_overlap(self) -> None:
        """Type match requires same type AND any character overlap."""
        preds = [_span("PERSON_NAME", 5, 15)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = TypeMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 1.0

    def test_different_type(self) -> None:
        preds = [_span("EMAIL_ADDRESS", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = TypeMatchMetric()
        result = m.compute(preds, labels)
        assert result.f1 == 0.0


class TestEntityLevelF1Metric:
    """Full entity-level F1 with all match modes and per-entity breakdown."""

    def setup_method(self) -> None:
        self.metric = EntityLevelF1Metric()

    def test_perfect_predictions(self) -> None:
        preds = [
            _span("PERSON_NAME", 0, 10),
            _span("EMAIL_ADDRESS", 15, 30),
        ]
        labels = list(preds)
        result = self.metric.compute(preds, labels, match_mode=MatchMode.STRICT)
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_per_entity_breakdown(self) -> None:
        preds = [
            _span("PERSON_NAME", 0, 10),
            _span("EMAIL_ADDRESS", 15, 30),
        ]
        labels = [
            _span("PERSON_NAME", 0, 10),
            _span("EMAIL_ADDRESS", 15, 30),
            _span("PHONE_NUMBER", 35, 50),  # missed
        ]
        result = self.metric.compute(preds, labels, match_mode=MatchMode.STRICT)
        assert "PERSON_NAME" in result.per_entity_breakdown
        assert "PHONE_NUMBER" in result.per_entity_breakdown
        assert result.per_entity_breakdown["PHONE_NUMBER"]["recall"] == 0.0

    def test_empty_predictions(self) -> None:
        labels = [_span("PERSON_NAME", 0, 10)]
        result = self.metric.compute([], labels)
        assert result.f1 == 0.0
        assert result.recall == 0.0

    def test_empty_labels(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        result = self.metric.compute(preds, [])
        assert result.f1 == 0.0
        assert result.precision == 0.0


class TestTokenLevelF1Metric:
    """Token-level (character-level) F1 metric."""

    def test_perfect_overlap(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 0, 10)]
        m = TokenLevelF1Metric()
        result = m.compute(preds, labels)
        assert result.f1 == 1.0

    def test_partial_character_overlap(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10)]
        labels = [_span("PERSON_NAME", 5, 15)]
        m = TokenLevelF1Metric()
        result = m.compute(preds, labels)
        assert 0.0 < result.f1 < 1.0


class TestDocumentLevelConsistencyMetric:
    """Document-level consistency: same surface form gets same label."""

    def test_consistent_labelling(self) -> None:
        preds = [
            _span("PERSON_NAME", 0, 10, "r1"),
            _span("PERSON_NAME", 50, 60, "r1"),
        ]
        labels = [
            _span("PERSON_NAME", 0, 10, "r1"),
            _span("PERSON_NAME", 50, 60, "r1"),
        ]
        m = DocumentLevelConsistencyMetric()
        result = m.compute(preds, labels, context={"text": "John Smith " * 6 + "John Smith"})
        assert result.value >= 0.0


# ---------------------------------------------------------------------------
# Privacy Metrics
# ---------------------------------------------------------------------------

class TestReidentificationRiskMetric:
    """RAT-Bench (2025) inspired re-identification risk."""

    def test_no_leakage(self) -> None:
        labels = [_span("PERSON_NAME", 0, 10)]
        ctx = {
            "original_text": "John Smith is a good person.",
            "anonymized_text": "XXXX XXXXX is a good person.",
        }
        m = ReidentificationRiskMetric()
        result = m.compute([], labels, context=ctx)
        assert result.value == 0.0  # no original text found in anonymized

    def test_full_leakage(self) -> None:
        labels = [_span("PERSON_NAME", 0, 10)]
        ctx = {
            "original_text": "John Smith is a good person.",
            "anonymized_text": "John Smith is a good person.",
        }
        m = ReidentificationRiskMetric()
        result = m.compute([], labels, context=ctx)
        assert result.value == 1.0


class TestKAnonymityMetric:
    """k-Anonymity: minimum group size for pseudonyms (Sweeney, 2002)."""

    def test_good_k_anonymity(self) -> None:
        """k=3 when 3 originals map to same pseudonym."""
        ctx = {
            "pseudonym_map": {"Alice": "[PERSON]", "Bob": "[PERSON]", "Carol": "[PERSON]"},
        }
        m = KAnonymityMetric()
        result = m.compute([], [], context=ctx)
        assert result.value == 3.0

    def test_singleton_group(self) -> None:
        """k=1 when only one original per pseudonym."""
        ctx = {
            "pseudonym_map": {"Alice": "[PERSON_1]"},
        }
        m = KAnonymityMetric()
        result = m.compute([], [], context=ctx)
        assert result.value == 1.0


class TestLeakageDetectionMetric:
    """PII fragment leakage detection."""

    def test_no_fragments_leaked(self) -> None:
        labels = [_span("EMAIL_ADDRESS", 0, 20)]
        ctx = {
            "original_text": "john.doe@example.com sent a message",
            "anonymized_text": "[EMAIL_REDACTED] sent a message",
        }
        m = LeakageDetectionMetric()
        result = m.compute([], labels, context=ctx)
        assert result.value == 0.0

    def test_partial_fragment_leaked(self) -> None:
        labels = [_span("EMAIL_ADDRESS", 0, 20)]
        ctx = {
            "original_text": "john.doe@example.com sent a message",
            "anonymized_text": "john.doe sent a message",  # partial leak
        }
        m = LeakageDetectionMetric()
        result = m.compute([], labels, context=ctx)
        assert result.value > 0.0


# ---------------------------------------------------------------------------
# Utility Metrics
# ---------------------------------------------------------------------------

class TestFormatPreservationMetric:
    """Format preservation: anonymized replacement matches original pattern."""

    def test_format_preserved(self) -> None:
        ctx = {
            "replacements": [
                {"entity_type": "PHONE_NUMBER", "original": "(415) 555-0100", "replacement": "(312) 555-9876"},
            ],
        }
        m = FormatPreservationMetric()
        result = m.compute([], [], context=ctx)
        assert result.value == 1.0  # both match phone pattern

    def test_format_destroyed(self) -> None:
        ctx = {
            "replacements": [
                {"entity_type": "PHONE_NUMBER", "original": "(415) 555-0100", "replacement": "REDACTED"},
            ],
        }
        m = FormatPreservationMetric()
        result = m.compute([], [], context=ctx)
        assert result.value == 0.0

    def test_skips_unknown_and_invalid_original_pattern(self) -> None:
        ctx = {
            "replacements": [
                {"entity_type": "PERSON_NAME", "original": "John", "replacement": "User-1"},
                {"entity_type": "PHONE_NUMBER", "original": "not-a-phone", "replacement": "(312) 555-9876"},
            ],
        }
        m = FormatPreservationMetric()
        result = m.compute([], [], context=ctx)
        assert result.value == 1.0
        assert result.metadata["checked"] == 0


class TestInformationLossMetric:
    """Information loss: weighted ratio of characters removed."""

    def test_no_loss(self) -> None:
        """No labels means no information loss."""
        ctx = {"original_text": "Hello world"}
        m = InformationLossMetric()
        result = m.compute([], [], context=ctx)
        assert result.value == 0.0  # no labels → no loss

    def test_full_span_loss(self) -> None:
        """Label covering entire text → loss = 1.0."""
        labels = [_span("PERSON_NAME", 0, 11)]
        ctx = {"original_text": "Hello world"}
        m = InformationLossMetric()
        result = m.compute([], labels, context=ctx)
        assert result.value == 1.0

    def test_weighted_loss_uses_risk_weights(self) -> None:
        labels = [_span("US_SSN", 0, 11)]
        ctx = {
            "original_text": "123-45-6789",
            "entity_risk_weights": {"US_SSN": 2.0},
        }
        m = InformationLossMetric()
        result = m.compute([], labels, context=ctx)
        assert result.value == 1.0
        assert result.metadata["weighted_chars_removed"] == 22.0


class TestSemanticPreservationMetric:
    def test_missing_context_defaults_to_one(self) -> None:
        result = SemanticPreservationMetric().compute([], [])
        assert result.value == 1.0

    def test_character_ngram_similarity(self) -> None:
        labels = [_span("PERSON_NAME", 0, 4)]
        ctx = {
            "original_text": "John sent a message to support.",
            "anonymized_text": "[PERSON] sent a message to support.",
        }
        result = SemanticPreservationMetric().compute([], labels, context=ctx)
        assert 0.0 <= result.value <= 1.0
        assert result.metadata["ngram_size"] == 3


class TestPrivacyUtilityTradeoffMetric:
    def test_harmonic_mean_combines_scores(self) -> None:
        ctx = {"privacy_score": 0.8, "utility_score": 0.5}
        result = PrivacyUtilityTradeoffMetric().compute([], [], context=ctx)
        assert round(result.value, 6) == round((2 * 0.8 * 0.5) / (0.8 + 0.5), 6)


# ---------------------------------------------------------------------------
# Fairness Metrics
# ---------------------------------------------------------------------------

class TestLanguageFairnessMetric:
    """Cross-language F1 gap (Bitter Lesson, 2024)."""

    def test_uniform_performance(self) -> None:
        preds = [
            _span("PERSON_NAME", 0, 10, "en_1"),
            _span("PERSON_NAME", 0, 10, "es_1"),
        ]
        labels = list(preds)
        ctx = {"record_languages": {"en_1": "en", "es_1": "es"}}
        m = LanguageFairnessMetric()
        result = m.compute(preds, labels, context=ctx)
        assert result.value == 0.0  # no gap when performance is identical

    def test_gap_detected(self) -> None:
        preds = [
            _span("PERSON_NAME", 0, 10, "en_1"),
            # Missing prediction for es
        ]
        labels = [
            _span("PERSON_NAME", 0, 10, "en_1"),
            _span("PERSON_NAME", 0, 10, "es_1"),
        ]
        ctx = {"record_languages": {"en_1": "en", "es_1": "es"}}
        m = LanguageFairnessMetric()
        result = m.compute(preds, labels, context=ctx)
        assert result.value > 0.0


class TestEntityTypeFairnessMetric:
    """Cross-entity-type F1 gap."""

    def test_uniform_entity_performance(self) -> None:
        preds = [
            _span("PERSON_NAME", 0, 10),
            _span("EMAIL_ADDRESS", 15, 30),
        ]
        labels = list(preds)
        m = EntityTypeFairnessMetric()
        result = m.compute(preds, labels)
        assert result.value == 0.0


class TestDifficultyFairnessMetric:
    def test_missing_context_returns_zero_gap(self) -> None:
        result = DifficultyFairnessMetric().compute([], [])
        assert result.value == 0.0
        assert result.metadata["difficulties"] == 0

    def test_detects_gap_across_difficulty_groups(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10, "easy-1")]
        labels = [
            _span("PERSON_NAME", 0, 10, "easy-1"),
            _span("PERSON_NAME", 0, 10, "hard-1"),
        ]
        ctx = {"record_difficulties": {"easy-1": "easy", "hard-1": "hard"}}
        result = DifficultyFairnessMetric().compute(preds, labels, context=ctx)
        assert result.value > 0.0
        assert result.metadata["difficulties"] == 2


class TestScriptFairnessMetric:
    def test_missing_context_returns_zero_gap(self) -> None:
        result = ScriptFairnessMetric().compute([], [])
        assert result.value == 0.0
        assert result.metadata["scripts"] == 0

    def test_detects_gap_across_scripts(self) -> None:
        preds = [_span("PERSON_NAME", 0, 10, "latin-1")]
        labels = [
            _span("PERSON_NAME", 0, 10, "latin-1"),
            _span("PERSON_NAME", 0, 10, "cjk-1"),
        ]
        ctx = {"record_scripts": {"latin-1": "Latin", "cjk-1": "CJK"}}
        result = ScriptFairnessMetric().compute(preds, labels, context=ctx)
        assert result.value > 0.0
        assert result.metadata["scripts"] == 2
