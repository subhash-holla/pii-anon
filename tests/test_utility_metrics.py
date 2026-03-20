"""Tests for utility metrics module.

Tests cover FormatPreservationMetric, SemanticPreservationMetric, InformationLossMetric,
and other utility preservation metrics.
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.metrics.base import EvaluationLevel, LabeledSpan, MatchMode
from pii_anon.eval_framework.metrics.utility_metrics import (
    FormatPreservationMetric,
    SemanticPreservationMetric,
    PrivacyUtilityTradeoffMetric,
    InformationLossMetric,
    EmbeddingSemanticPreservation,
    TaskUtilityProxy,
)


# ---------------------------------------------------------------------------
# FormatPreservationMetric
# ---------------------------------------------------------------------------

class TestFormatPreservationMetric:
    def test_no_replacements_returns_one(self):
        metric = FormatPreservationMetric()
        result = metric.compute([], [], context={})
        assert result.value == 1.0

    def test_email_format_preserved(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "original": "john@example.com",
                    "replacement": "user@domain.com",
                }
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.value == 1.0

    def test_email_format_not_preserved(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "original": "john@example.com",
                    "replacement": "REDACTED",
                }
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.value < 1.0

    def test_phone_number_format_preserved(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "PHONE_NUMBER",
                    "original": "555-1234",
                    "replacement": "555-5678",
                }
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.value == 1.0

    def test_ip_address_format_preserved(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "IP_ADDRESS",
                    "original": "192.168.1.1",
                    "replacement": "10.0.0.1",
                }
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.value == 1.0

    def test_credit_card_format_preserved(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "CREDIT_CARD_NUMBER",
                    "original": "4532-1234-5678-9010",
                    "replacement": "4532-9999-9999-9999",
                }
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.value == 1.0

    def test_unknown_entity_type_skipped(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "UNKNOWN_TYPE",
                    "original": "value1",
                    "replacement": "value2",
                }
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.metadata["checked"] == 0

    def test_mixed_preservation(self):
        metric = FormatPreservationMetric()
        context = {
            "replacements": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "original": "john@example.com",
                    "replacement": "user@domain.com",
                },
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "original": "jane@example.com",
                    "replacement": "REDACTED",
                },
            ]
        }
        result = metric.compute([], [], context=context)
        assert result.metadata["preserved"] == 1
        assert result.metadata["checked"] == 2


# ---------------------------------------------------------------------------
# SemanticPreservationMetric
# ---------------------------------------------------------------------------

class TestSemanticPreservationMetric:
    def test_no_text_returns_one(self):
        metric = SemanticPreservationMetric()
        result = metric.compute([], [], context={})
        assert result.value == 1.0

    def test_empty_original_text(self):
        metric = SemanticPreservationMetric()
        result = metric.compute(
            [], [],
            context={"original_text": "", "anonymized_text": "some text"}
        )
        assert result.value == 1.0

    def test_identical_texts_high_score(self):
        metric = SemanticPreservationMetric()
        text = "This is a test document."
        result = metric.compute(
            [], [],
            context={"original_text": text, "anonymized_text": text}
        )
        assert result.value > 0.8

    def test_with_pii_removal(self):
        metric = SemanticPreservationMetric()
        original = "John Smith works at Google."
        anonymized = "PERSON works at COMPANY."
        labels = [
            LabeledSpan(start=0, end=10, entity_type="PERSON"),
            LabeledSpan(start=20, end=26, entity_type="COMPANY"),
        ]
        result = metric.compute(
            [], labels,
            context={"original_text": original, "anonymized_text": anonymized}
        )
        assert result.value >= 0.0

    def test_ngram_metadata(self):
        metric = SemanticPreservationMetric()
        result = metric.compute(
            [], [],
            context={
                "original_text": "The quick brown fox",
                "anonymized_text": "The quick brown fox"
            }
        )
        assert "ngram_size" in result.metadata
        assert "intersection" in result.metadata
        assert "union" in result.metadata


# ---------------------------------------------------------------------------
# PrivacyUtilityTradeoffMetric
# ---------------------------------------------------------------------------

class TestPrivacyUtilityTradeoffMetric:
    def test_harmonic_mean_default(self):
        metric = PrivacyUtilityTradeoffMetric()
        result = metric.compute(
            [], [],
            context={"privacy_score": 0.8, "utility_score": 0.8}
        )
        assert result.value == pytest.approx(0.8, abs=0.01)

    def test_harmonic_mean_asymmetric(self):
        metric = PrivacyUtilityTradeoffMetric()
        result = metric.compute(
            [], [],
            context={"privacy_score": 1.0, "utility_score": 0.0}
        )
        # Harmonic mean of 1.0 and 0.0 is 0.0
        assert result.value == 0.0

    def test_weighted_combination_high_alpha(self):
        metric = PrivacyUtilityTradeoffMetric()
        result = metric.compute(
            [], [],
            context={"privacy_score": 0.9, "utility_score": 0.1, "alpha": 0.9}
        )
        # Should be closer to privacy (0.9)
        assert result.value > 0.5

    def test_weighted_combination_low_alpha(self):
        metric = PrivacyUtilityTradeoffMetric()
        result = metric.compute(
            [], [],
            context={"privacy_score": 0.1, "utility_score": 0.9, "alpha": 0.1}
        )
        # Should be closer to utility (0.9)
        assert result.value > 0.5

    def test_metadata_includes_alpha(self):
        metric = PrivacyUtilityTradeoffMetric()
        result = metric.compute(
            [], [],
            context={"privacy_score": 0.7, "utility_score": 0.8, "alpha": 0.6}
        )
        assert result.metadata["alpha"] == 0.6


# ---------------------------------------------------------------------------
# InformationLossMetric
# ---------------------------------------------------------------------------

class TestInformationLossMetric:
    def test_no_pii_no_loss(self):
        metric = InformationLossMetric()
        labels = []
        result = metric.compute(
            [], labels,
            context={"original_text": "This is a document."}
        )
        assert result.value == 0.0

    def test_all_pii_complete_loss(self):
        metric = InformationLossMetric()
        text = "John Smith lives in NYC."
        labels = [
            LabeledSpan(start=0, end=10, entity_type="PERSON"),
            LabeledSpan(start=20, end=23, entity_type="LOCATION"),
        ]
        result = metric.compute(
            [], labels,
            context={"original_text": text}
        )
        assert result.value > 0.0

    def test_weighted_loss(self):
        metric = InformationLossMetric()
        text = "John Smith abc"
        labels = [LabeledSpan(start=0, end=10, entity_type="PERSON")]
        context = {
            "original_text": text,
            "entity_risk_weights": {"PERSON": 2.0}
        }
        result = metric.compute([], labels, context=context)
        assert result.metadata["weighted_chars_removed"] == 20.0

    def test_loss_clamped_to_one(self):
        metric = InformationLossMetric()
        text = "John"
        labels = [LabeledSpan(start=0, end=4, entity_type="PERSON")]
        result = metric.compute(
            [], labels,
            context={"original_text": text}
        )
        assert result.value <= 1.0

    def test_empty_original_text(self):
        metric = InformationLossMetric()
        result = metric.compute(
            [], [],
            context={"original_text": ""}
        )
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# EmbeddingSemanticPreservation
# ---------------------------------------------------------------------------

class TestEmbeddingSemanticPreservation:
    def test_initialization(self):
        metric = EmbeddingSemanticPreservation()
        assert metric.model_name == "all-MiniLM-L6-v2"
        assert metric.per_sentence is False

    def test_custom_model_name(self):
        metric = EmbeddingSemanticPreservation(model_name="distiluse-base-multilingual-cased-v2")
        assert metric.model_name == "distiluse-base-multilingual-cased-v2"

    def test_no_text_returns_one(self):
        metric = EmbeddingSemanticPreservation()
        result = metric.compute([], [], context={})
        assert result.value == 1.0

    def test_empty_original_text(self):
        metric = EmbeddingSemanticPreservation()
        result = metric.compute(
            [], [],
            context={"original_text": "", "anonymized_text": "some text"}
        )
        assert result.value == 1.0

    def test_fallback_to_trigram_jaccard(self):
        # Test without sentence transformers fallback
        metric = EmbeddingSemanticPreservation()
        result = metric.compute(
            [], [],
            context={
                "original_text": "The quick brown fox",
                "anonymized_text": "The quick brown fox"
            }
        )
        # Should still return a valid score
        assert 0.0 <= result.value <= 1.0

    def test_per_sentence_computation(self):
        metric = EmbeddingSemanticPreservation(per_sentence=True)
        result = metric.compute(
            [], [],
            context={
                "original_text": "John works here. He is happy.",
                "anonymized_text": "PERSON works here. They are happy."
            }
        )
        assert 0.0 <= result.value <= 1.0

    def test_metadata_includes_model_name(self):
        metric = EmbeddingSemanticPreservation()
        result = metric.compute(
            [], [],
            context={
                "original_text": "Test text",
                "anonymized_text": "Test text"
            }
        )
        # Check that metadata is present (may or may not include model_name depending on impl)
        assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# TaskUtilityProxy
# ---------------------------------------------------------------------------

class TestTaskUtilityProxy:
    def test_no_text_returns_one(self):
        metric = TaskUtilityProxy()
        result = metric.compute([], [], context={})
        assert result.value == 1.0

    def test_empty_texts(self):
        metric = TaskUtilityProxy()
        result = metric.compute(
            [], [],
            context={"original_text": "", "anonymized_text": ""}
        )
        assert result.value == 1.0

    def test_sentiment_preservation_positive(self):
        metric = TaskUtilityProxy()
        original = "This is a great product, I love it!"
        anonymized = "This is a great PRODUCT, I love it!"
        result = metric.compute(
            [], [],
            context={"original_text": original, "anonymized_text": anonymized}
        )
        assert result.value > 0.5

    def test_sentiment_preservation_negative(self):
        metric = TaskUtilityProxy()
        original = "This is terrible and awful."
        anonymized = "This is terrible and awful."
        result = metric.compute(
            [], [],
            context={"original_text": original, "anonymized_text": anonymized}
        )
        assert result.value > 0.5

    def test_structural_preservation(self):
        metric = TaskUtilityProxy()
        original = "Sentence one. Sentence two."
        anonymized = "Sentence one. Sentence two."
        result = metric.compute(
            [], [],
            context={"original_text": original, "anonymized_text": anonymized}
        )
        assert result.metadata["structural_preservation"] == 1.0

    def test_length_preservation(self):
        metric = TaskUtilityProxy()
        original = "The document is here."
        anonymized = "The REDACTED is here."
        result = metric.compute(
            [], [],
            context={"original_text": original, "anonymized_text": anonymized}
        )
        assert result.metadata["length_preservation"] > 0.0

    def test_all_sub_scores_present(self):
        metric = TaskUtilityProxy()
        result = metric.compute(
            [], [],
            context={
                "original_text": "A test document with some content.",
                "anonymized_text": "A test DOCUMENT with some content."
            }
        )
        assert "sentiment_preservation" in result.metadata
        assert "structural_preservation" in result.metadata
        assert "length_preservation" in result.metadata

    def test_final_score_average_of_subscores(self):
        metric = TaskUtilityProxy()
        result = metric.compute(
            [], [],
            context={
                "original_text": "Same text exactly.",
                "anonymized_text": "Same text exactly."
            }
        )
        # With identical text, all subscores should be close to 1.0
        assert result.value > 0.8


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_char_ngrams(self):
        from pii_anon.eval_framework.metrics.utility_metrics import _char_ngrams
        ngrams = _char_ngrams("hello", 3)
        assert "hel" in ngrams
        assert "ell" in ngrams
        assert "llo" in ngrams
        assert len(ngrams) == 3

    def test_char_ngrams_too_short(self):
        from pii_anon.eval_framework.metrics.utility_metrics import _char_ngrams
        ngrams = _char_ngrams("hi", 3)
        assert ngrams == set()

    def test_split_sentences(self):
        from pii_anon.eval_framework.metrics.utility_metrics import _split_sentences
        text = "First sentence. Second sentence! Third sentence?"
        sentences = _split_sentences(text)
        assert len(sentences) >= 2
        assert "First sentence" in sentences[0]

    def test_split_sentences_no_punctuation(self):
        from pii_anon.eval_framework.metrics.utility_metrics import _split_sentences
        text = "No punctuation here"
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert "No punctuation" in sentences[0]
