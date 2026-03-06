"""Privacy-utility trade-off metrics for anonymized / pseudonymized text.

Evidence basis:
- TAB (Text Anonymization Benchmark, Lison et al. 2022):
  privacy-utility evaluation methodology for text anonymization
- Information-loss measurement frameworks from SDC Practice
"""

from __future__ import annotations

import re
from typing import Any

from .base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    MultiLevelMetric,
    safe_div,
)

# Check for sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False


class FormatPreservationMetric(MultiLevelMetric):
    """Measures structural format retention after anonymization.

    For each entity type with a known format pattern (phone numbers,
    emails, credit cards, etc.), checks whether the anonymized
    replacement preserves the same structural pattern.

    ``context`` must include ``original_text`` and ``anonymized_text``.

    References: TAB 2022 utility evaluation.
    """

    name = "format_preservation"

    # Structural signature patterns — entity-type to regex
    _FORMAT_PATTERNS: dict[str, re.Pattern[str]] = {
        "EMAIL_ADDRESS": re.compile(r"[^@]+@[^@]+\.[^@]+"),
        "PHONE_NUMBER": re.compile(r"[\d\s\-\(\)\+]{7,}"),
        "CREDIT_CARD_NUMBER": re.compile(r"[\d\s\-]{13,19}"),
        "IP_ADDRESS": re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
        "US_SSN": re.compile(r"\d{3}[\-\s]?\d{2}[\-\s]?\d{4}"),
        "DATE_OF_BIRTH": re.compile(r"\d{4}[\-/]\d{2}[\-/]\d{2}|\d{2}[\-/]\d{2}[\-/]\d{4}"),
        "IBAN": re.compile(r"[A-Z]{2}\d{2}[\s]?[\dA-Z\s]{10,30}"),
        "MAC_ADDRESS": re.compile(r"([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}"),
    }

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY, EvaluationLevel.DOCUMENT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        replacements: list[dict[str, Any]] = ctx.get("replacements", [])

        if not replacements:
            return EvalMetricResult(
                name=self.name, value=1.0, level=level, match_mode=match_mode,
                metadata={"preserved": 0, "checked": 0},
            )

        preserved = 0
        checked = 0
        for repl in replacements:
            entity_type = str(repl.get("entity_type", ""))
            original_value = str(repl.get("original", ""))
            replacement_value = str(repl.get("replacement", ""))
            pattern = self._FORMAT_PATTERNS.get(entity_type)
            if pattern is None:
                continue
            checked += 1
            orig_matches = bool(pattern.fullmatch(original_value.strip()))
            repl_matches = bool(pattern.fullmatch(replacement_value.strip()))
            if orig_matches and repl_matches:
                preserved += 1
            elif not orig_matches:
                # Original didn't match pattern, skip
                checked -= 1

        score = safe_div(preserved, checked) if checked > 0 else 1.0
        return EvalMetricResult(
            name=self.name, value=round(score, 6), level=level, match_mode=match_mode,
            metadata={"preserved": preserved, "checked": checked},
        )


class SemanticPreservationMetric(MultiLevelMetric):
    """Measures semantic meaning retention via normalised edit distance.

    Computes the character-level Levenshtein ratio between the original
    and anonymized text *excluding* the PII spans.  A ratio close to 1.0
    means the non-PII context is well-preserved.

    ``context`` must include ``original_text`` and ``anonymized_text``.

    References: TAB 2022 utility evaluation.
    """

    name = "semantic_preservation"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.DOCUMENT,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        original_text: str = ctx.get("original_text", "")
        anonymized_text: str = ctx.get("anonymized_text", "")
        if not original_text or not anonymized_text:
            return EvalMetricResult(name=self.name, value=1.0, level=level, match_mode=match_mode)

        # Strip out PII spans from original to get non-PII "skeleton"
        sorted_labels = sorted(labels, key=lambda s: s.start, reverse=True)
        non_pii_original = original_text
        for span in sorted_labels:
            non_pii_original = non_pii_original[:span.start] + non_pii_original[span.end:]

        # Normalised character overlap (simplified Jaccard on character n-grams)
        n = 3  # trigrams
        orig_ngrams = _char_ngrams(non_pii_original, n)
        anon_ngrams = _char_ngrams(anonymized_text, n)
        if not orig_ngrams:
            return EvalMetricResult(name=self.name, value=1.0, level=level, match_mode=match_mode)

        intersection = len(orig_ngrams & anon_ngrams)
        union = len(orig_ngrams | anon_ngrams)
        score = safe_div(intersection, union)
        return EvalMetricResult(
            name=self.name, value=round(score, 6), level=level, match_mode=match_mode,
            metadata={"ngram_size": n, "intersection": intersection, "union": union},
        )


class PrivacyUtilityTradeoffMetric(MultiLevelMetric):
    """Computes a combined privacy-utility score.

    Combines a privacy score (1 - leakage_rate) with a utility score
    (semantic preservation) into a single number using harmonic mean
    or weighted combination.

    ``context`` must include ``privacy_score`` (float 0-1) and
    ``utility_score`` (float 0-1). Optionally, ``alpha`` (float 0-1)
    for weighted combination: combined = alpha * privacy + (1-alpha) * utility.
    If alpha is not provided, harmonic mean is used.

    References: TAB 2022 evaluation methodology.
    """

    name = "privacy_utility_tradeoff"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.DOCUMENT,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        privacy_score: float = float(ctx.get("privacy_score", 0.0))
        utility_score: float = float(ctx.get("utility_score", 0.0))
        alpha = ctx.get("alpha")

        if alpha is not None:
            # Weighted combination
            alpha = float(alpha)
            combined = alpha * privacy_score + (1.0 - alpha) * utility_score
        else:
            # Harmonic mean (default)
            combined = safe_div(
                2.0 * privacy_score * utility_score,
                privacy_score + utility_score,
            )

        return EvalMetricResult(
            name=self.name, value=round(combined, 6), level=level, match_mode=match_mode,
            metadata={"privacy_score": privacy_score, "utility_score": utility_score, "alpha": alpha},
        )


class InformationLossMetric(MultiLevelMetric):
    """Quantifies how much useful information was destroyed by anonymization.

    Measures the ratio of characters replaced or removed relative to the
    original document length, weighted by entity risk level.

    ``context`` must include ``original_text``, ``anonymized_text``, and
    optionally ``entity_risk_weights`` (entity_type -> weight).
    """

    name = "information_loss"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.DOCUMENT,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        original_text: str = ctx.get("original_text", "")
        if not original_text or not labels:
            return EvalMetricResult(name=self.name, value=0.0, level=level, match_mode=match_mode)

        risk_weights: dict[str, float] = ctx.get("entity_risk_weights", {})
        total_chars = len(original_text)
        weighted_removed = 0.0
        for span in labels:
            span_len = span.end - span.start
            weight = risk_weights.get(span.entity_type, 1.0)
            weighted_removed += span_len * weight

        loss = safe_div(weighted_removed, total_chars)
        return EvalMetricResult(
            name=self.name, value=round(min(loss, 1.0), 6), level=level, match_mode=match_mode,
            metadata={"weighted_chars_removed": round(weighted_removed, 2), "total_chars": total_chars},
        )


class EmbeddingSemanticPreservation(MultiLevelMetric):
    """Measures semantic similarity using embeddings (sentence transformers).

    Encodes original and anonymized text using a pre-trained sentence
    transformer model and computes cosine similarity. If per_sentence=True,
    splits texts into sentences and computes per-sentence similarities,
    then averages them.

    Falls back to trigram Jaccard similarity if sentence-transformers is
    not available.

    ``context`` must include ``original_text`` and ``anonymized_text``.

    Args:
        model_name: Pre-trained sentence transformer model name (default: all-MiniLM-L6-v2).
        per_sentence: If True, compute similarity per sentence and average (default: False).
    """

    name = "embedding_semantic_preservation"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", per_sentence: bool = False):
        """Initialize the embedding semantic preservation metric.

        Args:
            model_name: Name of the sentence transformer model to use.
            per_sentence: Whether to compute similarity per sentence.
        """
        super().__init__()
        self.model_name = model_name
        self.per_sentence = per_sentence
        self._model = None

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def _load_model(self) -> Any:
        """Lazily load the sentence transformer model."""
        if self._model is None and _HAS_SBERT:
            self._model = SentenceTransformer(self.model_name)

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.DOCUMENT,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        original_text: str = ctx.get("original_text", "")
        anonymized_text: str = ctx.get("anonymized_text", "")

        if not original_text or not anonymized_text:
            return EvalMetricResult(name=self.name, value=1.0, level=level, match_mode=match_mode)

        if not _HAS_SBERT:
            # Fall back to trigram Jaccard
            n = 3
            orig_ngrams = _char_ngrams(original_text, n)
            anon_ngrams = _char_ngrams(anonymized_text, n)
            if not orig_ngrams:
                score = 1.0
            else:
                intersection = len(orig_ngrams & anon_ngrams)
                union = len(orig_ngrams | anon_ngrams)
                score = safe_div(intersection, union)
            return EvalMetricResult(
                name=self.name, value=round(score, 6), level=level, match_mode=match_mode,
                metadata={"fallback": "trigram_jaccard"},
            )

        self._load_model()

        if self._model is None:
            return EvalMetricResult(name=self.name, value=1.0, level=level, match_mode=match_mode)

        if self.per_sentence:
            # Split into sentences and compute per-sentence similarity
            orig_sentences = _split_sentences(original_text)
            anon_sentences = _split_sentences(anonymized_text)

            if not orig_sentences or not anon_sentences:
                return EvalMetricResult(name=self.name, value=1.0, level=level, match_mode=match_mode)

            # Pad shorter list with empty strings to match lengths
            max_len = max(len(orig_sentences), len(anon_sentences))
            orig_sentences += [""] * (max_len - len(orig_sentences))
            anon_sentences += [""] * (max_len - len(anon_sentences))

            similarities = []
            for orig_sent, anon_sent in zip(orig_sentences, anon_sentences):
                if not orig_sent or not anon_sent:
                    similarities.append(0.0)
                else:
                    orig_emb = self._model.encode([orig_sent], convert_to_numpy=True)
                    anon_emb = self._model.encode([anon_sent], convert_to_numpy=True)
                    sim = cosine_similarity(orig_emb, anon_emb)[0, 0]
                    similarities.append(float(sim))

            score = sum(similarities) / len(similarities) if similarities else 1.0
        else:
            # Encode full texts
            orig_emb = self._model.encode([original_text], convert_to_numpy=True)
            anon_emb = self._model.encode([anonymized_text], convert_to_numpy=True)
            score = float(cosine_similarity(orig_emb, anon_emb)[0, 0])

        return EvalMetricResult(
            name=self.name, value=round(score, 6), level=level, match_mode=match_mode,
            metadata={"model": self.model_name, "per_sentence": self.per_sentence},
        )


class TaskUtilityProxy(MultiLevelMetric):
    """Lightweight proxy for downstream task utility assessment.

    Computes three utility preservation sub-scores:
    1. Sentiment preservation: polarity-based score
    2. Structural preservation: sentence/paragraph count similarity
    3. Length preservation: character length ratio

    Final score is the average of the three sub-scores.

    ``context`` must include ``original_text`` and ``anonymized_text``.
    """

    name = "task_utility_proxy"

    # Simple sentiment word lists for rule-based approach
    _POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "beautiful", "perfect", "happy", "best", "awesome",
    }
    _NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "hate", "worst", "ugly",
        "sad", "angry", "poor", "disappointing", "fail", "broken",
    }

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def _compute_sentiment_polarity(self, text: str) -> float:
        """Compute sentiment polarity [-1, 1] based on word counts."""
        text_lower = text.lower()
        pos_count = sum(1 for word in self._POSITIVE_WORDS if word in text_lower)
        neg_count = sum(1 for word in self._NEGATIVE_WORDS if word in text_lower)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def _compute_structural_similarity(self, orig_text: str, anon_text: str) -> float:
        """Compute structural preservation score."""
        orig_sentences = len(_split_sentences(orig_text))
        anon_sentences = len(_split_sentences(anon_text))
        orig_paragraphs = len(orig_text.split("\n\n"))
        anon_paragraphs = len(anon_text.split("\n\n"))

        sent_max = max(orig_sentences, anon_sentences, 1)
        para_max = max(orig_paragraphs, anon_paragraphs, 1)

        sent_diff = abs(orig_sentences - anon_sentences) / sent_max
        para_diff = abs(orig_paragraphs - anon_paragraphs) / para_max

        # Average the two structural metrics
        struct_diff = (sent_diff + para_diff) / 2.0
        return 1.0 - struct_diff

    def _compute_length_preservation(self, orig_text: str, anon_text: str) -> float:
        """Compute length preservation score."""
        orig_len = len(orig_text)
        anon_len = len(anon_text)
        if orig_len == 0 and anon_len == 0:
            return 1.0
        return min(anon_len, orig_len) / max(anon_len, orig_len, 1)

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.DOCUMENT,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        original_text: str = ctx.get("original_text", "")
        anonymized_text: str = ctx.get("anonymized_text", "")

        if not original_text or not anonymized_text:
            return EvalMetricResult(name=self.name, value=1.0, level=level, match_mode=match_mode)

        # Compute sub-scores
        orig_polarity = self._compute_sentiment_polarity(original_text)
        anon_polarity = self._compute_sentiment_polarity(anonymized_text)
        sentiment_score = 1.0 - abs(orig_polarity - anon_polarity)

        structural_score = self._compute_structural_similarity(original_text, anonymized_text)
        length_score = self._compute_length_preservation(original_text, anonymized_text)

        # Final score is average of sub-scores
        final_score = (sentiment_score + structural_score + length_score) / 3.0

        return EvalMetricResult(
            name=self.name, value=round(final_score, 6), level=level, match_mode=match_mode,
            metadata={
                "sentiment_preservation": round(sentiment_score, 6),
                "structural_preservation": round(structural_score, 6),
                "length_preservation": round(length_score, 6),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> set[str]:
    """Return the set of character n-grams in *text*."""
    if len(text) < n:
        return set()
    return {text[i: i + n] for i in range(len(text) - n + 1)}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence delimiters (.!?)."""
    import re
    # Split on sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    # Strip whitespace and filter out empty strings
    return [s.strip() for s in sentences if s.strip()]
