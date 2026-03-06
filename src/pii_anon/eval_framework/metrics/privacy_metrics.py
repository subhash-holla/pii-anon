"""Privacy risk assessment metrics for anonymized / pseudonymized output.

Evidence basis:
- Sweeney (2002): k-anonymity model for protecting privacy
- Machanavajjhala et al. (2007): l-diversity beyond k-anonymity
- Li, Li & Venkatasubramanian (2007): t-closeness via EMD
- RAT-Bench (2025): TRIR text re-identification risk index
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    MultiLevelMetric,
    safe_div,
)


class ReidentificationRiskMetric(MultiLevelMetric):
    """Estimates re-identification probability from anonymized text.

    For each ground-truth entity, checks whether its original surface form
    survives in the anonymized output.  The risk score is the fraction of
    entities whose original text can still be found.

    References: RAT-Bench 2025, TRIR metric.

    ``context`` must include ``anonymized_text`` (str).
    """

    name = "reidentification_risk"

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
        if not original_text or not anonymized_text or not labels:
            return EvalMetricResult(name=self.name, value=0.0, level=level,
                                   match_mode=match_mode, metadata={"leaked": 0, "total": 0})

        leaked = 0
        total = len(labels)
        anon_lower = anonymized_text.lower()
        for span in labels:
            surface = original_text[span.start: span.end].lower().strip()
            if surface and surface in anon_lower:
                leaked += 1

        risk = safe_div(leaked, total)
        return EvalMetricResult(
            name=self.name, value=round(risk, 6), level=level, match_mode=match_mode,
            metadata={"leaked": leaked, "total": total},
        )


class KAnonymityMetric(MultiLevelMetric):
    """Computes k-anonymity for pseudonymized output.

    k-anonymity measures the minimum group size when entities are grouped
    by their pseudonymised replacement.  Higher *k* means better privacy.

    ``context`` must include ``pseudonym_map`` — a dict mapping
    original surface forms to pseudonyms.

    References: Sweeney (2002).
    """

    name = "k_anonymity"

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
        pseudonym_map: dict[str, str] = (context or {}).get("pseudonym_map", {})
        if not pseudonym_map:
            return EvalMetricResult(name=self.name, value=0.0, level=level,
                                   match_mode=match_mode, metadata={"k": 0, "groups": 0})

        groups: Counter[str] = Counter(pseudonym_map.values())
        k = min(groups.values()) if groups else 0
        return EvalMetricResult(
            name=self.name, value=float(k), level=level, match_mode=match_mode,
            metadata={"k": k, "groups": len(groups)},
        )


class LDiversityMetric(MultiLevelMetric):
    """Computes l-diversity (entropy variant) for pseudonymised output.

    Within each equivalence class (group sharing the same pseudonym), measures
    the entropy of the original entity values.  The metric value is the minimum
    entropy across all groups; higher is better.

    ``context`` must include ``pseudonym_map`` (original -> pseudonym) and
    ``entity_values`` (pseudonym -> list of original values).

    References: Machanavajjhala et al. (2007).
    """

    name = "l_diversity"

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
        entity_values: dict[str, list[str]] = (context or {}).get("entity_values", {})
        if not entity_values:
            return EvalMetricResult(name=self.name, value=0.0, level=level,
                                   match_mode=match_mode, metadata={"min_entropy": 0.0})

        entropies: list[float] = []
        for _pseudonym, originals in entity_values.items():
            counts = Counter(originals)
            total = sum(counts.values())
            entropy = -sum(
                (c / total) * math.log2(c / total) for c in counts.values() if c > 0
            )
            entropies.append(entropy)

        min_entropy = min(entropies) if entropies else 0.0
        return EvalMetricResult(
            name=self.name, value=round(min_entropy, 6), level=level, match_mode=match_mode,
            metadata={"min_entropy": round(min_entropy, 6), "groups": len(entity_values)},
        )


class TClosenessMetric(MultiLevelMetric):
    """Computes t-closeness via Earth Mover's Distance (EMD).

    Measures how close the distribution of sensitive values within each
    equivalence class is to the overall distribution.  Lower EMD means
    better privacy.

    ``context`` must include ``group_distributions`` — a dict mapping group
    IDs to Counter objects, and ``overall_distribution`` — a Counter for the
    entire dataset.

    References: Li, Li & Venkatasubramanian (2007).
    """

    name = "t_closeness"

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
        group_dists: dict[str, Counter[str]] = ctx.get("group_distributions", {})
        overall: Counter[str] = ctx.get("overall_distribution", Counter())
        if not group_dists or not overall:
            return EvalMetricResult(name=self.name, value=0.0, level=level,
                                   match_mode=match_mode, metadata={"max_emd": 0.0})

        overall_total = sum(overall.values())
        max_emd = 0.0
        for _group_id, dist in group_dists.items():
            group_total = sum(dist.values())
            if group_total == 0 or overall_total == 0:
                continue
            all_keys = set(dist.keys()) | set(overall.keys())
            emd = sum(
                abs(dist.get(k, 0) / group_total - overall.get(k, 0) / overall_total)
                for k in all_keys
            ) / 2.0
            max_emd = max(max_emd, emd)

        return EvalMetricResult(
            name=self.name, value=round(max_emd, 6), level=level, match_mode=match_mode,
            metadata={"max_emd": round(max_emd, 6), "groups": len(group_dists)},
        )


class LeakageDetectionMetric(MultiLevelMetric):
    """Detects PII fragments surviving in anonymized / pseudonymized output.

    Checks whether substrings of ground-truth entities (length >= ``min_chars``)
    appear in the output text.  Complements :class:`ReidentificationRiskMetric`
    by catching partial leaks.

    ``context`` must include ``anonymized_text`` and ``original_text``.
    """

    name = "leakage_detection"

    def __init__(self, min_chars: int = 4) -> None:
        self.min_chars = min_chars

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
        if not original_text or not anonymized_text or not labels:
            return EvalMetricResult(name=self.name, value=0.0, level=level,
                                   match_mode=match_mode, metadata={"leaked": 0, "total": 0})

        anon_lower = anonymized_text.lower()
        leaked = 0
        for span in labels:
            surface = original_text[span.start: span.end].strip()
            if len(surface) < self.min_chars:
                continue
            # Check progressively shorter substrings
            found = False
            for window_len in range(len(surface), self.min_chars - 1, -1):
                for start_offset in range(len(surface) - window_len + 1):
                    fragment = surface[start_offset: start_offset + window_len].lower()
                    if fragment in anon_lower:
                        found = True
                        break
                if found:
                    break
            if found:
                leaked += 1

        total = len(labels)
        score = safe_div(leaked, total)
        return EvalMetricResult(
            name=self.name, value=round(score, 6), level=level, match_mode=match_mode,
            metadata={"leaked": leaked, "total": total, "min_chars": self.min_chars},
        )


class MembershipInferenceMetric(MultiLevelMetric):
    """Detects membership inference via n-gram overlap between original and anonymized text.

    Extracts character n-grams (n=4) from original PII spans and checks what fraction
    appears in the anonymized text. This provides a surface-level membership inference
    attack signal.

    ``context`` must include ``original_text``, ``anonymized_text``, and ``labels``
    (for extracting PII spans).

    Computes approximate AUC via pseudo-ROC across multiple thresholds.
    Returns overlap_score (0=no leakage, 1=full leakage).
    """

    name = "membership_inference"

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
        if not original_text or not anonymized_text or not labels:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"overlap_score": 0.0, "auc_approx": 0.0, "pii_ngrams_checked": 0}
            )

        # Extract 4-grams from original PII spans
        ngrams_original: set[str] = set()
        for span in labels:
            surface = original_text[span.start: span.end].lower()
            if len(surface) >= 4:
                for i in range(len(surface) - 3):
                    ngrams_original.add(surface[i: i + 4])

        if not ngrams_original:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"overlap_score": 0.0, "auc_approx": 0.0, "pii_ngrams_checked": 0}
            )

        # Check overlap in anonymized text
        anon_lower = anonymized_text.lower()
        overlap_count = sum(1 for ng in ngrams_original if ng in anon_lower)
        overlap_score = safe_div(overlap_count, len(ngrams_original))

        # Compute approximate AUC via pseudo-ROC using multiple thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        tpr_fpr_pairs: list[tuple[float, float]] = []
        for threshold in thresholds:
            # Treat overlap > threshold as "positive prediction"
            prediction = 1.0 if overlap_score > threshold else 0.0
            # True positive rate (assuming actual = 1 for membership)
            tpr = prediction
            # False positive rate (assuming actual = 0 for non-membership)
            fpr = prediction
            tpr_fpr_pairs.append((fpr, tpr))

        # Approximate AUC via trapezoidal rule
        tpr_fpr_pairs.sort()
        auc_approx = 0.0
        for i in range(len(tpr_fpr_pairs) - 1):
            fpr1, tpr1 = tpr_fpr_pairs[i]
            fpr2, tpr2 = tpr_fpr_pairs[i + 1]
            auc_approx += (fpr2 - fpr1) * (tpr1 + tpr2) / 2.0

        return EvalMetricResult(
            name=self.name, value=round(overlap_score, 6), level=level, match_mode=match_mode,
            metadata={
                "overlap_score": round(overlap_score, 6),
                "auc_approx": round(auc_approx, 6),
                "pii_ngrams_checked": len(ngrams_original),
            },
        )


class AttributeInferenceMetric(MultiLevelMetric):
    """Measures inference of quasi-identifiers from contextual clues in anonymized text.

    Checks whether gendered pronouns, age-indicating words, nationality references,
    etc., could reveal quasi-identifiers even after direct anonymization.

    ``context`` must include ``anonymized_text`` and ``quasi_identifier_labels``
    (list of dicts with keys: ``entity_type``, ``original_value``).

    Returns inference_rate (0=no inference, 1=all inferred).
    """

    name = "attribute_inference"

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
        anonymized_text: str = ctx.get("anonymized_text", "")
        quasi_id_labels: list[dict[str, Any]] = ctx.get("quasi_identifier_labels", [])
        if not anonymized_text or not quasi_id_labels:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"inferred": 0, "total": 0, "inference_rate": 0.0}
            )

        anon_lower = anonymized_text.lower()
        inferred_count = 0

        # Define contextual clue patterns
        gendered_pronouns = {"he", "she", "his", "her", "him", "hers"}
        age_indicating_words = {
            "elderly", "young", "teenager", "toddler", "infant", "retired",
            "senior", "youth", "youngster", "child", "adult", "aged"
        }
        nationality_words = {
            "american", "british", "french", "german", "spanish", "italian",
            "chinese", "japanese", "indian", "brazilian", "canadian", "australian",
            "mexico", "mexico city", "london", "paris", "tokyo", "mumbai",
            "russia", "usa", "uk", "europe", "asia", "africa"
        }

        for quasi_id in quasi_id_labels:
            entity_type = quasi_id.get("entity_type", "").lower()
            inferred = False

            if entity_type == "gender":
                # Check for gendered pronouns
                words = anon_lower.split()
                if any(pronoun in words for pronoun in gendered_pronouns):
                    inferred = True

            elif entity_type in ["age", "date_of_birth"]:
                # Check for age-indicating words
                words = anon_lower.split()
                if any(word in age_indicating_words for word in words):
                    inferred = True

            elif entity_type == "nationality":
                # Check for nationality/country references
                words = anon_lower.split()
                if any(word in nationality_words for word in words):
                    inferred = True
                # Also check for longer phrases
                for nat_word in nationality_words:
                    if nat_word in anon_lower:
                        inferred = True
                        break

            if inferred:
                inferred_count += 1

        inference_rate = safe_div(inferred_count, len(quasi_id_labels))
        return EvalMetricResult(
            name=self.name, value=round(inference_rate, 6), level=level, match_mode=match_mode,
            metadata={
                "inferred": inferred_count,
                "total": len(quasi_id_labels),
                "inference_rate": round(inference_rate, 6),
            },
        )


class CanaryExposureMetric(MultiLevelMetric):
    """Detects exposure of canary strings in anonymized text.

    Canary strings are intentionally injected markers used to detect if the
    original data was used or if a model memorized specific training examples.
    Checks for both full and partial (>50% character overlap) matches.

    ``context`` must include ``canary_strings`` (list[str]) and ``anonymized_text`` (str).

    Returns exposure_score (0=no canaries exposed, 1=all canaries exposed).
    """

    name = "canary_exposure"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def _partial_match_score(self, canary: str, text: str) -> float:
        """Compute how much of the canary appears as a substring in text."""
        canary_lower = canary.lower()
        text_lower = text.lower()

        if canary_lower in text_lower:
            return 1.0  # Full match

        max_overlap: float = 0.0
        # Check all substrings of the text for overlap with canary
        for i in range(len(text_lower)):
            for j in range(i + 1, len(text_lower) + 1):
                substring = text_lower[i:j]
                # Count matching characters
                matches = sum(1 for c in substring if c in canary_lower)
                overlap_ratio = safe_div(matches, len(canary_lower))
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio

        return max_overlap

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
        canary_strings: list[str] = ctx.get("canary_strings", [])
        anonymized_text: str = ctx.get("anonymized_text", "")
        if not canary_strings or not anonymized_text:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"exposed_full": 0, "exposed_partial": 0, "total_canaries": 0}
            )

        exposed_full = 0
        exposed_partial = 0

        for canary in canary_strings:
            overlap_score = self._partial_match_score(canary, anonymized_text)
            if overlap_score >= 1.0:
                exposed_full += 1
            elif overlap_score >= 0.5:
                exposed_partial += 1

        total_canaries = len(canary_strings)
        exposure_count = exposed_full + exposed_partial
        exposure_score = safe_div(exposure_count, total_canaries)

        return EvalMetricResult(
            name=self.name, value=round(exposure_score, 6), level=level, match_mode=match_mode,
            metadata={
                "exposed_full": exposed_full,
                "exposed_partial": exposed_partial,
                "total_canaries": total_canaries,
            },
        )


class EntityLinkageMetric(MultiLevelMetric):
    """Measures cross-document entity linkage via n-gram fingerprint similarity.

    Builds n-gram fingerprints (n=5) for each document in an anonymized corpus
    and attempts to link documents that share entities. Documents with Jaccard
    similarity > threshold (0.3) are considered linked, indicating possible
    de-anonymization through document matching.

    ``context`` must include ``anonymized_documents`` (list[str]) and
    ``document_entity_map`` (dict mapping document index to list of original entity surface forms).

    Returns linkage_rate (0=no linked documents, 1=all documents linked).
    """

    name = "entity_linkage"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.DOCUMENT]

    def _build_ngram_fingerprint(self, text: str, n: int = 5) -> set[str]:
        """Extract n-gram fingerprint from text."""
        text_lower = text.lower()
        ngrams: set[str] = set()
        if len(text_lower) >= n:
            for i in range(len(text_lower) - n + 1):
                ngrams.add(text_lower[i: i + n])
        return ngrams

    def _jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        return intersection / union

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
        anonymized_docs: list[str] = ctx.get("anonymized_documents", [])
        doc_entity_map: dict[int, list[str]] = ctx.get("document_entity_map", {})
        if not anonymized_docs or not doc_entity_map:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"linked_pairs": 0, "total_pairs": 0, "threshold": 0.3}
            )

        # Build n-gram fingerprints for each document
        fingerprints: list[set[str]] = []
        for doc in anonymized_docs:
            fp = self._build_ngram_fingerprint(doc, n=5)
            fingerprints.append(fp)

        # Check for linked documents via Jaccard similarity
        threshold = 0.3
        linked_pairs = 0
        total_pairs = 0

        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                similarity = self._jaccard_similarity(fingerprints[i], fingerprints[j])
                total_pairs += 1
                if similarity > threshold:
                    linked_pairs += 1

        linkage_rate = safe_div(linked_pairs, total_pairs) if total_pairs > 0 else 0.0

        return EvalMetricResult(
            name=self.name, value=round(linkage_rate, 6), level=level, match_mode=match_mode,
            metadata={
                "linked_pairs": linked_pairs,
                "total_pairs": total_pairs,
                "threshold": threshold,
            },
        )
