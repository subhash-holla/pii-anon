"""Expert similarity detection to prevent redundant MoE experts.

Uses cosine similarity on entity_strengths vectors (pure stdlib, no numpy).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

from pii_anon.moe import ExpertSpec

logger = logging.getLogger(__name__)


@dataclass
class SimilarityCheckResult:
    """Result of checking a candidate expert against existing experts."""

    is_similar: bool
    most_similar_id: str | None
    similarity_score: float
    action_taken: Literal["allowed", "warned", "rejected"]


class ExpertSimilarityGuard:
    """Guard against registering experts too similar to existing ones.

    Parameters
    ----------
    threshold : float
        Cosine similarity above which experts are considered duplicates.
        Default 0.95 (spaCy/Stanza at ~0.998 would trigger).
    action : {"warn", "reject"}
        ``"warn"`` logs a warning but allows registration.
        ``"reject"`` raises ``ExpertManifestError``.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        action: Literal["warn", "reject"] = "warn",
    ) -> None:
        self.threshold = threshold
        self.action = action

    def check(
        self,
        candidate: ExpertSpec,
        existing: list[ExpertSpec],
    ) -> SimilarityCheckResult:
        """Check if a candidate expert is too similar to any existing expert.

        Parameters
        ----------
        candidate : ExpertSpec
            The expert being registered.
        existing : list[ExpertSpec]
            Currently registered experts.

        Returns
        -------
        SimilarityCheckResult
            Whether the candidate is similar and what action was taken.
        """
        if not existing:
            return SimilarityCheckResult(
                is_similar=False,
                most_similar_id=None,
                similarity_score=0.0,
                action_taken="allowed",
            )

        best_id: str | None = None
        best_score = 0.0

        for other in existing:
            if other.expert_id == candidate.expert_id:
                continue
            score = self._cosine(candidate.entity_strengths, other.entity_strengths)
            if score > best_score:
                best_score = score
                best_id = other.expert_id

        is_similar = best_score >= self.threshold

        if is_similar:
            if self.action == "reject":
                from pii_anon.errors import ExpertManifestError

                raise ExpertManifestError(
                    f"Expert {candidate.expert_id!r} is too similar to "
                    f"{best_id!r} (cosine={best_score:.4f} >= {self.threshold})"
                )
            logger.warning(
                "expert_similarity_warning",
                extra={
                    "candidate": candidate.expert_id,
                    "similar_to": best_id,
                    "cosine": round(best_score, 4),
                    "threshold": self.threshold,
                },
            )
            action_taken: Literal["allowed", "warned", "rejected"] = "warned"
        else:
            action_taken = "allowed"

        return SimilarityCheckResult(
            is_similar=is_similar,
            most_similar_id=best_id,
            similarity_score=round(best_score, 4),
            action_taken=action_taken,
        )

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse strength vectors.

        Fills missing keys with 0.0 across the union of entity types.
        Returns 0.0 if either vector has zero norm.
        """
        all_keys = set(a.keys()) | set(b.keys())
        if not all_keys:
            return 0.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for key in all_keys:
            va = a.get(key, 0.0)
            vb = b.get(key, 0.0)
            dot += va * vb
            norm_a += va * va
            norm_b += vb * vb

        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        if denom < 1e-12:
            return 0.0
        return dot / denom
