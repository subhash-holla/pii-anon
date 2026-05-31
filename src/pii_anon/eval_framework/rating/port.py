"""Structural Protocol port for rating engines (S3-01, FR-003).

``RatingEnginePort`` is the MINIMAL structural contract the production
callers rely on. It is a :class:`typing.Protocol` so concrete engines
satisfy it WITHOUT inheritance (zero call-site changes); the existing
:class:`~pii_anon.eval_framework.rating.elo.PIIRateEloEngine` is registered
as the ``glicko-legacy`` fallback tier and conforms structurally.

Design trace: D-EVAL DECISION 2 — RatingEnginePort 3-tier ladder. The richer
engine API (``run_reidentification_tournament``, ``tournament_summary``,
``evaluate_governance``, ``update_from_match``, ``get_leaderboard``) stays on
the concrete class and is intentionally NOT part of this port, so future
tiers (``bradley-terry-mle``, ``bayes-bt``) are not over-constrained.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .elo import EloRating, RatingUpdate

__all__ = ["RatingEnginePort"]


@runtime_checkable
class RatingEnginePort(Protocol):
    """Minimal structural contract for a pairwise rating engine.

    Only the two methods the four production callers use are part of the
    contract:

    * :meth:`run_round_robin` — rate an all-pairs round-robin from composite
      scores, returning the per-match :class:`RatingUpdate` records.
    * :meth:`get_rating` — fetch the current rating for a system, or ``None``.
    """

    def run_round_robin(self, composites: dict[str, float]) -> list[RatingUpdate]:
        """Run an all-pairs round-robin from ``name -> composite score``."""
        ...

    def get_rating(self, name: str) -> EloRating | None:
        """Return the current rating for ``name``, or ``None`` if unknown."""
        ...
