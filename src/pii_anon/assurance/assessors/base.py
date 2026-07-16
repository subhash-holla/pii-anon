"""Shared assessor primitives — re-exported from the dependency-free ``spans`` module
(kept here so existing ``from .base import ...`` call sites stay valid)."""

from __future__ import annotations

from ..spans import Span, _norm, match_counts, per_type_counts

__all__ = ["Span", "_norm", "match_counts", "per_type_counts"]
