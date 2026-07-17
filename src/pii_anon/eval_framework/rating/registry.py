"""Entry-point registry for rating engines (S3-01, NFR-026).

Mirrors :mod:`pii_anon.engines.registry` (thread-safe dict, ``register`` /
``get`` / ``discover_entrypoint_engines`` with the
``entry_points().select(group=...)`` + ``hasattr`` fallback and the
``ep.load()`` + type-check loop). The discovery group is
``pii_anon.rating_engines`` — a DISTINCT group from ``pii_anon.engines`` so
rating-engine discovery cannot affect detection-engine discovery.

Unlike :class:`~pii_anon.engines.registry.EngineRegistry` (which keys off
``EngineAdapter.adapter_id``), rating engines satisfy the structural
:class:`~pii_anon.eval_framework.rating.port.RatingEnginePort`, which has no
identity attribute — so they are keyed by the entry-point *name* (e.g.
``glicko-legacy``).

Discovery degrades gracefully (NFR-026): any failure to enumerate or load an
entry point yields an empty / partial list rather than raising, so absent
optional tiers (``bradley-terry-mle``, ``bayes-bt``) never break callers.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata
from threading import Lock

from .port import RatingEnginePort

__all__ = ["RatingEngineRegistry"]


class RatingEngineRegistry:
    """Thread-safe name-keyed registry of :class:`RatingEnginePort` engines."""

    def __init__(self) -> None:
        self._engines: dict[str, RatingEnginePort] = {}
        self._lock = Lock()

    def register(self, name: str, engine: RatingEnginePort) -> None:
        """Register ``engine`` under ``name`` (overwrites an existing entry)."""
        with self._lock:
            self._engines[name] = engine

    def unregister(self, name: str) -> None:
        with self._lock:
            self._engines.pop(name, None)

    def get(self, name: str) -> RatingEnginePort | None:
        with self._lock:
            return self._engines.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._engines.keys())

    def discover_entrypoint_engines(
        self, group: str = "pii_anon.rating_engines"
    ) -> list[str]:
        """Discover and register engines advertised under ``group``.

        Returns the list of registered entry-point names. Resolves both class
        targets (instantiated with no args) and pre-built instances. Any
        enumeration or load failure is swallowed (NFR-026 graceful
        degradation) so an absent / broken extra never raises.
        """
        discovered: list[str] = []
        try:
            entry_points = metadata.entry_points()
            selected: Iterable[metadata.EntryPoint]
            if hasattr(entry_points, "select"):
                selected = entry_points.select(group=group)
            else:
                selected = entry_points.get(group) or ()
        except Exception:
            return discovered

        for ep in selected:
            try:
                loaded = ep.load()
                if isinstance(loaded, type):
                    instance = loaded()
                else:
                    instance = loaded
                if not isinstance(instance, RatingEnginePort):
                    continue
            except Exception:
                continue

            self.register(ep.name, instance)
            discovered.append(ep.name)
        return discovered
