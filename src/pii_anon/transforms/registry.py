"""Thread-safe registry for transformation strategies.

Mirrors the ``EngineRegistry`` pattern used for detection engines,
providing a central place to register, look up, and enumerate
available transformation strategies.
"""

from __future__ import annotations

from threading import Lock

from pii_anon.transforms.base import StrategyMetadata, TransformStrategy


class StrategyRegistry:
    """Registry of available transformation strategies.

    Thread-safe dictionary-based storage.  Strategies are registered by
    their ``strategy_id`` and can be looked up, listed, or removed at
    runtime.

    Example
    -------
    >>> registry = StrategyRegistry()
    >>> registry.register(RedactionStrategy())
    >>> registry.get("redact")
    <RedactionStrategy ...>
    """

    def __init__(self) -> None:
        self._strategies: dict[str, TransformStrategy] = {}
        self._lock = Lock()

    def register(self, strategy: TransformStrategy) -> None:
        """Register a transformation strategy.

        Parameters
        ----------
        strategy : TransformStrategy
            The strategy instance to register.

        Raises
        ------
        ValueError
            If a strategy with the same ``strategy_id`` is already registered.
        """
        with self._lock:
            if strategy.strategy_id in self._strategies:
                raise ValueError(
                    f"Strategy '{strategy.strategy_id}' is already registered. "
                    f"Unregister it first to replace."
                )
            self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> TransformStrategy | None:
        """Look up a strategy by ID.

        Parameters
        ----------
        strategy_id : str
            The strategy identifier.

        Returns
        -------
        TransformStrategy | None
            The registered strategy, or ``None`` if not found.
        """
        with self._lock:
            return self._strategies.get(strategy_id)

    def unregister(self, strategy_id: str) -> None:
        """Remove a strategy from the registry.

        Parameters
        ----------
        strategy_id : str
            The strategy identifier to remove.

        Raises
        ------
        KeyError
            If the strategy is not registered.
        """
        with self._lock:
            if strategy_id not in self._strategies:
                raise KeyError(f"Strategy '{strategy_id}' is not registered.")
            del self._strategies[strategy_id]

    def list_strategies(self) -> list[str]:
        """Return sorted list of registered strategy IDs."""
        with self._lock:
            return sorted(self._strategies.keys())

    def list_metadata(self) -> list[StrategyMetadata]:
        """Return metadata for all registered strategies."""
        with self._lock:
            return [s.metadata() for s in self._strategies.values()]

    def __contains__(self, strategy_id: str) -> bool:
        with self._lock:
            return strategy_id in self._strategies

    def __len__(self) -> int:
        with self._lock:
            return len(self._strategies)
