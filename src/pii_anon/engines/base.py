"""Abstract base class and interface contract for PII detection engines.

The ``EngineAdapter`` defines the contract that all PII detection engines must
implement. Engines are responsible for scanning payloads and emitting
``EngineFinding`` instances (individual detections with confidence scores).

Engine Lifecycle
----------------
1. **Initialization**: ``initialize()`` loads runtime configuration.
2. **Dependency checks**: ``dependency_available()`` verifies native dependencies.
3. **Capability reporting**: ``capabilities()`` and ``health_check()`` inform
   orchestrators about what the engine supports.
4. **Detection**: ``detect()`` scans a payload and returns findings.
5. **Shutdown**: ``shutdown()`` cleans up resources (optional).

Typical Usage
~~~~~~~~~~~~~
Orchestrators instantiate engines, call ``initialize()`` with config, and then
invoke ``detect()`` during payload processing. Multiple engines can run in parallel,
with findings later merged by fusion strategies.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import Any

from pii_anon.types import EngineCapabilities, EngineFinding, Payload


class EngineAdapter(ABC):
    """Abstract base class defining the engine adapter contract.

    Subclasses must implement the ``detect()`` method and may override
    ``initialize()``, ``health_check()``, ``shutdown()``, and ``capabilities()``
    to provide domain-specific behavior.

    Attributes
    ----------
    adapter_id : str
        Unique identifier for this engine (e.g., "regex-oss", "presidio").
        Set as a class attribute by subclasses.
    native_dependency : str | None
        Name of the native Python dependency (e.g., "presidio_analyzer").
        Used to check if optional engines are available.
    enabled : bool
        Whether this engine is active. Disabled engines are skipped during
        ``detect()`` calls.
    """
    adapter_id: str = "unknown"
    native_dependency: str | None = None

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._config: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the engine with optional runtime configuration.

        Subclasses should override this to load engine-specific config
        (e.g., pattern registry, model weights, deny-lists).

        Parameters
        ----------
        config : dict[str, Any] | None
            Engine configuration dict. Keys may include:

            - ``enabled`` (bool): whether the engine is active.
            - Engine-specific keys: delegated to subclass implementations.

        Notes
        -----
        This base implementation stores config and handles the ``enabled`` key.
        """
        if not config:
            return
        self._config = dict(config)
        if "enabled" in config:
            self.enabled = bool(config["enabled"])

    def dependency_available(self) -> bool:
        """Check whether the engine's native dependency is installed.

        Returns ``True`` if no dependency is required, or if the dependency
        module is found in the Python environment.  The result is cached
        after the first call to avoid repeated filesystem lookups.

        Returns
        -------
        bool
            ``True`` if the native dependency is available or not required;
            ``False`` otherwise.

        Notes
        -----
        Used by orchestrators to determine which engines can be enabled and
        to report health status. Engines without native dependencies always
        return ``True``.
        """
        if self.native_dependency is None:
            return True
        # Cache the result — importlib.util.find_spec triggers filesystem
        # stat() calls on every invocation, which is expensive in hot loops.
        cached: bool | None = getattr(self, "_dep_available_cache", None)
        if cached is not None:
            return cached
        try:
            result = importlib.util.find_spec(self.native_dependency) is not None
        except Exception:
            result = False
        self._dep_available_cache: bool | None = result
        return result

    def capabilities(self) -> EngineCapabilities:
        """Report the engine's capabilities and feature support.

        Returns
        -------
        EngineCapabilities
            An object describing the engine's supported languages, streaming
            capability, runtime configuration flexibility, and dependency status.

        Notes
        -----
        Subclasses may override this to report additional capabilities.
        The orchestrator uses this to determine if an engine can handle
        specific language requests or streaming workloads.
        """
        return EngineCapabilities(
            adapter_id=self.adapter_id,
            native_dependency=self.native_dependency,
            dependency_available=self.dependency_available(),
        )

    def health_check(self) -> dict[str, Any]:
        """Perform a health check and return status information.

        Returns a diagnostic report indicating whether the engine is enabled,
        whether dependencies are available, and operational mode (native vs
        fallback).

        Returns
        -------
        dict[str, Any]
            A dict with keys:

            - ``adapter_id``: engine identifier
            - ``healthy``: always ``True`` (always returns success in base impl)
            - ``details``: "disabled", "native", or "fallback"
            - ``dependency_available``: whether native dependency is present

        Notes
        -----
        Subclasses may override to add custom health checks (e.g., API
        connectivity, model loading).
        """
        if not self.enabled:
            return {
                "adapter_id": self.adapter_id,
                "healthy": True,
                "details": "disabled",
                "dependency_available": self.dependency_available(),
            }
        available = self.dependency_available()
        details = "native" if available else "fallback"
        return {
            "adapter_id": self.adapter_id,
            "healthy": True,
            "details": details,
            "dependency_available": available,
        }

    def shutdown(self) -> None:
        """Clean up engine resources (optional).

        Called when the engine is no longer needed. Subclasses may override
        to release external connections, cache memory, or background threads.

        Notes
        -----
        The base implementation is a no-op.
        """
        return None

    @abstractmethod
    def detect(self, payload: Payload, context: dict[str, Any]) -> list[EngineFinding]:
        """Scan a payload and emit PII findings.

        This is the core detection method that all engines must implement.
        It iterates over the payload fields and returns a list of detected
        entities with confidence scores.

        Parameters
        ----------
        payload : Payload
            A dict of field names to scalar values. Typically contains text
            fields like "name", "email", "phone", etc.
        context : dict[str, Any]
            Runtime context (e.g., ``language``, ``policy_mode``).
            Used to customize detection behavior.

        Returns
        -------
        list[EngineFinding]
            A list of detected entities, each with ``entity_type``,
            ``confidence``, optional span positions, and ``engine_id``.

        Raises
        ------
        NotImplementedError
            In the base class. Subclasses must implement.
        """
        raise NotImplementedError
