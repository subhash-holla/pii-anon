"""Online calibration: EMA-based gating weight updates during a session."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OnlineCalibrationConfig:
    """Configuration for online (session-level) calibration.

    Attributes
    ----------
    enabled : bool
        Master switch for online calibration (default ``False``).
    ema_alpha : float
        Exponential moving average blend rate. Higher = faster adaptation.
    min_observations : int
        Minimum observations per (engine, entity_type) before updating.
    floor_strength : float
        Never let a strength decay below this value.
    persist_on_update : bool
        If ``True``, write updated strengths to ``CalibrationStore``
        after each observation.
    """

    enabled: bool = False
    ema_alpha: float = 0.05
    min_observations: int = 5
    floor_strength: float = 0.05
    persist_on_update: bool = False


class OnlineCalibrator:
    """EMA-based live weight updater for the MoE router.

    Updates ``ExpertSpec.entity_strengths`` in place during a running
    session based on observed detection performance. Off by default.

    Parameters
    ----------
    expert_registry : ExpertRegistry
        Registry whose specs will be mutated.
    router : MoERouter
        Router whose cache will be cleared on updates.
    config : OnlineCalibrationConfig | None
        Configuration. Defaults to disabled.
    store : CalibrationStore | None
        Optional store for persisting updates.
    """

    def __init__(
        self,
        expert_registry: Any,
        router: Any,
        config: OnlineCalibrationConfig | None = None,
        store: Any | None = None,
    ) -> None:
        self._expert_registry = expert_registry
        self._router = router
        self._config = config or OnlineCalibrationConfig()
        self._store = store
        self._lock = Lock()
        self._observations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # Snapshot original strengths at init for reset capability
        self._original_strengths: dict[str, dict[str, float]] = {}
        for spec in self._expert_registry.list_experts():
            self._original_strengths[spec.expert_id] = dict(spec.entity_strengths)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def observe(
        self,
        engine_id: str,
        entity_type: str,
        predicted_spans: set[tuple[int, int]],
        gold_spans: set[tuple[int, int]],
    ) -> float | None:
        """Record an observation and optionally update gating weight.

        Parameters
        ----------
        engine_id : str
            Engine that produced the predictions.
        entity_type : str
            Entity type being evaluated.
        predicted_spans : set[tuple[int, int]]
            Set of (start, end) spans predicted by the engine.
        gold_spans : set[tuple[int, int]]
            Set of (start, end) gold-standard spans.

        Returns
        -------
        float | None
            New strength value if updated, ``None`` if insufficient observations.
        """
        if not self._config.enabled:
            return None

        # Compute F1 for this observation
        tp = len(predicted_spans & gold_spans)
        fp = len(predicted_spans - gold_spans)
        fn = len(gold_spans - predicted_spans)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        observed_f1 = 2 * precision * recall / max(1e-9, precision + recall)

        with self._lock:
            self._observations[engine_id][entity_type].append(observed_f1)
            obs_count = len(self._observations[engine_id][entity_type])

            if obs_count < self._config.min_observations:
                return None

            spec = self._expert_registry.get_expert(engine_id)
            if spec is None:
                return None

            current = spec.entity_strengths.get(entity_type, spec.default_weight)
            alpha = self._config.ema_alpha
            new_strength = alpha * observed_f1 + (1.0 - alpha) * current
            new_strength = max(new_strength, self._config.floor_strength)

            spec.entity_strengths[entity_type] = new_strength
            self._router.clear_cache()

            logger.debug(
                "online_calibration_update",
                extra={
                    "engine_id": engine_id,
                    "entity_type": entity_type,
                    "old_strength": round(current, 4),
                    "new_strength": round(new_strength, 4),
                    "observed_f1": round(observed_f1, 4),
                    "observations": obs_count,
                },
            )

            if self._config.persist_on_update and self._store is not None:
                self._persist()

            return float(new_strength)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return current entity_strengths for all experts."""
        result: dict[str, dict[str, float]] = {}
        for spec in self._expert_registry.list_experts():
            result[spec.expert_id] = dict(spec.entity_strengths)
        return result

    def reset(self) -> None:
        """Restore original strengths and clear observations."""
        with self._lock:
            for spec in self._expert_registry.list_experts():
                original = self._original_strengths.get(spec.expert_id)
                if original is not None:
                    spec.entity_strengths.clear()
                    spec.entity_strengths.update(original)
            self._observations.clear()
            self._router.clear_cache()

    def _persist(self) -> None:
        """Write current strengths to the calibration store."""
        import time

        from pii_anon.calibration.store import CalibrationResult

        strengths = self.snapshot()
        result = CalibrationResult(
            schema_version="1.0",
            calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dataset="online_session",
            engine_entity_f1=strengths,
            sample_counts={},
            metadata={"source": "online_calibration"},
        )
        self._store.save(result)  # type: ignore[union-attr]
