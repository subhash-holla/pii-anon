"""Offline calibration: run engines against labeled benchmark data."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pii_anon.calibration.store import CalibrationResult, CalibrationStore
from pii_anon.errors import CalibrationError

logger = logging.getLogger(__name__)


@dataclass
class OfflineCalibrationConfig:
    """Configuration for offline calibration runs.

    Attributes
    ----------
    dataset : str
        Benchmark dataset name (loaded via ``load_benchmark_dataset``).
    max_samples : int | None
        Limit the number of records processed. ``None`` means all.
    min_entity_samples : int
        Minimum labeled examples per entity type to include in results.
    store_path : str | None
        Override path for ``CalibrationStore``.
    """

    dataset: str = "pii_anon_benchmark"
    max_samples: int | None = None
    min_entity_samples: int = 10
    store_path: str | None = None


class OfflineCalibrator:
    """Run each engine individually against labeled data and compute per-entity F1.

    Parameters
    ----------
    engine_registry : EngineRegistry
        Registry of available detection engines.
    expert_registry : ExpertRegistry
        MoE expert registry (updated when ``run_and_apply`` is called).
    config : OfflineCalibrationConfig | None
        Calibration parameters. Defaults to standard settings.
    """

    def __init__(
        self,
        engine_registry: Any,
        expert_registry: Any,
        config: OfflineCalibrationConfig | None = None,
    ) -> None:
        self._engine_registry = engine_registry
        self._expert_registry = expert_registry
        self._config = config or OfflineCalibrationConfig()

    def run(self) -> CalibrationResult:
        """Run calibration without modifying registries.

        Returns
        -------
        CalibrationResult
            Per-engine, per-entity-type F1 scores and sample counts.

        Raises
        ------
        CalibrationError
            If the benchmark dataset cannot be loaded.
        """
        from pii_anon.benchmarks import load_benchmark_dataset

        try:
            records = load_benchmark_dataset(self._config.dataset)
        except Exception as exc:
            raise CalibrationError(f"Failed to load dataset {self._config.dataset!r}: {exc}") from exc

        if self._config.max_samples is not None:
            records = records[: self._config.max_samples]

        engines = self._engine_registry.list_engines(include_disabled=False)
        engine_entity_f1: dict[str, dict[str, float]] = {}
        sample_counts: dict[str, dict[str, int]] = {}
        skipped: list[str] = []

        for engine in engines:
            try:
                engine_f1, engine_counts = self._evaluate_engine(engine, records)
                # Filter by min samples
                filtered_f1: dict[str, float] = {}
                for etype, f1 in engine_f1.items():
                    if engine_counts.get(etype, 0) >= self._config.min_entity_samples:
                        filtered_f1[etype] = round(f1, 4)
                engine_entity_f1[engine.adapter_id] = filtered_f1
                sample_counts[engine.adapter_id] = engine_counts
            except Exception:
                logger.warning(
                    "calibration_engine_skipped",
                    extra={"engine_id": engine.adapter_id},
                    exc_info=True,
                )
                skipped.append(engine.adapter_id)

        return CalibrationResult(
            schema_version="1.0",
            calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dataset=self._config.dataset,
            engine_entity_f1=engine_entity_f1,
            sample_counts=sample_counts,
            skipped_engines=skipped,
            metadata={
                "max_samples": self._config.max_samples,
                "min_entity_samples": self._config.min_entity_samples,
                "total_records": len(records),
            },
        )

    def run_and_apply(
        self,
        router: Any | None = None,
    ) -> CalibrationResult:
        """Run calibration, save results, and apply to expert registry.

        Parameters
        ----------
        router : MoERouter | None
            If provided, clears cache after applying calibrated weights.
        """
        result = self.run()
        store = CalibrationStore(path=Path(self._config.store_path) if self._config.store_path else None)
        store.save(result)
        store.apply_to_registry(
            self._expert_registry,
            router,
            min_samples=self._config.min_entity_samples,
        )
        return result

    def _evaluate_engine(
        self,
        engine: Any,
        records: list[Any],
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Compute per-entity-type F1 for a single engine.

        Returns (entity_type -> F1, entity_type -> sample_count).
        """
        # Accumulate TP/FP/FN per entity type
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)
        label_counts: dict[str, int] = defaultdict(int)

        for record in records:
            context: dict[str, Any] = {
                "language": getattr(record, "language", "en"),
                "policy_mode": "balanced",
            }
            findings = engine.detect({"text": record.text}, context)

            # Build span sets for comparison
            pred_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
            for f in findings:
                if f.span_start is not None and f.span_end is not None:
                    pred_by_type[f.entity_type].add((f.span_start, f.span_end))

            gold_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
            for label in record.labels:
                etype = str(label.get("entity_type", "UNKNOWN"))
                start = int(label.get("start", 0))
                end = int(label.get("end", 0))
                gold_by_type[etype].add((start, end))
                label_counts[etype] += 1

            # Compute per-entity-type TP/FP/FN
            all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
            for etype in all_types:
                preds = pred_by_type.get(etype, set())
                golds = gold_by_type.get(etype, set())
                tp[etype] += len(preds & golds)
                fp[etype] += len(preds - golds)
                fn[etype] += len(golds - preds)

        # Compute F1 per entity type
        entity_f1: dict[str, float] = {}
        for etype in set(tp.keys()) | set(fn.keys()):
            t = tp[etype]
            f = fp[etype]
            n = fn[etype]
            precision = t / max(1, t + f)
            recall = t / max(1, t + n)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            entity_f1[etype] = f1

        return entity_f1, dict(label_counts)
