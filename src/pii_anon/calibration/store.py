"""Portable JSON persistence for MoE calibration results."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pii_anon.errors import CalibrationError


_DEFAULT_PATH = Path.home() / ".pii_anon" / "calibration.json"


@dataclass
class CalibrationResult:
    """JSON-serializable record of one calibration run.

    Attributes
    ----------
    schema_version : str
        Schema version for forward compatibility (currently ``"1.0"``).
    calibrated_at : str
        ISO-8601 timestamp of when calibration was run.
    dataset : str
        Name of the benchmark dataset used.
    engine_entity_f1 : dict[str, dict[str, float]]
        Per-engine, per-entity-type F1 scores.
    sample_counts : dict[str, dict[str, int]]
        Number of labeled examples per (engine, entity_type).
    skipped_engines : list[str]
        Engines that failed during calibration.
    metadata : dict[str, Any]
        Arbitrary metadata (e.g., run parameters).
    """

    schema_version: str = "1.0"
    calibrated_at: str = ""
    dataset: str = ""
    engine_entity_f1: dict[str, dict[str, float]] = field(default_factory=dict)
    sample_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_engines: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationResult:
        version = data.get("schema_version", "")
        if not version.startswith("1."):
            raise CalibrationError(f"Unsupported calibration schema version: {version!r}")
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            calibrated_at=str(data.get("calibrated_at", "")),
            dataset=str(data.get("dataset", "")),
            engine_entity_f1=data.get("engine_entity_f1", {}),
            sample_counts=data.get("sample_counts", {}),
            skipped_engines=data.get("skipped_engines", []),
            metadata=data.get("metadata", {}),
        )


class CalibrationStore:
    """Load and save calibration results as a JSON file.

    Parameters
    ----------
    path : Path | None
        File path. Defaults to ``~/.pii_anon/calibration.json``.
        Can be overridden via the ``PII_ANON_CALIBRATION_PATH`` env var.
    """

    def __init__(self, path: Path | None = None) -> None:
        env_path = os.environ.get("PII_ANON_CALIBRATION_PATH")
        if path is not None:
            self._path = Path(path)
        elif env_path:
            self._path = Path(env_path)
        else:
            self._path = _DEFAULT_PATH

    @property
    def path(self) -> Path:
        return self._path

    def save(self, result: CalibrationResult) -> None:
        """Atomically write calibration results to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        data = result.to_dict()
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def load(self) -> CalibrationResult | None:
        """Load calibration results. Returns ``None`` if file does not exist."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise CalibrationError(f"Failed to read calibration file: {exc}") from exc
        return CalibrationResult.from_dict(data)

    def apply_to_registry(
        self,
        registry: Any,
        router: Any | None = None,
        *,
        min_samples: int = 10,
    ) -> dict[str, list[str]]:
        """Apply calibrated strengths to an ExpertRegistry.

        Updates ``ExpertSpec.entity_strengths`` in place for each
        (engine, entity_type) pair where the sample count meets the
        minimum threshold.

        Parameters
        ----------
        registry : ExpertRegistry
            The expert registry to update.
        router : MoERouter | None
            If provided, ``router.clear_cache()`` is called after updates.
        min_samples : int
            Minimum labeled examples required to accept calibrated F1.

        Returns
        -------
        dict[str, list[str]]
            Mapping of engine_id to list of entity types that were updated.
        """
        result = self.load()
        if result is None:
            return {}

        updated: dict[str, list[str]] = {}
        for engine_id, entity_f1 in result.engine_entity_f1.items():
            spec = registry.get_expert(engine_id)
            if spec is None:
                continue
            counts = result.sample_counts.get(engine_id, {})
            updated_types: list[str] = []
            for entity_type, f1 in entity_f1.items():
                if counts.get(entity_type, 0) >= min_samples:
                    spec.entity_strengths[entity_type] = f1
                    updated_types.append(entity_type)
            if updated_types:
                updated[engine_id] = updated_types

        if updated and router is not None:
            router.clear_cache()

        return updated
