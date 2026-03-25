"""Expert manifest loading and validation for MoE auto-registration.

Supports YAML and JSON manifest files co-located with engine adapter code.
Priority for profile resolution:
1. Manifest file on disk (``expert_manifest.yaml`` next to adapter)
2. ``engine.expert_profile()`` return value
3. Neither -> engine is not auto-registered in MoE
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

from pii_anon.errors import ExpertManifestError

logger = logging.getLogger(__name__)

_MANIFEST_FILENAMES = [
    "expert_manifest.yaml",
    "expert_manifest.yml",
    "expert_manifest.json",
]


class ExpertProfileData(TypedDict, total=False):
    """Typed shape for expert profile data (manifest or self-declaration).

    Required keys: ``expert_id``, ``display_name``, ``entity_strengths``.
    Optional keys have defaults applied during validation.
    """

    expert_id: str
    display_name: str
    entity_strengths: dict[str, float]
    entity_weaknesses: dict[str, float]
    default_weight: float
    metadata: dict[str, Any]


class ManifestLoader:
    """Load and validate expert manifest files."""

    def load_from_path(self, path: Path) -> ExpertProfileData:
        """Parse and validate a single manifest file.

        Parameters
        ----------
        path : Path
            Path to a YAML or JSON manifest file.

        Returns
        -------
        ExpertProfileData
            Validated profile data.

        Raises
        ------
        ExpertManifestError
            If the file cannot be parsed or fails validation.
        """
        if not path.exists():
            raise ExpertManifestError(f"Manifest file not found: {path}")

        text = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]

                data = yaml.safe_load(text)
            except ImportError:
                raise ExpertManifestError(f"PyYAML is required to load YAML manifests: {path}")
            except Exception as exc:
                raise ExpertManifestError(f"Failed to parse YAML manifest {path}: {exc}") from exc
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ExpertManifestError(f"Failed to parse JSON manifest {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ExpertManifestError(f"Manifest must be a mapping, got {type(data).__name__}: {path}")

        return self.validate(data, source=str(path))

    def find_manifest_for_adapter(self, adapter: Any) -> Path | None:
        """Find a co-located manifest file for an engine adapter.

        Looks for ``expert_manifest.{yaml,yml,json}`` in the same directory
        as the adapter's source module.

        Parameters
        ----------
        adapter : EngineAdapter
            Engine adapter instance.

        Returns
        -------
        Path | None
            Path to the manifest if found, ``None`` otherwise.
        """
        try:
            source_file = inspect.getfile(type(adapter))
        except (TypeError, OSError):
            return None

        adapter_dir = Path(source_file).parent

        # Try adapter-specific manifest first (expert_manifest_{adapter_id}.{ext})
        adapter_id = getattr(adapter, "adapter_id", "")
        if adapter_id:
            clean_id = adapter_id.replace("-", "_")
            for ext in ("yaml", "yml", "json"):
                specific = adapter_dir / f"expert_manifest_{clean_id}.{ext}"
                if specific.exists():
                    return specific

        # Fall back to generic manifest names
        for name in _MANIFEST_FILENAMES:
            candidate = adapter_dir / name
            if candidate.exists():
                return candidate

        return None

    def load_for_adapter(self, adapter: Any) -> ExpertProfileData | None:
        """Resolve profile data for an adapter (manifest > self-declaration).

        Parameters
        ----------
        adapter : EngineAdapter
            Engine adapter instance.

        Returns
        -------
        ExpertProfileData | None
            Resolved profile, or ``None`` if the engine opts out.
        """
        # Priority 1: manifest file on disk
        manifest_path = self.find_manifest_for_adapter(adapter)
        if manifest_path is not None:
            try:
                return self.load_from_path(manifest_path)
            except ExpertManifestError:
                logger.warning(
                    "manifest_load_failed",
                    extra={"adapter_id": getattr(adapter, "adapter_id", "?"), "path": str(manifest_path)},
                    exc_info=True,
                )

        # Priority 2: self-declaration via expert_profile()
        profile_fn = getattr(adapter, "expert_profile", None)
        if profile_fn is not None:
            profile = profile_fn()
            if profile is not None:
                return self.validate(dict(profile), source=f"{type(adapter).__name__}.expert_profile()")

        return None

    def validate(self, data: dict[str, Any], *, source: str = "<unknown>") -> ExpertProfileData:
        """Validate raw manifest data.

        Parameters
        ----------
        data : dict
            Raw manifest data.
        source : str
            Description of where the data came from (for error messages).

        Returns
        -------
        ExpertProfileData
            Validated and normalized profile data.

        Raises
        ------
        ExpertManifestError
            If required fields are missing or values are invalid.
        """
        # Check schema version if present
        schema_version = data.get("schema_version", "1.0")
        if isinstance(schema_version, str) and not schema_version.startswith("1."):
            raise ExpertManifestError(f"Unsupported manifest schema version {schema_version!r} in {source}")

        # Required fields
        expert_id = data.get("expert_id")
        if not expert_id or not isinstance(expert_id, str):
            raise ExpertManifestError(f"Missing or invalid 'expert_id' in {source}")

        display_name = data.get("display_name")
        if not display_name or not isinstance(display_name, str):
            raise ExpertManifestError(f"Missing or invalid 'display_name' in {source}")

        entity_strengths = data.get("entity_strengths", {})
        if not isinstance(entity_strengths, dict):
            raise ExpertManifestError(f"'entity_strengths' must be a mapping in {source}")

        # Validate strength values
        for key, val in entity_strengths.items():
            if not isinstance(val, (int, float)):
                raise ExpertManifestError(f"Strength for {key!r} must be numeric, got {type(val).__name__} in {source}")
            if val < 0.0 or val > 1.0:
                raise ExpertManifestError(f"Strength for {key!r} must be in [0.0, 1.0], got {val} in {source}")

        entity_weaknesses = data.get("entity_weaknesses", {})
        if not isinstance(entity_weaknesses, dict):
            entity_weaknesses = {}

        default_weight = data.get("default_weight", 1.0)
        if not isinstance(default_weight, (int, float)) or default_weight <= 0:
            raise ExpertManifestError(f"'default_weight' must be positive in {source}")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return ExpertProfileData(
            expert_id=expert_id,
            display_name=display_name,
            entity_strengths={k: float(v) for k, v in entity_strengths.items()},
            entity_weaknesses={k: float(v) for k, v in entity_weaknesses.items()},
            default_weight=float(default_weight),
            metadata=metadata,
        )
