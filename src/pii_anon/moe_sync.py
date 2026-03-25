"""MoE sync bridge: auto-register engines as MoE experts.

Mediates between ``EngineRegistry`` (execution) and ``ExpertRegistry``
(MoE gating weights). When an engine is registered/unregistered, the
bridge resolves its expert profile and updates the MoE registry.
"""

from __future__ import annotations

import logging
from typing import Any

from pii_anon.engines.manifest import ExpertProfileData, ManifestLoader
from pii_anon.moe import ExpertRegistry, ExpertSpec, MoERouter
from pii_anon.moe_similarity import ExpertSimilarityGuard

logger = logging.getLogger(__name__)


class MoeSyncBridge:
    """Bridge between EngineRegistry and ExpertRegistry.

    Resolves expert profiles from manifests or self-declaration, checks
    similarity, and maintains the MoE expert registry in sync with
    engine registration.

    Parameters
    ----------
    expert_registry : ExpertRegistry
        MoE expert registry to update.
    router : MoERouter | None
        Router whose cache is cleared on registration changes.
    similarity_guard : ExpertSimilarityGuard | None
        Guard for rejecting/warning on similar experts.
    manifest_loader : ManifestLoader | None
        Loader for disk-based manifests.
    default_strength : float
        Default entity strength for engines that declare entity types
        via capabilities but not via manifest/profile.
    """

    def __init__(
        self,
        expert_registry: ExpertRegistry,
        router: MoERouter | None = None,
        similarity_guard: ExpertSimilarityGuard | None = None,
        manifest_loader: ManifestLoader | None = None,
        default_strength: float = 0.50,
    ) -> None:
        self._expert_registry = expert_registry
        self._router = router
        self._similarity_guard = similarity_guard
        self._manifest_loader = manifest_loader or ManifestLoader()
        self._default_strength = default_strength

    def on_engine_registered(self, engine: Any) -> ExpertSpec | None:
        """Called after an engine is registered in EngineRegistry.

        Resolves the expert profile, checks similarity, and registers
        the expert in the MoE registry.

        Parameters
        ----------
        engine : EngineAdapter
            The newly registered engine.

        Returns
        -------
        ExpertSpec | None
            The registered expert spec, or ``None`` if the engine opts out.
        """
        adapter_id = getattr(engine, "adapter_id", None)
        if not adapter_id:
            return None

        # Don't overwrite existing experts (e.g., from build_default_registry)
        existing = self._expert_registry.get_expert(adapter_id)
        if existing is not None:
            return existing

        # Resolve profile: manifest > expert_profile() > capabilities fallback
        profile = self._manifest_loader.load_for_adapter(engine)

        if profile is None:
            # Fall back to capabilities-based seeding
            caps = engine.capabilities()
            entity_types = getattr(caps, "supported_entity_types", None)
            if entity_types:
                profile = ExpertProfileData(
                    expert_id=adapter_id,
                    display_name=adapter_id,
                    entity_strengths={et: self._default_strength for et in entity_types},
                )
            else:
                # Engine opts out of MoE entirely
                return None

        spec = self._profile_to_spec(profile)

        # Similarity check
        if self._similarity_guard is not None:
            try:
                self._similarity_guard.check(
                    spec,
                    self._expert_registry.list_experts(),
                )
            except Exception:
                logger.warning(
                    "expert_registration_blocked",
                    extra={"adapter_id": adapter_id},
                    exc_info=True,
                )
                return None

        self._expert_registry.register_expert(spec)

        if self._router is not None:
            self._router.clear_cache()

        logger.debug(
            "moe_expert_auto_registered",
            extra={
                "adapter_id": adapter_id,
                "entity_types": list(spec.entity_strengths.keys()),
            },
        )
        return spec

    def on_engine_unregistered(self, adapter_id: str) -> None:
        """Called after an engine is unregistered from EngineRegistry.

        Removes the corresponding expert from the MoE registry.

        Parameters
        ----------
        adapter_id : str
            The unregistered engine's ID.
        """
        try:
            self._expert_registry.unregister_expert(adapter_id)
        except KeyError:
            pass

        if self._router is not None:
            self._router.clear_cache()

    def _profile_to_spec(self, profile: ExpertProfileData) -> ExpertSpec:
        """Convert an ExpertProfileData dict to an ExpertSpec dataclass."""
        return ExpertSpec(
            expert_id=profile["expert_id"],
            display_name=profile.get("display_name", profile["expert_id"]),
            entity_strengths=dict(profile.get("entity_strengths", {})),
            entity_weaknesses=dict(profile.get("entity_weaknesses", {})),
            default_weight=float(profile.get("default_weight", 1.0)),
            is_available=True,
            metadata=dict(profile.get("metadata", {})),
        )


def create_default_bridge(
    expert_registry: ExpertRegistry,
    router: MoERouter | None = None,
    *,
    similarity_threshold: float = 0.95,
    similarity_action: str = "warn",
) -> MoeSyncBridge:
    """Create a bridge with default configuration.

    Parameters
    ----------
    expert_registry : ExpertRegistry
        MoE expert registry.
    router : MoERouter | None
        Router for cache invalidation.
    similarity_threshold : float
        Cosine similarity threshold for duplicate detection.
    similarity_action : str
        Action on similarity violation (``"warn"`` or ``"reject"``).
    """
    guard = ExpertSimilarityGuard(
        threshold=similarity_threshold,
        action=similarity_action,  # type: ignore[arg-type]
    )
    return MoeSyncBridge(
        expert_registry=expert_registry,
        router=router,
        similarity_guard=guard,
    )
