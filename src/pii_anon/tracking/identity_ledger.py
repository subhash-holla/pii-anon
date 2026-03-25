from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ClusterState:
    cluster_id: str
    entity_family: str
    canonical_text: str
    placeholder_index: int
    aliases: set[str] = field(default_factory=set)
    full_name_aliases: set[str] = field(default_factory=set)
    short_name_aliases: set[str] = field(default_factory=set)
    last_name_aliases: set[str] = field(default_factory=set)
    email_local_aliases: set[str] = field(default_factory=set)


class IdentityLedger:
    """Session/document-scoped entity linkage ledger."""

    def __init__(self, max_scopes: int | None = None) -> None:
        self._clusters_by_scope: dict[str, dict[str, ClusterState]] = {}
        self._alias_to_cluster: dict[str, dict[str, str]] = {}
        self._counter_by_scope: dict[str, int] = {}
        self._clusters_by_family: dict[str, dict[str, dict[str, ClusterState]]] = {}
        self._max_scopes = max_scopes
        self._scope_order: list[str] = []

    def all_clusters(self, scope: str) -> list[ClusterState]:
        return list(self._clusters_by_scope.get(scope, {}).values())

    def clusters_by_family(self, scope: str, family: str) -> list[ClusterState]:
        return list(self._clusters_by_family.get(scope, {}).get(family, {}).values())

    def clear_scope(self, scope: str) -> None:
        """Remove all data for a given scope."""
        self._clusters_by_scope.pop(scope, None)
        self._alias_to_cluster.pop(scope, None)
        self._counter_by_scope.pop(scope, None)
        self._clusters_by_family.pop(scope, None)
        try:
            self._scope_order.remove(scope)
        except ValueError:
            pass

    def find_by_alias(self, scope: str, alias_norm: str) -> ClusterState | None:
        cluster_id = self._alias_to_cluster.get(scope, {}).get(alias_norm)
        if cluster_id is None:
            return None
        return self._clusters_by_scope.get(scope, {}).get(cluster_id)

    def create_cluster(
        self,
        scope: str,
        *,
        entity_family: str,
        canonical_text: str,
        alias_norm: str,
        full_name_alias: str | None,
        short_name_alias: str | None,
        last_name_alias: str | None,
        email_local_alias: str | None,
    ) -> ClusterState:
        if scope not in self._clusters_by_scope:
            self._scope_order.append(scope)
            if self._max_scopes is not None and len(self._scope_order) > self._max_scopes:
                evict = self._scope_order.pop(0)
                self.clear_scope(evict)
        count = self._counter_by_scope.get(scope, 0) + 1
        self._counter_by_scope[scope] = count
        cluster_id = f"{entity_family}-{count:04d}"
        state = ClusterState(
            cluster_id=cluster_id,
            entity_family=entity_family,
            canonical_text=canonical_text,
            placeholder_index=count,
            aliases={alias_norm},
        )
        if full_name_alias:
            state.full_name_aliases.add(full_name_alias)
        if short_name_alias:
            state.short_name_aliases.add(short_name_alias)
        if last_name_alias:
            state.last_name_aliases.add(last_name_alias)
        if email_local_alias:
            state.email_local_aliases.add(email_local_alias)

        self._clusters_by_scope.setdefault(scope, {})[cluster_id] = state
        self._alias_to_cluster.setdefault(scope, {})[alias_norm] = cluster_id
        self._clusters_by_family.setdefault(scope, {}).setdefault(entity_family, {})[cluster_id] = state
        return state

    def register_alias(
        self,
        scope: str,
        cluster: ClusterState,
        *,
        alias_norm: str,
        full_name_alias: str | None,
        short_name_alias: str | None,
        last_name_alias: str | None,
        email_local_alias: str | None,
        canonical_text_candidate: str,
    ) -> None:
        cluster.aliases.add(alias_norm)
        self._alias_to_cluster.setdefault(scope, {})[alias_norm] = cluster.cluster_id

        if full_name_alias:
            cluster.full_name_aliases.add(full_name_alias)
            # Prefer explicit full-name canonical labels over weaker aliases.
            if cluster.canonical_text.count(" ") < 1:
                cluster.canonical_text = canonical_text_candidate
        if short_name_alias:
            cluster.short_name_aliases.add(short_name_alias)
        if last_name_alias:
            cluster.last_name_aliases.add(last_name_alias)
        if email_local_alias:
            cluster.email_local_aliases.add(email_local_alias)
