"""Configurable deny-list and allow-list for PII false-positive suppression.

The deny-list prevents known false positives from being reported (e.g.,
geographic names like "New York" detected as PERSON_NAME).  The allow-list
protects known safe values from ever being flagged (e.g., a company's own
public IP range).

Both lists are per-entity-type and support case-insensitive matching via
normalized ``set`` membership (O(1) lookup).

Configuration
-------------
Deny/allow-lists can be loaded from:

1. **Defaults** — ``DEFAULT_DENY_LISTS`` provides a curated set of geographic
   and test-data false positives for PERSON_NAME.
2. **Constructor** — Pass ``deny_list_config`` or ``allow_list_config`` dicts.
3. **Runtime** — Call ``DenyListManager.initialize()`` with a config dict,
   or ``add_entries()`` to extend programmatically.
"""

from __future__ import annotations

from typing import Any

# ── Default deny-list entries ──────────────────────────────────────────────
# Geographic names and test data that commonly trigger PERSON_NAME false
# positives due to their multi-word capitalized structure.

DEFAULT_DENY_LISTS: dict[str, set[str]] = {
    "PERSON_NAME": {
        "new york", "san francisco", "los angeles", "united states",
        "north america", "south america", "east coast", "west coast",
        "great britain", "new zealand", "south africa", "north korea",
        "south korea", "united kingdom", "hong kong", "el salvador",
        "costa rica", "puerto rico", "sri lanka", "saudi arabia",
        "test user", "sample data", "john doe", "jane doe",
    },
}


class DenyListManager:
    """Manages per-entity-type deny-lists and allow-lists.

    Parameters
    ----------
    deny_config:
        Optional config dict with ``enabled`` (bool) and ``lists``
        (dict mapping entity types to lists of denied terms).
    allow_config:
        Optional config dict with ``enabled`` (bool) and ``lists``
        (dict mapping entity types to lists of allowed terms).
    """

    def __init__(
        self,
        deny_config: dict[str, Any] | None = None,
        allow_config: dict[str, Any] | None = None,
    ) -> None:
        self._deny_lists: dict[str, set[str]] = {}
        self._allow_lists: dict[str, set[str]] = {}

        if deny_config:
            self._load_lists(deny_config, target=self._deny_lists)
        else:
            self._deny_lists = {k: set(v) for k, v in DEFAULT_DENY_LISTS.items()}

        if allow_config:
            self._load_lists(allow_config, target=self._allow_lists)

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Re-initialize from a configuration dict.

        Expected keys: ``deny_list`` and/or ``allow_list``, each with
        ``enabled`` (bool) and ``lists`` (dict[str, list[str]]).
        """
        if not config:
            return
        if "deny_list" in config:
            self._load_lists(config["deny_list"], target=self._deny_lists)
        if "allow_list" in config:
            self._load_lists(config["allow_list"], target=self._allow_lists)

    @staticmethod
    def _load_lists(cfg: dict[str, Any], *, target: dict[str, set[str]]) -> None:
        """Populate *target* from a config dict, clearing previous entries."""
        target.clear()
        if not cfg.get("enabled", True):
            return
        for entity_type, values in cfg.get("lists", {}).items():
            target[entity_type] = {v.lower() for v in values}

    def is_denied(self, entity_type: str, matched_text: str) -> bool:
        """Return *True* if *matched_text* is in the deny-list for *entity_type*.

        Matching is case-insensitive.
        """
        deny_set = self._deny_lists.get(entity_type)
        if not deny_set:
            return False
        return matched_text.lower() in deny_set

    def is_allowed(self, entity_type: str, matched_text: str) -> bool:
        """Return *True* if *matched_text* is in the allow-list for *entity_type*.

        When a value is allow-listed, it should be excluded from detection
        entirely (the inverse of deny-listing).
        """
        allow_set = self._allow_lists.get(entity_type)
        if not allow_set:
            return False
        return matched_text.lower() in allow_set

    def add_deny_entries(self, entity_type: str, entries: list[str]) -> None:
        """Add entries to the deny-list for *entity_type* at runtime."""
        if entity_type not in self._deny_lists:
            self._deny_lists[entity_type] = set()
        self._deny_lists[entity_type].update(v.lower() for v in entries)

    def add_allow_entries(self, entity_type: str, entries: list[str]) -> None:
        """Add entries to the allow-list for *entity_type* at runtime."""
        if entity_type not in self._allow_lists:
            self._allow_lists[entity_type] = set()
        self._allow_lists[entity_type].update(v.lower() for v in entries)
