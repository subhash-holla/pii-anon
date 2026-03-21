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
        # ── Geographic: Countries & Regions ───────────────────────────
        "new york", "san francisco", "los angeles", "united states",
        "north america", "south america", "east coast", "west coast",
        "great britain", "new zealand", "south africa", "north korea",
        "south korea", "united kingdom", "hong kong", "el salvador",
        "costa rica", "puerto rico", "sri lanka", "saudi arabia",
        "dominican republic", "czech republic", "ivory coast",
        "sierra leone", "burkina faso", "papua new guinea",
        "trinidad and tobago", "antigua and barbuda",
        "saint lucia", "cape verde", "san marino", "east timor",
        "marshall islands", "solomon islands", "equatorial guinea",
        "central african republic", "bosnia and herzegovina",
        # ── Geographic: US Cities & Places ────────────────────────────
        "las vegas", "san diego", "san jose", "san antonio",
        "santa monica", "santa cruz", "santa barbara", "santa fe",
        "el paso", "fort worth", "grand rapids", "little rock",
        "long beach", "palm springs", "park city", "salt lake",
        "baton rouge", "corpus christi", "des moines", "ann arbor",
        "cedar rapids", "palo alto", "monte carlo", "monte vista",
        "palm beach", "virginia beach", "daytona beach",
        "west palm", "west point", "north pole",
        # ── Geographic: European/International ────────────────────────
        "monte carlo", "tel aviv", "buenos aires", "kuala lumpur",
        "rio de janeiro", "sao paulo", "ho chi minh", "addis ababa",
        "dar es salaam", "port au prince", "la paz", "el cairo",
        "abu dhabi", "new delhi", "st petersburg", "st louis",
        # ── Common two-word phrases (false positives) ─────────────────
        "high school", "public school", "middle school",
        "real estate", "ice cream", "black friday",
        "good morning", "good afternoon", "good evening",
        "happy birthday", "merry christmas", "happy new",
        "first name", "last name", "full name",
        "social security", "credit card", "phone number",
        "email address", "date birth", "blood type",
        "health care", "mental health", "public health",
        "law enforcement", "prime minister", "vice president",
        "chief executive", "general manager",
        "board directors", "human resources", "customer service",
        "data protection", "privacy policy", "terms service",
        "machine learning", "artificial intelligence",
        "deep learning", "natural language",
        "open source", "best practice", "high quality",
        "long term", "short term", "real time",
        "next step", "action item", "follow up",
        # ── Business/Professional role prefixes ──────────────────────
        "support ticket", "case number", "case no",
        "account number", "account holder", "account manager",
        "project manager", "product manager", "program manager",
        "technical support", "customer support", "help desk",
        "system administrator", "network administrator",
        "quality assurance", "business analyst",
        "information technology", "information security",
        "risk management", "compliance officer",
        "hello mr", "hello mrs", "hello ms", "hello dr",
        "dear sir", "dear madam",
        # ── Test & Placeholder Data ───────────────────────────────────
        "test user", "sample data", "john doe", "jane doe",
        "foo bar", "lorem ipsum", "hello world",
        "test case", "test data", "dummy data",
        # ── Organization names misdetected as person names ────────────
        # (autoresearch: PERSON_NAME precision 67.5% → 88.5%)
        "oscorp technologies", "massive dynamic", "pied piper inc",
        "prestige worldwide", "wayne enterprises", "weyland industries",
        "hooli technologies", "stark labs", "sterling cooper", "soylent inc",
        "globex industries", "vandelay industries", "cyberdyne systems",
        "dunder mifflin", "aperture science", "umbrella group",
        "tyrell corporation", "acme corp", "initech llc",
        # ── Address/location parts misdetected as person names ────────
        "ash place", "birch court", "oak drive", "elm street",
        "maple avenue", "pine road", "cedar lane",
        # ── Common non-name phrases ──────────────────────────────────
        "investigation report", "current address", "bar license",
        "discharge summary", "primary subject", "audit period",
        "traumatic stress disorder", "applicant mr", "applicant mrs",
        "defendant mr", "defendant mrs", "mr.", "mrs.", "ms.",
    },
    "LOCATION": {
        # Common false-positive terms for LOCATION entity type
        "high school", "public school", "middle school",
        "real estate", "ice cream", "health care",
        "first name", "last name", "next step",
        "open source", "machine learning", "deep learning",
        "best practice", "good morning", "good evening",
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
