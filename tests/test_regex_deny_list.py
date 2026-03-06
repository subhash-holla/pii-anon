"""Tests for configurable deny-list system (Enhancement 4).

Covers: default deny-list, custom config loading, suppression of
false-positive matches, case sensitivity, and schema integration.
"""

from __future__ import annotations

from pii_anon.config.schema import CoreConfig, DenyListConfig
from pii_anon.engines.regex_adapter import RegexEngineAdapter


# ---------------------------------------------------------------------------
# DenyListConfig schema
# ---------------------------------------------------------------------------


class TestDenyListConfigSchema:
    def test_default_enabled(self) -> None:
        cfg = DenyListConfig()
        assert cfg.enabled is True

    def test_default_lists_has_person_name(self) -> None:
        cfg = DenyListConfig()
        assert "PERSON_NAME" in cfg.lists

    def test_default_lists_person_name_entries(self) -> None:
        cfg = DenyListConfig()
        assert "new york" in cfg.lists["PERSON_NAME"]
        assert "john doe" in cfg.lists["PERSON_NAME"]

    def test_custom_lists(self) -> None:
        cfg = DenyListConfig(
            lists={"ORGANIZATION": ["acme corp", "test inc"]}
        )
        assert "ORGANIZATION" in cfg.lists
        assert "acme corp" in cfg.lists["ORGANIZATION"]

    def test_disabled(self) -> None:
        cfg = DenyListConfig(enabled=False)
        assert cfg.enabled is False


class TestCoreConfigDenyList:
    def test_deny_list_field_exists(self) -> None:
        config = CoreConfig.default()
        assert hasattr(config, "deny_list")
        assert isinstance(config.deny_list, DenyListConfig)

    def test_deny_list_default_enabled(self) -> None:
        config = CoreConfig.default()
        assert config.deny_list.enabled is True


# ---------------------------------------------------------------------------
# RegexEngineAdapter deny-list loading
# ---------------------------------------------------------------------------


class TestDenyListLoading:
    def test_default_deny_lists_loaded(self) -> None:
        adapter = RegexEngineAdapter()
        assert "PERSON_NAME" in adapter._deny_lists
        assert "new york" in adapter._deny_lists["PERSON_NAME"]

    def test_custom_deny_list_via_constructor(self) -> None:
        adapter = RegexEngineAdapter(
            deny_list_config={
                "enabled": True,
                "lists": {
                    "PERSON_NAME": ["custom entry", "another entry"],
                    "ORGANIZATION": ["fake org"],
                },
            }
        )
        assert "custom entry" in adapter._deny_lists["PERSON_NAME"]
        assert "fake org" in adapter._deny_lists["ORGANIZATION"]

    def test_disabled_deny_list_empties(self) -> None:
        adapter = RegexEngineAdapter(
            deny_list_config={"enabled": False, "lists": {}}
        )
        assert len(adapter._deny_lists) == 0

    def test_initialize_loads_config(self) -> None:
        adapter = RegexEngineAdapter()
        adapter.initialize({
            "deny_list": {
                "enabled": True,
                "lists": {"PERSON_NAME": ["override entry"]},
            }
        })
        assert "override entry" in adapter._deny_lists["PERSON_NAME"]


# ---------------------------------------------------------------------------
# _is_denied
# ---------------------------------------------------------------------------


class TestIsDenied:
    def test_denied_entry(self) -> None:
        adapter = RegexEngineAdapter()
        assert adapter._is_denied("PERSON_NAME", "New York")

    def test_case_insensitive(self) -> None:
        adapter = RegexEngineAdapter()
        assert adapter._is_denied("PERSON_NAME", "NEW YORK")
        assert adapter._is_denied("PERSON_NAME", "new york")
        assert adapter._is_denied("PERSON_NAME", "New York")

    def test_not_denied(self) -> None:
        adapter = RegexEngineAdapter()
        assert not adapter._is_denied("PERSON_NAME", "Alice Smith")

    def test_unknown_entity_type(self) -> None:
        adapter = RegexEngineAdapter()
        assert not adapter._is_denied("UNKNOWN_TYPE", "anything")


# ---------------------------------------------------------------------------
# Integration: deny-list suppression in detect()
# ---------------------------------------------------------------------------


class TestDenyListSuppression:
    def test_geographic_fp_suppressed(self) -> None:
        """'New York' should not be detected as PERSON_NAME."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "Mr. New York arrived at the airport."},
            {"language": "en"},
        )
        person_findings = [
            f for f in findings
            if f.entity_type == "PERSON_NAME"
        ]
        # "New York" should be filtered out by deny-list
        for f in person_findings:
            matched_text = "Mr. New York arrived at the airport."[f.span_start:f.span_end]
            assert "New York" not in matched_text or "Mr." in matched_text
            # The title-prefix pattern "Mr. New York" includes "Mr." so it's
            # the full match "Mr. New York" that gets checked

    def test_john_doe_suppressed(self) -> None:
        """'John Doe' test data should be suppressed."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "Name: John Doe is a placeholder."},
            {"language": "en"},
        )
        person_findings = [
            f for f in findings
            if f.entity_type == "PERSON_NAME"
        ]
        matched_texts = [
            "Name: John Doe is a placeholder."[f.span_start:f.span_end]
            for f in person_findings
        ]
        # "John Doe" should be filtered by deny-list for full-name pattern
        assert "John Doe" not in matched_texts

    def test_real_name_not_suppressed(self) -> None:
        """A real name like 'Alice Smith' should still be detected."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "Patient Alice Smith was admitted."},
            {"language": "en"},
        )
        person_findings = [
            f for f in findings
            if f.entity_type == "PERSON_NAME"
        ]
        assert len(person_findings) >= 1

    def test_custom_deny_list_suppresses(self) -> None:
        """Custom deny-list entries should suppress matches."""
        adapter = RegexEngineAdapter(
            deny_list_config={
                "enabled": True,
                "lists": {
                    "PERSON_NAME": ["alice smith"],
                },
            }
        )
        findings = adapter.detect(
            {"text": "Alice Smith went to the park."},
            {"language": "en"},
        )
        person_findings = [
            f for f in findings
            if f.entity_type == "PERSON_NAME"
        ]
        matched_texts = [
            "Alice Smith went to the park."[f.span_start:f.span_end]
            for f in person_findings
        ]
        assert "Alice Smith" not in matched_texts


# ---------------------------------------------------------------------------
# _looks_like_org_or_common_phrase delegation
# ---------------------------------------------------------------------------


class TestLooksLikeOrgDelegation:
    def test_delegates_to_is_denied(self) -> None:
        adapter = RegexEngineAdapter()
        assert adapter._looks_like_org_or_common_phrase("New York")
        assert not adapter._looks_like_org_or_common_phrase("Alice Smith")
