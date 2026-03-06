"""Tests for the allow-list system (AllowListConfig and DenyListManager).

Covers: allow-list configuration, case-insensitive matching, integration
with detect() to exclude allowed values from detection, and backward compatibility.
"""

from __future__ import annotations

from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.engines.regex.deny_list import DenyListManager
from pii_anon.config.schema import AllowListConfig


# ═══════════════════════════════════════════════════════════════════════════
# AllowListConfig Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAllowListConfig:
    """Test AllowListConfig defaults and initialization."""

    def test_default_config_disabled(self) -> None:
        """AllowListConfig should be disabled by default."""
        config = AllowListConfig()
        assert config.enabled is False

    def test_default_config_empty_lists(self) -> None:
        """AllowListConfig should have empty lists by default."""
        config = AllowListConfig()
        assert config.lists == {}

    def test_enable_allowlist(self) -> None:
        """AllowListConfig can be enabled."""
        config = AllowListConfig(enabled=True)
        assert config.enabled is True

    def test_add_allowlist_entries(self) -> None:
        """AllowListConfig can have custom lists."""
        config = AllowListConfig(
            enabled=True,
            lists={
                "EMAIL_ADDRESS": ["admin@company.com", "support@company.com"],
                "PHONE_NUMBER": ["555-1212"],
            },
        )
        assert config.enabled is True
        assert "EMAIL_ADDRESS" in config.lists
        assert len(config.lists["EMAIL_ADDRESS"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# DenyListManager Allow-list Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDenyListManagerAllowList:
    """Test DenyListManager's allow-list functionality."""

    def test_manager_accepts_allow_config(self) -> None:
        """DenyListManager should accept allow_config."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert mgr._allow_lists is not None

    def test_manager_is_allowed_method(self) -> None:
        """DenyListManager should have is_allowed() method."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert hasattr(mgr, "is_allowed")
        assert callable(mgr.is_allowed)

    def test_is_allowed_returns_bool(self) -> None:
        """is_allowed() should return a boolean."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        result = mgr.is_allowed("EMAIL_ADDRESS", "admin@company.com")
        assert isinstance(result, bool)

    def test_is_allowed_returns_true_for_allowed(self) -> None:
        """is_allowed() should return True for allowed entries."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert mgr.is_allowed("EMAIL_ADDRESS", "admin@company.com")

    def test_is_allowed_returns_false_for_not_allowed(self) -> None:
        """is_allowed() should return False for non-allowed entries."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert not mgr.is_allowed("EMAIL_ADDRESS", "user@example.com")

    def test_is_allowed_case_insensitive(self) -> None:
        """is_allowed() should be case-insensitive."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert mgr.is_allowed("EMAIL_ADDRESS", "ADMIN@COMPANY.COM")
        assert mgr.is_allowed("EMAIL_ADDRESS", "Admin@Company.Com")

    def test_add_allow_entries(self) -> None:
        """DenyListManager should support add_allow_entries()."""
        mgr = DenyListManager()
        mgr.add_allow_entries("EMAIL_ADDRESS", ["test@example.com"])
        assert mgr.is_allowed("EMAIL_ADDRESS", "test@example.com")

    def test_add_allow_entries_multiple(self) -> None:
        """add_allow_entries() should support multiple entries."""
        mgr = DenyListManager()
        mgr.add_allow_entries(
            "EMAIL_ADDRESS",
            ["test1@example.com", "test2@example.com"],
        )
        assert mgr.is_allowed("EMAIL_ADDRESS", "test1@example.com")
        assert mgr.is_allowed("EMAIL_ADDRESS", "test2@example.com")

    def test_is_allowed_empty_allow_list(self) -> None:
        """is_allowed() should return False when allow-list is empty."""
        mgr = DenyListManager()
        assert not mgr.is_allowed("EMAIL_ADDRESS", "any@example.com")

    def test_is_allowed_disabled_allow_list(self) -> None:
        """is_allowed() should return False when allow-list is disabled."""
        allow_config = {
            "enabled": False,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert not mgr.is_allowed("EMAIL_ADDRESS", "admin@company.com")

    def test_is_allowed_multiple_entity_types(self) -> None:
        """Allow-list should work for multiple entity types."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
                "PHONE_NUMBER": ["555-1212"],
                "IP_ADDRESS": ["192.168.1.1"],
            },
        }
        mgr = DenyListManager(allow_config=allow_config)
        assert mgr.is_allowed("EMAIL_ADDRESS", "admin@company.com")
        assert mgr.is_allowed("PHONE_NUMBER", "555-1212")
        assert mgr.is_allowed("IP_ADDRESS", "192.168.1.1")
        assert not mgr.is_allowed("EMAIL_ADDRESS", "other@example.com")


# ═══════════════════════════════════════════════════════════════════════════
# RegexEngineAdapter Allow-list Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestRegexEngineAdapterAllowList:
    """Test allow-list integration with RegexEngineAdapter.detect()."""

    def test_adapter_accepts_allow_list_config(self) -> None:
        """RegexEngineAdapter should accept allow_list_config."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        assert adapter is not None

    def test_allowed_email_not_detected(self) -> None:
        """Allowed email should not be detected."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Email me at admin@company.com"},
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) == 0

    def test_non_allowed_email_detected(self) -> None:
        """Non-allowed email should still be detected."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Email me at user@example.com"},
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) >= 1
        assert emails[0].confidence > 0

    def test_allowed_phone_not_detected(self) -> None:
        """Allowed phone number should not be detected."""
        allow_config = {
            "enabled": True,
            "lists": {
                "PHONE_NUMBER": ["555-1212"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Call 555-1212"},
            {"language": "en"},
        )
        phones = [f for f in findings if f.entity_type == "PHONE_NUMBER"]
        assert len(phones) == 0

    def test_non_allowed_phone_detected(self) -> None:
        """Non-allowed phone number should still be detected."""
        allow_config = {
            "enabled": True,
            "lists": {
                "PHONE_NUMBER": ["555-1212"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Call (555) 123-4567"},
            {"language": "en"},
        )
        phones = [f for f in findings if f.entity_type == "PHONE_NUMBER"]
        assert len(phones) >= 1

    def test_allowed_ip_not_detected(self) -> None:
        """Allowed IP address should not be detected."""
        allow_config = {
            "enabled": True,
            "lists": {
                "IP_ADDRESS": ["192.168.1.1"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Server at 192.168.1.1"},
            {"language": "en"},
        )
        ips = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        assert len(ips) == 0

    def test_non_allowed_ip_detected(self) -> None:
        """Non-allowed IP address should still be detected."""
        allow_config = {
            "enabled": True,
            "lists": {
                "IP_ADDRESS": ["192.168.1.1"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Server at 192.168.1.2"},
            {"language": "en"},
        )
        ips = [f for f in findings if f.entity_type == "IP_ADDRESS"]
        assert len(ips) >= 1

    def test_allow_list_case_insensitive_in_detect(self) -> None:
        """Allow-list matching in detect() should be case-insensitive."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Email: ADMIN@COMPANY.COM"},
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) == 0  # Should be filtered out (case-insensitive)

    def test_allow_list_multiple_entries(self) -> None:
        """Multiple entries in allow-list should all be filtered."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": [
                    "admin@company.com",
                    "support@company.com",
                    "info@company.com",
                ],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {
                "text": "Emails: admin@company.com, support@company.com, info@company.com, user@example.com"
            },
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        # Should only detect user@example.com
        assert len(emails) >= 1
        assert all(e.field_path == "text" for e in emails)

    def test_disabled_allow_list_does_not_filter(self) -> None:
        """Disabled allow-list should not filter detections."""
        allow_config = {
            "enabled": False,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {"text": "Email me at admin@company.com"},
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) >= 1  # Should be detected despite being "allowed"

    def test_allow_list_for_different_entity_types(self) -> None:
        """Allow-list should work independently for different entity types."""
        allow_config = {
            "enabled": True,
            "lists": {
                "EMAIL_ADDRESS": ["admin@company.com"],
            },
        }
        adapter = RegexEngineAdapter(allow_list_config=allow_config)
        findings = adapter.detect(
            {
                "text": "Contact: admin@company.com or user@example.com"
            },
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        # admin@company.com should be allowed (filtered out)
        # user@example.com should still be detected
        assert len(emails) >= 1
        matched_texts = [
            "Contact: admin@company.com or user@example.com"[f.span_start:f.span_end]
            for f in emails
        ]
        assert "admin@company.com" not in matched_texts
        assert "user@example.com" in matched_texts


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAllowListBackwardCompatibility:
    """Test allow-list backward compatibility with existing code."""

    def test_adapter_without_allow_config(self) -> None:
        """Adapter should work without allow_list_config."""
        adapter = RegexEngineAdapter()
        findings = adapter.detect(
            {"text": "Email: admin@company.com"},
            {"language": "en"},
        )
        # Should detect normally
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) >= 1

    def test_allow_list_config_none(self) -> None:
        """Adapter should accept allow_list_config=None."""
        adapter = RegexEngineAdapter(allow_list_config=None)
        findings = adapter.detect(
            {"text": "Email: admin@company.com"},
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) >= 1

    def test_initialize_with_allow_list(self) -> None:
        """Adapter.initialize() should accept allow_list config."""
        adapter = RegexEngineAdapter()
        config = {
            "allow_list": {
                "enabled": True,
                "lists": {
                    "EMAIL_ADDRESS": ["admin@company.com"],
                },
            },
        }
        adapter.initialize(config)
        findings = adapter.detect(
            {"text": "Email: admin@company.com"},
            {"language": "en"},
        )
        emails = [f for f in findings if f.entity_type == "EMAIL_ADDRESS"]
        assert len(emails) == 0
