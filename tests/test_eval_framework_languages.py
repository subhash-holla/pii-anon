"""Tests for pii_anon.eval_framework.languages module.

Validates:
- 52 language profiles aligned with OpenNER 1.0
- Script coverage for major writing systems
- Resource-level classification
- Query methods
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.languages import (
    LanguageProfile,
    ResourceLevel,
    Script,
    SUPPORTED_LANGUAGES,
)


class TestSupportedLanguages:
    """SUPPORTED_LANGUAGES registry validation."""

    def test_count_at_least_52(self) -> None:
        """OpenNER 1.0 covers 52 languages; we must match or exceed."""
        assert SUPPORTED_LANGUAGES.count() >= 52

    def test_all_returns_list_of_profiles(self) -> None:
        profiles = SUPPORTED_LANGUAGES.all()
        assert len(profiles) >= 52
        assert all(isinstance(p, LanguageProfile) for p in profiles)

    def test_all_codes_unique(self) -> None:
        codes = SUPPORTED_LANGUAGES.all_codes()
        assert len(codes) == len(set(codes))

    def test_get_english(self) -> None:
        en = SUPPORTED_LANGUAGES.get("en")
        assert en is not None
        assert en.name == "English"
        assert en.script == Script.LATIN
        assert en.resource_level == ResourceLevel.HIGH

    def test_get_nonexistent_returns_none(self) -> None:
        assert SUPPORTED_LANGUAGES.get("xx") is None

    def test_major_languages_present(self) -> None:
        """Verify presence of major world languages."""
        for code in ["en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi", "pt", "ru"]:
            lang = SUPPORTED_LANGUAGES.get(code)
            assert lang is not None, f"Missing language: {code}"


class TestScriptCoverage:
    """Verify writing system coverage."""

    def test_at_least_10_scripts(self) -> None:
        """Major scripts: Latin, Cyrillic, Arabic, Devanagari, CJK, Hangul, etc."""
        assert len(Script) >= 10

    def test_by_script_latin(self) -> None:
        latin = SUPPORTED_LANGUAGES.by_script(Script.LATIN)
        assert len(latin) >= 15  # Most European languages use Latin

    def test_by_script_cyrillic(self) -> None:
        cyrillic = SUPPORTED_LANGUAGES.by_script(Script.CYRILLIC)
        assert len(cyrillic) >= 2  # At least Russian, Ukrainian

    def test_by_script_arabic(self) -> None:
        arabic = SUPPORTED_LANGUAGES.by_script(Script.ARABIC)
        assert len(arabic) >= 2  # Arabic, Farsi, Urdu


class TestResourceLevel:
    """Verify resource-level classification."""

    def test_four_resource_levels(self) -> None:
        assert len(ResourceLevel) == 4

    def test_high_resource_includes_english(self) -> None:
        high = SUPPORTED_LANGUAGES.by_resource_level(ResourceLevel.HIGH)
        codes = {lang_profile.iso639_1 for lang_profile in high}
        assert "en" in codes

    def test_low_resource_languages_exist(self) -> None:
        low = SUPPORTED_LANGUAGES.by_resource_level(ResourceLevel.LOW)
        very_low = SUPPORTED_LANGUAGES.by_resource_level(ResourceLevel.VERY_LOW)
        assert len(low) + len(very_low) > 0, "Should have under-resourced languages"


class TestLanguageFamilies:
    """Verify language family coverage."""

    def test_by_family_germanic(self) -> None:
        germanic = SUPPORTED_LANGUAGES.by_family("Germanic")
        assert len(germanic) >= 3  # English, German, Dutch, ...

    def test_by_family_romance(self) -> None:
        romance = SUPPORTED_LANGUAGES.by_family("Romance")
        assert len(romance) >= 4  # Spanish, French, Italian, Portuguese, Romanian

    def test_by_family_case_insensitive(self) -> None:
        a = SUPPORTED_LANGUAGES.by_family("germanic")
        b = SUPPORTED_LANGUAGES.by_family("Germanic")
        assert len(a) == len(b)


class TestOpenNERAlignment:
    """Verify alignment with OpenNER 1.0 benchmark."""

    def test_openner_supported_count(self) -> None:
        supported = SUPPORTED_LANGUAGES.openner_supported()
        assert len(supported) >= 40  # Most of our 52 should be OpenNER-supported

    def test_openner_flag_set_correctly(self) -> None:
        en = SUPPORTED_LANGUAGES.get("en")
        assert en is not None
        assert en.openner_supported is True


class TestLanguageProfileDataclass:
    """Verify LanguageProfile is frozen and well-formed."""

    def test_frozen(self) -> None:
        en = SUPPORTED_LANGUAGES.get("en")
        assert en is not None
        with pytest.raises(AttributeError):
            en.name = "Modified"  # type: ignore[misc]

    def test_profile_fields(self) -> None:
        en = SUPPORTED_LANGUAGES.get("en")
        assert en is not None
        assert en.iso639_1 == "en"
        assert len(en.iso639_3) == 3
        assert en.native_name != ""
        assert isinstance(en.script, Script)
        assert isinstance(en.resource_level, ResourceLevel)
        assert en.family != ""
