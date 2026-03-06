"""52 language profiles aligned with OpenNER 1.0 (2024) coverage.

Evidence basis:
- OpenNER 1.0 (Malmasi et al., 2024): Standardised NER datasets in 52 languages
- FiNERweb (2024): Coverage of 25+ scripts and long-tail languages
- The Bitter Lesson from 2,000+ Multilingual Benchmarks (2024):
  analysed resource distribution gaps across languages
- ISO 639-1 / 639-3: Language code standards
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Script(str, Enum):
    """Major writing systems covered by the evaluation framework."""

    LATIN = "Latin"
    CYRILLIC = "Cyrillic"
    ARABIC = "Arabic"
    DEVANAGARI = "Devanagari"
    CJK = "CJK"
    HANGUL = "Hangul"
    THAI = "Thai"
    GREEK = "Greek"
    HEBREW = "Hebrew"
    ETHIOPIC = "Ethiopic"
    GEORGIAN = "Georgian"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    BENGALI = "Bengali"
    KHMER = "Khmer"
    LAO = "Lao"
    MYANMAR = "Myanmar"


class ResourceLevel(str, Enum):
    """NER resource availability per OpenNER 1.0 classification.

    HIGH    – abundant annotated corpora, strong model baselines
    MEDIUM  – some corpora exist, community-maintained
    LOW     – sparse annotations, transfer-learning dependent
    VERY_LOW – near-zero dedicated NER resources
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass(frozen=True)
class LanguageProfile:
    """Immutable metadata for one of the 52 supported languages."""

    iso639_1: str
    iso639_3: str
    name: str
    native_name: str
    script: Script
    family: str
    resource_level: ResourceLevel
    openner_supported: bool
    locale_variants: tuple[str, ...] = ()
    notes: str = ""


# ---------------------------------------------------------------------------
# 52 Language profiles (sorted by family / region)
# ---------------------------------------------------------------------------

_PROFILES: tuple[LanguageProfile, ...] = (
    # ── Western European (12) ──────────────────────────────────────────
    LanguageProfile("en", "eng", "English", "English", Script.LATIN, "Indo-European / Germanic", ResourceLevel.HIGH, True, ("en_US", "en_GB", "en_AU", "en_IN")),
    LanguageProfile("es", "spa", "Spanish", "Espa\u00f1ol", Script.LATIN, "Indo-European / Romance", ResourceLevel.HIGH, True, ("es_ES", "es_MX", "es_AR")),
    LanguageProfile("fr", "fra", "French", "Fran\u00e7ais", Script.LATIN, "Indo-European / Romance", ResourceLevel.HIGH, True, ("fr_FR", "fr_CA", "fr_BE")),
    LanguageProfile("de", "deu", "German", "Deutsch", Script.LATIN, "Indo-European / Germanic", ResourceLevel.HIGH, True, ("de_DE", "de_AT", "de_CH")),
    LanguageProfile("it", "ita", "Italian", "Italiano", Script.LATIN, "Indo-European / Romance", ResourceLevel.HIGH, True, ("it_IT",)),
    LanguageProfile("pt", "por", "Portuguese", "Portugu\u00eas", Script.LATIN, "Indo-European / Romance", ResourceLevel.HIGH, True, ("pt_BR", "pt_PT")),
    LanguageProfile("nl", "nld", "Dutch", "Nederlands", Script.LATIN, "Indo-European / Germanic", ResourceLevel.HIGH, True, ("nl_NL", "nl_BE")),
    LanguageProfile("el", "ell", "Greek", "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac", Script.GREEK, "Indo-European / Hellenic", ResourceLevel.MEDIUM, True),
    LanguageProfile("sv", "swe", "Swedish", "Svenska", Script.LATIN, "Indo-European / Germanic", ResourceLevel.MEDIUM, True),
    LanguageProfile("no", "nor", "Norwegian", "Norsk", Script.LATIN, "Indo-European / Germanic", ResourceLevel.MEDIUM, True, ("nb_NO", "nn_NO")),
    LanguageProfile("da", "dan", "Danish", "Dansk", Script.LATIN, "Indo-European / Germanic", ResourceLevel.MEDIUM, True),
    LanguageProfile("fi", "fin", "Finnish", "Suomi", Script.LATIN, "Uralic / Finnic", ResourceLevel.MEDIUM, True),
    # ── Eastern European (8) ───────────────────────────────────────────
    LanguageProfile("pl", "pol", "Polish", "Polski", Script.LATIN, "Indo-European / Slavic", ResourceLevel.MEDIUM, True),
    LanguageProfile("cs", "ces", "Czech", "\u010ce\u0161tina", Script.LATIN, "Indo-European / Slavic", ResourceLevel.MEDIUM, True),
    LanguageProfile("hu", "hun", "Hungarian", "Magyar", Script.LATIN, "Uralic / Ugric", ResourceLevel.MEDIUM, True),
    LanguageProfile("ro", "ron", "Romanian", "Rom\u00e2n\u0103", Script.LATIN, "Indo-European / Romance", ResourceLevel.MEDIUM, True),
    LanguageProfile("ru", "rus", "Russian", "\u0420\u0443\u0441\u0441\u043a\u0438\u0439", Script.CYRILLIC, "Indo-European / Slavic", ResourceLevel.HIGH, True),
    LanguageProfile("uk", "ukr", "Ukrainian", "\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430", Script.CYRILLIC, "Indo-European / Slavic", ResourceLevel.MEDIUM, True),
    LanguageProfile("bg", "bul", "Bulgarian", "\u0411\u044a\u043b\u0433\u0430\u0440\u0441\u043a\u0438", Script.CYRILLIC, "Indo-European / Slavic", ResourceLevel.MEDIUM, True),
    LanguageProfile("sr", "srp", "Serbian", "\u0421\u0440\u043f\u0441\u043a\u0438", Script.CYRILLIC, "Indo-European / Slavic", ResourceLevel.LOW, True),
    # ── East Asian (6) ─────────────────────────────────────────────────
    LanguageProfile("zh", "zho", "Chinese", "\u4e2d\u6587", Script.CJK, "Sino-Tibetan", ResourceLevel.HIGH, True, ("zh_CN", "zh_TW")),
    LanguageProfile("ja", "jpn", "Japanese", "\u65e5\u672c\u8a9e", Script.CJK, "Japonic", ResourceLevel.HIGH, True),
    LanguageProfile("ko", "kor", "Korean", "\ud55c\uad6d\uc5b4", Script.HANGUL, "Koreanic", ResourceLevel.HIGH, True),
    LanguageProfile("th", "tha", "Thai", "\u0e44\u0e17\u0e22", Script.THAI, "Kra\u2013Dai", ResourceLevel.MEDIUM, True),
    LanguageProfile("vi", "vie", "Vietnamese", "Ti\u1ebfng Vi\u1ec7t", Script.LATIN, "Austroasiatic", ResourceLevel.MEDIUM, True),
    LanguageProfile("id", "ind", "Indonesian", "Bahasa Indonesia", Script.LATIN, "Austronesian", ResourceLevel.MEDIUM, True),
    # ── South Asian (4) ────────────────────────────────────────────────
    LanguageProfile("hi", "hin", "Hindi", "\u0939\u093f\u0928\u094d\u0926\u0940", Script.DEVANAGARI, "Indo-European / Indo-Aryan", ResourceLevel.MEDIUM, True),
    LanguageProfile("bn", "ben", "Bengali", "\u09ac\u09be\u0982\u09b2\u09be", Script.BENGALI, "Indo-European / Indo-Aryan", ResourceLevel.MEDIUM, True),
    LanguageProfile("ta", "tam", "Tamil", "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd", Script.TAMIL, "Dravidian", ResourceLevel.LOW, True),
    LanguageProfile("te", "tel", "Telugu", "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41", Script.TELUGU, "Dravidian", ResourceLevel.LOW, True),
    # ── Middle Eastern & North African (6) ─────────────────────────────
    LanguageProfile("ar", "ara", "Arabic", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629", Script.ARABIC, "Afro-Asiatic / Semitic", ResourceLevel.MEDIUM, True, ("ar_SA", "ar_EG", "ar_MA")),
    LanguageProfile("he", "heb", "Hebrew", "\u05e2\u05d1\u05e8\u05d9\u05ea", Script.HEBREW, "Afro-Asiatic / Semitic", ResourceLevel.MEDIUM, True),
    LanguageProfile("fa", "fas", "Persian", "\u0641\u0627\u0631\u0633\u06cc", Script.ARABIC, "Indo-European / Iranian", ResourceLevel.LOW, True),
    LanguageProfile("tr", "tur", "Turkish", "T\u00fcrk\u00e7e", Script.LATIN, "Turkic", ResourceLevel.MEDIUM, True),
    LanguageProfile("ur", "urd", "Urdu", "\u0627\u0631\u062f\u0648", Script.ARABIC, "Indo-European / Indo-Aryan", ResourceLevel.LOW, True),
    LanguageProfile("ps", "pus", "Pashto", "\u067e\u069a\u062a\u0648", Script.ARABIC, "Indo-European / Iranian", ResourceLevel.VERY_LOW, False),
    # ── Sub-Saharan African (8) ────────────────────────────────────────
    LanguageProfile("sw", "swa", "Swahili", "Kiswahili", Script.LATIN, "Niger\u2013Congo / Bantu", ResourceLevel.LOW, True),
    LanguageProfile("am", "amh", "Amharic", "\u12a0\u121b\u122d\u129b", Script.ETHIOPIC, "Afro-Asiatic / Semitic", ResourceLevel.LOW, True),
    LanguageProfile("yo", "yor", "Yoruba", "Yor\u00f9b\u00e1", Script.LATIN, "Niger\u2013Congo / Volta\u2013Niger", ResourceLevel.VERY_LOW, True),
    LanguageProfile("ig", "ibo", "Igbo", "As\u1ee5s\u1ee5 Igbo", Script.LATIN, "Niger\u2013Congo / Volta\u2013Niger", ResourceLevel.VERY_LOW, False),
    LanguageProfile("so", "som", "Somali", "Af Soomaali", Script.LATIN, "Afro-Asiatic / Cushitic", ResourceLevel.VERY_LOW, False),
    LanguageProfile("ha", "hau", "Hausa", "Hausa", Script.LATIN, "Afro-Asiatic / Chadic", ResourceLevel.VERY_LOW, True),
    LanguageProfile("mg", "mlg", "Malagasy", "Malagasy", Script.LATIN, "Austronesian", ResourceLevel.VERY_LOW, False),
    LanguageProfile("xh", "xho", "Xhosa", "isiXhosa", Script.LATIN, "Niger\u2013Congo / Bantu", ResourceLevel.VERY_LOW, False),
    # ── Additional (8) ─────────────────────────────────────────────────
    LanguageProfile("fil", "fil", "Filipino", "Filipino", Script.LATIN, "Austronesian", ResourceLevel.LOW, False, notes="Based on Tagalog."),
    LanguageProfile("ms", "msa", "Malay", "Bahasa Melayu", Script.LATIN, "Austronesian", ResourceLevel.LOW, True),
    LanguageProfile("km", "khm", "Khmer", "\u1797\u17b6\u179f\u17b6\u1781\u17d2\u1798\u17c2\u179a", Script.KHMER, "Austroasiatic", ResourceLevel.VERY_LOW, False),
    LanguageProfile("lo", "lao", "Lao", "\u0e9e\u0eb2\u0eaa\u0eb2\u0ea5\u0eb2\u0ea7", Script.LAO, "Kra\u2013Dai", ResourceLevel.VERY_LOW, False),
    LanguageProfile("my", "mya", "Burmese", "\u1019\u103c\u1014\u103a\u1019\u102c\u1018\u102c\u101e\u102c", Script.MYANMAR, "Sino-Tibetan", ResourceLevel.VERY_LOW, False),
    LanguageProfile("af", "afr", "Afrikaans", "Afrikaans", Script.LATIN, "Indo-European / Germanic", ResourceLevel.LOW, True),
    LanguageProfile("ka", "kat", "Georgian", "\u10e5\u10d0\u10e0\u10d7\u10e3\u10da\u10d8", Script.GEORGIAN, "Kartvelian", ResourceLevel.LOW, True),
    LanguageProfile("et", "est", "Estonian", "Eesti", Script.LATIN, "Uralic / Finnic", ResourceLevel.LOW, True),
)


class SUPPORTED_LANGUAGES:
    """Registry of all 52 supported language profiles.

    Usage::

        from pii_anon.eval_framework.languages import SUPPORTED_LANGUAGES

        profile = SUPPORTED_LANGUAGES.get("en")
        all_langs = SUPPORTED_LANGUAGES.all()
        latin_langs = SUPPORTED_LANGUAGES.by_script(Script.LATIN)
    """

    _by_code: dict[str, LanguageProfile] = {p.iso639_1: p for p in _PROFILES}

    @classmethod
    def get(cls, code: str) -> LanguageProfile | None:
        """Look up a language by ISO 639-1 code."""
        return cls._by_code.get(code)

    @classmethod
    def all(cls) -> list[LanguageProfile]:
        """Return all 52 profiles sorted by ISO 639-1 code."""
        return sorted(cls._by_code.values(), key=lambda p: p.iso639_1)

    @classmethod
    def all_codes(cls) -> list[str]:
        """Return all 52 ISO 639-1 codes (sorted)."""
        return sorted(cls._by_code.keys())

    @classmethod
    def by_script(cls, script: Script) -> list[LanguageProfile]:
        """Return languages using a given writing system."""
        return [p for p in cls._by_code.values() if p.script == script]

    @classmethod
    def by_resource_level(cls, level: ResourceLevel) -> list[LanguageProfile]:
        """Return languages at a given NER resource level."""
        return [p for p in cls._by_code.values() if p.resource_level == level]

    @classmethod
    def by_family(cls, family_substring: str) -> list[LanguageProfile]:
        """Return languages whose family contains *family_substring*."""
        needle = family_substring.lower()
        return [p for p in cls._by_code.values() if needle in p.family.lower()]

    @classmethod
    def openner_supported(cls) -> list[LanguageProfile]:
        """Return languages covered by OpenNER 1.0 (2024)."""
        return [p for p in cls._by_code.values() if p.openner_supported]

    @classmethod
    def count(cls) -> int:
        """Total number of registered languages."""
        return len(cls._by_code)
