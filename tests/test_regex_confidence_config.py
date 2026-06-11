"""Tests for regex confidence module configuration and adjustment logic.

Covers configure_from_config function, module-level globals, the
context-aware confidence adjustment system, and the negative-context
demotion for cross-identifier digit false positives (SP1 Task 9).
"""

from __future__ import annotations


from pii_anon.engines.regex import confidence


# ═══════════════════════════════════════════════════════════════════════════
# configure_from_config() Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_configure_from_config_sets_context_boost() -> None:
    """Test configure_from_config updates CONTEXT_BOOST global."""
    original = confidence.CONTEXT_BOOST
    try:
        confidence.configure_from_config(context_boost=0.25)
        assert confidence.CONTEXT_BOOST == 0.25
    finally:
        confidence.CONTEXT_BOOST = original


def test_configure_from_config_sets_context_penalty() -> None:
    """Test configure_from_config updates CONTEXT_PENALTY global."""
    original = confidence.CONTEXT_PENALTY
    try:
        confidence.configure_from_config(context_penalty=0.20)
        assert confidence.CONTEXT_PENALTY == 0.20
    finally:
        confidence.CONTEXT_PENALTY = original


def test_configure_from_config_sets_context_window() -> None:
    """Test configure_from_config updates CONTEXT_WINDOW global."""
    original = confidence.CONTEXT_WINDOW
    try:
        confidence.configure_from_config(context_window=100)
        assert confidence.CONTEXT_WINDOW == 100
    finally:
        confidence.CONTEXT_WINDOW = original


def test_configure_from_config_sets_confidence_cap() -> None:
    """Test configure_from_config updates CONFIDENCE_CAP global."""
    original = confidence.CONFIDENCE_CAP
    try:
        confidence.configure_from_config(confidence_cap=0.95)
        assert confidence.CONFIDENCE_CAP == 0.95
    finally:
        confidence.CONFIDENCE_CAP = original


def test_configure_from_config_sets_confidence_floor() -> None:
    """Test configure_from_config updates CONFIDENCE_FLOOR global."""
    original = confidence.CONFIDENCE_FLOOR
    try:
        confidence.configure_from_config(confidence_floor=0.35)
        assert confidence.CONFIDENCE_FLOOR == 0.35
    finally:
        confidence.CONFIDENCE_FLOOR = original


def test_configure_from_config_multiple_values() -> None:
    """Test configure_from_config can update multiple values at once."""
    originals = {
        'CONTEXT_BOOST': confidence.CONTEXT_BOOST,
        'CONTEXT_PENALTY': confidence.CONTEXT_PENALTY,
        'CONTEXT_WINDOW': confidence.CONTEXT_WINDOW,
        'CONFIDENCE_CAP': confidence.CONFIDENCE_CAP,
        'CONFIDENCE_FLOOR': confidence.CONFIDENCE_FLOOR,
    }
    try:
        confidence.configure_from_config(
            context_boost=0.12,
            context_penalty=0.18,
            context_window=75,
            confidence_cap=0.97,
            confidence_floor=0.42,
        )
        assert confidence.CONTEXT_BOOST == 0.12
        assert confidence.CONTEXT_PENALTY == 0.18
        assert confidence.CONTEXT_WINDOW == 75
        assert confidence.CONFIDENCE_CAP == 0.97
        assert confidence.CONFIDENCE_FLOOR == 0.42
    finally:
        confidence.CONTEXT_BOOST = originals['CONTEXT_BOOST']
        confidence.CONTEXT_PENALTY = originals['CONTEXT_PENALTY']
        confidence.CONTEXT_WINDOW = originals['CONTEXT_WINDOW']
        confidence.CONFIDENCE_CAP = originals['CONFIDENCE_CAP']
        confidence.CONFIDENCE_FLOOR = originals['CONFIDENCE_FLOOR']


def test_configure_from_config_none_values_ignored() -> None:
    """Test that None values in configure_from_config are ignored."""
    originals = {
        'CONTEXT_BOOST': confidence.CONTEXT_BOOST,
        'CONTEXT_PENALTY': confidence.CONTEXT_PENALTY,
    }
    try:
        confidence.configure_from_config(
            context_boost=None,
            context_penalty=0.22
        )
        # CONTEXT_BOOST should remain unchanged
        assert confidence.CONTEXT_BOOST == originals['CONTEXT_BOOST']
        # CONTEXT_PENALTY should be updated
        assert confidence.CONTEXT_PENALTY == 0.22
    finally:
        confidence.CONTEXT_BOOST = originals['CONTEXT_BOOST']
        confidence.CONTEXT_PENALTY = originals['CONTEXT_PENALTY']


def test_configure_from_config_all_none() -> None:
    """Test configure_from_config with all None values changes nothing."""
    originals = {
        'CONTEXT_BOOST': confidence.CONTEXT_BOOST,
        'CONTEXT_PENALTY': confidence.CONTEXT_PENALTY,
        'CONTEXT_WINDOW': confidence.CONTEXT_WINDOW,
        'CONFIDENCE_CAP': confidence.CONFIDENCE_CAP,
        'CONFIDENCE_FLOOR': confidence.CONFIDENCE_FLOOR,
    }
    try:
        confidence.configure_from_config(
            context_boost=None,
            context_penalty=None,
            context_window=None,
            confidence_cap=None,
            confidence_floor=None,
        )
        assert confidence.CONTEXT_BOOST == originals['CONTEXT_BOOST']
        assert confidence.CONTEXT_PENALTY == originals['CONTEXT_PENALTY']
        assert confidence.CONTEXT_WINDOW == originals['CONTEXT_WINDOW']
        assert confidence.CONFIDENCE_CAP == originals['CONFIDENCE_CAP']
        assert confidence.CONFIDENCE_FLOOR == originals['CONFIDENCE_FLOOR']
    finally:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Global Constants Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_module_globals_exist() -> None:
    """Test that all module-level configuration globals exist."""
    assert hasattr(confidence, 'CONTEXT_BOOST')
    assert hasattr(confidence, 'CONTEXT_PENALTY')
    assert hasattr(confidence, 'CONTEXT_WINDOW')
    assert hasattr(confidence, 'CONFIDENCE_CAP')
    assert hasattr(confidence, 'CONFIDENCE_FLOOR')


def test_module_globals_have_sensible_defaults() -> None:
    """Test that module-level globals have sensible default values."""
    assert isinstance(confidence.CONTEXT_BOOST, float)
    assert isinstance(confidence.CONTEXT_PENALTY, float)
    assert isinstance(confidence.CONTEXT_WINDOW, int)
    assert isinstance(confidence.CONFIDENCE_CAP, float)
    assert isinstance(confidence.CONFIDENCE_FLOOR, float)
    # Check ranges
    assert 0 < confidence.CONTEXT_BOOST < 1
    assert 0 < confidence.CONTEXT_PENALTY < 1
    assert confidence.CONTEXT_WINDOW > 0
    assert 0 < confidence.CONFIDENCE_CAP <= 1
    assert 0 <= confidence.CONFIDENCE_FLOOR < 1


# ═══════════════════════════════════════════════════════════════════════════
# adjust_confidence with Dynamic Globals Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_adjust_confidence_respects_confidence_cap() -> None:
    """Test that adjust_confidence respects CONFIDENCE_CAP global."""
    originals = {
        'CONTEXT_BOOST': confidence.CONTEXT_BOOST,
        'CONFIDENCE_CAP': confidence.CONFIDENCE_CAP,
    }
    try:
        # Set a low cap
        confidence.CONFIDENCE_CAP = 0.85
        confidence.CONTEXT_BOOST = 0.20

        text = "My name is John Doe"
        # This should find "name" context word and boost
        result = confidence.adjust_confidence(
            "PERSON_NAME",
            base_confidence=0.80,
            text=text,
            start=11,
            end=20
        )
        # Result should be capped at 0.85, not 1.0
        assert result <= 0.85
    finally:
        confidence.CONTEXT_BOOST = originals['CONTEXT_BOOST']
        confidence.CONFIDENCE_CAP = originals['CONFIDENCE_CAP']


def test_adjust_confidence_respects_confidence_floor() -> None:
    """Test that adjust_confidence respects CONFIDENCE_FLOOR global."""
    originals = {
        'CONTEXT_PENALTY': confidence.CONTEXT_PENALTY,
        'CONFIDENCE_FLOOR': confidence.CONFIDENCE_FLOOR,
    }
    try:
        # Set a high floor
        confidence.CONFIDENCE_FLOOR = 0.60
        confidence.CONTEXT_PENALTY = 0.30

        text = "123-45-6789"
        # HIGH_FP_TYPES includes US_SSN, will penalize without context
        result = confidence.adjust_confidence(
            "US_SSN",
            base_confidence=0.80,
            text=text,
            start=0,
            end=11
        )
        # Result should not go below 0.60
        assert result >= 0.60
    finally:
        confidence.CONTEXT_PENALTY = originals['CONTEXT_PENALTY']
        confidence.CONFIDENCE_FLOOR = originals['CONFIDENCE_FLOOR']


def test_adjust_confidence_uses_dynamic_context_boost() -> None:
    """Test adjust_confidence uses current CONTEXT_BOOST value."""
    originals = {
        'CONTEXT_BOOST': confidence.CONTEXT_BOOST,
        'CONFIDENCE_CAP': confidence.CONFIDENCE_CAP,
    }
    try:
        # Set custom boost value
        confidence.CONTEXT_BOOST = 0.30
        confidence.CONFIDENCE_CAP = 0.99

        text = "social security number 123-45-6789"
        result = confidence.adjust_confidence(
            "US_SSN",
            base_confidence=0.70,
            text=text,
            start=24,
            end=35
        )
        # With context boost of 0.30: 0.70 + 0.30 = 1.0, but capped at 0.99
        assert result == 0.99
    finally:
        confidence.CONTEXT_BOOST = originals['CONTEXT_BOOST']
        confidence.CONFIDENCE_CAP = originals['CONFIDENCE_CAP']


def test_adjust_confidence_uses_dynamic_context_penalty() -> None:
    """Test adjust_confidence uses current CONTEXT_PENALTY value."""
    originals = {
        'CONTEXT_PENALTY': confidence.CONTEXT_PENALTY,
        'CONFIDENCE_FLOOR': confidence.CONFIDENCE_FLOOR,
    }
    try:
        # Set custom penalty value
        confidence.CONTEXT_PENALTY = 0.40
        confidence.CONFIDENCE_FLOOR = 0.30

        text = "123-45-6789"  # No context keywords
        result = confidence.adjust_confidence(
            "US_SSN",  # HIGH_FP_TYPE
            base_confidence=0.75,
            text=text,
            start=0,
            end=11
        )
        # With penalty of 0.40: 0.75 - 0.40 = 0.35, but floored at 0.30
        assert result == 0.35
    finally:
        confidence.CONTEXT_PENALTY = originals['CONTEXT_PENALTY']
        confidence.CONFIDENCE_FLOOR = originals['CONFIDENCE_FLOOR']


def test_adjust_confidence_uses_dynamic_context_window() -> None:
    """Test adjust_confidence uses current CONTEXT_WINDOW value."""
    originals = {
        'CONTEXT_WINDOW': confidence.CONTEXT_WINDOW,
        'CONFIDENCE_PENALTY': confidence.CONTEXT_PENALTY,
        'CONFIDENCE_FLOOR': confidence.CONFIDENCE_FLOOR,
    }
    try:
        # With small window, context words further away won't be found
        confidence.CONTEXT_WINDOW = 5
        confidence.CONTEXT_PENALTY = 0.30
        confidence.CONFIDENCE_FLOOR = 0.30

        # Create a case where context would be outside a small window
        # "name" keyword is at position 0-4, span starts at 36
        text = "name is not relevant here..................John Doe"
        result = confidence.adjust_confidence(
            "PERSON_NAME",
            base_confidence=0.75,
            text=text,
            start=36,
            end=40
        )
        # With small window, "name" is outside and should be penalized
        assert result < 0.75
    finally:
        confidence.CONTEXT_WINDOW = originals['CONTEXT_WINDOW']
        confidence.CONTEXT_PENALTY = originals['CONFIDENCE_PENALTY']
        confidence.CONFIDENCE_FLOOR = originals['CONFIDENCE_FLOOR']


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


def test_full_workflow_configure_then_adjust() -> None:
    """Test full workflow: configure module then use adjust_confidence."""
    originals = {
        'CONTEXT_BOOST': confidence.CONTEXT_BOOST,
        'CONTEXT_PENALTY': confidence.CONTEXT_PENALTY,
        'CONTEXT_WINDOW': confidence.CONTEXT_WINDOW,
        'CONFIDENCE_CAP': confidence.CONFIDENCE_CAP,
        'CONFIDENCE_FLOOR': confidence.CONFIDENCE_FLOOR,
    }
    try:
        # Configure with custom values
        confidence.configure_from_config(
            context_boost=0.25,
            context_penalty=0.35,
            context_window=80,
            confidence_cap=0.96,
            confidence_floor=0.45
        )

        # Test with context word found
        # "name" is a context keyword for PERSON_NAME
        text = "The name John Doe is important"
        result = confidence.adjust_confidence(
            "PERSON_NAME",
            base_confidence=0.75,
            text=text,
            start=9,
            end=17
        )
        # Should be boosted (context found: "name")
        # 0.75 + 0.25 (boost) = 1.0, but capped at 0.96
        assert result == 0.96
    finally:
        confidence.CONTEXT_BOOST = originals['CONTEXT_BOOST']
        confidence.CONTEXT_PENALTY = originals['CONTEXT_PENALTY']
        confidence.CONTEXT_WINDOW = originals['CONTEXT_WINDOW']
        confidence.CONFIDENCE_CAP = originals['CONFIDENCE_CAP']
        confidence.CONFIDENCE_FLOOR = originals['CONFIDENCE_FLOOR']


# ═══════════════════════════════════════════════════════════════════════════
# Negative-Context Demotion (SP1 Task 9)
# ═══════════════════════════════════════════════════════════════════════════


class TestNegativeContextDemotion:
    """A match whose ±50-char window names a DIFFERENT identifier's field
    label is demoted below the emit floor. Kills the SSN-inside-NID-,
    PHONE-after-NPI: FP classes (n=10000 baseline: 829 + 727 FPs).

    Composition rule (pinned here and in confidence.py): the demotion
    fires when a negative phrase is anywhere in the ±CONTEXT_WINDOW
    window AND no OWN-type context keyword appears in the
    NEGATIVE_ADJACENCY_WINDOW chars immediately before the span (the
    field-label position). An adjacent own label ("Phone:", "SSN:")
    protects the finding even with a negative word elsewhere in the
    window; a DISTANT own-type word (the neighbouring field's
    "contact:" label) does NOT protect.
    """

    def _findings(self, text: str, entity_type: str):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == entity_type
        ]

    # ── FP classes: demoted below the emit floor ─────────────────────

    def test_ssn_not_emitted_inside_national_id(self) -> None:
        # 9-digit run inside an NID- value: "national id" / "nid-" in the
        # window, no SSN keyword adjacent -> 0.65 - 0.30 - 0.15 = 0.20
        # -> floored 0.40 < 0.50 emit floor -> suppressed.
        found = self._findings("National Id Number: NID-422736198", "US_SSN")
        assert found == [], f"US_SSN must not be emitted inside NID- value; got {found}"

    def test_ssn_not_emitted_after_routing_label(self) -> None:
        # 9-digit run after "Bank Routing Number:" -> "routing" negative.
        found = self._findings("Bank Routing Number: 827011129", "US_SSN")
        assert found == [], f"US_SSN must not be emitted after routing label; got {found}"

    def test_phone_not_emitted_after_npi_label(self) -> None:
        # 10-digit run after "NPI:" -> "npi" negative, no phone keyword
        # adjacent -> 0.80 - 0.30 - 0.15 = 0.35 -> floored 0.40 -> suppressed.
        found = self._findings("NPI: 3803252675", "PHONE_NUMBER")
        assert found == [], f"PHONE_NUMBER must not be emitted after NPI label; got {found}"

    def test_phone_not_emitted_after_account_label(self) -> None:
        # 10-digit run after "Bank Account Number:" -> "account number".
        found = self._findings("Bank Account Number: 2746750189", "PHONE_NUMBER")
        assert found == [], (
            f"PHONE_NUMBER must not be emitted after account label; got {found}"
        )

    def test_org_health_insurance_not_emitted(self) -> None:
        # Generic phrase "Health Insurance" (a field label, not an org):
        # the span itself puts "health insurance" in the window; no ORG
        # keyword adjacent -> 0.78 - 0.30 - 0.15 -> floored 0.40 -> suppressed.
        text = "Record for Maria Garcia, Health Insurance: HI-12345"
        spans = [
            text[f.span_start : f.span_end]
            for f in self._findings(text, "ORGANIZATION")
            if f.span_start is not None and f.span_end is not None
        ]
        assert "Health Insurance" not in spans, (
            f"'Health Insurance' must not be emitted as ORGANIZATION; got {spans!r}"
        )

    # ── Genuine forms: own label adjacent -> still emitted ───────────

    def test_genuine_ssn_still_emitted(self) -> None:
        found = self._findings("SSN: 536-90-4399", "US_SSN")
        assert len(found) == 1, f"Expected exactly 1 US_SSN, got {found}"

    def test_genuine_phone_still_emitted(self) -> None:
        found = self._findings("Phone: (415) 555-0142", "PHONE_NUMBER")
        assert len(found) == 1, f"Expected exactly 1 PHONE_NUMBER, got {found}"

    def test_genuine_phone_with_distant_negative_word(self) -> None:
        # Pins the composition rule end-to-end. The genuine phone has its
        # own label adjacent ("Phone: " directly before the span) and a
        # negative phrase ("account number") within the +-50 window -> the
        # adjacent own label wins -> boost path -> emitted. The 10-digit
        # account value in the SAME text has "phone" only DISTANTLY in its
        # window (26+ chars back) and "account number" directly before it
        # -> demoted -> NOT emitted. Net: exactly the genuine span remains.
        text = "Phone: (415) 555-0142, account number 9855276428"
        found = self._findings(text, "PHONE_NUMBER")
        assert len(found) == 1, f"Expected exactly 1 PHONE_NUMBER, got {found}"
        f = found[0]
        assert text[f.span_start : f.span_end] == "(415) 555-0142", (
            f"The genuine labelled phone must be the surviving finding; got "
            f"{text[f.span_start:f.span_end]!r}"
        )

    # ── Unit level: exact arithmetic anchors ─────────────────────────

    def test_demotion_unit_level(self) -> None:
        """Pins exact demotion arithmetic at the adjust_confidence level.

        Derivation: the demoted branch is
        ``max(CONFIDENCE_FLOOR, base - NEGATIVE_CONTEXT_PENALTY - CONTEXT_PENALTY)``
        for HIGH_FP_TYPES (the span is scored as context-ABSENT because the
        adjacent foreign label owns the digits, so the absence penalty
        stacks with the demotion), applied BEFORE the floor clamp.
        """
        # US_SSN nodash base 0.65 inside "National Id Number: NID-...":
        # 0.65 - 0.30 - 0.15 = 0.20 -> clamped to CONFIDENCE_FLOOR = 0.40.
        text = "National Id Number: NID-422736198"
        result = confidence.adjust_confidence(
            "US_SSN", base_confidence=0.65, text=text, start=24, end=33
        )
        assert result == 0.40, f"Expected exact floor 0.40, got {result}"

        # PHONE_NUMBER +1-format base 0.92 after "NPI:": unclamped branch,
        # 0.92 - 0.30 - 0.15 (exact float expression mirrors the
        # implementation's left-to-right subtraction) = 0.47000...,
        # strictly below the 0.50 adapter emit floor.
        text = "NPI: 3803252675"
        result = confidence.adjust_confidence(
            "PHONE_NUMBER", base_confidence=0.92, text=text, start=5, end=15
        )
        expected = (
            0.92
            - confidence.NEGATIVE_CONTEXT_PENALTY
            - confidence.CONTEXT_PENALTY
        )
        assert result == expected, f"Expected exact {expected}, got {result}"
        assert result < 0.50, "Demoted +1-format phone must drop below the emit floor"

        # Adjacent own label protects: identical negative word in window,
        # but "Phone: " directly before the span -> boost path, no demotion.
        text = "Phone: (415) 555-0142, account number 9855276428"
        result = confidence.adjust_confidence(
            "PHONE_NUMBER", base_confidence=0.80, text=text, start=7, end=21
        )
        expected = min(confidence.CONFIDENCE_CAP, 0.80 + confidence.CONTEXT_BOOST)
        assert result == expected, f"Expected boosted {expected}, got {result}"

    def test_negative_context_constants_shape(self) -> None:
        # The mechanism's constants: per-type frozensets of lowercase
        # substrings + the demotion penalty. BANK_ACCOUNT deliberately has
        # NO entry (IBAN truth maps canonically to BANK_ACCOUNT — those
        # detections are real census TPs; suppressing them regresses G6).
        assert confidence.NEGATIVE_CONTEXT_PENALTY == 0.30
        assert set(confidence.NEGATIVE_CONTEXT_WORDS) == {
            "US_SSN", "PHONE_NUMBER", "ORGANIZATION",
        }
        assert "BANK_ACCOUNT" not in confidence.NEGATIVE_CONTEXT_WORDS
        for words in confidence.NEGATIVE_CONTEXT_WORDS.values():
            assert isinstance(words, frozenset)
            assert all(w == w.lower() for w in words)

    def test_apply_negative_context_org_unit_level(self) -> None:
        """ORGANIZATION specs carry no context_type, so they reach the
        demotion through ``apply_negative_context`` (the adapter's
        no-context_type branch): same arbitration, no boost activation.
        """
        text = "Record for Maria Garcia, Health Insurance: HI-12345"
        # span "Health Insurance" at [25, 41): 0.78 - 0.30 - 0.15 -> 0.40 floor.
        result = confidence.apply_negative_context(
            "ORGANIZATION", base_confidence=0.78, text=text, start=25, end=41
        )
        assert result == 0.40, f"Expected exact floor 0.40, got {result}"

        # No negative phrase in window -> returned unchanged (identity).
        text = "The meeting covered Acme Corporation results"
        result = confidence.apply_negative_context(
            "ORGANIZATION", base_confidence=0.78, text=text, start=20, end=36
        )
        assert result == 0.78, f"Expected unchanged 0.78, got {result}"

        # Entity types without a NEGATIVE_CONTEXT_WORDS entry pass through
        # untouched even when another type's negative word is present.
        result = confidence.apply_negative_context(
            "NATIONAL_ID", base_confidence=0.88,
            text="National Id Number: NID-422736198", start=20, end=33,
        )
        assert result == 0.88, f"Expected unchanged 0.88, got {result}"
