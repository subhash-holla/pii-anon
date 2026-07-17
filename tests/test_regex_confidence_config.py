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
    """A match evidenced as a DIFFERENT identifier's field/value is
    demoted below the emit floor. Kills the SSN-inside-NID-,
    PHONE-after-NPI: FP classes (n=10000 baseline: 829 + 727 FPs).

    Composition rules (pinned here, in TestNegativeContextLeakFixes, and
    in confidence.py): RULE A (value-position: US_SSN, PHONE_NUMBER) —
    the demotion fires when a negative phrase hits the pre/post window
    segments (span text EXCLUDED) AND no OWN-type context keyword
    appears in the NEGATIVE_ADJACENCY_WINDOW chars immediately before
    the span (the field-label position; trailing whitespace stripped).
    RULE B (label-position: ORGANIZATION) — the demotion fires when a
    negative phrase starts INSIDE the span AND label-position evidence
    (qualifier + separator + value) follows the span. An adjacent own
    label ("Phone:", "SSN:") protects the finding either way; a DISTANT
    own-type word (the neighbouring field's "contact:" label) does NOT
    protect.
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
        # the span IS the negative phrase AND ": HI-12345" follows (RULE B
        # label-position evidence); no ORG keyword adjacent
        # -> 0.78 - 0.30 - 0.15 -> floored 0.40 -> suppressed.
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


# ═══════════════════════════════════════════════════════════════════════════
# Leak-direction review fixes: span-exclusion + label-position rule +
# word-boundary negatives (a suppressed TRUE detection = unredacted PII)
# ═══════════════════════════════════════════════════════════════════════════


class TestNegativeContextLeakFixes:
    """Pins the two leak-direction fixes to the demotion mechanism.

    LEAK 1 (ORG self-trigger): the scan window included the span itself,
    so a genuine org whose OWN NAME contains a negative phrase ("Social
    Security Administration", "National Health Insurance Company")
    self-triggered the demotion -> unredacted PII. Fixed by splitting the
    mechanism into two rules:

    - RULE A (value-position: US_SSN, PHONE_NUMBER): scan the pre/post
      window segments EXCLUDING the span text (spans are digit runs, so
      exclusion is safe and principled).
    - RULE B (label-position: ORGANIZATION): the FP class and the genuine
      mention are THE SAME TEXT SHAPE distinguished by what FOLLOWS.
      Demote ONLY when a negative phrase starts inside the span AND the
      span is followed by label-position evidence (optional qualifier +
      separator + value token); otherwise keep.

    LEAK 2 (substring-vs-token asymmetry): "policy" substring-hit
    "Policyholder", suppressing a genuine phone. Fixed by word-boundary
    anchoring SHORT BARE-WORD negative phrases; multi-word labels
    ("account number") and prefix-shaped forms ("nid-") stay substrings.
    """

    def _findings(self, text: str, entity_type: str):
        from pii_anon.engines.regex_adapter import RegexEngineAdapter

        adapter = RegexEngineAdapter()
        return [
            f
            for f in adapter.detect({"text": text}, {"language": "en"})
            if f.entity_type == entity_type
        ]

    def _spans(self, text: str, entity_type: str) -> list[str]:
        return sorted(
            text[f.span_start : f.span_end] for f in self._findings(text, entity_type)
        )

    # ── The four-case ORGANIZATION matrix ─────────────────────────────

    def test_org_case1_org_name_mention_emitted(self) -> None:
        # The org's own name contains the negative phrase "social
        # security" but is followed by " in Baltimore" — no separator +
        # value, so no label evidence -> NOT demoted. sp2 extent hygiene:
        # the old pin captured "the ... in" junk extents (full-pattern
        # IGNORECASE let the capture start at lowercase "the"); the agency
        # suffix list + case-sensitive capture + nested same-type dedup now
        # yield the one clean maximal span. The leak fix stays covered:
        # the genuine mention IS emitted.
        text = "She works at the Social Security Administration in Baltimore"
        assert self._spans(text, "ORGANIZATION") == [
            "Social Security Administration",
        ]

    def test_org_case2_sued_org_name_emitted(self) -> None:
        # Genuine org mention containing "health insurance"; followed by
        # end-of-string -> no label evidence -> emitted (was 0 = leak).
        # sp2: nested same-type dedup keeps only the maximal span.
        text = "He sued National Health Insurance Company"
        spans = self._spans(text, "ORGANIZATION")
        assert spans == ["National Health Insurance Company"]

    def test_org_case3_label_with_value_stays_dead(self) -> None:
        # The original FP class: "Health Insurance" here is a FIELD LABEL
        # (followed by ": HI-12345" — separator + value), not an org.
        # The negative phrase IS the span, so the label-position evidence
        # is what keeps this dead under span-aware scanning.
        text = "Record for Maria Garcia, Health Insurance: HI-12345"
        assert self._findings(text, "ORGANIZATION") == []

    def test_org_case4_ssn_inside_nid_stays_dead(self) -> None:
        # RULE A regression guard: excluding the span from the window scan
        # must not resurrect the SSN-inside-NID class — the negative
        # phrases ("national id", "nid-") live in the PRE segment.
        found = self._findings("National Id Number: NID-422736198", "US_SSN")
        assert found == [], f"US_SSN must stay dead inside NID- value; got {found}"

    # ── RULE B qualifier shapes (label evidence past a qualifier word) ─

    def test_org_label_with_id_qualifier_stays_dead(self) -> None:
        # "Migraine Insurance ID: INS-12345": the phrase "insurance id"
        # STARTS inside the span and the evidence regex's qualifier
        # absorbs " ID" before the separator -> still demoted.
        assert self._findings("Migraine Insurance ID: INS-12345", "ORGANIZATION") == []

    def test_org_label_with_handle_qualifier_stays_dead(self) -> None:
        # "Social Media Handle: @maria_g": "handle" is a measured label
        # qualifier (n=2000 seed=8314 FP class) -> still demoted.
        assert self._findings("Social Media Handle: @maria_g", "ORGANIZATION") == []

    def test_org_bare_qualifier_word_stays_dead(self) -> None:
        # Evidence shape B2 (tuning iteration 2): the deposition-dialogue
        # class names the FIELD without a separator + value ("…your Social
        # Security Number, for the record?") — the bare qualifier word
        # right after the span is label evidence; B1 alone resurrected 6
        # of these as ORG FPs on the n=2000 seed=8314 draw. The genuine
        # "Bluth Company" org in the same text must survive.
        text = (
            "worked at Bluth Company for approximately 8 years.  "
            "Q: And your Social Security Number, for the record?"
        )
        spans = self._spans(text, "ORGANIZATION")
        assert "Social Security" not in spans, (
            f"field-name 'Social Security Number' must not resurrect; got {spans!r}"
        )
        assert "Bluth Company" in spans

    def test_org_phrase_only_in_consumed_tail_not_demoted(self) -> None:
        # The negative phrase must START INSIDE the span: a phrase living
        # entirely in the consumed evidence tail (a parenthetical) must
        # not demote the real org name before it.
        text = "Acme Corp (Health Insurance): 22 Main St"
        result = confidence.apply_negative_context(
            "ORGANIZATION", base_confidence=0.80, text=text, start=0, end=9
        )
        assert result == 0.80, f"Expected unchanged 0.80, got {result}"

    def test_org_label_position_unit_anchors(self) -> None:
        # Org-name mention (no evidence tail) -> identity.
        text = "She works at the Social Security Administration in Baltimore"
        start = text.index("Social Security")
        result = confidence.apply_negative_context(
            "ORGANIZATION",
            base_confidence=0.78,
            text=text,
            start=start,
            end=start + len("Social Security"),
        )
        assert result == 0.78, f"Expected unchanged 0.78, got {result}"
        # Label + qualifier + value -> exact demotion floor.
        text = "Migraine Insurance ID: INS-12345"
        result = confidence.apply_negative_context(
            "ORGANIZATION", base_confidence=0.78, text=text, start=0, end=18
        )
        assert result == 0.40, f"Expected exact floor 0.40, got {result}"

    # ── LEAK 2: word-boundary negatives (value-position types) ────────

    def test_policyholder_phone_emitted(self) -> None:
        # "policy" must NOT substring-hit "Policyholder": the genuine
        # phone was suppressed (leak). Exactly 1 PHONE_NUMBER.
        text = "Policyholder John Smith can be reached at (415) 555-0142"
        found = self._findings(text, "PHONE_NUMBER")
        assert len(found) == 1, f"Expected exactly 1 PHONE_NUMBER, got {found}"
        assert text[found[0].span_start : found[0].span_end] == "(415) 555-0142"

    def test_npi_phone_stays_dead(self) -> None:
        # Keep-dead: "npi" still matches as a whole word.
        assert self._findings("NPI: 3803252675", "PHONE_NUMBER") == []

    def test_insurance_policy_ssn_stays_dead(self) -> None:
        # Keep-dead: "insurance" and "policy" appear as WHOLE words in the
        # pre-segment -> the valid-format 9-digit run is still demoted.
        found = self._findings("Insurance Policy Number: 536904399", "US_SSN")
        assert found == [], f"US_SSN must stay dead after policy label; got {found}"

    def test_has_negative_context_word_boundary_unit(self) -> None:
        # Bare-word negatives are word-boundary anchored...
        assert not confidence.has_negative_context(
            "PHONE_NUMBER", "policyholder john smith can be reached at "
        )
        assert not confidence.has_negative_context("US_SSN", "invoiced amounts: ")
        assert confidence.has_negative_context("PHONE_NUMBER", "insurance policy number: ")
        assert confidence.has_negative_context("US_SSN", "invoice #")
        # ...multi-word and prefix-shaped negatives stay substrings.
        assert confidence.has_negative_context("US_SSN", "national id number: nid-")
        assert confidence.has_negative_context("PHONE_NUMBER", "the account number is")

    # ── Own-label adjacency: trailing-whitespace strip ─────────────────

    def test_own_label_padding_still_protects(self) -> None:
        # Whitespace between the own label and the span is stripped before
        # taking the 25-char prefix, so "Phone:" + padding still protects
        # the genuine phone from the "account number" in its post-window;
        # the account value itself stays demoted (exactly 1 finding).
        text = "Phone:" + " " * 30 + "(415) 555-0142, account number 9855276428"
        found = self._findings(text, "PHONE_NUMBER")
        assert len(found) == 1, f"Expected exactly 1 PHONE_NUMBER, got {found}"
        assert text[found[0].span_start : found[0].span_end] == "(415) 555-0142"


# ═══════════════════════════════════════════════════════════════════════════
# Negative-context hygiene: config wiring, reachability, export
# ═══════════════════════════════════════════════════════════════════════════


def test_configure_from_config_sets_negative_context_penalty() -> None:
    """NEGATIVE_CONTEXT_PENALTY is wired into configure_from_config for
    symmetry with the other runtime-tunable confidence constants."""
    original = confidence.NEGATIVE_CONTEXT_PENALTY
    try:
        confidence.configure_from_config(negative_context_penalty=0.45)
        assert confidence.NEGATIVE_CONTEXT_PENALTY == 0.45
        confidence.configure_from_config(negative_context_penalty=None)
        assert confidence.NEGATIVE_CONTEXT_PENALTY == 0.45
    finally:
        confidence.NEGATIVE_CONTEXT_PENALTY = original


def test_demoted_finding_below_emit_floor_under_default_constants() -> None:
    """COUPLING PIN: under DEFAULT constants, every spec that can reach the
    negative-context demotion lands strictly below the adapter's 0.50 emit
    floor when demoted (the mechanism SUPPRESSES) — with exactly ONE
    measured exception: the dashed-SSN spec ('regex ssn', base 0.97),
    whose strongest-format value survives demotion at exactly
    0.97 - 0.30 - 0.15 = 0.52. That survival is pinned (not "fixed"):
    forcing a validated XXX-XX-XXXX value below the floor would be a
    suppression-strengthening change — the leak direction this task
    guards against. Any constant/spec drift must update this partition
    consciously."""
    from pii_anon.engines.regex.patterns import PATTERN_REGISTRY
    from pii_anon.engines.regex_adapter import _MIN_EMIT_CONFIDENCE

    assert confidence.CONFIDENCE_FLOOR < _MIN_EMIT_CONFIDENCE
    suppressed = 0
    survivors: dict[str, float] = {}
    for spec in PATTERN_REGISTRY:
        neg_key = spec.context_type or spec.entity_type
        if neg_key not in confidence.NEGATIVE_CONTEXT_WORDS:
            continue
        bases = [spec.base_confidence]
        if spec.valid_confidence is not None:
            bases.append(spec.valid_confidence)
        for base in bases:
            demoted = confidence._negative_context_demotion(neg_key, base)
            if demoted < _MIN_EMIT_CONFIDENCE:
                suppressed += 1
            else:
                survivors[spec.explanation] = demoted
    assert suppressed > 0, "No demotable specs found — registry coupling broken"
    assert set(survivors) == {"regex ssn"}, (
        f"Demotion-survivor set drifted from the pinned dashed-SSN-only "
        f"exception: {survivors!r}"
    )
    expected = 0.97 - confidence.NEGATIVE_CONTEXT_PENALTY - confidence.CONTEXT_PENALTY
    assert survivors["regex ssn"] == expected


def test_negative_context_keys_reachable_via_pattern_specs() -> None:
    """REACHABILITY GUARD: every NEGATIVE_CONTEXT_WORDS key must be
    reachable from the adapter's detect loop — via adjust_confidence
    (some spec's context_type == key) or via apply_negative_context
    (some spec with NO context_type has entity_type == key). A dead key
    is a silent no-op that would mask a broken suppression."""
    from pii_anon.engines.regex.patterns import PATTERN_REGISTRY

    reachable: set[str] = set()
    for spec in PATTERN_REGISTRY:
        if spec.context_type:
            reachable.add(spec.context_type)
        else:
            reachable.add(spec.entity_type)
    unreachable = set(confidence.NEGATIVE_CONTEXT_WORDS) - reachable
    assert unreachable == set(), (
        f"NEGATIVE_CONTEXT_WORDS keys unreachable via any PatternSpec: "
        f"{sorted(unreachable)}"
    )


def test_negative_adjacency_window_exported() -> None:
    """NEGATIVE_ADJACENCY_WINDOW is part of the mechanism's public surface
    alongside NEGATIVE_CONTEXT_WORDS / NEGATIVE_CONTEXT_PENALTY."""
    from pii_anon.engines import regex

    assert regex.NEGATIVE_ADJACENCY_WINDOW == confidence.NEGATIVE_ADJACENCY_WINDOW
    assert "NEGATIVE_ADJACENCY_WINDOW" in regex.__all__
