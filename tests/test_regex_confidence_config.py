"""Tests for regex confidence module configuration and adjustment logic.

Covers configure_from_config function, module-level globals, and the
context-aware confidence adjustment system.
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
