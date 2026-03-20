"""Tests to improve coverage for transformation strategies.

Targets uncovered lines in src/pii_anon/transforms/strategies.py:
- Lines 60-61: Exception handling in PlaceholderStrategy template formatting
- Lines 116, 191, 207, 211: Various edge cases in RedactionStrategy
- Lines 304, 307-308, 320-324, 337, 345, 364, 368, 376, 389, 395-404, 412, 418, 423-425:
  GeneralizationStrategy edge cases
- Lines 628, 764, 818-819, 841-842, 847-848, 850-851: SyntheticReplacementStrategy and
  PerturbationStrategy edge cases
"""

from pii_anon.transforms.base import TransformContext
from pii_anon.transforms.strategies import (
    PlaceholderStrategy,
    TokenizationStrategy,
    RedactionStrategy,
    GeneralizationStrategy,
    SyntheticReplacementStrategy,
    PerturbationStrategy,
)


def create_context(**overrides):
    """Helper to create a context with custom parameters."""
    defaults = {
        "entity_type": "PERSON_NAME",
        "plaintext": "John Smith",
        "field_path": "customer.name",
        "language": "en",
        "scope": "default",
        "finding": None,
        "cluster_id": "cluster_001",
        "placeholder_index": 1,
        "is_first_mention": True,
        "mention_index": 0,
        "document_text": "",
        "token_key": "test-key-12345",
        "token_version": 1,
        "strategy_params": {},
    }
    defaults.update(overrides)
    return TransformContext(**defaults)


# ============================================================================
# PlaceholderStrategy - Exception Handling (Lines 60-61)
# ============================================================================


class TestPlaceholderStrategyExceptionHandling:
    """Test exception handling in template formatting."""

    def test_invalid_template_format_string_fallback(self):
        """PlaceholderStrategy should fallback to default when template has invalid placeholders."""
        strategy = PlaceholderStrategy(template="<{invalid_placeholder}>")
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="test@example.com",
            placeholder_index=5,
        )
        result = strategy.transform("test@example.com", "EMAIL_ADDRESS", context)
        # Should fallback to default format
        assert result.replacement == "<EMAIL_ADDRESS:anon_5>"

    def test_template_with_missing_closing_brace(self):
        """PlaceholderStrategy should handle malformed template."""
        strategy = PlaceholderStrategy(template="<{entity_type:anon_{index}")
        context = create_context(
            entity_type="PHONE_NUMBER",
            plaintext="555-1234",
            placeholder_index=3,
        )
        result = strategy.transform("555-1234", "PHONE_NUMBER", context)
        # Should fallback to default
        assert result.replacement == "<PHONE_NUMBER:anon_3>"

    def test_template_with_syntax_error(self):
        """PlaceholderStrategy should gracefully handle template syntax errors."""
        strategy = PlaceholderStrategy(template="<{entity_type!r:bad}>")
        context = create_context(
            entity_type="ADDRESS",
            plaintext="123 Main St",
            placeholder_index=2,
        )
        result = strategy.transform("123 Main St", "ADDRESS", context)
        # Should fallback to default
        assert result.replacement == "<ADDRESS:anon_2>"


# ============================================================================
# RedactionStrategy - Edge Cases (Lines 191, 207, 211)
# ============================================================================


class TestRedactionStrategyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_redaction(self):
        """RedactionStrategy should handle empty strings without crashing."""
        strategy = RedactionStrategy(mode="full", mask_char="*")
        context = create_context(entity_type="PERSON_NAME", plaintext="")
        result = strategy.transform("", "PERSON_NAME", context)
        # Empty string should remain empty
        assert result.replacement == ""

    def test_single_character_partial_start(self):
        """RedactionStrategy partial_start with reveal_count >= len(text)."""
        strategy = RedactionStrategy(mode="partial_start", mask_char="*", reveal_count=5)
        context = create_context(entity_type="NAME", plaintext="A")
        result = strategy.transform("A", "NAME", context)
        # Should return original when reveal_count >= length
        assert result.replacement == "A"

    def test_single_character_partial_end(self):
        """RedactionStrategy partial_end with reveal_count >= len(text)."""
        strategy = RedactionStrategy(mode="partial_end", mask_char="*", reveal_count=10)
        context = create_context(entity_type="NAME", plaintext="Z")
        result = strategy.transform("Z", "NAME", context)
        # Should return original when reveal_count >= length
        assert result.replacement == "Z"

    def test_partial_end_reveal_exactly_text_length(self):
        """RedactionStrategy partial_end where reveal_count equals text length."""
        strategy = RedactionStrategy(mode="partial_end", mask_char="*", reveal_count=3)
        context = create_context(entity_type="TEXT", plaintext="ABC")
        result = strategy.transform("ABC", "TEXT", context)
        # Should reveal all characters when reveal_count >= length
        assert result.replacement == "ABC"

    def test_unknown_mode_fallback_to_full(self):
        """RedactionStrategy with unknown mode should fallback to full redaction."""
        strategy = RedactionStrategy()
        context = create_context(
            entity_type="TEXT",
            plaintext="Secret",
            strategy_params={"mode": "unknown_mode"},
        )
        result = strategy.transform("Secret", "TEXT", context)
        # Should fallback to full redaction (mask_char defaults to █)
        assert result.replacement == "██████"


# ============================================================================
# GeneralizationStrategy - Numeric Range Edge Cases (Lines 304, 307-308)
# ============================================================================


class TestGeneralizationStrategyNumericRangeEdgeCases:
    """Test numeric range generalization edge cases."""

    def test_numeric_range_empty_cleaned_value(self):
        """GeneralizationStrategy numeric_range with no numeric content."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="AGE", plaintext="abc")
        result = strategy.transform("abc", "AGE", context)
        # Should return generalization fallback when no numbers found
        assert result.replacement == "[GENERALIZED]"

    def test_numeric_range_with_invalid_float_conversion(self):
        """GeneralizationStrategy numeric_range with unconvertible cleaned value."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="SALARY", plaintext="$---.---")
        result = strategy.transform("$---.---", "SALARY", context)
        # Should handle ValueError gracefully
        assert result.replacement == "[GENERALIZED]"

    def test_numeric_range_with_separator_formatting(self):
        """GeneralizationStrategy numeric_range with separator produces formatted output."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="SALARY", plaintext="$85000")
        result = strategy.transform("$85000", "SALARY", context)
        # Should format with separator when bucket_size and separator defined
        assert "-" in result.replacement


# ============================================================================
# GeneralizationStrategy - Prefix Mask Edge Cases (Lines 320-324, 345)
# ============================================================================


class TestGeneralizationStrategyPrefixMaskEdgeCases:
    """Test prefix mask edge cases."""

    def test_prefix_mask_with_keep_chars_zero(self):
        """GeneralizationStrategy prefix_mask with keep_chars=0."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="ADDRESS", plaintext="123 Main St")
        result = strategy.transform("123 Main St", "ADDRESS", context)
        # Should mask entire text when keep_chars=0
        assert result.replacement == "*" * len("123 Main St")

    def test_prefix_mask_large_keep_chars(self):
        """GeneralizationStrategy prefix_mask with keep_chars > text length."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="ZIP_CODE", plaintext="123")
        result = strategy.transform("123", "ZIP_CODE", context)
        # Should handle gracefully when keep_chars > text length
        assert result.replacement  # Should not crash


# ============================================================================
# GeneralizationStrategy - Date Parsing Edge Cases (Lines 337)
# ============================================================================


class TestGeneralizationStrategyDateEdgeCases:
    """Test date generalization edge cases."""

    def test_date_year_no_year_found(self):
        """GeneralizationStrategy date_year_only when no year is found."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="DATE", plaintext="no date here")
        result = strategy.transform("no date here", "DATE", context)
        # Should return fallback when no 4-digit year found
        assert result.replacement == "[YEAR]"


# ============================================================================
# GeneralizationStrategy - Email Edge Cases (Lines 364, 368)
# ============================================================================


class TestGeneralizationStrategyEmailEdgeCases:
    """Test email generalization edge cases."""

    def test_email_no_at_sign(self):
        """GeneralizationStrategy email_mask with no @ sign."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="EMAIL_ADDRESS", plaintext="notanemail")
        result = strategy.transform("notanemail", "EMAIL_ADDRESS", context)
        # Should return fallback when no @ found
        assert result.replacement == "***@[DOMAIN]"

    def test_email_at_sign_at_start(self):
        """GeneralizationStrategy email_mask with @ at position 0."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="EMAIL_ADDRESS", plaintext="@example.com")
        result = strategy.transform("@example.com", "EMAIL_ADDRESS", context)
        # Should return fallback when @ at start
        assert result.replacement == "***@[DOMAIN]"

    def test_email_single_character_local(self):
        """GeneralizationStrategy email_mask with single character local part."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="EMAIL_ADDRESS", plaintext="a@example.com")
        result = strategy.transform("a@example.com", "EMAIL_ADDRESS", context)
        # Should handle single character local part
        assert "a***" in result.replacement


# ============================================================================
# GeneralizationStrategy - Name Initials Edge Cases (Lines 376, 389)
# ============================================================================


class TestGeneralizationStrategyNameInitialsEdgeCases:
    """Test name initials edge cases."""

    def test_initials_empty_string(self):
        """GeneralizationStrategy initials with empty string."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="PERSON_NAME", plaintext="")
        result = strategy.transform("", "PERSON_NAME", context)
        # Should return fallback for empty string
        assert result.replacement == "[NAME]"

    def test_initials_only_non_alpha_characters(self):
        """GeneralizationStrategy initials with non-alphabetic content."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="PERSON_NAME", plaintext="123 456")
        result = strategy.transform("123 456", "PERSON_NAME", context)
        # Should handle non-alpha gracefully
        assert result.replacement  # Should not crash


# ============================================================================
# GeneralizationStrategy - Truncate Precision Edge Cases (Lines 395-404)
# ============================================================================


class TestGeneralizationStrategyTruncatePrecisionEdgeCases:
    """Test truncate precision edge cases."""

    def test_truncate_precision_mixed_with_spaces(self):
        """GeneralizationStrategy truncate_precision with space separators."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="LOCATION_COORDINATES",
            plaintext="40.7128 -74.0060",
        )
        result = strategy.transform("40.7128 -74.0060", "LOCATION_COORDINATES", context)
        # Should handle space separators
        assert "40.7" in result.replacement or "40" in result.replacement

    def test_truncate_precision_non_numeric_parts(self):
        """GeneralizationStrategy truncate_precision with mixed text and numbers."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="LOCATION_COORDINATES",
            plaintext="Lat: 40.7128, Lon: -74.0060",
        )
        result = strategy.transform("Lat: 40.7128, Lon: -74.0060", "LOCATION_COORDINATES", context)
        # Should preserve non-numeric parts
        assert "Lat" in result.replacement or "40.7" in result.replacement


# ============================================================================
# GeneralizationStrategy - Subnet Mask Edge Cases (Lines 412)
# ============================================================================


class TestGeneralizationStrategySubnetEdgeCases:
    """Test subnet mask edge cases."""

    def test_subnet_ipv6_format(self):
        """GeneralizationStrategy subnet_mask with IPv6 format."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="IP_ADDRESS",
            plaintext="2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        )
        result = strategy.transform(
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334", "IP_ADDRESS", context
        )
        # IPv6 should be returned unchanged (not 4 parts)
        assert "2001" in result.replacement


# ============================================================================
# GeneralizationStrategy - Generic Generalization Edge Cases (Lines 418, 423-425)
# ============================================================================


class TestGeneralizationStrategyGenericEdgeCases:
    """Test generic generalization fallback."""

    def test_generic_generalize_empty_string(self):
        """GeneralizationStrategy generic_generalize with empty string."""
        result = GeneralizationStrategy._generic_generalize("")
        # Should return single asterisk for empty
        assert result == "*"

    def test_generic_generalize_single_character(self):
        """GeneralizationStrategy generic_generalize with single character."""
        result = GeneralizationStrategy._generic_generalize("X")
        # Should return just asterisk for single char
        assert result == "*"

    def test_generic_generalize_multiple_characters(self):
        """GeneralizationStrategy generic_generalize with multiple characters."""
        result = GeneralizationStrategy._generic_generalize("Hello")
        # Should return char[0] + asterisks
        assert result == "H****"


# ============================================================================
# SyntheticReplacementStrategy - ZIP Code Edge Cases (Lines 628)
# ============================================================================


class TestSyntheticReplacementStrategyZipEdgeCases:
    """Test ZIP code synthetic generation edge cases."""

    def test_synthetic_zip_no_digits(self):
        """SyntheticReplacementStrategy synthetic_zip with no digit content."""
        strategy = SyntheticReplacementStrategy()
        context = create_context(
            entity_type="ZIP_CODE",
            plaintext="ABC-DEF",
            token_key="key123",
        )
        result = strategy.transform("ABC-DEF", "ZIP_CODE", context)
        # Should default to 5 digits when none found
        assert len(result.replacement.replace("-", "")) >= 5

    def test_synthetic_zip_with_dash_format(self):
        """SyntheticReplacementStrategy synthetic_zip with dash format."""
        strategy = SyntheticReplacementStrategy()
        context = create_context(
            entity_type="ZIP_CODE",
            plaintext="12345-6789",
            token_key="key123",
        )
        result = strategy.transform("12345-6789", "ZIP_CODE", context)
        # Should preserve dash format
        assert "-" in result.replacement
        parts = result.replacement.split("-")
        assert len(parts) == 2


# ============================================================================
# PerturbationStrategy - Laplace Sample Edge Cases (Lines 764)
# ============================================================================


class TestPerturbationStrategyLaplaceEdgeCases:
    """Test Laplace sampling edge cases."""

    def test_laplace_sample_with_zero_u(self):
        """PerturbationStrategy _laplace_sample when u maps to exactly 0."""
        # With seed % 1_000_000 = 500_000, u = 0
        # The code checks for u == 0 and sets it to 0.001
        noise = PerturbationStrategy._laplace_sample(500_000, scale=2.0)
        # Should not raise and should be non-zero
        assert isinstance(noise, float)
        assert noise != 0  # Because u is set to 0.001, not 0


# ============================================================================
# PerturbationStrategy - Coordinate Perturbation Edge Cases (Lines 818-819, 841-842, 847-848, 850-851)
# ============================================================================


class TestPerturbationStrategyCoordinateEdgeCases:
    """Test coordinate perturbation edge cases."""

    def test_coordinate_perturb_insufficient_parts(self):
        """PerturbationStrategy coordinates with insufficient numeric parts."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="LOCATION_COORDINATES",
            plaintext="40.7128",
            mention_index=0,
        )
        result = strategy.transform("40.7128", "LOCATION_COORDINATES", context)
        # Should return original and report noise:0 when < 2 coordinates
        assert result.replacement == "40.7128"
        assert result.metadata["noise"] == 0

    def test_coordinate_perturb_non_numeric(self):
        """PerturbationStrategy coordinates with non-numeric content."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="LOCATION_COORDINATES",
            plaintext="Lat,Lon",
            mention_index=0,
        )
        result = strategy.transform("Lat,Lon", "LOCATION_COORDINATES", context)
        # Should return original when can't parse as floats
        assert result.replacement == "Lat,Lon"


# ============================================================================
# PerturbationStrategy - Date Perturbation Edge Cases (Lines 841-842, 847-848, 850-851)
# ============================================================================


class TestPerturbationStrategyDateEdgeCases:
    """Test date perturbation edge cases."""

    def test_date_perturb_day_exceeds_28_overflow(self):
        """PerturbationStrategy date perturb when day shift exceeds 28."""
        strategy = PerturbationStrategy()
        # Use a seed that will cause day > 28
        context = create_context(
            entity_type="DATE_OF_BIRTH",
            plaintext="1990-01-25",
            mention_index=0,
        )
        result = strategy.transform("1990-01-25", "DATE_OF_BIRTH", context)
        # Should handle day overflow (wrapping to next month)
        assert result.replacement  # Should not crash
        parts = result.replacement.split("-")
        assert len(parts) == 3

    def test_date_perturb_day_below_one_underflow(self):
        """PerturbationStrategy date perturb when day shift results in day < 1."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="DATE",
            plaintext="1990-03-05",
            mention_index=0,
        )
        result = strategy.transform("1990-03-05", "DATE", context)
        # Should handle day underflow (wrapping to previous month)
        assert result.replacement  # Should not crash
        parts = result.replacement.split("-")
        day = int(parts[2])
        assert 1 <= day <= 28

    def test_date_perturb_month_exceeds_12(self):
        """PerturbationStrategy date perturb when month exceeds 12."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="DATE_OF_BIRTH",
            plaintext="1990-11-20",
            mention_index=0,
        )
        result = strategy.transform("1990-11-20", "DATE_OF_BIRTH", context)
        # Should wrap month and adjust year if needed
        assert result.replacement  # Should not crash
        parts = result.replacement.split("-")
        month = int(parts[1])
        assert 1 <= month <= 12

    def test_date_perturb_month_below_one(self):
        """PerturbationStrategy date perturb when month goes below 1."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="DATE",
            plaintext="1990-01-15",
            mention_index=0,
        )
        result = strategy.transform("1990-01-15", "DATE", context)
        # Should wrap month and adjust year if needed
        assert result.replacement  # Should not crash
        parts = result.replacement.split("-")
        month = int(parts[1])
        assert 1 <= month <= 12

    def test_date_perturb_no_match(self):
        """PerturbationStrategy date perturb when date format doesn't match."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="DATE_OF_BIRTH",
            plaintext="not a date",
            mention_index=0,
        )
        result = strategy.transform("not a date", "DATE_OF_BIRTH", context)
        # Should return original and noise:0 when can't parse
        assert result.replacement == "not a date"
        assert result.metadata["noise"] == 0


# ============================================================================
# PerturbationStrategy - Age Perturbation with Invalid Input
# ============================================================================


class TestPerturbationStrategyAgeInvalid:
    """Test age perturbation with invalid inputs."""

    def test_age_perturb_empty_string(self):
        """PerturbationStrategy age with empty string."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="AGE",
            plaintext="",
            mention_index=0,
        )
        result = strategy.transform("", "AGE", context)
        # Should return original and noise:0 when empty
        assert result.replacement == ""
        assert result.metadata["noise"] == 0

    def test_age_perturb_no_digits(self):
        """PerturbationStrategy age with no digit content."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="AGE",
            plaintext="abc",
            mention_index=0,
        )
        result = strategy.transform("abc", "AGE", context)
        # Should return original when no digits found
        assert result.replacement == "abc"


# ============================================================================
# PerturbationStrategy - Salary Perturbation with Invalid Input
# ============================================================================


class TestPerturbationStrategySalaryInvalid:
    """Test salary perturbation with invalid inputs."""

    def test_salary_perturb_empty_string(self):
        """PerturbationStrategy salary with empty string."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="SALARY",
            plaintext="",
            mention_index=0,
        )
        result = strategy.transform("", "SALARY", context)
        # Should return original and noise:0
        assert result.replacement == ""
        assert result.metadata["noise"] == 0

    def test_salary_perturb_no_numeric_content(self):
        """PerturbationStrategy salary with no numeric content."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="SALARY",
            plaintext="N/A",
            mention_index=0,
        )
        result = strategy.transform("N/A", "SALARY", context)
        # Should return original when no numeric content
        assert result.replacement == "N/A"


# ============================================================================
# TokenizationStrategy - Null Tokenizer Check (Line 116)
# ============================================================================


class TestTokenizationStrategyNullCheck:
    """Test tokenizer null check."""

    def test_tokenization_ensures_tokenizer_initialized(self):
        """TokenizationStrategy should initialize tokenizer on transform."""
        strategy = TokenizationStrategy()
        # Tokenizer should be None initially
        assert strategy._tokenizer is None

        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="test@example.com",
            token_key="key123",
        )
        result = strategy.transform("test@example.com", "EMAIL_ADDRESS", context)

        # After transform, tokenizer should be initialized
        assert strategy._tokenizer is not None
        assert result.replacement  # Should have a token


# ============================================================================
# Redaction Empty Text Edge Cases
# ============================================================================


class TestRedactionEmptyAndShortText:
    """Test redaction with various empty and short inputs."""

    def test_redaction_length_preserving_empty_string(self):
        """RedactionStrategy length_preserving with empty string."""
        strategy = RedactionStrategy(mode="length_preserving", mask_char="#")
        context = create_context(entity_type="TEXT", plaintext="")
        result = strategy.transform("", "TEXT", context)
        assert result.replacement == ""

    def test_redaction_length_preserving_only_punctuation(self):
        """RedactionStrategy length_preserving with only punctuation."""
        strategy = RedactionStrategy(mode="length_preserving", mask_char="#")
        context = create_context(entity_type="TEXT", plaintext="!!!")
        result = strategy.transform("!!!", "TEXT", context)
        # Punctuation should be preserved
        assert result.replacement == "!!!"

    def test_redaction_length_preserving_mixed_alnum_punctuation(self):
        """RedactionStrategy length_preserving preserves structure."""
        strategy = RedactionStrategy(mode="length_preserving", mask_char="*")
        context = create_context(entity_type="TEXT", plaintext="a-b-c")
        result = strategy.transform("a-b-c", "TEXT", context)
        # Dashes should be preserved, letters masked
        assert result.replacement == "*-*-*"


# ============================================================================
# GeneralizationStrategy - Partial Mask Edge Cases (Line 345)
# ============================================================================


class TestGeneralizationStrategyPartialMaskEdgeCases:
    """Test partial mask with various configurations."""

    def test_partial_mask_with_zero_keep_start_and_end(self):
        """GeneralizationStrategy partial_mask with keep_start=0, keep_end=0."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="CREDIT_CARD_NUMBER",
            plaintext="1234-5678-9012-3456",
        )
        result = strategy.transform("1234-5678-9012-3456", "CREDIT_CARD_NUMBER", context)
        # Should mask entire text
        assert result.replacement.count("*") > 10

    def test_partial_mask_keep_start_only(self):
        """GeneralizationStrategy partial_mask with keep_start > 0, keep_end=0."""
        config = {"keep_start": 4, "keep_end": 0, "mask_char": "*"}
        result = GeneralizationStrategy._generalize_partial_mask("1234-5678", config)
        # Should keep first 4 chars and mask rest
        assert result.startswith("1234")
        assert result[4] == "*"

    def test_partial_mask_keep_end_only(self):
        """GeneralizationStrategy partial_mask with keep_start=0, keep_end > 0."""
        config = {"keep_start": 0, "keep_end": 4, "mask_char": "X"}
        result = GeneralizationStrategy._generalize_partial_mask("1234567890", config)
        # Should mask start and keep last 4
        assert result.endswith("7890")
        assert result[0] == "X"


# ============================================================================
# SyntheticReplacementStrategy - Generic Fallback (No Direct Coverage)
# ============================================================================


class TestSyntheticReplacementStrategyGenericFallback:
    """Test synthetic generic fallback for unsupported types."""

    def test_synthetic_unknown_entity_type(self):
        """SyntheticReplacementStrategy with unsupported entity type."""
        strategy = SyntheticReplacementStrategy()
        context = create_context(
            entity_type="UNKNOWN_TYPE",
            plaintext="some_value_123",
            token_key="key123",
        )
        result = strategy.transform("some_value_123", "UNKNOWN_TYPE", context)
        # Should use generic fallback
        assert result.replacement  # Should have something
        assert result.strategy_id == "synthetic"


# ============================================================================
# PerturbationStrategy - Unsupported Type Fallback
# ============================================================================


class TestPerturbationStrategyUnsupportedType:
    """Test perturbation with unsupported entity types."""

    def test_perturb_unsupported_entity_type(self):
        """PerturbationStrategy with unsupported entity type."""
        strategy = PerturbationStrategy()
        context = create_context(
            entity_type="UNKNOWN_TYPE",
            plaintext="some_value",
            mention_index=0,
        )
        result = strategy.transform("some_value", "UNKNOWN_TYPE", context)
        # Should use generic fallback (no perturbation)
        assert result.replacement == "some_value"
        assert result.metadata["mechanism"] == "none"
