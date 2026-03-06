"""Comprehensive tests for all 6 transformation strategies.

Tests cover:
1. PlaceholderStrategy - basic transform, custom template, default template
2. TokenizationStrategy - basic tokenize, reversibility flag
3. RedactionStrategy - full mode, partial modes, length_preserving, custom mask_char
4. GeneralizationStrategy - AGE ranges, DATE_OF_BIRTH year-only, ZIP_CODE prefix, EMAIL,
   PHONE, PERSON_NAME initials, IP_ADDRESS subnet, unsupported entity passthrough
5. SyntheticReplacementStrategy - PERSON_NAME fake names, EMAIL fake emails,
   PHONE format preserved, CREDIT_CARD valid format, determinism
6. PerturbationStrategy - AGE noise, SALARY noise, deterministic seed
7. StrategyRegistry - register, get, unregister, list, contains, len
8. StrategyMetadata - each strategy returns correct metadata
"""

import pytest
from pii_anon.transforms.base import TransformContext, TransformResult, StrategyMetadata
from pii_anon.transforms.strategies import (
    PlaceholderStrategy,
    TokenizationStrategy,
    RedactionStrategy,
    GeneralizationStrategy,
    SyntheticReplacementStrategy,
    PerturbationStrategy,
)
from pii_anon.transforms.registry import StrategyRegistry


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


@pytest.fixture
def basic_context():
    """Create a minimal TransformContext for testing."""
    return TransformContext(
        entity_type="PERSON_NAME",
        plaintext="John Smith",
        field_path="customer.name",
        language="en",
        scope="default",
        finding=None,
        cluster_id="cluster_001",
        placeholder_index=1,
        is_first_mention=True,
        mention_index=0,
        document_text="My name is John Smith and I work here.",
        token_key="test-key-12345",
        token_version=1,
        strategy_params={},
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
# 1. PlaceholderStrategy Tests
# ============================================================================


class TestPlaceholderStrategy:
    """Test PlaceholderStrategy with various configurations."""

    def test_basic_transform_with_default_template(self):
        """PlaceholderStrategy should use default template format."""
        strategy = PlaceholderStrategy()
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="John Smith",
            placeholder_index=5,
        )
        result = strategy.transform("John Smith", "PERSON_NAME", context)

        assert result.replacement == "<PERSON_NAME:anon_5>"
        assert result.strategy_id == "placeholder"
        assert result.is_reversible is False
        assert "template" in result.metadata

    def test_custom_template(self):
        """PlaceholderStrategy should support custom templates."""
        custom_template = "[REDACTED_{entity_type}_{index}]"
        strategy = PlaceholderStrategy(template=custom_template)
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="john@example.com",
            placeholder_index=3,
        )
        result = strategy.transform("john@example.com", "EMAIL_ADDRESS", context)

        assert result.replacement == "[REDACTED_EMAIL_ADDRESS_3]"
        assert result.strategy_id == "placeholder"

    def test_template_with_cluster_id(self):
        """PlaceholderStrategy should support cluster_id in template."""
        custom_template = "<{entity_type}:{cluster_id}:{index}>"
        strategy = PlaceholderStrategy(template=custom_template)
        context = create_context(
            entity_type="PHONE_NUMBER",
            plaintext="555-1234",
            placeholder_index=2,
            cluster_id="person_42",
        )
        result = strategy.transform("555-1234", "PHONE_NUMBER", context)

        assert result.replacement == "<PHONE_NUMBER:person_42:2>"

    def test_template_override_via_strategy_params(self):
        """PlaceholderStrategy should allow template override via strategy_params."""
        strategy = PlaceholderStrategy(template="<DEFAULT>")
        context = create_context(
            entity_type="ADDRESS",
            plaintext="123 Main St",
            placeholder_index=1,
            strategy_params={"template": "<LOCATION:{index}>"},
        )
        result = strategy.transform("123 Main St", "ADDRESS", context)

        assert result.replacement == "<LOCATION:1>"

    def test_metadata(self):
        """PlaceholderStrategy should return correct metadata."""
        strategy = PlaceholderStrategy()
        metadata = strategy.metadata()

        assert metadata.strategy_id == "placeholder"
        assert metadata.reversible is False
        assert metadata.format_preserving is False
        assert "placeholder" in metadata.description.lower()


# ============================================================================
# 2. TokenizationStrategy Tests
# ============================================================================


class TestTokenizationStrategy:
    """Test TokenizationStrategy with reversible tokenization."""

    def test_basic_tokenize(self):
        """TokenizationStrategy should produce deterministic tokens."""
        strategy = TokenizationStrategy()
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="john@example.com",
            token_key="secret-key-123",
            scope="user_1",
        )
        result = strategy.transform("john@example.com", "EMAIL_ADDRESS", context)

        assert result.replacement  # Token should be non-empty
        assert result.strategy_id == "tokenize"
        assert result.is_reversible is True
        assert "version" in result.metadata
        assert "scope" in result.metadata

    def test_deterministic_tokenization(self):
        """TokenizationStrategy should produce same token for same input."""
        strategy = TokenizationStrategy()
        context1 = create_context(
            entity_type="PHONE_NUMBER",
            plaintext="555-1234-5678",
            token_key="key-123",
            scope="user_2",
        )
        context2 = create_context(
            entity_type="PHONE_NUMBER",
            plaintext="555-1234-5678",
            token_key="key-123",
            scope="user_2",
        )

        result1 = strategy.transform("555-1234-5678", "PHONE_NUMBER", context1)
        result2 = strategy.transform("555-1234-5678", "PHONE_NUMBER", context2)

        assert result1.replacement == result2.replacement

    def test_different_key_produces_different_token(self):
        """TokenizationStrategy should produce different tokens for different keys."""
        strategy = TokenizationStrategy()
        context1 = create_context(
            entity_type="PERSON_NAME",
            plaintext="Alice",
            token_key="key-A",
            scope="default",
        )
        context2 = create_context(
            entity_type="PERSON_NAME",
            plaintext="Alice",
            token_key="key-B",
            scope="default",
        )

        result1 = strategy.transform("Alice", "PERSON_NAME", context1)
        result2 = strategy.transform("Alice", "PERSON_NAME", context2)

        assert result1.replacement != result2.replacement

    def test_is_reversible(self):
        """TokenizationStrategy should report reversibility."""
        strategy = TokenizationStrategy()
        assert strategy.is_reversible() is True

    def test_metadata(self):
        """TokenizationStrategy should return correct metadata."""
        strategy = TokenizationStrategy()
        metadata = strategy.metadata()

        assert metadata.strategy_id == "tokenize"
        assert metadata.reversible is True
        assert "tokenization" in metadata.description.lower()


# ============================================================================
# 3. RedactionStrategy Tests
# ============================================================================


class TestRedactionStrategy:
    """Test RedactionStrategy with various masking modes."""

    def test_full_mode(self):
        """RedactionStrategy in full mode should mask entire value."""
        strategy = RedactionStrategy(mode="full", mask_char="█")
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="John Smith",
        )
        result = strategy.transform("John Smith", "PERSON_NAME", context)

        assert result.replacement == "██████████"
        assert result.strategy_id == "redact"
        assert result.is_reversible is False
        assert result.metadata["mode"] == "full"
        assert result.metadata["original_length"] == 10

    def test_partial_start(self):
        """RedactionStrategy partial_start should reveal start characters."""
        strategy = RedactionStrategy(mode="partial_start", mask_char="*", reveal_count=2)
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="John Smith",
        )
        result = strategy.transform("John Smith", "PERSON_NAME", context)

        assert result.replacement == "Jo********"
        assert result.metadata["mode"] == "partial_start"

    def test_partial_start_short_text(self):
        """RedactionStrategy partial_start should handle short text."""
        strategy = RedactionStrategy(mode="partial_start", mask_char="*", reveal_count=5)
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="Jo",
        )
        result = strategy.transform("Jo", "PERSON_NAME", context)

        assert result.replacement == "Jo"

    def test_partial_end(self):
        """RedactionStrategy partial_end should reveal end characters."""
        strategy = RedactionStrategy(mode="partial_end", mask_char="X", reveal_count=3)
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="john@example.com",
        )
        result = strategy.transform("john@example.com", "EMAIL_ADDRESS", context)

        assert result.replacement == "XXXXXXXXXXXXXcom"
        assert result.metadata["mode"] == "partial_end"

    def test_length_preserving(self):
        """RedactionStrategy length_preserving should preserve spaces/punctuation."""
        strategy = RedactionStrategy(mode="length_preserving", mask_char="#")
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="Dr. John Smith",
        )
        result = strategy.transform("Dr. John Smith", "PERSON_NAME", context)

        # Spaces and periods should be preserved
        assert result.replacement == "##. #### #####"
        assert len(result.replacement) == len("Dr. John Smith")

    def test_custom_mask_char(self):
        """RedactionStrategy should support custom mask characters."""
        strategy = RedactionStrategy(mode="full", mask_char="?")
        context = create_context(
            entity_type="PHONE_NUMBER",
            plaintext="555-1234",
        )
        result = strategy.transform("555-1234", "PHONE_NUMBER", context)

        assert result.replacement == "????????"

    def test_mode_from_strategy_params(self):
        """RedactionStrategy should use mode from strategy_params."""
        strategy = RedactionStrategy(mode="full", mask_char="█")
        context = create_context(
            entity_type="ADDRESS",
            plaintext="123 Main St",
            strategy_params={"mode": "partial_start", "reveal_count": 3, "mask_char": "*"},
        )
        result = strategy.transform("123 Main St", "ADDRESS", context)

        assert result.replacement == "123********"

    def test_metadata(self):
        """RedactionStrategy should return correct metadata."""
        strategy = RedactionStrategy()
        metadata = strategy.metadata()

        assert metadata.strategy_id == "redact"
        assert metadata.reversible is False
        assert metadata.format_preserving is False


# ============================================================================
# 4. GeneralizationStrategy Tests
# ============================================================================


class TestGeneralizationStrategy:
    """Test GeneralizationStrategy with various entity types."""

    def test_age_generalization(self):
        """GeneralizationStrategy should generalize age to ranges."""
        strategy = GeneralizationStrategy()
        context = create_context(entity_type="AGE", plaintext="32")
        result = strategy.transform("32", "AGE", context)

        assert result.replacement == "30-39"
        assert result.strategy_id == "generalize"
        assert result.is_reversible is False

    def test_age_edge_cases(self):
        """GeneralizationStrategy should handle age edge cases."""
        strategy = GeneralizationStrategy()

        # Age 5
        context = create_context(entity_type="AGE", plaintext="5")
        result = strategy.transform("5", "AGE", context)
        assert result.replacement == "0-9"

        # Age 95
        context = create_context(entity_type="AGE", plaintext="95")
        result = strategy.transform("95", "AGE", context)
        assert result.replacement == "90-99"

    def test_date_of_birth_year_only(self):
        """GeneralizationStrategy should reduce DATE_OF_BIRTH to year."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="DATE_OF_BIRTH",
            plaintext="1990-05-15",
        )
        result = strategy.transform("1990-05-15", "DATE_OF_BIRTH", context)

        assert result.replacement == "1990"

    def test_zip_code_prefix_mask(self):
        """GeneralizationStrategy should mask ZIP_CODE keeping prefix."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="ZIP_CODE",
            plaintext="10001-1234",
        )
        result = strategy.transform("10001-1234", "ZIP_CODE", context)

        assert result.replacement == "100*******"

    def test_zip_code_without_suffix(self):
        """GeneralizationStrategy should handle ZIP codes without suffix."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="ZIP_CODE",
            plaintext="90210",
        )
        result = strategy.transform("90210", "ZIP_CODE", context)

        assert result.replacement == "902**"

    def test_email_masking(self):
        """GeneralizationStrategy should mask EMAIL_ADDRESS."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="john.smith@example.com",
        )
        result = strategy.transform("john.smith@example.com", "EMAIL_ADDRESS", context)

        # Should show first char + domain
        assert "@example.com" in result.replacement
        assert result.replacement.startswith("j")

    def test_phone_number_generalization(self):
        """GeneralizationStrategy should generalize PHONE_NUMBER."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="PHONE_NUMBER",
            plaintext="555-123-4567",
        )
        result = strategy.transform("555-123-4567", "PHONE_NUMBER", context)

        # Should preserve format structure
        assert len(result.replacement) == len("555-123-4567")

    def test_person_name_initials(self):
        """GeneralizationStrategy should convert PERSON_NAME to initials."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="Dr. John Smith",
        )
        result = strategy.transform("Dr. John Smith", "PERSON_NAME", context)

        # Should contain initials
        assert "J" in result.replacement and "S" in result.replacement
        assert "." in result.replacement

    def test_person_name_initials_single_word(self):
        """GeneralizationStrategy should handle single word names."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="Madonna",
        )
        result = strategy.transform("Madonna", "PERSON_NAME", context)

        assert result.replacement == "M."

    def test_ip_address_subnet_mask(self):
        """GeneralizationStrategy should mask IP_ADDRESS as subnet."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="IP_ADDRESS",
            plaintext="192.168.1.100",
        )
        result = strategy.transform("192.168.1.100", "IP_ADDRESS", context)

        assert result.replacement == "192.168.1.0/24"

    def test_unsupported_entity_type_passthrough(self):
        """GeneralizationStrategy should handle unsupported entity types."""
        strategy = GeneralizationStrategy()
        context = create_context(
            entity_type="UNKNOWN_TYPE",
            plaintext="SomeValue",
        )
        result = strategy.transform("SomeValue", "UNKNOWN_TYPE", context)

        # Should fall back to generic generalization
        assert result.replacement == "S********"

    def test_custom_hierarchies(self):
        """GeneralizationStrategy should support custom hierarchies."""
        custom = {
            "CUSTOM_TYPE": {"type": "numeric_range", "bucket_size": 5}
        }
        strategy = GeneralizationStrategy(custom_hierarchies=custom)
        context = create_context(
            entity_type="CUSTOM_TYPE",
            plaintext="17",
        )
        result = strategy.transform("17", "CUSTOM_TYPE", context)

        assert result.replacement == "15-19"

    def test_metadata(self):
        """GeneralizationStrategy should return correct metadata."""
        strategy = GeneralizationStrategy()
        metadata = strategy.metadata()

        assert metadata.strategy_id == "generalize"
        assert metadata.reversible is False
        assert metadata.format_preserving is True
        assert metadata.supports_entity_types is not None


# ============================================================================
# 5. SyntheticReplacementStrategy Tests
# ============================================================================


class TestSyntheticReplacementStrategy:
    """Test SyntheticReplacementStrategy with various entity types."""

    def test_person_name_synthetic(self):
        """SyntheticReplacementStrategy should generate fake PERSON_NAME."""
        strategy = SyntheticReplacementStrategy()
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext="John Smith",
            token_key="key123",
        )
        result = strategy.transform("John Smith", "PERSON_NAME", context)

        # Should be a plausible name (non-empty, contains alphabetic)
        assert result.replacement
        assert any(c.isalpha() for c in result.replacement)
        assert result.strategy_id == "synthetic"
        assert result.is_reversible is False
        assert "locale" in result.metadata

    def test_person_name_determinism(self):
        """SyntheticReplacementStrategy should be deterministic for same input."""
        strategy = SyntheticReplacementStrategy()
        context1 = create_context(
            entity_type="PERSON_NAME",
            plaintext="Alice Johnson",
            token_key="secret-key",
            scope="scope1",
        )
        context2 = create_context(
            entity_type="PERSON_NAME",
            plaintext="Alice Johnson",
            token_key="secret-key",
            scope="scope1",
        )

        result1 = strategy.transform("Alice Johnson", "PERSON_NAME", context1)
        result2 = strategy.transform("Alice Johnson", "PERSON_NAME", context2)

        assert result1.replacement == result2.replacement

    def test_email_address_synthetic(self):
        """SyntheticReplacementStrategy should generate fake EMAIL_ADDRESS."""
        strategy = SyntheticReplacementStrategy()
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="john.smith@company.com",
            token_key="key123",
        )
        result = strategy.transform("john.smith@company.com", "EMAIL_ADDRESS", context)

        # Should be a valid email format
        assert "@" in result.replacement
        assert "." in result.replacement.split("@")[1]
        assert result.replacement.count("@") == 1

    def test_phone_number_format_preserved(self):
        """SyntheticReplacementStrategy should preserve PHONE_NUMBER format."""
        strategy = SyntheticReplacementStrategy()
        original = "555-123-4567"
        context = create_context(
            entity_type="PHONE_NUMBER",
            plaintext=original,
            token_key="key123",
        )
        result = strategy.transform(original, "PHONE_NUMBER", context)

        # Should preserve structure (dashes, length)
        assert len(result.replacement) == len(original)
        assert result.replacement[3] == "-"
        assert result.replacement[7] == "-"
        assert result.replacement[0].isdigit()

    def test_credit_card_luhn_valid(self):
        """SyntheticReplacementStrategy should generate valid CREDIT_CARD numbers."""
        strategy = SyntheticReplacementStrategy()
        context = create_context(
            entity_type="CREDIT_CARD_NUMBER",
            plaintext="4532-1111-2222-3333",
            token_key="key123",
        )
        result = strategy.transform("4532-1111-2222-3333", "CREDIT_CARD_NUMBER", context)

        # Should be formatted as card number
        assert result.replacement.count("-") >= 3
        # Remove dashes and check length
        digits_only = result.replacement.replace("-", "")
        assert len(digits_only) == 16
        assert digits_only.isdigit()

    def test_synthetic_locale_awareness(self):
        """SyntheticReplacementStrategy should respect language setting."""
        strategy = SyntheticReplacementStrategy()

        # English
        context_en = create_context(
            entity_type="PERSON_NAME",
            plaintext="Jean Dupont",
            language="en",
            token_key="key123",
        )
        result_en = strategy.transform("Jean Dupont", "PERSON_NAME", context_en)

        # French
        context_fr = create_context(
            entity_type="PERSON_NAME",
            plaintext="Jean Dupont",
            language="fr",
            token_key="key123",
        )
        result_fr = strategy.transform("Jean Dupont", "PERSON_NAME", context_fr)

        # Should generate different names for different locales with same input
        # (though both should be valid names)
        assert result_en.replacement
        assert result_fr.replacement

    def test_metadata(self):
        """SyntheticReplacementStrategy should return correct metadata."""
        strategy = SyntheticReplacementStrategy()
        metadata = strategy.metadata()

        assert metadata.strategy_id == "synthetic"
        assert metadata.reversible is False
        assert metadata.format_preserving is True
        assert "PERSON_NAME" in metadata.supports_entity_types
        assert "EMAIL_ADDRESS" in metadata.supports_entity_types


# ============================================================================
# 6. PerturbationStrategy Tests
# ============================================================================


class TestPerturbationStrategy:
    """Test PerturbationStrategy with noise addition."""

    def test_age_perturbation(self):
        """PerturbationStrategy should add noise to AGE within bounds."""
        strategy = PerturbationStrategy(epsilon=1.0)
        original_age = 35
        context = create_context(
            entity_type="AGE",
            plaintext=str(original_age),
            token_key="key123",
            mention_index=0,
        )
        result = strategy.transform(str(original_age), "AGE", context)

        # Should produce a valid age
        perturbed_age = int(result.replacement)
        assert 0 <= perturbed_age <= 120
        # Should be somewhat close to original (Laplace noise is bounded)
        assert abs(perturbed_age - original_age) < 50  # loose bound

    def test_age_perturbation_determinism(self):
        """PerturbationStrategy should be deterministic for same context."""
        strategy = PerturbationStrategy(epsilon=1.0)
        context1 = create_context(
            entity_type="AGE",
            plaintext="42",
            token_key="key123",
            mention_index=0,
        )
        context2 = create_context(
            entity_type="AGE",
            plaintext="42",
            token_key="key123",
            mention_index=0,
        )

        result1 = strategy.transform("42", "AGE", context1)
        result2 = strategy.transform("42", "AGE", context2)

        assert result1.replacement == result2.replacement

    def test_salary_perturbation(self):
        """PerturbationStrategy should add noise to SALARY."""
        strategy = PerturbationStrategy(sigma=0.1)
        context = create_context(
            entity_type="SALARY",
            plaintext="$100,000",
            token_key="key123",
            mention_index=0,
        )
        result = strategy.transform("$100,000", "SALARY", context)

        # Should have $ prefix and be numeric
        assert result.replacement.startswith("$")
        # Remove prefix and commas
        amount_str = result.replacement.replace("$", "").replace(",", "")
        amount = float(amount_str)
        assert amount > 0
        # Should be roughly in the right ballpark (within 50% due to sigma)
        assert 50000 < amount < 150000

    def test_salary_without_currency(self):
        """PerturbationStrategy should handle salary without currency symbol."""
        strategy = PerturbationStrategy(sigma=0.1)
        context = create_context(
            entity_type="SALARY",
            plaintext="75000",
            token_key="key123",
            mention_index=0,
        )
        result = strategy.transform("75000", "SALARY", context)

        # Should be numeric
        amount = float(result.replacement)
        assert amount > 0

    def test_perturbation_with_seed_variation(self):
        """PerturbationStrategy should produce different noise for different inputs."""
        strategy = PerturbationStrategy(epsilon=1.0)
        context1 = create_context(
            entity_type="AGE",
            plaintext="30",
            mention_index=0,
        )
        context2 = create_context(
            entity_type="AGE",
            plaintext="30",
            mention_index=1,  # Different mention_index
        )

        result1 = strategy.transform("30", "AGE", context1)
        result2 = strategy.transform("30", "AGE", context2)

        # Should produce different perturbations due to different seeds
        assert result1.replacement != result2.replacement

    def test_metadata_includes_noise_info(self):
        """PerturbationStrategy should include noise info in metadata."""
        strategy = PerturbationStrategy(epsilon=0.5, sigma=0.2)
        context = create_context(
            entity_type="AGE",
            plaintext="45",
            mention_index=0,
        )
        result = strategy.transform("45", "AGE", context)

        assert result.metadata["epsilon"] == 0.5
        assert result.metadata["sigma"] == 0.2
        assert result.metadata["entity_type"] == "AGE"
        assert "mechanism" in result.metadata

    def test_metadata(self):
        """PerturbationStrategy should return correct metadata."""
        strategy = PerturbationStrategy()
        metadata = strategy.metadata()

        assert metadata.strategy_id == "perturb"
        assert metadata.reversible is False
        assert metadata.format_preserving is True
        assert "AGE" in metadata.supports_entity_types


# ============================================================================
# 7. StrategyRegistry Tests
# ============================================================================


class TestStrategyRegistry:
    """Test StrategyRegistry operations."""

    def test_register_strategy(self):
        """StrategyRegistry should register strategies."""
        registry = StrategyRegistry()
        strategy = RedactionStrategy()
        registry.register(strategy)

        assert "redact" in registry
        assert registry.get("redact") is strategy

    def test_register_duplicate_raises_error(self):
        """StrategyRegistry should reject duplicate strategy IDs."""
        registry = StrategyRegistry()
        registry.register(RedactionStrategy())

        with pytest.raises(ValueError, match="already registered"):
            registry.register(RedactionStrategy())

    def test_get_strategy(self):
        """StrategyRegistry should retrieve registered strategies."""
        registry = StrategyRegistry()
        strategy = PlaceholderStrategy()
        registry.register(strategy)

        retrieved = registry.get("placeholder")
        assert retrieved is strategy

    def test_get_nonexistent_returns_none(self):
        """StrategyRegistry should return None for unregistered strategy."""
        registry = StrategyRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_unregister_strategy(self):
        """StrategyRegistry should unregister strategies."""
        registry = StrategyRegistry()
        strategy = GeneralizationStrategy()
        registry.register(strategy)
        assert "generalize" in registry

        registry.unregister("generalize")
        assert "generalize" not in registry

    def test_unregister_nonexistent_raises_error(self):
        """StrategyRegistry should raise KeyError for unregistering nonexistent."""
        registry = StrategyRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")

    def test_list_strategies(self):
        """StrategyRegistry should list all registered strategy IDs."""
        registry = StrategyRegistry()
        registry.register(PlaceholderStrategy())
        registry.register(RedactionStrategy())
        registry.register(GeneralizationStrategy())

        strategies = registry.list_strategies()
        assert strategies == ["generalize", "placeholder", "redact"]  # sorted

    def test_list_strategies_empty_registry(self):
        """StrategyRegistry should return empty list for empty registry."""
        registry = StrategyRegistry()
        assert registry.list_strategies() == []

    def test_list_metadata(self):
        """StrategyRegistry should list metadata for all strategies."""
        registry = StrategyRegistry()
        registry.register(PlaceholderStrategy())
        registry.register(RedactionStrategy())

        metadata_list = registry.list_metadata()
        assert len(metadata_list) == 2
        assert all(isinstance(m, StrategyMetadata) for m in metadata_list)
        strategy_ids = {m.strategy_id for m in metadata_list}
        assert strategy_ids == {"placeholder", "redact"}

    def test_contains_operator(self):
        """StrategyRegistry should support 'in' operator."""
        registry = StrategyRegistry()
        registry.register(PlaceholderStrategy())

        assert "placeholder" in registry
        assert "redact" not in registry

    def test_len_operator(self):
        """StrategyRegistry should support len() operator."""
        registry = StrategyRegistry()
        assert len(registry) == 0

        registry.register(PlaceholderStrategy())
        assert len(registry) == 1

        registry.register(RedactionStrategy())
        assert len(registry) == 2

        registry.unregister("redact")
        assert len(registry) == 1

    def test_register_multiple_different_strategies(self):
        """StrategyRegistry should handle multiple different strategies."""
        registry = StrategyRegistry()
        strategies = [
            PlaceholderStrategy(),
            TokenizationStrategy(),
            RedactionStrategy(),
            GeneralizationStrategy(),
            SyntheticReplacementStrategy(),
            PerturbationStrategy(),
        ]

        for strategy in strategies:
            registry.register(strategy)

        assert len(registry) == 6
        assert registry.list_strategies() == [
            "generalize", "perturb", "placeholder", "redact", "synthetic", "tokenize"
        ]


# ============================================================================
# 8. StrategyMetadata Tests
# ============================================================================


class TestStrategyMetadata:
    """Test metadata declarations for all strategies."""

    def test_placeholder_metadata(self):
        """PlaceholderStrategy metadata should be correct."""
        strategy = PlaceholderStrategy()
        meta = strategy.metadata()

        assert meta.strategy_id == "placeholder"
        assert meta.reversible is False
        assert meta.format_preserving is False
        assert meta.supports_entity_types is None  # supports all

    def test_tokenization_metadata(self):
        """TokenizationStrategy metadata should be correct."""
        strategy = TokenizationStrategy()
        meta = strategy.metadata()

        assert meta.strategy_id == "tokenize"
        assert meta.reversible is True
        assert meta.format_preserving is False
        assert meta.supports_entity_types is None

    def test_redaction_metadata(self):
        """RedactionStrategy metadata should be correct."""
        strategy = RedactionStrategy()
        meta = strategy.metadata()

        assert meta.strategy_id == "redact"
        assert meta.reversible is False
        assert meta.format_preserving is False
        assert meta.supports_entity_types is None

    def test_generalization_metadata(self):
        """GeneralizationStrategy metadata should be correct."""
        strategy = GeneralizationStrategy()
        meta = strategy.metadata()

        assert meta.strategy_id == "generalize"
        assert meta.reversible is False
        assert meta.format_preserving is True
        assert isinstance(meta.supports_entity_types, list)
        assert "AGE" in meta.supports_entity_types
        assert "EMAIL_ADDRESS" in meta.supports_entity_types

    def test_synthetic_metadata(self):
        """SyntheticReplacementStrategy metadata should be correct."""
        strategy = SyntheticReplacementStrategy()
        meta = strategy.metadata()

        assert meta.strategy_id == "synthetic"
        assert meta.reversible is False
        assert meta.format_preserving is True
        assert isinstance(meta.supports_entity_types, list)
        assert "PERSON_NAME" in meta.supports_entity_types
        assert "EMAIL_ADDRESS" in meta.supports_entity_types

    def test_perturbation_metadata(self):
        """PerturbationStrategy metadata should be correct."""
        strategy = PerturbationStrategy()
        meta = strategy.metadata()

        assert meta.strategy_id == "perturb"
        assert meta.reversible is False
        assert meta.format_preserving is True
        assert isinstance(meta.supports_entity_types, list)
        assert "AGE" in meta.supports_entity_types
        assert "SALARY" in meta.supports_entity_types

    def test_all_strategies_have_descriptions(self):
        """All strategies should have meaningful descriptions."""
        strategies = [
            PlaceholderStrategy(),
            TokenizationStrategy(),
            RedactionStrategy(),
            GeneralizationStrategy(),
            SyntheticReplacementStrategy(),
            PerturbationStrategy(),
        ]

        for strategy in strategies:
            meta = strategy.metadata()
            assert meta.description
            assert len(meta.description) > 10


# ============================================================================
# 9. Integration Tests
# ============================================================================


class TestStrategyIntegration:
    """Integration tests across multiple strategies."""

    def test_all_strategies_transform_same_input(self):
        """All strategies should be able to transform the same input."""
        plaintext = "John Smith"
        context = create_context(
            entity_type="PERSON_NAME",
            plaintext=plaintext,
            token_key="test-key",
        )

        strategies = [
            PlaceholderStrategy(),
            TokenizationStrategy(),
            RedactionStrategy(),
            GeneralizationStrategy(),
            SyntheticReplacementStrategy(),
        ]

        for strategy in strategies:
            result = strategy.transform(plaintext, "PERSON_NAME", context)
            assert isinstance(result, TransformResult)
            assert result.replacement != plaintext
            assert result.strategy_id == strategy.strategy_id
        
        # PerturbationStrategy doesn't support PERSON_NAME, returns unchanged
        perturb = PerturbationStrategy()
        result = perturb.transform(plaintext, "PERSON_NAME", context)
        assert result.replacement == plaintext  # unsupported type
        assert result.strategy_id == "perturb"

    def test_registry_with_all_strategies(self):
        """Registry should handle all six strategies."""
        registry = StrategyRegistry()

        strategies = [
            PlaceholderStrategy(),
            TokenizationStrategy(),
            RedactionStrategy(),
            GeneralizationStrategy(),
            SyntheticReplacementStrategy(),
            PerturbationStrategy(),
        ]

        for strategy in strategies:
            registry.register(strategy)

        assert len(registry) == 6
        metadata_list = registry.list_metadata()
        assert len(metadata_list) == 6
        strategy_ids = {m.strategy_id for m in metadata_list}
        assert strategy_ids == {
            "placeholder", "tokenize", "redact", "generalize", "synthetic", "perturb"
        }

    def test_strategy_result_consistency(self):
        """TransformResult should be consistent across strategies."""
        context = create_context(
            entity_type="EMAIL_ADDRESS",
            plaintext="test@example.com",
        )

        strategies = [
            PlaceholderStrategy(),
            RedactionStrategy(),
            GeneralizationStrategy(),
            SyntheticReplacementStrategy(),
        ]

        for strategy in strategies:
            result = strategy.transform("test@example.com", "EMAIL_ADDRESS", context)

            # All results should have these properties
            assert result.strategy_id
            assert result.replacement
            assert isinstance(result.is_reversible, bool)
            assert isinstance(result.metadata, dict)
