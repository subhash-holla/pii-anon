"""Tests for transformation policies and compliance templates."""

import pytest

from pii_anon.transforms.policies import (
    EntityTransformRule,
    TransformPolicy,
    load_compliance_template,
    list_compliance_templates,
)


class TestEntityTransformRule:
    """Tests for EntityTransformRule dataclass."""

    def test_entity_transform_rule_creation(self):
        """Test creating an EntityTransformRule."""
        rule = EntityTransformRule(
            entity_type="EMAIL_ADDRESS",
            strategy="redact",
        )
        assert rule.entity_type == "EMAIL_ADDRESS"
        assert rule.strategy == "redact"
        assert rule.params == {}
        assert rule.min_confidence == 0.0
        assert rule.fallback_strategy == "redact"

    def test_entity_transform_rule_with_params(self):
        """Test EntityTransformRule with parameters."""
        rule = EntityTransformRule(
            entity_type="ZIP_CODE",
            strategy="generalize",
            params={"keep_chars": 3},
            min_confidence=0.5,
            fallback_strategy="redact",
        )
        assert rule.params == {"keep_chars": 3}
        assert rule.min_confidence == 0.5


class TestTransformPolicy:
    """Tests for TransformPolicy class."""

    def test_empty_policy(self):
        """Test creating an empty policy."""
        policy = TransformPolicy()
        assert len(policy) == 0
        assert policy.list_rules() == []
        assert policy.entity_types() == []

    def test_set_rule_simple(self):
        """Test setting a simple rule."""
        policy = TransformPolicy()
        policy.set_rule("EMAIL_ADDRESS", "redact")

        assert len(policy) == 1
        assert "EMAIL_ADDRESS" in policy
        rule = policy.get_rule("EMAIL_ADDRESS")
        assert rule is not None
        assert rule.entity_type == "EMAIL_ADDRESS"
        assert rule.strategy == "redact"

    def test_set_rule_with_params(self):
        """Test setting a rule with parameters."""
        policy = TransformPolicy()
        policy.set_rule("ZIP_CODE", "generalize", keep_chars=3)

        rule = policy.get_rule("ZIP_CODE")
        assert rule.params == {"keep_chars": 3}

    def test_set_rule_with_confidence_and_fallback(self):
        """Test setting rule with min_confidence and fallback_strategy."""
        policy = TransformPolicy()
        policy.set_rule(
            "PERSON_NAME",
            "synthetic",
            min_confidence=0.8,
            fallback_strategy="generalize",
        )

        rule = policy.get_rule("PERSON_NAME")
        assert rule.min_confidence == 0.8
        assert rule.fallback_strategy == "generalize"

    def test_set_rule_overwrite(self):
        """Test overwriting an existing rule."""
        policy = TransformPolicy()
        policy.set_rule("EMAIL_ADDRESS", "redact")
        policy.set_rule("EMAIL_ADDRESS", "synthetic")

        rule = policy.get_rule("EMAIL_ADDRESS")
        assert rule.strategy == "synthetic"
        assert len(policy) == 1

    def test_get_rule_nonexistent(self):
        """Test getting a rule that doesn't exist."""
        policy = TransformPolicy()
        rule = policy.get_rule("NONEXISTENT")
        assert rule is None

    def test_list_rules(self):
        """Test listing all rules."""
        policy = TransformPolicy()
        policy.set_rule("EMAIL_ADDRESS", "redact")
        policy.set_rule("PERSON_NAME", "synthetic")
        policy.set_rule("ZIP_CODE", "generalize")

        rules = policy.list_rules()
        assert len(rules) == 3
        strategies = {r.entity_type: r.strategy for r in rules}
        assert strategies == {
            "EMAIL_ADDRESS": "redact",
            "PERSON_NAME": "synthetic",
            "ZIP_CODE": "generalize",
        }

    def test_entity_types_sorted(self):
        """Test entity_types returns sorted list."""
        policy = TransformPolicy()
        policy.set_rule("ZEBRA", "redact")
        policy.set_rule("ALPHA", "redact")
        policy.set_rule("BETA", "redact")

        entity_types = policy.entity_types()
        assert entity_types == ["ALPHA", "BETA", "ZEBRA"]

    def test_len(self):
        """Test len() on policy."""
        policy = TransformPolicy()
        assert len(policy) == 0

        policy.set_rule("EMAIL_ADDRESS", "redact")
        assert len(policy) == 1

        policy.set_rule("PERSON_NAME", "synthetic")
        assert len(policy) == 2

    def test_contains(self):
        """Test 'in' operator on policy."""
        policy = TransformPolicy()
        policy.set_rule("EMAIL_ADDRESS", "redact")

        assert "EMAIL_ADDRESS" in policy
        assert "PERSON_NAME" not in policy

    def test_to_profile_overrides_empty(self):
        """Test to_profile_overrides on empty policy."""
        policy = TransformPolicy()
        entity_strategies, strategy_params = policy.to_profile_overrides()

        assert entity_strategies == {}
        assert strategy_params == {}

    def test_to_profile_overrides_simple(self):
        """Test to_profile_overrides with simple rules."""
        policy = TransformPolicy()
        policy.set_rule("EMAIL_ADDRESS", "redact")
        policy.set_rule("PERSON_NAME", "synthetic")

        entity_strategies, strategy_params = policy.to_profile_overrides()

        assert entity_strategies == {
            "EMAIL_ADDRESS": "redact",
            "PERSON_NAME": "synthetic",
        }
        assert strategy_params == {}

    def test_to_profile_overrides_with_params(self):
        """Test to_profile_overrides with parameters."""
        policy = TransformPolicy()
        policy.set_rule("ZIP_CODE", "generalize", keep_chars=3)
        policy.set_rule("LOCATION_COORDINATES", "perturb", sigma=0.5)

        entity_strategies, strategy_params = policy.to_profile_overrides()

        assert entity_strategies == {
            "ZIP_CODE": "generalize",
            "LOCATION_COORDINATES": "perturb",
        }
        assert strategy_params == {
            "generalize": {"keep_chars": 3},
            "perturb": {"sigma": 0.5},
        }

    def test_to_profile_overrides_same_strategy_multiple_entities(self):
        """Test to_profile_overrides with same strategy for multiple entities."""
        policy = TransformPolicy()
        policy.set_rule("EMAIL_ADDRESS", "redact", mode="full")
        policy.set_rule("PHONE_NUMBER", "redact", mode="full")

        entity_strategies, strategy_params = policy.to_profile_overrides()

        assert entity_strategies == {
            "EMAIL_ADDRESS": "redact",
            "PHONE_NUMBER": "redact",
        }
        assert strategy_params == {"redact": {"mode": "full"}}

    def test_merge_empty_policies(self):
        """Test merging two empty policies."""
        policy1 = TransformPolicy()
        policy2 = TransformPolicy()

        merged = policy1.merge(policy2)

        assert len(merged) == 0

    def test_merge_with_empty(self):
        """Test merging a policy with an empty one."""
        policy1 = TransformPolicy()
        policy1.set_rule("EMAIL_ADDRESS", "redact")

        policy2 = TransformPolicy()

        merged = policy1.merge(policy2)
        assert len(merged) == 1
        assert merged.get_rule("EMAIL_ADDRESS").strategy == "redact"

    def test_merge_disjoint_rules(self):
        """Test merging policies with disjoint rules."""
        policy1 = TransformPolicy()
        policy1.set_rule("EMAIL_ADDRESS", "redact")
        policy1.set_rule("PERSON_NAME", "synthetic")

        policy2 = TransformPolicy()
        policy2.set_rule("ZIP_CODE", "generalize")

        merged = policy1.merge(policy2)

        assert len(merged) == 3
        assert "EMAIL_ADDRESS" in merged
        assert "PERSON_NAME" in merged
        assert "ZIP_CODE" in merged

    def test_merge_overlapping_rules_other_takes_precedence(self):
        """Test that in merge, other policy takes precedence."""
        policy1 = TransformPolicy()
        policy1.set_rule("EMAIL_ADDRESS", "redact")

        policy2 = TransformPolicy()
        policy2.set_rule("EMAIL_ADDRESS", "synthetic")

        merged = policy1.merge(policy2)

        assert len(merged) == 1
        rule = merged.get_rule("EMAIL_ADDRESS")
        assert rule.strategy == "synthetic"

    def test_merge_does_not_modify_originals(self):
        """Test that merge returns a new policy and doesn't modify originals."""
        policy1 = TransformPolicy()
        policy1.set_rule("EMAIL_ADDRESS", "redact")

        policy2 = TransformPolicy()
        policy2.set_rule("EMAIL_ADDRESS", "synthetic")

        merged = policy1.merge(policy2)

        assert policy1.get_rule("EMAIL_ADDRESS").strategy == "redact"
        assert policy2.get_rule("EMAIL_ADDRESS").strategy == "synthetic"
        assert merged.get_rule("EMAIL_ADDRESS").strategy == "synthetic"


class TestComplianceTemplates:
    """Tests for compliance template loading and listing."""

    def test_load_hipaa_safe_harbor(self):
        """Test loading HIPAA Safe Harbor template."""
        policy = load_compliance_template("hipaa_safe_harbor")

        assert isinstance(policy, TransformPolicy)
        assert "PERSON_NAME" in policy
        assert "EMAIL_ADDRESS" in policy
        assert "US_SSN" in policy
        assert policy.get_rule("PERSON_NAME").strategy == "synthetic"

    def test_load_gdpr_pseudonymization(self):
        """Test loading GDPR pseudonymization template."""
        policy = load_compliance_template("gdpr_pseudonymization")

        assert isinstance(policy, TransformPolicy)
        assert "PERSON_NAME" in policy
        assert "EMAIL_ADDRESS" in policy
        # All entities should use tokenize
        for rule in policy.list_rules():
            assert rule.strategy == "tokenize"

    def test_load_gdpr_anonymization(self):
        """Test loading GDPR anonymization template."""
        policy = load_compliance_template("gdpr_anonymization")

        assert isinstance(policy, TransformPolicy)
        assert "PERSON_NAME" in policy
        assert "EMAIL_ADDRESS" in policy
        # Should have mix of synthetic and redact strategies
        strategies = {r.entity_type: r.strategy for r in policy.list_rules()}
        assert "synthetic" in strategies.values()
        assert "redact" in strategies.values()

    def test_load_ccpa_deidentification(self):
        """Test loading CCPA deidentification template."""
        policy = load_compliance_template("ccpa_deidentification")

        assert isinstance(policy, TransformPolicy)
        assert "PERSON_NAME" in policy
        assert "EMAIL_ADDRESS" in policy
        assert policy.get_rule("PERSON_NAME").strategy == "synthetic"

    def test_load_minimal_risk(self):
        """Test loading minimal risk template."""
        policy = load_compliance_template("minimal_risk")

        assert isinstance(policy, TransformPolicy)
        # Should only have high/critical risk types
        assert "US_SSN" in policy
        # Low-risk types should not be present
        assert "GENDER" not in policy

    def test_load_maximum_privacy(self):
        """Test loading maximum privacy template."""
        policy = load_compliance_template("maximum_privacy")

        assert isinstance(policy, TransformPolicy)
        # Should have many entity types
        assert len(policy) > 20
        # All should be redact
        for rule in policy.list_rules():
            assert rule.strategy == "redact"

    def test_load_invalid_template_raises_valueerror(self):
        """Test that loading invalid template raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_compliance_template("invalid_template_name")

        assert "Unknown compliance template" in str(exc_info.value)
        assert "invalid_template_name" in str(exc_info.value)

    def test_load_invalid_template_lists_available(self):
        """Test that error message lists available templates."""
        with pytest.raises(ValueError) as exc_info:
            load_compliance_template("nonexistent")

        error_msg = str(exc_info.value)
        assert "hipaa_safe_harbor" in error_msg
        assert "gdpr_pseudonymization" in error_msg
        assert "gdpr_anonymization" in error_msg
        assert "ccpa_deidentification" in error_msg
        assert "minimal_risk" in error_msg
        assert "maximum_privacy" in error_msg

    def test_list_compliance_templates(self):
        """Test listing all compliance templates."""
        templates = list_compliance_templates()

        assert isinstance(templates, list)
        assert len(templates) == 6
        assert templates == [
            "ccpa_deidentification",
            "gdpr_anonymization",
            "gdpr_pseudonymization",
            "hipaa_safe_harbor",
            "maximum_privacy",
            "minimal_risk",
        ]

    def test_list_compliance_templates_sorted(self):
        """Test that list_compliance_templates returns sorted list."""
        templates = list_compliance_templates()

        assert templates == sorted(templates)

    def test_all_templates_load_successfully(self):
        """Test that all listed templates can be loaded."""
        templates = list_compliance_templates()

        for template_name in templates:
            policy = load_compliance_template(template_name)
            assert isinstance(policy, TransformPolicy)
            assert len(policy) > 0

    def test_hipaa_specific_rules(self):
        """Test HIPAA Safe Harbor specific entity rules."""
        policy = load_compliance_template("hipaa_safe_harbor")

        # Check specific entity type rules match HIPAA requirements
        assert policy.get_rule("PERSON_NAME").strategy == "synthetic"
        assert policy.get_rule("EMAIL_ADDRESS").strategy == "redact"
        assert policy.get_rule("PHONE_NUMBER").strategy == "redact"
        assert policy.get_rule("US_SSN").strategy == "redact"

    def test_gdpr_pseudonymization_all_tokenize(self):
        """Test GDPR pseudonymization uses tokenize for all entities."""
        policy = load_compliance_template("gdpr_pseudonymization")

        for rule in policy.list_rules():
            assert rule.strategy == "tokenize", (
                f"Rule for {rule.entity_type} has strategy "
                f"{rule.strategy}, expected tokenize"
            )

    def test_maximum_privacy_all_redact(self):
        """Test maximum privacy uses redact for all entities."""
        policy = load_compliance_template("maximum_privacy")

        for rule in policy.list_rules():
            assert rule.strategy == "redact", (
                f"Rule for {rule.entity_type} has strategy "
                f"{rule.strategy}, expected redact"
            )

    def test_ccpa_has_required_entity_types(self):
        """Test CCPA template has required entity types."""
        policy = load_compliance_template("ccpa_deidentification")

        required = [
            "PERSON_NAME",
            "EMAIL_ADDRESS",
            "US_SSN",
            "CREDIT_CARD_NUMBER",
            "ADDRESS",
        ]
        for entity_type in required:
            assert entity_type in policy


class TestPolicyMergeAndCombination:
    """Tests for combining and merging multiple policies."""

    def test_merge_multiple_policies_chain(self):
        """Test merging multiple policies in chain."""
        policy1 = TransformPolicy()
        policy1.set_rule("EMAIL_ADDRESS", "redact")

        policy2 = TransformPolicy()
        policy2.set_rule("PERSON_NAME", "synthetic")

        policy3 = TransformPolicy()
        policy3.set_rule("ZIP_CODE", "generalize")

        merged = policy1.merge(policy2).merge(policy3)

        assert len(merged) == 3
        assert merged.get_rule("EMAIL_ADDRESS").strategy == "redact"
        assert merged.get_rule("PERSON_NAME").strategy == "synthetic"
        assert merged.get_rule("ZIP_CODE").strategy == "generalize"

    def test_to_profile_overrides_complex_scenario(self):
        """Test to_profile_overrides with complex multi-strategy setup."""
        policy = TransformPolicy()
        policy.set_rule("PERSON_NAME", "synthetic")
        policy.set_rule("EMAIL_ADDRESS", "redact", mode="full")
        policy.set_rule("ZIP_CODE", "generalize", keep_chars=3)
        policy.set_rule("PHONE_NUMBER", "redact", mode="full")
        policy.set_rule("LOCATION_COORDINATES", "perturb", sigma=0.5)

        entity_strategies, strategy_params = policy.to_profile_overrides()

        assert len(entity_strategies) == 5
        assert entity_strategies["PERSON_NAME"] == "synthetic"
        assert entity_strategies["EMAIL_ADDRESS"] == "redact"
        assert entity_strategies["ZIP_CODE"] == "generalize"
        assert entity_strategies["PHONE_NUMBER"] == "redact"
        assert entity_strategies["LOCATION_COORDINATES"] == "perturb"

        assert strategy_params["generalize"] == {"keep_chars": 3}
        assert strategy_params["perturb"] == {"sigma": 0.5}
        assert strategy_params["redact"] == {"mode": "full"}
