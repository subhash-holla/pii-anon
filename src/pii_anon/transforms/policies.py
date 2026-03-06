"""Transformation policies and compliance templates.

A ``TransformPolicy`` maps entity types to transformation rules, enabling
per-entity-type strategy selection.  Pre-built ``ComplianceTemplate``
instances encode regulatory requirements from HIPAA, GDPR, CCPA, etc.

Usage::

    policy = load_compliance_template("hipaa_safe_harbor")
    entity_strats, strat_params = policy.to_profile_overrides()
    profile.entity_strategies.update(entity_strats)
    profile.strategy_params.update(strat_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EntityTransformRule:
    """Transformation rule for a single entity type.

    Attributes
    ----------
    entity_type : str
        The entity type this rule applies to.
    strategy : str
        Strategy ID to use (e.g. ``"redact"``, ``"generalize"``).
    params : dict[str, Any]
        Strategy-specific parameters.
    min_confidence : float
        Minimum detection confidence to trigger transformation.
        Findings below this threshold are left untransformed.
    fallback_strategy : str
        Strategy to use if the primary strategy fails or is unavailable.
    """

    entity_type: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    min_confidence: float = 0.0
    fallback_strategy: str = "redact"


class TransformPolicy:
    """Collection of per-entity-type transformation rules.

    A policy maps entity types to ``EntityTransformRule`` instances,
    enabling fine-grained control over how each type of PII is
    transformed.

    Example
    -------
    >>> policy = TransformPolicy()
    >>> policy.set_rule("EMAIL_ADDRESS", "generalize")
    >>> policy.set_rule("AGE", "perturb", epsilon=1.0)
    >>> strats, params = policy.to_profile_overrides()
    """

    def __init__(self) -> None:
        self._rules: dict[str, EntityTransformRule] = {}

    def set_rule(
        self,
        entity_type: str,
        strategy: str,
        *,
        min_confidence: float = 0.0,
        fallback_strategy: str = "redact",
        **params: Any,
    ) -> None:
        """Set the transformation rule for an entity type.

        Parameters
        ----------
        entity_type : str
            Entity type name (e.g. ``"PERSON_NAME"``).
        strategy : str
            Strategy ID.
        min_confidence : float
            Minimum confidence threshold.
        fallback_strategy : str
            Fallback strategy ID.
        **params : Any
            Strategy-specific parameters.
        """
        self._rules[entity_type] = EntityTransformRule(
            entity_type=entity_type,
            strategy=strategy,
            params=dict(params),
            min_confidence=min_confidence,
            fallback_strategy=fallback_strategy,
        )

    def get_rule(self, entity_type: str) -> EntityTransformRule | None:
        """Look up the rule for an entity type."""
        return self._rules.get(entity_type)

    def list_rules(self) -> list[EntityTransformRule]:
        """Return all configured rules."""
        return list(self._rules.values())

    def entity_types(self) -> list[str]:
        """Return entity types with rules."""
        return sorted(self._rules.keys())

    def to_profile_overrides(self) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        """Convert policy to profile-compatible overrides.

        Returns
        -------
        tuple[dict[str, str], dict[str, dict[str, Any]]]
            A tuple of (entity_strategies, strategy_params) suitable for
            merging into ``ProcessingProfileSpec``.
        """
        entity_strategies: dict[str, str] = {}
        strategy_params: dict[str, dict[str, Any]] = {}

        for rule in self._rules.values():
            entity_strategies[rule.entity_type] = rule.strategy
            if rule.params:
                # Merge into per-strategy params bucket
                existing = strategy_params.get(rule.strategy, {})
                existing.update(rule.params)
                strategy_params[rule.strategy] = existing

        return entity_strategies, strategy_params

    def merge(self, other: "TransformPolicy") -> "TransformPolicy":
        """Merge another policy into this one (other takes precedence).

        Returns a new ``TransformPolicy`` with combined rules.
        """
        merged = TransformPolicy()
        for rule in self._rules.values():
            merged._rules[rule.entity_type] = rule
        for rule in other._rules.values():
            merged._rules[rule.entity_type] = rule
        return merged

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, entity_type: str) -> bool:
        return entity_type in self._rules


# ── Pre-built Compliance Templates ────────────────────────────────────────


def _build_hipaa_safe_harbor() -> TransformPolicy:
    """HIPAA Safe Harbor (45 CFR §164.514(b)).

    Implements the 18 Safe Harbor identifiers:
    1. Names → synthetic replacement
    2. Geographic data < state → generalize (3-digit ZIP)
    3. Dates (except year) → generalize (year only)
    4. Phone numbers → redact
    5. Fax numbers → redact
    6. Email addresses → redact
    7. SSN → redact (full)
    8. Medical record numbers → tokenize (reversible for authorized access)
    9. Health plan IDs → tokenize
    10. Account numbers → redact
    11. Certificate/license numbers → redact
    12. Vehicle identifiers → redact
    13. Device identifiers → redact
    14. URLs → redact
    15. IP addresses → generalize (subnet)
    16. Biometric identifiers → redact
    17. Full-face photographs → N/A (not text)
    18. Any other unique number → redact
    """
    policy = TransformPolicy()
    policy.set_rule("PERSON_NAME", "synthetic")
    policy.set_rule("ZIP_CODE", "generalize", keep_chars=3)
    policy.set_rule("ADDRESS", "generalize")
    policy.set_rule("DATE_OF_BIRTH", "generalize")
    policy.set_rule("DATE", "generalize")
    policy.set_rule("AGE", "redact")  # Ages >89 must be aggregated to 90+
    policy.set_rule("PHONE_NUMBER", "redact")
    policy.set_rule("EMAIL_ADDRESS", "redact")
    policy.set_rule("US_SSN", "redact", mode="full")
    policy.set_rule("MEDICAL_RECORD_NUMBER", "tokenize")
    policy.set_rule("HEALTH_INSURANCE_ID", "tokenize")
    policy.set_rule("BANK_ACCOUNT_NUMBER", "redact")
    policy.set_rule("CREDIT_CARD_NUMBER", "redact")
    policy.set_rule("DRIVERS_LICENSE", "redact")
    policy.set_rule("VEHICLE_IDENTIFICATION_NUMBER", "redact")
    policy.set_rule("DEVICE_ID", "redact")
    policy.set_rule("URL_WITH_PII", "redact")
    policy.set_rule("IP_ADDRESS", "generalize")
    policy.set_rule("BIOMETRIC_ID", "redact")
    policy.set_rule("PRESCRIPTION_NUMBER", "redact")
    policy.set_rule("LICENSE_PLATE", "redact")
    policy.set_rule("PASSPORT_NUMBER", "redact")
    policy.set_rule("NATIONAL_ID_NUMBER", "redact")
    return policy


def _build_gdpr_pseudonymization() -> TransformPolicy:
    """GDPR Article 4(5) pseudonymization.

    All personal data is tokenized (reversible) so that data subject
    access requests (DSAR) can be fulfilled.  Tokens are deterministic
    within a scope to maintain referential integrity.
    """
    policy = TransformPolicy()
    # All identifiers → reversible tokenization
    for entity_type in [
        "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS",
        "ZIP_CODE", "DATE_OF_BIRTH", "US_SSN", "PASSPORT_NUMBER",
        "DRIVERS_LICENSE", "NATIONAL_ID_NUMBER", "CREDIT_CARD_NUMBER",
        "BANK_ACCOUNT_NUMBER", "IBAN", "ROUTING_NUMBER", "TAX_ID",
        "MEDICAL_RECORD_NUMBER", "HEALTH_INSURANCE_ID", "EMPLOYEE_ID",
        "IP_ADDRESS", "MAC_ADDRESS", "DEVICE_ID", "URL_WITH_PII",
        "SOCIAL_MEDIA_HANDLE", "USERNAME", "BIOMETRIC_ID",
        "LOCATION_COORDINATES", "SALARY", "AGE",
    ]:
        policy.set_rule(entity_type, "tokenize")
    return policy


def _build_gdpr_anonymization() -> TransformPolicy:
    """GDPR-grade anonymization (irreversible, exits GDPR scope).

    Uses synthetic replacement for structured types and redaction for
    others.  Once applied, data is no longer considered personal data
    under GDPR.
    """
    policy = TransformPolicy()
    synthetic_types = [
        "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS",
        "ZIP_CODE", "DATE_OF_BIRTH", "CREDIT_CARD_NUMBER", "DATE",
    ]
    for et in synthetic_types:
        policy.set_rule(et, "synthetic")

    redact_types = [
        "US_SSN", "PASSPORT_NUMBER", "DRIVERS_LICENSE", "NATIONAL_ID_NUMBER",
        "BANK_ACCOUNT_NUMBER", "IBAN", "ROUTING_NUMBER", "TAX_ID",
        "MEDICAL_RECORD_NUMBER", "HEALTH_INSURANCE_ID", "BIOMETRIC_ID",
        "API_KEY", "AUTHENTICATION_TOKEN", "SOCIAL_MEDIA_HANDLE", "USERNAME",
        "DEVICE_ID", "MAC_ADDRESS", "PRESCRIPTION_NUMBER", "LICENSE_PLATE",
        "VEHICLE_IDENTIFICATION_NUMBER", "SWIFT_BIC_CODE", "CRYPTOCURRENCY_WALLET",
    ]
    for et in redact_types:
        policy.set_rule(et, "redact", mode="full")

    # Generalize remaining quasi-identifiers
    policy.set_rule("AGE", "generalize")
    policy.set_rule("SALARY", "generalize")
    policy.set_rule("IP_ADDRESS", "generalize")
    policy.set_rule("LOCATION_COORDINATES", "perturb", sigma=0.5)
    policy.set_rule("URL_WITH_PII", "redact")
    return policy


def _build_ccpa_deidentification() -> TransformPolicy:
    """CCPA de-identification template.

    Names and emails → synthetic, financial → redact,
    location → generalize, identifiers → tokenize.
    """
    policy = TransformPolicy()
    policy.set_rule("PERSON_NAME", "synthetic")
    policy.set_rule("EMAIL_ADDRESS", "synthetic")
    policy.set_rule("PHONE_NUMBER", "synthetic")
    policy.set_rule("ADDRESS", "generalize")
    policy.set_rule("ZIP_CODE", "generalize", keep_chars=3)
    policy.set_rule("DATE_OF_BIRTH", "generalize")
    policy.set_rule("AGE", "generalize")
    policy.set_rule("US_SSN", "redact", mode="full")
    policy.set_rule("CREDIT_CARD_NUMBER", "redact")
    policy.set_rule("BANK_ACCOUNT_NUMBER", "redact")
    policy.set_rule("DRIVERS_LICENSE", "tokenize")
    policy.set_rule("IP_ADDRESS", "generalize")
    policy.set_rule("LOCATION_COORDINATES", "generalize")
    policy.set_rule("SALARY", "generalize")
    return policy


def _build_minimal_risk() -> TransformPolicy:
    """Minimal risk: only transform HIGH/CRITICAL risk entity types.

    Low and moderate risk types pass through unchanged.  This maximizes
    data utility for use cases where only high-sensitivity PII needs
    protection.
    """
    policy = TransformPolicy()
    # HIGH/CRITICAL risk types only
    high_critical = [
        "US_SSN", "PASSPORT_NUMBER", "CREDIT_CARD_NUMBER", "IBAN",
        "BANK_ACCOUNT_NUMBER", "MEDICAL_RECORD_NUMBER", "HEALTH_INSURANCE_ID",
        "BIOMETRIC_ID", "TAX_ID", "DRIVERS_LICENSE", "NATIONAL_ID_NUMBER",
        "API_KEY", "AUTHENTICATION_TOKEN",
    ]
    for et in high_critical:
        policy.set_rule(et, "redact", mode="full")
    return policy


def _build_maximum_privacy() -> TransformPolicy:
    """Maximum privacy: redact everything with full masking.

    No data utility preservation.  Suitable for public release where
    no PII must be present in any form.
    """
    policy = TransformPolicy()
    all_types = [
        "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS",
        "ZIP_CODE", "DATE_OF_BIRTH", "DATE", "AGE", "GENDER", "NATIONALITY",
        "US_SSN", "PASSPORT_NUMBER", "DRIVERS_LICENSE", "NATIONAL_ID_NUMBER",
        "VISA_NUMBER", "LICENSE_PLATE", "VEHICLE_IDENTIFICATION_NUMBER",
        "CREDIT_CARD_NUMBER", "IBAN", "BANK_ACCOUNT_NUMBER", "ROUTING_NUMBER",
        "SWIFT_BIC_CODE", "CRYPTOCURRENCY_WALLET", "TAX_ID",
        "MEDICAL_RECORD_NUMBER", "HEALTH_INSURANCE_ID", "PRESCRIPTION_NUMBER",
        "MEDICAL_DIAGNOSIS", "BIOMETRIC_ID",
        "IP_ADDRESS", "MAC_ADDRESS", "API_KEY", "AUTHENTICATION_TOKEN",
        "DEVICE_ID", "URL_WITH_PII",
        "EMPLOYEE_ID", "ORGANIZATION", "JOB_TITLE", "SALARY", "EDUCATION_LEVEL",
        "LOCATION_COORDINATES", "ETHNIC_ORIGIN", "RELIGIOUS_BELIEF",
        "POLITICAL_OPINION", "SOCIAL_MEDIA_HANDLE", "USERNAME",
        "MARITAL_STATUS", "HOUSEHOLD_SIZE", "VEHICLE_MODEL",
    ]
    for et in all_types:
        policy.set_rule(et, "redact", mode="full")
    return policy


# ── Template Registry ─────────────────────────────────────────────────────

_COMPLIANCE_TEMPLATES: dict[str, Callable[..., TransformPolicy]] = {
    "hipaa_safe_harbor": _build_hipaa_safe_harbor,
    "gdpr_pseudonymization": _build_gdpr_pseudonymization,
    "gdpr_anonymization": _build_gdpr_anonymization,
    "ccpa_deidentification": _build_ccpa_deidentification,
    "minimal_risk": _build_minimal_risk,
    "maximum_privacy": _build_maximum_privacy,
}


def load_compliance_template(name: str) -> TransformPolicy:
    """Load a pre-built compliance template by name.

    Parameters
    ----------
    name : str
        Template name.  One of: ``"hipaa_safe_harbor"``,
        ``"gdpr_pseudonymization"``, ``"gdpr_anonymization"``,
        ``"ccpa_deidentification"``, ``"minimal_risk"``,
        ``"maximum_privacy"``.

    Returns
    -------
    TransformPolicy
        A configured policy with per-entity-type rules.

    Raises
    ------
    ValueError
        If the template name is not recognized.
    """
    factory = _COMPLIANCE_TEMPLATES.get(name)
    if factory is None:
        available = ", ".join(sorted(_COMPLIANCE_TEMPLATES.keys()))
        raise ValueError(
            f"Unknown compliance template: {name!r}. "
            f"Available templates: {available}"
        )
    return factory()


def list_compliance_templates() -> list[str]:
    """Return sorted list of available compliance template names."""
    return sorted(_COMPLIANCE_TEMPLATES.keys())
