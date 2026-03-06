"""Regulatory compliance validation for PII de-identification systems.

Validates that a system's entity-type coverage meets the requirements of
specific regulatory standards (NIST SP 800-122, GDPR, ISO 27701).

Evidence basis:
- NIST SP 800-122 (McCallister et al., 2010): US federal PII taxonomy and
  confidentiality impact levels (low / moderate / high).
- GDPR Articles 4 & 9 (European Parliament, 2016): Defines personal data
  (Art. 4) and special-category data (Art. 9) requiring explicit consent.
- ISO/IEC 27701:2019: Privacy information management controls including
  PII de-identification (A.7.4.5) and PII minimisation (A.7.4.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..taxonomy import (
    EntityTypeProfile,
    EntityTypeRegistry,
    PII_TAXONOMY,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Standards enumeration
# ---------------------------------------------------------------------------

class ComplianceStandard(str, Enum):
    """Supported regulatory standards for compliance checking."""

    NIST_SP_800_122 = "nist"
    GDPR = "gdpr"
    ISO_27701 = "iso27701"
    HIPAA = "hipaa"
    CCPA = "ccpa"


# ---------------------------------------------------------------------------
# Standard-specific required entity-type sets
# ---------------------------------------------------------------------------

# NIST SP 800-122 §2.1: "stand-alone PII" + "linked PII" categories that a
# compliant de-identification system MUST be capable of detecting.
_NIST_REQUIRED_TYPES: frozenset[str] = frozenset({
    PII_TAXONOMY.PERSON_NAME,
    PII_TAXONOMY.EMAIL_ADDRESS,
    PII_TAXONOMY.PHONE_NUMBER,
    PII_TAXONOMY.DATE_OF_BIRTH,
    PII_TAXONOMY.ADDRESS,
    PII_TAXONOMY.US_SSN,
    PII_TAXONOMY.PASSPORT_NUMBER,
    PII_TAXONOMY.DRIVERS_LICENSE,
    PII_TAXONOMY.NATIONAL_ID_NUMBER,
    PII_TAXONOMY.CREDIT_CARD_NUMBER,
    PII_TAXONOMY.BANK_ACCOUNT_NUMBER,
    PII_TAXONOMY.TAX_ID,
    PII_TAXONOMY.MEDICAL_RECORD_NUMBER,
    PII_TAXONOMY.BIOMETRIC_ID,
    PII_TAXONOMY.IP_ADDRESS,
    PII_TAXONOMY.EMPLOYEE_ID,
})

# GDPR Art. 4(1) + Art. 9(1): personal data and special categories.
_GDPR_REQUIRED_TYPES: frozenset[str] = frozenset({
    PII_TAXONOMY.PERSON_NAME,
    PII_TAXONOMY.EMAIL_ADDRESS,
    PII_TAXONOMY.PHONE_NUMBER,
    PII_TAXONOMY.DATE_OF_BIRTH,
    PII_TAXONOMY.ADDRESS,
    PII_TAXONOMY.IP_ADDRESS,
    PII_TAXONOMY.LOCATION_COORDINATES,
    PII_TAXONOMY.CREDIT_CARD_NUMBER,
    PII_TAXONOMY.IBAN,
    PII_TAXONOMY.NATIONAL_ID_NUMBER,
    PII_TAXONOMY.PASSPORT_NUMBER,
    # Special categories (Art. 9)
    PII_TAXONOMY.ETHNIC_ORIGIN,
    PII_TAXONOMY.RELIGIOUS_BELIEF,
    PII_TAXONOMY.POLITICAL_OPINION,
    PII_TAXONOMY.MEDICAL_DIAGNOSIS,
    PII_TAXONOMY.BIOMETRIC_ID,
    PII_TAXONOMY.MEDICAL_RECORD_NUMBER,
    PII_TAXONOMY.HEALTH_INSURANCE_ID,
})

# ISO 27701:2019 A.7.4.5 (PII de-identification) — covers a broad set
# including digital identifiers and credentials.
_ISO_REQUIRED_TYPES: frozenset[str] = frozenset({
    PII_TAXONOMY.PERSON_NAME,
    PII_TAXONOMY.EMAIL_ADDRESS,
    PII_TAXONOMY.PHONE_NUMBER,
    PII_TAXONOMY.ADDRESS,
    PII_TAXONOMY.DATE_OF_BIRTH,
    PII_TAXONOMY.CREDIT_CARD_NUMBER,
    PII_TAXONOMY.BANK_ACCOUNT_NUMBER,
    PII_TAXONOMY.MEDICAL_RECORD_NUMBER,
    PII_TAXONOMY.BIOMETRIC_ID,
    PII_TAXONOMY.IP_ADDRESS,
    PII_TAXONOMY.MAC_ADDRESS,
    PII_TAXONOMY.API_KEY,
    PII_TAXONOMY.AUTHENTICATION_TOKEN,
    PII_TAXONOMY.DEVICE_ID,
    PII_TAXONOMY.EMPLOYEE_ID,
})

# HIPAA Safe Harbor — 18 identifiers per 45 CFR §164.514(b)(2).
_HIPAA_REQUIRED_TYPES: frozenset[str] = frozenset({
    PII_TAXONOMY.PERSON_NAME,
    PII_TAXONOMY.ADDRESS,
    PII_TAXONOMY.ZIP_CODE,
    PII_TAXONOMY.DATE_OF_BIRTH,
    PII_TAXONOMY.AGE,
    PII_TAXONOMY.PHONE_NUMBER,
    PII_TAXONOMY.EMAIL_ADDRESS,
    PII_TAXONOMY.US_SSN,
    PII_TAXONOMY.MEDICAL_RECORD_NUMBER,
    PII_TAXONOMY.HEALTH_INSURANCE_ID,
    PII_TAXONOMY.BANK_ACCOUNT_NUMBER,
    PII_TAXONOMY.DRIVERS_LICENSE,
    PII_TAXONOMY.LICENSE_PLATE,
    PII_TAXONOMY.DEVICE_ID,
    PII_TAXONOMY.URL_WITH_PII,
    PII_TAXONOMY.IP_ADDRESS,
    PII_TAXONOMY.BIOMETRIC_ID,
})

# CCPA — California Consumer Privacy Act §1798.140(v).
_CCPA_REQUIRED_TYPES: frozenset[str] = frozenset({
    PII_TAXONOMY.PERSON_NAME,
    PII_TAXONOMY.EMAIL_ADDRESS,
    PII_TAXONOMY.PHONE_NUMBER,
    PII_TAXONOMY.ADDRESS,
    PII_TAXONOMY.US_SSN,
    PII_TAXONOMY.DRIVERS_LICENSE,
    PII_TAXONOMY.PASSPORT_NUMBER,
    PII_TAXONOMY.BANK_ACCOUNT_NUMBER,
    PII_TAXONOMY.CREDIT_CARD_NUMBER,
    PII_TAXONOMY.IP_ADDRESS,
    PII_TAXONOMY.BIOMETRIC_ID,
    PII_TAXONOMY.LOCATION_COORDINATES,
})

_STANDARD_REQUIREMENTS: dict[ComplianceStandard, frozenset[str]] = {
    ComplianceStandard.NIST_SP_800_122: _NIST_REQUIRED_TYPES,
    ComplianceStandard.GDPR: _GDPR_REQUIRED_TYPES,
    ComplianceStandard.ISO_27701: _ISO_REQUIRED_TYPES,
    ComplianceStandard.HIPAA: _HIPAA_REQUIRED_TYPES,
    ComplianceStandard.CCPA: _CCPA_REQUIRED_TYPES,
}

_STANDARD_LABELS: dict[ComplianceStandard, str] = {
    ComplianceStandard.NIST_SP_800_122: "NIST SP 800-122 (2010)",
    ComplianceStandard.GDPR: "GDPR Articles 4 & 9 (2016)",
    ComplianceStandard.ISO_27701: "ISO/IEC 27701:2019",
    ComplianceStandard.HIPAA: "HIPAA Safe Harbor (45 CFR §164.514)",
    ComplianceStandard.CCPA: "CCPA §1798.140(v)",
}


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageGap:
    """A single entity type required by a standard but missing from the system."""

    entity_type: str
    standard: str
    risk_level: str
    category: str
    remediation_hint: str


@dataclass
class ComplianceReport:
    """Full compliance validation report against one or more standards.

    Attributes:
        standard: The primary standard validated against.
        standard_label: Human-readable standard name.
        required_types: Set of entity types required by the standard.
        covered_types: Set of entity types the system supports.
        missing_types: Set of entity types not yet supported.
        coverage_ratio: Fraction of required types that are covered (0.0–1.0).
        compliant: True if coverage_ratio == 1.0.
        gaps: Detailed descriptions of each gap with remediation hints.
        risk_summary: Count of missing types per risk level.
        metadata: Arbitrary extra information.
    """

    standard: str
    standard_label: str
    required_types: frozenset[str]
    covered_types: frozenset[str]
    missing_types: frozenset[str]
    coverage_ratio: float
    compliant: bool
    gaps: list[CoverageGap] = field(default_factory=list)
    risk_summary: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-safe dictionary."""
        return {
            "standard": self.standard,
            "standard_label": self.standard_label,
            "required_types": sorted(self.required_types),
            "covered_types": sorted(self.covered_types),
            "missing_types": sorted(self.missing_types),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "compliant": self.compliant,
            "gaps": [
                {
                    "entity_type": g.entity_type,
                    "standard": g.standard,
                    "risk_level": g.risk_level,
                    "category": g.category,
                    "remediation_hint": g.remediation_hint,
                }
                for g in self.gaps
            ],
            "risk_summary": self.risk_summary,
            "metadata": self.metadata,
        }


@dataclass
class MultiStandardComplianceReport:
    """Aggregated compliance report across multiple standards."""

    reports: list[ComplianceReport] = field(default_factory=list)
    overall_coverage_ratio: float = 0.0
    all_missing_types: frozenset[str] = frozenset()
    fully_compliant_standards: list[str] = field(default_factory=list)
    non_compliant_standards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_coverage_ratio": round(self.overall_coverage_ratio, 4),
            "all_missing_types": sorted(self.all_missing_types),
            "fully_compliant_standards": self.fully_compliant_standards,
            "non_compliant_standards": self.non_compliant_standards,
            "reports": [r.to_dict() for r in self.reports],
        }


# ---------------------------------------------------------------------------
# Compliance validator
# ---------------------------------------------------------------------------

class ComplianceValidator:
    """Validates entity-type coverage against regulatory standards.

    Evidence basis:
    - NIST SP 800-122 §2.1: Stand-alone and linked PII identification.
    - GDPR Art. 4(1)/9(1): Personal data and special-category definitions.
    - ISO 27701 A.7.4.5: PII de-identification controls.
    - HIPAA 45 CFR §164.514(b)(2): Safe Harbor de-identification.
    - CCPA §1798.140(v): Personal information definition.

    Usage::

        validator = ComplianceValidator()
        report = validator.validate(
            entity_types=["PERSON_NAME", "EMAIL_ADDRESS", "US_SSN"],
            standard="nist",
        )
        print(report.coverage_ratio)  # 0.1875 (3 / 16 NIST types)
    """

    def __init__(self) -> None:
        self._registry = EntityTypeRegistry()

    # ── Primary validation ────────────────────────────────────────────

    def validate(
        self,
        entity_types: list[str],
        *,
        standard: str = "nist",
    ) -> ComplianceReport:
        """Validate *entity_types* against a single *standard*.

        Parameters
        ----------
        entity_types:
            Entity-type strings the system currently supports.
        standard:
            Standard identifier — one of ``"nist"``, ``"gdpr"``,
            ``"iso27701"``, ``"hipaa"``, ``"ccpa"``.

        Returns
        -------
        ComplianceReport
            Detailed coverage analysis with gap descriptions.
        """
        std_enum = self._resolve_standard(standard)
        required = _STANDARD_REQUIREMENTS[std_enum]
        covered = frozenset(entity_types) & required
        missing = required - covered
        ratio = len(covered) / len(required) if required else 1.0

        gaps: list[CoverageGap] = []
        risk_summary: dict[str, int] = {}

        for etype in sorted(missing):
            profile = self._registry.get(etype)
            risk = profile.risk_level.value if profile else "unknown"
            cat = profile.category.value if profile else "unknown"
            risk_summary[risk] = risk_summary.get(risk, 0) + 1
            gaps.append(
                CoverageGap(
                    entity_type=etype,
                    standard=std_enum.value,
                    risk_level=risk,
                    category=cat,
                    remediation_hint=self._build_hint(etype, profile, std_enum),
                ),
            )

        return ComplianceReport(
            standard=std_enum.value,
            standard_label=_STANDARD_LABELS[std_enum],
            required_types=required,
            covered_types=covered,
            missing_types=missing,
            coverage_ratio=round(ratio, 6),
            compliant=len(missing) == 0,
            gaps=gaps,
            risk_summary=risk_summary,
        )

    # ── Multi-standard validation ─────────────────────────────────────

    def validate_all(
        self,
        entity_types: list[str],
        *,
        standards: list[str] | None = None,
    ) -> MultiStandardComplianceReport:
        """Validate against multiple standards at once.

        Parameters
        ----------
        entity_types:
            Entity-type strings the system currently supports.
        standards:
            List of standard identifiers.  ``None`` = all supported standards.

        Returns
        -------
        MultiStandardComplianceReport
        """
        if standards is None:
            stds = list(ComplianceStandard)
        else:
            stds = [self._resolve_standard(s) for s in standards]

        reports: list[ComplianceReport] = []
        all_missing: set[str] = set()
        compliant_stds: list[str] = []
        non_compliant_stds: list[str] = []

        for std in stds:
            report = self.validate(entity_types, standard=std.value)
            reports.append(report)
            all_missing |= report.missing_types
            if report.compliant:
                compliant_stds.append(report.standard_label)
            else:
                non_compliant_stds.append(report.standard_label)

        all_required: set[str] = set()
        for std in stds:
            all_required |= _STANDARD_REQUIREMENTS[std]
        overall_covered = frozenset(entity_types) & all_required
        overall_ratio = len(overall_covered) / len(all_required) if all_required else 1.0

        return MultiStandardComplianceReport(
            reports=reports,
            overall_coverage_ratio=round(overall_ratio, 6),
            all_missing_types=frozenset(all_missing),
            fully_compliant_standards=compliant_stds,
            non_compliant_standards=non_compliant_stds,
        )

    # ── Category-level gap analysis ───────────────────────────────────

    def category_gap_analysis(
        self,
        entity_types: list[str],
        *,
        standard: str = "nist",
    ) -> dict[str, dict[str, Any]]:
        """Return per-category coverage statistics.

        Returns a dict mapping category names to::

            {
                "required": int,
                "covered": int,
                "missing": list[str],
                "coverage_ratio": float,
            }
        """
        report = self.validate(entity_types, standard=standard)
        by_cat: dict[str, dict[str, Any]] = {}

        for etype in report.required_types:
            profile = self._registry.get(etype)
            cat = profile.category.value if profile else "unknown"
            entry = by_cat.setdefault(cat, {"required": 0, "covered": 0, "missing": []})
            entry["required"] += 1
            if etype in report.covered_types:
                entry["covered"] += 1
            else:
                entry["missing"].append(etype)

        for entry in by_cat.values():
            entry["coverage_ratio"] = round(
                entry["covered"] / entry["required"] if entry["required"] else 1.0, 4,
            )

        return dict(sorted(by_cat.items()))

    # ── Risk-prioritised remediation plan ─────────────────────────────

    def remediation_plan(
        self,
        entity_types: list[str],
        *,
        standard: str = "nist",
    ) -> list[dict[str, str]]:
        """Return missing types ordered by descending risk severity.

        Each item has keys: ``entity_type``, ``risk_level``, ``category``,
        ``priority`` (1 = highest), and ``hint``.
        """
        report = self.validate(entity_types, standard=standard)
        _priority = {RiskLevel.CRITICAL: 1, RiskLevel.HIGH: 2, RiskLevel.MODERATE: 3, RiskLevel.LOW: 4}

        items: list[tuple[int, dict[str, str]]] = []
        for gap in report.gaps:
            prio = _priority.get(RiskLevel(gap.risk_level), 5) if gap.risk_level != "unknown" else 5
            items.append((prio, {
                "entity_type": gap.entity_type,
                "risk_level": gap.risk_level,
                "category": gap.category,
                "priority": str(prio),
                "hint": gap.remediation_hint,
            }))

        items.sort(key=lambda x: (x[0], x[1]["entity_type"]))
        return [item for _, item in items]

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_standard(raw: str) -> ComplianceStandard:
        """Resolve a string to a ComplianceStandard enum."""
        norm = raw.strip().lower().replace("-", "_").replace(" ", "_")
        aliases: dict[str, ComplianceStandard] = {
            "nist": ComplianceStandard.NIST_SP_800_122,
            "nist_sp_800_122": ComplianceStandard.NIST_SP_800_122,
            "nist800122": ComplianceStandard.NIST_SP_800_122,
            "gdpr": ComplianceStandard.GDPR,
            "iso27701": ComplianceStandard.ISO_27701,
            "iso_27701": ComplianceStandard.ISO_27701,
            "hipaa": ComplianceStandard.HIPAA,
            "ccpa": ComplianceStandard.CCPA,
        }
        result = aliases.get(norm)
        if result is None:
            # Try direct enum value match
            for member in ComplianceStandard:
                if member.value == norm:
                    return member
            valid = ", ".join(sorted(aliases.keys()))
            raise ValueError(f"Unknown standard {raw!r}. Valid aliases: {valid}")
        return result

    @staticmethod
    def _build_hint(
        entity_type: str,
        profile: EntityTypeProfile | None,
        standard: ComplianceStandard,
    ) -> str:
        """Build a human-readable remediation hint for a missing entity type."""
        if profile is None:
            return f"Add detection support for {entity_type}."

        risk = profile.risk_level.value
        cat = profile.category.value

        if profile.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            urgency = "HIGH PRIORITY"
        else:
            urgency = "Recommended"

        refs = [ref for ref in profile.regulatory_refs if standard.value.lower() in ref.standard.lower()]
        ref_str = ""
        if refs:
            ref_str = f" (ref: {refs[0].standard} {refs[0].section})"
        elif profile.regulatory_refs:
            ref_str = f" (ref: {profile.regulatory_refs[0].standard} {profile.regulatory_refs[0].section})"

        return (
            f"[{urgency}] Add {entity_type} detection "
            f"(risk={risk}, category={cat}){ref_str}. "
            f"{profile.description}"
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def validate_compliance(
    entity_types: list[str],
    *,
    standard: str = "nist",
) -> ComplianceReport:
    """Module-level convenience: validate entity types against a standard."""
    return ComplianceValidator().validate(entity_types, standard=standard)


def validate_all_standards(
    entity_types: list[str],
) -> MultiStandardComplianceReport:
    """Module-level convenience: validate against all supported standards."""
    return ComplianceValidator().validate_all(entity_types)
