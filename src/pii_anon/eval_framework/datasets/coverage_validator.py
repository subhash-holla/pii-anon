"""Coverage validation tool for the PII Anonymization Evaluation Dataset v1.0.0.

This module provides comprehensive validation that the evaluation dataset meets all
requirements for the 7 evaluation dimensions:

    1. Entity Tracking         (20%) — consistent coreference across context
    2. Multilingual & Dialect  (15%) — 52 languages, 17 scripts, locale variants
    3. Context Preservation    (20%) — semantic integrity in dialogues/narratives
    4. Diverse PII Types       (20%) — breadth across 48 entity types / 7 categories
    5. Edge Cases              (10%) — abbreviations, partial PII, ambiguity
    6. Data Format Variations  (10%) — structured/semi-structured/unstructured
    7. Temporal Consistency     (5%) — time-series entity evolution

Evidence basis:
    - Sweeney (2002): k-anonymity and re-identification via quasi-identifiers
    - Gebru et al. (2021): Datasheets for Datasets
    - Cochran (1977): Sampling Techniques (statistical power)
    - TAB (2022), PII-Bench (2025), RAT-Bench (2025): PII benchmark standards
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .schema import EvalBenchmarkRecord, VALID_DIMENSION_TAGS, DIMENSION_WEIGHTS, load_eval_dataset


# ──────────────────────────────────────────────────────────────────────────
# Sample size guidance (Cochran 1977, statistical power)
# ──────────────────────────────────────────────────────────────────────────

SAMPLE_SIZE_GUIDANCE: dict[str, int | str] = {
    "description": "Minimum sample sizes per stratum (Cochran 1977, 95% CI, 5% margin of error)",
    "per_language": 30,           # Minimum records per ISO 639-1 language
    "per_entity_type": 50,        # Minimum records per entity type
    "per_qi_group": 30,           # Minimum records per quasi-identifier group
    "per_format": 25,             # Minimum records per data format type
    "per_adversarial_type": 20,   # Minimum records per adversarial attack type
    "global_minimum": 500,        # Absolute minimum total records for representative evaluation
}


# ──────────────────────────────────────────────────────────────────────────
# Coverage Report Dataclass
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CoverageReport:
    """Comprehensive coverage validation report for PII Anonymization Eval Dataset v1.0.0.

    This report documents coverage across all 7 evaluation dimensions and validates
    that the dataset meets statistical requirements for representative evaluation.
    """

    timestamp: str
    dataset_name: str
    total_records: int

    dimension_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-dimension coverage: {dimension: {count, target, pct, adequate}}"""

    language_coverage: dict[str, int] = field(default_factory=dict)
    """Records per ISO 639-1 language code."""

    entity_type_coverage: dict[str, int] = field(default_factory=dict)
    """Records per entity type."""

    quasi_identifier_coverage: dict[str, int] = field(default_factory=dict)
    """Records per quasi-identifier group (e.g., "name+address", "email+phone")."""

    adversarial_coverage: dict[str, int] = field(default_factory=dict)
    """Records per adversarial attack type (or "clean" for non-adversarial)."""

    format_coverage: dict[str, int] = field(default_factory=dict)
    """Records per data format type."""

    coverage_gaps: list[dict[str, Any]] = field(default_factory=list)
    """List of uncovered or undercovered strata. Each dict: {stratum, count, target, gap}"""

    all_requirements_met: bool = False
    """True if all dimensions meet their target coverage."""

    def summary(self) -> str:
        """Generate a human-readable summary of coverage validation results.

        Returns
        -------
        str
            Multi-line summary report with dimension-by-dimension breakdown.
        """
        lines = [
            "=" * 80,
            "PII ANONYMIZATION EVAL DATASET v1.0.0 - COVERAGE VALIDATION REPORT",
            "=" * 80,
            f"Dataset: {self.dataset_name}",
            f"Timestamp: {self.timestamp}",
            f"Total Records: {self.total_records}",
            "",
            "DIMENSION COVERAGE (with weights):",
            "-" * 80,
        ]

        for dim in sorted(self.dimension_coverage.keys()):
            cov = self.dimension_coverage[dim]
            weight = DIMENSION_WEIGHTS.get(dim, 0.0)
            count = cov.get("count", 0)
            target = cov.get("target", 0)
            pct = cov.get("pct", 0.0)
            adequate = cov.get("adequate", False)
            status = "PASS" if adequate else "FAIL"

            lines.append(
                f"  {dim:30s}  {weight*100:5.1f}%  "
                f"({count:3d}/{target:3d})  {pct:5.1f}%  [{status}]"
            )

        lines.extend([
            "",
            "LANGUAGE COVERAGE (ISO 639-1):",
            "-" * 80,
        ])

        if self.language_coverage:
            for lang in sorted(self.language_coverage.keys()):
                count = self.language_coverage[lang]
                min_target = int(SAMPLE_SIZE_GUIDANCE["per_language"])
                status = "OK" if count >= min_target else "LOW"
                lines.append(f"  {lang:10s}  {count:3d} records  [{status}]")
        else:
            lines.append("  (no language data)")

        lines.extend([
            "",
            "ENTITY TYPE COVERAGE:",
            "-" * 80,
        ])

        if self.entity_type_coverage:
            for etype in sorted(self.entity_type_coverage.keys()):
                count = self.entity_type_coverage[etype]
                min_target = int(SAMPLE_SIZE_GUIDANCE["per_entity_type"])
                status = "OK" if count >= min_target else "LOW"
                lines.append(f"  {etype:30s}  {count:3d} records  [{status}]")
        else:
            lines.append("  (no entity type data)")

        lines.extend([
            "",
            "QUASI-IDENTIFIER COVERAGE:",
            "-" * 80,
        ])

        if self.quasi_identifier_coverage:
            for qi_group in sorted(self.quasi_identifier_coverage.keys()):
                count = self.quasi_identifier_coverage[qi_group]
                min_target = int(SAMPLE_SIZE_GUIDANCE["per_qi_group"])
                status = "OK" if count >= min_target else "LOW"
                lines.append(f"  {qi_group:40s}  {count:3d} records  [{status}]")
        else:
            lines.append("  (no quasi-identifier data)")

        lines.extend([
            "",
            "ADVERSARIAL COVERAGE:",
            "-" * 80,
        ])

        if self.adversarial_coverage:
            for adv_type in sorted(self.adversarial_coverage.keys()):
                count = self.adversarial_coverage[adv_type]
                min_target_raw = SAMPLE_SIZE_GUIDANCE.get(
                    "per_adversarial_type", 20
                ) if adv_type != "clean" else 0
                min_target = int(min_target_raw) if isinstance(min_target_raw, (int, str)) else 20
                status = "OK" if adv_type == "clean" or count >= min_target else "LOW"
                lines.append(f"  {adv_type:30s}  {count:3d} records  [{status}]")
        else:
            lines.append("  (no adversarial data)")

        lines.extend([
            "",
            "DATA FORMAT COVERAGE:",
            "-" * 80,
        ])

        if self.format_coverage:
            for fmt in sorted(self.format_coverage.keys()):
                count = self.format_coverage[fmt]
                min_target = int(SAMPLE_SIZE_GUIDANCE["per_format"])
                status = "OK" if count >= min_target else "LOW"
                lines.append(f"  {fmt:30s}  {count:3d} records  [{status}]")
        else:
            lines.append("  (no format data)")

        lines.extend([
            "",
            "COVERAGE GAPS (undercovered strata):",
            "-" * 80,
        ])

        if self.coverage_gaps:
            for gap in self.coverage_gaps:
                stratum = gap.get("stratum", "unknown")
                count = gap.get("count", 0)
                target = gap.get("target", 0)
                shortage = gap.get("gap", target - count)
                lines.append(f"  {stratum:40s}  +{shortage:3d} records needed")
        else:
            lines.append("  (no gaps detected)")

        lines.extend([
            "",
            "FINAL ASSESSMENT:",
            "-" * 80,
            f"All Requirements Met: {'YES' if self.all_requirements_met else 'NO'}",
            "=" * 80,
        ])

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Coverage Validator Class
# ──────────────────────────────────────────────────────────────────────────

class CoverageValidator:
    """Validates dataset coverage across 7 evaluation dimensions and statistical requirements.

    This validator checks:
      - Per-language sample sizes (ISO 639-1 codes)
      - Per-entity-type distribution
      - Per-quasi-identifier group coverage
      - Per-adversarial-attack type coverage
      - Per-data-format coverage
      - All 7 dimension tags are represented
      - Coverage adequacy against statistical targets (Cochran 1977)

    Parameters
    ----------
    min_per_language : int
        Minimum records required per language. Default: 30
    min_per_entity_type : int
        Minimum records required per entity type. Default: 50
    min_per_qi_group : int
        Minimum records required per quasi-identifier group. Default: 30
    """

    def __init__(
        self,
        min_per_language: int = 30,
        min_per_entity_type: int = 50,
        min_per_qi_group: int = 30,
    ) -> None:
        self.min_per_language = min_per_language
        self.min_per_entity_type = min_per_entity_type
        self.min_per_qi_group = min_per_qi_group

    def validate(self, records: list[EvalBenchmarkRecord]) -> CoverageReport:
        """Validate dataset coverage and generate comprehensive report.

        Parameters
        ----------
        records : list[EvalBenchmarkRecord]
            Evaluation benchmark records to validate.

        Returns
        -------
        CoverageReport
            Detailed coverage validation report.
        """
        if not records:
            return CoverageReport(
                timestamp=datetime.utcnow().isoformat() + "Z",
                dataset_name="(empty dataset)",
                total_records=0,
                all_requirements_met=False,
            )

        # Check all 7 dimensions
        dimension_coverage = self._check_dimensions(records)

        # Check language distribution
        language_coverage = self._check_languages(records)

        # Check entity types
        entity_type_coverage = self._check_entity_types(records)

        # Check quasi-identifier groups
        qi_coverage = self._check_quasi_identifiers(records)

        # Check adversarial coverage
        adversarial_coverage = self._check_adversarial(records)

        # Check data formats
        format_coverage = self._check_formats(records)

        # Identify coverage gaps
        coverage_gaps = self._identify_gaps(
            dimension_coverage,
            language_coverage,
            entity_type_coverage,
            qi_coverage,
            adversarial_coverage,
            format_coverage,
        )

        # Determine if all requirements are met
        all_requirements_met = all(
            cov.get("adequate", False) for cov in dimension_coverage.values()
        )

        return CoverageReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            dataset_name=records[0].source_id if records else "unknown",
            total_records=len(records),
            dimension_coverage=dimension_coverage,
            language_coverage=language_coverage,
            entity_type_coverage=entity_type_coverage,
            quasi_identifier_coverage=qi_coverage,
            adversarial_coverage=adversarial_coverage,
            format_coverage=format_coverage,
            coverage_gaps=coverage_gaps,
            all_requirements_met=all_requirements_met,
        )

    def _check_dimensions(self, records: list[EvalBenchmarkRecord]) -> dict[str, dict[str, Any]]:
        """Check coverage of the 7 evaluation dimensions.

        Returns dict mapping dimension name to {count, target, pct, adequate}.
        """
        dimension_counts: dict[str, int] = {dim: 0 for dim in VALID_DIMENSION_TAGS}
        total = len(records)

        for record in records:
            for dim in record.dimension_tags:
                if dim in dimension_counts:
                    dimension_counts[dim] += 1

        # Dimension targets: proportional to DIMENSION_WEIGHTS, minimum 50 records
        result: dict[str, dict[str, Any]] = {}
        for dim in sorted(VALID_DIMENSION_TAGS):
            weight = DIMENSION_WEIGHTS.get(dim, 0.05)
            target = max(int(total * weight), 50)
            count = dimension_counts[dim]
            pct = (count / target * 100.0) if target > 0 else 0.0
            adequate = count >= target

            result[dim] = {
                "count": count,
                "target": target,
                "pct": pct,
                "adequate": adequate,
            }

        return result

    def _check_languages(self, records: list[EvalBenchmarkRecord]) -> dict[str, int]:
        """Check language distribution. Target: min_per_language per language."""
        language_counts: dict[str, int] = {}
        for record in records:
            lang = record.language or "unknown"
            language_counts[lang] = language_counts.get(lang, 0) + 1
        return dict(sorted(language_counts.items()))

    def _check_entity_types(self, records: list[EvalBenchmarkRecord]) -> dict[str, int]:
        """Check entity type distribution. Target: min_per_entity_type per type."""
        entity_type_counts: dict[str, int] = {}
        for record in records:
            for etype in record.entity_types_present:
                entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1
        return dict(sorted(entity_type_counts.items()))

    def _check_quasi_identifiers(self, records: list[EvalBenchmarkRecord]) -> dict[str, int]:
        """Check quasi-identifier group coverage. Target: min_per_qi_group per group.

        Quasi-identifiers are combined attributes like (name, address, dob) that
        enable re-identification under k-anonymity (Sweeney 2002).
        """
        qi_counts: dict[str, int] = {}
        for record in records:
            # Serialize QI list as canonical string for grouping
            qi_list = sorted(record.quasi_identifiers_present)
            if qi_list:
                qi_group = "+".join(qi_list)
            else:
                # Infer from entity types present
                qi_group = "+".join(sorted(record.entity_types_present)) or "none"

            qi_counts[qi_group] = qi_counts.get(qi_group, 0) + 1

        return dict(sorted(qi_counts.items()))

    def _check_adversarial(self, records: list[EvalBenchmarkRecord]) -> dict[str, int]:
        """Check adversarial attack type coverage.

        Adversarial examples test robustness against attacks like:
        - typos/misspellings
        - case variations
        - unicode normalization attacks
        - semantic perturbations
        """
        adversarial_counts: dict[str, int] = {}
        for record in records:
            adv_type = record.adversarial_type or "clean"
            adversarial_counts[adv_type] = adversarial_counts.get(adv_type, 0) + 1

        return dict(sorted(adversarial_counts.items()))

    def _check_formats(self, records: list[EvalBenchmarkRecord]) -> dict[str, int]:
        """Check data format type coverage.

        Formats tested:
        - unstructured_text: free-form paragraphs
        - structured: tables, databases, fixed schemas
        - semi_structured: JSON, XML, CSV variants
        - code: source code, scripts
        - logs: event logs, access logs
        - mixed: documents with multiple embedded formats
        """
        format_counts: dict[str, int] = {}
        for record in records:
            fmt = record.data_type or "unknown"
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        return dict(sorted(format_counts.items()))

    def _identify_gaps(
        self,
        dimension_coverage: dict[str, dict[str, Any]],
        language_coverage: dict[str, int],
        entity_type_coverage: dict[str, int],
        qi_coverage: dict[str, int],
        adversarial_coverage: dict[str, int],
        format_coverage: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Identify undercovered strata across all dimensions.

        Returns list of {stratum, count, target, gap} dicts.
        """
        gaps: list[dict[str, Any]] = []

        # Dimension gaps
        for dim, cov in dimension_coverage.items():
            if not cov.get("adequate", False):
                count = cov.get("count", 0)
                target = cov.get("target", 0)
                gaps.append({
                    "stratum": f"dimension:{dim}",
                    "count": count,
                    "target": target,
                    "gap": max(0, target - count),
                })

        # Language gaps
        for lang, count in language_coverage.items():
            if count < self.min_per_language:
                gaps.append({
                    "stratum": f"language:{lang}",
                    "count": count,
                    "target": self.min_per_language,
                    "gap": max(0, self.min_per_language - count),
                })

        # Entity type gaps
        for etype, count in entity_type_coverage.items():
            if count < self.min_per_entity_type:
                gaps.append({
                    "stratum": f"entity_type:{etype}",
                    "count": count,
                    "target": self.min_per_entity_type,
                    "gap": max(0, self.min_per_entity_type - count),
                })

        # Quasi-identifier gaps
        for qi_group, count in qi_coverage.items():
            if count < self.min_per_qi_group:
                gaps.append({
                    "stratum": f"qi_group:{qi_group}",
                    "count": count,
                    "target": self.min_per_qi_group,
                    "gap": max(0, self.min_per_qi_group - count),
                })

        # Sort by gap size (largest first)
        gaps.sort(key=lambda x: x["gap"], reverse=True)
        return gaps


# ──────────────────────────────────────────────────────────────────────────
# Convenience Function
# ──────────────────────────────────────────────────────────────────────────

def validate_and_report(
    dataset_name: str = "pii_anon_eval_v1",
) -> CoverageReport:
    """Load a dataset and validate its coverage.

    Convenience function that loads the dataset by name and runs coverage
    validation, returning a comprehensive report.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (without .jsonl extension).
        Defaults to "pii_anon_eval_v1" (v1.0.0 unified dataset).

    Returns
    -------
    CoverageReport
        Detailed coverage validation report with all dimension metrics.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.

    Examples
    --------
    >>> report = validate_and_report("pii_anon_eval_v1")
    >>> print(report.summary())
    >>> print(f"All requirements met: {report.all_requirements_met}")
    """
    records = load_eval_dataset(dataset_name)
    validator = CoverageValidator()
    return validator.validate(records)
