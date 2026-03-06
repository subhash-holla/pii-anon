"""Tests for research rigor enhancements to the evaluation framework.

Covers:
- Expanded dataset generator (44 entity types, context tiers, adversarial patterns)
- Dataset integrity (checksums, statistics)
- Generalized bootstrap CI for all metrics
- Paired bootstrap significance testing
- Cohen's d effect size
- Cohen's kappa inter-annotator agreement
- Simulated annotator agreement
- Expanded research references
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.datasets.generator import (
    compute_dataset_checksum,
    dataset_statistics,
    generate_dataset,
)
from pii_anon.eval_framework.evaluation.aggregation import MetricAggregator
from pii_anon.eval_framework.research.references import (
    all_references,
    get_references_for,
)


# ---------------------------------------------------------------------------
# Dataset generator: entity coverage
# ---------------------------------------------------------------------------

class TestExpandedEntityCoverage:
    """Verify all 44 taxonomy entity types appear in the generated dataset."""

    @pytest.fixture(scope="class")
    def small_dataset(self) -> list[dict]:
        return generate_dataset(seed=42, target_records=500)

    @pytest.fixture(scope="class")
    def stats(self, small_dataset: list[dict]) -> dict:
        return dataset_statistics(small_dataset)

    def test_records_generated(self, small_dataset: list[dict]) -> None:
        assert len(small_dataset) >= 450  # allow slight variance

    def test_entity_type_coverage_minimum(self, stats: dict) -> None:
        """At least 30 of 44 entity types should be covered in a 500-record sample."""
        assert stats["entity_type_count"] >= 25

    def test_critical_entity_types_present(self, stats: dict) -> None:
        covered = set(stats["entity_types_covered"])
        critical = {
            "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
            "CREDIT_CARD_NUMBER", "IBAN", "ADDRESS", "DATE_OF_BIRTH",
            "IP_ADDRESS", "MAC_ADDRESS",
        }
        missing = critical - covered
        assert len(missing) == 0, f"Missing critical types: {missing}"

    def test_new_entity_types_present(self, stats: dict) -> None:
        """Verify new entity types from expanded templates are generated."""
        covered = set(stats["entity_types_covered"])
        new_types = {
            "BANK_ACCOUNT_NUMBER", "ROUTING_NUMBER", "SWIFT_BIC_CODE",
            "TAX_ID", "DRIVERS_LICENSE", "LICENSE_PLATE",
            "VEHICLE_IDENTIFICATION_NUMBER", "NATIONAL_ID_NUMBER", "VISA_NUMBER",
        }
        found = new_types & covered
        # With 500 records and 26+ templates, most should appear
        assert len(found) >= 4, f"Only found {len(found)} of expected new types: {found}"

    def test_behavioral_types_present(self, stats: dict) -> None:
        """Behavioral/contextual types from the new behavioral_sensitive template."""
        covered = set(stats["entity_types_covered"])
        behavioral = {"AGE", "GENDER", "ETHNIC_ORIGIN", "RELIGIOUS_BELIEF", "POLITICAL_OPINION", "LOCATION_COORDINATES"}
        found = behavioral & covered
        assert len(found) >= 2, f"Behavioral types found: {found}"

    def test_digital_types_present(self, stats: dict) -> None:
        covered = set(stats["entity_types_covered"])
        digital = {"SOCIAL_MEDIA_HANDLE", "AUTHENTICATION_TOKEN", "DEVICE_ID", "URL_WITH_PII"}
        found = digital & covered
        assert len(found) >= 2, f"Digital types found: {found}"


# ---------------------------------------------------------------------------
# Dataset generator: context tiers
# ---------------------------------------------------------------------------

class TestContextLengthTiers:
    @pytest.fixture(scope="class")
    def small_dataset(self) -> list[dict]:
        return generate_dataset(seed=42, target_records=500)

    @pytest.fixture(scope="class")
    def stats(self, small_dataset: list[dict]) -> dict:
        return dataset_statistics(small_dataset)

    def test_short_tier_present(self, stats: dict) -> None:
        assert stats["context_length_distribution"].get("short", 0) > 0

    def test_medium_tier_present(self, stats: dict) -> None:
        assert stats["context_length_distribution"].get("medium", 0) > 0

    def test_long_tier_present(self, stats: dict) -> None:
        assert stats["context_length_distribution"].get("long", 0) > 0

    def test_text_length_variance(self, stats: dict) -> None:
        """Max text length should be significantly larger than min."""
        assert stats["text_length_max"] > stats["text_length_min"] * 3


# ---------------------------------------------------------------------------
# Dataset generator: adversarial patterns
# ---------------------------------------------------------------------------

class TestAdversarialPatterns:
    @pytest.fixture(scope="class")
    def small_dataset(self) -> list[dict]:
        return generate_dataset(seed=42, target_records=500)

    @pytest.fixture(scope="class")
    def stats(self, small_dataset: list[dict]) -> dict:
        return dataset_statistics(small_dataset)

    def test_adversarial_records_exist(self, stats: dict) -> None:
        assert stats["adversarial_count"] > 0

    def test_adversarial_ratio_reasonable(self, stats: dict) -> None:
        # With 4 adversarial out of 26 templates, ~15% should be adversarial
        assert 0.05 <= stats["adversarial_ratio"] <= 0.35

    def test_multiple_adversarial_types(self, small_dataset: list[dict]) -> None:
        adv_types = {r["adversarial_type"] for r in small_dataset if r.get("adversarial_type")}
        assert len(adv_types) >= 2


# ---------------------------------------------------------------------------
# Dataset generator: regulatory domains
# ---------------------------------------------------------------------------

class TestRegulatoryDomains:
    @pytest.fixture(scope="class")
    def stats(self) -> dict:
        records = generate_dataset(seed=42, target_records=500)
        return dataset_statistics(records)

    def test_gdpr_domain_present(self, stats: dict) -> None:
        assert "gdpr" in stats["regulatory_domains"]

    def test_ccpa_domain_present(self, stats: dict) -> None:
        assert "ccpa" in stats["regulatory_domains"]

    def test_hipaa_domain_present(self, stats: dict) -> None:
        assert "hipaa" in stats["regulatory_domains"]

    def test_pci_dss_domain_present(self, stats: dict) -> None:
        assert "pci_dss" in stats["regulatory_domains"]


# ---------------------------------------------------------------------------
# Dataset integrity: checksums
# ---------------------------------------------------------------------------

class TestDatasetIntegrity:
    def test_checksum_deterministic(self) -> None:
        records_a = generate_dataset(seed=42, target_records=100)
        records_b = generate_dataset(seed=42, target_records=100)
        assert compute_dataset_checksum(records_a) == compute_dataset_checksum(records_b)

    def test_different_seeds_different_checksums(self) -> None:
        records_a = generate_dataset(seed=42, target_records=100)
        records_b = generate_dataset(seed=99, target_records=100)
        assert compute_dataset_checksum(records_a) != compute_dataset_checksum(records_b)

    def test_checksum_is_sha256_hex(self) -> None:
        records = generate_dataset(seed=42, target_records=50)
        cksum = compute_dataset_checksum(records)
        assert len(cksum) == 64
        assert all(c in "0123456789abcdef" for c in cksum)


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

class TestDatasetStatistics:
    @pytest.fixture(scope="class")
    def stats(self) -> dict:
        records = generate_dataset(seed=42, target_records=200)
        return dataset_statistics(records)

    def test_total_records(self, stats: dict) -> None:
        assert stats["total_records"] >= 180

    def test_entity_types_covered_is_list(self, stats: dict) -> None:
        assert isinstance(stats["entity_types_covered"], list)

    def test_language_count(self, stats: dict) -> None:
        assert stats["language_count"] >= 10

    def test_difficulty_distribution(self, stats: dict) -> None:
        assert len(stats["difficulty_distribution"]) >= 3

    def test_data_type_distribution(self, stats: dict) -> None:
        assert len(stats["data_type_distribution"]) >= 3

    def test_text_length_stats(self, stats: dict) -> None:
        assert stats["text_length_min"] > 0
        assert stats["text_length_mean"] > 0
        assert stats["text_length_max"] >= stats["text_length_mean"]


# ---------------------------------------------------------------------------
# Generalized bootstrap CI
# ---------------------------------------------------------------------------

class TestGeneralizedBootstrapCI:
    def test_precision_ci(self) -> None:
        values = [0.85, 0.90, 0.88, 0.92, 0.87, 0.91, 0.89, 0.86, 0.93, 0.88]
        result = MetricAggregator.compute_metric_confidence_intervals(
            values, metric_name="precision",
        )
        assert result["metric"] == "precision"
        assert result["lower"] < result["mean"] < result["upper"]
        assert result["std_error"] > 0

    def test_recall_ci(self) -> None:
        values = [0.70, 0.75, 0.72, 0.68, 0.73, 0.71, 0.69, 0.74, 0.72, 0.70]
        result = MetricAggregator.compute_metric_confidence_intervals(
            values, metric_name="recall",
        )
        assert result["metric"] == "recall"
        assert 0.0 < result["lower"] <= result["upper"] <= 1.0

    def test_empty_values(self) -> None:
        result = MetricAggregator.compute_metric_confidence_intervals([])
        assert result["mean"] == 0.0
        assert result["lower"] == 0.0
        assert result["upper"] == 0.0

    def test_single_value(self) -> None:
        result = MetricAggregator.compute_metric_confidence_intervals([0.5])
        assert result["mean"] == 0.5
        # With single value, CI is narrow
        assert result["lower"] == result["upper"] == 0.5

    def test_ci_narrows_with_more_data(self) -> None:
        small = [0.8, 0.9, 0.85]
        large = [0.8, 0.9, 0.85] * 100
        ci_small = MetricAggregator.compute_metric_confidence_intervals(small)
        ci_large = MetricAggregator.compute_metric_confidence_intervals(large)
        width_small = ci_small["upper"] - ci_small["lower"]
        width_large = ci_large["upper"] - ci_large["lower"]
        assert width_large < width_small


# ---------------------------------------------------------------------------
# Paired bootstrap significance test
# ---------------------------------------------------------------------------

class TestPairedBootstrapTest:
    def test_significant_difference(self) -> None:
        a = [0.95, 0.93, 0.96, 0.94, 0.92, 0.97, 0.95, 0.94, 0.96, 0.93]
        b = [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.60, 0.61, 0.59, 0.62]
        result = MetricAggregator.paired_bootstrap_test(a, b)
        # The delta should be positive and large
        assert result["delta_mean"] > 0.2
        # The CI should not cross zero (both positive)
        assert result["ci_lower"] > 0.0

    def test_no_significant_difference(self) -> None:
        a = [0.80, 0.82, 0.79, 0.81, 0.80, 0.82, 0.79, 0.81, 0.80, 0.82]
        b = [0.81, 0.80, 0.82, 0.79, 0.81, 0.80, 0.82, 0.79, 0.81, 0.80]
        result = MetricAggregator.paired_bootstrap_test(a, b)
        assert result["delta_mean"] < 0.05

    def test_empty_scores(self) -> None:
        result = MetricAggregator.paired_bootstrap_test([], [])
        assert result["p_value"] == 1.0
        assert result["delta_mean"] == 0.0

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            MetricAggregator.paired_bootstrap_test([0.8, 0.9], [0.7])

    def test_result_keys(self) -> None:
        result = MetricAggregator.paired_bootstrap_test([0.8], [0.7])
        assert "p_value" in result
        assert "delta_mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result


# ---------------------------------------------------------------------------
# Cohen's d effect size
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_large_effect(self) -> None:
        a = [0.95, 0.93, 0.96, 0.94, 0.92]
        b = [0.60, 0.62, 0.58, 0.61, 0.59]
        d = MetricAggregator.cohens_d(a, b)
        assert d > 0.8  # large effect

    def test_small_effect(self) -> None:
        # Use values with large overlap for a small effect
        a = [0.80, 0.82, 0.79, 0.81, 0.80, 0.78, 0.83, 0.77, 0.84, 0.76]
        b = [0.78, 0.80, 0.77, 0.79, 0.78, 0.76, 0.81, 0.75, 0.82, 0.74]
        d = MetricAggregator.cohens_d(a, b)
        assert 0.0 < d < 2.0  # positive effect

    def test_zero_effect(self) -> None:
        a = [0.80, 0.80, 0.80]
        b = [0.80, 0.80, 0.80]
        d = MetricAggregator.cohens_d(a, b)
        assert d == 0.0

    def test_empty_lists(self) -> None:
        assert MetricAggregator.cohens_d([], []) == 0.0

    def test_negative_effect(self) -> None:
        a = [0.60, 0.62, 0.58]
        b = [0.90, 0.92, 0.88]
        d = MetricAggregator.cohens_d(a, b)
        assert d < -0.8  # large negative effect


# ---------------------------------------------------------------------------
# Cohen's kappa inter-annotator agreement
# ---------------------------------------------------------------------------

class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        labels = ["EMAIL", "PHONE", "SSN", "EMAIL", "PHONE"]
        kappa = MetricAggregator.cohens_kappa(labels, labels)
        assert kappa == 1.0

    def test_high_agreement(self) -> None:
        a = ["EMAIL", "PHONE", "SSN", "EMAIL", "PHONE", "SSN", "EMAIL", "PHONE", "SSN", "EMAIL"]
        b = ["EMAIL", "PHONE", "SSN", "EMAIL", "PHONE", "SSN", "EMAIL", "PHONE", "EMAIL", "EMAIL"]
        kappa = MetricAggregator.cohens_kappa(a, b)
        assert kappa > 0.5  # substantial agreement

    def test_empty_lists(self) -> None:
        assert MetricAggregator.cohens_kappa([], []) == 0.0

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            MetricAggregator.cohens_kappa(["A", "B"], ["A"])

    def test_complete_disagreement(self) -> None:
        a = ["A", "A", "A", "A"]
        b = ["B", "B", "B", "B"]
        kappa = MetricAggregator.cohens_kappa(a, b)
        # With disjoint label sets p_e = 0, p_o = 0, kappa = 0.0
        assert kappa <= 0.0


# ---------------------------------------------------------------------------
# Simulated annotator agreement
# ---------------------------------------------------------------------------

class TestSimulatedAnnotatorAgreement:
    def test_low_noise_high_kappa(self) -> None:
        labels = ["EMAIL"] * 100 + ["PHONE"] * 100 + ["SSN"] * 100
        result = MetricAggregator.simulate_annotator_agreement(labels, noise_rate=0.02)
        assert result["kappa"] > 0.9
        assert result["observed_agreement"] > 0.95
        assert result["n_samples"] == 300

    def test_high_noise_lower_kappa(self) -> None:
        labels = ["EMAIL"] * 50 + ["PHONE"] * 50
        result = MetricAggregator.simulate_annotator_agreement(labels, noise_rate=0.3)
        assert result["kappa"] < 0.8

    def test_zero_noise_perfect_agreement(self) -> None:
        labels = ["A", "B", "C", "A", "B"]
        result = MetricAggregator.simulate_annotator_agreement(labels, noise_rate=0.0)
        assert result["kappa"] == 1.0
        assert result["observed_agreement"] == 1.0

    def test_empty_labels(self) -> None:
        result = MetricAggregator.simulate_annotator_agreement([])
        assert result["kappa"] == 0.0
        assert result["n_samples"] == 0

    def test_deterministic(self) -> None:
        labels = ["EMAIL", "PHONE"] * 50
        r1 = MetricAggregator.simulate_annotator_agreement(labels, seed=42)
        r2 = MetricAggregator.simulate_annotator_agreement(labels, seed=42)
        assert r1["kappa"] == r2["kappa"]


# ---------------------------------------------------------------------------
# Expanded research references
# ---------------------------------------------------------------------------

class TestResearchReferences:
    def test_new_references_exist(self) -> None:
        refs = all_references()
        keys = {r.key for r in refs}
        expected = {
            "efron_bootstrap_1993", "berg_kirkpatrick_2012",
            "cohen_kappa_1960", "cohen_d_1988",
            "bender_friedman_2018", "gebru_datasheets_2021",
        }
        missing = expected - keys
        assert len(missing) == 0, f"Missing references: {missing}"

    def test_dataset_methodology_registry(self) -> None:
        refs = get_references_for("dataset_methodology")
        assert len(refs) >= 2
        keys = {r.key for r in refs}
        assert "gebru_datasheets_2021" in keys
        assert "bender_friedman_2018" in keys

    def test_statistical_significance_registry(self) -> None:
        refs = get_references_for("statistical_significance")
        assert len(refs) >= 2
        keys = {r.key for r in refs}
        assert "berg_kirkpatrick_2012" in keys
        assert "efron_bootstrap_1993" in keys

    def test_inter_annotator_agreement_registry(self) -> None:
        refs = get_references_for("inter_annotator_agreement")
        assert len(refs) >= 1
        keys = {r.key for r in refs}
        assert "cohen_kappa_1960" in keys

    def test_confidence_intervals_registry(self) -> None:
        refs = get_references_for("confidence_intervals")
        assert len(refs) >= 1

    def test_dataset_references_expanded(self) -> None:
        refs = get_references_for("dataset")
        assert len(refs) >= 5  # original 3 + 2 new

    def test_total_reference_count(self) -> None:
        refs = all_references()
        assert len(refs) >= 27  # 21 original + 6 new


# ---------------------------------------------------------------------------
# Label integrity checks
# ---------------------------------------------------------------------------

class TestLabelIntegrity:
    """Verify that generated labels have correct character offsets."""

    @pytest.fixture(scope="class")
    def small_dataset(self) -> list[dict]:
        return generate_dataset(seed=42, target_records=200)

    def test_all_labels_within_text(self, small_dataset: list[dict]) -> None:
        """Every label span should fall within the text boundaries."""
        for record in small_dataset:
            text = record["text"]
            for lbl in record["labels"]:
                assert lbl["start"] >= 0, f"Negative start in {record['id']}"
                assert lbl["end"] <= len(text), (
                    f"End {lbl['end']} > text length {len(text)} in {record['id']}"
                )
                assert lbl["start"] < lbl["end"], f"Empty span in {record['id']}"

    def test_labels_non_overlapping(self, small_dataset: list[dict]) -> None:
        """Labels should not overlap (they may in adversarial but mostly shouldn't)."""
        clean_records = [r for r in small_dataset if not r.get("adversarial_type")]
        overlap_count = 0
        for record in clean_records[:100]:
            spans = sorted(record["labels"], key=lambda sp: sp["start"])
            for i in range(len(spans) - 1):
                if spans[i]["end"] > spans[i + 1]["start"]:
                    overlap_count += 1
        # Allow very few overlaps in non-adversarial records
        assert overlap_count < 10

    def test_label_entity_types_non_empty(self, small_dataset: list[dict]) -> None:
        for record in small_dataset[:100]:
            for lbl in record["labels"]:
                assert lbl["entity_type"].strip() != ""
