"""Statistical aggregation for evaluation metrics.

Provides micro/macro/weighted metric averaging, bootstrap confidence
intervals, paired significance testing, and inter-annotator agreement.

Evidence basis:
- seqeval (Nakayama, 2018): micro-averaged F1 for sequence labelling
- scikit-learn classification_report: micro/macro/weighted aggregation
- Efron & Tibshirani (1993): Bootstrap confidence intervals
- Berg-Kirkpatrick et al. (2012): Paired bootstrap significance testing for NLP
- Cohen (1960): Inter-annotator agreement via Cohen's kappa
- Cohen (1988): Effect size measures (Cohen's d)

Performance notes:
- ``cohens_kappa()`` uses ``collections.Counter`` for O(n) label frequency
  counting instead of the previous O(n·k) approach of iterating all labels
  and scanning the full list for each one.
- Bootstrap CI and paired bootstrap tests use **numpy** for vectorised
  resampling when available, falling back to pure-Python for environments
  without numpy.  At 50K records × 10K bootstrap iterations the numpy
  path is ~50-100× faster (seconds vs. minutes).
"""

from __future__ import annotations

import math
import random as random_module
from collections import Counter
from typing import Any

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from ..metrics.base import compute_f1, safe_div

# Maximum number of index rows to allocate at once in numpy bootstrap.
# Keeps peak memory below ~200 MB even at n=50K samples.
_NP_BOOTSTRAP_CHUNK = 500


def _np_bootstrap_means(
    arr: _np.ndarray[Any, _np.dtype[_np.float64]],
    n_bootstrap: int,
    rng: _np.random.Generator,
) -> _np.ndarray[Any, _np.dtype[_np.float64]]:
    """Vectorised bootstrap mean computation, chunked to bound memory.

    Instead of allocating a (n_bootstrap × n) index matrix all at once
    (which can exceed available RAM for large n), we process in chunks
    of ``_NP_BOOTSTRAP_CHUNK`` rows.
    """
    n = len(arr)
    result = _np.empty(n_bootstrap, dtype=_np.float64)
    offset = 0
    while offset < n_bootstrap:
        chunk = min(_NP_BOOTSTRAP_CHUNK, n_bootstrap - offset)
        indices = rng.integers(0, n, size=(chunk, n))
        result[offset : offset + chunk] = _np.mean(arr[indices], axis=1)
        offset += chunk
    return result


class MetricAggregator:
    """Aggregates per-entity-type metrics into micro/macro/weighted averages.

    Provides static methods for:
    - **Micro averaging**: TP/FP/FN summed globally, then precision/recall/F1.
      Each entity *instance* counts equally regardless of type.
    - **Macro averaging**: unweighted mean of per-type metrics.
      Each entity *type* counts equally regardless of frequency.
    - **Weighted averaging**: per-type metrics weighted by support (instance count).
      Balances type-level and instance-level perspectives.
    - **Bootstrap confidence intervals**: non-parametric CIs via resampling.
    - **Paired bootstrap test**: significance testing between two systems.
    - **Cohen's kappa**: inter-annotator agreement beyond chance.
    - **Cohen's d**: standardized effect size between two systems.
    """

    @staticmethod
    def compute_micro_averaged(
        per_entity_results: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Micro-averaged: sum all TP/FP/FN globally, then compute metrics.

        Each entity instance counts equally regardless of type.
        """
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        for _et, metrics in per_entity_results.items():
            support = metrics.get("support", 0)
            p = metrics.get("precision", 0.0)
            r = metrics.get("recall", 0.0)
            tp = r * support
            fn = support - tp
            fp = safe_div(tp, p) - tp if p > 0 else 0.0
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = safe_div(total_tp, total_tp + total_fp)
        recall = safe_div(total_tp, total_tp + total_fn)
        f1 = compute_f1(precision, recall)
        return {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    @staticmethod
    def compute_macro_averaged(
        per_entity_results: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Macro-averaged: unweighted average of per-class metrics.

        Each entity type counts equally regardless of frequency.
        """
        if not per_entity_results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        n = len(per_entity_results)
        total_p = sum(m.get("precision", 0.0) for m in per_entity_results.values())
        total_r = sum(m.get("recall", 0.0) for m in per_entity_results.values())
        total_f1 = sum(m.get("f1", 0.0) for m in per_entity_results.values())
        return {
            "precision": round(total_p / n, 6),
            "recall": round(total_r / n, 6),
            "f1": round(total_f1 / n, 6),
        }

    @staticmethod
    def compute_weighted_averaged(
        per_entity_results: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Weighted-averaged: weighted by support (instance count per type)."""
        total_support = sum(m.get("support", 0) for m in per_entity_results.values())
        if total_support == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        wp = sum(m.get("precision", 0.0) * m.get("support", 0) for m in per_entity_results.values())
        wr = sum(m.get("recall", 0.0) * m.get("support", 0) for m in per_entity_results.values())
        wf1 = sum(m.get("f1", 0.0) * m.get("support", 0) for m in per_entity_results.values())
        return {
            "precision": round(wp / total_support, 6),
            "recall": round(wr / total_support, 6),
            "f1": round(wf1 / total_support, 6),
        }

    @staticmethod
    def compute_confidence_intervals(
        per_record_f1s: list[float],
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Bootstrap confidence intervals for F1 scores.

        Returns (lower, upper) bounds at the given confidence level.

        Uses numpy vectorised resampling when available (~50-100× faster
        than pure-Python at scale).
        """
        if not per_record_f1s:
            return (0.0, 0.0)

        alpha = 1.0 - confidence_level

        if _HAS_NUMPY:
            arr = _np.asarray(per_record_f1s, dtype=_np.float64)
            rng = _np.random.default_rng(seed)
            bootstrap_means = _np_bootstrap_means(arr, n_bootstrap, rng)
            bootstrap_means.sort()
            lower_idx = max(0, int(math.floor(alpha / 2.0 * n_bootstrap)))
            upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)
            return (round(float(bootstrap_means[lower_idx]), 6), round(float(bootstrap_means[upper_idx]), 6))

        # Pure-Python fallback
        rng_py = random_module.Random(seed)
        n = len(per_record_f1s)
        bootstrap_means_list: list[float] = []
        for _ in range(n_bootstrap):
            sample = [rng_py.choice(per_record_f1s) for _ in range(n)]
            bootstrap_means_list.append(sum(sample) / n)
        bootstrap_means_list.sort()
        lower_idx = max(0, int(math.floor(alpha / 2.0 * n_bootstrap)))
        upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)
        return (round(bootstrap_means_list[lower_idx], 6), round(bootstrap_means_list[upper_idx], 6))

    @staticmethod
    def compute_metric_confidence_intervals(
        per_record_values: list[float],
        *,
        metric_name: str = "metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict[str, float | str]:
        """Generalized bootstrap confidence intervals for any metric.

        Returns a dict with metric_name, mean, lower, upper, and std_error.

        Evidence: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
        """
        if not per_record_values:
            return {
                "metric": metric_name,
                "mean": 0.0,
                "lower": 0.0,
                "upper": 0.0,
                "std_error": 0.0,
            }

        alpha = 1.0 - confidence_level

        if _HAS_NUMPY:
            arr = _np.asarray(per_record_values, dtype=_np.float64)
            observed_mean = float(arr.mean())
            rng = _np.random.default_rng(seed)
            bm = _np_bootstrap_means(arr, n_bootstrap, rng)
            bm.sort()
            lower_idx = max(0, int(math.floor(alpha / 2.0 * n_bootstrap)))
            upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)
            std_error = float(bm.std(ddof=1))
            return {
                "metric": metric_name,
                "mean": round(observed_mean, 6),
                "lower": round(float(bm[lower_idx]), 6),
                "upper": round(float(bm[upper_idx]), 6),
                "std_error": round(std_error, 6),
            }

        # Pure-Python fallback
        rng_py = random_module.Random(seed)
        n = len(per_record_values)
        observed_mean = sum(per_record_values) / n
        bootstrap_means: list[float] = []

        for _ in range(n_bootstrap):
            sample = [rng_py.choice(per_record_values) for _ in range(n)]
            bootstrap_means.append(sum(sample) / n)

        bootstrap_means.sort()
        lower_idx = max(0, int(math.floor(alpha / 2.0 * n_bootstrap)))
        upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)

        # Standard error of bootstrap distribution
        bm_list = bootstrap_means
        bm_mean = sum(bm_list) / len(bm_list)
        variance = sum((x - bm_mean) ** 2 for x in bm_list) / max(len(bm_list) - 1, 1)
        std_error = math.sqrt(variance)

        return {
            "metric": metric_name,
            "mean": round(observed_mean, 6),
            "lower": round(bm_list[lower_idx], 6),
            "upper": round(bm_list[upper_idx], 6),
            "std_error": round(std_error, 6),
        }

    @staticmethod
    def paired_bootstrap_test(
        system_a_scores: list[float],
        system_b_scores: list[float],
        *,
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Paired bootstrap significance test between two systems.

        Tests whether system A is significantly better than system B.

        Evidence: Berg-Kirkpatrick et al. (2012) "An Empirical Investigation of
        Statistical Significance in NLP"

        Uses numpy vectorised resampling when available (~50-100× faster
        than pure-Python at scale).

        Returns:
            p_value: probability that the observed difference is due to chance.
            delta_mean: observed mean difference (A - B).
            ci_lower, ci_upper: 95% CI for the difference.
        """
        if len(system_a_scores) != len(system_b_scores):
            raise ValueError("Score lists must have equal length for paired test.")
        n = len(system_a_scores)
        if n == 0:
            return {"p_value": 1.0, "delta_mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        if _HAS_NUMPY:
            arr_a = _np.asarray(system_a_scores, dtype=_np.float64)
            arr_b = _np.asarray(system_b_scores, dtype=_np.float64)
            deltas = arr_a - arr_b
            observed_delta = float(deltas.mean())

            rng = _np.random.default_rng(seed)
            bootstrap_deltas = _np_bootstrap_means(deltas, n_bootstrap, rng)

            # Two-sided p-value: fraction of bootstrap deltas with
            # magnitude >= observed magnitude.
            count_ge = int(_np.sum(_np.abs(bootstrap_deltas) >= abs(observed_delta)))
            p_value = count_ge / n_bootstrap

            bootstrap_deltas.sort()
            ci_lower = float(bootstrap_deltas[max(0, int(0.025 * n_bootstrap))])
            ci_upper = float(bootstrap_deltas[min(n_bootstrap - 1, int(0.975 * n_bootstrap) - 1)])

            return {
                "p_value": round(p_value, 6),
                "delta_mean": round(observed_delta, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
            }

        # Pure-Python fallback
        deltas_list = [a - b for a, b in zip(system_a_scores, system_b_scores)]
        observed_delta = sum(deltas_list) / n

        rng_py = random_module.Random(seed)
        count_ge = 0
        bootstrap_deltas_list: list[float] = []

        for _ in range(n_bootstrap):
            sample_idx = [rng_py.randint(0, n - 1) for _ in range(n)]
            sample_delta = sum(deltas_list[i] for i in sample_idx) / n
            bootstrap_deltas_list.append(sample_delta)
            # Two-sided: count how often bootstrap delta magnitude >= observed
            if abs(sample_delta) >= abs(observed_delta):
                count_ge += 1

        bootstrap_deltas_list.sort()
        p_value = count_ge / n_bootstrap
        ci_lower = bootstrap_deltas_list[max(0, int(0.025 * n_bootstrap))]
        ci_upper = bootstrap_deltas_list[min(n_bootstrap - 1, int(0.975 * n_bootstrap) - 1)]

        return {
            "p_value": round(p_value, 6),
            "delta_mean": round(observed_delta, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
        }

    @staticmethod
    def cohens_d(
        system_a_scores: list[float],
        system_b_scores: list[float],
    ) -> float:
        """Cohen's d effect size between two systems.

        Evidence: Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"

        Returns: effect size magnitude. Convention: 0.2 = small, 0.5 = medium, 0.8 = large.
        """
        if not system_a_scores or not system_b_scores:
            return 0.0

        if _HAS_NUMPY:
            arr_a = _np.asarray(system_a_scores, dtype=_np.float64)
            arr_b = _np.asarray(system_b_scores, dtype=_np.float64)
            n_a, n_b = len(arr_a), len(arr_b)
            var_a = float(arr_a.var(ddof=1)) if n_a > 1 else 0.0
            var_b = float(arr_b.var(ddof=1)) if n_b > 1 else 0.0
            pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
            pooled_sd = math.sqrt(pooled_var)
            if pooled_sd == 0.0:
                return 0.0
            return round(float(arr_a.mean() - arr_b.mean()) / pooled_sd, 6)

        n_a = len(system_a_scores)
        n_b = len(system_b_scores)
        mean_a = sum(system_a_scores) / n_a
        mean_b = sum(system_b_scores) / n_b

        var_a = sum((x - mean_a) ** 2 for x in system_a_scores) / max(n_a - 1, 1)
        var_b = sum((x - mean_b) ** 2 for x in system_b_scores) / max(n_b - 1, 1)

        # Pooled standard deviation
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
        pooled_sd = math.sqrt(pooled_var)

        if pooled_sd == 0.0:
            return 0.0
        return round((mean_a - mean_b) / pooled_sd, 6)

    @staticmethod
    def cohens_kappa(
        rater_a: list[str],
        rater_b: list[str],
    ) -> float:
        """Cohen's kappa inter-annotator agreement.

        Measures agreement between two raters beyond chance.

        Evidence: Cohen (1960) "A coefficient of agreement for nominal scales"

        Returns: kappa coefficient. <0 = worse than chance, 0 = chance,
        0.01-0.20 = slight, 0.21-0.40 = fair, 0.41-0.60 = moderate,
        0.61-0.80 = substantial, 0.81-1.00 = almost perfect.
        """
        if len(rater_a) != len(rater_b):
            raise ValueError("Rater lists must have equal length.")
        n = len(rater_a)
        if n == 0:
            return 0.0

        # Observed agreement: single-pass count of matching labels
        agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
        p_o = agree / n

        # Expected agreement by chance.
        # Uses Counter for O(n) frequency counting instead of O(n·k) where
        # k = number of distinct labels.  For large annotation sets with
        # many labels this is a significant improvement.
        counts_a = Counter(rater_a)
        counts_b = Counter(rater_b)
        all_labels = counts_a.keys() | counts_b.keys()
        p_e = 0.0
        for label in all_labels:
            p_a = counts_a.get(label, 0) / n
            p_b = counts_b.get(label, 0) / n
            p_e += p_a * p_b

        if p_e >= 1.0:
            return 1.0 if p_o >= 1.0 else 0.0
        return round((p_o - p_e) / (1.0 - p_e), 6)

    @staticmethod
    def simulate_annotator_agreement(
        labels: list[str],
        *,
        noise_rate: float = 0.05,
        seed: int = 42,
    ) -> dict[str, float]:
        """Simulate inter-annotator agreement by introducing controlled noise.

        Creates a simulated second annotator with a configurable error rate,
        then computes Cohen's kappa. This provides a lower bound on the
        inter-annotator agreement that the dataset would achieve.

        Parameters
        ----------
        labels:
            Ground-truth entity type labels.
        noise_rate:
            Probability that the simulated annotator disagrees (0.0-1.0).
        seed:
            Random seed for reproducibility.

        Returns:
            Dict with kappa, observed_agreement, noise_rate, and n_samples.
        """
        if not labels:
            return {"kappa": 0.0, "observed_agreement": 0.0, "noise_rate": noise_rate, "n_samples": 0}

        rng = random_module.Random(seed)
        unique_labels = sorted(set(labels))
        n = len(labels)

        # Create noisy annotations
        rater_b: list[str] = []
        for lbl in labels:
            if rng.random() < noise_rate:
                # Replace with random other label
                alternatives = [lab for lab in unique_labels if lab != lbl]
                rater_b.append(rng.choice(alternatives) if alternatives else lbl)
            else:
                rater_b.append(lbl)

        agree = sum(1 for a, b in zip(labels, rater_b) if a == b)
        kappa = MetricAggregator.cohens_kappa(labels, rater_b)

        return {
            "kappa": kappa,
            "observed_agreement": round(agree / n, 6),
            "noise_rate": noise_rate,
            "n_samples": n,
        }

    @staticmethod
    def per_entity_type_ci(
        per_entity_records: dict[str, list[float]],
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict[str, dict[str, float | str]]:
        """Compute confidence intervals for each entity type.

        Parameters
        ----------
        per_entity_records:
            Mapping of entity_type -> list of per-record F1 scores for that type.
        confidence_level:
            Confidence level for CI (default 0.95 = 95%).
        n_bootstrap:
            Number of bootstrap samples (default 1000).
        seed:
            Random seed for reproducibility (default 42).

        Returns:
            Dict with entity_type -> {"mean": ..., "lower": ..., "upper": ..., "std_error": ...}
        """
        result = {}
        rng_seed = seed
        for entity_type, scores in per_entity_records.items():
            ci_dict = MetricAggregator.compute_metric_confidence_intervals(
                scores,
                metric_name=entity_type,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=rng_seed,
            )
            result[entity_type] = {
                "mean": ci_dict["mean"],
                "lower": ci_dict["lower"],
                "upper": ci_dict["upper"],
                "std_error": ci_dict["std_error"],
            }
            # Increment seed for each entity type to maintain independence
            rng_seed += 1
        return result

    @staticmethod
    def per_language_ci(
        per_language_records: dict[str, list[float]],
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict[str, dict[str, float | str]]:
        """Compute confidence intervals for each language.

        Parameters
        ----------
        per_language_records:
            Mapping of language_code -> list of per-record scores for that language.
        confidence_level:
            Confidence level for CI (default 0.95 = 95%).
        n_bootstrap:
            Number of bootstrap samples (default 1000).
        seed:
            Random seed for reproducibility (default 42).

        Returns:
            Dict with language_code -> {"mean": ..., "lower": ..., "upper": ..., "std_error": ...}
        """
        result = {}
        rng_seed = seed
        for language, scores in per_language_records.items():
            ci_dict = MetricAggregator.compute_metric_confidence_intervals(
                scores,
                metric_name=language,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=rng_seed,
            )
            result[language] = {
                "mean": ci_dict["mean"],
                "lower": ci_dict["lower"],
                "upper": ci_dict["upper"],
                "std_error": ci_dict["std_error"],
            }
            # Increment seed for each language to maintain independence
            rng_seed += 1
        return result

    @staticmethod
    def per_dimension_ci(
        per_dimension_records: dict[str, list[float]],
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict[str, dict[str, float | str]]:
        """Compute confidence intervals for each dimension.

        Parameters
        ----------
        per_dimension_records:
            Mapping of dimension_tag -> list of per-record scores for that dimension.
        confidence_level:
            Confidence level for CI (default 0.95 = 95%).
        n_bootstrap:
            Number of bootstrap samples (default 1000).
        seed:
            Random seed for reproducibility (default 42).

        Returns:
            Dict with dimension_tag -> {"mean": ..., "lower": ..., "upper": ..., "std_error": ...}
        """
        result = {}
        rng_seed = seed
        for dimension, scores in per_dimension_records.items():
            ci_dict = MetricAggregator.compute_metric_confidence_intervals(
                scores,
                metric_name=dimension,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                seed=rng_seed,
            )
            result[dimension] = {
                "mean": ci_dict["mean"],
                "lower": ci_dict["lower"],
                "upper": ci_dict["upper"],
                "std_error": ci_dict["std_error"],
            }
            # Increment seed for each dimension to maintain independence
            rng_seed += 1
        return result

    @staticmethod
    def significance_matrix(
        group_scores: dict[str, list[float]],
        *,
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> dict[str, dict[str, float | bool]]:
        """Compute pairwise significance tests between all groups.

        Performs paired_bootstrap_test for every pair of groups and determines
        significance at the 0.05 and 0.01 levels.

        Parameters
        ----------
        group_scores:
            Mapping of group_name -> list of per-record scores for that group.
        n_bootstrap:
            Number of bootstrap samples (default 10000).
        seed:
            Random seed for reproducibility (default 42).

        Returns:
            Dict with keys like "group_a_vs_group_b" -> {
                "p_value": ...,
                "delta_mean": ...,
                "significant_at_05": bool,
                "significant_at_01": bool
            }
            Unequal-length groups are truncated to minimum length.
        """
        result = {}
        rng_seed = seed
        group_names = sorted(group_scores.keys())

        for i, group_a in enumerate(group_names):
            for j, group_b in enumerate(group_names):
                if i >= j:
                    # Only compute upper triangle (and skip diagonal)
                    continue

                scores_a = group_scores[group_a]
                scores_b = group_scores[group_b]

                # Truncate to minimum length for paired test
                min_len = min(len(scores_a), len(scores_b))
                if min_len == 0:
                    continue

                truncated_a = scores_a[:min_len]
                truncated_b = scores_b[:min_len]

                test_result = MetricAggregator.paired_bootstrap_test(
                    truncated_a,
                    truncated_b,
                    n_bootstrap=n_bootstrap,
                    seed=rng_seed,
                )

                p_value = test_result["p_value"]
                key = f"{group_a}_vs_{group_b}"
                result[key] = {
                    "p_value": p_value,
                    "delta_mean": test_result["delta_mean"],
                    "significant_at_05": p_value < 0.05,
                    "significant_at_01": p_value < 0.01,
                }
                rng_seed += 1

        return result

    @staticmethod
    def minimum_detectable_effect(
        sample_size: int,
        *,
        alpha: float = 0.05,
        power: float = 0.80,
        std_dev: float = 0.15,
    ) -> float:
        """Compute minimum detectable effect size (MDE).

        Based on Cohen's formula for power analysis:
        MDE = (z_alpha + z_beta) * std_dev * sqrt(2 / n)

        Where:
        - z_alpha is the critical value for the significance level (e.g., 1.96 for alpha=0.05)
        - z_beta is the critical value for the power level (e.g., 0.842 for power=0.80)
        - std_dev is the assumed standard deviation of the metric
        - n is the sample size

        Parameters
        ----------
        sample_size:
            Number of samples available (e.g., number of test records).
        alpha:
            Significance level (default 0.05 = 5%, two-tailed).
        power:
            Statistical power (default 0.80 = 80%).
        std_dev:
            Assumed standard deviation of the metric (default 0.15).

        Returns:
            Minimum detectable effect size (float). Smaller values indicate
            better ability to detect differences. Typical interpretation:
            - 0.2 = small effect
            - 0.5 = medium effect
            - 0.8 = large effect
        """
        if sample_size <= 0:
            return float("inf")

        # Standard normal critical values
        # For two-tailed test at alpha=0.05, z_alpha = 1.96
        # For power=0.80, z_beta = 0.842
        z_alpha_map = {
            0.10: 1.645,
            0.05: 1.96,
            0.01: 2.576,
        }
        z_beta_map = {
            0.80: 0.842,
            0.90: 1.282,
            0.95: 1.645,
        }

        z_alpha = z_alpha_map.get(alpha, 1.96)
        z_beta = z_beta_map.get(power, 0.842)

        mde = (z_alpha + z_beta) * std_dev * math.sqrt(2.0 / sample_size)
        return round(mde, 6)
