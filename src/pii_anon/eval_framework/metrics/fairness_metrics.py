"""Fairness assessment metrics across languages, entity types, and demographics.

Evidence basis:
- OpenNER 1.0 (2024): documents performance disparities across languages
- The Bitter Lesson from 2,000+ Multilingual Benchmarks (2024):
  reveals English overrepresentation in NER evaluation
- FairNLP literature: equitable NLP evaluation across demographics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .base import (
    EvalMetricResult,
    EvaluationLevel,
    LabeledSpan,
    MatchMode,
    MultiLevelMetric,
)
from .span_metrics import _aligned_prf


@dataclass
class FairnessReport:
    """Comprehensive fairness analysis for one grouping dimension."""
    max_gap: float = 0.0
    equalized_odds_gap: float = 0.0  # max |TPR_a - TPR_b| across pairs
    demographic_parity_gap: float = 0.0  # max |detection_rate_a - detection_rate_b|
    percentiles: dict[str, float] = field(default_factory=dict)  # p10, p25, p50, p75, p90
    within_group_std: float = 0.0
    floor_violations: list[str] = field(default_factory=list)  # groups below threshold
    per_group_f1: dict[str, float] = field(default_factory=dict)
    per_group_tpr: dict[str, float] = field(default_factory=dict)
    per_group_detection_rate: dict[str, float] = field(default_factory=dict)
    num_groups: int = 0


def _compute_fairness_report(
    per_group_data: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]],
    match_mode: MatchMode,
    fairness_floor: float = 0.7,
) -> FairnessReport:
    """Compute comprehensive fairness report for a grouping dimension.

    Args:
        per_group_data: dict mapping group name to (predictions, labels) tuples
        match_mode: MatchMode for _aligned_prf computation
        fairness_floor: minimum acceptable F1 score threshold

    Returns:
        FairnessReport with all computed fairness metrics
    """
    report = FairnessReport()

    if not per_group_data:
        return report

    per_group_f1: dict[str, float] = {}
    per_group_tpr: dict[str, float] = {}
    per_group_detection_rate: dict[str, float] = {}

    # Compute per-group metrics
    for group_name, (preds, lbls) in per_group_data.items():
        precision, recall, f1, *_ = _aligned_prf(preds, lbls, match_mode)
        per_group_f1[group_name] = round(f1, 6)
        per_group_tpr[group_name] = round(recall, 6)  # TPR = recall

        # Detection rate = len(predictions) / max(len(labels), 1)
        detection_rate = len(preds) / max(len(lbls), 1) if lbls else 0.0
        per_group_detection_rate[group_name] = round(detection_rate, 6)

    report.per_group_f1 = per_group_f1
    report.per_group_tpr = per_group_tpr
    report.per_group_detection_rate = per_group_detection_rate
    report.num_groups = len(per_group_data)

    if not per_group_f1:
        return report

    # Compute max_gap
    f1_values = list(per_group_f1.values())
    if len(f1_values) >= 2:
        report.max_gap = round(max(f1_values) - min(f1_values), 6)
    else:
        report.max_gap = 0.0

    # Compute equalized_odds_gap (max |TPR_a - TPR_b| across pairs)
    tpr_values = list(per_group_tpr.values())
    if len(tpr_values) >= 2:
        max_tpr_gap = 0.0
        for i in range(len(tpr_values)):
            for j in range(i + 1, len(tpr_values)):
                gap = abs(tpr_values[i] - tpr_values[j])
                max_tpr_gap = max(max_tpr_gap, gap)
        report.equalized_odds_gap = round(max_tpr_gap, 6)
    else:
        report.equalized_odds_gap = 0.0

    # Compute demographic_parity_gap (max |detection_rate_a - detection_rate_b|)
    det_values = list(per_group_detection_rate.values())
    if len(det_values) >= 2:
        max_det_gap = 0.0
        for i in range(len(det_values)):
            for j in range(i + 1, len(det_values)):
                gap = abs(det_values[i] - det_values[j])
                max_det_gap = max(max_det_gap, gap)
        report.demographic_parity_gap = round(max_det_gap, 6)
    else:
        report.demographic_parity_gap = 0.0

    # Compute percentiles from F1 values
    sorted_f1 = sorted(f1_values)
    n = len(sorted_f1)

    def percentile(values: list[float], p: float) -> float:
        """Compute percentile using linear interpolation."""
        if not values:
            return 0.0
        if p == 50:
            # Median
            if n % 2 == 1:
                return values[n // 2]
            else:
                return (values[n // 2 - 1] + values[n // 2]) / 2.0
        else:
            # Linear interpolation
            idx = (p / 100.0) * (n - 1)
            lower_idx = int(idx)
            upper_idx = min(lower_idx + 1, n - 1)
            if lower_idx == upper_idx:
                return values[lower_idx]
            fraction = idx - lower_idx
            return values[lower_idx] * (1 - fraction) + values[upper_idx] * fraction

    report.percentiles = {
        "p10": round(percentile(sorted_f1, 10), 6),
        "p25": round(percentile(sorted_f1, 25), 6),
        "p50": round(percentile(sorted_f1, 50), 6),
        "p75": round(percentile(sorted_f1, 75), 6),
        "p90": round(percentile(sorted_f1, 90), 6),
    }

    # Compute within_group_std
    if len(f1_values) >= 2:
        mean_f1 = sum(f1_values) / len(f1_values)
        variance = sum((v - mean_f1) ** 2 for v in f1_values) / len(f1_values)
        report.within_group_std = round(math.sqrt(variance), 6)
    else:
        report.within_group_std = 0.0

    # Find floor_violations
    report.floor_violations = [
        group for group, f1 in per_group_f1.items()
        if f1 < fairness_floor
    ]

    return report


class LanguageFairnessMetric(MultiLevelMetric):
    """Measures performance disparity across languages.

    Groups records by language and computes per-language F1, then reports
    the maximum gap between any two languages.  A gap of 0 means perfectly
    equitable performance.

    ``context`` must include ``record_languages`` — a list mapping each
    span's record_id to its language code.  Alternatively, pass a dict
    ``per_language_groups`` with language -> (predictions, labels) lists.
    """

    name = "language_fairness"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        per_lang: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = ctx.get(
            "per_language_groups", {}
        )

        if not per_lang:
            # Attempt to build from record_languages
            record_langs: dict[str, str] = ctx.get("record_languages", {})
            if record_langs:
                pred_by_lang: dict[str, list[LabeledSpan]] = {}
                label_by_lang: dict[str, list[LabeledSpan]] = {}
                for span in predictions:
                    lang = record_langs.get(span.record_id, "unknown")
                    pred_by_lang.setdefault(lang, []).append(span)
                for span in labels:
                    lang = record_langs.get(span.record_id, "unknown")
                    label_by_lang.setdefault(lang, []).append(span)
                all_langs = set(pred_by_lang.keys()) | set(label_by_lang.keys())
                per_lang = {
                    lang: (pred_by_lang.get(lang, []), label_by_lang.get(lang, []))
                    for lang in all_langs
                }

        if not per_lang:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"languages": 0, "max_gap": 0.0},
            )

        f1_by_lang: dict[str, float] = {}
        for lang, (preds, lbls) in per_lang.items():
            _, _, f1, *_ = _aligned_prf(preds, lbls, match_mode)
            f1_by_lang[lang] = round(f1, 6)

        f1_values = list(f1_by_lang.values())
        max_gap = max(f1_values) - min(f1_values) if len(f1_values) >= 2 else 0.0
        return EvalMetricResult(
            name=self.name, value=round(max_gap, 6), level=level, match_mode=match_mode,
            metadata={"per_language_f1": f1_by_lang, "max_gap": round(max_gap, 6),
                       "languages": len(f1_by_lang)},
        )


class EntityTypeFairnessMetric(MultiLevelMetric):
    """Measures performance disparity across entity types.

    Computes per-entity-type F1 and reports the max gap.
    """

    name = "entity_type_fairness"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        entity_types = sorted({s.entity_type for s in labels})
        f1_by_type: dict[str, float] = {}
        for et in entity_types:
            pf = [s for s in predictions if s.entity_type == et]
            lf = [s for s in labels if s.entity_type == et]
            _, _, f1, *_ = _aligned_prf(pf, lf, match_mode)
            f1_by_type[et] = round(f1, 6)

        vals = list(f1_by_type.values())
        max_gap = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        return EvalMetricResult(
            name=self.name, value=round(max_gap, 6), level=level, match_mode=match_mode,
            metadata={"per_entity_type_f1": f1_by_type, "max_gap": round(max_gap, 6),
                       "entity_types": len(f1_by_type)},
        )


class DifficultyFairnessMetric(MultiLevelMetric):
    """Measures performance disparity across difficulty levels.

    ``context`` must include ``record_difficulties`` — a dict mapping
    record_id to difficulty level string.
    """

    name = "difficulty_fairness"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        record_diffs: dict[str, str] = ctx.get("record_difficulties", {})
        if not record_diffs:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"difficulties": 0},
            )

        pred_by_diff: dict[str, list[LabeledSpan]] = {}
        label_by_diff: dict[str, list[LabeledSpan]] = {}
        for span in predictions:
            diff = record_diffs.get(span.record_id, "unknown")
            pred_by_diff.setdefault(diff, []).append(span)
        for span in labels:
            diff = record_diffs.get(span.record_id, "unknown")
            label_by_diff.setdefault(diff, []).append(span)

        all_diffs = sorted(set(pred_by_diff.keys()) | set(label_by_diff.keys()))
        f1_by_diff: dict[str, float] = {}
        for diff in all_diffs:
            pf = pred_by_diff.get(diff, [])
            lf = label_by_diff.get(diff, [])
            _, _, f1, *_ = _aligned_prf(pf, lf, match_mode)
            f1_by_diff[diff] = round(f1, 6)

        vals = list(f1_by_diff.values())
        max_gap = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        return EvalMetricResult(
            name=self.name, value=round(max_gap, 6), level=level, match_mode=match_mode,
            metadata={"per_difficulty_f1": f1_by_diff, "max_gap": round(max_gap, 6),
                       "difficulties": len(f1_by_diff)},
        )


class ScriptFairnessMetric(MultiLevelMetric):
    """Measures performance disparity across writing systems.

    ``context`` must include ``record_scripts`` — a dict mapping
    record_id to script name (e.g. "Latin", "CJK", "Arabic").
    """

    name = "script_fairness"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        record_scripts: dict[str, str] = ctx.get("record_scripts", {})
        if not record_scripts:
            return EvalMetricResult(
                name=self.name, value=0.0, level=level, match_mode=match_mode,
                metadata={"scripts": 0},
            )

        pred_by_script: dict[str, list[LabeledSpan]] = {}
        label_by_script: dict[str, list[LabeledSpan]] = {}
        for span in predictions:
            script = record_scripts.get(span.record_id, "unknown")
            pred_by_script.setdefault(script, []).append(span)
        for span in labels:
            script = record_scripts.get(span.record_id, "unknown")
            label_by_script.setdefault(script, []).append(span)

        all_scripts = sorted(set(pred_by_script.keys()) | set(label_by_script.keys()))
        f1_by_script: dict[str, float] = {}
        for script in all_scripts:
            pf = pred_by_script.get(script, [])
            lf = label_by_script.get(script, [])
            _, _, f1, *_ = _aligned_prf(pf, lf, match_mode)
            f1_by_script[script] = round(f1, 6)

        vals = list(f1_by_script.values())
        max_gap = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        return EvalMetricResult(
            name=self.name, value=round(max_gap, 6), level=level, match_mode=match_mode,
            metadata={"per_script_f1": f1_by_script, "max_gap": round(max_gap, 6),
                       "scripts": len(f1_by_script)},
        )


class ComprehensiveFairnessMetric(MultiLevelMetric):
    """Measures comprehensive fairness across multiple grouping dimensions.

    Computes FairnessReport for each available grouping (language, entity type,
    difficulty, script). Also checks for intersectional fairness if multiple
    groupings are available.

    ``context`` may include:
    - per_language_groups: dict[str, (predictions, labels)]
    - record_difficulties: dict[record_id, difficulty_level]
    - record_scripts: dict[record_id, script_name]

    Overall value is the average of max_gaps across all dimensions (lower = fairer).
    """

    name = "comprehensive_fairness"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}

        fairness_reports: dict[str, FairnessReport] = {}
        max_gaps: list[float] = []

        # Language fairness
        per_lang: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = ctx.get(
            "per_language_groups", {}
        )
        if per_lang:
            report = _compute_fairness_report(per_lang, match_mode)
            fairness_reports["language"] = report
            max_gaps.append(report.max_gap)

        # Difficulty fairness
        record_diffs: dict[str, str] = ctx.get("record_difficulties", {})
        if record_diffs:
            pred_by_diff: dict[str, list[LabeledSpan]] = {}
            label_by_diff: dict[str, list[LabeledSpan]] = {}
            for span in predictions:
                diff = record_diffs.get(span.record_id, "unknown")
                pred_by_diff.setdefault(diff, []).append(span)
            for span in labels:
                diff = record_diffs.get(span.record_id, "unknown")
                label_by_diff.setdefault(diff, []).append(span)
            all_diffs = set(pred_by_diff.keys()) | set(label_by_diff.keys())
            per_diff = {
                d: (pred_by_diff.get(d, []), label_by_diff.get(d, []))
                for d in all_diffs
            }
            if per_diff:
                report = _compute_fairness_report(per_diff, match_mode)
                fairness_reports["difficulty"] = report
                max_gaps.append(report.max_gap)

        # Script fairness
        record_scripts: dict[str, str] = ctx.get("record_scripts", {})
        if record_scripts:
            pred_by_script: dict[str, list[LabeledSpan]] = {}
            label_by_script: dict[str, list[LabeledSpan]] = {}
            for span in predictions:
                script = record_scripts.get(span.record_id, "unknown")
                pred_by_script.setdefault(script, []).append(span)
            for span in labels:
                script = record_scripts.get(span.record_id, "unknown")
                label_by_script.setdefault(script, []).append(span)
            all_scripts = set(pred_by_script.keys()) | set(label_by_script.keys())
            per_script = {
                s: (pred_by_script.get(s, []), label_by_script.get(s, []))
                for s in all_scripts
            }
            if per_script:
                report = _compute_fairness_report(per_script, match_mode)
                fairness_reports["script"] = report
                max_gaps.append(report.max_gap)

        # Entity type fairness
        entity_types = sorted({s.entity_type for s in labels})
        if entity_types:
            per_entity = {
                et: (
                    [s for s in predictions if s.entity_type == et],
                    [s for s in labels if s.entity_type == et]
                )
                for et in entity_types
            }
            if per_entity:
                report = _compute_fairness_report(per_entity, match_mode)
                fairness_reports["entity_type"] = report
                max_gaps.append(report.max_gap)

        # Intersectional fairness (if both language and entity type available)
        if per_lang and entity_types:
            intersectional_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = {}
            for lang in per_lang:
                for et in entity_types:
                    group_key = f"{lang}_{et}"
                    lang_preds, lang_labels = per_lang[lang]
                    preds_for_group = [s for s in lang_preds if s.entity_type == et]
                    labels_for_group = [s for s in lang_labels if s.entity_type == et]
                    intersectional_groups[group_key] = (preds_for_group, labels_for_group)

            if intersectional_groups:
                report = _compute_fairness_report(intersectional_groups, match_mode)
                fairness_reports["intersectional"] = report
                max_gaps.append(report.max_gap)

        # Compute overall value
        overall_value = sum(max_gaps) / len(max_gaps) if max_gaps else 0.0

        # Convert reports to dicts for metadata
        metadata = {
            "fairness_reports": {
                key: {
                    "max_gap": report.max_gap,
                    "equalized_odds_gap": report.equalized_odds_gap,
                    "demographic_parity_gap": report.demographic_parity_gap,
                    "percentiles": report.percentiles,
                    "within_group_std": report.within_group_std,
                    "floor_violations": report.floor_violations,
                    "per_group_f1": report.per_group_f1,
                    "per_group_tpr": report.per_group_tpr,
                    "per_group_detection_rate": report.per_group_detection_rate,
                    "num_groups": report.num_groups,
                }
                for key, report in fairness_reports.items()
            },
            "dimensions_analyzed": len(fairness_reports),
            "overall_max_gap_average": round(overall_value, 6),
        }

        return EvalMetricResult(
            name=self.name,
            value=round(overall_value, 6),
            level=level,
            match_mode=match_mode,
            metadata=metadata,
        )


class IntersectionalFairnessMetric(MultiLevelMetric):
    """Measures fairness across intersectional groupings.

    For example, combining language and entity type to identify performance
    disparities at the intersection of multiple demographic dimensions.

    ``context`` must include ``intersectional_groups`` — a dict mapping
    intersectional group names (e.g. "en_PERSON", "fr_EMAIL_ADDRESS") to
    (predictions, labels) tuples.
    """

    name = "intersectional_fairness"

    @property
    def supported_levels(self) -> list[EvaluationLevel]:
        return [EvaluationLevel.ENTITY]

    def compute(
        self,
        predictions: list[LabeledSpan],
        labels: list[LabeledSpan],
        *,
        level: EvaluationLevel = EvaluationLevel.ENTITY,
        match_mode: MatchMode = MatchMode.STRICT,
        context: dict[str, Any] | None = None,
    ) -> EvalMetricResult:
        ctx = context or {}
        intersectional_groups: dict[str, tuple[list[LabeledSpan], list[LabeledSpan]]] = ctx.get(
            "intersectional_groups", {}
        )

        if not intersectional_groups:
            return EvalMetricResult(
                name=self.name,
                value=0.0,
                level=level,
                match_mode=match_mode,
                metadata={"intersectional_groups": 0},
            )

        # Compute fairness report
        report = _compute_fairness_report(intersectional_groups, match_mode)

        metadata = {
            "max_gap": report.max_gap,
            "equalized_odds_gap": report.equalized_odds_gap,
            "demographic_parity_gap": report.demographic_parity_gap,
            "percentiles": report.percentiles,
            "within_group_std": report.within_group_std,
            "floor_violations": report.floor_violations,
            "per_group_f1": report.per_group_f1,
            "per_group_tpr": report.per_group_tpr,
            "per_group_detection_rate": report.per_group_detection_rate,
            "num_groups": report.num_groups,
        }

        return EvalMetricResult(
            name=self.name,
            value=report.max_gap,
            level=level,
            match_mode=match_mode,
            metadata=metadata,
        )
