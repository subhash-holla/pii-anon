"""Tests for eval_framework.evaluation.reporting module."""

from __future__ import annotations

import json

from pii_anon.eval_framework.evaluation.framework import (
    BatchEvaluationReport,
    ComprehensiveEvaluationReport,
    EvaluationFrameworkConfig,
    EvaluationReport,
)
from pii_anon.eval_framework.evaluation.reporting import ReportGenerator


def _single_report(**kwargs) -> EvaluationReport:
    defaults = dict(
        evaluation_id="eval-001",
        timestamp="2026-02-22T00:00:00Z",
        config=EvaluationFrameworkConfig(),
        language="en",
        precision=0.85,
        recall=0.90,
        f1=0.875,
        records_evaluated=100,
        privacy_score=0.95,
        fairness_score=0.88,
    )
    defaults.update(kwargs)
    return EvaluationReport(**defaults)


def _batch_report(**kwargs) -> BatchEvaluationReport:
    defaults = dict(
        evaluation_id="batch-001",
        timestamp="2026-02-22T00:00:00Z",
        records_evaluated=500,
        micro_averaged={"precision": 0.85, "recall": 0.90, "f1": 0.875},
        macro_averaged={"precision": 0.80, "recall": 0.85, "f1": 0.825},
        weighted_averaged={"precision": 0.82, "recall": 0.87, "f1": 0.845},
        per_entity_type={
            "PERSON_NAME": {"precision": 0.90, "recall": 0.95, "f1": 0.925, "support": 200},
            "EMAIL_ADDRESS": {"precision": 0.85, "recall": 0.80, "f1": 0.825, "support": 100},
        },
        per_language={
            "en": {"precision": 0.88, "recall": 0.92, "f1": 0.90},
            "de": {"precision": 0.80, "recall": 0.85, "f1": 0.825},
        },
        per_difficulty={
            "easy": {"precision": 0.95, "recall": 0.98, "f1": 0.965},
            "hard": {"precision": 0.70, "recall": 0.75, "f1": 0.725},
        },
        confidence_interval=(0.85, 0.90),
    )
    defaults.update(kwargs)
    return BatchEvaluationReport(**defaults)


def _comprehensive_report(**kwargs) -> ComprehensiveEvaluationReport:
    defaults = dict(
        evaluation_id="comp-001",
        timestamp="2026-02-22T00:00:00Z",
        records_evaluated=1000,
        micro_averaged={"precision": 0.85, "recall": 0.90, "f1": 0.875},
        macro_averaged={"precision": 0.80, "recall": 0.85, "f1": 0.825},
        weighted_averaged={"precision": 0.82, "recall": 0.87, "f1": 0.845},
        per_entity_type={
            "PERSON_NAME": {"precision": 0.90, "recall": 0.95, "f1": 0.925, "support": 500},
        },
        privacy_score=0.92,
        leakage_score=0.05,
        utility_score=0.88,
        format_preservation=0.95,
        information_loss=0.10,
        fairness_score=0.90,
        fairness_details={"overall_gap": 0.05},
        per_dimension={
            "boundary": {"precision": 0.75, "recall": 0.80, "f1": 0.775},
        },
    )
    defaults.update(kwargs)
    return ComprehensiveEvaluationReport(**defaults)


# ── to_json ──────────────────────────────────────────────────────────


class TestToJson:
    def test_single_report(self) -> None:
        report = _single_report()
        result = ReportGenerator.to_json(report)
        data = json.loads(result)
        assert data["evaluation_id"] == "eval-001"
        assert data["f1"] == 0.875

    def test_batch_report(self) -> None:
        report = _batch_report()
        result = ReportGenerator.to_json(report)
        data = json.loads(result)
        assert data["records_evaluated"] == 500
        assert "micro_averaged" in data

    def test_comprehensive_report(self) -> None:
        report = _comprehensive_report()
        result = ReportGenerator.to_json(report)
        data = json.loads(result)
        assert data["privacy_score"] == 0.92


# ── to_markdown ──────────────────────────────────────────────────────


class TestToMarkdown:
    def test_single_report_markdown(self) -> None:
        report = _single_report()
        md = ReportGenerator.to_markdown(report)
        assert "# PII Evaluation Report" in md
        assert "**Language:** en" in md
        assert "0.8750" in md

    def test_batch_report_markdown(self) -> None:
        report = _batch_report()
        md = ReportGenerator.to_markdown(report)
        assert "## Aggregated Metrics" in md
        assert "## Per-Language Performance" in md
        assert "## Per-Difficulty Performance" in md
        assert "95% Confidence Interval" in md

    def test_comprehensive_report_markdown(self) -> None:
        report = _comprehensive_report()
        md = ReportGenerator.to_markdown(report)
        assert "## Privacy & Security" in md
        assert "## Utility Metrics" in md
        assert "## Fairness" in md
        assert "Overall Gap" in md


# ── to_csv ───────────────────────────────────────────────────────────


class TestToCsv:
    def test_batch_csv(self) -> None:
        report = _batch_report()
        csv = ReportGenerator.to_csv(report)
        lines = csv.strip().split("\n")
        assert lines[0] == "entity_type,precision,recall,f1,support"
        assert len(lines) == 3  # header + 2 entity types

    def test_comprehensive_csv(self) -> None:
        report = _comprehensive_report()
        csv = ReportGenerator.to_csv(report)
        assert "PERSON_NAME" in csv


# ── to_latex ─────────────────────────────────────────────────────────


class TestToLatex:
    def test_batch_latex(self) -> None:
        report = _batch_report()
        tex = ReportGenerator.to_latex(report)
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex
        assert "PERSON_NAME" in tex
        assert "Micro-avg" in tex
        assert "Macro-avg" in tex
        assert "Weighted-avg" in tex

    def test_comprehensive_latex(self) -> None:
        report = _comprehensive_report()
        tex = ReportGenerator.to_latex(report)
        assert "PERSON_NAME" in tex


# ── to_dashboard_json ────────────────────────────────────────────────


class TestToDashboardJson:
    def test_batch_dashboard(self) -> None:
        report = _batch_report()
        result = ReportGenerator.to_dashboard_json(report)
        data = json.loads(result)
        assert data["summary"]["f1"] == 0.875
        assert "per_entity" in data
        assert "per_language" in data

    def test_single_report_dashboard(self) -> None:
        report = _single_report(f1=0.875)
        result = ReportGenerator.to_dashboard_json(report)
        data = json.loads(result)
        assert data["summary"]["f1"] == 0.875

    def test_comprehensive_dashboard(self) -> None:
        report = _comprehensive_report()
        result = ReportGenerator.to_dashboard_json(report)
        data = json.loads(result)
        assert data["summary"]["privacy"] == 0.92


# ── comparison_report ────────────────────────────────────────────────


class TestComparisonReport:
    def test_empty_systems(self) -> None:
        result = ReportGenerator.comparison_report({})
        assert "No systems" in result

    def test_multiple_systems(self) -> None:
        systems = {
            "system_a": {"f1": 0.85, "precision": 0.80, "recall": 0.90, "privacy_score": 0.95, "fairness_score": 0.88, "floor_gates_passed": True},
            "system_b": {"micro_averaged": {"f1": 0.75, "precision": 0.70, "recall": 0.80}, "floor_gates_passed": False},
        }
        md = ReportGenerator.comparison_report(systems)
        assert "system_a" in md
        assert "system_b" in md
        assert "✓" in md
        assert "✗" in md


# ── executive_summary ────────────────────────────────────────────────


class TestExecutiveSummary:
    def test_single_report_summary(self) -> None:
        report = _single_report()
        summary = ReportGenerator.executive_summary(report)
        assert "F1 of 0.8750" in summary

    def test_batch_with_ci(self) -> None:
        report = _batch_report()
        summary = ReportGenerator.executive_summary(report)
        assert "95% CI" in summary

    def test_comprehensive_with_privacy(self) -> None:
        report = _comprehensive_report()
        summary = ReportGenerator.executive_summary(report)
        assert "Privacy protection" in summary

    def test_language_fairness_gap(self) -> None:
        report = _batch_report()
        summary = ReportGenerator.executive_summary(report)
        assert "Fairness gap" in summary

    def test_worst_entity(self) -> None:
        report = _batch_report()
        summary = ReportGenerator.executive_summary(report)
        assert "EMAIL_ADDRESS" in summary


# ── render_comparison ────────────────────────────────────────────────


class TestRenderComparison:
    def test_multiple_reports(self) -> None:
        reports = [
            _single_report(evaluation_id="r1"),
            _single_report(evaluation_id="r2", f1=0.92),
        ]
        md = ReportGenerator.render_comparison(reports)
        assert "r1" in md
        assert "r2" in md
        assert "Comparison" in md
