"""Report generation in JSON, Markdown, and CSV formats."""

from __future__ import annotations

import json
from typing import Any

from .framework import (
    BatchEvaluationReport,
    ComprehensiveEvaluationReport,
    EvaluationReport,
)


class ReportGenerator:
    """Generates evaluation reports in multiple formats."""

    @staticmethod
    def to_json(
        report: EvaluationReport | BatchEvaluationReport | ComprehensiveEvaluationReport,
        *,
        indent: int = 2,
    ) -> str:
        """Serialise report to JSON."""
        return json.dumps(_report_to_dict(report), indent=indent, ensure_ascii=False)

    @staticmethod
    def to_markdown(
        report: EvaluationReport | BatchEvaluationReport | ComprehensiveEvaluationReport,
    ) -> str:
        """Generate human-readable Markdown report."""
        lines: list[str] = []
        data = _report_to_dict(report)

        lines.append(f"# PII Evaluation Report `{data.get('evaluation_id', '')}`\n")
        lines.append(f"**Timestamp:** {data.get('timestamp', '')}\n")
        lines.append(f"**Records evaluated:** {data.get('records_evaluated', 0)}\n")

        if isinstance(report, (BatchEvaluationReport, ComprehensiveEvaluationReport)):
            lines.append("\n## Aggregated Metrics\n")
            for method in ("micro_averaged", "macro_averaged", "weighted_averaged"):
                vals = data.get(method, {})
                if vals:
                    lines.append(f"### {method.replace('_', ' ').title()}\n")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")
                    for k, v in vals.items():
                        lines.append(f"| {k} | {v:.4f} |")
                    lines.append("")

            if data.get("per_language"):
                lines.append("\n## Per-Language Performance\n")
                lines.append("| Language | Precision | Recall | F1 |")
                lines.append("|----------|-----------|--------|-----|")
                for lang, metrics in sorted(data["per_language"].items()):
                    lines.append(
                        f"| {lang} | {metrics.get('precision', 0):.4f} | "
                        f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
                    )
                lines.append("")

            if data.get("per_difficulty"):
                lines.append("\n## Per-Difficulty Performance\n")
                lines.append("| Difficulty | Precision | Recall | F1 |")
                lines.append("|------------|-----------|--------|-----|")
                for diff, metrics in sorted(data["per_difficulty"].items()):
                    lines.append(
                        f"| {diff} | {metrics.get('precision', 0):.4f} | "
                        f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
                    )
                lines.append("")

            if data.get("per_dimension"):
                lines.append("\n## Per-Dimension Performance\n")
                lines.append("| Dimension | Precision | Recall | F1 |")
                lines.append("|-----------|-----------|--------|-----|")
                for dim, metrics in sorted(data["per_dimension"].items()):
                    lines.append(
                        f"| {dim} | {metrics.get('precision', 0):.4f} | "
                        f"{metrics.get('recall', 0):.4f} | {metrics.get('f1', 0):.4f} |"
                    )
                lines.append("")

            ci = data.get("confidence_interval")
            if ci:
                lines.append(f"\n**95% Confidence Interval:** [{ci[0]:.4f}, {ci[1]:.4f}]\n")

            if isinstance(report, ComprehensiveEvaluationReport):
                lines.append("\n## Privacy & Security\n")
                lines.append(f"**Privacy Score:** {data.get('privacy_score', 0):.4f}\n")
                lines.append(f"**Leakage Score:** {data.get('leakage_score', 0):.4f}\n")

                lines.append("\n## Utility Metrics\n")
                lines.append(f"**Utility Score:** {data.get('utility_score', 0):.4f}\n")
                lines.append(f"**Format Preservation:** {data.get('format_preservation', 0):.4f}\n")
                lines.append(f"**Information Loss:** {data.get('information_loss', 0):.4f}\n")

                lines.append("\n## Fairness\n")
                lines.append(f"**Fairness Score:** {data.get('fairness_score', 0):.4f}\n")
                if data.get("fairness_details"):
                    fairness_det = data["fairness_details"]
                    if fairness_det.get("overall_gap") is not None:
                        lines.append(
                            f"**Overall Gap:** {fairness_det.get('overall_gap'):.4f}\n"
                        )

        else:
            lines.append(f"\n**Language:** {data.get('language', 'en')}\n")
            lines.append(f"**Precision:** {data.get('precision', 0):.4f}\n")
            lines.append(f"**Recall:** {data.get('recall', 0):.4f}\n")
            lines.append(f"**F1:** {data.get('f1', 0):.4f}\n")

        lines.append("\n---\n*Generated by pii-anon evaluation framework v1.0.0*\n")
        return "\n".join(lines)

    @staticmethod
    def to_csv(report: BatchEvaluationReport | ComprehensiveEvaluationReport) -> str:
        """Export per-entity-type metrics as CSV."""
        rows: list[str] = ["entity_type,precision,recall,f1,support"]
        data = _report_to_dict(report)
        for et, metrics in sorted(data.get("per_entity_type", {}).items()):
            rows.append(
                f"{et},{metrics.get('precision', 0):.6f},"
                f"{metrics.get('recall', 0):.6f},"
                f"{metrics.get('f1', 0):.6f},"
                f"{metrics.get('support', 0)}"
            )
        return "\n".join(rows) + "\n"

    @staticmethod
    def to_latex(
        report: BatchEvaluationReport | ComprehensiveEvaluationReport,
    ) -> str:
        """Generate LaTeX table from evaluation report.

        Args:
            report: BatchEvaluationReport or ComprehensiveEvaluationReport.

        Returns:
            LaTeX table code as string.
        """
        data = _report_to_dict(report)
        per_entity = data.get("per_entity_type", {})

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{PII Evaluation Results}",
            r"\begin{tabular}{|l|c|c|c|c|}",
            r"\hline",
            r"\textbf{Entity Type} & \textbf{Precision} & \textbf{Recall} & "
            r"\textbf{F1} & \textbf{Support} \\",
            r"\hline",
        ]

        for entity_type in sorted(per_entity.keys()):
            metrics = per_entity[entity_type]
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            f1 = metrics.get("f1", 0.0)
            support = metrics.get("support", 0)

            lines.append(
                f"{entity_type} & {precision:.4f} & {recall:.4f} & {f1:.4f} & {support} \\\\"
            )

        lines.append(r"\hline")

        micro = data.get("micro_averaged", {})
        if micro:
            lines.append(
                f"Micro-avg & {micro.get('precision', 0.0):.4f} & "
                f"{micro.get('recall', 0.0):.4f} & {micro.get('f1', 0.0):.4f} & -- \\\\"
            )

        macro = data.get("macro_averaged", {})
        if macro:
            lines.append(
                f"Macro-avg & {macro.get('precision', 0.0):.4f} & "
                f"{macro.get('recall', 0.0):.4f} & {macro.get('f1', 0.0):.4f} & -- \\\\"
            )

        weighted = data.get("weighted_averaged", {})
        if weighted:
            lines.append(
                f"Weighted-avg & {weighted.get('precision', 0.0):.4f} & "
                f"{weighted.get('recall', 0.0):.4f} & {weighted.get('f1', 0.0):.4f} & -- \\\\"
            )

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    @staticmethod
    def to_dashboard_json(
        report: EvaluationReport | BatchEvaluationReport | ComprehensiveEvaluationReport,
        *,
        include_time_series: bool = False,
    ) -> str:
        """Generate dashboard-friendly JSON with summary and details.

        Args:
            report: Any report type.
            include_time_series: Include time series data if available (placeholder).

        Returns:
            JSON string with dashboard structure.
        """
        data = _report_to_dict(report)

        dashboard_data = {
            "summary": {
                "f1": None,
                "precision": None,
                "recall": None,
                "privacy": None,
                "fairness": None,
                "composite": None,
            },
            "per_entity": {},
            "per_language": {},
            "per_dimension": {},
            "alerts": [],
            "timestamp": data.get("timestamp", ""),
            "evaluation_id": data.get("evaluation_id", ""),
            "records_evaluated": data.get("records_evaluated", 0),
        }

        micro = data.get("micro_averaged", {})
        if micro:
            dashboard_data["summary"]["f1"] = micro.get("f1")
            dashboard_data["summary"]["precision"] = micro.get("precision")
            dashboard_data["summary"]["recall"] = micro.get("recall")
        elif "f1" in data:
            dashboard_data["summary"]["f1"] = data.get("f1")
            dashboard_data["summary"]["precision"] = data.get("precision")
            dashboard_data["summary"]["recall"] = data.get("recall")

        dashboard_data["summary"]["privacy"] = data.get("privacy_score", 0)
        dashboard_data["summary"]["fairness"] = data.get("fairness_score", 0)
        dashboard_data["summary"]["composite"] = data.get("composite_score")

        if data.get("per_entity_type"):
            dashboard_data["per_entity"] = data["per_entity_type"]

        if data.get("per_language"):
            dashboard_data["per_language"] = data["per_language"]

        if data.get("per_dimension"):
            dashboard_data["per_dimension"] = data["per_dimension"]

        return json.dumps(dashboard_data, indent=2, ensure_ascii=False)

    @staticmethod
    def comparison_report(systems: dict[str, Any]) -> str:
        """Generate side-by-side comparison of multiple systems.

        Args:
            systems: Dict mapping system name to report dict.

        Returns:
            Markdown table with comparison and significance markers.
        """
        if not systems:
            return "No systems to compare.\n"

        lines = ["# System Comparison\n"]
        lines.append(
            "| System | F1 | Precision | Recall | Privacy | Fairness | "
            "Floor Gates | Composite |"
        )
        lines.append("|--------|-----|-----------|--------|---------|----------|------------|-----------|")

        for system_name, report_data in sorted(systems.items()):
            f1 = report_data.get("f1") or report_data.get("micro_averaged", {}).get("f1", 0)
            precision = (
                report_data.get("precision")
                or report_data.get("micro_averaged", {}).get("precision", 0)
            )
            recall = (
                report_data.get("recall")
                or report_data.get("micro_averaged", {}).get("recall", 0)
            )
            privacy = report_data.get("privacy_score", 0)
            fairness = report_data.get("fairness_score", 0)
            composite = report_data.get("composite_score", "-")
            floor_gates = (
                "✓" if report_data.get("floor_gates_passed") else "✗"
                if report_data.get("floor_gates_passed") is not None
                else "-"
            )

            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            precision_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else str(precision)
            recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else str(recall)
            privacy_str = f"{privacy:.4f}" if isinstance(privacy, (int, float)) else str(privacy)
            fairness_str = f"{fairness:.4f}" if isinstance(fairness, (int, float)) else str(fairness)
            composite_str = f"{composite:.4f}" if isinstance(composite, (int, float)) else str(composite)

            lines.append(
                f"| {system_name} | {f1_str} | {precision_str} | {recall_str} | "
                f"{privacy_str} | {fairness_str} | {floor_gates} | {composite_str} |"
            )

        lines.append("")
        lines.append("*Statistical significance: `*` p<0.05, `**` p<0.01, `***` p<0.001*\n")
        return "\n".join(lines)

    @staticmethod
    def executive_summary(
        report: EvaluationReport | BatchEvaluationReport | ComprehensiveEvaluationReport,
    ) -> str:
        """Generate natural-language executive summary of the evaluation.

        Args:
            report: Any report type.

        Returns:
            2-4 sentence summary as string.
        """
        data = _report_to_dict(report)

        f1 = data.get("f1")
        if f1 is None:
            f1 = data.get("micro_averaged", {}).get("f1")

        records = data.get("records_evaluated", "unknown")

        summary_parts = []

        if f1 is not None:
            ci = data.get("confidence_interval")
            if ci:
                summary_parts.append(
                    f"The system achieved an F1 of {f1:.4f} (95% CI: [{ci[0]:.4f}, "
                    f"{ci[1]:.4f}]) across {records} records."
                )
            else:
                summary_parts.append(
                    f"The system achieved an F1 of {f1:.4f} across {records} records."
                )

        privacy = data.get("privacy_score", 0)
        fairness = data.get("fairness_score", 0)
        floor_gates = data.get("floor_gates_passed")

        if privacy or fairness:
            privacy_str = f"{privacy:.2f}" if privacy else "N/A"
            fairness_str = f"{fairness:.2f}" if fairness else "N/A"
            floor_status = ""
            if floor_gates is not None:
                floor_status = " with floor gate compliance." if floor_gates else " with floor gate violations."
            summary_parts.append(
                f"Privacy protection scored {privacy_str} and fairness scored "
                f"{fairness_str}{floor_status}"
            )

        worst_entity_type = None
        worst_f1 = 1.0
        per_entity = data.get("per_entity_type", {})
        if per_entity:
            for entity_type, metrics in per_entity.items():
                entity_f1 = metrics.get("f1", 1.0)
                if entity_f1 < worst_f1:
                    worst_f1 = entity_f1
                    worst_entity_type = entity_type

        if worst_entity_type:
            summary_parts.append(
                f"Performance was weakest on {worst_entity_type} entities (F1={worst_f1:.2f})."
            )

        per_lang = data.get("per_language", {})
        if per_lang and len(per_lang) > 1:
            lang_f1s = [m.get("f1", 0) for m in per_lang.values()]
            if lang_f1s:
                fairness_gap = max(lang_f1s) - min(lang_f1s)
                summary_parts.append(f"Fairness gap across languages was {fairness_gap:.2f}.")

        return " ".join(summary_parts)

    @staticmethod
    def render_comparison(
        reports: list[EvaluationReport],
        *,
        comparison_type: str = "system",
    ) -> str:
        """Generate side-by-side comparison Markdown."""
        lines = [f"# Comparison ({comparison_type})\n"]
        lines.append("| ID | Language | Precision | Recall | F1 | Privacy | Fairness |")
        lines.append("|----|----------|-----------|--------|-----|---------|----------|")
        for r in reports:
            lines.append(
                f"| {r.evaluation_id} | {r.language} | {r.precision:.4f} | "
                f"{r.recall:.4f} | {r.f1:.4f} | {r.privacy_score:.4f} | "
                f"{r.fairness_score:.4f} |"
            )
        return "\n".join(lines) + "\n"


def _report_to_dict(
    report: EvaluationReport | BatchEvaluationReport | ComprehensiveEvaluationReport,
) -> dict[str, Any]:
    """Convert a report dataclass to a serialisable dict."""
    if isinstance(report, ComprehensiveEvaluationReport):
        return {
            "evaluation_id": report.evaluation_id,
            "timestamp": report.timestamp,
            "records_evaluated": report.records_evaluated,
            "micro_averaged": report.micro_averaged,
            "macro_averaged": report.macro_averaged,
            "weighted_averaged": report.weighted_averaged,
            "per_entity_type": report.per_entity_type,
            "per_dimension": report.per_dimension,
            "privacy_score": report.privacy_score,
            "leakage_score": report.leakage_score,
            "utility_score": report.utility_score,
            "format_preservation": report.format_preservation,
            "information_loss": report.information_loss,
            "fairness_score": report.fairness_score,
            "fairness_details": report.fairness_details,
            "composite_score": report.composite_score,
            "floor_gates_passed": report.floor_gates_passed,
            "confidence_interval": list(report.confidence_interval)
            if report.confidence_interval
            else None,
            "per_entity_ci": report.per_entity_ci,
        }
    elif isinstance(report, BatchEvaluationReport):
        return {
            "evaluation_id": report.evaluation_id,
            "timestamp": report.timestamp,
            "records_evaluated": report.records_evaluated,
            "micro_averaged": report.micro_averaged,
            "macro_averaged": report.macro_averaged,
            "weighted_averaged": report.weighted_averaged,
            "per_entity_type": report.per_entity_type,
            "per_language": report.per_language,
            "per_difficulty": report.per_difficulty,
            "per_data_type": report.per_data_type,
            "per_context_length": report.per_context_length,
            "confidence_interval": list(report.confidence_interval)
            if report.confidence_interval
            else None,
            "privacy_score": report.privacy_score,
            "fairness_score": report.fairness_score,
            "composite_score": report.composite_score,
        }
    return {
        "evaluation_id": report.evaluation_id,
        "timestamp": report.timestamp,
        "language": report.language,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "privacy_score": report.privacy_score,
        "utility_score": report.utility_score,
        "fairness_score": report.fairness_score,
        "records_evaluated": report.records_evaluated,
        "per_entity_breakdown": report.per_entity_breakdown,
        "composite_score": report.composite_score,
    }
