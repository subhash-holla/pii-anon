#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

README_START = "<!-- BENCHMARK_SUMMARY_START -->"
README_END = "<!-- BENCHMARK_SUMMARY_END -->"
SUPPORTED_REPORT_SCHEMA = {"2026-02-15.v2", "2026-02-19.v3"}

# Dataset descriptions used in the benchmark summary section.
_DATASET_DESCRIPTIONS: dict[str, str] = {
    "pii_anon_benchmark_v1": (
        "Research-grade unified benchmark dataset (50 000 records, 22 entity types, "
        "12 languages, 7 evaluation dimensions). Covers core PII detection "
        "(35 700 records), long-context entity tracking (2 800 records), and "
        "dimension-specific probes (11 500 records) spanning entity consistency, "
        "multilingual support, context preservation, PII type coverage, edge cases, "
        "format variations, and temporal consistency."
    ),
}


def _validate_payload(payload: dict[str, Any]) -> None:
    schema_version = str(payload.get("report_schema_version", ""))
    if schema_version not in SUPPORTED_REPORT_SCHEMA:
        raise SystemExit(
            f"unsupported benchmark report schema `{schema_version}`; "
            "regenerate artifacts with the current benchmark runner."
        )

    # v3 combined reports have a different top-level structure.
    if schema_version == "2026-02-19.v3" and "by_dataset" in payload:
        # Combined report — validated separately.
        return

    required_top = {
        "report_schema_version",
        "dataset",
        "dataset_source",
        "warmup_samples",
        "measured_runs",
        "floor_pass",
        "qualification_gate_pass",
        "expected_competitors",
        "available_competitors",
        "unavailable_competitors",
        "all_competitors_available",
        "run_metadata",
        "failed_profiles",
        "required_profiles",
        "required_profiles_passed",
        "profile_results",
        "systems",
        "profiles",
    }
    missing = sorted(required_top.difference(payload))
    if missing:
        raise SystemExit(
            f"benchmark report missing required fields: {', '.join(missing)}. "
            "Regenerate artifacts with the current benchmark runner."
        )

    systems = payload.get("systems")
    if not isinstance(systems, list):
        raise SystemExit("benchmark report field `systems` must be a list")

    required_row = {"system", "available", "license_gate_passed", "evaluation_track"}
    for index, row in enumerate(systems):
        if not isinstance(row, dict):
            raise SystemExit(f"benchmark report system row {index} is not an object")
        row_missing = sorted(required_row.difference(row))
        if row_missing:
            raise SystemExit(
                f"benchmark report system row {index} missing fields: {', '.join(row_missing)}"
            )


def _best_value(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool = False) -> float:
    values = [
        float(item[key])
        for item in rows
        if item.get("available") and item.get("license_gate_passed", True) and key in item
    ]
    if not values:
        return 0.0
    return min(values) if lower_is_better else max(values)


def _strengths_and_weaknesses(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    core = next((item for item in rows if item.get("system") == "pii-anon"), None)
    available = [item for item in rows if item.get("available") and item.get("license_gate_passed", True)]
    if not core or not available:
        return ["No benchmark data available."], ["No benchmark data available."]

    strengths: list[str] = []
    weaknesses: list[str] = []

    metrics = [
        ("composite_score", False),
        ("precision", False),
        ("recall", False),
        ("f1", False),
        ("docs_per_hour", False),
        ("latency_p50_ms", True),
    ]

    for metric, lower_is_better in metrics:
        if metric not in core:
            continue
        best = _best_value(available, metric, lower_is_better=lower_is_better)
        cur = float(core[metric])
        if best <= 0:
            continue

        if lower_is_better:
            if cur <= best * 1.05:
                strengths.append(f"{metric}: within 5% of best ({cur:.3f} vs best {best:.3f}).")
            elif cur > best * 1.10:
                weaknesses.append(f"{metric}: more than 10% slower than best ({cur:.3f} vs best {best:.3f}).")
        else:
            if cur >= best * 0.95:
                strengths.append(f"{metric}: within 5% of best ({cur:.3f} vs best {best:.3f}).")
            elif cur < best * 0.90:
                weaknesses.append(f"{metric}: more than 10% below best ({cur:.3f} vs best {best:.3f}).")

    if not strengths:
        strengths = ["No metric met the strength threshold in this run."]
    if not weaknesses:
        weaknesses = ["No metric crossed the weakness threshold in this run."]
    return strengths, weaknesses


def _table(rows: list[dict[str, Any]]) -> str:
    out = [
        "| System | Status | Composite | F1 | 95% CI | Precision | Recall | p50 Latency (ms) | Docs/hour | Elo |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("system") != "pii-anon" and not row.get("license_gate_passed", True):
            status = f"excluded ({row.get('license_gate_reason')})"
        elif row.get("available"):
            status = "available"
        else:
            status = f"skipped ({row.get('skipped_reason')})"
        composite = float(row.get("composite_score", 0.0))
        elo = float(row.get("elo_rating", 0.0))
        f1_val = float(row.get("f1", 0.0))
        ci_lower = float(row.get("f1_ci_lower", 0.0))
        ci_upper = float(row.get("f1_ci_upper", 0.0))
        ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]" if ci_lower > 0 or ci_upper > 0 else "—"
        out.append(
            f"| {row.get('system')} | {status} | "
            f"{composite:.4f} | "
            f"{f1_val:.3f} | {ci_str} | "
            f"{float(row.get('precision', 0.0)):.3f} | {float(row.get('recall', 0.0)):.3f} | "
            f"{float(row.get('latency_p50_ms', 0.0)):.3f} | "
            f"{float(row.get('docs_per_hour', 0.0)):.2f} | "
            f"{elo:.0f} |"
        )
    return "\n".join(out)


def _render(rows: list[dict[str, Any]], dataset: str, warmup: int, measured: int) -> str:
    strengths, weaknesses = _strengths_and_weaknesses(rows)

    lines = [
        f"Benchmark dataset: `{dataset}`",
        f"Warm-up samples/system: `{warmup}`. Measured runs/system: `{measured}`.",
        "",
        _table(rows),
        "",
        "Strengths for `pii-anon`:",
        *[f"- {item}" for item in strengths],
        "",
        "Weaknesses for `pii-anon`:",
        *[f"- {item}" for item in weaknesses],
        "",
        "This section is generated from benchmark artifacts.",
    ]
    excluded = [row for row in rows if row.get("system") != "pii-anon" and not row.get("license_gate_passed", True)]
    if excluded:
        lines.append("")
        lines.append("Qualification notes:")
        for row in excluded:
            lines.append(
                f"- `{row.get('system')}` excluded from floor math: {row.get('license_gate_reason') or 'no qualification evidence'}."
            )
    return "\n".join(lines).strip() + "\n"


def _render_statistical_summary(payload: dict[str, Any]) -> str:
    """Render statistical significance section from benchmark report."""
    stats = payload.get("statistical_tests", {})
    if not stats:
        return ""

    lines: list[str] = [
        "### Statistical Significance",
        "",
    ]

    # Sample size and MDE
    n = stats.get("sample_size", 0)
    mde = stats.get("minimum_detectable_effect", 0.0)
    if n > 0:
        lines.append(
            f"Evaluated on **{n:,}** records. "
            f"Minimum detectable effect (MDE) at α=0.05, power=0.80: **{mde:.4f}** F1 points."
        )
        lines.append("")

    # Per-system confidence intervals table
    system_cis = stats.get("system_confidence_intervals", {})
    if system_cis:
        lines.append(
            "| System | F1 | 95% CI | Samples |"
        )
        lines.append("|---|---:|---|---:|")
        for sys_name in sorted(system_cis.keys()):
            ci = system_cis[sys_name]
            lines.append(
                f"| {sys_name} | {ci['f1']:.3f} | "
                f"[{ci['f1_ci_lower']:.3f}, {ci['f1_ci_upper']:.3f}] | "
                f"{ci['samples']:,} |"
            )
        lines.append("")

    # Pairwise significance tests
    pairwise = stats.get("pairwise_tests", [])
    if pairwise:
        lines.append("Pairwise comparisons (paired bootstrap, n=10,000 resamples):")
        lines.append("")
        lines.append(
            "| Comparison | ΔF1 | p-value | Significant | Effect Size |"
        )
        lines.append("|---|---:|---:|---|---|")
        for test in sorted(pairwise, key=lambda t: t.get("p_value", 1.0)):
            sig_marker = ""
            if test.get("significant_at_01"):
                sig_marker = "p<0.01"
            elif test.get("significant_at_05"):
                sig_marker = "p<0.05"
            else:
                sig_marker = "n.s."
            lines.append(
                f"| {test['system_a']} vs {test['system_b']} | "
                f"{test['delta_f1']:+.4f} | "
                f"{test['p_value']:.4f} | "
                f"{sig_marker} | "
                f"{test['effect_size']} (d={test['cohens_d']:+.3f}) |"
            )
        lines.append("")
        lines.append(
            "*Method: paired bootstrap significance test "
            "(Berg-Kirkpatrick et al., 2012). "
            "Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*"
        )
        lines.append("")

    return "\n".join(lines)


def _render_profile_floor_summary(profiles: list[dict[str, Any]]) -> str:
    if not profiles:
        return ""
    lines = [
        "Profile floor-gate results:",
    ]
    for item in profiles:
        lines.append(
            f"- `{item.get('profile')}` ({item.get('objective')}): floor_pass={item.get('floor_pass')}"
        )
    lines.append("")
    return "\n".join(lines)


def _aggregate_rows_for_objective(
    profiles: list[dict[str, Any]],
    *,
    objective: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    relevant = [item for item in profiles if str(item.get("objective", "")).lower() == objective]
    if not relevant:
        return [], []

    grouped: dict[str, list[dict[str, Any]]] = {}
    profile_names = [str(item.get("profile", "unknown")) for item in relevant]
    for profile in relevant:
        systems = profile.get("systems", [])
        if not isinstance(systems, list):
            continue
        for row in systems:
            if not isinstance(row, dict):
                continue
            system = str(row.get("system", "unknown"))
            grouped.setdefault(system, []).append(row)

    merged_rows: list[dict[str, Any]] = []
    for system in sorted(grouped.keys()):
        rows = grouped[system]
        weight_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        latency_sum = 0.0
        docs_per_hour_sum = 0.0
        available = True
        license_gate_passed = True
        reasons: list[str] = []
        license_reasons: list[str] = []
        qualification_status: str | None = None
        license_name: str | None = None
        license_source: str | None = None
        citation_url: str | None = None
        per_entity_num: dict[str, float] = {}
        per_entity_den: dict[str, float] = {}
        dominance: dict[str, bool] = {}

        composite_sum = 0.0
        elo_max = 0.0
        entity_detected_max = 0
        entity_total_max = 0

        for row in rows:
            sample_weight = float(max(1, int(row.get("samples", 0) or 0)))
            weight_sum += sample_weight
            precision_sum += float(row.get("precision", 0.0)) * sample_weight
            recall_sum += float(row.get("recall", 0.0)) * sample_weight
            f1_sum += float(row.get("f1", 0.0)) * sample_weight
            latency_sum += float(row.get("latency_p50_ms", 0.0)) * sample_weight
            docs_per_hour_sum += float(row.get("docs_per_hour", 0.0)) * sample_weight
            composite_sum += float(row.get("composite_score", 0.0)) * sample_weight
            elo_max = max(elo_max, float(row.get("elo_rating", 0.0)))
            entity_detected_max = max(entity_detected_max, int(row.get("entity_types_detected", 0)))
            entity_total_max = max(entity_total_max, int(row.get("entity_types_total", 0)))

            available = available and bool(row.get("available", False))
            license_gate_passed = license_gate_passed and bool(row.get("license_gate_passed", True))
            if row.get("skipped_reason"):
                reasons.append(str(row.get("skipped_reason")))
            if row.get("license_gate_reason"):
                license_reasons.append(str(row.get("license_gate_reason")))

            if qualification_status is None and row.get("qualification_status") is not None:
                qualification_status = str(row.get("qualification_status"))
            if license_name is None and row.get("license_name") is not None:
                license_name = str(row.get("license_name"))
            if license_source is None and row.get("license_source") is not None:
                license_source = str(row.get("license_source"))
            if citation_url is None and row.get("citation_url") is not None:
                citation_url = str(row.get("citation_url"))

            per_entity = row.get("per_entity_recall", {})
            if isinstance(per_entity, dict):
                for entity, value in per_entity.items():
                    entity_key = str(entity)
                    per_entity_num[entity_key] = per_entity_num.get(entity_key, 0.0) + (float(value) * sample_weight)
                    per_entity_den[entity_key] = per_entity_den.get(entity_key, 0.0) + sample_weight

            dominance_map = row.get("dominance_pass_by_profile", {})
            if isinstance(dominance_map, dict):
                for profile_key, passed in dominance_map.items():
                    dominance[str(profile_key)] = bool(passed)

        per_entity_recall = {
            entity: round(per_entity_num[entity] / per_entity_den[entity], 6)
            for entity in sorted(per_entity_num.keys())
            if per_entity_den.get(entity, 0.0) > 0
        }

        merged_rows.append(
            {
                "system": system,
                "available": available,
                "skipped_reason": "; ".join(sorted(set(reasons))) if reasons else None,
                "license_gate_passed": license_gate_passed,
                "license_gate_reason": "; ".join(sorted(set(license_reasons))) if license_reasons else None,
                "qualification_status": qualification_status,
                "license_name": license_name,
                "license_source": license_source,
                "citation_url": citation_url,
                "precision": round(precision_sum / weight_sum, 6) if weight_sum else 0.0,
                "recall": round(recall_sum / weight_sum, 6) if weight_sum else 0.0,
                "f1": round(f1_sum / weight_sum, 6) if weight_sum else 0.0,
                "latency_p50_ms": round(latency_sum / weight_sum, 3) if weight_sum else 0.0,
                "docs_per_hour": round(docs_per_hour_sum / weight_sum, 2) if weight_sum else 0.0,
                "composite_score": round(composite_sum / weight_sum, 6) if weight_sum else 0.0,
                "elo_rating": round(elo_max, 2),
                "entity_types_detected": entity_detected_max,
                "entity_types_total": entity_total_max,
                "samples": int(weight_sum),
                "per_entity_recall": per_entity_recall,
                "dominance_pass_by_profile": dominance,
                "evaluation_track": "detect_only",
            }
        )

    return merged_rows, profile_names


def _inject_into_readme(readme_path: Path, body: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit(
            f"README markers not found. Expected markers `{README_START}` and `{README_END}` in {readme_path}"
        )

    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)
    updated = text[:start] + "\n\n" + body.rstrip() + "\n\n" + text[end:]
    readme_path.write_text(updated, encoding="utf-8")


# ------------------------------------------------------------------
# Cross-dataset rendering (v3 combined reports)
# ------------------------------------------------------------------

def _render_cross_dataset_summary(combined: dict[str, Any]) -> str:
    """Render a markdown section analysing performance across datasets."""
    systems = combined.get("cross_dataset_summary", {}).get("systems", [])
    datasets = combined.get("datasets_evaluated", [])
    tiers = combined.get("engine_tiers_evaluated", [])

    lines: list[str] = [
        "## Cross-Dataset Performance Summary",
        "",
        f"Engine tiers evaluated: {', '.join(f'`{t}`' for t in tiers)}",
        f"Datasets evaluated: {', '.join(f'`{d}`' for d in datasets)}",
        "",
    ]

    # Dataset descriptions
    lines.append("### Dataset Characteristics")
    lines.append("")
    for ds in datasets:
        desc = _DATASET_DESCRIPTIONS.get(ds, "No description available.")
        lines.append(f"- **{ds}**: {desc}")
    lines.append("")

    # Cross-dataset table
    lines.append("### Aggregated Results (sample-weighted average)")
    lines.append("")
    lines.append(
        "| System | Datasets | F1 Avg | Precision Avg | Recall Avg | "
        "Latency Avg (ms) | Docs/hr Avg | Best F1 On | Worst F1 On |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")

    for sys_row in sorted(systems, key=lambda r: -r.get("f1_average", 0.0)):
        lines.append(
            f"| {sys_row['system']} | {sys_row['datasets_evaluated']} | "
            f"{sys_row['f1_average']:.3f} | {sys_row['precision_average']:.3f} | "
            f"{sys_row['recall_average']:.3f} | "
            f"{sys_row['latency_p50_ms_average']:.1f} | "
            f"{sys_row['docs_per_hour_average']:.0f} | "
            f"{sys_row['best_f1_dataset']} | {sys_row['worst_f1_dataset']} |"
        )
    lines.append("")

    # Per-dataset F1 breakdown for pii-anon tiers
    pii_systems = [s for s in systems if s["system"].startswith("pii-anon")]
    if pii_systems and len(datasets) > 1:
        lines.append("### pii-anon Tier Performance by Dataset")
        lines.append("")
        header = "| Tier |" + " | ".join(f" {ds} F1 " for ds in datasets) + " |"
        sep = "|---|" + "|".join("---:" for _ in datasets) + "|"
        lines.append(header)
        lines.append(sep)
        for sys_row in pii_systems:
            per_ds = sys_row.get("per_dataset", {})
            cols = []
            for ds in datasets:
                ds_data = per_ds.get(ds, {})
                cols.append(f"{ds_data.get('f1', 0.0):.3f}")
            lines.append(f"| {sys_row['system']} | " + " | ".join(cols) + " |")
        lines.append("")

    # Interpretation section
    lines.append("### Interpretation")
    lines.append("")
    lines.append(
        "Performance variation across datasets reveals different system strengths:"
    )
    lines.append("")
    lines.append(
        "- **Strong performance on `pii_anon_benchmark_v1`** indicates robust detection "
        "across core PII types, long-context coreference, and multi-language inputs."
    )
    lines.append(
        "- **`minimal` tier** trades accuracy for speed — ideal for real-time "
        "pipelines where latency budgets are tight."
    )
    lines.append(
        "- **`standard` tier** (the `auto` default for non-speed profiles) "
        "provides the best accuracy/speed balance for most workloads."
    )
    lines.append(
        "- **`full` tier** adds low-weight engines that may improve recall on "
        "niche entity types at the cost of increased latency and potential false positives."
    )
    lines.append("")

    return "\n".join(lines)


def _render_single_dataset(payload: dict[str, Any], *, require_floor_pass: bool) -> str:
    """Render a single-dataset benchmark report (v2 or v3 per-dataset)."""
    track_rows = payload.get("evaluation_tracks", {}).get("detect_only")
    if isinstance(track_rows, list) and track_rows:
        rows = [row for row in track_rows if str(row.get("evaluation_track", "")) == "detect_only"]
    else:
        rows = [
            row
            for row in list(payload.get("systems", []))
            if str(row.get("evaluation_track", "detect_only")) == "detect_only"
        ]

    floor_pass = bool(payload.get("floor_pass", True))
    qualification_gate_pass = bool(payload.get("qualification_gate_pass", payload.get("mit_gate_pass", True)))
    if require_floor_pass and not floor_pass:
        raise SystemExit("benchmark floor gate did not pass; refusing to render publishable summary")
    if require_floor_pass and not qualification_gate_pass:
        raise SystemExit("benchmark qualification gate did not pass; refusing to render publishable summary")
    if require_floor_pass and not bool(payload.get("all_competitors_available", False)):
        raise SystemExit("not all configured competitors were available; refusing to render publishable summary")
    if require_floor_pass and not bool(payload.get("required_profiles_passed", False)):
        raise SystemExit("required profile floor gate did not pass; refusing to render publishable summary")
    if require_floor_pass and not bool(payload.get("run_metadata", {}).get("canonical_claim_run", False)):
        raise SystemExit("benchmark run is not canonical publish-grade; refusing to render publishable summary")
    if require_floor_pass and str(payload.get("dataset_source", "auto")) != "package-only":
        raise SystemExit("benchmark dataset source is not package-only; refusing to render publishable summary")

    dataset_name = str(payload.get("dataset", "unknown"))
    warmup = int(payload.get("warmup_samples", 0))
    measured = int(payload.get("measured_runs", 0))
    profiles_raw = payload.get("profiles", [])

    body_sections: list[str] = []
    if isinstance(profiles_raw, list) and profiles_raw:
        objective_blocks: list[str] = []
        for objective in ("accuracy", "balanced", "speed"):
            objective_rows, objective_profiles = _aggregate_rows_for_objective(
                [item for item in profiles_raw if isinstance(item, dict)],
                objective=objective,
            )
            if not objective_rows:
                continue
            objective_header = (
                f"## {objective.title()} Objective (profiles: {', '.join(objective_profiles)})\n\n"
            )
            objective_blocks.append(
                objective_header
                + _render(
                    rows=objective_rows,
                    dataset=dataset_name,
                    warmup=warmup,
                    measured=measured,
                )
            )
        if objective_blocks:
            body_sections.extend(objective_blocks)

    if not body_sections:
        body_sections.append(
            _render(
                rows=rows,
                dataset=dataset_name,
                warmup=warmup,
                measured=measured,
            )
        )

    body = "\n".join(item.strip() for item in body_sections if item.strip()).strip() + "\n"
    profile_rows = payload.get("profile_results", payload.get("profiles", []))
    profile_summary = _render_profile_floor_summary(list(profile_rows))
    if profile_summary:
        body = body + "\n" + profile_summary

    # Append statistical significance section if available.
    stat_summary = _render_statistical_summary(payload)
    if stat_summary:
        body = body + "\n" + stat_summary

    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark summary markdown")
    parser.add_argument(
        "--input-json",
        action="append",
        default=[],
        dest="input_jsons",
        help=(
            "Benchmark report JSON(s). "
            "Pass once for single-dataset mode. "
            "Pass multiple times: the first is the combined report, "
            "subsequent are per-dataset reports."
        ),
    )
    parser.add_argument("--output-markdown", default="benchmark-summary.md")
    parser.add_argument("--update-readme", default="")
    parser.add_argument("--require-floor-pass", action="store_true")
    args = parser.parse_args()

    if not args.input_jsons:
        args.input_jsons = ["benchmark-results.json"]

    primary_path = Path(args.input_jsons[0])
    primary_payload = json.loads(primary_path.read_text(encoding="utf-8"))
    _validate_payload(primary_payload)

    schema_version = str(primary_payload.get("report_schema_version", ""))
    is_combined = schema_version == "2026-02-19.v3" and "by_dataset" in primary_payload

    body_parts: list[str] = []

    if is_combined:
        # Render cross-dataset summary from combined report.
        body_parts.append(_render_cross_dataset_summary(primary_payload))

        # Render each per-dataset report that was passed as additional --input-json.
        by_dataset = primary_payload.get("by_dataset", {})
        additional_jsons = args.input_jsons[1:]

        if additional_jsons:
            # Use explicitly provided per-dataset JSON files.
            for json_path_str in additional_jsons:
                ds_payload = json.loads(Path(json_path_str).read_text(encoding="utf-8"))
                _validate_payload(ds_payload)
                ds_name = str(ds_payload.get("dataset", "unknown"))
                body_parts.append(f"## Dataset: `{ds_name}`\n")
                body_parts.append(
                    _render_single_dataset(ds_payload, require_floor_pass=False)
                )
        elif by_dataset:
            # Fall back to embedded per-dataset data from combined report.
            for ds_name, ds_payload in by_dataset.items():
                body_parts.append(f"## Dataset: `{ds_name}`\n")
                body_parts.append(
                    _render_single_dataset(ds_payload, require_floor_pass=False)
                )
    else:
        # Single-dataset mode — original behavior.
        body_parts.append(
            _render_single_dataset(primary_payload, require_floor_pass=args.require_floor_pass)
        )

    body = "\n".join(part.strip() for part in body_parts if part.strip()).strip() + "\n"

    output = Path(args.output_markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(f"wrote {output}")

    if args.update_readme:
        _inject_into_readme(Path(args.update_readme), body)
        print(f"updated README benchmark section in {args.update_readme}")


if __name__ == "__main__":
    main()
