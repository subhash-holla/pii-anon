#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

README_START = "<!-- BENCHMARK_SUMMARY_START -->"
README_END = "<!-- BENCHMARK_SUMMARY_END -->"
COMPLEX_START = "<!-- COMPLEX_MODE_EXAMPLE_START -->"
COMPLEX_END = "<!-- COMPLEX_MODE_EXAMPLE_END -->"
MARKETING_START = "<!-- MARKETING_NARRATIVE_START -->"
MARKETING_END = "<!-- MARKETING_NARRATIVE_END -->"
SUPPORTED_REPORT_SCHEMA = {"2026-02-15.v2", "2026-02-19.v3"}


def _validate_report_payload(payload: dict[str, object]) -> None:
    version = str(payload.get("report_schema_version", ""))
    if version not in SUPPORTED_REPORT_SCHEMA:
        raise SystemExit(
            f"unsupported benchmark report schema `{version}`. "
            "Regenerate benchmark artifacts before updating README claims."
        )

    # v3 combined reports wrap per-dataset reports under ``by_dataset``.
    # Validate the first inner dataset report instead.
    if "by_dataset" in payload:
        by_dataset = payload["by_dataset"]
        if not isinstance(by_dataset, dict) or not by_dataset:
            raise SystemExit("v3 combined report has empty `by_dataset`.")
        first_ds = next(iter(by_dataset.values()))
        if not isinstance(first_ds, dict):
            raise SystemExit("v3 combined report: first dataset entry is not a dict.")
        _validate_single_dataset_payload(first_ds)
        return

    _validate_single_dataset_payload(payload)


def _validate_single_dataset_payload(payload: dict[str, object]) -> None:
    required = {
        "report_schema_version",
        "dataset",
        "dataset_source",
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
    missing = sorted(required.difference(payload))
    if missing:
        raise SystemExit(
            f"benchmark report missing required fields: {', '.join(missing)}. "
            "Regenerate benchmark artifacts before updating README claims."
        )


def _extract_publish_claims(payload: dict[str, object]) -> dict[str, object]:
    """Return the payload to check publish-claim gates against.

    For v3 combined reports, use the first per-dataset entry so the
    per-field gate checks work unchanged.
    """
    if "by_dataset" in payload:
        by_dataset = payload["by_dataset"]
        if isinstance(by_dataset, dict) and by_dataset:
            return dict(next(iter(by_dataset.values())))  # type: ignore[arg-type]
    return payload


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text or end_marker not in text:
        raise SystemExit(f"README markers missing: `{start_marker}` and `{end_marker}` are required.")
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    return text[start:end].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate README benchmark section matches generated summary")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--summary", default="benchmark-summary.md")
    parser.add_argument("--complex-summary", default="")
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    if args.report_json:
        report_path = Path(args.report_json)
        if not report_path.exists():
            raise SystemExit(f"report JSON not found: {report_path}")
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"failed to parse report JSON `{report_path}`: {exc}") from exc
        _validate_report_payload(payload)

        claims = _extract_publish_claims(payload)
        if not bool(claims.get("floor_pass", True)):
            raise SystemExit("README benchmark claims are blocked because floor gate is failing.")
        if not bool(claims.get("required_profiles_passed", False)):
            raise SystemExit("README benchmark claims are blocked because required profile floors are failing.")
        qualification_ok = bool(claims.get("qualification_gate_pass", claims.get("mit_gate_pass", True)))
        if not qualification_ok:
            raise SystemExit("README benchmark claims are blocked because qualification gate is failing.")
        if not bool(claims.get("all_competitors_available", False)):
            raise SystemExit("README benchmark claims are blocked because one or more competitors are unavailable.")
        canonical_claim_run = bool(claims.get("run_metadata", {}).get("canonical_claim_run", False))
        if not canonical_claim_run:
            raise SystemExit("README benchmark claims are blocked because benchmark run is not canonical publish-grade.")
        if str(claims.get("dataset_source", "auto")) != "package-only":
            raise SystemExit("README benchmark claims are blocked because dataset source is not package-only.")

    readme_path = Path(args.readme)
    summary_path = Path(args.summary)
    if not readme_path.exists():
        raise SystemExit(f"README not found: {readme_path}")
    if not summary_path.exists():
        raise SystemExit(f"benchmark summary not found: {summary_path}")
    readme_text = readme_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8").strip()

    section = _extract_section(readme_text, README_START, README_END)
    if section != summary_text:
        raise SystemExit(
            "README benchmark section is stale. Regenerate benchmark summary and update README section before merging."
        )

    if args.complex_summary:
        complex_path = Path(args.complex_summary)
        if not complex_path.exists():
            raise SystemExit(f"complex summary not found: {complex_path}")
        complex_text = complex_path.read_text(encoding="utf-8").strip()
        complex_section = _extract_section(readme_text, COMPLEX_START, COMPLEX_END)
        if complex_section != complex_text:
            raise SystemExit(
                "README complex mode section is stale. Regenerate complex example and update README section before merging."
            )

    # Validate marketing narrative markers if present.
    if MARKETING_START in readme_text and MARKETING_END in readme_text:
        marketing_section = _extract_section(readme_text, MARKETING_START, MARKETING_END)
        if not marketing_section.strip():
            raise SystemExit(
                "README marketing narrative section is empty. "
                "Run render_marketing_narrative.py to populate it."
            )

    print("README benchmark section matches generated summary")


if __name__ == "__main__":
    main()
