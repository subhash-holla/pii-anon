from __future__ import annotations

import json
import subprocess
import sys


def test_render_benchmark_summary_contains_strengths_and_weaknesses(tmp_path) -> None:
    payload = {
        "report_schema_version": "2026-02-19.v3",
        "dataset": "pii_anon_benchmark_v1",
        "dataset_source": "package-only",
        "warmup_samples": 100,
        "measured_runs": 3,
        "floor_pass": True,
        "qualification_gate_pass": True,
        "expected_competitors": ["presidio", "scrubadub", "gliner"],
        "available_competitors": ["presidio", "scrubadub", "gliner"],
        "unavailable_competitors": {},
        "all_competitors_available": True,
        "run_metadata": {"canonical_claim_run": True},
        "failed_profiles": [],
        "required_profiles": ["short_chat"],
        "required_profiles_passed": True,
        "profile_results": [],
        "profiles": [],
        "systems": [
            {
                "system": "pii-anon",
                "available": True,
                "skipped_reason": None,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "precision": 0.91,
                "recall": 0.90,
                "f1": 0.905,
                "latency_p50_ms": 20.0,
                "docs_per_hour": 12000.0,
                "per_entity_recall": {},
                "samples": 1500,
            },
            {
                "system": "presidio",
                "available": True,
                "skipped_reason": None,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "precision": 0.92,
                "recall": 0.88,
                "f1": 0.899,
                "latency_p50_ms": 25.0,
                "docs_per_hour": 10000.0,
                "per_entity_recall": {},
                "samples": 1500,
            },
            {
                "system": "scrubadub",
                "available": True,
                "skipped_reason": None,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "precision": 0.80,
                "recall": 0.78,
                "f1": 0.79,
                "latency_p50_ms": 15.0,
                "docs_per_hour": 15000.0,
                "per_entity_recall": {},
                "samples": 1500,
            },
        ],
    }

    input_json = tmp_path / "benchmark-results.json"
    output_md = tmp_path / "benchmark-summary.md"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_benchmark_summary.py",
            "--input-json",
            str(input_json),
            "--output-markdown",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    summary = output_md.read_text(encoding="utf-8")
    assert "Strengths for `pii-anon`:" in summary
    assert "Weaknesses for `pii-anon`:" in summary
    assert "docs_per_hour: more than 10% below best" in summary


def test_render_benchmark_summary_requires_floor_pass(tmp_path) -> None:
    payload = {
        "report_schema_version": "2026-02-19.v3",
        "dataset": "pii_anon_benchmark_v1",
        "dataset_source": "package-only",
        "warmup_samples": 1,
        "measured_runs": 1,
        "floor_pass": False,
        "qualification_gate_pass": True,
        "expected_competitors": ["presidio"],
        "available_competitors": ["presidio"],
        "unavailable_competitors": {},
        "all_competitors_available": True,
        "run_metadata": {"canonical_claim_run": False},
        "failed_profiles": ["short_chat"],
        "required_profiles": ["short_chat"],
        "required_profiles_passed": False,
        "profile_results": [],
        "systems": [],
        "profiles": [],
    }
    input_json = tmp_path / "benchmark-results.json"
    output_md = tmp_path / "benchmark-summary.md"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_benchmark_summary.py",
            "--input-json",
            str(input_json),
            "--output-markdown",
            str(output_md),
            "--require-floor-pass",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_render_benchmark_summary_requires_mit_gate_pass(tmp_path) -> None:
    payload = {
        "report_schema_version": "2026-02-19.v3",
        "dataset": "pii_anon_benchmark_v1",
        "dataset_source": "package-only",
        "warmup_samples": 1,
        "measured_runs": 1,
        "floor_pass": True,
        "qualification_gate_pass": False,
        "expected_competitors": ["presidio"],
        "available_competitors": [],
        "unavailable_competitors": {"presidio": "unavailable"},
        "all_competitors_available": False,
        "run_metadata": {"canonical_claim_run": False},
        "failed_profiles": [],
        "required_profiles": ["short_chat"],
        "required_profiles_passed": False,
        "profile_results": [],
        "systems": [],
        "profiles": [],
    }
    input_json = tmp_path / "benchmark-results.json"
    output_md = tmp_path / "benchmark-summary.md"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_benchmark_summary.py",
            "--input-json",
            str(input_json),
            "--output-markdown",
            str(output_md),
            "--require-floor-pass",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_render_benchmark_summary_rejects_legacy_schema(tmp_path) -> None:
    payload = {
        "dataset": "pii_anon_benchmark_v1",
        "warmup_samples": 1,
        "measured_runs": 1,
        "floor_pass": True,
        "qualification_gate_pass": True,
        "systems": [],
        "profiles": [],
    }
    input_json = tmp_path / "benchmark-results.json"
    output_md = tmp_path / "benchmark-summary.md"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_benchmark_summary.py",
            "--input-json",
            str(input_json),
            "--output-markdown",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_render_benchmark_summary_uses_objective_specific_profile_rows(tmp_path) -> None:
    payload = {
        "report_schema_version": "2026-02-19.v3",
        "dataset": "pii_anon_benchmark_v1",
        "dataset_source": "package-only",
        "warmup_samples": 100,
        "measured_runs": 3,
        "floor_pass": True,
        "qualification_gate_pass": True,
        "expected_competitors": ["presidio"],
        "available_competitors": ["presidio"],
        "unavailable_competitors": {},
        "all_competitors_available": True,
        "run_metadata": {"canonical_claim_run": True},
        "failed_profiles": [],
        "required_profiles": ["short_chat", "long_document"],
        "required_profiles_passed": True,
        "profile_results": [],
        "systems": [
            {
                "system": "pii-anon",
                "available": True,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "precision": 0.997,
                "recall": 0.261,
                "f1": 0.414,
                "latency_p50_ms": 0.009,
                "docs_per_hour": 1000.0,
                "per_entity_recall": {},
                "samples": 1000,
            },
            {
                "system": "presidio",
                "available": True,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "precision": 0.35,
                "recall": 0.51,
                "f1": 0.417,
                "latency_p50_ms": 9.6,
                "docs_per_hour": 500.0,
                "per_entity_recall": {},
                "samples": 1000,
            },
        ],
        "profiles": [
            {
                "profile": "short_chat",
                "objective": "speed",
                "floor_pass": True,
                "systems": [
                    {
                        "system": "pii-anon",
                        "available": True,
                        "license_gate_passed": True,
                        "evaluation_track": "detect_only",
                        "precision": 0.997,
                        "recall": 0.261,
                        "f1": 0.414,
                        "latency_p50_ms": 0.009,
                        "docs_per_hour": 1000.0,
                        "per_entity_recall": {},
                        "samples": 1000,
                    },
                    {
                        "system": "presidio",
                        "available": True,
                        "license_gate_passed": True,
                        "evaluation_track": "detect_only",
                        "precision": 0.35,
                        "recall": 0.51,
                        "f1": 0.417,
                        "latency_p50_ms": 9.6,
                        "docs_per_hour": 500.0,
                        "per_entity_recall": {},
                        "samples": 1000,
                    },
                ],
            },
            {
                "profile": "long_document",
                "objective": "accuracy",
                "floor_pass": True,
                "systems": [
                    {
                        "system": "pii-anon",
                        "available": True,
                        "license_gate_passed": True,
                        "evaluation_track": "detect_only",
                        "precision": 0.758,
                        "recall": 0.544,
                        "f1": 0.633,
                        "latency_p50_ms": 1.200,
                        "docs_per_hour": 120.0,
                        "per_entity_recall": {},
                        "samples": 1000,
                    },
                    {
                        "system": "presidio",
                        "available": True,
                        "license_gate_passed": True,
                        "evaluation_track": "detect_only",
                        "precision": 0.700,
                        "recall": 0.560,
                        "f1": 0.622,
                        "latency_p50_ms": 2.100,
                        "docs_per_hour": 90.0,
                        "per_entity_recall": {},
                        "samples": 1000,
                    },
                ],
            },
        ],
    }

    input_json = tmp_path / "benchmark-results.json"
    output_md = tmp_path / "benchmark-summary.md"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_benchmark_summary.py",
            "--input-json",
            str(input_json),
            "--output-markdown",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    summary = output_md.read_text(encoding="utf-8")
    assert "## Accuracy Objective" in summary
    assert "## Speed Objective" in summary
    # Table columns: System | Status | Composite | F1 | 95% CI | Precision | Recall | ...
    assert "| pii-anon | available | 0.0000 | 0.633 |" in summary
    assert "| 0.758 | 0.544 |" in summary
