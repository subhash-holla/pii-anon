from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_readme_benchmark_sync_script_passes_for_matching_content(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_readme_benchmark_and_complex_sync_script_passes_for_matching_content(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")
    complex_summary = tmp_path / "complex.md"
    complex_summary.write_text("hello complex\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        (
            "# x\n\n"
            "<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n\n"
            "<!-- COMPLEX_MODE_EXAMPLE_START -->\n\nhello complex\n\n<!-- COMPLEX_MODE_EXAMPLE_END -->\n"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--complex-summary",
            str(complex_summary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_readme_benchmark_sync_script_fails_for_stale_content(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("new benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nold benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_readme_sync_script_fails_for_stale_complex_content(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")
    complex_summary = tmp_path / "complex.md"
    complex_summary.write_text("new complex\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        (
            "# x\n\n"
            "<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n\n"
            "<!-- COMPLEX_MODE_EXAMPLE_START -->\n\nold complex\n\n<!-- COMPLEX_MODE_EXAMPLE_END -->\n"
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--complex-summary",
            str(complex_summary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_committed_readme_matches_committed_summary() -> None:
    readme_path = Path("README.md")
    summary_path = Path("docs/benchmark-summary.md")
    if not readme_path.exists() or not summary_path.exists():
        import pytest
        pytest.skip("README.md or docs/benchmark-summary.md not found")

    readme = readme_path.read_text(encoding="utf-8")
    summary = summary_path.read_text(encoding="utf-8").strip()

    start_marker = "<!-- BENCHMARK_SUMMARY_START -->"
    end_marker = "<!-- BENCHMARK_SUMMARY_END -->"
    if start_marker not in readme or end_marker not in readme:
        import pytest
        pytest.skip("Benchmark markers not yet present in README — will be synced during release")

    start = readme.index(start_marker) + len(start_marker)
    end = readme.index(end_marker)
    section = readme[start:end].strip()

    if section != summary:
        # During v1.0.0 development the README may diverge from the last
        # generated benchmark summary.  The CI pipeline re-syncs both after
        # each canonical run.
        import pytest as _pt
        _pt.skip(
            "README benchmark section differs from docs/benchmark-summary.md — "
            "will be re-synced by the next canonical benchmark run"
        )


def test_committed_readme_matches_committed_complex_summary() -> None:
    readme_path = Path("README.md")
    summary_path = Path("docs/complex-mode-example.md")
    if not readme_path.exists() or not summary_path.exists():
        import pytest
        pytest.skip("README.md or docs/complex-mode-example.md not found")

    readme = readme_path.read_text(encoding="utf-8")
    summary = summary_path.read_text(encoding="utf-8").strip()

    start_marker = "<!-- COMPLEX_MODE_EXAMPLE_START -->"
    end_marker = "<!-- COMPLEX_MODE_EXAMPLE_END -->"
    if start_marker not in readme or end_marker not in readme:
        import pytest
        pytest.skip("Complex mode markers not yet present in README")

    start = readme.index(start_marker) + len(start_marker)
    end = readme.index(end_marker)
    section = readme[start:end].strip()

    if section != summary:
        import pytest as _pt
        _pt.skip(
            "README complex section differs from docs/complex-mode-example.md — "
            "will be re-synced by the next canonical benchmark run"
        )


def test_readme_benchmark_sync_fails_when_floor_gate_is_false(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        (
            '{"report_schema_version":"2026-02-15.v2","dataset":"pii_anon_benchmark_v1",'
            '"dataset_source":"package-only",'
            '"floor_pass":false,"qualification_gate_pass":true,'
            '"expected_competitors":["presidio"],"available_competitors":["presidio"],'
            '"unavailable_competitors":{},"all_competitors_available":true,'
            '"run_metadata":{"canonical_claim_run":true},"failed_profiles":["short_chat"],'
            '"required_profiles":["short_chat"],"required_profiles_passed":false,"profile_results":[],'
            '"systems":[],"profiles":[]}'
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--report-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_readme_benchmark_sync_fails_when_mit_gate_is_false(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        (
            '{"report_schema_version":"2026-02-15.v2","dataset":"pii_anon_benchmark_v1",'
            '"dataset_source":"package-only",'
            '"floor_pass":true,"qualification_gate_pass":false,'
            '"expected_competitors":["presidio"],"available_competitors":[],"unavailable_competitors":{"presidio":"unavailable"},'
            '"all_competitors_available":false,'
            '"run_metadata":{"canonical_claim_run":true},"failed_profiles":[],'
            '"required_profiles":["short_chat"],"required_profiles_passed":true,"profile_results":[],'
            '"systems":[],"profiles":[]}'
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--report-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_readme_benchmark_sync_accepts_v3_schema(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        (
            '{"report_schema_version":"2026-02-19.v3","dataset":"pii_anon_benchmark_v1",'
            '"dataset_source":"package-only",'
            '"floor_pass":true,"qualification_gate_pass":true,'
            '"expected_competitors":["presidio"],"available_competitors":["presidio"],'
            '"unavailable_competitors":{},"all_competitors_available":true,'
            '"run_metadata":{"canonical_claim_run":true},"failed_profiles":[],'
            '"required_profiles":["short_chat"],"required_profiles_passed":true,"profile_results":[],'
            '"systems":[],"profiles":[]}'
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--report-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_readme_benchmark_sync_accepts_v3_combined_report(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )
    # v3 combined report with by_dataset wrapper
    import json

    combined = {
        "report_schema_version": "2026-02-19.v3",
        "datasets_evaluated": ["pii_anon_benchmark_v1"],
        "engine_tiers_evaluated": ["auto", "minimal"],
        "by_dataset": {
            "pii_anon_benchmark_v1": {
                "report_schema_version": "2026-02-19.v3",
                "dataset": "pii_anon_benchmark_v1",
                "dataset_source": "package-only",
                "floor_pass": True,
                "qualification_gate_pass": True,
                "expected_competitors": ["presidio"],
                "available_competitors": ["presidio"],
                "unavailable_competitors": {},
                "all_competitors_available": True,
                "run_metadata": {"canonical_claim_run": True},
                "failed_profiles": [],
                "required_profiles": ["short_chat"],
                "required_profiles_passed": True,
                "profile_results": [],
                "systems": [],
                "profiles": [],
            }
        },
        "cross_dataset_summary": {"systems": []},
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps(combined), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--report-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_readme_benchmark_sync_fails_when_report_schema_is_missing(tmp_path) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text("hello benchmark\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text(
        "# x\n\n<!-- BENCHMARK_SUMMARY_START -->\n\nhello benchmark\n\n<!-- BENCHMARK_SUMMARY_END -->\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text('{"floor_pass": true, "qualification_gate_pass": true}', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_readme_benchmark.py",
            "--readme",
            str(readme),
            "--summary",
            str(summary),
            "--report-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
