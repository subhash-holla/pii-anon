"""Tests for multi-dataset aggregation, combined report structure, and suite resilience."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _make_single_dataset_payload(
    dataset: str,
    *,
    systems: list[dict] | None = None,
) -> dict:
    """Build a minimal single-dataset benchmark payload."""
    if systems is None:
        systems = [
            {
                "system": "pii-anon",
                "available": True,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "f1": 0.9,
                "precision": 0.85,
                "recall": 0.95,
                "latency_p50_ms": 10.0,
                "docs_per_hour": 2000.0,
                "composite_score": 0.88,
                "elo_rating": 1050,
                "samples": 100,
            },
            {
                "system": "presidio",
                "available": True,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "f1": 0.8,
                "precision": 0.75,
                "recall": 0.85,
                "latency_p50_ms": 20.0,
                "docs_per_hour": 1500.0,
                "composite_score": 0.78,
                "elo_rating": 1000,
                "samples": 100,
            },
        ]
    return {
        "report_schema_version": "2026-02-19.v3",
        "dataset": dataset,
        "dataset_source": "auto",
        "warmup_samples": 10,
        "measured_runs": 1,
        "floor_pass": True,
        "qualification_gate_pass": True,
        "expected_competitors": ["presidio"],
        "available_competitors": ["presidio"],
        "unavailable_competitors": [],
        "all_competitors_available": True,
        "run_metadata": {"canonical_claim_run": False},
        "failed_profiles": [],
        "required_profiles": [],
        "required_profiles_passed": True,
        "profile_results": [],
        "systems": systems,
        "profiles": [],
    }


class TestAggregateDatasetReports:
    """Test the _aggregate_dataset_reports function from run_publish_grade_suite."""

    @staticmethod
    def _aggregate(per_dataset_jsons: dict[str, Path], engine_tiers: list[str]) -> dict:
        """Import and call _aggregate_dataset_reports."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_publish_grade_suite",
            Path(__file__).resolve().parents[1] / "scripts" / "run_publish_grade_suite.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod._aggregate_dataset_reports(per_dataset_jsons, engine_tiers_evaluated=engine_tiers)

    def test_combined_report_structure(self, tmp_path: Path) -> None:
        ds1 = tmp_path / "ds1.json"
        ds2 = tmp_path / "ds2.json"
        ds1.write_text(json.dumps(_make_single_dataset_payload("pii_anon_benchmark_v1")))
        ds2.write_text(json.dumps(_make_single_dataset_payload("other_dataset")))

        combined = self._aggregate(
            {"pii_anon_benchmark_v1": ds1, "other_dataset": ds2},
            engine_tiers=["auto", "minimal"],
        )

        assert combined["report_schema_version"] == "2026-02-19.v3"
        assert set(combined["datasets_evaluated"]) == {"pii_anon_benchmark_v1", "other_dataset"}
        assert combined["engine_tiers_evaluated"] == ["auto", "minimal"]
        assert "by_dataset" in combined
        assert "pii_anon_benchmark_v1" in combined["by_dataset"]
        assert "other_dataset" in combined["by_dataset"]
        assert "cross_dataset_summary" in combined
        assert "systems" in combined["cross_dataset_summary"]

    def test_cross_dataset_metrics(self, tmp_path: Path) -> None:
        ds1_systems = [
            {
                "system": "pii-anon",
                "available": True,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "f1": 0.9,
                "precision": 0.85,
                "recall": 0.95,
                "latency_p50_ms": 10.0,
                "docs_per_hour": 2000.0,
                "composite_score": 0.88,
                "elo_rating": 1050,
                "samples": 100,
            },
        ]
        ds2_systems = [
            {
                "system": "pii-anon",
                "available": True,
                "license_gate_passed": True,
                "evaluation_track": "detect_only",
                "f1": 0.8,
                "precision": 0.75,
                "recall": 0.85,
                "latency_p50_ms": 15.0,
                "docs_per_hour": 1800.0,
                "composite_score": 0.78,
                "elo_rating": 1020,
                "samples": 100,
            },
        ]
        ds1 = tmp_path / "ds1.json"
        ds2 = tmp_path / "ds2.json"
        ds1.write_text(json.dumps(_make_single_dataset_payload("ds_a", systems=ds1_systems)))
        ds2.write_text(json.dumps(_make_single_dataset_payload("ds_b", systems=ds2_systems)))

        combined = self._aggregate({"ds_a": ds1, "ds_b": ds2}, engine_tiers=["auto"])
        systems = combined["cross_dataset_summary"]["systems"]
        assert len(systems) == 1
        pii = systems[0]
        assert pii["system"] == "pii-anon"
        assert pii["datasets_evaluated"] == 2
        # Equal sample weights → simple average.
        assert abs(pii["f1_average"] - 0.85) < 0.01
        assert pii["best_f1_dataset"] == "ds_a"
        assert pii["worst_f1_dataset"] == "ds_b"
        assert "per_dataset" in pii
        assert set(pii["per_dataset"].keys()) == {"ds_a", "ds_b"}


class TestRenderBenchmarkSummaryV3:
    """Test v3 combined report rendering from render_benchmark_summary.py."""

    @staticmethod
    def _load_render_module():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "render_benchmark_summary",
            Path(__file__).resolve().parents[1] / "scripts" / "render_benchmark_summary.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_validate_v3_schema(self) -> None:
        mod = self._load_render_module()
        # Valid v3 combined payload should not raise.
        mod._validate_payload({
            "report_schema_version": "2026-02-19.v3",
            "by_dataset": {"ds1": {}},
            "datasets_evaluated": ["ds1"],
        })

    def test_validate_rejects_unknown_schema(self) -> None:
        mod = self._load_render_module()
        with pytest.raises(SystemExit, match="unsupported"):
            mod._validate_payload({"report_schema_version": "9999-01-01.v99"})

    def test_render_cross_dataset_analysis(self) -> None:
        mod = self._load_render_module()
        combined = {
            "datasets_evaluated": ["pii_anon_benchmark_v1", "other_dataset"],
            "engine_tiers_evaluated": ["auto", "minimal"],
            "cross_dataset_summary": {
                "systems": [
                    {
                        "system": "pii-anon",
                        "datasets_evaluated": 2,
                        "f1_average": 0.85,
                        "precision_average": 0.80,
                        "recall_average": 0.90,
                        "latency_p50_ms_average": 12.0,
                        "docs_per_hour_average": 1900.0,
                        "best_f1_dataset": "pii_anon_benchmark_v1",
                        "worst_f1_dataset": "other_dataset",
                        "per_dataset": {
                            "pii_anon_benchmark_v1": {"f1": 0.9, "samples": 100},
                            "other_dataset": {"f1": 0.8, "samples": 80},
                        },
                    },
                    {
                        "system": "pii-anon-minimal",
                        "datasets_evaluated": 2,
                        "f1_average": 0.75,
                        "precision_average": 0.70,
                        "recall_average": 0.80,
                        "latency_p50_ms_average": 5.0,
                        "docs_per_hour_average": 4000.0,
                        "best_f1_dataset": "pii_anon_benchmark_v1",
                        "worst_f1_dataset": "other_dataset",
                        "per_dataset": {
                            "pii_anon_benchmark_v1": {"f1": 0.8, "samples": 100},
                            "other_dataset": {"f1": 0.7, "samples": 80},
                        },
                    },
                ],
            },
        }
        md = mod._render_cross_dataset_summary(combined)

        # Must contain expected sections.
        assert "## Cross-Dataset Performance Summary" in md
        assert "### Dataset Characteristics" in md
        assert "### Aggregated Results" in md
        assert "### pii-anon Tier Performance by Dataset" in md
        assert "### Interpretation" in md
        assert "pii-anon" in md
        assert "pii-anon-minimal" in md
        # Check interpretation mentions key patterns.
        assert "`pii_anon_benchmark_v1`" in md
        assert "`other_dataset`" in md


class TestRunSoftResilience:
    """Verify _run_soft does not abort the process on failure."""

    @staticmethod
    def _load_suite_module():
        spec = importlib.util.spec_from_file_location(
            "run_publish_grade_suite",
            Path(__file__).resolve().parents[1] / "scripts" / "run_publish_grade_suite.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_run_soft_returns_success_on_zero_exit(self, tmp_path: Path) -> None:
        mod = self._load_suite_module()
        ok, msg = mod._run_soft(
            ["python3", "-c", "pass"],
            cwd=tmp_path,
            label="noop",
        )
        assert ok is True
        assert msg == ""

    def test_run_soft_returns_failure_without_raising(self, tmp_path: Path) -> None:
        mod = self._load_suite_module()
        ok, msg = mod._run_soft(
            ["python3", "-c", "raise SystemExit(42)"],
            cwd=tmp_path,
            label="failing step",
        )
        assert ok is False
        assert "failing step" in msg
        assert "42" in msg

    def test_run_hard_raises_on_failure(self, tmp_path: Path) -> None:
        mod = self._load_suite_module()
        with pytest.raises(SystemExit):
            mod._run(
                ["python3", "-c", "raise SystemExit(1)"],
                cwd=tmp_path,
            )

    def test_aggregation_skips_missing_datasets(self, tmp_path: Path) -> None:
        """If one dataset has no artifacts, aggregation should work with
        just the successful datasets."""
        mod = self._load_suite_module()
        ds1 = tmp_path / "ds1.json"
        ds1.write_text(json.dumps(_make_single_dataset_payload("pii_anon_benchmark_v1")))

        # Only one dataset — should produce a valid combined report.
        combined = mod._aggregate_dataset_reports(
            {"pii_anon_benchmark_v1": ds1},
            engine_tiers_evaluated=["auto"],
        )
        assert combined["report_schema_version"] == "2026-02-19.v3"
        assert combined["datasets_evaluated"] == ["pii_anon_benchmark_v1"]
        assert len(combined["cross_dataset_summary"]["systems"]) > 0
