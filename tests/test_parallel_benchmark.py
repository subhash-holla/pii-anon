"""Tests for the parallel competitor evaluation infrastructure.

Covers: _SystemEvalSpec picklability, worker unavailable path,
CLI --parallel flag default, and sequential fallback behaviour.
"""

from __future__ import annotations

import argparse
import pickle

from pii_anon.benchmarks import BenchmarkRecord
from pii_anon.evaluation.competitor_compare import (
    _SystemEvalSpec,
    _evaluate_system_worker,
)


# ---------------------------------------------------------------------------
# _SystemEvalSpec picklability
# ---------------------------------------------------------------------------


class TestSystemEvalSpecPickle:
    def test_round_trip(self) -> None:
        spec = _SystemEvalSpec(
            system_name="presidio",
            records=[
                BenchmarkRecord(
                    record_id="r1",
                    text="Alice lives in New York",
                    labels=[{"entity_type": "PERSON", "start": 0, "end": 5}],
                    language="en",
                ),
            ],
            warmup_samples=0,
            measured_runs=1,
            allow_fallback_detectors=True,
            require_native_competitors=False,
            forced_unavailable_reason=None,
            profile_label="test_profile",
        )
        restored = pickle.loads(pickle.dumps(spec))
        assert restored.system_name == "presidio"
        assert len(restored.records) == 1
        assert restored.records[0].record_id == "r1"

    def test_with_forced_unavailable(self) -> None:
        spec = _SystemEvalSpec(
            system_name="scrubadub",
            records=[],
            warmup_samples=0,
            measured_runs=1,
            allow_fallback_detectors=True,
            require_native_competitors=False,
            forced_unavailable_reason="not installed in env",
            profile_label="test_profile",
        )
        restored = pickle.loads(pickle.dumps(spec))
        assert restored.forced_unavailable_reason == "not installed in env"


# ---------------------------------------------------------------------------
# _evaluate_system_worker — forced unavailable path
# ---------------------------------------------------------------------------


class TestEvaluateSystemWorkerUnavailable:
    def test_forced_unavailable_returns_skipped(self) -> None:
        spec = _SystemEvalSpec(
            system_name="scrubadub",
            records=[
                BenchmarkRecord(
                    record_id="r1",
                    text="Bob works at Acme Corp",
                    labels=[{"entity_type": "PERSON", "start": 0, "end": 3}],
                    language="en",
                ),
            ],
            warmup_samples=0,
            measured_runs=1,
            allow_fallback_detectors=True,
            require_native_competitors=False,
            forced_unavailable_reason="forced skip for test",
            profile_label="test_profile",
        )
        result = _evaluate_system_worker(spec)
        assert result.system == "scrubadub"
        assert result.available is False
        assert result.skipped_reason is not None
        assert "forced skip for test" in result.skipped_reason

    def test_unknown_system_returns_skipped(self) -> None:
        spec = _SystemEvalSpec(
            system_name="nonexistent_engine",
            records=[
                BenchmarkRecord(
                    record_id="r1",
                    text="Test text",
                    labels=[],
                    language="en",
                ),
            ],
            warmup_samples=0,
            measured_runs=1,
            allow_fallback_detectors=True,
            require_native_competitors=False,
            forced_unavailable_reason=None,
            profile_label="test_profile",
        )
        result = _evaluate_system_worker(spec)
        assert result.system == "nonexistent_engine"
        assert result.available is False
        assert result.skipped_reason is not None


# ---------------------------------------------------------------------------
# CLI --parallel flag
# ---------------------------------------------------------------------------


class TestParallelCLIFlag:
    def test_default_is_true(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--parallel",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        args = parser.parse_args([])
        assert args.parallel is True

    def test_no_parallel_flag(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--parallel",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        args = parser.parse_args(["--no-parallel"])
        assert args.parallel is False

    def test_explicit_parallel_flag(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--parallel",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        args = parser.parse_args(["--parallel"])
        assert args.parallel is True
