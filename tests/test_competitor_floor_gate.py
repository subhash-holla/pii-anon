from __future__ import annotations

import pytest

from pii_anon.benchmarks import UseCaseProfile
from pii_anon.evaluation import competitor_compare as cc
from pii_anon.evaluation.competitor_compare import (
    FloorCheckResult,
    ProfileBenchmarkResult,
    SystemBenchmarkResult,
)


def _profile_result(*, floor_pass: bool) -> ProfileBenchmarkResult:
    systems = [
        SystemBenchmarkResult(
            system="pii-anon",
            available=True,
            skipped_reason=None,
            qualification_status="core",
            license_name="Apache-2.0",
            license_source="project",
            citation_url="https://github.com/subhash-holla/pii-anon",
            license_gate_passed=True,
            license_gate_reason=None,
            precision=0.8,
            recall=0.8,
            f1=0.8,
            latency_p50_ms=1.0,
            docs_per_hour=1000.0,
            per_entity_recall={},
            samples=10,
        ),
        SystemBenchmarkResult(
            system="presidio",
            available=True,
            skipped_reason=None,
            qualification_status="qualified",
            license_name="MIT License",
            license_source="classifier",
            citation_url="https://github.com/microsoft/presidio",
            license_gate_passed=True,
            license_gate_reason="MIT classifier evidence",
            precision=0.7,
            recall=0.7,
            f1=0.7,
            latency_p50_ms=2.0,
            docs_per_hour=900.0,
            per_entity_recall={},
            samples=10,
        ),
    ]
    return ProfileBenchmarkResult(
        profile="short_chat",
        objective="speed",
        systems=systems,
        floor_pass=floor_pass,
        floor_checks=[
            FloorCheckResult(
                metric="latency_p50_ms",
                comparator="presidio",
                target=2.0,
                actual=1.0,
                passed=floor_pass,
            )
        ],
        qualified_competitors=1,
        mit_qualified_competitors=1,
    )


def test_matrix_floor_gate_can_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cc,
        "load_use_case_matrix",
        lambda _path: [UseCaseProfile(profile="short_chat", objective="speed", required=True)],
    )
    monkeypatch.setattr(cc, "_evaluate_profile", lambda **kwargs: _profile_result(floor_pass=False))
    with pytest.raises(RuntimeError):
        cc.compare_competitors(matrix_path="matrix.json", enforce_floors=True, max_samples=10)


def test_matrix_floor_gate_report_is_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cc,
        "load_use_case_matrix",
        lambda _path: [UseCaseProfile(profile="short_chat", objective="speed", required=True)],
    )
    monkeypatch.setattr(cc, "_evaluate_profile", lambda **kwargs: _profile_result(floor_pass=True))
    report = cc.compare_competitors(matrix_path="matrix.json", enforce_floors=False, max_samples=10)
    assert report.floor_pass is True
    assert report.qualification_gate_pass is True
    assert report.profiles[0].profile == "short_chat"
