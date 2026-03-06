from __future__ import annotations

from pii_anon.benchmarks import BenchmarkRecord
from pii_anon.evaluation import compare_competitors
from pii_anon.evaluation import competitor_compare as cc


EXPECTED_SYSTEMS = {"pii-anon", "presidio", "scrubadub", "gliner"}


def _records() -> list[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            record_id="r1",
            text="Contact alice@example.com",
            labels=[{"entity_type": "EMAIL_ADDRESS", "start": 8, "end": 25}],
            language="en",
        ),
        BenchmarkRecord(
            record_id="r2",
            text="Call +1 415 555 0100",
            labels=[{"entity_type": "PHONE_NUMBER", "start": 5, "end": 20}],
            language="en",
        ),
    ]


def _prepare_lightweight_benchmark(monkeypatch) -> None:
    def _simple_detector(record: BenchmarkRecord):
        rows = []
        for label in record.labels:
            rows.append((record.record_id, str(label["entity_type"]), int(label["start"]), int(label["end"])))
        return rows

    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": _records())
    monkeypatch.setattr(cc, "_core_detector", lambda **kwargs: _simple_detector)
    monkeypatch.setattr(cc, "_core_end_to_end_detector", lambda **kwargs: _simple_detector)
    monkeypatch.setattr(cc, "_presidio_detector", lambda **kwargs: (_simple_detector, None))
    monkeypatch.setattr(cc, "_scrubadub_detector", lambda **kwargs: (_simple_detector, None))
    monkeypatch.setattr(cc, "_gliner_detector", lambda **kwargs: (_simple_detector, None))

    # Stub the license qualification gate so competitors are not excluded
    # when their packages are not installed in the test environment.
    _original_qualify = cc._qualify_oss_license

    def _fake_qualify(system: str) -> cc.QualificationEvidence:
        if system in {"presidio", "scrubadub", "gliner"}:
            return cc.QualificationEvidence(
                passed=True,
                reason=None,
                qualification_status="qualified",
                license_name="MIT",
                license_source="test-stub",
                citation_url="https://example.com",
            )
        return _original_qualify(system)

    monkeypatch.setattr(cc, "_qualify_oss_license", _fake_qualify)


def test_competitor_benchmark_returns_all_systems(monkeypatch) -> None:
    _prepare_lightweight_benchmark(monkeypatch)
    report = compare_competitors(
        dataset="pii_anon_benchmark_v1",
        warmup_samples=1,
        measured_runs=1,
        max_samples=2,
        enable_parallel=False,  # monkeypatches don't carry into spawned processes
    )
    systems = {item.system for item in report.systems}

    assert systems == EXPECTED_SYSTEMS
    core = next(item for item in report.systems if item.system == "pii-anon")
    assert core.available is True
    assert core.samples == 2
    assert core.license_gate_passed is True
    assert core.qualification_status in {"core", "qualified"}
    assert core.evaluation_track == "detect_only"
    assert report.profiles[0].profile == "default"
    assert report.profiles[0].end_to_end_systems
    assert report.profiles[0].end_to_end_systems[0].evaluation_track == "end_to_end"
    assert report.report_schema_version == "2026-02-19.v3"
    assert report.dataset_source == "auto"
    assert "presidio" in report.expected_competitors
    assert isinstance(report.unavailable_competitors, dict)
    assert report.qualification_gate_pass is True


def test_competitor_benchmark_is_reproducible_on_metrics(monkeypatch) -> None:
    _prepare_lightweight_benchmark(monkeypatch)
    first = compare_competitors(dataset="pii_anon_benchmark_v1", warmup_samples=1, measured_runs=1, max_samples=2, enable_parallel=False)
    second = compare_competitors(dataset="pii_anon_benchmark_v1", warmup_samples=1, measured_runs=1, max_samples=2, enable_parallel=False)

    first_core = next(item for item in first.systems if item.system == "pii-anon")
    second_core = next(item for item in second.systems if item.system == "pii-anon")

    assert first_core.precision == second_core.precision
    assert first_core.recall == second_core.recall
    assert first_core.f1 == second_core.f1
