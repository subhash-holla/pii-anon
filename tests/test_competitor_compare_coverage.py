"""Additional unit tests for competitor_compare.py to improve code coverage.

Focuses on:
- Native detector paths with mocked models
- Core detector speed/sync/multi-engine paths
- License checking edge cases
- Floor contract edge cases
- Elo rating application
- Competitor availability summarization
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from pii_anon.benchmarks import BenchmarkRecord
from pii_anon.evaluation import competitor_compare as cc
from pii_anon.evaluation.competitor_compare import (
    SystemBenchmarkResult,
)


def _sample_record(text: str = "Contact alice@example.com", rid: str = "r1") -> BenchmarkRecord:
    return BenchmarkRecord(
        record_id=rid,
        text=text,
        labels=[{"entity_type": "EMAIL_ADDRESS", "start": 8, "end": 25}],
        language="en",
    )


def _make_system(
    system: str,
    *,
    available: bool = True,
    f1: float = 0.8,
    precision: float = 0.8,
    recall: float = 0.8,
    latency_p50_ms: float = 10.0,
    docs_per_hour: float = 2000.0,
    license_gate_passed: bool = True,
    qualification_status: str = "qualified",
    skipped_reason: str | None = None,
) -> SystemBenchmarkResult:
    return SystemBenchmarkResult(
        system=system,
        available=available,
        skipped_reason=skipped_reason,
        qualification_status=qualification_status,
        license_name="MIT License",
        license_source="classifier",
        citation_url=f"https://example.com/{system}",
        license_gate_passed=license_gate_passed,
        license_gate_reason=None,
        precision=precision,
        recall=recall,
        f1=f1,
        latency_p50_ms=latency_p50_ms,
        docs_per_hour=docs_per_hour,
        per_entity_recall={},
        samples=10,
    )


# ---------------------------------------------------------------------------
# Core detector: speed objective path
# ---------------------------------------------------------------------------
class TestCoreDetectorSpeedPath:
    def test_speed_objective_uses_regex_only(self) -> None:
        detector = cc._core_detector(use_case="short_chat", objective="speed")
        record = BenchmarkRecord(
            record_id="s1",
            text="Email alice@example.com and call 415-555-1234",
            labels=[],
            language="en",
        )
        rows = detector(record)
        types = {row[1] for row in rows}
        assert "EMAIL_ADDRESS" in types
        assert "PHONE_NUMBER" in types

    def test_speed_no_false_positives_on_plain_text(self) -> None:
        detector = cc._core_detector(use_case="short_chat", objective="speed")
        record = BenchmarkRecord(
            record_id="s2", text="Hello world", labels=[], language="en"
        )
        assert detector(record) == []


# ---------------------------------------------------------------------------
# Core end-to-end detector
# ---------------------------------------------------------------------------
class TestCoreEndToEndDetector:
    def test_end_to_end_detector_runs_full_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeOrchestrator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def run(self, payload: Any, *, profile: Any, segmentation: Any, scope: Any, token_version: Any) -> dict[str, Any]:
                return {
                    "ensemble_findings": [
                        {"entity_type": "EMAIL_ADDRESS", "span": {"start": 8, "end": 25}},
                    ],
                    "transformed_payload": payload,
                    "link_audit": [],
                }

        monkeypatch.setattr(cc, "PIIOrchestrator", FakeOrchestrator)
        detector = cc._core_end_to_end_detector(use_case="short_chat", objective="balanced")
        rows = detector(_sample_record())
        assert ("r1", "EMAIL_ADDRESS", 8, 25) in rows

    def test_end_to_end_skips_findings_without_spans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeOrchestrator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def run(self, payload: Any, **kwargs: Any) -> dict[str, Any]:
                return {
                    "ensemble_findings": [
                        {"entity_type": "EMAIL_ADDRESS", "span": {}},
                        {"entity_type": "PHONE_NUMBER", "span": {"start": 0, "end": 12}},
                    ],
                    "transformed_payload": payload,
                    "link_audit": [],
                }

        monkeypatch.setattr(cc, "PIIOrchestrator", FakeOrchestrator)
        detector = cc._core_end_to_end_detector(use_case="short_chat", objective="balanced")
        rows = detector(_sample_record())
        assert len(rows) == 1
        assert rows[0][1] == "PHONE_NUMBER"


# ---------------------------------------------------------------------------
# GLiNER detector mocked paths
# ---------------------------------------------------------------------------
class TestGlinerDetector:
    def test_gliner_native_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gliner_mod = ModuleType("gliner")

        class FakeGLiNER:
            @classmethod
            def from_pretrained(cls, model_name: str) -> "FakeGLiNER":
                return cls()

            def predict_entities(self, text: str, labels: list[str], threshold: float = 0.5) -> list[dict[str, Any]]:
                return [
                    {"label": "name", "start": 8, "end": 25, "score": 0.92},
                ]

        setattr(gliner_mod, "GLiNER", FakeGLiNER)
        monkeypatch.setitem(sys.modules, "gliner", gliner_mod)

        detector, reason = cc._gliner_detector()
        assert reason is None
        assert detector is not None
        rows = detector(_sample_record())
        assert len(rows) == 1
        assert rows[0][1] == "PERSON_NAME"

    def test_gliner_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "gliner", raising=False)
        monkeypatch.setattr(
            cc.importlib, "import_module",
            lambda name: (_ for _ in ()).throw(ImportError("no gliner")),
        )
        detector, reason = cc._gliner_detector()
        assert detector is None
        assert "not installed" in str(reason)

    def test_gliner_model_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gliner_mod = ModuleType("gliner")

        class BrokenGLiNER:
            @classmethod
            def from_pretrained(cls, model_name: str) -> None:
                raise RuntimeError("model not found")

        setattr(gliner_mod, "GLiNER", BrokenGLiNER)
        monkeypatch.setitem(sys.modules, "gliner", gliner_mod)

        detector, reason = cc._gliner_detector(require_native=True)
        assert detector is None
        assert "unavailable" in str(reason)


# ---------------------------------------------------------------------------
# Scrubadub detector mocked paths
# ---------------------------------------------------------------------------
class TestScrubadubDetector:
    def test_scrubadub_native_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scrubadub_mod = ModuleType("scrubadub")

        class FakeFilth:
            beg = 8
            end = 25

        class EmailFilth(FakeFilth):
            pass

        class FakeScrubber:
            def iter_filth(self, text: str) -> list[FakeFilth]:
                return [EmailFilth()]

        setattr(scrubadub_mod, "Scrubber", FakeScrubber)
        monkeypatch.setitem(sys.modules, "scrubadub", scrubadub_mod)

        detector, reason = cc._scrubadub_detector()
        assert reason is None
        rows = detector(_sample_record())
        assert len(rows) == 1
        assert rows[0][1] == "EMAIL_ADDRESS"


# ---------------------------------------------------------------------------
# License checking edge cases
# ---------------------------------------------------------------------------
class TestLicenseChecking:
    class FakeMeta(dict):
        """Metadata dict that supports get_all() like importlib.metadata."""
        def get_all(self, key: str) -> list[str] | None:
            value = self.get(key)
            if value is None:
                return None
            if isinstance(value, list):
                return value
            return [value]

    def test_qualify_oss_license_core_system(self) -> None:
        evidence = cc._qualify_oss_license("pii-anon")
        assert evidence.passed is True
        assert evidence.qualification_status == "core"

    def test_qualify_oss_license_unknown_system(self) -> None:
        evidence = cc._qualify_oss_license("totally_unknown_system")
        assert evidence.passed is False
        assert evidence.qualification_status == "excluded"

    def test_qualify_oss_license_metadata_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc, "_metadata_for_package", lambda _name: None)
        evidence = cc._qualify_oss_license("presidio")
        assert evidence.passed is False
        assert evidence.qualification_status == "unavailable"

    def test_qualify_oss_license_missing_license_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc, "_metadata_for_package", lambda _name: self.FakeMeta({}))
        evidence = cc._qualify_oss_license("presidio")
        assert evidence.passed is False
        assert "license metadata missing" in str(evidence.reason)

    def test_qualify_oss_license_osi_approved_classifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meta = self.FakeMeta({
            "Classifier": [
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python",
            ]
        })
        monkeypatch.setattr(cc, "_metadata_for_package", lambda _name: meta)
        evidence = cc._qualify_oss_license("presidio")
        assert evidence.passed is True
        assert evidence.qualification_status == "qualified"

    def test_qualify_oss_license_blocked_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meta = self.FakeMeta({"License-Expression": "PROPRIETARY"})
        monkeypatch.setattr(cc, "_metadata_for_package", lambda _name: meta)
        evidence = cc._qualify_oss_license("presidio")
        assert evidence.passed is False
        assert "unqualified" in str(evidence.reason)

    def test_qualify_oss_license_spdx_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meta = self.FakeMeta({"License-Expression": "Apache-2.0"})
        monkeypatch.setattr(cc, "_metadata_for_package", lambda _name: meta)
        evidence = cc._qualify_oss_license("presidio")
        assert evidence.passed is True

    def test_license_from_metadata_license_field_fallback(self) -> None:
        meta = self.FakeMeta({"License": "BSD-3-Clause"})
        name, source = cc._license_from_metadata(meta)
        assert name == "BSD-3-Clause"
        assert source == "license-field"

    def test_license_from_metadata_no_info(self) -> None:
        meta = self.FakeMeta({})
        name, source = cc._license_from_metadata(meta)
        assert name is None
        assert source is None

    def test_license_from_metadata_non_callable_get_all(self) -> None:
        """When info doesn't have a callable get_all, fall back to empty classifiers."""
        meta: dict[str, Any] = {"License-Expression": "MIT"}
        name, source = cc._license_from_metadata(meta)
        assert name == "MIT"
        assert source == "spdx"

    def test_metadata_for_package_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc.metadata, "metadata", lambda _name: (_ for _ in ()).throw(Exception("boom")))
        result = cc._metadata_for_package("nonexistent-pkg")
        assert result is None

    def test_mit_license_gate_pii_anon(self) -> None:
        passed, reason = cc._mit_license_gate("pii-anon")
        assert passed is True

    def test_mit_license_gate_unknown_system(self) -> None:
        passed, reason = cc._mit_license_gate("unknown_system")
        assert passed is False

    def test_mit_license_gate_metadata_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cc.metadata, "metadata", lambda _name: (_ for _ in ()).throw(Exception("boom")))
        passed, reason = cc._mit_license_gate("presidio")
        assert passed is False
        assert "metadata unavailable" in str(reason)

    def test_mit_license_gate_spdx_mit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meta = self.FakeMeta({"License-Expression": "MIT"})
        monkeypatch.setattr(cc.metadata, "metadata", lambda _name: meta)
        passed, reason = cc._mit_license_gate("presidio")
        assert passed is True
        assert "SPDX" in str(reason)

    def test_mit_license_gate_no_evidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meta = self.FakeMeta({})
        monkeypatch.setattr(cc.metadata, "metadata", lambda _name: meta)
        passed, reason = cc._mit_license_gate("presidio")
        assert passed is False
        assert "no MIT" in str(reason)


# ---------------------------------------------------------------------------
# Floor contract edge cases
# ---------------------------------------------------------------------------
class TestFloorContractEdgeCases:
    def test_core_unavailable(self) -> None:
        systems = [_make_system("presidio")]
        passed, checks = cc._evaluate_floor_contract(systems, "balanced")
        assert passed is False
        assert checks[0].metric == "core_available"

    def test_core_present_but_not_available(self) -> None:
        systems = [_make_system("pii-anon", available=False)]
        passed, checks = cc._evaluate_floor_contract(systems, "balanced")
        assert passed is False
        assert checks[0].metric == "core_available"

    def test_no_qualified_competitors(self) -> None:
        systems = [
            _make_system("pii-anon"),
            _make_system("presidio", available=False),
        ]
        passed, checks = cc._evaluate_floor_contract(systems, "balanced")
        assert passed is False
        assert checks[0].metric == "qualified_competitor_available"

    def test_no_licensed_competitors(self) -> None:
        systems = [
            _make_system("pii-anon"),
            _make_system("presidio", license_gate_passed=False),
        ]
        passed, checks = cc._evaluate_floor_contract(systems, "accuracy")
        assert passed is False
        assert checks[0].metric == "qualified_competitor_available"

    def test_accuracy_floor_checks(self) -> None:
        systems = [
            _make_system("pii-anon", f1=0.9, recall=0.85),
            _make_system("presidio", f1=0.8, recall=0.8),
        ]
        passed, checks = cc._evaluate_floor_contract(systems, "accuracy")
        assert passed is True
        metrics = {c.metric for c in checks}
        assert "f1" in metrics
        assert "recall" in metrics

    def test_speed_floor_checks(self) -> None:
        systems = [
            _make_system("pii-anon", latency_p50_ms=5.0, docs_per_hour=5000.0),
            _make_system("presidio", latency_p50_ms=10.0, docs_per_hour=2000.0),
        ]
        passed, checks = cc._evaluate_floor_contract(systems, "speed")
        assert passed is True
        metrics = {c.metric for c in checks}
        assert "latency_p50_ms" in metrics
        assert "docs_per_hour" in metrics


# ---------------------------------------------------------------------------
# Normalize findings edge cases
# ---------------------------------------------------------------------------
class TestNormalizeFindings:
    def test_missing_span_fields_are_skipped(self) -> None:
        class NoSpan:
            entity_type = "EMAIL_ADDRESS"
            span_start = None
            span_end = None

        record = _sample_record()
        result = cc._normalize_findings(record, [NoSpan()])
        assert result == []

    def test_valid_findings_are_normalized(self) -> None:
        class GoodFinding:
            entity_type = "EMAIL_ADDRESS"
            span_start = 8
            span_end = 25

        record = _sample_record()
        result = cc._normalize_findings(record, [GoodFinding()])
        assert result == [("r1", "EMAIL_ADDRESS", 8, 25)]


# ---------------------------------------------------------------------------
# Elo rating edge cases
# ---------------------------------------------------------------------------
class TestEloRating:
    def test_fewer_than_two_systems_skips_elo(self) -> None:
        systems = [_make_system("pii-anon")]
        cc._apply_elo_ratings(systems)
        # elo_rating stays at default 0.0 when skipped
        assert systems[0].elo_rating == 0.0

    def test_elo_assigns_ratings_to_two_systems(self) -> None:
        s1 = _make_system("pii-anon", f1=0.9)
        s2 = _make_system("presidio", f1=0.7)
        # Must set composite_score > 0 for Elo engine to pick them up
        s1.composite_score = 0.85
        s2.composite_score = 0.65
        systems = [s1, s2]
        cc._apply_elo_ratings(systems)
        assert systems[0].elo_rating > 0
        assert systems[1].elo_rating > 0
        # Better composite should get higher Elo
        assert systems[0].elo_rating > systems[1].elo_rating

    def test_elo_skips_unavailable_systems(self) -> None:
        s1 = _make_system("pii-anon", f1=0.9)
        s1.composite_score = 0.85
        s2 = _make_system("presidio", available=False)
        s3 = _make_system("scrubadub", f1=0.7)
        s3.composite_score = 0.65
        systems = [s1, s2, s3]
        cc._apply_elo_ratings(systems)
        assert systems[0].elo_rating > 0
        assert systems[1].elo_rating == 0.0  # unavailable, no rating assigned
        assert systems[2].elo_rating > 0


# ---------------------------------------------------------------------------
# Competitor availability summarization
# ---------------------------------------------------------------------------
class TestCompetitorAvailabilitySummary:
    def test_summarize_all_available(self) -> None:
        from pii_anon.evaluation.competitor_compare import ProfileBenchmarkResult

        profile = ProfileBenchmarkResult(
            profile="default",
            objective="balanced",
            systems=[_make_system("pii-anon"), _make_system("presidio")],
            floor_pass=True,
            floor_checks=[],
            qualified_competitors=1,
            mit_qualified_competitors=1,
        )
        available, unavailable, all_ok = cc._summarize_competitor_availability(
            profile_reports=[profile],
            expected_competitors=["presidio"],
        )
        assert "presidio" in available
        assert not unavailable
        assert all_ok is True

    def test_summarize_mixed_availability(self) -> None:
        from pii_anon.evaluation.competitor_compare import ProfileBenchmarkResult

        profile = ProfileBenchmarkResult(
            profile="default",
            objective="balanced",
            systems=[
                _make_system("pii-anon"),
                _make_system("presidio"),
                _make_system("scrubadub", available=False, skipped_reason="not installed"),
            ],
            floor_pass=True,
            floor_checks=[],
            qualified_competitors=1,
            mit_qualified_competitors=1,
        )
        available, unavailable, all_ok = cc._summarize_competitor_availability(
            profile_reports=[profile],
            expected_competitors=["presidio", "scrubadub"],
        )
        assert "presidio" in available
        assert "scrubadub" in unavailable
        assert all_ok is False

    def test_summarize_missing_benchmark_row(self) -> None:
        from pii_anon.evaluation.competitor_compare import ProfileBenchmarkResult

        profile = ProfileBenchmarkResult(
            profile="default",
            objective="balanced",
            systems=[_make_system("pii-anon")],
            floor_pass=True,
            floor_checks=[],
            qualified_competitors=0,
            mit_qualified_competitors=0,
        )
        available, unavailable, all_ok = cc._summarize_competitor_availability(
            profile_reports=[profile],
            expected_competitors=["presidio"],
        )
        assert "presidio" in unavailable
        assert "missing benchmark row" in unavailable["presidio"]
        assert all_ok is False
