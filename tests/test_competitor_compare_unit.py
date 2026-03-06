from __future__ import annotations

import sys
from types import ModuleType

import pytest

from pii_anon.benchmarks import BenchmarkRecord, UseCaseProfile
from pii_anon.evaluation import competitor_compare as cc
from pii_anon.evaluation.competitor_compare import (
    FloorCheckResult,
    ProfileBenchmarkResult,
    SystemBenchmarkResult,
)


def _sample_record() -> BenchmarkRecord:
    return BenchmarkRecord(
        record_id="r1",
        text="Contact alice@example.com",
        labels=[{"entity_type": "EMAIL_ADDRESS", "start": 8, "end": 25}],
        language="en",
    )


def test_normalization_and_metrics_helpers() -> None:
    assert cc._normalize_entity_type("emailfilth") == "EMAIL_ADDRESS"
    assert cc._normalize_entity_type("phone") == "PHONE_NUMBER"
    assert cc._normalize_entity_type("creditcardfilth") == "CREDIT_CARD"
    assert cc._normalize_entity_type("ipaddressfilth") == "IP_ADDRESS"

    pred = [("r1", "EMAIL_ADDRESS", 8, 25)]
    labels = [("r1", "EMAIL_ADDRESS", 8, 25), ("r1", "PHONE_NUMBER", 0, 1)]
    # Inline metrics check (shared frozenset construction used in _evaluate_system)
    pred_set = frozenset(pred)
    label_set = frozenset(labels)
    true_pos = len(pred_set & label_set)
    precision = cc._safe_div(true_pos, len(pred_set))
    recall = cc._safe_div(true_pos, len(label_set))
    f1 = cc._safe_div(2.0 * precision * recall, precision + recall)
    assert precision == 1.0
    assert recall == 0.5
    assert round(f1, 3) == 0.667

    per_entity = cc._per_entity_recall(predictions=pred, labels=labels)
    assert per_entity["EMAIL_ADDRESS"] == 1.0
    assert per_entity["PHONE_NUMBER"] == 0.0


def test_evaluate_floor_contract_for_objectives() -> None:
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
            precision=0.9,
            recall=0.9,
            f1=0.9,
            latency_p50_ms=10.0,
            docs_per_hour=2000.0,
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
            precision=0.8,
            recall=0.85,
            f1=0.83,
            latency_p50_ms=20.0,
            docs_per_hour=1500.0,
            per_entity_recall={},
            samples=10,
        ),
    ]
    for objective in ("accuracy", "balanced", "speed"):
        passed, checks = cc._evaluate_floor_contract(systems, objective)
        assert passed is True
        assert checks


def test_deterministic_profile_records_respects_filters() -> None:
    rows = [
        BenchmarkRecord(record_id="a", text="a", labels=[], language="en"),
        BenchmarkRecord(record_id="b", text="b", labels=[], language="es"),
        BenchmarkRecord(record_id="c", text="c", labels=[], language="fr"),
    ]
    profile = UseCaseProfile(profile="short_chat", objective="speed", languages=["en", "es"], max_samples=1)
    out = cc._deterministic_profile_records(rows, profile=profile, max_samples=None)
    assert len(out) == 1
    assert out[0].language in {"en", "es"}


def test_compare_competitors_enforce_floor_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": [_sample_record()])
    monkeypatch.setattr(
        cc,
        "_evaluate_profile",
        lambda **kwargs: ProfileBenchmarkResult(
            profile="default",
            objective="balanced",
            systems=[],
            floor_pass=False,
            floor_checks=[FloorCheckResult("f1", "presidio", 0.5, 0.3, False)],
            qualified_competitors=0,
            mit_qualified_competitors=0,
        ),
    )
    with pytest.raises(RuntimeError):
        cc.compare_competitors(enforce_floors=True)


def test_compare_competitors_matrix_required_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": [_sample_record()])
    monkeypatch.setattr(
        cc,
        "load_use_case_matrix",
        lambda _path: [
            UseCaseProfile(profile="short_chat", objective="speed", required=True),
            UseCaseProfile(profile="optional_profile", objective="balanced", required=False),
        ],
    )

    def fake_eval(**kwargs):
        profile = kwargs["profile"]
        return ProfileBenchmarkResult(
            profile=profile,
            objective="speed" if profile == "short_chat" else "balanced",
            systems=[],
            floor_pass=profile != "short_chat",
            floor_checks=[],
            qualified_competitors=1 if profile != "short_chat" else 0,
            mit_qualified_competitors=1 if profile != "short_chat" else 0,
        )

    monkeypatch.setattr(cc, "_evaluate_profile", fake_eval)
    report = cc.compare_competitors(matrix_path="matrix.json", enforce_floors=False)
    assert report.floor_pass is False
    assert report.qualification_gate_pass is False
    assert len(report.profiles) == 2


def test_compare_competitors_require_all_competitors_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": [_sample_record()])
    monkeypatch.setattr(
        cc,
        "load_use_case_matrix",
        lambda _path: [UseCaseProfile(profile="short_chat", objective="speed", required=True)],
    )

    def fake_eval(**kwargs):
        _ = kwargs
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
                precision=0.9,
                recall=0.9,
                f1=0.9,
                latency_p50_ms=1.0,
                docs_per_hour=1000.0,
                per_entity_recall={},
                samples=1,
            ),
            SystemBenchmarkResult(
                system="presidio",
                available=False,
                skipped_reason="native init failed",
                qualification_status="unavailable",
                license_name="MIT License",
                license_source="classifier",
                citation_url="https://github.com/microsoft/presidio",
                license_gate_passed=True,
                license_gate_reason=None,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                latency_p50_ms=0.0,
                docs_per_hour=0.0,
                per_entity_recall={},
                samples=1,
            ),
        ]
        return ProfileBenchmarkResult(
            profile="short_chat",
            objective="speed",
            systems=systems,
            floor_pass=True,
            floor_checks=[],
            qualified_competitors=1,
            mit_qualified_competitors=1,
        )

    monkeypatch.setattr(cc, "_evaluate_profile", fake_eval)
    report = cc.compare_competitors(
        matrix_path="matrix.json",
        enforce_floors=False,
        require_all_competitors=True,
        expected_competitors=["presidio"],
    )
    assert report.all_competitors_available is False
    assert report.qualification_gate_pass is False
    assert "presidio" in report.unavailable_competitors


def test_presidio_detector_falls_back_when_native_analyzer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build fake presidio_analyzer module with AnalyzerEngine that raises on analyze().
    module = ModuleType("presidio_analyzer")

    class FakeAnalyzerEngine:
        def __init__(self, **kwargs):
            pass

        def analyze(self, **kwargs):
            raise RuntimeError("boom")

    setattr(module, "AnalyzerEngine", FakeAnalyzerEngine)

    # Also need a fake nlp_engine submodule so the import succeeds.
    nlp_module = ModuleType("presidio_analyzer.nlp_engine")

    class FakeNlpEngineProvider:
        def __init__(self, **kwargs):
            pass

        def create_engine(self):
            return object()

    setattr(nlp_module, "NlpEngineProvider", FakeNlpEngineProvider)

    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp_module)
    monkeypatch.setattr(
        cc,
        "_core_detector",
        lambda **kwargs: (lambda record: [(record.record_id, "EMAIL_ADDRESS", 8, 25)]),
    )

    detector, reason = cc._presidio_detector()
    assert reason is None
    assert detector is not None
    rows = detector(_sample_record())
    assert rows == [("r1", "EMAIL_ADDRESS", 8, 25)]


def test_mit_license_gate_uses_classifier_or_spdx(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMeta(dict):
        def get_all(self, key: str):
            value = self.get(key)
            if value is None:
                return None
            if isinstance(value, list):
                return value
            return [value]

    monkeypatch.setattr(
        cc.metadata,
        "metadata",
        lambda name: FakeMeta({"Classifier": ["License :: OSI Approved :: MIT License"]}),
    )
    passed, reason = cc._mit_license_gate("presidio")
    assert passed is True
    assert "MIT" in str(reason)

    monkeypatch.setattr(
        cc.metadata,
        "metadata",
        lambda name: FakeMeta({"License-Expression": "Apache-2.0"}),
    )
    passed, reason = cc._mit_license_gate("scrubadub")
    assert passed is False
    assert "not MIT-only" in str(reason)


def test_tier_system_name() -> None:
    from pii_anon.evaluation.competitor_compare import _tier_system_name

    assert _tier_system_name("auto") == "pii-anon"
    assert _tier_system_name("minimal") == "pii-anon-minimal"
    assert _tier_system_name("standard") == "pii-anon-standard"
    assert _tier_system_name("full") == "pii-anon-full"


def test_multi_tier_profile_evaluates_all_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """When engine_tiers is provided, _evaluate_profile should produce one
    core system per tier while competitors appear only once."""
    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": [_sample_record()])

    # Track calls to _core_detector to see which tiers are evaluated.
    core_tiers_seen: list[str] = []

    def fake_core_detector(**kwargs):
        tier = kwargs.get("engine_tier", "auto")
        core_tiers_seen.append(tier)
        return lambda record: [(record.record_id, "EMAIL_ADDRESS", 8, 25)]

    monkeypatch.setattr(cc, "_core_detector", fake_core_detector)

    # Stub competitors to be unavailable so they're quick.
    monkeypatch.setattr(cc, "_COMPETITOR_META", {})

    report = cc.compare_competitors(
        dataset="pii_anon_benchmark_v1",
        engine_tiers=["auto", "minimal", "standard", "full"],
        warmup_samples=1,
        measured_runs=1,
        enforce_floors=False,
        include_end_to_end=False,
        enable_parallel=False,
    )

    # All four tiers should have been evaluated.
    assert len(core_tiers_seen) == 4
    assert set(core_tiers_seen) == {"auto", "minimal", "standard", "full"}

    # The report should contain 4 pii-anon systems.
    pii_systems = [s for s in report.profiles[0].systems if s.system.startswith("pii-anon")]
    assert len(pii_systems) == 4
    pii_names = {s.system for s in pii_systems}
    assert pii_names == {"pii-anon", "pii-anon-minimal", "pii-anon-standard", "pii-anon-full"}


def test_multi_tier_competitors_evaluated_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Competitors should be evaluated exactly once, even when multiple engine tiers
    are requested for core pii-anon."""
    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": [_sample_record()])

    competitor_calls: list[str] = []

    def fake_core_detector(**kwargs):
        return lambda record: [(record.record_id, "EMAIL_ADDRESS", 8, 25)]

    monkeypatch.setattr(cc, "_core_detector", fake_core_detector)

    # Track presidio calls.
    def tracking_presidio(**kwargs):
        competitor_calls.append("presidio")
        return (None, "stubbed out for test")

    monkeypatch.setattr(cc, "_presidio_detector", tracking_presidio)
    monkeypatch.setattr(cc, "_scrubadub_detector", lambda **kw: (None, "stub"))

    cc.compare_competitors(
        dataset="pii_anon_benchmark_v1",
        engine_tiers=["auto", "minimal", "standard"],
        warmup_samples=1,
        measured_runs=1,
        enforce_floors=False,
        include_end_to_end=False,
        enable_parallel=False,
    )

    # presidio should only be called once, not 3 times.
    assert competitor_calls.count("presidio") == 1


def test_multi_tier_progress_work_units(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify progress TOTAL accounts for len(engine_tiers) core evaluations."""
    monkeypatch.setattr(cc, "load_benchmark_dataset", lambda _dataset, source="auto": [_sample_record()])
    monkeypatch.setattr(cc, "_COMPETITOR_META", {})

    def fake_core_detector(**kwargs):
        return lambda record: [(record.record_id, "EMAIL_ADDRESS", 8, 25)]

    monkeypatch.setattr(cc, "_core_detector", fake_core_detector)

    progress_msgs: list[str] = []
    cc.compare_competitors(
        dataset="pii_anon_benchmark_v1",
        engine_tiers=["auto", "minimal"],
        warmup_samples=1,
        measured_runs=1,
        enforce_floors=False,
        include_end_to_end=False,
        enable_parallel=False,
        progress_hook=lambda msg: progress_msgs.append(msg),
    )

    # Find the TOTAL message.
    total_msgs = [m for m in progress_msgs if m.startswith("TOTAL:")]
    assert len(total_msgs) == 1
    # With 2 tiers, 1 record, warmup=1, measured=1:
    # work per eval = min(1,1) + 1*1 = 2
    # total = 2 tiers * 2 work = 4 (no competitors, no e2e)
    total_val = int(total_msgs[0].split("|")[0].replace("TOTAL:", ""))
    assert total_val == 4


def test_floor_gate_uses_auto_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """The floor gate should use the 'pii-anon' (auto tier) as the canonical
    core system for floor comparison, not tiered variants."""
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
            precision=0.9,
            recall=0.9,
            f1=0.9,
            latency_p50_ms=10.0,
            docs_per_hour=2000.0,
            per_entity_recall={},
            samples=10,
        ),
        SystemBenchmarkResult(
            system="pii-anon-minimal",
            available=True,
            skipped_reason=None,
            qualification_status="core",
            license_name="Apache-2.0",
            license_source="project",
            citation_url="https://github.com/subhash-holla/pii-anon",
            license_gate_passed=True,
            license_gate_reason=None,
            precision=0.7,
            recall=0.7,
            f1=0.7,
            latency_p50_ms=5.0,
            docs_per_hour=4000.0,
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
            license_gate_reason=None,
            precision=0.8,
            recall=0.85,
            f1=0.83,
            latency_p50_ms=20.0,
            docs_per_hour=1500.0,
            per_entity_recall={},
            samples=10,
        ),
    ]
    # Floor contract checks core ("pii-anon") vs competitors.
    passed, checks = cc._evaluate_floor_contract(systems, "balanced")
    assert passed is True
    # Make sure no check references pii-anon-minimal as a comparator.
    for check in checks:
        assert "pii-anon-minimal" not in check.comparator


def test_core_detector_uses_orchestrator_detect_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"detect_only": 0}

    class FakeOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            _ = args, kwargs

        def detect_only(self, payload, *, profile, segmentation, scope, token_version):
            _ = payload, profile, segmentation, scope, token_version
            calls["detect_only"] += 1
            return {
                "ensemble_findings": [
                    {
                        "entity_type": "EMAIL_ADDRESS",
                        "span": {"start": 8, "end": 25},
                    }
                ]
            }

    monkeypatch.setattr(cc, "PIIOrchestrator", FakeOrchestrator)
    detector = cc._core_detector(use_case="short_chat", objective="accuracy")
    out = detector(_sample_record())
    assert calls["detect_only"] == 1
    assert out == [("r1", "EMAIL_ADDRESS", 8, 25)]


# ── Overlap matching tests ──────────────────────────────────────────────


class TestOverlapMatch:
    """Tests for _overlap_match and related helpers."""

    def test_exact_match_counts_as_tp(self) -> None:
        preds = [("r1", "EMAIL_ADDRESS", 10, 30)]
        labels = [("r1", "EMAIL_ADDRESS", 10, 30)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 1
        assert mp == {0}
        assert ml == {0}

    def test_no_overlap_yields_zero(self) -> None:
        preds = [("r1", "EMAIL_ADDRESS", 0, 10)]
        labels = [("r1", "EMAIL_ADDRESS", 20, 30)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 0

    def test_different_entity_type_no_match(self) -> None:
        preds = [("r1", "EMAIL_ADDRESS", 10, 30)]
        labels = [("r1", "PHONE_NUMBER", 10, 30)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 0

    def test_different_record_id_no_match(self) -> None:
        preds = [("r1", "EMAIL_ADDRESS", 10, 30)]
        labels = [("r2", "EMAIL_ADDRESS", 10, 30)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 0

    def test_high_overlap_matches(self) -> None:
        """Prediction off by 1 char should still match (IoU > 0.5)."""
        preds = [("r1", "PERSON_NAME", 10, 21)]
        labels = [("r1", "PERSON_NAME", 10, 20)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        # IoU = 10 / 11 = 0.909 > 0.5
        assert tp == 1

    def test_low_overlap_no_match(self) -> None:
        """Small overlap should not match when IoU < 0.5."""
        preds = [("r1", "PERSON_NAME", 10, 30)]
        labels = [("r1", "PERSON_NAME", 25, 50)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        # overlap = 5, union = 40, IoU = 0.125 < 0.5
        assert tp == 0

    def test_multiple_preds_per_label(self) -> None:
        """Each label matches at most one prediction (greedy by IoU)."""
        labels = [("r1", "EMAIL_ADDRESS", 10, 30)]
        preds = [
            ("r1", "EMAIL_ADDRESS", 10, 30),  # exact match, IoU = 1.0
            ("r1", "EMAIL_ADDRESS", 10, 31),  # slightly off, IoU < 1.0
        ]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 1
        assert len(mp) == 1
        assert len(ml) == 1

    def test_one_to_one_assignment(self) -> None:
        """Two labels should match two different predictions (no double counting)."""
        labels = [
            ("r1", "PERSON_NAME", 10, 20),
            ("r1", "PERSON_NAME", 30, 40),
        ]
        preds = [
            ("r1", "PERSON_NAME", 10, 21),
            ("r1", "PERSON_NAME", 30, 41),
        ]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 2
        assert len(mp) == 2
        assert len(ml) == 2

    def test_empty_predictions(self) -> None:
        labels = [("r1", "EMAIL_ADDRESS", 0, 10)]
        tp, mp, ml = cc._overlap_match([], labels)
        assert tp == 0

    def test_empty_labels(self) -> None:
        preds = [("r1", "EMAIL_ADDRESS", 0, 10)]
        tp, mp, ml = cc._overlap_match(preds, [])
        assert tp == 0

    def test_both_empty(self) -> None:
        tp, mp, ml = cc._overlap_match([], [])
        assert tp == 0

    def test_entity_type_normalization(self) -> None:
        """Prediction entity type 'emailfilth' should match label 'EMAIL_ADDRESS'."""
        preds = [("r1", "emailfilth", 10, 30)]
        labels = [("r1", "EMAIL_ADDRESS", 10, 30)]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 1

    def test_multi_record(self) -> None:
        """Spans from different records should not cross-match."""
        preds = [
            ("r1", "EMAIL_ADDRESS", 10, 30),
            ("r2", "EMAIL_ADDRESS", 10, 30),
        ]
        labels = [
            ("r1", "EMAIL_ADDRESS", 10, 30),
            ("r2", "EMAIL_ADDRESS", 10, 30),
        ]
        tp, mp, ml = cc._overlap_match(preds, labels)
        assert tp == 2


class TestBuildOverlapIndex:
    """Tests for _build_overlap_index."""

    def test_groups_by_record_and_type(self) -> None:
        spans = [
            ("r1", "EMAIL_ADDRESS", 0, 10),
            ("r1", "EMAIL_ADDRESS", 20, 30),
            ("r1", "PHONE_NUMBER", 5, 15),
            ("r2", "EMAIL_ADDRESS", 0, 10),
        ]
        idx = cc._build_overlap_index(spans)
        assert len(idx[("r1", "EMAIL_ADDRESS")]) == 2
        assert len(idx[("r1", "PHONE_NUMBER")]) == 1
        assert len(idx[("r2", "EMAIL_ADDRESS")]) == 1

    def test_empty_spans(self) -> None:
        idx = cc._build_overlap_index([])
        assert idx == {}


class TestPerEntityMetricsOverlap:
    """Tests for per-entity recall/precision with overlap indices."""

    def test_recall_with_overlap_indices(self) -> None:
        labels = [
            ("r1", "EMAIL_ADDRESS", 0, 10),
            ("r1", "PHONE_NUMBER", 20, 30),
        ]
        preds = [("r1", "EMAIL_ADDRESS", 0, 11)]
        # Only label index 0 is matched
        result = cc._per_entity_recall(preds, labels, matched_label_indices={0})
        assert result["EMAIL_ADDRESS"] == 1.0
        assert result["PHONE_NUMBER"] == 0.0

    def test_precision_with_overlap_indices(self) -> None:
        preds = [
            ("r1", "EMAIL_ADDRESS", 0, 10),
            ("r1", "PHONE_NUMBER", 20, 30),
        ]
        labels = [("r1", "EMAIL_ADDRESS", 0, 10)]
        # Only pred index 0 is matched
        result = cc._per_entity_precision(preds, labels, matched_pred_indices={0})
        assert result["EMAIL_ADDRESS"] == 1.0
        assert result["PHONE_NUMBER"] == 0.0

    def test_recall_legacy_pred_set_kwarg(self) -> None:
        """Backward compat: pred_set kwarg still works for exact matching."""
        pred = [("r1", "EMAIL_ADDRESS", 0, 10)]
        labels = [("r1", "EMAIL_ADDRESS", 0, 10), ("r1", "PHONE_NUMBER", 20, 30)]
        result = cc._per_entity_recall(pred, labels, pred_set=frozenset(pred))
        assert result["EMAIL_ADDRESS"] == 1.0
        assert result["PHONE_NUMBER"] == 0.0

    def test_precision_legacy_label_set_kwarg(self) -> None:
        """Backward compat: label_set kwarg still works for exact matching."""
        pred = [("r1", "EMAIL_ADDRESS", 0, 10), ("r1", "PHONE_NUMBER", 20, 30)]
        labels = [("r1", "EMAIL_ADDRESS", 0, 10)]
        result = cc._per_entity_precision(pred, labels, label_set=frozenset(labels))
        assert result["EMAIL_ADDRESS"] == 1.0
        assert result["PHONE_NUMBER"] == 0.0


# ── Diagnostic function tests ───────────────────────────────────────────


class TestComputePerEntityF1:
    def test_basic(self) -> None:
        prec = {"EMAIL_ADDRESS": 0.8, "PHONE_NUMBER": 1.0}
        rec = {"EMAIL_ADDRESS": 0.6, "PHONE_NUMBER": 0.5}
        result = cc._compute_per_entity_f1(prec, rec)
        # EMAIL: 2*0.8*0.6 / (0.8+0.6) = 0.96/1.4 ≈ 0.685714
        assert abs(result["EMAIL_ADDRESS"] - 0.685714) < 0.001
        # PHONE: 2*1.0*0.5 / (1.0+0.5) = 1.0/1.5 ≈ 0.666667
        assert abs(result["PHONE_NUMBER"] - 0.666667) < 0.001

    def test_zero_precision_and_recall(self) -> None:
        prec = {"EMAIL_ADDRESS": 0.0}
        rec = {"EMAIL_ADDRESS": 0.0}
        result = cc._compute_per_entity_f1(prec, rec)
        assert result["EMAIL_ADDRESS"] == 0.0

    def test_types_only_in_one_dict(self) -> None:
        prec = {"EMAIL_ADDRESS": 0.9}
        rec = {"PHONE_NUMBER": 0.7}
        result = cc._compute_per_entity_f1(prec, rec)
        # EMAIL: P=0.9, R=0.0 → F1=0
        assert result["EMAIL_ADDRESS"] == 0.0
        # PHONE: P=0.0, R=0.7 → F1=0
        assert result["PHONE_NUMBER"] == 0.0

    def test_empty(self) -> None:
        assert cc._compute_per_entity_f1({}, {}) == {}


class TestClassifyErrors:
    def test_all_true_positives(self) -> None:
        labels = [("r1", "EMAIL_ADDRESS", 0, 10)]
        preds = [("r1", "EMAIL_ADDRESS", 0, 10)]
        matched_preds = {0}
        matched_labels = {0}
        totals, per_ent = cc._classify_errors(preds, labels, matched_preds, matched_labels)
        assert totals["true_positive"] == 1
        assert totals["boundary_miss"] == 0
        assert totals["complete_miss"] == 0
        assert totals["spurious_fp"] == 0
        assert totals["type_confusion"] == 0
        assert per_ent["EMAIL_ADDRESS"]["true_positive"] == 1

    def test_complete_miss(self) -> None:
        labels = [("r1", "EMAIL_ADDRESS", 0, 10)]
        preds: list[cc.LabelSpan] = []
        totals, per_ent = cc._classify_errors(preds, labels, set(), set())
        assert totals["complete_miss"] == 1
        assert per_ent["EMAIL_ADDRESS"]["complete_miss"] == 1

    def test_spurious_fp(self) -> None:
        labels: list[cc.LabelSpan] = []
        preds = [("r1", "EMAIL_ADDRESS", 0, 10)]
        totals, per_ent = cc._classify_errors(preds, labels, set(), set())
        assert totals["spurious_fp"] == 1
        assert per_ent["EMAIL_ADDRESS"]["spurious_fp"] == 1

    def test_boundary_miss(self) -> None:
        """Prediction overlaps same type but IoU < threshold (0.5)."""
        # Label: 0..20, Pred: 15..40 → inter=5, union=40 → IoU=0.125 < 0.5
        labels = [("r1", "EMAIL_ADDRESS", 0, 20)]
        preds = [("r1", "EMAIL_ADDRESS", 15, 40)]
        # Neither matched (IoU too low)
        totals, per_ent = cc._classify_errors(preds, labels, set(), set())
        assert totals["boundary_miss"] == 1
        assert totals["spurious_fp"] == 1  # pred is also unmatched
        assert per_ent["EMAIL_ADDRESS"]["boundary_miss"] == 1

    def test_type_confusion(self) -> None:
        """Prediction overlaps label position but different entity type."""
        labels = [("r1", "EMAIL_ADDRESS", 0, 20)]
        preds = [("r1", "PHONE_NUMBER", 0, 20)]
        totals, per_ent = cc._classify_errors(preds, labels, set(), set())
        assert totals["type_confusion"] == 1
        assert per_ent["EMAIL_ADDRESS"]["type_confusion"] == 1

    def test_mixed_errors(self) -> None:
        """Multiple error types in a single evaluation."""
        labels = [
            ("r1", "EMAIL_ADDRESS", 0, 10),   # will be TP
            ("r1", "PHONE_NUMBER", 20, 30),    # complete miss
            ("r2", "PERSON_NAME", 0, 15),      # boundary miss
        ]
        preds = [
            ("r1", "EMAIL_ADDRESS", 0, 10),    # TP
            ("r1", "US_SSN", 50, 60),          # spurious FP
            ("r2", "PERSON_NAME", 10, 25),     # overlaps but IoU < 0.5
        ]
        matched_preds = {0}
        matched_labels = {0}
        totals, per_ent = cc._classify_errors(preds, labels, matched_preds, matched_labels)
        assert totals["true_positive"] == 1
        assert totals["complete_miss"] == 1
        assert totals["boundary_miss"] == 1
        assert totals["spurious_fp"] == 2  # SSN FP + unmatched PERSON_NAME pred

    def test_support_tracked(self) -> None:
        labels = [("r1", "EMAIL_ADDRESS", 0, 10), ("r1", "EMAIL_ADDRESS", 20, 30)]
        preds = [("r1", "EMAIL_ADDRESS", 0, 10)]
        matched_preds = {0}
        matched_labels = {0}
        totals, per_ent = cc._classify_errors(preds, labels, matched_preds, matched_labels)
        assert per_ent["EMAIL_ADDRESS"]["support"] == 2


class TestComputeDiagnostics:
    def _make_system(
        self,
        name: str,
        f1: float = 0.8,
        per_entity_f1: dict[str, float] | None = None,
        error_counts: dict[str, int] | None = None,
        per_entity_errors: dict[str, dict[str, int]] | None = None,
        is_core: bool = False,
    ) -> SystemBenchmarkResult:
        return SystemBenchmarkResult(
            system=name,
            available=True,
            skipped_reason=None,
            qualification_status="core" if is_core else "qualified",
            license_name="MIT",
            license_source="https://example.com",
            citation_url="https://example.com",
            license_gate_passed=True,
            license_gate_reason=None,
            precision=f1,
            recall=f1,
            f1=f1,
            latency_p50_ms=10.0,
            docs_per_hour=10000.0,
            per_entity_recall={},
            samples=100,
            per_entity_f1=per_entity_f1 or {},
            error_counts=error_counts or {},
            per_entity_errors=per_entity_errors or {},
        )

    def test_entity_head_to_head(self) -> None:
        sys_a = self._make_system("pii-anon", per_entity_f1={"EMAIL_ADDRESS": 0.9, "PHONE_NUMBER": 0.7}, is_core=True)
        sys_b = self._make_system("presidio", per_entity_f1={"EMAIL_ADDRESS": 0.8, "PHONE_NUMBER": 0.85})
        diag = cc._compute_diagnostics([sys_a, sys_b])
        h2h = diag["entity_head_to_head"]
        assert h2h["EMAIL_ADDRESS"]["best_system"] == "pii-anon"
        assert h2h["PHONE_NUMBER"]["best_system"] == "presidio"

    def test_improvement_opportunities(self) -> None:
        sys_a = self._make_system(
            "pii-anon",
            per_entity_f1={"EMAIL_ADDRESS": 0.9, "PHONE_NUMBER": 0.5},
            per_entity_errors={"EMAIL_ADDRESS": {"support": 100}, "PHONE_NUMBER": {"support": 200}},
            is_core=True,
        )
        sys_b = self._make_system("presidio", per_entity_f1={"EMAIL_ADDRESS": 0.85, "PHONE_NUMBER": 0.8})
        diag = cc._compute_diagnostics([sys_a, sys_b])
        opps = diag["improvement_opportunities"]["pii-anon"]
        # PHONE_NUMBER has bigger delta (0.3) × support (200) = 60 weighted impact
        assert opps[0]["entity_type"] == "PHONE_NUMBER"
        assert opps[0]["delta_f1"] == pytest.approx(0.3, abs=0.01)
        # EMAIL_ADDRESS: competitor is worse, so no opportunity
        assert len(opps) == 1

    def test_entity_difficulty_ranking(self) -> None:
        sys_a = self._make_system("sys_a", per_entity_f1={"EASY": 0.95, "HARD": 0.2})
        sys_b = self._make_system("sys_b", per_entity_f1={"EASY": 0.90, "HARD": 0.3})
        diag = cc._compute_diagnostics([sys_a, sys_b])
        ranking = diag["entity_difficulty_ranking"]
        # HARD should be first (lowest avg F1)
        assert ranking[0]["entity_type"] == "HARD"
        assert ranking[1]["entity_type"] == "EASY"

    def test_entity_type_wins(self) -> None:
        sys_a = self._make_system("pii-anon", per_entity_f1={"A": 0.9, "B": 0.9, "C": 0.5}, is_core=True)
        sys_b = self._make_system("presidio", per_entity_f1={"A": 0.8, "B": 0.8, "C": 0.9})
        diag = cc._compute_diagnostics([sys_a, sys_b])
        wins = diag["entity_type_wins"]
        assert wins["pii-anon"] == 2
        assert wins["presidio"] == 1

    def test_system_error_profiles(self) -> None:
        sys_a = self._make_system(
            "pii-anon",
            error_counts={"true_positive": 80, "boundary_miss": 10, "complete_miss": 5, "spurious_fp": 5, "type_confusion": 0},
            is_core=True,
        )
        diag = cc._compute_diagnostics([sys_a])
        profile = diag["system_error_profiles"]["pii-anon"]
        assert profile["total_errors"] == 20
        assert profile["error_rate"] == pytest.approx(0.2, abs=0.01)

    def test_empty_systems(self) -> None:
        assert cc._compute_diagnostics([]) == {}

    def test_unavailable_systems_excluded(self) -> None:
        sys_a = self._make_system("pii-anon", per_entity_f1={"A": 0.9}, is_core=True)
        sys_b = SystemBenchmarkResult(
            system="dead",
            available=False,
            skipped_reason="unavailable",
            qualification_status="unavailable",
            license_name=None,
            license_source=None,
            citation_url=None,
            license_gate_passed=False,
            license_gate_reason=None,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            latency_p50_ms=0.0,
            docs_per_hour=0.0,
            per_entity_recall={},
            samples=0,
        )
        diag = cc._compute_diagnostics([sys_a, sys_b])
        assert "dead" not in diag.get("system_error_profiles", {})
        h2h = diag["entity_head_to_head"]
        assert h2h["A"]["best_system"] == "pii-anon"
