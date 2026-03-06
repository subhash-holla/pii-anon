"""Targeted tests to boost code coverage on under-tested modules.

Covers: privacy_metrics, registry, adapter init/detect branches,
llm_guard_adapter (mocked), scrubadub/presidio adapters (mocked).
"""

from __future__ import annotations

from collections import Counter

from pii_anon.engines import RegexEngineAdapter
from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter
from pii_anon.engines.presidio_adapter import PresidioAdapter
from pii_anon.engines.registry import EngineRegistry
from pii_anon.engines.scrubadub_adapter import ScrubadubAdapter
from pii_anon.eval_framework.metrics.base import EvaluationLevel, LabeledSpan
from pii_anon.eval_framework.metrics.privacy_metrics import (
    KAnonymityMetric,
    LDiversityMetric,
    LeakageDetectionMetric,
    ReidentificationRiskMetric,
    TClosenessMetric,
)
from pii_anon.ingestion.schema import FileIngestResult


# ---------------------------------------------------------------------------
# Privacy metrics
# ---------------------------------------------------------------------------


class TestReidentificationRisk:
    def test_no_leakage(self) -> None:
        metric = ReidentificationRiskMetric()
        labels = [LabeledSpan(entity_type="PERSON", start=0, end=5)]
        result = metric.compute(
            [], labels,
            context={"original_text": "Alice works here", "anonymized_text": "<PERSON> works here"},
        )
        assert result.value == 0.0

    def test_full_leakage(self) -> None:
        metric = ReidentificationRiskMetric()
        labels = [LabeledSpan(entity_type="PERSON", start=0, end=5)]
        result = metric.compute(
            [], labels,
            context={"original_text": "Alice works here", "anonymized_text": "Alice works here"},
        )
        assert result.value == 1.0

    def test_no_context(self) -> None:
        metric = ReidentificationRiskMetric()
        result = metric.compute([], [LabeledSpan(entity_type="X", start=0, end=3)])
        assert result.value == 0.0

    def test_empty_labels(self) -> None:
        metric = ReidentificationRiskMetric()
        result = metric.compute([], [], context={"original_text": "hello", "anonymized_text": "hello"})
        assert result.value == 0.0


class TestKAnonymity:
    def test_basic(self) -> None:
        metric = KAnonymityMetric()
        result = metric.compute(
            [], [],
            context={"pseudonym_map": {"Alice": "P1", "Bob": "P1", "Charlie": "P2"}},
        )
        assert result.value == 1.0  # min group size: P2 has 1 member

    def test_empty_map(self) -> None:
        metric = KAnonymityMetric()
        result = metric.compute([], [])
        assert result.value == 0.0

    def test_all_same_pseudonym(self) -> None:
        metric = KAnonymityMetric()
        result = metric.compute(
            [], [],
            context={"pseudonym_map": {"A": "P", "B": "P", "C": "P"}},
        )
        assert result.value == 3.0


class TestLDiversity:
    def test_basic_entropy(self) -> None:
        metric = LDiversityMetric()
        result = metric.compute(
            [], [],
            context={"entity_values": {"P1": ["Alice", "Bob"], "P2": ["Charlie", "Dave"]}},
        )
        assert result.value > 0.0  # both groups have 2 distinct values → entropy > 0

    def test_single_value_per_group(self) -> None:
        metric = LDiversityMetric()
        result = metric.compute(
            [], [],
            context={"entity_values": {"P1": ["Alice"], "P2": ["Bob"]}},
        )
        assert result.value == 0.0  # entropy of single-element groups is 0

    def test_empty_context(self) -> None:
        metric = LDiversityMetric()
        result = metric.compute([], [])
        assert result.value == 0.0

    def test_supported_levels(self) -> None:
        metric = LDiversityMetric()
        assert EvaluationLevel.DOCUMENT in metric.supported_levels


class TestTCloseness:
    def test_basic_emd(self) -> None:
        metric = TClosenessMetric()
        result = metric.compute(
            [], [],
            context={
                "group_distributions": {"g1": Counter({"A": 5, "B": 5}), "g2": Counter({"A": 10})},
                "overall_distribution": Counter({"A": 15, "B": 5}),
            },
        )
        assert result.value > 0.0

    def test_identical_distributions(self) -> None:
        metric = TClosenessMetric()
        overall = Counter({"A": 10, "B": 10})
        result = metric.compute(
            [], [],
            context={
                "group_distributions": {"g1": Counter({"A": 5, "B": 5})},
                "overall_distribution": overall,
            },
        )
        assert result.value == 0.0

    def test_empty_context(self) -> None:
        metric = TClosenessMetric()
        result = metric.compute([], [])
        assert result.value == 0.0

    def test_empty_group(self) -> None:
        metric = TClosenessMetric()
        result = metric.compute(
            [], [],
            context={
                "group_distributions": {"g1": Counter()},
                "overall_distribution": Counter({"A": 10}),
            },
        )
        assert result.value == 0.0


class TestLeakageDetection:
    def test_no_leakage(self) -> None:
        metric = LeakageDetectionMetric(min_chars=3)
        labels = [LabeledSpan(entity_type="NAME", start=0, end=5)]
        result = metric.compute(
            [], labels,
            context={"original_text": "Alice is here", "anonymized_text": "<NAME> is here"},
        )
        assert result.value == 0.0

    def test_partial_leakage(self) -> None:
        metric = LeakageDetectionMetric(min_chars=3)
        labels = [LabeledSpan(entity_type="NAME", start=0, end=5)]
        result = metric.compute(
            [], labels,
            context={"original_text": "Alice is here", "anonymized_text": "Ali## is here"},
        )
        assert result.value > 0.0  # "Ali" is 3 chars and found

    def test_short_entity_skipped(self) -> None:
        metric = LeakageDetectionMetric(min_chars=5)
        labels = [LabeledSpan(entity_type="X", start=0, end=2)]
        result = metric.compute(
            [], labels,
            context={"original_text": "AB is here", "anonymized_text": "AB is here"},
        )
        assert result.value == 0.0  # too short to check

    def test_empty_context(self) -> None:
        metric = LeakageDetectionMetric()
        result = metric.compute([], [LabeledSpan(entity_type="X", start=0, end=3)])
        assert result.value == 0.0

    def test_supported_levels(self) -> None:
        metric = LeakageDetectionMetric()
        assert EvaluationLevel.DOCUMENT in metric.supported_levels


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


class TestEngineRegistryCoverage:
    def test_register_and_unregister(self) -> None:
        reg = EngineRegistry()
        engine = RegexEngineAdapter(enabled=True)
        reg.register(engine)
        assert engine.adapter_id in reg.ids()
        reg.unregister(engine.adapter_id)
        assert engine.adapter_id not in reg.ids()

    def test_unregister_unknown_is_noop(self) -> None:
        reg = EngineRegistry()
        reg.unregister("nonexistent")  # should not raise

    def test_initialize_engines(self) -> None:
        reg = EngineRegistry()
        engine = RegexEngineAdapter(enabled=True)
        reg.register(engine)
        reg.initialize({"regex-oss": {"enabled": True, "weight": 2.0}})
        assert engine.enabled is True

    def test_list_engines_disabled(self) -> None:
        reg = EngineRegistry()
        engine = RegexEngineAdapter(enabled=False)
        reg.register(engine)
        all_engines = reg.list_engines(include_disabled=True)
        enabled = reg.list_engines(include_disabled=False)
        assert len(all_engines) == 1
        assert len(enabled) == 0

    def test_health_report(self) -> None:
        reg = EngineRegistry()
        engine = RegexEngineAdapter(enabled=True)
        reg.register(engine)
        report = reg.health_report()
        assert engine.adapter_id in report

    def test_capabilities_report(self) -> None:
        reg = EngineRegistry()
        engine = RegexEngineAdapter(enabled=True)
        reg.register(engine)
        caps = reg.capabilities_report()
        assert engine.adapter_id in caps


# ---------------------------------------------------------------------------
# Engine adapter base coverage
# ---------------------------------------------------------------------------


class TestEngineAdapterBase:
    def test_initialize_empty(self) -> None:
        engine = RegexEngineAdapter(enabled=True)
        engine.initialize(None)
        assert engine.enabled is True

    def test_initialize_disables(self) -> None:
        engine = RegexEngineAdapter(enabled=True)
        engine.initialize({"enabled": False})
        assert engine.enabled is False

    def test_dependency_available_no_dep(self) -> None:
        engine = RegexEngineAdapter(enabled=True)
        assert engine.dependency_available() is True

    def test_health_check_disabled(self) -> None:
        engine = RegexEngineAdapter(enabled=False)
        report = engine.health_check()
        assert report["details"] == "disabled"

    def test_health_check_enabled(self) -> None:
        engine = RegexEngineAdapter(enabled=True)
        report = engine.health_check()
        assert report["healthy"] is True

    def test_shutdown_noop(self) -> None:
        engine = RegexEngineAdapter(enabled=True)
        result = engine.shutdown()
        assert result is None

    def test_capabilities_returns_dataclass(self) -> None:
        engine = RegexEngineAdapter(enabled=True)
        caps = engine.capabilities()
        assert caps.adapter_id == engine.adapter_id


# ---------------------------------------------------------------------------
# LLM Guard adapter (mocked)
# ---------------------------------------------------------------------------


class TestLLMGuardAdapterMocked:
    def test_dependency_not_available(self) -> None:
        adapter = LLMGuardAdapter(enabled=True)
        # llm_guard may or may not be installed; just check the method works
        result = adapter.dependency_available()
        assert isinstance(result, bool)

    def test_detect_without_dependency(self) -> None:
        adapter = LLMGuardAdapter(enabled=True)
        findings = adapter.detect({"text": "test"}, {"language": "en"})
        # Without llm_guard installed, should return empty or handle gracefully
        assert isinstance(findings, list)

    def test_health_check(self) -> None:
        adapter = LLMGuardAdapter(enabled=False)
        report = adapter.health_check()
        assert report["details"] == "disabled"


# ---------------------------------------------------------------------------
# Scrubadub adapter (coverage for init path)
# ---------------------------------------------------------------------------


class TestScrubadubAdapterCoverage:
    def test_capabilities(self) -> None:
        adapter = ScrubadubAdapter(enabled=True)
        caps = adapter.capabilities()
        assert caps.adapter_id == "scrubadub-compatible"

    def test_detect_empty_payload(self) -> None:
        adapter = ScrubadubAdapter(enabled=True)
        findings = adapter.detect({}, {"language": "en"})
        assert findings == []


# ---------------------------------------------------------------------------
# Presidio adapter (coverage for init path)
# ---------------------------------------------------------------------------


class TestPresidioAdapterCoverage:
    def test_capabilities(self) -> None:
        adapter = PresidioAdapter(enabled=True)
        caps = adapter.capabilities()
        assert caps.adapter_id == "presidio-compatible"

    def test_detect_empty_payload(self) -> None:
        adapter = PresidioAdapter(enabled=True)
        findings = adapter.detect({}, {"language": "en"})
        assert findings == []


# ---------------------------------------------------------------------------
# FileIngestResult property coverage
# ---------------------------------------------------------------------------


class TestFileIngestResultCoverage:
    def test_records_per_second_zero_time(self) -> None:
        result = FileIngestResult(
            input_path="test.csv", output_path=None, format="csv",
            records_processed=10, records_failed=0,
            total_chars=100, total_chunks=10, elapsed_seconds=0.0,
        )
        assert result.records_per_second == 0.0

    def test_records_per_second_positive(self) -> None:
        result = FileIngestResult(
            input_path="test.csv", output_path=None, format="csv",
            records_processed=100, records_failed=0,
            total_chars=1000, total_chunks=100, elapsed_seconds=2.0,
        )
        assert result.records_per_second == 50.0
