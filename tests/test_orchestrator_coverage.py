"""Tests to improve coverage for orchestrator.py.

Targets uncovered lines in src/pii_anon/orchestrator.py covering:
- Strategy resolution and transformation context building
- Transform mode to strategy mapping
- Placeholder formatting
- Confidence envelope calculations
- Finding dict conversion
- Various edge cases in detection and transformation pipelines
"""

import pytest
from unittest.mock import Mock
from pii_anon.orchestrator import AsyncPIIOrchestrator, PIIOrchestrator
from pii_anon.types import (
    ProcessingProfileSpec,
    SegmentationPlan,
    EnsembleFinding,
    EngineFinding,
)
from pii_anon.transforms.strategies import RedactionStrategy
from pii_anon.transforms.base import TransformContext


@pytest.fixture
def orchestrator():
    """Create a minimal orchestrator for testing."""
    return AsyncPIIOrchestrator(token_key="test-key-12345")


@pytest.fixture
def basic_profile():
    """Create a basic processing profile."""
    return ProcessingProfileSpec(
        profile_id="test",
        transform_mode="anonymize",
        mode="weighted_consensus",
        language="en",
    )


@pytest.fixture
def basic_segmentation():
    """Create a basic segmentation plan."""
    return SegmentationPlan(enabled=False)


# ============================================================================
# Strategy Resolution Tests (Lines 841-865)
# ============================================================================


class TestStrategyResolution:
    """Test strategy ID resolution."""

    def test_resolve_strategy_from_profile_entity_strategies(self, orchestrator, basic_profile):
        """_resolve_strategy_id should use profile.entity_strategies override."""
        profile = ProcessingProfileSpec(
            profile_id="test",
            transform_mode="anonymize",
            mode="weighted_consensus",
            entity_strategies={"PERSON_NAME": "synthetic"},
        )
        strategy_id = orchestrator._resolve_strategy_id("PERSON_NAME", profile)
        assert strategy_id == "synthetic"

    def test_resolve_strategy_from_config_entity_strategies(self, orchestrator):
        """_resolve_strategy_id should use config override if profile doesn't have one."""
        # Mock config with entity_strategies
        orchestrator.config.transform = Mock()
        orchestrator.config.transform.entity_strategies = {"EMAIL_ADDRESS": "redact"}

        profile = ProcessingProfileSpec(
            profile_id="test",
            transform_mode="anonymize",
            mode="weighted_consensus",
        )
        strategy_id = orchestrator._resolve_strategy_id("EMAIL_ADDRESS", profile)
        assert strategy_id == "redact"

    def test_resolve_strategy_from_transform_mode(self, orchestrator, basic_profile):
        """_resolve_strategy_id should map transform_mode when no entity override."""
        # Test each mapping
        for mode, expected_strategy in [
            ("anonymize", "placeholder"),
            ("pseudonymize", "tokenize"),
            ("redact", "redact"),
            ("generalize", "generalize"),
            ("synthetic", "synthetic"),
            ("perturb", "perturb"),
        ]:
            profile = ProcessingProfileSpec(
                profile_id="test",
                transform_mode=mode,
                mode="weighted_consensus",
            )
            strategy_id = orchestrator._resolve_strategy_id("ANY_TYPE", profile)
            assert strategy_id == expected_strategy

    def test_resolve_strategy_unknown_mode_returns_mode_as_is(self, orchestrator):
        """_resolve_strategy_id should return unknown mode as strategy ID."""
        profile = ProcessingProfileSpec(
            profile_id="test",
            transform_mode="custom_strategy_id",
            mode="weighted_consensus",
        )
        strategy_id = orchestrator._resolve_strategy_id("ANY_TYPE", profile)
        assert strategy_id == "custom_strategy_id"

    def test_resolve_strategy_profile_overrides_config(self, orchestrator):
        """Profile entity_strategies should override config entity_strategies."""
        orchestrator.config.transform = Mock()
        orchestrator.config.transform.entity_strategies = {"PERSON_NAME": "redact"}

        profile = ProcessingProfileSpec(
            profile_id="test",
            transform_mode="anonymize",
            mode="weighted_consensus",
            entity_strategies={"PERSON_NAME": "synthetic"},
        )
        strategy_id = orchestrator._resolve_strategy_id("PERSON_NAME", profile)
        # Profile should take precedence
        assert strategy_id == "synthetic"


# ============================================================================
# Transform Context Building Tests (Lines 867-903)
# ============================================================================


class TestBuildTransformContext:
    """Test TransformContext building."""

    def test_build_context_basic(self, orchestrator, basic_profile):
        """_build_transform_context should create valid context."""
        decision = Mock()
        decision.canonical_text = "John Smith"
        decision.cluster_id = "person_contact-1"
        decision.placeholder_index = 5

        finding = EnsembleFinding(
            entity_type="PERSON_NAME",
            confidence=0.95,
            engines=["regex"],
            field_path="customer.name",
            span_start=0,
            span_end=10,
        )

        context = orchestrator._build_transform_context(
            decision=decision,
            finding=finding,
            text="John Smith works here",
            scope="user_123",
            token_version=1,
            profile=basic_profile,
            mention_index=0,
        )

        assert isinstance(context, TransformContext)
        assert context.entity_type == "PERSON_NAME"
        assert context.plaintext == "John Smith"
        assert context.scope == "user_123"
        assert context.is_first_mention is True
        assert context.mention_index == 0

    def test_build_context_with_strategy_params(self, orchestrator):
        """_build_transform_context should include strategy params from profile."""
        profile = ProcessingProfileSpec(
            profile_id="test",
            transform_mode="redact",
            mode="weighted_consensus",
            strategy_params={
                "redact": {"mode": "partial_start", "reveal_count": 2}
            },
        )

        decision = Mock()
        decision.canonical_text = "secret"
        decision.cluster_id = "test_cluster"
        decision.placeholder_index = 1

        finding = EnsembleFinding(
            entity_type="PASSWORD",
            confidence=0.9,
            engines=["regex"],
        )

        context = orchestrator._build_transform_context(
            decision=decision,
            finding=finding,
            text="secret",
            scope="default",
            token_version=1,
            profile=profile,
            mention_index=0,
        )

        assert context.strategy_params["mode"] == "partial_start"
        assert context.strategy_params["reveal_count"] == 2

    def test_build_context_merge_config_and_profile_params(self, orchestrator):
        """Config params should be merged with profile params (profile wins)."""
        # Create a mock transform config with entity_strategies and strategy_params
        orchestrator.config.transform = Mock()
        orchestrator.config.transform.entity_strategies = {}
        orchestrator.config.transform.strategy_params = {
            "redact": {"mode": "full", "mask_char": "*"}
        }

        profile = ProcessingProfileSpec(
            profile_id="test",
            transform_mode="redact",
            mode="weighted_consensus",
            strategy_params={
                "redact": {"mode": "partial_start"}  # Override mode
            },
        )

        decision = Mock()
        decision.canonical_text = "value"
        decision.cluster_id = "test"
        decision.placeholder_index = 1

        finding = EnsembleFinding(
            entity_type="DATA",
            confidence=0.8,
            engines=["engine"],
        )

        context = orchestrator._build_transform_context(
            decision=decision,
            finding=finding,
            text="value",
            scope="default",
            token_version=1,
            profile=profile,
            mention_index=0,
        )

        # Profile should override config
        assert context.strategy_params["mode"] == "partial_start"
        # Config param should still be there if not overridden
        assert context.strategy_params["mask_char"] == "*"

    def test_build_context_non_first_mention(self, orchestrator, basic_profile):
        """is_first_mention should be False for mention_index > 0."""
        decision = Mock()
        decision.canonical_text = "John"
        decision.cluster_id = "person_1"
        decision.placeholder_index = 2

        finding = EnsembleFinding(
            entity_type="PERSON_NAME",
            confidence=0.9,
            engines=["regex"],
        )

        context = orchestrator._build_transform_context(
            decision=decision,
            finding=finding,
            text="John goes to John's store",
            scope="default",
            token_version=1,
            profile=basic_profile,
            mention_index=2,
        )

        assert context.is_first_mention is False
        assert context.mention_index == 2


# ============================================================================
# Placeholder Formatting Tests (Lines 1062-1072)
# ============================================================================


class TestFormatPlaceholder:
    """Test placeholder template formatting."""

    def test_format_placeholder_valid_template(self):
        """_format_placeholder should use valid template."""
        result = AsyncPIIOrchestrator._format_placeholder(
            template="<{entity_type}:anon_{index}>",
            entity_type="EMAIL_ADDRESS",
            index=5,
            cluster_id="email_1",
        )
        assert result == "<EMAIL_ADDRESS:anon_5>"

    def test_format_placeholder_with_cluster_id(self):
        """_format_placeholder should support cluster_id in template."""
        result = AsyncPIIOrchestrator._format_placeholder(
            template="{cluster_id}_{index}",
            entity_type="PHONE",
            index=3,
            cluster_id="contact_42",
        )
        assert result == "contact_42_3"

    def test_format_placeholder_invalid_template_fallback(self):
        """_format_placeholder should fallback on format error."""
        result = AsyncPIIOrchestrator._format_placeholder(
            template="<{invalid_var}>",
            entity_type="ADDRESS",
            index=2,
            cluster_id="addr_1",
        )
        # Should fallback to default
        assert result == "<ADDRESS:anon_2>"

    def test_format_placeholder_malformed_brace_fallback(self):
        """_format_placeholder should handle malformed braces."""
        result = AsyncPIIOrchestrator._format_placeholder(
            template="<{entity_type",
            entity_type="NAME",
            index=1,
            cluster_id="name_1",
        )
        assert result == "<NAME:anon_1>"


# ============================================================================
# Confidence Envelope Tests (Lines 1074-1120)
# ============================================================================


class TestConfidenceEnvelope:
    """Test confidence envelope calculation."""

    def test_confidence_envelope_empty_findings(self, orchestrator):
        """_confidence_envelope should handle empty findings list."""
        envelope = orchestrator._confidence_envelope([])

        assert envelope.score == 0.0
        assert envelope.risk_level == "high"
        assert "No findings" in envelope.notes[0]

    def test_confidence_envelope_single_finding(self, orchestrator):
        """_confidence_envelope with single finding."""
        findings = [
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.95,
                engines=["regex"],
            )
        ]

        envelope = orchestrator._confidence_envelope(findings)

        assert envelope.score == 0.95
        assert envelope.risk_level == "low"
        assert envelope.contributors == ["regex"]

    def test_confidence_envelope_multiple_findings(self, orchestrator):
        """_confidence_envelope should macro-average confidence."""
        findings = [
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.90,
                engines=["regex"],
            ),
            EnsembleFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.80,
                engines=["regex", "spacy"],
            ),
        ]

        envelope = orchestrator._confidence_envelope(findings)

        # Average of (0.90 + 0.80) / 2 = 0.85
        assert envelope.score == 0.85
        assert envelope.risk_level == "moderate"

    def test_confidence_envelope_low_confidence(self, orchestrator):
        """_confidence_envelope should mark low scores as high risk."""
        findings = [
            EnsembleFinding(
                entity_type="UNKNOWN",
                confidence=0.6,
                engines=["unknown_engine"],
            )
        ]

        envelope = orchestrator._confidence_envelope(findings)

        assert envelope.score == 0.6
        assert envelope.risk_level == "high"

    def test_confidence_envelope_per_entity_type_breakdown(self, orchestrator):
        """_confidence_envelope should include per-entity-type scores."""
        findings = [
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.95,
                engines=["regex"],
            ),
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.85,
                engines=["spacy"],
            ),
            EnsembleFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.90,
                engines=["regex"],
            ),
        ]

        envelope = orchestrator._confidence_envelope(findings)

        # PERSON_NAME average: (0.95 + 0.85) / 2 = 0.90
        # EMAIL_ADDRESS average: 0.90
        assert envelope.by_entity_type["PERSON_NAME"] == 0.9
        assert envelope.by_entity_type["EMAIL_ADDRESS"] == 0.9

    def test_confidence_envelope_multiple_engines_deduped(self, orchestrator):
        """Contributors should be deduplicated and sorted."""
        findings = [
            EnsembleFinding(
                entity_type="PERSON_NAME",
                confidence=0.9,
                engines=["regex", "spacy", "regex"],  # regex repeated
            ),
            EnsembleFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.85,
                engines=["spacy", "presidio"],
            ),
        ]

        envelope = orchestrator._confidence_envelope(findings)

        # Should be deduplicated and sorted
        assert envelope.contributors == ["presidio", "regex", "spacy"]


# ============================================================================
# Finding Dict Conversion Tests (Lines 1122-1146)
# ============================================================================


class TestFindingDictConversion:
    """Test finding-dict conversions."""

    def test_finding_to_dict(self):
        """_finding_to_dict should convert EnsembleFinding to dict."""
        finding = EnsembleFinding(
            entity_type="PERSON_NAME",
            confidence=0.95,
            engines=["regex", "spacy"],
            field_path="customer.name",
            span_start=0,
            span_end=10,
            explanation="Matched name pattern",
            language="en",
        )

        result = AsyncPIIOrchestrator._finding_to_dict(finding)

        assert result["entity_type"] == "PERSON_NAME"
        assert result["confidence"] == 0.95
        assert result["engines"] == ["regex", "spacy"]
        assert result["field_path"] == "customer.name"
        assert result["span"] == {"start": 0, "end": 10}
        assert result["explanation"] == "Matched name pattern"
        assert result["language"] == "en"

    def test_finding_to_dict_with_none_spans(self):
        """_finding_to_dict should handle None spans."""
        finding = EnsembleFinding(
            entity_type="TYPE",
            confidence=0.8,
            engines=["engine"],
        )

        result = AsyncPIIOrchestrator._finding_to_dict(finding)

        assert result["span"] == {"start": None, "end": None}

    def test_dict_to_finding(self):
        """_dict_to_finding should convert dict back to EnsembleFinding."""
        data = {
            "entity_type": "EMAIL_ADDRESS",
            "confidence": 0.92,
            "engines": ["regex"],
            "field_path": "contact.email",
            "span": {"start": 5, "end": 22},
            "explanation": "Email pattern",
            "language": "en",
        }

        result = AsyncPIIOrchestrator._dict_to_finding(data)

        assert result.entity_type == "EMAIL_ADDRESS"
        assert result.confidence == 0.92
        assert result.engines == ["regex"]
        assert result.field_path == "contact.email"
        assert result.span_start == 5
        assert result.span_end == 22

    def test_dict_to_finding_missing_fields(self):
        """_dict_to_finding should handle missing fields."""
        data = {}

        result = AsyncPIIOrchestrator._dict_to_finding(data)

        assert result.entity_type == "UNKNOWN"
        assert result.confidence == 0.0
        assert result.engines == []
        assert result.field_path is None


# ============================================================================
# Cluster Entity Type Tests (Lines 1055-1059)
# ============================================================================


class TestClusterEntityType:
    """Test cluster-based entity type resolution."""

    def test_cluster_entity_type_person_contact_cluster(self):
        """_cluster_entity_type should return PERSON_NAME for person_contact clusters."""
        result = AsyncPIIOrchestrator._cluster_entity_type(
            "person_contact-john-smith", "EMAIL_ADDRESS"
        )
        assert result == "PERSON_NAME"

    def test_cluster_entity_type_other_cluster(self):
        """_cluster_entity_type should return fallback for non-person-contact clusters."""
        result = AsyncPIIOrchestrator._cluster_entity_type(
            "email_cluster_1", "EMAIL_ADDRESS"
        )
        assert result == "EMAIL_ADDRESS"

    def test_cluster_entity_type_no_prefix_match(self):
        """_cluster_entity_type should use fallback if no person_contact- prefix."""
        result = AsyncPIIOrchestrator._cluster_entity_type(
            "person_info_1", "PHONE_NUMBER"
        )
        assert result == "PHONE_NUMBER"


# ============================================================================
# List Engines Tests (Line 258-260)
# ============================================================================


class TestListEngines:
    """Test engine listing."""

    def test_list_engines_include_disabled(self, orchestrator):
        """list_engines with include_disabled=True should list all."""
        ids = orchestrator.list_engines(include_disabled=True)
        # Should include all registered engines
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_list_engines_exclude_disabled(self, orchestrator):
        """list_engines with include_disabled=False should list only enabled."""
        ids = orchestrator.list_engines(include_disabled=False)
        # Should return list of enabled engine IDs
        assert isinstance(ids, list)


# ============================================================================
# Strategy Listing Tests (Lines 215-223)
# ============================================================================


class TestListStrategies:
    """Test strategy listing."""

    def test_list_strategies_returns_sorted_ids(self, orchestrator):
        """list_strategies should return sorted strategy IDs."""
        strategies = orchestrator.list_strategies()
        assert isinstance(strategies, list)
        # Should contain built-in strategies
        assert "placeholder" in strategies
        assert "redact" in strategies
        assert "generalize" in strategies

    def test_list_strategies_registered_order(self, orchestrator):
        """list_strategies should return strategies in sorted order."""
        strategies = orchestrator.list_strategies()
        assert strategies == sorted(strategies)


# ============================================================================
# Unregister Strategy Tests (Lines 205-213)
# ============================================================================


class TestUnregisterStrategy:
    """Test strategy unregistration."""

    def test_unregister_strategy_removes_from_registry(self, orchestrator):
        """unregister_strategy should remove strategy."""
        # Get initial count
        initial = orchestrator.list_strategies()
        initial_count = len(initial)

        # Unregister one
        orchestrator.unregister_strategy("redact")

        # Should have one fewer
        after = orchestrator.list_strategies()
        assert len(after) == initial_count - 1
        assert "redact" not in after

    def test_unregister_strategy_can_reregister(self, orchestrator):
        """Should be able to reregister a strategy after unregistering."""
        orchestrator.unregister_strategy("redact")
        assert "redact" not in orchestrator.list_strategies()

        # Reregister
        orchestrator.register_strategy(RedactionStrategy())
        assert "redact" in orchestrator.list_strategies()


# ============================================================================
# Apply Transform Tests - No Findings (Lines 935-936)
# ============================================================================


class TestApplyTransformNoFindings:
    """Test apply_transform with no findings."""

    def test_apply_transform_empty_findings(self, orchestrator, basic_profile):
        """_apply_transform should return original text when no findings."""
        text = "John works at ACME Corp"
        result_text, audit = orchestrator._apply_transform(
            text=text,
            findings=[],
            scope="user_1",
            token_version=1,
            profile=basic_profile,
        )

        assert result_text == text
        assert audit == []


# ============================================================================
# Offset Findings Tests (Lines 537-551)
# ============================================================================


class TestOffsetFindings:
    """Test finding offset adjustments."""

    def test_offset_findings_positive_offset(self):
        """_offset_findings should add offset to span positions."""
        findings = [
            EnsembleFinding(
                entity_type="NAME",
                confidence=0.9,
                engines=["test"],
                span_start=5,
                span_end=10,
            ),
            EnsembleFinding(
                entity_type="EMAIL",
                confidence=0.8,
                engines=["test"],
                span_start=15,
                span_end=25,
            ),
        ]

        AsyncPIIOrchestrator._offset_findings(findings, 100)

        assert findings[0].span_start == 105
        assert findings[0].span_end == 110
        assert findings[1].span_start == 115
        assert findings[1].span_end == 125

    def test_offset_findings_with_none_spans(self):
        """_offset_findings should handle None spans."""
        findings = [
            EnsembleFinding(
                entity_type="NAME",
                confidence=0.9,
                engines=["test"],
            )
        ]

        # Should not crash
        AsyncPIIOrchestrator._offset_findings(findings, 50)
        assert findings[0].span_start is None
        assert findings[0].span_end is None


# ============================================================================
# Offset Audits Tests (Lines 545-551)
# ============================================================================


class TestOffsetAudits:
    """Test audit record offset adjustments."""

    def test_offset_audits_positive_offset(self):
        """_offset_audits should add offset to span positions."""
        audit = Mock()
        audit.span_start = 5
        audit.span_end = 10

        audits = [audit]

        AsyncPIIOrchestrator._offset_audits(audits, 100)

        assert audits[0].span_start == 105
        assert audits[0].span_end == 110

    def test_offset_audits_with_none_spans(self):
        """_offset_audits should handle None spans."""
        audit = Mock()
        audit.span_start = None
        audit.span_end = None

        audits = [audit]

        # Should not crash
        AsyncPIIOrchestrator._offset_audits(audits, 50)
        assert audits[0].span_start is None
        assert audits[0].span_end is None


# ============================================================================
# Resolve Execution Plan Tests (Lines 727-756)
# ============================================================================


class TestResolveExecutionPlan:
    """Test execution plan resolution."""

    def test_resolve_execution_plan_basic(self, orchestrator, basic_profile):
        """_resolve_execution_plan should return ExecutionPlan."""
        payload = {"name": "John Smith", "email": "john@example.com"}
        plan = orchestrator._resolve_execution_plan(payload=payload, profile=basic_profile)

        assert plan is not None
        assert hasattr(plan, "plan_id")
        assert hasattr(plan, "engine_ids")

    def test_resolve_execution_plan_has_engine_ids(self, orchestrator, basic_profile):
        """_resolve_execution_plan should set engine_ids."""
        payload = {"text": "test"}
        plan = orchestrator._resolve_execution_plan(payload=payload, profile=basic_profile)

        # Should have at least one engine
        assert isinstance(plan.engine_ids, list)
        assert len(plan.engine_ids) > 0


# ============================================================================
# Engines for Plan Tests (Lines 758-762)
# ============================================================================


class TestEnginesForPlan:
    """Test engine selection for execution plan."""

    def test_engines_for_plan_selects_correct_engines(self, orchestrator):
        """_engines_for_plan should select engines matching plan IDs."""
        plan = Mock()
        plan.engine_ids = ["regex-oss"]

        engines = orchestrator._engines_for_plan(plan)
        assert len(engines) > 0
        assert any(e.adapter_id == "regex-oss" for e in engines)


# ============================================================================
# Merge with Audit Tests (Lines 764-772)
# ============================================================================


class TestMergeWithAudit:
    """Test merging with audit generation."""

    def test_merge_with_audit_returns_tuple(self, orchestrator):
        """_merge_with_audit should return (merged, audits) tuple."""
        from pii_anon.fusion import build_fusion

        fusion = build_fusion(
            "weighted_consensus",
            weights={"regex-oss": 1.0},
            min_consensus=1,
        )
        findings = [
            EngineFinding(
                entity_type="PERSON_NAME",
                confidence=0.9,
                engine_id="regex-oss",
                field_path="name",
                span_start=0,
                span_end=4,
            ),
        ]

        merged, audits = orchestrator._merge_with_audit(fusion=fusion, raw_findings=findings)

        assert isinstance(merged, list)
        assert isinstance(audits, list)


# ============================================================================
# Confidence Envelope with Custom Thresholds
# ============================================================================


class TestConfidenceEnvelopeWithCustomThresholds:
    """Test confidence envelope with custom risk thresholds."""

    def test_confidence_envelope_custom_low_threshold(self, orchestrator):
        """_confidence_envelope should respect custom low_risk_threshold."""
        orchestrator.config.risk = Mock()
        orchestrator.config.risk.low_risk_threshold = 0.85
        orchestrator.config.risk.moderate_risk_threshold = 0.70

        findings = [
            EnsembleFinding(
                entity_type="TYPE",
                confidence=0.80,
                engines=["engine"],
            ),
        ]

        envelope = orchestrator._confidence_envelope(findings)

        # 0.80 is below 0.85 but above 0.70, so moderate
        assert envelope.risk_level == "moderate"

    def test_confidence_envelope_default_thresholds(self, orchestrator):
        """_confidence_envelope should use default thresholds when not configured."""
        # Ensure config.risk doesn't exist
        if hasattr(orchestrator.config, "risk"):
            delattr(orchestrator.config, "risk")

        findings = [
            EnsembleFinding(
                entity_type="TYPE",
                confidence=0.80,
                engines=["engine"],
            ),
        ]

        envelope = orchestrator._confidence_envelope(findings)

        # With defaults: >= 0.90 is low, >= 0.75 is moderate
        # 0.80 should be moderate
        assert envelope.risk_level == "moderate"


# ============================================================================
# Synchronous Wrapper Tests
# ============================================================================


class TestPIIOrchestratorSync:
    """Test synchronous orchestrator wrapper."""

    def test_sync_orchestrator_initialization(self):
        """PIIOrchestrator should initialize without errors."""
        orchestrator = PIIOrchestrator(token_key="test-key")
        assert orchestrator is not None
        # Sync wrapper doesn't expose token_key directly, check it has async attr
        assert hasattr(orchestrator, "_async")

    def test_sync_orchestrator_init_with_config_path(self, tmp_path):
        """PIIOrchestrator should initialize with config_path."""
        # Create a minimal config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
logging:
  level: INFO
engines:
  regex-oss:
    weight: 1.0
""")

        orchestrator = PIIOrchestrator(
            token_key="test-key",
            config_path=str(config_file),
        )

        assert orchestrator is not None
        assert hasattr(orchestrator, "_async")
