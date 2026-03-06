"""Orchestration of PII detection, fusion, and transformation pipelines.

The orchestrator coordinates the entire detection workflow:

1. **Engine management**: Register/discover detection engines.
2. **Detection**: Run enabled engines on payloads (serial or parallel).
3. **Fusion**: Merge findings from multiple engines using configurable strategies.
4. **Transformation**: Replace/tokenize detected entities.
5. **Tracking**: Map entity mentions to identity clusters for consistency.

Two implementations are provided:

- ``AsyncPIIOrchestrator``: Async/await interface (recommended).
- ``PIIOrchestrator``: Synchronous interface (wraps async via threading).

Typical usage:

.. code-block:: python

    from pii_anon import PIIOrchestrator, ProcessingProfileSpec, SegmentationPlan

    orchestrator = PIIOrchestrator(token_key="my-secret-key")
    profile = ProcessingProfileSpec(
        profile_id="default",
        mode="weighted_consensus",
    )
    payload = {"email": "john@example.com", "phone": "555-1234"}

    result = orchestrator.run(
        payload,
        profile=profile,
        segmentation=SegmentationPlan(),
        scope="user_123",
        token_version=1,
    )
"""

from __future__ import annotations

import asyncio
import time
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any, Coroutine, TypeVar, cast

from pii_anon.config import ConfigManager, CoreConfig
from pii_anon.engines import (
    EngineAdapter,
    EngineRegistry,
    GLiNERAdapter,
    LLMGuardAdapter,
    PresidioAdapter,
    RegexEngineAdapter,
    SpacyNERAdapter,
    StanzaNERAdapter,
    ScrubadubAdapter,
)
from pii_anon.errors import EngineExecutionError
from pii_anon.fusion import build_fusion, build_fusion_audit
from pii_anon.observability import get_logger
from pii_anon.policy import ExecutionPlan, PolicyRouter
from pii_anon.segmentation import BoundaryReconciler, Segmenter
from pii_anon.tracking import IdentityLedger, link_findings
from pii_anon.tokenization import (
    DeterministicHMACTokenizer,
    InMemoryTokenStore,
    TokenStore,
    TokenizerProvider,
)
from pii_anon.transforms import (
    GeneralizationStrategy,
    PerturbationStrategy,
    PlaceholderStrategy,
    RedactionStrategy,
    StrategyRegistry,
    SyntheticReplacementStrategy,
    TokenizationStrategy,
    TransformContext,
    TransformStrategy,
)
from pii_anon.types import (
    ConfidenceEnvelope,
    EngineFinding,
    EnsembleFinding,
    ProcessingProfileSpec,
    FusionAuditRecord,
    Payload,
    RiskLevel,
    SegmentationPlan,
)

T = TypeVar("T")


class AsyncPIIOrchestrator:
    """Async orchestrator for PII detection, fusion, and transformation.

    Manages the full detection-to-transformation pipeline. Engines are run
    in parallel, findings are fused, entities are tracked for consistency,
    and output is transformed according to the processing profile.

    Parameters
    ----------
    token_key : str
        Secret key for HMAC tokenization (used in pseudonymization mode).
    config : CoreConfig | None
        Application configuration. If *None*, uses default config.
    tokenizer : TokenizerProvider | None
        Tokenization provider. If *None*, uses deterministic HMAC.
    token_store : TokenStore | None
        Token storage backend. If *None*, uses in-memory store.

    Attributes
    ----------
    registry : EngineRegistry
        Manages registered detection engines.
    config : CoreConfig
        Application configuration.
    identity_ledger : IdentityLedger
        Tracks entity mentions across documents for consistency.
    router : PolicyRouter
        Selects execution plans based on profile and payload.

    Notes
    -----
    This class should be instantiated once per application and reused
    for multiple payloads. Engine discovery and initialization happen
    during ``__init__``.
    """
    def __init__(
        self,
        token_key: str,
        *,
        config: CoreConfig | None = None,
        tokenizer: TokenizerProvider | None = None,
        token_store: TokenStore | None = None,
    ) -> None:
        self.token_key = token_key
        self.segmenter = Segmenter()
        self.reconciler = BoundaryReconciler()
        self.tokenizer = tokenizer or DeterministicHMACTokenizer()
        self.token_store = token_store or InMemoryTokenStore()
        self.config = config or CoreConfig.default()
        self.identity_ledger = IdentityLedger()
        self.router = PolicyRouter()
        self.logger = get_logger(
            "pii_anon.orchestrator",
            level=self.config.logging.level,
            structured=self.config.logging.structured,
        )
        self._semaphore = asyncio.Semaphore(max(1, self.config.stream.max_concurrency))

        self.registry = EngineRegistry()
        self._register_default_engines()
        if self.config.auto_discover_engines:
            self.registry.discover_entrypoint_engines()
        self.registry.initialize({k: v.model_dump() for k, v in self.config.engines.items()})

        self.strategy_registry = StrategyRegistry()
        self._register_default_strategies()

    @classmethod
    def from_config_path(
        cls,
        token_key: str,
        *,
        config_path: str | None = None,
        tokenizer: TokenizerProvider | None = None,
        token_store: TokenStore | None = None,
    ) -> "AsyncPIIOrchestrator":
        config = ConfigManager().load(config_path)
        return cls(token_key, config=config, tokenizer=tokenizer, token_store=token_store)

    def _register_default_engines(self) -> None:
        self.register_engine(RegexEngineAdapter(enabled=True))
        self.register_engine(PresidioAdapter(enabled=False))
        self.register_engine(LLMGuardAdapter(enabled=False))
        self.register_engine(ScrubadubAdapter(enabled=False))
        self.register_engine(SpacyNERAdapter(enabled=False))
        self.register_engine(StanzaNERAdapter(enabled=False))
        self.register_engine(GLiNERAdapter(enabled=False))

    def _register_default_strategies(self) -> None:
        """Register the six built-in transformation strategies."""
        self.strategy_registry.register(PlaceholderStrategy())
        self.strategy_registry.register(TokenizationStrategy())
        self.strategy_registry.register(RedactionStrategy())
        self.strategy_registry.register(GeneralizationStrategy())
        self.strategy_registry.register(SyntheticReplacementStrategy())
        self.strategy_registry.register(PerturbationStrategy())

    def register_strategy(self, strategy: TransformStrategy) -> None:
        """Register a custom transformation strategy.

        Parameters
        ----------
        strategy : TransformStrategy
            The strategy to register.
        """
        self.strategy_registry.register(strategy)

    def unregister_strategy(self, strategy_id: str) -> None:
        """Unregister a transformation strategy.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        """
        self.strategy_registry.unregister(strategy_id)

    def list_strategies(self) -> list[str]:
        """List registered transformation strategies.

        Returns
        -------
        list[str]
            Strategy identifiers.
        """
        return self.strategy_registry.list_strategies()

    def register_engine(self, engine: EngineAdapter) -> None:
        """Register a detection engine.

        Parameters
        ----------
        engine : EngineAdapter
            The engine to register.
        """
        self.registry.register(engine)

    def unregister_engine(self, adapter_id: str) -> None:
        """Unregister a detection engine.

        Parameters
        ----------
        adapter_id : str
            Engine identifier.
        """
        self.registry.unregister(adapter_id)

    def list_engines(self, *, include_disabled: bool = True) -> list[str]:
        """List registered engines.

        Parameters
        ----------
        include_disabled : bool
            Whether to include disabled engines (default True).

        Returns
        -------
        list[str]
            Engine identifiers.
        """
        if include_disabled:
            return self.registry.ids()
        return [engine.adapter_id for engine in self.registry.list_engines(include_disabled=False)]

    def discover_engines(self) -> list[str]:
        """Discover and register engines from entrypoints.

        Returns
        -------
        list[str]
            Newly discovered engine identifiers.
        """
        return self.registry.discover_entrypoint_engines()

    def health_check_engines(self) -> dict[str, dict[str, Any]]:
        """Get health status of all registered engines.

        Returns
        -------
        dict[str, dict[str, Any]]
            Per-engine health reports.
        """
        return self.registry.health_report()

    def capabilities(self) -> dict[str, dict[str, Any]]:
        """Report capabilities of all registered engines.

        Returns
        -------
        dict[str, dict[str, Any]]
            Per-engine capability info (languages, streaming support, etc.).
        """
        report = self.registry.capabilities_report()
        return {
            adapter_id: {
                "native_dependency": caps.native_dependency,
                "dependency_available": caps.dependency_available,
                "supports_languages": caps.supports_languages,
                "supports_streaming": caps.supports_streaming,
                "supports_runtime_configuration": caps.supports_runtime_configuration,
            }
            for adapter_id, caps in report.items()
        }

    async def run(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> dict[str, Any]:
        """Run detection and transformation on a payload.

        Orchestrates the full pipeline: detects PII, fuses results, and
        transforms matched entities.

        Parameters
        ----------
        payload : Payload
            Input data (field names to scalar values).
        profile : ProcessingProfileSpec
            Detection and transformation configuration.
        segmentation : SegmentationPlan
            Text chunking strategy for large documents.
        scope : str
            Tokenization scope (e.g., user ID).
        token_version : int
            Token version for consistency tracking.

        Returns
        -------
        dict[str, Any]
            Result dict with keys:

            - ``transformed_payload``: anonymized/pseudonymized output.
            - ``ensemble_findings``: detected entities (list of dicts).
            - ``confidence_envelope``: aggregate confidence metrics.
            - ``fusion_audit``: lineage of fusion decisions.
            - ``boundary_trace``: segmentation reconciliation stats.
            - ``execution_plan``: plan used (for debugging).
            - ``link_audit``: entity tracking decisions.
        """
        detect = await self.detect_only(
            payload,
            profile=profile,
            segmentation=segmentation,
            scope=scope,
            token_version=token_version,
        )
        findings: list[EnsembleFinding] = [self._dict_to_finding(item) for item in detect["ensemble_findings"]]
        transformed_payload: Payload = dict(payload)
        link_audit: list[dict[str, Any]] = []
        by_field: dict[str, list[EnsembleFinding]] = {}
        for finding in findings:
            if finding.field_path is None:
                continue
            by_field.setdefault(finding.field_path, []).append(finding)
        for field, value in payload.items():
            if not isinstance(value, str):
                continue
            transformed_value, field_audit = self._apply_transform(
                text=value,
                findings=by_field.get(field, []),
                scope=scope,
                token_version=token_version,
                profile=profile,
            )
            transformed_payload[field] = transformed_value
            for item in field_audit:
                item["field_path"] = field
            link_audit.extend(field_audit)
        return {
            "transformed_payload": transformed_payload,
            "ensemble_findings": detect["ensemble_findings"],
            "confidence_envelope": detect["confidence_envelope"],
            "fusion_audit": detect["fusion_audit"],
            "boundary_trace": detect["boundary_trace"],
            "execution_plan": detect["execution_plan"],
            "link_audit": link_audit,
        }

    async def detect_only(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> dict[str, Any]:
        """Run detection without transformation.

        Useful for inspection and audit. Returns findings and confidence
        metrics without modifying the payload.

        Parameters
        ----------
        payload : Payload
            Input data.
        profile : ProcessingProfileSpec
            Detection configuration.
        segmentation : SegmentationPlan
            Text chunking strategy.
        scope : str
            Tokenization scope.
        token_version : int
            Token version.

        Returns
        -------
        dict[str, Any]
            Result dict with keys: ``ensemble_findings``, ``confidence_envelope``,
            ``fusion_audit``, ``boundary_trace``, ``execution_plan``.
        """
        flat, fusion_audit, boundary_trace, plan = await self.detect_findings(
            payload,
            profile=profile,
            segmentation=segmentation,
            scope=scope,
            token_version=token_version,
        )
        envelope = self._confidence_envelope(flat)
        return {
            "ensemble_findings": [self._finding_to_dict(x) for x in flat],
            "confidence_envelope": {
                "score": envelope.score,
                "risk_level": envelope.risk_level,
                "contributors": envelope.contributors,
                "notes": envelope.notes,
                "by_entity_type": envelope.by_entity_type,
            },
            "fusion_audit": [self._audit_to_dict(item) for item in fusion_audit],
            "boundary_trace": boundary_trace,
            "execution_plan": self._execution_plan_to_dict(plan),
        }

    async def detect_findings(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> tuple[list[EnsembleFinding], list[FusionAuditRecord], dict[str, Any] | None, ExecutionPlan]:
        """Low-level detection: return findings without transformation.

        Parameters
        ----------
        payload : Payload
            Input data.
        profile : ProcessingProfileSpec
            Detection configuration.
        segmentation : SegmentationPlan
            Text chunking strategy.
        scope : str
            Tokenization scope.
        token_version : int
            Token version.

        Returns
        -------
        tuple[list[EnsembleFinding], list[FusionAuditRecord], dict[str, Any] | None, ExecutionPlan]
            Tuple of (findings, fusion audit records, boundary reconciliation trace, execution plan).
        """
        del scope
        del token_version
        findings_batches: list[list[EnsembleFinding]] = []
        boundary_trace: dict[str, Any] | None = None
        fusion_audit: list[FusionAuditRecord] = []
        plan = self._resolve_execution_plan(payload=payload, profile=profile)

        for field, value in payload.items():
            if not isinstance(value, str):
                continue

            should_segment = (segmentation.enabled or plan.segmentation_enabled) and len(value.split()) > segmentation.max_tokens
            if should_segment:
                findings, boundary_trace, audits = await self._detect_segmented_field(
                    field=field,
                    text=value,
                    profile=profile,
                    segmentation=segmentation,
                    plan=plan,
                )
            else:
                findings, audits, _source = await self._detect_on_text_field_async(
                    field,
                    value,
                    profile,
                    plan=plan,
                )

            findings_batches.append(findings)
            fusion_audit.extend(audits)

        flat = [item for batch in findings_batches for item in batch]
        return flat, fusion_audit, boundary_trace, plan

    async def _detect_segmented_field(
        self,
        *,
        field: str,
        text: str,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        plan: ExecutionPlan,
    ) -> tuple[list[EnsembleFinding], dict[str, Any], list[FusionAuditRecord]]:
        segments = self.segmenter.segment(
            text,
            max_tokens=segmentation.max_tokens,
            overlap_tokens=segmentation.overlap_tokens,
        )

        segment_findings: list[list[EnsembleFinding]] = []
        audits: list[FusionAuditRecord] = []
        for segment in segments:
            findings, segment_audit, _raw = await self._detect_on_text_field_async(
                field,
                segment.text,
                profile,
                plan=plan,
            )
            self._offset_findings(findings, segment.start_char)
            self._offset_audits(segment_audit, segment.start_char)
            segment_findings.append(findings)
            audits.extend(segment_audit)

        merged, trace = self.reconciler.reconcile(segment_findings, segmentation.overlap_tokens)
        boundary_trace = {
            "segments_processed": trace.segments_processed,
            "overlap_tokens": trace.overlap_tokens,
            "merged_spans": trace.merged_spans,
            "deduped_findings": trace.deduped_findings,
        }
        return merged, boundary_trace, audits

    @staticmethod
    def _offset_findings(findings: list[EnsembleFinding], offset: int) -> None:
        for finding in findings:
            if finding.span_start is not None:
                finding.span_start += offset
            if finding.span_end is not None:
                finding.span_end += offset

    @staticmethod
    def _offset_audits(audits: list[FusionAuditRecord], offset: int) -> None:
        for item in audits:
            if item.span_start is not None:
                item.span_start += offset
            if item.span_end is not None:
                item.span_end += offset

    async def _detect_on_text_field_async(
        self,
        field: str,
        text: str,
        profile: ProcessingProfileSpec,
        *,
        plan: ExecutionPlan,
    ) -> tuple[list[EnsembleFinding], list[FusionAuditRecord], list[EngineFinding]]:
        payload: Payload = {field: text}
        engines = self._engines_for_plan(plan)
        context = {
            "policy_mode": profile.policy_mode,
            "language": profile.language or self.config.default_language,
        }

        raw_findings: list[EngineFinding] = []
        if len(engines) == 1:
            try:
                raw_findings = engines[0].detect(payload, context)
            except Exception as exc:
                self.logger.warning(
                    "engine_execution_error",
                    extra={"payload": {"error": str(exc), "type": type(exc).__name__}},
                )
                raw_findings = []
            # Fast-path for single-engine execution plans where escalation is disabled.
            # This avoids fusion/audit overhead on strict speed profiles.
            if profile.objective == "speed" and not plan.escalate_on_low_confidence:
                merged = [
                    EnsembleFinding(
                        entity_type=item.entity_type,
                        confidence=item.confidence,
                        engines=[item.engine_id],
                        field_path=item.field_path,
                        span_start=item.span_start,
                        span_end=item.span_end,
                        explanation=item.explanation,
                        language=item.language,
                    )
                    for item in raw_findings
                ]
                return merged, [], raw_findings
        else:
            tasks = [
                asyncio.create_task(self._run_engine_detect_async(engine, payload, profile))
                for engine in engines
            ]

            if tasks:
                results: list[list[EngineFinding] | BaseException] = await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        self.logger.warning(
                            "engine_execution_error",
                            extra={"payload": {"error": str(result), "type": type(result).__name__}},
                        )
                        continue
                    raw_findings.extend(result)

        fusion = build_fusion(
            plan.fusion_mode,
            weights=self._resolve_weights(profile),
            min_consensus=profile.min_consensus,
        )
        merged, audits = self._merge_with_audit(fusion=fusion, raw_findings=raw_findings)
        merged, audits, raw_findings = await self._maybe_escalate(
            payload=payload,
            profile=profile,
            plan=plan,
            merged=merged,
            audits=audits,
            raw_findings=raw_findings,
        )
        return merged, audits, raw_findings

    async def _run_engine_detect_async(
        self,
        engine: EngineAdapter,
        payload: Payload,
        profile: ProcessingProfileSpec,
    ) -> list[EngineFinding]:
        async with self._semaphore:
            try:
                return await asyncio.to_thread(
                    engine.detect,
                    payload,
                    {
                        "policy_mode": profile.policy_mode,
                        "language": profile.language or self.config.default_language,
                    },
                )
            except Exception as exc:
                raise EngineExecutionError(f"Engine `{engine.adapter_id}` failed") from exc

    async def run_stream_async(
        self,
        payloads: AsyncIterable[Payload] | Iterable[Payload],
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream detection and transformation over multiple payloads.

        Yields results one at a time, enabling memory-efficient processing
        of large datasets.

        Parameters
        ----------
        payloads : AsyncIterable[Payload] | Iterable[Payload]
            Input stream (async or sync iterable).
        profile : ProcessingProfileSpec
            Detection configuration.
        segmentation : SegmentationPlan
            Text chunking strategy.
        scope : str
            Tokenization scope.
        token_version : int
            Token version.

        Yields
        ------
        dict[str, Any]
            Result dicts (same format as ``run()``).
        """
        if isinstance(payloads, AsyncIterable):
            async for payload in payloads:
                yield await self.run(
                    payload,
                    profile=profile,
                    segmentation=segmentation,
                    scope=scope,
                    token_version=token_version,
                )
            return

        for payload in payloads:
            yield await self.run(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope=scope,
                token_version=token_version,
            )

    def _resolve_weights(self, profile: ProcessingProfileSpec) -> dict[str, float]:
        if profile.engine_weights:
            return profile.engine_weights

        out: dict[str, float] = {}
        for adapter_id, cfg in self.config.engines.items():
            out[adapter_id] = cfg.weight
        return out

    def _resolve_execution_plan(
        self,
        *,
        payload: Payload,
        profile: ProcessingProfileSpec,
    ) -> ExecutionPlan:
        capabilities = dict(self.capabilities())
        external_ids = {"spacy-ner-compatible", "stanza-ner-compatible"}
        policy = self.config.competitor_policy
        runtime_external_allowed = policy.enabled and policy.runtime_leverage_enabled and profile.use_external_competitors
        allowlist = set(profile.external_competitor_allowlist or policy.allowed_adapters)
        if not runtime_external_allowed:
            for adapter_id in external_ids:
                capabilities.pop(adapter_id, None)
        else:
            for adapter_id in list(external_ids):
                if allowlist and adapter_id not in allowlist:
                    capabilities.pop(adapter_id, None)

        plan = self.router.select(payload, profile, capabilities)
        enabled_ids = {engine.adapter_id for engine in self.registry.list_engines(include_disabled=False)}
        if plan.plan_id == "default_compat":
            plan.engine_ids = sorted(enabled_ids)
        plan.engine_ids = [engine_id for engine_id in plan.engine_ids if engine_id in enabled_ids]
        if not plan.engine_ids:
            plan.engine_ids = sorted(enabled_ids) if enabled_ids else ["regex-oss"]
        plan.escalation_engine_ids = [
            engine_id for engine_id in plan.escalation_engine_ids if engine_id in enabled_ids and engine_id not in plan.engine_ids
        ]
        return plan

    def _engines_for_plan(self, plan: ExecutionPlan) -> list[EngineAdapter]:
        engines = self.registry.list_engines(include_disabled=False)
        by_id = {engine.adapter_id: engine for engine in engines}
        selected = [by_id[engine_id] for engine_id in plan.engine_ids if engine_id in by_id]
        return selected or engines

    def _merge_with_audit(
        self,
        *,
        fusion: Any,
        raw_findings: list[EngineFinding],
    ) -> tuple[list[EnsembleFinding], list[FusionAuditRecord]]:
        merged = fusion.merge(raw_findings)
        audits = build_fusion_audit(fusion, merged, raw_findings)
        return merged, audits

    async def _maybe_escalate(
        self,
        *,
        payload: Payload,
        profile: ProcessingProfileSpec,
        plan: ExecutionPlan,
        merged: list[EnsembleFinding],
        audits: list[FusionAuditRecord],
        raw_findings: list[EngineFinding],
    ) -> tuple[list[EnsembleFinding], list[FusionAuditRecord], list[EngineFinding]]:
        if not plan.escalate_on_low_confidence or not plan.escalation_engine_ids:
            return merged, audits, raw_findings
        if not merged:
            needs_escalation = True
        else:
            envelope = self._confidence_envelope(merged)
            needs_escalation = envelope.score < plan.low_confidence_threshold
        if not needs_escalation:
            return merged, audits, raw_findings

        engines = {engine.adapter_id: engine for engine in self.registry.list_engines(include_disabled=False)}
        escalation_engines = [engines[engine_id] for engine_id in plan.escalation_engine_ids if engine_id in engines]
        if not escalation_engines:
            return merged, audits, raw_findings

        tasks = [
            asyncio.create_task(self._run_engine_detect_async(engine, payload, profile))
            for engine in escalation_engines
        ]
        if not tasks:
            return merged, audits, raw_findings
        results: list[list[EngineFinding] | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)
        extra: list[EngineFinding] = []
        for result in results:
            if isinstance(result, BaseException):
                self.logger.warning(
                    "engine_escalation_error",
                    extra={"payload": {"error": str(result), "type": type(result).__name__}},
                )
                continue
            extra.extend(result)
        if not extra:
            return merged, audits, raw_findings

        all_raw = [*raw_findings, *extra]
        fusion = build_fusion(
            plan.fusion_mode,
            weights=self._resolve_weights(profile),
            min_consensus=profile.min_consensus,
        )
        escalated_merged, escalated_audit = self._merge_with_audit(fusion=fusion, raw_findings=all_raw)
        return escalated_merged, escalated_audit, all_raw

    # ── Strategy resolution helpers ────────────────────────────────────

    _TRANSFORM_MODE_TO_STRATEGY: dict[str, str] = {
        "anonymize": "placeholder",
        "pseudonymize": "tokenize",
        "redact": "redact",
        "generalize": "generalize",
        "synthetic": "synthetic",
        "perturb": "perturb",
    }

    def _resolve_strategy_id(
        self,
        entity_type: str,
        profile: ProcessingProfileSpec,
    ) -> str:
        """Determine the strategy ID for a given entity type and profile.

        Resolution order:
        1. ``profile.entity_strategies[entity_type]`` (per-type override)
        2. ``config.transform.entity_strategies[entity_type]`` (config override)
        3. ``profile.transform_mode`` mapped to strategy ID
        """
        # Per-type override from profile
        if entity_type in profile.entity_strategies:
            return profile.entity_strategies[entity_type]

        # Per-type override from config
        config_strats = getattr(self.config.transform, "entity_strategies", {})
        if entity_type in config_strats:
            return str(config_strats[entity_type])

        # Global transform_mode → strategy mapping
        return self._TRANSFORM_MODE_TO_STRATEGY.get(
            profile.transform_mode, profile.transform_mode
        )

    def _build_transform_context(
        self,
        *,
        decision: Any,
        finding: EnsembleFinding,
        text: str,
        scope: str,
        token_version: int,
        profile: ProcessingProfileSpec,
        mention_index: int,
    ) -> TransformContext:
        """Build a TransformContext for a single mention."""
        strategy_id = self._resolve_strategy_id(finding.entity_type, profile)
        params = dict(profile.strategy_params.get(strategy_id, {}))
        # Merge config-level strategy params
        config_params = getattr(self.config.transform, "strategy_params", {})
        if strategy_id in config_params:
            merged = dict(config_params[strategy_id])
            merged.update(params)
            params = merged

        return TransformContext(
            entity_type=finding.entity_type,
            plaintext=decision.canonical_text,
            field_path=finding.field_path,
            language=finding.language,
            scope=scope,
            finding=finding,
            cluster_id=decision.cluster_id,
            placeholder_index=decision.placeholder_index,
            is_first_mention=(mention_index == 0),
            mention_index=mention_index,
            document_text=text,
            token_key=self.token_key,
            token_version=token_version,
            strategy_params=params,
        )

    def _apply_transform(
        self,
        *,
        text: str,
        findings: list[EnsembleFinding],
        scope: str,
        token_version: int,
        profile: ProcessingProfileSpec,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Replace detected PII spans with strategy-determined replacements.

        Builds the output string in O(n) using segment assembly instead
        of repeated string slicing (which was O(n·m) where m = number of
        replacements and n = text length).

        Algorithm:
        1. Link findings to identity clusters via the linker.
        2. Sort decisions by span start (ascending) and deduplicate.
        3. For each finding, resolve the appropriate strategy and call
           ``strategy.transform()`` to obtain the replacement.
        4. Walk through the text, copying unmatched segments and inserting
           replacement values at matched spans.
        5. Join all segments in one ``str.join()`` call.

        Backward Compatibility
        ----------------------
        ``transform_mode="anonymize"`` routes through ``PlaceholderStrategy``,
        ``transform_mode="pseudonymize"`` routes through ``TokenizationStrategy``.
        Both produce identical output to the pre-strategy implementation.
        """
        if not findings:
            return text, []

        tracking_enabled = profile.entity_tracking_enabled and self.config.tracking.enabled
        decisions = link_findings(
            text=text,
            findings=findings,
            ledger=self.identity_ledger,
            scope=scope,
            enabled=tracking_enabled,
            min_link_score=self.config.tracking.min_link_score,
            allow_email_name_link=self.config.tracking.allow_email_name_link,
            require_unique_short_name=self.config.tracking.require_unique_short_name,
        )
        if not decisions:
            return text, []

        # Sort ascending by span position for forward traversal.
        sorted_decisions = sorted(
            decisions,
            key=lambda item: (item.finding.span_start or -1, item.finding.span_end or -1),
        )

        # Deduplicate overlapping spans and collect replacement info.
        seen_ranges: set[tuple[int, int]] = set()
        replacements: list[tuple[int, int, str, dict[str, Any]]] = []

        for mention_idx, decision in enumerate(sorted_decisions):
            finding = decision.finding
            if finding.span_start is None or finding.span_end is None:
                continue
            start = finding.span_start
            end = finding.span_end
            if start < 0 or end > len(text) or start >= end:
                continue
            if (start, end) in seen_ranges:
                continue
            seen_ranges.add((start, end))

            # Resolve strategy for this entity type
            strategy_id = self._resolve_strategy_id(finding.entity_type, profile)
            strategy = self.strategy_registry.get(strategy_id)

            if strategy is not None:
                # Build full context and delegate to strategy
                entity_type_resolved = self._cluster_entity_type(
                    decision.cluster_id, finding.entity_type
                )
                ctx = self._build_transform_context(
                    decision=decision,
                    finding=finding,
                    text=text,
                    scope=scope,
                    token_version=token_version,
                    profile=profile,
                    mention_index=mention_idx,
                )
                # Inject token store for tokenization strategy
                if strategy_id == "tokenize":
                    ctx.strategy_params["token_store"] = self.token_store

                result = strategy.transform(decision.canonical_text, entity_type_resolved, ctx)
                replacement_value = result.replacement
            else:
                # Fallback: use legacy paths for backward compat
                if profile.transform_mode == "anonymize":
                    replacement_value = self._format_placeholder(
                        template=profile.placeholder_template or self.config.transform.placeholder_template,
                        entity_type=self._cluster_entity_type(decision.cluster_id, finding.entity_type),
                        index=decision.placeholder_index,
                        cluster_id=decision.cluster_id,
                    )
                else:
                    token_entity_type = self._cluster_entity_type(decision.cluster_id, finding.entity_type)
                    token = self.tokenizer.tokenize(
                        token_entity_type,
                        decision.canonical_text,
                        scope,
                        token_version,
                        self.token_key,
                        store=self.token_store,
                    )
                    replacement_value = token.token

            audit_entry = {
                "entity_type": finding.entity_type,
                "span": {"start": start, "end": end},
                "mention_text": decision.mention_text,
                "cluster_id": decision.cluster_id,
                "canonical_text": decision.canonical_text,
                "placeholder_index": decision.placeholder_index,
                "score": round(decision.score, 4),
                "rule": decision.rule,
                "transform_mode": profile.transform_mode,
                "strategy_id": strategy_id if strategy else profile.transform_mode,
                "replacement": replacement_value,
            }
            replacements.append((start, end, replacement_value, audit_entry))

        if not replacements:
            return text, []

        # Build output string in O(n) via segment assembly.
        # Walk forward through the text, copying gaps between replacements
        # and inserting replacement values at each span.
        parts: list[str] = []
        link_audit: list[dict[str, Any]] = []
        cursor = 0
        for start, end, replacement_value, audit_entry in replacements:
            if start > cursor:
                parts.append(text[cursor:start])
            parts.append(replacement_value)
            link_audit.append(audit_entry)
            cursor = end
        # Append trailing text after the last replacement
        if cursor < len(text):
            parts.append(text[cursor:])

        return "".join(parts), link_audit

    @staticmethod
    def _cluster_entity_type(cluster_id: str, fallback_entity_type: str) -> str:
        if cluster_id.startswith("person_contact-"):
            return "PERSON_NAME"
        return fallback_entity_type

    @staticmethod
    def _format_placeholder(
        *,
        template: str,
        entity_type: str,
        index: int,
        cluster_id: str,
    ) -> str:
        try:
            return template.format(entity_type=entity_type, index=index, cluster_id=cluster_id)
        except Exception:
            return f"<{entity_type}:anon_{index}>"

    def _confidence_envelope(self, findings: list[EnsembleFinding]) -> ConfidenceEnvelope:
        """Compute aggregate confidence metrics for a batch of findings.

        The envelope provides a single ``score`` (macro-average confidence),
        a ``risk_level`` classification, and per-entity-type breakdowns.

        Risk level thresholds:
        - >= 0.90 → "low" risk (high detection confidence)
        - >= 0.75 → "moderate" risk
        - <  0.75 → "high" risk (low detection confidence)
        """
        if not findings:
            return ConfidenceEnvelope(
                score=0.0,
                risk_level="high",
                notes=["No findings from configured engines"],
            )

        # Macro-average confidence across all findings
        score = sum(item.confidence for item in findings) / max(1, len(findings))
        # Deduplicate engine IDs across all findings
        contributors = sorted({engine for item in findings for engine in item.engines})
        # Per-entity-type average confidence for detailed reporting
        by_type: dict[str, list[float]] = {}
        for item in findings:
            by_type.setdefault(item.entity_type, []).append(item.confidence)
        by_entity_type = {
            entity_type: round(sum(scores) / max(1, len(scores)), 4)
            for entity_type, scores in by_type.items()
        }

        risk_level: RiskLevel
        if score >= 0.9:
            risk_level = "low"
        elif score >= 0.75:
            risk_level = "moderate"
        else:
            risk_level = "high"

        return ConfidenceEnvelope(
            score=round(score, 4),
            risk_level=risk_level,
            contributors=contributors,
            by_entity_type=by_entity_type,
        )

    @staticmethod
    def _finding_to_dict(item: Any) -> dict[str, Any]:
        return {
            "entity_type": item.entity_type,
            "confidence": item.confidence,
            "engines": item.engines,
            "field_path": item.field_path,
            "span": {"start": item.span_start, "end": item.span_end},
            "explanation": item.explanation,
            "language": getattr(item, "language", "en"),
        }

    @staticmethod
    def _dict_to_finding(item: dict[str, Any]) -> EnsembleFinding:
        span = item.get("span", {})
        return EnsembleFinding(
            entity_type=str(item.get("entity_type", "UNKNOWN")),
            confidence=float(item.get("confidence", 0.0)),
            engines=[str(engine) for engine in item.get("engines", [])],
            field_path=item.get("field_path"),
            span_start=span.get("start"),
            span_end=span.get("end"),
            explanation=item.get("explanation"),
            language=str(item.get("language", "en")),
        )

    @staticmethod
    def _audit_to_dict(item: FusionAuditRecord) -> dict[str, Any]:
        return {
            "strategy": item.strategy,
            "entity_type": item.entity_type,
            "field_path": item.field_path,
            "span": {"start": item.span_start, "end": item.span_end},
            "source_engines": item.source_engines,
            "source_count": item.source_count,
            "fused_confidence": item.fused_confidence,
            "notes": item.notes,
        }

    @staticmethod
    def _execution_plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "engine_ids": plan.engine_ids,
            "fusion_mode": plan.fusion_mode,
            "segmentation_enabled": plan.segmentation_enabled,
            "concurrency_cap": plan.concurrency_cap,
            "escalate_on_low_confidence": plan.escalate_on_low_confidence,
            "low_confidence_threshold": plan.low_confidence_threshold,
            "escalation_engine_ids": plan.escalation_engine_ids,
        }


class PIIOrchestrator:
    """Synchronous wrapper around AsyncPIIOrchestrator.

    Provides a familiar synchronous interface for applications that don't
    use async/await. Internally uses threading to run async operations.

    Parameters
    ----------
    token_key : str
        Secret key for HMAC tokenization.
    config : CoreConfig | None
        Application configuration.
    config_path : str | None
        Path to YAML config file (alternative to *config*).
    tokenizer : TokenizerProvider | None
        Tokenization provider.
    token_store : TokenStore | None
        Token storage backend.

    Notes
    -----
    Synchronous methods block until completion. For high-throughput
    scenarios, prefer ``AsyncPIIOrchestrator`` with proper async runtime.
    """
    def __init__(
        self,
        token_key: str,
        *,
        config: CoreConfig | None = None,
        config_path: str | None = None,
        tokenizer: TokenizerProvider | None = None,
        token_store: TokenStore | None = None,
    ) -> None:
        if config is None and config_path is not None:
            config = ConfigManager().load(config_path)

        self._async = AsyncPIIOrchestrator(
            token_key,
            config=config,
            tokenizer=tokenizer,
            token_store=token_store,
        )

    def register_engine(self, engine: EngineAdapter) -> None:
        """Register a detection engine."""
        self._async.register_engine(engine)

    def unregister_engine(self, adapter_id: str) -> None:
        """Unregister a detection engine."""
        self._async.unregister_engine(adapter_id)

    def list_engines(self, *, include_disabled: bool = True) -> list[str]:
        """List registered engines."""
        return self._async.list_engines(include_disabled=include_disabled)

    def discover_engines(self) -> list[str]:
        """Discover and register engines from entrypoints."""
        return self._async.discover_engines()

    def health_check_engines(self) -> dict[str, dict[str, Any]]:
        """Get health status of all registered engines."""
        return self._async.health_check_engines()

    def capabilities(self) -> dict[str, dict[str, Any]]:
        """Report capabilities of all registered engines."""
        return self._async.capabilities()

    async def run_async(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
        ) -> dict[str, Any]:
        """Async version of ``run()``."""
        return await self._async.run(
            payload,
            profile=profile,
            segmentation=segmentation,
            scope=scope,
            token_version=token_version,
        )

    async def detect_only_async(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> dict[str, Any]:
        """Async version of ``detect_only()``."""
        return await self._async.detect_only(
            payload,
            profile=profile,
            segmentation=segmentation,
            scope=scope,
            token_version=token_version,
        )

    async def detect_findings_async(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> tuple[list[EnsembleFinding], list[FusionAuditRecord], dict[str, Any] | None, dict[str, Any]]:
        """Async version of ``detect_findings()``."""
        findings, audits, boundary_trace, plan = await self._async.detect_findings(
            payload,
            profile=profile,
            segmentation=segmentation,
            scope=scope,
            token_version=token_version,
        )
        return findings, audits, boundary_trace, self._async._execution_plan_to_dict(plan)

    def run(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> dict[str, Any]:
        """Run detection and transformation (synchronous).

        See ``AsyncPIIOrchestrator.run()`` for parameters and return value.
        """
        return _run_coroutine_sync(
            self.run_async(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope=scope,
                token_version=token_version,
            )
        )

    def detect_only(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> dict[str, Any]:
        """Run detection without transformation (synchronous).

        See ``AsyncPIIOrchestrator.detect_only()`` for parameters and return value.
        """
        return _run_coroutine_sync(
            self.detect_only_async(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope=scope,
                token_version=token_version,
            )
        )

    def detect_findings(
        self,
        payload: Payload,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> tuple[list[EnsembleFinding], list[FusionAuditRecord], dict[str, Any] | None, dict[str, Any]]:
        """Low-level detection (synchronous).

        See ``AsyncPIIOrchestrator.detect_findings()`` for parameters and return value.
        """
        return _run_coroutine_sync(
            self.detect_findings_async(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope=scope,
                token_version=token_version,
            )
        )

    def run_stream(
        self,
        payloads: Iterable[Payload],
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
    ) -> Iterator[dict[str, Any]]:
        """Stream detection and transformation over multiple payloads (synchronous).

        Yields results one at a time for memory-efficient batch processing.

        Parameters
        ----------
        payloads : Iterable[Payload]
            Input stream.
        profile : ProcessingProfileSpec
            Detection configuration.
        segmentation : SegmentationPlan
            Text chunking strategy.
        scope : str
            Tokenization scope.
        token_version : int
            Token version.

        Yields
        ------
        dict[str, Any]
            Result dicts (same format as ``run()``).
        """
        for payload in payloads:
            yield self.run(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope=scope,
                token_version=token_version,
            )

    def run_file(
        self,
        path: str | Path,
        *,
        profile: ProcessingProfileSpec,
        segmentation: SegmentationPlan,
        scope: str,
        token_version: int,
        ingest_config: Any | None = None,
        output_path: str | Path | None = None,
        output_format: Any | None = None,
    ) -> Any:
        """Process all records in a file and optionally write results.

        Parameters
        ----------
        path:
            Input file (CSV, JSON, JSONL, or TXT).
        profile:
            Processing profile for detection/transformation.
        segmentation:
            Segmentation plan — large records are automatically chunked.
        scope / token_version:
            Tokenization scope and version.
        ingest_config:
            Optional :class:`~pii_anon.ingestion.IngestConfig`.
        output_path:
            If provided, processed results are written to this path.
        output_format:
            Explicit output format; auto-detected from *output_path* if ``None``.

        Returns
        -------
        :class:`~pii_anon.ingestion.FileIngestResult`
        """
        from pii_anon.ingestion import (
            FileIngestResult,
            IngestConfig,
            read_file,
            write_results,
        )
        from pii_anon.segmentation.chunker import estimate_token_count

        config = ingest_config or IngestConfig()
        threshold = self._async.config.stream.large_text_threshold_tokens

        start = time.monotonic()
        records_processed = 0
        records_failed = 0
        total_chars = 0
        total_chunks = 0
        errors: list[str] = []
        results_buffer: list[dict[str, Any]] = []

        for record in read_file(path, config):
            text = record.text
            total_chars += len(text)

            # Use streaming segmentation for large text
            est_tokens = estimate_token_count(text)
            use_segmentation = segmentation.enabled or est_tokens > threshold
            active_seg = SegmentationPlan(
                enabled=use_segmentation,
                max_tokens=segmentation.max_tokens,
                overlap_tokens=segmentation.overlap_tokens,
            ) if use_segmentation and not segmentation.enabled else segmentation

            if use_segmentation and est_tokens > threshold:
                total_chunks += max(1, est_tokens // (segmentation.max_tokens - segmentation.overlap_tokens))
            else:
                total_chunks += 1

            try:
                result = self.run(
                    {config.text_column: text, **{k: str(v) for k, v in record.metadata.items()}},
                    profile=profile,
                    segmentation=active_seg,
                    scope=scope,
                    token_version=token_version,
                )
                result["metadata"] = record.metadata
                result["record_id"] = record.record_id
                results_buffer.append(result)
                records_processed += 1
            except Exception as exc:
                records_failed += 1
                errors.append(f"record {record.record_id}: {exc}")

        elapsed = time.monotonic() - start

        output_str: str | None = None
        if output_path is not None:
            output_str = str(output_path)
            write_results(iter(results_buffer), output_path, fmt=output_format)

        return FileIngestResult(
            input_path=str(path),
            output_path=output_str,
            format=str(config.format or "auto"),
            records_processed=records_processed,
            records_failed=records_failed,
            total_chars=total_chars,
            total_chunks=total_chunks,
            elapsed_seconds=round(elapsed, 4),
            errors=errors,
        )


def _run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    holder: dict[str, object] = {}

    def runner() -> None:
        try:
            holder["result"] = asyncio.run(coroutine)
        except Exception as exc:  # pragma: no cover
            holder["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in holder:
        raise cast(BaseException, holder["error"])
    if "result" not in holder:
        raise RuntimeError("coroutine did not return a result")
    return cast(T, holder["result"])
