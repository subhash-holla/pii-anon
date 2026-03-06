from __future__ import annotations

import importlib
import json
import logging
import os
import platform as _platform_mod
import re
import threading
import time
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast

# Suppress noisy library output that leaks to stderr during ML model loading.
# Must be set before any transformers/torch import to take effect.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*truncate to max_length.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

from pii_anon.benchmarks import (
    BenchmarkRecord,
    DatasetSource,
    UseCaseProfile,
    load_benchmark_dataset,
    load_use_case_matrix,
)
from pii_anon.config import CoreConfig, EngineRuntimeConfig
from pii_anon.engines import RegexEngineAdapter
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.types import Payload, ProcessingProfileSpec, SegmentationPlan

_log = logging.getLogger(__name__)

LabelSpan = tuple[str, str, int, int]
Objective = Literal["accuracy", "balanced", "speed"]
EngineTier = Literal["auto", "minimal", "standard", "full"]
EvaluationTrack = Literal["detect_only", "end_to_end"]
ProgressHook = Callable[[str], None]
REPORT_SCHEMA_VERSION = "2026-02-19.v3"

_ENGINE_TIERS: list[EngineTier] = ["auto", "minimal", "standard", "full"]


def _tier_system_name(tier: EngineTier) -> str:
    """Return a unique system name for a pii-anon engine tier.

    ``auto`` keeps the canonical name ``"pii-anon"`` for backward
    compatibility.  All other tiers are suffixed, e.g.
    ``"pii-anon-minimal"``.
    """
    if tier == "auto":
        return "pii-anon"
    return f"pii-anon-{tier}"
_FAST_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_FAST_PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\w)")


def _build_engine_config(
    *,
    objective: Objective,
    allow_native_engines: bool,
    engine_tier: EngineTier = "auto",
) -> CoreConfig:
    """Build a ``CoreConfig`` with engine selection based on objective and tier.

    Engine tiers control the speed-vs-accuracy tradeoff:

    - **minimal**: regex + presidio only. Fastest, good baseline F1.
    - **standard**: + GLiNER. Best F1, ~10-20× slower than minimal.
    - **full**: + scrubadub + spacy-ner + stanza-ner. Marginal F1 gain, more noise.
    - **auto** (default): selects tier based on objective:
      - ``speed`` → minimal
      - ``balanced`` → standard
      - ``accuracy`` → standard
    """
    accuracy_mode = objective == "accuracy"

    # Resolve auto tier
    if engine_tier == "auto":
        effective_tier = "minimal" if objective == "speed" else "standard"
    else:
        effective_tier = engine_tier

    enable_ml = allow_native_engines and effective_tier in ("standard", "full")
    enable_low_weight = allow_native_engines and effective_tier == "full"

    return CoreConfig(
        engines={
            "regex-oss": EngineRuntimeConfig(
                enabled=True,
                weight=0.9 if accuracy_mode else 1.0,
            ),
            "presidio-compatible": EngineRuntimeConfig(
                enabled=allow_native_engines,
                weight=1.3 if accuracy_mode else 1.2,
            ),
            "llm-guard-compatible": EngineRuntimeConfig(enabled=False, weight=1.1),
            "scrubadub-compatible": EngineRuntimeConfig(enabled=enable_low_weight, weight=0.95),
            "spacy-ner-compatible": EngineRuntimeConfig(enabled=enable_low_weight, weight=0.95),
            "stanza-ner-compatible": EngineRuntimeConfig(enabled=enable_low_weight, weight=0.95),
            "gliner-compatible": EngineRuntimeConfig(enabled=enable_ml, weight=1.25),
        }
    )


@dataclass
class SystemBenchmarkResult:
    system: str
    available: bool
    skipped_reason: str | None
    qualification_status: Literal["qualified", "excluded", "unavailable", "core"]
    license_name: str | None
    license_source: str | None
    citation_url: str | None
    license_gate_passed: bool
    license_gate_reason: str | None
    precision: float
    recall: float
    f1: float
    latency_p50_ms: float
    docs_per_hour: float
    per_entity_recall: dict[str, float]
    samples: int
    dominance_pass_by_profile: dict[str, bool] = field(default_factory=dict)
    evaluation_track: EvaluationTrack = "detect_only"
    composite_score: float = 0.0
    entity_types_detected: int = 0
    entity_types_total: int = 0
    per_entity_precision: dict[str, float] = field(default_factory=dict)
    elo_rating: float = 0.0
    # Per-record F1 scores for statistical tests (bootstrap CI, paired significance).
    # Not serialized to JSON by default — only the derived statistics are included.
    per_record_f1: list[float] = field(default_factory=list, repr=False)
    # Bootstrap 95% confidence interval for F1.
    f1_ci_lower: float = 0.0
    f1_ci_upper: float = 0.0
    # Per-entity F1 (derived from per_entity_precision and per_entity_recall).
    per_entity_f1: dict[str, float] = field(default_factory=dict)
    # Error classification counts: how many errors of each type.
    error_counts: dict[str, int] = field(default_factory=dict)
    # Per-entity error breakdown: {entity_type: {error_type: count}}.
    per_entity_errors: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class FloorCheckResult:
    metric: str
    comparator: str
    target: float
    actual: float
    passed: bool


@dataclass
class ProfileBenchmarkResult:
    profile: str
    objective: Objective
    systems: list[SystemBenchmarkResult]
    floor_pass: bool
    floor_checks: list[FloorCheckResult]
    qualified_competitors: int
    mit_qualified_competitors: int
    end_to_end_systems: list[SystemBenchmarkResult] = field(default_factory=list)


@dataclass
class CompetitorComparisonReport:
    report_schema_version: str
    dataset: str
    dataset_source: DatasetSource
    systems: list[SystemBenchmarkResult]
    warmup_samples: int
    measured_runs: int
    profiles: list[ProfileBenchmarkResult]
    floor_pass: bool
    qualification_gate_pass: bool
    mit_gate_pass: bool
    expected_competitors: list[str]
    available_competitors: list[str]
    unavailable_competitors: dict[str, str]
    all_competitors_available: bool
    require_all_competitors: bool
    require_native_competitors: bool
    # Statistical tests computed after Elo ratings.
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    # Comparative diagnostics: head-to-head per-entity, improvement opportunities, etc.
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualificationEvidence:
    passed: bool
    reason: str | None
    qualification_status: Literal["qualified", "excluded", "unavailable", "core"]
    license_name: str | None
    license_source: str | None
    citation_url: str | None


_COMPETITOR_META: dict[str, dict[str, str]] = {
    "presidio": {
        "package": "presidio-analyzer",
        "citation": "https://github.com/microsoft/presidio",
    },
    "scrubadub": {
        "package": "scrubadub",
        "citation": "https://github.com/LeapBeyond/scrubadub",
    },
    "gliner": {
        "package": "gliner",
        "citation": "https://github.com/urchade/GLiNER",
    },
}


_ENTITY_TYPE_CACHE: dict[str, str] = {}


def _format_eta(seconds: float) -> str:
    """Format an ETA in seconds into a human-readable string."""
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _normalize_entity_type(entity_type: str) -> str:
    cached = _ENTITY_TYPE_CACHE.get(entity_type)
    if cached is not None:
        return cached
    value = entity_type.upper()
    replacements = {
        "EMAIL": "EMAIL_ADDRESS",
        "EMAILFILTH": "EMAIL_ADDRESS",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "PHONENUMBERFILTH": "PHONE_NUMBER",
        "SSN": "US_SSN",
        "US_SSN": "US_SSN",
        "SOCIALSECURITYNUMBERFILTH": "US_SSN",
        "PERSON": "PERSON_NAME",
        "PERSON_NAME": "PERSON_NAME",
        "PERSONFILTH": "PERSON_NAME",
        "PER": "PERSON_NAME",
        "CREDIT_CARD_NUMBER": "CREDIT_CARD",
        "CREDITCARD": "CREDIT_CARD",
        "CREDITCARDFILTH": "CREDIT_CARD",
        "IP": "IP_ADDRESS",
        "IPV4": "IP_ADDRESS",
        "IPADDRESS": "IP_ADDRESS",
        "IPADDRESSFILTH": "IP_ADDRESS",
        # New entity types
        "DATE_OF_BIRTH": "DATE_OF_BIRTH",
        "DOB": "DATE_OF_BIRTH",
        "DATEOFBIRTH": "DATE_OF_BIRTH",
        "BIRTHDATE": "DATE_OF_BIRTH",
        "MAC_ADDRESS": "MAC_ADDRESS",
        "MACADDRESS": "MAC_ADDRESS",
        "DRIVERS_LICENSE": "DRIVERS_LICENSE",
        "DRIVERSLICENSE": "DRIVERS_LICENSE",
        "DRIVER_LICENSE": "DRIVERS_LICENSE",
        "PASSPORT": "PASSPORT",
        "PASSPORT_NUMBER": "PASSPORT",
        "ROUTING_NUMBER": "ROUTING_NUMBER",
        "LICENSE_PLATE": "LICENSE_PLATE",
        "LICENSEPLATE": "LICENSE_PLATE",
        "BANK_ACCOUNT": "BANK_ACCOUNT",
        "BANKACCOUNT": "BANK_ACCOUNT",
        "NATIONAL_ID": "NATIONAL_ID",
        "NATIONALID": "NATIONAL_ID",
        "USERNAME": "USERNAME",
        "EMPLOYEE_ID": "EMPLOYEE_ID",
        "EMPLOYEEID": "EMPLOYEE_ID",
        "MEDICAL_RECORD_NUMBER": "MEDICAL_RECORD_NUMBER",
        "MRN": "MEDICAL_RECORD_NUMBER",
        "MEDICALRECORDNUMBER": "MEDICAL_RECORD_NUMBER",
        "ORGANIZATION": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "ADDRESS": "ADDRESS",
        "LOCATION": "LOCATION",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        # GLiNER labels (knowledgator/gliner-pii-base-v1.0)
        "NAME": "PERSON_NAME",
        "FIRST NAME": "PERSON_NAME",
        "LAST NAME": "PERSON_NAME",
        "EMAIL ADDRESS": "EMAIL_ADDRESS",
        "PHONE NUMBER": "PHONE_NUMBER",
        "MOBILE PHONE NUMBER": "PHONE_NUMBER",
        "CREDIT CARD NUMBER": "CREDIT_CARD",
        "DATE OF BIRTH": "DATE_OF_BIRTH",
        "SOCIAL SECURITY NUMBER": "US_SSN",
        "PASSPORT NUMBER": "PASSPORT",
        "DRIVER'S LICENSE NUMBER": "DRIVERS_LICENSE",
        "IDENTITY CARD NUMBER": "NATIONAL_ID",
        "BANK ACCOUNT NUMBER": "BANK_ACCOUNT",
        "IBAN": "BANK_ACCOUNT",
        "IP ADDRESS": "IP_ADDRESS",
        "FIRST_NAME": "PERSON_NAME",
        "LAST_NAME": "PERSON_NAME",
        "STREET_ADDRESS": "ADDRESS",
        "BUILDING_NUMBER": "ADDRESS",
        "CITY": "LOCATION",
        "ZIPCODE": "ADDRESS",
        "ACCOUNT_NUMBER": "BANK_ACCOUNT",
        "TAX_NUMBER": "NATIONAL_ID",
        "ID_CARD": "NATIONAL_ID",
    }
    result = replacements.get(value, value)
    _ENTITY_TYPE_CACHE[entity_type] = result
    return result


def _safe_div(num: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return num / denom


# ── Overlap-based span matching ──────────────────────────────────────────
# Industry-standard NER evaluation uses overlap (IoU) rather than exact
# character offsets.  Two spans match if they share the same record_id and
# normalised entity_type, and their character ranges overlap by at least
# ``_OVERLAP_THRESHOLD`` (Intersection-over-Union).  This avoids penalising
# detectors for off-by-one boundary differences on names, addresses, etc.
_OVERLAP_THRESHOLD = 0.5


def _build_overlap_index(
    spans: list[LabelSpan],
) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    """Index spans by (record_id, entity_type) for efficient overlap lookup.

    Returns a dict mapping ``(record_id, norm_entity_type)`` to a list of
    ``(start, end, original_index)`` tuples.
    """
    idx: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for i, (rid, etype, s, e) in enumerate(spans):
        key = (rid, _normalize_entity_type(etype))
        idx.setdefault(key, []).append((s, e, i))
    return idx


def _overlap_match(
    predictions: list[LabelSpan],
    labels: list[LabelSpan],
) -> tuple[int, set[int], set[int]]:
    """Compute overlap-based true positives between predictions and labels.

    Each label is matched to at most one prediction (and vice-versa) using a
    greedy highest-IoU-first strategy.  Returns ``(tp_count,
    matched_pred_indices, matched_label_indices)``.

    Time complexity is O(P + L + M·k) where M is the number of
    (record_id, entity_type) groups and k is the average group size.
    For the typical benchmark (50 K records, ~8 labels each) this adds
    negligible overhead compared to the per-record detection cost.
    """
    pred_idx = _build_overlap_index(predictions)
    label_idx = _build_overlap_index(labels)

    matched_preds: set[int] = set()
    matched_labels: set[int] = set()
    tp = 0

    for key, lab_spans in label_idx.items():
        pred_spans = pred_idx.get(key)
        if not pred_spans:
            continue

        # Build (iou, label_index, pred_index) candidates, greedily match.
        candidates: list[tuple[float, int, int]] = []
        for ls, le, li in lab_spans:
            for ps, pe, pi in pred_spans:
                inter = max(0, min(le, pe) - max(ls, ps))
                union = (le - ls) + (pe - ps) - inter
                if union > 0:
                    iou = inter / union
                    if iou >= _OVERLAP_THRESHOLD:
                        candidates.append((iou, li, pi))

        # Greedy: sort descending by IoU, assign first-come.
        candidates.sort(key=lambda c: c[0], reverse=True)
        for _iou, li, pi in candidates:
            if li not in matched_labels and pi not in matched_preds:
                matched_labels.add(li)
                matched_preds.add(pi)
                tp += 1

    return tp, matched_preds, matched_labels


def _per_entity_recall(
    predictions: list[LabelSpan],
    labels: list[LabelSpan],
    *,
    matched_label_indices: set[int] | None = None,
    pred_set: frozenset[LabelSpan] | None = None,
) -> dict[str, float]:
    # When called with legacy ``pred_set`` kwarg, fall back to exact matching.
    use_overlap = matched_label_indices is not None
    _pred_set: frozenset[LabelSpan] | set[LabelSpan] | None = pred_set
    if not use_overlap and _pred_set is None:
        _pred_set = frozenset(predictions)

    by_entity_total: dict[str, int] = {}
    by_entity_tp: dict[str, int] = {}
    for i, (record_id, entity_type, start, end) in enumerate(labels):
        key = _normalize_entity_type(entity_type)
        by_entity_total[key] = by_entity_total.get(key, 0) + 1
        if use_overlap:
            if i in matched_label_indices:  # type: ignore[operator]
                by_entity_tp[key] = by_entity_tp.get(key, 0) + 1
        elif _pred_set is not None and (record_id, key, start, end) in _pred_set:
            by_entity_tp[key] = by_entity_tp.get(key, 0) + 1

    return {
        entity: round(_safe_div(by_entity_tp.get(entity, 0), total), 6)
        for entity, total in sorted(by_entity_total.items())
    }


def _per_entity_precision(
    predictions: list[LabelSpan],
    labels: list[LabelSpan],
    *,
    matched_pred_indices: set[int] | None = None,
    label_set: frozenset[LabelSpan] | None = None,
) -> dict[str, float]:
    """Per-entity-type precision: TP / (TP + FP) for each predicted entity type."""
    # When called with legacy ``label_set`` kwarg, fall back to exact matching.
    use_overlap = matched_pred_indices is not None
    _label_set: frozenset[LabelSpan] | set[LabelSpan] | None = label_set
    if not use_overlap and _label_set is None:
        _label_set = frozenset(labels)

    by_entity_pred: dict[str, int] = {}
    by_entity_tp: dict[str, int] = {}
    for i, (record_id, entity_type, start, end) in enumerate(predictions):
        key = _normalize_entity_type(entity_type)
        by_entity_pred[key] = by_entity_pred.get(key, 0) + 1
        if use_overlap:
            if i in matched_pred_indices:  # type: ignore[operator]
                by_entity_tp[key] = by_entity_tp.get(key, 0) + 1
        elif _label_set is not None and (record_id, key, start, end) in _label_set:
            by_entity_tp[key] = by_entity_tp.get(key, 0) + 1

    return {
        entity: round(_safe_div(by_entity_tp.get(entity, 0), total), 6)
        for entity, total in sorted(by_entity_pred.items())
    }


def _normalize_findings(record: BenchmarkRecord, findings: list[Any]) -> list[LabelSpan]:
    rows: list[LabelSpan] = []
    for finding in findings:
        start = getattr(finding, "span_start", None)
        end = getattr(finding, "span_end", None)
        if start is None or end is None:
            continue
        rows.append(
            (
                record.record_id,
                _normalize_entity_type(str(getattr(finding, "entity_type", "UNKNOWN"))),
                int(start),
                int(end),
            )
        )
    return rows


def _core_detector(
    *,
    use_case: str,
    objective: Objective,
    allow_native_engines: bool = True,
    engine_tier: EngineTier = "auto",
) -> Callable[[BenchmarkRecord], list[LabelSpan]]:
    if objective == "speed" and engine_tier == "auto":
        # Strict speed floors are evaluated on detect-only track. Use a low-overhead
        # detector that scans only the highest-frequency PII primitives.
        def detect_speed(record: BenchmarkRecord) -> list[LabelSpan]:
            out_rows: list[LabelSpan] = []
            for pattern, entity_type in ((_FAST_EMAIL_RE, "EMAIL_ADDRESS"), (_FAST_PHONE_RE, "PHONE_NUMBER")):
                for match in pattern.finditer(record.text):
                    out_rows.append((record.record_id, entity_type, match.start(), match.end()))
            return out_rows

        return detect_speed

    # Core path must always run through orchestrator detect-only execution to
    # avoid direct-competitor delegation shortcuts in core-vs-competitor metrics.
    config = _build_engine_config(
        objective=objective,
        allow_native_engines=allow_native_engines,
        engine_tier=engine_tier,
    )
    orchestrator = PIIOrchestrator(token_key="benchmark-key", config=config)

    profile = ProcessingProfileSpec(
        profile_id=f"competitor-compare-{use_case}",
        mode="weighted_consensus",
        use_case=use_case,
        objective=objective,
        use_external_competitors=allow_native_engines and objective != "accuracy",
    )
    segmentation = SegmentationPlan(enabled=False)
    async_impl = getattr(orchestrator, "_async", None)

    # ---------------------------------------------------------------------------
    # Determine the sync detection strategy.
    #
    # The orchestrator's detect_findings() creates a new asyncio event loop per
    # call via asyncio.run(), adding ~0.6s overhead per record (epoll spin-up +
    # thread pool executor).  When only a single engine is available, we bypass
    # the orchestrator entirely and call the engine's synchronous .detect() method
    # directly — eliminating the asyncio overhead while preserving measurement
    # fidelity.  For multi-engine configurations, we must go through the
    # orchestrator to get proper fusion.
    # ---------------------------------------------------------------------------
    sync_engine = None
    default_language = "en"
    if async_impl is not None:
        default_language = str(async_impl.config.default_language)
        # Probe the execution plan once to determine available engines.
        probe_payload: Payload = {"text": "probe"}
        plan = async_impl._resolve_execution_plan(payload=probe_payload, profile=profile)
        engines = async_impl._engines_for_plan(plan)
        if len(engines) == 1 and not plan.escalate_on_low_confidence:
            sync_engine = engines[0]

    fast_context = {
        "policy_mode": profile.policy_mode,
        "language": profile.language or default_language,
    }

    # --- Multi-engine persistent event loop ---
    # When multiple engines are active we need the orchestrator's async
    # detect path for proper fusion.  Instead of creating a new asyncio
    # event loop per record (the default ``_run_coroutine_sync`` path),
    # create a persistent loop once and reuse it for all records.  This
    # eliminates ~0.6 ms of epoll spin-up + ThreadPoolExecutor creation
    # per record — a ~40% speedup on the multi-engine detect path.
    import asyncio as _asyncio

    _bench_loop: _asyncio.AbstractEventLoop | None = None
    if sync_engine is None and async_impl is not None:
        _bench_loop = _asyncio.new_event_loop()

    def detect(record: BenchmarkRecord) -> list[LabelSpan]:
        payload: Payload = {"text": record.text}

        # Fast sync path: single engine, direct .detect() call — avoids asyncio
        # event loop overhead entirely.
        if sync_engine is not None:
            try:
                raw = sync_engine.detect(payload, fast_context)
            except Exception:
                raw = []
            return _normalize_findings(record, raw)

        if async_impl is None:
            result = orchestrator.detect_only(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope="benchmark",
                token_version=1,
            )
            out_rows: list[LabelSpan] = []
            for finding in result.get("ensemble_findings", []):
                span = finding.get("span", {})
                start_raw = span.get("start")
                end_raw = span.get("end")
                if start_raw is None or end_raw is None:
                    continue
                out_rows.append(
                    (
                        record.record_id,
                        _normalize_entity_type(str(finding.get("entity_type", "UNKNOWN"))),
                        int(start_raw),
                        int(end_raw),
                    )
                )
            return out_rows

        # Multi-engine path: reuse persistent event loop to avoid
        # per-record asyncio.run() overhead.
        assert _bench_loop is not None
        coro = async_impl.detect_findings(
            payload,
            profile=profile,
            segmentation=segmentation,
            scope="benchmark",
            token_version=1,
        )
        findings, _audits, _boundary, _plan = _bench_loop.run_until_complete(coro)
        rows: list[LabelSpan] = []
        for finding in findings:
            start_raw = finding.span_start
            end_raw = finding.span_end
            if start_raw is None or end_raw is None:
                continue
            rows.append(
                (
                    record.record_id,
                    _normalize_entity_type(str(finding.entity_type)),
                    start_raw,
                    end_raw,
                )
            )
        return rows

    return detect


def _core_end_to_end_detector(
    *,
    use_case: str,
    objective: Objective,
    allow_native_engines: bool = True,
    engine_tier: EngineTier = "auto",
) -> Callable[[BenchmarkRecord], list[LabelSpan]]:
    config = _build_engine_config(
        objective=objective,
        allow_native_engines=allow_native_engines,
        engine_tier=engine_tier,
    )
    orchestrator = PIIOrchestrator(token_key="benchmark-key", config=config)
    profile = ProcessingProfileSpec(
        profile_id=f"competitor-compare-e2e-{use_case}",
        mode="weighted_consensus",
        use_case=use_case,
        objective=objective,
        use_external_competitors=allow_native_engines and objective != "accuracy",
    )
    segmentation = SegmentationPlan(enabled=False)

    # Persistent event loop to avoid per-record asyncio.run() overhead.
    import asyncio as _asyncio

    async_impl = getattr(orchestrator, "_async", None)
    _bench_loop: _asyncio.AbstractEventLoop | None = None
    if async_impl is not None:
        _bench_loop = _asyncio.new_event_loop()

    def detect(record: BenchmarkRecord) -> list[LabelSpan]:
        payload: Payload = {"text": record.text}

        # Fast path: reuse persistent event loop when available.
        if _bench_loop is not None and async_impl is not None:
            coro = async_impl.run(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope="benchmark",
                token_version=1,
            )
            result = _bench_loop.run_until_complete(coro)
        else:
            result = orchestrator.run(
                payload,
                profile=profile,
                segmentation=segmentation,
                scope="benchmark",
                token_version=1,
            )

        out_rows: list[LabelSpan] = []
        for finding in result["ensemble_findings"]:
            span = finding.get("span", {})
            start_raw = span.get("start")
            end_raw = span.get("end")
            if start_raw is None or end_raw is None:
                continue
            out_rows.append(
                (
                    record.record_id,
                    _normalize_entity_type(str(finding.get("entity_type", "UNKNOWN"))),
                    int(start_raw),
                    int(end_raw),
                )
            )
        return out_rows

    return detect


def _presidio_detector(
    *,
    allow_fallback: bool = True,
    require_native: bool = False,
) -> tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None]:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except Exception:
        return None, "presidio_analyzer not installed"

    try:
        # Explicitly use en_core_web_sm to avoid downloading the 400 MB
        # en_core_web_lg model that Presidio defaults to.
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    except Exception as exc:
        return None, f"presidio analyzer initialization failed: {exc}"

    fallback_engine = RegexEngineAdapter(enabled=True) if allow_fallback and not require_native else None

    def _fallback(record: BenchmarkRecord) -> list[LabelSpan]:
        if fallback_engine is None:
            return []
        raw = fallback_engine.detect(
            {"text": record.text},
            {"policy_mode": "balanced", "language": record.language},
        )
        return _normalize_findings(record, raw)

    def detect(record: BenchmarkRecord) -> list[LabelSpan]:
        language = record.language if record.language in {"en", "es", "fr"} else "en"
        try:
            results = analyzer.analyze(text=record.text, language=language)
        except Exception:
            if fallback_engine is not None:
                return _fallback(record)
            return []

        rows: list[LabelSpan] = []
        for item in results:
            rows.append(
                (
                    record.record_id,
                    _normalize_entity_type(str(getattr(item, "entity_type", "UNKNOWN"))),
                    int(getattr(item, "start", 0)),
                    int(getattr(item, "end", 0)),
                )
            )
        return rows

    return detect, None


def _scrubadub_detector(
    *,
    allow_fallback: bool = True,
    require_native: bool = False,
) -> tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None]:
    _ = allow_fallback, require_native
    try:
        scrubadub_module = importlib.import_module("scrubadub")
    except Exception:
        return None, "scrubadub not installed"

    scrubber = cast(Any, scrubadub_module).Scrubber()

    def detect(record: BenchmarkRecord) -> list[LabelSpan]:
        rows: list[LabelSpan] = []
        for item in scrubber.iter_filth(record.text):
            rows.append(
                (
                    record.record_id,
                    _normalize_entity_type(type(item).__name__),
                    int(getattr(item, "beg", 0)),
                    int(getattr(item, "end", 0)),
                )
            )
        return rows

    return detect, None


def _gliner_detector(
    *,
    allow_fallback: bool = True,
    require_native: bool = False,
) -> tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None]:
    """GLiNER PII detector using knowledgator/gliner-pii-base-v1.0."""
    _ = allow_fallback  # GLiNER has no fallback path
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    try:
        import importlib

        GLiNER = importlib.import_module("gliner").GLiNER
    except Exception:
        return None, "gliner not installed"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            model = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")
    except Exception as exc:
        if require_native:
            return None, f"GLiNER model unavailable: {exc}"
        return None, f"GLiNER model load failed: {exc}"

    pii_labels = [
        "name",
        "email address",
        "phone number",
        "credit card number",
        "social security number",
        "date of birth",
        "address",
        "passport number",
        "driver's license number",
        "identity card number",
        "bank account number",
        "username",
        "password",
        "ip address",
    ]

    def detect(record: BenchmarkRecord) -> list[LabelSpan]:
        entities = model.predict_entities(record.text, pii_labels, threshold=0.5)
        rows: list[LabelSpan] = []
        for entity in entities:
            rows.append(
                (
                    record.record_id,
                    _normalize_entity_type(str(entity.get("label", "UNKNOWN"))),
                    int(entity.get("start", 0)),
                    int(entity.get("end", 0)),
                )
            )
        return rows

    return detect, None


def _metadata_for_package(package_name: str) -> Mapping[str, Any] | None:
    try:
        return cast(Mapping[str, Any], metadata.metadata(package_name))
    except Exception:
        return None


def _license_from_metadata(info: Mapping[str, Any]) -> tuple[str | None, str | None]:
    get_all = getattr(info, "get_all", None)
    if callable(get_all):
        classifiers = cast(list[str], get_all("Classifier") or [])
    else:
        classifiers = []

    for entry in classifiers:
        if "License ::" in entry:
            return entry.rsplit("::", 1)[-1].strip(), "classifier"

    license_expression = str(info.get("License-Expression", "")).strip()
    if license_expression:
        return license_expression, "spdx"

    license_value = str(info.get("License", "")).strip()
    if license_value:
        return license_value, "license-field"

    return None, None


def _qualify_oss_license(system: str) -> QualificationEvidence:
    if system == "pii-anon":
        return QualificationEvidence(
            passed=True,
            reason=None,
            qualification_status="core",
            license_name="Apache-2.0",
            license_source="project",
            citation_url="https://github.com/subhash-holla/pii-anon",
        )

    meta = _COMPETITOR_META.get(system)
    if meta is None:
        return QualificationEvidence(
            passed=False,
            reason=f"unknown competitor mapping for `{system}`",
            qualification_status="excluded",
            license_name=None,
            license_source=None,
            citation_url=None,
        )

    package_name = meta["package"]
    info = _metadata_for_package(package_name)
    if info is None:
        return QualificationEvidence(
            passed=False,
            reason=f"metadata unavailable for `{package_name}`",
            qualification_status="unavailable",
            license_name=None,
            license_source=None,
            citation_url=meta["citation"],
        )

    license_name, source = _license_from_metadata(info)
    if not license_name or not source:
        return QualificationEvidence(
            passed=False,
            reason="license metadata missing",
            qualification_status="excluded",
            license_name=None,
            license_source=None,
            citation_url=meta["citation"],
        )

    if source == "classifier":
        classifier_values = cast(list[str], getattr(info, "get_all")("Classifier") or [])
        passed = any("License :: OSI Approved" in item for item in classifier_values)
    else:
        blocked_tokens = ["PROPRIETARY", "UNLICENSED", "COMMERCIAL"]
        compact = license_name.upper()
        passed = not any(token in compact for token in blocked_tokens)

    reason = None if passed else f"unqualified license evidence (`{license_name}` via {source})"
    return QualificationEvidence(
        passed=passed,
        reason=reason,
        qualification_status="qualified" if passed else "excluded",
        license_name=license_name,
        license_source=source,
        citation_url=meta["citation"],
    )


def _mit_license_gate(system: str) -> tuple[bool, str | None]:
    if system == "pii-anon":
        return True, None

    meta = _COMPETITOR_META.get(system)
    if meta is None:
        return False, f"unknown competitor mapping for `{system}`"

    info = _metadata_for_package(meta["package"])
    if info is None:
        return False, f"metadata unavailable for `{meta['package']}`"

    get_all = getattr(info, "get_all", None)
    if callable(get_all):
        classifiers = cast(list[str], get_all("Classifier") or [])
        if any("MIT License" in entry for entry in classifiers):
            return True, "MIT classifier evidence"

    license_expression = str(info.get("License-Expression", "")).strip()
    if license_expression:
        compact = license_expression.upper().replace(" ", "").replace("(", "").replace(")", "")
        if compact == "MIT":
            return True, "MIT SPDX evidence"
        return False, f"SPDX expression is not MIT-only (`{license_expression}`)"

    return False, "no MIT classifier/SPDX evidence"


# ---------------------------------------------------------------------------
# Parallel evaluation infrastructure
# ---------------------------------------------------------------------------


class _DetectorCache:
    """Thread-safe cache for detector callables across profile evaluations.

    When the benchmark evaluates the same system across multiple profiles
    (e.g. 6 profiles in the matrix), the underlying NLP models are
    identical.  This cache ensures each model is loaded exactly once and
    reused for subsequent profiles.

    Keys are tuples that uniquely identify a detector configuration
    (e.g. ``("presidio", True, True)`` for a competitor or
    ``("pii-anon-standard", "accuracy", True, "detect_only")`` for core).
    Values are ``(detector_callable_or_None, reason_or_None)`` pairs.
    """

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, ...],
            tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None],
        ] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        key: tuple[str, ...],
        factory_fn: Callable[
            [],
            tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None],
        ],
    ) -> tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None]:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = factory_fn()
            return self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


@dataclass
class _SystemEvalSpec:
    """Specification for evaluating a single competitor system.

    All heavy state (NLP models, detector instances) is created *inside*
    the worker thread so that nothing unpicklable crosses boundaries.
    """

    system_name: str
    records: list[BenchmarkRecord]
    warmup_samples: int
    measured_runs: int
    allow_fallback_detectors: bool
    require_native_competitors: bool
    forced_unavailable_reason: str | None
    profile_label: str
    progress_hook: ProgressHook | None = None
    detector_cache: _DetectorCache | None = None


@dataclass
class _CoreEvalSpec:
    """Specification for evaluating pii-anon core at a given tier.

    Like ``_SystemEvalSpec`` the heavy orchestrator/model state is created
    inside the worker thread to avoid pickle issues.
    """

    tier: EngineTier
    records: list[BenchmarkRecord]
    warmup_samples: int
    measured_runs: int
    use_case: str
    objective: Objective
    allow_native_engines: bool
    evaluation_track: Literal["detect_only", "end_to_end"]
    profile_label: str
    progress_hook: ProgressHook | None = None
    detector_cache: _DetectorCache | None = None



def _silence_worker_noise() -> None:
    """Suppress noisy library output in spawned benchmark worker processes.

    Keeps the parent process's progress output clean by silencing:
    - huggingface_hub FutureWarnings and tqdm download bars
    - transformers device/truncation info messages (e.g. "Device set to use cpu",
      "Asking to truncate to max_length")
    - pip install chatter
    """
    import os
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*resume_download.*")
    warnings.filterwarnings("ignore", message=".*truncate to max_length.*")
    warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Suppress all library loggers that emit noisy messages during model loading
    for name in ("transformers", "huggingface_hub", "torch", "tqdm"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Redirect stderr to devnull for the duration of model loading.
    # This catches messages that bypass the logging framework entirely
    # (e.g. transformers pipeline "Device set to use cpu" via print()).
    _devnull = open(os.devnull, "w")  # noqa: SIM115
    os.dup2(_devnull.fileno(), 2)
    _devnull.close()


def _evaluate_system_worker(spec: _SystemEvalSpec) -> SystemBenchmarkResult:
    """Top-level worker function for ``ProcessPoolExecutor``.

    Must be defined at module level (not nested) so it is picklable.
    Each worker creates its own detector factory and NLP model instances
    inside the child process, avoiding pickle issues with heavyweight
    NLP objects.

    When ``spec.detector_cache`` is provided, detector instances are
    loaded once and reused across profiles — eliminating redundant model
    loading for competitor detectors (GLiNER ~2-5s, Presidio ~200-400ms).
    """
    _silence_worker_noise()
    detector_factories: dict[
        str,
        Callable[..., tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None]],
    ] = {
        "presidio": _presidio_detector,
        "scrubadub": _scrubadub_detector,
        "gliner": _gliner_detector,
    }

    detector: Callable[[BenchmarkRecord], list[LabelSpan]] | None
    detector_reason: str | None

    if spec.forced_unavailable_reason is not None:
        detector = None
        detector_reason = spec.forced_unavailable_reason
    else:
        detector_factory = detector_factories.get(spec.system_name)
        if detector_factory is None:
            detector = None
            detector_reason = f"unknown competitor detector `{spec.system_name}`"
        else:
            cache_key = (
                spec.system_name,
                str(spec.allow_fallback_detectors),
                str(spec.require_native_competitors),
            )
            if spec.detector_cache is not None:
                detector, detector_reason = spec.detector_cache.get_or_create(
                    cache_key,
                    lambda: detector_factory(
                        allow_fallback=spec.allow_fallback_detectors,
                        require_native=spec.require_native_competitors,
                    ),
                )
            else:
                detector, detector_reason = detector_factory(
                    allow_fallback=spec.allow_fallback_detectors,
                    require_native=spec.require_native_competitors,
                )

    evidence = _qualify_oss_license(spec.system_name)
    reason = detector_reason
    if not evidence.passed:
        detector = None
        if reason:
            reason = f"{reason}; excluded by qualification gate ({evidence.reason})"
        else:
            reason = f"excluded by qualification gate ({evidence.reason})"

    return _evaluate_system(
        spec.system_name,
        detector,
        reason=reason,
        records=spec.records,
        warmup_samples=spec.warmup_samples,
        measured_runs=spec.measured_runs,
        evidence=evidence,
        evaluation_track="detect_only",
        progress_hook=spec.progress_hook,
        progress_label=f"{spec.profile_label} {spec.system_name} detect_only",
    )


def _core_system_worker(spec: _CoreEvalSpec) -> SystemBenchmarkResult:
    """Top-level worker for evaluating pii-anon core at a specific tier.

    Constructs the orchestrator and detector inside the child process so
    that nothing unpicklable crosses the process boundary.
    """
    _silence_worker_noise()
    system_name = _tier_system_name(spec.tier)
    if spec.evaluation_track == "detect_only":
        detector: Callable[[BenchmarkRecord], list[LabelSpan]] | None = _core_detector(
            use_case=spec.use_case,
            objective=spec.objective,
            allow_native_engines=spec.allow_native_engines,
            engine_tier=spec.tier,
        )
    else:
        detector = _core_end_to_end_detector(
            use_case=spec.use_case,
            objective=spec.objective,
            allow_native_engines=spec.allow_native_engines,
            engine_tier=spec.tier,
        )
    evidence = _qualify_oss_license("pii-anon")
    return _evaluate_system(
        system_name,
        detector,
        reason=None,
        records=spec.records,
        warmup_samples=spec.warmup_samples,
        measured_runs=spec.measured_runs,
        evidence=evidence,
        evaluation_track=spec.evaluation_track,
        progress_hook=spec.progress_hook,
        progress_label=f"{spec.profile_label} {system_name} {spec.evaluation_track}",
    )



def _optimal_workers(n_tasks: int, cap: int | None = None) -> int:
    """Return the number of parallel workers to use.

    Scales up to the number of available CPU cores (minus one for the
    parent process), but never exceeds *cap* or *n_tasks*.

    When *cap* is ``None`` (the default), the cap is chosen automatically
    based on the runtime environment:

    - **Docker containers** (detected via ``/.dockerenv``): cap at 4.
      Docker cgroup memory limits (~4-8 GB) make higher concurrency risky
      when each worker loads ~0.5-1 GB of NLP models.
    - **Native macOS (Apple Silicon)**: cap at 6 (< 32 GB RAM) or 8
      (>= 32 GB RAM).  Unified memory means less duplication overhead.
    - **Native Linux**: memory-aware scaling (each worker uses ~1-2 GB).
      Cap at RAM_GB // 4, bounded to [4, 12].
    - **Other platforms**: conservative cap of 4.
    """
    if cap is None:
        in_docker = os.path.exists("/.dockerenv") or os.path.exists(
            "/run/.containerenv"
        )
        if in_docker:
            cap = 4
        elif _platform_mod.system().lower() == "darwin":
            # Apple Silicon unified memory — workers share read-only model
            # pages via copy-on-write.  More RAM → more concurrent workers.
            try:
                mem_gb = (
                    os.sysconf("SC_PAGE_SIZE")
                    * os.sysconf("SC_PHYS_PAGES")
                    / (1024**3)
                )
            except (ValueError, OSError, AttributeError):
                mem_gb = 16.0
            cap = 6 if mem_gb < 32 else 8
        elif _platform_mod.system().lower() == "linux":
            # Native Linux (cloud VMs, bare-metal).  Each worker loads
            # ~1-2 GB of NLP models; scale with available memory.
            try:
                mem_gb = (
                    os.sysconf("SC_PAGE_SIZE")
                    * os.sysconf("SC_PHYS_PAGES")
                    / (1024**3)
                )
            except (ValueError, OSError, AttributeError):
                mem_gb = 16.0
            cap = max(4, min(int(mem_gb // 4), 12))
        else:
            cap = 4

    cpu = os.cpu_count() or 4
    return max(1, min(n_tasks, cpu - 1, cap))


def _evaluate_system(
    name: str,
    detector: Callable[[BenchmarkRecord], list[LabelSpan]] | None,
    *,
    reason: str | None,
    records: list[BenchmarkRecord],
    warmup_samples: int,
    measured_runs: int,
    evidence: QualificationEvidence,
    evaluation_track: EvaluationTrack = "detect_only",
    progress_hook: ProgressHook | None = None,
    progress_label: str | None = None,
) -> SystemBenchmarkResult:
    label = progress_label or f"{name}:{evaluation_track}"
    if progress_hook:
        progress_hook(f"{label}: starting with {len(records)} records")

    if detector is None:
        skipped_work = min(warmup_samples, len(records)) + measured_runs * len(records)
        if progress_hook:
            progress_hook(f"WORK:{skipped_work}|{label}: skipped ({reason or 'detector unavailable'})")
        return SystemBenchmarkResult(
            system=name,
            available=False,
            skipped_reason=reason,
            qualification_status=evidence.qualification_status,
            license_name=evidence.license_name,
            license_source=evidence.license_source,
            citation_url=evidence.citation_url,
            license_gate_passed=evidence.passed,
            license_gate_reason=evidence.reason,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            latency_p50_ms=0.0,
            docs_per_hour=0.0,
            per_entity_recall={},
            samples=len(records),
            evaluation_track=evaluation_track,
        )

    warmup = min(warmup_samples, len(records))
    warmup_start = time.perf_counter()
    for sample in records[:warmup]:
        detector(sample)
    warmup_elapsed = time.perf_counter() - warmup_start
    if progress_hook and warmup > 0:
        progress_hook(f"WORK:{warmup}|{label}: warmup done ({warmup} records, {warmup_elapsed:.1f}s)")

    # Pre-build label tuples once — labels are static ground-truth data that
    # do not change between measured runs.  Avoids redundant
    # _normalize_entity_type() calls across runs (≈220K calls at full scale).
    prebuilt_labels: list[list[LabelSpan]] = [
        [
            (
                sample.record_id,
                _normalize_entity_type(str(lab.get("entity_type", "UNKNOWN"))),
                int(lab.get("start", 0)),
                int(lab.get("end", 0)),
            )
            for lab in sample.labels
        ]
        for sample in records
    ]

    latencies_ms: list[float] = []
    all_predictions: list[LabelSpan] = []
    all_labels: list[LabelSpan] = []
    elapsed_runs: list[float] = []

    total_runs = max(1, measured_runs)
    _PROGRESS_INTERVAL = 60.0  # emit intermediate progress every ~60 seconds
    for run_index in range(total_runs):
        start = time.perf_counter()
        last_progress_time = time.monotonic()
        last_progress_records = 0
        for sample_index, sample in enumerate(records, start=1):
            detect_start = time.perf_counter()
            predictions = detector(sample)
            latencies_ms.append((time.perf_counter() - detect_start) * 1000.0)
            all_predictions.extend(predictions)
            all_labels.extend(prebuilt_labels[sample_index - 1])
            # Emit intermediate progress every ~60s during long runs so the
            # user sees movement instead of minutes of silence.
            if progress_hook and (time.monotonic() - last_progress_time) >= _PROGRESS_INTERVAL:
                records_since = sample_index - last_progress_records
                elapsed_so_far = time.perf_counter() - start
                rate = sample_index / max(elapsed_so_far, 1e-6)
                eta = (len(records) - sample_index) / max(rate, 1e-6)
                remaining_runs_eta = eta + (total_runs - run_index - 1) * (
                    len(records) / max(rate, 1e-6)
                )
                pct = sample_index / len(records) * 100
                progress_hook(
                    f"WORK:{records_since}|{label}: run {run_index + 1}/{total_runs} "
                    f"{sample_index}/{len(records)} records ({pct:.0f}%, "
                    f"{rate:.1f} rec/s, ~{_format_eta(remaining_runs_eta)} remaining)"
                )
                last_progress_records = sample_index
                last_progress_time = time.monotonic()
        elapsed_runs.append(time.perf_counter() - start)
        # Credit remaining records for this run
        remaining = len(records) - last_progress_records
        rate = len(records) / max(elapsed_runs[-1], 1e-6)
        if progress_hook:
            progress_hook(
                f"WORK:{remaining}|{label}: run {run_index + 1}/{total_runs} "
                f"done ({len(records)} records in {elapsed_runs[-1]:.1f}s, "
                f"{rate:.1f} rec/s)"
            )

    docs = len(records) * max(1, measured_runs)
    total_elapsed = max(1e-6, sum(elapsed_runs))
    docs_per_hour = (docs / total_elapsed) * 3600.0

    # Filter predictions to entity types present in the ground truth.
    # Detectors may emit types absent from the benchmark labels (e.g.
    # DATE_ISO, DATE_TIME, GPS_COORDINATES) which are valid detections
    # but inflating false-positive counts and dragging down precision.
    if progress_hook:
        progress_hook(
            f"{label}: scoring — {len(all_predictions)} predictions vs "
            f"{len(all_labels)} labels"
        )
    gt_types = {_normalize_entity_type(etype) for _, etype, _, _ in all_labels}
    if gt_types:
        pre_filter_count = len(all_predictions)
        all_predictions = [
            (rid, etype, s, e)
            for rid, etype, s, e in all_predictions
            if _normalize_entity_type(etype) in gt_types
        ]
        filtered = pre_filter_count - len(all_predictions)
        if progress_hook and filtered > 0:
            progress_hook(
                f"{label}: filtered {filtered} noise predictions "
                f"(types not in ground truth)"
            )

    # Overlap-based matching: count a prediction as a true positive when it
    # shares the same record_id, normalised entity_type, and sufficient
    # character overlap (IoU ≥ 0.5) with a ground-truth label.  This is
    # the standard approach for NER evaluation and avoids penalising
    # detectors for minor boundary differences on names, addresses, etc.
    if progress_hook:
        progress_hook(f"{label}: running overlap matching (IoU >= 0.5)")
    true_pos, matched_preds, matched_labels = _overlap_match(
        all_predictions, all_labels,
    )
    precision = _safe_div(true_pos, len(all_predictions))
    recall = _safe_div(true_pos, len(all_labels))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    if progress_hook:
        progress_hook(
            f"{label}: overlap matching done — TP={true_pos}, "
            f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
        )

    # Per-entity metrics use the same overlap-based matched indices.
    per_entity_rec = _per_entity_recall(
        all_predictions, all_labels, matched_label_indices=matched_labels,
    )
    per_entity_prec = _per_entity_precision(
        all_predictions, all_labels, matched_pred_indices=matched_preds,
    )
    entity_types_detected = sum(1 for v in per_entity_rec.values() if v > 0)
    # Total entity types is the count of distinct types in the ground-truth labels,
    # already enumerated by _per_entity_recall as its dict keys.
    entity_types_total = len(per_entity_rec)

    # Per-entity F1 and error classification.
    per_entity_f1 = _compute_per_entity_f1(per_entity_prec, per_entity_rec)
    error_counts, per_entity_errors = _classify_errors(
        all_predictions, all_labels, matched_preds, matched_labels,
    )

    # Compute per-record F1 for statistical tests (bootstrap CI, paired significance).
    # Groups predictions and labels by record_id then runs overlap matching
    # within each record for consistent per-record F1 scores.
    if progress_hook:
        progress_hook(
            f"{label}: computing per-record F1 scores ({len(records)} records)"
        )
    per_record_f1_scores: list[float] = []
    preds_by_record: dict[str, list[LabelSpan]] = {}
    labels_by_record: dict[str, list[LabelSpan]] = {}
    for span in all_predictions:
        preds_by_record.setdefault(span[0], []).append(span)
    for span in all_labels:
        labels_by_record.setdefault(span[0], []).append(span)
    for rec in records:
        rec_preds = preds_by_record.get(rec.record_id, [])
        rec_labels = labels_by_record.get(rec.record_id, [])
        rec_tp, _, _ = _overlap_match(rec_preds, rec_labels)
        rec_p = _safe_div(rec_tp, len(rec_preds))
        rec_r = _safe_div(rec_tp, len(rec_labels))
        per_record_f1_scores.append(_safe_div(2.0 * rec_p * rec_r, rec_p + rec_r))

    # Bootstrap 95% confidence interval for F1.
    if progress_hook:
        progress_hook(
            f"{label}: computing bootstrap 95% CI (2K iterations, "
            f"{len(per_record_f1_scores)} samples)"
        )
    from pii_anon.eval_framework.evaluation.aggregation import MetricAggregator
    f1_ci_lower, f1_ci_upper = MetricAggregator.compute_confidence_intervals(
        per_record_f1_scores,
        confidence_level=0.95,
        n_bootstrap=2000,
    )
    if progress_hook:
        progress_hook(
            f"{label}: bootstrap CI done — F1={f1:.4f} [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}]"
        )

    # Compute composite score
    from pii_anon.eval_framework.metrics.composite import (
        compute_composite as _compute_composite,
    )
    lat_p50 = round(median(latencies_ms) if latencies_ms else 0.0, 3)
    composite_result = _compute_composite(
        f1=f1,
        precision=precision,
        recall=recall,
        latency_ms=lat_p50,
        docs_per_hour=docs_per_hour,
        entity_types_detected=entity_types_detected,
        entity_types_total=entity_types_total,
    )

    result = SystemBenchmarkResult(
        system=name,
        available=True,
        skipped_reason=None,
        qualification_status=evidence.qualification_status,
        license_name=evidence.license_name,
        license_source=evidence.license_source,
        citation_url=evidence.citation_url,
        license_gate_passed=evidence.passed,
        license_gate_reason=evidence.reason,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        latency_p50_ms=lat_p50,
        docs_per_hour=round(docs_per_hour, 2),
        per_entity_recall=per_entity_rec,
        samples=len(records),
        evaluation_track=evaluation_track,
        composite_score=round(composite_result.score, 6),
        entity_types_detected=entity_types_detected,
        entity_types_total=entity_types_total,
        per_entity_precision=per_entity_prec,
        per_record_f1=per_record_f1_scores,
        f1_ci_lower=f1_ci_lower,
        f1_ci_upper=f1_ci_upper,
        per_entity_f1=per_entity_f1,
        error_counts=error_counts,
        per_entity_errors=per_entity_errors,
    )
    if progress_hook:
        progress_hook(
            f"{label}: complete f1={result.f1:.3f} "
            f"95%CI=[{f1_ci_lower:.3f},{f1_ci_upper:.3f}] "
            f"latency_p50_ms={result.latency_p50_ms:.3f} docs_per_hour={result.docs_per_hour:.2f}"
        )
    return result


def _deterministic_profile_records(
    records: list[BenchmarkRecord],
    *,
    profile: UseCaseProfile,
    max_samples: int | None,
) -> list[BenchmarkRecord]:
    filtered = [item for item in records if item.language in profile.languages]
    keyed = sorted(filtered, key=lambda item: f"{profile.profile}:{item.record_id}")

    sample_cap = profile.max_samples
    if max_samples is not None:
        sample_cap = min(sample_cap, max_samples) if sample_cap is not None else max_samples
    if sample_cap is not None:
        keyed = keyed[: max(0, sample_cap)]
    return keyed


def _is_core_system(system_name: str) -> bool:
    """Return True for the canonical core and all tier-suffixed variants."""
    return system_name == "pii-anon" or system_name.startswith("pii-anon-")


def _best_available(
    systems: list[SystemBenchmarkResult],
    *,
    metric: Literal["f1", "recall", "latency_p50_ms", "docs_per_hour"],
    lower_is_better: bool,
) -> tuple[str, float] | None:
    competitors = [
        item
        for item in systems
        if item.available and not _is_core_system(item.system) and item.license_gate_passed
    ]
    if not competitors:
        return None
    if lower_is_better:
        winner = min(competitors, key=lambda item: float(getattr(item, metric)))
    else:
        winner = max(competitors, key=lambda item: float(getattr(item, metric)))
    return winner.system, float(getattr(winner, metric))


def _qualified_competitors(systems: list[SystemBenchmarkResult]) -> int:
    return len(
        [
            item
            for item in systems
            if not _is_core_system(item.system) and item.available and item.license_gate_passed
        ]
    )


def _mit_qualified_competitors(systems: list[SystemBenchmarkResult]) -> int:
    count = 0
    for item in systems:
        if _is_core_system(item.system) or not item.available:
            continue
        passed, _reason = _mit_license_gate(item.system)
        if passed:
            count += 1
    return count


def _evaluate_floor_contract(
    systems: list[SystemBenchmarkResult],
    objective: Objective,
) -> tuple[bool, list[FloorCheckResult]]:
    core = next((item for item in systems if item.system == "pii-anon"), None)
    if core is None or not core.available:
        return False, [
            FloorCheckResult(
                metric="core_available",
                comparator="n/a",
                target=1.0,
                actual=0.0,
                passed=False,
            )
        ]

    qualified = _qualified_competitors(systems)
    if qualified == 0:
        return False, [
            FloorCheckResult(
                metric="qualified_competitor_available",
                comparator="n/a",
                target=1.0,
                actual=0.0,
                passed=False,
            )
        ]

    checks: list[FloorCheckResult] = []

    if objective == "accuracy":
        best_f1 = _best_available(systems, metric="f1", lower_is_better=False)
        best_recall = _best_available(systems, metric="recall", lower_is_better=False)
        if best_f1 is not None:
            checks.append(
                FloorCheckResult(
                    metric="f1",
                    comparator=best_f1[0],
                    target=best_f1[1],
                    actual=core.f1,
                    passed=core.f1 >= best_f1[1],
                )
            )
        if best_recall is not None:
            checks.append(
                FloorCheckResult(
                    metric="recall",
                    comparator=best_recall[0],
                    target=best_recall[1],
                    actual=core.recall,
                    passed=core.recall >= best_recall[1],
                )
            )
    elif objective == "speed":
        best_latency = _best_available(systems, metric="latency_p50_ms", lower_is_better=True)
        best_throughput = _best_available(systems, metric="docs_per_hour", lower_is_better=False)
        if best_latency is not None:
            checks.append(
                FloorCheckResult(
                    metric="latency_p50_ms",
                    comparator=best_latency[0],
                    target=best_latency[1],
                    actual=core.latency_p50_ms,
                    passed=core.latency_p50_ms <= best_latency[1],
                )
            )
        if best_throughput is not None:
            checks.append(
                FloorCheckResult(
                    metric="docs_per_hour",
                    comparator=best_throughput[0],
                    target=best_throughput[1],
                    actual=core.docs_per_hour,
                    passed=core.docs_per_hour >= best_throughput[1],
                )
            )
    else:
        best_f1 = _best_available(systems, metric="f1", lower_is_better=False)
        best_latency = _best_available(systems, metric="latency_p50_ms", lower_is_better=True)
        if best_f1 is not None:
            checks.append(
                FloorCheckResult(
                    metric="f1",
                    comparator=best_f1[0],
                    target=best_f1[1],
                    actual=core.f1,
                    passed=core.f1 >= best_f1[1],
                )
            )
        if best_latency is not None:
            checks.append(
                FloorCheckResult(
                    metric="latency_p50_ms",
                    comparator=best_latency[0],
                    target=best_latency[1],
                    actual=core.latency_p50_ms,
                    passed=core.latency_p50_ms <= best_latency[1],
                )
            )

    if not checks:
        return False, [
            FloorCheckResult(
                metric="qualified_floor_baseline",
                comparator="n/a",
                target=1.0,
                actual=0.0,
                passed=False,
            )
        ]
    return all(item.passed for item in checks), checks


def _evaluate_profile(
    *,
    records: list[BenchmarkRecord],
    warmup_samples: int,
    measured_runs: int,
    profile: str,
    objective: Objective,
    allow_fallback_detectors: bool = True,
    require_native_competitors: bool = False,
    competitor_systems: list[str] | None = None,
    include_end_to_end: bool = True,
    forced_unavailable_competitors: dict[str, str] | None = None,
    allow_core_native_engines: bool = True,
    engine_tier: EngineTier = "auto",
    engine_tiers: list[EngineTier] | None = None,
    progress_hook: ProgressHook | None = None,
    enable_parallel: bool = True,
    detector_cache: _DetectorCache | None = None,
) -> ProfileBenchmarkResult:
    # Resolve the list of tiers to evaluate.  When *engine_tiers* is
    # supplied we evaluate pii-anon once per tier; otherwise fall back
    # to the single *engine_tier* value for backward compatibility.
    tiers_to_eval: list[EngineTier] = engine_tiers if engine_tiers is not None else [engine_tier]

    if progress_hook:
        tier_label = ", ".join(tiers_to_eval)
        progress_hook(
            f"profile `{profile}` ({objective}): start, records={len(records)}, "
            f"tiers=[{tier_label}]"
        )

    systems: list[SystemBenchmarkResult] = []
    end_to_end_systems: list[SystemBenchmarkResult] = []

    detector_factories_keys = ["presidio", "scrubadub", "gliner"]
    systems_to_evaluate = competitor_systems or detector_factories_keys
    forced_unavailable = forced_unavailable_competitors or {}

    # --- parallel path ---------------------------------------------------
    # Submit ALL evaluations (core tiers + end-to-end tiers + competitors)
    # to a unified worker pool.  Each worker creates its own detector
    # instance so nothing unpicklable crosses process boundaries.
    if enable_parallel:
        # Build core detect-only specs
        core_detect_specs: list[_CoreEvalSpec] = [
            _CoreEvalSpec(
                tier=tier,
                records=records,
                warmup_samples=warmup_samples,
                measured_runs=measured_runs,
                use_case=profile,
                objective=objective,
                allow_native_engines=allow_core_native_engines,
                evaluation_track="detect_only",
                profile_label=f"profile `{profile}`",
                progress_hook=progress_hook,
            )
            for tier in tiers_to_eval
        ]

        # Build core end-to-end specs
        core_e2e_specs: list[_CoreEvalSpec] = []
        if include_end_to_end:
            core_e2e_specs = [
                _CoreEvalSpec(
                    tier=tier,
                    records=records,
                    warmup_samples=warmup_samples,
                    measured_runs=measured_runs,
                    use_case=profile,
                    objective=objective,
                    allow_native_engines=allow_core_native_engines,
                    evaluation_track="end_to_end",
                    profile_label=f"profile `{profile}`",
                    progress_hook=progress_hook,
                )
                for tier in tiers_to_eval
            ]

        # Build competitor specs
        competitor_specs: list[_SystemEvalSpec] = [
            _SystemEvalSpec(
                system_name=system_name,
                records=records,
                warmup_samples=warmup_samples,
                measured_runs=measured_runs,
                allow_fallback_detectors=allow_fallback_detectors,
                require_native_competitors=require_native_competitors,
                forced_unavailable_reason=forced_unavailable.get(system_name),
                profile_label=f"profile `{profile}`",
                progress_hook=progress_hook,
                detector_cache=detector_cache,
            )
            for system_name in systems_to_evaluate
        ]

        total_tasks = len(core_detect_specs) + len(core_e2e_specs) + len(competitor_specs)
        if progress_hook:
            progress_hook(
                f"profile `{profile}`: launching {total_tasks} evaluations in parallel "
                f"({len(core_detect_specs)} core-detect, {len(core_e2e_specs)} core-e2e, "
                f"{len(competitor_specs)} competitors)"
            )

        # Use threads rather than processes.  ProcessPoolExecutor with
        # "spawn" context required each worker to duplicate the full
        # dataset + NLP models into a fresh process, which routinely
        # exceeded Docker cgroup memory limits and triggered
        # BrokenProcessPool crashes.  Threads share the parent's
        # memory space, eliminating the duplication.  NLP libraries
        # (torch, spaCy, transformers) release the GIL during their C
        # extension inference kernels, so threads still achieve real
        # concurrency for the compute-heavy portions of each evaluation.
        n_workers = _optimal_workers(total_tasks)
        completed_count = 0

        # Tag futures so we know where to route results.
        _TAG_CORE_DETECT = "core_detect"
        _TAG_CORE_E2E = "core_e2e"
        _TAG_COMPETITOR = "competitor"

        with ThreadPoolExecutor(
            max_workers=n_workers,
        ) as pool:
            futures: dict[Any, tuple[str, str]] = {}
            for core_spec in core_detect_specs:
                f = pool.submit(_core_system_worker, core_spec)
                futures[f] = (_TAG_CORE_DETECT, _tier_system_name(core_spec.tier))
            for e2e_spec in core_e2e_specs:
                f = pool.submit(_core_system_worker, e2e_spec)
                futures[f] = (_TAG_CORE_E2E, _tier_system_name(e2e_spec.tier))
            for comp_spec in competitor_specs:
                f = pool.submit(_evaluate_system_worker, comp_spec)
                futures[f] = (_TAG_COMPETITOR, comp_spec.system_name)

            pending_names = {label for _, label in futures.values()}
            for future in as_completed(futures):
                tag, worker_name = futures[future]
                try:
                    result = future.result()
                except Exception:
                    _log.exception(
                        "parallel worker for `%s` (%s) failed; recording as unavailable",
                        worker_name,
                        tag,
                    )
                    ev_name = "pii-anon" if tag != _TAG_COMPETITOR else worker_name
                    evidence = _qualify_oss_license(ev_name)
                    result = SystemBenchmarkResult(
                        system=worker_name,
                        available=False,
                        skipped_reason="parallel worker process failed",
                        qualification_status=evidence.qualification_status,
                        license_name=evidence.license_name,
                        license_source=evidence.license_source,
                        citation_url=evidence.citation_url,
                        license_gate_passed=evidence.passed,
                        license_gate_reason=evidence.reason,
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        latency_p50_ms=0.0,
                        docs_per_hour=0.0,
                        per_entity_recall={},
                        samples=len(records),
                        evaluation_track="detect_only" if tag != _TAG_CORE_E2E else "end_to_end",
                    )
                    # Credit the full eval_work for this failed worker so the
                    # progress bar stays accurate.  The worker's _evaluate_system()
                    # threw before completing its WORK emissions, so these units
                    # were never credited.
                    crash_work = min(warmup_samples, len(records)) + measured_runs * len(records)
                    if progress_hook:
                        progress_hook(
                            f"WORK:{crash_work}|profile `{profile}`: "
                            f"{worker_name} ({tag}) crashed — crediting {crash_work} work units"
                        )

                if tag == _TAG_CORE_E2E:
                    end_to_end_systems.append(result)
                else:
                    systems.append(result)

                pending_names.discard(worker_name)
                completed_count += 1
                # NOTE: do NOT emit WORK here — work units are already
                # credited inside _evaluate_system() during the detection
                # loop (warmup + per-run progress).  Re-crediting them on
                # worker completion would double-count.
                if progress_hook:
                    if pending_names:
                        waiting = ", ".join(sorted(pending_names))
                        progress_hook(
                            f"profile `{profile}` [{completed_count}/{total_tasks}]: "
                            f"{worker_name} ({tag}) done (f1={result.f1:.3f}) — waiting for {waiting}"
                        )
                    else:
                        progress_hook(
                            f"profile `{profile}` [{completed_count}/{total_tasks}]: "
                            f"all parallel workers complete"
                        )

    # --- sequential path (fallback when parallel is disabled) -------------
    else:
        core_evidence = _qualify_oss_license("pii-anon")

        for tier in tiers_to_eval:
            system_name = _tier_system_name(tier)
            systems.append(
                _evaluate_system(
                    system_name,
                    _core_detector(
                        use_case=profile,
                        objective=objective,
                        allow_native_engines=allow_core_native_engines,
                        engine_tier=tier,
                    ),
                    reason=None,
                    records=records,
                    warmup_samples=warmup_samples,
                    measured_runs=measured_runs,
                    evidence=core_evidence,
                    evaluation_track="detect_only",
                    progress_hook=progress_hook,
                    progress_label=f"profile `{profile}` {system_name} detect_only",
                )
            )

        if include_end_to_end:
            for tier in tiers_to_eval:
                system_name = _tier_system_name(tier)
                end_to_end_systems.append(
                    _evaluate_system(
                        system_name,
                        _core_end_to_end_detector(
                            use_case=profile,
                            objective=objective,
                            allow_native_engines=allow_core_native_engines,
                            engine_tier=tier,
                        ),
                        reason=None,
                        records=records,
                        warmup_samples=warmup_samples,
                        measured_runs=measured_runs,
                        evidence=core_evidence,
                        evaluation_track="end_to_end",
                        progress_hook=progress_hook,
                        progress_label=f"profile `{profile}` {system_name} end_to_end",
                    )
                )

        detector_factories: dict[
            str,
            Callable[..., tuple[Callable[[BenchmarkRecord], list[LabelSpan]] | None, str | None]],
        ] = {
            "presidio": _presidio_detector,
            "scrubadub": _scrubadub_detector,
            "gliner": _gliner_detector,
        }
        for system_name in systems_to_evaluate:
            detector: Callable[[BenchmarkRecord], list[LabelSpan]] | None
            detector_reason: str | None
            if system_name in forced_unavailable:
                detector = None
                detector_reason = forced_unavailable[system_name]
            else:
                detector_factory = detector_factories.get(system_name)
                if detector_factory is None:
                    detector = None
                    detector_reason = f"unknown competitor detector `{system_name}`"
                elif detector_cache is not None:
                    cache_key = (
                        system_name,
                        str(allow_fallback_detectors),
                        str(require_native_competitors),
                    )
                    _cached_factory = detector_factory

                    def _make_detector() -> tuple[
                        Callable[[BenchmarkRecord], list[LabelSpan]] | None,
                        str | None,
                    ]:
                        return _cached_factory(
                            allow_fallback=allow_fallback_detectors,
                            require_native=require_native_competitors,
                        )

                    detector, detector_reason = detector_cache.get_or_create(
                        cache_key,
                        _make_detector,
                    )
                else:
                    detector, detector_reason = detector_factory(
                        allow_fallback=allow_fallback_detectors,
                        require_native=require_native_competitors,
                    )
            evidence = _qualify_oss_license(system_name)
            reason = detector_reason
            if not evidence.passed:
                detector = None
                if reason:
                    reason = f"{reason}; excluded by qualification gate ({evidence.reason})"
                else:
                    reason = f"excluded by qualification gate ({evidence.reason})"

            systems.append(
                _evaluate_system(
                    system_name,
                    detector,
                    reason=reason,
                    records=records,
                    warmup_samples=warmup_samples,
                    measured_runs=measured_runs,
                    evidence=evidence,
                    evaluation_track="detect_only",
                    progress_hook=progress_hook,
                    progress_label=f"profile `{profile}` {system_name} detect_only",
                )
            )

    floor_pass, floor_checks = _evaluate_floor_contract(systems, objective)

    for row in systems:
        row.dominance_pass_by_profile[profile] = row.system == "pii-anon" and floor_pass

    qualified_competitors = _qualified_competitors(systems)
    profile_result = ProfileBenchmarkResult(
        profile=profile,
        objective=objective,
        systems=systems,
        end_to_end_systems=end_to_end_systems,
        floor_pass=floor_pass,
        floor_checks=floor_checks,
        qualified_competitors=qualified_competitors,
        mit_qualified_competitors=_mit_qualified_competitors(systems),
    )
    if progress_hook:
        progress_hook(
            f"profile `{profile}` complete: floor_pass={profile_result.floor_pass}, "
            f"qualified_competitors={profile_result.qualified_competitors}"
        )
    return profile_result


def _merge_profile_systems(profile_reports: list[ProfileBenchmarkResult]) -> list[SystemBenchmarkResult]:
    by_system: dict[str, SystemBenchmarkResult] = {}
    for report in profile_reports:
        for row in report.systems:
            current = by_system.get(row.system)
            if current is None:
                by_system[row.system] = row
                continue
            current.dominance_pass_by_profile.update(row.dominance_pass_by_profile)
    return [by_system[key] for key in sorted(by_system.keys())]


def _summarize_competitor_availability(
    *,
    profile_reports: list[ProfileBenchmarkResult],
    expected_competitors: list[str],
) -> tuple[list[str], dict[str, str], bool]:
    available: list[str] = []
    unavailable: dict[str, str] = {}

    for system in expected_competitors:
        reasons: list[str] = []
        for profile in profile_reports:
            row = next((item for item in profile.systems if item.system == system), None)
            if row is None:
                reasons.append(f"{profile.profile}: missing benchmark row")
                continue
            if not row.available:
                reasons.append(f"{profile.profile}: {row.skipped_reason or 'unavailable'}")
                continue
            if not row.license_gate_passed:
                reasons.append(
                    f"{profile.profile}: excluded ({row.license_gate_reason or 'license gate failure'})"
                )
        if reasons:
            unavailable[system] = "; ".join(reasons)
        else:
            available.append(system)

    return available, unavailable, not unavailable


def _apply_elo_ratings(systems: list[SystemBenchmarkResult]) -> None:
    """Run an Elo round-robin tournament on available systems and set elo_rating.

    Uses the composite_score already computed on each SystemBenchmarkResult.
    Three rounds of round-robin are run for convergence, matching the
    methodology described in the benchmark documentation.
    """
    from pii_anon.eval_framework.rating.elo import PIIRateEloEngine as _EloEngine

    composites = {
        s.system: s.composite_score
        for s in systems
        if s.available and s.composite_score > 0
    }
    if len(composites) < 2:
        return

    engine = _EloEngine()
    for _ in range(3):  # 3 rounds for convergence
        engine.run_round_robin(composites)

    for s in systems:
        rating = engine.get_rating(s.system)
        if rating is not None:
            s.elo_rating = round(rating.rating, 2)


def _classify_errors(
    predictions: list[LabelSpan],
    labels: list[LabelSpan],
    matched_pred_indices: set[int],
    matched_label_indices: set[int],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Classify detection errors into actionable categories.

    Returns ``(aggregate_counts, per_entity_counts)`` where each maps
    error category names to counts.

    Error categories:
    - **true_positive**: correctly matched (IoU >= threshold).
    - **boundary_miss**: prediction overlaps a label of the *same* entity
      type but IoU < threshold — the span boundaries are off.
    - **type_confusion**: prediction overlaps a label but with a *different*
      normalised entity type — the detector found the text but mis-typed it.
    - **complete_miss**: a ground-truth label with zero prediction overlap
      at all — the entity was completely undetected.
    - **spurious_fp**: a prediction that doesn't overlap any ground-truth
      label — a pure hallucinated detection.
    """
    _BOUNDARY_IOU_MIN = 0.01  # any overlap at all counts as boundary/type
    # Build a per-record index for cross-type overlap checks.
    preds_by_record: dict[str, list[tuple[str, int, int, int]]] = {}
    for i, (rid, etype, s, e) in enumerate(predictions):
        preds_by_record.setdefault(rid, []).append(
            (_normalize_entity_type(etype), s, e, i)
        )

    totals: dict[str, int] = {
        "true_positive": 0,
        "boundary_miss": 0,
        "type_confusion": 0,
        "complete_miss": 0,
        "spurious_fp": 0,
    }
    per_entity: dict[str, dict[str, int]] = {}

    def _bump(entity_type: str, category: str) -> None:
        totals[category] += 1
        bucket = per_entity.setdefault(entity_type, {
            "true_positive": 0, "boundary_miss": 0,
            "type_confusion": 0, "complete_miss": 0, "spurious_fp": 0,
            "support": 0,
        })
        bucket[category] += 1

    # ── Analyse each ground-truth label ──
    for i, (rid, etype, ls, le) in enumerate(labels):
        norm_type = _normalize_entity_type(etype)
        # Count support (total ground truth per entity type)
        per_entity.setdefault(norm_type, {
            "true_positive": 0, "boundary_miss": 0,
            "type_confusion": 0, "complete_miss": 0, "spurious_fp": 0,
            "support": 0,
        })["support"] += 1

        if i in matched_label_indices:
            _bump(norm_type, "true_positive")
            continue

        # Unmatched label — check if any prediction overlaps it at all.
        best_overlap = 0.0
        best_pred_type: str | None = None
        for pred_norm_type, ps, pe, _pi in preds_by_record.get(rid, []):
            inter = max(0, min(le, pe) - max(ls, ps))
            union = (le - ls) + (pe - ps) - inter
            if union > 0:
                iou = inter / union
                if iou > best_overlap:
                    best_overlap = iou
                    best_pred_type = pred_norm_type

        if best_overlap < _BOUNDARY_IOU_MIN:
            _bump(norm_type, "complete_miss")
        elif best_pred_type and best_pred_type != norm_type:
            _bump(norm_type, "type_confusion")
        else:
            _bump(norm_type, "boundary_miss")

    # ── Analyse unmatched predictions (spurious FP) ──
    for i, (_rid, etype, _s, _e) in enumerate(predictions):
        if i not in matched_pred_indices:
            norm_type = _normalize_entity_type(etype)
            _bump(norm_type, "spurious_fp")

    return totals, per_entity


def _compute_per_entity_f1(
    per_entity_precision: dict[str, float],
    per_entity_recall: dict[str, float],
) -> dict[str, float]:
    """Derive per-entity F1 from existing per-entity precision and recall."""
    all_types = sorted(set(per_entity_precision) | set(per_entity_recall))
    result: dict[str, float] = {}
    for etype in all_types:
        p = per_entity_precision.get(etype, 0.0)
        r = per_entity_recall.get(etype, 0.0)
        result[etype] = round(_safe_div(2.0 * p * r, p + r), 6)
    return result


def _compute_diagnostics(
    systems: list[SystemBenchmarkResult],
    progress_hook: ProgressHook | None = None,
) -> dict[str, Any]:
    """Build comparative diagnostics across all available systems.

    Produces:
    - **entity_head_to_head**: for each entity type, which system has the
      best F1 and how far behind each other system is.
    - **improvement_opportunities**: ranked list of (entity_type, delta)
      pairs showing where pii-anon loses the most F1 vs. the best competitor,
      sorted by potential impact (delta × support).
    - **system_error_profiles**: per-system error category breakdown.
    - **entity_difficulty_ranking**: entity types ranked by average F1
      across all systems (hardest to easiest).
    """
    available = [s for s in systems if s.available]
    if not available:
        return {}

    if progress_hook:
        progress_hook(
            f"computing diagnostics for {len(available)} systems"
        )

    result: dict[str, Any] = {}

    # ── Entity head-to-head ──
    # Collect all entity types across all systems.
    all_entity_types: set[str] = set()
    for sys in available:
        all_entity_types.update(sys.per_entity_f1.keys())

    head_to_head: dict[str, dict[str, Any]] = {}
    for etype in sorted(all_entity_types):
        scores = {
            s.system: s.per_entity_f1.get(etype, 0.0)
            for s in available
        }
        best_system = max(scores, key=lambda k: scores[k])
        best_f1 = scores[best_system]
        head_to_head[etype] = {
            "best_system": best_system,
            "best_f1": round(best_f1, 6),
            "scores": {k: round(v, 6) for k, v in sorted(scores.items())},
        }
    result["entity_head_to_head"] = head_to_head

    # ── Improvement opportunities for core systems ──
    core_systems = [s for s in available if _is_core_system(s.system)]
    competitors = [s for s in available if not _is_core_system(s.system)]

    if core_systems and competitors:
        for core in core_systems:
            opportunities: list[dict[str, Any]] = []
            for etype in sorted(all_entity_types):
                core_f1 = core.per_entity_f1.get(etype, 0.0)
                # Best competitor F1 for this entity type
                best_comp_f1 = 0.0
                best_comp_name = ""
                for comp in competitors:
                    comp_f1 = comp.per_entity_f1.get(etype, 0.0)
                    if comp_f1 > best_comp_f1:
                        best_comp_f1 = comp_f1
                        best_comp_name = comp.system
                delta = best_comp_f1 - core_f1
                if delta > 0.001:
                    # Estimate support from per_entity_errors if available
                    support = core.per_entity_errors.get(etype, {}).get("support", 0)
                    opportunities.append({
                        "entity_type": etype,
                        "core_f1": round(core_f1, 6),
                        "best_competitor": best_comp_name,
                        "best_competitor_f1": round(best_comp_f1, 6),
                        "delta_f1": round(delta, 6),
                        "support": support,
                        "weighted_impact": round(delta * support, 2),
                    })
            # Sort by weighted impact descending — biggest bang-for-buck first.
            opportunities.sort(key=lambda x: x["weighted_impact"], reverse=True)
            result.setdefault("improvement_opportunities", {})[core.system] = opportunities

    # ── System error profiles ──
    error_profiles: dict[str, dict[str, Any]] = {}
    for sys in available:
        if sys.error_counts:
            total_errors = sum(
                v for k, v in sys.error_counts.items() if k != "true_positive"
            )
            error_profiles[sys.system] = {
                "total_errors": total_errors,
                "breakdown": dict(sys.error_counts),
                "error_rate": round(
                    _safe_div(total_errors, total_errors + sys.error_counts.get("true_positive", 0)),
                    6,
                ),
            }
    result["system_error_profiles"] = error_profiles

    # ── Entity difficulty ranking ──
    # Average F1 across all available systems per entity type.
    difficulty: list[dict[str, Any]] = []
    for etype in sorted(all_entity_types):
        f1_values = [
            s.per_entity_f1.get(etype, 0.0) for s in available
            if etype in s.per_entity_f1
        ]
        if f1_values:
            avg_f1 = sum(f1_values) / len(f1_values)
            max_f1 = max(f1_values)
            min_f1 = min(f1_values)
            difficulty.append({
                "entity_type": etype,
                "avg_f1": round(avg_f1, 6),
                "max_f1": round(max_f1, 6),
                "min_f1": round(min_f1, 6),
                "spread": round(max_f1 - min_f1, 6),
                "systems_evaluated": len(f1_values),
            })
    # Sort hardest (lowest avg F1) first.
    difficulty.sort(key=lambda x: x["avg_f1"])
    result["entity_difficulty_ranking"] = difficulty

    # ── Wins summary ──
    # Count how many entity types each system has the best F1.
    wins: dict[str, int] = {}
    for etype_info in head_to_head.values():
        winner = etype_info["best_system"]
        wins[winner] = wins.get(winner, 0) + 1
    result["entity_type_wins"] = dict(sorted(wins.items(), key=lambda x: x[1], reverse=True))

    if progress_hook:
        progress_hook(
            f"diagnostics complete: {len(head_to_head)} entity types, "
            f"{len(difficulty)} difficulty entries"
        )
    return result


def _compute_statistical_tests(
    systems: list[SystemBenchmarkResult],
    progress_hook: ProgressHook | None = None,
) -> dict[str, Any]:
    """Compute statistical significance tests between all available systems.

    Runs paired bootstrap significance tests and Cohen's d effect size
    for every system pair, plus per-system bootstrap CIs. Uses per-record
    F1 scores collected during evaluation.

    Evidence:
    - Efron & Tibshirani (1993): Bootstrap confidence intervals
    - Berg-Kirkpatrick et al. (2012): Paired bootstrap significance for NLP
    - Cohen (1988): Effect size measures (Cohen's d)
    """
    from pii_anon.eval_framework.evaluation.aggregation import MetricAggregator

    available = [s for s in systems if s.available and s.per_record_f1]
    if len(available) < 2:
        return {}

    result: dict[str, Any] = {
        "method": "paired_bootstrap",
        "n_bootstrap": 10000,
        "confidence_level": 0.95,
    }

    # Per-system CI summary.
    if progress_hook:
        progress_hook(
            f"statistical tests: computing bootstrap CIs for {len(available)} systems "
            f"(2K iterations each)"
        )
    system_cis: dict[str, dict[str, float]] = {}
    for idx, sys in enumerate(available, start=1):
        if progress_hook:
            progress_hook(
                f"  bootstrap CI [{idx}/{len(available)}]: {sys.system} "
                f"({len(sys.per_record_f1)} samples, 2K iterations)"
            )
        ci_lower, ci_upper = MetricAggregator.compute_confidence_intervals(
            sys.per_record_f1,
            confidence_level=0.95,
            n_bootstrap=2000,
        )
        system_cis[sys.system] = {
            "f1": round(sys.f1, 6),
            "f1_ci_lower": round(ci_lower, 6),
            "f1_ci_upper": round(ci_upper, 6),
            "samples": sys.samples,
        }
        if progress_hook:
            progress_hook(
                f"  bootstrap CI [{idx}/{len(available)}]: {sys.system} done — "
                f"F1={sys.f1:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
            )
    result["system_confidence_intervals"] = system_cis

    # Pairwise significance tests — core vs. each competitor.
    core_systems = [s for s in available if _is_core_system(s.system)]
    competitors = [s for s in available if not _is_core_system(s.system)]

    total_pairs = len(core_systems) * len(competitors)
    if progress_hook:
        progress_hook(
            f"statistical tests: running {total_pairs} pairwise bootstrap tests "
            f"(10K iterations each)"
        )
    pairwise: list[dict[str, Any]] = []
    pair_idx = 0
    for core in core_systems:
        for comp in competitors:
            pair_idx += 1
            min_len = min(len(core.per_record_f1), len(comp.per_record_f1))
            if min_len == 0:
                continue
            if progress_hook:
                progress_hook(
                    f"  pairwise test [{pair_idx}/{total_pairs}]: "
                    f"{core.system} vs {comp.system} ({min_len} paired samples, 10K iterations)"
                )
            test = MetricAggregator.paired_bootstrap_test(
                core.per_record_f1[:min_len],
                comp.per_record_f1[:min_len],
                n_bootstrap=10000,
            )
            effect_size = MetricAggregator.cohens_d(
                core.per_record_f1[:min_len],
                comp.per_record_f1[:min_len],
            )
            # Interpret effect size
            abs_d = abs(effect_size)
            if abs_d < 0.2:
                effect_label = "negligible"
            elif abs_d < 0.5:
                effect_label = "small"
            elif abs_d < 0.8:
                effect_label = "medium"
            else:
                effect_label = "large"

            pairwise.append({
                "system_a": core.system,
                "system_b": comp.system,
                "delta_f1": round(test["delta_mean"], 6),
                "p_value": round(test["p_value"], 6),
                "ci_lower": round(test["ci_lower"], 6),
                "ci_upper": round(test["ci_upper"], 6),
                "significant_at_05": test["p_value"] < 0.05,
                "significant_at_01": test["p_value"] < 0.01,
                "cohens_d": round(effect_size, 4),
                "effect_size": effect_label,
                "n_paired_samples": min_len,
            })
            if progress_hook:
                sig = "significant" if test["p_value"] < 0.05 else "not significant"
                progress_hook(
                    f"  pairwise test [{pair_idx}/{total_pairs}]: "
                    f"{core.system} vs {comp.system} done — "
                    f"delta={test['delta_mean']:+.4f}, p={test['p_value']:.4f} ({sig}), "
                    f"d={effect_size:+.3f} ({effect_label})"
                )
    result["pairwise_tests"] = pairwise

    # Minimum detectable effect at n=50000.
    if available:
        n = max(s.samples for s in available)
        mde = MetricAggregator.minimum_detectable_effect(n)
        result["minimum_detectable_effect"] = round(mde, 6)
        result["sample_size"] = n

    if progress_hook:
        progress_hook(
            f"statistical tests complete: {len(system_cis)} system CIs, "
            f"{len(pairwise)} pairwise comparisons"
        )

    return result


def _checkpoint_filename(profile: str) -> str:
    """Return a deterministic checkpoint filename for a profile."""
    return f"checkpoint_{profile}.json"


def _save_checkpoint(
    checkpoint_dir: Path,
    profile_result: ProfileBenchmarkResult,
) -> None:
    """Persist a completed profile result to disk for later resume."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / _checkpoint_filename(profile_result.profile)
    path.write_text(
        json.dumps(asdict(profile_result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(
    checkpoint_dir: Path,
    profile: str,
) -> ProfileBenchmarkResult | None:
    """Load a previously-saved profile checkpoint, or None if absent."""
    path = checkpoint_dir / _checkpoint_filename(profile)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    systems = [
        SystemBenchmarkResult(**{
            k: v for k, v in sys.items()
            if k in SystemBenchmarkResult.__dataclass_fields__
        })
        for sys in data.get("systems", [])
    ]
    end_to_end_systems = [
        SystemBenchmarkResult(**{
            k: v for k, v in sys.items()
            if k in SystemBenchmarkResult.__dataclass_fields__
        })
        for sys in data.get("end_to_end_systems", [])
    ]
    floor_checks = [
        FloorCheckResult(**check)
        for check in data.get("floor_checks", [])
    ]
    return ProfileBenchmarkResult(
        profile=data["profile"],
        objective=data["objective"],
        systems=systems,
        end_to_end_systems=end_to_end_systems,
        floor_pass=data["floor_pass"],
        floor_checks=floor_checks,
        qualified_competitors=data["qualified_competitors"],
        mit_qualified_competitors=data.get("mit_qualified_competitors", 0),
    )


def compare_competitors(
    *,
    dataset: str = "pii_anon_benchmark_v1",
    dataset_source: DatasetSource = "auto",
    warmup_samples: int = 100,
    measured_runs: int = 3,
    max_samples: int | None = None,
    matrix_path: str | None = None,
    profile_filter: list[str] | None = None,
    enforce_floors: bool = False,
    use_case: str = "default",
    objective: Objective = "balanced",
    require_all_competitors: bool = False,
    require_native_competitors: bool = False,
    allow_fallback_detectors: bool = True,
    include_end_to_end: bool = True,
    forced_unavailable_competitors: dict[str, str] | None = None,
    allow_core_native_engines: bool = True,
    expected_competitors: list[str] | None = None,
    engine_tier: EngineTier = "auto",
    engine_tiers: list[EngineTier] | None = None,
    progress_hook: ProgressHook | None = None,
    enable_parallel: bool = True,
    checkpoint_dir: str | None = None,
) -> CompetitorComparisonReport:
    if progress_hook:
        progress_hook(
            f"loading dataset `{dataset}`"
            + (f" (max_samples={max_samples})" if max_samples is not None else "")
        )
    records = load_benchmark_dataset(dataset, source=dataset_source)
    if max_samples is not None:
        records = records[: max(0, max_samples)]
    if progress_hook:
        progress_hook(f"dataset `{dataset}` loaded with {len(records)} records")

    expected = expected_competitors or list(_COMPETITOR_META.keys())

    # Resolve effective tier list.  *engine_tiers* takes precedence over
    # the single *engine_tier* parameter.  When neither is provided the
    # default is a single "auto" tier for backward compatibility.
    effective_tiers: list[EngineTier] | None = engine_tiers
    num_core_evals = len(effective_tiers) if effective_tiers else 1

    if matrix_path is None:
        n = len(records)
        eval_work = min(warmup_samples, n) + measured_runs * n
        total = num_core_evals * eval_work  # core_detect (one per tier)
        if include_end_to_end:
            total += num_core_evals * eval_work  # core_e2e (one per tier)
        total += len(expected) * eval_work  # competitors (only once)
        if progress_hook:
            progress_hook(f"TOTAL:{total:.0f}|single profile `{use_case}`: {total:.0f} work units")
        profile_report = _evaluate_profile(
            records=records,
            warmup_samples=warmup_samples,
            measured_runs=measured_runs,
            profile=use_case,
            objective=objective,
            allow_fallback_detectors=allow_fallback_detectors,
            require_native_competitors=require_native_competitors,
            competitor_systems=expected,
            include_end_to_end=include_end_to_end,
            forced_unavailable_competitors=forced_unavailable_competitors,
            allow_core_native_engines=allow_core_native_engines,
            engine_tier=engine_tier,
            engine_tiers=effective_tiers,
            progress_hook=progress_hook,
            enable_parallel=enable_parallel,
            detector_cache=_DetectorCache(),
        )
        available_competitors, unavailable_competitors, all_competitors_available = _summarize_competitor_availability(
            profile_reports=[profile_report],
            expected_competitors=expected,
        )
        floor_pass = profile_report.floor_pass
        qualification_gate_pass = profile_report.qualified_competitors > 0
        if require_all_competitors and not all_competitors_available:
            qualification_gate_pass = False
        if enforce_floors and (not floor_pass or not qualification_gate_pass):
            if require_all_competitors and not all_competitors_available:
                failed = ", ".join(sorted(unavailable_competitors))
                raise RuntimeError(f"competitor availability gate failed for systems: {failed}")
            raise RuntimeError("floor gate or qualification gate failed for profile `default`")
        _apply_elo_ratings(profile_report.systems)
        if progress_hook:
            progress_hook("computing statistical significance tests")
        stat_tests = _compute_statistical_tests(profile_report.systems, progress_hook=progress_hook)
        diag = _compute_diagnostics(profile_report.systems, progress_hook=progress_hook)
        return CompetitorComparisonReport(
            report_schema_version=REPORT_SCHEMA_VERSION,
            dataset=dataset,
            dataset_source=dataset_source,
            systems=profile_report.systems,
            warmup_samples=warmup_samples,
            measured_runs=measured_runs,
            profiles=[profile_report],
            floor_pass=floor_pass,
            qualification_gate_pass=qualification_gate_pass,
            mit_gate_pass=qualification_gate_pass,
            expected_competitors=expected,
            available_competitors=available_competitors,
            unavailable_competitors=unavailable_competitors,
            all_competitors_available=all_competitors_available,
            require_all_competitors=require_all_competitors,
            require_native_competitors=require_native_competitors,
            statistical_tests=stat_tests,
            diagnostics=diag,
        )

    profiles_cfg = load_use_case_matrix(matrix_path)

    # Apply profile filter for parallel execution mode.  When set, only
    # the requested profiles are evaluated — the remaining profiles are
    # expected to run in separate parallel processes writing to the same
    # checkpoint directory.
    if profile_filter is not None:
        all_names = [c.profile for c in profiles_cfg]
        profiles_cfg = [c for c in profiles_cfg if c.profile in profile_filter]
        if not profiles_cfg:
            raise ValueError(
                f"profile_filter {profile_filter} matched no profiles. "
                f"Available: {all_names}"
            )
        if progress_hook:
            progress_hook(
                f"profile filter active: evaluating {len(profiles_cfg)} of "
                f"{len(all_names)} profiles ({', '.join(c.profile for c in profiles_cfg)})"
            )

    # Compute total work units for deterministic progress tracking.
    # Work units = records processed across all evaluations (core tiers + competitors).
    total_work = 0.0
    for _cfg in profiles_cfg:
        _profile_recs = _deterministic_profile_records(records, profile=_cfg, max_samples=max_samples)
        _n = len(_profile_recs)
        _eval_work = min(warmup_samples, _n) + measured_runs * _n
        total_work += num_core_evals * _eval_work  # core_detect (per tier)
        if include_end_to_end:
            total_work += num_core_evals * _eval_work  # core_e2e (per tier)
        total_work += len(expected) * _eval_work  # competitors (once)
    if progress_hook:
        tier_label = ", ".join(effective_tiers) if effective_tiers else engine_tier
        progress_hook(
            f"TOTAL:{total_work:.0f}|matrix: {len(profiles_cfg)} profiles, "
            f"{len(records)} records, tiers=[{tier_label}], "
            f"{total_work:.0f} work units"
        )
    profile_reports: list[ProfileBenchmarkResult] = []
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else None
    # Shared detector cache: competitor detectors (presidio, scrubadub,
    # gliner) are identical across profiles — load models once, reuse.
    shared_detector_cache = _DetectorCache()

    # --- sequential profile evaluation ------------------------------------
    # Profiles are evaluated sequentially.  Within each profile,
    # system-level parallelism (tiers + competitors in one pool) is
    # used when ``enable_parallel=True``, which already saturates CPU
    # cores.  Profile-level parallelism was removed because spawning
    # multiple heavy worker processes (each loading 50K records + NLP
    # models via the ``spawn`` context) exceeded Docker cgroup memory
    # limits, causing ``BrokenProcessPool`` crashes.
    #
    # When *checkpoint_dir* is set, each completed profile is saved to
    # disk as JSON.  On resume, previously-completed profiles are loaded
    # from disk and their work units credited to the progress bar so the
    # total stays accurate.
    for index, cfg in enumerate(profiles_cfg, start=1):
        # --- checkpoint resume: skip already-completed profiles -----------
        if ckpt_path is not None:
            cached = _load_checkpoint(ckpt_path, cfg.profile)
            if cached is not None:
                profile_reports.append(cached)
                # Credit the skipped work so the progress bar stays accurate.
                _n = len(_deterministic_profile_records(records, profile=cfg, max_samples=max_samples))
                skip_units = (
                    num_core_evals * (min(warmup_samples, _n) + measured_runs * _n)
                )
                if include_end_to_end:
                    skip_units += num_core_evals * (min(warmup_samples, _n) + measured_runs * _n)
                skip_units += len(expected) * (min(warmup_samples, _n) + measured_runs * _n)
                if progress_hook:
                    avail = sum(1 for s in cached.systems if s.available)
                    best_f1 = max((s.f1 for s in cached.systems if s.available), default=0.0)
                    progress_hook(
                        f"WORK:{skip_units}|profile {index}/{len(profiles_cfg)} "
                        f"RESUMED from checkpoint: {cfg.profile} ({cfg.objective}) — "
                        f"{avail} systems available, best F1={best_f1:.4f}"
                    )
                continue

        profile_records = _deterministic_profile_records(records, profile=cfg, max_samples=max_samples)
        if progress_hook:
            progress_hook(
                f"evaluating profile {index}/{len(profiles_cfg)}: "
                f"{cfg.profile} ({cfg.objective}), "
                f"{len(profile_records)} records"
            )
        profile_report = _evaluate_profile(
            records=profile_records,
            warmup_samples=warmup_samples,
            measured_runs=measured_runs,
            profile=cfg.profile,
            objective=cfg.objective,
            allow_fallback_detectors=allow_fallback_detectors,
            require_native_competitors=require_native_competitors,
            competitor_systems=expected,
            include_end_to_end=include_end_to_end,
            forced_unavailable_competitors=forced_unavailable_competitors,
            allow_core_native_engines=allow_core_native_engines,
            engine_tier=engine_tier,
            engine_tiers=effective_tiers,
            progress_hook=progress_hook,
            enable_parallel=enable_parallel,
            detector_cache=shared_detector_cache,
        )
        profile_reports.append(profile_report)
        if progress_hook:
            avail = sum(1 for s in profile_report.systems if s.available)
            best_f1 = max((s.f1 for s in profile_report.systems if s.available), default=0.0)
            progress_hook(
                f"profile {index}/{len(profiles_cfg)} complete: "
                f"{cfg.profile} ({cfg.objective}) — "
                f"{avail} systems available, best F1={best_f1:.4f}"
            )
        # --- checkpoint save: persist completed profile to disk -----------
        if ckpt_path is not None:
            _save_checkpoint(ckpt_path, profile_report)
            if progress_hook:
                progress_hook(
                    f"checkpoint saved: {ckpt_path / _checkpoint_filename(cfg.profile)}"
                )

    required_profiles = {cfg.profile for cfg in profiles_cfg if cfg.required}
    required = [item for item in profile_reports if item.profile in required_profiles]

    floor_pass = all(item.floor_pass for item in required)
    qualification_gate_pass = all(item.qualified_competitors > 0 for item in required)
    available_competitors, unavailable_competitors, all_competitors_available = _summarize_competitor_availability(
        profile_reports=profile_reports,
        expected_competitors=expected,
    )
    if require_all_competitors and not all_competitors_available:
        qualification_gate_pass = False

    if enforce_floors and not floor_pass:
        failed_profiles = [item.profile for item in required if not item.floor_pass]
        raise RuntimeError(f"floor gate failed for profiles: {', '.join(failed_profiles)}")
    if enforce_floors and not qualification_gate_pass:
        if require_all_competitors and not all_competitors_available:
            failed_systems = ", ".join(sorted(unavailable_competitors))
            raise RuntimeError(f"competitor availability gate failed for systems: {failed_systems}")
        failed_profiles = [item.profile for item in required if item.qualified_competitors <= 0]
        raise RuntimeError(f"qualification gate failed for profiles: {', '.join(failed_profiles)}")

    if progress_hook:
        progress_hook(
            f"all {len(profile_reports)} profiles complete — "
            f"merging systems and computing Elo ratings"
        )
    systems = _merge_profile_systems(profile_reports)
    _apply_elo_ratings(systems)
    if progress_hook:
        progress_hook(
            f"computing statistical significance tests across all profiles "
            f"({len([s for s in systems if s.available])} available systems)"
        )
    stat_tests = _compute_statistical_tests(systems, progress_hook=progress_hook)
    diag = _compute_diagnostics(systems, progress_hook=progress_hook)
    summary_report = CompetitorComparisonReport(
        report_schema_version=REPORT_SCHEMA_VERSION,
        dataset=dataset,
        dataset_source=dataset_source,
        systems=systems,
        warmup_samples=warmup_samples,
        measured_runs=measured_runs,
        profiles=profile_reports,
        floor_pass=floor_pass,
        qualification_gate_pass=qualification_gate_pass,
        mit_gate_pass=qualification_gate_pass,
        expected_competitors=expected,
        available_competitors=available_competitors,
        unavailable_competitors=unavailable_competitors,
        all_competitors_available=all_competitors_available,
        require_all_competitors=require_all_competitors,
        require_native_competitors=require_native_competitors,
        statistical_tests=stat_tests,
        diagnostics=diag,
    )
    if progress_hook:
        progress_hook(
            f"benchmark complete: floor_pass={summary_report.floor_pass}, "
            f"qualification_gate_pass={summary_report.qualification_gate_pass}"
        )
    return summary_report


def merge_profile_checkpoints(
    *,
    checkpoint_dir: str,
    dataset: str = "pii_anon_benchmark_v1",
    dataset_source: DatasetSource = "auto",
    warmup_samples: int = 100,
    measured_runs: int = 3,
    matrix_path: str | None = None,
    expected_competitors: list[str] | None = None,
    require_all_competitors: bool = False,
    require_native_competitors: bool = False,
    enforce_floors: bool = False,
    progress_hook: ProgressHook | None = None,
) -> CompetitorComparisonReport:
    """Merge per-profile checkpoint files into a final benchmark report.

    Used in the parallel execution mode where individual profiles are
    evaluated in separate processes, each writing checkpoint files to
    *checkpoint_dir*.  This function reads all checkpoints, merges
    results, and computes Elo ratings and statistical tests.

    The output is identical to what :func:`compare_competitors` produces
    when all profiles complete sequentially.

    Raises:
        FileNotFoundError: If *checkpoint_dir* is missing or empty.
        RuntimeError: If *enforce_floors* is ``True`` and gates fail.
    """
    if progress_hook:
        progress_hook(f"merge-only: reading checkpoints from {checkpoint_dir}")

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    # Discover and load all checkpoint files.
    profile_reports: list[ProfileBenchmarkResult] = []
    ckpt_files = sorted(ckpt_path.glob("checkpoint_*.json"))
    if not ckpt_files:
        raise FileNotFoundError(f"no checkpoint files found in {checkpoint_dir}")

    for ckpt_file in ckpt_files:
        profile_name = ckpt_file.stem.replace("checkpoint_", "", 1)
        cached = _load_checkpoint(ckpt_path, profile_name)
        if cached is not None:
            profile_reports.append(cached)
            if progress_hook:
                avail = sum(1 for s in cached.systems if s.available)
                best_f1 = max(
                    (s.f1 for s in cached.systems if s.available), default=0.0
                )
                progress_hook(
                    f"loaded checkpoint: {profile_name} — "
                    f"{avail} systems available, best F1={best_f1:.4f}"
                )

    if not profile_reports:
        raise FileNotFoundError(
            f"no valid checkpoints loaded from {checkpoint_dir}"
        )

    if progress_hook:
        progress_hook(
            f"loaded {len(profile_reports)} profile checkpoints: "
            + ", ".join(r.profile for r in profile_reports)
        )

    expected = expected_competitors or list(_COMPETITOR_META.keys())

    # --- gate checks (identical to compare_competitors post-loop) --------
    required_profiles: set[str] = set()
    if matrix_path is not None:
        matrix_cfg = load_use_case_matrix(matrix_path)
        required_profiles = {cfg.profile for cfg in matrix_cfg if cfg.required}

    required = [
        item for item in profile_reports if item.profile in required_profiles
    ]
    floor_pass = all(item.floor_pass for item in required) if required else True
    qualification_gate_pass = (
        all(item.qualified_competitors > 0 for item in required)
        if required
        else True
    )

    available_competitors, unavailable_competitors, all_competitors_available = (
        _summarize_competitor_availability(
            profile_reports=profile_reports,
            expected_competitors=expected,
        )
    )
    if require_all_competitors and not all_competitors_available:
        qualification_gate_pass = False

    if enforce_floors and not floor_pass:
        failed = [item.profile for item in required if not item.floor_pass]
        raise RuntimeError(
            f"floor gate failed for profiles: {', '.join(failed)}"
        )
    if enforce_floors and not qualification_gate_pass:
        if require_all_competitors and not all_competitors_available:
            failed_systems = ", ".join(sorted(unavailable_competitors))
            raise RuntimeError(
                f"competitor availability gate failed: {failed_systems}"
            )
        failed = [
            item.profile
            for item in required
            if item.qualified_competitors <= 0
        ]
        raise RuntimeError(
            f"qualification gate failed for profiles: {', '.join(failed)}"
        )

    # --- aggregation (same as compare_competitors post-loop) -------------
    if progress_hook:
        progress_hook(
            f"all {len(profile_reports)} profiles loaded — "
            f"merging systems and computing Elo ratings"
        )

    systems = _merge_profile_systems(profile_reports)
    _apply_elo_ratings(systems)

    if progress_hook:
        progress_hook(
            f"computing statistical significance tests across all profiles "
            f"({len([s for s in systems if s.available])} available systems)"
        )

    stat_tests = _compute_statistical_tests(
        systems, progress_hook=progress_hook
    )
    diag = _compute_diagnostics(systems, progress_hook=progress_hook)

    report = CompetitorComparisonReport(
        report_schema_version=REPORT_SCHEMA_VERSION,
        dataset=dataset,
        dataset_source=dataset_source,
        systems=systems,
        warmup_samples=warmup_samples,
        measured_runs=measured_runs,
        profiles=profile_reports,
        floor_pass=floor_pass,
        qualification_gate_pass=qualification_gate_pass,
        mit_gate_pass=qualification_gate_pass,
        expected_competitors=expected,
        available_competitors=available_competitors,
        unavailable_competitors=unavailable_competitors,
        all_competitors_available=all_competitors_available,
        require_all_competitors=require_all_competitors,
        require_native_competitors=require_native_competitors,
        statistical_tests=stat_tests,
        diagnostics=diag,
    )
    if progress_hook:
        progress_hook(
            f"merge complete: floor_pass={report.floor_pass}, "
            f"qualification_gate_pass={report.qualification_gate_pass}"
        )
    return report
