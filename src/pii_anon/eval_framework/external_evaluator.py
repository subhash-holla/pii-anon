"""End-to-end evaluation for user-supplied PII pipelines.

This module is the public entry point for anyone who wants to score
*their own* PII detector against the ``pii-anon`` benchmark and get a
pii-rate-elo ranking alongside the published baselines (``pii-anon``,
``pii-anon-swarm``, Presidio, GLiNER, Scrubadub) without having to
install every competitor package or reimplement the scoring pipeline.

Typical usage::

    from pii_anon.eval_framework import (
        evaluate_external_system,
        load_baseline_leaderboard,
    )

    def my_detector(text: str) -> list[tuple[str, int, int]]:
        # Return (entity_type, start, end) tuples.
        ...

    scorecard = evaluate_external_system(
        my_detector,
        system_name="my-detector",
        max_records=2_000,
    )
    leaderboard = load_baseline_leaderboard().with_scorecard(scorecard)
    print(leaderboard.to_markdown())

The underlying machinery — ``load_eval_dataset``, ``compute_composite``,
``PIIRateEloEngine`` — is unchanged; this module wires them together so
callers do not need to know the full graph.
"""

from __future__ import annotations

import importlib
import json
import statistics
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Iterable

from .datasets.schema import load_eval_dataset
from .metrics.composite import (
    CompositeConfig,
    CompositeScore,
    DeploymentProfile,
    compute_composite,
)
from .rating.leaderboard import Leaderboard
from .rating.scorecard import BenchmarkScorecard, SystemScorecard


# ---------------------------------------------------------------------------
# User-facing types
# ---------------------------------------------------------------------------

#: Predictor signature accepted by :func:`evaluate_external_system`.
#:
#: The callable takes a plain Unicode string and returns an iterable of
#: ``(entity_type, start, end)`` tuples where ``start``/``end`` are
#: 0-indexed half-open offsets into the input string.  Extra tuple
#: elements (e.g. confidence, a fourth field) are ignored.
Predictor = Callable[[str], Iterable[tuple[str, int, int]]]


@dataclass
class ExternalEvaluationResult:
    """Full result bundle from :func:`evaluate_external_system`.

    Attributes
    ----------
    scorecard:
        ``SystemScorecard`` suitable for insertion into a ``Leaderboard``.
    composite:
        Full ``CompositeScore`` with per-component breakdown.
    per_record_f1:
        F1 computed on each record — used for bootstrap CI and paired
        significance testing against baseline systems.
    latency_ms_samples:
        Raw per-record latency samples (wall-clock milliseconds).
    records_evaluated:
        Number of records the predictor was actually run on.
    skipped_records:
        Records dropped because the predictor raised an exception.
    errors:
        First few predictor errors (truncated for readability).
    """

    scorecard: SystemScorecard
    composite: CompositeScore
    per_record_f1: list[float] = field(default_factory=list)
    latency_ms_samples: list[float] = field(default_factory=list)
    records_evaluated: int = 0
    skipped_records: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON export."""
        f1s = self.per_record_f1
        return {
            "scorecard": self.scorecard.to_dict(),
            "composite": self.composite.to_dict(),
            "records_evaluated": self.records_evaluated,
            "skipped_records": self.skipped_records,
            "f1_mean": statistics.fmean(f1s) if f1s else 0.0,
            "f1_stdev": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Label / prediction bookkeeping
# ---------------------------------------------------------------------------

def _labels_as_span_set(labels: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    return {(lbl["entity_type"], lbl["start"], lbl["end"]) for lbl in labels}


def _predictions_as_span_set(
    predictions: Iterable[tuple[str, int, int]] | Iterable[Any],
    text_length: int,
) -> set[tuple[str, int, int]]:
    """Coerce the predictor output into a strict span set.

    Predictions with out-of-bounds offsets, reversed spans, or empty
    entity types are dropped silently — this mirrors how the main
    evaluation framework treats malformed spans.
    """
    spans: set[tuple[str, int, int]] = set()
    for item in predictions:
        try:
            entity_type, start, end = item[0], int(item[1]), int(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not entity_type or start < 0 or end <= start or end > text_length:
            continue
        spans.add((str(entity_type), start, end))
    return spans


def _record_f1(
    pred_spans: set[tuple[str, int, int]],
    label_spans: set[tuple[str, int, int]],
) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` for strict span matching on one record."""
    tp = len(pred_spans & label_spans)
    fp = len(pred_spans - label_spans)
    fn = len(label_spans - pred_spans)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_external_system(
    predictor: Predictor,
    *,
    system_name: str = "external-system",
    dataset: str = "pii_anon",
    language: str | None = None,
    max_records: int | None = None,
    warmup_records: int = 10,
    deployment_profile: DeploymentProfile | None = None,
    composite_config: CompositeConfig | None = None,
    on_error: str = "skip",
) -> ExternalEvaluationResult:
    """Score a user-supplied PII detector against the ``pii-anon`` benchmark.

    Parameters
    ----------
    predictor:
        Callable with signature ``predict(text: str) -> Iterable[tuple[str, int, int]]``
        returning ``(entity_type, start, end)`` tuples.  See :data:`Predictor`.
    system_name:
        Display name for the resulting scorecard — appears in the
        leaderboard.  Keep it short (e.g. ``"my-detector"``).
    dataset:
        Dataset filename to load (without extension).  Defaults to the
        v1.1+ canonical ``"pii_anon"``; legacy names are probed as fallbacks.
    language:
        Optional BCP 47 filter (e.g. ``"en"``).
    max_records:
        Cap the number of evaluation records.  ``None`` = all records.
    warmup_records:
        Number of records to warm the predictor on before measuring
        latency.  Warmup predictions still contribute to F1/precision/recall.
    deployment_profile:
        One of ``"standard"``, ``"high_security"``, ``"high_throughput"``
        — selects a preset weight mix via
        :meth:`CompositeConfig.for_deployment`.  Mutually exclusive with
        *composite_config*.
    composite_config:
        Fully-custom ``CompositeConfig``.  Wins over *deployment_profile*
        if both are provided.
    on_error:
        Behavior when the predictor raises on a record:
        ``"skip"`` (default) — drop the record and continue;
        ``"raise"`` — re-raise the original exception with the record id
        attached.

    Returns
    -------
    ExternalEvaluationResult
        Scorecard, full composite breakdown, and per-record diagnostics
        suitable for bootstrap CI and leaderboard insertion.
    """
    if composite_config is not None and deployment_profile is not None:
        raise ValueError(
            "pass either `composite_config` or `deployment_profile`, not both"
        )
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")

    records = load_eval_dataset(dataset, language=language)
    if max_records is not None:
        records = records[: max(0, max_records)]
    if not records:
        raise ValueError(
            f"dataset `{dataset}` yielded zero records for the given filters"
        )

    if composite_config is not None:
        cfg = composite_config
    elif deployment_profile is not None:
        cfg = CompositeConfig.for_deployment(deployment_profile)
    else:
        cfg = CompositeConfig()

    # Warmup — run the predictor on the first *warmup_records* rows so
    # any JIT, weight-loading, or caching overhead is excluded from the
    # measured latency distribution.  Warmup records are later processed
    # in the main loop for F1 scoring, but their latency samples are not
    # recorded (their first-call timings would skew p50 downward for
    # stateful predictors that populate caches on first invocation).
    warmup_count = min(warmup_records, len(records))
    for record in records[:warmup_count]:
        try:
            list(predictor(record.text))
        except Exception:
            # Warmup errors are non-fatal — we still want latency
            # measurement to proceed.  They surface again in the main loop.
            pass

    entity_types_seen: set[str] = set()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_record_f1: list[float] = []
    latency_ms_samples: list[float] = []
    errors: list[str] = []
    skipped = 0
    evaluated = 0

    for idx, record in enumerate(records):
        label_spans = _labels_as_span_set(record.labels)
        try:
            # Materialise the predictor output (in case it's a generator)
            # *before* stopping the clock so span-coercion overhead is
            # not folded into the measured inference time.
            start = time.perf_counter()
            raw_predictions = list(predictor(record.text))
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            pred_spans = _predictions_as_span_set(
                raw_predictions, text_length=len(record.text)
            )
        except Exception as exc:  # noqa: BLE001 — user-code boundary
            if on_error == "raise":
                raise type(exc)(
                    f"predictor raised on record_id={record.record_id}: {exc}"
                ) from exc
            skipped += 1
            if len(errors) < 5:
                errors.append(f"{record.record_id}: {exc!r}")
            continue

        evaluated += 1
        # Only record latency after warmup so cold-start overhead
        # doesn't pull p50 down for stateful predictors.
        if idx >= warmup_count:
            latency_ms_samples.append(elapsed_ms)

        tp, fp, fn = _record_f1(pred_spans, label_spans)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        denom = 2 * tp + fp + fn
        per_record_f1.append((2 * tp) / denom if denom > 0 else 1.0)

        for et, _, _ in pred_spans:
            entity_types_seen.add(et)
        for et, _, _ in label_spans:
            entity_types_seen.add(et)

    if evaluated == 0:
        raise RuntimeError(
            f"predictor produced zero successful predictions across {len(records)} records"
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    # Fall back to the full distribution (including warmup) if every
    # measured record hit an error — avoids ``median([])``.
    measured = latency_ms_samples or [0.0]
    latency_p50 = statistics.median(measured)
    docs_per_hour = 3_600_000.0 / latency_p50 if latency_p50 > 0 else 0.0

    # Count entity-type coverage against the ground-truth types seen
    # in the *evaluated slice* — keeps the denominator honest when the
    # user filters the benchmark (e.g. English-only runs).
    gold_types: set[str] = set()
    for record in records:
        for label in record.labels:
            gold_types.add(label["entity_type"])
    entity_types_detected = sum(1 for et in gold_types if et in entity_types_seen)
    entity_types_total = len(gold_types)

    composite = compute_composite(
        f1=f1,
        precision=precision,
        recall=recall,
        latency_ms=latency_p50,
        docs_per_hour=docs_per_hour,
        entity_types_detected=entity_types_detected,
        entity_types_total=entity_types_total,
        config=cfg,
    )

    scorecard = SystemScorecard(
        system_name=system_name,
        available=True,
        f1=f1,
        precision=precision,
        recall=recall,
        latency_p50_ms=latency_p50,
        docs_per_hour=docs_per_hour,
        composite_score=composite.score,
        samples=evaluated,
        evaluation_track="detect_only",
    )

    return ExternalEvaluationResult(
        scorecard=scorecard,
        composite=composite,
        per_record_f1=per_record_f1,
        latency_ms_samples=latency_ms_samples,
        records_evaluated=evaluated,
        skipped_records=skipped,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Baseline leaderboard (no competitor packages required)
# ---------------------------------------------------------------------------

# The canonical benchmark artifact shipped with the library — produced by
# ``scripts/run_full_benchmark.py`` and vendored into the package tree
# (``eval_framework/baselines/benchmark-results.json``) so ``pip install
# pii-anon`` users can compare against the published numbers without
# re-running competitors.
_PACKAGE_BASELINE_ARTIFACT = "benchmark-results.json"
_REPO_BENCHMARK_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "benchmarks"
    / "benchmark-results.json"
)


def _default_baseline_artifact_path() -> Path:
    """Resolve the baseline benchmark artifact.

    Tries the package-relative location first (works from wheels), then
    falls back to the repo-relative path for source checkouts that have
    run ``make benchmark-full``.
    """
    try:
        pkg_path = resources.files("pii_anon.eval_framework.baselines").joinpath(
            _PACKAGE_BASELINE_ARTIFACT,
        )
        candidate = Path(str(pkg_path))
        if candidate.exists():
            return candidate
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    return _REPO_BENCHMARK_ARTIFACT


def _scorecard_from_system_entry(entry: dict[str, Any]) -> SystemScorecard:
    """Reconstruct a :class:`SystemScorecard` from one ``systems[i]`` entry."""
    return SystemScorecard(
        system_name=str(entry.get("system", "unknown")),
        available=bool(entry.get("available", True)),
        f1=float(entry.get("f1", 0.0) or 0.0),
        precision=float(entry.get("precision", 0.0) or 0.0),
        recall=float(entry.get("recall", 0.0) or 0.0),
        latency_p50_ms=float(entry.get("latency_p50_ms", 0.0) or 0.0),
        docs_per_hour=float(entry.get("docs_per_hour", 0.0) or 0.0),
        composite_score=float(entry.get("composite_score", 0.0) or 0.0),
        elo_rating=float(entry.get("elo_rating", 1500.0) or 1500.0),
        elo_rd=float(entry.get("elo_rd", 350.0) or 350.0),
        samples=int(entry.get("samples", 0) or 0),
        evaluation_track=str(entry.get("evaluation_track", "detect_only")),
        license_name=entry.get("license_name"),
    )


def load_baseline_leaderboard(
    artifact_path: Path | str | None = None,
) -> "BaselineLeaderboard":
    """Load the checked-in baseline leaderboard.

    Returns scorecards for the systems that ship alongside the library
    (``pii-anon``, ``pii-anon-swarm``, and — when a fresh benchmark has
    been run — Presidio, GLiNER, Scrubadub).  No competitor packages need
    to be installed to read these numbers; they come from the committed
    ``artifacts/benchmarks/benchmark-results.json``.

    Parameters
    ----------
    artifact_path:
        Override the default artifact location.  If ``None``, reads the
        committed benchmark artifact.

    Returns
    -------
    BaselineLeaderboard
        Baselines ready for comparison via ``.with_scorecard(...)``.
    """
    path = (
        Path(artifact_path)
        if artifact_path is not None
        else _default_baseline_artifact_path()
    )
    if not path.exists():
        raise FileNotFoundError(
            f"benchmark artifact not found at {path}. The library ships "
            "with a vendored baseline at "
            "`pii_anon/eval_framework/baselines/benchmark-results.json`; "
            "if you see this error the package-data entry is missing. "
            "Run `make benchmark-full` to regenerate locally, or pass "
            "`artifact_path=` explicitly."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    systems = payload.get("systems") or []

    scorecards: list[SystemScorecard] = [
        _scorecard_from_system_entry(e) for e in systems if e.get("system")
    ]
    return BaselineLeaderboard(
        dataset=str(payload.get("dataset", "pii_anon")),
        max_samples=payload.get("max_samples"),
        scorecards=scorecards,
    )


@dataclass
class BaselineLeaderboard:
    """Baseline scorecards with convenience methods for head-to-head comparison.

    Returned by :func:`load_baseline_leaderboard`.  Users call
    :meth:`with_scorecard` to splice in their own system and get a
    ranked :class:`~pii_anon.eval_framework.rating.leaderboard.Leaderboard`.
    """

    dataset: str
    scorecards: list[SystemScorecard]
    max_samples: int | None = None

    def system_names(self) -> list[str]:
        """Return the names of the baseline systems, sorted alphabetically."""
        return sorted(sc.system_name for sc in self.scorecards)

    def with_scorecard(
        self,
        scorecard: SystemScorecard,
        *,
        replace: bool = True,
    ) -> Leaderboard:
        """Splice *scorecard* into the baselines and produce a ranked leaderboard.

        The Elo tournament is re-run across all systems so *scorecard*
        gets a rating grounded in the same round-robin that produced the
        baseline ratings.  Prior baseline ratings are discarded — only
        composite scores drive the tournament.

        Parameters
        ----------
        scorecard:
            User system scorecard (typically from
            :func:`evaluate_external_system`).
        replace:
            When ``True`` (default) a baseline scorecard with the same
            ``system_name`` is silently overwritten — the common case of
            iterating on your own detector.  Set to ``False`` to raise
            if you explicitly want to avoid shadowing a baseline.

        Returns
        -------
        Leaderboard
            Sorted by Elo rating, high to low.
        """
        existing_names = {sc.system_name for sc in self.scorecards}
        if scorecard.system_name in existing_names and not replace:
            raise ValueError(
                f"baseline already contains a system named "
                f"`{scorecard.system_name}`; pass replace=True to overwrite"
            )
        merged: list[SystemScorecard] = [
            sc for sc in self.scorecards if sc.system_name != scorecard.system_name
        ]
        merged.append(scorecard)

        bench = BenchmarkScorecard(
            benchmark_name="pii-rate-elo-external",
            dataset_name=self.dataset,
        )
        for sc in merged:
            bench.add_system(sc)
        return Leaderboard.from_benchmark_scorecard(bench)


# ---------------------------------------------------------------------------
# CLI helper — resolve "module:callable" paths to predictors
# ---------------------------------------------------------------------------

def resolve_predictor_path(path: str) -> Predictor:
    """Resolve a ``"module.submod:callable"`` string to the predictor.

    Used by the ``pii-anon rate-elo`` CLI so users can run::

        pii-anon rate-elo --predictor my_pkg.detector:predict

    without writing a Python shim.  The resolved object must be callable;
    anything else raises :class:`TypeError`.
    """
    if ":" not in path:
        raise ValueError(
            "predictor path must be `module.submod:callable` "
            f"(got {path!r})"
        )
    module_name, _, attr = path.partition(":")
    module = importlib.import_module(module_name)
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {attr!r}"
        ) from exc
    if not callable(obj):
        raise TypeError(
            f"{path} resolved to non-callable {type(obj).__name__}"
        )
    # We cannot statically verify the signature of user-provided code —
    # the cast is a deliberate contract boundary.  Runtime errors from a
    # malformed return type surface via :func:`_predictions_as_span_set`.
    return obj  # type: ignore[no-any-return]


__all__ = [
    "Predictor",
    "ExternalEvaluationResult",
    "BaselineLeaderboard",
    "evaluate_external_system",
    "load_baseline_leaderboard",
    "resolve_predictor_path",
]
