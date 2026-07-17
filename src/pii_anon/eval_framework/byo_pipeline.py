"""BYO-pipeline SDK adapter + identical-incumbent scoring (S6-04, DC-12).

Implements **FR-001** (BYO-pipeline adapter contract — score any system with
zero harness-core edits; MUST) and **FR-002** (identical scoring path for
incumbents; SHOULD). Upholds **NFR-026** (entry-point discovery degrades
gracefully — never raises), **AX-001** (no real PII anywhere in this module
or its tests) and **AX-002** (the scoring fields are deterministic; latency
is the only wall-clock-dependent component and is zero-weightable).

Two halves:

1. **The SDK half (FR-001).** :class:`BYOPipelineRegistry` discovers
   :data:`~pii_anon.eval_framework.external_evaluator.Predictor` callables
   advertised under the ``pii_anon.byo_pipelines`` entry-point group — a
   DISTINCT group from ``pii_anon.engines`` / ``pii_anon.rating_engines`` so
   BYO discovery cannot affect detection- or rating-engine discovery. A
   third-party package adds::

       [project.entry-points."pii_anon.byo_pipelines"]
       my-detector = "my_pkg.detector:predict"

   and is discovered with zero pii-anon edits. Discovery resolves function
   objects only — it never constructs an engine, so heavy native models can
   never be pulled at enumeration/import time.

2. **The identical-path half (FR-002).** The in-tree incumbents
   (Presidio / GLiNER / Scrubadub / spaCy / Stanza) are expressed as
   ordinary module-level predictors via :func:`engine_predictor`, making
   them "BYO pipelines that happen to ship in-tree". :func:`evaluate_incumbent`
   is a single delegation to the literal
   :func:`~pii_anon.eval_framework.external_evaluator.evaluate_external_system`
   — it has NO scoring logic of its own (contract-test-pinned), so the
   incumbent scoring path and the BYO scoring path are the same function by
   construction. :func:`build_identical_path_leaderboard` ranks any mix of
   incumbents and BYO systems through that one path.

Honesty note (the committed benchmark artifact): the rows in
``artifacts/benchmarks/benchmark-results.json`` were produced by the frozen
legacy path (``evaluation/competitor_compare.py`` — overlap matching over the
benchmark dataset). The identical path here uses strict span matching over
the eval dataset, so identical-path incumbent numbers will legitimately
differ from the committed artifact rows. FR-002 pins the *path* for all
future scoring, not retroactive artifact parity.

# SWITCH-POINT(ORCH) RESOLVED (sp2): ``pii_anon_predictor`` /
# ``pii_anon_swarm_predictor`` are now LIVE below via the ENGINE seam
# (RegexEngineAdapter + the swarm fusion strategy directly) — the
# orchestrator is not involved, so the S2-03 user-WIP block never applied to
# this path. An orchestrator-mediated variant (full policy/segmentation
# pipeline) remains available as a future enhancement once the user-WIP
# settles.
# SWITCH-POINT(DATA): regenerating the committed benchmark artifact rows
# through the identical path at full census is eval-data Pass-2 (the
# benchmark script also rewrites user-WIP files and must not be run).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib import metadata
from threading import Lock
from typing import TYPE_CHECKING

from .external_evaluator import (
    ExternalEvaluationResult,
    Predictor,
    evaluate_external_system,
)
from .metrics.composite import CompositeConfig, DeploymentProfile
from .rating.leaderboard import Leaderboard
from .rating.scorecard import BenchmarkScorecard

if TYPE_CHECKING:
    from pii_anon.engines.base import EngineAdapter

__all__ = [
    "BYOPipelineRegistry",
    "FIRST_PARTY_SYSTEMS",
    "INCUMBENT_SYSTEMS",
    "build_identical_path_leaderboard",
    "engine_predictor",
    "evaluate_incumbent",
    "first_party_predictor",
    "gliner_predictor",
    "incumbent_predictor",
    "pii_anon_predictor",
    "pii_anon_swarm_predictor",
    "presidio_predictor",
    "scrubadub_predictor",
    "spacy_ner_predictor",
    "stanza_ner_predictor",
]


# ---------------------------------------------------------------------------
# The SDK registry (FR-001)
# ---------------------------------------------------------------------------

class BYOPipelineRegistry:
    """Thread-safe name-keyed registry of BYO :data:`Predictor` callables.

    Mirrors :class:`~pii_anon.eval_framework.rating.registry.RatingEngineRegistry`
    (thread-safe dict, ``register`` / ``get`` / ``names`` / entry-point
    discovery with the ``entry_points().select(group=...)`` + ``hasattr``
    fallback). Predictors have no identity attribute, so they are keyed by
    the entry-point *name* (e.g. ``my-detector``).

    Discovery degrades gracefully (NFR-026): any failure to enumerate or
    load an entry point yields an empty / partial list rather than raising,
    and a target that is not callable is skipped — an absent or broken
    third-party package never breaks callers.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Predictor] = {}
        self._lock = Lock()

    def register(self, name: str, predictor: Predictor) -> None:
        """Register ``predictor`` under ``name`` (overwrites an existing entry).

        Fail-closed: a non-callable is refused with :class:`TypeError` —
        a registry entry that cannot be invoked must never exist.
        """
        if not callable(predictor):
            raise TypeError(
                f"BYO pipeline {name!r} must be callable (a Predictor); "
                f"got {type(predictor).__name__}"
            )
        with self._lock:
            self._pipelines[name] = predictor

    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry (no-op when absent)."""
        with self._lock:
            self._pipelines.pop(name, None)

    def get(self, name: str) -> Predictor | None:
        """Return the predictor registered under ``name``, or ``None``."""
        with self._lock:
            return self._pipelines.get(name)

    def names(self) -> list[str]:
        """Return the registered pipeline names, sorted."""
        with self._lock:
            return sorted(self._pipelines.keys())

    def discover_entrypoint_pipelines(
        self, group: str = "pii_anon.byo_pipelines"
    ) -> list[str]:
        """Discover and register predictors advertised under ``group``.

        Returns the list of registered entry-point names. Targets must be
        callables (module-level functions are the canonical shape); classes
        are NOT instantiated — a class object is itself callable and would
        construct on first *use*, never at discovery time. Any enumeration
        or load failure is swallowed (NFR-026 graceful degradation).
        """
        discovered: list[str] = []
        try:
            entry_points = metadata.entry_points()
            selected: Iterable[metadata.EntryPoint]
            if hasattr(entry_points, "select"):
                selected = entry_points.select(group=group)
            else:
                selected = entry_points.get(group) or ()
        except Exception:
            return discovered

        for ep in selected:
            try:
                loaded = ep.load()
            except Exception:
                continue
            if not callable(loaded):
                continue
            self.register(ep.name, loaded)
            discovered.append(ep.name)
        return discovered


# ---------------------------------------------------------------------------
# EngineAdapter -> Predictor bridge (the FR-002 mechanism)
# ---------------------------------------------------------------------------

def engine_predictor(
    engine: EngineAdapter, *, language: str = "en"
) -> Predictor:
    """Express an :class:`~pii_anon.engines.base.EngineAdapter` as a Predictor.

    The returned callable feeds the text through
    ``engine.detect({"text": text}, {"language": language})`` and maps each
    finding to an ``(entity_type, start, end)`` tuple. Findings with absent
    (``None``) spans are dropped, and **bool spans are dropped explicitly**
    — ``bool`` is an ``int`` subclass in Python, so ``int(True) == 1`` would
    otherwise silently coerce into a phantom offset downstream.

    Engine exceptions PROPAGATE: they hit the exact same ``on_error``
    user-code boundary inside :func:`evaluate_external_system` that any BYO
    predictor hits — error handling is part of the identical path.
    """

    def _predict(text: str) -> list[tuple[str, int, int]]:
        findings = engine.detect({"text": text}, {"language": language})
        spans: list[tuple[str, int, int]] = []
        for finding in findings:
            start = finding.span_start
            end = finding.span_end
            if isinstance(start, bool) or isinstance(end, bool):
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            spans.append((str(finding.entity_type), start, end))
        return spans

    return _predict


# ---------------------------------------------------------------------------
# The five in-tree incumbents, expressed as ordinary BYO predictors
# ---------------------------------------------------------------------------
# Engines are imported and constructed LAZILY on first predictor call —
# never at module import or entry-point discovery time — so optional native
# dependencies (e.g. the GLiNER model) cannot be pulled by enumeration.

def _gliner_factory() -> Predictor:
    from pii_anon.engines.gliner_adapter import GLiNERAdapter

    return engine_predictor(GLiNERAdapter(enabled=True))


def _presidio_factory() -> Predictor:
    from pii_anon.engines.presidio_adapter import PresidioAdapter

    return engine_predictor(PresidioAdapter(enabled=True))


def _scrubadub_factory() -> Predictor:
    from pii_anon.engines.scrubadub_adapter import ScrubadubAdapter

    return engine_predictor(ScrubadubAdapter(enabled=True))


def _spacy_ner_factory() -> Predictor:
    from pii_anon.engines.spacy_adapter import SpacyNERAdapter

    return engine_predictor(SpacyNERAdapter(enabled=True))


def _stanza_ner_factory() -> Predictor:
    from pii_anon.engines.stanza_adapter import StanzaNERAdapter

    return engine_predictor(StanzaNERAdapter(enabled=True))


_INCUMBENT_FACTORIES: dict[str, Callable[[], Predictor]] = {
    "gliner": _gliner_factory,
    "presidio": _presidio_factory,
    "scrubadub": _scrubadub_factory,
    "spacy_ner": _spacy_ner_factory,
    "stanza_ner": _stanza_ner_factory,
}

#: The in-tree incumbent systems scorable on the identical path (sorted).
INCUMBENT_SYSTEMS: tuple[str, ...] = tuple(sorted(_INCUMBENT_FACTORIES))

_PREDICTOR_CACHE: dict[str, Predictor] = {}
_CACHE_LOCK = Lock()


def incumbent_predictor(name: str) -> Predictor:
    """Return the (lazily constructed, cached) predictor for an incumbent.

    Unknown names are refused fail-closed with a domain-named
    :class:`ValueError`. Construction happens once per process under a lock;
    caching never changes outputs (AX-002 — the engines are deterministic
    given their fixed configuration).
    """
    factory = _INCUMBENT_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"unknown incumbent pipeline {name!r}; "
            f"expected one of {INCUMBENT_SYSTEMS}"
        )
    with _CACHE_LOCK:
        predictor = _PREDICTOR_CACHE.get(name)
        if predictor is None:
            predictor = factory()
            _PREDICTOR_CACHE[name] = predictor
    return predictor


# Module-level entry-point targets (``pii_anon.byo_pipelines`` group).
# Plain functions: ``ep.load()`` resolves them without constructing engines.

def gliner_predictor(text: str) -> list[tuple[str, int, int]]:
    """GLiNER expressed as an ordinary BYO ``Predictor`` (FR-002)."""
    return list(incumbent_predictor("gliner")(text))


def presidio_predictor(text: str) -> list[tuple[str, int, int]]:
    """Presidio expressed as an ordinary BYO ``Predictor`` (FR-002)."""
    return list(incumbent_predictor("presidio")(text))


def scrubadub_predictor(text: str) -> list[tuple[str, int, int]]:
    """Scrubadub expressed as an ordinary BYO ``Predictor`` (FR-002)."""
    return list(incumbent_predictor("scrubadub")(text))


def spacy_ner_predictor(text: str) -> list[tuple[str, int, int]]:
    """spaCy NER expressed as an ordinary BYO ``Predictor`` (FR-002)."""
    return list(incumbent_predictor("spacy_ner")(text))


def stanza_ner_predictor(text: str) -> list[tuple[str, int, int]]:
    """Stanza NER expressed as an ordinary BYO ``Predictor`` (FR-002)."""
    return list(incumbent_predictor("stanza_ner")(text))


# ---------------------------------------------------------------------------
# First-party systems (sp2): pii-anon itself, expressed as BYO predictors
# ---------------------------------------------------------------------------
# The first-party factories live in the sibling ``first_party`` module so the
# engine/fusion imports they need stay OUT of this BYO-SDK module — preserving
# its A13 import-boundary audit (no swarm/moe/fusion/orchestrator imports).
# Re-exported here for backward-compatible call sites. A relative re-export is
# A13-clean (head is ``first_party``, not ``pii_anon.fusion``).
from .first_party import (  # noqa: E402 - re-export after engine_predictor is defined
    FIRST_PARTY_SYSTEMS,
    first_party_predictor,
    pii_anon_predictor,
    pii_anon_swarm_predictor,
)


# ---------------------------------------------------------------------------
# The identical-path entry points (FR-002)
# ---------------------------------------------------------------------------

def evaluate_incumbent(
    name: str,
    *,
    dataset: str = "pii_anon",
    language: str | None = None,
    max_records: int | None = None,
    warmup_records: int = 10,
    deployment_profile: DeploymentProfile | None = None,
    composite_config: CompositeConfig | None = None,
    on_error: str = "skip",
) -> ExternalEvaluationResult:
    """Score an in-tree incumbent through the IDENTICAL BYO scoring path.

    The body is a single delegation to the literal
    :func:`evaluate_external_system` — this function carries NO scoring
    logic of its own (the FR-002 contract, pinned by the story's A5/A6
    contract tests). Every keyword is passed through unchanged.
    """
    return evaluate_external_system(
        incumbent_predictor(name),
        system_name=name,
        dataset=dataset,
        language=language,
        max_records=max_records,
        warmup_records=warmup_records,
        deployment_profile=deployment_profile,
        composite_config=composite_config,
        on_error=on_error,
    )


def build_identical_path_leaderboard(
    systems: Mapping[str, Predictor],
    *,
    dataset: str = "pii_anon",
    language: str | None = None,
    max_records: int | None = None,
    warmup_records: int = 10,
    deployment_profile: DeploymentProfile | None = None,
    composite_config: CompositeConfig | None = None,
    on_error: str = "skip",
) -> tuple[Leaderboard, dict[str, ExternalEvaluationResult]]:
    """Rank any mix of incumbents and BYO systems through the ONE path.

    Every system — incumbent or third-party, indistinguishable here — is
    evaluated by :func:`evaluate_external_system` in sorted-name order
    (AX-002: a deterministic evaluation order), assembled into one
    :class:`BenchmarkScorecard`, and ranked by the same Elo round-robin.
    Same dataset, same strict matching, same composite, for every row.

    Returns the ranked :class:`Leaderboard` plus the full per-system
    :class:`ExternalEvaluationResult` bundles keyed by system name.
    """
    if not systems:
        raise ValueError(
            "systems must contain at least one (name, Predictor) pair"
        )
    results: dict[str, ExternalEvaluationResult] = {}
    bench = BenchmarkScorecard(
        benchmark_name="pii-rate-elo-identical-path",
        dataset_name=dataset,
    )
    for name in sorted(systems):
        result = evaluate_external_system(
            systems[name],
            system_name=name,
            dataset=dataset,
            language=language,
            max_records=max_records,
            warmup_records=warmup_records,
            deployment_profile=deployment_profile,
            composite_config=composite_config,
            on_error=on_error,
        )
        results[name] = result
        bench.add_system(result.scorecard)
    return Leaderboard.from_benchmark_scorecard(bench), results
