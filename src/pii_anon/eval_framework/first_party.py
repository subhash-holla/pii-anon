"""First-party systems (sp2): pii-anon itself, expressed as BYO predictors.

Native-label expression of pii-anon's own two detection surfaces — the
vanilla regex engine and the swarm fusion pipeline — as ordinary
:data:`~pii_anon.eval_framework.external_evaluator.Predictor` callables.
Labels are the engines' NATIVE entity types (``EMAIL_ADDRESS``, ``IBAN``,
``US_SSN``, ...), NOT the benchmark-canonical folding
``evaluation/competitor_compare.py`` applies (which e.g. collapses ``IBAN``
into ``BANK_ACCOUNT``): an external harness owns its own label projection,
and a lossy pre-fold here would forfeit types its taxonomy distinguishes.
Built on the ENGINE seam — the orchestrator is not involved.

Import discipline: this module lives OUTSIDE ``byo_pipeline.py`` so the BYO
SDK module keeps its A13 audit (no swarm/moe/fusion/orchestrator imports —
it stays standalone). Here, every engine/fusion import happens INSIDE the
factory functions, so importing this module (or ``byo_pipeline``, which
re-exports the public names) still pulls no detection machinery.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from threading import Lock
from typing import TYPE_CHECKING

from .external_evaluator import Predictor

if TYPE_CHECKING:
    from pii_anon.engines.base import EngineAdapter
    from pii_anon.types import EngineFinding

__all__ = [
    "FIRST_PARTY_SYSTEMS",
    "first_party_predictor",
    "pii_anon_predictor",
    "pii_anon_swarm_predictor",
]


def _pii_anon_factory() -> Predictor:
    from pii_anon.engines.regex_adapter import RegexEngineAdapter

    from .byo_pipeline import engine_predictor

    # eval_cross_type_arbitration=True: this is the EVAL path (scoring, never
    # masking), so the benchmark-precision arbitration (drop a PERSON FP on a
    # real job title / org / condition) is correct and safe here.
    return engine_predictor(RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True))


#: (find_spec probe, adapter module, adapter class) per optional swarm engine.
#: The pool mirrors the canonical pii-anon-swarm ensemble composition
#: (``evaluation/competitor_compare._ensemble_detector``: presidio +
#: scrubadub + gliner over the regex shared layer). Stanza is deliberately
#: NOT pooled: it is not part of the canonical ensemble and its CPU
#: inference (~300ms/record) dominates swarm latency for marginal recall.
_SWARM_POOL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("gliner", "pii_anon.engines.gliner_adapter", "GLiNERAdapter"),
    ("presidio_analyzer", "pii_anon.engines.presidio_adapter", "PresidioAdapter"),
    ("scrubadub", "pii_anon.engines.scrubadub_adapter", "ScrubadubAdapter"),
)


def _swarm_pool() -> list[EngineAdapter]:
    """The swarm engine pool: regex always; heavy engines join when importable.

    Probes with ``find_spec`` first so a missing optional dependency costs no
    import attempt; a present-but-broken engine is skipped (degraded host),
    never fatal — the pool always retains the regex shared layer.
    """
    from pii_anon.engines.regex_adapter import RegexEngineAdapter

    # EVAL path → arbitration on (benchmark precision); see _pii_anon_factory.
    engines: list[EngineAdapter] = [
        RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)
    ]
    for probe, module_name, cls_name in _SWARM_POOL_SPECS:
        if importlib.util.find_spec(probe) is None:
            continue
        try:
            adapter_cls = getattr(importlib.import_module(module_name), cls_name)
            engines.append(adapter_cls(enabled=True))
        except Exception:  # noqa: BLE001 - a broken optional engine degrades the pool, never kills it
            continue
    return engines


def _swarm_predictor_from_engines(
    engines: Sequence[EngineAdapter], *, language: str = "en"
) -> Predictor:
    """Predictor running an engine pool through the swarm fusion strategy.

    Each engine's raw findings (native labels, engine-stamped ``engine_id``)
    are pooled and merged by ``build_fusion("swarm")`` — the floor-projecting
    swarm strategy, so a regex-oss span the Layer-4 gate drops is re-injected
    (FR-016/AX-003). A single failing engine is skipped (the swarm degrades;
    it never dies with its weakest expert). Span guards mirror
    :func:`~pii_anon.eval_framework.byo_pipeline.engine_predictor` (bool
    offsets dropped explicitly — ``bool`` is an ``int`` subclass, so
    ``int(True)`` would otherwise coerce into a phantom offset downstream).
    """
    from pii_anon.fusion import build_fusion

    fusion = build_fusion("swarm", weights={}, min_consensus=1)

    def _predict(text: str) -> list[tuple[str, int, int]]:
        pooled: list[EngineFinding] = []
        for engine in engines:
            try:
                pooled.extend(engine.detect({"text": text}, {"language": language}))
            except Exception:  # noqa: BLE001 - degraded engine: its findings are simply absent
                continue
        spans: list[tuple[str, int, int]] = []
        for finding in fusion.merge(pooled):
            start = finding.span_start
            end = finding.span_end
            if isinstance(start, bool) or isinstance(end, bool):
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            spans.append((str(finding.entity_type), start, end))
        return spans

    return _predict


def _pii_anon_swarm_factory() -> Predictor:
    return _swarm_predictor_from_engines(_swarm_pool())


_FIRST_PARTY_FACTORIES: dict[str, Callable[[], Predictor]] = {
    "pii_anon": _pii_anon_factory,
    "pii_anon_swarm": _pii_anon_swarm_factory,
}

#: The first-party pii-anon systems scorable as ordinary BYO pipelines (sorted).
FIRST_PARTY_SYSTEMS: tuple[str, ...] = tuple(sorted(_FIRST_PARTY_FACTORIES))

_FIRST_PARTY_CACHE: dict[str, Predictor] = {}
_FIRST_PARTY_CACHE_LOCK = Lock()


def first_party_predictor(name: str) -> Predictor:
    """Return the (lazily constructed, cached) predictor for a first-party system.

    Unknown names are refused fail-closed with a domain-named
    :class:`ValueError` (mirrors
    :func:`~pii_anon.eval_framework.byo_pipeline.incumbent_predictor`).
    Construction is once-per-process under a lock; heavy engine weights load
    on first predictor CALL, never at construction.
    """
    factory = _FIRST_PARTY_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"unknown first-party pipeline {name!r}; "
            f"expected one of {FIRST_PARTY_SYSTEMS}"
        )
    with _FIRST_PARTY_CACHE_LOCK:
        predictor = _FIRST_PARTY_CACHE.get(name)
        if predictor is None:
            predictor = factory()
            _FIRST_PARTY_CACHE[name] = predictor
    return predictor


def pii_anon_predictor(text: str) -> list[tuple[str, int, int]]:
    """Vanilla pii-anon (the regex engine) as an ordinary BYO ``Predictor``."""
    return list(first_party_predictor("pii_anon")(text))


def pii_anon_swarm_predictor(text: str) -> list[tuple[str, int, int]]:
    """pii-anon-swarm (engine pool + swarm fusion) as an ordinary BYO ``Predictor``."""
    return list(first_party_predictor("pii_anon_swarm")(text))
