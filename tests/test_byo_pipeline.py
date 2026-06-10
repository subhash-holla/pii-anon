"""S6-04 — BYO-pipeline SDK adapter + identical-incumbent scoring (DC-12).

Implements FR-001 (BYO-pipeline adapter contract — score any system, zero
harness-core edits; MUST) + FR-002 (identical scoring path for incumbents;
SHOULD). Upholds NFR-026 (discovery degrades gracefully), AX-001 (synthetic
fixtures only) and AX-002 (deterministic scoring fields).

Anchors A1–A13 per the story contract
(dev-assist-artifacts/04-development/02-stories/sprint-6/
S6-04-byo-pipeline-sdk-adapter.md). Exact-value anchors throughout — never
bounds (the recurring S5-02/S5-03/S6-01 review lesson). No native models are
ever invoked (stub engines only; the always-available smoke uses the
stdlib-backed ``RegexEngineAdapter``).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

import pii_anon.eval_framework.byo_pipeline as byo
import pii_anon.eval_framework.external_evaluator as ext
from pii_anon.engines.base import EngineAdapter
from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.eval_framework import (
    INCUMBENT_SYSTEMS,
    BYOPipelineRegistry,
    CompositeConfig,
    build_identical_path_leaderboard,
    engine_predictor,
    evaluate_external_system,
    evaluate_incumbent,
    incumbent_predictor,
    load_baseline_leaderboard,
)
from pii_anon.types import EngineFinding, Payload


# ---------------------------------------------------------------------------
# Synthetic fixtures (AX-001 — no real PII; synthetic-shaped values only)
# ---------------------------------------------------------------------------

class _MiniRecord:
    """Duck-typed stand-in for ``EvalBenchmarkRecord`` (record_id/text/labels)."""

    def __init__(self, record_id: str, text: str, labels: list[dict[str, Any]]):
        self.record_id = record_id
        self.text = text
        self.labels = labels


def _mini_dataset(*_args: Any, **_kwargs: Any) -> list[_MiniRecord]:
    """3-record synthetic mini-dataset with hand-countable tp/fp/fn.

    Record r1: 1 label, predictor hits it exactly        -> tp=1 fp=0 fn=0
    Record r2: 1 label, predictor returns nothing        -> tp=0 fp=0 fn=1
    Record r3: 2 labels, predictor hits exactly one      -> tp=1 fp=0 fn=1
    Totals: tp=2 fp=0 fn=2 -> precision=1.0, recall=0.5, f1=2/3.
    """
    return [
        _MiniRecord(
            "r1",
            "mail zz-alias@synthetic.test today",
            [{"entity_type": "EMAIL_ADDRESS", "start": 5, "end": 28}],
        ),
        _MiniRecord(
            "r2",
            "call 000-0000 now",
            [{"entity_type": "PHONE_NUMBER", "start": 5, "end": 13}],
        ),
        _MiniRecord(
            "r3",
            "Zz Aa and Yy Bb",
            [
                {"entity_type": "PERSON_NAME", "start": 0, "end": 5},
                {"entity_type": "PERSON_NAME", "start": 10, "end": 15},
            ],
        ),
    ]


def _stub_predictor(text: str) -> list[tuple[str, int, int]]:
    """Deterministic predictor matched to ``_mini_dataset`` (see its math)."""
    if text.startswith("mail "):
        return [("EMAIL_ADDRESS", 5, 28)]
    if text.startswith("Zz Aa"):
        return [("PERSON_NAME", 0, 5)]
    return []


def _empty_predictor(_text: str) -> list[tuple[str, int, int]]:
    return []


_ZERO_LATENCY_CFG = CompositeConfig(weight_latency=0.0, weight_throughput=0.0)


class _StubEngine(EngineAdapter):
    """Stub EngineAdapter emitting one good + two malformed findings."""

    adapter_id = "stub-engine"

    def detect(
        self, payload: Payload, context: dict[str, Any]
    ) -> list[EngineFinding]:
        del payload, context
        return [
            EngineFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.9,
                span_start=5,
                span_end=28,
            ),
            # Malformed: span offsets absent -> must be dropped.
            EngineFinding(entity_type="PHONE_NUMBER", confidence=0.9),
            # Malformed: bool spans (bool-is-int trap) -> must be dropped.
            EngineFinding(
                entity_type="US_SSN",
                confidence=0.9,
                span_start=True,  # type: ignore[arg-type]
                span_end=True,  # type: ignore[arg-type]
            ),
        ]


# ---------------------------------------------------------------------------
# A1 — registry round-trip
# ---------------------------------------------------------------------------

def test_a1_registry_round_trip_and_fail_closed_on_non_callable() -> None:
    """[UNIT-TEST] A1: register/get/names(sorted)/unregister work; a
    non-callable registration is refused with TypeError (fail-closed)."""
    reg = BYOPipelineRegistry()
    reg.register("zeta", _stub_predictor)
    reg.register("alpha", _empty_predictor)
    assert reg.get("zeta") is _stub_predictor
    assert reg.names() == ["alpha", "zeta"]
    reg.unregister("zeta")
    assert reg.get("zeta") is None
    assert reg.names() == ["alpha"]
    with pytest.raises(TypeError, match="callable"):
        reg.register("bad", "not-a-callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A2 — real-metadata entry-point discovery
# ---------------------------------------------------------------------------

def test_a2_discovers_exactly_the_five_incumbents_without_building_engines() -> None:
    """[UNIT-TEST] A2: the installed ``pii_anon.byo_pipelines`` group yields
    exactly the five in-tree incumbents (exact sorted list — the S3-01
    precedent) and discovery never constructs an engine (lazy cache empty)."""
    byo._PREDICTOR_CACHE.clear()
    reg = BYOPipelineRegistry()
    discovered = reg.discover_entrypoint_pipelines()
    assert sorted(discovered) == [
        "gliner",
        "presidio",
        "scrubadub",
        "spacy_ner",
        "stanza_ner",
    ]
    assert reg.names() == sorted(discovered)
    for name in discovered:
        assert callable(reg.get(name))
    # Discovery resolves function objects only — no engine was constructed.
    assert byo._PREDICTOR_CACHE == {}


# ---------------------------------------------------------------------------
# A3 — graceful degradation (NFR-026)
# ---------------------------------------------------------------------------

def test_a3_discovery_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """[UNIT-TEST] A3: an absent group returns []; a raising loader and a
    non-callable target are skipped while the good one registers."""
    reg = BYOPipelineRegistry()
    assert reg.discover_entrypoint_pipelines(group="pii_anon.no_such_group") == []

    class _EP:
        def __init__(self, name: str, target: Any, raises: bool = False):
            self.name = name
            self._target = target
            self._raises = raises

        def load(self) -> Any:
            if self._raises:
                raise RuntimeError("simulated broken entry point")
            return self._target

    class _EPS:
        def select(self, group: str) -> list[_EP]:
            del group
            return [
                _EP("broken", None, raises=True),
                _EP("not-callable", object()),
                _EP("good", _stub_predictor),
            ]

    monkeypatch.setattr(byo.metadata, "entry_points", lambda: _EPS())
    reg2 = BYOPipelineRegistry()
    assert reg2.discover_entrypoint_pipelines() == ["good"]
    assert reg2.get("good") is _stub_predictor
    assert reg2.get("broken") is None
    assert reg2.get("not-callable") is None


# ---------------------------------------------------------------------------
# A4 — EngineAdapter -> Predictor bridge
# ---------------------------------------------------------------------------

def test_a4_engine_predictor_maps_findings_and_drops_malformed_spans() -> None:
    """[UNIT-TEST] A4: a good finding maps to its exact tuple; None-span and
    bool-span (bool-is-int trap) findings are dropped."""
    predictor = engine_predictor(_StubEngine())
    assert predictor("mail zz-alias@synthetic.test today") == [
        ("EMAIL_ADDRESS", 5, 28)
    ]


# ---------------------------------------------------------------------------
# A5 — identical-path delegation (contract)
# ---------------------------------------------------------------------------

def test_a5_evaluate_incumbent_delegates_to_the_literal_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[CONTRACT-TEST] A5: the byo module re-exports the literal
    ``evaluate_external_system`` function object, and ``evaluate_incumbent``
    makes exactly one delegating call with no scoring logic of its own."""
    assert byo.evaluate_external_system is ext.evaluate_external_system

    calls: list[dict[str, Any]] = []

    def _spy(predictor: Any, **kwargs: Any) -> str:
        calls.append({"predictor": predictor, **kwargs})
        return "sentinel-result"

    monkeypatch.setattr(byo, "evaluate_external_system", _spy)
    result = evaluate_incumbent("presidio", max_records=5)
    assert result == "sentinel-result"
    assert len(calls) == 1
    assert calls[0]["system_name"] == "presidio"
    assert callable(calls[0]["predictor"])
    assert calls[0]["max_records"] == 5


# ---------------------------------------------------------------------------
# A6 — identical-scoring proof (contract)
# ---------------------------------------------------------------------------

def test_a6_incumbent_and_byo_paths_score_exactly_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[CONTRACT-TEST] A6: ONE stub predictor scored (a) directly as a BYO
    system and (b) through the incumbent path produces EXACTLY equal
    deterministic scoring fields — the FR-002 identity, by construction."""
    monkeypatch.setattr(ext, "load_eval_dataset", _mini_dataset)
    monkeypatch.setitem(
        byo._INCUMBENT_FACTORIES, "fake-incumbent", lambda: _stub_predictor
    )
    byo._PREDICTOR_CACHE.pop("fake-incumbent", None)

    as_byo = evaluate_external_system(
        _stub_predictor,
        system_name="fake-incumbent",
        warmup_records=0,
        composite_config=_ZERO_LATENCY_CFG,
    )
    as_incumbent = evaluate_incumbent(
        "fake-incumbent",
        warmup_records=0,
        composite_config=_ZERO_LATENCY_CFG,
    )

    assert as_incumbent.scorecard.f1 == as_byo.scorecard.f1
    assert as_incumbent.scorecard.precision == as_byo.scorecard.precision
    assert as_incumbent.scorecard.recall == as_byo.scorecard.recall
    assert as_incumbent.per_record_f1 == as_byo.per_record_f1
    assert as_incumbent.records_evaluated == as_byo.records_evaluated
    assert as_incumbent.composite.score == as_byo.composite.score

    byo._PREDICTOR_CACHE.pop("fake-incumbent", None)


# ---------------------------------------------------------------------------
# A7 — zero-edit FR-001 proof
# ---------------------------------------------------------------------------

def test_a7_third_party_predictor_ranks_with_zero_harness_edits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[UNIT-TEST] A7: an in-test third-party predictor is evaluated and
    spliced into a baseline leaderboard using ONLY public eval_framework
    imports — no harness module is modified."""
    monkeypatch.setattr(ext, "load_eval_dataset", _mini_dataset)
    artifact = tmp_path / "baseline.json"
    artifact.write_text(
        json.dumps(
            {
                "dataset": "synthetic-mini",
                "systems": [
                    {
                        "system": "baseline-stub",
                        "f1": 0.5,
                        "precision": 0.5,
                        "recall": 0.5,
                        "latency_p50_ms": 10.0,
                        "docs_per_hour": 1000.0,
                        "composite_score": 0.4,
                        "samples": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_external_system(
        _stub_predictor,
        system_name="third-party-detector",
        warmup_records=0,
        composite_config=_ZERO_LATENCY_CFG,
    )
    board = load_baseline_leaderboard(artifact_path=artifact).with_scorecard(
        result.scorecard
    )
    names = [sc.system_name for sc in board.systems]
    assert "third-party-detector" in names
    assert "baseline-stub" in names


# ---------------------------------------------------------------------------
# A8 — exact-score anchor
# ---------------------------------------------------------------------------

def test_a8_exact_score_anchor_on_hand_counted_mini_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[UNIT-TEST] A8: per-record F1s and aggregate precision/recall/F1 are
    anchored at the exact hand-computed reference values (never bounds)."""
    monkeypatch.setattr(ext, "load_eval_dataset", _mini_dataset)
    result = evaluate_external_system(
        _stub_predictor,
        system_name="anchor-stub",
        warmup_records=0,
        composite_config=_ZERO_LATENCY_CFG,
    )
    assert result.records_evaluated == 3
    assert result.skipped_records == 0
    assert result.per_record_f1 == [1.0, 0.0, pytest.approx(2 / 3)]
    assert result.scorecard.precision == 1.0
    assert result.scorecard.recall == 0.5
    assert result.scorecard.f1 == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# A9 — determinism (AX-002)
# ---------------------------------------------------------------------------

def test_a9_deterministic_scoring_fields_across_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[UNIT-TEST] A9: two back-to-back runs produce identical deterministic
    scoring fields and an identical zero-latency-weight composite; the
    identical-path leaderboard ranking replays identically."""
    monkeypatch.setattr(ext, "load_eval_dataset", _mini_dataset)
    monkeypatch.setitem(
        byo._INCUMBENT_FACTORIES, "fake-incumbent", lambda: _stub_predictor
    )
    byo._PREDICTOR_CACHE.pop("fake-incumbent", None)

    runs = [
        evaluate_incumbent(
            "fake-incumbent",
            warmup_records=0,
            composite_config=_ZERO_LATENCY_CFG,
        )
        for _ in range(2)
    ]
    assert runs[0].per_record_f1 == runs[1].per_record_f1
    assert runs[0].scorecard.f1 == runs[1].scorecard.f1
    assert runs[0].scorecard.precision == runs[1].scorecard.precision
    assert runs[0].scorecard.recall == runs[1].scorecard.recall
    assert runs[0].composite.score == runs[1].composite.score

    systems = {"good-stub": _stub_predictor, "bad-stub": _empty_predictor}
    boards = [
        build_identical_path_leaderboard(
            systems,
            warmup_records=0,
            composite_config=_ZERO_LATENCY_CFG,
        )[0]
        for _ in range(2)
    ]
    assert [sc.system_name for sc in boards[0].systems] == [
        sc.system_name for sc in boards[1].systems
    ]

    byo._PREDICTOR_CACHE.pop("fake-incumbent", None)


# ---------------------------------------------------------------------------
# A10 — fail-closed + signature-drift pin
# ---------------------------------------------------------------------------

def test_a10_unknown_incumbent_fails_closed_and_signature_pins() -> None:
    """[UNIT-TEST] A10: an unknown incumbent name is refused with a
    domain-named ValueError; ``evaluate_incumbent``'s passthrough keywords
    stay a subset of ``evaluate_external_system``'s (drift pin)."""
    with pytest.raises(ValueError, match="incumbent"):
        incumbent_predictor("no-such-engine")

    incumbent_params = set(inspect.signature(evaluate_incumbent).parameters)
    external_params = set(inspect.signature(evaluate_external_system).parameters)
    assert (incumbent_params - {"name"}) <= external_params


# ---------------------------------------------------------------------------
# A11 — always-available incumbent smoke
# ---------------------------------------------------------------------------

def test_a11_regex_engine_bridge_smoke_and_lazy_incumbents() -> None:
    """[UNIT-TEST] A11: the stdlib-backed regex engine bridges to an exact
    span with no native dependency; all five incumbent predictors resolve to
    callables WITHOUT invoking any engine (no model load)."""
    predictor = engine_predictor(RegexEngineAdapter(enabled=True))
    text = "mail zz-alias@synthetic.test today"
    spans = predictor(text)
    assert ("EMAIL_ADDRESS", 5, 28) in spans
    assert text[5:28] == "zz-alias@synthetic.test"

    assert INCUMBENT_SYSTEMS == (
        "gliner",
        "presidio",
        "scrubadub",
        "spacy_ner",
        "stanza_ner",
    )
    for name in INCUMBENT_SYSTEMS:
        assert callable(incumbent_predictor(name))


# ---------------------------------------------------------------------------
# A12 — identical-path leaderboard
# ---------------------------------------------------------------------------

def test_a12_identical_path_leaderboard_ranks_hand_computed_winner_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[UNIT-TEST] A12: two stub systems scored on the identical path produce
    a board with exactly those names and the hand-computed winner first."""
    monkeypatch.setattr(ext, "load_eval_dataset", _mini_dataset)
    board, results = build_identical_path_leaderboard(
        {"good-stub": _stub_predictor, "bad-stub": _empty_predictor},
        warmup_records=0,
        composite_config=_ZERO_LATENCY_CFG,
    )
    names = [sc.system_name for sc in board.systems]
    assert sorted(names) == ["bad-stub", "good-stub"]
    assert names[0] == "good-stub"
    assert set(results) == {"bad-stub", "good-stub"}
    assert results["good-stub"].scorecard.f1 == pytest.approx(2 / 3)
    assert results["bad-stub"].scorecard.f1 == 0.0


# ---------------------------------------------------------------------------
# A13 — import-boundary audit
# ---------------------------------------------------------------------------

def test_a13_module_imports_nothing_forbidden() -> None:
    """[AUDIT] A13: byo_pipeline.py imports nothing from swarm/moe/fusion/
    orchestrator (a standalone eval-framework module)."""
    import ast

    src = Path(byo.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {"swarm", "moe", "fusion", "orchestrator"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")
            assert not (
                len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden
            ), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")
                assert not (
                    len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden
                ), alias.name
