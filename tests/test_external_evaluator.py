"""Tests for the external-system evaluation API.

Covers the public workflow documented in docs/evaluate-your-pipeline.md:
user supplies a predict callable, :func:`evaluate_external_system` scores
it against the benchmark, and :func:`load_baseline_leaderboard` splices
the result into a leaderboard that ranks it alongside the checked-in
baselines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pii_anon.eval_framework import (
    BaselineLeaderboard,
    CompositeConfig,
    ExternalEvaluationResult,
    Leaderboard,
    SystemScorecard,
    evaluate_external_system,
    load_baseline_leaderboard,
    resolve_predictor_path,
)
from pii_anon.eval_framework.datasets.schema import resolve_eval_dataset_path


_DATASET_AVAILABLE = resolve_eval_dataset_path() is not None
requires_dataset = pytest.mark.skipif(
    not _DATASET_AVAILABLE,
    reason="pii-anon-datasets not installed",
)


# ---------------------------------------------------------------------------
# Helper predictors
# ---------------------------------------------------------------------------

def _empty_predictor(_text: str):
    return []


def _copy_label_predictor(records):
    """Return a predictor that perfectly reproduces the labels on a known slice."""
    labels_by_text = {
        rec.text: [(lbl["entity_type"], lbl["start"], lbl["end"]) for lbl in rec.labels]
        for rec in records
    }

    def predict(text: str):
        return labels_by_text.get(text, [])

    return predict


def _raising_predictor(_text: str):
    raise RuntimeError("simulated detector failure")


# ---------------------------------------------------------------------------
# resolve_predictor_path
# ---------------------------------------------------------------------------

def test_resolve_predictor_path_happy():
    """A ``module:callable`` path resolves to the callable."""
    resolved = resolve_predictor_path("math:floor")
    assert callable(resolved)
    assert resolved(3.7) == 3


def test_resolve_predictor_path_missing_colon():
    with pytest.raises(ValueError, match="module.submod:callable"):
        resolve_predictor_path("not_a_valid_path")


def test_resolve_predictor_path_non_callable():
    with pytest.raises(TypeError, match="non-callable"):
        resolve_predictor_path("math:pi")


def test_resolve_predictor_path_unknown_attr():
    with pytest.raises(AttributeError):
        resolve_predictor_path("math:definitely_not_a_function")


# ---------------------------------------------------------------------------
# evaluate_external_system
# ---------------------------------------------------------------------------

@requires_dataset
def test_evaluate_external_system_empty_predictor_is_rank_zero():
    """A predictor that returns nothing produces F1=0 but still completes."""
    result = evaluate_external_system(
        _empty_predictor,
        system_name="empty",
        max_records=20,
        language="en",
        warmup_records=2,
    )
    assert isinstance(result, ExternalEvaluationResult)
    assert result.scorecard.system_name == "empty"
    assert result.scorecard.f1 == 0.0
    assert result.scorecard.precision == 0.0
    assert result.scorecard.recall == 0.0
    # Composite still has a value in (0, 1) — zero predictions = free
    # latency, so efficiency sub-score dominates.
    assert 0.0 <= result.scorecard.composite_score <= 1.0
    assert result.records_evaluated == 20
    assert result.skipped_records == 0
    # Latency samples exclude the 2 warmup records (see
    # test_warmup_records_excluded_from_latency_samples).
    assert len(result.latency_ms_samples) == 18


@requires_dataset
def test_evaluate_external_system_perfect_predictor_hits_high_f1():
    """Feeding back the gold labels yields F1=1.0."""
    from pii_anon.eval_framework.datasets.schema import load_eval_dataset

    records = load_eval_dataset(language="en")[:15]
    result = evaluate_external_system(
        _copy_label_predictor(records),
        system_name="oracle",
        max_records=15,
        language="en",
        warmup_records=2,
    )
    # A few records may have dropped labels from span normalization;
    # the oracle should still be near-perfect.
    assert result.scorecard.f1 >= 0.95
    assert result.scorecard.precision >= 0.95
    assert result.scorecard.recall >= 0.95


@requires_dataset
def test_evaluate_external_system_skips_partial_errors():
    """A predictor that fails on half the records keeps going and records both."""
    from pii_anon.eval_framework.datasets.schema import load_eval_dataset

    records = load_eval_dataset(language="en")[:20]
    good_predictor = _copy_label_predictor(records)
    call_count = {"n": 0}

    def flaky(text: str):
        call_count["n"] += 1
        if call_count["n"] % 3 == 0:
            raise ValueError("every third record bombs")
        return good_predictor(text)

    result = evaluate_external_system(
        flaky,
        system_name="flaky",
        max_records=18,
        language="en",
        warmup_records=0,
        on_error="skip",
    )
    assert result.skipped_records > 0
    assert result.records_evaluated > 0
    assert result.records_evaluated + result.skipped_records == 18
    # We should have captured the first few error strings for diagnosis.
    assert len(result.errors) > 0
    assert "every third record bombs" in result.errors[0]


@requires_dataset
def test_evaluate_external_system_raises_when_all_records_fail():
    """If every record fails, evaluator raises to avoid silently returning zeros."""
    with pytest.raises(RuntimeError, match="zero successful"):
        evaluate_external_system(
            _raising_predictor,
            system_name="broken",
            max_records=5,
            language="en",
            warmup_records=0,
            on_error="skip",
        )


@requires_dataset
def test_evaluate_external_system_reraises_on_error_raise():
    """``on_error='raise'`` propagates the first failure."""
    with pytest.raises(RuntimeError, match="simulated detector failure"):
        evaluate_external_system(
            _raising_predictor,
            system_name="broken",
            max_records=3,
            language="en",
            warmup_records=0,
            on_error="raise",
        )


@requires_dataset
def test_evaluate_external_system_rejects_dual_config():
    """Passing both ``deployment_profile`` and ``composite_config`` is a user error."""
    with pytest.raises(ValueError, match="not both"):
        evaluate_external_system(
            _empty_predictor,
            system_name="x",
            max_records=5,
            language="en",
            deployment_profile="standard",
            composite_config=CompositeConfig(),
        )


@requires_dataset
def test_evaluate_external_system_honors_deployment_profile():
    """``deployment_profile='high_security'`` changes the composite weights."""
    result_standard = evaluate_external_system(
        _empty_predictor, system_name="a", max_records=10, language="en",
        deployment_profile="standard",
    )
    result_secure = evaluate_external_system(
        _empty_predictor, system_name="b", max_records=10, language="en",
        deployment_profile="high_security",
    )
    # Config objects should differ; composite scores may too, since the
    # weights differ even when the inputs are identical.
    assert result_standard.composite.config.deployment_profile == "standard"
    assert result_secure.composite.config.deployment_profile == "high_security"


@requires_dataset
def test_evaluate_external_system_rejects_bad_on_error():
    with pytest.raises(ValueError, match="on_error"):
        evaluate_external_system(
            _empty_predictor,
            system_name="x",
            max_records=5,
            language="en",
            on_error="explode",
        )


@requires_dataset
def test_warmup_records_excluded_from_latency_samples():
    """Warmup predictions contribute to F1 but not to the latency distribution.

    Regression test: an earlier revision folded warmup timings into the
    measured distribution, so stateful predictors whose first call was
    abnormally slow got penalised twice (once as warmup, once in p50).
    """
    result = evaluate_external_system(
        _empty_predictor,
        system_name="x",
        max_records=20,
        language="en",
        warmup_records=5,
    )
    # 20 evaluated records, 5 were warmup → only 15 latency samples.
    assert result.records_evaluated == 20
    assert len(result.latency_ms_samples) == 15


@requires_dataset
def test_external_evaluation_result_to_dict_is_json_safe():
    """``.to_dict()`` must round-trip through ``json.dumps``."""
    result = evaluate_external_system(
        _empty_predictor, system_name="x", max_records=10, language="en",
    )
    payload = result.to_dict()
    # Must not raise.
    encoded = json.dumps(payload)
    assert "scorecard" in json.loads(encoded)


# ---------------------------------------------------------------------------
# load_baseline_leaderboard
# ---------------------------------------------------------------------------

def test_load_baseline_leaderboard_reads_committed_artifact():
    """The checked-in benchmark artifact produces a non-empty baseline."""
    baseline = load_baseline_leaderboard()
    assert isinstance(baseline, BaselineLeaderboard)
    names = baseline.system_names()
    # We expect pii-anon and pii-anon-swarm to always be present — they
    # ship with the library.
    assert "pii-anon" in names
    assert "pii-anon-swarm" in names


def test_load_baseline_leaderboard_missing_artifact():
    with pytest.raises(FileNotFoundError, match="benchmark artifact not found"):
        load_baseline_leaderboard(Path("/nonexistent/path.json"))


def test_baseline_with_scorecard_adds_user_system():
    """User scorecard lands in the leaderboard with a real Elo rating."""
    baseline = load_baseline_leaderboard()
    my_card = SystemScorecard(
        system_name="my-detector",
        f1=0.72, precision=0.70, recall=0.74,
        latency_p50_ms=3.0, docs_per_hour=1_200_000,
        composite_score=0.68,
    )
    merged = baseline.with_scorecard(my_card)
    assert isinstance(merged, Leaderboard)
    found = next((s for s in merged.systems if s.system_name == "my-detector"), None)
    assert found is not None
    # Elo must move from the 1500 default after the tournament runs.
    assert found.elo_rating != 1500.0
    # RD must shrink from the untrained 350 baseline.
    assert found.elo_rd < 350.0


def test_baseline_with_scorecard_rejects_duplicate_name_when_replace_false():
    """Explicit ``replace=False`` raises on a baseline-name collision."""
    baseline = load_baseline_leaderboard()
    clash = SystemScorecard(system_name="pii-anon", composite_score=0.99)
    with pytest.raises(ValueError, match="already contains"):
        baseline.with_scorecard(clash, replace=False)


def test_baseline_with_scorecard_default_replaces():
    """The default behavior silently replaces a duplicate name.

    Repeated iterative runs with the same ``system_name`` should not
    require the caller to thread ``replace=True`` through.
    """
    baseline = load_baseline_leaderboard()
    replacement = SystemScorecard(
        system_name="pii-anon", composite_score=0.99, f1=0.99,
    )
    merged = baseline.with_scorecard(replacement)
    found = next(s for s in merged.systems if s.system_name == "pii-anon")
    assert found.composite_score == 0.99
    assert sum(1 for s in merged.systems if s.system_name == "pii-anon") == 1


# ---------------------------------------------------------------------------
# Leaderboard.from_benchmark_scorecard
# ---------------------------------------------------------------------------

def test_from_benchmark_scorecard_runs_tournament_when_no_engine_provided():
    from pii_anon.eval_framework import BenchmarkScorecard

    bench = BenchmarkScorecard(benchmark_name="test", dataset_name="pii_anon")
    bench.add_system(SystemScorecard(system_name="A", composite_score=0.8, f1=0.8))
    bench.add_system(SystemScorecard(system_name="B", composite_score=0.5, f1=0.5))
    bench.add_system(SystemScorecard(system_name="C", composite_score=0.65, f1=0.65))

    board = Leaderboard.from_benchmark_scorecard(bench)
    # Sorted by Elo desc — A has the highest composite so it should win
    # the round-robin and lead the table.
    assert board.systems[0].system_name == "A"
    assert board.systems[-1].system_name == "B"
    # Every system has a non-default Elo rating now.
    for sc in board.systems:
        assert sc.elo_rating != 1500.0


def test_from_benchmark_scorecard_invalid_sort_by():
    from pii_anon.eval_framework import BenchmarkScorecard

    bench = BenchmarkScorecard(benchmark_name="test", dataset_name="pii_anon")
    bench.add_system(SystemScorecard(system_name="A", composite_score=0.5))
    with pytest.raises(ValueError, match="sort_by"):
        Leaderboard.from_benchmark_scorecard(bench, sort_by="garbage")


def test_from_benchmark_scorecard_does_not_mutate_source():
    """Building a leaderboard from a BenchmarkScorecard must not mutate it.

    Regression test for a subtle bug where a second call would start
    the Elo tournament from previously-written ratings rather than the
    untrained 1500/350 defaults, producing non-deterministic Elo output.
    """
    from pii_anon.eval_framework import BenchmarkScorecard

    bench = BenchmarkScorecard(benchmark_name="stable", dataset_name="pii_anon")
    bench.add_system(SystemScorecard(system_name="A", composite_score=0.8, f1=0.8))
    bench.add_system(SystemScorecard(system_name="B", composite_score=0.5, f1=0.5))

    board_first = Leaderboard.from_benchmark_scorecard(bench)
    ratings_first = {sc.system_name: sc.elo_rating for sc in board_first.systems}

    # The source scorecards must still hold their pristine Elo defaults.
    for sc in bench.system_scorecards.values():
        assert sc.elo_rating == 1500.0, (
            "from_benchmark_scorecard mutated its input — second calls "
            "will build on stale Elo state"
        )
        assert sc.elo_rd == 350.0

    # And a second call from the same source must produce the same
    # tournament results, not drifted ones.
    board_second = Leaderboard.from_benchmark_scorecard(bench)
    ratings_second = {sc.system_name: sc.elo_rating for sc in board_second.systems}
    assert ratings_first == ratings_second
