from __future__ import annotations

from pathlib import Path

import pytest

from pii_anon import cli
from pii_anon.cli import create_app
from pii_anon.evaluation.pipeline import PipelineEvaluationReport
from pii_anon.types import StrategyComparisonResult


@pytest.fixture
def runner():
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    return CliRunner()


def test_cli_detect_and_detect_stream_commands(runner, tmp_path: Path) -> None:
    app = create_app()

    config = tmp_path / "core.json"
    config.write_text('{"logging":{"level":"DEBUG"}}', encoding="utf-8")

    detect = runner.invoke(
        app,
        ["detect", "Contact alice@example.com", "--config", str(config), "--output", "text"],
    )
    assert detect.exit_code == 0
    assert "transformed_payload" in detect.stdout

    stream_file = tmp_path / "payloads.txt"
    stream_file.write_text("alice@example.com\n123-45-6789\n", encoding="utf-8")
    stream = runner.invoke(app, ["detect-stream", str(stream_file), "--output", "json"])
    assert stream.exit_code == 0
    assert '"count": 2' in stream.stdout


def test_cli_tokenize_benchmark_and_evaluate_commands(runner, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app()

    tok = runner.invoke(app, ["tokenize", "alice@example.com", "--output", "json"])
    assert tok.exit_code == 0
    assert '"token"' in tok.stdout

    bench = runner.invoke(app, ["benchmark", "--dataset", "pii_anon_benchmark_v1", "--max-samples", "5", "--output", "json"])
    assert bench.exit_code == 0
    assert '"docs_per_hour"' in bench.stdout

    class FakeReport:
        winner = "weighted_consensus"
        results = [
            StrategyComparisonResult(
                strategy="weighted_consensus",
                span_fbeta=1.0,
                findings_count=1,
                avg_confidence=1.0,
            )
        ]

    monkeypatch.setattr(
        cli.StrategyEvaluator,
        "compare_strategies",
        lambda self, strategies, dataset="": FakeReport(),
    )
    eval_result = runner.invoke(app, ["evaluate", "--output", "json"])
    assert eval_result.exit_code == 0
    assert '"winner": "weighted_consensus"' in eval_result.stdout

    monkeypatch.setattr(
        cli,
        "evaluate_pipeline",
        lambda *args, **kwargs: PipelineEvaluationReport(
            dataset="pii_anon_benchmark_v1",
            samples=5,
            transform_mode="pseudonymize",
            precision=1.0,
            recall=1.0,
            f1=1.0,
            privacy_score=1.0,
            fairness_score=1.0,
            avg_findings_per_record=2.0,
            avg_link_audit_per_record=2.0,
        ),
    )
    pipe = runner.invoke(app, ["evaluate-pipeline", "--max-samples", "5", "--output", "json"])
    assert pipe.exit_code == 0
    assert '"dataset": "pii_anon_benchmark_v1"' in pipe.stdout

    class FakeBatch:
        records_evaluated = 10
        micro_averaged = {"f1": 1.0}
        macro_averaged = {"f1": 1.0}
        weighted_averaged = {"f1": 1.0}
        privacy_score = 0.9
        fairness_score = 0.95

    monkeypatch.setattr(cli, "load_eval_dataset", lambda *args, **kwargs: [object()] * 10)
    monkeypatch.setattr(cli.EvaluationFramework, "evaluate_batch", lambda self, records: FakeBatch())
    fw = runner.invoke(app, ["eval-framework", "--max-records", "10", "--output", "json"])
    assert fw.exit_code == 0
    assert '"records_evaluated": 10' in fw.stdout


def test_cli_main_importerror_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():
        raise ImportError("missing")

    monkeypatch.setattr(cli, "create_app", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "CLI dependencies are not installed" in str(exc.value)
