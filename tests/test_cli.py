import subprocess
from pathlib import Path

import pytest

from pii_anon import cli
from pii_anon.cli import create_app


def test_cli_version_command() -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "1.0.0" in result.stdout


def test_cli_health_json() -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(app, ["health", "--output", "json"])

    assert result.exit_code == 0
    assert "engines" in result.stdout


def test_cli_capabilities_json() -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(app, ["capabilities", "--output", "json"])

    assert result.exit_code == 0
    assert "capabilities" in result.stdout


def test_cli_compare_competitors_json(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    class _Row:
        def __init__(self, system: str) -> None:
            self.system = system
            self.available = True
            self.skipped_reason = None
            self.qualification_status = "core"
            self.license_name = "Apache-2.0"
            self.license_source = "project"
            self.citation_url = "https://example.com"
            self.license_gate_passed = True
            self.license_gate_reason = None
            self.precision = 1.0
            self.recall = 1.0
            self.f1 = 1.0
            self.latency_p50_ms = 0.1
            self.docs_per_hour = 1000.0
            self.per_entity_recall = {}
            self.samples = 10
            self.dominance_pass_by_profile = {"default": True}
            self.evaluation_track = "detect_only"

    class _Profile:
        profile = "default"
        objective = "balanced"
        floor_pass = True
        qualified_competitors = 1
        mit_qualified_competitors = 1
        floor_checks = []
        systems = [_Row("pii-anon")]
        end_to_end_systems = [_Row("pii-anon")]

    class _Report:
        report_schema_version = "2026-02-19.v3"
        dataset = "pii_anon_benchmark_v1"
        dataset_source = "package-only"
        warmup_samples = 1
        measured_runs = 1
        floor_pass = True
        qualification_gate_pass = True
        mit_gate_pass = True
        expected_competitors = ["presidio", "scrubadub", "gliner"]
        available_competitors = ["presidio", "scrubadub", "gliner"]
        unavailable_competitors = {}
        all_competitors_available = True
        require_all_competitors = True
        require_native_competitors = True
        systems = [_Row("pii-anon")]
        profiles = [_Profile()]

    monkeypatch.setattr(cli, "compare_competitors", lambda **kwargs: _Report())

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(
        app,
        [
            "compare-competitors",
            "--dataset",
            "pii_anon_benchmark_v1",
            "--warmup-samples",
            "1",
            "--measured-runs",
            "1",
            "--max-samples",
            "10",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert "pii-anon" in result.stdout
    assert "systems" in result.stdout
    assert "floor_pass" in result.stdout


def test_cli_benchmark_preflight_json(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    monkeypatch.setattr(
        cli,
        "run_benchmark_runtime_preflight",
        lambda **kwargs: {
            "ready": True,
            "strict_runtime": kwargs.get("strict_runtime", False),
            "all_competitors_available": True,
        },
    )

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(app, ["benchmark-preflight", "--output", "json"])
    assert result.exit_code == 0
    assert "all_competitors_available" in result.stdout


def test_cli_benchmark_publish_suite_runs_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    script = tmp_path / "scripts" / "run_publish_grade_suite.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src" / "pii_anon" / "cli.py"))

    class _Proc:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Proc())

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(app, ["benchmark-publish-suite", "--output", "json"])
    assert result.exit_code == 0
    assert "exit_code" in result.stdout


def test_cli_benchmark_publish_suite_portable_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    script = tmp_path / "scripts" / "run_publish_grade_suite.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src" / "pii_anon" / "cli.py"))
    captured: dict[str, object] = {}

    class _Proc:
        returncode = 0

    def _fake_run(cmd: object, **kwargs: object) -> _Proc:
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(
        app,
        [
            "benchmark-publish-suite",
            "--dataset-source",
            "auto",
            "--no-strict-runtime",
            "--no-require-all-competitors",
            "--no-require-native-competitors",
            "--no-include-end-to-end",
            "--no-allow-core-native-engines",
            "--no-enforce-floors",
            "--no-enforce-publish-claims",
            "--no-validate-readme-sync",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0
    command = captured.get("cmd")
    assert isinstance(command, list)
    joined = " ".join(str(item) for item in command)
    assert "--dataset-source auto" in joined
    assert "--no-strict-runtime" in joined
    assert "--no-require-all-competitors" in joined
    assert "--no-require-native-competitors" in joined
    assert "--no-include-end-to-end" in joined
    assert "--no-allow-core-native-engines" in joined
    assert "--no-enforce-floors" in joined
    assert "--no-enforce-publish-claims" in joined
    assert "--no-validate-readme-sync" in joined


def test_cli_benchmark_publish_suite_dataset_source_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    script = tmp_path / "scripts" / "run_publish_grade_suite.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src" / "pii_anon" / "cli.py"))

    runner = CliRunner()
    app = create_app()
    result = runner.invoke(
        app,
        ["benchmark-publish-suite", "--dataset-source", "invalid-source", "--output", "json"],
    )
    assert result.exit_code != 0
