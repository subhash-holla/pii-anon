"""Tests targeting coverage gaps across low-coverage modules.

Covers: cli.py, llm_guard_adapter.py, engines/registry.py,
        tokenization/providers.py, tokenization/store.py,
        competitor_compare.py (selected paths).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# CLI coverage (targets: detect, detect-stream, process-file, tokenize,
#   evaluate, evaluate-pipeline, benchmark, eval-framework, text output mode)
# ---------------------------------------------------------------------------


class TestCLIDetect:
    def test_detect_text_output(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["detect", "hello alice@example.com", "--output", "text"],
        )
        assert result.exit_code == 0

    def test_detect_bad_transform_mode(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["detect", "hello", "--transform-mode", "invalid"],
        )
        assert result.exit_code != 0

    def test_detect_anonymize_mode(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["detect", "hello alice@example.com", "--transform-mode", "anonymize", "--output", "json"],
        )
        assert result.exit_code == 0


class TestCLIDetectStream:
    def test_detect_stream(self, tmp_path: Path) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        f = tmp_path / "input.txt"
        f.write_text("alice@example.com\n555-12-3456\n", encoding="utf-8")

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["detect-stream", str(f), "--output", "json"],
        )
        assert result.exit_code == 0
        assert "count" in result.stdout

    def test_detect_stream_bad_transform_mode(self, tmp_path: Path) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        f = tmp_path / "input.txt"
        f.write_text("test\n", encoding="utf-8")

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["detect-stream", str(f), "--transform-mode", "invalid"],
        )
        assert result.exit_code != 0


class TestCLIProcessFile:
    def test_process_file_csv(self, tmp_path: Path) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        f = tmp_path / "data.csv"
        f.write_text("text\nhello alice@example.com\n", encoding="utf-8")
        out = tmp_path / "output.csv"

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["process-file", str(f), "--output", "json", "-o", str(out)],
        )
        assert result.exit_code == 0
        assert "records_processed" in result.stdout

    def test_process_file_bad_format(self, tmp_path: Path) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        f = tmp_path / "data.csv"
        f.write_text("text\nhello\n", encoding="utf-8")

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["process-file", str(f), "--format", "xyzformat"],
        )
        assert result.exit_code != 0

    def test_process_file_bad_transform_mode(self, tmp_path: Path) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        f = tmp_path / "data.csv"
        f.write_text("text\nhello\n", encoding="utf-8")

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["process-file", str(f), "--transform-mode", "invalid"],
        )
        assert result.exit_code != 0


class TestCLITokenize:
    def test_tokenize_json(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["tokenize", "alice@example.com", "--entity-type", "EMAIL", "--output", "json"],
        )
        assert result.exit_code == 0
        assert "token" in result.stdout

    def test_tokenize_text(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["tokenize", "test", "--output", "text"],
        )
        assert result.exit_code == 0
        assert "token:" in result.stdout


class TestCLIEvaluate:
    def test_evaluate_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon import cli
        from pii_anon.cli import create_app

        class _FakeResult:
            winner = "weighted_consensus"
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            strategy = "weighted_consensus"
            latency_ms = 0.1

        class _FakeReport:
            winner = "weighted_consensus"
            results = [_FakeResult()]

        class _FakeEval:
            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def compare_strategies(self, *a: object, **kw: object) -> _FakeReport:
                return _FakeReport()

        monkeypatch.setattr(cli, "StrategyEvaluator", _FakeEval)

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["evaluate", "--strategies", "weighted_consensus", "--output", "json"],
        )
        assert result.exit_code == 0
        assert "winner" in result.stdout

    def test_evaluate_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon import cli
        from pii_anon.cli import create_app

        class _FakeEval:
            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def compare_strategies(self, *a: object, **kw: object) -> None:
                raise FileNotFoundError("dataset not found")

        monkeypatch.setattr(cli, "StrategyEvaluator", _FakeEval)

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(app, ["evaluate", "--output", "json"])
        assert result.exit_code != 0


class TestCLIEvalPipeline:
    def test_evaluate_pipeline_bad_transform(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["evaluate-pipeline", "--transform-mode", "invalid"],
        )
        assert result.exit_code != 0

    def test_evaluate_pipeline_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon import cli
        from pii_anon.cli import create_app

        monkeypatch.setattr(
            cli,
            "evaluate_pipeline",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("missing")),
        )

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["evaluate-pipeline", "--output", "json"],
        )
        assert result.exit_code != 0


class TestCLIBenchmark:
    def test_benchmark_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon import cli
        from pii_anon.cli import create_app

        monkeypatch.setattr(
            cli,
            "run_benchmark",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("missing")),
        )

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(app, ["benchmark", "--output", "json"])
        assert result.exit_code != 0


class TestCLICompareValidation:
    def test_compare_competitors_bad_objective(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["compare-competitors", "--objective", "invalid"],
        )
        assert result.exit_code != 0

    def test_compare_competitors_bad_dataset_source(self) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon.cli import create_app

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["compare-competitors", "--dataset-source", "invalid"],
        )
        assert result.exit_code != 0


class TestCLIEvalFramework:
    def test_eval_framework_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon import cli
        from pii_anon.cli import create_app

        monkeypatch.setattr(
            cli,
            "load_eval_dataset",
            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("missing")),
        )

        runner = CliRunner()
        app = create_app()
        result = runner.invoke(app, ["eval-framework", "--output", "json"])
        assert result.exit_code != 0


class TestCLIPreflightFailing:
    def test_benchmark_preflight_not_ready(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        pytest.importorskip("typer")
        from typer.testing import CliRunner
        from pii_anon import cli
        from pii_anon.cli import create_app

        monkeypatch.setattr(
            cli,
            "run_benchmark_runtime_preflight",
            lambda **kw: {"ready": False, "failures": ["test"]},
        )
        out = tmp_path / "preflight.json"
        runner = CliRunner()
        app = create_app()
        result = runner.invoke(
            app,
            ["benchmark-preflight", "--output-file", str(out), "--output", "json"],
        )
        assert result.exit_code == 1
        assert out.exists()


# ---------------------------------------------------------------------------
# LLM Guard Adapter coverage
# ---------------------------------------------------------------------------


class TestLLMGuardAdapter:
    def test_capabilities(self) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        caps = adapter.capabilities()
        assert "en" in caps.supports_languages

    def test_detect_disabled(self) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=False)
        assert adapter.detect({"text": "hello"}, {}) == []

    def test_fallback_detect_ssn_and_email(self) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        findings = adapter._fallback_detect({"text": "123-45-6789 alice@example.com"}, "en")
        types = {f.entity_type for f in findings}
        assert "US_SSN" in types
        assert "EMAIL_ADDRESS" in types

    def test_fallback_detect_non_string(self) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        findings = adapter._fallback_detect({"count": 42}, "en")
        assert findings == []

    def test_detect_falls_back_when_native_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: None)
        findings = adapter.detect({"text": "123-45-6789"}, {"language": "en"})
        assert len(findings) > 0
        assert findings[0].entity_type == "US_SSN"

    def test_probe_native_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("no subprocess")),
        )
        assert adapter._probe_native_import() is False
        # Cached
        assert adapter._probe_native_import() is False

    def test_probe_native_import_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _Proc:
            returncode = 1

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Proc())
        assert adapter._probe_native_import() is False

    def test_native_detect_tuple_scan_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> tuple[str, bool, float]:
                return ("sanitized", False, 0.5)

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "sensitive data"}, {"language": "en"})
        assert len(findings) == 1
        assert findings[0].entity_type == "SENSITIVE_PII"

    def test_native_detect_tuple_float_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> tuple[str, float]:
                return ("sanitized", 0.5)

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "sensitive data"}, {})
        assert len(findings) == 1

    def test_native_detect_bool_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> bool:
                return False

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "sensitive"}, {})
        assert len(findings) == 1

    def test_native_detect_string_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> str:
                return "REDACTED"

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "sensitive"}, {})
        assert len(findings) == 1

    def test_native_detect_exception_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> None:
                raise RuntimeError("scan error")

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "123-45-6789"}, {})
        assert any(f.entity_type == "US_SSN" for f in findings)

    def test_native_detect_non_string_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> str:
                return text

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"count": 42}, {})
        assert findings == []

    def test_native_detect_single_element_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> tuple[str]:
                return ("REDACTED",)

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "sensitive"}, {})
        assert len(findings) == 1

    def test_load_native_scanner_returns_cached(self) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        sentinel = object()
        adapter._native_scanner = sentinel
        assert adapter._load_native_scanner() is sentinel

    def test_load_native_scanner_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        adapter._native_import_probe_ok = True  # skip subprocess probe

        # Patch the import to fail
        import builtins
        original_import = builtins.__import__

        def _fail_import(name: str, *args: object, **kwargs: object) -> object:
            if "llm_guard" in name:
                raise ImportError("no llm_guard")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail_import)
        assert adapter._load_native_scanner() is None

    def test_load_native_scanner_all_candidates_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)
        adapter._native_import_probe_ok = True

        # Create a fake module with scanner classes that raise
        fake_scanners = MagicMock()
        fake_scanners.Sensitive = MagicMock(side_effect=RuntimeError("init fail"))
        fake_scanners.Anonymize = MagicMock(side_effect=RuntimeError("init fail"))
        fake_scanners.Regex = MagicMock(side_effect=RuntimeError("init fail"))

        fake_llm_guard = MagicMock()
        fake_llm_guard.input_scanners = fake_scanners

        monkeypatch.setitem(sys.modules, "llm_guard", fake_llm_guard)
        monkeypatch.setitem(sys.modules, "llm_guard.input_scanners", fake_scanners)

        assert adapter._load_native_scanner() is None

    def test_native_detect_no_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.llm_guard_adapter import LLMGuardAdapter

        adapter = LLMGuardAdapter(enabled=True)

        class _FakeScanner:
            def scan(self, text: str) -> tuple[str, bool]:
                return (text, True)  # not flagged, text unchanged

        monkeypatch.setattr(adapter, "_load_native_scanner", lambda: _FakeScanner())
        findings = adapter.detect({"text": "safe"}, {})
        assert findings == []


# ---------------------------------------------------------------------------
# EngineRegistry coverage
# ---------------------------------------------------------------------------


class TestEngineRegistryCoverage:
    def test_unregister_nonexistent(self) -> None:
        from pii_anon.engines.registry import EngineRegistry

        reg = EngineRegistry()
        reg.unregister("no-such-engine")  # should not raise

    def test_list_engines_include_disabled(self) -> None:
        from pii_anon.engines.registry import EngineRegistry
        from pii_anon.engines.base import EngineAdapter

        class _Disabled(EngineAdapter):
            adapter_id = "test-disabled"

            def detect(self, payload: Any, context: Any) -> list:
                return []

        reg = EngineRegistry()
        engine = _Disabled(enabled=False)
        reg.register(engine)
        assert len(reg.list_engines(include_disabled=False)) == 0
        assert len(reg.list_engines(include_disabled=True)) == 1

    def test_discover_bad_entrypoint_no_crash(self) -> None:
        from pii_anon.engines.registry import EngineRegistry

        reg = EngineRegistry()
        # Using a non-existent group should return empty
        discovered = reg.discover_entrypoint_engines(group="nonexistent.group.xyz")
        assert isinstance(discovered, list)

    def test_unregister_existing(self) -> None:
        from pii_anon.engines.registry import EngineRegistry
        from pii_anon.engines.base import EngineAdapter

        class _TestEngine(EngineAdapter):
            adapter_id = "test-unreg"
            _shutdown_called = False

            def detect(self, payload: Any, context: Any) -> list:
                return []

            def shutdown(self) -> None:
                self._shutdown_called = True

        reg = EngineRegistry()
        engine = _TestEngine(enabled=True)
        reg.register(engine)
        assert reg.get("test-unreg") is not None
        reg.unregister("test-unreg")
        assert reg.get("test-unreg") is None
        assert engine._shutdown_called

    def test_discover_with_non_adapter_entrypoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.registry import EngineRegistry
        from importlib import metadata

        class _FakeEP:
            name = "bad"
            group = "pii_anon.engines"

            def load(self) -> type:
                return str  # not an EngineAdapter subclass

        class _FakeEPs:
            def select(self, group: str = "") -> list:
                return [_FakeEP()]

        monkeypatch.setattr(metadata, "entry_points", _FakeEPs)
        reg = EngineRegistry()
        discovered = reg.discover_entrypoint_engines()
        assert discovered == []

    def test_discover_exception_during_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.registry import EngineRegistry
        from importlib import metadata

        class _FakeEP:
            name = "broken"
            group = "pii_anon.engines"

            def load(self) -> None:
                raise ImportError("broken")

        class _FakeEPs:
            def select(self, group: str = "") -> list:
                return [_FakeEP()]

        monkeypatch.setattr(metadata, "entry_points", _FakeEPs)
        reg = EngineRegistry()
        discovered = reg.discover_entrypoint_engines()
        assert discovered == []

    def test_discover_exception_in_entry_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pii_anon.engines.registry import EngineRegistry
        from importlib import metadata

        monkeypatch.setattr(metadata, "entry_points", lambda: (_ for _ in ()).throw(Exception("fail")))
        reg = EngineRegistry()
        discovered = reg.discover_entrypoint_engines()
        assert discovered == []

    def test_health_and_capabilities(self) -> None:
        from pii_anon.engines.registry import EngineRegistry
        from pii_anon.engines.base import EngineAdapter

        class _TestEngine(EngineAdapter):
            adapter_id = "test-health"

            def detect(self, payload: Any, context: Any) -> list:
                return []

        reg = EngineRegistry()
        reg.register(_TestEngine(enabled=True))
        health = reg.health_report()
        assert "test-health" in health
        caps = reg.capabilities_report()
        assert "test-health" in caps


# ---------------------------------------------------------------------------
# TokenStore (SQLite) coverage
# ---------------------------------------------------------------------------


class TestSQLiteTokenStoreCoverage:
    def test_put_get_with_scope(self) -> None:
        from pii_anon.tokenization.store import SQLiteTokenStore, TokenMapping

        store = SQLiteTokenStore()
        mapping = TokenMapping(scope="test", token="tok1", plaintext="hello", entity_type="NAME", version=1)
        store.put(mapping)
        result = store.get("tok1", scope="test")
        assert result is not None
        assert result.plaintext == "hello"

    def test_get_without_scope(self) -> None:
        from pii_anon.tokenization.store import SQLiteTokenStore, TokenMapping

        store = SQLiteTokenStore()
        mapping = TokenMapping(scope="s1", token="tok2", plaintext="world", entity_type="NAME", version=1)
        store.put(mapping)
        result = store.get("tok2")
        assert result is not None
        assert result.plaintext == "world"

    def test_get_missing(self) -> None:
        from pii_anon.tokenization.store import SQLiteTokenStore

        store = SQLiteTokenStore()
        assert store.get("nonexistent") is None

    def test_close(self) -> None:
        from pii_anon.tokenization.store import SQLiteTokenStore

        store = SQLiteTokenStore()
        store.close()  # should not raise


class TestInMemoryTokenStoreCoverage:
    def test_get_without_scope_fallback(self) -> None:
        from pii_anon.tokenization.store import InMemoryTokenStore, TokenMapping

        store = InMemoryTokenStore()
        mapping = TokenMapping(scope="s1", token="tok1", plaintext="hello", entity_type="NAME", version=1)
        store.put(mapping)
        # Get without scope should find by token
        result = store.get("tok1")
        assert result is not None
        assert result.plaintext == "hello"


# ---------------------------------------------------------------------------
# Tokenization providers coverage
# ---------------------------------------------------------------------------


class TestTokenizerProvidersCoverage:
    def test_aessiv_detokenize_via_store(self) -> None:
        from pii_anon.tokenization.providers import AESSIVTokenizer
        from pii_anon.tokenization.store import InMemoryTokenStore

        tokenizer = AESSIVTokenizer()
        store = InMemoryTokenStore()
        record = tokenizer.tokenize("EMAIL", "alice@example.com", "test", 1, "my-key", store=store)
        # Detokenize via store
        result = tokenizer.detokenize(record, key="my-key", store=store)
        assert result == "alice@example.com"

    def test_aessiv_detokenize_via_cipher(self) -> None:
        from pii_anon.tokenization.providers import AESSIVTokenizer

        tokenizer = AESSIVTokenizer()
        record = tokenizer.tokenize("EMAIL", "alice@example.com", "test", 1, "my-key")
        # Detokenize without store (must decrypt)
        result = tokenizer.detokenize(record, key="my-key")
        assert result == "alice@example.com"

    def test_aessiv_detokenize_invalid_token_with_mapping(self) -> None:
        from pii_anon.tokenization.providers import AESSIVTokenizer, TokenRecord

        tokenizer = AESSIVTokenizer()
        record = TokenRecord(entity_type="EMAIL", version=1, token="<EMAIL:v1:bad_token>", scope="test")
        result = tokenizer.detokenize(record, key="my-key", mapping={"<EMAIL:v1:bad_token>": "fallback"})
        assert result == "fallback"

    def test_aessiv_detokenize_invalid_token_no_mapping(self) -> None:
        from pii_anon.tokenization.providers import AESSIVTokenizer, TokenRecord

        tokenizer = AESSIVTokenizer()
        record = TokenRecord(entity_type="EMAIL", version=1, token="<EMAIL:v1:bad_token>", scope="test")
        result = tokenizer.detokenize(record, key="my-key")
        assert result is None

    def test_hmac_detokenize_no_store_with_mapping(self) -> None:
        from pii_anon.tokenization.providers import DeterministicHMACTokenizer

        tokenizer = DeterministicHMACTokenizer()
        record = tokenizer.tokenize("EMAIL", "alice@example.com", "test", 1, "my-key")
        result = tokenizer.detokenize(record, key="my-key", mapping={record.token: "mapped"})
        assert result == "mapped"

    def test_hmac_detokenize_no_store_no_mapping(self) -> None:
        from pii_anon.tokenization.providers import DeterministicHMACTokenizer

        tokenizer = DeterministicHMACTokenizer()
        record = tokenizer.tokenize("EMAIL", "alice@example.com", "test", 1, "my-key")
        result = tokenizer.detokenize(record, key="my-key")
        assert result is None


# ---------------------------------------------------------------------------
# competitor_compare coverage (selected helper paths)
# ---------------------------------------------------------------------------


class TestCompetitorCompareHelpers:
    def test_normalize_entity_type_cache(self) -> None:
        from pii_anon.evaluation.competitor_compare import _normalize_entity_type

        # First call populates cache
        result1 = _normalize_entity_type("emailfilth")
        assert result1 == "EMAIL_ADDRESS"
        # Second call hits cache
        result2 = _normalize_entity_type("emailfilth")
        assert result2 == "EMAIL_ADDRESS"

    def test_per_entity_recall_with_pred_set(self) -> None:
        from pii_anon.evaluation.competitor_compare import _per_entity_recall

        pred = [("r1", "EMAIL_ADDRESS", 0, 5)]
        labels = [("r1", "EMAIL_ADDRESS", 0, 5), ("r1", "PHONE_NUMBER", 10, 15)]
        pred_set = frozenset(pred)
        result = _per_entity_recall(pred, labels, pred_set=pred_set)
        assert result["EMAIL_ADDRESS"] == 1.0
        assert result["PHONE_NUMBER"] == 0.0

    def test_per_entity_precision_with_label_set(self) -> None:
        from pii_anon.evaluation.competitor_compare import _per_entity_precision

        pred = [("r1", "EMAIL_ADDRESS", 0, 5), ("r1", "PHONE_NUMBER", 10, 15)]
        labels = [("r1", "EMAIL_ADDRESS", 0, 5)]
        label_set = frozenset(labels)
        result = _per_entity_precision(pred, labels, label_set=label_set)
        assert result["EMAIL_ADDRESS"] == 1.0
        assert result["PHONE_NUMBER"] == 0.0

    def test_safe_div_zero_denominator(self) -> None:
        from pii_anon.evaluation.competitor_compare import _safe_div

        assert _safe_div(10, 0) == 0.0
        assert _safe_div(0, 0) == 0.0

    def test_safe_div_normal(self) -> None:
        from pii_anon.evaluation.competitor_compare import _safe_div

        assert _safe_div(1, 2) == 0.5
