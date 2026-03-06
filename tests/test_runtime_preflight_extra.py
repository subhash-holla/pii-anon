from __future__ import annotations

import subprocess

import pytest

from pii_anon.evaluation import runtime_preflight as rp


def test_run_python_probe_timeout_and_execution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python", "-c", "x"], timeout=1)

    monkeypatch.setattr(rp.subprocess, "run", _timeout)
    ok, code, output = rp._run_python_probe(python_executable="python", code="print('x')")
    assert ok is False
    assert code == 124
    assert "timed out" in output

    def _explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rp.subprocess, "run", _explode)
    ok, code, output = rp._run_python_probe(python_executable="python", code="print('x')")
    assert ok is False
    assert code == 125
    assert "probe failed to execute" in output


def test_run_python_probe_success_combines_stdout_and_stderr() -> None:
    ok, code, output = rp._run_python_probe(
        python_executable=rp.sys.executable,
        code="import sys; print('hello'); print('warn', file=sys.stderr)",
    )
    assert ok is True
    assert code == 0
    assert "hello" in output
    assert "warn" in output


def test_check_shared_memory_failure_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force Linux code-path so /dev/shm checks are exercised
    monkeypatch.setattr(rp.platform, "system", lambda: "Linux")

    monkeypatch.setattr(rp.os.path, "exists", lambda _path: False)
    ok, path, reason = rp._check_shared_memory()
    assert ok is False
    assert path == "/dev/shm"
    assert "missing" in reason

    monkeypatch.setattr(rp.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(rp.os.path, "isdir", lambda _path: False)
    ok, _path, reason = rp._check_shared_memory()
    assert ok is False
    assert "not a directory" in reason

    monkeypatch.setattr(rp.os.path, "isdir", lambda _path: True)
    monkeypatch.setattr(rp.os, "access", lambda *_args, **_kwargs: False)
    ok, _path, reason = rp._check_shared_memory()
    assert ok is False
    assert "not writable" in reason


def test_check_shared_memory_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force Linux code-path so /dev/shm checks are exercised
    monkeypatch.setattr(rp.platform, "system", lambda: "Linux")

    monkeypatch.setattr(rp.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(rp.os.path, "isdir", lambda _path: True)
    monkeypatch.setattr(rp.os, "access", lambda *_args, **_kwargs: True)
    ok, path, reason = rp._check_shared_memory()
    assert ok is True
    assert path == "/dev/shm"
    assert reason == "ok"


def test_runtime_preflight_reports_model_hint_for_native_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rp.platform, "release", lambda: "6.8")
    monkeypatch.setattr(rp.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(rp, "_check_shared_memory", lambda: (True, "/dev/shm", "ok"))

    def _probe(*, python_executable: str, code: str, timeout_sec: int = 60):
        _ = python_executable, timeout_sec
        if "gliner-pii-base" in code:
            return False, 2, "GLiNER model not found"
        return True, 0, ""

    monkeypatch.setattr(rp, "_run_python_probe", _probe)

    report = rp.run_benchmark_runtime_preflight(
        strict_runtime=False,
        require_all_competitors=True,
        require_native_competitors=True,
    )
    assert report["ready"] is False
    assert report["all_competitors_available"] is False
    assert "gliner" in report["unavailable_competitors"]
    assert "Install GLiNER" in report["competitors"]["gliner"]["reason"]
