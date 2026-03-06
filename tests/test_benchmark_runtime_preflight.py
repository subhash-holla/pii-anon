from __future__ import annotations

from pii_anon.evaluation import runtime_preflight as rp


def test_runtime_preflight_passes_on_darwin_when_strict(monkeypatch) -> None:
    """macOS (Darwin) is a supported runtime — strict mode should pass."""
    monkeypatch.setattr(rp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rp.platform, "release", lambda: "23.0")
    monkeypatch.setattr(rp.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(rp, "_run_python_probe", lambda **kwargs: (True, 0, ""))

    report = rp.run_benchmark_runtime_preflight(
        strict_runtime=True,
        require_all_competitors=True,
        require_native_competitors=True,
        python_executable="/tmp/fake-python",
    )
    assert report["ready"] is True
    assert report["linux_runtime_ok"] is True
    assert report["shared_memory_ok"] is True
    assert "darwin" not in " ".join(report["failures"]).lower()


def test_runtime_preflight_passes_on_linux_when_strict(monkeypatch) -> None:
    """Linux remains a supported runtime."""
    monkeypatch.setattr(rp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rp.platform, "release", lambda: "6.8")
    monkeypatch.setattr(rp.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(rp, "_check_shared_memory", lambda: (True, "/dev/shm", "ok"))
    monkeypatch.setattr(rp, "_run_python_probe", lambda **kwargs: (True, 0, ""))

    report = rp.run_benchmark_runtime_preflight(
        strict_runtime=True,
        require_all_competitors=True,
        require_native_competitors=True,
        python_executable="/tmp/fake-python",
    )
    assert report["ready"] is True
    assert report["linux_runtime_ok"] is True


def test_runtime_preflight_fails_on_unsupported_os_when_strict(monkeypatch) -> None:
    """Unsupported platforms (e.g. Windows) should fail under strict mode."""
    monkeypatch.setattr(rp.platform, "system", lambda: "Windows")
    monkeypatch.setattr(rp.platform, "release", lambda: "10.0")
    monkeypatch.setattr(rp.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(rp, "_check_shared_memory", lambda: (False, "/dev/shm", "missing /dev/shm"))
    monkeypatch.setattr(rp, "_run_python_probe", lambda **kwargs: (True, 0, ""))

    report = rp.run_benchmark_runtime_preflight(
        strict_runtime=True,
        require_all_competitors=True,
        require_native_competitors=True,
        python_executable="/tmp/fake-python",
    )
    assert report["ready"] is False
    assert any("unsupported runtime" in msg for msg in report["failures"])


def test_runtime_preflight_reports_unavailable_competitors(monkeypatch) -> None:
    monkeypatch.setattr(rp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rp.platform, "release", lambda: "6.8")
    monkeypatch.setattr(rp.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(rp, "_check_shared_memory", lambda: (True, "/dev/shm", "ok"))

    def _probe(**kwargs):
        code = kwargs["code"]
        if "import gliner" in code:
            return False, 134, "gliner not installed"
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


def test_shared_memory_ok_on_darwin(monkeypatch) -> None:
    """macOS should report shared memory as OK without /dev/shm."""
    monkeypatch.setattr(rp.platform, "system", lambda: "Darwin")
    ok, path, reason = rp._check_shared_memory()
    assert ok is True
    assert "darwin" in path.lower() or "posix" in path.lower()
    assert "ok" in reason.lower()
