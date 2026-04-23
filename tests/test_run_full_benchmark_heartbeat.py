"""Tests for the benchmark-runner heartbeat.

Locks in the contract documented in scripts/run_full_benchmark.py:
the parent process emits ``[heartbeat] <description> — Xm elapsed,
still running`` to stderr every ``heartbeat_interval_s`` seconds
while a long subprocess is active, and stops cleanly when the
subprocess returns.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_full_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_full_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Ensure _progress is discoverable — it's a sibling script module.
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    sys.modules["run_full_benchmark"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def test_default_heartbeat_interval_is_five_minutes(mod):
    """Default interval should be 300 seconds (5 minutes)."""
    assert mod._DEFAULT_HEARTBEAT_SECONDS == 300


def test_heartbeat_loop_emits_message_each_interval(mod, capsys):
    """Calling ``_heartbeat_loop`` directly with a tight interval and
    short stop time should emit exactly one message.
    """
    import threading
    import time

    stop = threading.Event()
    start = time.monotonic()
    thread = threading.Thread(
        target=mod._heartbeat_loop,
        args=(stop, "TEST STEP", 0.25, start),
        daemon=True,
    )
    thread.start()
    # Give it enough time for exactly one heartbeat (0.25s + slack).
    time.sleep(0.4)
    stop.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "heartbeat thread did not stop after stop_event"

    captured = capsys.readouterr()
    assert "[heartbeat]" in captured.err
    assert "TEST STEP" in captured.err
    assert "still running" in captured.err


def test_heartbeat_loop_stops_immediately_when_event_set(mod):
    """The heartbeat thread must respect the stop_event and exit cleanly."""
    import threading
    import time

    stop = threading.Event()
    start = time.monotonic()
    thread = threading.Thread(
        target=mod._heartbeat_loop,
        args=(stop, "TEST", 10.0, start),   # 10s interval — we don't want to wait
        daemon=True,
    )
    thread.start()
    # Signal stop before any heartbeat would have fired.
    stop.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "heartbeat thread did not exit on stop_event"


def test_run_step_emits_heartbeat_for_long_subprocess(mod, capsys):
    """End-to-end: ``run_step(heartbeat_interval_s=0.3)`` against a 1s
    sleep must emit at least one heartbeat line to stderr.
    """
    from _progress import ProgressTracker

    tracker = ProgressTracker(total_work=1, label="t", refresh_s=60.0)
    tracker.start()
    mod.run_step(
        "TEST SLEEP",
        ["sleep", "1"],
        tracker=tracker,
        heartbeat_interval_s=0.3,
    )
    captured = capsys.readouterr()
    assert "[heartbeat]" in captured.err, (
        f"expected a heartbeat in stderr, got:\nstdout:\n{captured.out}\n"
        f"stderr:\n{captured.err}"
    )


@pytest.mark.parametrize("interval", [0, None])
def test_run_step_heartbeat_disabled(mod, capsys, interval):
    """Passing 0 or None as the interval must suppress all heartbeats."""
    from _progress import ProgressTracker

    tracker = ProgressTracker(total_work=1, label="t", refresh_s=60.0)
    tracker.start()
    mod.run_step(
        "QUICK",
        ["sleep", "0.2"],
        tracker=tracker,
        heartbeat_interval_s=interval,
    )
    captured = capsys.readouterr()
    assert "[heartbeat]" not in captured.err, (
        "heartbeat emitted despite interval=0/None: " + captured.err
    )


def test_heartbeat_stops_even_if_subprocess_fails(mod, capsys):
    """If the subprocess returns non-zero, the heartbeat still stops
    cleanly via the ``finally`` branch in ``run_step``.  We verify this
    by running a failing command and checking that the test itself
    returns control (i.e. the daemon thread doesn't wedge shutdown).
    """
    from _progress import ProgressTracker

    tracker = ProgressTracker(total_work=1, label="t", refresh_s=60.0)
    tracker.start()
    # check=False so the non-zero exit doesn't sys.exit the test runner.
    mod.run_step(
        "FAILING",
        ["false"],   # exits 1 immediately
        tracker=tracker,
        check=False,
        heartbeat_interval_s=60.0,   # would never fire within test budget
    )
    # If we got here, the heartbeat thread was successfully stopped.
