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
    # Either the meaningful-progress form (if a snapshot existed) or
    # the fallback form — both are valid heartbeat outputs.  This test
    # doesn't supply a state_file so the fallback branch fires.
    assert (
        "has not emitted" in captured.err
        or "progress:" in captured.err
        or "still running" in captured.err
    )


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


# ---------------------------------------------------------------------------
# State-file heartbeat — meaningful progress via cross-process JSON snapshot
# ---------------------------------------------------------------------------

def test_progress_tracker_writes_state_file(tmp_path):
    """``ProgressTracker`` with ``state_file=`` must write atomic JSON
    snapshots that the parent's heartbeat can read.
    """
    import sys
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent / "scripts"),
    )
    from _progress import ProgressTracker

    state_path = tmp_path / ".progress.json"
    tracker = ProgressTracker(
        total_work=18_777_444,
        label="test",
        refresh_s=0.01,
        state_file=str(state_path),
    )
    tracker.start()
    tracker.set_phase("testing", phase_total=18_777_444)
    tracker.advance(1_234_567)
    tracker._render(force=True)   # ensure a write lands

    assert state_path.exists(), "state file was not written"
    import json
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["total"] == 18_777_444
    assert payload["completed"] == 1_234_567
    # 1_234_567 / 18_777_444 * 100 ≈ 6.5744%
    assert 6.0 < payload["pct"] < 7.0
    assert "phase" in payload
    tracker.finish()


def test_progress_tracker_picks_up_env_var(tmp_path, monkeypatch):
    """When the env var ``PII_ANON_PROGRESS_FILE`` is set, a tracker
    without an explicit ``state_file`` kwarg must pick it up.
    """
    import sys
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent / "scripts"),
    )
    from _progress import ProgressTracker

    state_path = tmp_path / "env.json"
    monkeypatch.setenv("PII_ANON_PROGRESS_FILE", str(state_path))
    tracker = ProgressTracker(total_work=100, refresh_s=0.01)
    tracker.start()
    tracker.advance(50)
    tracker._render(force=True)
    assert state_path.exists()
    tracker.finish()


def test_heartbeat_emits_real_progress_numbers(mod, tmp_path, capsys):
    """When the state file exists, the heartbeat emits real
    completed/total/pct numbers, not a "still running" placeholder.
    """
    import json
    state_path = tmp_path / ".progress.json"
    # Pre-populate the state file so the heartbeat has something to read.
    state_path.write_text(
        json.dumps({
            "pct": 6.5723,
            "completed": 1_234_567,
            "total": 18_777_444,
            "rate_units_per_s": 4321.0,
            "eta_seconds": 2520.0,
            "elapsed_seconds": 900.0,
            "phase": "profile short_chat",
            "updated_at_monotonic": 0.0,
            "updated_at_wall_s": __import__("time").time(),
        }),
        encoding="utf-8",
    )

    import threading
    import time
    stop = threading.Event()
    thread = threading.Thread(
        target=mod._heartbeat_loop,
        args=(stop, "Step 2/6", 0.1, time.monotonic(), str(state_path)),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)   # let one heartbeat fire
    stop.set()
    thread.join(timeout=1.0)

    captured = capsys.readouterr()
    assert "[heartbeat]" in captured.err
    # The meaningful numbers must be in the output.
    assert "1,234,567" in captured.err
    assert "18,777,444" in captured.err
    assert "6.5723%" in captured.err
    assert "profile short_chat" in captured.err


def test_heartbeat_falls_back_when_no_snapshot(mod, tmp_path, capsys):
    """Before the child has emitted its first render, the state file
    doesn't exist — the heartbeat should say "not yet emitted" rather
    than crash or report nonsense values.
    """
    missing = str(tmp_path / "not-written-yet.json")

    import threading
    import time
    stop = threading.Event()
    thread = threading.Thread(
        target=mod._heartbeat_loop,
        args=(stop, "Step 2/6", 0.1, time.monotonic(), missing),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=1.0)

    captured = capsys.readouterr()
    assert "[heartbeat]" in captured.err
    assert "has not emitted" in captured.err or "engines still loading" in captured.err


def test_heartbeat_flags_stale_snapshot(mod, tmp_path, capsys):
    """When the snapshot is older than 2x the heartbeat interval, the
    heartbeat should flag it as a suspected hang so the user knows to
    investigate.
    """
    import json
    import time
    state_path = tmp_path / ".progress.json"
    state_path.write_text(
        json.dumps({
            "pct": 10.0, "completed": 100, "total": 1000,
            "rate_units_per_s": 0.0, "eta_seconds": 0.0,
            "elapsed_seconds": 60.0, "phase": "stuck",
            "updated_at_monotonic": 0.0,
            # Snapshot is 5 minutes old — way beyond 2x 0.1s heartbeat.
            "updated_at_wall_s": time.time() - 300.0,
        }),
        encoding="utf-8",
    )

    import threading
    stop = threading.Event()
    thread = threading.Thread(
        target=mod._heartbeat_loop,
        args=(stop, "Step 2/6", 0.1, time.monotonic(), str(state_path)),
        daemon=True,
    )
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=1.0)

    captured = capsys.readouterr()
    assert "may be hung" in captured.err or "stuck on a slow record" in captured.err
