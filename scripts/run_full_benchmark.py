#!/usr/bin/env python3
"""Run the complete pii-anon benchmark suite and update all documentation.

End-to-end script that:
1. Verifies all competitor engines are available
2. Runs the full competitor benchmark on pii-anon-eval-data
3. Renders the benchmark summary markdown
4. Updates the README benchmark section (between markers)
5. Renders the complex mode pseudonymization example
6. Validates README stays in sync with benchmark data

Usage:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --max-samples 5000
    python scripts/run_full_benchmark.py --skip-preflight
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "benchmarks"
README_PATH = ROOT_DIR / "README.md"
# Cross-process progress snapshot.  The child (competitor benchmark)
# writes its ProgressTracker state here atomically; the parent's
# heartbeat thread reads the file to emit meaningful progress updates.
# Hidden-dotfile name so it doesn't land in committed artifact listings.
STATE_FILE = ARTIFACTS_DIR / ".progress.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _progress import ProgressTracker, format_elapsed  # noqa: E402

# How many suite steps contribute to the step-level progress bar.
_TOTAL_STEPS = 6

# Default heartbeat interval for long-running steps.  Five minutes gives
# a usable "still alive" signal without flooding the log.  Override via
# ``--heartbeat-interval`` on the CLI.
_DEFAULT_HEARTBEAT_SECONDS = 300


def _format_eta_seconds(seconds: float) -> str:
    """Convert an ETA-in-seconds to a human string.

    Short windows use ``{X}m``; longer windows use ``{X}h{Y}m``.  The
    existing ``format_elapsed`` helper already does this for the
    elapsed case; we re-use it so the two fields render consistently.
    """
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    return format_elapsed(seconds)


def _read_progress_state(state_file: str) -> dict[str, float | int | str] | None:
    """Read the child's progress snapshot.  Returns None on any failure.

    The child writes atomically so we never see half-written JSON — but
    we might still hit a race where the file doesn't exist yet (child
    hasn't emitted its first render), or a disk/encoding glitch.  All
    failures return None; the heartbeat loop falls back to a less
    informative message rather than crashing.
    """
    try:
        with open(state_file, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _heartbeat_loop(
    stop_event: threading.Event,
    description: str,
    interval_s: float,
    start_time: float,
    state_file: str | None = None,
) -> None:
    """Emit a meaningful progress heartbeat every *interval_s*.

    When *state_file* is set and the child has written a valid
    snapshot, emits a multi-field status:

        [heartbeat] Step 2/6 — 1,234,567 / 18,777,444 records (6.5723%)
                    rate 4,321 u/s — ETA 42m — elapsed 15m — phase "..."

    When no state file (or no child writes yet), falls back to the
    original "still running" form so the user still sees something.

    When the child has not updated the state file in more than
    ``2 × interval_s`` seconds, prints a suspected-hang notice.
    """
    while not stop_event.wait(timeout=interval_s):
        parent_elapsed = time.monotonic() - start_time
        snap = _read_progress_state(state_file) if state_file else None

        if snap is None:
            # No snapshot yet — child still initialising or file write
            # failed.  Fall back to the proof-of-life form.
            sys.stderr.write(
                f"\n  [heartbeat] {description} — "
                f"{format_elapsed(parent_elapsed)} elapsed, child has not emitted a "
                f"progress snapshot yet (engines still loading?)\n"
            )
            sys.stderr.flush()
            continue

        pct = float(snap.get("pct", 0.0))
        completed = int(snap.get("completed", 0))
        total = int(snap.get("total", 0))
        rate = float(snap.get("rate_units_per_s", 0.0))
        eta_s = float(snap.get("eta_seconds", 0.0))
        elapsed_s = float(snap.get("elapsed_seconds", 0.0))
        phase = str(snap.get("phase", ""))
        wall_s = float(snap.get("updated_at_wall_s", 0.0))

        # Staleness check — if the child hasn't written in
        # 2x the heartbeat interval, surface the concern.
        stale_threshold_s = 2 * interval_s
        stale_s = (time.time() - wall_s) if wall_s > 0 else 0.0
        stale_marker = ""
        if stale_s > stale_threshold_s:
            stale_marker = (
                f" ⚠ child has not updated state in {format_elapsed(stale_s)} — "
                "may be hung or stuck on a slow record"
            )

        # Multi-line block so the heartbeat is obviously distinct from
        # the child's own in-place progress bar.
        sys.stderr.write(
            f"\n  [heartbeat] {description}\n"
            f"              progress: {completed:,} / {total:,} "
            f"({pct:.4f}%)\n"
            f"              rate:     {rate:,.0f} units/s  "
            f"elapsed: {format_elapsed(elapsed_s)}  "
            f"ETA: {_format_eta_seconds(eta_s)}\n"
            f"              phase:    {phase}{stale_marker}\n"
        )
        sys.stderr.flush()


def run_step(
    description: str,
    cmd: list[str],
    *,
    check: bool = True,
    tracker: ProgressTracker | None = None,
    heartbeat_interval_s: float | None = None,
    state_file: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess step with logging.

    If ``tracker`` is provided, the step's start is painted on the
    single-line suite bar, the subprocess takes over the terminal for
    its own in-place bar (if any), and on completion the step outcome is
    appended to the tracker's phase log.

    If *heartbeat_interval_s* is non-None and positive, a daemon thread
    prints a progress heartbeat every *heartbeat_interval_s* seconds
    for the duration of the subprocess.  When *state_file* is also
    provided, the heartbeat reads the child's atomic JSON snapshot and
    emits real completed/total/rate/ETA numbers at 0.01% fidelity.
    Otherwise falls back to a proof-of-life "still running" message.

    The child picks up the state-file path from the
    ``PII_ANON_PROGRESS_FILE`` env var set here — any subprocess that
    uses :class:`~_progress.ProgressTracker` will write JSON snapshots
    to that path on every render without the caller having to pass
    the path explicitly.
    """
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  Command: {' '.join(cmd)}")
    print()
    if tracker is not None:
        tracker.set_phase(description, phase_total=1)
    t0 = time.monotonic()

    # Clear any stale state from a previous run so the first heartbeat
    # doesn't report last run's completed count.
    if state_file:
        try:
            Path(state_file).unlink(missing_ok=True)
        except OSError:
            pass

    # Build the environment for the child so it knows where to write
    # its progress snapshot.  Inherit the parent's environment.
    env = dict(__import__("os").environ)
    if state_file:
        env["PII_ANON_PROGRESS_FILE"] = state_file

    stop_heartbeat: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    if heartbeat_interval_s and heartbeat_interval_s > 0:
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(stop_heartbeat, description, heartbeat_interval_s, t0, state_file),
            daemon=True,
            name="benchmark-heartbeat",
        )
        heartbeat_thread.start()

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT_DIR), check=False, text=True, env=env,
        )
    finally:
        # Always stop the heartbeat, even if subprocess.run raised.
        if stop_heartbeat is not None:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                # join() with a short timeout — the thread checks the
                # event at its next wakeup, which is bounded by the
                # interval; we don't want to block exit if the caller
                # picks a very long interval.
                heartbeat_thread.join(timeout=1.0)

    elapsed = time.monotonic() - t0
    if tracker is not None:
        tracker.advance(1)
        ok = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
        tracker.finish_phase(f"{description}: {ok}, {format_elapsed(elapsed)}")
    if check and result.returncode != 0:
        print(f"\nERROR: Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


# Rough per-record latencies observed on a 2024 M-series MacBook
# native-macOS path (ms/record, measured at warm cache).  Used only
# for the up-front time estimate — real numbers come from the
# benchmark itself.  The slowest system sets the profile wall-time
# because all systems run in parallel per profile.
_EST_MS_PER_RECORD = {
    "pii-anon-core": 0.5,          # regex + checksum only
    "pii-anon-swarm-e2e": 120.0,   # NER fusion + Dawid-Skene + meta-learner
    "presidio": 15.0,
    "scrubadub": 0.3,
    "gliner": 90.0,
}
_EST_DEFAULT_DATASET_SIZE = 160_000
_EST_PROFILE_COUNT = 6
# Fixed per-profile startup cost (engine init, GLiNER model load,
# Presidio recogniser warmup).  This is why every profile sits at
# "0%" for 1-5 minutes even on tiny sample caps.
_EST_PROFILE_STARTUP_SECONDS = 120.0


def _print_volume_estimate(max_samples: int | None, warmup: int, runs: int) -> None:
    """Print a human-readable work-volume + time estimate before launch.

    The estimate models actual parallel execution: within a profile all
    systems run concurrently, so wall-time is driven by the slowest
    system (pii-anon-swarm / GLiNER) not the sum of all systems.
    Profiles run sequentially, so total wall-time is ``profile_count ×
    per_profile_wall``.

    Without this estimate the benchmark looks "hung" for hours — users
    have reported multi-hour waits with zero feedback.  Surface the
    budget here so they can Ctrl-C before committing to an over-long
    run.
    """
    per_profile_samples = max_samples if max_samples else _EST_DEFAULT_DATASET_SIZE
    # Per-system per-profile record-operations: warmup is a one-shot
    # pre-measurement pass; measured runs iterate the sample set.
    per_system_ops = warmup + per_profile_samples * runs
    # Wall-time per profile is dominated by the slowest system, since
    # systems run in parallel.
    slowest_ms = max(_EST_MS_PER_RECORD.values())
    per_profile_seconds = (
        _EST_PROFILE_STARTUP_SECONDS + per_system_ops * slowest_ms / 1000.0
    )
    total_est_seconds = _EST_PROFILE_COUNT * per_profile_seconds
    total_est_minutes = total_est_seconds / 60.0
    total_est_hours = total_est_seconds / 3600.0

    print()
    print("─" * 62)
    print("  Benchmark run — work-volume estimate")
    print("─" * 62)
    cap_str = f"capped at {max_samples:,}/profile" if max_samples \
        else f"uncapped ({_EST_DEFAULT_DATASET_SIZE:,}/profile default)"
    print(f"  Samples:       {per_profile_samples:,} per profile ({cap_str})")
    print(f"  Profiles:      {_EST_PROFILE_COUNT} (short_chat, long_document,")
    print("                 structured_form_accuracy/latency, log_lines, multilingual_mix)")
    print(f"  Passes/sample: {warmup} warmup + {runs} measured")
    print(f"  Per-profile:   ~{per_profile_seconds / 60.0:.1f} min "
          f"(dominated by slowest system: ~{slowest_ms:.0f} ms/record)")
    if total_est_hours < 1.5:
        print(f"  Total wall-time estimate: ~{total_est_minutes:.0f} min")
    else:
        print(f"  Total wall-time estimate: ~{total_est_hours:.1f} h")
    if total_est_hours > 3.0:
        print()
        print("  ⚠  Estimated runtime exceeds 3 hours.  Consider:")
        print("     - make benchmark-full BENCH_MAX_SAMPLES=5000   (~30-60 min)")
        print("     - make benchmark-full BENCH_MAX_SAMPLES=1000   (~5-10 min)")
        print("     - The run is NOT hung if Step 2's progress bar sits at 0%")
        print("       for the first 1-5 minutes — engines are still loading.")
    print("─" * 62)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark and update documentation")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit benchmark samples (for fast runs)")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the preflight check")
    parser.add_argument("--dataset", default="pii_anon_benchmark", help="Dataset name")
    parser.add_argument("--dataset-source", default="auto", help="Dataset source (auto, package-only)")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up samples per system")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per system")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=_DEFAULT_HEARTBEAT_SECONDS,
        help=(
            "Seconds between 'still running' heartbeat messages emitted "
            "during long-running steps.  0 disables heartbeats.  Default "
            f"{_DEFAULT_HEARTBEAT_SECONDS} (5 minutes) — long enough not "
            "to spam the log, short enough to confirm progress on a "
            "multi-hour run."
        ),
    )
    args = parser.parse_args()

    python = args.python
    t_start = time.time()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Up-front work volume estimate ─────────────────────────────────
    # Benchmarks regularly spawn parallel sub-processes whose per-record
    # progress is invisible to this top-level tracker (a known
    # limitation: subprocess stdout interleaves badly with the
    # overwriting progress bar, so sub-process progress lines are
    # dropped).  The surface effect is that Step 2 looks "hung" for
    # hours when it's really grinding through millions of detections.
    # Surfacing a work-volume estimate + time budget up-front turns a
    # "is it hung?" question into a "is the estimate reasonable?"
    # decision the user can make before committing to the run.
    _print_volume_estimate(
        max_samples=args.max_samples,
        warmup=args.warmup,
        runs=args.runs,
    )

    tracker = ProgressTracker(
        total_work=_TOTAL_STEPS,
        label="Benchmark suite",
        refresh_s=60.0,
    )
    tracker.start()

    # ── Step 1: Preflight check ───────────────────────────────────────
    if not args.skip_preflight:
        run_step(
            "Step 1/6: Benchmark preflight check",
            [python, str(SCRIPTS_DIR / "check_benchmark_runtime.py"),
             "--output-json", str(ARTIFACTS_DIR / "runtime-preflight.json")],
            tracker=tracker,
        )

        # Report available engines
        preflight_path = ARTIFACTS_DIR / "runtime-preflight.json"
        if preflight_path.exists():
            preflight = json.loads(preflight_path.read_text())
            available = preflight.get("available_competitors", {})
            unavailable = preflight.get("unavailable_competitors", {})
            print(f"\n  Available: {', '.join(sorted(available)) or 'none'}")
            if unavailable:
                print(f"  Unavailable: {', '.join(sorted(unavailable))}")
            if not preflight.get("ready", False):
                print("\n  WARNING: Not all competitors available. Benchmark may be incomplete.")
    else:
        print("\n  Skipping preflight check (--skip-preflight)")
        tracker.advance(1)
        tracker.finish_phase("Step 1/6: preflight skipped")

    # ── Step 2: Run competitor benchmark ──────────────────────────────
    bench_cmd = [
        python, str(SCRIPTS_DIR / "run_competitor_benchmark.py"),
        "--dataset", args.dataset,
        "--dataset-source", args.dataset_source,
        "--matrix", "src/pii_anon/benchmarks/matrix/use_case_matrix.json",
        "--output-json", str(ARTIFACTS_DIR / "benchmark-results.json"),
        "--output-csv", str(ARTIFACTS_DIR / "benchmark-raw.csv"),
        "--output-floor-report", str(ARTIFACTS_DIR / "floor-gate-report.md"),
        "--output-baseline", str(ARTIFACTS_DIR / "floor-baseline.json"),
        "--preflight-output-json", str(ARTIFACTS_DIR / "runtime-preflight.json"),
    ]
    if args.max_samples:
        bench_cmd.extend(["--max-samples", str(args.max_samples)])

    # Step 2 is the long pole (typically 90%+ of total wall-time).
    # The child emits its own single-line progress bar to stdout which
    # the terminal renders directly; we just tell the user what to
    # look for so a "still at 0%" line doesn't look like a freeze.
    print()
    print("  Step 2 is the long-running phase (runs all detection systems")
    print(f"  against the benchmark dataset in parallel, {args.warmup} warmup + "
          f"{args.runs} measured runs each).")
    if args.max_samples:
        print(f"  Sample cap active: {args.max_samples:,} records per profile.")
    else:
        print("  UNCAPPED run — full ~160K records per profile × 5 systems.")
    print("  The child's own progress bar updates below; it may sit at")
    print("  0% for 1-5 minutes while engines initialise (GLiNER model load,")
    print("  Presidio recognizer warmup).")
    print()

    # Step 2 is the only step that routinely exceeds 5 minutes, so it's
    # the only one that gets the heartbeat.  The shorter steps would
    # interleave heartbeat output with their own terse completion
    # lines for no benefit.  The child writes its ProgressTracker
    # state to ``STATE_FILE`` on every render; the heartbeat reads
    # that snapshot and emits real completed/total/rate/ETA values
    # at 0.01% fidelity.
    run_step(
        "Step 2/6: Running competitor benchmark",
        bench_cmd,
        tracker=tracker,
        heartbeat_interval_s=args.heartbeat_interval,
        state_file=str(STATE_FILE),
    )

    # ── Step 3: Render benchmark summary ──────────────────────────────
    run_step(
        "Step 3/6: Rendering benchmark summary",
        [python, str(SCRIPTS_DIR / "render_benchmark_summary.py"),
         "--input-json", str(ARTIFACTS_DIR / "benchmark-results.json"),
         "--output-markdown", "docs/benchmark-summary.md"],
        tracker=tracker,
    )

    # ── Step 4: Update README with benchmark results ──────────────────
    print(f"\n{'=' * 60}")
    print("  Step 4/6: Updating README benchmark section")
    print(f"{'=' * 60}")
    tracker.set_phase("Step 4/6: Updating README benchmark section", phase_total=1)

    step4_t0 = time.monotonic()
    readme_text = README_PATH.read_text(encoding="utf-8")
    summary_path = ROOT_DIR / "docs" / "benchmark-summary.md"
    step4_outcome = "skipped"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8")
        start_marker = "<!-- BENCHMARK_SUMMARY_START -->"
        end_marker = "<!-- BENCHMARK_SUMMARY_END -->"
        start_idx = readme_text.find(start_marker)
        end_idx = readme_text.find(end_marker)
        if start_idx >= 0 and end_idx >= 0:
            new_readme = (
                readme_text[:start_idx]
                + start_marker + "\n\n"
                + summary_text + "\n\n"
                + end_marker
                + readme_text[end_idx + len(end_marker):]
            )
            README_PATH.write_text(new_readme, encoding="utf-8")
            print("  README benchmark section updated.")
            step4_outcome = "ok"
        else:
            print("  WARNING: Benchmark markers not found in README. Skipping update.")
    else:
        print("  WARNING: docs/benchmark-summary.md not found. Skipping README update.")
    tracker.advance(1)
    tracker.finish_phase(
        f"Step 4/6: README update {step4_outcome}, "
        f"{format_elapsed(time.monotonic() - step4_t0)}"
    )

    # ── Step 4b: Render the pii-rate-elo value-prop block ─────────────
    # This is the "why composite over F1 alone" section.  Runs alongside
    # the benchmark-summary update so the two marketing surfaces stay in
    # sync after every benchmark.  Non-fatal if the markers are missing
    # from README — the renderer logs a warning and exits cleanly.
    value_script = SCRIPTS_DIR / "render_pii_rate_elo_value.py"
    if value_script.exists():
        run_step(
            "Step 4b/6: Rendering pii-rate-elo value block",
            [python, str(value_script),
             "--input-json", str(ARTIFACTS_DIR / "benchmark-results.json"),
             "--readme", "README.md",
             "--output-markdown", str(ROOT_DIR / "docs" / "pii-rate-elo-value.md")],
            check=False,
            tracker=None,   # fold under Step 4's progress counter
        )

    # ── Step 5: Render complex mode example ───────────────────────────
    complex_script = SCRIPTS_DIR / "render_complex_mode_example.py"
    if complex_script.exists():
        run_step(
            "Step 5/6: Rendering complex mode example",
            [python, str(complex_script)],
            check=False,
            tracker=tracker,
        )
    else:
        print("\n  Skipping complex mode example (script not found)")
        tracker.advance(1)
        tracker.finish_phase("Step 5/6: complex mode example skipped")

    # ── Step 6: Validate README sync ──────────────────────────────────
    check_script = SCRIPTS_DIR / "check_readme_benchmark.py"
    if check_script.exists():
        run_step(
            "Step 6/6: Validating README benchmark sync",
            [python, str(check_script),
             "--readme", "README.md",
             "--summary", "docs/benchmark-summary.md",
             "--report-json", str(ARTIFACTS_DIR / "benchmark-results.json")],
            check=False,
            tracker=tracker,
        )
    else:
        print("\n  Skipping README sync check (script not found)")
        tracker.advance(1)
        tracker.finish_phase("Step 6/6: README sync check skipped")

    tracker.finish()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results_path = ARTIFACTS_DIR / "benchmark-results.json"

    print()
    print("── Suite Log " + "─" * 47)
    for entry in tracker.phase_log:
        print(f"  {entry}")
    print("─" * 60)
    print()
    print(f"\n{'=' * 60}")
    print("  Benchmark Complete")
    print(f"{'=' * 60}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if results_path.exists():
        results = json.loads(results_path.read_text())
        systems = results.get("systems", [])
        if systems:
            print(f"\n  Results ({len(systems)} systems evaluated):")
            for sys_data in systems:
                name = sys_data.get("system", sys_data.get("system_name", "?"))
                f1 = sys_data.get("f1", 0)
                p = sys_data.get("precision", 0)
                r = sys_data.get("recall", 0)
                print(f"    {name:25s} F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

    print(f"\n  Artifacts: {ARTIFACTS_DIR}")
    print("  Results:   benchmark-results.json")
    print("  README:    Updated (benchmark section)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
