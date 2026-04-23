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
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "benchmarks"
README_PATH = ROOT_DIR / "README.md"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _progress import ProgressTracker, format_elapsed  # noqa: E402

# How many suite steps contribute to the step-level progress bar.
_TOTAL_STEPS = 6


def run_step(
    description: str,
    cmd: list[str],
    *,
    check: bool = True,
    tracker: ProgressTracker | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess step with logging.

    If ``tracker`` is provided, the step's start is painted on the
    single-line suite bar, the subprocess takes over the terminal for
    its own in-place bar (if any), and on completion the step outcome is
    appended to the tracker's phase log.
    """
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  Command: {' '.join(cmd)}")
    print()
    if tracker is not None:
        tracker.set_phase(description, phase_total=1)
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False, text=True)
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

    run_step("Step 2/6: Running competitor benchmark", bench_cmd, tracker=tracker)

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
