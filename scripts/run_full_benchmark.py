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


def run_step(description: str, cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess step with logging."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    print(f"  Command: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False, text=True)
    if check and result.returncode != 0:
        print(f"\nERROR: Step failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


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

    # ── Step 1: Preflight check ───────────────────────────────────────
    if not args.skip_preflight:
        run_step(
            "Step 1/6: Benchmark preflight check",
            [python, str(SCRIPTS_DIR / "check_benchmark_runtime.py"),
             "--output-json", str(ARTIFACTS_DIR / "runtime-preflight.json")],
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

    run_step("Step 2/6: Running competitor benchmark", bench_cmd)

    # ── Step 3: Render benchmark summary ──────────────────────────────
    run_step(
        "Step 3/6: Rendering benchmark summary",
        [python, str(SCRIPTS_DIR / "render_benchmark_summary.py"),
         "--input-json", str(ARTIFACTS_DIR / "benchmark-results.json"),
         "--output-markdown", "docs/benchmark-summary.md"],
    )

    # ── Step 4: Update README with benchmark results ──────────────────
    print(f"\n{'=' * 60}")
    print("  Step 4/6: Updating README benchmark section")
    print(f"{'=' * 60}")

    readme_text = README_PATH.read_text(encoding="utf-8")
    summary_path = ROOT_DIR / "docs" / "benchmark-summary.md"
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
        else:
            print("  WARNING: Benchmark markers not found in README. Skipping update.")
    else:
        print("  WARNING: docs/benchmark-summary.md not found. Skipping README update.")

    # ── Step 5: Render complex mode example ───────────────────────────
    complex_script = SCRIPTS_DIR / "render_complex_mode_example.py"
    if complex_script.exists():
        run_step(
            "Step 5/6: Rendering complex mode example",
            [python, str(complex_script)],
            check=False,
        )
    else:
        print("\n  Skipping complex mode example (script not found)")

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
        )
    else:
        print("\n  Skipping README sync check (script not found)")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    results_path = ARTIFACTS_DIR / "benchmark-results.json"

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
