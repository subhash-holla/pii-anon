#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


class _SuiteProgress:
    """Track publish-suite pipeline steps with a compact percentage bar.

    Renders a single self-updating line on TTYs::

        [===============>              ]  50.0%  [3/6] Competitor benchmark evaluation
    """

    _BAR_WIDTH = 30

    def __init__(self, total_steps: int) -> None:
        self._step = 0
        self._total = total_steps
        self._start_time = time.monotonic()
        self._label = "starting"
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def _render(self) -> str:
        pct = (self._step / self._total * 100.0) if self._total else 0.0
        filled = int(self._BAR_WIDTH * pct / 100.0)
        bar = "=" * filled
        if filled < self._BAR_WIDTH:
            bar += ">"
            bar += " " * (self._BAR_WIDTH - filled - 1)
        elapsed = int(time.monotonic() - self._start_time)
        minutes, seconds = divmod(elapsed, 60)
        return (
            f"[{bar}] {pct:5.1f}%  "
            f"{minutes:02d}:{seconds:02d}  "
            f"[{self._step}/{self._total}] {self._label}"
        )

    def step(self, label: str) -> None:
        self._step += 1
        self._label = label
        if self._is_tty:
            line = self._render()
            sys.stderr.write(f"\r{line:<120}\r")
            sys.stderr.flush()
        else:
            print(f"[suite {self._step}/{self._total}] {label}", flush=True)

    def close(self) -> None:
        self._step = self._total
        self._label = "done"
        if self._is_tty:
            line = self._render()
            sys.stderr.write(f"\r{line:<120}\n")
            sys.stderr.flush()


def _run(cmd: Sequence[str], *, cwd: Path) -> None:
    rendered = " ".join(cmd)
    print(f"[suite] {rendered}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {rendered}")


def _run_soft(cmd: Sequence[str], *, cwd: Path, label: str) -> tuple[bool, str]:
    """Run a command, returning (success, message) instead of aborting.

    Used for evaluation steps so that a single dataset or continuity
    failure doesn't prevent the remaining evaluations from running.
    """
    rendered = " ".join(cmd)
    print(f"[suite] {rendered}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        msg = f"{label} failed (exit {proc.returncode})"
        print(f"[suite] WARNING: {msg}", flush=True)
        return False, msg
    return True, ""


def _pick_single(pattern: str, *, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise SystemExit(f"missing artifact matching `{pattern}` in {root}")
    return matches[-1]


def _aggregate_dataset_reports(
    per_dataset_jsons: dict[str, Path],
    *,
    engine_tiers_evaluated: list[str],
) -> dict[str, Any]:
    """Load per-dataset benchmark JSONs and produce a combined report.

    The combined report wraps individual dataset reports under
    ``by_dataset`` and adds a ``cross_dataset_summary`` with averaged
    metrics per system.
    """
    by_dataset: dict[str, Any] = {}
    all_systems: dict[str, list[dict[str, Any]]] = {}  # system -> list of metrics dicts

    for dataset_name, json_path in per_dataset_jsons.items():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        by_dataset[dataset_name] = payload
        for row in payload.get("systems", []):
            system = str(row.get("system", "unknown"))
            all_systems.setdefault(system, []).append(
                {
                    "dataset": dataset_name,
                    "f1": float(row.get("f1", 0.0)),
                    "precision": float(row.get("precision", 0.0)),
                    "recall": float(row.get("recall", 0.0)),
                    "latency_p50_ms": float(row.get("latency_p50_ms", 0.0)),
                    "docs_per_hour": float(row.get("docs_per_hour", 0.0)),
                    "composite_score": float(row.get("composite_score", 0.0)),
                    "elo_rating": float(row.get("elo_rating", 0.0)),
                    "available": bool(row.get("available", False)),
                    "samples": int(row.get("samples", 0)),
                }
            )

    # Build cross-dataset summary: weighted average per system.
    cross_dataset_systems: list[dict[str, Any]] = []
    for system, entries in sorted(all_systems.items()):
        total_samples = sum(e["samples"] for e in entries)
        if total_samples <= 0:
            total_samples = len(entries)  # fallback

        def _wavg(key: str) -> float:
            num = sum(e[key] * max(1, e["samples"]) for e in entries)
            den = sum(max(1, e["samples"]) for e in entries)
            return round(num / den, 6) if den > 0 else 0.0

        best_dataset = max(entries, key=lambda e: e["f1"])["dataset"]
        worst_dataset = min(entries, key=lambda e: e["f1"])["dataset"]

        cross_dataset_systems.append(
            {
                "system": system,
                "datasets_evaluated": len(entries),
                "f1_average": _wavg("f1"),
                "precision_average": _wavg("precision"),
                "recall_average": _wavg("recall"),
                "latency_p50_ms_average": _wavg("latency_p50_ms"),
                "docs_per_hour_average": _wavg("docs_per_hour"),
                "composite_average": _wavg("composite_score"),
                "best_f1_dataset": best_dataset,
                "worst_f1_dataset": worst_dataset,
                "per_dataset": {e["dataset"]: {k: v for k, v in e.items() if k != "dataset"} for e in entries},
            }
        )

    return {
        "report_schema_version": "2026-02-19.v3",
        "datasets_evaluated": list(per_dataset_jsons.keys()),
        "engine_tiers_evaluated": engine_tiers_evaluated,
        "by_dataset": by_dataset,
        "cross_dataset_summary": {
            "systems": cross_dataset_systems,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run publish-grade benchmark suite (build/install/preflight/benchmark/continuity/docs-sync)"
    )
    parser.add_argument("--dataset", default="pii_anon_benchmark_v1")
    # Multi-dataset: overrides --dataset when provided (legacy, rarely needed).
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Evaluate against multiple datasets. "
            "Overrides --dataset when provided. "
            "Example: --datasets pii_anon_benchmark_v1"
        ),
    )
    parser.add_argument(
        "--matrix",
        default="src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json",
    )
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--dataset-source", choices=["auto", "package-only"], default="package-only")
    parser.add_argument("--artifacts-dir", default="artifacts/benchmarks")
    parser.add_argument("--checkpoint-dir", default="", help="Directory for per-profile checkpoint files (enables resume)")
    parser.add_argument("--work-dir", default=".publish-suite")
    parser.add_argument(
        "--strict-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require Linux runtime with shared-memory support",
    )
    parser.add_argument(
        "--require-all-competitors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require all configured competitors to be available",
    )
    parser.add_argument(
        "--require-native-competitors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require native competitor initialization readiness",
    )
    parser.add_argument(
        "--enforce-floors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail suite when benchmark floor/qualification gates fail",
    )
    parser.add_argument(
        "--include-end-to-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include core end-to-end track in competitor benchmark execution",
    )
    parser.add_argument(
        "--allow-core-native-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow native optional engines in core benchmark execution path",
    )
    parser.add_argument(
        "--enforce-publish-claims",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce canonical claim gates during summary/README validation",
    )
    parser.add_argument(
        "--validate-readme-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate README benchmark section sync against generated artifacts",
    )
    parser.add_argument(
        "--engine-tiers",
        nargs="+",
        choices=["auto", "minimal", "standard", "full"],
        default=None,
        help=(
            "Evaluate multiple engine tiers as separate pii-anon variants. "
            "Example: --engine-tiers auto minimal standard full"
        ),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-readme-update", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--reuse-current-env",
        action="store_true",
        help="Run suite checks in the current Python environment instead of creating an isolated venv",
    )
    parser.add_argument(
        "--install-no-deps",
        action="store_true",
        help="Install built wheels with `--no-deps` (use when dependencies are already provisioned)",
    )
    args = parser.parse_args()

    # Resolve dataset list: --datasets overrides --dataset.
    datasets: list[str] = args.datasets if args.datasets else [args.dataset]
    engine_tiers: list[str] | None = args.engine_tiers

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    work_dir = (repo_root / args.work_dir).resolve()
    dist_dir = work_dir / "dist"
    venv_dir = work_dir / "venv"
    python_bin = Path(args.python)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Count suite steps for progress bar.
    # Phase 0: setup
    step_count = 2  # install + preflight
    if not args.skip_build:
        step_count += 1
    # Phase 1: evaluations (always run to completion)
    step_count += len(datasets)  # one benchmark run per dataset
    step_count += 1  # continuity
    # Phase 2: post-processing
    if len(datasets) > 1:
        step_count += 1  # aggregation step
        step_count += 1  # marketing narrative
    step_count += 1  # render summary
    if args.validate_readme_sync:
        step_count += 1
    progress = _SuiteProgress(total_steps=step_count)

    if not args.skip_build:
        progress.step("Building wheel + sdist")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        dist_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                args.python,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--sdist",
                "--outdir",
                str(dist_dir),
                ".",
            ],
            cwd=repo_root,
        )
        # NOTE: pii-anon-datasets is now in its own repo (pii-anon-eval-data)
        # and must be built/published separately.

    if args.reuse_current_env:
        pip_command_prefix = [str(python_bin), "-m", "pip"]
    else:
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        _run([args.python, "-m", "venv", str(venv_dir)], cwd=repo_root)
        scripts_dir = venv_dir / ("Scripts" if sys.platform.startswith("win") else "bin")
        python_name = "python.exe" if sys.platform.startswith("win") else "python"
        pip_name = "pip.exe" if sys.platform.startswith("win") else "pip"
        python_bin = scripts_dir / python_name
        pip_command_prefix = [str(scripts_dir / pip_name)]
        _run([*pip_command_prefix, "install", "--upgrade", "pip"], cwd=repo_root)

    progress.step("Installing wheel into environment")
    core_wheel = _pick_single("pii_anon-*.whl", root=dist_dir)
    # pii-anon-datasets is in its own repo; install from local/PyPI separately.
    install_cmd = [*pip_command_prefix, "install"]
    if args.reuse_current_env:
        install_cmd.append("--force-reinstall")
    if args.install_no_deps:
        install_cmd.append("--no-deps")
    install_cmd.append(str(core_wheel))
    _run(install_cmd, cwd=repo_root)

    progress.step("Runtime preflight checks")
    preflight_json = artifacts_dir / "runtime-preflight.json"
    preflight_cmd = [
        str(python_bin),
        "scripts/check_benchmark_runtime.py",
        "--output-json",
        str(preflight_json),
    ]
    if args.strict_runtime:
        preflight_cmd.append("--strict-runtime")
    if args.require_all_competitors:
        preflight_cmd.append("--require-all-competitors")
    if args.require_native_competitors:
        preflight_cmd.append("--require-native-competitors")
    _run(
        preflight_cmd,
        cwd=repo_root,
    )

    # ==================================================================
    # PHASE 1: All evaluations (run to completion regardless of failures)
    #
    # Every time-consuming evaluation runs inside _run_soft so that a
    # single dataset or continuity failure does NOT prevent the remaining
    # evaluations from completing.  Failures are collected and reported
    # at the very end, after results are rendered.
    # ==================================================================
    evaluation_failures: list[str] = []
    per_dataset_jsons: dict[str, Path] = {}
    per_dataset_status: dict[str, bool] = {}
    primary_benchmark_json: Path | None = None

    for ds_index, dataset_name in enumerate(datasets, start=1):
        ds_suffix = f"-{dataset_name}" if len(datasets) > 1 else ""
        progress.step(
            f"Competitor benchmark: {dataset_name} [{ds_index}/{len(datasets)}]"
        )
        benchmark_json = artifacts_dir / f"benchmark-results{ds_suffix}.json"
        benchmark_csv = artifacts_dir / f"benchmark-raw{ds_suffix}.csv"
        floor_report = artifacts_dir / f"floor-gate-report{ds_suffix}.md"
        baseline_json = artifacts_dir / f"floor-baseline{ds_suffix}.v1.0.0.json"

        benchmark_cmd = [
            str(python_bin),
            "scripts/run_competitor_benchmark.py",
            "--dataset",
            dataset_name,
            "--matrix",
            args.matrix,
            "--dataset-source",
            args.dataset_source,
            "--warmup-samples",
            str(args.warmup_samples),
            "--measured-runs",
            str(args.measured_runs),
            "--output-json",
            str(benchmark_json),
            "--output-csv",
            str(benchmark_csv),
            "--output-floor-report",
            str(floor_report),
            "--output-baseline",
            str(baseline_json),
            "--preflight-output-json",
            str(preflight_json),
        ]
        if args.strict_runtime:
            benchmark_cmd.append("--strict-runtime")
        if args.require_all_competitors:
            benchmark_cmd.append("--require-all-competitors")
        if args.require_native_competitors:
            benchmark_cmd.append("--require-native-competitors")
        # NOTE: floor enforcement is deferred to the post-evaluation phase
        # so that all datasets finish even if one fails its gate.
        # Artifacts are always written (--write-artifacts-on-fail defaults to
        # True) so we can still render partial results.
        benchmark_cmd.append(
            "--include-end-to-end" if args.include_end_to_end else "--no-include-end-to-end"
        )
        benchmark_cmd.append(
            "--allow-core-native-engines"
            if args.allow_core_native_engines
            else "--no-allow-core-native-engines"
        )
        if engine_tiers:
            benchmark_cmd.extend(["--engine-tiers", *engine_tiers])
        # Checkpoint support: default to artifacts_dir/checkpoints/<dataset>
        # so that interrupted runs can be resumed by re-running the suite.
        ckpt_dir = args.checkpoint_dir
        if not ckpt_dir:
            ckpt_dir = str(artifacts_dir / "checkpoints" / dataset_name)
        benchmark_cmd.extend(["--checkpoint-dir", ckpt_dir])

        ok, msg = _run_soft(benchmark_cmd, cwd=repo_root, label=f"benchmark/{dataset_name}")
        per_dataset_status[dataset_name] = ok
        if ok and benchmark_json.exists():
            per_dataset_jsons[dataset_name] = benchmark_json
            if primary_benchmark_json is None:
                primary_benchmark_json = benchmark_json
        elif benchmark_json.exists():
            # The process failed but artifacts were written (--write-artifacts-on-fail).
            # Include them so partial results are still visible.
            per_dataset_jsons[dataset_name] = benchmark_json
            if primary_benchmark_json is None:
                primary_benchmark_json = benchmark_json
            evaluation_failures.append(msg)
        else:
            evaluation_failures.append(msg)

    # Continuity benchmark (also an evaluation — run it before post-processing).
    progress.step("Continuity benchmark")
    continuity_json = artifacts_dir / "continuity-results.json"
    continuity_md = artifacts_dir / "continuity-gate-report.md"
    cont_ok, cont_msg = _run_soft(
        [
            str(python_bin),
            "scripts/run_continuity_benchmark.py",
            "--max-samples",
            "0",
            "--enforce",
            "--output-json",
            str(continuity_json),
            "--output-markdown",
            str(continuity_md),
        ],
        cwd=repo_root,
        label="continuity benchmark",
    )
    if not cont_ok:
        evaluation_failures.append(cont_msg)

    # ==================================================================
    # PHASE 2: Post-processing (aggregation, rendering, validation)
    #
    # These are lightweight and only run if at least one dataset produced
    # artifacts.
    # ==================================================================
    combined_json: Path | None = None

    if not per_dataset_jsons:
        progress.close()
        print("[suite] ERROR: no dataset evaluations produced artifacts", flush=True)
        raise SystemExit("all benchmark evaluations failed; no artifacts to render")

    if len(per_dataset_jsons) > 1:
        progress.step("Aggregating multi-dataset results")
        combined_json = artifacts_dir / "benchmark-combined.json"
        combined_payload = _aggregate_dataset_reports(
            per_dataset_jsons,
            engine_tiers_evaluated=engine_tiers or ["auto"],
        )
        combined_json.write_text(
            json.dumps(combined_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {combined_json}")

    # Render summary.  When a combined report exists, pass it so the
    # renderer can produce cross-dataset analysis.
    progress.step("Rendering benchmark summary")
    benchmark_summary = artifacts_dir / "benchmark-summary.md"
    render_input = str(combined_json) if combined_json else str(primary_benchmark_json)
    render_cmd = [
        str(python_bin),
        "scripts/render_benchmark_summary.py",
        "--input-json",
        render_input,
    ]
    if combined_json:
        for ds_json_path in per_dataset_jsons.values():
            render_cmd.extend(["--input-json", str(ds_json_path)])
    render_cmd.extend(["--output-markdown", str(benchmark_summary)])
    # Do NOT pass --require-floor-pass during rendering when we are running
    # evaluations-first; floor enforcement happens below after rendering.
    if not args.skip_readme_update:
        render_cmd.extend(["--update-readme", "README.md"])
    _run(render_cmd, cwd=repo_root)

    # Render marketing narrative (only when a combined report exists).
    if combined_json:
        progress.step("Rendering marketing narrative")
        marketing_md = artifacts_dir / "marketing-narrative.md"
        marketing_cmd = [
            str(python_bin),
            "scripts/render_marketing_narrative.py",
            "--input-json",
            render_input,
            "--output-markdown",
            str(marketing_md),
        ]
        if not args.skip_readme_update:
            marketing_cmd.extend(["--update-readme", "README.md"])
        _run(marketing_cmd, cwd=repo_root)

    if args.validate_readme_sync:
        progress.step("Validating README sync")
        check_cmd = [
            str(python_bin),
            "scripts/check_readme_benchmark.py",
            "--readme",
            "README.md",
            "--summary",
            str(benchmark_summary),
        ]
        if args.enforce_publish_claims:
            check_cmd.extend(["--report-json", str(primary_benchmark_json)])
        complex_summary = repo_root / "docs" / "complex-mode-example.md"
        if complex_summary.exists():
            check_cmd.extend(["--complex-summary", str(complex_summary)])
        _run(check_cmd, cwd=repo_root)

    # ==================================================================
    # PHASE 3: Final summary & deferred enforcement
    # ==================================================================
    progress.close()

    # Print a per-dataset status table.
    passed_ds = [ds for ds, ok in per_dataset_status.items() if ok]

    print("", flush=True)
    print("=" * 72, flush=True)
    print("  BENCHMARK EVALUATION SUMMARY", flush=True)
    print("=" * 72, flush=True)
    for ds in datasets:
        status = "PASS" if per_dataset_status.get(ds) else "FAIL"
        print(f"  [{status}]  {ds}", flush=True)
    continuity_status = "PASS" if cont_ok else "FAIL"
    print(f"  [{continuity_status}]  continuity", flush=True)
    print("-" * 72, flush=True)
    print(
        f"  {len(passed_ds)}/{len(datasets)} datasets passed, "
        f"continuity {'passed' if cont_ok else 'FAILED'}",
        flush=True,
    )
    print(f"  artifacts in {artifacts_dir}", flush=True)
    print("=" * 72, flush=True)
    print("", flush=True)

    # Deferred floor enforcement: only fail AFTER everything has been
    # evaluated, aggregated, and rendered so the operator can review the
    # full results.
    if evaluation_failures and args.enforce_floors:
        details = "; ".join(evaluation_failures)
        raise SystemExit(
            f"suite completed with evaluation failures (--enforce-floors): {details}"
        )


if __name__ == "__main__":
    main()
