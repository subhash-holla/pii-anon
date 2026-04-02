#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Event, Thread
from typing import Any

from pii_anon.evaluation import compare_competitors, run_benchmark_runtime_preflight
from pii_anon.benchmarks import load_use_case_matrix, resolve_benchmark_dataset_path


REPORT_SCHEMA_VERSION = "2026-02-19.v3"

# Exclude bulky per-record/per-entity detail from the main JSON report.
# These are surfaced in the separate diagnostics artifact instead.
_EXCLUDE_KEYS = {"per_record_f1", "per_entity_errors", "error_counts"}


def _strip_per_record(d: dict[str, Any]) -> dict[str, Any]:
    """Remove per-record score lists from serialized system dicts."""
    return {k: v for k, v in d.items() if k not in _EXCLUDE_KEYS}


def _strip_per_record_profile(d: dict[str, Any]) -> dict[str, Any]:
    """Strip per-record scores from all systems in a profile dict."""
    result = dict(d)
    if "systems" in result and isinstance(result["systems"], list):
        result["systems"] = [_strip_per_record(s) for s in result["systems"]]
    if "end_to_end_systems" in result and isinstance(result["end_to_end_systems"], list):
        result["end_to_end_systems"] = [_strip_per_record(s) for s in result["end_to_end_systems"]]
    return result


def _progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[progress {timestamp}] {message}", flush=True)


def _read_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
    return out or "unknown"


def _winner_for_metric(
    systems: list[dict[str, Any]],
    *,
    metric: str,
    lower_is_better: bool = False,
) -> dict[str, Any] | None:
    pool = [
        item
        for item in systems
        if bool(item.get("available"))
        and item.get("system") != "pii-anon"
        and bool(item.get("license_gate_passed", False))
    ]
    if not pool:
        return None

    def _key_fn(item: dict[str, Any]) -> float:
        return float(item.get(metric, 0.0))

    winner = min(pool, key=_key_fn) if lower_is_better else max(pool, key=_key_fn)
    return {
        "system": str(winner.get("system", "unknown")),
        "metric": metric,
        "value": float(winner.get(metric, 0.0)),
    }


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds into a compact human-readable string."""
    if seconds < 0:
        return "--:--"
    total = int(seconds)
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}"


class _ProgressReporter:
    """High-fidelity progress reporter with work-unit tracking.

    The evaluation pipeline emits structured progress messages:

    - ``TOTAL:1234|message`` — declares total work units for 100%.
    - ``WORK:100|message``   — increments completed work by 100 units.
    - Plain messages          — info-only, no progress change.

    Percentage is computed as ``work_done / total_work * 100``.

    On TTYs, renders a **live single-line progress bar** (via rich) that
    updates in-place at 0.01% granularity with a constantly-updating
    ETA estimate.

    On non-TTYs (Docker/CI), emits **exactly one aggregated line per
    minute** via a heartbeat thread.  All ``__call__`` invocations only
    update internal state; the heartbeat thread is the sole emitter.
    This ensures a clean, predictable log regardless of how many
    parallel workers are active.

    Each non-TTY line includes a text progress bar, 0.01% percentage,
    elapsed time, ETA, work-unit counts, and the latest status message.
    """

    _RE_TOTAL = re.compile(r"^TOTAL:([\d.]+)\|(.*)$")
    _RE_WORK = re.compile(r"^WORK:([\d.]+)\|(.*)$")

    # Width of the ASCII progress bar (in characters).
    _BAR_WIDTH = 30

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._total_work = 0.0
        self._work_done = 0.0
        self._percent = 0.0
        self._last_message = "starting"
        self._stop = Event()
        self._thread: Thread | None = None
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        # Rich live display for TTY mode.
        self._live: Any = None

    # ------------------------------------------------------------------
    # ETA estimation
    # ------------------------------------------------------------------
    def _eta_seconds(self) -> float:
        """Estimate remaining seconds from work-unit throughput."""
        elapsed = time.monotonic() - self._start_time
        if self._work_done <= 0 or elapsed <= 0 or self._total_work <= 0:
            return -1.0
        rate = self._work_done / elapsed
        remaining = self._total_work - self._work_done
        if remaining <= 0:
            return 0.0
        return remaining / rate

    # ------------------------------------------------------------------
    # Rendering — TTY (rich live display)
    # ------------------------------------------------------------------
    def _render_rich_str(self) -> str:
        """Build a rich-markup progress line for the TTY live display."""
        bar_width = 40
        filled = int(bar_width * self._percent / 100.0)
        bar_filled = "\u2588" * filled
        bar_empty = "\u2591" * (bar_width - filled)
        elapsed = time.monotonic() - self._start_time
        eta = self._eta_seconds()
        eta_str = _format_duration(eta) if eta >= 0 else "--:--"
        return (
            f"[cyan]{bar_filled}{bar_empty}[/cyan]  "
            f"[bold green]{self._percent:6.2f}%[/bold green]  "
            f"[dim]{_format_duration(elapsed)} elapsed[/dim]  "
            f"[yellow]ETA {eta_str}[/yellow]  "
            f"{self._last_message}"
        )

    def _update_live(self) -> None:
        """Update the rich live display if in TTY mode."""
        if self._live is not None:
            try:
                from rich.text import Text
                self._live.update(Text.from_markup(self._render_rich_str()))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Rendering — non-TTY (Docker/CI one-line-per-minute)
    # ------------------------------------------------------------------
    def _render_non_tty(self) -> None:
        """Print a single aggregated progress line for Docker/CI logs.

        Includes a text progress bar, 0.01% percentage, elapsed, ETA,
        work-unit counts, and the latest status message.
        """
        filled = int(self._BAR_WIDTH * self._percent / 100.0)
        bar = "\u2588" * filled + "\u2591" * (self._BAR_WIDTH - filled)

        elapsed = time.monotonic() - self._start_time
        eta = self._eta_seconds()
        eta_str = _format_duration(eta) if eta >= 0 else "--:--"
        work_str = (
            f"{self._work_done:.0f}/{self._total_work:.0f}"
            if self._total_work else "n/a"
        )
        _progress(
            f"|{bar}| {self._percent:6.2f}% | "
            f"{_format_duration(elapsed)} elapsed | "
            f"ETA {eta_str} | "
            f"work: {work_str} | "
            f"{self._last_message}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def __call__(self, message: str) -> None:
        total_match = self._RE_TOTAL.match(message)
        work_match = self._RE_WORK.match(message)

        if total_match:
            self._total_work = float(total_match.group(1))
            self._last_message = total_match.group(2)
        elif work_match:
            self._work_done += float(work_match.group(1))
            self._last_message = work_match.group(2)
            if self._total_work > 0:
                self._percent = min(99.99, (self._work_done / self._total_work) * 100.0)
        else:
            self._last_message = message

        # TTY mode: update the rich live display on every message.
        # Non-TTY mode: state is updated above; the heartbeat thread
        # handles all output so we never emit here.
        if self._is_tty:
            self._update_live()

    def start(self) -> None:
        self._start_time = time.monotonic()
        if self._is_tty:
            try:
                from rich.live import Live
                from rich.text import Text
                self._live = Live(
                    Text.from_markup(self._render_rich_str()),
                    console=None,
                    refresh_per_second=4,
                    transient=False,
                )
                self._live.start()
            except ImportError:
                # Fallback: rich not available, degrade to non-TTY mode.
                self._is_tty = False
        if not self._is_tty:
            # Emit the initial state line immediately, then start a
            # heartbeat thread that emits exactly one line per minute.
            self._render_non_tty()

            def _heartbeat() -> None:
                while not self._stop.wait(timeout=60):
                    self._render_non_tty()

            self._thread = Thread(target=_heartbeat, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._percent = 100.0
        self._last_message = "complete"
        if self._live is not None:
            try:
                from rich.text import Text
                self._live.update(Text.from_markup(self._render_rich_str()))
                self._live.stop()
            except Exception:
                pass
            self._live = None
        else:
            self._render_non_tty()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _update_readme_benchmark_section(
    report: Any,
    payload: dict[str, Any],
) -> None:
    """Replace the benchmark summary section in README.md with latest results.

    Looks for ``<!-- BENCHMARK_SUMMARY_START -->`` and ``<!-- BENCHMARK_SUMMARY_END -->``
    markers and replaces everything between them with a fresh summary table
    generated from the current benchmark run.  If the markers or the README
    don't exist, this is a no-op.
    """
    # Walk up from the script to find the repo root README.md
    readme_candidates = [
        Path(__file__).resolve().parent.parent / "README.md",
        Path.cwd() / "README.md",
    ]
    readme_path: Path | None = None
    for candidate in readme_candidates:
        if candidate.exists():
            readme_path = candidate
            break
    if readme_path is None:
        return

    start_marker = "<!-- BENCHMARK_SUMMARY_START -->"
    end_marker = "<!-- BENCHMARK_SUMMARY_END -->"

    content = readme_path.read_text(encoding="utf-8")
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        return

    # --- Build the markdown summary ---
    lines: list[str] = [start_marker, ""]

    # Group profiles by objective for separate tables
    from collections import defaultdict

    by_objective: dict[str, list[Any]] = defaultdict(list)
    for profile in report.profiles:
        by_objective[profile.objective].append(profile)

    objective_labels = {"accuracy": "Accuracy", "speed": "Speed", "balanced": "Balanced"}

    for objective, profiles in sorted(by_objective.items()):
        profile_names = ", ".join(p.profile for p in profiles)
        label = objective_labels.get(objective, objective.title())
        lines.append(f"## {label} Objective (profiles: {profile_names})")
        lines.append("")
        lines.append(f"Benchmark dataset: `{report.dataset}`")
        lines.append(
            f"Warm-up samples/system: `{report.warmup_samples}`. "
            f"Measured runs/system: `{report.measured_runs}`."
        )
        lines.append("")

        # Collect all systems across profiles in this objective
        all_systems: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            for sys in profile.systems:
                if sys.system not in all_systems:
                    status = "available" if sys.available else f"skipped ({sys.skipped_reason or 'unavailable'})"
                    all_systems[sys.system] = {
                        "status": status,
                        "composite": sys.composite_score,
                        "f1": sys.f1,
                        "ci_lower": sys.f1_ci_lower,
                        "ci_upper": sys.f1_ci_upper,
                        "precision": sys.precision,
                        "recall": sys.recall,
                        "latency": sys.latency_p50_ms,
                        "docs_per_hour": sys.docs_per_hour,
                        "elo": sys.elo_rating,
                    }

        # Render table
        lines.append(
            "| System | Status | Composite | F1 | 95% CI | Precision | Recall "
            "| p50 Latency (ms) | Docs/hour | Elo |"
        )
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
        for sys_name in sorted(all_systems.keys()):
            s = all_systems[sys_name]
            ci_str = "—"
            if s["ci_lower"] and s["ci_upper"] and s["ci_lower"] > 0:
                ci_str = f"[{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]"
            lines.append(
                f"| {sys_name} | {s['status']} | {s['composite']:.4f} | {s['f1']:.3f} "
                f"| {ci_str} | {s['precision']:.3f} | {s['recall']:.3f} "
                f"| {s['latency']:.3f} | {s['docs_per_hour']:.2f} | {s['elo']} |"
            )

        # Strengths/weaknesses for pii-anon
        pii_anon = all_systems.get("pii-anon")
        if pii_anon:
            available_systems = {k: v for k, v in all_systems.items() if v["status"] == "available"}
            strength_lines: list[str] = []
            weakness_lines: list[str] = []
            for metric, key, lower_better in [
                ("composite_score", "composite", False),
                ("f1", "f1", False),
                ("precision", "precision", False),
                ("recall", "recall", False),
                ("docs_per_hour", "docs_per_hour", False),
                ("latency_p50_ms", "latency", True),
            ]:
                values = [v[key] for v in available_systems.values() if v[key] > 0]
                if not values:
                    continue
                best = min(values) if lower_better else max(values)
                val = pii_anon[key]
                if best == 0:
                    continue
                if lower_better:
                    ratio = val / best if best > 0 else 999
                    if ratio <= 1.05:
                        strength_lines.append(f"- {metric}: within 5% of best ({val:.3f} vs best {best:.3f}).")
                    elif ratio > 1.10:
                        weakness_lines.append(f"- {metric}: more than 10% slower than best ({val:.3f} vs best {best:.3f}).")
                else:
                    ratio = val / best if best > 0 else 0
                    if ratio >= 0.95:
                        strength_lines.append(f"- {metric}: within 5% of best ({val:.3f} vs best {best:.3f}).")
                    elif ratio < 0.90:
                        weakness_lines.append(f"- {metric}: more than 10% below best ({val:.3f} vs best {best:.3f}).")
            lines.append("")
            if strength_lines:
                lines.append(f"Strengths for `pii-anon`:")
                lines.extend(strength_lines)
                lines.append("")
            if weakness_lines:
                lines.append(f"Weaknesses for `pii-anon`:")
                lines.extend(weakness_lines)
                lines.append("")

        lines.append("This section is generated from benchmark artifacts.")

    # Floor-gate results
    lines.append("")
    lines.append("Profile floor-gate results:")
    for profile in report.profiles:
        lines.append(f"- `{profile.profile}` ({profile.objective}): floor_pass={profile.floor_pass}")
    lines.append("")

    # Statistical significance
    stat_tests = report.statistical_tests
    if stat_tests and isinstance(stat_tests, dict):
        samples = stat_tests.get("total_samples", 0)
        mde = stat_tests.get("mde", 0)
        lines.append("### Statistical Significance")
        lines.append("")
        if samples and mde:
            lines.append(
                f"Evaluated on **{samples:,}** records. "
                f"Minimum detectable effect (MDE) at \u03b1=0.05, power=0.80: **{mde:.4f}** F1 points."
            )
            lines.append("")
        per_system = stat_tests.get("per_system", {})
        if per_system:
            lines.append("| System | F1 | 95% CI | Samples |")
            lines.append("|---|---:|---|---:|")
            for sys_name, sys_stats in sorted(per_system.items()):
                if isinstance(sys_stats, dict):
                    f1 = sys_stats.get("f1", 0)
                    ci = sys_stats.get("ci_95", [0, 0])
                    n = sys_stats.get("samples", samples)
                    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and len(ci) == 2 else "—"
                    lines.append(f"| {sys_name} | {f1:.3f} | {ci_str} | {n:,} |")
            lines.append("")
        pairwise = stat_tests.get("pairwise", [])
        if pairwise:
            lines.append("Pairwise comparisons (paired bootstrap, n=10,000 resamples):")
            lines.append("")
            lines.append("| Comparison | \u0394F1 | p-value | Significant | Effect Size |")
            lines.append("|---|---:|---:|---|---|")
            for pair in pairwise:
                if isinstance(pair, dict):
                    comp = pair.get("comparison", "")
                    delta = pair.get("delta_f1", 0)
                    pval = pair.get("p_value", 1)
                    sig = "**yes**" if pval < 0.05 else "n.s."
                    effect = pair.get("effect_size", "")
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"| {comp} | {sign}{delta:.4f} | {pval:.4f} | {sig} | {effect} |")
            lines.append("")
            lines.append(
                "*Method: paired bootstrap significance test (Berg-Kirkpatrick et al., 2012). "
                "Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8).*"
            )
            lines.append("")

    lines.append("")
    lines.append(end_marker)

    # Splice into README
    new_content = content[: start_idx] + "\n".join(lines) + content[end_idx + len(end_marker) :]
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"updated {readme_path} with latest benchmark results")


def _auto_cleanup_old_artifacts(*output_paths: str) -> None:
    """Archive previous benchmark artifacts before writing new ones.

    Scans the parent directory of the first output path for existing benchmark
    artifacts (``*.json``, ``*.csv``, ``*.md``) and moves them into an
    ``_archive/<timestamp>/`` subdirectory.  This ensures each benchmark run
    starts clean and only the latest results are visible at the top level.

    Skips silently if the output directory does not yet exist.
    """
    import shutil

    # Determine the artifact directory from the first output path
    artifact_dirs: set[Path] = set()
    for p in output_paths:
        if p:
            artifact_dirs.add(Path(p).parent)
    if not artifact_dirs:
        return

    for artifact_dir in artifact_dirs:
        if not artifact_dir.exists():
            continue

        # Gather stale artifact files
        stale: list[Path] = []
        for ext in ("*.json", "*.csv", "*.md"):
            stale.extend(artifact_dir.glob(ext))
        # Also grab any checkpoints subdirectory
        ckpt_dir = artifact_dir / "checkpoints"
        if ckpt_dir.is_dir():
            stale.append(ckpt_dir)

        if not stale:
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = artifact_dir / "_archive" / ts
        archive_dir.mkdir(parents=True, exist_ok=True)

        for item in stale:
            dest = archive_dir / item.name
            try:
                shutil.move(str(item), str(dest))
            except Exception:
                pass  # Best effort; don't block the benchmark

        print(f"archived {len(stale)} old artifact(s) to {archive_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run competitor benchmark and write report artifacts")
    parser.add_argument("--dataset", default="pii_anon_benchmark")
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--matrix", default="")
    parser.add_argument("--dataset-source", choices=["auto", "package-only"], default="auto")
    parser.add_argument("--enforce-floors", action="store_true")
    parser.add_argument("--strict-runtime", action="store_true")
    parser.add_argument("--require-all-competitors", action="store_true")
    parser.add_argument("--require-native-competitors", action="store_true")
    parser.add_argument(
        "--include-end-to-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include core end-to-end evaluation track in benchmark artifacts",
    )
    parser.add_argument(
        "--allow-core-native-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow native optional engines in core benchmark execution path",
    )
    parser.add_argument("--preflight-output-json", default="")
    parser.add_argument("--output-json", default="benchmark-results.json")
    parser.add_argument("--output-csv", default="benchmark-raw.csv")
    parser.add_argument("--output-diagnostics", default="benchmark-diagnostics.json")
    parser.add_argument("--output-floor-report", default="floor-gate-report.md")
    parser.add_argument("--output-baseline", default="")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate competitors in parallel (default: enabled; use --no-parallel to disable)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help=(
            "Directory for per-profile checkpoint files.  When set, each "
            "completed profile is saved to this directory as JSON.  On "
            "resume, previously-completed profiles are loaded from disk "
            "and skipped, saving hours on long benchmark runs.  Use the "
            "same directory on re-run to resume."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help=(
            "Evaluate only the listed profile names (for parallel execution). "
            "Each profile writes its own checkpoint file.  Example: "
            "--profiles short_chat long_document"
        ),
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Skip profile evaluation entirely.  Instead, read all checkpoint "
            "files from --checkpoint-dir, merge systems, compute Elo ratings "
            "and statistical tests, then write final report artifacts.  "
            "Requires --checkpoint-dir."
        ),
    )
    parser.add_argument("--quiet-progress", action="store_true", help="Disable periodic progress output")
    parser.add_argument("--write-artifacts-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-after-write", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress-interval-sec", type=int, default=30)
    args = parser.parse_args()

    fail_after_write = bool(args.enforce_floors) if args.fail_after_write is None else bool(args.fail_after_write)
    reporter = None if args.quiet_progress else _ProgressReporter()
    progress_hook = None if reporter is None else reporter

    # ------------------------------------------------------------------
    # Auto-cleanup: remove stale artifacts before each new run so only
    # the latest results are present.  Old artifacts are moved to an
    # `_archive/<timestamp>/` subdirectory alongside the output files.
    # ------------------------------------------------------------------
    _auto_cleanup_old_artifacts(args.output_json, args.output_csv)

    # ------------------------------------------------------------------
    # Merge-only mode: skip evaluation, just merge checkpoint files.
    # ------------------------------------------------------------------
    if args.merge_only:
        if not args.checkpoint_dir:
            raise SystemExit("--merge-only requires --checkpoint-dir to be set")

        from pii_anon.evaluation import merge_profile_checkpoints

        if reporter is not None:
            reporter.start()
            progress_hook("starting merge-only phase")

        try:
            report = merge_profile_checkpoints(
                checkpoint_dir=args.checkpoint_dir,
                dataset=args.dataset,
                dataset_source=args.dataset_source,
                warmup_samples=args.warmup_samples,
                measured_runs=args.measured_runs,
                matrix_path=args.matrix or None,
                require_all_competitors=bool(args.require_all_competitors),
                require_native_competitors=bool(args.require_native_competitors),
                enforce_floors=False,
                progress_hook=progress_hook,
            )
        finally:
            if reporter is not None:
                reporter.stop()

        # Preflight report not available in merge-only mode.
        preflight_report: dict[str, Any] = {"ready": True, "merge_only": True}

        # Fall through to artifact writing below (line with dataset_path =).
        # We skip to after the normal try/finally block.

    else:
        # ------------------------------------------------------------------
        # Normal evaluation mode.
        # ------------------------------------------------------------------
        if reporter is not None:
            reporter.start()
            progress_hook("starting competitor benchmark run")

        preflight_report = run_benchmark_runtime_preflight(
            strict_runtime=bool(args.strict_runtime),
            require_all_competitors=bool(args.require_all_competitors),
            require_native_competitors=bool(args.require_native_competitors),
        )
        if args.preflight_output_json:
            preflight_out = Path(args.preflight_output_json)
            preflight_out.parent.mkdir(parents=True, exist_ok=True)
            preflight_out.write_text(json.dumps(preflight_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {preflight_out}")
        if not bool(preflight_report.get("ready", False)):
            failures = preflight_report.get("failures", [])
            if isinstance(failures, list) and failures:
                raise SystemExit("runtime preflight failed: " + "; ".join(str(item) for item in failures))
            raise SystemExit("runtime preflight failed")

        forced_unavailable_competitors = {
            str(key): str(value)
            for key, value in (preflight_report.get("unavailable_competitors", {}) or {}).items()
        }

        try:
            report = compare_competitors(
                dataset=args.dataset,
                dataset_source=args.dataset_source,
                warmup_samples=args.warmup_samples,
                measured_runs=args.measured_runs,
                max_samples=args.max_samples if args.max_samples > 0 else None,
                matrix_path=args.matrix or None,
                profile_filter=args.profiles,
                enforce_floors=False,
                require_all_competitors=bool(args.require_all_competitors),
                require_native_competitors=bool(args.require_native_competitors),
                allow_fallback_detectors=not bool(args.require_native_competitors),
                include_end_to_end=bool(args.include_end_to_end),
                forced_unavailable_competitors=forced_unavailable_competitors,
                allow_core_native_engines=bool(args.allow_core_native_engines),
                progress_hook=progress_hook,
                enable_parallel=bool(args.parallel),
                checkpoint_dir=args.checkpoint_dir or None,
            )
        finally:
            if reporter is not None:
                reporter.stop()

    if progress_hook:
        progress_hook(
            f"benchmark run finished: floor_pass={report.floor_pass}, "
            f"qualification_gate_pass={report.qualification_gate_pass}"
        )

    dataset_path = resolve_benchmark_dataset_path(args.dataset, source=args.dataset_source)
    matrix_path = Path(args.matrix) if args.matrix else None
    run_metadata: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "dataset_sha256": _read_sha256(dataset_path) if dataset_path is not None else None,
        "dataset_source": args.dataset_source,
        "matrix_path": str(matrix_path) if matrix_path is not None else None,
        "matrix_sha256": _read_sha256(matrix_path) if matrix_path is not None else None,
        "strict_runtime": bool(args.strict_runtime),
        "require_all_competitors": bool(args.require_all_competitors),
        "require_native_competitors": bool(args.require_native_competitors),
    }

    profile_results: list[dict[str, Any]] = []
    failed_profiles: list[str] = []
    for profile in report.profiles:
        if not profile.floor_pass:
            failed_profiles.append(profile.profile)
        systems_rows = [asdict(item) for item in profile.systems]
        winners: dict[str, Any] = {}
        objective = profile.objective
        if objective == "accuracy":
            winners["f1"] = _winner_for_metric(systems_rows, metric="f1", lower_is_better=False)
            winners["recall"] = _winner_for_metric(systems_rows, metric="recall", lower_is_better=False)
        elif objective == "speed":
            winners["latency_p50_ms"] = _winner_for_metric(systems_rows, metric="latency_p50_ms", lower_is_better=True)
            winners["docs_per_hour"] = _winner_for_metric(systems_rows, metric="docs_per_hour", lower_is_better=False)
        else:
            winners["f1"] = _winner_for_metric(systems_rows, metric="f1", lower_is_better=False)
            winners["latency_p50_ms"] = _winner_for_metric(systems_rows, metric="latency_p50_ms", lower_is_better=True)
        profile_results.append(
            {
                "profile": profile.profile,
                "objective": profile.objective,
                "evaluation_track": "detect_only",
                "floor_pass": profile.floor_pass,
                "qualified_competitors": profile.qualified_competitors,
                "mit_qualified_competitors": profile.mit_qualified_competitors,
                "floor_checks": [asdict(item) for item in profile.floor_checks],
                "winners": winners,
            }
        )

    required_profiles: list[str] = []
    if matrix_path is not None and matrix_path.exists():
        required_profiles = [
            item.profile
            for item in load_use_case_matrix(str(matrix_path))
            if item.required
        ]
    required_profiles_passed = all(
        profile.floor_pass for profile in report.profiles if profile.profile in required_profiles
    ) if required_profiles else False
    default_matrix_path = (Path("src") / "pii_anon" / "benchmarks" / "matrix" / "use_case_matrix.json").resolve()
    canonical_claim_run = bool(
        matrix_path is not None
        and matrix_path.resolve() == default_matrix_path
        and args.max_samples <= 0
        and args.warmup_samples >= 100
        and args.measured_runs >= 3
        and required_profiles_passed
        and report.all_competitors_available
        and args.dataset_source == "package-only"
        and bool(args.require_all_competitors)
        and bool(args.require_native_competitors)
        and bool(args.strict_runtime)
    )
    run_metadata["canonical_claim_run"] = canonical_claim_run
    run_metadata["required_profiles"] = required_profiles
    run_metadata["required_profiles_passed"] = required_profiles_passed

    payload: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "dataset": report.dataset,
        "warmup_samples": report.warmup_samples,
        "measured_runs": report.measured_runs,
        "max_samples": args.max_samples if args.max_samples > 0 else None,
        "dataset_source": report.dataset_source,
        "floor_pass": report.floor_pass,
        "qualification_gate_pass": report.qualification_gate_pass,
        "mit_gate_pass": report.mit_gate_pass,
        "expected_competitors": report.expected_competitors,
        "available_competitors": report.available_competitors,
        "unavailable_competitors": report.unavailable_competitors,
        "all_competitors_available": report.all_competitors_available,
        "require_all_competitors": report.require_all_competitors,
        "require_native_competitors": report.require_native_competitors,
        "run_metadata": run_metadata,
        "runtime_preflight": preflight_report,
        "failed_profiles": failed_profiles,
        "required_profiles": required_profiles,
        "required_profiles_passed": required_profiles_passed,
        "profile_results": profile_results,
        "systems": [_strip_per_record(asdict(item)) for item in report.systems],
        "profiles": [_strip_per_record_profile(asdict(item)) for item in report.profiles],
        "evaluation_tracks": {
            "detect_only": [_strip_per_record(asdict(item)) for item in report.systems],
            "end_to_end": [_strip_per_record(asdict(item)) for profile in report.profiles for item in profile.end_to_end_systems],
        },
        "statistical_tests": report.statistical_tests,
        "diagnostics": report.diagnostics,
    }

    if args.write_artifacts_on_fail or (report.floor_pass and report.qualification_gate_pass):
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "system",
                    "available",
                    "skipped_reason",
                    "license_gate_passed",
                    "license_gate_reason",
                    "qualification_status",
                    "license_name",
                    "license_source",
                    "citation_url",
                    "precision",
                    "recall",
                    "f1",
                    "f1_ci_lower",
                    "f1_ci_upper",
                    "latency_p50_ms",
                    "docs_per_hour",
                    "composite_score",
                    "elo_rating",
                    "entity_types_detected",
                    "entity_types_total",
                    "samples",
                    "per_entity_recall",
                    "per_entity_precision",
                    "per_entity_f1",
                    "dominance_pass_by_profile",
                    "evaluation_track",
                ],
            )
            writer.writeheader()
            systems_rows = payload.get("systems", [])
            if not isinstance(systems_rows, list):
                raise SystemExit("invalid benchmark payload: `systems` must be a list")
            for row in systems_rows:
                if not isinstance(row, dict):
                    raise SystemExit("invalid benchmark payload: system rows must be objects")
                writer.writerow(
                    {
                        **row,
                        "per_entity_recall": json.dumps(row.get("per_entity_recall", {}), sort_keys=True),
                        "per_entity_precision": json.dumps(row.get("per_entity_precision", {}), sort_keys=True),
                        "per_entity_f1": json.dumps(row.get("per_entity_f1", {}), sort_keys=True),
                        "dominance_pass_by_profile": json.dumps(row.get("dominance_pass_by_profile", {}), sort_keys=True),
                    }
                )

        floor_lines = [
            f"# Floor Gate Report ({report.dataset})",
            "",
            f"Overall floor pass: `{report.floor_pass}`",
            f"Overall qualification gate pass: `{report.qualification_gate_pass}`",
            f"All competitors available: `{report.all_competitors_available}`",
            f"Failed profiles: `{', '.join(failed_profiles) if failed_profiles else 'none'}`",
            "",
        ]
        if report.unavailable_competitors:
            floor_lines.append("Unavailable competitors:")
            for system, reason in sorted(report.unavailable_competitors.items()):
                floor_lines.append(f"- `{system}`: {reason}")
            floor_lines.append("")
        for profile in report.profiles:
            floor_lines.append(f"## Profile `{profile.profile}` ({profile.objective})")
            floor_lines.append(f"- floor_pass: `{profile.floor_pass}`")
            floor_lines.append(f"- qualified_competitors: `{profile.qualified_competitors}`")
            if profile.floor_checks:
                for check in profile.floor_checks:
                    floor_lines.append(
                        f"- {check.metric}: actual={check.actual:.3f}, target={check.target:.3f}, comparator={check.comparator}, passed={check.passed}"
                    )
            else:
                floor_lines.append("- no floor checks evaluated")
            winners = next((item for item in profile_results if item["profile"] == profile.profile), None)
            if isinstance(winners, dict):
                floor_lines.append(f"- winners: `{json.dumps(winners.get('winners', {}), sort_keys=True)}`")
            excluded = [row for row in profile.systems if row.system != "pii-anon" and not row.license_gate_passed]
            for row in excluded:
                floor_lines.append(
                    f"- excluded competitor `{row.system}`: {row.license_gate_reason or 'qualification evidence missing'}"
                )
            floor_lines.append("")
        floor_report = Path(args.output_floor_report)
        floor_report.parent.mkdir(parents=True, exist_ok=True)
        floor_report.write_text("\n".join(floor_lines).strip() + "\n", encoding="utf-8")

        # Write detailed diagnostics artifact with per-system error breakdowns.
        diag_payload: dict[str, Any] = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "timestamp_utc": run_metadata.get("timestamp_utc", ""),
            "dataset": report.dataset,
            "diagnostics": report.diagnostics,
            "per_system_detail": {},
        }
        for sys in report.systems:
            if sys.available:
                sys_detail: dict[str, Any] = {
                    "f1": sys.f1,
                    "precision": sys.precision,
                    "recall": sys.recall,
                    "per_entity_f1": sys.per_entity_f1,
                    "per_entity_precision": sys.per_entity_precision,
                    "per_entity_recall": sys.per_entity_recall,
                    "error_counts": sys.error_counts,
                    "per_entity_errors": sys.per_entity_errors,
                    "entity_types_detected": sys.entity_types_detected,
                    "entity_types_total": sys.entity_types_total,
                }
                diag_payload["per_system_detail"][sys.system] = sys_detail
        diag_out = Path(args.output_diagnostics)
        diag_out.parent.mkdir(parents=True, exist_ok=True)
        diag_out.write_text(json.dumps(diag_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if args.output_baseline:
            baseline = Path(args.output_baseline)
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {baseline}")

        print(f"wrote {output_json}")
        print(f"wrote {output_csv}")
        print(f"wrote {floor_report}")
        print(f"wrote {diag_out}")

        # Auto-update README.md with latest benchmark results
        _update_readme_benchmark_section(report, payload)

    should_fail = (
        bool(args.enforce_floors)
        and fail_after_write
        and (
            not report.floor_pass
            or not report.qualification_gate_pass
            or (bool(args.require_all_competitors) and not report.all_competitors_available)
        )
    )
    if should_fail:
        if bool(args.require_all_competitors) and not report.all_competitors_available:
            failed_systems = ", ".join(sorted(report.unavailable_competitors.keys()))
            raise SystemExit(f"competitor availability gate failed for systems: {failed_systems}")
        failed = ", ".join(failed_profiles) if failed_profiles else "unknown"
        raise SystemExit(f"floor gate failed for profiles: {failed}")


if __name__ == "__main__":
    main()
