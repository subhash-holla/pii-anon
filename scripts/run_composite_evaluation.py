#!/usr/bin/env python3
"""Comprehensive composite metric evaluation for pii-anon v1.0.0.

Evaluates ALL systems (pii-anon + competitors) against the full
benchmark dataset using:

1. Standard metrics: Precision, Recall, F1, Latency, Throughput
2. Tier 1 composite scoring via ``compute_composite()``
3. PII-Rate-Elo round-robin tournament via ``PIIRateEloEngine``
4. Ranked leaderboard generation with JSON/MD/CSV export

Usage::

    python scripts/run_composite_evaluation.py
    python scripts/run_composite_evaluation.py --dataset eval_framework_v1 --max-samples 50000
    python scripts/run_composite_evaluation.py --output-dir results/composite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the source tree is on sys.path when running from repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Patch environment for offline operation (blank spacy models, etc.)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
try:
    from _env_setup import setup_offline_env
    setup_offline_env()
except ImportError:
    pass  # Not available in all environments

from pii_anon.evaluation.competitor_compare import (
    CompetitorComparisonReport,
    SystemBenchmarkResult,
    compare_competitors,
)
from pii_anon.eval_framework.metrics.composite import (
    CompositeConfig,
    CompositeScore,
    compute_composite,
)
from pii_anon.eval_framework.rating.elo import PIIRateEloEngine
from pii_anon.eval_framework.rating.scorecard import (
    BenchmarkScorecard,
    SystemScorecard,
)
from pii_anon.eval_framework.rating.leaderboard import (
    Leaderboard,
    LeaderboardExporter,
)


def _progress(msg: str) -> None:
    """Print a timestamped progress message."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_composite_scores(
    systems: list[SystemBenchmarkResult],
    config: CompositeConfig,
) -> dict[str, CompositeScore]:
    """Compute composite scores for every available system."""
    scores: dict[str, CompositeScore] = {}
    for sys_result in systems:
        if not sys_result.available:
            continue
        cs = compute_composite(
            f1=sys_result.f1,
            precision=sys_result.precision,
            recall=sys_result.recall,
            latency_ms=sys_result.latency_p50_ms,
            docs_per_hour=sys_result.docs_per_hour,
            config=config,
        )
        scores[sys_result.system] = cs
    return scores


def _build_scorecards(
    systems: list[SystemBenchmarkResult],
    composite_scores: dict[str, CompositeScore],
    elo_ratings: dict[str, tuple[float, float]],
) -> BenchmarkScorecard:
    """Assemble SystemScorecard objects and pack into BenchmarkScorecard."""
    benchmark = BenchmarkScorecard(
        benchmark_name="pii-anon composite evaluation v1.0.0",
        dataset_name="eval_framework_v1",
    )
    for sys_result in systems:
        cs = composite_scores.get(sys_result.system)
        elo_rating, elo_rd = elo_ratings.get(sys_result.system, (1500.0, 350.0))
        sc = SystemScorecard(
            system_name=sys_result.system,
            available=sys_result.available,
            f1=sys_result.f1,
            precision=sys_result.precision,
            recall=sys_result.recall,
            latency_p50_ms=sys_result.latency_p50_ms,
            docs_per_hour=sys_result.docs_per_hour,
            composite_score=cs.score if cs else 0.0,
            elo_rating=elo_rating,
            elo_rd=elo_rd,
            samples=sys_result.samples,
            evaluation_track=sys_result.evaluation_track,
            license_name=sys_result.license_name,
        )
        benchmark.add_system(sc)
    return benchmark


def run_evaluation(
    *,
    dataset: str = "eval_framework_v1",
    max_samples: int | None = None,
    warmup_samples: int = 50,
    measured_runs: int = 1,
    output_dir: str = "results/composite",
    elo_rounds: int = 3,
) -> dict:
    """Run the full composite evaluation pipeline.

    Returns a dictionary with the complete evaluation results.
    """
    config = CompositeConfig()  # Default Tier 1 weights

    _progress(f"Starting composite evaluation on dataset='{dataset}'")
    _progress(f"  max_samples={max_samples}, warmup={warmup_samples}, measured_runs={measured_runs}")
    _progress(f"  Composite config: F1={config.weight_detection_f1}, "
              f"P={config.weight_detection_precision}, R={config.weight_detection_recall}, "
              f"Lat={config.weight_latency}, Thr={config.weight_throughput}")

    # ── Step 1: Run competitor comparison benchmark ────────────────────
    _progress("Step 1: Running competitor comparison benchmark...")
    report: CompetitorComparisonReport = compare_competitors(
        dataset=dataset,
        warmup_samples=warmup_samples,
        measured_runs=measured_runs,
        max_samples=max_samples,
        objective="accuracy",
        use_case="default",
        allow_fallback_detectors=True,
        require_native_competitors=False,
        include_end_to_end=False,
        allow_core_native_engines=False,
        progress_hook=_progress,
    )
    _progress(f"Benchmark complete: {len(report.systems)} systems evaluated")

    # ── Step 2: Compute composite scores ──────────────────────────────
    _progress("Step 2: Computing composite scores...")
    composite_scores = _build_composite_scores(report.systems, config)
    for name, cs in sorted(composite_scores.items(), key=lambda x: -x[1].score):
        _progress(f"  {name}: composite={cs.score:.4f} "
                  f"(detection={cs.detection_sub:.4f}, efficiency={cs.efficiency_sub:.4f})")

    # ── Step 3: Run PII-Rate-Elo tournament ───────────────────────────────
    _progress(f"Step 3: Running PII-Rate-Elo round-robin tournament ({elo_rounds} rounds)...")
    elo_engine = PIIRateEloEngine()
    composites_map = {name: cs.score for name, cs in composite_scores.items()}

    for round_num in range(elo_rounds):
        updates = elo_engine.run_round_robin(composites_map)
        _progress(f"  Round {round_num + 1}: {len(updates)} rating updates")

    elo_ratings: dict[str, tuple[float, float]] = {}
    elo_leaderboard = elo_engine.get_leaderboard()
    for er in elo_leaderboard:
        elo_ratings[er.system_name] = (er.rating, er.rd)
        _progress(f"  {er.system_name}: Elo={er.rating:.0f} (RD={er.rd:.0f})")

    # ── Step 4: Build scorecards & leaderboard ────────────────────────
    _progress("Step 4: Building scorecards and leaderboard...")
    benchmark_sc = _build_scorecards(report.systems, composite_scores, elo_ratings)

    leaderboard = Leaderboard(
        benchmark_name="pii-anon v1.0.0 — Composite Evaluation",
        systems=list(benchmark_sc.system_scorecards.values()),
    )
    leaderboard.sort_by_composite()

    # ── Step 5: Export results ────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _progress(f"Step 5: Exporting results to {out_path}/...")
    exported = LeaderboardExporter.export(
        leaderboard,
        out_path,
        formats=["json", "md", "csv"],
        filename_prefix="composite_leaderboard",
    )
    for fmt, path in exported.items():
        _progress(f"  Exported {fmt}: {path}")

    # Full results JSON
    full_results = {
        "report_schema_version": report.report_schema_version,
        "dataset": report.dataset,
        "systems": [],
        "composite_config": {
            "weight_detection_f1": config.weight_detection_f1,
            "weight_detection_precision": config.weight_detection_precision,
            "weight_detection_recall": config.weight_detection_recall,
            "weight_latency": config.weight_latency,
            "weight_throughput": config.weight_throughput,
        },
        "elo_rounds": elo_rounds,
    }

    for sys_result in report.systems:
        cs = composite_scores.get(sys_result.system)
        elo_r, elo_rd = elo_ratings.get(sys_result.system, (1500.0, 350.0))
        entry = {
            "system": sys_result.system,
            "available": sys_result.available,
            "precision": sys_result.precision,
            "recall": sys_result.recall,
            "f1": sys_result.f1,
            "latency_p50_ms": sys_result.latency_p50_ms,
            "docs_per_hour": sys_result.docs_per_hour,
            "per_entity_recall": sys_result.per_entity_recall,
            "composite_score": cs.score if cs else 0.0,
            "composite_detection_sub": cs.detection_sub if cs else 0.0,
            "composite_efficiency_sub": cs.efficiency_sub if cs else 0.0,
            "elo_rating": round(elo_r, 2),
            "elo_rd": round(elo_rd, 2),
            "samples": sys_result.samples,
        }
        full_results["systems"].append(entry)

    # Sort by composite score descending
    full_results["systems"].sort(key=lambda x: -x["composite_score"])

    results_path = out_path / "composite_evaluation_results.json"
    results_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    _progress(f"  Full results: {results_path}")

    # ── Step 6: Print summary ─────────────────────────────────────────
    _progress("")
    _progress("=" * 72)
    _progress("COMPOSITE EVALUATION SUMMARY")
    _progress("=" * 72)
    _progress("")
    _progress(f"{'System':<16} {'Composite':>10} {'F1':>8} {'Prec':>8} {'Recall':>8} "
              f"{'Lat(ms)':>10} {'Elo':>8}")
    _progress("-" * 72)
    for entry in full_results["systems"]:
        if not entry["available"]:
            _progress(f"{entry['system']:<16} {'(unavailable)':>10}")
            continue
        _progress(
            f"{entry['system']:<16} "
            f"{entry['composite_score']:>10.4f} "
            f"{entry['f1']:>8.4f} "
            f"{entry['precision']:>8.4f} "
            f"{entry['recall']:>8.4f} "
            f"{entry['latency_p50_ms']:>10.3f} "
            f"{entry['elo_rating']:>8.0f}"
        )
    _progress("-" * 72)

    # Determine winner
    available = [e for e in full_results["systems"] if e["available"]]
    if available:
        winner = available[0]
        _progress(f"\nWINNER: {winner['system']} "
                  f"(composite={winner['composite_score']:.4f}, "
                  f"Elo={winner['elo_rating']:.0f})")

        pii_anon = next((e for e in available if e["system"] == "pii-anon"), None)
        if pii_anon and pii_anon["system"] == winner["system"]:
            _progress("pii-anon is the TOP-RANKED system on the composite metric!")
        elif pii_anon:
            _progress(f"pii-anon composite: {pii_anon['composite_score']:.4f} "
                      f"(Elo: {pii_anon['elo_rating']:.0f})")

    _progress(f"\nResults saved to: {out_path}/")
    return full_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run composite metric evaluation for pii-anon v1.0.0"
    )
    parser.add_argument(
        "--dataset", default="eval_framework_v1",
        help="Benchmark dataset name (default: eval_framework_v1)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Warmup samples per system (default: 50)",
    )
    parser.add_argument(
        "--measured-runs", type=int, default=1,
        help="Number of measured benchmark runs (default: 1)",
    )
    parser.add_argument(
        "--output-dir", default="results/composite",
        help="Output directory for results (default: results/composite)",
    )
    parser.add_argument(
        "--elo-rounds", type=int, default=3,
        help="Number of Elo round-robin tournament rounds (default: 3)",
    )

    args = parser.parse_args()

    run_evaluation(
        dataset=args.dataset,
        max_samples=args.max_samples,
        warmup_samples=args.warmup,
        measured_runs=args.measured_runs,
        output_dir=args.output_dir,
        elo_rounds=args.elo_rounds,
    )


if __name__ == "__main__":
    main()
