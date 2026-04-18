#!/usr/bin/env python3
"""Train the pii-anon-swarm pipeline with stratified K-fold cross-validation.

End-to-end script that:
1. Loads training data from pii-anon-eval-data (and optionally external datasets)
2. Runs stratified K-fold cross-validation to estimate swarm performance
3. Retrains on ALL data to produce final deployed artifacts
4. Deploys artifacts to ~/.pii_anon/swarm/

Usage:
    python scripts/train_swarm.py
    python scripts/train_swarm.py --max-records 5000 --kfold 5
    python scripts/train_swarm.py --datasets pii_anon_eval,ai4privacy --kfold 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers.
for _noisy in (
    "presidio_analyzer", "presidio_anonymizer", "presidio-analyzer",
    "presidio_analyzer.nlp_engine", "presidio_analyzer.analyzer_engine",
    "presidio_analyzer.recognizer_registry",
    "stanza", "gliner", "transformers", "sentence_transformers",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Add src to path for development installs; _progress is a sibling script module.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src"))
sys.path.insert(0, str(_SCRIPTS_DIR))

from _progress import ProgressTracker, format_elapsed  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────


def _primary_entity_type(rec: Any) -> str:
    """Get the most common entity type in a record (for stratification)."""
    labels = getattr(rec, "labels", [])
    if not labels:
        return "_NONE"
    counts: dict[str, int] = {}
    for lbl in labels:
        etype = lbl.get("entity_type", "_NONE")
        counts[etype] = counts.get(etype, 0) + 1
    return max(counts, key=counts.__getitem__)


def _stratified_kfold_split(
    records: list[Any], k: int, seed: int,
) -> list[tuple[list[Any], list[Any]]]:
    """Split records into K stratified folds."""
    rng = random.Random(seed)
    strata: dict[str, list[Any]] = defaultdict(list)
    for rec in records:
        strata[_primary_entity_type(rec)].append(rec)

    fold_assignments: list[int] = [0] * len(records)
    rec_to_idx = {id(rec): i for i, rec in enumerate(records)}
    for stratum_records in strata.values():
        rng.shuffle(stratum_records)
        for j, rec in enumerate(stratum_records):
            fold_assignments[rec_to_idx[id(rec)]] = j % k

    folds: list[tuple[list[Any], list[Any]]] = []
    for fold_i in range(k):
        train = [rec for rec, fa in zip(records, fold_assignments) if fa != fold_i]
        test = [rec for rec, fa in zip(records, fold_assignments) if fa == fold_i]
        rng.shuffle(train)
        rng.shuffle(test)
        folds.append((train, test))
    return folds


def _process_single_record(
    rec: Any, engines: list[Any],
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Process one record through all engines. Thread-safe."""
    record_votes: dict[str, str] = {}
    record_confs: dict[str, list[float]] = defaultdict(list)
    for engine in engines:
        try:
            findings = engine.detect(
                {"text": rec.text},
                {"language": rec.language, "policy_mode": "balanced"},
            )
            for f in findings:
                record_confs[engine.adapter_id].append(f.confidence)
                if f.entity_type:
                    record_votes[engine.adapter_id] = f.entity_type
        except Exception:
            continue
    return record_votes, record_confs


def _process_batch(
    batch: list[tuple[str, str]], engine_ids_to_use: list[str],
) -> list[tuple[dict[str, str], dict[str, list[float]]]]:
    """Process a batch of (text, language) pairs through all engines.

    Runs in a worker process with its own engine instances.
    Uses only JSON-serializable types for safe cross-process transfer.
    """
    from pii_anon import PIIOrchestrator
    orch = PIIOrchestrator(token_key="swarm-worker")
    all_engines = orch._async.registry.list_engines(include_disabled=True)
    for engine in all_engines:
        if engine.dependency_available() and not engine.enabled:
            engine.enabled = True
    engines = [e for e in all_engines if e.enabled]

    results = []
    for text, language in batch:
        record_votes: dict[str, str] = {}
        record_confs: dict[str, list[float]] = defaultdict(list)
        for engine in engines:
            try:
                findings = engine.detect(
                    {"text": text}, {"language": language, "policy_mode": "balanced"},
                )
                for f in findings:
                    record_confs[engine.adapter_id].append(f.confidence)
                    if f.entity_type:
                        record_votes[engine.adapter_id] = f.entity_type
            except Exception:
                continue
        results.append((record_votes, dict(record_confs)))
    return results


def _run_engines_on_records(
    records: list[Any],
    engines: list[Any],
    progress: ProgressTracker,
    phase_name: str,
    max_workers: int = 4,
) -> tuple[list[dict[str, str]], dict[str, list[float]]]:
    """Run all engines on records with progress tracking.

    Uses multiprocessing (ProcessPoolExecutor) for true parallelism
    when workers > 1.  Each worker spawns its own engine instances
    to avoid serialization issues.  Data is passed as plain strings
    and dicts (JSON-safe types only).
    """
    annotations: list[dict[str, str]] = []
    engine_confidences: dict[str, list[float]] = defaultdict(list)
    progress.set_phase(phase_name, len(records))
    engine_ids = [e.adapter_id for e in engines]

    # Auto-select strategy: multiprocessing is only worth it for large datasets
    # because each worker must load all NER models (~10s startup).
    # Threshold: ~500 records per worker makes the startup cost < 5% of total.
    min_records_for_mp = max_workers * 500
    use_mp = max_workers > 1 and len(records) >= min_records_for_mp

    if not use_mp:
        if max_workers > 1 and len(records) < min_records_for_mp:
            logger.debug("  Dataset too small for multiprocessing (%d < %d); using sequential",
                         len(records), min_records_for_mp)
        for rec in records:
            votes, confs = _process_single_record(rec, engines)
            if votes:
                annotations.append(votes)
            for eid, cs in confs.items():
                engine_confidences[eid].extend(cs)
            progress.advance()
        return annotations, engine_confidences

    # Parallel path: split into batches for worker processes.
    logger.debug("  Using %d worker processes (batch parallelism)", max_workers)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    batch_size = max(50, len(records) // (max_workers * 4))
    batches: list[list[tuple[str, str]]] = []
    current_batch: list[tuple[str, str]] = []
    for rec in records:
        current_batch.append((rec.text, getattr(rec, "language", "en")))
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_process_batch, batch, engine_ids): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            for votes, confs in batch_results:
                if votes:
                    annotations.append(votes)
                for eid, cs in confs.items():
                    engine_confidences[eid].extend(cs)
                progress.advance()

    return annotations, engine_confidences


def _evaluate_fold(
    test_records: list[Any],
    engines: list[Any],
    ds: Any,
    temp_scaler: Any,
    info_scorer: Any,
    output_dir: Path,
    progress: ProgressTracker,
    phase_name: str,
    max_eval: int = 500,
    max_workers: int = 4,
) -> tuple[float, float, float]:
    """Evaluate swarm on test records with progress tracking."""
    from pii_anon.swarm import SwarmConfig, SwarmFusionStrategy
    from pii_anon.types import EngineFinding

    config = SwarmConfig(
        ds_params_path=str(output_dir / "ds_params.json") if (output_dir / "ds_params.json").exists() else None,
        calibration_path=str(output_dir) if (output_dir / "temperature.json").exists() else None,
    )
    swarm = SwarmFusionStrategy(
        config=config, ds_aggregator=ds,
        temperature_scaler=temp_scaler, informativeness_scorer=info_scorer,
    )

    eval_records = test_records[:max_eval]
    progress.set_phase(phase_name, len(eval_records))
    tp_total, fp_total, fn_total = 0, 0, 0

    for rec in eval_records:
        gold_spans = set()
        for lbl in rec.labels:
            gold_spans.add((lbl.get("entity_type", ""), lbl.get("start", 0), lbl.get("end", 0)))

        raw_findings: list[EngineFinding] = []
        for engine in engines:
            try:
                findings = engine.detect(
                    {"text": rec.text}, {"language": rec.language, "policy_mode": "balanced"},
                )
                raw_findings.extend(findings)
            except Exception:
                continue

        ensemble = swarm.merge(raw_findings)
        pred_spans = {(ef.entity_type, ef.span_start, ef.span_end) for ef in ensemble}
        tp_total += len(pred_spans & gold_spans)
        fp_total += len(pred_spans - gold_spans)
        fn_total += len(gold_spans - pred_spans)
        progress.advance()

    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return f1, precision, recall


def _train_artifacts(
    annotations: list[dict[str, str]],
    engine_confidences: dict[str, list[float]],
) -> tuple[Any, Any, Any]:
    """Train DS, temperature scaler, and informativeness scorer."""
    from pii_anon.swarm import DawidSkeneAggregator, InformativenessScorer, TemperatureScaler

    ds = DawidSkeneAggregator.train_em(annotations, max_iter=50)

    temperatures: dict[str, float] = {}
    for eid, confs in engine_confidences.items():
        if len(confs) < 10:
            temperatures[eid] = 1.0
            continue
        mean_c = sum(confs) / len(confs)
        variance = sum((c - mean_c) ** 2 for c in confs) / len(confs)
        temperatures[eid] = max(0.5, min(3.0, 1.0 / max(math.sqrt(variance) * 3, 0.33)))
    temp_scaler = TemperatureScaler(temperatures=temperatures)
    info_scorer = InformativenessScorer.from_engine_findings(dict(engine_confidences))

    return ds, temp_scaler, info_scorer


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pii-anon-swarm pipeline")
    parser.add_argument("--datasets", default="pii_anon_eval", help="Comma-separated dataset names")
    parser.add_argument("--max-records", type=int, default=10000,
                        help="Max records per dataset (0 = unlimited)")
    parser.add_argument("--kfold", type=int, default=5, help="Number of cross-validation folds (1 = no CV)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for engine execution (default 4, use 1 to disable)")
    parser.add_argument("--output", default=str(Path.home() / ".pii_anon" / "swarm"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    max_records = args.max_records if args.max_records > 0 else None
    k = max(1, args.kfold)
    workers = max(1, args.workers)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    t_start = time.time()

    # Silence info-level noise from pii_anon modules so the progress line
    # stays the only visible output during the run. Warnings/errors still
    # surface. This is restored in a `finally` below.
    pii_anon_logger = logging.getLogger("pii_anon")
    _prev_level = pii_anon_logger.level
    pii_anon_logger.setLevel(logging.WARNING)

    try:
        # ── Load data ─────────────────────────────────────────────────────
        t_load = time.time()
        from pii_anon.swarm_datasets import load_training_data
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        records = load_training_data(datasets=dataset_names, max_records_per_dataset=max_records)
        if not records:
            logger.error("No training data loaded. Install pii-anon-datasets or check dataset names.")
            sys.exit(1)
        load_elapsed = time.time() - t_load

        # ── Initialize engines ────────────────────────────────────────────
        t_init = time.time()
        from pii_anon import PIIOrchestrator
        orch = PIIOrchestrator(token_key="swarm-training")
        all_engines = orch._async.registry.list_engines(include_disabled=True)
        for engine in all_engines:
            if engine.dependency_available() and not engine.enabled:
                engine.enabled = True
        engines = [e for e in all_engines if e.enabled]
        engine_ids = [e.adapter_id for e in engines]
        init_elapsed = time.time() - t_init

        # ── Compute weighted total_work for 0.01% overall accuracy ────────
        # records_work: every record pass through engines (fold train + eval + final retrain)
        # load_work:    ~2% of total, credited up-front since load already finished
        # finalize_work: ~1% of total, advanced during artifact save
        if k > 1:
            fold_train_size = int(len(records) * (k - 1) / k)
            fold_eval_size = min(500, int(len(records) / k))
            records_work = k * (fold_train_size + fold_eval_size) + len(records)
        else:
            holdout_size = min(500, max(1, int(len(records) * 0.15)))
            records_work = len(records) + holdout_size

        load_work = max(100, len(records) // 50)
        finalize_work = max(100, records_work // 100)
        total_work = load_work + records_work + finalize_work

        # Pre-credit the load phase (it completed before the bar was built).
        progress = ProgressTracker(
            total_work, label="Swarm Training",
            refresh_s=60.0, initial_completed=load_work,
        )
        progress.phase_log.append(
            f"[{load_work / total_work * 100:6.2f}%] Loaded {len(records):,} records "
            f"from {dataset_names} ({format_elapsed(load_elapsed)})"
        )
        progress.phase_log.append(
            f"[{load_work / total_work * 100:6.2f}%] Initialized {len(engines)} engines: "
            f"{engine_ids} ({format_elapsed(init_elapsed)})"
        )
        progress.set_phase("Starting training", 0)

        # ── K-fold cross-validation ───────────────────────────────────────
        fold_results: list[dict[str, float]] = []
        if k > 1:
            folds = _stratified_kfold_split(records, k=k, seed=args.seed)

            for fold_i, (fold_train, fold_test) in enumerate(folds):
                fold_annotations, fold_confidences = _run_engines_on_records(
                    fold_train, engines, progress, f"Fold {fold_i + 1}/{k} train",
                    max_workers=workers,
                )
                progress.finish_phase(
                    f"Fold {fold_i + 1}/{k} train: {len(fold_train)} records, "
                    f"{len(fold_annotations)} annotations"
                )

                fold_ds, fold_temp, fold_info = _train_artifacts(fold_annotations, fold_confidences)

                f1, prec, rec = _evaluate_fold(
                    fold_test, engines, fold_ds, fold_temp, fold_info, output_dir,
                    progress, f"Fold {fold_i + 1}/{k} eval",
                    max_workers=workers,
                )
                fold_results.append({"fold": fold_i + 1, "f1": f1, "precision": prec, "recall": rec})
                progress.finish_phase(
                    f"Fold {fold_i + 1}/{k} eval: F1={f1:.4f} P={prec:.4f} R={rec:.4f}"
                )

            mean_f1 = sum(r["f1"] for r in fold_results) / k
            std_f1 = math.sqrt(sum((r["f1"] - mean_f1) ** 2 for r in fold_results) / k)
            mean_p = sum(r["precision"] for r in fold_results) / k
            mean_r = sum(r["recall"] for r in fold_results) / k
            progress.phase_log.append(
                f"[{progress._completed / progress._total * 100:6.2f}%] "
                f"CV summary: F1={mean_f1:.4f} +/- {std_f1:.4f}  P={mean_p:.4f}  R={mean_r:.4f}"
            )
        else:
            mean_f1 = mean_p = mean_r = std_f1 = 0.0

        # ── Final training on ALL data ────────────────────────────────────
        all_annotations, all_confidences = _run_engines_on_records(
            records, engines, progress, "Final retrain (all data)",
            max_workers=workers,
        )
        progress.finish_phase(
            f"Final retrain: {len(records)} records, {len(all_annotations)} annotations"
        )

        ds, temp_scaler, info_scorer = _train_artifacts(all_annotations, all_confidences)

        # ── Save artifacts ────────────────────────────────────────────────
        progress.set_phase("Saving artifacts", finalize_work)
        save_step = max(1, finalize_work // 5)

        ds.save(output_dir / "ds_params.json")
        progress.advance(save_step)
        temp_scaler.save(output_dir / "temperature.json")
        progress.advance(save_step)
        info_scorer.save(output_dir / "informativeness.json")
        progress.advance(save_step)

        try:
            import xgboost  # noqa: F401
            xgboost_note = "XGBoost available; logistic fallback until trained on pipeline output"
        except Exception as exc:
            xgboost_note = f"XGBoost not available ({type(exc).__name__}); logistic fallback active"

        # ── Final metrics ─────────────────────────────────────────────────
        if k > 1 and fold_results:
            final_f1, final_p, final_r = mean_f1, mean_p, mean_r
        else:
            random.shuffle(records)
            holdout_n = max(1, int(len(records) * 0.15))
            holdout = records[-holdout_n:]
            final_f1, final_p, final_r = _evaluate_fold(
                holdout, engines, ds, temp_scaler, info_scorer, output_dir,
                progress, "Holdout eval",
                max_workers=workers,
            )
            progress.finish_phase(f"Holdout eval: F1={final_f1:.4f}")

        # ── Save manifest ─────────────────────────────────────────────────
        manifest: dict[str, Any] = {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datasets_used": dataset_names,
            "total_records": len(records),
            "engines": engine_ids,
            "kfold": k,
            "cv_results": fold_results,
            "final_f1": round(final_f1, 4),
            "final_precision": round(final_p, 4),
            "final_recall": round(final_r, 4),
            "temperatures": dict(temp_scaler._temps),
            "seed": args.seed,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Close out finalize phase (reach 100.00% exactly).
        remaining_finalize = finalize_work - 3 * save_step
        if remaining_finalize > 0:
            progress.advance(remaining_finalize)
        progress.finish_phase(f"Saved artifacts to {output_dir}; {xgboost_note}")
        progress.finish()
    finally:
        pii_anon_logger.setLevel(_prev_level)

    # ── End-of-run output ─────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print("── Training Log " + "─" * 44)
    for entry in progress.phase_log:
        print(f"  {entry}")
    print("─" * 60)
    print()
    print("=" * 60)
    print("  pii-anon-swarm Training Complete")
    print("=" * 60)
    print(f"  Datasets:   {', '.join(dataset_names)}")
    print(f"  Records:    {len(records)}")
    if k > 1:
        print(f"  K-fold:     {k}-fold stratified cross-validation")
        print(f"  CV F1:      {mean_f1:.4f} +/- {std_f1:.4f}")
        for r in fold_results:
            print(f"    Fold {r['fold']}: F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")
    print(f"  Final F1:   {final_f1:.4f}")
    print(f"  Precision:  {final_p:.4f}")
    print(f"  Recall:     {final_r:.4f}")
    print(f"  Time:       {elapsed_total:.0f}s ({elapsed_total / 60:.1f}m)")
    print(f"  Artifacts:  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
