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

# Add src to path for development installs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Progress Tracker ────────────────────────────────────────────────────


class ProgressTracker:
    """High-fidelity progress indicator with 0.01% granularity.

    Tracks progress across the entire training pipeline as one unified bar.
    Estimates ETA based on observed throughput with exponential smoothing.
    """

    def __init__(self, total_records: int, label: str = "Training") -> None:
        self._total = total_records
        self._completed = 0
        self._label = label
        self._t_start = time.time()
        self._last_print = 0.0
        self._rate_ema = 0.0  # exponential moving average of rec/s
        self._phase = ""
        self._phase_start = 0.0
        self._phase_completed = 0
        self._phase_total = 0

    def set_phase(self, name: str, phase_total: int) -> None:
        """Start a new phase (e.g., 'Fold 1/3 train', 'Final retrain')."""
        self._phase = name
        self._phase_start = time.time()
        self._phase_completed = 0
        self._phase_total = phase_total

    def _format_eta(self, eta_s: float) -> str:
        if eta_s >= 3600:
            return f"{eta_s / 3600:.1f}h"
        if eta_s >= 60:
            return f"{eta_s / 60:.1f}m"
        return f"{eta_s:.0f}s"

    def _format_elapsed(self, s: float) -> str:
        if s >= 3600:
            return f"{s / 3600:.1f}h"
        if s >= 60:
            return f"{s / 60:.1f}m"
        return f"{s:.0f}s"

    def advance(self, n: int = 1) -> None:
        """Mark n records as completed. Updates the in-place progress bar."""
        self._completed += n
        self._phase_completed += n

        elapsed = time.time() - self._t_start
        if elapsed <= 0:
            return

        # Compute rate.
        rate = self._completed / elapsed
        if self._rate_ema <= 0:
            self._rate_ema = rate
        else:
            self._rate_ema = 0.95 * self._rate_ema + 0.05 * rate

        now = time.time()
        # Update the bar at most every 0.3s (avoids flickering).
        if now - self._last_print < 0.3:
            return
        self._last_print = now

        pct = self._completed / self._total * 100 if self._total > 0 else 0
        remaining = self._total - self._completed
        eta_s = remaining / self._rate_ema if self._rate_ema > 0 else 0

        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        # Single overwriting line.
        line = (
            f"\r  {bar} {pct:6.2f}% | {self._completed:,}/{self._total:,} | "
            f"{self._rate_ema:.0f} rec/s | ETA {self._format_eta(eta_s)} | "
            f"{self._phase}"
        )
        # Pad to clear previous longer lines.
        sys.stderr.write(line.ljust(120))
        sys.stderr.flush()

        # Log a summary line once per minute (on a new line so it persists in scrollback).
        if not hasattr(self, "_last_summary") or now - self._last_summary >= 60:
            self._last_summary = now
            sys.stderr.write("\n")
            logger.info(
                "[%5.2f%%] %s — %s elapsed, %s remaining, %.0f rec/s",
                pct, self._phase, self._format_elapsed(elapsed),
                self._format_eta(eta_s), self._rate_ema,
            )

    def finish_phase(self, message: str = "") -> None:
        """Mark current phase as complete. Prints a permanent summary line."""
        phase_elapsed = time.time() - self._phase_start
        pct = self._completed / self._total * 100 if self._total > 0 else 0
        sys.stderr.write("\n")
        sys.stderr.flush()
        if message:
            logger.info("[%5.2f%%] %s (%s)", pct, message, self._format_elapsed(phase_elapsed))

    def finish(self) -> None:
        """Mark the entire pipeline as complete."""
        elapsed = time.time() - self._t_start
        sys.stderr.write("\n")
        sys.stderr.flush()
        logger.info("Pipeline complete: %s records in %s (%.1f rec/s)",
                    f"{self._completed:,}", self._format_elapsed(elapsed),
                    self._completed / max(elapsed, 1))


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
            logger.info("  Dataset too small for multiprocessing (%d < %d); using sequential",
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
    logger.info("  Using %d worker processes (batch parallelism)", max_workers)
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

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading training data...")
    from pii_anon.swarm_datasets import load_training_data
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    records = load_training_data(datasets=dataset_names, max_records_per_dataset=max_records)
    if not records:
        logger.error("No training data loaded. Install pii-anon-datasets or check dataset names.")
        sys.exit(1)
    logger.info("Loaded %d records from %s", len(records), dataset_names)

    # ── Initialize engines ────────────────────────────────────────────
    logger.info("Initializing detection engines...")
    from pii_anon import PIIOrchestrator
    orch = PIIOrchestrator(token_key="swarm-training")
    all_engines = orch._async.registry.list_engines(include_disabled=True)
    for engine in all_engines:
        if engine.dependency_available() and not engine.enabled:
            engine.enabled = True
            logger.info("  Enabled engine: %s", engine.adapter_id)
    engines = [e for e in all_engines if e.enabled]
    engine_ids = [e.adapter_id for e in engines]
    logger.info("Parallel workers: %d", workers)
    logger.info("Active engines (%d): %s", len(engines), engine_ids)

    # ── Compute total work for progress tracking ──────────────────────
    # Each record is processed once per: K fold trains + K fold evals + 1 final retrain
    # Fold train size ≈ records * (k-1)/k, fold eval size ≈ records / k (capped at 500)
    if k > 1:
        fold_train_size = int(len(records) * (k - 1) / k)
        fold_eval_size = min(500, int(len(records) / k))
        total_work = k * (fold_train_size + fold_eval_size) + len(records)  # K folds + final retrain
    else:
        holdout_size = min(500, max(1, int(len(records) * 0.15)))
        total_work = len(records) + holdout_size  # retrain + holdout eval

    progress = ProgressTracker(total_work, label="Swarm Training")
    logger.info("Total work: %d record-passes (%d-fold CV + final retrain)", total_work, k)

    # ── K-fold cross-validation ───────────────────────────────────────
    fold_results: list[dict[str, float]] = []
    if k > 1:
        folds = _stratified_kfold_split(records, k=k, seed=args.seed)

        for fold_i, (fold_train, fold_test) in enumerate(folds):
            # Train on this fold.
            fold_annotations, fold_confidences = _run_engines_on_records(
                fold_train, engines, progress, f"Fold {fold_i + 1}/{k} train",
                max_workers=workers,
            )
            progress.finish_phase(
                f"Fold {fold_i + 1}/{k} train: {len(fold_train)} records, "
                f"{len(fold_annotations)} annotations"
            )

            fold_ds, fold_temp, fold_info = _train_artifacts(fold_annotations, fold_confidences)

            # Evaluate on this fold.
            f1, prec, rec = _evaluate_fold(
                fold_test, engines, fold_ds, fold_temp, fold_info, output_dir,
                progress, f"Fold {fold_i + 1}/{k} eval",
                max_workers=workers,
            )
            fold_results.append({"fold": fold_i + 1, "f1": f1, "precision": prec, "recall": rec})
            progress.finish_phase(f"Fold {fold_i + 1}/{k} eval: F1={f1:.4f} P={prec:.4f} R={rec:.4f}")

        mean_f1 = sum(r["f1"] for r in fold_results) / k
        std_f1 = math.sqrt(sum((r["f1"] - mean_f1) ** 2 for r in fold_results) / k)
        mean_p = sum(r["precision"] for r in fold_results) / k
        mean_r = sum(r["recall"] for r in fold_results) / k
        logger.info("CV summary: F1=%.4f +/- %.4f  P=%.4f  R=%.4f", mean_f1, std_f1, mean_p, mean_r)
    else:
        mean_f1 = mean_p = mean_r = std_f1 = 0.0

    # ── Final training on ALL data ────────────────────────────────────
    all_annotations, all_confidences = _run_engines_on_records(
        records, engines, progress, "Final retrain (all data)",
        max_workers=workers,
    )
    progress.finish_phase(f"Final retrain: {len(records)} records, {len(all_annotations)} annotations")

    ds, temp_scaler, info_scorer = _train_artifacts(all_annotations, all_confidences)
    ds.save(output_dir / "ds_params.json")
    temp_scaler.save(output_dir / "temperature.json")
    info_scorer.save(output_dir / "informativeness.json")

    # ── XGBoost ───────────────────────────────────────────────────────
    try:
        import xgboost  # noqa: F401
        logger.info("XGBoost available; meta-learner uses logistic fallback until trained on pipeline output.")
    except Exception as exc:
        logger.info("XGBoost not available (%s). Logistic fallback active.", type(exc).__name__)

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

    progress.finish()

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

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
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
