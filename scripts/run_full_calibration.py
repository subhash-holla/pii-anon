#!/usr/bin/env python
"""Full MoE weight calibration against pii-anon-eval-data.

Runs each engine individually against the complete labeled dataset,
computes per-entity-type F1, and updates the MoE expert registry.
Prints progress updates every 60 seconds.

Slow neural engines (stanza, spacy, gliner) are given a configurable
subsample ceiling to keep total runtime practical.
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Engines below this throughput threshold (rec/s) are considered "slow"
# and will be capped at SLOW_ENGINE_SAMPLES records.
SLOW_ENGINE_IDS = {"stanza-ner-compatible", "spacy-ner-compatible", "gliner-compatible"}
SLOW_ENGINE_SAMPLES = 10_000  # 10K subsample for slow NER engines
MIN_ENTITY_SAMPLES = 10
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────
# 1. Setup
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  MoE Full Weight Calibration — pii-anon-eval-data")
print("=" * 70)
print()

t_global = time.time()

print("[setup] Loading pii-anon-eval-data …")
import pii_anon_datasets

all_records = pii_anon_datasets.load_dataset()
print(f"[setup] Loaded {len(all_records):,} records")

# Filter to records that have annotations with start/end spans
records = [r for r in all_records if r.get("annotations")]
print(f"[setup] {len(records):,} records have annotations")

# Prepare a stratified subsample for slow engines
rng = random.Random(RANDOM_SEED)
slow_records = rng.sample(records, min(SLOW_ENGINE_SAMPLES, len(records)))
print(f"[setup] Slow-engine subsample: {len(slow_records):,} records")

print("[setup] Initializing orchestrator and engines …")
from pii_anon.moe import get_default_registry, reset_default_registry
from pii_anon.orchestrator import PIIOrchestrator

reset_default_registry()
orch = PIIOrchestrator(token_key="calibration-key")
engine_registry = orch._async.registry

engines = list(engine_registry._engines.values())
for eng in engines:
    eng.enabled = True
print(f"[setup] Enabled {len(engines)} engines: {[e.adapter_id for e in engines]}")

expert_registry = get_default_registry()

# Show pre-calibration weights
print("\n[pre-calibration] Expert strengths:")
for spec in expert_registry.list_experts():
    top = sorted(spec.entity_strengths.items(), key=lambda x: -x[1])[:3]
    print(f"  {spec.expert_id}: {dict(top)}")

# ──────────────────────────────────────────────────────────────────────
# 2. Per-engine calibration with progress reporting
# ──────────────────────────────────────────────────────────────────────

engine_entity_f1: dict[str, dict[str, float]] = {}
sample_counts: dict[str, dict[str, int]] = {}
skipped_engines: list[str] = []

total_engines = len(engines)
for eng_idx, engine in enumerate(engines, 1):
    adapter_id = engine.adapter_id

    # Choose record set based on engine speed
    is_slow = adapter_id in SLOW_ENGINE_IDS
    engine_records = slow_records if is_slow else records
    tag = f"(subsample {len(engine_records):,})" if is_slow else f"(full {len(engine_records):,})"

    print(f"\n{'─' * 70}")
    print(f"[engine {eng_idx}/{total_engines}] Calibrating: {adapter_id} {tag}")
    print(f"{'─' * 70}")

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)

    t_engine = time.time()
    last_report = time.time()
    n_records = len(engine_records)
    errors = 0

    for i, record in enumerate(engine_records):
        # ── Progress report every 60s ──
        now = time.time()
        if now - last_report >= 60.0 or i == 0:
            elapsed_eng = now - t_engine
            rate = (i / elapsed_eng) if elapsed_eng > 0 and i > 0 else 0
            eta = ((n_records - i) / rate) if rate > 0 else 0
            elapsed_total = now - t_global
            print(
                f"  [{adapter_id}] record {i:>7,}/{n_records:,} "
                f"({100 * i / n_records:5.1f}%) | "
                f"{rate:,.0f} rec/s | "
                f"engine elapsed {elapsed_eng:,.0f}s | "
                f"ETA {eta:,.0f}s | "
                f"total elapsed {elapsed_total / 60:,.1f}min | "
                f"errors {errors}",
                flush=True,
            )
            last_report = now

        # ── Run engine detection ──
        text = record.get("text", "")
        lang = record.get("language", "en")
        context: dict[str, Any] = {"language": lang, "policy_mode": "balanced"}

        try:
            findings = engine.detect({"text": text}, context)
        except Exception:
            errors += 1
            continue

        # ── Build predicted spans ──
        pred_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for f in findings:
            if f.span_start is not None and f.span_end is not None:
                pred_by_type[f.entity_type].add((f.span_start, f.span_end))

        # ── Build gold spans from annotations ──
        gold_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for ann in record.get("annotations", []):
            etype = str(ann.get("entity_type", "UNKNOWN"))
            start = ann.get("start")
            end = ann.get("end")
            if start is not None and end is not None:
                gold_by_type[etype].add((int(start), int(end)))
                label_counts[etype] += 1

        # ── Compute TP/FP/FN ──
        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
        for etype in all_types:
            preds = pred_by_type.get(etype, set())
            golds = gold_by_type.get(etype, set())
            tp[etype] += len(preds & golds)
            fp[etype] += len(preds - golds)
            fn[etype] += len(golds - preds)

    # ── Compute F1 per entity type ──
    elapsed_eng = time.time() - t_engine
    entity_f1: dict[str, float] = {}
    for etype in sorted(set(tp.keys()) | set(fn.keys())):
        t_val, f_val, n_val = tp[etype], fp[etype], fn[etype]
        prec = t_val / max(1, t_val + f_val)
        rec = t_val / max(1, t_val + n_val)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        entity_f1[etype] = round(f1, 4)

    # Filter by min samples
    filtered_f1: dict[str, float] = {}
    for etype, f1 in entity_f1.items():
        if label_counts.get(etype, 0) >= MIN_ENTITY_SAMPLES:
            filtered_f1[etype] = f1

    engine_entity_f1[adapter_id] = filtered_f1
    sample_counts[adapter_id] = dict(label_counts)

    print(f"\n  [{adapter_id}] Done in {elapsed_eng:,.1f}s ({elapsed_eng / 60:,.1f}min) | errors={errors}")
    print(f"  [{adapter_id}] Entity types scored: {len(filtered_f1)}")
    top_f1 = sorted(filtered_f1.items(), key=lambda x: -x[1])[:5]
    for etype, f1 in top_f1:
        print(f"    {etype}: F1={f1:.4f} (n={label_counts.get(etype, 0):,})")
    if len(filtered_f1) > 5:
        bot_f1 = sorted(filtered_f1.items(), key=lambda x: x[1])[:3]
        for etype, f1 in bot_f1:
            print(f"    {etype}: F1={f1:.4f} (n={label_counts.get(etype, 0):,})")

# ──────────────────────────────────────────────────────────────────────
# 3. Save & apply
# ──────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("  Saving calibration results")
print(f"{'=' * 70}")

from pii_anon.calibration.store import CalibrationResult, CalibrationStore

result = CalibrationResult(
    schema_version="1.0",
    calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    dataset="pii_anon_eval_data_full",
    engine_entity_f1=engine_entity_f1,
    sample_counts=sample_counts,
    skipped_engines=skipped_engines,
    metadata={
        "source": "pii-anon-eval-data (full dataset)",
        "total_records": len(records),
        "slow_engine_subsample": SLOW_ENGINE_SAMPLES,
        "slow_engines": sorted(SLOW_ENGINE_IDS),
        "min_entity_samples": MIN_ENTITY_SAMPLES,
    },
)

store = CalibrationStore()
store.save(result)
print(f"[save] Written to {store.path}")

# Apply to registry
updated = store.apply_to_registry(expert_registry, min_samples=MIN_ENTITY_SAMPLES)
print(f"[apply] Updated {sum(len(v) for v in updated.values())} entity-type weights across {len(updated)} engines")

# ──────────────────────────────────────────────────────────────────────
# 4. Final summary
# ──────────────────────────────────────────────────────────────────────

total_elapsed = time.time() - t_global
print(f"\n{'=' * 70}")
print(f"  Calibration complete — {total_elapsed / 60:,.1f} minutes total")
print(f"{'=' * 70}")

print("\nFull F1 results:")
for engine_id in sorted(engine_entity_f1):
    f1_map = engine_entity_f1[engine_id]
    if not f1_map:
        print(f"\n  {engine_id}: (no entity types above threshold)")
        continue
    sorted_f1 = sorted(f1_map.items(), key=lambda x: -x[1])
    n_samples = "subsample" if engine_id in SLOW_ENGINE_IDS else "full"
    print(f"\n  {engine_id} [{n_samples}]:")
    for etype, f1 in sorted_f1:
        n = sample_counts.get(engine_id, {}).get(etype, 0)
        print(f"    {etype:30s} F1={f1:.4f}  (n={n:>7,})")

print("\nPost-calibration expert strengths (top 5 per engine):")
for spec in expert_registry.list_experts():
    top = sorted(spec.entity_strengths.items(), key=lambda x: -x[1])[:5]
    print(f"  {spec.expert_id}: {dict(top)}")

print(f"\nDone. Calibration JSON: {store.path}")
