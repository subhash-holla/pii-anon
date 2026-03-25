#!/usr/bin/env python3
"""Per-profile and per-segment evaluation of pii-anon and pii-anon-swarm.

Since all 6 profiles use the same language set (yielding identical records),
this script also segments by difficulty, scenario, datatype, and language
to surface performance differences across real data dimensions.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from pii_anon.benchmarks.datasets import (
    BenchmarkRecord,
    load_benchmark_dataset,
    load_use_case_matrix,
)
from pii_anon.evaluation.competitor_compare import (
    _core_detector,
    _ensemble_detector,
    _normalize_entity_type,
    _deterministic_profile_records,
)

LabelSpan = tuple[str, str, int, int]

SAMPLE_PER_SEGMENT = 300  # records per segment to evaluate


def _labels_to_spans(record: BenchmarkRecord) -> list[LabelSpan]:
    spans: list[LabelSpan] = []
    for label in record.labels:
        entity_type = _normalize_entity_type(label["entity_type"])
        if entity_type == "_BENCHMARK_IGNORE":
            continue
        spans.append((record.record_id, entity_type, label["start"], label["end"]))
    return spans


def _compute_metrics(predictions: list[LabelSpan], ground_truth: list[LabelSpan]) -> dict:
    pred_set = set(predictions)
    gt_set = set(ground_truth)
    tp_all = len(pred_set & gt_set)
    fp_all = len(pred_set - gt_set)
    fn_all = len(gt_set - pred_set)
    p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    all_et = sorted({s[1] for s in pred_set} | {s[1] for s in gt_set})
    per_entity: dict[str, dict] = {}
    for et in all_et:
        et_pred = {s for s in pred_set if s[1] == et}
        et_gt = {s for s in gt_set if s[1] == et}
        tp = len(et_pred & et_gt)
        fp = len(et_pred - et_gt)
        fn = len(et_gt - et_pred)
        pe = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fe = 2 * pe * re / (pe + re) if (pe + re) > 0 else 0.0
        per_entity[et] = {"precision": pe, "recall": re, "f1": fe, "tp": tp, "fp": fp, "fn": fn}

    return {"overall": {"precision": p, "recall": r, "f1": f1, "tp": tp_all, "fp": fp_all, "fn": fn_all}, "per_entity": per_entity}


def _run_on_records(detect_fn, records: list[BenchmarkRecord]) -> tuple[list[LabelSpan], float]:
    t0 = time.time()
    preds: list[LabelSpan] = []
    for rec in records:
        preds.extend(detect_fn(rec))
    elapsed = time.time() - t0
    return preds, elapsed


def _print_table(name: str, metrics: dict):
    o = metrics["overall"]
    print(f"    {name}: P={o['precision']:.4f}  R={o['recall']:.4f}  F1={o['f1']:.4f}  (TP={o['tp']} FP={o['fp']} FN={o['fn']})")


def _print_entity_table(metrics: dict, show_only_weak=False):
    per = metrics["per_entity"]
    if show_only_weak:
        per = {et: m for et, m in per.items() if m["f1"] < 0.70 and (m["tp"] + m["fn"] > 0 or m["fp"] > 3)}
    if not per:
        print("      (all entity types F1 >= 0.70)")
        return
    print(f"      {'Entity Type':<28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print(f"      {'-'*28} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4}")
    for et, m in sorted(per.items(), key=lambda x: x[1]["f1"]):
        print(f"      {et:<28s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['tp']:4d} {m['fp']:4d} {m['fn']:4d}")


def _deterministic_sample(records: list[BenchmarkRecord], n: int) -> list[BenchmarkRecord]:
    """Take first n records after stable sort by record_id."""
    return sorted(records, key=lambda r: r.record_id)[:n]


def main():
    print("=" * 90)
    print("COMPREHENSIVE PER-SEGMENT EVALUATION: pii-anon & pii-anon-swarm")
    print("=" * 90)

    records = load_benchmark_dataset("pii_anon_benchmark_v1")
    print(f"\nDataset: {len(records)} records")

    # Build detectors
    print("Building detectors...")
    regex_detect = _core_detector(use_case="benchmark", objective="accuracy")
    ensemble_detect = _ensemble_detector(use_case="benchmark", allow_fallback_detectors=True)

    all_results = {}

    # =========================================================================
    # 1. OVERALL (full dataset sample)
    # =========================================================================
    print(f"\n{'='*90}")
    print("SECTION 1: OVERALL (1000 records, evenly sampled)")
    print(f"{'='*90}")
    overall_sample = _deterministic_sample(records, 1000)
    gt = []
    for r in overall_sample:
        gt.extend(_labels_to_spans(r))
    print(f"  Records: {len(overall_sample)}, Labels: {len(gt)}")

    regex_preds, regex_t = _run_on_records(regex_detect, overall_sample)
    regex_m = _compute_metrics(regex_preds, gt)
    _print_table("Regex", regex_m)

    ens_preds, ens_t = _run_on_records(ensemble_detect, overall_sample)
    ens_m = _compute_metrics(ens_preds, gt)
    _print_table("Ensemble", ens_m)

    print("\n  Weak entity types (F1 < 0.70) — Regex:")
    _print_entity_table(regex_m, show_only_weak=True)
    all_results["overall"] = {"regex": regex_m, "ensemble": ens_m, "records": len(overall_sample)}

    # =========================================================================
    # 2. BY DIFFICULTY LEVEL
    # =========================================================================
    print(f"\n{'='*90}")
    print("SECTION 2: BY DIFFICULTY LEVEL")
    print(f"{'='*90}")
    difficulty_groups: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for r in records:
        difficulty_groups[r.difficulty_level].append(r)

    for diff in sorted(difficulty_groups.keys()):
        group = difficulty_groups[diff]
        sample = _deterministic_sample(group, SAMPLE_PER_SEGMENT)
        gt_seg = []
        for r in sample:
            gt_seg.extend(_labels_to_spans(r))
        if not gt_seg:
            continue
        print(f"\n  [{diff.upper()}] — {len(sample)} records, {len(gt_seg)} labels")

        rp, _ = _run_on_records(regex_detect, sample)
        rm = _compute_metrics(rp, gt_seg)
        _print_table("Regex", rm)

        ep, _ = _run_on_records(ensemble_detect, sample)
        em = _compute_metrics(ep, gt_seg)
        _print_table("Ensemble", em)

        print("    Weak entities (Regex, F1<0.70):")
        _print_entity_table(rm, show_only_weak=True)
        all_results[f"difficulty_{diff}"] = {"regex": rm, "ensemble": em, "records": len(sample)}

    # =========================================================================
    # 3. BY SCENARIO TYPE (grouped for readability)
    # =========================================================================
    print(f"\n{'='*90}")
    print("SECTION 3: BY SCENARIO TYPE")
    print(f"{'='*90}")
    scenario_groups: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for r in records:
        scenario_groups[r.scenario_id].append(r)

    # Group smaller scenarios
    SCENARIO_MAP = {
        "baseline": "baseline",
        "context_loss": "context_loss",
        "context_preservation": "context_preservation",
        "entity_consistency": "entity_consistency",
        "continuity_tracking": "continuity",
        "continuity_ambiguous": "continuity",
        "temporal_consistency": "temporal_consistency",
        "format_xml": "format_structured",
        "format_json": "format_structured",
        "format_table": "format_structured",
        "format_csv": "format_structured",
        "edge_case_overlapping": "edge_cases",
        "edge_case_false_positive": "edge_cases",
        "edge_case_dense_pii": "edge_cases",
        "edge_case_pii_in_url": "edge_cases",
        "edge_case_code_embedded": "edge_cases",
        "edge_case_unicode": "edge_cases",
    }
    grouped_scenarios: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for scenario, recs in scenario_groups.items():
        group_name = SCENARIO_MAP.get(scenario, scenario)
        grouped_scenarios[group_name].extend(recs)

    for sname in sorted(grouped_scenarios.keys()):
        group = grouped_scenarios[sname]
        sample = _deterministic_sample(group, SAMPLE_PER_SEGMENT)
        gt_seg = []
        for r in sample:
            gt_seg.extend(_labels_to_spans(r))
        if not gt_seg:
            continue
        print(f"\n  [{sname}] — {len(sample)} records, {len(gt_seg)} labels")

        rp, _ = _run_on_records(regex_detect, sample)
        rm = _compute_metrics(rp, gt_seg)
        _print_table("Regex", rm)

        ep, _ = _run_on_records(ensemble_detect, sample)
        em = _compute_metrics(ep, gt_seg)
        _print_table("Ensemble", em)

        print("    Weak entities (Regex, F1<0.70):")
        _print_entity_table(rm, show_only_weak=True)
        all_results[f"scenario_{sname}"] = {"regex": rm, "ensemble": em, "records": len(sample)}

    # =========================================================================
    # 4. BY LANGUAGE GROUP
    # =========================================================================
    print(f"\n{'='*90}")
    print("SECTION 4: BY LANGUAGE GROUP")
    print(f"{'='*90}")
    lang_groups: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for r in records:
        lang_groups[r.language].append(r)

    LANG_BUCKETS = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it+pt+nl": "Other European",
        "ja+zh+ko": "CJK",
        "ar+hi": "Arabic+Hindi",
    }
    lang_bucket_records: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for lang, recs in lang_groups.items():
        if lang in ("it", "pt", "nl"):
            lang_bucket_records["it+pt+nl"].extend(recs)
        elif lang in ("ja", "zh", "ko"):
            lang_bucket_records["ja+zh+ko"].extend(recs)
        elif lang in ("ar", "hi"):
            lang_bucket_records["ar+hi"].extend(recs)
        else:
            lang_bucket_records[lang].extend(recs)

    for bucket in ["en", "es", "fr", "de", "it+pt+nl", "ja+zh+ko", "ar+hi"]:
        if bucket not in lang_bucket_records:
            continue
        group = lang_bucket_records[bucket]
        sample = _deterministic_sample(group, SAMPLE_PER_SEGMENT)
        gt_seg = []
        for r in sample:
            gt_seg.extend(_labels_to_spans(r))
        if not gt_seg:
            continue
        label = LANG_BUCKETS.get(bucket, bucket)
        print(f"\n  [{label}] — {len(sample)} records, {len(gt_seg)} labels")

        rp, _ = _run_on_records(regex_detect, sample)
        rm = _compute_metrics(rp, gt_seg)
        _print_table("Regex", rm)

        ep, _ = _run_on_records(ensemble_detect, sample)
        em = _compute_metrics(ep, gt_seg)
        _print_table("Ensemble", em)

        print("    Weak entities (Regex, F1<0.70):")
        _print_entity_table(rm, show_only_weak=True)
        all_results[f"lang_{bucket}"] = {"regex": rm, "ensemble": em, "records": len(sample)}

    # =========================================================================
    # 5. BY DATATYPE GROUP
    # =========================================================================
    print(f"\n{'='*90}")
    print("SECTION 5: BY DATATYPE GROUP")
    print(f"{'='*90}")
    dt_groups: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for r in records:
        dt_groups[r.datatype_group].append(r)

    for dtname in sorted(dt_groups.keys()):
        group = dt_groups[dtname]
        sample = _deterministic_sample(group, SAMPLE_PER_SEGMENT)
        gt_seg = []
        for r in sample:
            gt_seg.extend(_labels_to_spans(r))
        if not gt_seg:
            continue
        print(f"\n  [{dtname}] — {len(sample)} records, {len(gt_seg)} labels")

        rp, _ = _run_on_records(regex_detect, sample)
        rm = _compute_metrics(rp, gt_seg)
        _print_table("Regex", rm)

        ep, _ = _run_on_records(ensemble_detect, sample)
        em = _compute_metrics(ep, gt_seg)
        _print_table("Ensemble", em)

        print("    Weak entities (Regex, F1<0.70):")
        _print_entity_table(rm, show_only_weak=True)
        all_results[f"datatype_{dtname}"] = {"regex": rm, "ensemble": em, "records": len(sample)}

    # =========================================================================
    # GRAND SUMMARY
    # =========================================================================
    print(f"\n{'='*90}")
    print("GRAND SUMMARY")
    print(f"{'='*90}")
    print(f"{'Segment':<35s} {'Regex F1':>9s} {'Ens F1':>8s} {'Recs':>5s} {'Labels':>7s}")
    print(f"{'-'*35} {'-'*9} {'-'*8} {'-'*5} {'-'*7}")
    for seg, data in all_results.items():
        rf1 = data["regex"]["overall"]["f1"]
        ef1 = data["ensemble"]["overall"]["f1"]
        recs = data["records"]
        labels = data["regex"]["overall"]["tp"] + data["regex"]["overall"]["fn"]
        print(f"{seg:<35s} {rf1:9.4f} {ef1:8.4f} {recs:5d} {labels:7d}")

    # Collect all weak entities across segments
    print(f"\n{'='*90}")
    print("CROSS-SEGMENT WEAKNESS ANALYSIS")
    print(f"{'='*90}")
    weakness_counts: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for seg, data in all_results.items():
        for et, m in data["regex"]["per_entity"].items():
            if m["f1"] < 0.70 and (m["tp"] + m["fn"] > 0 or m["fp"] > 3):
                weakness_counts[et].append((seg, m["f1"]))

    print(f"\n  Entity types that are weak (F1<0.70) across multiple segments:")
    for et in sorted(weakness_counts.keys()):
        segs = weakness_counts[et]
        if len(segs) >= 3:  # weak in 3+ segments
            avg_f1 = sum(f for _, f in segs) / len(segs)
            print(f"    {et}: weak in {len(segs)} segments, avg F1={avg_f1:.4f}")
            for seg, f1 in sorted(segs, key=lambda x: x[1]):
                print(f"      {seg}: F1={f1:.4f}")

    # Save
    output_path = _ROOT / "pdlc-artifacts" / "development" / "profile-eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
