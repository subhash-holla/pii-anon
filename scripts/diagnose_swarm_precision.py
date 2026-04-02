#!/usr/bin/env python3
"""Diagnose precision gap between regex-only and swarm (MoE) detection.

Runs both detectors on 100 benchmark records, compares predicted spans against
gold labels, and reports:
  1. Per-entity-type TP / FP / FN counts for both modes
  2. Which engines contributed to FP findings in the swarm
  3. Aggregate precision / recall / F1 for both modes

Usage:
    python scripts/diagnose_swarm_precision.py [--records 100]
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress noisy library output before any ML imports.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*truncate to max_length.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")

from pii_anon.benchmarks.datasets import load_benchmark_dataset, BenchmarkRecord
from pii_anon.evaluation.competitor_compare import (
    _build_engine_config,
    _normalize_entity_type,
    _normalize_findings,
    _OVERLAP_THRESHOLD,
)
from pii_anon.engines import RegexEngineAdapter
from pii_anon.types import EngineFinding, EnsembleFinding

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger(__name__)

LabelSpan = tuple[str, str, int, int]  # (record_id, entity_type, start, end)


# ---------------------------------------------------------------------------
# Extended span type that also tracks contributing engines
# ---------------------------------------------------------------------------
@dataclass
class EngineSpan:
    record_id: str
    entity_type: str
    start: int
    end: int
    engines: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Gold labels extraction
# ---------------------------------------------------------------------------
def _gold_labels(records: list[BenchmarkRecord]) -> list[LabelSpan]:
    labels: list[LabelSpan] = []
    for rec in records:
        for lab in rec.labels:
            etype = _normalize_entity_type(str(lab["entity_type"]))
            if etype == "_BENCHMARK_IGNORE":
                continue
            labels.append((rec.record_id, etype, int(lab["start"]), int(lab["end"])))
    return labels


# ---------------------------------------------------------------------------
# Regex-only detector
# ---------------------------------------------------------------------------
def build_regex_detector():
    engine = RegexEngineAdapter(enabled=True)

    def detect(record: BenchmarkRecord) -> list[LabelSpan]:
        try:
            raw = engine.detect(
                {"text": record.text},
                {"policy_mode": "balanced", "language": record.language or "en"},
            )
            return _normalize_findings(record, raw)
        except Exception:
            return []

    return detect


# ---------------------------------------------------------------------------
# Swarm (MoE) detector -- returns EngineSpan with engine attribution
# ---------------------------------------------------------------------------
def build_swarm_detector():
    from pii_anon.moe import MoEFusionStrategy, get_default_registry

    regex_engine = RegexEngineAdapter(enabled=True)
    default_language = "en"

    # Try to load competitor detectors
    from pii_anon.evaluation.competitor_compare import (
        _presidio_detector,
        _scrubadub_detector,
        _gliner_detector,
    )

    _ENGINE_ID_MAP = {
        "presidio": "presidio-compatible",
        "gliner": "gliner-compatible",
        "scrubadub": "scrubadub-compatible",
    }

    competitor_detectors: dict[str, Any] = {}
    for name, factory in [
        ("presidio", _presidio_detector),
        ("scrubadub", _scrubadub_detector),
        ("gliner", _gliner_detector),
    ]:
        detector, reason = factory(allow_fallback=True, require_native=False)
        if detector is not None:
            competitor_detectors[name] = detector
            print(f"  [swarm] loaded competitor: {name}")
        else:
            print(f"  [swarm] competitor unavailable: {name} ({reason})")

    moe_fusion = MoEFusionStrategy(
        registry=get_default_registry(),
        top_k=3,
        iou_threshold=0.5,
        performance_floor=True,
        min_expert_weight=0.15,
    )

    _HIGH_FP_SEMANTIC_TYPES = frozenset({
        "PERSON_NAME", "ORGANIZATION", "LOCATION", "ADDRESS",
        "DRIVERS_LICENSE", "PASSPORT", "NATIONAL_ID",
    })
    _REGEX_ENGINE_ID = "regex-oss"

    def detect(record: BenchmarkRecord) -> list[EngineSpan]:
        all_findings: list[EngineFinding] = []

        # 1) Regex findings
        try:
            raw_regex = regex_engine.detect(
                {"text": record.text},
                {"policy_mode": "balanced", "language": record.language or default_language},
            )
            for f in raw_regex:
                all_findings.append(EngineFinding(
                    entity_type=str(getattr(f, "entity_type", "UNKNOWN")),
                    confidence=float(getattr(f, "confidence", 0.8)),
                    field_path=getattr(f, "field_path", None),
                    span_start=getattr(f, "span_start", None),
                    span_end=getattr(f, "span_end", None),
                    explanation=getattr(f, "explanation", None),
                    engine_id="regex-oss",
                    language=record.language or default_language,
                ))
        except Exception:
            pass

        # 2) Competitor findings
        for comp_name, comp_detector in competitor_detectors.items():
            engine_id = _ENGINE_ID_MAP.get(comp_name, comp_name)
            try:
                spans = comp_detector(record)
                for _rid, entity_type, start, end in spans:
                    all_findings.append(EngineFinding(
                        entity_type=entity_type,
                        confidence=0.85,
                        field_path=None,
                        span_start=start,
                        span_end=end,
                        explanation=f"competitor:{comp_name}",
                        engine_id=engine_id,
                        language=record.language or default_language,
                    ))
            except Exception:
                pass

        if not all_findings:
            return []

        # 3) Fuse through MoE
        ensemble_findings = moe_fusion.merge(all_findings)

        # 4) Convert to EngineSpan with corroboration filter
        results: list[EngineSpan] = []
        for ef in ensemble_findings:
            if ef.span_start is None or ef.span_end is None:
                continue
            etype = _normalize_entity_type(str(ef.entity_type))
            if etype == "_BENCHMARK_IGNORE":
                continue
            # Corroboration check
            has_regex = _REGEX_ENGINE_ID in ef.engines
            if (
                not has_regex
                and etype in _HIGH_FP_SEMANTIC_TYPES
                and len(ef.engines) < 2
            ):
                continue
            results.append(EngineSpan(
                record_id=record.record_id,
                entity_type=etype,
                start=ef.span_start,
                end=ef.span_end,
                engines=list(ef.engines),
                confidence=ef.confidence,
            ))
        return results

    return detect


# ---------------------------------------------------------------------------
# Overlap matching (same logic as competitor_compare)
# ---------------------------------------------------------------------------
def _compute_tp_fp_fn(
    predictions: list[LabelSpan],
    gold: list[LabelSpan],
) -> tuple[int, int, int, set[int], set[int]]:
    """Returns (tp, fp, fn, matched_pred_indices, matched_gold_indices)."""
    # Build index by (record_id, entity_type)
    def _index(spans):
        idx: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
        for i, (rid, etype, s, e) in enumerate(spans):
            key = (rid, _normalize_entity_type(etype))
            idx.setdefault(key, []).append((s, e, i))
        return idx

    pred_idx = _index(predictions)
    gold_idx = _index(gold)

    matched_preds: set[int] = set()
    matched_golds: set[int] = set()
    tp = 0

    for key, gold_spans in gold_idx.items():
        pred_spans = pred_idx.get(key)
        if not pred_spans:
            continue

        candidates: list[tuple[float, int, int]] = []
        for gs, ge, gi in gold_spans:
            for ps, pe, pi in pred_spans:
                inter = max(0, min(ge, pe) - max(gs, ps))
                union = (ge - gs) + (pe - ps) - inter
                if union > 0:
                    iou = inter / union
                    if iou >= _OVERLAP_THRESHOLD:
                        candidates.append((iou, gi, pi))

        candidates.sort(key=lambda c: c[0], reverse=True)
        for _iou, gi, pi in candidates:
            if gi not in matched_golds and pi not in matched_preds:
                matched_golds.add(gi)
                matched_preds.add(pi)
                tp += 1

    fp = len(predictions) - len(matched_preds)
    fn = len(gold) - len(matched_golds)
    return tp, fp, fn, matched_preds, matched_golds


# ---------------------------------------------------------------------------
# Per-entity-type breakdown
# ---------------------------------------------------------------------------
def _per_entity_breakdown(
    predictions: list[LabelSpan],
    gold: list[LabelSpan],
) -> dict[str, dict[str, int]]:
    """Returns {entity_type: {"tp": N, "fp": N, "fn": N}}."""
    _tp, _fp, _fn, matched_preds, matched_golds = _compute_tp_fp_fn(predictions, gold)

    breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # TPs and FPs from predictions
    for i, (rid, etype, s, e) in enumerate(predictions):
        if i in matched_preds:
            breakdown[etype]["tp"] += 1
        else:
            breakdown[etype]["fp"] += 1

    # FNs from gold
    for i, (rid, etype, s, e) in enumerate(gold):
        if i not in matched_golds:
            breakdown[etype]["fn"] += 1

    return dict(breakdown)


# ---------------------------------------------------------------------------
# Engine attribution for FPs in swarm
# ---------------------------------------------------------------------------
def _engine_fp_attribution(
    swarm_results: list[EngineSpan],
    gold: list[LabelSpan],
) -> dict[str, dict[str, int]]:
    """Returns {engine_id: {entity_type: fp_count}} for FP findings in swarm."""
    # Convert EngineSpan to LabelSpan for matching
    predictions = [
        (es.record_id, es.entity_type, es.start, es.end)
        for es in swarm_results
    ]
    _tp, _fp, _fn, matched_preds, _matched_golds = _compute_tp_fp_fn(predictions, gold)

    # For each unmatched prediction (FP), attribute to its engines
    engine_fps: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, es in enumerate(swarm_results):
        if i not in matched_preds:
            for eng in es.engines:
                engine_fps[eng][es.entity_type] += 1

    return {k: dict(v) for k, v in engine_fps.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose swarm precision gap")
    parser.add_argument("--records", type=int, default=100, help="Number of records to sample")
    args = parser.parse_args()

    n_records = args.records

    print(f"Loading benchmark dataset (first {n_records} records)...")
    all_records = load_benchmark_dataset()
    records = all_records[:n_records]
    print(f"  Loaded {len(records)} records (of {len(all_records)} total)")

    gold = _gold_labels(records)
    print(f"  Gold labels: {len(gold)}")

    # ---- Regex-only ----
    print("\n=== Regex-Only Detection ===")
    regex_detect = build_regex_detector()
    t0 = time.time()
    regex_preds: list[LabelSpan] = []
    for rec in records:
        regex_preds.extend(regex_detect(rec))
    regex_time = time.time() - t0
    print(f"  Predictions: {len(regex_preds)} ({regex_time:.2f}s)")

    # ---- Swarm (MoE) ----
    print("\n=== Swarm (MoE) Detection ===")
    swarm_detect = build_swarm_detector()
    t0 = time.time()
    swarm_engine_results: list[EngineSpan] = []
    for rec in records:
        swarm_engine_results.extend(swarm_detect(rec))
    swarm_time = time.time() - t0
    swarm_preds: list[LabelSpan] = [
        (es.record_id, es.entity_type, es.start, es.end)
        for es in swarm_engine_results
    ]
    print(f"  Predictions: {len(swarm_preds)} ({swarm_time:.2f}s)")

    # ---- Aggregate metrics ----
    regex_tp, regex_fp, regex_fn, _, _ = _compute_tp_fp_fn(regex_preds, gold)
    swarm_tp, swarm_fp, swarm_fn, _, _ = _compute_tp_fp_fn(swarm_preds, gold)

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    regex_p, regex_r, regex_f1 = _prf(regex_tp, regex_fp, regex_fn)
    swarm_p, swarm_r, swarm_f1 = _prf(swarm_tp, swarm_fp, swarm_fn)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"{'Metric':<20} {'Regex-Only':>12} {'Swarm (MoE)':>12} {'Delta':>10}")
    print("-" * 54)
    print(f"{'TP':<20} {regex_tp:>12} {swarm_tp:>12} {swarm_tp - regex_tp:>+10}")
    print(f"{'FP':<20} {regex_fp:>12} {swarm_fp:>12} {swarm_fp - regex_fp:>+10}")
    print(f"{'FN':<20} {regex_fn:>12} {swarm_fn:>12} {swarm_fn - regex_fn:>+10}")
    print(f"{'Precision':<20} {regex_p:>12.4f} {swarm_p:>12.4f} {swarm_p - regex_p:>+10.4f}")
    print(f"{'Recall':<20} {regex_r:>12.4f} {swarm_r:>12.4f} {swarm_r - regex_r:>+10.4f}")
    print(f"{'F1':<20} {regex_f1:>12.4f} {swarm_f1:>12.4f} {swarm_f1 - regex_f1:>+10.4f}")

    # ---- Per-entity-type breakdown ----
    regex_breakdown = _per_entity_breakdown(regex_preds, gold)
    swarm_breakdown = _per_entity_breakdown(swarm_preds, gold)

    all_entity_types = sorted(set(list(regex_breakdown.keys()) + list(swarm_breakdown.keys())))

    print("\n" + "=" * 70)
    print("PER-ENTITY-TYPE BREAKDOWN (sorted by swarm FP count desc)")
    print("=" * 70)

    # Sort by swarm FP descending
    entity_by_swarm_fp = sorted(
        all_entity_types,
        key=lambda et: swarm_breakdown.get(et, {}).get("fp", 0),
        reverse=True,
    )

    print(f"\n{'Entity Type':<28} {'--- Regex ---':>16} {'--- Swarm ---':>16} {'FP Delta':>10}")
    print(f"{'':28} {'TP/FP/FN':>16} {'TP/FP/FN':>16}")
    print("-" * 72)
    for et in entity_by_swarm_fp:
        rb = regex_breakdown.get(et, {"tp": 0, "fp": 0, "fn": 0})
        sb = swarm_breakdown.get(et, {"tp": 0, "fp": 0, "fn": 0})
        r_str = f"{rb['tp']}/{rb['fp']}/{rb['fn']}"
        s_str = f"{sb['tp']}/{sb['fp']}/{sb['fn']}"
        fp_delta = sb["fp"] - rb["fp"]
        marker = " <<<" if fp_delta > 2 else ""
        print(f"{et:<28} {r_str:>16} {s_str:>16} {fp_delta:>+10}{marker}")

    # ---- Engine FP attribution ----
    engine_fps = _engine_fp_attribution(swarm_engine_results, gold)

    print("\n" + "=" * 70)
    print("ENGINE FP ATTRIBUTION (which engines caused FPs in swarm)")
    print("=" * 70)

    for engine_id in sorted(engine_fps.keys()):
        entity_counts = engine_fps[engine_id]
        total = sum(entity_counts.values())
        print(f"\n  {engine_id} (total FPs contributed to: {total})")
        for et in sorted(entity_counts.keys(), key=lambda k: entity_counts[k], reverse=True):
            print(f"    {et:<28} {entity_counts[et]:>4}")

    # ---- Top FP examples ----
    print("\n" + "=" * 70)
    print("TOP FP EXAMPLES (swarm FPs not in regex, with text context)")
    print("=" * 70)

    _, _, _, regex_matched_preds, _ = _compute_tp_fp_fn(regex_preds, gold)
    _, _, _, swarm_matched_preds, _ = _compute_tp_fp_fn(swarm_preds, gold)

    # Build a lookup for records by ID
    record_map = {r.record_id: r for r in records}

    # Collect swarm FPs with engine info
    swarm_fp_examples: list[tuple[str, str, int, int, list[str], str]] = []
    for i, es in enumerate(swarm_engine_results):
        if i not in swarm_matched_preds:
            text = record_map.get(es.record_id, BenchmarkRecord("?", "", [])).text
            ctx_start = max(0, es.start - 20)
            ctx_end = min(len(text), es.end + 20)
            context = text[ctx_start:ctx_end]
            swarm_fp_examples.append((
                es.record_id, es.entity_type, es.start, es.end,
                es.engines, context,
            ))

    # Show up to 30 FP examples
    for rid, etype, start, end, engines, context in swarm_fp_examples[:30]:
        print(f"  [{rid}] {etype} [{start}:{end}] engines={engines}")
        print(f"    context: ...{context}...")
        print()

    # ---- Generate output dict for markdown report ----
    output = {
        "n_records": n_records,
        "n_gold_labels": len(gold),
        "regex": {
            "tp": regex_tp, "fp": regex_fp, "fn": regex_fn,
            "precision": round(regex_p, 4),
            "recall": round(regex_r, 4),
            "f1": round(regex_f1, 4),
        },
        "swarm": {
            "tp": swarm_tp, "fp": swarm_fp, "fn": swarm_fn,
            "precision": round(swarm_p, 4),
            "recall": round(swarm_r, 4),
            "f1": round(swarm_f1, 4),
        },
        "per_entity_regex": {et: regex_breakdown.get(et, {"tp": 0, "fp": 0, "fn": 0}) for et in all_entity_types},
        "per_entity_swarm": {et: swarm_breakdown.get(et, {"tp": 0, "fp": 0, "fn": 0}) for et in all_entity_types},
        "engine_fp_attribution": {k: dict(v) for k, v in engine_fps.items()},
        "fp_examples": [
            {
                "record_id": rid, "entity_type": etype,
                "start": start, "end": end,
                "engines": engines, "context": context,
            }
            for rid, etype, start, end, engines, context in swarm_fp_examples[:30]
        ],
    }

    return output


def generate_markdown(data: dict) -> str:
    lines = []
    lines.append("# Swarm Precision Diagnosis Report")
    lines.append("")
    lines.append(f"**Records analyzed**: {data['n_records']}")
    lines.append(f"**Gold labels**: {data['n_gold_labels']}")
    lines.append("")

    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Metric | Regex-Only | Swarm (MoE) | Delta |")
    lines.append("|--------|-----------|-------------|-------|")
    for metric in ["tp", "fp", "fn"]:
        rv = data["regex"][metric]
        sv = data["swarm"][metric]
        lines.append(f"| {metric.upper()} | {rv} | {sv} | {sv - rv:+d} |")
    for metric in ["precision", "recall", "f1"]:
        rv = data["regex"][metric]
        sv = data["swarm"][metric]
        lines.append(f"| {metric.capitalize()} | {rv:.4f} | {sv:.4f} | {sv - rv:+.4f} |")
    lines.append("")

    lines.append("## Per-Entity-Type Breakdown (sorted by swarm FP desc)")
    lines.append("")
    lines.append("| Entity Type | Regex TP/FP/FN | Swarm TP/FP/FN | FP Delta |")
    lines.append("|-------------|---------------|----------------|----------|")
    all_types = sorted(data["per_entity_swarm"].keys(),
                       key=lambda et: data["per_entity_swarm"].get(et, {}).get("fp", 0),
                       reverse=True)
    for et in all_types:
        rb = data["per_entity_regex"].get(et, {"tp": 0, "fp": 0, "fn": 0})
        sb = data["per_entity_swarm"].get(et, {"tp": 0, "fp": 0, "fn": 0})
        r_str = f"{rb['tp']}/{rb['fp']}/{rb['fn']}"
        s_str = f"{sb['tp']}/{sb['fp']}/{sb['fn']}"
        fp_delta = sb["fp"] - rb["fp"]
        lines.append(f"| {et} | {r_str} | {s_str} | {fp_delta:+d} |")
    lines.append("")

    lines.append("## Engine FP Attribution")
    lines.append("")
    lines.append("Which engines contributed to false positive findings in the swarm:")
    lines.append("")
    for engine_id, entity_counts in sorted(data["engine_fp_attribution"].items()):
        total = sum(entity_counts.values())
        lines.append(f"### {engine_id} ({total} FP contributions)")
        lines.append("")
        lines.append("| Entity Type | FP Count |")
        lines.append("|-------------|----------|")
        for et in sorted(entity_counts.keys(), key=lambda k: entity_counts[k], reverse=True):
            lines.append(f"| {et} | {entity_counts[et]} |")
        lines.append("")

    lines.append("## FP Examples (first 30)")
    lines.append("")
    for ex in data.get("fp_examples", [])[:30]:
        lines.append(f"- **[{ex['record_id']}]** `{ex['entity_type']}` [{ex['start']}:{ex['end']}]")
        lines.append(f"  - Engines: {', '.join(ex['engines'])}")
        lines.append(f"  - Context: `...{ex['context']}...`")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")

    # Compute which entity types have the most excess FPs
    excess_fps = []
    for et in all_types:
        rb = data["per_entity_regex"].get(et, {"tp": 0, "fp": 0, "fn": 0})
        sb = data["per_entity_swarm"].get(et, {"tp": 0, "fp": 0, "fn": 0})
        delta = sb["fp"] - rb["fp"]
        if delta > 0:
            excess_fps.append((et, delta, sb["fp"]))

    excess_fps.sort(key=lambda x: x[1], reverse=True)
    if excess_fps:
        lines.append("**Entity types with most excess FPs in swarm:**")
        lines.append("")
        for et, delta, total_fp in excess_fps[:10]:
            lines.append(f"- {et}: +{delta} FPs (total {total_fp})")
        lines.append("")

    # Compute which engines are most responsible
    engine_totals = {}
    for engine_id, entity_counts in data["engine_fp_attribution"].items():
        engine_totals[engine_id] = sum(entity_counts.values())
    engine_ranked = sorted(engine_totals.items(), key=lambda x: x[1], reverse=True)
    if engine_ranked:
        lines.append("**Engines most responsible for FPs:**")
        lines.append("")
        for eng, total in engine_ranked:
            lines.append(f"- {eng}: {total} FP contributions")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    data = main()
    if data is None:
        sys.exit(1)

    # Save markdown report
    report_dir = Path(__file__).resolve().parents[1] / "pdlc-artifacts" / "swarm" / "discovery"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "precision-diagnosis.md"
    md = generate_markdown(data)
    report_path.write_text(md, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Also save raw JSON for further analysis
    json_path = report_dir / "precision-diagnosis.json"
    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"JSON data saved to: {json_path}")
