"""SP1 per-change detection measurement harness (core-only, seconds-fast).

Scores the CORE detector (``_core_detector(use_case="default",
objective="balanced")`` — the same detector the canonical census measures)
against the DATA corpus with overlap-match semantics, per-entity P/R and
micro P/R/F2, plus p50 latency. Read-only on all eval-framework modules.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/sp1_detection_delta.py \
      --n 2000 --seed 8314 --out artifacts/sp1/baseline-n2000.json
  PYTHONPATH=src .venv/bin/python scripts/sp1_detection_delta.py \
      --n 2000 --seed 8314 --compare artifacts/sp1/baseline-n2000.json
  ... --dump-fp PERSON_NAME --dump-fn EMAIL_ADDRESS   # example inspection
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScoreStats:
    per_entity: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f2: float = 0.0
    latency_p50_ms: float = 0.0
    n_records: int = 0


def _f2(p: float, r: float) -> float:
    return 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0


def score_records(pairs) -> ScoreStats:
    """pairs: iterable of (record, found) where found is list of
    (record_id, entity_type, start, end) tuples (the LabelSpan shape)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    n = 0
    for rec, found in pairs:
        n += 1
        truth = [
            (t["entity_type"], t["start"], t["end"])
            for t in rec.labels
            if t["entity_type"] != "_BENCHMARK_IGNORE"
        ]
        spans = [(f[1], f[2], f[3]) for f in found]
        for fe, fs, fend in spans:
            if any(te == fe and not (fend <= ts or fs >= tend) for te, ts, tend in truth):
                counts[fe][0] += 1
            else:
                counts[fe][1] += 1
        for te, ts, tend in truth:
            if not any(fe == te and not (fend <= ts or fs >= tend) for fe, fs, fend in spans):
                counts[te][2] += 1
    tp = sum(v[0] for v in counts.values())
    fp = sum(v[1] for v in counts.values())
    fn = sum(v[2] for v in counts.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return ScoreStats(
        per_entity={k: tuple(v) for k, v in counts.items()},
        micro_precision=p,
        micro_recall=r,
        micro_f2=_f2(p, r),
        n_records=n,
    )


def run_measurement(n: int, seed: int) -> tuple[ScoreStats, list]:
    from pii_anon.benchmarks.datasets import load_benchmark_dataset
    from pii_anon.evaluation.competitor_compare import _core_detector

    records = load_benchmark_dataset("pii_anon", source="package-only")
    rng = random.Random(seed)
    sample = rng.sample(records, n)
    detect = _core_detector(use_case="default", objective="balanced")
    latencies: list[float] = []
    pairs = []
    for rec in sample:
        t0 = time.perf_counter()
        found = detect(rec)
        latencies.append((time.perf_counter() - t0) * 1000)
        pairs.append((rec, found))
    stats = score_records(pairs)
    latencies.sort()
    stats.latency_p50_ms = latencies[len(latencies) // 2]
    return stats, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=8314)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None)
    ap.add_argument("--dump-fp", type=str, default=None)
    ap.add_argument("--dump-fn", type=str, default=None)
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    stats, pairs = run_measurement(args.n, args.seed)
    print(
        f"n={stats.n_records} seed={args.seed} | micro P={stats.micro_precision:.4f} "
        f"R={stats.micro_recall:.4f} F2={stats.micro_f2:.4f} "
        f"p50={stats.latency_p50_ms:.3f}ms"
    )

    for flag, kind in ((args.dump_fp, "fp"), (args.dump_fn, "fn")):
        if not flag:
            continue
        shown = 0
        for rec, found in pairs:
            truth = [
                (t["entity_type"], t["start"], t["end"])
                for t in rec.labels
                if t["entity_type"] != "_BENCHMARK_IGNORE"
            ]
            spans = [(f[1], f[2], f[3]) for f in found]
            if kind == "fp":
                items = [
                    (fs, fend)
                    for fe, fs, fend in spans
                    if fe == flag
                    and not any(
                        te == fe and not (fend <= ts or fs >= tend) for te, ts, tend in truth
                    )
                ]
            else:
                items = [
                    (ts, tend)
                    for te, ts, tend in truth
                    if te == flag
                    and not any(
                        fe == te and not (fend <= ts or fs >= tend) for fe, fs, fend in spans
                    )
                ]
            for s, e in items:
                if shown >= args.examples:
                    break
                ctx = rec.text[max(0, s - 30) : min(len(rec.text), e + 15)].replace("\n", " ")
                print(f'  {kind.upper()} {flag} [{rec.language}] "{rec.text[s:e][:42]}" | …{ctx}…')
                shown += 1

    payload = {
        "n": stats.n_records,
        "seed": args.seed,
        "micro": {
            "precision": round(stats.micro_precision, 6),
            "recall": round(stats.micro_recall, 6),
            "f2": round(stats.micro_f2, 6),
        },
        "latency_p50_ms": round(stats.latency_p50_ms, 4),
        "per_entity": {k: list(v) for k, v in sorted(stats.per_entity.items())},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")
    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        bm, cm = base["micro"], payload["micro"]
        print(
            f"DELTA vs {args.compare}: "
            f"P {bm['precision']:.4f}->{cm['precision']:.4f} ({cm['precision'] - bm['precision']:+.4f}) "
            f"R {bm['recall']:.4f}->{cm['recall']:.4f} ({cm['recall'] - bm['recall']:+.4f}) "
            f"F2 {bm['f2']:.4f}->{cm['f2']:.4f} ({cm['f2'] - bm['f2']:+.4f}) "
            f"p50 {base['latency_p50_ms']:.3f}->{payload['latency_p50_ms']:.3f}ms"
        )
        bpe = {k: v for k, v in base["per_entity"].items()}
        for et in sorted(set(bpe) | set(payload["per_entity"])):
            b = bpe.get(et, [0, 0, 0])
            c = payload["per_entity"].get(et, [0, 0, 0])
            if b != c:
                print(f"  {et}: TP {b[0]}->{c[0]}  FP {b[1]}->{c[1]}  FN {b[2]}->{c[2]}")


if __name__ == "__main__":
    main()
