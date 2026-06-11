"""SP1 per-change detection measurement harness (core-only, seconds-fast).

Scores the CORE detector (``_core_detector(use_case="default",
objective="balanced")`` — the same detector the canonical census measures)
against the DATA corpus with overlap-match semantics on CANONICAL entity
types, per-entity P/R and micro P/R/F2, plus p50 latency. Read-only on all
eval-framework modules.

Canonical-label semantics (mirrors ``competitor_compare.py``):

- BOTH sides are normalized through ``_normalize_entity_type``. The canonical
  eval normalizes ground-truth labels when pre-building label tuples
  (competitor_compare.py:1730-1741) and predictions inside its gt-type filter
  and overlap index. Comparing raw truth types against canonical detection
  types double-penalizes renamed families (e.g. truth ``IBAN`` vs detection
  ``BANK_ACCOUNT``: a real match recorded as 1 FP + 1 FN).
- Truth spans normalizing to ``_BENCHMARK_IGNORE`` are EXCLUDED from
  per_entity/micro counting here and surfaced as ``unmatchable_truth_spans``.
  The canonical eval KEEPS them: labels are normalized but never dropped
  (competitor_compare.py:1730-1741, 1759), so they stay in its micro recall
  denominator (``recall = tp / len(all_labels)``, line 1828) as permanent
  misses unless an ignore-mapped prediction happens to overlap one. The
  bucket is constant for a fixed (n, seed), so per-change DELTAS are
  unaffected either way.
- FP-universe rule: canonical drops predictions whose canonical type is
  absent from the run-wide truth type set BEFORE FP counting
  (competitor_compare.py:1802-1809, "noise predictions"). The harness
  replicates this using the SCORED (non-ignore) truth-type universe, which
  also drops ignore-mapped detections — consistent with excluding the
  ``_BENCHMARK_IGNORE`` bucket on both sides (e.g. ``SWIFT_BIC`` detections
  are dropped because ``SWIFT_BIC_CODE`` truth maps to ignore).

ACCEPTED divergences from canonical (intentional, pinned by exact-anchor
tests — do NOT change): any-character-overlap matching (canonical: IoU >=
0.5, competitor_compare.py:567) and non-exclusive matching (canonical:
greedy 1:1, ``_overlap_match`` at line 585).

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
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pii_anon.evaluation.competitor_compare import _core_detector, _normalize_entity_type

_IGNORE = "_BENCHMARK_IGNORE"

# (entity_type, start, end) — canonical type, record-local offsets.
Span = tuple[str, int, int]


@dataclass
class ScoreStats:
    per_entity: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f2: float = 0.0
    latency_p50_ms: float = 0.0
    n_records: int = 0
    unmatchable_truth_spans: int = 0


def _f2(p: float, r: float) -> float:
    return 5 * p * r / (4 * p + r) if (4 * p + r) > 0 else 0.0


def _overlaps(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Any-character overlap between [s1, e1) and [s2, e2).

    Touching spans (e1 == s2) do NOT overlap. ACCEPTED divergence from the
    canonical IoU >= 0.5 rule (competitor_compare.py:567).
    """
    return s1 < e2 and s2 < e1


def _truth_spans(rec) -> tuple[list[Span], int]:
    """Canonical-normalized truth spans + the unmatchable (ignore-mapped) count.

    Mirrors the canonical label pre-build (competitor_compare.py:1730-1741)
    but EXCLUDES ``_BENCHMARK_IGNORE``-mapped spans from the scored list,
    returning their count separately (see module docstring).
    """
    spans: list[Span] = []
    unmatchable = 0
    for lab in rec.labels:
        etype = _normalize_entity_type(str(lab.get("entity_type", "UNKNOWN")))
        if etype == _IGNORE:
            unmatchable += 1
        else:
            spans.append((etype, int(lab.get("start", 0)), int(lab.get("end", 0))))
    return spans, unmatchable


def _found_spans(found) -> list[Span]:
    """Canonical-normalized detection spans (LabelSpan -> (type, start, end))."""
    return [(_normalize_entity_type(str(f[1])), int(f[2]), int(f[3])) for f in found]


def score_records(pairs) -> ScoreStats:
    """pairs: iterable of (record, found) where found is list of
    (record_id, entity_type, start, end) tuples (the LabelSpan shape).

    Entity types on BOTH sides are canonicalized via ``_normalize_entity_type``.
    Detections whose canonical type is absent from the run-wide scored
    truth-type universe are dropped before FP counting (the canonical
    FP-universe rule, competitor_compare.py:1802-1809).
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    n = 0
    unmatchable = 0
    prepared: list[tuple[list[Span], list[Span]]] = []
    universe: set[str] = set()
    for rec, found in pairs:
        n += 1
        truth, ignored = _truth_spans(rec)
        unmatchable += ignored
        universe.update(te for te, _, _ in truth)
        prepared.append((truth, _found_spans(found)))
    for truth, detected in prepared:
        # FP-universe rule (competitor_compare.py:1802-1809): drop detections
        # whose canonical type has no truth anywhere in this run.
        spans = [(fe, fs, fend) for fe, fs, fend in detected if fe in universe]
        for fe, fs, fend in spans:
            if any(te == fe and _overlaps(fs, fend, ts, tend) for te, ts, tend in truth):
                counts[fe][0] += 1
            else:
                counts[fe][1] += 1
        for te, ts, tend in truth:
            if not any(fe == te and _overlaps(fs, fend, ts, tend) for fe, fs, fend in spans):
                counts[te][2] += 1
    tp = sum(v[0] for v in counts.values())
    fp = sum(v[1] for v in counts.values())
    fn = sum(v[2] for v in counts.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return ScoreStats(
        per_entity={k: (v[0], v[1], v[2]) for k, v in counts.items()},
        micro_precision=p,
        micro_recall=r,
        micro_f2=_f2(p, r),
        n_records=n,
        unmatchable_truth_spans=unmatchable,
    )


def run_measurement(n: int, seed: int) -> tuple[ScoreStats, list]:
    from pii_anon.benchmarks.datasets import load_benchmark_dataset

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


def build_payload(stats: ScoreStats, seed: int) -> dict:
    """The on-disk baseline contract (kept in sync by tests/test_sp1_harness.py)."""
    return {
        "n": stats.n_records,
        "seed": seed,
        "micro": {
            "precision": round(stats.micro_precision, 6),
            "recall": round(stats.micro_recall, 6),
            "f2": round(stats.micro_f2, 6),
        },
        "latency_p50_ms": round(stats.latency_p50_ms, 4),
        "per_entity": {k: list(v) for k, v in sorted(stats.per_entity.items())},
        "unmatchable_truth_spans": stats.unmatchable_truth_spans,
    }


def _dump_examples(pairs, flag: str, kind: str, max_examples: int) -> None:
    """Print FP/FN examples for one (canonicalized) entity type."""
    canon = _normalize_entity_type(flag)
    if canon == _IGNORE:
        print(
            f"  {kind.upper()} {flag} (unmatchable): maps to {_IGNORE} — counted in "
            f"unmatchable_truth_spans, never in per_entity"
        )
        return
    prepared = [(rec, _truth_spans(rec)[0], _found_spans(found)) for rec, found in pairs]
    universe = {te for _, truth, _ in prepared for te, _, _ in truth}
    if kind == "fp" and canon not in universe:
        print(
            f"  FP {flag}: dropped by the FP-universe rule "
            f"(no {canon} truth in this sample; competitor_compare.py:1802-1809)"
        )
        return
    shown = 0
    for rec, truth, detected in prepared:
        spans = [(fe, fs, fend) for fe, fs, fend in detected if fe in universe]
        if kind == "fp":
            items = [
                (fs, fend)
                for fe, fs, fend in spans
                if fe == canon
                and not any(te == fe and _overlaps(fs, fend, ts, tend) for te, ts, tend in truth)
            ]
        else:
            items = [
                (ts, tend)
                for te, ts, tend in truth
                if te == canon
                and not any(fe == te and _overlaps(fs, fend, ts, tend) for fe, fs, fend in spans)
            ]
        for s, e in items:
            if shown >= max_examples:
                break
            ctx = rec.text[max(0, s - 30) : min(len(rec.text), e + 15)].replace("\n", " ")
            print(f'  {kind.upper()} {flag} [{rec.language}] "{rec.text[s:e][:42]}" | …{ctx}…')
            shown += 1
        if shown >= max_examples:
            break  # cap reached — stop scanning records entirely


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
        f"p50={stats.latency_p50_ms:.3f}ms "
        f"unmatchable={stats.unmatchable_truth_spans}"
    )

    for flag, kind in ((args.dump_fp, "fp"), (args.dump_fn, "fn")):
        if flag:
            _dump_examples(pairs, flag, kind, args.examples)

    payload = build_payload(stats, args.seed)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")
    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        if base.get("n") != payload["n"] or base.get("seed") != payload["seed"]:
            print(
                f"ERROR: baseline {args.compare} is not congruent with this run: "
                f"baseline n={base.get('n')} seed={base.get('seed')} vs "
                f"run n={payload['n']} seed={payload['seed']}. "
                f"Deltas are only meaningful for an identical (n, seed) draw.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        bm, cm = base["micro"], payload["micro"]
        print(
            f"DELTA vs {args.compare}: "
            f"P {bm['precision']:.4f}->{cm['precision']:.4f} ({cm['precision'] - bm['precision']:+.4f}) "
            f"R {bm['recall']:.4f}->{cm['recall']:.4f} ({cm['recall'] - bm['recall']:+.4f}) "
            f"F2 {bm['f2']:.4f}->{cm['f2']:.4f} ({cm['f2'] - bm['f2']:+.4f}) "
            f"p50 {base['latency_p50_ms']:.3f}->{payload['latency_p50_ms']:.3f}ms"
        )
        bpe = base["per_entity"]
        for et in sorted(set(bpe) | set(payload["per_entity"])):
            b = bpe.get(et, [0, 0, 0])
            c = payload["per_entity"].get(et, [0, 0, 0])
            if b != c:
                print(f"  {et}: TP {b[0]}->{c[0]}  FP {b[1]}->{c[1]}  FN {b[2]}->{c[2]}")


if __name__ == "__main__":
    main()
