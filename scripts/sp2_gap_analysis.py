#!/usr/bin/env python3
"""sp2 gap analysis over a pii-anon-eval-data assessment artifact.

Reads a ``baseline_results.json`` (pii-anon-baseline-results/v1) and, for one
detector, ranks entity types by micro-F2 HEADROOM: how much the detector's
overall micro-F2 would rise if that type's false negatives all became true
positives (recall headroom) or its false positives vanished (precision
headroom). Micro F2 from pooled counts: ``F2 = 5*TP / (5*TP + 4*FN + FP)``.

Read-only, pure stdlib. Usage:

  python scripts/sp2_gap_analysis.py results/baselines/sp2-dev-iter0-vanilla/baseline_results.json pii_anon
  python scripts/sp2_gap_analysis.py <artifact> <detector> --top 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f2(tp: int, fp: int, fn: int) -> float:
    denom = 5 * tp + 4 * fn + fp
    return 5 * tp / denom if denom else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("detector")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    det = data["detectors"].get(args.detector)
    if det is None:
        raise SystemExit(
            f"unknown detector {args.detector!r}; have {sorted(data['detectors'])}"
        )

    by_entity = det["by_entity_type"]
    rows = []
    total_tp = total_fp = total_fn = 0
    for etype, row in by_entity.items():
        counts = row["counts"]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        rows.append((etype, tp, fp, fn, row.get("recall"), row.get("precision")))

    base = _f2(total_tp, total_fp, total_fn)
    print(f"detector: {args.detector}   micro-F2 (recomputed from counts): {base:.4f}")
    print(f"pooled counts: TP {total_tp}  FP {total_fp}  FN {total_fn}")
    print()

    scored = []
    for etype, tp, fp, fn, recall, precision in rows:
        if tp + fp + fn == 0:
            continue
        # Headroom: recall — all this type's FN become TP; precision — all
        # this type's FP vanish. Both vs the SAME baseline pooled counts.
        recall_gain = _f2(total_tp + fn, total_fp, total_fn - fn) - base
        precision_gain = _f2(total_tp, total_fp - fp, total_fn) - base
        scored.append((etype, tp, fp, fn, recall, precision, recall_gain, precision_gain))

    print(f"{'entity type':32s} {'gold':>6s} {'tp':>6s} {'fp':>6s} {'fn':>6s} "
          f"{'recall':>7s} {'prec':>7s} {'+F2 if recall=1':>16s} {'+F2 if fp=0':>12s}")
    key = lambda r: -(r[6] + r[7])  # noqa: E731 - combined headroom
    for etype, tp, fp, fn, recall, precision, rg, pg in sorted(scored, key=key)[: args.top]:
        rec = f"{recall:.3f}" if isinstance(recall, (int, float)) else "—"
        prec = f"{precision:.3f}" if isinstance(precision, (int, float)) else "—"
        print(f"{etype:32s} {tp + fn:>6d} {tp:>6d} {fp:>6d} {fn:>6d} "
              f"{rec:>7s} {prec:>7s} {rg:>+16.4f} {pg:>+12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
