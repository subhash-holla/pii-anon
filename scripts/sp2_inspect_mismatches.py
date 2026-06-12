#!/usr/bin/env python3
"""sp2 mismatch inspector — show strict-match FP/FN examples for one entity type.

Runs a first-party predictor (vanilla by default) over the pii-anon-eval-data
dev split, projects native labels through the DATA adapter's LABEL_MAP, and
prints concrete false-positive / false-negative examples for ONE canonical
entity type under strict (start, end, entity_type) matching — the evidence a
detection fix starts from. DEV SPLIT ONLY (the test split is reserved for the
final reported run).

Usage (needs both pii_anon and pii_anon_datasets importable — miniforge env):

  python scripts/sp2_inspect_mismatches.py PERSON_NAME --limit 2000 --examples 15
  python scripts/sp2_inspect_mismatches.py TIMESTAMP --system pii_anon_swarm
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_type")
    parser.add_argument("--system", default="pii_anon", help="pii_anon | pii_anon_swarm")
    parser.add_argument("--limit", type=int, default=2000, help="dev records to scan")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--context", type=int, default=45, help="chars around span")
    args = parser.parse_args()

    # The DATA adapter owns the projection — reuse it verbatim (single source).
    repo_root = Path(__file__).resolve().parents[1].parent / "pii-anon-eval-data"
    sys.path.insert(0, str(repo_root))
    from baselines.pii_anon_baseline import ADAPTER as VANILLA  # noqa: E402
    from baselines.pii_anon_swarm_baseline import ADAPTER as SWARM  # noqa: E402
    from pii_anon_datasets import load_dataset  # noqa: E402

    adapter = {"pii_anon": VANILLA, "pii_anon_swarm": SWARM}[args.system]
    model = adapter.build()
    records = load_dataset(split="dev", language="en")[: args.limit]

    target = args.entity_type
    n_gold = n_tp = 0
    fps: list[tuple[str, int, int, str]] = []  # (text, start, end, snippet)
    fns: list[tuple[str, int, int, str]] = []
    fn_overlap_types = Counter()  # what we DID predict overlapping a missed gold

    def snippet(text: str, start: int, end: int) -> str:
        lo = max(0, start - args.context)
        hi = min(len(text), end + args.context)
        return (
            text[lo:start].replace("\n", "\\n")
            + "⟦" + text[start:end].replace("\n", "\\n") + "⟧"
            + text[end:hi].replace("\n", "\\n")
        )

    for rec in records:
        text = rec["text"]
        gold = {
            (a["start"], a["end"])
            for a in rec.get("annotations", [])
            if a.get("entity_type") == target
        }
        preds_all = adapter.detect(text, model)
        preds = {(s.start, s.end) for s in preds_all if s.entity_type == target}
        n_gold += len(gold)
        n_tp += len(gold & preds)
        for start, end in sorted(preds - gold):
            if len(fps) < 400:
                fps.append((text, start, end, snippet(text, start, end)))
        for start, end in sorted(gold - preds):
            if len(fns) < 400:
                fns.append((text, start, end, snippet(text, start, end)))
            # what did we predict that OVERLAPS this gold (boundary/type clue)?
            for s in preds_all:
                if s.start < end and start < s.end:
                    fn_overlap_types[
                        s.entity_type if (s.start, s.end) != (start, end) else "SAME-SPAN-OTHER-TYPE"
                    ] += 1

    n_fp, n_fn = len(fps), len(fns)
    print(f"{args.system} · {target} · {len(records)} dev records")
    print(f"gold {n_gold}  tp {n_tp}  fn {n_gold - n_tp}  fp(sampled) {n_fp}")
    if fn_overlap_types:
        print(f"FN overlap clues (prediction overlapping a missed gold): {dict(fn_overlap_types.most_common(8))}")
    print()
    print(f"=== FALSE NEGATIVES (gold missed) — first {args.examples} ===")
    for _text, start, end, snip in fns[: args.examples]:
        print(f"  [{start}:{end}] …{snip}…")
    print()
    print(f"=== FALSE POSITIVES (spurious/boundary) — first {args.examples} ===")
    for _text, start, end, snip in fps[: args.examples]:
        print(f"  [{start}:{end}] …{snip}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
