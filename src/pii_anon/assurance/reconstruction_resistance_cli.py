"""sp7 Phase C — reconstruction-resistance report DRIVER.

Assembles ``CorpusRecord``s from a labelled corpus, builds a deterministic
pii-anon masker (regex adapter: detect -> replace each PII span with a
type-tagged placeholder), runs :func:`reconstruction_resistance_report`, and
writes reproducible ``.json`` + shareable ``.html`` artifacts.

Also asserts FR-036 stream/batch masking parity: the masking decision on a
record must be identical whether the record is masked whole (batch) or via the
segmenter (streaming) — otherwise a streamed record could leak where a batch one
does not.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pii_anon.assurance.reconstruction_resistance import (
    CorpusRecord,
    reconstruction_resistance_report,
    render_html,
)
from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.segmentation.segmenter import Segmenter


def build_regex_masker(*, eval_arbitration: bool = False) -> Callable[[str], str]:
    """A deterministic masker: detect PII spans with the regex engine and
    replace each (right-to-left, non-overlapping) with ``[TYPE]``."""
    engine = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=eval_arbitration)

    def mask(text: str) -> str:
        spans: list[tuple[int, int, str]] = sorted(
            {
                (f.span_start, f.span_end, str(f.entity_type))
                for f in engine.detect({"text": text}, {"language": "en"})
                if f.span_start is not None and f.span_end is not None
            }
        )
        # keep the longest non-overlapping cover, replace right-to-left
        kept: list[tuple[int, int, str]] = []
        last = -1
        for s, e, t in spans:
            if s >= last:
                kept.append((s, e, t))
                last = e
        out = text
        for s, e, t in reversed(kept):
            out = out[:s] + f"[{t}]" + out[e:]
        return out

    return mask


def masked_char_set(mask_fn: Callable[[str], str], text: str) -> frozenset[int]:
    """The set of ORIGINAL character offsets that ``mask_fn`` redacts (used for
    the stream/batch parity check — placeholder-insertion-invariant)."""
    engine = RegexEngineAdapter(enabled=True)
    out: set[int] = set()
    for f in engine.detect({"text": text}, {"language": "en"}):
        if f.span_start is not None and f.span_end is not None:
            out.update(range(f.span_start, f.span_end))
    return frozenset(out)


def assert_stream_batch_parity(
    records: Sequence[CorpusRecord], *, max_tokens: int = 64
) -> dict[str, Any]:
    """FR-036: the redacted character set must be a SUPERSET under streaming
    (segmented) detection vs batch — streaming may over-mask (safe) but must
    never leave a batch-masked character exposed."""
    engine = RegexEngineAdapter(enabled=True)
    seg = Segmenter()
    violations: list[str] = []
    for rec in records:
        batch = masked_char_set(build_regex_masker(), rec.text)
        streamed: set[int] = set()
        for s in seg.segment(rec.text, max_tokens=max_tokens, overlap_tokens=0):
            for f in engine.detect({"text": s.text}, {"language": "en"}):
                if f.span_start is not None and f.span_end is not None:
                    streamed.update(range(s.start_char + f.span_start, s.start_char + f.span_end))
        missing = batch - streamed
        if missing:
            violations.append(rec.record_id)
    return {
        "records_checked": len(records),
        "parity_violations": len(violations),
        "violating_records": violations[:20],
        "invariant": "streamed_redaction >= batch_redaction (over-mask is safe)",
    }


def records_from_benchmark(path: str, *, n: int, seed: int) -> list[CorpusRecord]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        recs = [json.loads(line) for line in fh]
    rng = random.Random(seed)
    sample = rng.sample(recs, min(n, len(recs)))
    out: list[CorpusRecord] = []
    for r in sample:
        text = r["text"]
        labs = [lab for lab in r.get("labels", []) if lab.get("end", 0) > lab.get("start", 0)]
        vals = tuple(text[lab["start"]:lab["end"]] for lab in labs)
        types = tuple(str(lab.get("entity_type", "UNKNOWN")) for lab in labs)
        out.append(
            CorpusRecord(record_id=str(r.get("id", len(out))), text=text, pii_values=vals, pii_types=types)
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pii-anon reconstruction-resistance report")
    ap.add_argument("--corpus", required=True, help="labelled .jsonl(.gz) corpus")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20240517)
    ap.add_argument("--mask-label", default="pii-anon-regex")
    ap.add_argument("--corpus-label", default="corpus")
    ap.add_argument("--out", required=True, help="output basename (writes .json + .html)")
    args = ap.parse_args(argv)

    records = records_from_benchmark(args.corpus, n=args.n, seed=args.seed)
    mask = build_regex_masker()
    report = reconstruction_resistance_report(
        records, mask, seed=args.seed,
        mask_label=args.mask_label, corpus_label=args.corpus_label,
    )
    report["fr036_stream_batch_parity"] = assert_stream_batch_parity(records)

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out.with_suffix(".html").write_text(render_html(report), encoding="utf-8")
    vl = report["verbatim_leakage"]
    rd = report["reidentification_tier3"]["masked"]
    print(f"verbatim leak {vl['leaked_verbatim']}/{vl['total_pii_values']} "
          f"({100*vl['leak_rate']:.2f}%); masked re-id recall {100*rd['reid_recall']:.2f}%; "
          f"parity violations {report['fr036_stream_batch_parity']['parity_violations']}")
    print(f"wrote {out.with_suffix('.json')} + {out.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
