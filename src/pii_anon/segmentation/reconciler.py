from __future__ import annotations

from pii_anon.types import BoundaryReconciliationTrace, EnsembleFinding


class BoundaryReconciler:
    def reconcile(self, segment_findings: list[list[EnsembleFinding]], overlap_tokens: int) -> tuple[list[EnsembleFinding], BoundaryReconciliationTrace]:
        dedup: dict[tuple[str, str | None, int | None, int | None], EnsembleFinding] = {}
        merged_spans = 0
        for batch in segment_findings:
            for finding in batch:
                key = (finding.entity_type, finding.field_path, finding.span_start, finding.span_end)
                if key in dedup:
                    existing = dedup[key]
                    existing.engines = sorted(set(existing.engines + finding.engines))
                    existing.confidence = max(existing.confidence, finding.confidence)
                    merged_spans += 1
                else:
                    dedup[key] = finding

        merged = list(dedup.values())
        trace = BoundaryReconciliationTrace(
            segments_processed=len(segment_findings),
            overlap_tokens=overlap_tokens,
            merged_spans=merged_spans,
            deduped_findings=max(0, sum(len(x) for x in segment_findings) - len(merged)),
        )
        return merged, trace
