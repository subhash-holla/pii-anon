"""The paired/differential substrate (Approach-B core) + silver-mode adjudication.

The substrate holds, per record, every system's predicted spans + transformed
output + the reference (gold in labeled mode, silver in unlabeled mode). Raw text
lives here in memory only; ``__repr__`` elides it.

Silver-mode honesty (red-team SHOWSTOPPER): pii-anon helps build the silver
reference it would be graded against, so silver mode NEVER yields a publishable
head-to-head winner. Instead we report span-level agreement (Jaccard), a
chance-adjusted character-level Cohen's κ (with a proper negative universe), the
only-A / only-B / both disagreement breakdown, and a union-vs-intersection
sensitivity swing — all ADVISORY.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .claim_strength import MeasurementMode
from .spans import Span, _norm, match_counts
from .stats import Counts, pooled_f1


@dataclass
class RecordEval:
    """Per-record evaluation inputs. ``original_text`` is RAW — in-memory only."""

    record_id: str
    original_text: str
    language: str
    reference: tuple[Span, ...]
    pred_by_system: dict[str, tuple[Span, ...]]
    transform_by_system: dict[str, str | None]
    transform_error_by_system: dict[str, str | None] = field(default_factory=dict)
    group: str | None = None

    def __repr__(self) -> str:  # defensive: record_id may carry PII — never echo it
        return (
            f"RecordEval(len(text)={len(self.original_text)}, "
            f"systems={sorted(self.pred_by_system)}, n_ref={len(self.reference)})"
        )


@dataclass(frozen=True)
class Substrate:
    records: tuple[RecordEval, ...]
    systems: tuple[str, ...]
    mode: MeasurementMode
    reference_kind: str  # "gold" | "silver"

    def __repr__(self) -> str:
        return (
            f"Substrate(n={len(self.records)}, systems={list(self.systems)}, "
            f"mode={self.mode.value}, reference={self.reference_kind})"
        )

    def raw_corpus(self) -> list[tuple[str, str]]:
        """(label, raw_string) pairs for the PII-egress scan — PROVENANCE-COMPLETE:
        every raw string a record carries (text, language, group, record_id, and each
        reference span's entity_type), so PII placed in any field is covered, not just
        ``original_text``."""
        out: list[tuple[str, str]] = []
        for r in self.records:
            out.append((r.record_id, r.original_text))
            if r.language:
                out.append((f"{r.record_id}:lang", r.language))
            if r.group:
                out.append((f"{r.record_id}:group", r.group))
            out.append((f"{r.record_id}:id", r.record_id))
            for etype, _s, _e in r.reference:
                out.append((f"{r.record_id}:type", etype))
        return out


# ---------------------------------------------------------------------------
# Silver reference construction
# ---------------------------------------------------------------------------


def union_reference(pred_by_system: Mapping[str, Sequence[Span]]) -> tuple[Span, ...]:
    """Recall-favouring union of all systems' spans (so missed PII is not silently
    under-counted in the silver reference). De-duplicated, deterministic order."""
    seen: dict[tuple[str, int, int], Span] = {}
    for spans in pred_by_system.values():
        for s in spans:
            key = (s[0].casefold(), int(s[1]), int(s[2]))
            seen.setdefault(key, (s[0], int(s[1]), int(s[2])))
    return tuple(sorted(seen.values(), key=lambda s: (s[1], s[2], s[0])))


def intersection_reference(pred_by_system: Mapping[str, Sequence[Span]]) -> tuple[Span, ...]:
    """Spans every system agrees on (for the sensitivity analysis)."""
    keysets = [
        {(s[0].casefold(), int(s[1]), int(s[2])) for s in spans}
        for spans in pred_by_system.values()
    ]
    if not keysets:
        return ()
    common = set.intersection(*keysets) if len(keysets) > 1 else keysets[0]
    # recover original-cased tuples from the first system that has each key
    out: list[Span] = []
    for spans in pred_by_system.values():
        for s in spans:
            if (s[0].casefold(), int(s[1]), int(s[2])) in common:
                out.append((s[0], int(s[1]), int(s[2])))
    dedup = {(s[0].casefold(), s[1], s[2]): s for s in out}
    return tuple(sorted(dedup.values(), key=lambda s: (s[1], s[2], s[0])))


# ---------------------------------------------------------------------------
# Agreement / disagreement / sensitivity (silver advisory story)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport:
    system_a: str
    system_b: str
    span_jaccard: float | None    # |A∩B| / |A∪B| over pooled spans (None if no spans at all)
    char_kappa: float | None      # chance-adjusted char-level Cohen's κ (None if vacuous)
    only_a: int
    only_b: int
    both: int
    reference_robust: bool        # ordering stable under union vs intersection
    ordering_union: tuple[str, ...]
    ordering_intersection: tuple[str, ...]


def _char_coverage(text_len: int, spans: Sequence[Span]) -> list[bool]:
    cov = [False] * text_len
    for _t, start, end in spans:
        for i in range(max(0, start), min(text_len, end)):
            cov[i] = True
    return cov


def agreement_report(records: Sequence[RecordEval], sys_a: str, sys_b: str) -> AgreementReport:
    """Pairwise agreement between two systems across the corpus."""
    only_a = only_b = both = 0
    n11 = n10 = n01 = n00 = 0
    for rec in records:
        a = rec.pred_by_system.get(sys_a, ())
        b = rec.pred_by_system.get(sys_b, ())
        ka = _norm(a, type_sensitive=True)
        kb = _norm(b, type_sensitive=True)
        both += len(ka & kb)
        only_a += len(ka - kb)
        only_b += len(kb - ka)
        # char-level confusion (proper negative universe)
        n = len(rec.original_text)
        ca = _char_coverage(n, a)
        cb = _char_coverage(n, b)
        for i in range(n):
            if ca[i] and cb[i]:
                n11 += 1
            elif ca[i] and not cb[i]:
                n10 += 1
            elif cb[i] and not ca[i]:
                n01 += 1
            else:
                n00 += 1
    union = only_a + only_b + both
    # No spans found by EITHER system ⇒ agreement is undefined, not a vacuous "perfect 1.0".
    jaccard: float | None = (both / union) if union else None
    total = n11 + n10 + n01 + n00
    positives = n11 + n10 + n01
    kappa: float | None
    if total == 0 or positives == 0:
        kappa = None  # neither system flagged any character ⇒ κ is vacuous, not 1.0
    else:
        po = (n11 + n00) / total
        pa1 = (n11 + n10) / total
        pb1 = (n11 + n01) / total
        pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
        kappa = None if (1 - pe) == 0 else (po - pe) / (1 - pe)
    ordering_u = _ordering(records, sys_a, sys_b, "union")
    ordering_i = _ordering(records, sys_a, sys_b, "intersection")
    robust = ordering_u == ordering_i
    return AgreementReport(
        system_a=sys_a, system_b=sys_b, span_jaccard=jaccard, char_kappa=kappa,
        only_a=only_a, only_b=only_b, both=both,
        reference_robust=robust, ordering_union=ordering_u, ordering_intersection=ordering_i,
    )


def _ordering(records: Sequence[RecordEval], sys_a: str, sys_b: str, ref_kind: str) -> tuple[str, ...]:
    """Order the two systems by F1 against a union- or intersection-built reference."""
    a_counts: list[Counts] = []
    b_counts: list[Counts] = []
    for rec in records:
        preds = rec.pred_by_system
        ref = union_reference(preds) if ref_kind == "union" else intersection_reference(preds)
        a_counts.append(match_counts(preds.get(sys_a, ()), ref))
        b_counts.append(match_counts(preds.get(sys_b, ()), ref))
    fa, fb = pooled_f1(a_counts), pooled_f1(b_counts)
    if fa > fb:
        return (sys_a, sys_b)
    if fb > fa:
        return (sys_b, sys_a)
    return tuple(sorted((sys_a, sys_b)))
