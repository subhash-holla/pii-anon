"""Residual-leakage assessor — a fail-closed LOWER BOUND.

For each referenced (gold/silver) PII span, check whether its surface text
survives verbatim in the pipeline's transformed output — independent of the
pipeline's own detector recall (red-team: measure against the reference's surface
text, not by re-detecting). Lower is better.

Fail CLOSED toward MORE leakage / honest abstention:
* no transform / transform failure / non-str output ⇒ that record is unscoreable;
* an empty (or whitespace-only) output is degenerate ⇒ unscoreable (NOT 0%);
* if too few records are scoreable ⇒ the whole dimension is NOT_ASSESSABLE;
* a no-op transform (output == input) ⇒ ~100% leakage by construction.

Specificity guard: only count a referenced surface of length >= ``min_specificity_len``
(avoids common-substring false leaks). The value is ALWAYS labelled a lower bound.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import replace

from ..adjudication import Substrate
from ..claim_strength import DimensionResult, MeasurementMode, PowerThresholds, not_assessable, resolve
from ..stats import Counts, cluster_bootstrap_ci

DIMENSION = "leakage"

_LOWER_BOUND_CAVEAT = (
    "residual-leakage is a LOWER BOUND: counts verbatim OR separator-normalized survival of "
    "DISTINCTIVE referenced entities only — emails, numeric IDs (>=3 digits), and names that "
    "are multi-word or contain an uppercase letter. A lowercase single-token name is NOT "
    "counted and may leak undetected; true leakage is >= the reported value"
)


def _leak_rate(records: Sequence[Counts]) -> float:
    leaked = sum(c.tp for c in records)
    total = sum(c.tp + c.fn for c in records)
    return leaked / total if total else 0.0


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _norm_text(s: str) -> str:
    """Case-, unicode-, format-char-, and whitespace-insensitive view. A length-preserving
    transform (lowercase / uppercase / NFKC / whitespace tweak / inserting a zero-width or
    soft-hyphen format char inside the PII) leaves the value human-readable but would defeat
    a verbatim match — normalize so such a no-op still scores as a leak (§6.2). Unicode Cf
    (format) chars — U+200B ZWSP, U+00AD SHY, U+FEFF, U+2060, ZWNJ/ZWJ — are STRIPPED."""
    nfkc = unicodedata.normalize("NFKC", s)
    no_format = "".join(c for c in nfkc if unicodedata.category(c) != "Cf")
    return re.sub(r"\s+", " ", no_format.casefold())


def _is_distinctive(surface: str, min_len: int) -> bool:
    """Only score surfaces specific enough that a chance recurrence elsewhere in the
    output is implausible (avoids the short-common-substring / common-word FALSE leak,
    which would inflate a competitor's leakage)."""
    s = surface.strip()
    if len(s) < min_len:
        return False
    if "@" in s or sum(c.isdigit() for c in s) >= 3:
        return True
    # pure-alphabetic: count only proper-noun / multi-word shapes, NOT a lone common
    # lowercase word (e.g. "information"), which would coincidentally "leak".
    return len(s) >= 8 and (" " in s or any(c.isupper() for c in s))


def _surface_leaked(surface: str, out: str, out_digits: str, out_norm: str) -> bool:
    """Leaked if the surface survives verbatim, OR case/unicode/whitespace-normalized
    (so a lowercase/uppercase/NFKC 'redactor' that leaves PII readable is still counted),
    OR (for numeric PII) its digit string survives separator reformatting. The digit path
    requires >= 7 digits so a short 5-6 digit run cannot match by coincidence."""
    if surface in out:
        return True
    if _norm_text(surface) in out_norm:
        return True
    d = _digits(surface)
    return len(d) >= 7 and d in out_digits


def assess_leakage(
    substrate: Substrate,
    *,
    thresholds: PowerThresholds,
    seed: int,
    n_resamples: int = 500,
    min_specificity_len: int = 4,
    min_scoreable_fraction: float = 0.5,
) -> dict[str, DimensionResult]:
    results: dict[str, DimensionResult] = {}
    total_records = len(substrate.records)
    for sys in substrate.systems:
        per_record: list[Counts] = []
        scoreable = unscoreable = 0
        total_gold = total_leaked = 0
        for rec in substrate.records:
            out = rec.transform_by_system.get(sys)
            if out is None:
                unscoreable += 1
                continue
            surfaces = [
                rec.original_text[s:e]
                for (_t, s, e) in rec.reference
                if _is_distinctive(rec.original_text[s:e], min_specificity_len)
            ]
            out_digits = _digits(out)
            out_norm = _norm_text(out)
            leaked = (
                sum(1 for surf in surfaces if _surface_leaked(surf, out, out_digits, out_norm))
                if surfaces else 0
            )
            # SAFETY-CRITICAL ordering: check the LEAK first. A record whose surviving PII
            # is the worst case — a short output that IS the leaked email/SSN — must ALWAYS
            # be scored as a leak, never dropped as "degenerate" by an output-length test
            # (that would launder real leakage to 0%, the unsafe direction).
            if leaked > 0:
                scoreable += 1
                total_gold += len(surfaces)
                total_leaked += leaked
                per_record.append(Counts(tp=leaked, fp=0, fn=len(surfaces) - leaked))
                continue
            # No leak detected: a content-destroying / empty output cannot be fairly scored
            # for redaction quality ⇒ unscoreable (NOT a clean 0). A substantial output with
            # no surviving PII is a genuine 0-leak record.
            stripped = out.strip()
            input_len = len(rec.original_text.strip())
            if len(stripped) < 3 or (input_len >= 10 and len(stripped) < 0.3 * input_len):
                unscoreable += 1
                continue
            scoreable += 1
            if surfaces:
                total_gold += len(surfaces)
                per_record.append(Counts(tp=0, fp=0, fn=len(surfaces)))
            else:
                per_record.append(Counts(0, 0, 0))

        if scoreable == 0 or total_gold == 0:
            results[sys] = not_assessable(
                DIMENSION, sys,
                "no scoreable transform output with referenced PII (no transform, or all outputs degenerate)",
            )
            continue
        if scoreable < min_scoreable_fraction * total_records:
            results[sys] = not_assessable(
                DIMENSION, sys,
                f"only {scoreable}/{total_records} records had scoreable transform output (fail-closed)",
            )
            continue

        leak_rate = total_leaked / total_gold
        _point, ci = cluster_bootstrap_ci(per_record, _leak_rate, seed=seed, n_resamples=n_resamples)
        res = resolve(
            dimension=DIMENSION, system=sys, mode=substrate.mode, value=leak_rate,
            support=total_gold, n_clusters=len(per_record), confidence_interval=ci, significant=None,
            thresholds=thresholds, caveats=(_LOWER_BOUND_CAVEAT,),
            silver_reasons=("reference is silver (advisory); leakage measured vs silver spans",)
            if substrate.mode is MeasurementMode.SILVER else (),
        )
        res = replace(res, metadata={
            **res.metadata,
            "lower_bound": True,
            "scoreable_records": scoreable,
            "unscoreable_records": unscoreable,
            "total_referenced_pii": total_gold,
            "leaked": total_leaked,
            "direction": "lower_is_better",
        })
        results[sys] = res
    return results
