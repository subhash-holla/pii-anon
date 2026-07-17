"""Re-identification resistance assessor — honestly labeled (spec §6.3).

DEFAULT ``NOT_ASSESSABLE`` for real attacks: a real candidate pool + gold
persona-target links + ≥128 shadow models are absent on an ordinary user dataset,
and the lightweight MIA ``auc_approx`` proxy is provably degenerate, so it is
EXCLUDED entirely (never contributes here).

What IS computable — and reported ``ADVISORY`` only, as a fail-closed LOWER BOUND —
is a **deterministic representative-adversary residual-linkability stand-in**: over a
closed world of the dataset's own records, does each system's ANONYMIZED output still
contain a real, distinctive identifier (its OWN or — for a broken surrogate that
permutes identifiers across records — ANOTHER record's) in a human-readable form? The
value is the LINKABILITY = fraction of outputs in which some real identifier survives
readably. Higher ⇒ more residual re-identifiability ⇒ weaker resistance. Lower is better.

Why a COMPACTED-SUBSTRING adversary rather than a word-token one (close rounds 2 & 5):

* A word-token / Jaccard adversary is obfuscation-fragile and produced two unsafe
  DIRECTION inversions — a cross-record identifier PERMUTATION (leaks a real name into
  another record) and CHARACTER-DOTTING (``A.l.p.h.o.n.s.e``) both defeated token
  matching while leaving the PII human-readable, so a strictly-weaker transform reported
  a strictly-LOWER risk (overstating anonymity, the leak-direction class).
* The compacted view (NFKC + casefold + strip every non-alphanumeric) recovers
  separator-obfuscated identifiers (dotting / spacing / hyphenation) and a substring test
  catches a real identifier surviving anywhere in the output (own record OR another's), so
  the measure is MONOTONE-safe: more readable PII surviving ⇒ ≥ linkability, never an
  inversion. It errs toward OVER-reporting risk (the safe direction).

Honesty constraints (spec §6.3, all enforced here):

* NEVER ``MEASURED`` — a representative adversary is not a trained-shadow attack.
* NEVER built from a silver reference (identifiers come from gold only); silver ⇒ NOT_ASSESSABLE.
* Requires a ``transform`` (no anonymized text to attack otherwise) ⇒ NOT_ASSESSABLE without one.
* A LOWER BOUND: a more capable adversary (homoglyphs, alnum-noise insertion) may
  re-identify more. The non-strippable :data:`ANTI_ANONYMITY_CAVEAT` is surfaced verbatim
  ("a low rate is NOT a guarantee of anonymity"); the closed-world ``|C|`` is stated.
"""

from __future__ import annotations

import unicodedata

from ..adjudication import Substrate
from ..claim_strength import (
    ClaimStrength,
    DimensionResult,
    MeasurementMode,
    PowerThresholds,
    advisory,
    not_assessable,
)

DIMENSION = "reidentification"

# A compacted identifier must be at least this long to count, so a short compacted name does not
# match another record's output by coincidence (a distinctiveness floor for the substring test).
_MIN_QI_LEN = 5
_MIN_CLOSED_WORLD = 2  # a 1-record "closed world" is a trivial link; need >= 2 candidates


def _compact(s: str) -> str:
    """Obfuscation-robust canonical view: NFKC-fold (fullwidth / ligatures), casefold, then keep
    only alphanumerics — so separator-based obfuscation (``A.l.p.h.o.n.s.e``, ``A l p h``,
    ``a-l-p-h``) collapses back to the readable identifier it still is to a human."""
    folded = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def assess_reidentification(
    substrate: Substrate,
    *,
    thresholds: PowerThresholds,  # accepted for signature parity; re-id is never MEASURED
    seed: int,
    n_resamples: int = 500,
) -> dict[str, DimensionResult]:
    # heavy import inside the function (mirrors the package convention)
    from pii_anon.eval_framework.attacks.reid import ANTI_ANONYMITY_CAVEAT

    results: dict[str, DimensionResult] = {}

    if substrate.mode is MeasurementMode.SILVER:
        for sys in substrate.systems:
            results[sys] = not_assessable(
                DIMENSION, sys,
                "re-identification stand-in requires gold quasi-identifiers; silver reference "
                "is detector-derived (circular) — not assessable",
            )
        return results

    # The closed-world distinctive identifiers (compacted gold PII surfaces), shared across systems.
    identifiers: set[str] = set()
    records_with_pii = 0
    for rec in substrate.records:
        has_qi = False
        for _t, s, e in rec.reference:
            c = _compact(rec.original_text[s:e])
            if len(c) >= _MIN_QI_LEN:
                identifiers.add(c)
                has_qi = True
        if has_qi:
            records_with_pii += 1

    if records_with_pii < _MIN_CLOSED_WORLD or not identifiers:
        for sys in substrate.systems:
            results[sys] = not_assessable(
                DIMENSION, sys,
                f"too few records carry a distinctive referenced identifier for a closed-world "
                f"linkage experiment (|C|={records_with_pii} < {_MIN_CLOSED_WORLD})",
            )
        return results

    candidate_set_size = records_with_pii
    sorted_ids = sorted(identifiers)  # deterministic iteration order

    for sys in substrate.systems:
        # only records that carry a distinctive identifier are targets (consistent with |C|)
        targets = [
            rec for rec in substrate.records
            if any(len(_compact(rec.original_text[s:e])) >= _MIN_QI_LEN for _t, s, e in rec.reference)
        ]
        missing_transform = any(rec.transform_by_system.get(sys) is None for rec in targets)
        if missing_transform:
            results[sys] = not_assessable(
                DIMENSION, sys,
                "pipeline exposes no transform; there is no anonymized output to attack",
            )
            continue

        linkable = 0
        for rec in targets:
            out = rec.transform_by_system.get(sys) or ""
            compact_out = _compact(out)
            # a confident link = SOME real distinctive identifier (own or another record's, to
            # catch a cross-record permutation) survives readably in this output.
            if any(qi in compact_out for qi in sorted_ids):
                linkable += 1
        n_targets = len(targets)
        linkability = (linkable / n_targets) if n_targets else 0.0

        results[sys] = advisory(
            DIMENSION, sys, linkability,
            reasons=(
                "representative deterministic compacted-substring linkage adversary, NOT a "
                "trained-shadow attack; ADVISORY only — never a publishable re-id claim",
                "value = residual LINKABILITY, a LOWER BOUND (fraction of outputs in which some "
                "real distinctive identifier survives readably, including cross-record leaks); "
                f"closed-world |C|={candidate_set_size}",
            ),
            caveats=(ANTI_ANONYMITY_CAVEAT,),
            support=n_targets,
            metadata={
                "linkability": linkability,
                "linkable_targets": linkable,
                "n_targets": n_targets,
                "candidate_set_size": candidate_set_size,
                "lower_bound": True,
                "adversary": "compacted-substring-linkage@v1",
                "direction": "lower_is_better",
                "real_attack_status": "NOT_ASSESSABLE: no candidate pool / persona-target links / "
                "shadow models on user data; MIA auc_approx excluded (degenerate)",
            },
        )

    # re-id is per-system ADVISORY by construction — never MEASURED, never a head-to-head winner.
    assert all(
        r.claim_strength in (ClaimStrength.ADVISORY, ClaimStrength.NOT_ASSESSABLE)
        for r in results.values()
    )
    return results
