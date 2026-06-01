"""Record-level paired-outcome assembly — temp-local builder (S3-04, FR-004).

# SWITCH-POINT(S6)
----------------
The design's *canonical* ``assemble_paired_set`` lives in the ``pii-anon-eval-data``
package (S6), which is **VERIFIED ABSENT** today. To keep S3-04 (and the
downstream SDO ``CompetitiveSupremacyGate``) from being DATA-blocked, this module
ships a self-contained, thin builder behind a small interface. When S6 lands,
swap this implementation for the eval-data primitive *behind the same interface*
(:func:`assemble_paired_set` / :class:`PairedComparisonSet`) — this marker and
the thin surface are deliberate so the swap is a drop-in.

PURE STDLIB + NUMPY. Imports nothing from the detection / orchestration /
``evaluation`` layers (S3-05 boundary). It takes a plain
``dict[str, list[float]]`` of per-record F1 (aligned by record index) — NOT an
``ExternalEvaluationResult`` — so the rating layer stays import-isolated.

What it does
------------
Each record contributes a pairwise outcome between every pair of systems from
their per-record F1: ``|f1_i − f1_j| ≤ tie_eps`` ⇒ a **tie**, else a win for the
larger F1. Across ``N`` records and ``K`` systems this yields ``C(K, 2)`` paired
tallies ``(wins_i, wins_j, ties, n)`` with ``wins_i + wins_j + ties = n = N``.

Ragged inputs are rejected LOUDLY
---------------------------------
Per-record outcomes require the systems' F1 lists to be index-aligned over the
SAME records. Mismatched lengths are a *data error*, so :func:`assemble_paired_set`
raises :class:`ValueError` naming the systems and their lengths. This DELIBERATELY
DIVERGES from the silent ``min_len`` truncation at ``competitor_compare.py:2871``
(which aligns ragged lists to the shorter): truncating would silently compare
different record sets. A reviewer must not "fix" the loud raise back into a
silent truncation — the divergence is intentional.

The two adapter views
---------------------
* :meth:`PairedComparisonSet.to_paired_counts_with_ties` → the 4-tuple
  ``{(i, j): (wins_i, wins_j, ties, n)}`` the Davidson tie likelihood consumes.
* :meth:`PairedComparisonSet.to_paired_counts` → the S3-02 3-tuple
  ``{(i, j): (wins_i, wins_j, n')}`` that :meth:`BayesBTEngine.fit_paired_posterior`
  consumes, with **ties EXCLUDED** (``n' = wins_i + wins_j``) so all totals stay
  **integer**. Ties are dropped rather than half-split, because
  ``_normalize_comparisons`` rounds ``wins_i`` with ``int(round(...))`` and a
  half-integer (from splitting an odd tie count) would corrupt the Binomial
  counts.

Evidence basis:
    - Davidson (1970) "On Extending the Bradley-Terry Model to Accommodate Ties".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Shape aliases mirroring the rating layer's paired-count conventions.
# 3-tuple == S3-02's ``PairedCounts`` value (wins_i, wins_j, n) — integer totals.
PairedCounts = dict[tuple[str, str], tuple[int, int, int]]
# 4-tuple == the Davidson path's value (wins_i, wins_j, ties, n).
PairedCountsWithTies = dict[tuple[str, str], tuple[int, int, int, int]]

# Default tie band — a tiny float-equality tolerance, so only numerically
# identical F1 counts as a tie unless the caller widens it.
_DEFAULT_TIE_EPS = 1e-9


@dataclass(frozen=True)
class PairedComparisonSet:
    """Record-level pairwise tallies between systems, with ties.

    Attributes
    ----------
    systems:
        Sorted system names. Pair keys are ``(systems[a], systems[b])`` with
        ``a < b``.
    n_records:
        The (shared) number of records every system was scored on.
    counts_with_ties:
        ``{(sys_i, sys_j): (wins_i, wins_j, ties, n)}`` with
        ``wins_i + wins_j + ties = n = n_records``. The canonical internal tally;
        the adapters below are pure views over it.
    """

    systems: list[str]
    n_records: int
    counts_with_ties: PairedCountsWithTies

    def to_paired_counts_with_ties(self) -> PairedCountsWithTies:
        """The 4-tuple view ``{(i, j): (wins_i, wins_j, ties, n)}`` (Davidson).

        Keeps the FULL ``n = wins_i + wins_j + ties`` — the tie counts the
        Davidson likelihood needs.
        """
        return dict(self.counts_with_ties)

    def to_paired_counts(self) -> PairedCounts:
        """The S3-02 3-tuple view ``{(i, j): (wins_i, wins_j, n')}`` (plain BT).

        Ties are **EXCLUDED**: ``n' = wins_i + wins_j`` (NOT ``n``), and the
        counts are returned as plain Python ``int`` so the downstream
        ``int(round(...))`` normalization is exact. Ties are dropped, never
        half-split, to avoid half-integer corruption of the Binomial totals.
        """
        out: PairedCounts = {}
        for pair, (wins_i, wins_j, _ties, _n) in self.counts_with_ties.items():
            out[pair] = (int(wins_i), int(wins_j), int(wins_i + wins_j))
        return out


def assemble_paired_set(
    per_record_f1: dict[str, list[float]],
    *,
    tie_eps: float = _DEFAULT_TIE_EPS,
) -> PairedComparisonSet:
    """Assemble record-level paired outcomes from per-system per-record F1.

    Parameters
    ----------
    per_record_f1:
        ``{system_name: [f1_record_0, f1_record_1, ...]}`` — F1 per record,
        index-aligned across systems (record ``r`` is the SAME record for every
        system). A plain dict, deliberately NOT an ``ExternalEvaluationResult``,
        so the rating layer stays import-isolated.
    tie_eps:
        Tie band: ``|f1_i − f1_j| ≤ tie_eps`` ⇒ a tie (inclusive at the boundary).
        Must be ``≥ 0``. Default ``1e-9`` (numeric float-equality).

    Returns
    -------
    :class:`PairedComparisonSet` with ``C(K, 2)`` tallies.

    Raises
    ------
    ValueError
        If fewer than 2 systems; if any per-record list is empty (0 records);
        if the lists are ragged (mismatched lengths — named loudly, NOT
        truncated); if ``tie_eps < 0``; or if any F1 value is NaN.
    """
    if tie_eps < 0.0:
        raise ValueError(f"tie_eps must be >= 0; got {tie_eps}")

    systems = sorted(per_record_f1)
    if len(systems) < 2:
        raise ValueError(
            "assemble_paired_set needs at least two systems to form a pair; "
            f"got {len(systems)}: {systems}"
        )

    # Length / raggedness check — reject loudly, naming systems + lengths. This
    # is the DELIBERATE divergence from competitor_compare.py:2871's silent
    # min_len truncation: ragged input is a data error, not something to align
    # to the shorter list.
    lengths = {name: len(per_record_f1[name]) for name in systems}
    distinct_lengths = set(lengths.values())
    if len(distinct_lengths) != 1:
        detail = ", ".join(f"{name}={lengths[name]}" for name in systems)
        raise ValueError(
            "per-record F1 lists are ragged (mismatched lengths); "
            "record-level paired outcomes require index-aligned lists over the "
            f"same records — refusing to truncate. Lengths: {detail}"
        )

    n_records = distinct_lengths.pop()
    if n_records == 0:
        raise ValueError(
            "per-record F1 lists are empty (0 records); cannot assemble paired "
            "outcomes from no records"
        )

    # NaN guard: a NaN F1 cannot be classified (NaN comparisons are silently
    # False), so reject loudly rather than silently mis-bucket it as a tie/win.
    for name in systems:
        for value in per_record_f1[name]:
            if math.isnan(value):
                raise ValueError(
                    f"per-record F1 for system {name!r} contains NaN; "
                    "cannot classify a NaN outcome"
                )

    counts: PairedCountsWithTies = {}
    for a in range(len(systems)):
        for b in range(a + 1, len(systems)):
            sys_i, sys_j = systems[a], systems[b]
            f1_i = per_record_f1[sys_i]
            f1_j = per_record_f1[sys_j]
            wins_i = 0
            wins_j = 0
            ties = 0
            for r in range(n_records):
                diff = f1_i[r] - f1_j[r]
                if abs(diff) <= tie_eps:
                    ties += 1
                elif diff > 0.0:
                    wins_i += 1
                else:
                    wins_j += 1
            counts[(sys_i, sys_j)] = (wins_i, wins_j, ties, n_records)

    return PairedComparisonSet(
        systems=systems,
        n_records=n_records,
        counts_with_ties=counts,
    )
