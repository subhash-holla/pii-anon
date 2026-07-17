"""Reproducible statistics for assurance reports.

Pure-Python (seeded :class:`random.Random`) so every CI / p-value is byte-
deterministic given a seed and never depends on a numpy version (the red-team
flagged numpy/pure-Python drift as a reproducibility risk).

Red-team-mandated discipline:

* **Cluster (per-record) bootstrap over pooled counts** for every headline —
  resample *records*, re-pool tp/fp/fn, recompute the END metric. Never bootstrap
  a per-record F1 for a micro headline; never bootstrap per entity.
* **Paired bootstrap** for an A-vs-B comparison — resample the same record indices
  for both systems and bootstrap the per-record delta vector.
* **Holm-Bonferroni** multiplicity control across the reported comparison family,
  with the family size stated by the caller.
* **Effect size** alongside p (at large N a trivial delta becomes "significant").
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Counts:
    """Confusion counts for one record (or pooled)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def prf1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from confusion counts. Empty (all-zero) → (0, 0, 0)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    """F-beta (beta>1 favours recall — privacy-first)."""
    precision, recall, _ = prf1_from_counts(tp, fp, fn)
    b2 = beta * beta
    denom = b2 * precision + recall
    return ((1 + b2) * precision * recall / denom) if denom else 0.0


def _pool(records: Sequence[Counts]) -> Counts:
    total = Counts()
    for c in records:
        total = total + c
    return total


def pooled_f1(records: Sequence[Counts]) -> float:
    t = _pool(records)
    return prf1_from_counts(t.tp, t.fp, t.fn)[2]


def pooled_precision(records: Sequence[Counts]) -> float:
    t = _pool(records)
    return prf1_from_counts(t.tp, t.fp, t.fn)[0]


def pooled_recall(records: Sequence[Counts]) -> float:
    t = _pool(records)
    return prf1_from_counts(t.tp, t.fp, t.fn)[1]


def pooled_fbeta(records: Sequence[Counts], beta: float = 2.0) -> float:
    t = _pool(records)
    return fbeta_from_counts(t.tp, t.fp, t.fn, beta)


# A committed floor on bootstrap resamples for a PUBLISHABLE CI: too few resamples collapse
# the percentile CI toward zero width, which would forge precision (and a MEASURED label).
# Below this, the CI is None -> the power gate demotes to ADVISORY (an exploratory low-B run
# still produces point estimates, just never a publishable claim).
_MIN_RESAMPLES_FOR_CI = 200


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile of an ALREADY-SORTED list, q in [0,1]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def cluster_bootstrap_ci(
    per_record: Sequence[Counts],
    metric: Callable[[Sequence[Counts]], float],
    *,
    seed: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float] | None]:
    """Point estimate (metric on full sample) + percentile CI via the record bootstrap.

    Returns ``(point, (lo, hi))``. CI is ``None`` when there are too few records to
    bootstrap (fail closed — the power gate then demotes the value).
    """
    n = len(per_record)
    point = metric(per_record)
    if n < 2 or n_resamples < _MIN_RESAMPLES_FOR_CI:  # fail closed: too few clusters/resamples -> no CI -> ADVISORY
        return point, None
    rng = random.Random(seed)
    draws: list[float] = []
    idx_range = range(n)
    for _ in range(n_resamples):
        sample = [per_record[rng.choice(idx_range)] for _ in idx_range]
        draws.append(metric(sample))
    draws.sort()
    lo = _percentile(draws, alpha / 2)
    hi = _percentile(draws, 1 - alpha / 2)
    return point, (lo, hi)


def mean_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float] | None]:
    """Point mean + percentile CI via the record bootstrap, for a per-record FLOAT metric
    (e.g. utility preservation) — the float analogue of :func:`cluster_bootstrap_ci`.

    Resamples records with replacement and re-means at the pinned seed. Fail closed:
    ``None`` CI when there are too few records / resamples (the power gate then demotes)."""
    n = len(values)
    point = statistics.fmean(values) if values else 0.0
    if n < 2 or n_resamples < _MIN_RESAMPLES_FOR_CI:
        return point, None
    rng = random.Random(seed)
    draws: list[float] = []
    idx_range = range(n)
    for _ in range(n_resamples):
        sample = [values[rng.choice(idx_range)] for _ in idx_range]
        draws.append(statistics.fmean(sample))
    draws.sort()
    lo = _percentile(draws, alpha / 2)
    hi = _percentile(draws, 1 - alpha / 2)
    return point, (lo, hi)


@dataclass(frozen=True)
class PairedResult:
    """Result of a paired A-vs-B bootstrap on the per-record delta of a metric."""

    delta: float                       # metric(A) - metric(B) on the full pooled sample
    ci: tuple[float, float] | None     # bootstrap CI of the delta
    p_value: float                     # two-sided achieved significance level
    effect_size: float | None          # Cohen's d; None == undefined (maximal consistent gap)
    n: int


def _per_record_metric(c: Counts, metric_name: str) -> float | None:
    """A per-record scalar for effect-size only (None where undefined)."""
    if (c.tp + c.fp + c.fn) == 0:
        return None
    p, r, f1 = prf1_from_counts(c.tp, c.fp, c.fn)
    return {"f1": f1, "precision": p, "recall": r}.get(metric_name, f1)


def cohens_d_paired(deltas: Sequence[float]) -> float | None:
    """Standardized mean of paired deltas.

    Zero variance with a ZERO mean is genuinely no effect (0.0). Zero variance with a
    NON-ZERO mean is a perfectly consistent, MAXIMAL effect — Cohen's d is undefined
    (infinite), NOT zero — so we return None (the renderer shows 'maximal'), never a
    misleading 0.0 next to a significant p-value."""
    values = list(deltas)
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd == 0.0:
        return 0.0 if mean == 0.0 else None
    return mean / sd


def paired_bootstrap(
    per_record_a: Sequence[Counts],
    per_record_b: Sequence[Counts],
    pooled_metric: Callable[[Sequence[Counts]], float],
    *,
    metric_name: str = "f1",
    seed: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired record bootstrap of ``metric(A) - metric(B)``.

    Same resampled record indices applied to both systems (the correct unit for an
    A-vs-B comparison). Two-sided achieved-significance-level p-value.
    """
    if len(per_record_a) != len(per_record_b):
        raise ValueError("paired_bootstrap requires equal-length per-record sequences")
    n = len(per_record_a)
    delta_full = pooled_metric(per_record_a) - pooled_metric(per_record_b)
    # Effect size over per-record metric deltas (records where either side defined).
    per_deltas: list[float] = []
    for ca, cb in zip(per_record_a, per_record_b):
        ma = _per_record_metric(ca, metric_name)
        mb = _per_record_metric(cb, metric_name)
        if ma is None or mb is None:  # only records where BOTH metrics are defined
            continue
        per_deltas.append(ma - mb)
    effect = cohens_d_paired(per_deltas)
    if n < 2 or n_resamples < 1:  # fail closed: no CI, neutral p
        return PairedResult(delta_full, None, 1.0, effect, n)
    rng = random.Random(seed)
    idx_range = range(n)
    draws: list[float] = []
    le = 0  # resampled deltas <= 0
    ge = 0  # resampled deltas >= 0
    for _ in range(n_resamples):
        idxs = [rng.choice(idx_range) for _ in idx_range]
        a = [per_record_a[i] for i in idxs]
        b = [per_record_b[i] for i in idxs]
        d = pooled_metric(a) - pooled_metric(b)
        draws.append(d)
        if d <= 0.0:
            le += 1
        if d >= 0.0:
            ge += 1
    draws.sort()
    ci = (_percentile(draws, alpha / 2), _percentile(draws, 1 - alpha / 2))
    # Floor at 1/B: a finite resample can never establish p < 1/n_resamples, so a raw
    # 0.0 is reported as the resolution limit rather than an impossible exact zero.
    p = max(2.0 * min(le, ge) / n_resamples, 1.0 / n_resamples)
    p = min(1.0, p)
    return PairedResult(delta_full, ci, p, effect, n)


def holm_bonferroni(
    pvalues: dict[str, float], alpha: float = 0.05
) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns name -> is_significant after correction.

    Controls the family-wise error rate across ALL comparisons passed in one call —
    so the caller must pass the entire reported family together (the family size is
    ``len(pvalues)``).
    """
    if not pvalues:
        return {}
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    result: dict[str, bool] = {}
    rejected_so_far = True
    for rank, (name, p) in enumerate(ordered):
        adjusted_alpha = alpha / (m - rank)
        if rejected_so_far and p <= adjusted_alpha:
            result[name] = True
        else:
            # once one fails to reject, all subsequent (larger p) also fail
            rejected_so_far = False
            result[name] = False
    return result
