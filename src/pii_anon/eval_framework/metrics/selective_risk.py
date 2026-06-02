"""Per-class calibration & selective-risk reporting (S4-03 — completes SDO G4).

The sibling :mod:`pii_anon.eval_framework.metrics.calibration` already provides
the **global** Expected/Maximum Calibration Error, the reliability-diagram data,
and a :class:`~pii_anon.eval_framework.metrics.calibration.TemperatureScaler`.
This module adds the *per-entity-class* selective-risk surface that AX-005
(calibrated abstention) requires and that the SDO **G4** guarantee reads:

* **per-class ECE** — one reliability diagram per entity type (REUSES
  :func:`~pii_anon.eval_framework.metrics.calibration.reliability_diagram_data`;
  this module never re-derives ECE — RISK-1 keeps ``calibration.py`` byte-identical);
* **per-class Brier** + the Murphy reliability/resolution/uncertainty
  decomposition (NFR-018);
* **per-class AURC** + a **monotone-non-increasing** risk-coverage curve
  (NFR-019 — risk never drops as coverage grows / never rises as you abstain
  more; the emitted curve is the running-max monotone envelope of the
  confidence-ranked cumulative error, the canonical optimal selective-risk
  curve);
* an **abstention operating-point table** with ≥3 rows at the selective-risk
  targets {1%, 2%, 5%} (NFR-021);
* a **calibrated-confidence-coverage** audit (NFR-020, the lone MUST) — the
  fraction of findings carrying BOTH a calibrated confidence (a real probability
  in ``[0, 1]``) AND a provenance string; a bare logit or a provenance-less
  finding drops coverage below ``1.0`` and is surfaced (teeth, not a no-op);
* post-temperature-scaling per-class ECE (REUSES ``TemperatureScaler``;
  NFR-017 — ``ECE_post ≤ ECE_pre`` always; held to the 0.05 high-resource bar or
  the 0.08 long-tail bar by class support).

Evidence basis:
- Geifman & El-Yaniv (2017): Selective classification / risk-coverage + AURC.
- Murphy (1973): the reliability/resolution/uncertainty Brier decomposition.
- Naeini et al. (2015) / Guo et al. (2017): ECE + temperature scaling (reused).

Boundary (story §2a): this module imports ONLY numpy / stdlib + the sibling
``calibration`` metric. It imports NOTHING from ``swarm`` / ``moe`` / ``fusion`` /
``policy`` (CI-guarded by ``tests/test_selective_risk.py``). It is eval REPORTING
— decoupled from the runtime ``calibration/`` package. Determinism (AX-002): the
reporter is a pure function of its inputs (the ordering tie-breaks are stable).
All corpora are SYNTHETIC (AX-001 — never real PII).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import reliability_diagram_data
from .calibration import TemperatureScaler

__all__ = [
    "ScoredFinding",
    "BrierDecomposition",
    "RiskCoveragePoint",
    "AbstentionOperatingPoint",
    "SelectiveRiskReport",
    "SelectiveRiskReporter",
    "HIGH_RESOURCE_ECE_BAR",
    "LONG_TAIL_ECE_BAR",
    "LONG_TAIL_SUPPORT",
    "ABSTENTION_TARGETS",
]

# -- NFR-017 thresholds: per-class post-scaling ECE bars (the contract). ------
HIGH_RESOURCE_ECE_BAR: float = 0.05
LONG_TAIL_ECE_BAR: float = 0.08
# A class with fewer than this many findings is "long-tail" (the looser bar).
LONG_TAIL_SUPPORT: int = 50

# -- NFR-021: the abstention operating-point selective-risk targets. ----------
ABSTENTION_TARGETS: tuple[float, ...] = (0.01, 0.02, 0.05)


@dataclass(frozen=True)
class ScoredFinding:
    """One scored prediction for the selective-risk report (synthetic — AX-001).

    Attributes
    ----------
    entity_type:
        The entity class (e.g. ``"EMAIL_ADDRESS"``) — the per-class slice key.
    confidence:
        The model's confidence. For a CALIBRATED finding this is a probability in
        ``[0, 1]``; a value outside ``[0, 1]`` is a bare logit (an NFR-020 breach).
    correct:
        Whether the prediction was correct (the calibration target).
    calibrated:
        Whether the confidence is a calibrated probability (vs a raw logit).
        Drives the NFR-020 coverage audit.
    provenance:
        The calibration provenance (e.g. the fusion strategy / calibrator id).
        ``None`` ⇒ provenance-less ⇒ an NFR-020 breach even if in range.
    """

    entity_type: str
    confidence: float
    correct: bool
    calibrated: bool = True
    provenance: str | None = None


@dataclass(frozen=True)
class BrierDecomposition:
    """The Murphy (1973) calibration/refinement decomposition of a Brier score.

    The exact identity holds for the **binned** Brier (the score of the
    bin-mean-confidence predictions): ``binned_brier == reliability - resolution
    + uncertainty`` (to machine epsilon). The RAW Brier additionally carries a
    non-negative within-bin variance term, so ``raw_brier ≥ binned_brier``;
    ``binned_brier`` is exposed precisely so the decomposition reconciles exactly.

    ``reliability`` low is good (bin confidence tracks bin accuracy);
    ``resolution`` high is good (the bins separate accuracy from the base rate);
    ``uncertainty`` is the irreducible base-rate variance ``p̄(1-p̄)``.
    """

    reliability: float
    resolution: float
    uncertainty: float
    binned_brier: float


@dataclass(frozen=True)
class RiskCoveragePoint:
    """One ``(coverage, risk)`` point on the monotone risk-coverage curve."""

    coverage: float
    risk: float


@dataclass(frozen=True)
class AbstentionOperatingPoint:
    """One abstention operating point: a target selective-risk → achieved coverage.

    ``achieved_coverage`` is the LARGEST coverage whose (monotone-envelope)
    selective risk is ≤ ``target_risk``; ``achieved_risk`` is the envelope risk at
    that coverage (≤ ``target_risk`` whenever any coverage qualifies).
    """

    target_risk: float
    achieved_coverage: float
    achieved_risk: float


@dataclass(frozen=True)
class SelectiveRiskReport:
    """The per-class calibration + selective-risk surface (what G4 reads).

    All per-class mappings are keyed by ``entity_type`` and sorted by key for
    determinism (AX-002). The ``global_*`` fields pool every finding.
    """

    # Per-class calibration (ECE reused from metrics/calibration.py).
    per_class_ece: dict[str, float]
    per_class_ece_post: dict[str, float]  # post-temperature-scaling (NFR-017)
    per_class_brier: dict[str, float]
    per_class_brier_decomposition: dict[str, BrierDecomposition]
    per_class_aurc: dict[str, float]
    per_class_risk_coverage: dict[str, tuple[RiskCoveragePoint, ...]]
    per_class_support: dict[str, int]

    # Global (pooled) calibration.
    global_ece: float
    global_aurc: float
    global_risk_coverage: tuple[RiskCoveragePoint, ...]

    # NFR-021: the ≥3-point abstention operating-point table (global).
    abstention_table: tuple[AbstentionOperatingPoint, ...]

    # NFR-020 (the lone MUST): calibrated-confidence-coverage audit.
    calibrated_confidence_coverage: float
    bare_logit_count: int
    uncalibrated_findings: tuple[int, ...]  # indices of the offending findings

    # NFR-017: per-class ECE bar (0.05 high-resource / 0.08 long-tail).
    per_class_ece_threshold: dict[str, float] = field(default_factory=dict)

    def ece_threshold_for(self, entity_type: str) -> float:
        """The NFR-017 ECE bar this class is held to (0.05 or 0.08)."""
        return self.per_class_ece_threshold[entity_type]


class SelectiveRiskReporter:
    """Compute a :class:`SelectiveRiskReport` from scored findings (reusing ECE).

    Pure + deterministic (AX-002): given the same findings the report is
    bit-for-bit identical (stable confidence-descending ordering; sorted class
    keys). Reuses :func:`reliability_diagram_data` + :class:`TemperatureScaler`
    from the sibling ``calibration`` module — never re-deriving ECE (RISK-1).
    """

    def __init__(self, num_bins: int = 15) -> None:
        """Initialise the reporter.

        Args:
            num_bins: Number of equal-width confidence bins for ECE/Brier
                decomposition (default 15, matching the global metric).
        """
        self.num_bins = num_bins

    # -- public API ---------------------------------------------------------

    def report(self, findings: list[ScoredFinding]) -> SelectiveRiskReport:
        """Build the full per-class + global selective-risk report."""
        # NFR-020 audit first (over ALL findings, before any per-class slicing).
        coverage, bare_count, offenders = self._coverage_audit(findings)

        # Per-class slices (sorted keys → deterministic).
        by_class: dict[str, list[ScoredFinding]] = {}
        for f in findings:
            by_class.setdefault(f.entity_type, []).append(f)

        per_class_ece: dict[str, float] = {}
        per_class_ece_post: dict[str, float] = {}
        per_class_brier: dict[str, float] = {}
        per_class_decomp: dict[str, BrierDecomposition] = {}
        per_class_aurc: dict[str, float] = {}
        per_class_curve: dict[str, tuple[RiskCoveragePoint, ...]] = {}
        per_class_support: dict[str, int] = {}
        per_class_threshold: dict[str, float] = {}

        for et in sorted(by_class):
            slice_ = by_class[et]
            # Only CALIBRATED, in-range findings feed the calibration math; a bare
            # logit cannot be binned as a probability (it is an NFR-020 breach the
            # coverage audit already flagged). This keeps ECE/Brier well-defined.
            confs = [f.confidence for f in slice_ if _is_probability(f.confidence)]
            correct = [f.correct for f in slice_ if _is_probability(f.confidence)]
            support = len(confs)
            per_class_support[et] = support

            per_class_ece[et] = reliability_diagram_data(
                confs, correct, num_bins=self.num_bins
            ).ece
            per_class_ece_post[et] = self._post_scaling_ece(confs, correct)
            per_class_brier[et] = _brier(confs, correct)
            per_class_decomp[et] = self._brier_decomposition(confs, correct)
            cov, risk = _monotone_risk_coverage(confs, correct)
            per_class_curve[et] = tuple(
                RiskCoveragePoint(c, r) for c, r in zip(cov, risk)
            )
            per_class_aurc[et] = float(np.mean(risk)) if len(risk) else 0.0
            per_class_threshold[et] = (
                LONG_TAIL_ECE_BAR if support < LONG_TAIL_SUPPORT else HIGH_RESOURCE_ECE_BAR
            )

        # Global (pooled).
        g_confs = [f.confidence for f in findings if _is_probability(f.confidence)]
        g_correct = [f.correct for f in findings if _is_probability(f.confidence)]
        global_ece = reliability_diagram_data(
            g_confs, g_correct, num_bins=self.num_bins
        ).ece
        g_cov, g_risk = _monotone_risk_coverage(g_confs, g_correct)
        global_curve = tuple(RiskCoveragePoint(c, r) for c, r in zip(g_cov, g_risk))
        global_aurc = float(np.mean(g_risk)) if len(g_risk) else 0.0

        abstention_table = _abstention_table(g_cov, g_risk)

        return SelectiveRiskReport(
            per_class_ece=per_class_ece,
            per_class_ece_post=per_class_ece_post,
            per_class_brier=per_class_brier,
            per_class_brier_decomposition=per_class_decomp,
            per_class_aurc=per_class_aurc,
            per_class_risk_coverage=per_class_curve,
            per_class_support=per_class_support,
            global_ece=global_ece,
            global_aurc=global_aurc,
            global_risk_coverage=global_curve,
            abstention_table=abstention_table,
            calibrated_confidence_coverage=coverage,
            bare_logit_count=bare_count,
            uncalibrated_findings=offenders,
            per_class_ece_threshold=per_class_threshold,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _coverage_audit(
        findings: list[ScoredFinding],
    ) -> tuple[float, int, tuple[int, ...]]:
        """NFR-020 audit — fraction of findings with a calibrated confidence AND
        provenance; the bare/provenance-less count + their indices.

        A finding is COVERED iff it is flagged ``calibrated`` AND its confidence is
        a probability in ``[0, 1]`` AND it carries a non-empty ``provenance``. An
        empty corpus is vacuously fully covered (``1.0``) — never a divide-by-zero.
        """
        if not findings:
            return 1.0, 0, ()
        offenders: list[int] = []
        for i, f in enumerate(findings):
            covered = (
                f.calibrated
                and _is_probability(f.confidence)
                and bool(f.provenance)
            )
            if not covered:
                offenders.append(i)
        bare = len(offenders)
        coverage = (len(findings) - bare) / len(findings)
        return coverage, bare, tuple(offenders)

    def _post_scaling_ece(self, confs: list[float], correct: list[bool]) -> float:
        """NFR-017: the ECE after the optimal temperature scaling.

        Reuses ``TemperatureScaler`` (the grid search always includes ``T=1.0``,
        so the post-scaling ECE can never be WORSE than the pre-scaling ECE —
        ``ECE_post ≤ ECE_pre``). Returns the pre-scaling ECE unchanged for an
        empty slice.
        """
        if not confs:
            return 0.0
        scaler = TemperatureScaler()
        optimal_t = scaler.find_optimal_temperature(
            confs, correct, num_bins=self.num_bins
        )
        scaled = TemperatureScaler(temperature=optimal_t).scale(confs)
        return reliability_diagram_data(scaled, correct, num_bins=self.num_bins).ece

    def _brier_decomposition(
        self, confs: list[float], correct: list[bool]
    ) -> BrierDecomposition:
        """The Murphy (1973) reliability/resolution/uncertainty decomposition.

        Computed over the SAME equal-width bins as the reliability diagram. The
        exact identity ``binned_brier == reliability - resolution + uncertainty``
        holds for the bin-mean-confidence predictions (to machine epsilon); the
        ``binned_brier`` it reconciles with is returned alongside the components.
        An empty slice → all-zero.
        """
        n = len(confs)
        if n == 0:
            return BrierDecomposition(0.0, 0.0, 0.0, 0.0)
        c = np.clip(np.asarray(confs, dtype=float), 0.0, 1.0)
        y = np.asarray(correct, dtype=float)
        base_rate = float(y.mean())
        uncertainty = base_rate * (1.0 - base_rate)

        reliability = 0.0
        resolution = 0.0
        binned_pred = np.zeros(n, dtype=float)
        edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        for k in range(self.num_bins):
            lo, hi = edges[k], edges[k + 1]
            if k == self.num_bins - 1:
                mask = (c >= lo) & (c <= hi)
            else:
                mask = (c >= lo) & (c < hi)
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            conf_k = float(c[mask].mean())
            acc_k = float(y[mask].mean())
            binned_pred[mask] = conf_k
            reliability += n_k * (conf_k - acc_k) ** 2
            resolution += n_k * (acc_k - base_rate) ** 2
        reliability /= n
        resolution /= n
        binned_brier = float(np.mean((binned_pred - y) ** 2))
        return BrierDecomposition(reliability, resolution, uncertainty, binned_brier)


# ---------------------------------------------------------------------------
# Pure helpers (module-level — deterministic functions over numpy arrays)
# ---------------------------------------------------------------------------


def _is_probability(x: float) -> bool:
    """True iff ``x`` is a finite probability in ``[0, 1]`` (not a bare logit)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return np.isfinite(xf) and 0.0 <= xf <= 1.0


def _brier(confs: list[float], correct: list[bool]) -> float:
    """Mean-squared-error Brier score ``mean((conf - correct)²)`` (0.0 if empty)."""
    if not confs:
        return 0.0
    c = np.clip(np.asarray(confs, dtype=float), 0.0, 1.0)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((c - y) ** 2))


def _monotone_risk_coverage(
    confs: list[float], correct: list[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """The monotone (non-decreasing in coverage) selective-risk curve.

    Orders findings by DESCENDING confidence (most-confident first — the natural
    abstention order keeps the most-confident predictions and abstains on the
    least-confident), takes the cumulative error rate at each coverage
    ``k/n``, then applies the running maximum to emit the **monotone envelope**:
    the canonical optimal risk-coverage curve, non-decreasing as coverage grows
    (equivalently: selective risk never increases as you abstain more — NFR-019).
    Ties in confidence are broken stably for determinism (AX-002).

    Returns ``(coverage, risk)`` arrays ordered by ascending coverage. Empty for
    an empty slice.
    """
    n = len(confs)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    c = np.asarray(confs, dtype=float)
    err = (~np.asarray(correct, dtype=bool)).astype(float)
    order = np.argsort(-c, kind="stable")  # descending confidence, stable
    err_sorted = err[order]
    cum_risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    monotone = np.maximum.accumulate(cum_risk)  # non-decreasing in coverage
    coverage = np.arange(1, n + 1, dtype=float) / n
    return coverage, monotone


def _abstention_table(
    coverage: np.ndarray, risk: np.ndarray
) -> tuple[AbstentionOperatingPoint, ...]:
    """The ≥3-point abstention operating-point table at :data:`ABSTENTION_TARGETS`.

    For each target selective-risk ``t`` the achieved coverage is the LARGEST
    coverage whose (monotone-envelope) risk is ≤ ``t``. Because the envelope is
    non-decreasing in coverage, looser targets admit ≥ the coverage of tighter
    ones (NFR-021 / the abstention-table consistency invariant). When no coverage
    meets ``t`` the achieved coverage is ``0.0`` (you would abstain on everything).
    """
    # An empty corpus has no operating points (honest empty table, not 3 zero
    # rows that would imply a measured-but-vacuous abstention surface).
    if len(coverage) == 0:
        return ()
    points: list[AbstentionOperatingPoint] = []
    for t in ABSTENTION_TARGETS:
        qualifying = np.where(risk <= t)[0]
        if len(qualifying) == 0:
            points.append(AbstentionOperatingPoint(t, 0.0, float(risk[0])))
        else:
            i = int(qualifying[-1])  # monotone ⇒ qualifying is a prefix
            points.append(
                AbstentionOperatingPoint(t, float(coverage[i]), float(risk[i]))
            )
    return tuple(points)


def selective_risk_report_to_dict(report: SelectiveRiskReport) -> dict[str, Any]:
    """Serialize a :class:`SelectiveRiskReport` to a JSON-able dict.

    The shape mirrors what the SDO ``_g4_calibration_selective_risk`` gate reads
    off a benchmark system block (per-class ECE + threshold, the AURC, the
    risk-coverage curve, the abstention operating points, and the
    calibrated-confidence-coverage) so a canonical run can emit it directly.
    """
    return {
        "per_class_ece": {k: round(v, 6) for k, v in report.per_class_ece.items()},
        "per_class_ece_post": {
            k: round(v, 6) for k, v in report.per_class_ece_post.items()
        },
        "per_class_ece_threshold": dict(report.per_class_ece_threshold),
        "per_class_brier": {
            k: round(v, 6) for k, v in report.per_class_brier.items()
        },
        "per_class_aurc": {k: round(v, 6) for k, v in report.per_class_aurc.items()},
        "selective_risk_aurc": round(report.global_aurc, 6),
        "risk_coverage_curve": [
            {"coverage": round(p.coverage, 6), "risk": round(p.risk, 6)}
            for p in report.global_risk_coverage
        ],
        "abstention_operating_points": [
            {
                "target_risk": p.target_risk,
                "achieved_coverage": round(p.achieved_coverage, 6),
                "achieved_risk": round(p.achieved_risk, 6),
            }
            for p in report.abstention_table
        ],
        "calibrated_confidence_coverage": round(
            report.calibrated_confidence_coverage, 6
        ),
    }
