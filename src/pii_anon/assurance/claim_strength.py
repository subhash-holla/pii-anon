"""The three-valued claim-strength model — the heart of a defendable report.

Every number an assurance report emits carries exactly one :class:`ClaimStrength`:

* ``MEASURED``        — publishable / advertisable. Computed against user-supplied
                        gold labels AND the power gate passes (run-level provenance
                        + adversarial-close gating is applied on top by the runner).
* ``ADVISORY``        — honest but explicitly hedged; never a headline claim.
                        Silver-reference, deterministic stand-in, or under-powered.
* ``NOT_ASSESSABLE``  — required input absent or only one system can supply it.
                        Rendered with the reason; NEVER as 0 or as a pass.

Rules enforced here / by callers (all tested):

* A ``MEASURED`` value MUST trace to gold labels + a passing power cell.
* Silver mode NEVER produces a ``MEASURED`` head-to-head winner.
* A dimension only one system can supply ⇒ the other system is ``NOT_ASSESSABLE``
  and NO "win" is rendered (mirrors the SDO gate's G2 phantom-0 rule).
* ``MEASURED`` and ``ADVISORY`` numbers are never blended into one headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class ClaimStrength(str, Enum):
    """How defensible a single emitted number is."""

    MEASURED = "measured"
    ADVISORY = "advisory"
    NOT_ASSESSABLE = "not_assessable"


class MeasurementMode(str, Enum):
    """Whether the dataset carries gold labels (publishable) or not (advisory)."""

    LABELED = "labeled"
    SILVER = "silver"


@dataclass(frozen=True)
class DimensionResult:
    """One dimension's outcome for one system.

    Holds NO raw user text — only metrics, labels, and reasons — so it is safe to
    serialize. ``__repr__`` is overridden defensively in case ``metadata`` ever
    carries a derived string.
    """

    dimension: str
    system: str
    claim_strength: ClaimStrength
    value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    effect_size: float | None = None
    support: int = 0
    components: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # defensive: never echo metadata contents
        return (
            f"DimensionResult(dimension={self.dimension!r}, system={self.system!r}, "
            f"claim_strength={self.claim_strength.value!r}, value={self.value!r}, "
            f"support={self.support})"
        )

    @property
    def is_publishable(self) -> bool:
        return self.claim_strength is ClaimStrength.MEASURED


# ---------------------------------------------------------------------------
# Constructors — make the three states impossible to confuse
# ---------------------------------------------------------------------------


def not_assessable(dimension: str, system: str, *reasons: str, **kw: Any) -> DimensionResult:
    """A dimension whose required input is absent. Value is ``None`` — never 0."""
    if not reasons:
        reasons = ("required input not available",)
    return DimensionResult(
        dimension=dimension,
        system=system,
        claim_strength=ClaimStrength.NOT_ASSESSABLE,
        value=None,
        reasons=tuple(reasons),
        **kw,
    )


def advisory(
    dimension: str,
    system: str,
    value: float | None,
    *,
    reasons: tuple[str, ...] = (),
    caveats: tuple[str, ...] = (),
    **kw: Any,
) -> DimensionResult:
    """An honest-but-hedged value (silver / stand-in / under-powered)."""
    return DimensionResult(
        dimension=dimension,
        system=system,
        claim_strength=ClaimStrength.ADVISORY,
        value=value,
        reasons=reasons,
        caveats=caveats,
        **kw,
    )


def demote_to_advisory(result: DimensionResult, *reasons: str) -> DimensionResult:
    """Run-level demotion: a ``MEASURED`` value becomes ``ADVISORY`` (carrying the
    extra reasons) when a run-level gate (provenance / harness-close) is not met.
    A non-``MEASURED`` result is returned unchanged."""
    if result.claim_strength is not ClaimStrength.MEASURED:
        return result
    return replace(
        result,
        claim_strength=ClaimStrength.ADVISORY,
        reasons=result.reasons + tuple(reasons),
    )


def measured(
    dimension: str,
    system: str,
    value: float,
    *,
    support: int,
    **kw: Any,
) -> DimensionResult:
    """A publishable value. Callers MUST have established gold + a passing power gate."""
    return DimensionResult(
        dimension=dimension,
        system=system,
        claim_strength=ClaimStrength.MEASURED,
        value=value,
        support=support,
        **kw,
    )


# ---------------------------------------------------------------------------
# The power gate — decides whether a labeled number is strong enough to publish
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerThresholds:
    """The committed bar a labeled value must clear to be ``MEASURED``."""

    min_support: int = 50
    max_ci_halfwidth: float = 0.05
    require_significance: bool = False
    # the resampling-UNIT (record/cluster) floor — distinct from span support. The cluster
    # bootstrap is meaningless at tiny record counts (a 2-record dataset collapses the CI to
    # zero width and forges precision), so a publishable label needs enough clusters.
    min_clusters: int = 20


def power_gate(
    *,
    support: int,
    n_clusters: int,
    confidence_interval: tuple[float, float] | None,
    significant: bool | None,
    thresholds: PowerThresholds,
) -> tuple[bool, tuple[str, ...]]:
    """Is a labeled measurement powered enough to publish?

    Returns ``(ok, reasons)``. ``reasons`` is non-empty only when ``ok`` is False,
    each entry a human-readable demotion cause (e.g. ``"insufficient support: n=12 < 50"``).
    Fail CLOSED: missing support / CI / too few clusters ⇒ not powered.
    """
    reasons: list[str] = []
    if n_clusters < thresholds.min_clusters:
        reasons.append(f"insufficient clusters: n={n_clusters} < {thresholds.min_clusters} records")
    if support < thresholds.min_support:
        reasons.append(f"insufficient support: n={support} < {thresholds.min_support}")
    if confidence_interval is None:
        reasons.append("no confidence interval computed")
    else:
        lo, hi = confidence_interval
        if hi < lo:
            reasons.append("inverted confidence interval (lo > hi)")
        else:
            halfwidth = (hi - lo) / 2.0
            if halfwidth > thresholds.max_ci_halfwidth:
                reasons.append(
                    f"CI too wide: ±{halfwidth:.3f} > ±{thresholds.max_ci_halfwidth}"
                )
    if thresholds.require_significance and not significant:
        reasons.append("difference vs comparator not statistically significant")
    return (not reasons, tuple(reasons))


def resolve(
    *,
    dimension: str,
    system: str,
    mode: MeasurementMode,
    value: float | None,
    support: int,
    n_clusters: int,
    confidence_interval: tuple[float, float] | None,
    significant: bool | None,
    thresholds: PowerThresholds,
    silver_reasons: tuple[str, ...] = (),
    caveats: tuple[str, ...] = (),
) -> DimensionResult:
    """Resolve a computed value into the correct claim-strength.

    * ``value is None`` ⇒ ``NOT_ASSESSABLE``.
    * silver mode ⇒ ``ADVISORY`` (never publishable — silver is circular for a
      head-to-head; see module docstring).
    * labeled + power gate passes ⇒ ``MEASURED``.
    * labeled + power gate fails ⇒ ``ADVISORY`` with the power reasons.
    """
    if value is None:
        return not_assessable(dimension, system, *(silver_reasons or ("value not computed",)))
    if mode is MeasurementMode.SILVER:
        reasons = silver_reasons or ("silver reference: advisory only, no publishable winner",)
        return advisory(
            dimension, system, value,
            reasons=reasons, caveats=caveats, support=support,
            confidence_interval=confidence_interval,
        )
    ok, reasons = power_gate(
        support=support,
        n_clusters=n_clusters,
        confidence_interval=confidence_interval,
        significant=significant,
        thresholds=thresholds,
    )
    if ok:
        return measured(
            dimension, system, value,
            support=support, confidence_interval=confidence_interval, caveats=caveats,
        )
    return advisory(
        dimension, system, value,
        reasons=reasons, caveats=caveats, support=support,
        confidence_interval=confidence_interval,
    )
