"""The assembled assurance report + the numeric no-fabrication gate.

``AssuranceReport.to_dict()`` is where the numeric gate runs: EVERY emitted value
is routed through the public no-fabrication validators. A unit-interval value
(score / rate / p-value / CI bound) that is not finite-and-in-[0,1], or any
non-finite number anywhere in the payload, raises :class:`NoFabricationError`
(fail closed) — a bug or an injected artifact value can never be serialized.

This gate is numeric-integrity ONLY. PII safety is a SEPARATE gate (``pii_egress``)
run on the serialized bytes by the renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pii_anon.eval_framework.validation import finite_unit_score, is_finite_number

from .adjudication import AgreementReport
from .claim_strength import DimensionResult
from .provenance import Provenance


class NoFabricationError(RuntimeError):
    """A value failed numeric-integrity validation and must not be emitted."""


def _unit(value: Any, ctx: str) -> float | None:
    if value is None:
        return None
    v = finite_unit_score(value)
    if v is None:
        raise NoFabricationError(f"non-[0,1] or non-finite value at {ctx}: rejected")
    return v


def _real(value: Any, ctx: str) -> float | None:
    if value is None:
        return None
    if not is_finite_number(value):
        raise NoFabricationError(f"non-finite value at {ctx}: rejected")
    return float(value)


def _finite_int(value: Any, ctx: str) -> int:
    """A finite integer (int form preserved; MAY be negative — e.g. a seed). Rejects a forged
    bool / non-int / non-finite / digit-limit-busting int (the int->str DoV class) fail-closed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise NoFabricationError(f"non-int at {ctx}: rejected")
    ivalue: int = value  # confirmed int; bind before the TypeGuard re-widens `value`
    if not is_finite_number(value):  # rejects the int->str digit-limit DoV (10**5000)
        raise NoFabricationError(f"non-finite int at {ctx}: rejected")
    return ivalue


def _count(value: Any, ctx: str) -> int:
    """A NON-NEGATIVE finite integer count. Rejects (fail-closed) a forged bool / non-int /
    negative / digit-limit-busting int (the int->str DoV class) with a NoFabricationError, so a
    bad count can never serialize a nonsensical value nor crash json.dumps with a raw ValueError."""
    v = _finite_int(value, ctx)
    if v < 0:
        raise NoFabricationError(f"negative count at {ctx}: rejected")
    return v


def _ci(ci: tuple[float, float] | None, ctx: str) -> list[float] | None:
    """Unit-interval CI (for a [0,1] score / rate / agreement)."""
    if ci is None:
        return None
    lo = _unit(ci[0], f"{ctx}.lo")
    hi = _unit(ci[1], f"{ctx}.hi")
    return None if lo is None or hi is None else [lo, hi]


def _signed_unit(value: Any, ctx: str) -> float | None:
    """A signed unit value in [-1, 1] (a paired-F1 DELTA) — finite AND in range. Rejects a
    forged delta outside [-1, 1] (an F1 difference cannot exceed that)."""
    if value is None:
        return None
    if not is_finite_number(value) or not (-1.0 <= float(value) <= 1.0):
        raise NoFabricationError(f"delta out of [-1,1] or non-finite at {ctx}: rejected")
    return float(value)


def _signed_ci(ci: tuple[float, float] | None, ctx: str) -> list[float] | None:
    """Signed CI (for a PAIRED DELTA, natural range [-1, 1]) — range-checked, not clamped
    to [0,1] (a delta CI is legitimately negative when the baseline outperforms the user)."""
    if ci is None:
        return None
    lo = _signed_unit(ci[0], f"{ctx}.lo")
    hi = _signed_unit(ci[1], f"{ctx}.hi")
    return None if lo is None or hi is None else [lo, hi]


# Two independent guards, keyed by the metadata field name:
#  * a BOOL is a forged score (a Python bool IS an int -> renders as 1.0000/0.0000) UNLESS its
#    key is a known FLAG. Fail-closed: reject bool everywhere except the flag allow-list. (The
#    Phase-2 close caught the inverse — an allow-SOME-score-keys design let a forged bool slip
#    under any NEW score key: reid_recall, overall_coverage_ratio, per_standard_coverage.<std>, …
#    Inverting the rule gates every present and future score field automatically.)
#  * a FLOAT is range-checked to [0,1] by DEFAULT (fail-closed: a future [0,1] rate under a
#    new key is gated automatically); only an explicitly-UNBOUNDED key (Cohen's d) is exempt.
_FLAG_KEYS = frozenset({"lower_bound", "require_significance", "type_schema_mismatch"})
_UNBOUNDED_META_KEYS = frozenset({"effect_size_vs_baseline"})
# Int keys that are legitimately UNBOUNDED counts. Every OTHER int-valued entry is treated as a
# [0,1] rate and range-checked, symmetric with the float path (the Phase-2 close caught a forged
# out-of-[0,1] int — e.g. overall_coverage_ratio=42 — slipping through finiteness-only int
# validation while its float twin 42.0 was correctly rejected). Fail-closed: a NEW unbounded
# count omitted here fails LOUDLY on a real run rather than silently admitting a forged int rate.
_COUNT_KEYS = frozenset({
    "support", "per_entity_type_total", "scoreable_records", "unscoreable_records",
    "total_referenced_pii", "leaked", "correct", "n_targets", "n_guesses", "linkable_targets",
    "candidate_set_size", "detected_type_count", "min_support", "min_clusters", "n", "n_powered",
    "only_a", "only_b", "both", "intersection", "union", "ngram_size",
})


def _sanitize(obj: Any, ctx: str, *, key: str | None = None) -> Any:
    """Recursively validate a metadata payload (range-/finiteness-checked per the guards
    above). Rejects out-of-range scores, bool-as-score, and non-finite numbers / huge ints."""
    if isinstance(obj, bool):
        if key not in _FLAG_KEYS:
            raise NoFabricationError(
                f"bool under non-flag key {key!r} at {ctx}: rejected (a bool renders as a 1.0/0.0 score)"
            )
        return obj  # a known flag (lower_bound / require_significance / type_schema_mismatch)
    if isinstance(obj, int):  # finiteness first (rejects 10**5000 via the digit-limit guard)
        if not is_finite_number(obj):
            raise NoFabricationError(f"non-finite int at {ctx}: rejected")
        if key in _COUNT_KEYS:
            if obj < 0:  # a count is non-negative; a forged negative count fails closed
                raise NoFabricationError(f"negative count under {key!r} at {ctx}: rejected")
            return obj
        if key in _UNBOUNDED_META_KEYS:
            return obj  # a legitimately-unbounded (possibly signed) metric
        # any other int is a [0,1] rate (a rate-as-int can validly only be 0 or 1); a forged
        # out-of-range int rate is rejected, symmetric with the float path.
        if finite_unit_score(obj) is None:
            raise NoFabricationError(f"non-[0,1] int under non-count key {key!r} at {ctx}: rejected")
        return obj
    if isinstance(obj, float):
        if key in _UNBOUNDED_META_KEYS:
            if not is_finite_number(obj):
                raise NoFabricationError(f"non-finite number at {ctx}: rejected")
            return obj
        v = finite_unit_score(obj)
        if v is None:
            raise NoFabricationError(f"non-[0,1]/non-finite value at {ctx}: rejected")
        return v
    if isinstance(obj, dict):
        return {k: _sanitize(v, f"{ctx}.{k}", key=k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, f"{ctx}[{i}]", key=key) for i, v in enumerate(obj)]
    return obj


def _result_to_dict(r: DimensionResult) -> dict[str, Any]:
    ctx = f"{r.dimension}.{r.system}"
    return {
        "system": r.system,
        "claim_strength": r.claim_strength.value,
        "value": _unit(r.value, f"{ctx}.value"),
        "confidence_interval": _ci(r.confidence_interval, f"{ctx}.ci"),
        "p_value": _unit(r.p_value, f"{ctx}.p_value"),
        "effect_size": _real(r.effect_size, f"{ctx}.effect_size"),
        "support": _count(r.support, f"{ctx}.support"),
        "reasons": list(r.reasons),
        "caveats": list(r.caveats),
        "metadata": _sanitize(dict(r.metadata), f"{ctx}.metadata"),
    }


@dataclass
class AssuranceReport:
    dataset_name: str
    dataset_fingerprint: str
    mode: str
    reference_kind: str
    n_records: int
    systems: tuple[str, ...]
    baseline: str
    provenance: Provenance
    dimensions: dict[str, dict[str, DimensionResult]]
    detection_comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    agreement: AgreementReport | None = None
    methodology: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    generated_at: str | None = None
    schema_version: str = "assurance-report/1"
    # statistics provenance — makes the CIs / p-values backing a MEASURED label exactly
    # recomputable (spec §11.1/§11.2): resample count, alpha, and the Holm family size.
    n_resamples: int = 0
    alpha: float = 0.05
    comparison_family_size: int = 0
    power_thresholds: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize + run the numeric no-fabrication gate over every value."""
        dims: dict[str, Any] = {}
        for dim, by_system in self.dimensions.items():
            dims[dim] = {sys: _result_to_dict(res) for sys, res in by_system.items()}

        comparisons: dict[str, Any] = {}
        for sys, cmp in self.detection_comparisons.items():
            # A comparison is publishable only if it is MEASURED; otherwise the
            # significance verdict is hedged (significant_holm forced False unless MEASURED).
            claim = cmp.get("claim_strength", "advisory")
            sig = bool(cmp.get("significant_holm", False)) and claim == "measured"
            comparisons[sys] = {
                "claim_strength": claim,
                "delta": _signed_unit(cmp.get("delta"), f"cmp.{sys}.delta"),
                "ci": _signed_ci(cmp.get("ci"), f"cmp.{sys}.ci"),
                "p_value": _unit(cmp.get("p_value"), f"cmp.{sys}.p_value"),
                "p_value_holm": _unit(cmp.get("p_value_holm"), f"cmp.{sys}.p_holm")
                if cmp.get("p_value_holm") is not None else None,
                "significant_holm": sig,
                "significant_holm_raw": bool(cmp.get("significant_holm", False)),
                "effect_size": _real(cmp.get("effect_size"), f"cmp.{sys}.effect"),
                "n": _count(cmp.get("n", 0), f"cmp.{sys}.n"),
            }

        agreement = None
        if self.agreement is not None:
            a = self.agreement
            agreement = {
                "system_a": a.system_a,
                "system_b": a.system_b,
                "span_jaccard": _unit(a.span_jaccard, "agreement.jaccard"),
                "char_kappa": _real(a.char_kappa, "agreement.kappa"),
                "only_a": _count(a.only_a, "agreement.only_a"),
                "only_b": _count(a.only_b, "agreement.only_b"),
                "both": _count(a.both, "agreement.both"),
                "reference_robust": bool(a.reference_robust),
                "ordering_union": list(a.ordering_union),
                "ordering_intersection": list(a.ordering_intersection),
            }

        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "dataset": {
                "name": self.dataset_name,
                "fingerprint": self.dataset_fingerprint,
                "n_records": _count(self.n_records, "dataset.n_records"),
                "mode": self.mode,
                "reference_kind": self.reference_kind,
            },
            "systems": list(self.systems),
            "baseline": self.baseline,
            "provenance": self.provenance.to_dict(),
            "statistics": {
                "bootstrap_resamples": _count(self.n_resamples, "statistics.bootstrap_resamples"),
                "alpha": _unit(self.alpha, "statistics.alpha"),
                "comparison_family_size": _count(self.comparison_family_size, "statistics.comparison_family_size"),
                "bootstrap_unit": "record (cluster)",
                "multiplicity_correction": "holm-bonferroni",
                # seed may be NEGATIVE -> finiteness-only int (keep int form), not _count
                "seed": _finite_int(self.provenance.seed, "statistics.seed"),
                "power_thresholds": _sanitize(dict(self.power_thresholds), "statistics.power_thresholds"),
            },
            "dimensions": dims,
            "detection_comparisons": comparisons,
            "agreement": agreement,
            "methodology": list(self.methodology),
            "limitations": list(self.limitations),
        }
