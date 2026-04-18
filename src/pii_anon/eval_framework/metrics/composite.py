"""PII-Rate-Elo composite metric for PII de-identification system evaluation.

Combines detection quality, operational efficiency, privacy protection,
utility preservation, fairness, and re-identification resistance into a
single interpretable score [0,1].

**Tier 1 (Competitive Benchmark)** — uses F2 (recall-weighted), F1,
precision, recall, latency, throughput.  F2-based composite weighting
(δ=0.4) directly encodes privacy-first principles: missing PII carries
greater regulatory exposure than false positives.  Available for every
system including competitors.

**Tier 2 (Full Evaluation)** — adds privacy, utility, and fairness sub-scores.
Available when the full ``EvaluationFramework`` pipeline is used.

**Tier 3 (Re-identification Resistance)** — in response to Lermen et al.
(2026), adds the Re-identification Resistance Score (RRS), Quasi-Identifier
Coverage (QIC), and Behavioral Signal Leakage (BSL).  These metrics
evaluate resistance to LLM-powered semantic re-identification attacks
that succeed even after all explicit PII entities are removed.

Theory:
    All component metrics are normalized to [0,1] via monotone bounded
    functions (higher = better).  Sub-scores are combined with configurable
    weights.  Privacy and utility are balanced with an α parameter
    (C = α·P + (1-α)·U) following the PII-Rate-Elo formulation.

    Tier 3 composite::

        C = w_D · D + w_O · O + w_R · RRS

    where D = detection sub-score, O = operational sub-score, RRS = 1 -
    (re-identification recall × precision).  Deployment profiles
    (standard, high-security, high-throughput) select appropriate weights.

Evidence basis:
    - PII-Rate-Elo theory (composite scoring, normalization, privacy-utility α)
    - Bradley-Terry (1952) paired-comparison model
    - TAB 2022 (privacy-utility trade-off, F2 scoring)
    - NIST SP 800-122/800-188 (risk-calibrated thresholds)
    - Lermen et al. (2026) — ESRC pipeline for LLM-based re-identification
    - Repo et al. (2025) — detection-privacy gap

See ``docs/composite-metric-evidence.md`` for full research backing.
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Deployment profile constants (Tier 3 weights)
# ---------------------------------------------------------------------------

DeploymentProfile = Literal["standard", "high_security", "high_throughput"]

# Recommended Tier 3 composite weights: C = w_D · D + w_O · O + w_R · RRS
_DEPLOYMENT_PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "standard": {"w_detection": 0.5, "w_operational": 0.2, "w_reidentification": 0.3},
    "high_security": {"w_detection": 0.3, "w_operational": 0.1, "w_reidentification": 0.6},
    "high_throughput": {"w_detection": 0.4, "w_operational": 0.4, "w_reidentification": 0.2},
}


# ---------------------------------------------------------------------------
# Normalization functions — monotone, bounded to [0,1]
# ---------------------------------------------------------------------------

def normalize_identity(value: float) -> float:
    """Clamp *value* to [0,1].  Used for metrics already in that range
    (F1, precision, recall, privacy/utility/fairness scores)."""
    return max(0.0, min(1.0, value))


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Compute F_beta score from precision and recall.

    F_beta = (1 + β²) · (P · R) / (β² · P + R)

    F2 (β=2) double-weights recall, preferred for privacy-first evaluation
    per TAB 2022: missing PII has greater regulatory exposure than false
    positives.  F0.5 (β=0.5) double-weights precision.

    Parameters
    ----------
    precision:
        Precision in [0, 1].
    recall:
        Recall in [0, 1].
    beta:
        Recall-weighting factor.  β > 1 emphasizes recall; β < 1 emphasizes
        precision.  β = 1 yields the standard F1 score.

    Returns
    -------
    float
        F_beta score in [0, 1].  Returns 0 if both precision and recall
        are zero.
    """
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * (precision * recall) / denominator


def normalize_latency(latency_ms: float, reference_ms: float = 100.0) -> float:
    """Normalize latency (lower is better) via inverse-square sigmoid.

    Formula::

        norm(lat) = 1 / (1 + (lat / ref)²)

    At *reference_ms* the score equals 0.5.  Sub-millisecond latencies
    approach 1.0; very high latencies approach 0.0.
    """
    if latency_ms <= 0.0:
        return 1.0
    ratio = latency_ms / reference_ms
    return 1.0 / (1.0 + ratio * ratio)


def normalize_throughput(
    docs_per_hour: float,
    reference_dph: float = 1_000_000.0,
) -> float:
    """Normalize throughput (higher is better) via hyperbolic sigmoid.

    Formula::

        norm(dph) = dph / (dph + ref)

    At *reference_dph* the score equals 0.5.  Very high throughput
    approaches 1.0; zero throughput yields 0.0.
    """
    if docs_per_hour <= 0.0:
        return 0.0
    return docs_per_hour / (docs_per_hour + reference_dph)


# ---------------------------------------------------------------------------
# Tier 2 adversarial normalization functions (Section 4.4 of the paper)
# ---------------------------------------------------------------------------

def normalize_attack_success_rate(asr: float) -> float:
    """Normalize attack success rate (lower is better).

    Formula::

        N(ASR) = 1 - ASR

    where ASR is the fraction of ground-truth PII entities recovered by
    an adversarial extraction probe.  ASR=0 (perfect defense) → 1.0;
    ASR=1 (complete extraction) → 0.0.

    Reference: Cheng et al. (2025) — Effective PII Extraction from LLMs.
    """
    return max(0.0, min(1.0, 1.0 - asr))


def normalize_mia_auc(auc: float) -> float:
    """Normalize Membership Inference Attack AUC (lower AUC is better).

    Formula::

        N(AUC) = clip(2 · (1 - AUC), 0, 1)

    Maps AUC=0.5 (no adversarial advantage, random guessing) → 1.0
    and AUC=1.0 (perfect attack) → 0.0.

    Reference: Shokri et al. (2017) — Membership Inference Attacks.
    """
    return max(0.0, min(1.0, 2.0 * (1.0 - auc)))


def normalize_canary_exposure(exposure: float, c: float = 5.0) -> float:
    """Normalize canary exposure (lower exposure is better).

    Formula::

        N(E) = exp(-E / c)

    where *c* is a policy-calibrated constant.  E=0 → 1.0 (no exposure);
    large E → 0.0 (high exposure).

    Reference: Carlini et al. (2019) — The Secret Sharer.
    """
    if exposure <= 0.0:
        return 1.0
    return math.exp(-exposure / c)


def normalize_k_anonymity(k: int, k_max: int = 100) -> float:
    """Normalize k-anonymity (higher k is better).

    Formula::

        N(k) = clip(log(k) / log(k_max), 0, 1)

    Logarithmic scaling reflects diminishing marginal privacy gains.
    k=1 → 0.0; k=k_max → 1.0.

    Reference: Sweeney (2002) — k-anonymity.
    """
    if k <= 1:
        return 0.0
    if k_max <= 1:
        return 1.0
    return max(0.0, min(1.0, math.log(k) / math.log(k_max)))


def normalize_epsilon_dp(
    epsilon: float,
    epsilon_0: float = 1.0,
    delta: float | None = None,
    delta_threshold: float = 1e-5,
) -> float:
    """Normalize differential privacy epsilon (lower is better).

    Formula::

        N(ε) = exp(-ε / ε₀)

    with an optional δ gate: if δ > δ_threshold, the score is halved
    to penalize weak (ε, δ)-DP guarantees.

    ε=0 → 1.0 (perfect privacy); large ε → 0.0 (weak privacy).

    Reference: Abadi et al. (2016) — Deep Learning with Differential Privacy.
    """
    if epsilon <= 0.0:
        score = 1.0
    else:
        score = math.exp(-epsilon / epsilon_0)
    # δ gate: penalize if δ exceeds threshold
    if delta is not None and delta > delta_threshold:
        score *= 0.5
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Tier 3 normalization functions — LLM-powered re-identification resistance
# (In response to Lermen et al. 2026; see recommendations-dataset-metric-
# evolution.md)
# ---------------------------------------------------------------------------

def normalize_reidentification_resistance(
    reid_recall: float,
    reid_precision: float,
) -> float:
    """Re-identification Resistance Score (RRS) — Tier 3 primary metric.

    Formula::

        RRS = 1 - (reid_recall × reid_precision)

    Measures resistance to LLM-based semantic re-identification attacks
    (e.g., Lermen et al. 2026 ESRC pipeline) applied to de-identified text.
    *reid_recall* is the fraction of users an attacker successfully matches
    to their true identity; *reid_precision* is the precision of those
    matches.  Higher RRS is better: 1.0 means the attacker achieved zero
    effective re-identification; 0.0 means perfect attack success (both
    recall=1.0 and precision=1.0).

    Example: Lermen et al. report 67% recall × 90% precision on HN→LinkedIn
    matching → RRS = 1 - 0.603 = 0.397 (moderately resistant, but clearly
    vulnerable).

    Parameters
    ----------
    reid_recall:
        Attack recall in [0, 1].
    reid_precision:
        Attack precision in [0, 1].

    Returns
    -------
    float
        RRS in [0, 1].  Higher = better privacy.

    Reference: Lermen et al. (2026) — Large-scale online deanonymization
    with LLMs.
    """
    reid_recall = normalize_identity(reid_recall)
    reid_precision = normalize_identity(reid_precision)
    # Both inputs are clamped to [0,1] above, so the product is in [0,1]
    # and ``1 - product`` is already in [0,1] — no outer clamp needed.
    return 1.0 - reid_recall * reid_precision


def normalize_quasi_identifier_coverage(
    removed_signals: int,
    total_signals: int,
    signal_weights: list[float] | None = None,
) -> float:
    """Quasi-Identifier Coverage (QIC) — fraction of QIs removed/obfuscated.

    Formula (unweighted)::

        QIC = removed / total

    Formula (weighted by re-identification contribution)::

        QIC = Σ(w_i · removed_i) / Σ(w_i · total_i)

    Measures what fraction of quasi-identifier signals (writing style,
    professional domain, interests, temporal patterns, implicit location
    references, personal anecdotes) a system removes or obfuscates during
    de-identification.  Current entity-only systems score 0.0 on this
    metric because they ignore quasi-identifiers entirely.

    Parameters
    ----------
    removed_signals:
        Number of quasi-identifier signals successfully removed/obfuscated.
    total_signals:
        Total number of quasi-identifier signals in the original text.
    signal_weights:
        Optional per-signal weights reflecting re-identification
        contribution (e.g., writing style > topic mentions).  If provided,
        must have len(signal_weights) == total_signals and *removed_signals*
        is interpreted as a sum of indicator-weighted contributions (use
        ``weighted_coverage`` helper for proper construction).

    Returns
    -------
    float
        QIC in [0, 1].  Higher = more quasi-identifiers removed.

    Reference: Lermen et al. (2026) — ESRC demonstrates LLMs extract
    identity features from behavioral signals after PII removal.
    """
    if total_signals <= 0:
        return 0.0
    if signal_weights is not None:
        total_weight = sum(signal_weights)
        if total_weight <= 0.0:
            return 0.0
        # Assume removed_signals is now a weighted sum of indicator × weight.
        return max(0.0, min(1.0, removed_signals / total_weight))
    return max(0.0, min(1.0, removed_signals / total_signals))


def normalize_behavioral_signal_leakage(
    embedding_similarity: float,
) -> float:
    """Behavioral Signal Leakage (BSL) — resistance to embedding-level leaks.

    Formula::

        BSL = 1 - embedding_similarity

    Measures how much behavioral/stylometric information leaks through
    de-identification by comparing embedding similarity between the
    original and de-identified text using a stylometry-trained embedding
    model.  High similarity = high leakage = low BSL score.

    Parameters
    ----------
    embedding_similarity:
        Cosine similarity in [0, 1] between original and de-identified
        text embeddings (stylometry model).

    Returns
    -------
    float
        BSL in [0, 1].  Higher = less behavioral information leaked.
    """
    embedding_similarity = normalize_identity(embedding_similarity)
    # Input is clamped to [0,1] above, so ``1 - x`` is already in [0,1].
    return 1.0 - embedding_similarity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompositeConfig:
    """Configuration for composite metric computation.

    **Tier 1 weights** (detection + efficiency) are used for all systems.
    **Tier 2 weights** (privacy, utility, fairness) default to 0 — set them
    to non-zero values when the full evaluation pipeline is used.
    **Tier 3 weights** (re-identification resistance, quasi-identifier
    coverage, behavioral signal leakage) default to 0 — enable them for
    LLM-era privacy evaluation per Lermen et al. (2026).

    The final score is the weighted average of all non-zero-weighted
    components.  Weights do **not** need to sum to 1; they are
    automatically normalized during computation.

    **F2-based composite weighting** (Paper v9):
        Setting ``weight_detection_f2 > 0`` encodes privacy-first principles
        directly.  Recommended default: weight_f2=0.40, weight_p=0.10,
        weight_r=0.20, weight_latency=0.15, weight_throughput=0.15.
        Use ``CompositeConfig.f2_privacy_first()`` for this preset.

    **Deployment profiles** (Tier 3):
        ``CompositeConfig.for_deployment("high_security")`` produces a
        config with w_D=0.3, w_O=0.1, w_R=0.6 distributed across the
        underlying component weights, suitable for public data release.
    """

    # Privacy-utility balance for Tier 2: C_pu = α·P + (1-α)·U
    alpha_privacy: float = 0.5

    # β controls Tier 1 vs Tier 2 balance:
    #   C_s = β · C_s^(1) + (1-β) · C_s^(2)
    # β=1.0 means Tier 1 only (default); β=0.5 means equal weight.
    beta_tier_balance: float = 1.0

    # F_beta parameter for F2-based composite (β=2 double-weights recall).
    fbeta_beta: float = 2.0

    # --- Tier 1: detection quality ---
    # Classical F1-weighted config is the default for backward compatibility.
    # Use ``CompositeConfig.f2_privacy_first()`` for F2-based weighting.
    weight_detection_f1: float = 0.50
    weight_detection_f2: float = 0.0  # Enable with f2_privacy_first() preset.
    weight_detection_precision: float = 0.15
    weight_detection_recall: float = 0.15

    # --- Tier 1: operational efficiency ---
    weight_latency: float = 0.10
    weight_throughput: float = 0.10

    # --- Tier 1: entity-type coverage (optional, default off) ---
    weight_coverage: float = 0.0

    # --- Tier 2: optional sub-scores (off by default) ---
    weight_privacy: float = 0.0
    weight_utility: float = 0.0
    weight_fairness: float = 0.0

    # --- Tier 3: LLM-era re-identification resistance (off by default) ---
    # Set via ``CompositeConfig.for_deployment(...)`` or manually.
    weight_reidentification_resistance: float = 0.0
    weight_quasi_identifier_coverage: float = 0.0
    weight_behavioral_signal_leakage: float = 0.0

    # --- Reference values for normalization ---
    reference_latency_ms: float = 100.0
    reference_throughput_dph: float = 1_000_000.0

    # --- Tier 2 reference values ---
    reference_canary_c: float = 5.0
    reference_k_max: int = 100
    reference_epsilon_0: float = 1.0
    reference_delta_threshold: float = 1e-5

    # --- Deployment profile (informational; affects defaults in classmethod) ---
    deployment_profile: DeploymentProfile | None = None

    # --- Floor gates configuration ---
    floor_gates: "FloorGateConfig" = field(default_factory=lambda: FloorGateConfig())

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def f2_privacy_first(cls) -> "CompositeConfig":
        """F2-based composite weighting preset (Paper v9).

        Weights: F2=0.40, P=0.10, R=0.20, latency=0.15, throughput=0.15.
        F2 (β=2) double-weights recall, encoding privacy-first principles
        per TAB 2022: missing PII creates greater regulatory exposure than
        false positives.

        Returns
        -------
        CompositeConfig
            Preset configured for privacy-first F2-based evaluation.
        """
        return cls(
            weight_detection_f1=0.0,
            weight_detection_f2=0.40,
            weight_detection_precision=0.10,
            weight_detection_recall=0.20,
            weight_latency=0.15,
            weight_throughput=0.15,
        )

    @classmethod
    def for_deployment(cls, profile: DeploymentProfile) -> "CompositeConfig":
        """Tier 3 deployment profile preset.

        Three profiles (from ``recommendations-dataset-metric-evolution.md``):

        - ``"standard"``:    w_D=0.5, w_O=0.2, w_R=0.3 — balanced.
        - ``"high_security"``: w_D=0.3, w_O=0.1, w_R=0.6 — public data
          release, emphasizes re-identification resistance.
        - ``"high_throughput"``: w_D=0.4, w_O=0.4, w_R=0.2 — streaming
          pipelines, emphasizes operational efficiency.

        Within each bundle, weights are distributed across component
        metrics: detection → (F2, P, R, coverage); operational → (latency,
        throughput); re-identification → (RRS, QIC, BSL).

        Parameters
        ----------
        profile:
            One of ``"standard"``, ``"high_security"``, ``"high_throughput"``.

        Returns
        -------
        CompositeConfig
            Config preset for the given deployment context.

        Raises
        ------
        ValueError
            If *profile* is not a known deployment profile.
        """
        if profile not in _DEPLOYMENT_PROFILE_WEIGHTS:
            raise ValueError(
                f"Unknown deployment profile {profile!r}.  "
                f"Must be one of {sorted(_DEPLOYMENT_PROFILE_WEIGHTS)}."
            )
        bundle = _DEPLOYMENT_PROFILE_WEIGHTS[profile]
        w_D = bundle["w_detection"]
        w_O = bundle["w_operational"]
        w_R = bundle["w_reidentification"]

        # Distribute detection weight across F2, P, R, coverage.
        # F2 gets the lion's share (privacy-first), then R, P, coverage.
        return cls(
            deployment_profile=profile,
            weight_detection_f1=0.0,
            weight_detection_f2=w_D * 0.55,
            weight_detection_precision=w_D * 0.15,
            weight_detection_recall=w_D * 0.20,
            weight_coverage=w_D * 0.10,
            weight_latency=w_O * 0.50,
            weight_throughput=w_O * 0.50,
            # Tier 3: RRS gets the primary weight, QIC and BSL complement.
            weight_reidentification_resistance=w_R * 0.60,
            weight_quasi_identifier_coverage=w_R * 0.25,
            weight_behavioral_signal_leakage=w_R * 0.15,
        )

    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if weights or alpha/beta are invalid."""
        if not 0.0 <= self.alpha_privacy <= 1.0:
            raise ValueError(
                f"alpha_privacy must be in [0, 1], got {self.alpha_privacy}"
            )
        if not 0.0 <= self.beta_tier_balance <= 1.0:
            raise ValueError(
                f"beta_tier_balance must be in [0, 1], got {self.beta_tier_balance}"
            )
        if self.fbeta_beta <= 0.0:
            raise ValueError(
                f"fbeta_beta must be > 0, got {self.fbeta_beta}"
            )
        for attr in (
            "weight_detection_f1",
            "weight_detection_f2",
            "weight_detection_precision",
            "weight_detection_recall",
            "weight_latency",
            "weight_throughput",
            "weight_coverage",
            "weight_privacy",
            "weight_utility",
            "weight_fairness",
            "weight_reidentification_resistance",
            "weight_quasi_identifier_coverage",
            "weight_behavioral_signal_leakage",
        ):
            val = getattr(self, attr)
            if val < 0.0:
                raise ValueError(f"{attr} must be >= 0, got {val}")
        if self.reference_latency_ms <= 0.0:
            raise ValueError(
                f"reference_latency_ms must be > 0, got {self.reference_latency_ms}"
            )
        if self.reference_throughput_dph <= 0.0:
            raise ValueError(
                f"reference_throughput_dph must be > 0, got {self.reference_throughput_dph}"
            )

    @property
    def total_weight(self) -> float:
        """Sum of all active weights."""
        return (
            self.weight_detection_f1
            + self.weight_detection_f2
            + self.weight_detection_precision
            + self.weight_detection_recall
            + self.weight_latency
            + self.weight_throughput
            + self.weight_coverage
            + self.weight_privacy
            + self.weight_utility
            + self.weight_fairness
            + self.weight_reidentification_resistance
            + self.weight_quasi_identifier_coverage
            + self.weight_behavioral_signal_leakage
        )

    @property
    def tier1_weight(self) -> float:
        """Sum of Tier 1 weights only."""
        return (
            self.weight_detection_f1
            + self.weight_detection_f2
            + self.weight_detection_precision
            + self.weight_detection_recall
            + self.weight_latency
            + self.weight_throughput
            + self.weight_coverage
        )

    @property
    def tier2_weight(self) -> float:
        """Sum of Tier 2 weights only."""
        return self.weight_privacy + self.weight_utility + self.weight_fairness

    @property
    def tier3_weight(self) -> float:
        """Sum of Tier 3 weights only (re-identification resistance)."""
        return (
            self.weight_reidentification_resistance
            + self.weight_quasi_identifier_coverage
            + self.weight_behavioral_signal_leakage
        )


# ---------------------------------------------------------------------------
# Floor gate configuration and result
# ---------------------------------------------------------------------------

@dataclass
class FloorGateConfig:
    """Minimum thresholds that must be met. If any fails, composite score is capped.

    Attributes
    ----------
    min_f1:
        Minimum F1 score threshold [0, 1].
    min_privacy:
        Minimum privacy score threshold [0, 1].
    min_fairness:
        Minimum fairness score threshold [0, 1].
    min_entity_coverage:
        Minimum entity coverage threshold [0, 1].
    cap_score:
        Score cap applied when any gate fails (default 0.40).
    enabled:
        Whether floor gates are active.  Disabled by default for backward
        compatibility — set ``enabled=True`` when you want safety guardrails.
    """
    min_f1: float = 0.60
    min_privacy: float = 0.70
    min_fairness: float = 0.50
    min_entity_coverage: float = 0.80
    min_f2: float = 0.0
    cap_score: float = 0.40
    enabled: bool = False

    @classmethod
    def industry_leadership(cls) -> "FloorGateConfig":
        """Return the paper v10 industry-leadership floor-gate preset.

        Paper v10 §4.1.5 states that a system qualifies as a
        next-generation benchmark leader only when it clears all of:
        ``F1 ≥ 0.60``, ``F2 ≥ 0.65`` (privacy-first β=2 score),
        ``entity coverage ≥ 0.80``, ``fairness ≥ 0.50``.  The overall
        composite must additionally reach ``≥ 0.75``.  Pair this with
        :meth:`GovernanceThresholds.industry_leadership` to gate on
        both the metric layer and the Elo layer.
        """
        return cls(
            min_f1=0.60,
            min_f2=0.65,
            min_privacy=0.70,
            min_fairness=0.50,
            min_entity_coverage=0.80,
            cap_score=0.40,
            enabled=True,
        )


@dataclass
class FloorGateResult:
    """Result of floor gate evaluation.

    Attributes
    ----------
    all_passed:
        Whether all enabled gates passed.
    gates:
        Dictionary mapping gate name to {threshold, actual, passed}.
    capped:
        Whether the score was capped due to gate failures.
    cap_score:
        The cap applied (if capped=True).
    remediation:
        List of remediation suggestions for failed gates.
    """
    all_passed: bool
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    capped: bool = False
    cap_score: float = 0.40
    remediation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "all_passed": self.all_passed,
            "gates": self.gates,
            "capped": self.capped,
            "cap_score": round(self.cap_score, 6),
            "remediation": self.remediation,
        }


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompositeScore:
    """Result of a composite metric computation.

    *score* is the primary output — a single [0,1] value.
    Sub-scores and component breakdowns are provided for interpretability.
    """

    score: float

    # Sub-scores
    detection_sub: float = 0.0
    efficiency_sub: float = 0.0
    privacy_sub: float = 0.0
    utility_sub: float = 0.0
    fairness_sub: float = 0.0
    # Tier 3 sub-score: re-identification resistance (RRS + QIC + BSL)
    reidentification_sub: float = 0.0

    # All normalized component values
    components: dict[str, float] = field(default_factory=dict)

    # Raw input values (for audit trail)
    raw_inputs: dict[str, float] = field(default_factory=dict)

    # Configuration used
    config: CompositeConfig = field(default_factory=CompositeConfig)

    # Floor gate result
    floor_gate_result: FloorGateResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "score": round(self.score, 6),
            "detection_sub": round(self.detection_sub, 6),
            "efficiency_sub": round(self.efficiency_sub, 6),
            "privacy_sub": round(self.privacy_sub, 6),
            "utility_sub": round(self.utility_sub, 6),
            "fairness_sub": round(self.fairness_sub, 6),
            "reidentification_sub": round(self.reidentification_sub, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "raw_inputs": {k: round(v, 6) for k, v in self.raw_inputs.items()},
        }
        if self.config.deployment_profile is not None:
            result["deployment_profile"] = self.config.deployment_profile
        if self.floor_gate_result is not None:
            result["floor_gate_result"] = self.floor_gate_result.to_dict()
        return result


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------

def _weighted_avg(values: list[tuple[float, float]]) -> float:
    """Weighted average of *(weight, value)* pairs.  Returns 0 if total weight is 0."""
    total_w = sum(w for w, _ in values)
    if total_w <= 0.0:
        return 0.0
    return sum(w * v for w, v in values) / total_w


def normalize_entity_coverage(
    detected_types: int,
    total_types: int,
) -> float:
    """Normalize entity-type coverage breadth.

    Formula::

        norm(coverage) = detected / total

    A system detecting 13 of 13 entity types → 1.0; one detecting
    1 of 13 → 0.077.  This directly addresses the entity-type blind
    spot identified in Section 1 of the paper.
    """
    if total_types <= 0:
        return 0.0
    return max(0.0, min(1.0, detected_types / total_types))


def evaluate_floor_gates(
    f1: float,
    privacy_score: float,
    fairness_score: float,
    entity_coverage: float,
    config: FloorGateConfig,
    *,
    f2: float | None = None,
) -> FloorGateResult:
    """Evaluate floor gates and generate remediation suggestions.

    Parameters
    ----------
    f1:
        F1 score [0, 1].
    privacy_score:
        Privacy score [0, 1].
    fairness_score:
        Fairness score [0, 1].
    entity_coverage:
        Entity coverage score [0, 1].
    config:
        FloorGateConfig with thresholds.
    f2:
        Optional F2 (β=2, privacy-first) score [0, 1].  Only enforced
        when both *f2* is supplied and ``config.min_f2 > 0`` (the paper
        v10 industry-leadership preset sets this).  Callers that do
        not compute F2 can omit it; older floor-gate configs default
        ``min_f2=0`` so this parameter becomes a no-op.

    Returns
    -------
    FloorGateResult
        Result with per-gate pass/fail and remediation suggestions.
    """
    gates: dict[str, dict[str, Any]] = {}
    failed_gates: list[str] = []
    remediation: list[str] = []

    # F1 gate
    f1_passed = f1 >= config.min_f1
    gates["f1"] = {
        "threshold": config.min_f1,
        "actual": round(f1, 6),
        "passed": f1_passed,
    }
    if not f1_passed:
        failed_gates.append("f1")
        remediation.append(
            f"Improve PII detection F1 score (current {f1:.3f}, target {config.min_f1:.3f})"
        )

    # F2 gate — only enforced when both a threshold and a score are
    # supplied (paper v10 §4.1.5 industry-leadership preset).
    if config.min_f2 > 0 and f2 is not None:
        f2_passed = f2 >= config.min_f2
        gates["f2"] = {
            "threshold": config.min_f2,
            "actual": round(f2, 6),
            "passed": f2_passed,
        }
        if not f2_passed:
            failed_gates.append("f2")
            remediation.append(
                f"Lift privacy-first F2 (current {f2:.3f}, target {config.min_f2:.3f}) — "
                "missed entities carry more regulatory weight than false positives"
            )

    # Privacy gate
    privacy_passed = privacy_score >= config.min_privacy
    gates["privacy"] = {
        "threshold": config.min_privacy,
        "actual": round(privacy_score, 6),
        "passed": privacy_passed,
    }
    if not privacy_passed:
        failed_gates.append("privacy")
        remediation.append(
            f"Enhance privacy protection (current {privacy_score:.3f}, target {config.min_privacy:.3f})"
        )

    # Fairness gate
    fairness_passed = fairness_score >= config.min_fairness
    gates["fairness"] = {
        "threshold": config.min_fairness,
        "actual": round(fairness_score, 6),
        "passed": fairness_passed,
    }
    if not fairness_passed:
        failed_gates.append("fairness")
        remediation.append(
            f"Improve fairness across demographics (current {fairness_score:.3f}, target {config.min_fairness:.3f})"
        )

    # Entity coverage gate
    coverage_passed = entity_coverage >= config.min_entity_coverage
    gates["entity_coverage"] = {
        "threshold": config.min_entity_coverage,
        "actual": round(entity_coverage, 6),
        "passed": coverage_passed,
    }
    if not coverage_passed:
        failed_gates.append("entity_coverage")
        remediation.append(
            f"Expand entity type coverage (current {entity_coverage:.3f}, target {config.min_entity_coverage:.3f})"
        )

    all_passed = len(failed_gates) == 0

    return FloorGateResult(
        all_passed=all_passed,
        gates=gates,
        capped=not all_passed,
        cap_score=config.cap_score,
        remediation=remediation,
    )


# Cached default config for the common ``config=None`` path — avoids
# allocating a fresh ``CompositeConfig`` (and nested ``FloorGateConfig``)
# plus running ``validate()`` on every call.  Known-valid, read-only in
# practice: consumers always pass an explicit config when customizing.
_DEFAULT_CONFIG = CompositeConfig()


def compute_composite(
    f1: float,
    precision: float,
    recall: float,
    latency_ms: float,
    docs_per_hour: float,
    *,
    privacy_score: float = 0.0,
    utility_score: float = 0.0,
    fairness_score: float = 0.0,
    entity_types_detected: int = 0,
    entity_types_total: int = 0,
    # Tier 3: re-identification resistance (LLM-era privacy)
    reidentification_recall: float | None = None,
    reidentification_precision: float | None = None,
    quasi_identifiers_removed: int | None = None,
    quasi_identifiers_total: int | None = None,
    behavioral_signal_similarity: float | None = None,
    config: CompositeConfig | None = None,
) -> CompositeScore:
    """Compute composite score from raw metric values.

    **Tier 1** inputs (required): *f1*, *precision*, *recall*, *latency_ms*,
    *docs_per_hour*.  Optionally *entity_types_detected* / *entity_types_total*
    for coverage scoring.  F2 is computed automatically from precision
    and recall when ``config.weight_detection_f2 > 0``.

    **Tier 2** inputs (optional): *privacy_score*, *utility_score*,
    *fairness_score* — all expected in [0, 1].

    **Tier 3** inputs (optional, LLM-era re-identification resistance):
        - *reidentification_recall* / *reidentification_precision*:
          attack metrics from running an ESRC-style pipeline on the
          de-identified output (Lermen et al. 2026).  Composed into the
          Re-identification Resistance Score (RRS).
        - *quasi_identifiers_removed* / *quasi_identifiers_total*:
          counts of quasi-identifier signals removed.  Composed into
          the Quasi-Identifier Coverage (QIC) metric.
        - *behavioral_signal_similarity*: embedding similarity in [0, 1]
          between original and de-identified text using a stylometry
          model.  Composed into Behavioral Signal Leakage (BSL).

    When both Tier 1 and Tier 2 are provided, the final score uses the
    β-weighted formula from the paper::

        C_s = β · C_s^(1) + (1-β) · C_s^(2)

    where β = ``config.beta_tier_balance`` (default 1.0 = Tier 1 only).

    Parameters
    ----------
    f1, precision, recall:
        PII detection quality metrics, each in [0, 1].
    latency_ms:
        Median (p50) processing latency in milliseconds.
    docs_per_hour:
        Processing throughput.
    privacy_score:
        Privacy protection score in [0, 1] (higher = better privacy).
    utility_score:
        Utility preservation score in [0, 1].
    fairness_score:
        Fairness score in [0, 1] (higher = more equitable).
    entity_types_detected:
        Number of entity types with non-zero recall.
    entity_types_total:
        Total number of entity types in the benchmark.
    reidentification_recall:
        LLM-based attack recall on de-identified text (lower = better).
    reidentification_precision:
        LLM-based attack precision on de-identified text.
    quasi_identifiers_removed:
        Number of quasi-identifier signals removed/obfuscated.
    quasi_identifiers_total:
        Total number of quasi-identifier signals present originally.
    behavioral_signal_similarity:
        Cosine similarity [0, 1] between original and de-identified text
        embeddings from a stylometry model (higher = more leakage).
    config:
        Weighting and normalization configuration.  If ``None``, the
        default ``CompositeConfig`` is used (Tier 1 only).

    Returns
    -------
    CompositeScore
        Composite result with overall score, sub-scores, and full breakdown.
    """
    if config is None:
        # ``_DEFAULT_CONFIG`` is a cached, known-valid singleton so we
        # skip ``validate()`` here.  A shallow copy is handed to the
        # returned :class:`CompositeScore` below (see ``config=``) so
        # callers who mutate ``result.config`` cannot leak changes back
        # into the module-level singleton.
        cfg = _DEFAULT_CONFIG
    else:
        cfg = config
        cfg.validate()

    # ── Normalize Tier 1 components ───────────────────────────────────
    n_f1 = normalize_identity(f1)
    n_prec = normalize_identity(precision)
    n_rec = normalize_identity(recall)
    n_f2 = fbeta_score(n_prec, n_rec, beta=cfg.fbeta_beta)
    n_lat = normalize_latency(latency_ms, cfg.reference_latency_ms)
    n_thr = normalize_throughput(docs_per_hour, cfg.reference_throughput_dph)
    n_cov = normalize_entity_coverage(entity_types_detected, entity_types_total)

    # ── Normalize Tier 2 components ───────────────────────────────────
    n_priv = normalize_identity(privacy_score)
    n_util = normalize_identity(utility_score)
    n_fair = normalize_identity(fairness_score)

    # ── Normalize Tier 3 components (default to 0 if inputs missing) ──
    if reidentification_recall is not None and reidentification_precision is not None:
        n_rrs = normalize_reidentification_resistance(
            reidentification_recall, reidentification_precision,
        )
    else:
        n_rrs = 0.0
    if quasi_identifiers_removed is not None and quasi_identifiers_total is not None:
        n_qic = normalize_quasi_identifier_coverage(
            quasi_identifiers_removed, quasi_identifiers_total,
        )
    else:
        n_qic = 0.0
    if behavioral_signal_similarity is not None:
        n_bsl = normalize_behavioral_signal_leakage(behavioral_signal_similarity)
    else:
        n_bsl = 0.0

    # ── Build component pairs once, reuse across sub-scores ───────────
    # Avoids rebuilding the same (weight, value) tuples 3× below.
    detection_pairs: list[tuple[float, float]] = [
        (cfg.weight_detection_f1, n_f1),
        (cfg.weight_detection_f2, n_f2),
        (cfg.weight_detection_precision, n_prec),
        (cfg.weight_detection_recall, n_rec),
    ]
    efficiency_pairs: list[tuple[float, float]] = [
        (cfg.weight_latency, n_lat),
        (cfg.weight_throughput, n_thr),
    ]

    # ── Tier 1 sub-scores ─────────────────────────────────────────────
    detection_sub = _weighted_avg(detection_pairs)
    efficiency_sub = _weighted_avg(efficiency_pairs)

    # ── Tier 1 composite: C_s^(1) ────────────────────────────────────
    tier1_components: list[tuple[float, float]] = detection_pairs + efficiency_pairs
    if cfg.weight_coverage > 0:
        tier1_components.append((cfg.weight_coverage, n_cov))
    tier1_score = _weighted_avg(tier1_components)

    # ── Tier 2 sub-scores ─────────────────────────────────────────────
    # Privacy-utility: C_pu = α·P + (1-α)·U
    privacy_utility_sub = (
        cfg.alpha_privacy * n_priv + (1.0 - cfg.alpha_privacy) * n_util
    )

    # Tier 2 composite: C_s^(2) — weighted average of privacy-utility
    # and fairness
    tier2_components: list[tuple[float, float]] = []
    pu_weight = cfg.weight_privacy + cfg.weight_utility
    if pu_weight > 0:
        tier2_components.append((pu_weight, privacy_utility_sub))
    if cfg.weight_fairness > 0:
        tier2_components.append((cfg.weight_fairness, n_fair))
    tier2_score = _weighted_avg(tier2_components) if tier2_components else 0.0

    # ── Tier 3 sub-score: re-identification resistance ────────────────
    tier3_components: list[tuple[float, float]] = [
        (cfg.weight_reidentification_resistance, n_rrs),
        (cfg.weight_quasi_identifier_coverage, n_qic),
        (cfg.weight_behavioral_signal_leakage, n_bsl),
    ]
    reidentification_sub = _weighted_avg(tier3_components)

    # ── Overall composite ─────────────────────────────────────────────
    # When Tier 3 weights are active, the composite is a unified weighted
    # average across all tiers (simpler and more interpretable than
    # nested β-balanced tiers).  When only Tier 1 + Tier 2 are active,
    # we preserve the original β-weighted formula for backward compat.
    has_tier2 = cfg.tier2_weight > 0
    has_tier3 = cfg.tier3_weight > 0

    if has_tier3:
        # Unified weighted average across all active components — reuse
        # the pair lists we've already built to avoid re-allocating tuples.
        all_components: list[tuple[float, float]] = [
            *detection_pairs,
            *efficiency_pairs,
            (cfg.weight_coverage, n_cov),
            (cfg.weight_privacy, n_priv),
            (cfg.weight_utility, n_util),
            (cfg.weight_fairness, n_fair),
            *tier3_components,
        ]
        overall = _weighted_avg(all_components)
    else:
        beta = cfg.beta_tier_balance
        if has_tier2 and beta < 1.0:
            overall = beta * tier1_score + (1.0 - beta) * tier2_score
        else:
            overall = tier1_score

    # ── Evaluate floor gates ───────────────────────────────────────────
    floor_gate_result = None
    final_score = round(overall, 6)

    if cfg.floor_gates.enabled:
        floor_gate_result = evaluate_floor_gates(
            n_f1, n_priv, n_fair, n_cov, cfg.floor_gates, f2=n_f2,
        )
        if floor_gate_result.capped:
            final_score = round(cfg.floor_gates.cap_score, 6)

    return CompositeScore(
        score=final_score,
        detection_sub=round(detection_sub, 6),
        efficiency_sub=round(efficiency_sub, 6),
        privacy_sub=round(n_priv, 6),
        utility_sub=round(n_util, 6),
        fairness_sub=round(n_fair, 6),
        reidentification_sub=round(reidentification_sub, 6),
        components={
            "f1_normalized": round(n_f1, 6),
            "f2_normalized": round(n_f2, 6),
            "precision_normalized": round(n_prec, 6),
            "recall_normalized": round(n_rec, 6),
            "latency_normalized": round(n_lat, 6),
            "throughput_normalized": round(n_thr, 6),
            "coverage_normalized": round(n_cov, 6),
            "privacy_normalized": round(n_priv, 6),
            "utility_normalized": round(n_util, 6),
            "fairness_normalized": round(n_fair, 6),
            "reidentification_resistance_normalized": round(n_rrs, 6),
            "quasi_identifier_coverage_normalized": round(n_qic, 6),
            "behavioral_signal_leakage_normalized": round(n_bsl, 6),
            "tier1_score": round(tier1_score, 6),
            "tier2_score": round(tier2_score, 6),
            "tier3_score": round(reidentification_sub, 6),
        },
        raw_inputs={
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "latency_ms": latency_ms,
            "docs_per_hour": docs_per_hour,
            "entity_types_detected": float(entity_types_detected),
            "entity_types_total": float(entity_types_total),
            "privacy_score": privacy_score,
            "utility_score": utility_score,
            "fairness_score": fairness_score,
            "reidentification_recall": (
                reidentification_recall if reidentification_recall is not None else 0.0
            ),
            "reidentification_precision": (
                reidentification_precision if reidentification_precision is not None else 0.0
            ),
            "quasi_identifiers_removed": float(quasi_identifiers_removed or 0),
            "quasi_identifiers_total": float(quasi_identifiers_total or 0),
            "behavioral_signal_similarity": (
                behavioral_signal_similarity if behavioral_signal_similarity is not None else 0.0
            ),
        },
        # Defensive copy: when ``cfg`` is the module-level
        # ``_DEFAULT_CONFIG`` singleton, a mutating caller must not be
        # able to corrupt it through ``result.config``.
        config=dataclasses.replace(cfg) if cfg is _DEFAULT_CONFIG else cfg,
        floor_gate_result=floor_gate_result,
    )


def compute_composite_from_benchmark_result(
    result: Any,
    config: CompositeConfig | None = None,
) -> CompositeScore:
    """Compute composite from a ``SystemBenchmarkResult``.

    Convenience function for integration with the competitor comparison
    pipeline.  Reads ``precision``, ``recall``, ``f1``, ``latency_p50_ms``,
    and ``docs_per_hour`` from the result object.  Also forwards optional
    Tier 2/Tier 3 fields when present: ``privacy_score``, ``utility_score``,
    ``fairness_score``, ``entity_types_detected``, ``entity_types_total``,
    ``reidentification_recall``, ``reidentification_precision``,
    ``quasi_identifiers_removed``, ``quasi_identifiers_total``,
    ``behavioral_signal_similarity``.

    Parameters
    ----------
    result:
        A ``SystemBenchmarkResult`` (or any object with the required
        attributes).
    config:
        Weighting and normalization configuration.

    Returns
    -------
    CompositeScore
    """
    return compute_composite(
        f1=getattr(result, "f1", 0.0),
        precision=getattr(result, "precision", 0.0),
        recall=getattr(result, "recall", 0.0),
        latency_ms=getattr(result, "latency_p50_ms", 0.0),
        docs_per_hour=getattr(result, "docs_per_hour", 0.0),
        privacy_score=getattr(result, "privacy_score", 0.0),
        utility_score=getattr(result, "utility_score", 0.0),
        fairness_score=getattr(result, "fairness_score", 0.0),
        entity_types_detected=getattr(result, "entity_types_detected", 0),
        entity_types_total=getattr(result, "entity_types_total", 0),
        reidentification_recall=getattr(result, "reidentification_recall", None),
        reidentification_precision=getattr(result, "reidentification_precision", None),
        quasi_identifiers_removed=getattr(result, "quasi_identifiers_removed", None),
        quasi_identifiers_total=getattr(result, "quasi_identifiers_total", None),
        behavioral_signal_similarity=getattr(result, "behavioral_signal_similarity", None),
        config=config,
    )


# ---------------------------------------------------------------------------
# Pareto Frontier Analysis
# ---------------------------------------------------------------------------

class ParetoFrontierAnalyzer:
    """Pareto frontier analysis for privacy-utility tradeoff.

    Analyzes multiple systems in terms of privacy vs. utility scores
    and identifies Pareto-optimal systems (non-dominated solutions).

    Attributes
    ----------
    _systems:
        Internal mapping of system name → (privacy_score, utility_score).
    """

    def __init__(self) -> None:
        """Initialize an empty analyzer."""
        self._systems: dict[str, tuple[float, float]] = {}

    def add_system(self, name: str, privacy_score: float, utility_score: float) -> None:
        """Add or update a system with privacy and utility scores.

        Parameters
        ----------
        name:
            System identifier.
        privacy_score:
            Privacy protection score [0, 1].
        utility_score:
            Utility preservation score [0, 1].
        """
        privacy_score = max(0.0, min(1.0, privacy_score))
        utility_score = max(0.0, min(1.0, utility_score))
        self._systems[name] = (privacy_score, utility_score)

    def compute_frontier(self) -> list[str]:
        """Return names of Pareto-optimal systems.

        A system is Pareto-optimal if no other system dominates it
        (i.e., is strictly better on both privacy and utility).

        Returns
        -------
        list[str]
            System names on the Pareto frontier, sorted by privacy score.
        """
        if not self._systems:
            return []

        frontier: list[str] = []
        for name, (priv, util) in self._systems.items():
            dominated = False
            for other_name, (other_priv, other_util) in self._systems.items():
                if other_name != name:
                    # Check if other dominates name
                    if other_priv >= priv and other_util >= util:
                        if other_priv > priv or other_util > util:
                            dominated = True
                            break
            if not dominated:
                frontier.append(name)

        # Sort by privacy score for readability
        frontier.sort(key=lambda n: self._systems[n][0])
        return frontier

    def is_dominated(self, name: str) -> bool:
        """Check if a system is dominated by another.

        Parameters
        ----------
        name:
            System identifier.

        Returns
        -------
        bool
            True if the system is dominated (not on frontier).
        """
        if name not in self._systems:
            return False
        return name not in self.compute_frontier()

    def distance_to_frontier(self, name: str) -> float:
        """Compute Euclidean distance to the nearest Pareto-optimal point.

        Parameters
        ----------
        name:
            System identifier.

        Returns
        -------
        float
            Distance to nearest frontier point. Returns 0 if system is on frontier.
        """
        if name not in self._systems:
            return float("inf")

        priv, util = self._systems[name]
        frontier = self.compute_frontier()

        if name in frontier:
            return 0.0

        min_dist = float("inf")
        for frontier_name in frontier:
            f_priv, f_util = self._systems[frontier_name]
            dist = math.sqrt((priv - f_priv) ** 2 + (util - f_util) ** 2)
            min_dist = min(min_dist, dist)

        return min_dist

    def frontier_data(self) -> dict[str, Any]:
        """Return all data for visualization and analysis.

        Returns
        -------
        dict
            Dictionary with keys:
            - "systems": all system data as {name: {privacy, utility, on_frontier}}
            - "frontier": list of frontier system names
            - "dominated": list of dominated system names
        """
        frontier = self.compute_frontier()
        systems_data: dict[str, dict[str, Any]] = {}

        for name, (priv, util) in self._systems.items():
            systems_data[name] = {
                "privacy": round(priv, 6),
                "utility": round(util, 6),
                "on_frontier": name in frontier,
                "distance_to_frontier": round(self.distance_to_frontier(name), 6),
            }

        return {
            "systems": systems_data,
            "frontier": frontier,
            "dominated": [name for name in self._systems if name not in frontier],
        }

    def reset(self) -> None:
        """Clear all systems from the analyzer."""
        self._systems.clear()
