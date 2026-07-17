"""S5-03 — representative membership-inference adversary + Secret-Sharer + TPR@low-FPR.

DC-09 ``attacks/`` package. Implements the membership-inference half of the
attack surface (UC-10; FR-013/NFR-013), conforming to the S5-01
:class:`~pii_anon.eval_framework.attacks.reid.MiaAttack` Protocol. The REAL
LiRA@128 (≥128 trained shadow models on in/out splits) + the real Secret-Sharer
canary in/out splits are Pass-2 / cross-repo (``# SWITCH-POINT(DATA)``); the
in-tree REPRESENTATIVE adversary is the deterministic stand-in.

What lives here
---------------
* :class:`MiaRecord` — a frozen record carrying ONLY the attacker-observable
  signal (``observed_loss``); the gold membership label is never on it.
* :class:`RepresentativeMiaAttack` — a deterministic ``MiaAttack``: scores
  membership from ``-observed_loss`` (the LiRA intuition — a member's loss is
  lower). De-circularized (FR-013): ``membership_scores`` has no label access.
* :func:`tpr_at_fpr` — the LiRA reporting primitive: TPR at a committed low FPR
  over the synthetic member/non-member scores (integer-count ROC; 0.0 when the
  FPR target is unachievable — never fabricated).
* :class:`MiaSuccessReport` / :func:`score_mia_attack` — TPR@FPR∈{1e-3,1e-2} +
  AUC + the Wilson CI (REUSES the S5-02 :func:`wilson_interval`) + the
  non-strippable NFR-016 caveat.
* :func:`canary_exposure` / :class:`SecretSharerReport` — the Secret-Sharer
  exposure metric (``log2(n) - log2(rank)``).
* :class:`MiaPowerReport` / :func:`assess_mia_power` — the NFR-013 ≥128-shadow
  power assertion.
* :func:`mia_attack_runner` + :data:`MIA_ATTACK_REGISTRY` — the allow-listed
  sandbox runner (scalar JSON-string args), additively merged into the default
  registry.

Determinism (AX-002) + isolation (AX-001 / RISK)
------------------------------------------------
Pure-stdlib over scalar inputs; deterministic scoring + pure ROC/exposure math;
imports NO ``random`` / ``uuid`` / ``time`` / ``secrets`` and nothing from
``swarm`` / ``moe`` / ``fusion`` / ``policy`` (the package AST + import-boundary
guards auto-scan it). ZERO unsafe-deserialization / subprocess / shell-out /
dynamic-eval. All records are SYNTHETIC.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pii_anon.eval_framework.attacks.reid import ANTI_ANONYMITY_CAVEAT
from pii_anon.eval_framework.attacks.reid_tier3 import wilson_interval
from pii_anon.eval_framework.attacks.sandbox import AttackCallable

__all__ = [
    "MiaRecord",
    "RepresentativeMiaAttack",
    "tpr_at_fpr",
    "MiaSuccessReport",
    "score_mia_attack",
    "canary_exposure",
    "SecretSharerReport",
    "MiaPowerReport",
    "assess_mia_power",
    "mia_attack_runner",
    "MIA_ATTACK_REGISTRY",
    "MIA_MIN_SHADOW_MODELS",
    "MIA_FPR_TARGETS",
]

_MIA_ADVERSARY_ID: Final[str] = "representative-mia@v1"

# NFR-013 — the committed low-FPR reporting points + the shadow-model floor.
MIA_FPR_TARGETS: Final[tuple[float, float]] = (1e-3, 1e-2)
MIA_MIN_SHADOW_MODELS: Final[int] = 128


# --------------------------------------------------------------------------- #
# The attacker-observable record + the representative adversary.              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MiaRecord:
    """One candidate record AS THE ATTACKER OBSERVES IT.

    Carries ONLY the observable signal — the model's ``observed_loss`` on the
    record (lower ⟹ more member-like, the LiRA intuition). The gold membership
    label is NOT a field here (de-circularization, FR-013): the attacker cannot
    read it; it lives only in the scorer's ``gold_membership`` argument.
    """

    record_id: str
    observed_loss: float


class RepresentativeMiaAttack:
    """A deterministic representative membership-inference adversary (FR-013).

    The honest in-tree stand-in for the real LiRA@128 adversary: scores
    membership as a monotone transform of ``-observed_loss`` (a member's loss is
    lower, so its membership score is higher). Conforms to the S5-01
    :class:`~pii_anon.eval_framework.attacks.reid.MiaAttack` Protocol. Pure +
    deterministic (AX-002): no RNG/clock. De-circularized — ``membership_scores``
    reads ONLY ``observed_loss``, never the gold label.

    # SWITCH-POINT(DATA): the real LiRA computes a per-record likelihood ratio
    against ≥128 shadow models' in/out loss distributions.
    """

    def __init__(self, *, adversary_id: str = _MIA_ADVERSARY_ID) -> None:
        self.adversary_id: str = adversary_id
        self.deterministic: bool = True

    def membership_scores(self, records: Sequence[Any]) -> list[float]:
        """Return a membership score per record (higher ⟹ more likely member).

        A pure function of each record's ``observed_loss`` only (the LiRA-shape
        stand-in: ``score = -observed_loss``). Never consults a gold membership
        label (de-circularization, FR-013/AX-002).
        """
        return [-float(record.observed_loss) for record in records]


# --------------------------------------------------------------------------- #
# TPR@low-FPR — the LiRA reporting primitive (integer-count ROC).            #
# --------------------------------------------------------------------------- #
def tpr_at_fpr(
    scores: Sequence[float],
    gold_membership: Sequence[bool],
    *,
    fpr_target: float,
) -> float:
    """The true-positive rate at the largest threshold whose FPR ≤ ``fpr_target``.

    The LiRA reporting convention: sweep the membership-score threshold (a member
    is predicted when ``score ≥ threshold``), and among thresholds whose empirical
    false-positive rate (non-members predicted member / total non-members) is
    ``≤ fpr_target``, report the TPR (members predicted member / total members) at
    the one admitting the MOST true positives. Returns ``0.0`` when no threshold
    achieves the target FPR (honest — never a fabricated TPR).

    Integer-count + deterministic. A corrupt ``fpr_target`` (outside ``[0, 1]``)
    or a ``scores``/``gold_membership`` length mismatch is refused LOUDLY with a
    domain-named :class:`ValueError` (the S5-02 SEC-01 fail-loud discipline).
    """
    if not 0.0 <= fpr_target <= 1.0:
        raise ValueError(
            f"tpr_at_fpr domain error: fpr_target must be in [0, 1] (got {fpr_target})"
        )
    if len(scores) != len(gold_membership):
        raise ValueError(
            f"tpr_at_fpr domain error: scores/gold_membership length mismatch "
            f"({len(scores)} vs {len(gold_membership)})"
        )
    n_members = sum(1 for is_member in gold_membership if is_member)
    n_non_members = len(gold_membership) - n_members
    if n_members == 0 or n_non_members == 0:
        return 0.0
    # Candidate thresholds = each distinct score (predict member when score >= t).
    # Sweep high→low; track the best TPR among thresholds with FPR <= target.
    best_tpr = 0.0
    for threshold in sorted(set(scores), reverse=True):
        tp = sum(
            1
            for score, is_member in zip(scores, gold_membership)
            if is_member and score >= threshold
        )
        fp = sum(
            1
            for score, is_member in zip(scores, gold_membership)
            if (not is_member) and score >= threshold
        )
        fpr = fp / n_non_members
        if fpr <= fpr_target:
            best_tpr = max(best_tpr, tp / n_members)
    return best_tpr


def _auc(scores: Sequence[float], gold_membership: Sequence[bool]) -> float:
    """The ROC AUC via the Mann-Whitney statistic (ties counted as 0.5).

    Pure + deterministic: AUC = P(score(member) > score(non-member)) + 0.5·P(=).
    Returns ``0.0`` when either class is empty.
    """
    members = [s for s, m in zip(scores, gold_membership) if m]
    non_members = [s for s, m in zip(scores, gold_membership) if not m]
    if not members or not non_members:
        return 0.0
    wins = 0.0
    for ms in members:
        for ns in non_members:
            if ms > ns:
                wins += 1.0
            elif ms == ns:
                wins += 0.5
    return wins / (len(members) * len(non_members))


@dataclass(frozen=True)
class MiaSuccessReport:
    """Membership-inference success metrics (with a non-strippable caveat).

    Carries TPR at the two committed low FPRs (NFR-013), the ROC AUC, the
    member/non-member counts, and the Wilson CI on the TPR@1e-2 true-positive
    count (REUSES the S5-02 :func:`wilson_interval`). ``caveat`` defaults to
    :data:`ANTI_ANONYMITY_CAVEAT` and CANNOT be cleared (NFR-016).
    """

    tpr_at_1e_3: float
    tpr_at_1e_2: float
    auc: float
    n_members: int
    n_non_members: int
    ci_low: float
    ci_high: float
    adversary_id: str
    deterministic: bool
    caveat: str = ANTI_ANONYMITY_CAVEAT

    def __post_init__(self) -> None:
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError(
                "MiaSuccessReport.caveat is the non-strippable anti-anonymity "
                "caveat (NFR-016) and may not be blank/cleared"
            )

    def as_outcome(self) -> dict[str, Any]:
        """Serialize to the sandbox runner's outcome Mapping (carries the caveat)."""
        return {
            "tpr_at_1e_3": self.tpr_at_1e_3,
            "tpr_at_1e_2": self.tpr_at_1e_2,
            "auc": self.auc,
            "n_members": self.n_members,
            "n_non_members": self.n_non_members,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "adversary_id": self.adversary_id,
            "deterministic": self.deterministic,
            "caveat": self.caveat,
        }


def score_mia_attack(
    scores: Sequence[float],
    gold_membership: Sequence[bool],
    *,
    adversary_id: str,
    deterministic: bool,
) -> MiaSuccessReport:
    """Score membership-inference scores into a :class:`MiaSuccessReport`.

    Computes TPR at both committed FPRs (NFR-013), the ROC AUC, and a Wilson CI
    on the TPR@1e-2 true-positive count over the member population. Carries the
    non-strippable NFR-016 caveat. Pure + deterministic (AX-002).
    """
    tpr_lo = tpr_at_fpr(scores, gold_membership, fpr_target=MIA_FPR_TARGETS[0])
    tpr_hi = tpr_at_fpr(scores, gold_membership, fpr_target=MIA_FPR_TARGETS[1])
    n_members = sum(1 for is_member in gold_membership if is_member)
    n_non_members = len(gold_membership) - n_members
    # The Wilson CI on the TPR@1e-2 true-positive proportion over members.
    true_positives = int(round(tpr_hi * n_members))
    ci_low, ci_high = wilson_interval(true_positives, n_members)
    return MiaSuccessReport(
        tpr_at_1e_3=tpr_lo,
        tpr_at_1e_2=tpr_hi,
        auc=_auc(scores, gold_membership),
        n_members=n_members,
        n_non_members=n_non_members,
        ci_low=ci_low,
        ci_high=ci_high,
        adversary_id=adversary_id,
        deterministic=deterministic,
    )


# --------------------------------------------------------------------------- #
# Secret-Sharer canary exposure.                                             #
# --------------------------------------------------------------------------- #
def canary_exposure(rank: int, n_candidates: int) -> float:
    """The Secret-Sharer exposure of a canary at ``rank`` among ``n_candidates``.

    ``exposure = log2(n_candidates) - log2(rank)`` — higher ⟹ more memorization
    ⟹ worse privacy (rank 1 = the most-likely sequence ⟹ maximal exposure; rank
    ``n`` ⟹ 0). Bounded ``≥ 0``. A corrupt ``rank`` (``< 1`` or ``> n``) or
    ``n_candidates < 1`` is refused LOUDLY with a domain-named :class:`ValueError`
    (never an out-of-range exposure). # SWITCH-POINT(DATA): the real canary in/out
    splits.
    """
    if n_candidates < 1:
        raise ValueError(
            f"canary_exposure domain error: n_candidates must be >= 1 (got {n_candidates})"
        )
    if rank < 1 or rank > n_candidates:
        raise ValueError(
            f"canary_exposure domain error: require 1 <= rank <= n_candidates "
            f"(got rank={rank}, n_candidates={n_candidates})"
        )
    return math.log2(n_candidates) - math.log2(rank)


@dataclass(frozen=True)
class SecretSharerReport:
    """A Secret-Sharer canary-exposure report (with a non-strippable caveat).

    ``caveat`` defaults to :data:`ANTI_ANONYMITY_CAVEAT` and CANNOT be cleared
    (NFR-016 — ``__post_init__`` re-asserts it).
    """

    rank: int
    n_candidates: int
    exposure: float
    caveat: str = ANTI_ANONYMITY_CAVEAT

    def __post_init__(self) -> None:
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError(
                "SecretSharerReport.caveat is the non-strippable anti-anonymity "
                "caveat (NFR-016) and may not be blank/cleared"
            )

    @classmethod
    def build(cls, *, rank: int, n_candidates: int) -> SecretSharerReport:
        """Build a report, computing the exposure (validates rank/n_candidates)."""
        return cls(
            rank=rank,
            n_candidates=n_candidates,
            exposure=canary_exposure(rank, n_candidates),
        )

    def as_outcome(self) -> dict[str, Any]:
        """Serialize to an audit Mapping (always carries the caveat)."""
        return {
            "rank": self.rank,
            "n_candidates": self.n_candidates,
            "exposure": self.exposure,
            "caveat": self.caveat,
        }


# --------------------------------------------------------------------------- #
# MIA power assertion — NFR-013 >=128 shadow models (with the caveat).       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MiaPowerReport:
    """The NFR-013 MIA-power verdict (≥128 shadow models; non-strippable caveat).

    ``powered`` is ``shadow_model_count >= MIA_MIN_SHADOW_MODELS``. The real
    training of ≥128 shadow models is Pass-2 (``# SWITCH-POINT(DATA)``); this
    reports the count + the verdict.
    """

    shadow_model_count: int
    min_required: int
    powered: bool
    caveat: str = ANTI_ANONYMITY_CAVEAT

    def __post_init__(self) -> None:
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError(
                "MiaPowerReport.caveat is the non-strippable anti-anonymity "
                "caveat (NFR-016) and may not be blank/cleared"
            )

    def as_outcome(self) -> dict[str, Any]:
        """Serialize to an audit Mapping (always carries the caveat)."""
        return {
            "shadow_model_count": self.shadow_model_count,
            "min_required": self.min_required,
            "powered": self.powered,
            "caveat": self.caveat,
        }


def assess_mia_power(shadow_model_count: int) -> MiaPowerReport:
    """The NFR-013 power verdict: ``powered`` iff ≥ :data:`MIA_MIN_SHADOW_MODELS`."""
    return MiaPowerReport(
        shadow_model_count=shadow_model_count,
        min_required=MIA_MIN_SHADOW_MODELS,
        powered=shadow_model_count >= MIA_MIN_SHADOW_MODELS,
    )


# --------------------------------------------------------------------------- #
# The allow-listed sandbox runner + the registry (additive merge).           #
# --------------------------------------------------------------------------- #
def _records_from_json(records_json: str) -> list[MiaRecord]:
    """Decode MIA records from a JSON array of plain mappings (stdlib parser only).

    Reads only the two scalar fields ``record_id`` + ``observed_loss`` — it NEVER
    constructs objects named in the data (no unsafe-deserialization). A non-array
    / non-object element is refused loudly. Mirrors the reid.py decode discipline.
    """
    decoded = json.loads(records_json)
    if not isinstance(decoded, list):
        raise ValueError(f"records must decode to a JSON array (got {type(decoded).__name__})")
    out: list[MiaRecord] = []
    for element in decoded:
        if not isinstance(element, Mapping):
            raise ValueError(f"each record must be a JSON object (got {type(element).__name__})")
        observed_loss = float(element["observed_loss"])
        # Fail LOUD on a non-finite observed_loss (NaN / ±inf) at the decode
        # boundary — the same fail-loud-domain-named discipline tpr_at_fpr /
        # canary_exposure use. A NaN loss would otherwise propagate a NaN
        # membership score silently (it stays bounded/non-fabricated in the ROC,
        # but a corrupt real-loss feed at the Pass-2 LiRA switch-point should be
        # refused, not laundered). Honest synthetic inputs are always finite.
        if not math.isfinite(observed_loss):
            raise ValueError(
                f"record observed_loss must be a finite number (got {observed_loss!r})"
            )
        out.append(
            MiaRecord(record_id=str(element["record_id"]), observed_loss=observed_loss)
        )
    return out


def _gold_from_json(gold_membership_json: str) -> list[bool]:
    """Decode the gold membership labels from a JSON array of booleans (stdlib only)."""
    decoded = json.loads(gold_membership_json)
    if not isinstance(decoded, list):
        raise ValueError(
            f"gold_membership must decode to a JSON array (got {type(decoded).__name__})"
        )
    # Each label must be a genuine JSON boolean — fail LOUD rather than silently
    # coerce ``null`` → False (a dropped membership label read as non-member) or
    # an int/string to truthiness. Same fail-loud-domain-named discipline as the
    # record decoder; honest inputs are always bool.
    out: list[bool] = []
    for value in decoded:
        if not isinstance(value, bool):
            raise ValueError(
                f"each gold_membership element must be a JSON boolean (got {type(value).__name__})"
            )
        out.append(value)
    return out


def mia_attack_runner(
    *,
    records_json: str,
    gold_membership_json: str,
    shadow_model_count: int,
) -> dict[str, Any]:
    """The allow-listed representative MIA runner (the sandbox entrypoint).

    Scalar JSON-string args (the ``AttackSpec`` rule): JSON-decode the records +
    gold labels (stdlib parser only — no unsafe-deserialization), run
    :class:`RepresentativeMiaAttack`, score via :func:`score_mia_attack`, and
    return :meth:`MiaSuccessReport.as_outcome` augmented with the NFR-013 power
    verdict — a ``Mapping[str, Any]`` (the sandbox contract). Pure + deterministic
    over its scalar inputs (AX-002).
    """
    records = _records_from_json(records_json)
    gold = _gold_from_json(gold_membership_json)
    adversary = RepresentativeMiaAttack()
    scores = adversary.membership_scores(records)
    report = score_mia_attack(
        scores,
        gold,
        adversary_id=adversary.adversary_id,
        deterministic=adversary.deterministic,
    )
    outcome = report.as_outcome()
    outcome["power"] = assess_mia_power(shadow_model_count).as_outcome()
    return outcome


# The in-code allow-list of the MIA runner (a plain dict — the only way a spec
# selects this code). Additively merged into the sandbox default registry (see
# ``attacks/__init__``); keys are DISJOINT from S5-01/S5-02's (no clobber).
MIA_ATTACK_REGISTRY: Final[Mapping[str, AttackCallable]] = {
    "pii_anon.eval_framework.attacks.mia.mia_attack_runner": mia_attack_runner,
    "mia_representative": mia_attack_runner,
}
