"""Two STRUCTURALLY DISTINCT de-identification scoring families (S4-01).

AX-004 — the headline project axiom — mandates that **anonymization**
(irreversible de-identification: "is the original unrecoverable?") and
**pseudonymization** (reversible-under-controlled-key: "is reversal
authorized-only, referentially consistent, and key-separated?") are **never**
merged into a single de-id score. They answer different questions and carry
different risk semantics.

This module ships the two families as two independent scorers producing two
independent frozen sub-records, carried together — but never combined — by
:class:`DeidFamilyScores`. There is **no** ``combined`` / ``deid_score`` /
``privacy_score`` field anywhere on the record (the no-merge CI guard in
``tests/test_deid_families.py`` is the standing AX-004 regression gate).

Layering / boundary (story §2a)
-------------------------------
Lives in ``eval_framework/metrics`` and imports ONLY this package's
:mod:`base` + :mod:`privacy_metrics` (the existing irreversibility metrics) +
the stdlib. It imports nothing from ``swarm`` / ``moe`` / ``fusion`` /
``policy`` (import-boundary CI guard). The two scorers share no mutable state and
neither imports/instantiates the other.

Families (story §2 / FR-006 / FR-009 / FR-010)
----------------------------------------------
* :class:`AnonymizationScorer` → :class:`AnonymizationScore` — IRREVERSIBLE.
  Aggregates the existing :class:`ReidentificationRiskMetric`,
  :class:`LeakageDetectionMetric`, :class:`CanaryExposureMetric` (all "lower is
  riskier-survives"); the derived ``irreversibility_score`` is
  ``1 - max(risk axes)`` (higher = the original is less recoverable).
* :class:`PseudonymizationIntegrityScorer` →
  :class:`PseudonymizationIntegrityScore` — REVERSIBLE-UNDER-KEY. The FR-009
  axes: ``unauthorized_reversal_rate`` (NFR-014: must = 0), ``referential_
  integrity`` (same plaintext → same surrogate, = 1.0), ``collision_rate``
  (distinct plaintext → same surrogate is a collision), and
  ``key_state_separation_ok`` (NFR-015 / Art-4(5): the artifact alone must not
  re-join — the external secret is required). The derived ``integrity_score``
  combines the reversible-integrity axes (NOT the anonymization axes).

Determinism (AX-002): both scorers are pure functions of their inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .base import EvaluationLevel, LabeledSpan
from .privacy_metrics import (
    CanaryExposureMetric,
    LeakageDetectionMetric,
    ReidentificationRiskMetric,
)

__all__ = [
    "AnonymizationScore",
    "PseudonymizationIntegrityScore",
    "DeidFamilyScores",
    "AnonymizationScorer",
    "PseudonymizationIntegrityScorer",
]


# ---------------------------------------------------------------------------
# Family A — anonymization (irreversible). Carries ONLY anonymization axes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnonymizationScore:
    """Irreversible-de-identification sub-record ("is the original unrecoverable?").

    Carries ONLY anonymization axes — it never holds a pseudonymization-integrity
    number (the AX-004 wall). All axes are in ``[0, 1]``.

    Attributes
    ----------
    reidentification_risk:
        Fraction of ground-truth entities whose original surface survives in the
        anonymized text (:class:`ReidentificationRiskMetric`). 0.0 = none survive.
    leakage:
        Fraction of entities with a surviving substring fragment
        (:class:`LeakageDetectionMetric`). 0.0 = no partial leaks.
    canary_exposure:
        Fraction of injected canary strings exposed in the output
        (:class:`CanaryExposureMetric`). 0.0 = no canary leaked.
    irreversibility_score:
        Derived ``1 - max(reidentification_risk, leakage, canary_exposure)`` —
        higher means the original is less recoverable (the anonymization goal).
    """

    reidentification_risk: float
    leakage: float
    canary_exposure: float
    irreversibility_score: float


# ---------------------------------------------------------------------------
# Family B — pseudonymization integrity (reversible-under-key). Carries ONLY
# the reversibility/integrity axes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PseudonymizationIntegrityScore:
    """Reversible-under-key sub-record (FR-009 five-axis pseudonymization family).

    Carries ONLY pseudonymization-integrity axes — it never holds an
    anonymization (re-identification/leakage/canary) number (the AX-004 wall).

    Attributes
    ----------
    unauthorized_reversal_rate:
        Fraction of reversal attempts that SUCCEEDED WITHOUT the authorized key
        (NFR-014: must be 0.0; a wrong-key reversal that nonetheless succeeds is a
        leak). In ``[0, 1]``.
    referential_integrity:
        Fraction of surrogates that are referentially consistent (a surrogate is
        consistent iff exactly one plaintext maps to it). 1.0 = fully consistent.
    collision_rate:
        Fraction of surrogates that two-or-more DISTINCT plaintexts share (a
        collision breaks reversibility). In ``[0, 1]``; complements
        ``referential_integrity``.
    key_state_separation_ok:
        NFR-015 / Art-4(5): ``True`` iff the artifact ALONE cannot re-join (the
        external secret is required to reverse); ``False`` flags an artifact-alone
        re-join (a key/state-separation breach).
    integrity_score:
        Derived overall pseudonymization-integrity in ``[0, 1]``: the referential
        integrity, zeroed out if unauthorized-reversal occurred or key/state
        separation is breached (a reversible scheme with either failure has no
        integrity). Combines ONLY the reversible-integrity axes — never the
        anonymization axes.
    """

    unauthorized_reversal_rate: float
    referential_integrity: float
    collision_rate: float
    key_state_separation_ok: bool
    integrity_score: float

    @property
    def unauthorized_reversal_ok(self) -> bool:
        """NFR-014 predicate: ``True`` iff the unauthorized-reversal rate is 0."""
        return self.unauthorized_reversal_rate == 0.0


# ---------------------------------------------------------------------------
# The carrier record — TWO separate sub-records, NEVER a merged number.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeidFamilyScores:
    """The de-identification scores as TWO structurally distinct families.

    AX-004: there is NO ``combined`` / ``deid_score`` / ``privacy_score`` field on
    this record. The anonymization and pseudonymization-integrity families are
    carried as two separate, separately-addressable sub-records and are never
    folded into one de-id scalar.
    """

    anonymization: AnonymizationScore
    pseudonymization_integrity: PseudonymizationIntegrityScore


# ---------------------------------------------------------------------------
# Scorer A — anonymization (aggregates the existing irreversibility metrics).
# ---------------------------------------------------------------------------


class AnonymizationScorer:
    """Aggregate the irreversibility metrics into an :class:`AnonymizationScore`.

    Composes the EXISTING :class:`ReidentificationRiskMetric`,
    :class:`LeakageDetectionMetric`, :class:`CanaryExposureMetric` from
    :mod:`privacy_metrics`. Pure / stateless; does not import or reference the
    pseudonymization family.
    """

    def __init__(self) -> None:
        self._reid = ReidentificationRiskMetric()
        self._leakage = LeakageDetectionMetric()
        self._canary = CanaryExposureMetric()

    def score(
        self,
        *,
        labels: Sequence[LabeledSpan],
        original_text: str,
        anonymized_text: str,
        canary_strings: Sequence[str] | None = None,
    ) -> AnonymizationScore:
        """Score the IRREVERSIBLE anonymization family for one synthetic result.

        Parameters
        ----------
        labels:
            Ground-truth PII spans over ``original_text`` (synthetic).
        original_text / anonymized_text:
            The pre/post anonymization document text (synthetic; AX-001).
        canary_strings:
            Optional injected canary markers to probe for exposure.
        """
        labels_list = list(labels)
        context = {"original_text": original_text, "anonymized_text": anonymized_text}

        reid = self._reid.compute(
            [], labels_list, level=EvaluationLevel.DOCUMENT, context=context
        ).value
        leakage = self._leakage.compute(
            [], labels_list, level=EvaluationLevel.DOCUMENT, context=context
        ).value
        canary = self._canary.compute(
            [],
            labels_list,
            level=EvaluationLevel.DOCUMENT,
            context={
                "anonymized_text": anonymized_text,
                "canary_strings": list(canary_strings or []),
            },
        ).value

        irreversibility = round(1.0 - max(reid, leakage, canary), 6)
        return AnonymizationScore(
            reidentification_risk=reid,
            leakage=leakage,
            canary_exposure=canary,
            irreversibility_score=irreversibility,
        )


# ---------------------------------------------------------------------------
# Scorer B — pseudonymization integrity (FR-009 five axes, reversible-under-key).
# ---------------------------------------------------------------------------


class PseudonymizationIntegrityScorer:
    """Compute the FR-009 reversible-under-key axes → :class:`PseudonymizationIntegrityScore`.

    Pure / stateless; does not import or reference the anonymization family. All
    inputs are synthetic (AX-001) — the unauthorized-reversal probe is driven by a
    caller-supplied list of ``(key, succeeded)`` attempts using a SYNTHETIC wrong
    key, never a real secret.
    """

    def score(
        self,
        *,
        pseudonym_map: dict[str, str],
        authorized_key: str,
        reversal_attempts: Sequence[tuple[str, bool]],
        artifact_alone_rejoinable: bool = False,
    ) -> PseudonymizationIntegrityScore:
        """Score the REVERSIBLE pseudonymization-integrity family (FR-009).

        Parameters
        ----------
        pseudonym_map:
            ``{plaintext: surrogate}`` (synthetic). Referential integrity holds
            when each surrogate has exactly one source plaintext; a surrogate
            shared by two-or-more distinct plaintexts is a collision.
        authorized_key:
            The single synthetic key authorized to reverse the mapping. A reversal
            that succeeds with any OTHER key is an unauthorized reversal (NFR-014).
        reversal_attempts:
            A sequence of ``(key_used, reversal_succeeded)`` probes (synthetic). An
            attempt is an unauthorized reversal iff ``key_used != authorized_key``
            AND ``reversal_succeeded`` is ``True``.
        artifact_alone_rejoinable:
            NFR-015 / Art-4(5): whether the pseudonymization artifact ALONE (no
            external secret) can re-join the originals. ``True`` is a key/state-
            separation breach; ``False`` (the desired state) means the external
            secret is required.
        """
        # NFR-014 — unauthorized-reversal rate (a wrong-key success is a leak).
        attempts = list(reversal_attempts)
        if attempts:
            unauthorized = sum(
                1 for key_used, succeeded in attempts
                if key_used != authorized_key and succeeded
            )
            unauthorized_rate = round(unauthorized / len(attempts), 6)
        else:
            unauthorized_rate = 0.0

        # Referential integrity + collisions: group plaintexts by surrogate.
        surrogate_sources: dict[str, set[str]] = {}
        for plaintext, surrogate in pseudonym_map.items():
            surrogate_sources.setdefault(surrogate, set()).add(plaintext)
        total_surrogates = len(surrogate_sources)
        if total_surrogates:
            colliding = sum(1 for sources in surrogate_sources.values() if len(sources) > 1)
            collision_rate = round(colliding / total_surrogates, 6)
            referential_integrity = round(1.0 - collision_rate, 6)
        else:
            collision_rate = 0.0
            referential_integrity = 1.0

        # NFR-015 — key/state separation (artifact alone must NOT re-join).
        key_state_separation_ok = not artifact_alone_rejoinable

        # Derived integrity: referential integrity, voided by any unauthorized
        # reversal or a key/state-separation breach (reversible-integrity axes
        # ONLY — never the anonymization axes).
        if unauthorized_rate > 0.0 or not key_state_separation_ok:
            integrity_score = 0.0
        else:
            integrity_score = referential_integrity

        return PseudonymizationIntegrityScore(
            unauthorized_reversal_rate=unauthorized_rate,
            referential_integrity=referential_integrity,
            collision_rate=collision_rate,
            key_state_separation_ok=key_state_separation_ok,
            integrity_score=integrity_score,
        )
