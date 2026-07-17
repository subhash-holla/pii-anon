"""S5-02 — Tier-3 representative re-identification adversary + RRS power.

DC-09 ``attacks/`` package. Builds on the S5-01 :class:`ReidAttack` foundation
with a materially stronger, still-deterministic re-identification adversary (the
honest in-tree stand-in for the real Tier-3 LLM adversary, FR-011) plus the
NFR-012 re-identification-risk-study (RRS) statistical-power machinery (Wilson
confidence intervals on integer counts + the 2-rung paired-persona ladder).

What makes it "Tier-3 representative" (vs the S5-01 baseline)
------------------------------------------------------------
* **QIC + BSL fused scoring (FR-011).** For each pseudonymous target the
  adversary fuses two signals an LLM adversary would exploit:
  - **QIC** (quasi-identifier-combination) — weighted-Jaccard of the target's
    surviving signal tokens against the candidate persona's quasi-identifiers;
  - **BSL** (background-signal-linkage) — weighted-Jaccard of the same target
    tokens against the persona's flattened ``auxiliary_knowledge`` values (the
    background knowledge the baseline ignores).
  ``score = (1-λ)·QIC + λ·BSL`` with a fixed blend :data:`_QIC_BSL_LAMBDA`.
* **Margin-based commit / abstain (FR-012 de-circularization).** The adversary
  commits the top candidate ONLY when ``top_score > 0`` and (single candidate,
  or ``top_score − runner_up_score ≥`` :data:`_MARGIN_THRESHOLD`); otherwise it
  abstains. So ``reid_precision`` measures real confidence, not a vacuous argmax,
  and the adversary never guesses on an ambiguous link.
* **No gold-link access (FR-012 contamination control).** ``attack(...)`` sees
  ONLY ``(targets, candidates, candidate_set_size)`` — never the ground-truth
  link (which lives in :func:`~pii_anon.eval_framework.attacks.reid.score_reid_attack`).
  The adversary follows the SIGNAL; it structurally cannot "cheat" to the gold.

NFR-012 RRS power
-----------------
* :func:`wilson_interval` — the Wilson score interval for a binomial proportion
  on INTEGER counts (never a normal-approx that misbehaves at the extremes).
* :class:`ReidPowerCell` / :class:`ReidPowerLadder` / :func:`assess_rrs_power` —
  the 2-rung paired-persona ladder (:data:`RRS_RUNG_REID_LOW` 385 /
  :data:`RRS_RUNG_REID_HIGH` 897) with a per-cell ``powered`` verdict + Wilson
  CIs on the success rate, carrying the non-strippable NFR-016 caveat.

Switch-points (Pass-2 / cross-repo)
-----------------------------------
* ``# SWITCH-POINT(LLM)`` — the real Tier-3 LLM-call adversary (lazy/optional).
* ``# SWITCH-POINT(DATA)`` — the real offline DATA adversary (IDF-weighted QIC +
  behavioral signals) + the real ≥385/≥897 paired-persona cohort
  (``assemble_paired_set``, VERIFIED ABSENT today). The power-ASSERTION machinery
  ships now; feeding it a real cohort is the cross-repo Pass-2.

Determinism (AX-002) + isolation (AX-001 / RISK)
------------------------------------------------
Pure-stdlib over scalar inputs; the ranking is a total order and the Wilson math
is pure ``math``; this module imports NO ``random`` / ``uuid`` / ``time`` /
``secrets`` and nothing from ``swarm`` / ``moe`` / ``fusion`` / ``policy`` (the
package AST + import-boundary guards auto-scan it). ZERO unsafe-deserialization /
subprocess / shell-out / dynamic-eval. All personas/targets are SYNTHETIC.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pii_anon.eval_framework.attacks.reid import (
    ANTI_ANONYMITY_CAVEAT,
    ReidGuess,
    ReidPersona,
    ReidTarget,
    score_reid_attack,
)
from pii_anon.eval_framework.attacks.sandbox import AttackCallable

__all__ = [
    "RepresentativeTier3ReidAttack",
    "wilson_interval",
    "ReidPowerCell",
    "ReidPowerLadder",
    "assess_rrs_power",
    "tier3_reid_attack_runner",
    "TIER3_REID_ATTACK_REGISTRY",
    "RRS_RUNG_REID_LOW",
    "RRS_RUNG_REID_HIGH",
]

# A version-pinned id (results cited as "vs adversary@version").
_TIER3_ADVERSARY_ID: Final[str] = "representative-tier3-reid@v1"

# The QIC/BSL blend: the share of the fused score the background-signal-linkage
# (auxiliary_knowledge) contributes. 0.5 = equal weight to surviving-QI overlap
# and background knowledge. # SWITCH-POINT(DATA): the real adversary learns this.
_QIC_BSL_LAMBDA: Final[float] = 0.5

# The de-circularization commit margin (FR-012): the top candidate must beat the
# runner-up by at least this much (in fused-score units) to be committed; else
# the adversary abstains. Small enough to commit clear links, large enough to
# refuse genuine ambiguity.
_MARGIN_THRESHOLD: Final[float] = 0.05

# NFR-012 — the 2-rung RRS paired-persona power ladder (the committed literals).
RRS_RUNG_REID_LOW: Final[int] = 385
RRS_RUNG_REID_HIGH: Final[int] = 897
_RUNG_MIN: Final[Mapping[str, int]] = {
    "REID_LOW": RRS_RUNG_REID_LOW,
    "REID_HIGH": RRS_RUNG_REID_HIGH,
}

# The default Wilson z (95% two-sided).
_WILSON_Z_95: Final[float] = 1.96


# --------------------------------------------------------------------------- #
# Token helpers — pure, deterministic (local; reid.py stays byte-identical).   #
# --------------------------------------------------------------------------- #
def _normalise_tokens(tokens: Sequence[str]) -> frozenset[str]:
    """Lower-cased, stripped, non-blank token set."""
    return frozenset(t.strip().lower() for t in tokens if t and t.strip())


def _weighted_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Weighted-Jaccard overlap of two token sets in ``[0, 1]`` (unit weights).

    Faithful to the S5-01 baseline's ``_weighted_jaccard``: ``|a∩b| / |a∪b|``;
    empty-vs-anything is 0.0 (no signal == no link). Re-implemented locally so
    ``reid.py`` stays byte-identical (its helper is module-private).
    # SWITCH-POINT(DATA): the real adversary weights tokens by inverse document
    frequency.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _aux_tokens(persona: ReidPersona) -> frozenset[str]:
    """Flatten a persona's ``auxiliary_knowledge`` VALUES into a token set (BSL).

    The background-knowledge side-channel the real adversary exploits. Only the
    mapping VALUES are tokenised (the keys are schema labels, not signal); a
    value is ``str``-ified, split on whitespace, and normalised. Deterministic.
    """
    tokens: list[str] = []
    for value in persona.auxiliary_knowledge.values():
        tokens.extend(str(value).split())
    return _normalise_tokens(tuple(tokens))


# --------------------------------------------------------------------------- #
# The representative Tier-3 adversary (conforms to the S5-01 ReidAttack).      #
# --------------------------------------------------------------------------- #
class RepresentativeTier3ReidAttack:
    """A deterministic, de-circularized QIC+BSL re-identification adversary.

    The honest in-tree stand-in for the real Tier-3 LLM adversary (FR-011):
    fuses surviving-quasi-identifier overlap (QIC) with a background-knowledge
    prior (BSL) and commits only on a clear margin (FR-012). Conforms to the
    S5-01 :class:`~pii_anon.eval_framework.attacks.reid.ReidAttack` Protocol
    (``@runtime_checkable``). Deterministic total order (AX-002): candidates
    ranked ``(fused-score DESC, persona_id ASC)``.
    """

    def __init__(self, *, adversary_id: str = _TIER3_ADVERSARY_ID) -> None:
        self.adversary_id: str = adversary_id
        self.deterministic: bool = True

    def _fused_score(
        self, target_tokens: frozenset[str], persona: ReidPersona
    ) -> float:
        """The QIC+BSL fused similarity of a target to a candidate persona."""
        qic = _weighted_jaccard(target_tokens, _normalise_tokens(persona.quasi_identifiers))
        bsl = _weighted_jaccard(target_tokens, _aux_tokens(persona))
        return (1.0 - _QIC_BSL_LAMBDA) * qic + _QIC_BSL_LAMBDA * bsl

    def attack(
        self,
        targets: Sequence[ReidTarget],
        candidates: Sequence[ReidPersona],
        candidate_set_size: int,
    ) -> list[ReidGuess]:
        """Link each pseudonymous target to its best candidate persona (or abstain).

        For each target, scores every candidate by the fused QIC+BSL similarity
        and commits the top-ranked persona ONLY when the margin over the
        runner-up clears :data:`_MARGIN_THRESHOLD` (FR-012 de-circularization);
        otherwise returns an abstain (``guessed_persona_id=None``). An empty
        candidate pool abstains. Sees ONLY ``(targets, candidates,
        candidate_set_size)`` — never the ground-truth link (AX-002 total order;
        the gold link lives in :func:`score_reid_attack`).
        """
        guesses: list[ReidGuess] = []
        for target in targets:
            t_tokens = _normalise_tokens(target.observed_signals)
            # Score every candidate, then rank by the total order
            # (score DESC, persona_id ASC) — input-order-invariant.
            scored = sorted(
                ((self._fused_score(t_tokens, p), p.persona_id) for p in candidates),
                key=lambda s: (-s[0], s[1]),
            )
            if not scored:
                guesses.append(ReidGuess(target_id=target.target_id, guessed_persona_id=None, score=0.0))
                continue
            top_score, top_id = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            # Commit only on a clear margin (FR-012); else abstain.
            committed = top_score > 0.0 and (
                len(scored) == 1 or (top_score - runner_up) >= _MARGIN_THRESHOLD
            )
            if committed:
                guesses.append(
                    ReidGuess(target_id=target.target_id, guessed_persona_id=top_id, score=top_score)
                )
            else:
                guesses.append(
                    ReidGuess(target_id=target.target_id, guessed_persona_id=None, score=top_score)
                )
        return guesses


# --------------------------------------------------------------------------- #
# Wilson score interval — NFR-012 CIs on integer counts (pure math).          #
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, *, z: float = _WILSON_Z_95) -> tuple[float, float]:
    """The Wilson score interval ``(low, high)`` for ``successes`` of ``n``.

    The NFR-012 confidence interval on INTEGER counts — preferred over the normal
    approximation, which misbehaves near 0 and 1 and for small ``n``. ``n == 0``
    ⇒ ``(0.0, 0.0)`` (no observations, no interval). The result is clamped to
    ``[0, 1]`` and brackets the point estimate ``successes / n``. Pure +
    deterministic (``math.sqrt`` only).

    A corrupt count (``successes < 0`` or ``successes > n``, or a negative ``n``)
    is non-physical and is refused LOUDLY with a DOMAIN-named ``ValueError``
    rather than left to surface as a lower-level ``math domain error`` from the
    negative-radicand ``sqrt`` (the S5-02 security-sast SEC-01 observation). Honest
    callers (a success count is, by construction, ``0 ≤ successes ≤ n``) never trip
    it; this only converts a corrupt-input crash into a self-describing one — it
    NEVER fabricates an out-of-``[0, 1]`` interval.
    """
    if successes < 0 or n < 0 or successes > n:
        raise ValueError(
            f"wilson_interval domain error: require 0 <= successes <= n "
            f"(got successes={successes}, n={n})"
        )
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    return (low, high)


# --------------------------------------------------------------------------- #
# RRS power ladder — NFR-012 2-rung paired-persona power (with the caveat).    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReidPowerCell:
    """One paired-persona RRS cell's power verdict (with a non-strippable caveat).

    ``powered`` is ``n_paired_personas >= min_required`` (the cell's rung
    threshold). ``ci_low``/``ci_high`` are the Wilson interval on the success
    rate. ``caveat`` defaults to :data:`ANTI_ANONYMITY_CAVEAT` and CANNOT be
    cleared (NFR-016 — ``__post_init__`` re-asserts a non-blank caveat).
    """

    cell_id: str
    n_paired_personas: int
    successes: int
    success_rate: float
    ci_low: float
    ci_high: float
    rung: str
    min_required: int
    powered: bool
    caveat: str = ANTI_ANONYMITY_CAVEAT

    def __post_init__(self) -> None:
        # NFR-016: the anti-anonymity caveat is non-strippable.
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError(
                "ReidPowerCell.caveat is the non-strippable anti-anonymity caveat "
                "(NFR-016) and may not be blank/cleared"
            )

    @classmethod
    def build(
        cls, *, cell_id: str, n_paired_personas: int, successes: int, rung: str
    ) -> ReidPowerCell:
        """Build a cell from raw integer counts (computes rate + Wilson CI + powered).

        ``rung`` must be ``"REID_LOW"`` or ``"REID_HIGH"`` (the 2-rung ladder); an
        unknown rung is refused loudly (it cannot name a power threshold).
        """
        if rung not in _RUNG_MIN:
            raise ValueError(f"unknown RRS rung {rung!r} (expected one of {sorted(_RUNG_MIN)})")
        min_required = _RUNG_MIN[rung]
        rate = (successes / n_paired_personas) if n_paired_personas > 0 else 0.0
        low, high = wilson_interval(successes, n_paired_personas)
        return cls(
            cell_id=cell_id,
            n_paired_personas=n_paired_personas,
            successes=successes,
            success_rate=rate,
            ci_low=low,
            ci_high=high,
            rung=rung,
            min_required=min_required,
            powered=n_paired_personas >= min_required,
        )

    def as_outcome(self) -> dict[str, Any]:
        """Serialize to an audit Mapping (always carries the caveat)."""
        return {
            "cell_id": self.cell_id,
            "n_paired_personas": self.n_paired_personas,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "rung": self.rung,
            "min_required": self.min_required,
            "powered": self.powered,
            "caveat": self.caveat,
        }


@dataclass(frozen=True)
class ReidPowerLadder:
    """The aggregate RRS power verdict over a set of paired-persona cells.

    ``all_powered`` is ``True`` iff EVERY cell is powered (and there is ≥1 cell).
    Carries the non-strippable NFR-016 caveat in :meth:`as_outcome`.
    """

    cells: tuple[ReidPowerCell, ...]
    all_powered: bool
    caveat: str = ANTI_ANONYMITY_CAVEAT

    def __post_init__(self) -> None:
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError(
                "ReidPowerLadder.caveat is the non-strippable anti-anonymity "
                "caveat (NFR-016) and may not be blank/cleared"
            )

    def as_outcome(self) -> dict[str, Any]:
        """Serialize to an audit Mapping (always carries the caveat)."""
        return {
            "all_powered": self.all_powered,
            "cells": [cell.as_outcome() for cell in self.cells],
            "caveat": self.caveat,
        }


def assess_rrs_power(cells: Sequence[ReidPowerCell]) -> ReidPowerLadder:
    """Aggregate per-cell RRS power verdicts into a :class:`ReidPowerLadder`.

    ``all_powered`` is ``True`` iff there is at least one cell and every cell is
    powered. The real ≥385/≥897 paired-persona cohort assembly is DATA-absent
    (``# SWITCH-POINT(DATA)``); this assesses whatever cells it is given.
    """
    cells_t = tuple(cells)
    all_powered = bool(cells_t) and all(cell.powered for cell in cells_t)
    return ReidPowerLadder(cells=cells_t, all_powered=all_powered)


# --------------------------------------------------------------------------- #
# The allow-listed sandbox runner + the registry (additive merge).            #
# --------------------------------------------------------------------------- #
def _decode_object_array(source_json: str, *, label: str) -> list[Mapping[str, Any]]:
    """Decode a JSON array of plain objects (stdlib parser; no unsafe-deser)."""
    decoded = json.loads(source_json)
    if not isinstance(decoded, list):
        raise ValueError(f"{label} must decode to a JSON array (got {type(decoded).__name__})")
    items: list[Mapping[str, Any]] = []
    for element in decoded:
        if not isinstance(element, Mapping):
            raise ValueError(f"each {label} element must be a JSON object (got {type(element).__name__})")
        items.append(element)
    return items


def _string_tuple(item: Mapping[str, Any], key: str, *, label: str) -> tuple[str, ...]:
    """Read ``item[key]`` as a JSON array of strings → a str tuple (missing → ())."""
    raw = item.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"{label}.{key} must be a JSON array of strings (got {type(raw).__name__})")
    return tuple(str(v) for v in raw)


def _targets_from_json(targets_json: str) -> list[ReidTarget]:
    """Decode targets from a JSON array of plain mappings (stdlib parser only).

    Reads scalar fields out of plain mappings via the shared object-array
    decoder — it NEVER constructs objects named in the data (no
    unsafe-deserialization). Mirrors the S5-01 ``reid.py`` decode discipline.
    """
    return [
        ReidTarget(
            target_id=str(item["target_id"]),
            anonymized_text=str(item.get("anonymized_text", "")),
            observed_signals=_string_tuple(item, "observed_signals", label="target"),
        )
        for item in _decode_object_array(targets_json, label="target")
    ]


def _personas_from_json(candidates_json: str) -> list[ReidPersona]:
    """Decode candidate personas from a JSON array of plain mappings.

    Reads scalar fields + the ``auxiliary_knowledge`` object (defaulting to
    ``{}`` when absent, refused loudly when non-object) out of plain mappings via
    the shared stdlib decoder — it NEVER constructs objects named in the data (no
    unsafe-deserialization). Mirrors the S5-01 ``reid.py`` decode discipline.
    """
    out: list[ReidPersona] = []
    for item in _decode_object_array(candidates_json, label="candidate"):
        aux = item.get("auxiliary_knowledge", {})
        if not isinstance(aux, Mapping):
            raise ValueError(
                f"candidate.auxiliary_knowledge must be a JSON object (got {type(aux).__name__})"
            )
        out.append(
            ReidPersona(
                persona_id=str(item["persona_id"]),
                quasi_identifiers=_string_tuple(item, "quasi_identifiers", label="candidate"),
                auxiliary_knowledge=dict(aux),
                source_text=str(item.get("source_text", "")),
            )
        )
    return out


def tier3_reid_attack_runner(
    *,
    targets_json: str,
    candidates_json: str,
    candidate_set_size: int,
) -> dict[str, Any]:
    """The allow-listed Tier-3 representative runner (the sandbox entrypoint).

    Scalar JSON-string args (the ``AttackSpec`` rule): JSON-decode the targets +
    candidates into frozen value objects (stdlib parser only — no
    unsafe-deserialization), run :class:`RepresentativeTier3ReidAttack`, score via
    the S5-01 :func:`~pii_anon.eval_framework.attacks.reid.score_reid_attack`, and
    return :meth:`ReidSuccessMetrics.as_outcome` (a ``Mapping[str, Any]`` — the
    sandbox contract). Pure + deterministic over its scalar inputs (AX-002).
    """
    targets = _targets_from_json(targets_json)
    candidates = _personas_from_json(candidates_json)
    adversary = RepresentativeTier3ReidAttack()
    guesses = adversary.attack(targets, candidates, candidate_set_size)
    metrics = score_reid_attack(
        guesses,
        targets,
        adversary_id=adversary.adversary_id,
        deterministic=adversary.deterministic,
        candidate_set_size=candidate_set_size,
    )
    return metrics.as_outcome()


# The in-code allow-list of the Tier-3 runner (a plain dict — the only way a spec
# selects this code). Additively merged into the sandbox default registry (see
# ``attacks/__init__``); keys are DISJOINT from S5-01's (no clobber).
TIER3_REID_ATTACK_REGISTRY: Final[Mapping[str, AttackCallable]] = {
    "pii_anon.eval_framework.attacks.reid_tier3.tier3_reid_attack_runner": tier3_reid_attack_runner,
    "reid_tier3_representative": tier3_reid_attack_runner,
}
