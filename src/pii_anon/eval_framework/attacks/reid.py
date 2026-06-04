"""S5-01 — ``ReidAttack`` protocol + deterministic baseline re-identification body.

DC-09 ``attacks/`` package (the FR-011/FR-013 protocol foundation). This module
adds the re-identification attack contract + a deterministic *baseline* adversary
body that runs ONLY inside the S5-04 sandbox (``run_attack_under_sandbox``), plus
the **non-strippable anti-anonymity caveat** every exported privacy metric must
carry (NFR-016 — a MUST, satisfied here via :class:`ReidSuccessMetrics`).

It shape-mirrors the DATA sibling's ``Adversary``/``Persona``/``Target``/``Guess``
contract (``pii_anon_datasets.scoring.adversary.base``) so swapping the baseline
for the real offline adversary (Wilson CIs on integer counts) is a structural
Pass-2 / cross-repo change — the ``# SWITCH-POINT(DATA)`` markers tag the swap.

What lives here:

- frozen value objects ``ReidPersona`` / ``ReidTarget`` / ``ReidGuess``;
- ``ReidSuccessMetrics`` (frozen) — integer-count success metrics whose
  ``caveat`` defaults to the canonical :data:`ANTI_ANONYMITY_CAVEAT` and CANNOT
  be cleared (``__post_init__`` re-asserts it; NFR-016) + :meth:`as_outcome`;
- ``@runtime_checkable`` :class:`ReidAttack` Protocol (FR-011) + a
  :class:`MiaAttack` Protocol seam (FR-013; consumed by S5-03);
- :class:`BaselineDeterministicReidAttack` — weighted-Jaccard over the surviving
  quasi-identifier tokens + observed signals, ranked ``(similarity desc,
  persona_id asc)`` (a TOTAL order — AX-002); abstains (``None``) on no signal;
- :func:`score_reid_attack` — integer-count recall / precision / success-rate;
- :func:`reid_attack_runner` — the **allow-listed** sandbox runner (scalar
  JSON-string args per the ``AttackSpec`` scalar-param rule), returning a Mapping;
- :data:`REID_ATTACK_REGISTRY` — ``{"reid_baseline": reid_attack_runner}``,
  additively merged into the sandbox default allow-list registry.

Determinism (AX-002): the ranking is a pure total order; this module imports NO
``random`` / ``uuid`` / ``time`` / ``secrets`` (an AST guard pins that). It is
pure-stdlib over scalar inputs — ZERO unsafe-deserialization, ZERO subprocess /
shell-out, ZERO dynamic-eval (the package AST/source guard pins that; the
patterns are referenced in prose via reworded forms only). The package imports
nothing from ``swarm`` / ``moe`` / ``fusion`` / ``policy`` (the import-boundary
test pins that).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

# The non-strippable anti-anonymity caveat (NFR-016): 100% of exported privacy
# artifacts must carry it. It is the default + re-asserted value of
# ``ReidSuccessMetrics.caveat`` and is always present in ``as_outcome()``.
ANTI_ANONYMITY_CAVEAT: Final[str] = (
    "ANTI-ANONYMITY CAVEAT: re-identification metrics measure residual privacy "
    "RISK under a bounded adversary; a low measured re-identification rate is "
    "NOT a guarantee of anonymity and MUST NOT be cited as one. Results are "
    "scoped to the stated adversary, candidate set, and synthetic evaluation "
    "corpus (AX-001)."
)

# A version-pinned id so results can be cited as "vs adversary@version" (mirrors
# the DATA sibling's ``adversary_id`` contract).
_BASELINE_ADVERSARY_ID: Final[str] = "baseline-deterministic-reid@v1"


# --------------------------------------------------------------------------- #
# Frozen value objects — mirror the DATA sibling Persona/Target/Guess shape.  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReidPersona:
    """The REAL side of a paired record — the ground-truth identity (read-only).

    ``quasi_identifiers`` are the surviving auxiliary-knowledge tokens the
    adversary may link against (e.g. coarse attributes). ``auxiliary_knowledge``
    is an opaque read-only side-channel the real adversary (S5-02) may consult;
    the baseline uses only the token overlap. # SWITCH-POINT(DATA): the real
    offline adversary reads the richer DATA ``Persona`` (sorted ``(type, value)``
    quasi-identifiers + a behavioral-signals block).
    """

    persona_id: str
    quasi_identifiers: tuple[str, ...]
    auxiliary_knowledge: Mapping[str, Any] = field(default_factory=dict)
    source_text: str = ""


@dataclass(frozen=True)
class ReidTarget:
    """The PSEUDONYMOUS side the adversary attacks.

    ``observed_signals`` are RE-EXTRACTED from ``anonymized_text`` (never copied
    from gold) — the load-bearing reidx-01 discipline the DATA sibling documents;
    the baseline links these surviving signals against persona quasi-identifiers.
    """

    target_id: str
    anonymized_text: str
    observed_signals: tuple[str, ...]


@dataclass(frozen=True)
class ReidGuess:
    """An adversary's ranked attribution for one target.

    ``guessed_persona_id is None`` == abstain (excluded from the precision
    denominator); a string == a commitment. ``score`` is the similarity used for
    ranking (a total order — AX-002).
    """

    target_id: str
    guessed_persona_id: str | None
    score: float


@dataclass(frozen=True)
class ReidSuccessMetrics:
    """Integer-count re-identification success metrics with a non-strippable caveat.

    ``correct`` / ``n_targets`` / ``n_guesses`` are integer counts;
    ``reid_recall = correct / n_targets``, ``reid_precision = correct / n_guesses``
    (precision degrades to 0.0 when there is no committed guess),
    ``reid_success_rate = reid_recall * reid_precision``.

    NFR-016: ``caveat`` defaults to :data:`ANTI_ANONYMITY_CAVEAT` and CANNOT be
    cleared — ``__post_init__`` re-asserts a non-blank caveat (clearing/stripping
    it raises ``ValueError``). The object is frozen, so the field cannot be
    mutated away post-construction either. :meth:`as_outcome` (the sandbox
    runner's return Mapping) ALWAYS carries the caveat.
    """

    reid_recall: float
    reid_precision: float
    reid_success_rate: float
    correct: int
    n_targets: int
    n_guesses: int
    candidate_set_size: int
    adversary_id: str
    deterministic: bool
    caveat: str = ANTI_ANONYMITY_CAVEAT

    def __post_init__(self) -> None:
        # NFR-016: the anti-anonymity caveat is non-strippable. A blank /
        # whitespace-only caveat is a strip attempt and is refused loudly.
        if not isinstance(self.caveat, str) or not self.caveat.strip():
            raise ValueError(
                "ReidSuccessMetrics.caveat is the non-strippable anti-anonymity "
                "caveat (NFR-016) and may not be blank/cleared"
            )

    def as_outcome(self) -> dict[str, Any]:
        """Return the sandbox runner's outcome Mapping (always carries the caveat)."""
        return {
            "reid_recall": self.reid_recall,
            "reid_precision": self.reid_precision,
            "reid_success_rate": self.reid_success_rate,
            "correct": self.correct,
            "n_targets": self.n_targets,
            "n_guesses": self.n_guesses,
            "candidate_set_size": self.candidate_set_size,
            "adversary_id": self.adversary_id,
            "deterministic": self.deterministic,
            "caveat": self.caveat,
        }


# --------------------------------------------------------------------------- #
# Protocols — the FR-011 ReidAttack contract + the FR-013 MiaAttack seam.      #
# --------------------------------------------------------------------------- #
@runtime_checkable
class ReidAttack(Protocol):
    """Outbound port: an interchangeable re-identification attacker (FR-011).

    A conforming object exposes a version-pinned ``adversary_id`` (so results can
    be cited as "vs adversary@version"), a ``deterministic`` flag (the headline
    adversary is deterministic — AX-002), and an ``attack`` method mapping a set
    of pseudonymous :class:`ReidTarget` s against a candidate pool of real
    :class:`ReidPersona` s, under a closed-world ``candidate_set_size`` (|C|), to
    ranked :class:`ReidGuess` es (``guessed_persona_id=None`` to abstain).
    Implementations MUST be pure in ``(targets, candidates, candidate_set_size)``
    when ``deterministic`` is ``True``.
    """

    adversary_id: str
    deterministic: bool

    def attack(
        self,
        targets: Sequence[ReidTarget],
        candidates: Sequence[ReidPersona],
        candidate_set_size: int,
    ) -> list[ReidGuess]: ...


@runtime_checkable
class MiaAttack(Protocol):
    """Outbound port seam: a membership-inference attacker (FR-013).

    Declared here so the ``attacks`` package boundary + the protocol family land
    together; the real LiRA@128 / Secret-Sharer body is S5-03 (it reuses this
    seam + the package import-boundary test). A conforming object exposes a
    version-pinned ``adversary_id`` + a ``deterministic`` flag and maps a sequence
    of opaque records to per-record membership scores.
    """

    adversary_id: str
    deterministic: bool

    def membership_scores(self, records: Sequence[Any]) -> list[float]: ...


# --------------------------------------------------------------------------- #
# Weighted-Jaccard similarity (shared helper) — pure, deterministic.          #
# --------------------------------------------------------------------------- #
def _weighted_jaccard(target_tokens: frozenset[str], persona_tokens: frozenset[str]) -> float:
    """Weighted-Jaccard overlap of two token sets in ``[0, 1]``.

    ``|intersection| / |union|`` over the surviving quasi-identifier / observed
    signal tokens (faithful to the DATA sibling's ``_weighted_jaccard``; here all
    tokens carry unit weight). Empty-vs-empty is 0.0 (no signal == no link). Pure
    + deterministic — the sole similarity primitive the baseline ranks on.
    # SWITCH-POINT(DATA): the real offline adversary weights tokens by inverse
    document frequency and consults the behavioral-signals block.
    """
    if not target_tokens or not persona_tokens:
        return 0.0
    intersection = len(target_tokens & persona_tokens)
    if intersection == 0:
        return 0.0
    union = len(target_tokens | persona_tokens)
    return intersection / union


def _target_tokens(target: ReidTarget) -> frozenset[str]:
    """The surviving signal tokens of a target (observed signals, normalised)."""
    return frozenset(s.strip().lower() for s in target.observed_signals if s and s.strip())


def _persona_tokens(persona: ReidPersona) -> frozenset[str]:
    """The linkable quasi-identifier tokens of a candidate persona (normalised)."""
    return frozenset(q.strip().lower() for q in persona.quasi_identifiers if q and q.strip())


# --------------------------------------------------------------------------- #
# The deterministic baseline adversary body.                                  #
# --------------------------------------------------------------------------- #
class BaselineDeterministicReidAttack:
    """A deterministic baseline re-identification adversary (FR-011 foundation).

    Links each pseudonymous :class:`ReidTarget` to the candidate
    :class:`ReidPersona` with the highest weighted-Jaccard overlap of surviving
    signal vs quasi-identifier tokens. Candidates are ranked by the TOTAL order
    ``(similarity DESC, persona_id ASC)`` — so the result is invariant to
    candidate input order and ties resolve deterministically by ``persona_id``
    ascending (AX-002). Abstains (``guessed_persona_id=None``, ``score=0.0``) when
    no candidate shares any surviving signal.

    Conforms to the :class:`ReidAttack` Protocol (``@runtime_checkable``). This is
    the structural stand-in for the real offline DATA adversary; the swap is a
    Pass-2 / cross-repo change (``# SWITCH-POINT(DATA)``).
    """

    def __init__(self, *, adversary_id: str = _BASELINE_ADVERSARY_ID) -> None:
        self.adversary_id: str = adversary_id
        self.deterministic: bool = True

    def attack(
        self,
        targets: Sequence[ReidTarget],
        candidates: Sequence[ReidPersona],
        candidate_set_size: int,
    ) -> list[ReidGuess]:
        # Pre-tokenise candidates once; deterministic (input-order-independent).
        candidate_tokens: list[tuple[str, frozenset[str]]] = [
            (persona.persona_id, _persona_tokens(persona)) for persona in candidates
        ]
        guesses: list[ReidGuess] = []
        for target in targets:
            t_tokens = _target_tokens(target)
            best_id: str | None = None
            best_score = 0.0
            # Total order: pick the max similarity; on a tie, the smaller
            # persona_id wins. We iterate candidates sorted by persona_id ASC so
            # the FIRST strictly-greater similarity, and the earliest id among
            # equal similarities, is selected — a deterministic total order.
            for persona_id, p_tokens in sorted(candidate_tokens, key=lambda c: c[0]):
                score = _weighted_jaccard(t_tokens, p_tokens)
                if score > best_score:
                    best_score = score
                    best_id = persona_id
            if best_id is None or best_score <= 0.0:
                guesses.append(ReidGuess(target_id=target.target_id, guessed_persona_id=None, score=0.0))
            else:
                guesses.append(
                    ReidGuess(target_id=target.target_id, guessed_persona_id=best_id, score=best_score)
                )
        return guesses


# --------------------------------------------------------------------------- #
# The integer-count success scorer.                                           #
# --------------------------------------------------------------------------- #
def score_reid_attack(
    guesses: Sequence[ReidGuess],
    targets: Sequence[ReidTarget],
    *,
    adversary_id: str,
    deterministic: bool,
    candidate_set_size: int,
) -> ReidSuccessMetrics:
    """Score re-identification guesses into integer-count success metrics.

    A guess is *correct* when its committed ``guessed_persona_id`` equals the
    true source persona of its target — by the DATA sibling's ground-truth-link
    convention, the target's own ``target_id`` IS the true ``persona_id``
    (``Target.target_id`` "equals the persona_id of its true source"). Abstains
    (``guessed_persona_id is None``) are excluded from the precision denominator.

    ``correct`` / ``n_targets`` / ``n_guesses`` are integers;
    ``reid_recall = correct / n_targets`` (0.0 if no targets),
    ``reid_precision = correct / n_guesses`` (0.0 if no committed guess),
    ``reid_success_rate = reid_recall * reid_precision``. Carries the
    non-strippable anti-anonymity caveat (NFR-016).
    """
    n_targets = len(targets)
    committed = [g for g in guesses if g.guessed_persona_id is not None]
    n_guesses = len(committed)
    correct = sum(1 for g in committed if g.guessed_persona_id == g.target_id)

    reid_recall = (correct / n_targets) if n_targets else 0.0
    reid_precision = (correct / n_guesses) if n_guesses else 0.0
    reid_success_rate = reid_recall * reid_precision

    return ReidSuccessMetrics(
        reid_recall=reid_recall,
        reid_precision=reid_precision,
        reid_success_rate=reid_success_rate,
        correct=correct,
        n_targets=n_targets,
        n_guesses=n_guesses,
        candidate_set_size=candidate_set_size,
        adversary_id=adversary_id,
        deterministic=deterministic,
    )


# --------------------------------------------------------------------------- #
# The allow-listed sandbox runner (scalar JSON-string args) + the registry.   #
# --------------------------------------------------------------------------- #
def _decode_object_array(source_json: str, *, label: str) -> list[Mapping[str, Any]]:
    """Decode ``source_json`` into a list of plain JSON objects (stdlib only).

    Shared decode path for the runner's two inputs: parse via the stdlib
    structured parser (no unsafe-deserialization), require a JSON array top level,
    and require every element to be a JSON object. It NEVER constructs objects
    named in the data; a malformed document / non-array / non-object element is
    refused loudly, naming ``label``.
    """
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
    """Read ``item[key]`` as a JSON array of strings -> a normalised str tuple.

    A missing key defaults to the empty tuple; a non-array value is refused
    loudly (the field is a token list, never a scalar/object).
    """
    raw = item.get(key, [])
    if not isinstance(raw, list):
        raise ValueError(f"{label}.{key} must be a JSON array of strings (got {type(raw).__name__})")
    return tuple(str(v) for v in raw)


def _personas_from_json(candidates_json: str) -> list[ReidPersona]:
    """Decode candidate personas from a JSON array of plain mappings.

    Reads scalar fields out of plain mappings via the shared object-array
    decoder — it NEVER constructs objects named in the data (no
    unsafe-deserialization).
    """
    return [
        ReidPersona(
            persona_id=str(item["persona_id"]),
            quasi_identifiers=_string_tuple(item, "quasi_identifiers", label="candidate"),
            source_text=str(item.get("source_text", "")),
        )
        for item in _decode_object_array(candidates_json, label="candidate")
    ]


def _targets_from_json(targets_json: str) -> list[ReidTarget]:
    """Decode targets from a JSON array of plain mappings (shared decoder)."""
    return [
        ReidTarget(
            target_id=str(item["target_id"]),
            anonymized_text=str(item.get("anonymized_text", "")),
            observed_signals=_string_tuple(item, "observed_signals", label="target"),
        )
        for item in _decode_object_array(targets_json, label="target")
    ]


def reid_attack_runner(
    *,
    targets_json: str,
    candidates_json: str,
    candidate_set_size: int,
) -> dict[str, Any]:
    """The allow-listed re-identification runner (the sandbox entrypoint).

    Scalar JSON-string args (per the ``AttackSpec`` scalar-param rule): JSON-decode
    the targets + candidates into frozen value objects (stdlib parser only — no
    unsafe-deserialization), run :class:`BaselineDeterministicReidAttack`, score,
    and return :meth:`ReidSuccessMetrics.as_outcome` — a ``Mapping[str, Any]`` (the
    sandbox contract; a non-Mapping return would be a ``SandboxViolation``). Pure
    + deterministic over its scalar inputs (AX-002).
    """
    targets = _targets_from_json(targets_json)
    candidates = _personas_from_json(candidates_json)
    adversary = BaselineDeterministicReidAttack()
    guesses = adversary.attack(targets, candidates, int(candidate_set_size))
    metrics = score_reid_attack(
        guesses,
        targets,
        adversary_id=adversary.adversary_id,
        deterministic=adversary.deterministic,
        candidate_set_size=int(candidate_set_size),
    )
    return metrics.as_outcome()


# The in-code allow-list of re-identification runners (a plain dict — the ONLY
# way a spec selects this code). Additively merged into the sandbox default
# registry (see ``attacks/__init__``) so the recon runner stays registered too.
REID_ATTACK_REGISTRY: Final[Mapping[str, Any]] = {
    "pii_anon.eval_framework.attacks.reid.reid_attack_runner": reid_attack_runner,
    "reid_baseline": reid_attack_runner,
}
