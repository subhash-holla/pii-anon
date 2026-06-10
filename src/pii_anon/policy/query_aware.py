"""S6-01 — query-aware masking gate (retain query-relevant PII, mask the rest).

DC-13 ``policy/`` surface (the FR-023 query-aware gate, UC-19). When PII is
processed in service of a *query*, this gate decides per detected span whether to
**retain** it (the query legitimately needs it) or **mask** it (the rest). It is
**subtractive-on-mask**: relative to the privacy-safe baseline (mask everything),
the gate SUBTRACTS from the masked set only the spans it can positively justify
as query-relevant. It **defaults to mask** and retains ONLY on a positive,
reason-stamped relevance signal — so the dangerous error (false-retention: leaking
a query-irrelevant PII span) cannot happen by default; over-redaction (masking a
query-relevant span) is the safe error.

What lives here
---------------
* :class:`MaskCandidate` — a detected PII span the gate decides over (the caller
  supplies the surface text; the gate NEVER re-detects).
* :class:`QueryAwareDecision` — the per-candidate retain/mask outcome + a reason.
* :class:`QueryAwareMaskingGate` — :meth:`decide` over (candidates, query).
* :class:`QueryAwareBoundReport` / :func:`score_query_aware_bound` — the
  representative FR-024 bound (over-redaction-rate + false-retention-rate vs the
  mask-all baseline) over a labelled candidate set.

Scope (Pass-2 switch-points)
----------------------------
* ``# SWITCH-POINT(ORCH)`` — the real router-pre-filter wire-in (DC-13: intercept
  + mask in ``policy/router`` before engines/prompt assembly) requires the
  orchestrator detect path, which is protected user-WIP (``orchestrator.py``); the
  gate ships as the pure primitive the wire-in will call.
* ``# SWITCH-POINT(DATA)`` — the real DATA query-aware scorer (FR-024 bound vs the
  curated UC-19 baseline corpus with CIs) + a learned relevance model are
  eval-data S5 Pass-2; the in-tree representative gate + bound ship now.

Determinism (AX-002) + isolation (AX-001)
-----------------------------------------
Pure-stdlib; the decision + the bound are pure functions; imports NO ``random`` /
``uuid`` / ``time`` / ``secrets`` and nothing from ``swarm`` / ``moe`` / ``fusion``
/ ``orchestrator`` (an AST test pins it). All spans + queries are SYNTHETIC.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MaskCandidate",
    "QueryAwareDecision",
    "QueryAwareMaskingGate",
    "QueryAwareBoundReport",
    "score_query_aware_bound",
]

# A token is a maximal run of word characters (lower-cased downstream). Used for
# both the query and the candidate-surface tokenisation — a pure, deterministic
# split (no locale / no RNG).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    """The lower-cased word-token set of ``text`` (deterministic, pure)."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


# A small fixed synonym map for the common entity types (module-level constant —
# allocated once, not per call; the Pass-2 router-pre-filter wire-in calls the
# gate per span). Deterministic. EVERY synonym token MUST be producible by
# :data:`_TOKEN_RE` (i.e. a ``[a-z0-9]+`` run) — a hyphenated/multi-word value
# (e.g. ``"e-mail"``) can never match a tokenised query and would be silent dead
# weight. # SWITCH-POINT(DATA): the real DATA scorer uses a learned relevance
# model instead of this fixed map.
_ENTITY_SYNONYMS: dict[str, frozenset[str]] = {
    "EMAIL_ADDRESS": frozenset({"email", "mail"}),
    "PHONE_NUMBER": frozenset({"phone", "telephone", "mobile", "cell"}),
    "PERSON_NAME": frozenset({"name", "person", "who"}),
    "US_SSN": frozenset({"ssn", "social", "security"}),
    "CREDIT_CARD": frozenset({"card", "credit", "payment"}),
    "STREET_ADDRESS": frozenset({"address", "street", "location"}),
}


def _entity_aliases(entity_type: str) -> frozenset[str]:
    """The query-facing alias tokens for an entity type.

    An entity type is query-named when the query mentions any of its alias tokens
    — its own underscore-split parts (``EMAIL_ADDRESS`` → ``{email, address}``)
    plus the fixed :data:`_ENTITY_SYNONYMS` map for the common types.
    Deterministic + pure.

    Note (a known heuristic limitation): underscore-split parts can be SHARED
    across entity types — e.g. ``"address"`` is a part of both ``EMAIL_ADDRESS``
    and ``STREET_ADDRESS``, so a query naming "address" matches BOTH; ``"security"``
    (a US_SSN synonym) matches a "security" query. This deterministic
    approximation is replaced by a learned relevance model at the
    ``# SWITCH-POINT(DATA)`` seam, which disambiguates with calibrated confidence.
    The collision is on the privacy-safe side relative to under-retention only when
    the query genuinely names the shared token.
    """
    parts = frozenset(p for p in entity_type.lower().split("_") if p)
    return parts | _ENTITY_SYNONYMS.get(entity_type, frozenset())


@dataclass(frozen=True)
class MaskCandidate:
    """A detected PII span the query-aware gate decides over.

    The caller supplies ``surface`` (the detected text) — the gate NEVER
    re-detects. ``field_path`` is the optional source field the span came from.
    """

    entity_type: str
    span_start: int
    span_end: int
    surface: str
    field_path: str | None = None


@dataclass(frozen=True)
class QueryAwareDecision:
    """The per-candidate retain/mask outcome with a stamped reason.

    ``retain=True`` means the gate justified keeping the span visible for the
    query; ``retain=False`` (the default) means mask it. ``reason`` is always a
    non-empty string when ``retain`` is ``True`` (the positive signal that fired);
    for a masked span it records why nothing justified retention.
    """

    candidate: MaskCandidate
    retain: bool
    reason: str


class QueryAwareMaskingGate:
    """A deterministic, subtractive-on-mask query-aware masking gate (FR-023).

    :meth:`decide` keeps a span ONLY when a positive relevance signal fires
    (the query names the entity type, or the query references the span's surface
    tokens); otherwise it masks (the privacy-safe default). Pure + deterministic
    (AX-002): the decision is a function of ``(candidates, query)`` only.
    """

    def decide(
        self, candidates: Sequence[MaskCandidate], *, query: str
    ) -> list[QueryAwareDecision]:
        """Decide retain-vs-mask for each candidate against ``query``.

        Default-to-mask: a candidate is retained ONLY on a positive signal —
        (i) the query mentions an alias of the candidate's entity type, or
        (ii) the query tokens overlap the candidate's surface tokens. An
        empty / whitespace query (no tokens) retains nothing — the gate never
        leaks by default. Order-independent + deterministic.
        """
        query_tokens = _tokens(query)
        decisions: list[QueryAwareDecision] = []
        for candidate in candidates:
            if not query_tokens:
                decisions.append(
                    QueryAwareDecision(candidate, False, "masked: empty/blank query (no signal)")
                )
                continue
            entity_hit = bool(_entity_aliases(candidate.entity_type) & query_tokens)
            surface_hit = bool(_tokens(candidate.surface) & query_tokens)
            if entity_hit and surface_hit:
                reason = "retained: entity-type AND surface-token match the query"
            elif entity_hit:
                reason = f"retained: query names the {candidate.entity_type} entity type"
            elif surface_hit:
                reason = "retained: query references the span's surface tokens"
            else:
                decisions.append(
                    QueryAwareDecision(candidate, False, "masked: no query-relevance signal")
                )
                continue
            decisions.append(QueryAwareDecision(candidate, True, reason))
        return decisions


@dataclass(frozen=True)
class QueryAwareBoundReport:
    """The representative FR-024 bound: over-redaction + false-retention vs baseline.

    * ``false_retention_rate`` — (retained ∧ ¬gold-relevant) / total: the LEAK
      rate (the bounded-dangerous error).
    * ``over_redaction_rate`` — (masked ∧ gold-relevant) / total: the SAFE error.
    * ``baseline`` — the mask-all baseline (false-retention 0; over-redaction =
      gold-relevant / total) the gate is compared against.

    # SWITCH-POINT(DATA): the real DATA scorer computes these over the curated
    UC-19 corpus with confidence intervals.
    """

    over_redaction_rate: float
    false_retention_rate: float
    n_candidates: int
    n_query_relevant_gold: int
    baseline: dict[str, float]


def score_query_aware_bound(
    decisions: Sequence[QueryAwareDecision],
    gold_relevant: Sequence[bool],
) -> QueryAwareBoundReport:
    """Score gate decisions against per-candidate gold query-relevance labels.

    Pure + deterministic integer-count rates. A ``decisions`` / ``gold_relevant``
    length mismatch is refused LOUDLY with a domain-named :class:`ValueError`
    (the S5-02/S5-03 fail-loud-domain-named discipline).
    """
    if len(decisions) != len(gold_relevant):
        raise ValueError(
            f"score_query_aware_bound length mismatch: {len(decisions)} decisions "
            f"vs {len(gold_relevant)} gold labels"
        )
    total = len(decisions)
    n_relevant = sum(1 for g in gold_relevant if g)
    false_retention = sum(
        1 for d, g in zip(decisions, gold_relevant) if d.retain and not g
    )
    over_redaction = sum(
        1 for d, g in zip(decisions, gold_relevant) if (not d.retain) and g
    )

    def _rate(n: int) -> float:
        return n / total if total else 0.0

    return QueryAwareBoundReport(
        over_redaction_rate=_rate(over_redaction),
        false_retention_rate=_rate(false_retention),
        n_candidates=total,
        n_query_relevant_gold=n_relevant,
        # The mask-all baseline: nothing retained ⇒ 0 leak, but every gold-relevant
        # span is over-redacted.
        baseline={
            "false_retention_rate": 0.0,
            "over_redaction_rate": _rate(n_relevant),
        },
    )
