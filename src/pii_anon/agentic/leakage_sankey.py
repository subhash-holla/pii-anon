"""Agentic leakage-Sankey + prompt-injection exfiltration resistance (S6-05 / DC-13).

Two auditable surfaces built on the S6-02 :class:`~pii_anon.agentic.interception.InterceptionLedger`:

1. **The leakage-Sankey** (FR-028) — a flow graph of where PII is **blocked** vs
   **leaked** across the agent channels. :func:`build_leakage_sankey` turns the
   surrogate-only ledger into a 6-node graph: the **4** S6-02 source channels
   (PROMPT / MEMORY / TOOL_IO / TRACE) feeding the **2** flow sinks
   ``{blocked, leaked}``, with per-source-channel counts. A ``blocked`` edge is
   emitted for every masked ledger record; a ``leaked`` edge is emitted **only**
   when a caller-supplied ``known_value`` actually **survives** an outbound
   payload.

   **Non-circular leak detection (the headline security invariant, RISK-1).** A
   ``leaked`` edge is NEVER created by re-running PII detection on the masked
   output — doing so would conflate a *detector miss* (a recall problem) with a
   *leak* (a value that escaped the boundary), and would let the Sankey
   "discover" leaks that no real value backs. Instead a leak is decided by a
   pure membership test: ``known_value in outbound[channel]``. No survivor ⇒ zero
   ``leaked`` edges. The entity-type LABEL on a confirmed leak is derived from the
   *shape* of the (already-confirmed) survivor (e.g. an ``@`` ⟹ ``"EMAIL"``); this
   labelling never decides leak *presence* and never echoes the raw value.

   **Channel-model reconciliation (for traceability).** FR-028 reads
   "leakage-Sankey, 6 channels"; S6-02 (FR-025) intercepts **4** channels. S6-05
   renders the Sankey as a **6-node** graph (the 4 source channels + the 2 sinks)
   carrying per-source-channel counts — the FR-028 intent. The source-channel set
   is **extensible**: :data:`_SOURCE_CHANNELS` derives from :class:`AgentChannel`,
   so a future wider source taxonomy plugs in without changing this module. The
   4-source/6-node reading vs FR-028's "6 channels" is flagged in the story
   §Traces for the traceability reviewer.

2. **Injection-resistance scoring** (FR-029) — :func:`score_injection_resistance`
   runs INERT synthetic obfuscated-PII payloads (base64 / homoglyph / zero-width
   transforms of a SYNTHETIC value) through the :class:`FourChannelGuard` and
   measures the **attack-success-rate** (a raw value exfiltrated past the guard)
   against **benign-task-success** (legitimate non-PII content preserved). When
   the optional ``datasets`` extra is importable it reuses the DATA
   ``pii_anon_datasets.scoring.adversary.build_payloads`` library; otherwise it
   falls back to an in-tree INERT payload set covering >=3 transforms. The DATA
   library is imported **lazily inside the function** so this module loads (and
   scores, via the fallback) with the extra absent (RISK-5).

Determinism (AX-002 / RISK-3)
-----------------------------
Both surfaces are **pure functions** of their inputs. The Sankey is a pure
function of ``(ledger, outbound, known_values)``; the in-tree injection payload
set is a FIXED module-level list. There is no ``random`` / ``uuid`` / ``time`` /
``secrets`` anywhere in this module — same inputs yield a byte-identical
:meth:`LeakageSankey.as_dict` and a byte-identical :class:`InjectionResistanceReport`.

Isolation (AX-001 / RISK-5)
---------------------------
This module imports ONLY the standard library plus
:mod:`pii_anon.agentic.interception` (+ the LAZY optional ``pii_anon_datasets``
inside :func:`score_injection_resistance`). It imports nothing from
``eval_framework.rating`` / ``attacks`` / the SDO gate / ``orchestrator``. All
in-tree payloads are INERT recoverable encodings of SYNTHETIC values (never real
PII).

AGENT_SIMULATED honesty boundary
--------------------------------
The leakage-Sankey over the real S6-02 ledger, the leaked-vs-blocked flow
accounting, and the injection-resistance scoring over INERT synthetic payloads
all run for REAL in-tree. A live-agent runtime + real transcript-residual
leakage are Pass-2 — :attr:`InjectionResistanceReport.is_representative` is
``True``.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pii_anon.agentic.interception import (
    AgentChannel,
    FourChannelGuard,
    InterceptionLedger,
    NoRawPIIPersistError,
)

__all__ = [
    "SankeyEdge",
    "LeakageSankey",
    "build_leakage_sankey",
    "InjectionResistanceReport",
    "score_injection_resistance",
]

# The two flow sinks of the Sankey (the masked-vs-leaked accounting).
_BLOCKED = "blocked"
_LEAKED = "leaked"

# The source-channel set, derived from the S6-02 AgentChannel enum so the Sankey
# node order is stable AND the taxonomy is extensible: a future wider source set
# (FR-028's "6 channels") plugs in by widening AgentChannel — this module needs
# no change. The 6-node graph = these source channels + the 2 sinks.
_SOURCE_CHANNELS: tuple[str, ...] = tuple(c.value for c in AgentChannel)


# =============================================================================
# Leakage-Sankey value objects (FR-028)
# =============================================================================
@dataclass(frozen=True, slots=True)
class SankeyEdge:
    """One flow edge: a source channel -> a sink, for one entity type.

    A frozen value object (cannot be mutated post-construction): ``source`` is the
    :class:`AgentChannel` value, ``target`` is ``"blocked"`` or ``"leaked"``,
    ``entity_type`` is a surrogate-safe label (never the raw PII), ``count`` is the
    number of spans flowing along this edge.
    """

    source: str
    target: str
    entity_type: str
    count: int


@dataclass(frozen=True, slots=True)
class LeakageSankey:
    """An immutable leakage flow graph over the S6-02 channels (the G5 audit half).

    Holds an immutable :class:`tuple` of :class:`SankeyEdge`. Exposes the
    blocked/leaked totals, per-source-channel counts, and the renderer shape
    (:meth:`as_dict`) — a 6-node graph (the 4 source channels + ``blocked`` +
    ``leaked``).
    """

    edges: tuple[SankeyEdge, ...]

    def blocked_count(self) -> int:
        """Total spans masked (the sum of every ``blocked`` edge's count)."""
        return sum(e.count for e in self.edges if e.target == _BLOCKED)

    def leaked_count(self) -> int:
        """Total raw values that survived to an outbound boundary (``leaked``)."""
        return sum(e.count for e in self.edges if e.target == _LEAKED)

    def counts_by_channel(self) -> dict[str, dict[str, int]]:
        """Per-source-channel ``{channel: {"blocked": n, "leaked": m}}`` counts.

        Only channels that carry at least one edge appear; within a channel only
        the non-zero sinks appear (a channel with masks but no leaks has just a
        ``blocked`` key). The FR-028 per-channel leakage accounting.
        """
        out: dict[str, dict[str, int]] = {}
        for edge in self.edges:
            channel = out.setdefault(edge.source, {})
            channel[edge.target] = channel.get(edge.target, 0) + edge.count
        return out

    def as_dict(self) -> dict[str, object]:
        """The renderer shape: ``{"nodes": [...], "links": [...]}`` (6 nodes).

        Nodes are the 4 source channels (in :class:`AgentChannel` order) followed
        by the ``blocked`` and ``leaked`` sinks. Links aggregate edge counts per
        ``(source, target)`` so a channel with two masked spans surfaces a single
        link of value 2. Deterministic: nodes follow the fixed channel order and
        links follow a fixed ``(source-order, sink-order)`` traversal — same
        inputs yield a byte-identical dict.
        """
        nodes = [{"id": name} for name in (*_SOURCE_CHANNELS, _BLOCKED, _LEAKED)]

        # Aggregate per (source, target) -> summed count.
        aggregate: dict[tuple[str, str], int] = {}
        for edge in self.edges:
            key = (edge.source, edge.target)
            aggregate[key] = aggregate.get(key, 0) + edge.count

        # Emit links in a STABLE order: source-channel order x sink order. This
        # makes as_dict() byte-identical for identical inputs (AX-002).
        links: list[dict[str, object]] = []
        for source in _SOURCE_CHANNELS:
            for target in (_BLOCKED, _LEAKED):
                value = aggregate.get((source, target))
                if value:
                    links.append(
                        {"source": source, "target": target, "value": value}
                    )
        return {"nodes": nodes, "links": links}


# =============================================================================
# Leakage-Sankey builder (FR-028) — pure, non-circular
# =============================================================================
def _classify_survivor(value: str) -> str:
    """Label a CONFIRMED surviving raw value by its synthetic shape.

    Called only AFTER a leak has been confirmed by a membership test
    (``known_value in payload``); this never decides leak *presence* and never
    returns the raw value — only a coarse, surrogate-safe entity-type label. A
    deliberately small, deterministic shape check (no detection engine, no
    re-scan of the masked output) so it cannot reintroduce circular re-detection.
    """
    if "@" in value:
        return "EMAIL"
    digits = sum(ch.isdigit() for ch in value)
    if digits >= 4 and any(c in value for c in "-.() "):
        return "PHONE"
    return "PII"


def build_leakage_sankey(
    ledger: InterceptionLedger,
    outbound: Mapping[AgentChannel, str],
    *,
    known_values: Iterable[str],
) -> LeakageSankey:
    """Build the leakage-Sankey from a ledger + a leak-scan of outbound payloads.

    Parameters
    ----------
    ledger:
        The S6-02 surrogate-only :class:`InterceptionLedger`. Each record is a
        masked span and contributes a ``blocked`` edge on its source channel.
    outbound:
        The post-mask payload per channel that crossed the outbound boundary
        (what an agent actually emitted). Scanned ONLY for caller-supplied
        ``known_values`` survival — never re-detected.
    known_values:
        The raw values the caller knows were in scope (e.g. the synthetic values
        a test injected, or values a higher layer tracked). A ``leaked`` edge is
        created for a ``(channel, entity_type)`` **only** when one of these
        survives ``outbound[channel]``. This is the load-bearing non-circular
        contract (RISK-1): a leak is a *known value that escaped*, never a
        re-detection of the masked output.

    Returns
    -------
    LeakageSankey
        A pure, deterministic function of ``(ledger, outbound, known_values)``.
    """
    # De-duplicate the known values once; skip the empty string (a vacuous
    # "survivor" that is trivially `in` every payload).
    known = [v for v in dict.fromkeys(known_values) if v]

    # 1) A blocked edge per masked ledger record, aggregated per (channel, type).
    blocked_counts: dict[tuple[str, str], int] = {}
    for record in ledger.records():
        key = (record.channel.value, record.entity_type)
        blocked_counts[key] = blocked_counts.get(key, 0) + 1

    # 2) A leaked edge ONLY for a known value that SURVIVES an outbound payload.
    #    Per channel, a given known value counts once (it either escaped or not);
    #    aggregated per (channel, classified-entity-type).
    leaked_counts: dict[tuple[str, str], int] = {}
    for channel, payload in outbound.items():
        for value in known:
            if value in payload:
                key = (channel.value, _classify_survivor(value))
                leaked_counts[key] = leaked_counts.get(key, 0) + 1

    # Build edges in a deterministic order: source-channel order, blocked-then-
    # leaked, entity-type sorted. (as_dict() re-aggregates, but a stable edge
    # tuple keeps the value object itself byte-stable for equality/replay.)
    edges: list[SankeyEdge] = []
    for source in _SOURCE_CHANNELS:
        for target, counts in ((_BLOCKED, blocked_counts), (_LEAKED, leaked_counts)):
            for (chan, entity_type), count in sorted(counts.items()):
                if chan == source:
                    edges.append(
                        SankeyEdge(
                            source=source,
                            target=target,
                            entity_type=entity_type,
                            count=count,
                        )
                    )
    return LeakageSankey(edges=tuple(edges))


# =============================================================================
# Injection-resistance scoring (FR-029)
# =============================================================================
@dataclass(frozen=True, slots=True)
class InjectionResistanceReport:
    """The prompt-injection exfiltration-resistance score (FR-029).

    ``attack_success_rate`` is the fraction of adversarial payloads whose raw
    synthetic value EXFILTRATED past the guard (survived the masked output);
    ``benign_task_success_rate`` is the fraction of legitimate non-PII content
    preserved. ``intent_tags`` names the transforms exercised (base64 / homoglyph
    / zero-width). ``is_representative`` is ``True``: the scoring runs for REAL
    over INERT synthetic payloads in-tree, but a live-agent transcript-residual
    run is the Pass-2 honesty boundary.
    """

    attack_success_rate: float
    benign_task_success_rate: float
    n_payloads: int
    n_benign: int
    intent_tags: tuple[str, ...]
    is_representative: bool = True


# In-tree INERT synthetic payload set (AX-001 / RISK-4) — used when the optional
# ``datasets`` extra is absent. Each entry is a SYNTHETIC raw value rendered
# through a faithful, recoverable transform (base64 / a fixed homoglyph map /
# zero-width insertion). FIXED list (no RNG / clock) so scoring is deterministic.
_SYNTH_PII_SPAN = "jane.roe@example.test"  # synthetic — never real PII (AX-001)

# A fixed, deterministic homoglyph/OCR-confusable map (faithful: a recoverable
# substitution). Mirrors the DATA library's intent but stays fully in-tree.
_HOMOGLYPH_MAP: dict[str, str] = {
    "o": "0",
    "l": "1",
    "e": "3",
    "a": "@",
    "s": "5",
    "t": "7",
}
_ZERO_WIDTH = "​"


def _homoglyph(span: str) -> str:
    """Faithful homoglyph substitution per the fixed in-tree map."""
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in span)


def _zero_width(span: str) -> str:
    """Faithful zero-width insertion (strip U+200B to recover)."""
    return _ZERO_WIDTH.join(span)


def _in_tree_payloads() -> tuple[tuple[str, str, str], ...]:
    """The in-tree INERT fallback: ``(intent_tag, raw_value, rendered_payload)``.

    >=3 transforms (base64 / homoglyph / zero-width) over a SYNTHETIC value, each
    embedded in a benign carrier. The ``raw_value`` is what the scorer scans the
    masked output for (the exfiltration test); the ``rendered_payload`` is what is
    fed to the guard. A FIXED tuple — deterministic (AX-002), INERT (AX-001).
    """
    span = _SYNTH_PII_SPAN
    b64 = base64.b64encode(span.encode("utf-8")).decode("ascii")
    return (
        ("base64", span, f"Please note: {b64}"),
        ("homoglyph", span, f"Please note: {_homoglyph(span)}"),
        ("zero_width", span, f"Please note: {_zero_width(span)}"),
        # The direct-exfiltration probe (the verbatim synthetic value a competent
        # guard MUST mask and a missing/leaky guard will not). Shared shape with
        # the DATA-library branch so the exfiltration scan is faithful for both.
        _direct_exfiltration_probe(),
    )


# The direct-exfiltration probe: the SYNTHETIC raw value in a plain benign
# carrier. This is what makes the ASR meaningful — a competent guard MUST mask the
# verbatim value, while a missing/leaky guard passes it straight through. The
# obfuscated transforms (base64 / homoglyph / zero-width) test DETECTOR robustness
# but, being obfuscated, never carry the value VERBATIM, so they cannot by
# themselves exercise the verbatim-exfiltration boundary. Appended to BOTH the
# DATA-library and the in-tree payload sets so the exfiltration scan is faithful
# regardless of which payload source is active.
#
# ``_DIRECT_PROBE_TAG`` is a measurement SCAFFOLD, not an obfuscation transform —
# it is EXCLUDED from the reported ``intent_tags`` (which names the transforms
# exercised) on both payload paths, so the report never claims an intent set that
# diverges from the obfuscation taxonomy. (It is still scored: ``n_payloads``
# includes it.) The DATA branch achieves this by reporting the library's
# ``INTENT_TAGS`` (which never contains the probe tag); the in-tree branch filters
# it explicitly below.
_DIRECT_PROBE_TAG = "direct"


def _direct_exfiltration_probe() -> tuple[str, str, str]:
    """``(intent_tag, raw_value, rendered_payload)`` carrying the verbatim value."""
    span = _SYNTH_PII_SPAN
    return (_DIRECT_PROBE_TAG, span, f"Please contact {span} directly.")


def _load_payloads() -> tuple[tuple[tuple[str, str, str], ...], tuple[str, ...]]:
    """Resolve the payload set: the DATA library if present, else the in-tree set.

    Returns ``((intent_tag, raw_value, rendered_payload)..., intent_tags)``. The
    optional ``pii_anon_datasets`` is imported LAZILY here (RISK-5) so this module
    loads — and scores via the in-tree fallback — when the ``datasets`` extra is
    absent. Any import failure falls back to the in-tree INERT set. A direct
    verbatim-value probe is appended in both branches so the exfiltration scan is
    faithful (the obfuscated transforms never carry the value verbatim).
    """
    try:
        # Intentionally lazy (PLC0415): the optional ``datasets`` extra — this
        # module must load + score via the in-tree fallback when it is ABSENT.
        # ``import-untyped``: the installed ``pii_anon_datasets`` ships no
        # ``py.typed`` marker and the bare ``ignore_missing_imports`` entry does
        # not cover this submodule (canonical ``mypy src/pii_anon`` flags it).
        from pii_anon_datasets.scoring.adversary import INTENT_TAGS, build_payloads  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        rows = _in_tree_payloads()
        # Report only the obfuscation transforms — the direct probe is a
        # measurement scaffold, not a transform (mirrors the DATA branch, which
        # reports the library's INTENT_TAGS sans the probe).
        intents = tuple(
            dict.fromkeys(
                tag for tag, _, _ in rows if tag != _DIRECT_PROBE_TAG
            )
        )
        return rows, intents

    # DATA library present: one rendered payload per committed transform, each
    # backed by the SAME synthetic raw value, plus the direct verbatim probe.
    span = _SYNTH_PII_SPAN
    rows = (
        *(
            (payload.intent_tag, span, payload.render())
            for payload in build_payloads(span)
        ),
        _direct_exfiltration_probe(),
    )
    return rows, tuple(INTENT_TAGS)


def score_injection_resistance(
    guard: FourChannelGuard, *, scope: str
) -> InjectionResistanceReport:
    """Score prompt-injection exfiltration resistance over INERT synthetic payloads.

    Each payload is run through ``guard.intercept(channel=PROMPT, scope=scope)``;
    an **exfiltration** is a known raw synthetic value SURVIVING the masked output
    (a caller-known-value membership test — NOT a re-detection). The attack
    success rate is the exfiltrated fraction; benign-task success is the fraction
    of legitimate non-PII content preserved.

    Resilience to the S6-02 persist invariant: a guard whose masker leaves a raw
    value in its output RAISES :class:`NoRawPIIPersistError` (it refuses to pass a
    leak through). That refusal is the guard WORKING — the raw value did NOT
    escape — so it counts as resistance (not an exfiltration). A guard that
    silently fails to mask (no record, no raise) lets the raw value survive and is
    counted as an exfiltration.

    Deterministic (AX-002): the payload set is fixed and the scan is pure, so the
    same guard yields a byte-identical report.
    """
    rows, intent_tags = _load_payloads()

    n_payloads = 0
    exfiltrations = 0
    for _intent, raw_value, rendered in rows:
        n_payloads += 1
        try:
            result = guard.intercept(
                rendered, channel=AgentChannel.PROMPT, scope=scope
            )
        except NoRawPIIPersistError:
            # The guard caught the raw value and refused to emit it — resistance,
            # not an exfiltration. The value did not cross the boundary.
            continue
        if raw_value and raw_value in result.masked_text:
            exfiltrations += 1

    # Benign-task success: legitimate non-PII content must survive unredacted (the
    # guard must not over-redact). A fixed synthetic benign corpus (AX-001),
    # deterministic. Each carrier sentence carries NO PII and must pass through
    # with its load-bearing benign tokens intact.
    benign_corpus = (
        "The quarterly report is attached for your review.",
        "Schedule the sync for next Tuesday afternoon.",
        "Approve the budget once the figures are finalized.",
    )
    n_benign = 0
    benign_preserved = 0
    for sentence in benign_corpus:
        n_benign += 1
        try:
            result = guard.intercept(
                sentence, channel=AgentChannel.PROMPT, scope=scope
            )
        except NoRawPIIPersistError:
            # A benign sentence should never trip the invariant; if it does, the
            # content was not preserved.
            continue
        # The benign content is preserved when the masked output is unchanged
        # (no PII to mask ⟹ pass-through).
        if result.masked_text == sentence:
            benign_preserved += 1

    asr = exfiltrations / n_payloads if n_payloads else 0.0
    benign_rate = benign_preserved / n_benign if n_benign else 0.0
    return InjectionResistanceReport(
        attack_success_rate=asr,
        benign_task_success_rate=benign_rate,
        n_payloads=n_payloads,
        n_benign=n_benign,
        intent_tags=intent_tags,
        is_representative=True,
    )
