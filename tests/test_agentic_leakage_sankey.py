"""S6-05 — agentic leakage-Sankey + prompt-injection exfiltration resistance (DC-13).

Consumes the S6-02 ``InterceptionLedger`` (DONE) to build a **leakage-Sankey** —
a 6-node flow graph (the 4 ``AgentChannel`` source channels + the ``blocked`` and
``leaked`` sinks) accounting for where PII is masked (``blocked``) vs where a raw
value survived into an outbound payload (``leaked``) — and scores
**prompt-injection exfiltration resistance** (attack-success-rate vs benign-task
success) by running INERT synthetic payloads through the ``FourChannelGuard``.

These tests run for REAL in-tree against SYNTHETIC text (AX-001). A live-agent
runtime + real transcript-residual leakage are the Pass-2 / AGENT_SIMULATED
honesty boundary (``is_representative=True``).

Headline security invariant (A2/A3): a ``leaked`` edge is created ONLY when a
caller-supplied ``known_value`` survives an outbound payload — NEVER by re-running
detection on the masked output (which would conflate a detector miss with a leak,
i.e. circular). No survivor ⇒ ``leaked_count() == 0``.

Acceptance map (story §3):
    A1  Sankey blocked edges sum to the intercepted total   [UNIT]
    A2  leaked edge when a raw value survives outbound       [SECURITY]
    A3  no leaked edge without a survivor (non-circular)     [SECURITY]
    A4  blocked edge from each masked ledger record          [UNIT]
    A5  as_dict() renderer shape — 6 nodes                   [CONTRACT]
    A6  deterministic Sankey + report                        [PROPERTY]
    A7  injection ASR zero when all masked                    [SECURITY]
    A8  injection detects exfiltration                        [SECURITY]
    A9  benign-task-success tracked                           [UNIT]
    A10 in-tree payloads w/o datasets extra + >=3 intents     [UNIT][AUDIT]
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import json
from pathlib import Path

import pytest

from pii_anon.agentic.interception import (
    AgentChannel,
    ChannelResult,
    FourChannelGuard,
    InterceptionLedger,
    InterceptionRecord,
)
from pii_anon.agentic.leakage_sankey import (
    InjectionResistanceReport,
    LeakageSankey,
    SankeyEdge,
    build_leakage_sankey,
    score_injection_resistance,
)

# --- synthetic fixtures (AX-001 — never real PII) ------------------------------
SYNTH_EMAIL = "jane.roe@example.test"
SYNTH_EMAIL_2 = "john.doe@example.test"
SYNTH_PHONE = "555-0100"
SCOPE = "case_s6_05"
FIXED_KEY = b"\x11" * 32

# Sink-node names (the 2 flow sinks of the 6-node graph).
BLOCKED = "blocked"
LEAKED = "leaked"


# --- helpers -------------------------------------------------------------------
def _ledger_with(*records: InterceptionRecord) -> InterceptionLedger:
    """A ledger pre-populated with the given surrogate records."""
    ledger = InterceptionLedger()
    ledger.append(records)
    return ledger


def _rec(
    channel: AgentChannel,
    entity_type: str = "EMAIL",
    surrogate: str = "<EMAIL:v1:tok_aaaaaaaaaaaa>",
    span_start: int = 0,
    span_end: int = 5,
) -> InterceptionRecord:
    """A surrogate-only interception record (the ledger's unit)."""
    return InterceptionRecord(
        channel=channel,
        entity_type=entity_type,
        surrogate=surrogate,
        span_start=span_start,
        span_end=span_end,
        scope=SCOPE,
    )


class _MissingMasker:
    """A masker that FAILS to detect the payload PII (returns text unchanged).

    Used by A8: it neither raises (it reports no record, so the S6-02
    ``_assert_no_raw_pii`` invariant derives no known-value from records and
    passes) NOR masks anything — so a caller-supplied known raw value survives in
    the masked output. The injection scorer's own caller-supplied known-value scan
    (NOT a re-detection) catches the exfiltration. This models a real masker that
    misses obfuscated PII, without tripping the upstream persist invariant.
    """

    def mask(
        self, text: str, *, channel: AgentChannel, scope: str
    ) -> ChannelResult:
        return ChannelResult(channel=channel, masked_text=text, records=())


# ==============================================================================
# A1 — Sankey blocked edges sum to the intercepted total [UNIT] (FR-028)
# ==============================================================================
def test_fr028_a1_blocked_edges_sum_to_intercepted_total() -> None:
    """N ledger records ⟹ blocked edges summing to N (+ any leaked survivors)."""
    ledger = _ledger_with(
        _rec(AgentChannel.PROMPT),
        _rec(AgentChannel.MEMORY),
        _rec(AgentChannel.TOOL_IO),
        _rec(AgentChannel.TRACE),
    )
    sankey = build_leakage_sankey(ledger, outbound={}, known_values=[])

    assert isinstance(sankey, LeakageSankey)
    # Exactly N=4 masked spans ⟹ blocked_count == 4, no survivors ⟹ leaked == 0.
    assert sankey.blocked_count() == 4
    assert sankey.leaked_count() == 0
    # The blocked edge counts sum to the ledger record total.
    blocked_total = sum(
        e.count for e in sankey.edges if e.target == BLOCKED
    )
    assert blocked_total == len(ledger.records())


# ==============================================================================
# A2 — leaked edge when a raw value survives outbound [SECURITY] (headline)
# ==============================================================================
def test_fr028_a2_leaked_edge_when_known_value_survives_outbound() -> None:
    """A known raw value still present in an outbound payload ⟹ a leaked edge."""
    ledger = _ledger_with(_rec(AgentChannel.PROMPT, entity_type="EMAIL"))
    # The outbound PROMPT payload STILL contains the raw synthetic email — a leak.
    outbound = {AgentChannel.PROMPT: f"forwarded: {SYNTH_EMAIL} (oops)"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[SYNTH_EMAIL]
    )

    assert sankey.leaked_count() > 0
    leaked_edges = [e for e in sankey.edges if e.target == LEAKED]
    assert len(leaked_edges) == 1
    leak = leaked_edges[0]
    assert leak.source == AgentChannel.PROMPT.value
    assert leak.target == LEAKED
    # per-channel counts surface the leak on the PROMPT source channel.
    by_channel = sankey.counts_by_channel()
    assert by_channel[AgentChannel.PROMPT.value][LEAKED] == 1


def test_fr028_a2_leaked_entity_type_inferred_from_surviving_value() -> None:
    """The leaked edge carries an entity_type derived from the survivor, not raw."""
    ledger = _ledger_with(_rec(AgentChannel.TOOL_IO))
    outbound = {AgentChannel.TOOL_IO: f"call(arg={SYNTH_EMAIL})"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[SYNTH_EMAIL]
    )
    leak = next(e for e in sankey.edges if e.target == LEAKED)
    # entity_type is a non-empty label; it is NEVER the raw PII value itself.
    assert leak.entity_type
    assert SYNTH_EMAIL not in leak.entity_type


# ==============================================================================
# A3 — NO leaked edge without a survivor [SECURITY] (non-circular)
# ==============================================================================
def test_fr028_a3_no_leaked_edge_without_survivor_non_circular() -> None:
    """Outbound payloads with NO known-value survivor ⟹ leaked_count() == 0.

    The headline non-circular guarantee: a leaked edge is NEVER created by
    re-detecting PII on the masked output — only by a caller-supplied known_value
    actually surviving. A fully-masked outbound (surrogates only) has no survivor.
    """
    ledger = _ledger_with(
        _rec(AgentChannel.PROMPT), _rec(AgentChannel.MEMORY)
    )
    # Outbound payloads carry ONLY surrogates — no raw value present at all.
    outbound = {
        AgentChannel.PROMPT: "forwarded: <EMAIL:v1:tok_aaaaaaaaaaaa> ok",
        AgentChannel.MEMORY: "noted <EMAIL:v1:tok_bbbbbbbbbbbb>",
    }
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[SYNTH_EMAIL, SYNTH_EMAIL_2]
    )
    assert sankey.leaked_count() == 0
    assert all(e.target == BLOCKED for e in sankey.edges)


def test_fr028_a3_masked_output_is_never_re_detected() -> None:
    """Even if the masked output *looks* like it contains PII-shaped text, no
    leaked edge appears unless a caller-supplied known_value literally survives.

    This pins that detection is NEVER re-run on the masked output: an outbound
    payload containing an email-SHAPED surrogate-ish token that is NOT in
    known_values yields zero leaks (no circular re-detection).
    """
    ledger = _ledger_with(_rec(AgentChannel.PROMPT))
    # Contains an email-LOOKING string, but it is NOT a caller-declared
    # known_value, so it must NOT be counted as a leak (no re-detection).
    outbound = {AgentChannel.PROMPT: "audit ref: noreply@masked.invalid"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[SYNTH_EMAIL]
    )
    assert sankey.leaked_count() == 0


def test_fr028_a3_empty_known_values_yields_zero_leaks() -> None:
    """With no known_values at all, leaked_count is always 0 regardless of outbound."""
    ledger = _ledger_with(_rec(AgentChannel.PROMPT))
    outbound = {AgentChannel.PROMPT: f"raw {SYNTH_EMAIL} present but undeclared"}
    sankey = build_leakage_sankey(ledger, outbound=outbound, known_values=[])
    assert sankey.leaked_count() == 0


# ==============================================================================
# A4 — blocked edge from each masked ledger record [UNIT] (FR-028)
# ==============================================================================
def test_fr028_a4_blocked_edge_per_masked_record_on_its_channel() -> None:
    """Each masked ledger record yields a blocked edge on its source channel."""
    ledger = _ledger_with(
        _rec(AgentChannel.PROMPT, entity_type="EMAIL"),
        _rec(AgentChannel.PROMPT, entity_type="PHONE"),
        _rec(AgentChannel.MEMORY, entity_type="EMAIL"),
    )
    sankey = build_leakage_sankey(ledger, outbound={}, known_values=[])
    by_channel = sankey.counts_by_channel()
    # 2 blocked on PROMPT, 1 on MEMORY.
    assert by_channel[AgentChannel.PROMPT.value][BLOCKED] == 2
    assert by_channel[AgentChannel.MEMORY.value][BLOCKED] == 1
    # Channels with no records carry no blocked count.
    assert AgentChannel.TRACE.value not in by_channel


# ==============================================================================
# A5 — as_dict() renderer shape — 6 nodes [CONTRACT] (FR-028)
# ==============================================================================
def test_fr028_a5_as_dict_renderer_shape_six_nodes() -> None:
    """as_dict() ⟹ {"nodes":[...],"links":[...]} with the 4 channels + 2 sinks."""
    ledger = _ledger_with(
        _rec(AgentChannel.PROMPT),
        _rec(AgentChannel.MEMORY),
    )
    outbound = {AgentChannel.PROMPT: f"leak {SYNTH_EMAIL}"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[SYNTH_EMAIL]
    )
    payload = sankey.as_dict()

    assert set(payload) == {"nodes", "links"}
    node_ids = [n["id"] for n in payload["nodes"]]
    # Exactly the 6-node graph: 4 channel source nodes + blocked + leaked sinks.
    assert node_ids == [
        AgentChannel.PROMPT.value,
        AgentChannel.MEMORY.value,
        AgentChannel.TOOL_IO.value,
        AgentChannel.TRACE.value,
        BLOCKED,
        LEAKED,
    ]
    assert len(payload["nodes"]) == 6
    # Links carry per-channel source->target counts.
    for link in payload["links"]:
        assert set(link) >= {"source", "target", "value"}
        assert link["source"] in node_ids
        assert link["target"] in {BLOCKED, LEAKED}
        assert isinstance(link["value"], int)
    # The whole artifact round-trips JSON and never leaks the raw value.
    blob = json.dumps(payload)
    assert SYNTH_EMAIL not in blob


def test_fr028_a5_as_dict_links_aggregate_per_channel_target() -> None:
    """A channel with 2 blocked spans surfaces a single link of value 2."""
    ledger = _ledger_with(
        _rec(AgentChannel.MEMORY),
        _rec(AgentChannel.MEMORY),
    )
    sankey = build_leakage_sankey(ledger, outbound={}, known_values=[])
    links = sankey.as_dict()["links"]
    mem_blocked = [
        link
        for link in links
        if link["source"] == AgentChannel.MEMORY.value
        and link["target"] == BLOCKED
    ]
    assert len(mem_blocked) == 1
    assert mem_blocked[0]["value"] == 2


# ==============================================================================
# A6 — deterministic Sankey + report [PROPERTY] (AX-002)
# ==============================================================================
def test_ax002_a6_sankey_is_byte_identical_for_same_inputs() -> None:
    """Same ledger + outbound + known_values ⟹ byte-identical as_dict()."""
    def _build() -> dict[str, object]:
        ledger = _ledger_with(
            _rec(AgentChannel.PROMPT),
            _rec(AgentChannel.MEMORY),
            _rec(AgentChannel.TOOL_IO),
        )
        outbound = {AgentChannel.PROMPT: f"leak {SYNTH_EMAIL}"}
        return build_leakage_sankey(
            ledger, outbound=outbound, known_values=[SYNTH_EMAIL]
        ).as_dict()

    first = json.dumps(_build(), sort_keys=True)
    second = json.dumps(_build(), sort_keys=True)
    assert first == second
    # And the *unsorted* serialization is also identical (stable ordering).
    assert json.dumps(_build()) == json.dumps(_build())


def test_ax002_a6_injection_report_is_byte_identical() -> None:
    """Same guard inputs ⟹ byte-identical InjectionResistanceReport."""
    def _report() -> InjectionResistanceReport:
        guard = FourChannelGuard(masker=None, surrogate_key=FIXED_KEY)
        return score_injection_resistance(guard, scope=SCOPE)

    a = _report()
    b = _report()
    assert a == b
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_ax002_a6_no_nondeterministic_imports_in_module() -> None:
    """The leakage_sankey module imports no random/uuid/time/secrets (AX-002)."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pii_anon"
        / "agentic"
        / "leakage_sankey.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    banned = {"random", "uuid", "time", "secrets"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), f"non-deterministic import(s): {imported & banned}"


# ==============================================================================
# A7 — injection ASR zero when all masked [SECURITY] (FR-029)
# ==============================================================================
def test_fr029_a7_asr_zero_when_guard_masks_everything() -> None:
    """When the guard masks every payload's PII, attack_success_rate == 0.0."""
    guard = FourChannelGuard(masker=None, surrogate_key=FIXED_KEY)
    report = score_injection_resistance(guard, scope=SCOPE)

    assert isinstance(report, InjectionResistanceReport)
    assert report.attack_success_rate == 0.0  # no exfiltration
    assert report.n_payloads > 0
    assert report.is_representative is True


# ==============================================================================
# A8 — injection detects exfiltration [SECURITY] (FR-029 headline)
# ==============================================================================
def test_fr029_a8_asr_positive_when_guard_leaks_a_raw_value() -> None:
    """A guard that fails to mask a payload's raw value ⟹ ASR > 0.0.

    The exfiltration is detected by the scorer's caller-supplied known-value scan
    over the masked output (NOT by re-detection): a raw synthetic value surviving
    the guard's output is an exfiltration.
    """
    leaky_guard = FourChannelGuard(masker=_MissingMasker())
    report = score_injection_resistance(leaky_guard, scope=SCOPE)
    assert report.attack_success_rate > 0.0


def test_fr029_a8_asr_is_a_unit_interval_rate() -> None:
    """attack_success_rate is a proper rate in [0.0, 1.0]."""
    leaky = score_injection_resistance(
        FourChannelGuard(masker=_MissingMasker()), scope=SCOPE
    )
    masked = score_injection_resistance(
        FourChannelGuard(masker=None, surrogate_key=FIXED_KEY), scope=SCOPE
    )
    for report in (leaky, masked):
        assert 0.0 <= report.attack_success_rate <= 1.0
        assert 0.0 <= report.benign_task_success_rate <= 1.0
    # The leaky guard's ASR is strictly higher than the masking guard's.
    assert leaky.attack_success_rate > masked.attack_success_rate


# ==============================================================================
# A9 — benign-task-success tracked [UNIT] (FR-029)
# ==============================================================================
def test_fr029_a9_benign_task_success_tracked() -> None:
    """Legitimate (non-PII) content surviving masking ⟹ benign_task_success_rate."""
    guard = FourChannelGuard(masker=None, surrogate_key=FIXED_KEY)
    report = score_injection_resistance(guard, scope=SCOPE)
    # A well-behaved guard preserves benign content (it does not over-redact).
    assert report.n_benign > 0
    assert report.benign_task_success_rate == 1.0


# ==============================================================================
# A10 — in-tree payloads w/o datasets extra + >=3 intents [UNIT][AUDIT]
# ==============================================================================
def test_fr029_a10_in_tree_payloads_without_datasets_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With pii_anon_datasets ABSENT, the in-tree INERT fallback is used and
    covers >=3 transform intents (base64 / homoglyph / zero-width)."""
    real_import = builtins.__import__

    def _no_datasets(name: str, *args: object, **kwargs: object) -> object:
        if name == "pii_anon_datasets" or name.startswith("pii_anon_datasets."):
            raise ModuleNotFoundError("No module named 'pii_anon_datasets'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_datasets)

    guard = FourChannelGuard(masker=None, surrogate_key=FIXED_KEY)
    report = score_injection_resistance(guard, scope=SCOPE)

    # The fallback still scores and covers >=3 distinct transform intents.
    assert report.n_payloads > 0
    assert len(set(report.intent_tags)) >= 3
    assert {"base64", "zero_width"}.issubset(set(report.intent_tags))
    # A homoglyph/ocr-style transform is present (named 'homoglyph' or 'ocr').
    assert {"homoglyph", "ocr"} & set(report.intent_tags)


def test_ax001_a10_import_isolation_ast() -> None:
    """AST scan: leakage_sankey.py imports only stdlib + pii_anon.agentic
    (+ the LAZY pii_anon_datasets inside the function) and NOTHING from
    eval_framework.rating / attacks / the SDO gate / orchestrator."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pii_anon"
        / "agentic"
        / "leakage_sankey.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))

    forbidden_substrings = (
        "eval_framework.rating",
        "eval_framework.attacks",
        "attacks",
        "competitive_supremacy",
        "competitor_compare",
        "orchestrator",
    )
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for mod in mods:
            for bad in forbidden_substrings:
                assert bad not in mod, f"forbidden import {mod!r} (matched {bad!r})"

    # MODULE-LEVEL pii_anon imports may reference ONLY pii_anon.agentic.
    # (pii_anon_datasets is imported LAZILY inside the function — see below.)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("pii_anon") and not node.module.startswith(
                "pii_anon_datasets"
            ):
                assert node.module.startswith("pii_anon.agentic"), (
                    f"module-level import of disallowed pii_anon module "
                    f"{node.module!r}"
                )

    # pii_anon_datasets must NOT be a module-level import (it is OPTIONAL — it
    # must be imported lazily inside the function so the module loads without it).
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            assert not n.startswith("pii_anon_datasets"), (
                "pii_anon_datasets must be imported LAZILY inside the function, "
                "not at module level (the datasets extra is optional)"
            )


def test_ax001_a10_in_tree_payloads_are_inert_synthetic() -> None:
    """The in-tree fallback payloads encode SYNTHETIC values only (AX-001).

    With the datasets extra forced absent, the scorer runs the in-tree INERT set;
    the report is still well-formed and covers the >=3 transforms. No real PII is
    ever present (the payloads are recoverable encodings of synthetic values).
    """
    real_import = builtins.__import__

    def _no_datasets(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("pii_anon_datasets"):
            raise ModuleNotFoundError("No module named 'pii_anon_datasets'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    import builtins as _b

    _b.__import__ = _no_datasets  # type: ignore[assignment]
    try:
        report = score_injection_resistance(
            FourChannelGuard(masker=None, surrogate_key=FIXED_KEY), scope=SCOPE
        )
    finally:
        _b.__import__ = real_import
    assert report.n_payloads >= 3
    assert report.is_representative is True


# ==============================================================================
# Frozen-value-object contract (defense-in-depth on the dataclasses)
# ==============================================================================
def test_fr028_sankey_edge_is_frozen() -> None:
    """SankeyEdge is a frozen value object (cannot be mutated post-construction)."""
    edge = SankeyEdge(
        source=AgentChannel.PROMPT.value,
        target=BLOCKED,
        entity_type="EMAIL",
        count=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        edge.count = 99  # type: ignore[misc]


def test_fr028_leakage_sankey_is_frozen() -> None:
    """LeakageSankey is frozen and holds an immutable edge tuple."""
    sankey = LeakageSankey(edges=())
    assert sankey.edges == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sankey.edges = (1,)  # type: ignore[misc,assignment]


def test_fr029_injection_report_defaults_is_representative_true() -> None:
    """InjectionResistanceReport.is_representative defaults to True (Pass-2 note)."""
    report = InjectionResistanceReport(
        attack_success_rate=0.0,
        benign_task_success_rate=1.0,
        n_payloads=3,
        n_benign=3,
        intent_tags=("base64", "homoglyph", "zero_width"),
    )
    assert report.is_representative is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.attack_success_rate = 1.0  # type: ignore[misc]


# ==============================================================================
# REFACTOR-phase additive edge tests (no production change — close edge paths)
# ==============================================================================
class _RaisingMasker:
    """A masker that reports a record for the email span but LEAVES the raw text.

    The S6-02 ``_assert_no_raw_pii`` invariant derives the raw value from the
    reported span, sees it survive in ``masked_text``, and RAISES
    ``NoRawPIIPersistError`` — the guard refuses to emit a leak. The injection
    scorer must treat that refusal as RESISTANCE (the value did not cross the
    boundary), not as an exfiltration. Models a guard that detects but fails to
    substitute, where the upstream invariant is the backstop.
    """

    def mask(
        self, text: str, *, channel: AgentChannel, scope: str
    ) -> ChannelResult:
        idx = text.find(SYNTH_EMAIL)
        if idx < 0:
            return ChannelResult(channel=channel, masked_text=text, records=())
        rec = InterceptionRecord(
            channel=channel,
            entity_type="EMAIL",
            surrogate="<EMAIL:v1:tok_raising0000>",
            span_start=idx,
            span_end=idx + len(SYNTH_EMAIL),
            scope=scope,
        )
        # Reports a record but returns the raw text unchanged ⟹ invariant RAISES.
        return ChannelResult(channel=channel, masked_text=text, records=(rec,))


def test_fr029_edge_guard_raise_counts_as_resistance_not_exfiltration() -> None:
    """A guard that RAISES NoRawPIIPersistError ⟹ that payload is resistance.

    The raw value never crossed the boundary (the guard refused to emit it), so
    the attack_success_rate is NOT incremented for it: ASR stays 0.0.
    """
    guard = FourChannelGuard(masker=_RaisingMasker())
    report = score_injection_resistance(guard, scope=SCOPE)
    # Every direct/verbatim probe RAISES (resistance), nothing exfiltrates.
    assert report.attack_success_rate == 0.0
    assert report.n_payloads > 0


def test_fr028_edge_classify_survivor_phone_shape() -> None:
    """A surviving phone-shaped synthetic value is labelled PHONE (not the raw)."""
    ledger = _ledger_with(_rec(AgentChannel.TOOL_IO, entity_type="PHONE"))
    # A synthetic phone-shaped value survives the outbound payload.
    outbound = {AgentChannel.TOOL_IO: f"dialed {SYNTH_PHONE} just now"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[SYNTH_PHONE]
    )
    leak = next(e for e in sankey.edges if e.target == LEAKED)
    assert leak.entity_type == "PHONE"
    assert SYNTH_PHONE not in leak.entity_type  # never echoes the raw value


def test_fr028_edge_classify_survivor_generic_pii_shape() -> None:
    """A surviving value with no email/phone shape is labelled the generic PII."""
    ledger = _ledger_with(_rec(AgentChannel.MEMORY, entity_type="NAME"))
    survivor = "Jane Roe"  # synthetic name — neither '@' nor a phone shape
    outbound = {AgentChannel.MEMORY: f"remember the client {survivor} please"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=[survivor]
    )
    leak = next(e for e in sankey.edges if e.target == LEAKED)
    assert leak.entity_type == "PII"


def test_fr028_edge_as_dict_blocked_only_has_no_leaked_link() -> None:
    """A blocked-only Sankey renders a leaked node but no link into it."""
    ledger = _ledger_with(_rec(AgentChannel.PROMPT))
    payload = build_leakage_sankey(
        ledger, outbound={}, known_values=[]
    ).as_dict()
    # The leaked sink node still exists (stable 6-node graph) ...
    assert {n["id"] for n in payload["nodes"]} >= {BLOCKED, LEAKED}
    # ... but no link targets it (no survivor).
    assert all(link["target"] != LEAKED for link in payload["links"])


def test_fr028_edge_empty_known_value_string_is_not_a_survivor() -> None:
    """An empty-string known_value is skipped (it would match every payload)."""
    ledger = _ledger_with(_rec(AgentChannel.PROMPT))
    outbound = {AgentChannel.PROMPT: "any non-empty outbound payload"}
    sankey = build_leakage_sankey(
        ledger, outbound=outbound, known_values=["", SYNTH_EMAIL]
    )
    # The empty string is not treated as a (vacuous) survivor.
    assert sankey.leaked_count() == 0


def test_fr028_edge_empty_ledger_and_outbound_is_empty_graph() -> None:
    """No records + no outbound ⟹ zero edges, all-zero counts, stable 6 nodes."""
    sankey = build_leakage_sankey(
        InterceptionLedger(), outbound={}, known_values=[SYNTH_EMAIL]
    )
    assert sankey.edges == ()
    assert sankey.blocked_count() == 0
    assert sankey.leaked_count() == 0
    assert sankey.counts_by_channel() == {}
    payload = sankey.as_dict()
    assert len(payload["nodes"]) == 6  # the 6-node graph is always present
    assert payload["links"] == []
