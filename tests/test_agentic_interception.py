"""S6-02 — 4-channel least-privilege interception + no-raw-PII-persist (DC-13).

A NEW ``pii_anon.agentic`` package intercepts PII on all four agent channels
(prompt / memory / tool-I/O / trace) under **least-privilege** (AX-006) and
GUARANTEES no raw PII is persisted to any channel after masking (FR-026). The
per-channel ``InterceptionLedger`` it emits is a direct **G5 audit input** + the
source the S6-05 leakage-Sankey consumes.

These tests run for REAL in-tree against SYNTHETIC text (AX-001) + the real
S6-03 ``EncryptedSQLiteTokenStore`` AEAD vault. A live agent runtime + real
transcript-residual leakage are the Pass-2 / AGENT_SIMULATED honesty boundary.

Test fixtures use ONLY synthetic PII (AX-001): ``"jane.roe@example.test"``,
``"555-0100"`` — never real values. The headline (A4) is that even a masker
that *forgets* one of these synthetic values is caught by ``_assert_no_raw_pii``
and the guard RAISES (loud), never silently persists raw PII.

Acceptance map (story §3):
    A1  exactly four channels                       [CONTRACT]
    A2  mask PII in the prompt channel              [UNIT]
    A3  record carries no plaintext                 [CONTRACT][SECURITY]
    A4  no-raw-PII-persist raises on leak           [SECURITY]
    A5  least-privilege: TRACE never reverses       [SECURITY][AUDIT]
    A6  memory channel persists only a surrogate    [UNIT]
    A7  intercept_all covers every channel          [UNIT]
    A8  ledger is surrogate-only + feeds G5 shape   [AUDIT]
    A9  deterministic                               [PROPERTY]
    A10 import isolation + no-plaintext audit       [AUDIT]
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pii_anon.agentic.interception import (
    AgentChannel,
    ChannelMasker,
    ChannelResult,
    FourChannelGuard,
    InterceptionLedger,
    InterceptionRecord,
    NoRawPIIPersistError,
)
from pii_anon.errors import PiiAnonError
from pii_anon.tokenization.encrypted_store import (
    EncryptedSQLiteTokenStore,
    StaticTestKeyProvider,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# --- synthetic fixtures (AX-001 — never real PII) ------------------------------
SYNTH_EMAIL = "jane.roe@example.test"
SYNTH_PHONE = "555-0100"
SCOPE = "case_s6_02"

# A surrogate using the canonical token format reused from
# tokenization/reidentification.py: ``<ENTITY:vN:tok_XXX>``.
EMAIL_SURROGATE_RE = r"<EMAIL:v\d+:tok_[A-Za-z0-9_-]+>"


# --- helpers / fixtures --------------------------------------------------------
@pytest.fixture()
def store() -> Iterator[EncryptedSQLiteTokenStore]:
    """A real S6-03 AEAD-encrypted token store (in-memory, synthetic KEK)."""
    provider = StaticTestKeyProvider(b"0123456789abcdef0123456789abcdef", key_id="s6-02-kek")
    s = EncryptedSQLiteTokenStore(":memory:", key_provider=provider)
    try:
        yield s
    finally:
        s.close()


class _LeakyMasker:
    """A deliberately-incomplete masker that FORGETS to mask one entity.

    Used by A4: it claims (via its returned record) to have masked an email but
    leaves the raw value in ``masked_text`` — the ``_assert_no_raw_pii`` invariant
    must catch the survival and RAISE, never silently persist the leak.
    """

    def __init__(self, raw_value: str) -> None:
        self._raw = raw_value

    def mask(
        self, text: str, *, channel: AgentChannel, scope: str
    ) -> ChannelResult:
        # The masker LIES: it emits a record for a surrogate but returns the raw
        # text unchanged (the entity survives un-masked).
        record = InterceptionRecord(
            channel=channel,
            entity_type="EMAIL",
            surrogate="<EMAIL:v1:tok_leaky00>",
            span_start=text.find(self._raw),
            span_end=text.find(self._raw) + len(self._raw),
            scope=scope,
        )
        return ChannelResult(channel=channel, masked_text=text, records=(record,))


# ==============================================================================
# A1 — exactly four channels [CONTRACT]
# ==============================================================================
def test_fr025_a1_exactly_four_channels() -> None:
    """``set(AgentChannel) == {PROMPT, MEMORY, TOOL_IO, TRACE}`` — pinned."""
    assert {c.value for c in AgentChannel} == {"PROMPT", "MEMORY", "TOOL_IO", "TRACE"}
    assert set(AgentChannel) == {
        AgentChannel.PROMPT,
        AgentChannel.MEMORY,
        AgentChannel.TOOL_IO,
        AgentChannel.TRACE,
    }
    # Exactly four — adding/removing a channel fails this test.
    assert len(AgentChannel) == 4
    # str-Enum: the value IS the channel name (stable serialization).
    assert AgentChannel.PROMPT == "PROMPT"
    assert isinstance(AgentChannel.TRACE, str)


# ==============================================================================
# A2 — mask PII in the prompt channel [UNIT]
# ==============================================================================
def test_fr025_a2_mask_prompt_channel() -> None:
    """Prompt-channel intercept returns a surrogate (not raw) + a record."""
    guard = FourChannelGuard(masker=None)  # default in-tree masker
    text = f"Please email {SYNTH_EMAIL} about the matter."
    result = guard.intercept(text, channel=AgentChannel.PROMPT, scope=SCOPE)

    assert isinstance(result, ChannelResult)
    assert result.channel is AgentChannel.PROMPT
    assert SYNTH_EMAIL not in result.masked_text  # raw email gone
    import re

    assert re.search(EMAIL_SURROGATE_RE, result.masked_text)  # surrogate present
    assert len(result.records) == 1
    rec = result.records[0]
    assert rec.channel is AgentChannel.PROMPT
    assert rec.entity_type == "EMAIL"
    assert rec.surrogate in result.masked_text
    # The recorded span points at where the raw value WAS in the source text.
    assert text[rec.span_start : rec.span_end] == SYNTH_EMAIL


# ==============================================================================
# A3 — record carries no plaintext [CONTRACT][SECURITY]
# ==============================================================================
def test_fr026_a3_record_has_no_plaintext_field() -> None:
    """Introspect ``InterceptionRecord`` fields + values — no raw PII anywhere."""
    guard = FourChannelGuard(masker=None)
    text = f"Reach {SYNTH_EMAIL} now."
    rec = guard.intercept(text, channel=AgentChannel.PROMPT, scope=SCOPE).records[0]

    field_names = {f.name for f in dataclasses.fields(rec)}
    # No field is named for raw content.
    assert "plaintext" not in field_names
    assert "raw" not in field_names
    assert "value" not in field_names
    assert "original" not in field_names
    # By construction the allowed fields are surrogate-only metadata.
    assert field_names == {
        "channel",
        "entity_type",
        "surrogate",
        "span_start",
        "span_end",
        "scope",
    }
    # No field VALUE equals (or contains) the raw PII.
    for f in dataclasses.fields(rec):
        val = getattr(rec, f.name)
        assert SYNTH_EMAIL not in str(val)
    # The record is frozen (cannot be mutated to smuggle plaintext later).
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.surrogate = SYNTH_EMAIL  # type: ignore[misc]


# ==============================================================================
# A4 — no-raw-PII-persist raises on leak [SECURITY] (FR-026 headline)
# ==============================================================================
def test_fr026_a4_no_raw_persist_raises_on_masked_text_leak() -> None:
    """A masker that leaves a raw value in ``masked_text`` ⟹ guard RAISES."""
    guard = FourChannelGuard(masker=_LeakyMasker(SYNTH_EMAIL))
    text = f"Contact {SYNTH_EMAIL} please."
    with pytest.raises(NoRawPIIPersistError):
        guard.intercept(text, channel=AgentChannel.PROMPT, scope=SCOPE)


def test_fr026_a4_no_raw_persist_raises_via_intercept_all() -> None:
    """The leak invariant also fires through the ``intercept_all`` path."""
    guard = FourChannelGuard(masker=_LeakyMasker(SYNTH_EMAIL))
    with pytest.raises(NoRawPIIPersistError):
        guard.intercept_all(
            {AgentChannel.MEMORY: f"note: {SYNTH_EMAIL}"}, scope=SCOPE
        )


def test_fr026_a4_error_is_pii_anon_error_subclass() -> None:
    """``NoRawPIIPersistError`` is a ``PiiAnonError`` (catchable by base)."""
    assert issubclass(NoRawPIIPersistError, PiiAnonError)


# ==============================================================================
# A5 — least-privilege: TRACE never reverses [SECURITY][AUDIT] (AX-006)
# ==============================================================================
def test_ax006_a5_trace_never_reverses_by_default(
    store: EncryptedSQLiteTokenStore,
) -> None:
    """TRACE persists NOTHING to the store; only MEMORY (reversible) persists."""
    guard = FourChannelGuard(
        masker=None,
        token_store=store,
        reversible_channels=frozenset({AgentChannel.MEMORY}),
    )
    assert store.count() == 0

    # TRACE channel — non-reversible: writes NOTHING to the store.
    guard.intercept(
        f"trace log emitted for {SYNTH_EMAIL}",
        channel=AgentChannel.TRACE,
        scope=SCOPE,
    )
    assert store.count() == 0, "TRACE (non-reversible) must not touch the store"

    # MEMORY channel — reversible: persists exactly one surrogate->token row.
    guard.intercept(
        f"remember {SYNTH_EMAIL}", channel=AgentChannel.MEMORY, scope=SCOPE
    )
    assert store.count() == 1, "MEMORY (reversible) persists one mapping"


def test_ax006_a5_trace_reversible_only_on_explicit_opt_in(
    store: EncryptedSQLiteTokenStore,
) -> None:
    """TRACE is reversible ONLY when explicitly placed in reversible_channels."""
    guard = FourChannelGuard(
        masker=None,
        token_store=store,
        reversible_channels=frozenset({AgentChannel.TRACE}),  # explicit opt-in
    )
    guard.intercept(
        f"trace {SYNTH_EMAIL}", channel=AgentChannel.TRACE, scope=SCOPE
    )
    assert store.count() == 1, "explicitly-reversible TRACE persists"


# ==============================================================================
# A6 — memory channel persists only a surrogate [UNIT]
# ==============================================================================
def test_fr026_a6_memory_persists_only_surrogate(
    store: EncryptedSQLiteTokenStore,
) -> None:
    """A reversible MEMORY channel writes a surrogate->token row; reading back
    the stored mapping yields the surrogate (ciphertext at rest), never raw."""
    guard = FourChannelGuard(
        masker=None,
        token_store=store,
        reversible_channels=frozenset({AgentChannel.MEMORY}),
    )
    text = f"scratchpad: {SYNTH_EMAIL}"
    result = guard.intercept(text, channel=AgentChannel.MEMORY, scope=SCOPE)

    surrogate = result.records[0].surrogate
    assert store.count(scope=SCOPE) == 1
    mapping = store.get(surrogate, scope=SCOPE)
    assert mapping is not None
    # The stored token IS the surrogate (the on-disk payload is AEAD ciphertext).
    assert mapping.token == surrogate
    assert mapping.entity_type == "EMAIL"


# ==============================================================================
# A7 — intercept_all covers every channel independently [UNIT]
# ==============================================================================
def test_fr025_a7_intercept_all_covers_every_channel() -> None:
    """One masked ``ChannelResult`` per channel, each with per-channel records."""
    guard = FourChannelGuard(masker=None)
    payloads = {
        AgentChannel.PROMPT: f"prompt {SYNTH_EMAIL}",
        AgentChannel.MEMORY: f"memory {SYNTH_EMAIL}",
        AgentChannel.TOOL_IO: f"tool {SYNTH_EMAIL}",
        AgentChannel.TRACE: f"trace {SYNTH_EMAIL}",
    }
    results = guard.intercept_all(payloads, scope=SCOPE)

    assert set(results) == set(AgentChannel)
    for channel, result in results.items():
        assert result.channel is channel
        assert SYNTH_EMAIL not in result.masked_text
        assert len(result.records) == 1
        assert result.records[0].channel is channel


# ==============================================================================
# A8 — ledger is surrogate-only + feeds the G5 shape [AUDIT]
# ==============================================================================
def test_fr026_a8_ledger_surrogate_only_g5_shape() -> None:
    """``InterceptionLedger.as_dict()`` carries only surrogates + per-channel
    counts; the shape is a consumable G5 audit artifact."""
    ledger = InterceptionLedger()
    guard = FourChannelGuard(masker=None, ledger=ledger)
    guard.intercept_all(
        {
            AgentChannel.PROMPT: f"p {SYNTH_EMAIL}",
            AgentChannel.MEMORY: f"m {SYNTH_PHONE}",
        },
        scope=SCOPE,
    )

    counts = ledger.counts_by_channel()
    assert counts[AgentChannel.PROMPT] == 1
    assert counts[AgentChannel.MEMORY] == 1

    records = ledger.records()
    assert len(records) == 2
    assert all(isinstance(r, InterceptionRecord) for r in records)

    payload = ledger.as_dict()
    # Round-trips through JSON (audit artifact must be serializable).
    blob = json.dumps(payload)
    # No raw PII anywhere in the serialized G5 artifact.
    assert SYNTH_EMAIL not in blob
    assert SYNTH_PHONE not in blob
    # Counts are part of the audit shape.
    assert payload["counts_by_channel"]["PROMPT"] == 1
    assert payload["counts_by_channel"]["MEMORY"] == 1
    assert len(payload["records"]) == 2


def test_fr026_a8_ledger_copy_on_read_is_immutable() -> None:
    """Mutating the returned records tuple/list does not corrupt the ledger."""
    ledger = InterceptionLedger()
    guard = FourChannelGuard(masker=None, ledger=ledger)
    guard.intercept(f"x {SYNTH_EMAIL}", channel=AgentChannel.PROMPT, scope=SCOPE)
    snapshot = ledger.records()
    assert isinstance(snapshot, tuple)  # copy-on-read snapshot
    # A second read after another interception reflects the append.
    guard.intercept(f"y {SYNTH_EMAIL}", channel=AgentChannel.MEMORY, scope=SCOPE)
    assert len(ledger.records()) == len(snapshot) + 1


# ==============================================================================
# A9 — deterministic [PROPERTY] (AX-002)
# ==============================================================================
def test_ax002_a9_deterministic_replay() -> None:
    """Same payloads + scope ⟹ byte-identical masked outputs + identical ledger."""
    payloads = {
        AgentChannel.PROMPT: f"prompt {SYNTH_EMAIL} and {SYNTH_PHONE}",
        AgentChannel.TOOL_IO: f"tool {SYNTH_EMAIL}",
    }

    ledger_a = InterceptionLedger()
    out_a = FourChannelGuard(masker=None, ledger=ledger_a).intercept_all(
        payloads, scope=SCOPE
    )

    ledger_b = InterceptionLedger()
    out_b = FourChannelGuard(masker=None, ledger=ledger_b).intercept_all(
        payloads, scope=SCOPE
    )

    for channel in payloads:
        assert out_a[channel].masked_text == out_b[channel].masked_text
        assert out_a[channel].records == out_b[channel].records
    # Identical ledger (as_dict equal byte-for-byte under JSON).
    assert json.dumps(ledger_a.as_dict(), sort_keys=True) == json.dumps(
        ledger_b.as_dict(), sort_keys=True
    )


def test_ax002_a9_no_nondeterministic_imports_on_path() -> None:
    """The interception module imports no ``random``/``uuid``/``time``/``secrets``."""
    src = Path(__file__).resolve().parents[1] / "src" / "pii_anon" / "agentic"
    module_src = (src / "interception.py").read_text(encoding="utf-8")
    tree = ast.parse(module_src)
    banned = {"random", "uuid", "time", "secrets"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), f"non-deterministic import(s): {imported & banned}"


# ==============================================================================
# A10 — import isolation + no-plaintext audit [AUDIT]
# ==============================================================================
def test_ax001_a10_import_isolation() -> None:
    """AST scan: ``agentic/interception.py`` imports nothing from
    ``eval_framework.rating`` / ``attacks`` / the SDO gate / ``orchestrator``."""
    src = Path(__file__).resolve().parents[1] / "src" / "pii_anon" / "agentic"
    module_src = (src / "interception.py").read_text(encoding="utf-8")
    tree = ast.parse(module_src)

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
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for mod in mods:
            for bad in forbidden_substrings:
                assert bad not in mod, f"forbidden import {mod!r} (matched {bad!r})"

    # agentic may import ONLY stdlib + pii_anon.tokenization + pii_anon.errors.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("pii_anon"):
                assert node.module.startswith(
                    ("pii_anon.tokenization", "pii_anon.errors")
                ), f"agentic imported disallowed pii_anon module {node.module!r}"


def test_ax001_a10_no_raw_value_field_audit() -> None:
    """The ledger/record types expose no raw-value field (defense-in-depth)."""
    rec_fields = {f.name for f in dataclasses.fields(InterceptionRecord)}
    res_fields = {f.name for f in dataclasses.fields(ChannelResult)}
    forbidden = {"plaintext", "raw", "value", "original", "raw_value"}
    assert not (rec_fields & forbidden)
    assert not (res_fields & forbidden)


def test_ax001_a10_channel_masker_is_runtime_checkable() -> None:
    """``ChannelMasker`` is a runtime-checkable Protocol the default satisfies."""
    guard = FourChannelGuard(masker=None)
    assert isinstance(guard.masker, ChannelMasker)
    # The leaky test double also structurally satisfies the protocol.
    assert isinstance(_LeakyMasker(SYNTH_EMAIL), ChannelMasker)
