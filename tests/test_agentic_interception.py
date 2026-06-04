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
import hashlib
import json
import re
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
    _keyed_token_id,
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

# A FIXED surrogate key so keyed-HMAC surrogates are reproducible in tests
# (the surrogate id is a keyed HMAC — security-S6-02-01 remediation — so
# determinism is per-key; an injected fixed key pins byte-identical output).
# A second key proves keyed surrogates DIFFER across keys (dictionary resistance).
FIXED_KEY = b"\x11" * 32
FIXED_KEY_2 = b"\x22" * 32

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
    """Same payloads + scope + KEY ⟹ byte-identical masked outputs + ledger.

    The surrogate is a KEYED HMAC (security-S6-02-01 remediation): determinism
    is *per fixed key*. We INJECT a fixed ``surrogate_key`` into both guards so
    the per-mask path is byte-identical and reproducible (the FR-030 path the
    canonical run uses). Key generation is the only non-deterministic step and it
    is skipped entirely when a key is supplied — so no RNG runs on the per-mask
    path here.
    """
    payloads = {
        AgentChannel.PROMPT: f"prompt {SYNTH_EMAIL} and {SYNTH_PHONE}",
        AgentChannel.TOOL_IO: f"tool {SYNTH_EMAIL}",
    }

    ledger_a = InterceptionLedger()
    out_a = FourChannelGuard(
        masker=None, surrogate_key=FIXED_KEY, ledger=ledger_a
    ).intercept_all(payloads, scope=SCOPE)

    ledger_b = InterceptionLedger()
    out_b = FourChannelGuard(
        masker=None, surrogate_key=FIXED_KEY, ledger=ledger_b
    ).intercept_all(payloads, scope=SCOPE)

    for channel in payloads:
        assert out_a[channel].masked_text == out_b[channel].masked_text
        assert out_a[channel].records == out_b[channel].records
    # Identical ledger (as_dict equal byte-for-byte under JSON).
    assert json.dumps(ledger_a.as_dict(), sort_keys=True) == json.dumps(
        ledger_b.as_dict(), sort_keys=True
    )


def test_ax002_a9_no_nondeterministic_imports_on_path() -> None:
    """The PER-MASK path is deterministic: no ``random``/``uuid``/``time`` import.

    Reconciliation with the keyed-surrogate remediation (security-S6-02-01):
    ``secrets`` / ``os`` are now ALLOWED because the masker mints a random
    per-instance ``surrogate_key`` at CONSTRUCTION (secure-by-default — the
    persisted G5 ledger is then non-dictionary-reversible). That key generation
    is a one-time construction step, never invoked on the per-mask path, so the
    per-mask path stays deterministic *given a fixed key*. ``random`` / ``uuid``
    / ``time`` remain banned outright — none of them appears in the module, and
    the keyed surrogate must never depend on wall-clock / RNG per call.

    The companion :func:`test_ax002_a9_deterministic_replay` (and the dedicated
    fixed-key determinism assertion below) prove the per-mask path is
    byte-identical when a fixed ``surrogate_key`` is injected.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "pii_anon" / "agentic"
    module_src = (src / "interception.py").read_text(encoding="utf-8")
    tree = ast.parse(module_src)
    # ``secrets``/``os`` deliberately EXCLUDED — construction-time key gen only.
    banned = {"random", "uuid", "time"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & banned), f"non-deterministic import(s): {imported & banned}"

    # ``secrets`` IS imported (for the construction-time key) but must NOT be
    # referenced anywhere inside ``_DefaultMasker.mask`` / ``_keyed_token_id`` /
    # ``_make_surrogate`` (the per-mask path). Assert the only ``secrets`` use is
    # in ``FourChannelGuard.__post_init__`` (construction).
    per_mask_funcs = {"mask", "_keyed_token_id", "_make_surrogate"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in per_mask_funcs:
            names = {
                n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            }
            assert "secrets" not in names, (
                f"{node.name} (per-mask path) must not call secrets — "
                "RNG belongs only in construction-time key generation"
            )

    # Determinism is per fixed key: inject one and assert byte-identical replay.
    text = f"reach {SYNTH_EMAIL}"
    out1 = FourChannelGuard(masker=None, surrogate_key=FIXED_KEY).intercept(
        text, channel=AgentChannel.PROMPT, scope=SCOPE
    )
    out2 = FourChannelGuard(masker=None, surrogate_key=FIXED_KEY).intercept(
        text, channel=AgentChannel.PROMPT, scope=SCOPE
    )
    assert out1.masked_text == out2.masked_text
    assert out1.records == out2.records


# ==============================================================================
# A11 — KEYED surrogate resists offline dictionary re-identification
#       [SECURITY] (FR-026 / AX-006 — security-S6-02-01 remediation)
# ==============================================================================
def test_fr026_a11_keyed_surrogate_resists_dictionary_reidentification() -> None:
    """A surrogate is NOT recoverable to its raw value without the key.

    The inverse of the reviewer's break-probe (which recovered the raw email
    from a KEYLESS hash + a candidate dictionary with NO key). With the surrogate
    now a KEYED HMAC:

    1. Masking the SAME value under two DIFFERENT keys yields DIFFERENT
       surrogates (the surrogate is key-dependent, not a public hash of the raw).
    2. An attacker holding the persisted (surrogate-only) G5 ledger + a candidate
       dictionary that CONTAINS the true raw value, but WITHOUT the key, cannot
       reconstruct the surrogate for any candidate — so no candidate matches the
       real surrogate. (We reproduce the attacker's offline keyless guess exactly
       and assert it never lands.)
    """
    text = f"email {SYNTH_EMAIL} now"

    # (1) Same raw value, two different keys -> different surrogates.
    ledger_a = InterceptionLedger()
    rec_a = (
        FourChannelGuard(masker=None, surrogate_key=FIXED_KEY, ledger=ledger_a)
        .intercept(text, channel=AgentChannel.MEMORY, scope=SCOPE)
        .records[0]
    )
    rec_b = (
        FourChannelGuard(masker=None, surrogate_key=FIXED_KEY_2)
        .intercept(text, channel=AgentChannel.MEMORY, scope=SCOPE)
        .records[0]
    )
    assert rec_a.surrogate != rec_b.surrogate, (
        "two different keys must produce different surrogates for the same raw "
        "value — otherwise the surrogate is a keyless (guessable) hash"
    )

    # (2) Offline dictionary attack WITHOUT the key fails. The attacker has the
    #     published G5 ledger (surrogate-only) + a candidate dictionary that even
    #     CONTAINS the true synthetic value + knows the (low-cardinality)
    #     scope/entity + the open-source surrogate construction — but NOT the key.
    target_surrogate = ledger_a.as_dict()["records"][0]["surrogate"]  # type: ignore[index]
    candidate_dictionary = [
        SYNTH_EMAIL,  # the true value is IN the dictionary
        "john.doe@example.test",
        "alice@example.test",
        "555-0100",
    ]

    # The attacker's best offline move: recompute the surrogate for each
    # candidate using a KEYLESS digest (they have no key). Reproduce the OLD
    # keyless construction exactly — it must NOT reproduce the keyed surrogate.
    def _keyless_guess(entity: str, raw: str) -> str:
        keyless_id = hashlib.blake2b(
            f"{SCOPE}\x1f{entity}\x1f{raw}".encode(), digest_size=8
        ).hexdigest()[:12]
        return f"<{entity}:v1:tok_{keyless_id}>"

    keyless_matches = [
        c for c in candidate_dictionary if _keyless_guess("EMAIL", c) == target_surrogate
    ]
    assert keyless_matches == [], (
        "a keyless dictionary guess must NOT recover the keyed surrogate — "
        "the surrogate is no longer offline-reversible without the key"
    )

    # Even brute-forcing with a WRONG key (the attacker guesses a key) misses;
    # only the exact key reproduces the surrogate.
    wrong_key_id = _keyed_token_id(FIXED_KEY_2, SCOPE, "EMAIL", SYNTH_EMAIL)
    assert f"<EMAIL:v1:tok_{wrong_key_id}>" != target_surrogate
    right_key_id = _keyed_token_id(FIXED_KEY, SCOPE, "EMAIL", SYNTH_EMAIL)
    assert f"<EMAIL:v1:tok_{right_key_id}>" == target_surrogate  # only WITH the key


def test_fr026_a11_default_key_is_random_per_instance_secure_by_default() -> None:
    """No injected key ⟹ a fresh random per-instance key (secure-by-default).

    Two default-constructed guards (no ``surrogate_key``) mask the same value but
    produce DIFFERENT surrogates, because each mints its own random key at
    construction — so the persisted G5 ledger from the keyless-default era is
    replaced by a non-dictionary-reversible one even when the operator injects
    nothing. (Determinism-when-needed is opt-in via a fixed key, proven by A9.)
    """
    text = f"reach {SYNTH_EMAIL}"
    s1 = FourChannelGuard(masker=None).intercept(
        text, channel=AgentChannel.PROMPT, scope=SCOPE
    ).records[0].surrogate
    s2 = FourChannelGuard(masker=None).intercept(
        text, channel=AgentChannel.PROMPT, scope=SCOPE
    ).records[0].surrogate
    assert s1 != s2, (
        "default (no-key) guards must each use a fresh random key — identical "
        "surrogates across instances would mean a fixed/keyless construction"
    )
    # Both are still well-formed canonical surrogates (and never the raw value).
    assert SYNTH_EMAIL not in s1 and SYNTH_EMAIL not in s2
    assert re.match(EMAIL_SURROGATE_RE, s1) and re.match(EMAIL_SURROGATE_RE, s2)


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


# ==============================================================================
# REFACTOR-phase additive edge tests (no production change — close edge paths)
# ==============================================================================
def test_ax001_edge_lazy_reexport_resolves_public_names() -> None:
    """The package's PEP 562 lazy ``__getattr__`` re-exports resolve correctly."""
    import pii_anon.agentic as agentic

    # Each public name resolves lazily to the same object as the submodule's.
    from pii_anon.agentic import interception as impl

    for name in (
        "AgentChannel",
        "InterceptionRecord",
        "ChannelResult",
        "ChannelMasker",
        "FourChannelGuard",
        "InterceptionLedger",
        "NoRawPIIPersistError",
    ):
        assert getattr(agentic, name) is getattr(impl, name)
    # __dir__ advertises exactly the public surface.
    assert set(dir(agentic)) >= {"FourChannelGuard", "AgentChannel"}
    # An unknown attribute still raises AttributeError (no silent None).
    with pytest.raises(AttributeError):
        _ = agentic.does_not_exist  # type: ignore[attr-defined]


def test_fr025_edge_default_masker_dedupes_overlapping_detectors() -> None:
    """Overlapping detector matches are claimed once (no double-mask)."""
    guard = FourChannelGuard(masker=None)
    # An email whose local-part visually contains a phone-like run: the email
    # detector claims the span first; the looser phone detector must skip it.
    text = "ping 555-0100-user@example.test please"
    result = guard.intercept(text, channel=AgentChannel.PROMPT, scope=SCOPE)
    # Exactly one record (the email), not an email + an overlapping phone.
    assert len(result.records) == 1
    assert result.records[0].entity_type == "EMAIL"


def test_fr026_edge_known_values_skips_out_of_bounds_span() -> None:
    """A record with an out-of-bounds span is ignored by the raw-value scan.

    Defensive: a misbehaving masker that reports ``span_start=-1`` (no real
    source span) must not crash ``_known_raw_values`` — the guard simply derives
    no raw value from that record and the masked text (no raw survival) passes.
    """

    class _BadSpanMasker:
        def mask(
            self, text: str, *, channel: AgentChannel, scope: str
        ) -> ChannelResult:
            rec = InterceptionRecord(
                channel=channel,
                entity_type="EMAIL",
                surrogate="<EMAIL:v1:tok_badspan0>",
                span_start=-1,  # out of bounds
                span_end=9999,
                scope=scope,
            )
            # Masked text carries NO raw value, so the invariant must pass.
            return ChannelResult(
                channel=channel,
                masked_text="<EMAIL:v1:tok_badspan0>",
                records=(rec,),
            )

    guard = FourChannelGuard(masker=_BadSpanMasker())
    result = guard.intercept("anything", channel=AgentChannel.PROMPT, scope=SCOPE)
    assert result.masked_text == "<EMAIL:v1:tok_badspan0>"


def test_fr026_edge_persist_noop_without_store() -> None:
    """A reversible channel with no injected store persists nothing (no crash)."""
    guard = FourChannelGuard(
        masker=None,
        token_store=None,
        reversible_channels=frozenset({AgentChannel.MEMORY}),
    )
    # MEMORY is reversible but there is no store — the persist step is a no-op.
    result = guard.intercept(
        f"remember {SYNTH_EMAIL}", channel=AgentChannel.MEMORY, scope=SCOPE
    )
    assert SYNTH_EMAIL not in result.masked_text
    assert len(result.records) == 1


def test_fr025_edge_empty_text_yields_no_records() -> None:
    """Empty / PII-free text produces a clean pass-through with no records."""
    guard = FourChannelGuard(masker=None)
    result = guard.intercept("", channel=AgentChannel.TRACE, scope=SCOPE)
    assert result.masked_text == ""
    assert result.records == ()
    clean = guard.intercept(
        "no pii here at all", channel=AgentChannel.TOOL_IO, scope=SCOPE
    )
    assert clean.records == ()
    assert clean.masked_text == "no pii here at all"
