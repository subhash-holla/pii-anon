"""4-channel least-privilege agent interception + no-raw-PII-persist (S6-02 / DC-13).

An agent pipeline leaks PII through four distinct channels — the **prompt** it
sends an LLM, its **memory** / scratchpad, its **tool I/O** (function args +
results), and its **trace** / telemetry. This module intercepts all four under
**least-privilege** (AX-006): each channel is masked, and only channels
EXPLICITLY declared reversible may persist a surrogate -> raw mapping (via an
injected token store — the S6-03 :class:`EncryptedSQLiteTokenStore` so any raw
at rest is AEAD-encrypted, defense-in-depth); the others — especially
**trace** — can never reverse.

The load-bearing guarantee (FR-026 / AX-006) is **no raw PII is persisted after
masking**: :meth:`FourChannelGuard._assert_no_raw_pii` scans the masked output
+ every record produced and RAISES :class:`NoRawPIIPersistError` if any raw PII
value survives. The per-channel :class:`InterceptionLedger` records only
surrogates (never plaintext) and is the auditable **G5** artifact (+ the S6-05
leakage-Sankey source).

Determinism (AX-002) + keyed-surrogate de-identification
--------------------------------------------------------
The per-mask path is **deterministic** — there is no ``random`` / ``uuid`` /
``time`` on it: the default in-tree masker derives each surrogate token via a
**keyed** ``HMAC-SHA256`` of ``(scope, entity_type, raw_value)`` under a
per-instance ``surrogate_key`` (mirroring the canonical keyed
:class:`~pii_anon.tokenization.providers.DeterministicHMACTokenizer` /
``encrypted_store.py``'s blind-index HMAC — no new crypto is rolled). Given a
**fixed** key, the same input always yields the same masked output and the same
ledger (the FR-030 "byte-identical given seed/key/scope" reproducible path).

The key is **keyed, not keyless**, on purpose: a keyless hash of low-entropy PII
(email / phone / name) is offline dictionary/rainbow-reversible — an attacker
holding the persisted surrogate-only :class:`InterceptionLedger` (a G5 audit
artifact + the S6-05 Sankey source) plus a candidate dictionary and the
low-cardinality scope/entity could brute-force-match a surrogate back to its raw
value with NO secret. Keying the HMAC removes that offline guessability while
preserving within-key determinism (closes the keyless-hash re-identification
finding).

``secrets`` is imported and used **only at CONSTRUCTION** to mint a random
per-instance key when none is injected (secure-by-default: the ledger is then
NON-dictionary-reversible). It is never called on the per-mask path, so
determinism-under-a-fixed-key is unaffected. ``random`` / ``uuid`` / ``time`` do
not appear anywhere in this module.

Isolation (AX-001 / RISK-5)
---------------------------
This module imports ONLY the standard library plus ``pii_anon.tokenization``
and ``pii_anon.errors``. It imports nothing from ``eval_framework.rating`` /
``attacks`` / the SDO gate / ``orchestrator`` (the protected user-WIP). All
fixtures exercising it use synthetic PII only.

AGENT_SIMULATED honesty boundary
--------------------------------
The 4-channel interception, the no-raw-PII-persist invariant, the
least-privilege reversibility policy, the surrogate-only ledger, and
determinism all run for REAL in-tree against SYNTHETIC text + the real S6-03
AEAD store. A live agent runtime + real transcript-residual leakage are Pass-2.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pii_anon.errors import PiiAnonError

if TYPE_CHECKING:
    from pii_anon.tokenization.store import TokenStore

__all__ = [
    "AgentChannel",
    "InterceptionRecord",
    "ChannelResult",
    "ChannelMasker",
    "FourChannelGuard",
    "InterceptionLedger",
    "NoRawPIIPersistError",
]


class NoRawPIIPersistError(PiiAnonError):
    """Raised when the post-mask invariant detects a raw PII value surviving.

    The survival may be in the masked output *or* in a record about to be
    persisted/audited (FR-026 / AX-006). This error is **loud by design**: the
    guard never silently passes a leak through. Kept in this module (not
    centralized in :mod:`pii_anon.errors`) so the agentic surface stays
    file-disjoint from unrelated work-streams.
    """


class AgentChannel(str, Enum):
    """The four agent channels that can leak PII.

    EXACTLY four — a contract test (A1) pins the membership; adding or removing
    a channel fails it. A :class:`str`-:class:`~enum.Enum` so a channel
    serializes as its own stable name (e.g. for the G5 ledger ``as_dict``).
    """

    PROMPT = "PROMPT"
    MEMORY = "MEMORY"
    TOOL_IO = "TOOL_IO"
    TRACE = "TRACE"


@dataclass(frozen=True, slots=True)
class InterceptionRecord:
    """A single surrogate substitution on one channel.

    **No plaintext field by construction** (FR-026): the record cannot carry a
    raw PII value even by accident — it holds only the channel, the entity type,
    the surrogate token, the span the raw value *occupied* in the source text,
    and the tokenization scope. A contract test (A3) asserts no field holds the
    raw value.
    """

    channel: AgentChannel
    entity_type: str
    surrogate: str
    span_start: int
    span_end: int
    scope: str

    def as_dict(self) -> dict[str, object]:
        """Serialize to a surrogate-only audit dict (the G5 record shape)."""
        return {
            "channel": self.channel.value,
            "entity_type": self.entity_type,
            "surrogate": self.surrogate,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """The masked result for one channel: masked text + its surrogate records."""

    channel: AgentChannel
    masked_text: str
    records: tuple[InterceptionRecord, ...]


@runtime_checkable
class ChannelMasker(Protocol):
    """Detect + replace PII on one channel, returning a :class:`ChannelResult`.

    Implementations MUST be deterministic (AX-002): the same ``(text, channel,
    scope)`` yields the same masked text + records. The default in-tree
    :class:`_DefaultMasker` reuses the canonical surrogate-token format
    (``<ENTITY:vN:tok_XXX>``) from :mod:`pii_anon.tokenization.reidentification`
    and needs no detection-engine dependency (CI-fast, deterministic).
    """

    def mask(
        self, text: str, *, channel: AgentChannel, scope: str
    ) -> ChannelResult:
        """Return a masked :class:`ChannelResult` for ``text`` on ``channel``."""
        ...


# Canonical surrogate-token format reused from
# ``tokenization/reidentification.py`` (``<ENTITY:vN:tok_XXX>``). The default
# masker EMITS tokens in exactly this shape so the ledger/store stay
# interoperable with the re-identification service.
_SURROGATE_VERSION = 1
_TOKEN_ID_LEN = 12  # hex chars of the keyed digest used as the token id
_SURROGATE_KEY_BYTES = 32  # 256-bit per-instance surrogate HMAC key


# Synthetic-PII detection patterns for the dependency-free default masker. These
# match the synthetic fixtures (emails + simple phone numbers) used in-tree; a
# production deployment injects a real engine-backed :class:`ChannelMasker`.
# Order matters: more specific (email) is applied before looser (phone).
_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ),
    (
        "PHONE",
        re.compile(r"\b\d{3}[-.\s]\d{4}\b"),
    ),
)


def _keyed_token_id(
    surrogate_key: bytes, scope: str, entity_type: str, raw_value: str
) -> str:
    """Derive a stable, **keyed** token id from ``(scope, entity_type, raw)``.

    A keyed :func:`hmac.new` over SHA-256 — the same keyed-pseudonym primitive
    the canonical :class:`~pii_anon.tokenization.providers.DeterministicHMACTokenizer`
    and ``encrypted_store.py``'s blind index use (no new crypto). Two properties
    matter here:

    * **Within a fixed key it is deterministic** (no ``random`` / ``uuid`` /
      ``time`` on this path), so the same raw value in the same scope always
      produces the same surrogate — the FR-030 / AX-002 reproducible path.
    * **Without the key it is NOT offline-reversible.** A keyless hash of
      low-entropy PII is dictionary/rainbow-reversible from the surrogate alone;
      keying the HMAC defeats that — an attacker holding the surrogate-only
      ledger plus a candidate dictionary but WITHOUT ``surrogate_key`` cannot
      brute-force a surrogate back to its raw value (closes the keyless-hash
      re-identification finding). Only the hex *image* of the keyed digest is
      ever emitted.
    """
    msg = f"{scope}\x1f{entity_type}\x1f{raw_value}".encode("utf-8")
    return hmac.new(surrogate_key, msg, hashlib.sha256).hexdigest()[:_TOKEN_ID_LEN]


def _make_surrogate(
    surrogate_key: bytes, scope: str, entity_type: str, raw_value: str
) -> str:
    """Build a canonical ``<ENTITY:vN:tok_XXX>`` surrogate for a raw value."""
    token_id = _keyed_token_id(surrogate_key, scope, entity_type, raw_value)
    return f"<{entity_type}:v{_SURROGATE_VERSION}:tok_{token_id}>"


class _DefaultMasker:
    """Dependency-free, **keyed**-surrogate default :class:`ChannelMasker`.

    Detects synthetic email / phone PII via small regexes and replaces each
    match with a canonical surrogate token whose id is a KEYED HMAC of the raw
    value under this masker's ``surrogate_key`` (see :func:`_keyed_token_id`).
    Keying makes the emitted surrogates — which flow into the persisted G5
    :class:`InterceptionLedger` — non-dictionary-reversible without the key,
    while staying deterministic given a fixed key. Has no detection-engine
    dependency, so it stays CI-fast and deterministic. A production deployment
    injects a richer engine-backed masker implementing the same
    :class:`ChannelMasker` protocol.

    Parameters
    ----------
    surrogate_key:
        The secret key for the surrogate HMAC. Determinism is per-key: the same
        ``(key, scope, entity, raw)`` always yields the same surrogate.
    """

    __slots__ = ("_surrogate_key",)

    def __init__(self, surrogate_key: bytes) -> None:
        self._surrogate_key = surrogate_key

    def mask(
        self, text: str, *, channel: AgentChannel, scope: str
    ) -> ChannelResult:
        records: list[InterceptionRecord] = []
        # Collect all matches across detectors, then apply them left-to-right so
        # spans (recorded against the ORIGINAL text) stay correct regardless of
        # surrogate length, and overlapping detectors do not double-mask.
        spans: list[tuple[int, int, str, str]] = []  # (start, end, entity, raw)
        claimed: list[tuple[int, int]] = []
        for entity_type, pattern in _DETECTORS:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if any(s < end and start < e for s, e in claimed):
                    continue  # already covered by a more-specific detector
                claimed.append((start, end))
                spans.append((start, end, entity_type, m.group(0)))

        spans.sort(key=lambda s: s[0])
        out_parts: list[str] = []
        cursor = 0
        for start, end, entity_type, raw in spans:
            surrogate = _make_surrogate(
                self._surrogate_key, scope, entity_type, raw
            )
            out_parts.append(text[cursor:start])
            out_parts.append(surrogate)
            cursor = end
            records.append(
                InterceptionRecord(
                    channel=channel,
                    entity_type=entity_type,
                    surrogate=surrogate,
                    span_start=start,
                    span_end=end,
                    scope=scope,
                )
            )
        out_parts.append(text[cursor:])
        return ChannelResult(
            channel=channel,
            masked_text="".join(out_parts),
            records=tuple(records),
        )


class InterceptionLedger:
    """Thread-safe, surrogate-only audit ledger (the G5 artifact source).

    Mirrors :class:`~pii_anon.tokenization.reidentification.ReidentificationService`'s
    ``audit_log`` pattern: a :class:`~threading.Lock`-guarded append plus
    copy-on-read so a caller never holds a live reference into the internal list.
    Records carry only surrogates (never plaintext); :meth:`as_dict` is the G5
    audit shape and is also consumed by the S6-05 leakage-Sankey.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[InterceptionRecord] = []

    def append(self, records: Iterable[InterceptionRecord]) -> None:
        """Append records (thread-safe)."""
        with self._lock:
            self._records.extend(records)

    def records(self) -> tuple[InterceptionRecord, ...]:
        """Return a copy-on-read snapshot of the recorded substitutions."""
        with self._lock:
            return tuple(self._records)

    def counts_by_channel(self) -> dict[AgentChannel, int]:
        """Return the count of surrogate substitutions per channel."""
        with self._lock:
            counts: dict[AgentChannel, int] = {}
            for rec in self._records:
                counts[rec.channel] = counts.get(rec.channel, 0) + 1
            return counts

    def as_dict(self) -> dict[str, object]:
        """Serialize to the G5 audit-artifact shape (surrogate-only)."""
        with self._lock:
            records = list(self._records)
        counts: dict[str, int] = {}
        for rec in records:
            counts[rec.channel.value] = counts.get(rec.channel.value, 0) + 1
        return {
            "counts_by_channel": counts,
            "records": [rec.as_dict() for rec in records],
        }


@dataclass(slots=True)
class FourChannelGuard:
    """Least-privilege 4-channel interceptor with a no-raw-PII-persist invariant.

    Parameters
    ----------
    masker:
        The :class:`ChannelMasker` used to detect + replace PII. ``None`` selects
        the dependency-free deterministic default (:class:`_DefaultMasker`),
        keyed by ``surrogate_key``. (When a custom ``masker`` is supplied it owns
        its own keying and ``surrogate_key`` is ignored.)
    surrogate_key:
        The secret key the default masker uses to derive surrogate token ids
        (a KEYED HMAC — see :func:`_keyed_token_id`). ``None`` (the default)
        mints a fresh random per-instance key (:func:`secrets.token_bytes`) at
        CONSTRUCTION — **secure-by-default**: the emitted surrogates (and the
        persisted G5 ledger) are then NOT offline-dictionary-reversible without
        this process's key. Supply a fixed key for the FR-030 reproducible path
        (byte-identical masked output + ledger given the same key/seed/scope).
        Key generation is the ONLY non-deterministic step and it happens once at
        construction, never on the per-mask path.
    token_store:
        Optional injected store for reversible channels — prefer the S6-03
        :class:`~pii_anon.tokenization.encrypted_store.EncryptedSQLiteTokenStore`
        so any raw at rest is AEAD-encrypted. A non-reversible channel NEVER
        touches the store.
    reversible_channels:
        The channels (a :class:`frozenset`) permitted to persist a
        surrogate -> surrogate linkage record (the ``plaintext`` column receives
        the SURROGATE, never raw PII — see :meth:`_persist`). **TRACE is
        reversible only if explicitly opted in** (default: never —
        least-privilege, AX-006). True authorized reversal-to-raw is FR-027
        (session pseudonyms) and is OUT OF S6-02's scope.
    ledger:
        Optional :class:`InterceptionLedger` to record every substitution into
        (the G5 audit source). One is created if omitted.
    """

    masker: ChannelMasker | None = None
    surrogate_key: bytes | None = None
    token_store: TokenStore | None = None
    reversible_channels: frozenset[AgentChannel] = frozenset()
    ledger: InterceptionLedger = field(default_factory=InterceptionLedger)
    # Resolved, never-None masker (set in __post_init__). Not a constructor arg.
    _masker: ChannelMasker = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ``masker=None`` is the ergonomic "use the default" sentinel; resolve it
        # to the dependency-free deterministic in-tree KEYED masker. The default
        # masker is keyed by ``surrogate_key`` (or a fresh random per-instance
        # key when ``surrogate_key is None`` — secure-by-default; this is the
        # ONLY construction-time use of ``secrets`` and it never touches the
        # per-mask path, so per-mask determinism-under-a-fixed-key holds). Both
        # the public ``masker`` attribute (for introspection) and the private
        # ``_masker`` (the never-None one the hot path reads) end up concrete.
        if self.masker is not None:
            resolved: ChannelMasker = self.masker
        else:
            key = (
                self.surrogate_key
                if self.surrogate_key is not None
                else secrets.token_bytes(_SURROGATE_KEY_BYTES)
            )
            resolved = _DefaultMasker(key)
        self.masker = resolved
        self._masker = resolved

    # -- public surface ----------------------------------------------------
    def intercept(
        self, text: str, *, channel: AgentChannel, scope: str
    ) -> ChannelResult:
        """Mask ``text`` on ``channel``, enforce no-raw-persist, audit + persist.

        Raises
        ------
        NoRawPIIPersistError
            If any raw PII value survives in the masked text or a record after
            masking (FR-026 / AX-006).
        """
        result = self._masker.mask(text, channel=channel, scope=scope)
        # The set of raw values this masking claims to have removed — exactly
        # what must NOT survive anywhere post-mask.
        known_values = self._known_raw_values(text, result)
        self._assert_no_raw_pii(result.masked_text, result.records, known_values)
        self.ledger.append(result.records)
        if channel in self.reversible_channels:
            self._persist(result, scope=scope)
        return result

    def intercept_all(
        self, payloads: Mapping[AgentChannel, str], *, scope: str
    ) -> dict[AgentChannel, ChannelResult]:
        """Intercept every channel in ``payloads`` independently."""
        return {
            channel: self.intercept(text, channel=channel, scope=scope)
            for channel, text in payloads.items()
        }

    # -- invariants --------------------------------------------------------
    def _assert_no_raw_pii(
        self,
        masked_text: str,
        records: tuple[InterceptionRecord, ...],
        known_values: Iterable[str],
    ) -> None:
        """RAISE if any raw PII value survives the masking (FR-026 headline).

        Scans both the ``masked_text`` AND the serialized form of every record
        (the about-to-be-persisted/audited surface) for each known raw value.
        Loud by design — never a silent pass.
        """
        for raw in known_values:
            if not raw:
                continue
            if raw in masked_text:
                raise NoRawPIIPersistError(
                    "raw PII value survived masking in masked_text "
                    "(entity unmasked) — refusing to persist a leak"
                )
            for rec in records:
                if raw in repr(rec.as_dict()):
                    raise NoRawPIIPersistError(
                        "raw PII value survived masking in an interception "
                        "record — refusing to persist a leak"
                    )

    def _known_raw_values(
        self, source_text: str, result: ChannelResult
    ) -> set[str]:
        """The raw values masking claims to have removed.

        For each record, the source substring it spans is the raw value that
        must NOT survive. A masker that LIES (reports a record but leaves the raw
        text) is caught because the spanned source value is still present in the
        unchanged masked text.
        """
        values: set[str] = set()
        for rec in result.records:
            if 0 <= rec.span_start < rec.span_end <= len(source_text):
                values.add(source_text[rec.span_start : rec.span_end])
        return values

    # -- persistence (reversible channels only) ----------------------------
    def _persist(self, result: ChannelResult, *, scope: str) -> None:
        """Persist a surrogate -> surrogate linkage record for a reversible channel.

        The ``plaintext`` column of each :class:`TokenMapping` receives the
        **surrogate** value (``plaintext=rec.surrogate``), not the raw PII — no
        raw value is ever written to the store (FR-026: the strongest reading of
        no-raw-PII-persist; the on-disk payload is additionally AEAD-encrypted
        ciphertext at rest, defense-in-depth). This persists a least-privilege
        linkage record gated by ``reversible_channels``; it does NOT enable
        authorized reversal-to-raw. **True surrogate -> raw reversal is FR-027
        (session pseudonyms) and is out of S6-02's scope** — a future story that
        needs it would thread the raw value through a reversible masker, at which
        point the EncryptedSQLiteTokenStore AEAD-at-rest property becomes
        load-bearing.

        Imports :class:`TokenMapping` lazily so importing this module never
        forces the tokenization store import (keeps the agentic surface light +
        the import graph minimal). A non-reversible channel never reaches here.
        """
        if self.token_store is None:
            return
        from pii_anon.tokenization.store import TokenMapping

        for rec in result.records:
            self.token_store.put(
                TokenMapping(
                    scope=scope,
                    token=rec.surrogate,
                    plaintext=rec.surrogate,
                    entity_type=rec.entity_type,
                    version=_SURROGATE_VERSION,
                    created_at=0.0,
                    expires_at=None,
                )
            )
