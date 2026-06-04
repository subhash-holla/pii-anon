# S6-02 — 4-channel least-privilege interception + no-raw-PII-persist (DC-13)

> **Cold-pickup invariant**: executable cold. A NEW `src/pii_anon/agentic/` package (off the cycle-prone `routing/` path) that intercepts PII on all four agent channels (prompt / memory / tool-I/O / trace) under least-privilege (AX-006) and GUARANTEES no raw PII is persisted to any channel after masking (FR-026). The per-channel `InterceptionLedger` it emits is a direct **G5 audit input** + the source the S6-05 leakage-Sankey consumes. Fully in-tree (the default masker reuses the canonical surrogate-token regex; the optional encrypted token store is the S6-03 `EncryptedSQLiteTokenStore`). No new hard dependency.

| Field | Value |
|---|---|
| Epic | **E6 Agentic interception** (DC-13: router pre-filter + query-aware gate + 4-channel least-privilege + no-raw-PII-persist) |
| State | **TODO** |
| provisional_status | AGENT_SIMULATED (the 4-channel interception, the no-raw-PII-persist invariant, the least-privilege per-channel reversibility policy, the surrogate-only ledger, and determinism all run for REAL in-tree against SYNTHETIC text + the real S6-03 AEAD store; a live agent runtime + real transcript-residual leakage are Pass-2) |
| Implements | **FR-025** (intercept all four agent channels prompt/memory/tool-I/O/trace under least-privilege — MUST); **FR-026** (persist NO raw PII to any channel after masking — MUST, **AX-006**); upholds **AX-006** (least-privilege — the TRACE channel can never reverse; only explicitly-reversible channels touch the token store), **AX-002** (determinism), **AX-001** (synthetic fixtures). Feeds **FR-028** (the leakage-Sankey, S6-05) + the SDO **G5** audit half. |
| Traces | Design **DC-13** (`D-implementation-ready-design.md:23,49` — "4-channel least-privilege interception (prompt/memory/tool-I/O/trace, AX-006); **no raw PII persisted post-masking** (FR-026)"). UC-20. |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[SECURITY-TEST]` `[PROPERTY-TEST]` `[AUDIT]` |
| Files owned | `src/pii_anon/agentic/__init__.py` (**new** — lazy re-exports). `src/pii_anon/agentic/interception.py` (**new** — `AgentChannel` enum, `InterceptionRecord`, `ChannelResult`, `ChannelMasker` Protocol, `FourChannelGuard`, `InterceptionLedger`, `NoRawPIIPersistError`). `tests/test_agentic_interception.py` (**new**). **Additive** to `src/pii_anon/errors.py` ONLY IF `NoRawPIIPersistError` is centralized there (else keep it in `interception.py`). |
| Depends on | None hard. CONSUMES (read-only, byte-identical): `src/pii_anon/tokenization/reidentification.py` (the canonical surrogate-token regex + the `ReidentificationService.audit_log` Lock-guarded-append pattern to mirror), `src/pii_anon/tokenization/encrypted_store.py` (S6-03 `EncryptedSQLiteTokenStore` — the optional injected reversible store), `src/pii_anon/errors.py` (`PiiAnonError` base). |
| Blocks | **S6-05** (leakage-Sankey consumes the `InterceptionLedger`). Feeds the SDO **G5** + the S7 canonical-run audit evidence. |
| Size | **M** (one new package, one module, ~10 acceptance tests). |

## 1. Intent
An agent pipeline leaks PII through four distinct channels — the **prompt** it sends an LLM, its **memory**/scratchpad, its **tool I/O** (function args + results), and its **trace**/telemetry. S6-02 intercepts all four under **least-privilege**: each channel is masked, and only channels EXPLICITLY declared reversible may persist a surrogate→raw mapping (via the injected S6-03 encrypted store); the others — especially **trace** — can never reverse. The load-bearing guarantee (FR-026/AX-006) is **no raw PII is persisted after masking**: a post-mask invariant scans the masked output + every persisted record and RAISES (`NoRawPIIPersistError`) if any raw PII value survives. The per-channel `InterceptionLedger` records only surrogates (never plaintext) and is the auditable G5 artifact (+ the S6-05 Sankey source). All in-tree against synthetic text; a live agent runtime is Pass-2.

## 2. Approach / scope
- **`AgentChannel(str, Enum)`** — EXACTLY `{PROMPT, MEMORY, TOOL_IO, TRACE}` (four, no more — a contract test pins the membership).
- **`InterceptionRecord`** (frozen) — `channel: AgentChannel`, `entity_type: str`, `surrogate: str`, `span_start: int`, `span_end: int`, `scope: str`. **No plaintext field by construction** (FR-026 — the record cannot carry raw PII even by accident; a contract test asserts no field holds the raw value).
- **`ChannelResult`** (frozen) — `channel`, `masked_text: str`, `records: tuple[InterceptionRecord, ...]`.
- **`NoRawPIIPersistError(PiiAnonError)`** — raised when the post-mask invariant detects a raw PII value surviving in the masked output or a persisted record.
- **`ChannelMasker` Protocol** (`@runtime_checkable`) — `mask(text: str, *, channel: AgentChannel, scope: str) -> ChannelResult`. The default in-tree `_DEFAULT_MASKER` detects+replaces using the canonical surrogate-token regex from `tokenization/reidentification.py` (no engine deps → CI-fast, deterministic).
- **`FourChannelGuard(masker, *, token_store=None, reversible_channels=frozenset())`** — `intercept(text, *, channel, scope) -> ChannelResult` and `intercept_all(payloads: Mapping[AgentChannel, str], *, scope) -> dict[AgentChannel, ChannelResult]`. After masking, `_assert_no_raw_pii(masked_text, records, known_values)` RAISES on any raw survival. A reversible channel (∈ `reversible_channels`) persists surrogate→token via the injected `token_store` (prefer `EncryptedSQLiteTokenStore` so any raw at rest is AEAD-encrypted — AX-006 defense-in-depth); a non-reversible channel NEVER touches the store. **TRACE is reversible only if explicitly opted in** (default: never — least-privilege).
- **`InterceptionLedger`** — Lock-guarded append + copy-on-read (mirrors `ReidentificationService.audit_log`); `counts_by_channel() -> dict[AgentChannel, int]`, `records() -> tuple[InterceptionRecord, ...]`, `as_dict() -> dict` (the G5 artifact shape; surrogate-only).

## 2a. Pre-claim de-risk (verify against live code on claim)
- **RISK-1 (FR-026 no-raw-PII-persist is the headline):** the `_assert_no_raw_pii` invariant must catch a raw value surviving in BOTH the masked text AND any persisted record; it RAISES (loud), never silently passes. A `[SECURITY-TEST]` injects a masker that "forgets" to mask one entity → the guard must raise.
- **RISK-2 (AX-006 least-privilege):** the TRACE channel must never reverse by default; only channels in `reversible_channels` persist a mapping; a `[SECURITY-TEST]` asserts a non-reversible channel writes nothing to the store.
- **RISK-3 (no plaintext in the record/ledger):** `InterceptionRecord` has no raw-value field; the ledger's `as_dict()` carries only surrogates. A contract test introspects the dataclass fields + the serialized dict for any raw value.
- **RISK-4 (determinism, AX-002):** no `random`/`uuid`/`time`/`secrets` on the interception path; same input → same masked output + same ledger. (Surrogate generation is deterministic from the existing tokenizer/store.)
- **RISK-5 (no new hard dep + import isolation):** `agentic/` imports only stdlib + `tokenization` + `errors`; it imports NOTHING from `eval_framework.rating`/`attacks`/the SDO gate. It MAY import `tokenization`. It must NOT import `orchestrator.py` (protected user-WIP).
- **RISK-6 (off-limits):** `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `evaluation/competitor_compare.py` byte-identical (RISK-6); `tokenization/encrypted_store.py` + `reidentification.py` consumed read-only. AX-001 synthetic fixtures only.

## 3. Given / When / Then (acceptance)
- **A1 — exactly four channels `[CONTRACT-TEST]`.** `set(AgentChannel) == {PROMPT, MEMORY, TOOL_IO, TRACE}` (membership pinned; adding/removing a channel fails the test).
- **A2 — mask PII in the prompt channel `[UNIT-TEST]`.** Given text with a synthetic email, `intercept(text, channel=PROMPT, scope=s)` returns a `ChannelResult` whose `masked_text` contains a surrogate (not the raw email) + a matching `InterceptionRecord`.
- **A3 — record carries no plaintext `[CONTRACT-TEST]` `[SECURITY-TEST]`.** Introspect `InterceptionRecord` fields + `as_dict()` — no field/value equals the raw PII; only the surrogate is present.
- **A4 — no-raw-PII-persist raises on leak `[SECURITY-TEST]`.** Given a deliberately-incomplete masker that leaves one raw value, `intercept`/`intercept_all` RAISES `NoRawPIIPersistError` (the invariant catches the survival in masked_text OR a persisted record).
- **A5 — least-privilege: TRACE never reverses `[SECURITY-TEST]` `[AUDIT]`.** With an injected token store and `reversible_channels={MEMORY}`, intercepting on TRACE persists NOTHING to the store; only MEMORY persists a surrogate→token mapping.
- **A6 — memory channel persists only a surrogate `[UNIT-TEST]`.** A reversible MEMORY channel writes a surrogate→token row to the (encrypted) store; the stored row carries no plaintext PII (read-back is the surrogate/ciphertext, never the raw value).
- **A7 — intercept_all covers every channel independently `[UNIT-TEST]`.** `intercept_all({PROMPT:.., MEMORY:.., TOOL_IO:.., TRACE:..})` returns a `ChannelResult` per channel, each masked, with per-channel records.
- **A8 — ledger is surrogate-only + feeds the G5 shape `[AUDIT]`.** After interceptions, `InterceptionLedger.as_dict()` carries only surrogates + `counts_by_channel()`; the shape is consumable as a G5 audit artifact (and by the S6-05 Sankey).
- **A9 — deterministic `[PROPERTY-TEST]`.** Same payloads + scope → byte-identical masked outputs + identical ledger (`as_dict()` equal); no random/uuid/time on the path.
- **A10 — import isolation + no-plaintext audit `[AUDIT]`.** AST scan: `agentic/interception.py` imports nothing from `eval_framework.rating`/`attacks`/SDO gate/`orchestrator`; the ledger/record types expose no raw-value field.

## 5. Notes / non-goals
- **Non-goal:** the leakage-Sankey + prompt-injection resistance (S6-05 — consumes this ledger) + the query-aware masking gate (S6-01) + the BYO-SDK adapter (S6-04).
- **Non-goal:** a live agent runtime / real transcript-residual leakage — Pass-2 (in-tree teeth use synthetic text + the real S6-03 AEAD store).
- **Non-goal:** wiring into `orchestrator.py` (protected user-WIP) — `agentic/` is a standalone library surface consumed via construction seams.

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[CONTRACT-TEST]` `[SECURITY-TEST]` `[PROPERTY-TEST]` `[AUDIT]`. **Reviewers (canonical 5-gate story set):** **axiom-compliance** (**PRIMARY** — AX-006 least-privilege + AX-002 + AX-001) + **security-sast** (the no-raw-PII-persist invariant + the no-plaintext record/ledger + the reversibility policy) + **code-quality** + **requirements-coverage** (FR-025/026 MUST coverage) + **traceability** (DC-13 → FR-025/026). All five APPROVE.
> **Adversarial close:** RECOMMENDED at the S6 work-stream close (a security/least-privilege surface) — independent attempts to leak a raw value past `_assert_no_raw_pii`, reverse a non-reversible channel, or get plaintext into the ledger. Bar = 0 upheld.

## 12. Definition of Done
- [ ] **RED**: `tests/test_agentic_interception.py` (A1–A10) written first & failing (`ModuleNotFoundError` on `pii_anon.agentic.interception`). RED precedes GREEN (git-evidenced).
- [ ] **GREEN**: `agentic/__init__.py` + `agentic/interception.py` — all A1–A10 green.
- [ ] **REFACTOR**: tidy; additive edge tests only.
- [ ] **Quality gate**: full suite green (no regression vs the post-S2-04 baseline); ruff clean; mypy clean under BOTH `mypy src/pii_anon` AND `mypy src/pii_anon --strict` (parametrize any numpy → `npt.NDArray[np.float64]` — likely none, pure-stdlib); coverage ≥84% (new module ≥90% by own tests).
- [ ] **Security (headline FR-026/AX-006)**: A4 (no-raw-persist raises) + A5 (TRACE never reverses) + A3 (no-plaintext record).
- [ ] **Determinism (AX-002)**: A9 byte-identical replay; no random/uuid/time/secrets.
- [ ] **Untouched / user-WIP**: `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `competitor_compare.py` (md5 `7cae16c89f4c97136e1a12394dae2025`) + `tokenization/*` byte-identical; `artifacts/benchmarks/*` never written; narrow `git add` of owned files only.
- [ ] **Story-gate APPROVE** — `_reviews/story/S6-02/`, all 5 reviewers APPROVE; substantive MINOR + ALL MAJOR remediated in-loop.

## Evidence (filled on completion)
*Provisional status: AGENT_SIMULATED. The 4-channel interception, the no-raw-PII-persist invariant, the least-privilege reversibility policy, the surrogate-only ledger, and determinism run for REAL in-tree against SYNTHETIC text + the real S6-03 AEAD store. A live agent runtime + real transcript-residual leakage are Pass-2.*
