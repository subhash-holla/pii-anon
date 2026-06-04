# S6-02 — 4-channel least-privilege interception + no-raw-PII-persist (DC-13)

> **Cold-pickup invariant**: executable cold. A NEW `src/pii_anon/agentic/` package (off the cycle-prone `routing/` path) that intercepts PII on all four agent channels (prompt / memory / tool-I/O / trace) under least-privilege (AX-006) and GUARANTEES no raw PII is persisted to any channel after masking (FR-026). The per-channel `InterceptionLedger` it emits is a direct **G5 audit input** + the source the S6-05 leakage-Sankey consumes. Fully in-tree (the default masker reuses the canonical surrogate-token regex; the optional encrypted token store is the S6-03 `EncryptedSQLiteTokenStore`). No new hard dependency.

| Field | Value |
|---|---|
| Epic | **E6 Agentic interception** (DC-13: router pre-filter + query-aware gate + 4-channel least-privilege + no-raw-PII-persist) |
| State | **REVIEW** |
| Owner | `dev-assist-development-executor` (worktree `agent-a9a00540b7d40ba30`, branch `worktree-agent-a9a00540b7d40ba30`, base `2464641`) |
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
- [x] **RED**: `tests/test_agentic_interception.py` (A1–A10) written first & failing (`ModuleNotFoundError` on `pii_anon.agentic.interception`). RED precedes GREEN (git-evidenced — RED `f47aa05` is an ancestor of HEAD).
- [x] **GREEN**: `agentic/__init__.py` + `agentic/interception.py` — all A1–A10 green.
- [x] **REFACTOR**: tidy; additive edge tests only (production unchanged since GREEN; +5 edge tests → 100% module coverage).
- [x] **Quality gate**: full suite green (3353 passed / 17 skipped / 9 deselected; no regression — +22 over the 3331 base); ruff clean; mypy clean under BOTH `mypy src/pii_anon` AND `mypy src/pii_anon --strict` (pure-stdlib — no numpy); coverage 87.37% repo-wide, `interception.py` + `__init__.py` both 100% by own tests.
- [x] **Security (headline FR-026/AX-006)**: A4 (no-raw-persist raises) + A5 (TRACE never reverses) + A3 (no-plaintext record).
- [x] **Determinism (AX-002)**: A9 byte-identical replay; no random/uuid/time/secrets (AST-pinned).
- [x] **Untouched / user-WIP**: `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `competitor_compare.py` (md5 `7cae16c89f4c97136e1a12394dae2025`) + `tokenization/*` + `errors.py` byte-identical; `artifacts/benchmarks/*` never written; narrow explicit `git add` of owned files only.
- [ ] **Story-gate APPROVE** — `_reviews/story/S6-02/`, all 5 reviewers APPROVE; substantive MINOR + ALL MAJOR remediated in-loop. *(orchestrator-dispatched; pending)*

## Evidence (filled on completion)
*Provisional status: AGENT_SIMULATED. The 4-channel interception, the no-raw-PII-persist invariant, the least-privilege reversibility policy, the surrogate-only ledger, and determinism run for REAL in-tree against SYNTHETIC text + the real S6-03 AEAD store. A live agent runtime + real transcript-residual leakage are Pass-2.*

### Execution (worktree `agent-a9a00540b7d40ba30`, branch `worktree-agent-a9a00540b7d40ba30`)
- **Base commit:** `2464641` (branch `pdlc/sota-program` HEAD — the commit that authored this story). The worktree was initially checked out at an ancestor (`2761a27`); fast-forwarded to `2464641` (the dispatched base) before claim. RED-precedes-GREEN is git-evidenced (`git merge-base --is-ancestor f47aa05 HEAD` ⇒ TRUE).
- **Commits (`git log --oneline 2464641..HEAD`):**
  - `f47aa05` — `test: S6-02 RED — pin FR-025/026 + AX-006 four-channel interception (A1–A10)` (tests-only; failed with `ModuleNotFoundError: No module named 'pii_anon.agentic'`).
  - `88319bd` — `feat: S6-02 GREEN — implement FR-025/026 + AX-006 four-channel interception (DC-13)`.
  - `fab9b1e` — `refactor: S6-02 — additive edge tests close agentic coverage to 100%` (test-only; production byte-identical to GREEN).

### Files owned (new)
- `src/pii_anon/agentic/__init__.py` — PEP 562 lazy re-exports (100% cov).
- `src/pii_anon/agentic/interception.py` — `AgentChannel`, `InterceptionRecord`, `ChannelResult`, `ChannelMasker` Protocol + default in-tree masker, `FourChannelGuard`, `InterceptionLedger`, `NoRawPIIPersistError` (100% line+branch cov).
- `tests/test_agentic_interception.py` — 22 tests (17 RED-pinned A1–A10 + 5 REFACTOR edge).

### Acceptance (A1–A10) — all green
| ID | Test(s) | Result |
|---|---|---|
| A1 exactly four channels | `test_fr025_a1_exactly_four_channels` | PASS — `set(AgentChannel) == {PROMPT, MEMORY, TOOL_IO, TRACE}`, `len == 4` |
| A2 mask prompt channel | `test_fr025_a2_mask_prompt_channel` | PASS — surrogate present, raw email absent, span points at source |
| A3 record no plaintext | `test_fr026_a3_record_has_no_plaintext_field` | PASS — fields == surrogate-only set; frozen; no value contains raw |
| A4 no-raw-persist raises | `test_fr026_a4_no_raw_persist_raises_on_masked_text_leak`, `…_via_intercept_all`, `…_error_is_pii_anon_error_subclass` | PASS — `_assert_no_raw_pii` RAISES `NoRawPIIPersistError` on a leaky masker |
| A5 TRACE never reverses | `test_ax006_a5_trace_never_reverses_by_default`, `…_reversible_only_on_explicit_opt_in` | PASS — TRACE writes 0 rows; only `reversible_channels` persist |
| A6 memory only-surrogate | `test_fr026_a6_memory_persists_only_surrogate` | PASS — store row token == surrogate (AEAD ciphertext at rest) |
| A7 intercept_all all channels | `test_fr025_a7_intercept_all_covers_every_channel` | PASS — one `ChannelResult` per channel, per-channel records |
| A8 ledger surrogate-only G5 | `test_fr026_a8_ledger_surrogate_only_g5_shape`, `…_copy_on_read_is_immutable` | PASS — `as_dict()` JSON-serializable, no raw PII, per-channel counts |
| A9 deterministic | `test_ax002_a9_deterministic_replay`, `…_no_nondeterministic_imports_on_path` | PASS — byte-identical replay; AST: no `random`/`uuid`/`time`/`secrets` |
| A10 import isolation | `test_ax001_a10_import_isolation`, `…_no_raw_value_field_audit`, `…_channel_masker_is_runtime_checkable` | PASS — AST: no `eval_framework.rating`/`attacks`/SDO/`orchestrator` imports |

### Quality gate (verified in-worktree)
- `ruff check src tests` — **All checks passed.**
- `mypy src/pii_anon` (config `strict=true`) — **Success: no issues (133 files).**
- `mypy src/pii_anon --strict` (explicit flag) — **Success: no issues (133 files).**
- `PYTHONPATH=src pytest` (full) — **3353 passed, 17 skipped, 9 deselected**; coverage **87.37%** (gate ≥84). New module own-coverage **100%** (line + branch).

### Protected / user-WIP byte-identical (md5, base==final)
- `src/pii_anon/orchestrator.py` = `4a837c52ccdb27925d1f7885e71667d0`
- `tests/test_moe_enhancements.py` = `a96d86248989e2bb5bb7fff9d65602b0`
- `src/pii_anon/evaluation/competitor_compare.py` = `7cae16c89f4c97136e1a12394dae2025`
- `src/pii_anon/errors.py` = `1d3ed9f784d425b25e76ed776a215a95` (NOT edited — `NoRawPIIPersistError` kept in `interception.py`)
- `src/pii_anon/tokenization/encrypted_store.py` = `132fa3c1b6ee14a98a7d3c6c8eb0b94f`
- `src/pii_anon/tokenization/reidentification.py` = `c2367ed62cb3e2a123ba88b8a3697751`
- `artifacts/benchmarks/*` — never written (no diff).

## History Log
- **2026-06-03 — `dev-assist-development-executor` (worktree `agent-a9a00540b7d40ba30`):** TODO → CLAIMED → IN_PROGRESS → REVIEW. Pre-claim validation passed (12 sections present; FR-025/026 + AX-001/002/006 confirmed in `requirements-document.md` / `D-implementation-ready-design.md`; consumed `reidentification.py` regex + `encrypted_store.py` store verified read-only). Strict TDD: RED `f47aa05` (tests-only, `ModuleNotFoundError`) → GREEN `88319bd` → REFACTOR `fab9b1e` (test-only). All gates green; all 6 protected md5s unchanged. Awaiting orchestrator-dispatched 5-gate story review (axiom-compliance PRIMARY + security-sast + code-quality + requirements-coverage + traceability). Merge-back is the orchestrator's responsibility.
