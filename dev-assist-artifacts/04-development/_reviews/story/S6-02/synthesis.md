# Story Gate Synthesis — S6-02 (4-channel least-privilege interception + no-raw-PII-persist, DC-13)

**Aggregate verdict: APPROVE** (iteration 2). 5/5 reviewers APPROVE; 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 MAJOR open. The only open findings are the standing traceability-matrix backfill (deferred to the S6 sprint gate, consistent with the S2-01/02/04 precedent).

## Reviewer set + verdicts

| Reviewer | iter-1 | iter-2 | Blocking (iter-1) | Disposition |
|---|---|---|---|---|
| axiom-compliance (PRIMARY) | APPROVE | **APPROVE** | — (2 OBS) | AX-006/AX-002/AX-001 upheld; OBS-2 (surrogate→surrogate) resolved by honest docstrings |
| **security-sast** | REQUEST_CHANGES | **APPROVE** | **MAJOR S6-02-01**: keyless-BLAKE2b surrogate dictionary-reversible (the G5 ledger wasn't de-identified) | CLOSED by keyed HMAC-SHA256; reviewer re-ran its own break-probe → resists |
| code-quality | APPROVE | **APPROVE** | — (2 MINOR) | both MINOR CLOSED (honest docstrings + FR/AX test renames) |
| requirements-coverage | APPROVE | — | — (1 MINOR matrix) | FR-025/026 fully in-tree-verified; matrix MINOR deferred |
| traceability | APPROVE | — | — (1 "MAJOR" = standing matrix gap) | DC-13→FR-025/026 clean; matrix backfill deferred (batched) |

## The blocking finding + remediation

**security-sast MAJOR (S6-02-01):** The default masker's surrogate token id was a **keyless BLAKE2b** of `(scope, entity_type, raw_value)`. The reviewer DEMONSTRATED an offline re-identification: from a surrogate + a candidate dictionary + the low-cardinality scope/entity, it recovered the synthetic raw value with no key. Because surrogates flow into the persisted `InterceptionLedger` (the G5 audit artifact) + the S6-05 leakage-Sankey — artifacts meant to be shareable de-identified evidence — a keyless hash of low-entropy PII is dictionary-reversible. This is a real privacy hole that the per-story functional tests (which only checked "surrogate ≠ raw value") could not surface — exactly what an independent security review catches.

**Remediation (`83b73bf`):** keyed the surrogate via `_keyed_token_id` = `hmac.new(surrogate_key, msg, sha256)` — byte-for-byte the canonical `DeterministicHMACTokenizer` / `encrypted_store` blind-index construction (no new crypto). `FourChannelGuard(surrogate_key: bytes | None = None)`: `None` → a random per-instance key via `secrets.token_bytes(32)` at construction (secure-by-default — the ledger is non-dictionary-reversible); an injected key → the FR-030 byte-identical reproducible path the canonical run uses. RNG is construction-only; the per-mask path stays deterministic given a fixed key (A9 reconciled to inject a fixed key + allow `secrets`/`os` for key-gen while AST-pinning the per-mask path RNG-free). Added `test_fr026_a11_keyed_surrogate_resists_dictionary_reidentification` (the inverse of the reviewer's probe). Both the security AND axiom reviewers independently re-ran the attack/determinism probes and confirmed closure.

**code-quality MINORs:** `_persist`/`reversible_channels` docstrings corrected to honest surrogate→surrogate (raw never at rest; true reversal-to-raw is FR-027, out of scope); the 5 edge tests renamed with FR/AX prefixes.

## Process note (worktree mis-allocation)

S6-02 was dispatched as a parallel worktree-isolated executor, but the harness allocated the worktree at a **stale base** (`2761a27`, not HEAD `2464641` — stale `.claude/worktrees/`). The executor self-healed (fast-forwarded its worktree branch to `2464641`, then did the work); its parallel sibling (S5-01) correctly REFUSED on the same stale base. The S6-02 branch (clean, 4 disjoint files) was ff-merged to `pdlc/sota-program`; the remediation ran in-tree. **The worktree mechanism is unreliable in this environment → remaining stories run sequentially in the main tree.**

## Next action

APPROVE → story REVIEW → DONE. The standing matrix backfill batches to the S6 sprint gate. S6-02's `InterceptionLedger` is a G5 audit input + the S6-05 leakage-Sankey source. A whole-S6 adversarial close is recommended at the S6 work-stream close (the agentic security surface).
