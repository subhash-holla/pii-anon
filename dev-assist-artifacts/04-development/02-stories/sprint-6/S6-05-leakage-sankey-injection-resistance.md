# S6-05 — agentic leakage-Sankey + prompt-injection exfiltration resistance (DC-13)

> **Cold-pickup invariant**: executable cold. Consumes the S6-02 `InterceptionLedger` (DONE) to build a **leakage-Sankey** — a flow graph of where PII is blocked vs leaked across the agent channels — and scores **prompt-injection exfiltration resistance** (attack-success-rate vs benign-task-success). Fully in-tree: the Sankey is a pure function of the ledger + a leak-scan of outbound payloads; the injection scorer uses the DATA `build_payloads` when the `datasets` extra is importable, else an in-tree INERT synthetic payload set. A live-agent transcript-residual run is Pass-2.

| Field | Value |
|---|---|
| Epic | **E6 Agentic interception** (DC-13) |
| State | **REVIEW** (owner: dev-assist-development-executor; main-tree, branch `pdlc/sota-program`; RED `9b6b3c3` → GREEN `5b35a12` → REFACTOR `5d34c11` → REFACTOR/remediation `d23ca23`) |
| provisional_status | AGENT_SIMULATED (the leakage-Sankey over the real S6-02 ledger, the leaked-vs-blocked flow accounting, and the injection-resistance scoring over INERT synthetic payloads all run for REAL in-tree; a live-agent runtime + real transcript-residual leakage are Pass-2 → `is_representative=True`) |
| Implements | **FR-028** (per-channel agentic leakage counts — the leakage-Sankey — MUST); **FR-029** (prompt-injection exfiltration resistance: ASR vs benign-task-success — MUST); upholds **AX-002** (deterministic Sankey + scoring), **AX-001** (INERT synthetic payloads — no real PII). Feeds the SDO **G5** audit half + the S7 canonical-run audit evidence. |
| Traces | Design **DC-13** (`D-implementation-ready-design.md:23,49`). UC-22 (leakage-Sankey) + UC-23 (injection resistance). **Channel-model reconciliation (for traceability):** FR-028 says "leakage-Sankey, 6 channels"; S6-02 (FR-025) intercepts the **4** agent channels (PROMPT/MEMORY/TOOL_IO/TRACE). S6-05 builds the Sankey as a **6-node flow graph** — the 4 interception source-channels + the 2 flow sinks `{blocked, leaked}` — with **per-source-channel** leakage counts (the FR-028 intent). The channel set is **extensible** (a future 6-source taxonomy plugs in); the in-tree default exercises the S6-02 4-channel ledger. Flagged explicitly so the traceability reviewer can confirm the 4-source/6-node reading satisfies FR-028 or request a wider source taxonomy. |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[SECURITY-TEST]` `[PROPERTY-TEST]` `[AUDIT]` |
| Files owned | `src/pii_anon/agentic/leakage_sankey.py` (**new** — `SankeyEdge`, `LeakageSankey`, `build_leakage_sankey`, `InjectionResistanceReport`, `score_injection_resistance` + an in-tree INERT payload set). **Additive** to `src/pii_anon/agentic/__init__.py` (re-exports). `tests/test_agentic_leakage_sankey.py` (**new**). |
| Depends on | **S6-02 DONE** (`InterceptionLedger` + `FourChannelGuard` + `AgentChannel`). Optionally consumes the DATA `pii_anon_datasets.scoring.adversary.{build_payloads, INTENT_TAGS}` when the `datasets` extra is present (else the in-tree INERT fallback). |
| Blocks | Nothing hard. Feeds the SDO **G5** + the S7 canonical-run audit evidence. |
| Size | **M** (one new module, ~10 acceptance tests). |

## 1. Intent
The S6-02 ledger records WHERE PII was masked per channel; S6-05 turns that into the auditable **leakage-Sankey** (a flow graph: each source channel → `blocked` for masked spans, → `leaked` for any raw value that survived into an outbound payload) and a **prompt-injection resistance** score. A leak edge is created only when a KNOWN raw value survives an outbound payload (detected by the caller-supplied `known_values` — never by re-detecting, which would be circular). The injection scorer runs adversarial payloads (base64 / homoglyph / zero-width transforms) through the `FourChannelGuard` and measures attack-success-rate (a raw value exfiltrated) against benign-task-success (legitimate content preserved). All in-tree, deterministic, INERT-synthetic; a live agent runtime is Pass-2.

## 2. Approach / scope
- **`SankeyEdge`** (frozen) — `source: str` (the channel), `target: str` (`"blocked"` | `"leaked"`), `entity_type: str`, `count: int`.
- **`LeakageSankey`** (frozen) — `edges: tuple[SankeyEdge, ...]`; `leaked_count() -> int`, `blocked_count() -> int`, `counts_by_channel() -> dict[str, dict[str, int]]`, `as_dict() -> dict` (the renderer shape: `{"nodes": [...], "links": [...]}` — a 6-node graph for the 4 channels + `blocked`/`leaked`).
- **`build_leakage_sankey(ledger: InterceptionLedger, outbound: Mapping[AgentChannel, str], *, known_values: Iterable[str]) -> LeakageSankey`** — for each ledger record: a `blocked` edge (the span was masked). For each `known_value` that survives an `outbound[channel]` payload: a `leaked` edge (raw PII reached the outbound boundary). Pure, deterministic; never re-detects.
- **`InjectionResistanceReport`** (frozen) — `attack_success_rate: float`, `benign_task_success_rate: float`, `n_payloads: int`, `n_benign: int`, `intent_tags: tuple[str, ...]`, `is_representative: bool = True`.
- **`score_injection_resistance(guard: FourChannelGuard, *, scope: str) -> InjectionResistanceReport`** — try `from pii_anon_datasets.scoring.adversary import build_payloads, INTENT_TAGS` (the `datasets` extra); else an in-tree INERT synthetic payload set with ≥3 transform intents (base64 / a fixed homoglyph map / zero-width). Run each payload through `guard.intercept(channel=PROMPT, scope=scope)`; an **exfiltration** = a known raw value surviving the masked output → counts toward ASR; benign-task-success = legitimate (non-PII) content preserved. `is_representative=True` (a live-agent transcript-residual run is Pass-2).

## 2a. Pre-claim de-risk (verify against live code on claim)
- **RISK-1 (no circular leak detection):** a `leaked` edge is created ONLY from a caller-supplied `known_value` surviving an outbound payload — never by re-running detection on the masked output (which would conflate detector misses with leakage). A `[SECURITY-TEST]` pins that a masked output with no known-value survivor yields ZERO `leaked` edges.
- **RISK-2 (FR-028 per-channel counts):** the Sankey carries per-source-channel `blocked`/`leaked` counts (`counts_by_channel()`); the `as_dict()` is a 6-node graph (4 channels + `blocked` + `leaked`). The 4-source/6-node reconciliation (vs FR-028's "6 channels") is flagged in §Traces for the reviewer.
- **RISK-3 (determinism, AX-002):** the Sankey is a pure function of the ledger + outbound + known_values; the in-tree payloads are a fixed list (no `random`/`uuid`/`time`); same inputs → byte-identical Sankey + report.
- **RISK-4 (INERT synthetic, AX-001):** the in-tree injection payloads are INERT (the transforms are recoverable encodings of SYNTHETIC values, never real PII); no live exfiltration.
- **RISK-5 (import isolation + optional dep):** `leakage_sankey.py` imports only stdlib + `pii_anon.agentic` (+ the LAZY optional `pii_anon_datasets` inside the function); imports NOTHING from `eval_framework.rating`/`attacks`/the SDO gate/`orchestrator`. `score_injection_resistance` works with the `datasets` extra ABSENT (the in-tree fallback).
- **RISK-6 (off-limits):** `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `competitor_compare.py` byte-identical (RISK-6); `agentic/interception.py` consumed read-only (S6-02). AX-001 synthetic.

## 3. Given / When / Then (acceptance)
- **A1 — Sankey edges sum to the intercepted total `[UNIT-TEST]`.** Given a ledger with N records, `build_leakage_sankey` yields `blocked` edges summing to N (+ any `leaked` edges from survivors).
- **A2 — leaked edge when a raw value survives outbound `[SECURITY-TEST]`.** Given an outbound payload that still contains a `known_value`, a `leaked` edge for that channel/entity_type is created; `leaked_count() > 0`.
- **A3 — no leaked edge without a survivor `[SECURITY-TEST]`.** Given outbound payloads with NO known-value survivor, `leaked_count() == 0` (no circular re-detection).
- **A4 — blocked edge from the ledger `[UNIT-TEST]`.** Each masked ledger record yields a `blocked` edge on its channel.
- **A5 — `as_dict()` renderer shape `[CONTRACT-TEST]`.** `as_dict()` returns `{"nodes": [...], "links": [...]}` with the 4 channel nodes + `blocked` + `leaked` (6 nodes); links carry per-channel counts.
- **A6 — deterministic `[PROPERTY-TEST]`.** Same ledger + outbound + known_values → byte-identical Sankey `as_dict()`; same guard → byte-identical `InjectionResistanceReport`.
- **A7 — injection ASR zero when all masked `[SECURITY-TEST]`.** When the guard masks every payload's PII, `attack_success_rate == 0.0` (no exfiltration).
- **A8 — injection detects exfiltration `[SECURITY-TEST]`.** Given a deliberately-incomplete guard that leaks one payload's raw value, `attack_success_rate > 0.0`.
- **A9 — benign-task-success tracked `[UNIT-TEST]`.** Legitimate (non-PII) content survives masking → `benign_task_success_rate` reflects it.
- **A10 — in-tree payloads without the datasets extra + ≥3 intents `[UNIT-TEST]` `[AUDIT]`.** With `pii_anon_datasets` absent, `score_injection_resistance` uses the in-tree INERT payloads; `intent_tags` covers ≥3 transforms (base64 / homoglyph / zero-width); import isolation holds (AST).

## 5. Notes / non-goals
- **Non-goal:** a live agent runtime / real transcript-residual leakage — Pass-2 (`is_representative=True`).
- **Non-goal:** the DATA query-aware scorer (S6-01) + the BYO-SDK adapter (S6-04) + session pseudonyms (FR-027, a SHOULD).
- **Non-goal:** widening the agent-channel source taxonomy beyond the S6-02 4 channels (if FR-028 strictly needs 6 SOURCE channels, that's a documented follow-up — the channel set is extensible).

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[CONTRACT-TEST]` `[SECURITY-TEST]` `[PROPERTY-TEST]` `[AUDIT]`. **Reviewers (canonical 5-gate story set):** **security-sast** (**PRIMARY** — the non-circular leak detection + the injection-resistance ASR) + **axiom-compliance** (AX-001 INERT + AX-002 determinism + import isolation) + **code-quality** + **requirements-coverage** (FR-028/029 MUST coverage + the 4-vs-6 channel reconciliation honesty) + **traceability** (DC-13 → FR-028/029 + the channel-model note). All five APPROVE.
> **Adversarial close:** folded into the S6 work-stream close (the agentic security surface) — independent attempts to create a phantom `leaked` edge, hide an exfiltration, or break determinism. Bar = 0 upheld.

## 12. Definition of Done
- [x] **RED**: `tests/test_agentic_leakage_sankey.py` (A1–A10) written first & failing (`ModuleNotFoundError`). RED precedes GREEN.
- [x] **GREEN**: `agentic/leakage_sankey.py` + additive `agentic/__init__` re-exports — all A1–A10 green.
- [x] **REFACTOR**: tidy; additive edge tests only.
- [x] **Quality gate**: full suite green (no regression); ruff clean; mypy clean under BOTH invocations; coverage ≥84% (new module 98% ≥90%).
- [x] **Security (headline)**: A2/A3 (non-circular leak detection) + A7/A8 (injection ASR).
- [x] **Determinism (AX-002)**: A6 byte-identical; no random/uuid/time/secrets.
- [x] **Untouched / user-WIP**: `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `competitor_compare.py` (md5 `7cae16c89f4c97136e1a12394dae2025`) + `agentic/interception.py` byte-identical; `artifacts/benchmarks/*` never written; narrow `git add` of owned files only.
- [ ] **Story-gate APPROVE** — `_reviews/story/S6-05/`, all 5 reviewers APPROVE; substantive MINOR + ALL MAJOR remediated in-loop. *(orchestrator-dispatched; State→REVIEW)*

## Evidence (filled on completion)
*Provisional status: AGENT_SIMULATED. The leakage-Sankey over the real S6-02 ledger, the leaked-vs-blocked flow accounting, and the injection-resistance scoring over INERT synthetic payloads run for REAL in-tree. A live-agent runtime + real transcript-residual leakage are Pass-2 (`is_representative=True`).*

### Commits (strict TDD, main tree, branch `pdlc/sota-program`)
- **RED** `9b6b3c3` — `test: S6-05 RED — pin FR-028 leakage-Sankey + FR-029 injection resistance (A1–A10)` (tests-only: `tests/test_agentic_leakage_sankey.py` + story).
- **GREEN** `5b35a12` — `feat: S6-05 GREEN — implement FR-028 leakage-Sankey + FR-029 injection resistance` (`src/pii_anon/agentic/leakage_sankey.py` + additive `src/pii_anon/agentic/__init__.py`).
- **REFACTOR** `5d34c11` — `refactor: S6-05 — additive edge tests (classifier + resistance-on-raise paths)` (tests-only; module coverage 92% → 98%).
- **REFACTOR/remediation** `d23ca23` — `refactor: S6-05 — report only obfuscation transforms in intent_tags (path symmetry)` (small `intent_tags` tweak so the in-tree fallback path reports the SAME obfuscation taxonomy as the DATA path — the `direct` verbatim probe is a measurement scaffold, still scored but excluded from `intent_tags`; + its pinning test). New module coverage 98%.

Full S6-05 diff scope = exactly the 4 owned paths (story + `agentic/__init__.py` + `agentic/leakage_sankey.py` + `tests/test_agentic_leakage_sankey.py`); `git diff --stat e569761..HEAD` = 1280 insertions / 2 deletions (the `__all__`/`__getattr__` restructure), nothing outside owned files.

### RED-before-GREEN proof (git-evidenced)
- RED commit `9b6b3c3` adds ONLY `tests/test_agentic_leakage_sankey.py` (+ the story) — `git show --name-status 9b6b3c3` shows two `A` entries, no production module.
- `git ls-tree 9b6b3c3 -- src/pii_anon/agentic/leakage_sankey.py` is **empty** (module ABSENT at RED); the same path is a blob at GREEN `5b35a12`. Running the RED test at `9b6b3c3` fails `ModuleNotFoundError: No module named 'pii_anon.agentic.leakage_sankey'` (verified directly; 1 collection error, 0 passed) — RED genuinely precedes GREEN.

### Tests — 29 cases (A1–A10 + frozen-contract + REFACTOR edges), FR/AX IDs in names
A1 `test_fr028_a1_blocked_edges_sum_to_intercepted_total`; A2 `test_fr028_a2_leaked_edge_when_known_value_survives_outbound`, `…a2_leaked_entity_type_inferred_from_surviving_value`; A3 (non-circular headline) `…a3_no_leaked_edge_without_survivor_non_circular`, `…a3_masked_output_is_never_re_detected`, `…a3_empty_known_values_yields_zero_leaks`; A4 `…a4_blocked_edge_per_masked_record_on_its_channel`; A5 `…a5_as_dict_renderer_shape_six_nodes`, `…a5_as_dict_links_aggregate_per_channel_target`; A6 (determinism) `test_ax002_a6_sankey_is_byte_identical_for_same_inputs`, `…a6_injection_report_is_byte_identical`, `…a6_no_nondeterministic_imports_in_module`; A7 `test_fr029_a7_asr_zero_when_guard_masks_everything`; A8 `…a8_asr_positive_when_guard_leaks_a_raw_value`, `…a8_asr_is_a_unit_interval_rate`; A9 `…a9_benign_task_success_tracked`; A10 `…a10_in_tree_payloads_without_datasets_extra`, `test_ax001_a10_import_isolation_ast`, `…a10_in_tree_payloads_are_inert_synthetic`. Frozen contract: `…sankey_edge_is_frozen`, `…leakage_sankey_is_frozen`, `…injection_report_defaults_is_representative_true`. REFACTOR edges: `…edge_guard_raise_counts_as_resistance_not_exfiltration`, `…edge_classify_survivor_phone_shape`, `…edge_classify_survivor_generic_pii_shape`, `…edge_as_dict_blocked_only_has_no_leaked_link`, `…edge_empty_known_value_string_is_not_a_survivor`, `…edge_empty_ledger_and_outbound_is_empty_graph`, `…edge_intent_tags_name_only_obfuscation_transforms`.

### Quality gates (all clean)
- `ruff check src tests` → **All checks passed**.
- `mypy src/pii_anon` (canonical) → **Success: no issues found in 135 source files**.
- `mypy src/pii_anon --strict` → **Success: no issues found in 135 source files** (the lazy optional `pii_anon_datasets.scoring.adversary` carries `# type: ignore[import-untyped]  # noqa: PLC0415` — ordering matters: mypy only honours `type: ignore` when it precedes the `noqa`).
- Coverage: new module `leakage_sankey.py` **98%** (≥90% target); `agentic/interception.py` stays **100%** (consumed read-only); module-targeted total 97.95% ≥84.
- Full suite (`PYTHONPATH=src python -m pytest`, canonical addopts incl. `--cov-fail-under=84`), two independent clean runs:
  - GREEN-state (production complete, pre the REFACTOR tests): **3414 passed, 16 skipped, 9 deselected, 0 failed**, total coverage **87.50%** (1108s).
  - After REFACTOR `5d34c11` (+6 tests-only edge cases): **3420 passed, 16 skipped, 9 deselected, 0 failed**, total coverage **87.44%** (1060s). The +6 exactly accounts for the new edge cases (3414 − 22 + 28 = 3420) — zero regressions.
  - The final `d23ca23` state adds +1 test (`…edge_intent_tags_name_only_obfuscation_transforms`) and a 1-line `intent_tags` tweak isolated to this module (nothing else imports `leakage_sankey`), so the final full count is **3421 passed / 0 failed** (the clean authoritative run on `d23ca23` reconfirms this). No `artifacts/benchmarks/*` written by any run; coverage ≥84 every run.

### Security headline (RISK-1 non-circular leak detection)
A `leaked` edge is created ONLY by a caller-supplied `known_value` surviving an `outbound[channel]` payload (a pure membership test); detection is NEVER re-run on the masked output. No survivor ⇒ `leaked_count()==0` — pinned by A3 ×3 (incl. a payload containing an email-SHAPED but undeclared token yielding zero leaks). The leak entity-type LABEL is derived from the confirmed survivor's shape and never echoes the raw value. Injection ASR (A7/A8): a guard that raises `NoRawPIIPersistError` counts as resistance (value did not cross the boundary); a guard that silently fails to mask is an exfiltration; ASR ∈ [0,1].

### Channel-model reconciliation (for traceability reviewer)
FR-028 reads "6 channels"; the Sankey is a **6-node** graph = the 4 S6-02 `AgentChannel` source nodes (PROMPT/MEMORY/TOOL_IO/TRACE) + the `blocked`/`leaked` sinks, with per-source-channel counts. `_SOURCE_CHANNELS` derives from `AgentChannel`, so a wider source taxonomy plugs in without changing the module. The 4-source/6-node reading vs FR-028's "6 channels" is flagged here for the reviewer to confirm or request a wider source taxonomy (documented non-goal §5).

### Import isolation + optional dep (RISK-5)
`leakage_sankey.py` imports only stdlib (`base64`, `collections.abc`, `dataclasses`) + `pii_anon.agentic.interception`; the optional `pii_anon_datasets.scoring.adversary` is imported LAZILY inside `score_injection_resistance`. Nothing from `eval_framework.rating`/`attacks`/SDO gate/`orchestrator` (AST-pinned by `test_ax001_a10_import_isolation_ast`). With the `datasets` extra forced absent (monkeypatched `__import__`), the in-tree INERT fallback scores with ≥3 transforms (base64 / homoglyph / zero-width) over a SYNTHETIC value (AX-001).

### Protected / user-WIP byte-identical (verified post-REFACTOR)
- `src/pii_anon/orchestrator.py` md5 `0afc6deed62bbd0653ae1051b723bace` ✓ (unchanged).
- `tests/test_moe_enhancements.py` md5 `910e9cd66ad6e38c7bb64a9c51ecb1cb` ✓ (unchanged).
- `src/pii_anon/evaluation/competitor_compare.py` md5 `7cae16c89f4c97136e1a12394dae2025` ✓ (unchanged).
- `src/pii_anon/agentic/interception.py` consumed READ-ONLY — md5 `7c306f41205c7bab38d907a4f29cee82` unchanged across RED/GREEN/REFACTOR.
- The SDO gate `competitive_supremacy.py` never touched. `artifacts/benchmarks/*` never written by this story (the pre-existing `M` entries are user-WIP from the session start, untouched). Every commit used a narrow `git add` of owned paths only (no `-A`/`-u`/`.`/`commit -a`).
