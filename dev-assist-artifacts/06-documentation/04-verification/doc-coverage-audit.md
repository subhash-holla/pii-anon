# doc-coverage-audit — pii-anon Stage 6 Documentation (D4 Verification, Axis 1 + Axis 3)

> **Wave D4 output.** Read-only verification of the six D3-authored deliverables under
> `06-documentation/03-authoring/` against the MUST FR/NFR + DC census (D1 §3/§4, the
> `requirements-document.md:§R7` MUST set). This file carries **Axis 1 (coverage)** and
> **Axis 3 (accuracy-against-code, api-reference)**. Trace/link integrity is in the sibling
> `doc-trace-integrity.md`. **Read-only on all sources; this verifier authored only the two
> files under `04-verification/`. Fixes happen by D3 re-author loopback, not here.**
>
> Deliverable abbreviations: **JRN** = project-journey.md · **CHG** = changelog.md ·
> **ADR** = architecture-and-adr.md · **API** = api-reference.md · **USR** = user-operator-guide.md ·
> **CON** = contributor-handbook.md.

---

## 0. Verdict (Axis 1 + Axis 3)

| Axis | Verdict | Basis |
|---|---|---|
| **Axis 1 — Coverage (MUST FR/NFR + DC)** | **PASS — zero documentation orphans** | All 23 MUST FRs, all 13 MUST NFRs, and all 15 DCs have ≥1 documenting deliverable (grep-confirmed). The DOCUMENTED bar is met. |
| **Axis 3 — Accuracy-against-code (api-reference D-5)** | **PASS for D-5 itself (>18 symbols re-verified)** | Every symbol the api-reference documents was found in live code. **BUT** a CATASTROPHIC accuracy defect lives in a *different* deliverable (CON), plus MAJOR/MINOR non-runnable examples in USR/CON — see §3. |
| **Net D4 disposition** | **LOOPBACK TO D3** | One CATASTROPHIC (CON RatingEnginePort fabricated contract) triggers the D3 re-author loop (cap 3). Coverage itself is clean. |

The coverage bar is the DOCUMENTED bar and it is **met with zero orphans** — this is a mature-brownfield
corpus with real authored signal, so the coverage table is dense (most MUSTs land in 4–6 deliverables).
The loopback is driven by **accuracy**, not coverage.

---

## 1. MUST FR coverage (23 MUST FRs — `requirements-document.md:§R7`)

ID → documenting deliverable(s). Every MUST FR has ≥1 home → **0 orphans**.

| MUST FR | Title (abbrev) | Documenting deliverables | Orphan? |
|---|---|---|---|
| FR-001 | BYO-pipeline adapter contract | JRN · CHG · ADR · API · USR · CON | no |
| FR-003 | Bayesian Bradley-Terry rating engine | JRN · CHG · ADR · CON | no |
| FR-004 | Coherent significance | JRN · CHG · ADR | no |
| FR-006 | Pseudonymization integrity distinct family | JRN · CHG · API · USR | no |
| FR-007 | CI ship/no-ship gate + per-language floor | JRN · CHG · API · USR | no |
| FR-008 | Canonical-run / provenance gate | JRN · CHG · API · USR | no |
| FR-009 | Pseudonymization integrity 5-axis family | JRN · CHG · API · USR | no |
| FR-010 | Anon vs pseudo distinct families (no-merge) | JRN · CHG · ADR · API · USR · CON | no |
| FR-011 | Real Tier-3 LLM-adversary re-id | JRN · CHG · ADR · API · USR | no |
| FR-013 | Full-power MIA (LiRA@128 + Secret-Sharer) | JRN · CHG · API · USR | no |
| FR-016 | Recall-floor by construction | JRN · CHG · ADR · API · USR · CON | no |
| FR-017 | Per-language recall-floor CI gate | CHG · ADR | no |
| FR-018 | MoE-router (learned routing + early-exit) | JRN · CHG · ADR · USR · CON | no |
| FR-019 | Reversible pseudonymization + key rotation | JRN · CHG · ADR · API · USR | no |
| FR-025 | Intercept four agent channels, least-privilege | JRN · CHG · ADR · API · CON | no |
| FR-026 | Persist no raw PII after masking (AX-006) | CHG · ADR · API | no |
| FR-028 | Per-channel agentic leakage counts | CHG · ADR · API | no |
| FR-029 | Prompt-injection exfiltration resistance | CHG · ADR · API | no |
| FR-031 | Native-format readers (Iterator[IngestRecord]) | JRN · CHG · ADR · API · USR · CON | no |
| FR-032 | Round-trip reconstruction preserves payload | CHG · ADR · API · USR · CON | no |
| FR-034 | Per-modality recall benchmark | CHG · API | no |
| FR-035 | CI gate on multimodal reader recall regression | CHG · ADR · API · USR | no |
| FR-036 | Identical scrub decisions stream/batch/offline | CHG · ADR · API · USR | no |

**Thin-but-covered (OBSERVATION, not orphan):**
- **FR-017** (2 deliverables: CHG, ADR) — covered; not in API/USR. Acceptable for an eval-internal CI-gate MUST.
- **FR-004** (3: JRN, CHG, ADR) — covered; correctly out-of-scope for API/USR (statistical-engine internal).
- **FR-034** (2: CHG, API) — covered with an explicit API callout; per-modality benchmark is DATA-track Pass-2.
- The D1 §9 thin-trace watch-list (**FR-033/FR-035/FR-036**) is each *explicitly* called out in API §8 and CHG
  Sprint-7 notes per the §5 callout requirement — present, not merely implied. **PASS** on the watch-list.

---

## 2. MUST NFR coverage (13 MUST NFRs) + DC coverage (DC-01..15)

### 2a. MUST NFRs — every MUST NFR has ≥1 home → **0 orphans**

| MUST NFR | Title (abbrev) | Documenting deliverables | Orphan? |
|---|---|---|---|
| NFR-001 | Bradley-Terry MCMC convergence | JRN · CHG · ADR · CON | no |
| NFR-002 | Significance coherence | CHG · ADR | no |
| NFR-004 | Statistical-power consumption (risk-tiered) | JRN · API · USR | no |
| NFR-005 | Scoring-run determinism | JRN · CHG · ADR · API · USR | no |
| NFR-006 | Canonical-run provenance | JRN · CHG · ADR · USR | no |
| NFR-011 | Router-on recall floor | JRN · CHG · ADR · API · USR · CON | no |
| NFR-012 | Tier-3 RRS power | JRN · CHG · ADR · API | no |
| NFR-013 | MIA power | JRN · CHG · API | no |
| NFR-014 | Pseudonymization integrity | JRN · CHG · ADR · API · USR | no |
| NFR-016 | Non-strippable re-id caveat | JRN · CHG · API · USR | no |
| NFR-020 | Calibrated confidence on every finding | CHG · ADR | no |
| NFR-023 | Stream/batch/offline parity | USR | no (thin — see O-01) |
| NFR-024 | No real PII in repo/fixtures/logs | CHG · ADR · CON | no |

**Thin-but-covered (OBSERVATION):**
- **NFR-023** (MUST) appears as a string token in only **USR §8f** (under the FR-036 parity heading). It is
  covered, but a MUST parity NFR in a single deliverable is the thinnest MUST-NFR row. ADR §3 lists "NFR-022/023"
  against DC-14 in the module-map cell (so the concept is in ADR too) — see O-01.
- **NFR-002 / NFR-020** (2 each: CHG + ADR) — covered; correctly eval-internal, out-of-scope for USR/API.

### 2b. Design Cases — every DC has ≥1 home → **0 orphans**

| DC | Title (abbrev) | Documenting deliverables | Orphan? |
|---|---|---|---|
| DC-01 | SharedLayerProjector recall-floor | JRN · CHG · ADR · API · USR · CON | no |
| DC-02 | MoE-router DistilledTopKGate + early-exit | JRN · CHG · ADR · API · CON | no |
| DC-03 | Aux-loss-free SLA selection-bias | JRN · CHG · ADR | no |
| DC-04 | Reversible pseudonymization + key rotation | JRN · ADR · API · USR | no |
| DC-05 | 6 transforms + legal-regime + orchestrate incumbents | CHG · ADR | no (thin by design) |
| DC-06 | RatingEnginePort + Registry (3-tier ladder) | JRN · CHG · ADR · CON | no |
| DC-07 | Coherent significance + Davidson ties | CHG · ADR | no |
| DC-08 | Distinct anon-vs-pseudo families | JRN · CHG · ADR · API · USR · CON | no |
| DC-09 | attacks/ Tier-3 + LiRA@128 MIA | JRN · CHG · ADR · API · CON | no |
| DC-10 | Calibration + selective-risk reporter | JRN · CHG · ADR | no |
| DC-11 | CanonicalRunGate + provenance + CI gate | JRN · CHG · ADR · API · USR | no |
| DC-12 | BYO-pipeline SDK + identical-incumbent scoring | JRN · CHG · ADR · API · USR · CON | no |
| DC-13 | Agentic interception (router pre-filter + 4-channel) | JRN · CHG · ADR · API · CON | no |
| DC-14 | Multimodal readers + per-modality benchmark + parity | JRN · CHG · ADR · API · USR · CON | no |
| DC-15 | Multilingual context + fairness gate + no-real-PII | JRN · CHG · ADR · API · USR | no |

**DC-05** (2 deliverables: CHG, ADR) is the least-elaborated DC — exactly as D1 §9 / D2 §5 predicted (1 source
file; FR-020/021/022 SHOULD + partially deferred). ADR §3 marks it PARTIAL and "least-elaborated (1 source file)"
explicitly. Covered with the required caveat → **OBSERVATION**, not an orphan or a gap.

### 2c. Orphan roll-up by ID family

| ID family | Total checked | ORPHANS | Documented |
|---|---|---|---|
| MUST FR | 23 | **0** | 23 |
| MUST NFR | 13 | **0** | 13 |
| DC | 15 | **0** | 15 |
| **Total** | **51** | **0** | **51** |

**Zero documentation orphans on MUST FR/NFR + DC — the DOCUMENTED bar is met.** This is the central
Axis-1 result and is the evidence D5 cites that no MUST is undocumented.

---

## 3. Axis 3 — Accuracy-against-code (api-reference D-5, plus contract examples in USR/CON)

**Method.** The api-reference claims 38 verification points + 2 declared GAPS. D4 independently
re-verified **>18 load-bearing symbols** against `src/pii_anon/` (grepping `class`/`def`/constant
declarations). **Every symbol the api-reference (D-5) documents exists in live code — D-5 itself has
NO absent-symbol finding.** The accuracy defects are in code examples in the *other* deliverables
(USR, CON) and in two D-5 semantic/signature inaccuracies. Code is ground truth; findings are against the doc.

### 3a. Independently re-verified present (sample, all PASS)

| # | Documented symbol | Live-code location | Result |
|---|---|---|---|
| 1 | `QueryAwareMaskingGate.decide` | `policy/query_aware.py:133/142` | PRESENT |
| 2 | `evaluate_incumbent` | `eval_framework/byo_pipeline.py:317` | PRESENT |
| 3 | `build_identical_path_leaderboard` | `byo_pipeline.py:348` | PRESENT |
| 4 | `INCUMBENT_SYSTEMS` / `incumbent_predictor` / `engine_predictor` / 5 named predictors | `byo_pipeline.py:257/263/178/288…` | PRESENT |
| 5 | `AnonymizationScorer` / `PseudonymizationIntegrityScorer` / `DeidFamilyScores` | `metrics/deid_families.py:172/238/154` | PRESENT |
| 6 | `evaluate_language_fairness` / `FairnessGateReport` / `LanguageGroupSlice` | `metrics/fairness_gate.py:79/65/51` | PRESENT |
| 7 | `NativeReaderRegistry.discover_entrypoint_readers` | `ingestion/native.py:108/144` | PRESENT |
| 8 | `PdfTextReader` + `.read` | `ingestion/native_pdf.py:190/211` | PRESENT (NOT at "lines 1–50" — see F-06) |
| 9 | `EncryptedSQLiteTokenStore` / `KeyEnvelope` / `StaticTestKeyProvider` / `EnvelopeKeyProvider` | `tokenization/encrypted_store.py:228/116/164/137` | PRESENT |
| 10 | `_g1_recall_floor` … `_g7_certified_run` (all 7) | `eval_framework/evaluation/competitive_supremacy.py:652…1774` | PRESENT |
| 11 | `_finite_unit_score` / `_is_finite_number` / `_is_nonblank_str` | `competitive_supremacy.py:795/772/1762` | PRESENT |
| 12 | `Verdict` / `GuaranteeResult` / `SupremacyVerdict` / `.from_artifacts` | `competitive_supremacy.py:180/229…` | PRESENT |
| 13 | SDO threshold constants (`J_BAR`, `EPS_F2`, `ENTITY_COVERAGE_MIN`, …) | `competitive_supremacy.py` const block | PRESENT |
| 14 | `produce_canonical_artifact` / `CanonicalRunGate` | `evaluation/canonical_run.py:1287/916` | PRESENT |
| 15 | `LatencyCeiling` / `COMMITTED_LATENCY_CEILINGS` / `ceiling_for` | `latency_ceilings.py:58/85/106` | PRESENT |
| 16 | `evaluate_external_system` / `ExternalEvaluationResult` / `Predictor` / `load_baseline_leaderboard` | `external_evaluator.py:158/69/65/414` | PRESENT |
| 17 | agentic: `FourChannelGuard`/`InterceptionLedger`/`AgentChannel`/`ChannelMasker`/`NoRawPIIPersistError`/`ChannelResult`/`InterceptionRecord` | `agentic/interception.py:344/298/99/153/88/144/114` | PRESENT |
| 18 | leakage: `LeakageSankey`/`SankeyEdge`/`build_leakage_sankey`/`score_injection_resistance`/`InjectionResistanceReport` | `agentic/leakage_sankey.py:122/106/205/419/280` | PRESENT |
| 19 | attacks: `RepresentativeTier3ReidAttack`/`assess_rrs_power`/`RRS_RUNG_REID_LOW`=385/`RRS_RUNG_REID_HIGH`=897/`RepresentativeMiaAttack`/`MIA_MIN_SHADOW_MODELS`=128/`MIA_FPR_TARGETS`=(1e-3,1e-2) | `attacks/reid_tier3.py` + `attacks/mia.py` | PRESENT (constants value-confirmed) |
| 20 | `SharedLayerProjector.project`/`ProjectionResult.violations_blocked`/`span_key_engine`/`is_shared_floor` | `routing/shared_layer.py:73/85/54/67/36/48` | PRESENT |
| 21 | 19 CLI commands incl. `supremacy`, `canonical-run`, `benchmark-publish-suite`, `compare-competitors` | `cli.py` `@app.command(...)` | PRESENT (all 19 match USR §9 census exactly) |
| 22 | 4 entry-point groups (`engines`/`rating_engines`/`byo_pipelines`/`readers`) | `pyproject.toml:61/70/78/87` | PRESENT (exact lines D2 cited) |

**The api-reference's own symbol inventory is accurate** — no documented public symbol in D-5 is absent from code.

### 3b. Accuracy DEFECTS (findings — code is ground truth, finding is against the doc)

- **★ F-01 (CATASTROPHIC) — `contributor-handbook.md §2b`: fabricated `RatingEnginePort` contract.**
  The handbook documents the rating-engine plugin contract as three methods — `fit(...)`, `ratings()`,
  `rank_one_probability(...)`. The **live `RatingEnginePort` Protocol** (`eval_framework/rating/port.py:26`)
  declares **only two**: `run_round_robin(composites: dict[str, float]) -> list[RatingUpdate]` and
  `get_rating(name: str) -> EloRating | None`. The docstring is explicit ("Only the two methods the four
  production callers use are part of the contract"). `grep "def fit"/"def ratings"` over `rating/` returns
  **nothing** — these methods are absent from the entire rating surface. A third-party engine built to the
  handbook's documented contract **fails `@runtime_checkable isinstance(x, RatingEnginePort)`** and is not a
  valid plugin. This is a **documented public contract absent from code** in the deliverable whose purpose is
  to tell contributors how to implement it — the canonical CATASTROPHIC accuracy trigger. Aggravating: the
  handbook's own `## Sources` row asserts `rating/port.py (verified: RatingEnginePort)` — the "verified" claim
  is false for the method surface. **Loopback target: re-author CON §2b against `rating/port.py:26` —
  the contract is `run_round_robin` + `get_rating`, NOT `fit`/`ratings`/`rank_one_probability`.**

- **F-02 (MAJOR) — `contributor-handbook.md §2d`: fabricated `ReaderCapabilities` constructor + wrong
  `capabilities` shape.** The handbook's `NativeReader` plugin example constructs
  `ReaderCapabilities(formats={...}, languages={...}, extraction_fidelity=0.85)`. The **live**
  `ReaderCapabilities` (`ingestion/native.py:61`) has fields `format_name`, `native_dependency`,
  `dependency_available`, `extracts_text`, `supports_reconstruction`, `notes` — **none** of `formats`,
  `languages`, `extraction_fidelity` exist; the constructor raises `TypeError`. The handbook also decorates
  `capabilities` as `@property`, but the live `NativeReader` Protocol declares it as a **method**
  (`def capabilities(self) -> ReaderCapabilities`). A reader built to this example does not satisfy the
  Protocol. (api-reference D-5 documents `ReaderCapabilities` and `NativeReader.capabilities()`
  **correctly** — CON contradicts the correct D-5 and the code.) **Loopback target: re-author CON §2d
  reader contract against `ingestion/native.py:61–92`.**

- **F-03 (MAJOR) — `user-operator-guide.md §8b`: non-runnable `EncryptedSQLiteTokenStore` example +
  absent `KeyEnvelope.from_env`.** USR §8b shows:
  `EncryptedSQLiteTokenStore(db_path="tokens.db", key_envelope=KeyEnvelope.from_env("PII_ANON_TOKEN_KEY"))`.
  Against code (`tokenization/encrypted_store.py:228`): the constructor takes a **required keyword-only**
  `key_provider: EnvelopeKeyProvider`, NOT `key_envelope=` → `TypeError`. And **`KeyEnvelope.from_env` does
  not exist** — `KeyEnvelope` is a frozen dataclass (`key_id`, `wrapped_dek`, `created_at`) with no
  classmethods (`grep from_env` → none) → `AttributeError`. The example is not copy-paste runnable and cites
  an **absent classmethod on a public type**. (api-reference D-5 documents the same constructor **correctly**
  as `key_provider: EnvelopeKeyProvider` → USR contradicts the correct D-5 and the code.) **Loopback target:
  re-author USR §8b to construct with `key_provider=StaticTestKeyProvider(...)` or a real `EnvelopeKeyProvider`;
  drop `KeyEnvelope.from_env`.**

- **F-04 (MINOR) — `user-operator-guide.md §6b`: `reader_capabilities()` iterated as a dict.** USR §6b does
  `caps = reader_capabilities(); for name, cap in caps.items(): ...`. Live `reader_capabilities()`
  (`ingestion/native.py`) returns **`list[ReaderCapabilities]`** (sorted by name), which has no `.items()`
  → `AttributeError`; the unpack `for name, cap in <list>` also fails. (D-5 documents the `list` return
  **correctly**.) **Loopback target: iterate the list:
  `for cap in reader_capabilities(): print(cap.format_name, cap.dependency_available)`.**

- **F-05 (MINOR) — `api-reference.md §10a`: `CanonicalRunGate.validate` signature + behavior inverted.**
  D-5 documents `def validate(self, artifact: dict[str, Any]) -> None: ...` with a comment "Raises on
  non-dict payload…". The **live** method (`evaluation/canonical_run.py:928`) is
  `def validate(self, payload: dict[str, Any]) -> tuple[bool, list[str]]:` whose docstring states **"Fail
  CLOSED, never raise"** (returns `(ok, missing)`). The documented contract is inverted (raises vs
  never-raises) and the return type (`None` vs `tuple[bool, list[str]]`) and param name (`artifact` vs
  `payload`) are wrong. The symbol EXISTS and is correctly located → MINOR, not CATASTROPHIC, but it
  misstates a control-path gate's contract. **Loopback target: correct D-5 §10a to the real signature/return.**

- **F-06 (MINOR) — `api-reference.md` Methodology: stale/incorrect line-number citations.** Several
  "verified at line N" claims do not match current source: `PdfTextReader` is at `native_pdf.py:190` (doc
  says "lines 1–50"; the author's GAPS note even concedes only lines 1–50 were read — so the *class* was
  NOT in the read window, making the "class declaration verified" line self-contradictory);
  `StaticTestKeyProvider` is at `encrypted_store.py:164` (doc says 180); `EncryptedSQLiteTokenStore.__init__`
  is documented at `:252` but the class is at `:228`. Symbols all exist; only the line anchors drift → MINOR
  (auditor-traceability hygiene). **Loopback target: refresh the line anchors or drop them.**

### 3c. Honest-gap acknowledgements (no finding — these are correctly disclosed)

- D-5 declares 2 GAPS: `produce_canonical_artifact` full kwargs not verified to keyword level, and the
  agentic submodules verified via `__all__` only. D4 independently confirmed both symbols **exist**
  (`produce_canonical_artifact` at `canonical_run.py:1287`; agentic `__all__` names all resolve to real
  `class`/`def` declarations). The disclosed gaps are honest and do not affect the present-in-code verdict.
- D-5 correctly marks CODE-local vs DATA-track on BYO / attacks / fairness surfaces (no integrator will
  assume a DATA-track dependency ships in-repo).

---

## 4. Brownfield framing (for D5's verdict)

This is a **mature-brownfield** corpus with real authored signal, so — unlike a thin brownfield input —
coverage is **not** the gap: zero MUST/DC orphans, dense multi-deliverable coverage, the thin-trace
watch-list (FR-033/035/036, NFR-008, DC-05) each explicitly called out with the required caveats. The
accuracy-against-code axis (the most reliable ground truth) is where the real defects surfaced — exactly
where D2 said to lean hardest. The single CATASTROPHIC (F-01) is a **fabricated plugin contract**, not a
coverage hole, and it is the hard-failure class that D2/D5 reserve the loopback for. The MAJOR/MINOR
example bugs (F-02/F-03/F-04) cluster in copy-paste code blocks where the canonical D-5 reference is
*correct* and the consuming deliverable drifted — a re-author against D-5 + code resolves them.

---

## 5. Findings index (this file)

| ID | Severity | Location | One-line |
|---|---|---|---|
| F-01 | **CATASTROPHIC** | CON §2b | `RatingEnginePort` contract fabricated (`fit`/`ratings`/`rank_one_probability`); real port = `run_round_robin` + `get_rating` (`rating/port.py:26`) — plugin built to doc fails isinstance |
| F-02 | MAJOR | CON §2d | `ReaderCapabilities(formats=/languages=/extraction_fidelity=)` fabricated; real fields `format_name`/`native_dependency`/…; `capabilities` is a method not `@property` |
| F-03 | MAJOR | USR §8b | `EncryptedSQLiteTokenStore(key_envelope=KeyEnvelope.from_env(...))` non-runnable; ctor needs `key_provider`; `KeyEnvelope.from_env` absent |
| F-04 | MINOR | USR §6b | `reader_capabilities()` iterated as `.items()`; it returns `list[ReaderCapabilities]` |
| F-05 | MINOR | API §10a | `CanonicalRunGate.validate` documented `-> None`/"raises"; real `-> tuple[bool, list[str]]`/"never raise" |
| F-06 | MINOR | API Methodology | stale line-number citations (PdfTextReader, StaticTestKeyProvider, EncryptedSQLiteTokenStore) |
| O-01 | OBSERVATION | NFR-023 | MUST parity NFR covered as a string token only in USR §8f (thinnest MUST-NFR row) — covered, not orphan |
| O-02 | OBSERVATION | FR-017 / DC-05 / FR-004 | thin-but-covered (2–3 deliverables); DC-05 PARTIAL caveat correctly carried |

> Coverage roll-up here; trace/link integrity + epistemic-honesty + source-mapping conformance findings
> (incl. CON's MEMORY.md out-of-mapping citation, the likely root of F-01) are in
> `doc-trace-integrity.md`. A11y disposition (N/A-WITH-REASON) is in `doc-trace-integrity.md §4`.

---

## Retry-1 re-verification (D4 addendum)

> **Scope:** confirm the 8 loopback fixes from the round-0 D4 audit are GENUINELY closed
> against live source (each deliverable section read AND the cited source file read).
> Re-verified 2026-06-10. Read-only except this append.

### Per-finding verdict

| # | Finding | Verdict | Evidence (deliverable §  ↔  live source) |
|---|---|---|---|
| 1 | **F-01** (was CATASTROPHIC) — CON §2b fabricated `RatingEnginePort` | **CLOSED** | CON §2b now documents EXACTLY `run_round_robin(self, composites: dict[str, float]) -> list[RatingUpdate]` + `get_rating(self, name: str) -> EloRating \| None`, matching `rating/port.py:37,41`. `grep "def fit"/"ratings()"/"rank_one_probability"` over CON → NONE. (Those names DO exist on concrete `bayes_bt.py`/`bradley_terry.py`/`significance.py` — but NOT on the Protocol, and CON no longer cites them.) The richer-API exclusion list matches `port.py:10-13` verbatim. Sources row no longer makes a false "verified" claim — it cites the real 2-method contract. |
| 2 | **F-02** (was MAJOR) — CON §2d fabricated `ReaderCapabilities` fields | **CLOSED** | CON §2d documents the six REAL fields `format_name / native_dependency / dependency_available / extracts_text / supports_reconstruction / notes` (matches `native.py:71-76`), `capabilities()` as a METHOD, and `read(self, path, config) -> Iterator[IngestRecord]` (matches `native.py:85-92`). `grep "formats="/"languages="/"extraction_fidelity="/"@property"` over CON → NONE. |
| 3 | **T-03** — MEMORY.md citations in CON | **CLOSED** | `grep -i "MEMORY.md\|auto-memory\|.claude/projects"` over CON → NONE. |
| 4 | **F-03** (was MAJOR) — USR §8b non-runnable `EncryptedSQLiteTokenStore` + absent `KeyEnvelope.from_env` | **CLOSED** | USR §8b example now constructs `EncryptedSQLiteTokenStore(db_path="tokens.db", key_provider=provider, algorithm="aesgcm")` with `provider = StaticTestKeyProvider(test_kek=kek, key_id="my-kek-v1")` — a REAL provider class — matching `encrypted_store.py:252` (`db_path, *, key_provider, algorithm="aesgcm"`). `grep "from_env\|key_envelope"` over USR returns ONLY Methodology-prose mentions documenting the fix (lines 818/819/836); the §8b code block carries no `KeyEnvelope.from_env`. |
| 5 | **F-04** (was MINOR) — USR §6b `reader_capabilities()` iterated as `.items()` | **CLOSED** | USR §6b code block (lines 460-466) now does `caps = reader_capabilities()` then `for cap in caps: print(cap.format_name, ...)` — list iteration, matching `native.py:360` (`-> list[ReaderCapabilities]`). The only `.items()` hit in USR is line 839 (Methodology prose describing the fix), NOT the code example. |
| 6 | **T-01** (was MINOR) — release-ops sourced from `docs/release-guide.md` | **CLOSED** | USR §8g "Release and publish operations" present; Methodology + Sources both cite `docs/release-guide.md` (§§1–8, "read retry-1"). The prior "subsection omitted" gap note is resolved. |
| 7 | **F-05** (was MINOR) — API §10a `CanonicalRunGate.validate` signature inverted | **CLOSED** | API §10a now documents `def validate(self, payload: dict[str, Any]) -> tuple[bool, list[str]]` with "fail-closed boolean + reasons list; never raises", matching the live def at `canonical_run.py:928` and the tail return `(not missing), missing` at `canonical_run.py:1159` (read in full — the method has NO raise path; non-dict/malformed inputs return `(False, [...])`). Param name `payload` and the `(ok, missing)` return are both correct. |
| 8 | **F-06** (was MINOR) — stale line-number citations in API Methodology | **CLOSED** | Spot-check (`EncryptedSQLiteTokenStore.__init__`): API now cites class line **228**; live `class EncryptedSQLiteTokenStore` is at `encrypted_store.py:228` (the `__init__` def at :252, the class header at :228 — the doc cites the class line, consistent with its file:symbol convention). Cross-checked the other two: `PdfTextReader` cited 190 ↔ live `native_pdf.py:190`; `StaticTestKeyProvider` cited 164 ↔ live `encrypted_store.py:164`. All three anchors current. |

**Methodology loopback notes present:** CON carries `**D4 LOOPBACK NOTE (retry-1, 2026-06-10)**` (line 541); USR carries `**Retry-1 corrections:**` (line 835, naming F-03/F-04/T-01); API carries `**Retry-1 corrections (D3 LOOPBACK):**` (line 983, naming F-05/F-06). All three fixed files document the loopback in their `## Methodology`.

### Fabrication sweep (3 fixed files)

Swept the three fixed files for any OTHER `ClassName(method-or-kwarg` pattern carried with a "verified" claim. Result: **CLEAN — no new fabrication.** Every "verified" assertion checked resolves to live code. 4 randomly-selected spot-checks against source:
1. CON §2c `BYOPipelineRegistry.get` / `.discover_entrypoint_pipelines` → PRESENT (`byo_pipeline.py:92/130/140`).
2. USR §8b `StaticTestKeyProvider(test_kek=..., key_id=...)` → matches `encrypted_store.py:180`.
3. API §6 `EncryptedSQLiteTokenStore.rotate` / `.rewrap_all` / `.list_key_envelopes` / `.list_key_ids` / `.row_key_id` → ALL PRESENT (`encrypted_store.py:731/744/779/771/790`).
4. CON §2a `EngineAdapter` + mandatory `detect()` → PRESENT (`engines/base.py:35/205`).

### Retry-1 disposition (Axis 3 — accuracy-against-code)

**All 8 findings CLOSED. Fabrication sweep CLEAN. Axis-3 disposition: PASS.** No SHOWSTOPPER/CATASTROPHIC remains — the round-0 CATASTROPHIC (F-01) and both MAJORs (F-02/F-03) are genuinely closed against live source, so **no further D3 loopback is warranted** (round-0 was loop iteration 1 of the cap-3 budget; this retry-1 confirmation closes it). Axis-1 coverage was already PASS (0 orphans) and is unaffected by these accuracy fixes.
