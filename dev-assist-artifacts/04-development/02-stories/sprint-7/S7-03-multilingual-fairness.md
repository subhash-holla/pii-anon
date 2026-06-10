# S7-03 — multilingual context activation (CJK/Hangul/Arabic) + powered worst-group fairness gate

| Field | Value |
|---|---|
| Story | S7-03 |
| Sprint | 7 |
| State | **DONE** (2026-06-09; SO-22. 5/5 first-pass story gate APPROVE — security-sast ZERO findings (the fail-closed power semantics + the monotonic-additive containment pass independently verified); code-quality 1 MINOR (Literal verdict type) remediated in-loop `0e35431`; axiom-compliance 5 upheld-with-evidence OBS incl. the AX-003 IBAN inverted-polarity proof. NO SDO close (gate + canonical_run byte-identical). See §Evidence.) |
| provisional_status | **AGENT_SIMULATED** — the CJK/Hangul/Arabic context-keyword activation runs for REAL in detection (the keywords already shipped in `CONTEXT_WORDS`; S7-03 makes them able to fire) and the fairness gate computes for REAL on synthetic per-language fixtures (AX-001). What stays Pass-2: the corpus-scale fairness number over the 60-language eval set (`# SWITCH-POINT(DATA)`; NFR-004 powered slices need the eval-data corpus), and wiring the gate verdict into `canonical_run.py` — that is a control-path-artifact change which would MANDATE the adversarial SDO close, explicitly deferred (`# SWITCH-POINT(CANONICAL)`). |
| Size | S |
| Implements | **FR-038** (multilingual non-EN context feature active in detection; UC-28, swarm, SHOULD) + **FR-039** (per-language fairness gap bounded + gated; UC-28, both, SHOULD) + **NFR-025** (multilingual worst-group fairness — worst-group recall gap ≤ 0.10 across POWERED language groups; SHOULD). Upholds **NFR-004** (power tiers define "powered" — long-tail floor 200 gold positives as the default), **NFR-024/AX-001** (synthetic fixtures), **AX-002** (pure deterministic gate). |
| Traces | Design **DC-15** (`D-implementation-ready-design.md:25` — "Multilingual context + fairness gate + no-real-PII + optional-dep degradation (cross-cutting)"). UC-28. CONSUMES read-only: `eval_framework/metrics/base.py` `_aligned_prf` (the same primitive S1-03 and `fairness_metrics.py` use), `eval_framework/languages.py` (`LanguageProfile` metadata), S1-03's per-language recall-floor gate test (consistency pin), S1-05's swarm language propagation (the language plumbing that feeds per-language slices). |
| Files owned | **additive** `src/pii_anon/engines/regex/confidence.py` (the FR-038 containment pass — see Approach (a)), `src/pii_anon/eval_framework/metrics/fairness_gate.py` (**new**), **additive** `src/pii_anon/eval_framework/metrics/__init__.py` re-exports (if the package re-exports metrics there), `tests/test_multilingual_fairness.py` (**new**). |
| Depends on | S1-03 (per-language recall floor — DONE) + S1-05 (language propagation — DONE). Order-independent with S7-01 (disjoint modules); sequenced after it per SO-19. |

## 1. Intent
**FR-038 (the demonstrable RED):** `CONTEXT_WORDS` has carried ZH/JA/KO/AR context keywords (e.g. PHONE_NUMBER: "电话", "電話", "전화", "هاتف"; EMAIL_ADDRESS: "邮箱", "メール", "이메일", "بريد") since the multilingual expansion — but they can NEVER fire: `has_context_words` intersects the keyword set against `_WORD_RE = [A-Za-zÀ-ÿ]+` tokens, and that tokenizer cannot produce a CJK/Hangul/Arabic token. The comment above `_WORD_RE` even *claims* "keyword matches there rely on substring containment" — but no containment pass exists. So the multilingual context feature is INACTIVE for every non-Latin script: a Chinese phone-context sentence ("请拨打电话…") gets no confidence boost that the equivalent English sentence gets — precisely the per-language quality gap NFR-025 is about. S7-03 activates it: keywords with **no Latin-alphabet runs** match by lowercase substring containment; every Latin/ASCII keyword keeps the token path **byte-identical** (regression-pinned). **FR-039/NFR-025 (the gate):** ships `evaluate_language_fairness` — a pure primitive computing per-language recall over labelled spans (the same `_aligned_prf` S1-03 uses), filtering to POWERED groups (gold positives ≥ NFR-004 long-tail floor 200 by default; test-scale floors injectable), computing `worst_group_recall_gap = max − min`, and returning a fail-closed verdict: `PASS` (gap ≤ threshold), `FAIL` (gap > threshold, violators named), or `INSUFFICIENT_POWER` (zero/one powered group — NEVER a PASS without evidence). Unpowered groups are reported observationally with `LanguageProfile` metadata, never silently dropped.

## 2. Approach / scope — the carried DESIGN decisions

### (a) FR-038 — the containment pass (additive, `confidence.py`)
* Module-level precomputed `_NON_LATIN_CONTEXT_KW: dict[str, tuple[str, ...]]` — for each entity type, the (sorted) keywords containing **zero** `_WORD_RE` runs (i.e. scripts the tokenizer cannot produce: CJK/Hangul/Arabic). Predicate `not _WORD_RE.search(kw)` — this selects EXACTLY the non-Latin keywords; "teléfono" (é ∈ À-ÿ) stays token-path; dead-but-Latin aliases like "e-mail"/"ss#" are deliberately NOT revived (out of scope — ASCII behavior byte-identical).
* `has_context_words`: after the existing token-set intersection misses, scan the entity's non-Latin keywords for lowercase substring containment in the (already-lowercased) context window. Deterministic, allocation-light (precomputed tuples — the S6-01 hoist lesson), pure.
* Document: the precomputed map snapshots `CONTEXT_WORDS` at import; runtime mutation of `CONTEXT_WORDS` (not a supported surface) won't refresh it.

### (b) FR-039/NFR-025 — the fairness gate (`metrics/fairness_gate.py`, new, pure)
* **`LanguageGroupSlice`** (frozen): `{language, gold: list[LabeledSpan], predicted: list[LabeledSpan]}` — or equivalently the caller passes per-language `(gold, predicted)` span collections; recall computed via the SAME `_aligned_prf` primitive S1-03's gate and `fairness_metrics.py` use (consistency pin A10).
* **`FairnessGateReport`** (frozen): `{verdict: "PASS"|"FAIL"|"INSUFFICIENT_POWER", worst_group_recall_gap: float | None, gap_threshold, power_floor, per_language_recall: dict, powered_groups: list, unpowered_groups: list, violating_groups: list, n_powered}`.
* **`evaluate_language_fairness(slices, *, gap_threshold=0.10, power_floor=200, match_mode=strict) -> FairnessGateReport`**:
  - per-language recall over gold/predicted spans; a group is POWERED iff its gold-positive count ≥ `power_floor` (NFR-004: 1522/753/200 tiers — long-tail 200 is the default; tests inject small floors to exercise the logic at fixture scale).
  - `worst_group_recall_gap = max(powered recalls) − min(powered recalls)`; `PASS` iff gap ≤ threshold (≤ semantics pinned at the boundary); violators = powered groups at distance > threshold below the max.
  - **Fail-closed:** 0 powered groups ⇒ `INSUFFICIENT_POWER` (gap None, never PASS); 1 powered group ⇒ `INSUFFICIENT_POWER` (a gap needs ≥2 groups). Domain-named `ValueError` on corrupt input (negative counts, gap_threshold outside [0,1], power_floor < 1 — the S5-02/03 fail-loud discipline).
  - Unpowered groups carried observationally (`unpowered_groups` with their recalls + n) — visible, not gate-driving.
* **Seams:** `# SWITCH-POINT(CANONICAL)` — emitting this verdict from `canonical_run.py` makes it a control-path artifact ⇒ MANDATES the adversarial SDO close; deferred Pass-2. `# SWITCH-POINT(DATA)` — the corpus-scale run over the 60-language eval set with NFR-004 full tiers.

## 2a. Pre-claim de-risk
- **RISK-1 (the ASCII regression):** the containment pass must not change ANY Latin-keyword behavior — A3 pins exact prior True/False cases (incl. "teléfono" token-path, "e-mail" still dead); the full ingestion/engine suites guard.
- **RISK-2 (fail-closed gate):** INSUFFICIENT_POWER on 0 AND 1 powered groups (A9); a PASS requires ≥2 powered groups with gap ≤ threshold; boundary `==` pinned (A7).
- **RISK-3 (consistency with S1-03):** the gate's per-language recalls must equal the S1-03 helper's on a shared fixture (A10) — one recall definition program-wide.
- **RISK-4 (no canonical_run touch):** `canonical_run.py` + SDO gate byte-identical ⇒ no SDO close; the wire-in is explicitly Pass-2.
- **RISK-5 (AX-001/AX-002):** synthetic multilingual fixtures only; gate + containment pure/deterministic (A5).
- **RISK-6 (exact anchors):** all rates integer-count-derived, anchored exactly (the recurring lesson).

## 3. Given / When / Then (acceptance)
- **A1 — ZH context fires (FR-038 RED→GREEN) `[UNIT-TEST]`.** `has_context_words("PHONE_NUMBER", "请拨打电话联系")` is True (provably False pre-fix); `adjust_confidence` returns exactly `base + CONTEXT_BOOST` (capped) for a ZH phone-context window.
- **A2 — JA/KO/AR keywords fire `[UNIT-TEST]`.** "メール" (EMAIL_ADDRESS, JA), "전화" (PHONE_NUMBER, KO), "هاتف" (PHONE_NUMBER, AR), "信用卡" (CREDIT_CARD, ZH) each containment-match in a synthetic sentence — exact booleans.
- **A3 — ASCII/Latin behavior byte-identical `[UNIT-TEST]`.** Exact prior-behavior pins: "call me" True / "nothing here" False (PHONE_NUMBER); "teléfono" True via the TOKEN path (and absent from `_NON_LATIN_CONTEXT_KW`); "e-mail" still dead (the literal "e-mail" alone without "mail" as a token — construct via punctuation-free check on the keyword's reachability, e.g. assert "e-mail" ∉ tokens ∧ containment list excludes it); unknown entity type False.
- **A4 — the non-Latin keyword partition is exact `[UNIT-TEST]`.** `_NON_LATIN_CONTEXT_KW["PHONE_NUMBER"]` == the exact sorted tuple of the ZH/JA/KO/AR phone keywords (and contains NO Latin-run keyword for any entity type — property over the whole map).
- **A5 — determinism (AX-002) `[UNIT-TEST]`.** 5 identical `has_context_words`/`adjust_confidence`/gate replays byte-identical.
- **A6 — gate FAIL exact `[UNIT-TEST]`.** en 4/4, es 3/4, zh 2/2 with `power_floor=2` ⇒ powered = all three, gap exactly `0.25`, verdict FAIL, `violating_groups == ["es"]`.
- **A7 — PASS boundary at the threshold `[UNIT-TEST]`.** Recalls 1.0 and 0.9 (integer-count fixtures), `gap_threshold=0.10` ⇒ gap exactly `0.1` (float-exact via construction) ⇒ PASS (≤ semantics); one epsilon worse ⇒ FAIL.
- **A8 — power filter exact `[UNIT-TEST]`.** A 1-gold-positive language is EXCLUDED from the gap and appears in `unpowered_groups` (with its recall reported observationally); the gap computes over the remaining powered groups only.
- **A9 — fail-closed `[SECURITY-TEST]`.** 0 powered groups ⇒ INSUFFICIENT_POWER (never PASS); exactly 1 powered group ⇒ INSUFFICIENT_POWER; corrupt input (power_floor=0, gap_threshold=1.5) ⇒ domain-named ValueError.
- **A10 — consistency with S1-03 `[CONTRACT-TEST]`.** On a shared synthetic fixture, the gate's per-language recalls equal the `_aligned_prf`-derived recalls the S1-03 gate computes — exact equality.
- **A11 — import-boundary audit `[AUDIT]`.** `fairness_gate.py` imports nothing from `swarm`/`moe`/`fusion`/`orchestrator`; `canonical_run.py` + SDO gate + `competitor_compare.py` + `orchestrator.py` byte-identical.

## 5. Notes / non-goals
- **Non-goal:** wiring the fairness verdict into `canonical_run.py` / the SDO gate — a control-path-artifact change MANDATING the adversarial close; `# SWITCH-POINT(CANONICAL)` Pass-2.
- **Non-goal:** the corpus-scale fairness number over the 60-language eval set at NFR-004 full power tiers — `# SWITCH-POINT(DATA)` (eval-data).
- **Non-goal:** reviving dead Latin aliases ("e-mail", "ss#") — a separate hygiene item; out of FR-038 scope (ASCII behavior byte-identical is the regression contract).
- **Non-goal:** non-Latin tokenization for OTHER consumers of `_WORD_RE` — only `has_context_words` gains the containment pass.

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[SECURITY-TEST]` `[CONTRACT-TEST]` `[AUDIT]`. **Reviewers (canonical 5-gate story set):** code-quality + traceability (DC-15 → FR-038/039/NFR-025 + UC-28) + requirements-coverage (the representative-vs-corpus split + the CANONICAL deferral tracked) + axiom-compliance (AX-001/002 + the fail-closed gate + no-canonical-touch) + security-sast ([AUDIT]/[SECURITY-TEST]; fail-closed power semantics — an unpowered PASS would be a fabricated fairness claim). **No SDO adversarial close** (no control-path change — explicitly verified at close).

## 12. Definition of Done
- [ ] **RED**: `tests/test_multilingual_fairness.py` (A1–A11) first & failing (A1/A2 fail on the inactive keywords; gate tests fail on `ModuleNotFoundError`). RED precedes GREEN.
- [ ] **GREEN**: the containment pass + `fairness_gate.py` — all A1–A11 green.
- [ ] **Quality gate**: full xdist suite green (the ASCII-regression guard); ruff clean; mypy clean (both modes); coverage ≥84% (new module ≥90%).
- [ ] **Untouched / off-limits**: `orchestrator.py` + `competitive_supremacy.py` + `competitor_compare.py` + `canonical_run.py` byte-identical; user-WIP never staged; narrow `git add`.
- [ ] **Story-gate APPROVE** — `_reviews/story/S7-03/`, all 5 APPROVE.
- [ ] **SDO verdict UNCHANGED** — a fairness-gate primitive flips no guarantee (it is not wired into the canonical artifact).

## Evidence (filled on completion)

**Commits (RED→GREEN→remediation, on `pdlc/sota-program`):** RED `4209ed6` (tests-only; ImportError on `_NON_LATIN_CONTEXT_KW` + ModuleNotFoundError on `fairness_gate`) → GREEN `e8cc897` (the containment map + fallthrough; `fairness_gate.py`) → remediation `0e35431` (the Literal verdict type).

**Files:** additive `src/pii_anon/engines/regex/confidence.py` (`_NON_LATIN_CONTEXT_KW` import-time map + the containment fallthrough — Latin keywords token-path-exclusive, ASCII behavior byte-identical), `src/pii_anon/eval_framework/metrics/fairness_gate.py` (new — `LanguageGroupSlice`, `FairnessGateReport` with `Literal` verdict, `evaluate_language_fairness`; 100% module coverage), `tests/test_multilingual_fairness.py` (A1–A11).

**Acceptance → tests (A1–A11, all green):** A1 ZH activation + exact `base+CONTEXT_BOOST`; A2 JA/KO/AR/ZH keywords fire + non-matching CJK stays False; A3 Latin token-path pins (call/teléfono/underscored/unknown-type); A4 exact partition (the map == keywords with zero Latin runs; both-direction completeness; dead Latin aliases NOT revived); A5 determinism ×5; A6 FAIL exact (gap 0.25, violators `["es"]`); A7 dyadic boundary (1.0 vs 7/8 @ 0.125 → PASS inclusive; 6/8 → FAIL); A8 power filter (unpowered observational); A9 fail-closed (0 AND 1 powered → INSUFFICIENT_POWER; corrupt-input ValueErrors); A10 [CONTRACT] gate recalls == `_aligned_prf` STRICT recalls; A11 import-boundary audit.

**Story gate (5/5 first-pass APPROVE; `_reviews/story/S7-03/`; run `wf_bb94aa83-837`):** 0 MAJOR / 1 MINOR (Literal type — remediated `0e35431`) / 9 OBSERVATION. security-sast ZERO findings. axiom-compliance highlights: AX-003 upheld BY CONSTRUCTION (the containment pass runs only after the token miss, flips False→True never True→False; the one inverted-polarity caller — `regex_adapter.py:674` IBAN/SWIFT suppression — is provably unaffected since IBAN carries only Latin keywords); the fail-closed gate upholds the no-fabrication spirit (an unpowered cohort can never manufacture a PASS). Doc-hygiene OBS (stale S1 dev-log cell) remediated at close. The scribe wrote synthesis.md; per-reviewer YAMLs transcribed.

**SDO — UNCHANGED:** `canonical_run.py` `d8f0f80e…` + `competitive_supremacy.py` `3b842e81…` + `competitor_compare.py` `7cae16c8…` + `orchestrator.py` `0afc6dee…` byte-identical; the gate writes no artifacts; the `# SWITCH-POINT(CANONICAL)` wire-in stays Pass-2 (it would mandate the adversarial close).

**Quality:** owned tests 11/11; `fairness_gate.py` 100%; full xdist suite EXIT=0 @ 88.87% (pre-remediation; the remediation is annotation-only, re-verified by owned tests + BOTH-mypy); ruff clean; mypy clean BOTH modes (144 files); confidence neighbor suites green (95 tests).

**DoD:** all checkboxes met. Pass-2 (tracked): the corpus-scale fairness number (`# SWITCH-POINT(DATA)`); the canonical wire-in (`# SWITCH-POINT(CANONICAL)`, SDO-close-gated); the NFR-025 matrix annotation (epic/sprint snapshot).
