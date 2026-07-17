# S5-03 — representative membership-inference adversary (LiRA-shape) + Secret-Sharer + TPR@low-FPR

| Field | Value |
|---|---|
| Story | S5-03 |
| Sprint | 5 |
| State | **DONE** (2026-06-09; SO-18. 5/5-APPROVE story gate — security-sast (PRIMARY) + axiom-compliance + traceability + requirements-coverage + code-quality; 1 MAJOR (provisional_status field) + substantive MINORs remediated/dispositioned in-loop. NO SDO close (no `competitive_supremacy.py` change — gate md5 `3b842e81…` byte-identical). See §Evidence.) |
| provisional_status | **AGENT_SIMULATED** — the `RepresentativeMiaAttack`, the `tpr_at_fpr`/`score_mia_attack` ROC machinery, `canary_exposure`, the ≥128-shadow power assertion, the sandbox run, the de-circularization, and the import/AST guards all run for REAL in-tree against SYNTHETIC records (AX-001). The real **LiRA@128** shadow-model training + the real **Secret-Sharer** canary in/out splits are DATA-absent (`pii_anon_datasets` S6) → Pass-2 / cross-repo (`# SWITCH-POINT(DATA)`). Mirrors the source FR-013/NFR-013/UC-10 AGENT_SIMULATED rows (`traceability-matrix.md`). |
| Size | M |
| Implements | **FR-013** (full-power MIA: LiRA@128 + Secret-Sharer, TPR@low-FPR — the in-tree REPRESENTATIVE stand-in; the real ≥128-shadow-model LiRA training + DATA canary splits are Pass-2) + **NFR-013** (MIA power: ≥128 shadow models + Secret-Sharer; report TPR@FPR∈{1e-3,1e-2}). Upholds **AX-001** (synthetic-only), **AX-002** (deterministic membership scoring + pure ROC/exposure math), **NFR-016** (the non-strippable anti-anonymity caveat — REUSED from S5-01, carried on every emitted MIA report). |
| Traces | Design **DC-09** (`D-implementation-ready-design.md:19` — "`attacks/` package: real Tier-3 LLM-adversary (de-circularized) + LiRA@128 MIA"). UC-10. The S5-01 `MiaAttack` Protocol seam (`membership_scores(records) -> list[float]`) + the `ANTI_ANONYMITY_CAVEAT` — **consumed, not changed.** The S5-04 sandbox substrate (`run_attack_under_sandbox`) — consumed, not changed. The S5-02 `wilson_interval` (NFR-012 power CI) — REUSED for the MIA power CI. |
| Files owned | `src/pii_anon/eval_framework/attacks/mia.py` (**new** — `MiaRecord`, `RepresentativeMiaAttack`, `tpr_at_fpr`/`MiaSuccessReport`/`score_mia_attack`, `canary_exposure`/`SecretSharerReport`, `MiaPowerReport`/`assess_mia_power`, `mia_attack_runner` + `MIA_ATTACK_REGISTRY`). **Additive** to `src/pii_anon/eval_framework/attacks/__init__.py` (re-exports + merge `MIA_ATTACK_REGISTRY`). `tests/test_attack_mia.py` (**new**). |
| Depends on | **S5-01 DONE** (the `MiaAttack` Protocol seam + the non-strippable caveat). **S5-04 DONE** (the sandbox substrate). **S5-02 DONE** (the `wilson_interval` power-CI primitive — REUSED). The DATA sibling (the real ≥128 shadow-model LiRA + the canary in/out splits) is **VERIFIED ABSENT** → the representative adversary + the TPR@low-FPR/exposure/power-assertion machinery ship now; the real shadow training + canary splits are `# SWITCH-POINT(DATA)` Pass-2. CONSUMES read-only: `sandbox.py`/`spec.py`/`reid.py` (byte-identical). The existing `privacy_metrics.py::MembershipInferenceMetric` (a different framework — surface n-gram MIA) is NOT touched. |

## 1. Intent
Membership inference is the second half of the DC-09 attack surface (UC-10; the privacy-quadrant MUST alongside the S5-01/S5-02 re-identification family). S5-03 adds a **representative membership-inference adversary** conforming to the S5-01 `MiaAttack` Protocol, plus the **NFR-013 reporting machinery**: the LiRA-convention **TPR@low-FPR** (true-positive rate at FPR∈{1e-3,1e-2}), the **Secret-Sharer canary-exposure** metric, and a **≥128-shadow-model power assertion**. The real LiRA needs ≥128 trained shadow models + real canary in/out splits (Pass-2, cross-repo `# SWITCH-POINT(DATA)`); the in-tree representative is the deterministic stand-in: it scores membership from an observable per-record signal (the LiRA intuition — a member's loss/confidence is distinguishable), reports TPR at the two committed low FPRs over synthetic member/non-member scores, computes a representative canary exposure, and carries the non-strippable NFR-016 caveat. It runs under the S5-04 sandbox and is **de-circularized** (FR-013/AX-002): `membership_scores(records)` sees ONLY the observable signal, never the gold membership label (which lives only in the scorer).

## 2. Approach / scope — the two carried DESIGN decisions

### (a) The representative MIA adversary (vs the real LiRA@128)
* **`MiaRecord`** (frozen) — `{record_id, observed_loss: float}`: what the attacker observes (the model's loss/confidence on a candidate record; lower loss ⟹ more member-like, the LiRA signal). The gold `is_member` is NOT on the record the attacker reads.
* **`RepresentativeMiaAttack`** (conforms to `MiaAttack`; `adversary_id="representative-mia@v1"`, `deterministic=True`) — `membership_scores(records) -> list[float]` returns a deterministic membership score per record (higher ⟹ more likely member): a monotone transform of `-observed_loss` (the calibrated LiRA-shape stand-in). Pure; no RNG/clock. `# SWITCH-POINT(DATA)`: the real LiRA computes a per-record likelihood ratio against ≥128 shadow models' in/out loss distributions.
* **De-circularization (FR-013):** `membership_scores` has no access to the gold membership; the gold lives only in `score_mia_attack`. An attack that read the label would "cheat" — a `[SECURITY-TEST]` proves the score is a pure function of `observed_loss` only.

### (b) The NFR-013 reporting machinery (TPR@low-FPR + Secret-Sharer + ≥128-shadow power)
* **`tpr_at_fpr(scores, gold_membership, *, fpr_target) -> float`** — the LiRA reporting primitive: sweep the score threshold over the synthetic member/non-member scores, find the largest threshold whose empirical FPR ≤ `fpr_target`, report the TPR there (0.0 when no threshold achieves the target FPR — honest, never fabricated). Pure + deterministic; integer-count ROC.
* **`MiaSuccessReport`** (frozen) — `tpr_at_1e_3`, `tpr_at_1e_2`, `auc`, `n_members`, `n_non_members`, `adversary_id`, `deterministic`, the Wilson CI (REUSE `wilson_interval` on the TP/member counts), + the non-strippable `caveat`. `score_mia_attack(scores, gold_membership, *, adversary_id, deterministic)` builds it.
* **Secret-Sharer:** `canary_exposure(rank, n_candidates) -> float` = `log2(n_candidates) - log2(rank)` (the standard exposure: higher ⟹ more memorization ⟹ worse privacy; rank 1 = most-likely canary). `SecretSharerReport` (frozen) carries the exposure + the caveat. `# SWITCH-POINT(DATA)`: the real canary in/out splits.
* **`MiaPowerReport`/`assess_mia_power(shadow_model_count, ...)`** — the NFR-013 power assertion: `powered = shadow_model_count >= MIA_MIN_SHADOW_MODELS` (128). The representative reports the count + the powered verdict; training the real ≥128 shadows is Pass-2. Carries the caveat.
* **Sandbox runner** `mia_attack_runner(*, records_json, gold_membership_json, shadow_model_count) -> dict` — scalar JSON-string args (the `AttackSpec` rule), stdlib JSON decode (no unsafe-deserialization), run `RepresentativeMiaAttack` + `score_mia_attack`, return `as_outcome()` (a Mapping). `MIA_ATTACK_REGISTRY = {"mia_representative": mia_attack_runner, "<module-path>": mia_attack_runner}` — additively merged (disjoint keys from S5-01/S5-02; reid_baseline + reid_tier3_representative survive).

## 2a. Pre-claim de-risk (verify against live code on claim)
- **RISK-1 (sandbox):** `mia_attack_runner` runs ONLY via `run_attack_under_sandbox`; returns a Mapping (else `SandboxViolation`). `[SECURITY-TEST]`.
- **RISK-2 (import isolation):** `mia.py` auto-scanned by `tests/test_attacks_import_boundary.py` (globs `*.py`) — 0 imports from `{swarm, moe, fusion, policy}`; an explicit `mia.py in scanned-files` assertion.
- **RISK-3 (no dangerous primitives):** the `attacks/` AST source-guard auto-covers `mia.py` — 0 unsafe-deserialization / subprocess / shell-out / dynamic-eval.
- **RISK-4 (determinism, AX-002):** membership scoring + ROC + exposure are pure; no `random`/`uuid`/`time`/`secrets`; `[PROPERTY-TEST]` pins replay + sandboxed replay-equal.
- **RISK-5 (NFR-016 caveat):** `MiaSuccessReport`/`SecretSharerReport`/`MiaPowerReport` carry the non-strippable caveat (re-asserted; blank raises) — REUSE the S5-01 `ANTI_ANONYMITY_CAVEAT`.
- **RISK-6 (off-limits + AX-001):** `orchestrator.py` + `test_moe_enhancements.py` + `competitor_compare.py` (`7cae16c8…`) + `competitive_supremacy.py` (`3b842e8…`) + `sandbox.py`/`spec.py`/`reid.py`/`reid_tier3.py` byte-identical; the SDO gate NOT touched (**no SDO close**). All records SYNTHETIC.
- **RISK-7 (de-circularization, FR-013):** `membership_scores(records)` has no gold-label access; `[SECURITY-TEST]` proves the score depends only on `observed_loss`.

## 3. Given / When / Then (acceptance)
- **A1 — representative MIA conforms to `MiaAttack` `[CONTRACT-TEST]`.** `isinstance(RepresentativeMiaAttack(), MiaAttack)` is True; exposes `adversary_id` + `deterministic=True`.
- **A2 — membership scores rank members above non-members `[UNIT-TEST]`.** On a fixture where members have lower `observed_loss`, `membership_scores` assigns members strictly higher scores.
- **A3 — de-circularization: score depends only on the observable signal `[SECURITY-TEST]`.** Two records with identical `observed_loss` but opposite gold membership receive identical scores (the attack cannot read the label).
- **A4 — `tpr_at_fpr` is a correct integer-count ROC `[UNIT-TEST]`.** On a separable fixture, `tpr_at_fpr(...,fpr_target=1e-2)==1.0`; on a non-separable fixture TPR < 1; an unachievable FPR target ⟹ 0.0 (never fabricated).
- **A5 — TPR reported at BOTH committed FPRs `[CONTRACT-TEST]`.** `MiaSuccessReport` carries `tpr_at_1e_3` AND `tpr_at_1e_2` (NFR-013).
- **A6 — Secret-Sharer exposure is correct `[UNIT-TEST]`.** `canary_exposure(rank=1, n_candidates=2**20)==20.0`; exposure decreases as rank grows; `rank==n_candidates` ⟹ 0.0; bounded ≥ 0.
- **A7 — ≥128-shadow power assertion `[CONTRACT-TEST]`.** `MIA_MIN_SHADOW_MODELS==128`; `assess_mia_power(127)` powered=False, `assess_mia_power(128)` powered=True.
- **A8 — every MIA report carries the non-strippable caveat (NFR-016) `[SECURITY-TEST]`.** `MiaSuccessReport`/`SecretSharerReport`/`MiaPowerReport` default to `ANTI_ANONYMITY_CAVEAT`; blank raises; `as_outcome()` always carries it.
- **A9 — deterministic replay `[PROPERTY-TEST]`.** Two runs (+ record-order permutation where order-independent) yield identical scores/reports.
- **A10 — runner runs under the sandbox, returns a Mapping `[SECURITY-TEST]` `[INTEGRATION-TEST]`.** `run_attack_under_sandbox` over `mia_representative` returns an `AttackResult` whose outcome is a Mapping carrying the TPR fields + caveat.
- **A11 — registry merge is additive `[CONTRACT-TEST]`.** `mia_representative` AND `reid_baseline` (S5-01) AND `reid_tier3_representative` (S5-02) all resolve; no clobber.
- **A12 — `mia.py` is import-boundary scanned `[AUDIT]`.** `tests/test_attacks_import_boundary.py` includes `mia.py`; 0 forbidden imports.
- **A13 — corrupt-input safety `[SECURITY-TEST]`.** `tpr_at_fpr` with an out-of-range `fpr_target` (or mismatched score/label lengths) is refused with a domain-named `ValueError`; `canary_exposure` with `rank<1`/`rank>n`/`n<1` refused (never an out-of-range exposure). (Mirrors the S5-02 SEC-01 fail-loud-domain-named discipline.)
- **A14 — sandboxed replay-equal `[PROPERTY-TEST]`.** Two sandboxed runs produce equal `AttackResult` (wall-clock excluded).

## 5. Notes / non-goals
- **Non-goal:** the real LiRA@128 (training ≥128 shadow models on in/out splits) — `# SWITCH-POINT(DATA)`; Pass-2/cross-repo (eval-data). The real Secret-Sharer canary in/out splits — `# SWITCH-POINT(DATA)`; Pass-2.
- **Non-goal:** changing the S5-04 sandbox / the S5-01 `reid.py` / the S5-02 `reid_tier3.py` (all byte-identical) or `privacy_metrics.py::MembershipInferenceMetric` (a different framework; not the LiRA attack body).
- **Non-goal:** touching the SDO gate — S5-03 is a privacy-attack feature; `competitive_supremacy.py` NOT changed, so **no adversarial SDO close** (the S5 sprint-close adversarial run remains the standing recommendation for the attack surface).
- **Pass-2 flags:** real LiRA@128 shadow training; real canary in/out splits; the live TPR@low-FPR at real scale (current ships the reporting machinery + a representative scorer).

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[SECURITY-TEST]` `[INTEGRATION-TEST]` `[AUDIT]`. **Reviewers (canonical 5-gate story set):** **security-sast** (**PRIMARY** — runs under the sandbox; import-boundary + dangerous-call guards; de-circularization; fail-loud domain guards) + **axiom-compliance** (AX-001 + AX-002 + import-isolation + NFR-016) + **code-quality** + **requirements-coverage** (FR-013 representative scope + NFR-013 TPR@low-FPR + ≥128-shadow power MUST; Pass-2 flags tracked) + **traceability** (DC-09 → FR-013/NFR-013 + UC-10). All five APPROVE. **No SDO adversarial close** (no `competitive_supremacy.py` change).

## 12. Definition of Done
- [ ] **RED**: `tests/test_attack_mia.py` (A1–A14) first & failing (`ModuleNotFoundError` on `eval_framework.attacks.mia`). RED precedes GREEN.
- [ ] **GREEN**: `attacks/mia.py` + additive `__init__` — all A1–A14 green.
- [ ] **REFACTOR**: tidy; additive edge tests only.
- [ ] **Quality gate**: full xdist suite green; ruff clean; mypy clean under BOTH `mypy src/pii_anon` AND `--strict`; coverage ≥84% (new module ≥90%).
- [ ] **Security (headline)**: A10 (sandbox) + A12 (import-boundary) + the AST dangerous-call guard + A8 (caveat) + A3/A7/A13 (de-circularization + power + fail-loud).
- [ ] **Determinism (AX-002)**: A9 + A14 byte-identical replay; no random/uuid/time/secrets.
- [ ] **Untouched / off-limits**: `orchestrator.py` + `test_moe_enhancements.py` + `competitor_compare.py`/`competitive_supremacy.py`/`sandbox.py`/`spec.py`/`reid.py`/`reid_tier3.py` byte-identical; `artifacts/benchmarks/*` never written; narrow `git add` of owned files only.
- [ ] **Story-gate APPROVE** — `_reviews/story/S5-03/`, all 5 reviewers APPROVE; substantive MINOR + ALL MAJOR remediated in-loop.
- [ ] **SDO verdict UNCHANGED** — recompute `pii-anon supremacy` (a MIA feature flips no guarantee; expect NOT_YET / `canonical_claim_run=False` byte-stable).

## Evidence (filled on completion)

*Provisional status: AGENT_SIMULATED (see the metadata `provisional_status` field). The `RepresentativeMiaAttack`, the `tpr_at_fpr`/`score_mia_attack` ROC machinery, `canary_exposure`, the ≥128-shadow power assertion, the sandbox run, the de-circularization, and the import/AST guards run for REAL in-tree against SYNTHETIC records. The real LiRA@128 shadow-model training + real Secret-Sharer canary in/out splits are Pass-2 / cross-repo (eval-data S6, `# SWITCH-POINT(DATA)`).*

**Commits (RED→GREEN→remediation, on `pdlc/sota-program`):** RED `32913d7` (tests-only; `ModuleNotFoundError` on `eval_framework.attacks.mia`) → GREEN `350b9cd` (the module + additive `__init__` merge) → remediation `f23de77` (the story-gate findings).

**Files:** `src/pii_anon/eval_framework/attacks/mia.py` (new), `src/pii_anon/eval_framework/attacks/__init__.py` (additive re-exports + the `{**REID,**TIER3,**MIA}` registry merge), `tests/test_attack_mia.py` (new — A1–A14, 24 cases). `reid.py`/`reid_tier3.py`/`sandbox.py`/`spec.py` byte-identical (consumed read-only).

**Acceptance → tests (A1–A14, 24 cases):** A1 MiaAttack conformance; A2 members rank above non-members; A3 de-circularization (identical observed_loss → identical score regardless of gold); A4 ROC anchored at 1.0 / 0.0 / **0.5** (the exact partial-separation intermediate — RC-01 fix); A5 TPR@both committed FPRs; A6 Secret-Sharer exposure (exact + monotone); A7 ≥128-shadow power; A8 non-strippable caveat on all three reports; A9 deterministic scores; A10 runs under the sandbox; A11 additive registry merge (reid_baseline + reid_tier3_representative survive); A12 import-boundary scanned; A13 corrupt-input fail-loud (tpr_at_fpr + canary_exposure + **the runner decode boundary** — non-finite observed_loss / non-bool gold); A14 sandboxed replay-equal.

**Story gate (5/5 APPROVE; `_reviews/story/S5-03/`):** security-sast (PRIMARY) + axiom-compliance + traceability + requirements-coverage + code-quality. **0 unremediated MAJOR.** Remediated in-loop (`f23de77`): TRACE-01 (the provisional_status field — MAJOR), RC-01 (the tpr_at_fpr intermediate anchor — also resolves CQ-01), CQ-02 (drop slots from MiaRecord), and the two decode-boundary OBSERVATIONs (security NaN/inf + CQ-03 bool-coercion) → `_records_from_json`/`_gold_from_json` now fail loud. Dispositioned NO-ACTION: RC-02 + CQ-04 (the caveat non-blank-vs-exact-string pattern, inherited from S5-01) + TRACE-03; TRACE-02 (matrix backfill) deferred to the S5 sprint gate.

**SDO — UNCHANGED (a MIA feature flips no guarantee):** `pii-anon supremacy` on the committed smoke artifact reads **NOT_YET / `canonical_claim_run=False` (G7)** — byte-identical to SO-17. The SDO gate `competitive_supremacy.py` is byte-identical (md5 `3b842e81c3f03eafd11f9c655c1789a0`), so **no adversarial SDO close was required**. Off-limits `competitor_compare.py` `7cae16c8…` + user-WIP `orchestrator.py` `0afc6dee…` / `test_moe_enhancements.py` `910e9cd6…` byte-identical.

**Quality:** full xdist suite green (see SO-18 for the count); ruff clean (src+tests); mypy clean under BOTH `mypy src/pii_anon` AND `--strict` (139 files). The attacks import-boundary + dangerous-call AST guards auto-cover `mia.py`.

**DoD:** all checkboxes met. Pass-2 (tracked): the real LiRA@128 shadow training; the real Secret-Sharer canary in/out splits; the live TPR@low-FPR at real scale.
