# S5-01 — `ReidAttack` protocol + baseline re-identification body + the attacks import-boundary test (DC-09)

> **Cold-pickup invariant**: executable cold. The S5-04 sandbox substrate is DONE (`eval_framework/attacks/` has `sandbox.py` `run_attack_under_sandbox()` + the `DEFAULT_ATTACK_REGISTRY` allow-list, `spec.py` `AttackSpec`). This story adds the **re-identification attack protocol** + a deterministic **baseline body** that runs via `run_attack_under_sandbox`, plus the standing **attacks import-boundary CI test** (the `__init__.py` docstring already promises it). It shape-mirrors the DATA sibling's `Adversary`/`Persona`/`Target`/`Guess` contract (`../pii-anon-eval-data/.../scoring/adversary/base.py`) so the Pass-2 swap to the real offline adversary is structural. Satisfies the **NFR-016 non-strippable-caveat MUST** now; lays the FR-011/FR-013 protocol foundation (the real Tier-3 LLM adversary = S5-02; MIA = S5-03). Fully in-tree, deterministic, synthetic.

| Field | Value |
|---|---|
| Epic | **E5 Attacks** (DC-09: `attacks/` real Tier-3 LLM-adversary + LiRA@128 MIA) |
| State | **REVIEW** (owner: dev-assist-development-executor; branch: `pdlc/sota-program` [main working tree, no worktree]) |
| provisional_status | AGENT_SIMULATED (the ReidAttack protocol, the deterministic baseline body, the success-metrics scorer, the non-strippable caveat, the sandbox-run, and the import-boundary test all run for REAL in-tree against SYNTHETIC personas/targets; a real Tier-3 LLM adversary [S5-02] + a real offline DATA adversary with Wilson CIs are Pass-2 / cross-repo) |
| Implements | **NFR-016** (non-strippable re-id caveat — 100% of exported privacy artifacts carry the anti-anonymity caveat — MUST, satisfied now via `ReidSuccessMetrics`'s non-strippable `caveat`); the **FR-011/FR-013 protocol foundation** (the `ReidAttack` + `MiaAttack` Protocols + a representative baseline ReidAttack body — the real Tier-3 LLM adversary is S5-02, the real LiRA@128 MIA is S5-03); upholds **AX-001** (synthetic), **AX-002** (deterministic total-order ranking), and the **attacks import-isolation invariant** (attacks ⊄ swarm/moe/fusion/policy — the standing CI guard this story adds). |
| Traces | Design **DC-09** (`D-implementation-ready-design.md:19` — "`attacks/` package: real Tier-3 LLM-adversary (de-circularized) + LiRA@128 MIA"; `:58` — "Tier-3 de-circularization lives in `attacks/`"). UC-09. The S5-04 sandbox substrate (`run_attack_under_sandbox`) — consumed, not changed. |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[SECURITY-TEST]` `[AUDIT]` |
| Files owned | `src/pii_anon/eval_framework/attacks/reid.py` (**new** — `ReidPersona`/`ReidTarget`/`ReidGuess` value objects, `ReidSuccessMetrics`, `@runtime_checkable ReidAttack` + `MiaAttack` Protocols, `BaselineDeterministicReidAttack`, `score_reid_attack`, `reid_attack_runner` + `REID_ATTACK_REGISTRY`). **Additive** to `src/pii_anon/eval_framework/attacks/__init__.py` (re-exports + merge `REID_ATTACK_REGISTRY` into the sandbox default registry). `tests/test_attack_reid_protocol.py` (**new**). `tests/test_attacks_import_boundary.py` (**new** — the standing CI guard the `__init__` docstring promises). |
| Depends on | **S5-04 DONE** (the sandbox substrate: `run_attack_under_sandbox`, `AttackSpec`, `DEFAULT_ATTACK_REGISTRY`). The DATA sibling `Adversary`/`Persona`/`Target`/`Guess` shape is mirrored (read-only reference). |
| Blocks | **S5-02** (Tier-3 LLM adversary — reuses the `ReidAttack` Protocol + ranking + scorer) + **S5-03** (MIA — reuses the `MiaAttack` Protocol seam + the package boundary test). Feeds the SDO **G5** audit half + Paper 1 Tier-2/3. |
| Size | **M** (one new module + the boundary test; ~11 acceptance tests). |

## 1. Intent
Re-identification resistance is the audit half of G5 (and Paper 1's Tier-2/3 evidence). S5-01 lays the foundation: a `ReidAttack` Protocol (input: anonymized/pseudonymized targets + auxiliary persona knowledge; output: re-identification guesses + success metrics) + a deterministic **baseline** adversary (weighted-Jaccard over surviving quasi-identifier tokens, ranked by `(similarity desc, persona_id asc)` — a total order, AX-002) that runs ONLY inside the S5-04 sandbox (the attack body is allow-listed + invoked via `run_attack_under_sandbox`). The success metrics carry a **non-strippable anti-anonymity caveat** (NFR-016 — `__post_init__` re-asserts it; it cannot be cleared). The protocol mirrors the DATA sibling's `Adversary` contract so swapping the baseline for the real offline adversary (Wilson CIs on integer counts) is a one-line Pass-2/cross-repo change. This story also lands the standing **attacks import-boundary CI test** (the package imports nothing from swarm/moe/fusion/policy — the `__init__` docstring already promises it).

## 2. Approach / scope
- **Value objects (frozen):** `ReidPersona(persona_id, quasi_identifiers: tuple[str,...], auxiliary_knowledge: Mapping, source_text)`, `ReidTarget(target_id, anonymized_text, observed_signals: tuple[str,...])`, `ReidGuess(target_id, guessed_persona_id: str|None, score: float)`.
- **`ReidSuccessMetrics`** (frozen) — `reid_recall`, `reid_precision`, `reid_success_rate`, `correct: int`, `n_targets: int`, `n_guesses: int`, `candidate_set_size: int`, `adversary_id: str`, `deterministic: bool`, `caveat: str = ANTI_ANONYMITY_CAVEAT`. `__post_init__` RE-ASSERTS the caveat (a non-strippable-caveat guard — clearing it raises; NFR-016) + `as_outcome() -> dict` (the sandbox runner's return Mapping).
- **`@runtime_checkable ReidAttack(Protocol)`** — `adversary_id: str`, `deterministic: bool`, `attack(targets, candidates, candidate_set_size) -> list[ReidGuess]`. Plus a **`MiaAttack(Protocol)`** seam (`adversary_id`, `deterministic`, `membership_scores(records) -> list[float]`) consumed by S5-03 (declared here so the package boundary + the protocol family land together).
- **`BaselineDeterministicReidAttack`** — weighted-Jaccard over the surviving quasi-identifier tokens + observed signals (faithful to the DATA sibling's `_weighted_jaccard`), ranked `(similarity desc, persona_id asc)` → a TOTAL order (AX-002). Abstains (`guessed_persona_id=None`) when no signal survives.
- **`score_reid_attack(guesses, targets, *, adversary_id, deterministic, candidate_set_size) -> ReidSuccessMetrics`** — integer counts: `correct = #(guessed_persona_id == true target persona)`, `reid_recall = correct/n_targets`, `reid_precision = correct/n_guesses`, `reid_success_rate = recall·precision`.
- **`reid_attack_runner(*, targets_json: str, candidates_json: str, candidate_set_size: int) -> dict`** — the **allow-listed** runner (scalar JSON-string args per the `AttackSpec` scalar-param rule): JSON-decode → frozen objects → `BaselineDeterministicReidAttack().attack(...)` → `score_reid_attack(...).as_outcome()` (a Mapping, per the sandbox contract). `REID_ATTACK_REGISTRY = {"reid_baseline": reid_attack_runner}`, merged into the sandbox default registry (additive to `attacks/__init__`).

## 2a. Pre-claim de-risk (verify against live code on claim)
- **RISK-1 (every body runs under the sandbox):** the baseline is invoked via `run_attack_under_sandbox(spec, inputs=...)` with `attack_kind="reid"` (already a `Literal` in `spec.py`); the runner returns a Mapping (else `SandboxViolation`). A `[SECURITY-TEST]` runs it under the sandbox + asserts the outcome is a Mapping.
- **RISK-2 (import isolation — the standing CI guard):** `tests/test_attacks_import_boundary.py` AST-walks `pii_anon.eval_framework.attacks` and asserts it imports NOTHING from `{swarm, moe, fusion, policy}` (+ a "≥1 file scanned" guard) — a verbatim adaptation of `tests/test_rating_import_boundary.py`.
- **RISK-3 (no dangerous primitives in attack bodies):** an `[AUDIT]` AST source-guard over `attacks/` asserts zero unsafe-deserialization / subprocess / shell-out / dynamic-eval call signatures (mirrors the S5-04 guard) — attack bodies are pure-stdlib over scalar inputs.
- **RISK-4 (determinism, AX-002):** the ranking is a total order `(similarity desc, persona_id asc)`; no `random`/`uuid`/`time`/`secrets`; the sandbox `AttackResult` equality already excludes wall-clock. A `[PROPERTY-TEST]` pins replay-equality.
- **RISK-5 (NFR-016 non-strippable caveat):** `ReidSuccessMetrics.caveat` defaults to `ANTI_ANONYMITY_CAVEAT` and `__post_init__` re-asserts it — a test that constructs with `caveat=""` (or strips it) must raise/restore.
- **RISK-6 (off-limits + AX-001):** `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `evaluation/competitor_compare.py` byte-identical (RISK-6); `sandbox.py`/`spec.py` consumed read-only. All personas/targets SYNTHETIC.

## 3. Given / When / Then (acceptance)
- **A1 — `ReidAttack` Protocol is runtime-checkable `[CONTRACT-TEST]`.** `isinstance(BaselineDeterministicReidAttack(), ReidAttack)` is True; the Protocol is `@runtime_checkable`.
- **A2 — baseline links an exact quasi-identifier match `[UNIT-TEST]`.** Given a target whose surviving QI tokens uniquely match one persona, the baseline guesses that persona (highest weighted-Jaccard).
- **A3 — baseline abstains on no signal `[UNIT-TEST]`.** Given a fully-anonymized target (no surviving QI/signal), the baseline returns `guessed_persona_id=None`.
- **A4 — deterministic total-order ranking `[PROPERTY-TEST]`.** Two runs (and a candidate-order permutation) yield identical guesses; ties resolve by `persona_id asc`.
- **A5 — success metrics are integer-count-based `[UNIT-TEST]`.** `score_reid_attack` returns `reid_recall=correct/n_targets`, `reid_precision=correct/n_guesses`, `reid_success_rate=recall·precision` with `correct`/`n_targets`/`n_guesses` integers.
- **A6 — non-strippable caveat (NFR-016) `[SECURITY-TEST]` `[CONTRACT-TEST]`.** `ReidSuccessMetrics(...).caveat == ANTI_ANONYMITY_CAVEAT`; constructing with `caveat=""`/stripping it raises (or restores) — the caveat cannot be removed; `as_outcome()` always carries it.
- **A7 — runner runs under the sandbox `[SECURITY-TEST]` `[INTEGRATION-TEST]`.** `run_attack_under_sandbox` over the `reid_baseline` runner (scalar JSON args) returns an `AttackResult` whose outcome is a Mapping with the metric fields.
- **A8 — runner outcome is a Mapping `[CONTRACT-TEST]`.** `reid_attack_runner(...)` returns a `Mapping[str, Any]` (the sandbox contract; a non-Mapping return would be a `SandboxViolation`).
- **A9 — deterministic AttackResult equality `[PROPERTY-TEST]`.** Two sandboxed runs produce equal `AttackResult` (wall-clock excluded).
- **A10 — attacks package imports nothing forbidden `[AUDIT]`.** `tests/test_attacks_import_boundary.py`: AST-walk `eval_framework.attacks` → 0 imports from `{swarm, moe, fusion, policy}`; ≥1 file scanned.
- **A11 — no dangerous call signatures in attacks `[AUDIT]` `[SECURITY-TEST]`.** AST source-guard over `attacks/`: 0 unsafe-deserialization / subprocess / shell-out / dynamic-eval signatures.

## 5. Notes / non-goals
- **Non-goal:** the real Tier-3 LLM adversary (RRS/QIC/BSL) — **S5-02** (reuses this protocol + scorer; the LLM path is lazy/optional, the real run Pass-2). The real LiRA@128 MIA — **S5-03** (reuses the `MiaAttack` seam; real run Pass-2, canary splits absent in DATA).
- **Non-goal:** the real offline DATA adversary + Wilson CIs — Pass-2/cross-repo (the baseline mirrors the shape; `# SWITCH-POINT(DATA)` marks the swap).
- **Non-goal:** changing the S5-04 sandbox substrate (`sandbox.py`/`spec.py` byte-identical; bodies plug the allow-list).

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[SECURITY-TEST]` `[AUDIT]`. **Reviewers (canonical 5-gate story set):** **security-sast** (**PRIMARY** — every body runs under the sandbox; the import-boundary + dangerous-call-signature guards; no egress) + **axiom-compliance** (AX-001 synthetic + AX-002 determinism + the import-isolation invariant + NFR-016 non-strippable caveat) + **code-quality** + **requirements-coverage** (NFR-016 MUST + honest FR-011/013-foundation scope) + **traceability** (DC-09 → NFR-016 + the FR-011/013 foundation). All five APPROVE.
> **Adversarial close:** RECOMMENDED at the S5 work-stream close (a security/attack surface) — independent attempts to run an attack body outside the sandbox, strip the caveat, or import a forbidden module. Bar = 0 upheld.

## 12. Definition of Done
- [ ] **RED**: `tests/test_attack_reid_protocol.py` + `tests/test_attacks_import_boundary.py` (A1–A11) written first & failing (`ModuleNotFoundError` on `eval_framework.attacks.reid`). RED precedes GREEN (git-evidenced).
- [ ] **GREEN**: `attacks/reid.py` + additive `attacks/__init__` re-exports/registry-merge — all A1–A11 green.
- [ ] **REFACTOR**: tidy (share weighted-Jaccard with a small helper); additive edge tests only.
- [ ] **Quality gate**: full suite green (no regression); ruff clean; mypy clean under BOTH `mypy src/pii_anon` AND `mypy src/pii_anon --strict`; coverage ≥84% (new module ≥90% by own tests).
- [ ] **Security (headline)**: A7 (runs under sandbox) + A10 (import-boundary) + A11 (no dangerous signatures) + A6 (non-strippable caveat).
- [ ] **Determinism (AX-002)**: A4 + A9 byte-identical replay; no random/uuid/time/secrets.
- [ ] **Untouched / user-WIP**: `orchestrator.py` + `tests/test_moe_enhancements.py` byte-identical; `competitor_compare.py` (md5 `7cae16c89f4c97136e1a12394dae2025`) + `sandbox.py`/`spec.py` byte-identical; `artifacts/benchmarks/*` never written; narrow `git add` of owned files only.
- [ ] **Story-gate APPROVE** — `_reviews/story/S5-01/`, all 5 reviewers APPROVE; substantive MINOR + ALL MAJOR remediated in-loop.

## Evidence (filled on completion)
*Provisional status: AGENT_SIMULATED. The ReidAttack/MiaAttack protocols, the deterministic baseline body, the success scorer, the non-strippable caveat, the sandbox-run, and the import-boundary + dangerous-signature guards run for REAL in-tree against SYNTHETIC personas/targets. A real Tier-3 LLM adversary (S5-02) + a real offline DATA adversary with Wilson CIs are Pass-2 / cross-repo.*

### Execution (TDD; main working tree, branch `pdlc/sota-program`)
- **RED** `59b8404` — `test: S5-01 RED — pin FR-011/FR-013 ReidAttack protocol + baseline + NFR-016 non-strippable caveat (A1–A11)`. Tests-only; both new files fail with `ModuleNotFoundError: No module named 'pii_anon.eval_framework.attacks.reid'` (and `test_attacks_import_boundary.py` independently asserts `reid.py` must be scanned). Files: `tests/test_attack_reid_protocol.py`, `tests/test_attacks_import_boundary.py`.
- **GREEN** `68c5d67` — `feat: S5-01 GREEN — implement FR-011/FR-013 ReidAttack protocol + deterministic baseline body (DC-09)`. New `src/pii_anon/eval_framework/attacks/reid.py`; additive `src/pii_anon/eval_framework/attacks/__init__.py` (re-exports + in-place merge of `REID_ATTACK_REGISTRY` into the sandbox default allow-list — `sandbox.py`/`spec.py` untouched, byte-identical). A1–A11 green.
- **REFACTOR** `9aa39bc` — `refactor: S5-01 — consolidate the runner's two JSON decoders behind a shared object-array helper`. Extracted `_decode_object_array` + `_string_tuple`; additive edge tests only. `reid.py` → 100% line+branch coverage.
- **RED-before-GREEN proof:** `git merge-base --is-ancestor 59b8404 68c5d67` ⇒ exit 0; `… 68c5d67 9aa39bc` ⇒ exit 0 (strict RED→GREEN→REFACTOR ancestry).

### Acceptance → tests (A1–A11; 34 new cases = 32 + 2)
- **A1** (runtime-checkable `ReidAttack`/`MiaAttack`): `test_fr011_a1_reid_attack_protocol_is_runtime_checkable`, `test_fr011_a1_baseline_exposes_required_protocol_attributes`, `test_fr013_a1_mia_attack_protocol_seam_is_runtime_checkable`.
- **A2** (exact-QI link): `test_fr011_a2_baseline_links_exact_quasi_identifier_match`, `test_fr011_a2_baseline_prefers_higher_overlap_persona`.
- **A3** (abstain on no signal): `test_fr011_a3_baseline_abstains_on_no_surviving_signal`, `test_fr011_a3_baseline_abstains_when_signal_matches_no_candidate`.
- **A4** (deterministic total-order ranking): `test_ax002_a4_ranking_is_deterministic_across_runs`, `test_ax002_a4_candidate_permutation_yields_identical_guesses`, `test_ax002_a4_ties_resolve_by_persona_id_ascending`.
- **A5** (integer-count metrics): `test_nfr016_a5_score_reid_attack_uses_integer_counts`, `test_nfr016_a5_zero_guesses_does_not_divide_by_zero`.
- **A6** (non-strippable caveat, NFR-016): `test_nfr016_a6_metrics_carry_anti_anonymity_caveat_by_default`, `test_nfr016_a6_constructing_with_blank_caveat_is_refused[…]` (3 params), `test_nfr016_a6_caveat_cannot_be_stripped_post_construction`, `test_nfr016_a6_as_outcome_always_carries_caveat`.
- **A7** (runs under sandbox): `test_fr011_a7_runner_runs_under_sandbox_returns_mapping`, `test_fr011_a7_reid_runner_is_registered_in_sandbox_default_registry`.
- **A8** (Mapping outcome): `test_nfr016_a8_runner_returns_mapping_outcome`.
- **A9** (equal `AttackResult`, wall-clock excluded): `test_ax002_a9_two_sandboxed_runs_yield_equal_attack_result`.
- **A10** (import-boundary AST guard): `tests/test_attacks_import_boundary.py::test_ax_isolation_attacks_layer_has_no_forbidden_imports`, `…::test_ax_isolation_at_least_one_attacks_module_scanned` (verbatim adaptation of `test_rating_import_boundary.py`).
- **A11** (no dangerous signatures): `test_ax002_a11_reid_body_imports_no_nondeterminism_sources`, `test_ax002_a11_reid_body_has_no_unsafe_call_signatures` (reid-scoped) + the package-wide `tests/test_attack_sandbox.py::test_attacks_package_has_no_unsafe_execution_call_signatures` now also scans `reid.py` (`scanned >= 3` ⇒ 4 files, green).
- REFACTOR additive edge tests (coverage→100%): `test_runner_rejects_non_array_targets_json`, `…_non_array_candidates_json`, `…_non_object_target_element`, `…_non_array_observed_signals_field`, `…_non_array_quasi_identifiers_field`, `test_runner_empty_inputs_yield_zeroed_metrics_with_caveat`, `test_baseline_normalises_token_case_and_whitespace`, `test_baseline_custom_adversary_id_is_pinned`.

### Quality gate (all green)
- `ruff check src tests` → **All checks passed**.
- `mypy src/pii_anon` → **Success: no issues found in 134 source files**; `mypy src/pii_anon --strict` → **Success: no issues found in 134 source files** (both invocations clean — the numpy-typing gotcha is N/A here, pure-stdlib module).
- `PYTHONPATH=src python -m pytest` (full suite, coverage gate) → exit 0; **Total coverage 87.48% ≥ 84%** ("Required test coverage of 84% reached"). Pass count **3401 passed / 16 skipped** (3367 pre-change baseline + 34 new; zero failures, zero regressions). `reid.py` own coverage **100%** (109 stmts / 20 branch, 0 miss).
- AGENT-SIMULATED note: pytest/ruff/mypy run for real locally on this host; no real CI runner / benchmark hardware — a Pass-2 real-CI run is scheduled per the program's epistemic-honesty discipline.

### Protected / user-WIP byte-identical (working-tree md5 re-verified post-REFACTOR)
- `src/pii_anon/orchestrator.py` = `0afc6deed62bbd0653ae1051b723bace` ✓ (user WIP, untouched)
- `tests/test_moe_enhancements.py` = `910e9cd66ad6e38c7bb64a9c51ecb1cb` ✓ (user WIP, untouched)
- `src/pii_anon/evaluation/competitor_compare.py` = `7cae16c89f4c97136e1a12394dae2025` ✓ (RISK-6)
- `src/pii_anon/eval_framework/attacks/sandbox.py` = `1c60b03ef1464f88112d9d814d59d6a1` ✓ (consumed read-only)
- `src/pii_anon/eval_framework/attacks/spec.py` = `bd452e05236137109c650815a648c622` ✓ (consumed read-only)
- SDO gate `src/pii_anon/eval_framework/evaluation/competitive_supremacy.py` — NOT touched (no diff). `artifacts/benchmarks/*` — never written. The 3 commits touch only owned paths: `reid.py`, `attacks/__init__.py`, `tests/test_attack_reid_protocol.py`, `tests/test_attacks_import_boundary.py`, this story file (narrow explicit `git add`; no `-A`/`-u`/`.`/`commit -a`).

### Review polish (post-APPROVE, in-loop; non-behavioral, State stays REVIEW)
- The 5-gate story set returned **APPROVE (0 MAJOR)**. Light in-loop polish of the 3 code-quality MINORs (type/comment only — NO behavioral change, NO new tests): **CQ-S5-01-02** retype `REID_ATTACK_REGISTRY: Final[Mapping[str, AttackCallable]]` (import the sandbox's `AttackCallable` runner type; registries cannot drift); **CQ-S5-01-03** drop the two redundant inner `int(candidate_set_size)` casts (annotation stays the honest `int`; reached as `int` on BOTH the sandbox-`inputs` (A7) + direct (A8) paths); **CQ-S5-01-01** expand the `# type: ignore[index]` registry-merge suppressor into a full `# SAFETY:` note (same mutable dict — `attacks.DEFAULT_ATTACK_REGISTRY is sandbox.DEFAULT_ATTACK_REGISTRY`; additive+idempotent, recon runner never overridden; switch to an `extend_default_registry()` helper IF `sandbox.py` ever wraps it in `MappingProxyType`). Touches ONLY `attacks/reid.py` + `attacks/__init__.py` (`sandbox.py`/`spec.py`/`competitor_compare.py` byte-identical; SDO gate untouched; `artifacts/benchmarks/*` never written). The security/axiom **MappingProxyType-immutability** + **registry-mutation** OBSERVATIONs are deferred to the **S5 adversarial close** (the standing catch-net for this attack-surface work-stream). Re-verify: `ruff check src tests` clean; `mypy src/pii_anon` AND `mypy src/pii_anon --strict` both clean (134 files); full suite **3401 passed / 16 skipped / 0 failed**, coverage **87.48% ≥ 84%**. Commit `refactor: S5-01 review polish — tighten REID_ATTACK_REGISTRY type + single candidate_set_size cast + SAFETY note on the sandbox-registry merge`.

### History
- IN_PROGRESS → REVIEW. Awaiting the canonical 5-gate story set (security-sast PRIMARY + axiom-compliance + code-quality + requirements-coverage + traceability). Adversarial close RECOMMENDED at the S5 work-stream close (bar = 0 upheld).
- REVIEW (APPROVE, 0 MAJOR) → in-loop code-quality-MINOR polish (3 type/comment-only fixes; no behavioral change, no new tests) → **State stays REVIEW** (the security/axiom MappingProxyType + registry-mutation OBSERVATIONs deferred to the S5 adversarial close).
