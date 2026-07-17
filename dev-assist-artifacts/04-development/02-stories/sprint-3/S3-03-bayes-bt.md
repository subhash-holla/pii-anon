# S3-03 — bayes-bt (NumPyro NUTS claim-grade tier + hard NFR-001 convergence gate)

> **Cold-pickup invariant**: executable cold by any executor agent. The NumPyro model + the pure-numpy convergence gate are fully specified below. The current `.venv` has **numpy but NOT numpyro/jax/arviz** — read §2 (Environment fork) FIRST; it governs the whole DoD.

| Field | Value |
|---|---|
| Epic | E3 Eval rating engine (DC-06 ladder / DC-07 significance) |
| State | **DONE** (gate APPROVE 2026-05-31; `_reviews/story/S3-03-gate.yaml`; 5 reviewers APPROVE; 1 substantive MINOR remediated in-loop) |
| provisional_status | AGENT_SIMULATED (real-NUTS run is a `bayes-eval`-CI / Pass-2 step) |
| Implements | FR-003 (rating abstraction — **third / claim-grade** tier), **NFR-001 (MCMC convergence gate: split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param ∧ 0 divergences — the resolved CATASTROPHIC eval-01)**, NFR-026 (graceful degradation when `bayes-eval` absent) |
| Traces | Design D-EVAL DECISION 2 — "only `bayes-bt` is claim-grade; a hard convergence gate refuses claim-grade leaderboard emission on failure (fails loud)." Resolves SME CATASTROPHIC eval-01. |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[INTEGRATION-TEST]` |
| Files owned | `src/pii_anon/eval_framework/rating/bayes_bt.py` (new), `src/pii_anon/eval_framework/rating/convergence.py` (new — pure-numpy diagnostics gate), `rating/__init__.py` (additive export), `pyproject.toml` (`bayes-eval` extra + entry-point row + mypy overrides), `tests/test_bayes_bt.py` (new), `tests/test_convergence_gate.py` (new) |
| Depends on | **S3-01 DONE** (port/registry), **S3-02 DONE** (BT MLE — bayes reuses the paired-outcome shape + is validated against MLE on well-identified designs). |
| Blocks | S3-04 (coherent significance BY CONSTRUCTION consumes this ONE joint posterior + adds the Davidson tie term), the SDO `CompetitiveSupremacyGate` **J = P(rank(pii-anon)=1 | joint posterior)** (J literally needs this posterior), S4 CanonicalRunGate (wires the convergence gate into leaderboard emission). |
| Size | L (focused: engine + gate; ties/coherence-by-construction are S3-04) |

## 1. Intent
Land the **claim-grade** third tier of the rating ladder: a `BayesBTEngine` that fits a Bayesian Bradley–Terry model by **NUTS (NumPyro)** and produces a **joint posterior** over per-system strengths, behind the existing `RatingEnginePort`. Pair it with the **hard NFR-001 convergence gate** (`convergence.py`, pure-numpy): split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param ∧ 0 divergences. The gate **fails loud** — it refuses to mark a fit claim-grade when diagnostics miss the bar (resolves SME CATASTROPHIC eval-01: frequentist tiers satisfy NFR-001 only by-substitution, so MLE/glicko are smoke/fallback only; **only bayes-bt is claim-grade**). This is the posterior the SDO J-meter and S3-04's by-construction coherence are built on.

## 2. Environment fork (READ FIRST — governs the DoD)
The dev `.venv` has **numpy 2.4.2** but **numpyro / jax / jaxlib / arviz are ABSENT**, and they are heavy, platform-specific wheels. **DO NOT `pip install` numpyro/jax/arviz into the `.venv`** — that env is shared with the user's WIP and the program treats real-dep verification as Pass-2. Architect for this:
- **Module import-safe WITHOUT jax.** `bayes_bt.py` must import (and be `ep.load()`-able) with numpyro absent — achieved by **lazy-importing numpyro/jax INSIDE the sampling method**, never at module top level. So the engine stays *discoverable* (the registry lists it), and only *sampling* requires the extra.
- **Use-without-extra fails loud, not silent.** Calling `run_round_robin`/`fit_paired_posterior` with numpyro absent raises a clear `MissingOptionalDependencyError` (define it, or reuse an existing optional-dep error if the repo has one — check `pii_anon` for a prior-art pattern first) whose message names the `bayes-eval` extra. NEVER fall back to a non-Bayesian answer and label it claim-grade (that would re-introduce eval-01).
- **The convergence gate is pure-numpy** → fully testable in THIS venv. It is the load-bearing NFR-001 artifact; it must be real and green in-tree with teeth.
- **Real NUTS = `pytest.importorskip("numpyro")`** integration test → SKIPPED here, real in the `bayes-eval` CI lane. Honest: the claim-grade sampler exists and is correct-by-construction; executing it is a tracked Pass-2 item.

## 3. Given/When/Then (acceptance)
- **Structural port (FR-003).** `isinstance(BayesBTEngine(), RatingEnginePort) is True`; module imports with numpyro absent; mypy --strict clean (numpyro/jax/arviz added to the mypy ignore-missing-imports overrides).
- **Graceful degradation (NFR-026).** **Given** numpyro absent, **when** `run_round_robin(composites)` is called, **then** it raises `MissingOptionalDependencyError` naming `bayes-eval` (NOT a silent non-Bayesian fallback). **And** registry discovery still lists `bayes-bt` among the tiers (module is import-safe) → `discover_entrypoint_engines("pii_anon.rating_engines")` returns `['bayes-bt','bradley-terry-mle','glicko-legacy']` (sorted). Update the S3-01/S3-02 discovery assertions to the 3-tier list.
- **NFR-001 convergence gate — PASS path (importorskip numpyro).** **Given** a synthetic well-identified BT design (≥4 systems, separable-but-not-perfect, ≥N records), **when** NUTS runs (≥4 chains, ≥1000 warmup + ≥1000 draws, target_accept≥0.8), **then** `ConvergenceReport` has split-R̂ ≤ 1.01 (all params) ∧ bulk-ESS ≥ 400/param ∧ 0 divergences, `report.claim_grade is True`, and the posterior-mean ordering recovers the planted ordering (and is consistent with S3-02 MLE on the same design).
- **NFR-001 convergence gate — TEETH (pure-numpy, no jax).** **Given** deliberately NON-converged posterior samples (e.g., 2 chains sampled at different locations → split-R̂ ≫ 1.01; or an injected divergence count > 0; or ESS < 400), **when** `assert_claim_grade()` runs, **then** it RAISES `ConvergenceError` (fails loud) and `report.claim_grade is False` with the binding diagnostic named. **Negative**: a clean synthetic posterior (well-mixed chains) → `claim_grade is True`, no raise.
- **Pure-numpy diagnostics correctness.** `split_rhat`, `bulk_ess`, `count_divergences` match known reference values on canned chains (cross-check the split-R̂ and bulk-ESS formulas against the Vehtari et al. 2021 / Gelman-Rubin definitions; assert on hand-computable tiny cases + a near-iid case where R̂≈1.0 and ESS≈n_draws·n_chains).
- **Determinism (AX-002).** Fixed `seed` → identical posterior summary + identical `ConvergenceReport` (NUTS path: pass the PRNGKey from seed; gate path: pure-numpy deterministic).
- **Import isolation (S3-05).** `bayes_bt.py` + `convergence.py` import nothing from `{swarm,moe,fusion,policy}` — `test_rating_import_boundary.py` stays GREEN.

## 4. Approach (model + gate specifics)
- **NumPyro model (BT, claim-grade).** Latent strengths `θ ~ Normal(0, σ)` with hierarchical `σ ~ HalfNormal(1.0)`; **sum-to-zero anchored** identifiability (subtract mean, or use the standard `θ = θ_raw - mean(θ_raw)` deterministic). Pairwise outcome likelihood: for each comparison record between i,j, `wins_i ~ Binomial(n_ij, logits = θ_i − θ_j)` (Bernoulli-logit per record, or Binomial on aggregated counts — Binomial on the `(wins_i, wins_j, n)` counts is cheaper and exact). **No Davidson tie term yet** — that is S3-04; leave a `# DAVIDSON(S3-04)` seam where the tie likelihood factor wires in. Sampler: `NUTS(model, target_accept_prob=0.9)`, `MCMC(num_warmup, num_samples, num_chains>=4, chain_method='sequential')` (sequential avoids jax pmap device issues in CI). Seeded `random.PRNGKey`.
- **Posterior → port types.** `get_rating(name)` returns `EloRating`: `rating = 1500 + 400·mean(θ_i)/ln10` (Elo scale, matches elo.py + S3-02); `rd ← 400·sd(θ_i)/ln10` (posterior SD on the Elo scale — a genuine uncertainty, unlike the legacy match-count-only RD). `run_round_robin(composites)` maps point composites → a synthetic paired design (reuse S3-02's sigmoid soft-outcome mapping so the port-compat path is consistent across tiers), samples, returns `list[RatingUpdate]` (old=prior, new=posterior-mean) for audit parity. The CLAIM-GRADE entry point is `fit_paired_posterior(comparisons)` (record-level counts, the real shape) — NOT on the port (engine-only, like S3-02's `fit_paired`); it returns a `Posterior` object carrying the joint samples (the thing S3-04 + the SDO J-meter consume).
- **`convergence.py` (pure-numpy, the NFR-001 teeth).** Input: a samples array shaped `(n_chains, n_draws, n_params)` + optional `divergences` count from NUTS extra fields. Implement:
  - `split_rhat(samples)` → per-param **split-**R̂ (split each chain in half → 2·n_chains half-chains; rank-normalized split-R̂ per Vehtari 2021 is ideal — at minimum the split Gelman-Rubin).
  - `bulk_ess(samples)` → per-param bulk effective sample size (autocorrelation-based; the standard ESS estimator).
  - `ConvergenceReport` dataclass: `max_rhat`, `min_bulk_ess`, `n_divergences`, `n_params`, `claim_grade: bool`, `binding_constraint: str` (which check failed + by how much — so the program always knows the next thing to fix, per the SDO philosophy).
  - `assert_claim_grade(report)` → raises `ConvergenceError(binding_constraint)` if not claim-grade.
  - Thresholds are NFR-001 literals: `RHAT_MAX=1.01`, `ESS_MIN_PER_PARAM=400`, `MAX_DIVERGENCES=0`.
- **Deps / pyproject.** Add `bayes-eval = ["numpyro>=0.13", "jax>=0.4", "arviz>=0.17", "numpy>=1.24"]`. Add entry-point row `bayes-bt = "pii_anon.eval_framework.rating.bayes_bt:BayesBTEngine"`. Add `numpyro.*`, `jax.*`, `jaxlib.*`, `arviz.*` to `[[tool.mypy.overrides]] ignore_missing_imports`. (numpy is typed — no override needed.)

## 5. Notes / scope (non-goals)
- NO Davidson ties, NO one-joint-posterior coherence wiring into the leaderboard (S3-04). NO CanonicalRunGate leaderboard wiring (S4 DC-11). NO change to the default rating path (glicko-legacy stays default; bayes-bt is opt-in claim-grade).
- Does NOT touch `elo.py`, `bradley_terry.py` internals (may import EloRating/RatingUpdate types only), or the 7 callers. Additive.
- **DO NOT install numpyro/jax/arviz into the `.venv`.** USER WIP (`orchestrator.py`, `test_moe_enhancements.py`, `artifacts/benchmarks/*`, `benchmark-diagnostics.json`, `README.md`, `docs/*`) OFF-LIMITS — narrow `git add` only; md5 must stay unchanged.

## 9. Test-type tags
`[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[INTEGRATION-TEST]` — reviewer set: code-quality + axiom-compliance + traceability (always); security-sast (entry-point path); requirements-coverage (NFR-001 is a MUST). The `[INTEGRATION-TEST]` (real NUTS) is importorskip-gated.

## 12. Definition of Done
- [ ] **RED**: `tests/test_convergence_gate.py` (pure-numpy teeth — non-converged → raises; converged → claim_grade True; formula correctness on canned chains) + `tests/test_bayes_bt.py` (isinstance port; numpyro-absent → MissingOptionalDependencyError naming bayes-eval; 3-tier discovery; importorskip real-NUTS PASS + planted-ordering recovery) — all written first & failing. RED precedes GREEN.
- [ ] **GREEN**: `convergence.py` (pure-numpy gate) + `bayes_bt.py` (lazy-NUTS engine) + additive `__init__.py` exports + pyproject (`bayes-eval` extra + `bayes-bt` entry-point row + mypy overrides). `pip install -e . --no-deps` → discovery lists 3 tiers.
- [ ] **Quality**: full suite green (numpyro integration test SKIPS, everything else passes; ≥ 2727 prior + new unit tests, no regressions); ruff clean; `mypy src/pii_anon` --strict clean (numpyro/jax/arviz overridden).
- [ ] **NFR-001 teeth proven**: a non-converged synthetic posterior makes `assert_claim_grade` RAISE (git-evidenced test). The gate names the binding constraint.
- [ ] **Isolation**: `test_rating_import_boundary.py` GREEN over both new files. Discovery assertions in S3-01/S3-02 tests updated to the 3-tier list.
- [ ] **Untouched**: `elo.py` + 7 callers byte-identical; user WIP md5 unchanged; numpyro/jax NOT installed.
- [ ] **Story-gate APPROVE** (`_reviews/story/S3-03-gate.yaml`).

## Evidence (agent-simulated execution)
- **RED** `33f389f` (tests first, failed at collection — modules absent) → **GREEN** `2b2110b` (`convergence.py` pure-numpy gate + `bayes_bt.py` lazy-NUTS engine + additive `__init__` exports + `bayes-eval` extra + `bayes-bt` entry-point + numpyro/jax/arviz mypy overrides) → **REFACTOR** `69cd45c` (drop a dead truthiness guard in `split_rhat`). RED precedes GREEN (git-evidenced).
- **Structural port**: `isinstance(BayesBTEngine(), RatingEnginePort) is True` (live-verified). `get_rating` numpyro-free.
- **NFR-026 graceful degradation**: numpyro absent → `run_round_robin`/`fit_paired_posterior` raise `MissingOptionalDependencyError` naming `bayes-eval` (live-verified, mentions=True); registry discovery still lists all 3 tiers → `['bayes-bt','bradley-terry-mle','glicko-legacy']` (module import-safe via lazy NUTS import).
- **NFR-001 gate TEETH (pure-numpy, runs in-tree)**: `test_teeth_rhat_failure_…`, `test_teeth_divergence_failure_…`, `test_teeth_ess_failure_…` — non-converged synthetic samples → `assert_claim_grade` RAISES `ConvergenceError(binding_constraint=…)`; clean posterior → `claim_grade is True`. split-R̂ / bulk-ESS validated against Gelman-Rubin / near-iid reference cases.
- **`pytest.importorskip("numpyro")`** real-NUTS integration test → **SKIPPED** in `.venv` (numpyro absent — honest); real in the `bayes-eval` CI lane / Pass-2. numpyro/jax NOT installed (env protected).
- **Quality**: full suite **exit 0** (~2754 passed / ~14 skipped, +27 new: convergence-gate 15, bayes_bt 12+2-skip; 2777 collected), coverage **86.03%**; ruff `All checks passed`; `mypy src/pii_anon` --strict `Success: 118 files`; import-boundary GREEN over both new files; `elo.py` + `bradley_terry.py` + 7 callers byte-identical.
- *AGENT_SIMULATED (local `.venv` py3.12); real-NUTS execution + R̂/ESS/divergence verification on real data = tracked Pass-2 / `bayes-eval`-CI.*
