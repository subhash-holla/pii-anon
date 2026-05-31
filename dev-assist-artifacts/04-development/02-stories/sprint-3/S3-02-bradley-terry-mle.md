# S3-02 — bradley-terry-mle (pure-stdlib MM + paired bootstrap behind the RatingEnginePort)

> **Cold-pickup invariant**: this file is executable cold by any executor agent without prior conversation context. If you'd need context not in this file, that's a story-design defect — escalate, don't proceed.

| Field | Value |
|---|---|
| Epic | E3 Eval rating engine (DC-06 ladder / DC-07 significance foundation) |
| State | **IN_PROGRESS** (2026-05-31; claimer=dev-assist-development-executor) |
| Implements | FR-003 (rating-engine abstraction — **second** ladder tier), NFR-002/NFR-003 (significance-coherence foundation via paired bootstrap), NFR-026 (graceful degradation) |
| Traces | Design D-EVAL DECISION 2 — `RatingEnginePort` 3-tier ladder: `glicko-legacy` → **`bradley-terry-mle`** → `bayes-bt`. "pure-stdlib MM + paired bootstrap; fast PR-CI/smoke tier." |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` |
| Files owned | `src/pii_anon/eval_framework/rating/bradley_terry.py` (new), `rating/__init__.py` (additive export), `pyproject.toml` (entry-point row only), `tests/test_bradley_terry_mle.py` (new) |
| Depends on | **S3-01 DONE** (`RatingEnginePort`/`RatingEngineRegistry` — satisfied). ⛓ Cross-repo: eval-data S6 `stats/bradley_terry.py` **VERIFIED ABSENT** → ship temp-local MM behind the port (documented switch-point); swap to the eval-data primitive when S6 lands. |
| Blocks | S3-03 (`bayes-bt` claim-grade tier), S3-04 (coherent significance + record-level paired outcomes + Davidson ties), and the SDO `CompetitiveSupremacyGate` J-fallback (MLE-BT path before bayes lands) |
| Size | M |

## 1. Intent
Add the **second tier** of the rating ladder: a `BradleyTerryMLEEngine` that fits Bradley–Terry (1952) strengths by **minorization-maximization (Hunter 2004)** — pure standard library, ZERO new hard/optional deps — and quantifies uncertainty by **paired (record-resampling) bootstrap**. It satisfies the existing `RatingEnginePort` structurally (zero call-site changes) and is registered as the `bradley-terry-mle` entry point in group `pii_anon.rating_engines`. This is the fast PR-CI / smoke tier of the ladder; the claim-grade default (`bayes-bt`, NumPyro NUTS) follows in S3-03. `elo.py` and all 7 production callers stay UNTOUCHED.

## 2. Why a temp-local impl (cross-repo honesty)
The design's canonical source for the BT primitive is `pii-anon-eval-data` `stats/bradley_terry.py`, **verified absent today** (DATA repo is mid-S5; BT lands at S6). Per the design switch-point, CODE ships a self-contained MM impl behind the port so the eval-integrity critical path (S3-02→03→04) and the SDO J-meter are not DATA-blocked. When eval-data S6 lands, the local impl is swapped for a thin adapter over the frozen primitive — the port keeps both call-sites stable. The module docstring + a `# SWITCH-POINT(S6)` marker record this explicitly.

## 3. Given/When/Then (acceptance)
- **Structural-port (FR-003).** **Given** `BradleyTerryMLEEngine`, **then** `isinstance(BradleyTerryMLEEngine(), RatingEnginePort) is True` and mypy --strict confirms structural conformance with ZERO call-site changes. **When** `run_round_robin(composites)` runs, **then** it returns `list[RatingUpdate]` and `get_rating(name)` returns an `EloRating` for every system in `composites` (and `None` for unknown).
- **MM correctness (FR-003).** **Given** a paired-outcome design with known asymmetric win counts, **when** the MM iteration runs to convergence, **then** the recovered strength ordering matches the true ordering AND the fitted log-strengths satisfy the BT stationarity condition `Σ_j n_ij·(π_i/(π_i+π_j)) == W_i` within tol. **Negative**: a perfect total order (degenerate composites) does NOT diverge — the soft-outcome mapping (sigmoid of composite gaps, mirroring `elo._sigmoid`) keeps strengths finite and convergence reached in ≤ max_iter.
- **Paired bootstrap (NFR-002/003).** **Given** record-level paired outcomes, **when** `paired_bootstrap(B, seed)` runs, **then** it returns per-system 95% CIs that are deterministic for a fixed seed, monotone-consistent (point estimate ∈ CI), and **coherent** (a system whose CI strictly exceeds another's ⇒ ranked above it — sign↔verdict cannot disagree). This is the *frequentist foundation* of NFR-002; the by-construction joint-posterior coherence is S3-03/04.
- **Graceful degradation (NFR-026).** **Given** the `bradley-terry-mle` entry point, **when** `discover_entrypoint_engines("pii_anon.rating_engines")` runs after `pip install -e .`, **then** the registry lists `['bradley-terry-mle', 'glicko-legacy']` (sorted) and each resolves to a `RatingEnginePort`. Absent optional tiers never raise.
- **Determinism (AX-002).** Same input + same seed ⇒ byte-identical ratings + CIs across runs (no `set`/dict-order nondeterminism; sorted system iteration like `elo.run_round_robin`).
- **Import isolation (S3-05 guard).** `bradley_terry.py` imports nothing from `{swarm, moe, fusion, policy}` — the existing `tests/test_rating_import_boundary.py` AST walk must stay GREEN over the new file.

## 4. Approach (algorithm + port mapping)
- **Strength model.** Parameterize per-system strength `π_i = exp(θ_i)`; report on the **Elo scale** via `rating = 1500 + scale·θ_i / ln(10)` (so deltas read like Elo points; `scale=400` matches `elo.py`), `EloRating.rd` ← bootstrap SE mapped to the same scale (fallback: Fisher-information SE when only one round-robin is available). Anchor identifiability with **sum-to-zero** `Σ θ_i = 0` (re-centre each iteration) — mirrors the design's "anchored identifiability."
- **MM iteration (Hunter 2004).** `π_i ← W_i / Σ_{j≠i} n_ij/(π_i+π_j)`, renormalize (geometric-mean to 1 ⇔ sum-to-zero in θ), iterate to `max|Δ log π| < tol` (tol=1e-9, max_iter=1000). `W_i` = total (fractional) wins of i; `n_ij` = comparisons between i,j. Add a tiny smoothing prior (+ε wins on the complete-design diagonal) so a single round-robin with separable outcomes still has a finite MLE (standard BT regularization; documented).
- **Port-compat path — `run_round_robin(composites: dict[str,float])`.** Point composites are a perfect total order ⇒ raw BT MLE degenerates. Map to **fractional** pairwise outcomes `s_ij = sigmoid(γ·(C_i − C_j))` (reuse the `elo._sigmoid` shape, γ default 10) → `w_ij += s_ij`, `n_ij += 1`. Fit MM on this complete design. Emit one `RatingUpdate` per system per match (old=prior rating, new=fitted) for audit parity with the Elo engine's history contract. This keeps the fast tier a faithful drop-in.
- **Claim-grade path (NOT in the port) — `fit_paired(comparisons)`.** Accept record-level paired outcomes `{(sys_i, sys_j): (wins_i, wins_j, n)}` (the shape S3-04 derives from `per_record_f1`). Fit MM directly on integer counts (no soft mapping needed — real designs aren't perfectly separable). `paired_bootstrap(records, B, seed)` resamples *records* with replacement, refits, and returns percentile CIs. Ties are **out of scope** here (plain BT) — the **Davidson tie term** is S3-04 (bayes). Leave a `# DAVIDSON(S3-04)` marker at the tie-handling seam.
- **No new deps.** Pure `math` + `random.Random(seed)` + stdlib. `numpy` stays optional/absent — do not import it. (The `bayes-eval` extra is reserved for S3-03.)

## 5. Notes / scope (explicit non-goals)
- Does NOT touch `elo.py`, `leaderboard.py`, `scorecard.py`, `competitor_composite.py`, `competitor_compare.py`, or any of the 7 callers. Additive only.
- Does NOT implement Bayesian inference, NUTS, R-hat/ESS/divergence gates (that's S3-03) or Davidson ties / one-joint-posterior coherence (that's S3-04).
- Does NOT wire BT into the default leaderboard path — `glicko-legacy` remains the default; BT is opt-in via the registry. (Switching the default to claim-grade is S3-03/S4 CanonicalRunGate work.)
- The user WIP files (`orchestrator.py`, `test_moe_enhancements.py`, `artifacts/benchmarks/*`, `benchmark-diagnostics.json`, `README.md`, `docs/*`) are OFF-LIMITS — do not stage/commit them; verify md5 unchanged.

## 9. Test-type tags
`[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` — implies reviewer set: code-quality + axiom-compliance + traceability (always); security-sast (conditional — entry-point load path, same trust surface as S3-01, expect APPROVE).

## 12. Definition of Done
- [ ] **RED**: `tests/test_bradley_terry_mle.py` written first & failing — contract (`isinstance … RatingEnginePort`), MM-recovers-known-ordering, stationarity-condition, degenerate-no-divergence, paired-bootstrap determinism+coherence, registry-discovery-lists-both, determinism property (`@given` seeded). RED commit precedes GREEN (git-evidenced).
- [ ] **GREEN**: `bradley_terry.py` + additive `__init__.py` export + `pyproject.toml` entry-point row (`bradley-terry-mle = "...bradley_terry:BradleyTerryMLEEngine"`). `pip install -e .` then discovery lists both tiers.
- [ ] **Quality**: ruff clean; **mypy --strict** clean; full suite green (≥ 2699 prior, no regressions); coverage ≥ 84%.
- [ ] **Isolation**: `test_rating_import_boundary.py` stays GREEN (rating ⊄ detection).
- [ ] **Untouched**: `elo.py` + 7 callers byte-identical (git-verified); user WIP md5 unchanged.
- [ ] **Story-gate review APPROVE** (`_reviews/story/S3-02-gate.yaml`).

## Evidence (filled on completion)
- RED commit · GREEN commit · REFACTOR commit (if any)
- `isinstance(BradleyTerryMLEEngine(), RatingEnginePort)` → True; `discover_entrypoint_engines("pii_anon.rating_engines")` → `['bradley-terry-mle', 'glicko-legacy']`
- MM recovers planted ordering; stationarity residual < tol; degenerate composites converge (no overflow)
- Paired-bootstrap CIs deterministic for fixed seed; point ∈ CI; rank↔CI coherence holds
- ruff / mypy --strict / full-suite / coverage results
- *Agent-simulated CI (local `.venv`); Pass-2 real-CI confirmation scheduled by orchestrator.*
