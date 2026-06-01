# S3-04 — coherent significance BY CONSTRUCTION + record-level paired outcomes + Davidson ties

> **Cold-pickup invariant**: executable cold. Consumes the S3-03 `Posterior` (joint θ draws). The significance + rank-probability primitives are **pure-numpy** (operate on the joint samples array) → testable in-tree WITHOUT numpyro; the Davidson tie term extends the NumPyro model and is `importorskip`-gated for the real run.

| Field | Value |
|---|---|
| Epic | E3 Eval rating engine (DC-07 coherent significance) |
| State | **IN_PROGRESS** (claimer=dev-assist-development-executor; claimed_at=2026-06-01; red_at=2026-06-01) |
| provisional_status | AGENT_SIMULATED (real-NUTS Davidson run is a `bayes-eval`-CI / Pass-2 step) |
| Implements | FR-004 (coherent significance), NFR-002 (significance coherence — now **BY CONSTRUCTION**), NFR-003; FR-010/AX-004 (anon≠pseudo never merged — the Tier-3 RRS Davidson sub-model stays separate) |
| Traces | Design D-EVAL DECISION 2 / DC-07: "one joint posterior (point∈CI, sign↔verdict, significant-iff-CI-excludes-0 cannot disagree) — eliminates elo.py:243/542/561 fabricated-outcome/fake-CI/decoupled-significance defects. Record-level paired outcomes (N·C(K,2) from per_record_f1) + Davidson tie term. Separate Davidson sub-model for Tier-3 RRS (never merged — FR-010)." |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[INTEGRATION-TEST]` |
| Files owned | `src/pii_anon/eval_framework/rating/significance.py` (new — pure-numpy coherent significance + rank-prob from a joint posterior), `src/pii_anon/eval_framework/rating/paired_set.py` (new — temp-local `assemble_paired_set` from per_record_f1, `# SWITCH-POINT(S6)`), `bayes_bt.py` (extend `_bt_model` with the Davidson tie term at the existing `# DAVIDSON(S3-04)` seam + a `fit_paired_posterior(..., ties=...)` path), `rating/__init__.py` (exports), `tests/test_coherent_significance.py` (new), `tests/test_paired_set.py` (new) |
| Depends on | **S3-03 DONE** (the `Posterior` / joint θ samples this consumes), S3-02 DONE (paired-count shape). |
| Blocks | the SDO `CompetitiveSupremacyGate` — **J = P(rank(pii-anon)=1 | joint posterior)** is literally `rank_one_probability(posterior, "pii-anon")` from this story; the per-pair coherent verdicts feed the supremacy margins. |
| Size | L |

## 1. Intent
Make pairwise significance **coherent by construction** by reading it off the ONE joint posterior S3-03 produces, eliminating the three verified `elo.py` defects (`:243` sigmoid fabricates a match outcome from a composite gap; `:542` CI = rating ± 1.96·RD is a fake normal-approx CI decoupled from any data; `:561` significance = rating-diff > 2·√(RD²+RD²) is a third, independent computation that can disagree with both). From one posterior over θ, every significance statement is the SAME object: the sign of `mean(θ_i − θ_j)`, the credible interval of `θ_i − θ_j`, and "significant ⟺ that interval excludes 0" CANNOT disagree. Add **record-level paired outcomes** (each record → a pairwise win/loss/**tie** between two systems from their per-record F1) and a **Davidson (1970) tie term** so ties are modeled, not dropped. Keep the **Tier-3 RRS** resistance model a SEPARATE Davidson sub-model (never merged into the de-id score — FR-010/AX-004). Expose `rank_one_probability(posterior, system)` — the SDO J-meter's exact quantity.

## 2. Cross-repo honesty (temp-local `assemble_paired_set`)
The design's canonical `assemble_paired_set` lives in `pii-anon-eval-data` (S6 — **VERIFIED ABSENT** today, same as the BT primitive). Ship a self-contained `paired_set.py` behind a small interface so S3-04 (and the SDO gate) are not DATA-blocked; swap for the eval-data primitive when S6 lands (`# SWITCH-POINT(S6)` marker + docstring). The temp-local builder takes `{system: [per_record_f1...]}` (aligned by record index) and emits `{(sys_i,sys_j): (wins_i, wins_j, ties, n)}` using a tie-band ε (|f1_i − f1_j| ≤ ε → tie).

## 3. Given/When/Then (acceptance)
- **Coherence BY CONSTRUCTION (NFR-002, the headline).** **Given** any joint posterior `theta_samples` (n_draws, n_systems) — synthetic in-tree — **when** `pairwise_significance(posterior)` runs, **then** for every pair: (a) `point = mean(θ_i − θ_j)`; (b) `ci = (q2.5, q97.5)` of `θ_i − θ_j`; (c) `significant = (0 < ci.lo) or (ci.hi < 0)`; (d) `sign(point)` agrees with which side of 0 the CI sits; and these three CANNOT contradict because they are read off the SAME draws. **Property test** (`@given` synthetic posteriors): point ∈ ci ALWAYS; significant ⟹ point ≠ 0 with matching sign ALWAYS; never "significant but CI spans 0", never "sign positive but point negative".
- **Eliminates the elo defects (FR-004).** A regression test documents that `significance.py` derives ALL of {point, CI, verdict} from one sample array — NO sigmoid-fabricated outcome, NO rating±1.96·RD normal CI, NO separate rating-diff>threshold test. (elo.py stays untouched as the legacy tier; this is the claim-grade path.)
- **rank-1 probability = SDO J (the optimization meter).** **Given** a joint posterior, **when** `rank_one_probability(posterior, name)` runs, **then** it returns the fraction of joint draws in which `name` has the maximum θ; Σ over systems = 1.0 (it's a proper distribution over "who is #1"); deterministic; pure-numpy. **Negative**: a clearly-dominant synthetic system → prob ≈ 1.0; a tied pair → ≈ 0.5 each.
- **Record-level paired outcomes (FR-004).** `assemble_paired_set({sys:[f1...]})` → counts with ties; N records × C(K,2) pairs; tie-band ε respected; deterministic; rejects ragged (mismatched-length) inputs loudly.
- **Davidson ties (importorskip numpyro).** `_bt_model` extended with a Davidson tie term: tie parameter `ν ~ HalfNormal`, tie/win/loss multinomial-logit per record sharing the SAME θ. Real-NUTS integration test (importorskip) on a synthetic design WITH ties → converges (NFR-001 gate passes), recovers ordering, and `ν` posterior is identifiable. The tie likelihood math is ALSO unit-tested pure-numpy (Davidson P(tie), P(i>j) formulas on hand values) so it has in-tree teeth.
- **Tier-3 separation (FR-010/AX-004).** A separate `fit_rrs_posterior` (or a `kind="rrs"` flag) builds a DISTINCT Davidson sub-model for re-identification-resistance; a CI guard test asserts the RRS posterior is NEVER merged into the detection/de-id significance (separate object, separate call, no shared mutable state). 
- **Determinism (AX-002) + import isolation (S3-05).** Pure-numpy paths deterministic; `significance.py`/`paired_set.py` import nothing from swarm/moe/fusion/policy (boundary test GREEN).

## 4. Approach
- **`significance.py` (pure-numpy).** `PairwiseVerdict` dataclass (i, j, point, ci_lo, ci_hi, p_i_beats_j, significant). `pairwise_significance(theta_samples, names) -> list[PairwiseVerdict]`: vectorized differences over draws, percentile CIs, `p_i_beats_j = mean(θ_i > θ_j)`. `rank_one_probability(theta_samples, names) -> dict[name,float]`: `argmax` over the system axis per draw, normalized counts (this IS J). `significant` derived ONLY from the CI excluding 0 → coherence by construction. All consume `Posterior.theta_samples` from S3-03 (or a raw array, for tests).
- **`paired_set.py` (temp-local, `# SWITCH-POINT(S6)`).** `PairedComparisonSet` (counts + ties) + `assemble_paired_set(per_record_f1, *, tie_eps=1e-9)`. Pure-stdlib/numpy. Interface kept thin so the eval-data S6 primitive drops in behind it.
- **Davidson term in `_bt_model`.** At the `# DAVIDSON(S3-04)` seam: add `nu = sample("nu", HalfNormal(1))`; per record the three-way outcome (i-wins / tie / j-wins) follows Davidson(1970): `P(i>j) ∝ exp(θ_i)`, `P(tie) ∝ ν·exp((θ_i+θ_j)/2)`, `P(j>i) ∝ exp(θ_j)`. Implement as a Categorical/Multinomial over (win_i, tie, win_j) counts. Plain-BT path (no ties) stays the default; ties path activated when the paired set carries tie counts. Shares the SAME θ → significance off ONE posterior.
- **mypy/deps:** no new deps (numpy already in bayes-eval; significance/paired_set are numpy-only and live with the bayes tier). Add the two new entry points NOT required (these are library functions, not engines).

## 5. Notes / scope (non-goals)
- Does NOT wire significance into the leaderboard emission or the CanonicalRunGate (S4 DC-11) — it provides the primitives the gate calls. Does NOT build the SDO `CompetitiveSupremacyGate` itself (separate successor story) — but provides its J (`rank_one_probability`) + per-pair verdicts.
- Does NOT touch `elo.py` (legacy significance stays as-is for the glicko-legacy tier; the coherent path is the claim-grade replacement consumed by S4). Does NOT touch the 7 callers.
- DO NOT install numpyro/jax. USER WIP OFF-LIMITS — narrow `git add`; md5 unchanged.

## 9. Test-type tags
`[UNIT-TEST]` `[CONTRACT-TEST]` `[PROPERTY-TEST]` `[INTEGRATION-TEST]` — reviewers: code-quality + axiom-compliance + traceability (always); requirements-coverage (FR-004/NFR-002 MUST); security-sast (if entry-point/load path touched — likely not).

## 12. Definition of Done
- [ ] **RED**: `tests/test_coherent_significance.py` (coherence-by-construction property tests: point∈CI, sign↔verdict, never-contradict; rank_one_probability sums to 1 + dominant≈1 + tie≈0.5; Davidson tie-formula pure-numpy teeth) + `tests/test_paired_set.py` (assemble from per_record_f1, ties, ragged-reject) — written first & failing.
- [ ] **GREEN**: `significance.py` + `paired_set.py` + Davidson extension to `_bt_model` (+ importorskip real-NUTS-with-ties integration test) + additive `__init__.py` exports.
- [ ] **Coherence proven**: property tests show the three significance statements cannot disagree (the NFR-002 by-construction claim has teeth). The three elo defects are documented as NOT reproducible in the claim-grade path.
- [ ] **Quality**: full suite green (numpyro integration tests SKIP); ruff + mypy --strict clean; import-boundary GREEN; coverage ≥ 84%.
- [ ] **Tier-3 separation**: CI guard asserts RRS posterior never merged into de-id significance (FR-010/AX-004).
- [ ] **Untouched**: elo.py + 7 callers byte-identical; numpyro/jax NOT installed; user WIP md5 unchanged.
- [ ] **Story-gate APPROVE** (`_reviews/story/S3-04-gate.yaml`).

## Evidence (filled on completion)
- RED/GREEN/REFACTOR SHAs · coherence property results · rank_one_probability (J) behavior · Davidson teeth · Tier-3 separation guard · importorskip integration status · ruff/mypy/suite/coverage · *AGENT_SIMULATED; real-NUTS Davidson run = Pass-2.*
