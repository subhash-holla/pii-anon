# Project Journey — pii-anon SOTA Program

> **Wave D3 authoring deliverable (the signature one).** The story of one autonomous PDLC
> pass over the `pii-anon-code` library, from a brownfield assessment on 2026-05-30 (SO-01)
> through a complete feature surface on 2026-06-09 (SO-23). It is the problem → solution →
> build → proof narrative, threaded along the 23-sign-off chronological spine.
>
> **Read this first.** It is the front door to the documentation set; the architecture, API,
> user/operator, contributor, and changelog deliverables each go deeper on one slice of what
> follows. Every load-bearing claim cites its source sign-off (SO-NN) or canonical artifact.
>
> **One honesty note up front, because it is the spine of the whole story.** The program's
> own completion-criterion gate returns an honest **NOT_YET** verdict — it refuses to certify
> the headline "we are state of the art" claim on the current data draw. That is not a failure
> hidden at the bottom of this document; it is the *point* of the document. This program
> optimized for the **honesty of the claim machinery** over the claim itself, and the machinery
> works precisely because it tells the truth. See `## Methodology` for what is authored-from-
> artifact vs agent-inferred, and the closing sections for the honest verdict and the Pass-2
> horizon.

---

## 0. The shape of the arc

| Milestone / wave | What happened | Spine |
|---|---|---|
| **M1 — Brownfield + bootstrap** | A mature v1.4.0 library assessed: 12 MAJOR findings, 0 SHOWSTOPPER; 6 axioms authored; 24 legacy files migrated to 5 citation artifacts (0 deletions). | SO-01 |
| **Discovery → Requirements → Design** | A POV pivot to *measurement-first*; 7 personas, 28 use cases; 39 FR + 26 NFR; 15 Design Cases + 3 Pugh-won decisions; one CATASTROPHIC eval finding resolved in design. | SO-02 · SO-03 · SO-04 |
| **Sprint-1 — the recall floor** | The single load-bearing correctness MUST (FR-016/NFR-011/AX-003) made true *by construction* and wired live on the production fusion path. | SO-05 · SO-06 · SO-07 |
| **The eval-integrity ladder + the SDO gate going LIVE** | The 3-tier rating ladder + coherent significance; then the SOTA-Dominance-Objective gate becomes a first-class, code-computable completion criterion. | SO-08 · SO-09 |
| **The 3 security MUSTs + adversarial closes** | Sign the control-path gate artifact; encrypt the token store; sandbox the attack harness — hardened under live break-probes that caught a real sandbox escape a 5/5 gate missed. | SO-10 |
| **The G2/G4 guarantees + fabrication hardening** | Two SDO guarantees made code-computable; an adversarial close caught a no-fabrication MAJOR in *both* gate functions. | SO-11 |
| **The MoE-router core** | Learned, feature-conditioned routing behind a fail-closed, advisory, recall-floor-bounded gate; the SLA selection-bias mechanism. | SO-12 · SO-13 |
| **Phase-B audit surface** | The agentic interception ledger + leakage-Sankey + the re-identification attack seam — the FEATURE half of the latency/audit guarantee. | SO-14 |
| **★ The S7-02 keystone** | The certified canonical run. A 7-round adversarial close found the *inherited* SDO gate deeply unsound — 11 holes / 5 fabrications. The close, not the audit, is the only reliable certifier. | SO-15 |
| **G5 — the last placeholder** | The latency + audit guarantee goes computed; a 6th fabrication (a breach-bury) caught in round 1. Every guarantee now computes a real value on a certified run. | SO-16 |
| **The feature surface** | Tier-3 re-id + MIA adversaries; query-aware masking; the BYO-pipeline SDK; native-format readers (zip-bomb catch); multilingual fairness (dead-CJK-keyword activation); docs with teeth (phantom-filename catch). | SO-17 … SO-23 |

The arc is unusual in one respect worth stating plainly: the *hardest, most valuable* work was
not building features — it was building a gate that **cannot lie**, and then proving, round
after adversarial round, that it could not. That story is the keystone (SO-15), and everything
before it is the foundation the keystone rests on.

---

## 1. The starting condition — a mature library with an honesty problem (M1, SO-01)

pii-anon arrived at this program not as a greenfield idea but as a **published v1.4.0 dual-pillar
privacy library**: a `pii-rate-elo` evaluation framework and a detection + anonymization /
pseudonymization engine (a fast regex path plus a 4-layer "swarm"). It was, by code metrics,
*unusually mature for its age* — ~37,000 lines of Python, ~2,548 test functions, strict `mypy`,
a full CI gate chain across an OS × version matrix (brownfield assessment §1–2). The maturity
was in **code and CI, not in formal PDLC prose** — exactly the brownfield profile where this
program's value is *formalizing the implicit and closing the gaps*.

The brownfield assessment (SO-01) found **12 MAJOR findings, 0 SHOWSTOPPER, 0 CATASTROPHIC**
(assessment §3). Four of them were the load-bearing problems the entire program would go on to
solve:

1. **The published benchmark was an uncertified 50-sample smoke run.** `benchmark-results.json`
   carried `max_samples=50`, `canonical_claim_run=False`, `floor_pass=False` — yet the README
   published a headline F1 and throughput as a "Production/Stable" claim. *The harness itself
   refused to certify the numbers it was being cited for* (assessment §3, MAJOR-1).
2. **The statistical-significance reporting was internally incoherent.** Every pairwise
   comparison read "n.s." at p ≈ 0.49–0.50 even where Cohen's *d* was large, while 95% CIs were
   ~0.002 wide and several did not bracket their own point estimate — a computation that *cannot*
   be all three at once on ~149K samples (assessment §3, MAJOR-2).
3. **The recall-floor axiom (AX-003) was NOT guaranteed by construction.** Two divergent floor
   mechanisms existed in the pipeline; a shared-layer finding could be silently dropped by the
   swarm's emission gate (`swarm.py:651–661`) (assessment §3, MAJOR-5).
4. **The agentic-interception axiom (AX-006) had zero realization** — a declared axiom with no
   module, port, or workflow in `src/` (assessment §3, MAJOR-8).

The steering decision at the M1 close set the program's whole tone: the eval-integrity findings
(smoke-run numbers, incoherent significance, presentation) were folded into the overhaul *as a
first-class redesign of how the library computes and presents results* — **not a pre-PDLC
hotfix** (SO-01, `decisions_acknowledged`). All current README/benchmark numbers were declared
**PROVISIONAL** until a certified canonical run and a corrected significance pipeline landed.
That declaration — *we will not claim what we cannot certify* — is the seed of the no-fabrication
invariant that becomes the program's load-bearing discipline.

Six project axioms were authored (AX-001 no-real-PII, AX-002 determinism, AX-003 recall-floor,
AX-004 anon≠pseudo, AX-005 calibration, AX-006 agentic least-privilege). AX-003 was named the
load-bearing Theme-1 invariant. The migration preserved all 24 legacy originals in place (0
deletions, deviating from the move-to-archive default by user instruction) (SO-01).

> **Honesty boundary, stated at the source.** SO-01's signer field is `AGENT_SIMULATED`. Every
> sign-off in this program carries that marker. The "user" is the documented **Pass-2 cohort** —
> the program ran autonomously at representative scale; no real-user research, NPS, or adoption
> evidence was collected (assessment §3, MAJOR-10). This journey describes an **agent-run PDLC**,
> not a real-user-validated product launch.

---

## 2. Discovery, Requirements, Design — a measurement-first pivot in one day (SO-02 → SO-04)

All three upstream stages closed on **2026-05-30**, the same day as the brownfield assessment —
an autonomous, single-session pass at representative scale.

**Discovery (SO-02)** produced the pivot that defines the product's competitive identity. All
three POV critics held-with-changes on a **measurement-first** framing: the `pii-rate-elo`
evaluation pillar is the headline; **pseudonymization-integrity scoring is the defensible empty
quadrant** that no public benchmark (RAT-Bench 2026, TAB, PIIBench, PrivaCI) occupies; and the
swarm is **re-scoped off the raw-F1 arms race** (where OpenAI's Privacy Filter at F1≈0.96 and
Presidio already compete) and onto *reversibility, recall-floor-by-construction, audit, and
orchestration*. Seven personas (two eval-dedicated), 28 use cases (15 eval / 12 swarm / 1 both).
The concept-value study is explicitly AGENT_SIMULATED (SO-02, `methodology.scale`). This pivot is
the reason the program can land honestly at NOT_YET on raw F2 and still ship: *raw-F1 supremacy
was never the claim.*

**Requirements (SO-03)** authored **39 FRs + 26 NFRs**, evaluation-led and co-equal, with 0
orphans. The eval-integrity foundation — FR-003 (Bayesian Bradley-Terry), FR-004 (coherent
significance), FR-008 (canonical-run), NFR-001/002/006 — was marked MUST and the critical path.
The headline novelty became MUST FRs: the pseudonymization-integrity 5-axis family (FR-009), the
anon-vs-pseudo no-merge invariant (FR-010), and a real Tier-3 LLM-adversary (FR-011). Six NFR
thresholds were stress-tested; **0 DIVERGED**. SME caveats were encoded structurally — EDPB Art
4(5) framed as an *engineering proxy* not a settled legal verdict (NFR-015), Tier-3 circularity
controlled (FR-012). Latency NFRs were kept **HONEST** (no sub-0.24ms parity claim) (SO-03).

**Design (SO-04)** produced **15 Design Cases** and three headline decisions via a Pugh
DIVERGE/CONVERGE, plus one resolved CATASTROPHIC:

- **DECISION 1 (Pugh A, 8.4):** the `SharedLayerProjector` — ONE post-fusion chokepoint making
  `entities(ensemble) ⊇ entities(shared)` *by construction* (the AX-003 fix), plus a
  `DistilledTopKGate` (an advisory, artifact-gated meta-learner) and rules-first depth-1
  early-exit.
- **DECISION 2 (Pugh A, 8.6):** a `RatingEnginePort` + 3-tier ladder (glicko-legacy → MLE-BT →
  Bayesian-BT NUTS as the claim-grade default), with **significance coherence by construction
  from one joint posterior**. This decision *resolved the CATASTROPHIC eval-01 finding*: only the
  Bayesian engine clears NFR-001's MCMC diagnostics (R̂ ≤ 1.01, ESS ≥ 400/param, 0 divergences),
  so the frequentist tiers are smoke/fallback only.
- **DECISION 3 (Pugh A):** agentic interception via a router pre-filter + query-aware gate +
  unified floor + 4-channel least-privilege (AX-006).

Five SME reviewers issued REQUEST_CHANGES carrying findings forward into Development as
**MUST-address** items — including three **Security MAJORs** (sign the gate artifact, encrypt the
token store, sandbox the attack harness) that became the SO-10 work-stream, and three **Docs
MAJORs** (surface the distinct anon-vs-pseudo APIs, update the evaluate-your-pipeline doc, fix the
divergent recall-floor docs) that became the SO-23 work-stream (SO-04, `findings_carried_forward`).

> **Honesty boundary.** The 5-SME panel is an *agent-simulated heuristic evaluation*, not human
> SME review (SO-04, `methodology`). The 0-DIVERGED thresholds and the resolved CATASTROPHIC are
> real artifacts of that process — but the validation is agent-run, not human-validated.

---

## 3. The Sprint-1 foundation — the recall floor, by construction (SO-05 → SO-07)

The first real code was the single most load-bearing MUST in the program: **recall-floor by
construction** (FR-016 / NFR-011 / AX-003). `SharedLayerProjector` (`routing/shared_layer.py`)
enforces `entities(output) ⊇ entities(shared)` structurally and closes the `swarm.py:654/658–660`
drop leak. Shipped under strict TDD (RED `ef85166` → GREEN `548f576`), 7/7 green including a
2,000-case property suite with 0 violations, `mypy --strict` + `ruff` clean, with **no
public-API change and 78 adjacent swarm/fusion/moe tests unaffected** (SO-05, `scope_done`;
release-readiness-report `## Evidence`).

The Stage-5 **release-readiness verdict** at this point was deliberately split and honest:
**SHIP-WITH-CAVEATS** for the recall-floor foundation (real, tested, non-regressing, discharges
the load-bearing MUST by construction) and **DEFER** for the full 4-theme redesign (~29 stories
remaining, several blocked on the sibling `pii-anon-eval-data` track). The NFR matrix read 2
VERIFIED + 2 PARTIAL + 22 DEFERRED + **0 FAIL** (SO-06; release-readiness-report `## Verdict`).
This SHIP-WITH-CAVEATS verdict is itself a small instance of the program's discipline: ship what
is genuinely done, defer honestly what is not.

**Sprint-1 then closed in full (SO-07).** Five stories (S1-01…05) made the floor *live on the
production path*: `FloorProjectingFusion` wraps both the swarm and mixture-of-experts strategies
at the `build_fusion` seam; a per-language recall ε-gate (ε ≤ 0.005) with a teeth-verified
regression guard; property coverage migrated to Hypothesis `@given` (closing the brownfield's
"zero `@given`" finding). The sprint-close ran an **11-agent verification workflow** (6 reviewer
dimensions + 5 adversaries); the adversarial run upheld **0 of 5 refutations**. The full suite was
2,690 passed / 0 failed at 86.22% coverage (SO-07). This is the first appearance of the pattern
that recurs through the whole program: *a multi-reviewer gate, then an adversarial close on top
of it.*

---

## 4. The eval-integrity ladder, and the SDO gate going LIVE (SO-08 → SO-09)

With the floor live, the program turned to the **critical path** the brownfield surfaced: making
the evaluation pillar's numbers *trustworthy*.

**SO-08 — the rating ladder.** The 3-tier ladder went live behind the `S3-01 RatingEnginePort`:
glicko-legacy → Bradley-Terry MLE (pure-stdlib MM / Hunter-2004 + paired bootstrap) → Bayesian-BT
(NumPyro NUTS, the claim-grade default). A structural `Protocol` meant **zero call-site changes**
— `elo.py` and 7 callers stayed byte-identical. `convergence.py` is the pure-numpy NFR-001 gate
(split-R̂ ≤ 1.01 ∧ bulk-ESS ≥ 400/param ∧ 0 divergences) that **fails loud and names the binding
constraint** — formally resolving the SME CATASTROPHIC eval-01: only the Bayesian engine is
claim-grade. The architecture was *env-honest* — numpyro/jax/arviz are not installed in the venv
(heavy platform wheels), so the module is import-safe with a lazy NUTS import; calling without the
Bayesian extra raises a loud `MissingOptionalDependencyError` rather than silently falling back to
a non-Bayesian engine (SO-08, `decisions_acknowledged`).

**SO-09 — the SDO gate goes LIVE.** This is the structural turning point. S3-04 (coherent
significance *by construction* + the `rank_one_probability` J primitive + Davidson ties) and
S4-CS-01 (the `CompetitiveSupremacyGate`) made the **SOTA-Dominance-Objective** a first-class,
code-computable gate. `pii-anon supremacy` now prints exactly one verdict
{`CLAIM_GRADE_SOTA` | `PROVISIONAL_SOTA` | `NOT_YET`} plus the single binding constraint, every
run. Coherence is by construction: point estimate, CI, sign, and verdict all derive from **one
joint-posterior difference vector**, so the three significance statements *cannot disagree* — the
brownfield's incoherent-significance MAJOR is closed at the structural level (SO-09,
`decisions_acknowledged`). The `RecallFloorVerdictGuard` (AX-003) was implemented, not deferred: a
recall-floor-breaching system can never top-rank.

The honest verdict at this landing: **NOT_YET**, binding constraint `canonical_claim_run=False`
(G7) — *the published numbers are still a 50-sample smoke run*. Guarantees read G1 PENDING, G2
PENDING (←S4-01), G3 PASS, G4 PENDING (←S4-03), G5 PENDING, G6 PASS, G7 FAIL; J = 1.0 from the
in-tree MLE-bootstrap fallback. The verdict was **quadruple-verified** (CLI + hand-recompute + two
independent reviewer reproductions, all agreeing) (SO-09, `sdo_verdict`). The #1 binding objective
was named: the S7 canonical run.

---

## 5. The three security MUSTs, and the first adversarial close that earned its keep (SO-10)

The three Design-stage Security MAJORs were closed and hardened together (SO-10): **S2-05** (sign
+ verify the MoE control-path `gate_v1.json` under detached HMAC-SHA256 with fail-loud
verify-on-load), **S6-03** (encrypt the reversible-pseudonymization token store at rest under
AEAD, the brownfield's plaintext-PII MAJOR), and **S5-04** (sandbox the adversarial attack-harness
runner under capability + resource isolation). Each passed the canonical 5-reviewer story gate.

Then the methodology that becomes the program's spine asserted itself. A **between-work-streams
adversarial close** ran live break-probes against all three and **caught a MAJOR all five iter-1
reviewers missed**: `SEC-S504-PATHTRAVERSAL` — the S5-04 path allow-list used a purely lexical
`PurePosixPath.relative_to` that never collapses `..`, so a `/grant/../../../etc/passwd` target
with `allowed_paths={'/grant'}` was *returned*, defeating both the runtime shim and the load-time
guard (demonstrated reading an `/etc/passwd`-shaped target plus a real symlink escape). The `..`
case had no test (SO-10, `adversarial_close.major_caught`). It was remediated under strict TDD
(path canonicalization via realpath/normpath) and a focused RE-CLOSE returned 0 still-broken.

The lesson, recorded in the sign-off and carried forward verbatim through the rest of the
program: **a per-story 5-reviewer gate is necessary but not sufficient for security-MUST and
control-path work; the adversarial close (live break-probes) is the catch-net** (SO-10,
`decisions_acknowledged`). It found a real sandbox escape after a clean 5/5 gate. Keep it standing.

The SDO verdict stayed **NOT_YET** — the security MUSTs harden posture but write no benchmark
artifact and flip no guarantee (SO-10, `sdo_verdict`). *(All real crypto-key / KEK / HSM custody
and real OS-level sandbox syscalls are Pass-2 — never agent-simulated as real; the in-tree HMAC +
AEAD + in-process guards run for real against deterministic non-production test keys.)*

---

## 6. The G2/G4 guarantees, and a no-fabrication MAJOR in both gate functions (SO-11)

Two SDO guarantees moved from hardcoded `_pending_guarantee(...)` placeholders to real,
three-valued, teeth-tested functions: **G2** (pseudonymization-integrity strict dominance, via
S4-01's distinct anon-vs-pseudo scorers + a no-merge CI guard — AX-004 enforced structurally) and
**G4** (calibration / selective-risk, via S4-03's per-class ECE/Brier/AURC + abstention reporter).
Each passed the 5-reviewer gate (SO-11, `stories_closed`).

The adversarial close then **caught a no-fabrication MAJOR in *both* gate functions** that the
per-story gates missed:

- **G2:** `_g2` used `max(competitor_pis, default=0.0)`. A SUT-only scorer (pii-anon carries a
  pseudonymization-integrity score; *no competitor does, the natural shape, and no contract test
  required it*) would vacuously "strictly dominate" a **phantom 0.0** → false PASS → CLAIM_GRADE.
- **G4:** `_g4` trusted artifact-supplied values — a NaN per-class ECE slipped `ece > bar`; a
  coverage > 1.0 / +inf was admitted by `>=`; an artifact-supplied threshold loosened the bar
  unclamped → false PASS → CLAIM_GRADE (SO-11, `adversarial_close.major_caught`).

The remediation introduced the validators that become the program's load-bearing no-fabrication
discipline: `_finite_unit_score` (rejects bool — *Python's bool is an int* — non-finite, and
out-of-[0,1] values, via the `_is_finite_number` TypeGuard), a G2 no-real-comparator → PENDING
guard (no phantom-0.0 win), a tighten-only ECE-threshold clamp (`min(artifact, 0.08)`), and a
coverage that must be finite ∧ [0,1] ∧ == 1.0. Sixteen RED vectors that FALSE-PASSed before the
fix read PENDING/FAIL after it (git-verified ancestor), and a RE-CLOSE confirmed 0
fabrication-possible (SO-11, `adversarial_close.remediation` + `reclose`).

The verdict stayed **NOT_YET** — both guarantees honestly read **PENDING-on-missing-fields** on
the smoke artifact (the false-PASS shapes require the S7 canonical run's artifact; they were
*fixed before they could matter*). The principle was stated in the sign-off: **the gate is the
program's completion criterion — it must be un-fabricatable** (SO-11, `decisions_acknowledged`).
The adversarial close had now earned its keep **twice** (the S5-04 path-traversal and these two
fabrication holes).

---

## 7. The MoE-router core, and the latency-bias mechanism (SO-12 → SO-13)

With the gate hardened, the program built out the **learned-routing core** of the swarm (DC-02).
**SO-12:** S2-01 widened `MoERouter.route()` to feature-conditioned routing and added an additive,
byte-for-byte backward-compatible v2 fusion-construction seam (the three Design-stage seam-
correction API-MAJORs); S2-02 shipped the runtime `DistilledTopKGate` — entered *only* through the
S2-05 fail-closed verify-on-load boundary, advisory, and structurally unable to drop a floored
span — plus the offline distillation trainer that emits the **signed** `gate_v1.json`. The
control-path artifact got an adversarial close (verify-bypass, forge/fabricate with 35+
malformed-but-signed payloads, and a 400-trial floor-defeat fuzz against the real
`FloorProjectingFusion`): **0 upheld**, with the verify-on-load entry-gate, the no-fabrication
shape validator (defense-in-depth: *the validator, not the signature, rejects NaN/Infinity that
survive json+HMAC*), and the advisory-floor bound all holding under live attack (SO-12,
`adversarial_close`). A clean close still earned its keep — it independently confirmed the
invariants.

Two stories were honestly deferred and remain so: **S2-03** (rules-first early-exit) is
**BLOCKED** because its only viable hook lives in `orchestrator.py`, which is the user's protected
WIP this entire program (re-checked byte-identical at every sign-off); **S2-04** moved to the next
session (SO-12, `stories_deferred`).

**SO-13:** S2-04 then landed the **aux-loss-free SLA selection-bias** (DeepSeek-V3 style,
selection-logits only) — a per-expert latency penalty that biases routing toward faster experts,
advisory and selection-only (never mutates confidence, never alters shared-layer membership, never
drops a floored span). The story gate ran iter-1 REQUEST_CHANGES with **two MAJORs that were the
same defect class** found on two independent ingress paths: a `reference_ms=0.0` →
ZeroDivisionError and a `latency_cost_ms=10**400` → OverflowError in `math.isfinite` (an int
wider than a C double), both meaning *the numeric ingress was not total over hostile input*. One
hardening pass closed both — exactly the robustness hole a multi-dimension gate is meant to
surface (SO-13, `story_gate`). The "power-tier trap" was structurally avoided: the bias reads only
`metadata['latency_cost_ms']` (NFR-009/010 latency) and *never* `default_weight` (the NFR-004
statistical-power signal), which the design's phrasing had conflated.

The verdict stayed **NOT_YET** through both — a swarm-routing feature writes no benchmark artifact
and flips no guarantee (SO-12 / SO-13, `sdo_verdict`).

---

## 8. The Phase-B audit surface — interception, leakage, and the attack seam (SO-14)

Three FEATURE modules landed together as the **audit half** of the as-yet-PENDING G5 guarantee
(SO-14): **S6-02** (the 4-channel least-privilege `FourChannelGuard` + the no-raw-PII-persist
invariant + the `InterceptionLedger` audit artifact), **S5-01** (the `ReidAttack` / `MiaAttack`
Protocols + a deterministic baseline + the attacks import-boundary CI test + the non-strippable
NFR-016 anti-anonymity caveat), and **S6-05** (the leakage-Sankey flow graph + prompt-injection
exfiltration resistance).

The work-stream gate caught and remediated a sharp one in S6-02: the iter-1 surrogate used a
**keyless BLAKE2b** that was *demonstrably dictionary-reversible* — meaning the G5 audit ledger
was not actually de-identified — fixed to a keyed HMAC-SHA256 surrogate, with the security
reviewer re-running its own break-probe to confirm it resists (SO-14, `work_stream_gate_summary`).

Two process facts from this sign-off matter for the rest of the story. First, **the worktree-
isolated parallel-dispatch mechanism proved unreliable in this environment** (it mis-allocated
both worktrees to a stale base; one executor self-healed, the other correctly refused) → **all
remaining stories run sequentially in the main tree** (SO-14, `decisions_acknowledged`). Second,
FR-011/FR-013 were honestly tracked as a *protocol foundation* only — the real Tier-3 LLM
adversary and real LiRA@128 MIA were named Pass-2, never over-claimed as discharged.

The verdict stayed **NOT_YET**: G5 becomes code-computable only when the S7 canonical run emits the
audit + latency fields *and* the gate's `_g5_*` method is wired (a tracked follow-up, exactly
analogous to S4-01/S4-03 for G2/G4) (SO-14, `sdo_verdict`).

---

## 9. ★ The S7-02 keystone — the certified run, and the gate that turned out to be unsound (SO-15)

This is the center of the whole program, and it is the section to read if you read only one.

The **keystone** is the canonical-run *producer* (`produce_canonical_artifact`) plus the
fail-closed `CanonicalRunGate` (`evaluation/canonical_run.py`). The producer runs **real
detection** (via `compare_competitors`) at representative scale and emits a **certified** artifact
carrying the G1 per-language ε, the G2 distinct anon/pseudo family fields (for *every* system,
including competitors — the SO-11 fix, so G2 has a real comparator), the G4 calibration block, and
the G7 provenance stamp — all **reusing the existing in-tree scorers**. The fail-closed gate sets
`canonical_claim_run=True` *only* when every required field is present-and-valid (SO-15, `scope`).
This is what finally makes the #1 binding constraint — a certified run — *manufacturable* rather
than merely demanded.

The story passed its 5/5 gate. Then the **mandatory adversarial close ran SEVEN ROUNDS**
(close-4 through close-10) and uncovered that the **inherited SDO gate (`competitive_supremacy.py`)
was deeply unsound** — it forged the terminal `CLAIM_GRADE_SOTA` (the *highest* verdict) through
nearly every axis. **11 holes / 5 fabrications, including 1 CATASTROPHIC and 2 SHOWSTOPPERs, plus
6 crashes** (SO-15, `work_stream_gate_summary`). The fabrications, each closed RED→GREEN:

- **close-5 — G7 provenance fail-OPEN:** a truthy-but-invalid stamp (whitespace `'   '`, an int,
  a bool) was read as "present" → forged G7 PASS → PROVISIONAL_SOTA. Fixed with `_is_nonblank_str`.
- **close-6 — `canonical_claim_run` coercion:** `bool(...)` coerced the *string* `"false"` /
  `"no"` / `"0"` to True → forged certified run. Fixed with strict `is True` at both read sites.
- **★ close-7 — CATASTROPHIC NaN-curve:** a single NaN row in a `risk_coverage_curve` laundered a
  *non-monotone* curve into a "monotone" G4 PASS → `CLAIM_GRADE_SOTA` (NaN compares False to
  everything, corrupting both the sort and the `>=` monotonicity check). Fixed by validating each
  row via `_is_finite_number`.
- **★ close-9 — two SHOWSTOPPER fabrications:** a bool / +inf / huge-int value inside a *nested*
  `per_entity_recall` dict counted as a "detected" entity — inflating G6 entity-coverage past the
  0.80 bar AND masking a real G1 ensemble miss. Both fixed by `_detected_entity_names` (a detection
  requires a valid [0,1] score > 0) (SO-15, `fabrications_closed`).

Two findings from this close are the most important sentences in the entire program, and they are
recorded verbatim in the sign-off:

> **★ THE CLOSE IS THE ONLY RELIABLE CERTIFIER — NOT the agent's manual audit.** The agent
> *explicitly dismissed both close-9 SHOWSTOPPERs as "not a fabrication" before the close caught
> them.* S7-02 was declared DONE only on a 0-upheld close where the fabrication-pass refuter
> *completed* — never on an audit (SO-15, `decisions_acknowledged`).

> **★ NO-FABRICATION is the program's LOAD-BEARING invariant.** *Every* artifact value read in the
> gate — top-level **and nested** (`per_entity_recall` rows, `risk_coverage_curve` rows,
> `per_class_ece`, provenance, the canonical flag, scores, deltas) — must route through a
> validator. A single un-validated read forges a moat-axis PASS. The nested reads were the blind
> spot a top-level audit missed (SO-15, `decisions_acknowledged`).

Round-7's fabrication refuter had *errored without emitting structured output* — so the fabrication
class was unconfirmed and the close could not be declared passed. Round-8 split the fabrication
hunt into flat-scalar + nested-value refuters with an explicit structured-output mandate; **both
completed**, and the confirmatory round-8 close (`wf_3239f1fa-0c4`) returned **RECLOSE_PASS — 0
upheld / 517 probes**, off-limits files byte-identical (SO-15, `final_close`).

A producer-correctness fix also landed: the `_BENCHMARK_IGNORE` sentinel leaked into the emitted
`per_entity_recall` (a competitor "detects" it, the ladder does not), causing a spurious G1 FAIL on
a non-entity; stripping it from the *emitted* copy only (source/composite untouched, so composite
and J stay byte-identical) auto-corrected G6 coverage 0.778 → 0.824 (SO-15, `producer_correctness_fix`).

**The honest endpoint of the keystone.** The freshly *produced* certified artifact reads
**NOT_YET**, `canonical_claim_run=True`, with the binding constraint now the **honest,
draw-sensitive raw-detection axis G6**: core F2 = 0.7214 vs the best Tier-R competitor (gliner) F2
= 0.75 (ε_F = 0.01); entity coverage 0.824 (≥ 0.80). Guarantees: G1 PASS, G2 PASS, G3 PASS, G4
PASS, G5 PENDING (honest — that is S7-04), G6 FAIL, G7 PASS; J = 0.0 at representative scale
(SO-15, `produced_canonical_artifact`).

**And this is the through-line of the whole program: the G6 FAIL is NOT a regression** (decision
A, `f2-gap-attribution.md`). Old code (`2761a27`) is *byte-identical* to current HEAD at
`use_case=default` — the detection path additions changed nothing measurable. pii-anon *wins
rank-1 on the `pii-rate-elo` COMPOSITE* (0.7035 > gliner 0.6643 — its operational latency /
throughput + entity-coverage moat) but loses raw F2 to gliner on the *current* non-deterministic
dataset draw (census sha `044fec59` → current `abfe651d`). **G6 is draw-sensitive** — pii-anon and
gliner trade raw-F2 leadership between corpus draws (it PASSED on the census draw, 0.779 > 0.697,
and FAILS on the current draw). The diagnostic's verdict, carried forward as program policy: *do
NOT "fix" F2 and do NOT weaken G6 — continue to RC accepting the honest NOT_YET*. Raising raw F2 to
robustly clear G6 on every draw is a detection-quality *enhancement* (a Pass-2 product item), not a
regression fix, and **changing a gate axis to force a PASS is exactly what the no-fabrication
invariant forbids** (f2-gap-attribution `## Refined conclusion`; SO-15, `decisions_acknowledged`).

---

## 10. G5 — the last placeholder, and the 6th fabrication (SO-16)

S7-04 made **G5 the last `_pending_guarantee` placeholder to go computed**, modeled exactly on how
G2/G4 went placeholder → computed: (1) a new committed NFR-009 latency-ceiling registry
(`latency_ceilings.py`, stdlib-only, frozen, per-objective p50/p95/p99; the full-swarm ensemble
budget is a real **500/1000/2000 ms** — sub-2s p99 with ~6× quiet headroom, *not* a sub-0.24ms
parity claim and not vacuous); (2) the gate's G5 placeholder replaced by `_g5_audit_latency` =
audit-pass ∧ latency-within-ceiling, three-valued, every artifact value validated; (3) the producer
emits real measured full-swarm per-record latency (min-of-3, contention-robust) plus the real
S6-02/S6-05 audit surface under a seed-derived key (SO-16, `scope`).

Because this touched `competitive_supremacy.py`, the **mandatory adversarial close** ran — and
round 1 caught a **6th fabrication**: a **breach-bury MAJOR**. A *real* breach in the present G5
half (over-budget latency or a leaked span) was buried into PENDING by omitting the *other* block,
escalating the verdict NOT_YET → PROVISIONAL_SOTA (PENDING blocks CLAIM_GRADE but not PROVISIONAL).
The breach-outranks-missing rule only fired when *both* blocks were present dicts; whole-block
absence short-circuited to PENDING before the combiner ran. The fix made an absent block a
missing-shape `_G5Half(ok=None)` so the rule applies at the whole-block level too (SO-16,
`fabrications_closed`). The recorded lesson:

> The breach-bury was the **same root cause as the close-9 nested-value SHOWSTOPPERs, one level
> up** — a correctness rule implemented for the inner case but not the outer whole-block case. A
> correctness rule must hold at **every level it applies.** The close caught it; the 5/5 story gate
> (which had APPROVED the buggy code) did not (SO-16, `lesson`).

The confirmatory round 2 (`wf_4c4df480-634`) returned **CLOSE_PASS — 0 upheld / 764 probes**.

**The completeness milestone.** The freshly produced certified artifact now reads NOT_YET,
`canonical_claim_run=True`, with **G1 · G2 · G3 · G4 · G5 · G7 ALL PASS and ZERO pending axes** —
*every guarantee now code-computes a real value on a certified run*. The binding constraint remains
the honest, draw-sensitive G6 (SO-16, `produced_canonical_artifact`). **G5 PASS does NOT flip the
verdict — G6 binds.** G5 is the program's completeness milestone, not a verdict flip. A deliberate
line was drawn: an over-budget-but-shape-valid latency still *certifies* the run and then G5
honestly FAILS at the gate (exactly like G6's F2 on a certified run); but an **audit-integrity
breach** (a leak, a corrupt persist stamp, an exfiltration) *refuses* certification — the artifact
itself is untrustworthy (SO-16, `decisions_acknowledged`, the certification-vs-performance line).

---

## 11. The feature surface — six stories, three sharp catches (SO-17 → SO-23)

With every guarantee computing, the program built out the remaining feature surface — **sequential
in the main tree** (worktree parallelism having proved unreliable). None of these touch
`competitive_supremacy.py`, so none required the SDO adversarial close; each passed the standing
5-reviewer story gate. *Every one flips no SDO guarantee — the gate stays byte-identical (md5
`3b842e81…`) across all six.*

- **SO-17 — S5-02 Tier-3 re-id adversary.** A materially stronger (still deterministic) in-tree
  stand-in for the real Tier-3 LLM adversary: QIC (quasi-identifier-combination) overlap fused with
  a background-signal-linkage prior, **de-circularized** via a margin-based commit/abstain (FR-012 —
  it commits only on a clear margin and *never consults the gold link*). Plus the NFR-012 power
  half: a Wilson score interval + the 2-rung paired-persona ladder (≥385 / ≥897). The recurring
  *exact-rate-anchor* lesson appeared again (RC-02: anchor the representative-metric test at an
  exact textbook value, not a bound) (SO-17).
- **SO-18 — S5-03 representative MIA.** A LiRA-shape membership-inference adversary + Secret-Sharer
  + TPR@low-FPR, scoring membership from attacker-observable signal only (de-circularized — FR-013).
  `tpr_at_fpr` returns 0.0 (not a fabricated TPR) when no threshold achieves the target FPR. The
  decode boundary was hardened to fail loud on a non-finite loss / non-bool gold label. The privacy
  attack surface is now **complete in representative form** (re-id + MIA, both with power machinery
  and the non-strippable caveat) (SO-18).
- **SO-19 — S6-01 query-aware masking.** A gate that decides per-span retain-vs-mask, **subtractive-
  on-mask and DEFAULT-TO-MASK**: it retains a span *only* on a positive reason-stamped relevance
  signal, so **false-retention (the leak) cannot occur by default** — over-redaction is the safe
  error. Confirmed under live adversarial probing (empty / whitespace / NBSP / zero-width queries
  all mask everything). Shipped as a standalone pure primitive; the orchestrator router-pre-filter
  wire-in is the S2-03 block again, honestly flagged Pass-2 (SO-19).
- **SO-20 — S6-04 the BYO-pipeline SDK.** FR-001's missing SDK half (a MUST): a third-party package
  can now advertise a `Predictor` under the new `pii_anon.byo_pipelines` entry-point group and be
  discovered with zero harness-core edits — and the five in-tree incumbents are scored through the
  **literally same function** (`evaluate_incumbent` is a single delegation with no scoring logic of
  its own), so BYO and incumbent paths are identical *by construction*. An honesty note ships in the
  module and docs: identical-path numbers will legitimately *differ* from the frozen legacy
  benchmark rows — FR-002 pins the *path*, not retroactive artifact parity (SO-20).
- **SO-21 — S7-01 native-format readers, and the zip-bomb catch.** PDF / image / screenshot / DICOM
  / audio behind the same `Iterator[IngestRecord]` contract (FR-031); one *real* extractor ships
  (the pure-stdlib PDF text reader), the rest are capability-honest (they report
  `dependency_available` truthfully and **raise loudly naming the install extra — never silent
  empty text**, which would silently drop every PII span in a modality). **The story gate earned its
  keep at iter-1:** the security reviewer caught an **unbounded FlateDecode inflate on untrusted PDF
  bytes — a zip-bomb memory-DoS** (~1000× amplification; a ~200KB stream inflated to ~200MB, peak
  RSS ~459MB). Fixed with a chunked decompress loop under a 64 MiB per-stream ceiling, boundary-exact
  (== ceiling passes, +1 raises). A real security hole on the exact untrusted-bytes path that is
  UC-24's whole point, caught before any close (SO-21, `findings_remediated_in_loop`).
- **SO-22 — S7-03 multilingual fairness, and the dead-keyword activation.** A genuine RED: the
  ZH/JA/KO/AR context keywords the library had advertised since its multilingual expansion were
  **provably DEAD** — the `[A-Za-zÀ-ÿ]+` tokenizer cannot produce a non-Latin token, and
  the containment fallthrough the code's comment *promised* was never implemented. S7-03 activates
  them (keywords with zero Latin runs match by substring containment after the token-set miss; every
  Latin keyword stays token-path-exclusive, so ASCII behavior is byte-identical). Plus a
  **fail-closed** powered worst-group fairness gate (FR-039 / NFR-025): zero or one powered group can
  *never* PASS — an unpowered cohort cannot fabricate a fairness claim. The canonical-run wire-in is
  deliberately Pass-2 *because* it would make the verdict a control-path field and mandate the
  adversarial close — **the deferral is the discipline, not an omission** (SO-22). This sign-off
  marks the **feature surface COMPLETE** — every story on the keystone's feature list (S5-02/03,
  S6-01/04, S7-01/03) is DONE.
- **SO-23 — S7-05 docs discoverability, and the phantom-filename catch.** The LAST feature story,
  closing the three D6 SME Docs MAJORs: the new `docs/anonymization-vs-pseudonymization.md` (the two
  scorer families + the AX-004 no-merge invariant verbatim + vanilla-vs-swarm positioning), the
  updated evaluate-your-pipeline doc, and the verified-live recall-floor doc. The standing teeth are
  `tests/test_docs_discoverability.py` (index completeness, zero broken links, headline-symbol
  discoverability, CLI-help pins). **The gate caught a real docs-accuracy defect a substring test
  could not:** the certify-a-run example was **not copy-paste-runnable** — step 2 read
  `./certified/benchmark-results.json`, *a file the canonical-run producer never writes* (its sole
  emission is `<output-dir>/canonical-run.json`); verbatim execution raised `BadParameter`. The fix
  corrected the path *and upgraded the teeth* so the whole class of defect (example-path drift) is
  now test-caught. A bonus fix repaired a `make docs-smoke` that had been broken since before the
  program (a stale notebook path that never existed in git) (SO-23, `findings_remediated_in_loop`).

Across all six, the SDO verdict on the committed smoke artifact stayed **byte-stable NOT_YET** — a
privacy-attack, masking, SDK, reader, fairness, or docs feature flips no guarantee (SO-17 … SO-23,
`sdo_verdict`).

---

## 12. The honest landing — where the program actually stands

There are **two honest verdicts**, and conflating them is the one thing the docs must not do
(SO-15 / SO-16, `sdo_verdict`):

| Artifact | Verdict | Binding constraint | Guarantees |
|---|---|---|---|
| **Committed smoke artifact** (`pii-anon supremacy` default) | NOT_YET | `canonical_claim_run=False` (G7) | G1·G2·G4·G5 PENDING; G3·G6 PASS; G7 FAIL; J=1.0 |
| **Produced certified artifact** (the keystone's teeth) | NOT_YET | **G6 FAIL** (core F2 0.7214 vs gliner 0.75; coverage 0.824) | **G1·G2·G3·G4·G5·G7 ALL PASS**; G6 FAIL; J=0.0 |

The committed smoke artifact predates the producer and carries no certified provenance — so its
binding constraint is honestly G7 (no certified run). The *produced* certified artifact is the
keystone's output: it manufactures the certified run, flips G7 to PASS, makes every other guarantee
compute a real value, and lands on the one axis the program will not fake — **G6, raw-F2
non-inferiority, which is draw-sensitive and currently FAILS** because pii-anon's robust win is the
*composite* (its operational moat), not raw F2 on the hardest mixed slice (f2-gap-attribution
`## Refined conclusion`).

The NFR matrix reads **2 VERIFIED + 2 PARTIAL + 22 DEFERRED + 0 FAIL** (release-readiness-report
`## Verdict`). The full suite at the final feature close was ~3,800 passed / 0 failed at 88.87%
coverage, `ruff` + `mypy` clean under both invocation modes (SO-23, `quality`).

**This is the program's defining choice, and it is worth naming once more.** Faced with a gate that
forged the highest verdict through nearly every axis, the program did not weaken the gate to make
the claim true — it *hardened the gate until it could only tell the truth*, then accepted the truth
the hardened gate returned: **NOT_YET on a draw-sensitive raw-detection axis, with every moat axis
(G1/G2/G3/G4/G5/G7) passing.** Benchmark numbers remain **PROVISIONAL** (a smoke run until a
full-census canonical regeneration) (release-readiness-report `## Caveats`). The program optimized
for the **honesty of the claim machinery over the claim itself** — and that is exactly why the
machinery can be trusted.

---

## 13. The Pass-2 horizon, and the RC state

The release-candidate ceremony is the immediate next step: Stage-6 Documentation (this compile) →
the release gate (an honest **SHIP-WITH-CAVEATS** given the G6/NOT_YET verdict) → version 1.4.0 →
1.5.0rc1 + CHANGELOG → tag **locally only** (the release workflow auto-publishes any pushed tag, so
the tag is never pushed) → `make build` + twine-check (sdist + wheel built, **not published**)
(SO-23, `next.rc_close`).

**The Pass-2 horizon** — every item below is an honest, documented follow-up, and **all program
requirements are AGENT_SIMULATED; the user is the Pass-2 cohort** (SO-06, `pass2`;
release-readiness-report `## Pass-2 commitments`):

- **★ The full-census G6 run on current code** (the headline Pass-2). The producer defaults to the
  full dataset (~25–48h, infeasible in one session); the census methodology (matrix + composite) is
  where pii-anon reaches rank-1. *Even so, G6 is draw-sensitive, so a full-census run is not
  guaranteed to reach PROVISIONAL_SOTA* — whether strict raw-F2 vs the single strongest cloud NER is
  even the right bar for a composite/operational-moat product is a **requirements question for the
  PO**, deliberately not changed here (f2-gap-attribution `## Refined conclusion`).
- **Real adversaries and real keys:** the real Tier-3 LLM adversary + the ≥385 paired-persona DATA
  cohort (S5-02); the real LiRA@128 shadow training + canary splits (S5-03); real Tier-C cloud runs
  (aws-comprehend / azure-ai-language / openai-privacy-filter — no keys); real-NUTS Bayesian J
  (numpyro/jax absent). The in-tree representative adversaries + power machinery run for real against
  synthetic data; the real runs are Pass-2 (SO-17 / SO-18; SO-15, `findings_waived_or_deferred`).
- **The ORCH-blocked seams:** the rules-first early-exit (S2-03), the query-aware router pre-filter
  wire-in (S6-01), the pii-anon-itself BYO predictor (S6-04), and the reader capability surface
  (S7-01) all wait on the user's protected `orchestrator.py` WIP, which stayed byte-identical
  (md5 `0afc6dee…`) through the entire program; land each when the WIP clears (SO-12 / SO-19 / SO-20
  / SO-21). The canonical fairness wire-in (S7-03) is Pass-2 *and* SDO-close-gated by design.
- **Real extraction strength + custody:** real OCR/DICOM/ASR extraction + span-level coordinates
  (S7-01); per-detector-class latency ceilings + a full-census latency measurement (the S7-04
  `detector_class` seam); real crypto-key / KEK / HSM custody + real OS-level sandbox syscalls
  (SO-10) — never agent-simulated as real.
- **The sibling tracks:** the DATA track (`../pii-anon-eval-data`: `bradley_terry.py`,
  `assemble_paired_set`, canary, the query-aware scorer) and the Papers (`../pii-anon-research-paper`:
  Paper 1 PII-Rate-Elo, Paper 2 the library/benchmark) are **flagged follow-on** — this run was
  code-only (SO-15 / SO-23, `pass2_followons`). The publishable contribution is the pivot itself:
  pseudonymization-integrity scoring + Bayesian-BT ratings + real Tier-3 re-id, in the empty quadrant
  no public benchmark occupies (PDLC-JOURNEY `## Defensibility`).

**Milestone state:** M4 (CODE v2) is **COMPLETE pending the RC ceremony** (SO-23, `milestone`).
Every actionable feature story across S1–S7 is DONE; only the orchestrator-blocked S2-03 (and its
dependent wire-ins) and the Pass-2 list remain. The keystone is LIVE — the program can manufacture
a certified canonical run on demand — and the honest endpoint is NOT_YET on the draw-sensitive G6
axis, with all moat axes passing. That is not the verdict the program would have *preferred*. It is
the verdict it can *defend* — which, for a measurement-first privacy library whose entire pitch is
trustworthy evaluation, is the only verdict worth shipping.

---

## Methodology

**This journey is COMPILED FROM ARTIFACTS by an agent (the D3 `project-journey` author).** It is
not itself a primary record; it is a narrative synthesis of primary records. What that means
concretely, stated for epistemic honesty per the D2 contract (§4) and the D1 observations:

- **The narrative spine is the 23-sign-off ledger + the MANIFEST `### S*-DONE` sections — NOT
  authored doc-seeds.** All five per-stage `_doc/doc-seed.md` files are **absent** (D1 O-1,
  confirmed). The plain-language stage narrative that would have lived in those doc-seeds is
  *reconstructed* here from the substitute sources the D2 architecture named authoritative: the SO
  `scope:` fields, the SO `decisions_acknowledged` / `adversarial_close` / `sdo_verdict` blocks, the
  brownfield `assessment-2026-05-30.md` findings, the `release-readiness-report.md` verdict, and the
  `f2-gap-attribution.md` diagnostic. Authoring the five doc-seeds is a recommended follow-up pass
  to convert this inferred narrative into explicit authored narrative (D1 O-1 recommendation).
- **What is authored-directly-from-artifact vs agent-inferred.** Every dated event, verdict,
  guarantee state, commit-hash reference, hole/fabrication count, probe count, and md5 is read
  *directly* from a sign-off or canonical artifact and cited inline (SO-NN). The **connective prose**
  — the framing of "the through-line," the section transitions, the characterization of the
  no-fabrication discipline as "the program's spine," the "verdict it can defend" close — is
  *agent-synthesized narrative inference*, drawn from the recorded `decisions_acknowledged` and
  `lesson` fields but composed by the author. It is labeled as narrative, not presented as a
  primary finding. Where a sign-off states a lesson verbatim (the "close is the only reliable
  certifier"; "a correctness rule must hold at every level"), it is quoted as such.
- **★ ALL cohort / persona / concept-value / SME research in this program was AGENT_SIMULATED —
  never real-user (D1 O-3; `00-validation/` is empty save `.gitkeep`).** Every sign-off's signer is
  `AGENT_SIMULATED`. No sentence in this journey claims real-user validation; the Discovery personas,
  the concept-value study, and the 5-SME design panel are all agent-simulated heuristic evaluations.
  The brownfield's MAJOR-10 (no real adoption evidence for the published v1.4.0) is unresolved by
  this pass and is a documented Pass-2 item. **The user is the Pass-2 cohort.**
- **The SDO verdict is reported honestly as NOT_YET with a binding G6 FAIL, and that is a
  methodology gap, not a regression** (D2 honesty-constraint 3; `f2-gap-attribution.md`). Old code is
  byte-identical to current HEAD at `use_case=default`; G6 is draw-sensitive. This is stated as the
  through-line, not buried.
- **Generated / WIP sources were NOT cited** (D2 honesty-constraint 4 / D1 O-7):
  `docs/benchmark-summary.md` (auto-rewritten by the competitor benchmark script) and
  `docs/pii-rate-elo-value.md` (user-WIP, excluded from the docs gate) are deliberately absent from
  `## Sources`; benchmark numbers trace to the stable `release-readiness-report.md` / the keystone
  sign-off instead.
- **Bounded context.** This deliverable is read-only on its mapped sources and authors exactly one
  file. API signatures, CLI invocation detail, ADR derivations, and plugin-authoring how-to are
  *out of scope* here (they live in the api-reference, user-operator-guide, architecture-and-adr, and
  contributor-handbook deliverables respectively, per D2 §5). No source outside the D2 `source_mapping`
  for D-1 was read; no gap was filled by guessing.

---

## Sources

Every load-bearing claim above traces to a row here. Citations are `file:section` (paths
abbreviated: `~/` = `dev-assist-artifacts/`) with the SO-NN / DC-NN / FR-NNN / NFR-NNN trace IDs the
source supplied. This list matches the D2 `source_mapping` for D-1.

| Source (file:section) | Trace IDs supplied | Used for |
|---|---|---|
| `~/_signoffs/SO-01-m1.yaml:scope/findings/decisions` | SO-01; AX-001..006; the 12 MAJORs | §0–1 starting condition; the eval-integrity steering decision |
| `~/_signoffs/SO-02-discovery.yaml:decisions/methodology` | SO-02; UC-01..28 (28 UCs); 7 personas | §2 the measurement-first POV pivot; AGENT_SIMULATED concept-value |
| `~/_signoffs/SO-03-requirements.yaml:decisions/cross_repo` | SO-03; FR-001..039, NFR-001..026; 0-DIVERGED | §2 requirement counts; eval-integrity critical path; honest latency NFRs |
| `~/_signoffs/SO-04-design.yaml:decisions/findings_carried_forward` | SO-04; DC-01..15; DECISION 1/2/3; the Security + Docs MAJORs | §2 the 3 Pugh decisions; resolved CATASTROPHIC eval-01; carried-forward MUSTs |
| `~/_signoffs/SO-05-development.yaml:scope_done/honesty` | SO-05; FR-016/NFR-011/AX-003 (DC-01) | §3 S1-01 recall floor (RED/GREEN commits, property test) |
| `~/_signoffs/SO-06-testing.yaml:verdict/caveats/pass2` | SO-06; FR-016/NFR-011 | §3 SHIP-WITH-CAVEATS/DEFER verdict; §13 Pass-2 (AGENT_SIMULATED) |
| `~/_signoffs/SO-07-sprint1.yaml:decisions/gate/invariants` | SO-07; FR-016/017, NFR-011/024/025 (DC-01) | §3 floor live on production path; 0/5 adversarial refutations |
| `~/_signoffs/SO-08-s3-eval-integrity.yaml:decisions/sota_dominance_objective` | SO-08; FR-003/004 (DC-06/07); NFR-001/002 | §4 the rating ladder; the convergence gate; env-honest architecture |
| `~/_signoffs/SO-09-sdo-gate-live.yaml:scope/decisions/sdo_verdict` | SO-09; FR-007/008 (DC-11); NFR-006; SDO-J | §4 the SDO gate going LIVE; coherence-by-construction; NOT_YET/G7 |
| `~/_signoffs/SO-10-security-musts.yaml:scope/adversarial_close/decisions` | SO-10; FR-018/019/026, NFR-014/015 (DC-04); AX-006 | §5 the 3 security MUSTs; the SEC-S504-PATHTRAVERSAL catch; the catch-net rule |
| `~/_signoffs/SO-11-g2-g4-guarantees.yaml:adversarial_close/decisions/sdo_verdict` | SO-11; FR-006/009/010 (DC-08), FR-005 (DC-10); NFR-014..021 | §6 G2/G4 computable; the no-fabrication MAJOR in both gate functions |
| `~/_signoffs/SO-12-s2-moe-router-core.yaml:scope/stories_deferred/adversarial_close` | SO-12; FR-018 (DC-02); NFR-005/026; AX-003 | §7 the MoE-router core; the control-path close (0 upheld); S2-03 blocked |
| `~/_signoffs/SO-13-s2-04-sla-bias.yaml:story_gate/decisions` | SO-13; NFR-009/010 (DC-03); AX-003 | §7 the SLA selection-bias; the two-MAJOR robustness hole; the power-tier trap |
| `~/_signoffs/SO-14-phase-b-core-g5-audit.yaml:scope/work_stream_gate_summary/decisions` | SO-14; FR-025/026/028/029 (DC-13), NFR-016 (DC-09) | §8 the G5 audit surface; the keyless-surrogate MAJOR; worktree-parallelism unreliable |
| `~/_signoffs/SO-15-keystone-close.yaml:scope/work_stream_gate_summary/decisions/sdo_verdict` | SO-15; FR-008 (DC-11); NFR-005/006; the 7-round close | §9 the keystone; 11 holes/5 fabrications; the close-is-the-only-certifier finding |
| `~/_signoffs/SO-16-s7-04-latency-g5.yaml:scope/adversarial_close/decisions/sdo_verdict` | SO-16; NFR-009 (DC-11/02) | §10 G5 the last placeholder; the breach-bury 6th fabrication; completeness milestone |
| `~/_signoffs/SO-17-s5-02-tier3-adversary.yaml:scope/decisions` | SO-17; FR-011/012, NFR-012 (DC-09); NFR-016 | §11 Tier-3 representative adversary; de-circularization; the exact-anchor lesson |
| `~/_signoffs/SO-18-s5-03-mia-representative.yaml:scope/decisions` | SO-18; FR-013, NFR-013 (DC-09); NFR-016 | §11 representative MIA; attack surface complete; decode-boundary hardening |
| `~/_signoffs/SO-19-s6-01-query-aware-masking.yaml:scope/decisions` | SO-19; FR-023/024 (DC-13); AX-006 | §11 query-aware default-to-mask gate; the S2-03 orchestrator block |
| `~/_signoffs/SO-20-s6-04-byo-pipeline-sdk.yaml:scope/decisions` | SO-20; FR-001/002 (DC-12); NFR-026 | §11 the BYO-pipeline SDK; identical-path-by-construction; the FR-019 erratum |
| `~/_signoffs/SO-21-s7-01-native-readers.yaml:scope/findings_remediated_in_loop/decisions` | SO-21; FR-031..035 (DC-14); NFR-026/023 | §11 native readers; the zip-bomb FlateDecode catch; loud-degradation NFR-026 |
| `~/_signoffs/SO-22-s7-03-multilingual-fairness.yaml:scope/decisions` | SO-22; FR-038/039, NFR-025/004 (DC-15); AX-003 | §11 dead-CJK-keyword activation; the fail-closed fairness gate; feature surface COMPLETE |
| `~/_signoffs/SO-23-s7-05-docs-discoverability.yaml:scope/findings_remediated_in_loop/next` | SO-23; FR-010 surface; the Docs MUST | §11 docs teeth; the phantom-filename catch; §13 the RC ceremony |
| `~/00-brownfield-assessment/assessment-2026-05-30.md:§1-3 (Profile/Signal/Findings)` | the 12 MAJORs (0 SHOWSTOPPER); AX gaps | §1 the starting condition; the four load-bearing problems |
| `~/05-testing/release-readiness-report.md:##Verdict/##Caveats/##Pass-2 commitments/##End-of-PDLC handoff` | FR-016/NFR-011/AX-003; the NFR tally | §3 SHIP-WITH-CAVEATS; §12 the NFR matrix + PROVISIONAL numbers; §13 Pass-2 |
| `~/05-testing/_diagnostics/f2-gap-attribution.md:##Findings/##Refined conclusion` | G6, NFR-011 | §9, §12, §13 the G6-is-not-a-regression / draw-sensitive through-line |
| `~/PDLC-JOURNEY.md:##Traceability spine/##Defensibility` | FR-016, NFR-011; the publishable pivot | §0 the spine framing; §13 the Papers defensibility note |

**Deliberately NOT cited (per D2 honesty-constraint 4 / D1 O-7):** `docs/benchmark-summary.md`
(generated / auto-rewritten — volatile) and `docs/pii-rate-elo-value.md` (user-WIP, excluded from
the docs gate). Where a number was needed it was sourced from the stable
`release-readiness-report.md` and the keystone sign-off (`SO-15`) instead.
