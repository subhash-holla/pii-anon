# pii-anon — Architecture and Architecture Decision Records (ADR)

> **Stage 6 D3 authored deliverable (D-4).** Compiled 2026-06-10 from canonical design
> artifacts. Audience: maintainers, contributors, auditors. Depth: standard (all 15 DCs +
> 3 headline decisions; decision-level depth, not line-by-line internals).

---

## Overview

pii-anon is a packaged, installable OSS Python library providing PII detection,
pseudonymization, and privacy evaluation. The library completed a full PDLC pass
(brownfield assessment → Discovery → Requirements → Design → Development → Testing) under
`pdlc/sota-program`. The architecture is organized into two pillars — **swarm** (detection,
routing, masking) and **eval** (benchmarking, rating, SDO certification) — sharing a single
public entry point (`PIIOrchestrator`), a common type contract, and six cross-cutting axioms.

---

## 1. System Architecture

### 1.1 The Four-Layer Swarm

The detection path is a four-layer pipeline sourced in `docs/swarm-architecture.md` and
realized in `src/pii_anon/swarm.py`:

```
Layer 1 — Regex fast-pass (structured PII)
  Luhn / IBAN mod-97 / ABA checksum validators
  confidence >= fast_pass_threshold (default 0.90) -> emit directly

Layer 2 — Heterogeneous NER engines
  GLiNER (zero-shot transformer), Presidio (spaCy-backed), Stanza, regex tail
  redundancy pruning via IoU overlap (default 0.3)

Layer 3 — Dawid-Skene + XGBoost meta-learner
  Bayesian aggregation of engine votes (EM-trained confusion matrices)
  Temperature calibration per engine, 21-feature vector -> meta_score in [0,1]

Layer 4 — Validation and corroboration
  Emission gate: meta_score >= emission_threshold (default 0.50)
  Corroboration gate for SEMANTIC_TYPES (require >= corroboration_min engines)
  Deduplication against Layer 1 fast-pass results
```

The always-on `regex-oss` baseline (the standalone `pii-anon` engine) holds three
privileged positions: Layer 1 fast-pass, pinned past the Layer 2 Jaccard pruner, and
dedicated feature slots (8/9/14) in the XGBoost meta-learner. This is a hard contract, not
an optional add-on.

### 1.2 The Shared-Layer Recall Floor (AX-003, DC-01)

The critical architectural invariant introduced by the SOTA program is the
`SharedLayerProjector`. It is a single post-fusion chokepoint that ensures:

```
entities(ensemble) ⊇ entities(shared_layer)   ZERO violations
```

The "shared layer" is the always-on `RegexEngineAdapter` (checksum/keyword-gated, ~0.7 ms/rec,
20 types) computed deterministically per chunk BEFORE any gate or threshold. After fusion,
`SharedLayerProjector.project(output, shared)` re-injects any shared span that a downstream
gate dropped, tagged `provenance='shared_floor'`. The floor is decoupled from the router and
therefore holds for any future routing implementation.

Both `MoEFusionStrategy` and `SwarmFusionStrategy` delegate to this chokepoint. A
`FloorProjectingFusion` wrapper (`routing/floor_fusion.py`) provides the integration seam.

Implementing modules: `pii_anon/routing/shared_layer.py`, `pii_anon/routing/floor_fusion.py`.
Pinned by: a property test asserting zero span-set violations across all modes x gate-on/off x
chunk-boundary, plus a per-language CI recall gate (NFR-011, ε ≤ 0.005).

### 1.3 The MoE Router (DC-02, DC-03)

The Mixture-of-Experts routing layer (`pii_anon/routing/distilled_gate.py`) adds learned
routing on top of the floor guarantee. The `DistilledTopKGate` is distilled offline from a
frozen XGBoost meta-learner as survival oracle (BCE/KL over per-(entity_type, expert) survival
labels). The gate artifact `gate_v1.json` carries the oracle-hash and `gate_feature_version`;
absent gate → static `entity_strengths` softmax (NFR-026 graceful degradation). The gate is
ADVISORY only — it can never drop a shared-layer span.

Rules-first Depth-1 early-exit: chunks whose shared spans are ALL checksum/keyword-gated exit
before heavy NER (structurally provable correctness, deterministic); the learned gate is
consulted only for UNCERTAIN chunks. The early-exit seam is at the orchestrator hook in
`_detect_on_text_field_async` — the only place engine-skip is achievable post-`merge()`.

Note: the early-exit was blocked by S2-03 (a SWITCH-POINT) at program completion. The seam
exists; the orchestrator wire-in is a Pass-2 item.

Aux-loss-free SLA bias (DC-03): DeepSeek-V3-style bias on selection logits only (never
fused confidence, never shared membership), nudged toward the LOCKED 1522/753/200 power-tier
latency budgets. This is a SHOULD; LIVE as of S2-04 / SO-13.

### 1.4 The Eval Framework

The eval pillar is organized as a ports-adapters architecture (`DC-06`) with an explicit
import boundary: `eval_framework.rating` imports nothing from swarm/moe/fusion/policy (pinned
by a CI import-boundary property test).

Key subsystems:

- **Rating engine ladder (`eval_framework/rating/`):** `RatingEnginePort` + `RatingEngineRegistry`
  with a 3-tier graceful-degrade ladder: `glicko-legacy` (verbatim fallback) →
  `bradley-terry-mle` (pure-stdlib, PR-CI/smoke tier) → `bayes-bt` (NumPyro Davidson/BT,
  NUTS — **claim-grade only**). Only `bayes-bt` satisfies NFR-001 (MCMC diagnostics: split-R̂
  ≤ 1.01, bulk-ESS ≥ 400/param, 0 divergences). A hard convergence gate refuses claim-grade
  emission on failure.

- **BYO-pipeline SDK adapter (`eval_framework/byo_pipeline.py`):** `BYOPipelineRegistry` +
  `engine_predictor`/`incumbent_predictor` protocols allow third-party pipelines to score
  on the identical evaluation path as the library (FR-001/FR-002, DC-12).

- **Attacks package (`eval_framework/attacks/`):** real Tier-3 LLM-adversary (de-circularized,
  DC-09) + LiRA@128 MIA. The attack harness runs in a sandbox (SO-10 security MUST).
  DATA-track dependency: the representative adversary and canary splits depend on
  `pii-anon-eval-data` (external_ref `DATA:`), not this repo.

- **SDO gate (`eval_framework/evaluation/competitive_supremacy.py`):** seven guarantee
  functions G1–G7 assessing recall floor, pseudonymization integrity, recall dominance,
  calibration, latency+audit, raw non-inferiority (F2), and certified-run provenance.
  All G1–G7 are computed (no placeholders) as of SO-16. The honest produced verdict is
  **NOT_YET / binding G6 FAIL** (F2 0.7214 vs 0.75 threshold) — this is a methodology gap,
  not a regression.

### 1.5 The SDO Gate / Producer Control Path (DC-11)

The Canonical Run Gate (`evaluation/canonical_run.py:CanonicalRunGate`) and the
Competitive Supremacy Gate (`eval_framework/evaluation/competitive_supremacy.py`) form a
trust chain: no artifact claiming `canonical_claim_run=True` is accepted without a
provenance stamp (seed / key / scope / dataset-hash / power-cell counts). The gate is
fabrication-hardened through a 7-round adversarial close (SO-15 keystone, SO-16 G5 fix)
that uncovered and closed 11 holes including 5 fabrications.

All control-path artifact changes require a mandatory adversarial close at bar = 0 upheld.

### 1.6 The Agentic Surface (DC-13)

The agentic interception layer (`pii_anon/agentic/`) provides a `FourChannelGuard` that
intercepts all four agent channels (prompt / memory / tool-I/O / trace) with least-privilege
scope (AX-006). No raw PII is persisted to any channel after masking (FR-026). The guard
operates via a router pre-filter (`policy/router.py:PolicyRouter`) that applies masking before
engines / prompt assembly. A `QueryAwareMaskingGate` (`policy/query_aware.py`) implements
subtractive-on-mask, default-to-mask policy — false-retention cannot occur by default.

The orchestrator wire-in of the router pre-filter is a Pass-2 item (SWITCH-POINT ORCH in
the codebase). The gate, guard, and leakage-Sankey surfaces are LIVE as standalone primitives.

### 1.7 The Ingestion Surface (DC-14)

Native-format readers (`pii_anon/ingestion/native.py`) expose a `NativeReaderRegistry` and a
`NativeReader` Protocol. Adapters for PDF (`PdfTextReader`), image-OCR (`ImageOcrReader`),
DICOM (`DicomReader`), and audio (`AudioReader`) all emit `Iterator[IngestRecord]` behind
lazy optional-deps. Offsets map back to source coordinates for extraction-fidelity assertion.
Round-trip reconstruction preserves non-PII payload (FR-032). OCR/DICOM/audio extraction at
full strength is a Pass-2 item (the reader seam is LIVE; the extraction backends are iter-1).

### 1.8 Cross-Repo Boundaries

The program divides labor between this repo (CODE-local) and `pii-anon-eval-data` (DATA-track):

| Surface | Track | Notes |
|---|---|---|
| `routing/`, `eval_framework/rating/`, BYO-SDK, agentic interception | CODE-local | Fully in this repo |
| `stats/bradley_terry.py`, `assemble_paired_set`, canary splits | DATA-track (S6) | `pii-anon-eval-data` external_ref |
| Tier-3 representative adversary power | DATA-track (S5) | blocking on eval-data |
| Query-aware scorer (learned model) | DATA-track (S5, SWITCH-POINT DATA) | standalone gate is CODE-local; learned scorer is DATA-track |
| Agentic oracle | DATA-track (S7) | future |

---

## 2. Architecture Decision Records

Three headline Pugh decisions from `D-implementation-ready-design.md:§DECISION 1/2/3` are
recorded here as ADR-style entries. No separate `D-decision.md` diamond files exist (O-5);
all decision context is consolidated in the synthesis document.

---

### ADR-001 — Swarm MoE-Router: SharedLayerProjector + DistilledTopKGate + Rules-first Early-Exit

**Status:** LIVE (S1-01 / S2-01 / S2-02; SO-07 / SO-12)

**Context (DC-01, DC-02, DC-03):**
The brownfield assessment identified that the swarm had TWO divergent floor mechanisms —
the MoE path's `non_routed_floor` parameter (`moe.py:354-378`) and the swarm path's
Layer-4 emission gate (`swarm.py:651-661`) — with the guarantee NOT by-construction across
the whole pipeline (`moe-architecture-and-guarantee.md`). The verified swarm.py:654/661 leak
could suppress a shared-layer (regex-oss) finding that fell below fast-pass and entered
fusion. The SOTA program needed learned routing (DistilledTopKGate), latency budgeting, and
early-exit while preserving or strengthening the recall-floor invariant.

The three Pugh proposals scored: Proposal A spine + grafts = **8.4**, Proposal B
always-on TinyNER in Shared = 7.7, Proposal C full speculative-verification machine = 7.4.

**Decision:**

1. **`SharedLayerProjector` (DC-01, AX-003 fix):** One post-fusion chokepoint that both
   `MoEFusionStrategy` and `SwarmFusionStrategy` delegate to. The shared span set is
   computed BEFORE any gate; the projector re-injects any dropped shared span tagged
   `provenance='shared_floor'`. Floor is decoupled from the router — holds for any future
   router by construction.

2. **`DistilledTopKGate` (DC-02):** Gate distilled offline from the frozen XGBoost
   meta-learner. ADVISORY only — cannot drop a shared span. Artifact carries oracle-hash +
   `gate_feature_version`; absent → static softmax (NFR-026).

3. **Rules-first Depth-1 early-exit (DC-02):** Chunks whose shared spans are ALL
   checksum/keyword-gated exit before heavy NER. Learned gate is consulted for UNCERTAIN
   chunks only. (Orchestrator hook, not the fusion strategy — `merge()` runs after engines.)

4. **Aux-loss-free SLA bias (DC-03, SHOULD):** Bias on selection logits only; never touches
   fused confidence or shared membership.

**Alternates rejected:**

- Proposal B: always-on TinyNER in the default shared set — risk of never-gated precision
  loss from an under-validated model.
- Proposal C: full speculative-verification machine — reserved as a switch-point if latency
  budgets are unmet in a future pass.

**Consequences:**

- FR-016 (recall floor by construction) MUST is satisfied; zero NFR-011 violations confirmed
  by property test and CI gate.
- DC-02 early-exit is BLOCKED by S2-03 at program completion (the orchestrator wire-in is a
  Pass-2 switch-point). The gate seam is live; the performance benefit is deferred.
- NFR-008 (early-exit chunk latency p50 ≤ 1 ms, p95 ≤ 2 ms) is documented via
  `latency_ceilings.py` but not gated by a named acceptance test (SHOULD gap, O-6).
- The gate artifact `gate_v1.json` is signed + verified (S2-05, SO-10 security MUST) to
  prevent privilege-escalation attacks on the control-path artifact.

**Satisfies:** FR-016, FR-017, FR-018, NFR-007, NFR-008 (doc-only), NFR-009, NFR-011, AX-003.

---

### ADR-002 — Eval Rating Engine: Bayesian-BT Spine with 3-Tier Graceful-Degrade Ladder

**Status:** LIVE (S3-01 / S3-02 / S3-03 / S3-04; SO-08)

**Context (DC-06, DC-07):**
NFR-001 requires MCMC diagnostics literally (split-R̂ ≤ 1.01, bulk-ESS ≥ 400/param, 0
divergences). A frequentist rating engine satisfies this only by substitution, not by
construction. The brownfield assessment found verified fabricated-outcome, fake-CI, and
decoupled-significance defects at `elo.py:243/542/561`. The program needed a rating engine
that (a) satisfies NFR-001 by construction at claim-grade, (b) preserves the ~7 callers of
the existing `PIIRateEloEngine`, and (c) degrades gracefully without hard deps.

The three Pugh proposals: Bayesian-BT spine = **8.6**, pure-frequentist = 7.4, hybrid = 7.1.

**Decision (CATASTROPHIC eval-01 resolved):**

1. **`RatingEnginePort` + `RatingEngineRegistry` (DC-06):** mirrors the `engines/registry.py`
   entry-point pattern (`pii_anon.rating_engines` group). `PIIRateEloEngine` becomes a thin
   facade, preserving existing callers.

2. **Three-tier ladder:** `glicko-legacy` (instant rollback, verbatim) → `bradley-terry-mle`
   (pure-stdlib MM + paired bootstrap; PR-CI / smoke tier) → `bayes-bt` (NumPyro Davidson/BT,
   NUTS — **claim-grade default**). Only `bayes-bt` satisfies NFR-001; a hard convergence
   gate refuses claim-grade emission on MCMC failure (fails loud).

3. **Coherent significance by construction (DC-07):** one joint posterior eliminates the
   verified decoupled-significance defect (point in CI, sign matches verdict, significant iff
   CI excludes 0 — cannot disagree). Davidson tie term (eval-02) is included. Record-level
   paired outcomes (N × C(K,2) from `per_record_f1`). Sum-to-zero + HalfNormal hierarchical
   prior for identifiability.

4. **Hard import boundary:** `eval_framework.rating` imports nothing from
   swarm/moe/fusion/policy — pinned by a CI import-boundary property test.

**Alternates rejected:**

- MLE-BT as claim-grade: satisfies NFR-001 only by substitution, not by construction.
- Glicko-only: cannot produce the joint posterior required for NFR-002.
- The switch-point exists: if the NumPyro/JAX dep footprint is rejected, demote `bayes-bt`
  to CI-only.

**Consequences:**

- The `bayes-eval` optional extra (`numpyro/jax/arviz`) is required for claim-grade runs.
- The `stats/bradley_terry.py` primitive is CODE-local in this repo; the frozen
  `PairedComparisonSet` comes from `pii-anon-eval-data` (DATA-track dependency, external_ref).
- G3 (recall dominance) and G4 (calibration selective-risk) contribute to the SDO verdict via
  `CanonicalRunGate`-emitted fields and the `CompetitiveSupremacyGate` function `_g3` / `_g4`.
- The anon vs. pseudo distinction is enforced by a separate Davidson sub-model for Tier-3 RRS
  — the two families are NEVER merged (FR-010 / AX-004).
- `RecallFloorVerdictGuard` (fail-closed) bars a floor-breaching system from claim-grade
  top-rank even if the rating engine awards it composite rank-1.

**Satisfies:** FR-003, FR-004, FR-005, FR-006, NFR-001, NFR-002, NFR-003, AX-004, AX-005.

---

### ADR-003 — Agentic Interception: Router Pre-Filter + Query-Aware Gate + Unified Floor

**Status:** LIVE (S6-01 / S6-02 / S6-05; SO-14 / SO-19); orchestrator wire-in Pass-2

**Context (DC-13):**
The SOTA program needed to intercept all four agentic channels (prompt / memory / tool-I/O /
trace) with least-privilege scope while maintaining the recall floor for masked spans. Three
option variants were evaluated: Option A router pre-filter + unified floor = **winner**,
Option B gateway facade, Option C reveal-only overlay.

**Decision:**

1. **Router pre-filter surface (DC-13):** intercept and mask in `policy/router` BEFORE
   engines or prompt assembly. Floor is wrapped at the `build_fusion` seam (not a registered
   decorator — verified that a decorator does not compose over built-in swarm/moe at
   `fusion.py:500-512`).

2. **Query-aware masking gate (`policy/query_aware.py:QueryAwareMaskingGate`):** subtractive-on-
   mask, default-to-mask policy — retains a span ONLY on a positive entity-alias/surface-token
   relevance signal. False-retention (information leak) cannot occur by default. The
   `score_query_aware_bound` function provides the FR-024 over-redaction / false-retention
   bound vs mask-all (a standalone pure primitive). The learned data scorer is a DATA-track
   SWITCH-POINT (Pass-2).

3. **Four-channel least-privilege interception (`agentic/interception.py:FourChannelGuard`):**
   covers prompt, memory, tool-I/O, and trace with `AX-006` scope. No raw PII persisted
   post-masking — `NoRawPIIPersistError` is raised on violation (FR-026).

4. **Leakage-Sankey + injection resistance (`agentic/leakage_sankey.py`):** per-channel
   leakage counts (FR-028) + injection-resistance scoring (FR-029). Observability surface
   for G5 audit evidence.

**Alternates rejected:**

- Option B gateway facade: mandatory if sole-writer-to-memory is required in a future context.
  Remains a documented switch-point.
- Option C reveal-only overlay: reserved if utility must expose masked spans to the agent.

**Consequences:**

- The orchestrator router pre-filter wire-in (`# SWITCH-POINT(ORCH)`) is a Pass-2 item.
  The gate, guard, and leakage-Sankey surfaces are live as standalone primitives.
- Query-aware over-redaction bound (`score_query_aware_bound`) tests are pinned at exact
  reference metric values (not bounds) per the exact-rate-anchor lesson (S5-02 / S5-03 /
  S6-01 recurrence).
- C's widened-shared-set (Option C graft) is a CI-gated WATCH item.

**Satisfies:** FR-023, FR-024, FR-025, FR-026, FR-027, FR-028, FR-029, FR-030, AX-006.

---

## 3. DC-01..DC-15 Module Map

All 15 Design Cases, their FR/NFR, implementing modules, and program status.
Source: `D-implementation-ready-design.md:§D1 Design Cases`.

| DC | Title | Implements | Implementing Module(s) | Status |
|---|---|---|---|---|
| DC-01 | SharedLayerProjector — recall-floor by construction | FR-016, NFR-011, AX-003 | `routing/shared_layer.py`, `routing/floor_fusion.py` | LIVE (S1-01, SO-07) |
| DC-02 | MoE-router: DistilledTopKGate + rules-first Depth-1 early-exit | FR-018, NFR-007/008/009 | `routing/distilled_gate.py`, `routing/gate_distillation.py`, orchestrator hook | LIVE (S2-01/02, SO-12); **early-exit blocked (S2-03, Pass-2)** |
| DC-03 | Aux-loss-free SLA selection-bias | NFR-009/010 | `routing/distilled_gate.py` (selection-logits) | LIVE (S2-04, SO-13) |
| DC-04 | Reversible pseudonymization + auditable key rotation | FR-019, NFR-014/015 | `tokenization/encrypted_store.py` (`EncryptedSQLiteTokenStore`, `KeyEnvelope`) | IN-TREE (S6-03, SO-10) |
| DC-05 | 6 transforms + legal-regime mapping + orchestrate incumbents | FR-020/021/022 | `transforms/`, `eval_framework/byo_pipeline.py` | PARTIAL; FR-020/021/022 are SHOULD; orchestration SHOULD deferred. **Least-elaborated DC (1 source file).** |
| DC-06 | RatingEnginePort + RatingEngineRegistry (3-tier ladder) | FR-003, NFR-001/026 | `eval_framework/rating/` (`RatingEnginePort`, `RatingEngineRegistry`, `BayesBTEngine`, `BradleyTerryMLEEngine`, `PIIRateEloEngine`) | LIVE (S3-01/02/03, SO-08) |
| DC-07 | Coherent significance + Davidson ties | FR-004, NFR-002/003 | `eval_framework/rating/` (joint posterior; `convergence.py`) | LIVE (S3-04, SO-08) |
| DC-08 | Distinct anon-vs-pseudo scoring families | FR-006/009/010, NFR-014/015 | `eval_framework/metrics/deid_families.py` (`AnonymizationScorer`, `PseudonymizationIntegrityScorer`, `DeidFamilyScores`) | LIVE (S4-01, SO-11) |
| DC-09 | attacks/ package: Tier-3 LLM-adversary + LiRA@128 MIA | FR-011/012/013, NFR-012/013 | `eval_framework/attacks/` (`reid.py`, `reid_tier3.py`, `mia.py`, `sandbox.py`, `spec.py`) | LIVE (S5-01/02/03/04, SO-14/17/18). DATA-track: representative adversary + canary splits depend on `pii-anon-eval-data`. |
| DC-10 | Calibration and selective-risk reporter | FR-005, NFR-017/018/019/020/021 | `calibration/` (offline, online, store, dominance); `eval_framework/metrics/composite.py` | LIVE (S4-03, SO-11) |
| DC-11 | CanonicalRunGate + provenance + CI ship/no-ship + ε-gate | FR-007/008, NFR-006/011(ε) | `evaluation/canonical_run.py`, `eval_framework/evaluation/competitive_supremacy.py`, `eval_framework/evaluation/latency_ceilings.py` | LIVE (S7-02/04, SO-15/16) |
| DC-12 | BYO-pipeline SDK adapter + identical-incumbent scoring | FR-001/002 | `eval_framework/byo_pipeline.py` (`BYOPipelineRegistry`, `engine_predictor`, `incumbent_predictor`, `build_identical_path_leaderboard`) | LIVE (S6-04, SO-20) |
| DC-13 | Agentic interception: router pre-filter + query-aware + 4-channel | FR-023..030, AX-006 | `policy/query_aware.py`, `policy/router.py`, `agentic/interception.py`, `agentic/leakage_sankey.py` | LIVE (S6-01/02/05, SO-14/19); **orchestrator wire-in Pass-2** |
| DC-14 | Multimodal readers + per-modality benchmark + parity | FR-031..037, NFR-022/023 | `ingestion/native.py`, `ingestion/native_pdf.py` | LIVE (S7-01, SO-21); **OCR/DICOM/audio extraction at full strength Pass-2**. FR-033/035/036 thin-trace: extraction-fidelity assertion, CI regression gate, stream/batch/offline parity all addressed in S7-01. |
| DC-15 | Multilingual context + fairness gate + no-real-PII | FR-038/039, NFR-024/025/026, AX-001 | `eval_framework/metrics/fairness_gate.py` (`evaluate_language_fairness`, `FairnessGateReport`); feature 21 in XGBoost meta-learner | LIVE (S7-03, SO-22) |

**Notes on thin-trace DCs:**
- **DC-05** is the least-elaborated decision (1 source file in dev-assist-artifacts/). Its
  FR-020/021/022 are SHOULD and partially deferred. The transform surface exists (`transforms/`);
  the legal-regime mapping and incumbent orchestration are Pass-2.
- **DC-14 thin trace (FR-033, FR-035, FR-036 — 5-6 files each):** all four carried by S7-01.
  FR-033 (extraction-fidelity assertion), FR-035 (CI regression gate), FR-036 (stream/batch/
  offline parity) are documented but have thin artifact coverage outside the S7-01 story.
- **NFR-008 (3 files):** early-exit chunk latency (p50 ≤ 1 ms, p95 ≤ 2 ms) is documented in
  `latency_ceilings.py` (the committed registry) but has no dedicated acceptance test story
  (SHOULD gap, O-6).

---

## 4. Cross-Cutting Axioms (AX-001..AX-006)

Source: `dev-assist-artifacts/00-axioms/project-axioms.yaml`.

| ID | Name | One-line semantics | Enforced by |
|---|---|---|---|
| AX-001 | `synthetic-only-no-real-pii` | No real person's identifiable data in any test fixture, example, doc snippet, benchmark slice, or bundled corpus. A PII tool embedding real PII is itself regulated personal data. | SAST reviewer scan; DC-15 no-real-PII gate; NFR-024 threshold 0 SHOWSTOPPER/CATASTROPHIC |
| AX-002 | `deterministic-reproducible-pseudonymization` | Same (value, key, scope) → identical surrogate across documents and runs. LLM-based steps record model id + prompt + decoding params + seed. | N=5 determinism replay property test; NFR-005 byte-identical scoring; code-quality reviewer check |
| AX-003 | `ensemble-recall-floor-guarantee` | The swarm must never detect fewer true entities than its always-on shared (regex/checksum) layer: `entities(ensemble) ⊇ entities(shared)`, by construction. | `SharedLayerProjector` (DC-01); property test zero-violations; per-language CI recall gate (NFR-011, ε ≤ 0.005); mutation tests |
| AX-004 | `anonymization-pseudonymization-separation` | Anonymization and pseudonymization are distinct end-states scored by different metric families and never collapsed into one "redaction quality" score. | Distinct scoring entry points (`deid_families.py`); DC-08 CI guard test; no merged de-id field; FR-010 MUST |
| AX-005 | `calibrated-uncertainty-and-auditable-abstention` | Detections carry a calibrated confidence; the pipeline can abstain on uncertain spans (route to human review) rather than silently emit or drop. Routing decisions are auditable (`router_path`/`abstain_reason`). | Calibration reporter (DC-10); NFR-020 100% calibrated confidence; abstention/escalation outcome reachable |
| AX-006 | `least-privilege-agentic-interception` | When pii-anon intercepts agent runtimes, the privacy layer accesses only the data + tools needed to detect and transform PII. No raw sensitive data is persisted beyond the masking step. Scope is bounded and documented. | `FourChannelGuard` (DC-13); `NoRawPIIPersistError` on violation; FR-026 MUST; security SAST reviewer |

---

## 5. Verification Properties and Standing Gates

The following verification properties bind the architecture to the axioms:

1. **Recall-floor property test:** `spans(output) ⊇ spans(shared)` ZERO violations across all
   modes × gate-on/off × chunk-boundary (AX-003, NFR-011).
2. **Import-boundary AST test:** `eval_framework.rating` imports nothing from detection
   (scoped to the 2 gate modules; DC-06).
3. **Distinct-family CI guard:** no merged de-id field; `AnonymizationScore` and
   `PseudonymizationIntegrityScore` are distinct types (AX-004, DC-08).
4. **Gate-artifact signature test:** `gate_v1.json` signed + verified on load (S2-05, SO-10).
5. **NFR-001 convergence gate:** `bayes-bt` refuses claim-grade emission on MCMC failure
   (split-R̂, bulk-ESS, divergence checks — `eval_framework/rating/convergence.py`).
6. **N=5 determinism replay:** scoring runs produce byte-identical output across 5 replays
   within a key epoch (AX-002, NFR-005).
7. **No-real-PII scan:** SAST scan on src/, tests/, docs/ for real-PII leakage (AX-001,
   NFR-024). Any finding is a CATASTROPHIC release blocker.
8. **Adversarial close (mandatory for control-path changes):** any change to
   `competitive_supremacy.py`, `canonical_run.py`, or gate artifact producers requires a
   mandatory close at bar = 0 upheld. The S7-02 keystone close ran 7 rounds (SO-15).

---

## Methodology

This deliverable was authored directly from the following canonical artifacts:

- `D-implementation-ready-design.md` — the primary source for all three ADRs and the DC
  table. The three `###DECISION` sections and the `§D1` table provided decision context,
  alternates, and consequences. No per-diamond `D-decision.md` files exist (O-5 confirmed);
  the ADR structure here is compiled from the synthesis document, not from dispersed ADR files.
- `moe-architecture-and-guarantee.md` — the canonical design record for the recall-floor
  invariant, including the proof sketch, the historical violation root cause, and the design
  mandate for the SharedLayerProjector. Cited directly for ADR-001 context.
- `doc-source-index.md` — the D1 harvest index, which provides the DC table with program
  status, the FR/NFR inventory, the SO ledger, and the cross-repo boundary summary.
- `doc-architecture.md` — the D2 architecture document, which provides the per-deliverable
  source mapping, the caveats to carry, and the audience framing.
- `project-axioms.yaml` — the authoritative registry for AX-001..AX-006, read verbatim for
  section 4.
- `docs/swarm-architecture.md` — the user-facing four-layer pipeline diagram, the fast-pass
  eligibility matrix, the SEMANTIC_TYPES corroboration gate. Read directly for section 1.1.

The narrative in sections 1–4 is compiled from these canonical artifacts, not from
agent-synthesized imagination. Where section 1.3 states "early-exit was blocked by S2-03,"
that status is taken directly from the DC table in `doc-source-index.md` and the synthesis
doc's switch-point text.

The following honesty constraints apply:

- The design was produced by an AGENT_SIMULATED process (17 agents, Pugh scoring, SME
  heuristic evaluation). The SME panel is agent-simulated, not human SME review. No claim of
  human-validated design decisions is made here.
- The SDO verdict is honest NOT_YET / binding G6 FAIL (F2 0.7214 vs 0.75 threshold; G1/G2/G3/
  G4/G5/G7 all PASS). The architecture sections citing the SDO gate reflect this accurately.
- `docs/benchmark-summary.md` was not cited (O-7: generated/volatile).
- `docs/pii-rate-elo-value.md` was not cited (O-7: user-WIP).
- DC-05's thinness (1 source file) is noted explicitly in the module map.
- FR-033/035/036 and NFR-008 thin-trace gaps are called out explicitly per the D2 mandate.

---

## Sources

| Source | Section(s) read | Trace IDs supplied |
|---|---|---|
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§D1 Design Cases` | DC table (DC-01..15), pillar, implements, program status | DC-01..DC-15; FR-016/018/019/020..030/031..037/038/039; NFR-007/008/009/011/014/015/022/023 |
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§DECISION 1` | SharedLayerProjector + DistilledTopKGate + rules-first early-exit; Pugh scores; alternates; seam corrections | DC-01/02/03; FR-016/018; NFR-007/008/009/011; AX-003 |
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§DECISION 2` | RatingEnginePort 3-tier ladder; CATASTROPHIC eval-01 resolution; DC-07 coherent significance | DC-06/07/08/10/11; FR-003/004/005/006/009/010; NFR-001/002/003; AX-004/005 |
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§DECISION 3` | Agentic interception; router pre-filter; query-aware gate; 4-channel guard | DC-13; FR-023..030; AX-006 |
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§SME findings` | CATASTROPHIC eval-01 context; security MAJORs; Docs MAJORs | DC-11; FR-007/008; NFR-006 |
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§Switch-points` | Documented alternates for all three ADRs | DC-02/06/13 |
| `dev-assist-artifacts/03-design/06-synthesis/D-implementation-ready-design.md:§Cross-repo` | DATA-track vs CODE-local division of labor | FR-001/002/003/011/013 |
| `dev-assist-artifacts/03-design/moe-architecture-and-guarantee.md` | Ensemble Superset Guarantee theorem; historical violation root cause; design mandate | DC-01/02; AX-003; NFR-011 |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§4 Decision Inventory` | DC table with program status; DECISION 1/2/3 summaries | DC-01..DC-15; SO-07..SO-23 |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§3 Requirement Inventory` | FR/NFR inventory including MoSCoW; file counts; thin-trace orphan candidates | FR-001..039; NFR-001..026 |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§8 Code Surface` | 8b new module table (symbols + FR/DC); 8c baseline modules | All §8b modules |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§2 Sign-off Ledger` | SO-07..SO-23 scope lines for program status column | SO-07..SO-23 |
| `dev-assist-artifacts/06-documentation/02-architecture/doc-architecture.md:§5 D-4 source mapping` | Per-deliverable source mapping; caveats to carry; O-5 note | O-5/O-6; DC-05; FR-033/035/036; NFR-008 |
| `dev-assist-artifacts/00-axioms/project-axioms.yaml` | AX-001..AX-006 definitions verbatim; verification criteria; reviewer hooks | AX-001..AX-006; NFR-024/025/026 |
| `docs/swarm-architecture.md` | Four-layer pipeline architecture; fast-pass eligibility matrix; SEMANTIC_TYPES gate; regex-oss positioning | DC-01/02; FR-016/018; NFR-007/008 |
