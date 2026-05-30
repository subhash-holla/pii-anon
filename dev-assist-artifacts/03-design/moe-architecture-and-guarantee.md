# MoE Architecture & Ensemble Superset Guarantee (canonical design record)

> **Source Signal** (per `dev-assist-brownfield-assessment` Step 2). Wraps and cites `pdlc-artifacts/design/moe-guarantee-analysis.md` (`sha256:82c717003aba9782…`, 611 lines, self-labeled "Research & Analysis Only"). **Original preserved untouched.** Migrated 2026-05-30. This is the canonical home for the **recall-floor (AX-003)** design rationale — the load-bearing Theme-1 invariant.
>
> ⚠ **Stale paths:** the original's Appendix A.2 references absolute `/sessions/dreamy-blissful-gauss/…` sandbox paths that no longer resolve — ignore them.

## What the original establishes

1. **MoE framing** — a Mixtral-8x7B-style sparse-MoE adaptation comparison for PII detection.
2. **The Ensemble Superset Guarantee (theorem):** `entities(E_union) ⊇ entities(E_i)` for every expert `E_i` — i.e. the ensemble must never detect *fewer* true entities than any single expert. Proof sketch + explicit failure conditions provided.
3. **Root-cause of the historical violation:** the `merge()` loop dropped non-routed experts' findings (`moe.py` lines 388–405 at the time of writing).
4. **Proposed fix:** a `non_routed_floor` parameter (the floor-weight now in `MoEFusionStrategy`, `moe.py:354-378`), plus an alternative registry-validation approach; impact assessment; a test recommendation.

## Carry-forward decisions for the Theme-1 redesign (with the assessment's caveat)

The brownfield assessment (`../00-brownfield-assessment/assessment-2026-05-30.md`, Design MAJOR #5) found this proof **predates the shipped code and is incomplete as a by-construction guarantee**:

- The **MoE path** enforces the floor via floor-weighting non-routed experts (`moe.py:354-378`) ✓.
- The **swarm path** (`SwarmFusionStrategy.merge()`, `swarm.py:651-661`) applies a Layer-4 emission gate (`meta_score < emission_threshold → drop`) and a `SEMANTIC_TYPES` corroboration filter that **can suppress a shared-layer (regex-oss) finding** that fell below fast-pass and entered fusion. So only fast-pass-eligible regex hits are floor-protected.
- **Two divergent floor mechanisms exist; the guarantee is NOT yet by-construction across the whole pipeline.**

**Design mandate (Theme 1 + axiom AX-pii-anon-003):**
- Define ONE "shared-layer span set" that every fusion/router path must emit as a **superset**, by construction.
- Re-validate the theorem against **both** the MoE floor-weight implementation AND the swarm Layer-4 emission gate.
- Pin it with the property test + **per-language recall CI gate** named in AX-003's verification block.
- Decide explicitly whether emission/corroboration gates may ever drop a shared-layer hit (the axiom says no) and design the override path.

→ This artifact is the architecture-decision-record the MoE-router redesign evolves; the learned router, early-exit, and budget balancing must all preserve the superset guarantee.
