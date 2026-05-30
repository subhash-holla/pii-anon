# Swarm / MoE Prior-Art — Design Inputs (Theme 1)

> **Source Signal** (per `dev-assist-brownfield-assessment` Step 2). This canonical artifact **wraps and cites** prior-art produced by an earlier (non-developer-assistant) PDLC run for the swarm. **All originals are preserved untouched** in `pdlc-artifacts/swarm/`. This dossier is a design INPUT for the Theme-1 swarm MoE-router redesign — not current requirements. Migrated 2026-05-30 (see `../../00-brownfield-assessment/migration-log.md`).

Provenance index: `pdlc-artifacts/swarm/MANIFEST.md` (`sha256:d585625ef1e9d745…`) — goal F1 0.648 → ≥0.85; 5 validated key decisions; full artifact map.

## A. Root-cause analysis (Discovery)

**Source:** `pdlc-artifacts/swarm/discovery/discovery-report.md` (`sha256:87e8f81e35041be0…`)
Validated three-tier root cause of swarm precision failure:
- **Span duplication**: +381 false positives per 100 records — the dominant defect (not the aggregation algorithm).
- **5/6 engines contribute zero marginal value** (heavy redundancy).
- **3/6 engines emit fixed (uncalibrated) confidence.**
Includes current/proposed workflow maps + a phased quick-wins plan with F1-point estimates.
→ *Feeds:* the redesign's span-dedup + selective-activation + calibration decisions.

## B. Empirical precision evidence

**Source:** `pdlc-artifacts/swarm/discovery/precision-diagnosis.md` (`sha256:30e06df68bbe8383…`), backed by raw data `precision-diagnosis.json` (`sha256:96fda3bcd4ba3934…`, KEEP_AS_IS).
Per-entity TP/FP/FN deltas (regex vs swarm) over 100 records / 690 gold labels; per-engine FP attribution: **gliner 412, presidio 397, scrubadub 210, regex 32** FP contributions; 30 concrete FP examples with context.
→ *Feeds:* justification for span-dedup + corroboration gating in the new router; the FP-attribution per engine informs routing priors.

## C. Engine redundancy & diversity

**Source:** `pdlc-artifacts/swarm/discovery/engine-correlation.md` (`sha256:7570e950281694cb…`)
Pairwise Jaccard (e.g. **spaCy+Stanza = 0.891** — near-duplicate); unique-contribution table (**regex 0.074; presidio/gliner/spacy/stanza/scrubadub ≈ 0.000**); co-miss / conditional-co-miss matrices; per-entity recall by engine.
→ *Feeds:* the learned-router's expert-selection / set-cover logic and the shared-vs-routed expert split (DeepSeekMoE shared-expert isolation).

## D. Confidence calibration

**Source:** `pdlc-artifacts/swarm/discovery/confidence-analysis.md` (`sha256:f00c4cf3d01e8a10…`)
Fixed-vs-variable confidence classification per engine (with code-level citations); weighted-fusion inflation/dilution math; 6 recommendations (extract spaCy/Stanza model scores; Platt/temperature scaling; calibration-aware weighting; ECE evaluation).
→ *Feeds:* the redesign's calibration layer + AX-005 (calibrated-abstention).

## E. Prior requirements (constraints/targets — to RE-BASELINE)

**Sources (one prior-art set):** `requirements-document.md` (`sha256:aa7379c0d08bc4a2…`), `functional-requirements.md` (`sha256:097f2ce98790af08…`, FR-001…018 with Given/When/Then), `non-functional-requirements.md` (`sha256:6f0dc47a6a8b067e…`, NFR-001…015).
Quantified targets: latency ≤200ms p50 (Layer-1 ≤2ms), throughput ≥10K rec/hr, meta-learner ≤5ms, **F1 ≥0.85, precision ≥0.80, recall ≥0.83, dominance within 0.02**, reproducibility, backward-compat.
> ⚠ **These targets are DRIFTED vs shipped reality** (shipped swarm F1=0.610, precision=0.486; fails 3/6 floor profiles — see assessment MAJOR #4). The Theme-1 redesign must **re-baseline** to measured/realistic targets, not inherit the aspirational 0.85/0.80.
→ *Feeds:* the new Requirements-stage NFRs for the swarm (with actual-vs-target columns).

## F. Prior baseline architecture (to EVOLVE)

**Source:** `pdlc-artifacts/swarm/design/final-architecture.md` (`sha256:99d8ae6ee61c5b3d…`)
Hybrid 4-layer architecture (Proposal A base + B's selective-activation/informativeness/manifest/dominance); ~2,200 LOC inventory; data structures (`SpanCandidate`, `SwarmConfig`, `TrainingManifest`); pure-Python Dawid-Skene EM; 20-feature XGBoost meta-learner table; training pipeline; risk-mitigation table.
→ *Feeds:* the **baseline-to-evolve** for the MoE-router redesign; the new design documents deltas against it.

## G. Design alternatives (DIVERGE set — MERGE G1)

**Sources:** `proposal-A.md` (`sha256:8ef6a283ccdaef84…`, simplicity: pure-Python DS, logistic fallback, single-file core) + `proposal-B.md` (`sha256:8c5ce80f8e00265a…`, research-aligned: selective activation via greedy set-cover, InformativenessScorer, token-level Dawid-Skene, TrainingManifest, Platt scaling).
→ *Feeds:* documented alternates for the Theme-1 Design diamond; several B-only ideas (set-cover selection, informativeness) are directly relevant to the learned-router work.

## H. Design decision rationale (CONVERGE)

**Source:** `pdlc-artifacts/swarm/design/design-critique.md` (`sha256:bffc342210b01dc0…`)
7-criterion weighted Pugh matrix (**A 8.3 vs B 6.0**) with arithmetic; per-criterion analysis; "unique ideas worth preserving" per proposal; hybrid recommendation + element-source table.
→ *Feeds:* the **template** for the Theme-1 redesign Pugh cascade (the assessment flagged this as exemplary DIVERGE/CONVERGE rigor to mirror).
