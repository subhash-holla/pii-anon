# Legacy Artifact Inventory — pii-anon-code `pdlc-artifacts/`

> Consumed by `/dev-assist-migrate` (interactive, per-item approval; never deletes; preserves originals; computes content fingerprints at migration time). Produced by the M1 brownfield assessment fan-out (2 inventory agents), read-only.
>
> **24 files inventoried · 0 ARCHIVE · 0 deletions.** The legacy tree is a foreign-schema PDLC from an older (non-developer-assistant) tool. Discipline: WRAP > MIGRATE > ARCHIVE. Most files are high-value **Theme-1 (swarm MoE redesign) design inputs** → WRAP into `03-design/_inputs/`; the eval evidence → `05-testing/benchmark-evidence/`.
>
> **Status-drift flag:** the foreign `pdlc-artifacts/MANIFEST.md` marks all 6 stages COMPLETE (2026-03-16), but 4 of 6 stage dirs (`discovery/ requirements/ testing/ management/`) are **empty on disk**. The canonical `dev-assist-artifacts/MANIFEST.md` must NOT inherit the false-COMPLETE claim.

```yaml
artifact_inventory:
  generated_at: "2026-05-30"
  source_directory: ./pdlc-artifacts/
  files_inventoried: 24
  classifications: { KEEP_AS_IS: 4, MIGRATE: 0, WRAP: 15, MERGE: 5, ARCHIVE: 0 }
  notes: >
    content_fingerprints are computed by /dev-assist-migrate at apply time.
    MERGE groups: (G1) proposal-A.md + proposal-B.md -> swarm-design-alternatives.md;
    (G2) ROUND3_EVAL_SUMMARY.md + round3-eval.txt -> detection-accuracy-round3-postfix.md;
    (G3) profile-eval-round2.txt + profile-eval.json -> profile-eval-round2-per-segment.{md,json}.
    WRAP set: the three swarm/requirements/*.md are cited as ONE prior-requirements wrapper.

  items:
    # ---- Group A: technical subprojects (swarm / ensemble-v2 / speed-boost) — Theme-1 design inputs ----
    - path: pdlc-artifacts/swarm/MANIFEST.md
      classification: KEEP_AS_IS
      confidence: STRONG
      rationale: Provenance index for the swarm investigation (goal F1 0.648->0.85, 5 key decisions, artifact map). Cited, not transformed.
      action: Leave in place; cite as the authoritative index of the swarm prior-art bundle.
      target_canonical_path: ""
      target_canonical_section: Prior-art provenance index (cited from 00-brownfield-assessment + 03-design)

    - path: pdlc-artifacts/swarm/discovery/discovery-report.md
      classification: WRAP
      confidence: STRONG
      rationale: Validated 3-tier root cause (span duplication +381 FP/100 rec; 5/6 engines zero marginal value; 3/6 fixed confidence) + workflow maps + phased plan. Canonical-shaped but from a prior run.
      action: Create a citing wrapper that maps each finding (P/G/O) to a new-design decision; original untouched.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-discovery-root-cause.md
      target_canonical_section: Design inputs / Prior-art root-cause analysis

    - path: pdlc-artifacts/swarm/discovery/precision-diagnosis.md
      classification: WRAP
      confidence: STRONG
      rationale: Crown-jewel precision evidence — per-entity TP/FP/FN regex-vs-swarm (100 rec/690 labels), per-engine FP attribution (gliner 412, presidio 397...), 30 FP examples. Justifies span-dedup/corroboration in the redesign.
      action: Citing wrapper pulling headline numbers; references the .json sibling as source-of-truth.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-precision-diagnosis.md
      target_canonical_section: Design inputs / Empirical precision-failure evidence

    - path: pdlc-artifacts/swarm/discovery/precision-diagnosis.json
      classification: KEEP_AS_IS
      confidence: STRONG
      rationale: Raw machine-readable backing data for precision-diagnosis.md. No transform appropriate for a raw data file.
      action: Leave in place; referenced by the precision-diagnosis WRAP wrapper as the raw dataset.
      target_canonical_path: ""
      target_canonical_section: Design inputs / raw dataset behind swarm-precision-diagnosis

    - path: pdlc-artifacts/swarm/discovery/engine-correlation.md
      classification: WRAP
      confidence: STRONG
      rationale: Redundancy/diversity study — pairwise Jaccard (spaCy+Stanza 0.891), unique-contribution table (regex 0.074; others 0.000), co-miss matrices. Direct basis for engine-pruning / selective-activation in the MoE router.
      action: Citing wrapper summarizing the redundancy verdict; feeds expert-selection/set-cover design.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-engine-correlation.md
      target_canonical_section: Design inputs / Engine redundancy and diversity evidence

    - path: pdlc-artifacts/swarm/discovery/confidence-analysis.md
      classification: WRAP
      confidence: STRONG
      rationale: Calibration investigation with code-level citations — fixed-vs-variable confidence per engine, fusion inflation/dilution math, 6 recommendations (Platt/temperature scaling, ECE). Informs the calibration layer.
      action: Citing wrapper capturing the R1-R6 recommendations; links to the design's confidence-calibration section.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-confidence-calibration.md
      target_canonical_section: Design inputs / Confidence calibration evidence and recommendations

    - path: pdlc-artifacts/swarm/requirements/requirements-document.md
      classification: WRAP
      confidence: STRONG
      rationale: Canonical-shaped req summary (18 FR + 15 NFR + traceability) but from a PRIOR run; must not be mis-attributed as current Requirements. NFR targets are drifted (F1 0.85 vs shipped 0.610).
      action: Cite as prior-art constraints/targets feeding the new design; do NOT relocate into 02-requirements as current.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-prior-requirements.md
      target_canonical_section: Design inputs / Prior-art requirements and targets

    - path: pdlc-artifacts/swarm/requirements/functional-requirements.md
      classification: WRAP
      confidence: PARTIAL
      rationale: FR-001..018 with Given/When/Then (the desired swarm behaviors). Backs the requirements-document summary (could MERGE); WRAPped as one prior-requirements set.
      action: Cover under the same prior-requirements wrapper (cite all three req files as one prior-art set).
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-prior-requirements.md
      target_canonical_section: Design inputs / Prior-art functional requirements (FR-001..018)

    - path: pdlc-artifacts/swarm/requirements/non-functional-requirements.md
      classification: WRAP
      confidence: PARTIAL
      rationale: NFR-001..015 with quantified thresholds (latency<=200ms p50, F1>=0.85, dominance<=0.02...). The binding targets the new design must honor/re-baseline.
      action: Cover under the same prior-requirements wrapper (cite quantified NFR thresholds as design targets to re-baseline).
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-prior-requirements.md
      target_canonical_section: Design inputs / Prior-art non-functional thresholds (NFR-001..015)

    - path: pdlc-artifacts/swarm/design/final-architecture.md
      classification: WRAP
      confidence: STRONG
      rationale: Convergence output — hybrid 4-layer architecture, file inventory (~2,200 LOC), data structures, pure-Python Dawid-Skene, 20-feature XGBoost table, training pipeline. The single most reusable Theme-1 baseline-to-evolve.
      action: Citing wrapper framing this as the prior baseline; the new design documents deltas against it.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-prior-architecture.md
      target_canonical_section: Design inputs / Prior-art baseline architecture (to evolve)

    - path: pdlc-artifacts/swarm/design/design-proposals/proposal-A.md
      classification: MERGE
      confidence: PARTIAL
      rationale: Simplicity DIVERGE alternate (pure-Python DS, logistic fallback, single-file core). Merge with proposal-B into one alternatives dossier (G1).
      action: MERGE with proposal-B into a canonical alternates dossier preserving both proposals' distinctive ideas; cite both originals untouched.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-design-alternatives.md
      target_canonical_section: Design inputs / Alternatives considered (Proposal A — simplicity)

    - path: pdlc-artifacts/swarm/design/design-proposals/proposal-B.md
      classification: MERGE
      confidence: PARTIAL
      rationale: Research-aligned DIVERGE alternate (selective activation/set-cover, informativeness, token-level DS, training manifest — several adopted into final-architecture). Merge with proposal-A (G1).
      action: MERGE with proposal-A into the alternates dossier; preserve B's distinctive ideas; cite both originals.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-design-alternatives.md
      target_canonical_section: Design inputs / Alternatives considered (Proposal B — research-aligned)

    - path: pdlc-artifacts/swarm/design/design-critique.md
      classification: WRAP
      confidence: STRONG
      rationale: CONVERGE decision record — 7-criterion weighted Pugh matrix (A 8.3 vs B 6.0), per-criterion analysis, hybrid recommendation + element-source table. The "why" behind the baseline architecture; the template for the Theme-1 redesign diamond.
      action: Citing wrapper capturing the decision; link from the architecture wrapper and the alternatives dossier.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/swarm-design-decision-rationale.md
      target_canonical_section: Design inputs / Design decision rationale (Pugh-style trade study)

    - path: pdlc-artifacts/ensemble-v2/MANIFEST.md
      classification: KEEP_AS_IS
      confidence: STRONG
      rationale: Management record of the "Beat GLiNER" investigation (swarm F1 0.6265->0.8622, +8.8 pts vs GLiNER; 3 changes). Provenance index, cited not transformed.
      action: Leave in place; cite as the ensemble-v2 prior-art index.
      target_canonical_path: ""
      target_canonical_section: Prior-art provenance index (ensemble-v2)

    - path: pdlc-artifacts/ensemble-v2/discovery/discovery-report.md
      classification: WRAP
      confidence: STRONG
      rationale: 500-rec FP-source diagnosis — 9x FP explosion split into Category 1 (missing entity-type normalization, 65% of FPs) vs Category 2 (genuine over-detection, 35%). Distinct, actionable input for the redesign's normalization layer.
      action: Citing wrapper capturing the two-category FP root cause; feeds normalization + corroboration design.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/ensemble-v2-fp-source-diagnosis.md
      target_canonical_section: Design inputs / Competitor FP-source diagnosis (normalization gaps)

    - path: pdlc-artifacts/ensemble-v2/testing/evaluation-results.md
      classification: WRAP
      confidence: STRONG
      rationale: Validated outcome — 1000-rec F1 0.8622, ~4,100 FPs eliminated, per-segment results incl. the edge_cases regression (-5.7 pts), 2253/0 tests. Proof normalization+corroboration work, with the documented caveat.
      action: Citing wrapper summarizing the validated gain + edge_cases caveat; pair with the FP-source wrapper.
      target_canonical_path: dev-assist-artifacts/03-design/_inputs/ensemble-v2-evaluation-evidence.md
      target_canonical_section: Design inputs / Validated ensemble fixes and per-segment results

    - path: pdlc-artifacts/speed-boost/MANIFEST.md
      classification: KEEP_AS_IS
      confidence: STRONG
      rationale: Self-contained speed-profile fix record — speed detector scanned only EMAIL/PHONE (24% of labels); swap to full RegexEngineAdapter (~0.7ms/rec, F1 0.407->0.889). Complete in one doc.
      action: Leave in place; cite from 00-brownfield-assessment / 03-design performance notes.
      target_canonical_path: ""
      target_canonical_section: Prior-art provenance + speed-profile evidence

    # ---- Group B: foreign-schema stage dirs + root MANIFEST ----
    - path: pdlc-artifacts/MANIFEST.md
      classification: WRAP
      confidence: STRONG
      rationale: Foreign-schema 6-stage root manifest. A canonical MANIFEST already exists, so do NOT migrate over it. Carries a "Changes Made" log (6 code fixes w/ line refs) + status drift (4 empty dirs vs 6 COMPLETE).
      action: Provenance note citing it; transcribe the "Changes Made" log into the dev change log; FLAG the false-COMPLETE drift so it is not inherited.
      target_canonical_path: dev-assist-artifacts/04-development/_provenance/legacy-pdlc-manifest-moe-guarantee.md
      target_canonical_section: Provenance / legacy artifact index + Development change log

    - path: pdlc-artifacts/design/moe-guarantee-analysis.md
      classification: WRAP
      confidence: STRONG
      rationale: 611-line formal analysis — Mixtral-vs-pii-anon MoE comparison, the "Ensemble Superset Guarantee" theorem + proof + failure conditions, merge() bug root cause (moe.py:388-405), proposed non_routed_floor fix. The academic backbone for AX-003 (recall-floor). Note stale /sessions/... paths in Appendix.
      action: Canonical System/Architecture artifact summarizing the MoE decision + superset invariant, citing this as the authoritative proof; carry the fix forward with traceability. Flag stale appendix paths.
      target_canonical_path: dev-assist-artifacts/03-design/moe-architecture-and-guarantee.md
      target_canonical_section: System & Architecture — ensemble fusion design + MoE performance-floor invariant

    - path: pdlc-artifacts/development/round1-eval.txt
      classification: WRAP
      confidence: PARTIAL
      rationale: Baseline per-entity eval (pii_anon_benchmark_v1, 200 rec/1636 labels) — the "before" snapshot cited by later rounds. Measurement output → Testing/NFR evidence.
      action: Cite as the Round-1 baseline "before" snapshot in a detection-accuracy benchmark-evidence artifact; preserve verbatim.
      target_canonical_path: dev-assist-artifacts/05-testing/benchmark-evidence/detection-accuracy-round1-baseline.txt
      target_canonical_section: NFR verification — detection accuracy baseline

    - path: pdlc-artifacts/development/round3-eval.txt
      classification: WRAP
      confidence: PARTIAL
      rationale: Post-fix re-eval (200 rec, synthetic fallback 142 labels) incl. "MOE UNION GUARANTEE TEST = PASSED" — empirical confirmation of the AX-003 theorem. Self-cautions the R1-vs-R3 comparison is cross-dataset.
      action: Cite as the Round-3 "after" snapshot + union-guarantee verification; carry the synthetic + cross-dataset caveats into canonical text.
      target_canonical_path: dev-assist-artifacts/05-testing/benchmark-evidence/detection-accuracy-round3-postfix.txt
      target_canonical_section: NFR verification — detection accuracy "after" + MoE superset-guarantee verification

    - path: pdlc-artifacts/development/ROUND3_EVAL_SUMMARY.md
      classification: MERGE
      confidence: PARTIAL
      rationale: Narrative companion to round3-eval.txt (same event/metrics + how-to-run + bug-fixes-verified). MERGE the prose (analysis) with the .txt (raw) into one Round-3 artifact (G2).
      action: Fold as the analysis section of the canonical Round-3 evidence doc; cite round3-eval.txt as raw data; preserve both originals.
      target_canonical_path: dev-assist-artifacts/05-testing/benchmark-evidence/detection-accuracy-round3-postfix.md
      target_canonical_section: NFR verification — Round-3 analysis + MoE union-guarantee summary

    - path: pdlc-artifacts/development/profile-eval-round2.txt
      classification: MERGE
      confidence: STRONG
      rationale: Per-segment summary (1000 rec) — 5 regex fixes, P 0.758->0.873 / R 0.840->0.910 / F1 0.797->0.891, FP -50.6%. Field-for-field identical to profile-eval.json top-level aggregates. Textbook MERGE pair (G3).
      action: MERGE with profile-eval.json — .txt as the human-readable summary citing the JSON; preserve both originals.
      target_canonical_path: dev-assist-artifacts/05-testing/benchmark-evidence/profile-eval-round2-per-segment.md
      target_canonical_section: NFR verification — per-segment detection accuracy (Round-2 post-fix)

    - path: pdlc-artifacts/development/profile-eval.json
      classification: MERGE
      confidence: STRONG
      rationale: 7272-line per-segment dataset (difficulty × scenario × language × entity-type, regex+ensemble P/R/F1/TP/FP/FN). The structured data behind profile-eval-round2.txt (verified match). High-value, never archive (G3).
      action: MERGE — register as the machine-readable dataset under benchmark-evidence; the .md summary points at it; preserve original.
      target_canonical_path: dev-assist-artifacts/05-testing/benchmark-evidence/profile-eval-round2-per-segment.json
      target_canonical_section: NFR verification — raw per-segment metrics dataset
```
