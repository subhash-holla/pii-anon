# Discovery Report: pii-anon-swarm

**Date**: 2026-03-27
**Status**: COMPLETE

---

## 1. Problem Statement

pii-anon-swarm (the MoE ensemble offering) achieves F1=0.648 on the 117K-record benchmark — **worse than the regex-only engine** (F1=0.792) and worse than GLiNER alone (F1=0.763). The swarm was designed to improve accuracy by combining 6 engines, but it degrades rather than improves quality. This must be fixed for the swarm to be a viable product.

## 2. Root Cause Analysis (Validated)

### 2.1 Primary Root Cause: Span Duplication (P-1)
**Impact**: Critical | **Validated**: Yes (100-record diagnostic)

The swarm generates **+381 false positives** compared to regex-only while gaining only **+8 true positives**. The root cause: multiple engines detect the **same entity** with slightly different span boundaries, and the IoU-based clustering (threshold 0.5) fails to merge them. This creates duplicate findings that each count as a false positive.

Top FP-producing entity types (excess over regex-only):
- EMAIL_ADDRESS: +112 FPs
- PERSON_NAME: +103 FPs
- US_SSN: +74 FPs
- PHONE_NUMBER: +51 FPs

### 2.2 Secondary Root Cause: Engine Redundancy (P-2)
**Impact**: High | **Validated**: Yes (50-record correlation analysis)

5 of 6 engines have **zero marginal value** — every entity they detect is already caught by regex. spaCy and Stanza are essentially identical (Jaccard similarity 0.891, conditional co-miss 94.6%). Adding redundant engines only adds noise without improving recall.

| Engine | Recall | Marginal Value | Unique Contributions |
|--------|--------|---------------|---------------------|
| regex | 0.988 | 0.074 | 25 entities |
| gliner | 0.727 | 0.000 | 0 |
| presidio | 0.691 | 0.000 | 0 |
| stanza | 0.374 | 0.000 | 0 |
| spacy | 0.344 | 0.000 | 0 |
| scrubadub | 0.252 | 0.000 | 0 |

### 2.3 Tertiary Root Cause: Uncalibrated Confidence (P-3)
**Impact**: High | **Validated**: Yes (code analysis)

3 of 6 engines use completely **fixed confidence scores** (scrubadub=0.84, spaCy=0.82, stanza=0.80). When weighted fusion averages these with regex's calibrated scores, FP confidence gets inflated (e.g., regex says 0.55 but fused becomes 0.74) and TP confidence gets diluted.

| Engine | Confidence Type | Signal Quality |
|--------|----------------|----------------|
| regex-oss | Variable, tiered (0.40-0.99) | High |
| presidio | Variable (passthrough) | Moderate |
| gliner | Variable (narrow range) | Weak |
| scrubadub | Fixed (0.84) | None |
| spacy | Fixed (0.82) | None |
| stanza | Fixed (0.80) | None |

### 2.4 Research-Identified Issues (Partially Applicable)

| Research Finding | Applies to Our System? | Notes |
|-----------------|----------------------|-------|
| Dawid-Skene > majority voting | Partially | Our fusion uses weighted average, not majority vote. DS would help but the bigger win is fixing span duplication. |
| HMM sequence correction | Low priority | We work at span level, not BIO token level. Span merging is our equivalent. |
| Correlated engine errors | Confirmed | spaCy+Stanza correlation = 0.891. But even diverse engines don't help because regex already has 0.988 recall. |
| Confidence calibration (ECE) | Confirmed | 3 engines have ECE > 0.50 (fixed scores). |
| Four-layer architecture | Good fit | We already have layer 1 (regex) and layer 2 (NER). Missing: meta-learner and validation. |
| Fine-tuned DeBERTa-v3 | Future work | Requires new dependency and training pipeline. |

## 3. Personas

### 3.1 Library Developer (Primary)
- **Role**: Python developer integrating pii-anon into data pipelines
- **Goal**: Use pii-anon-swarm for highest possible F1 on mixed PII types
- **Pain**: Swarm is slower AND less accurate than regex-only, making it pointless
- **Technical Level**: Advanced Python, familiar with NER concepts

### 3.2 Compliance Engineer
- **Role**: Ensures PII processing meets regulatory requirements
- **Goal**: Maximize recall (missed PII is categorically worse than false positives)
- **Pain**: Swarm has best recall (0.839) but produces too many false alarms for practical use
- **Technical Level**: Moderate Python, needs clear configuration options

## 4. Key Findings Summary

| ID | Finding | Type | Severity | Entity Count |
|----|---------|------|----------|--------------|
| P-1 | Span duplication: engines re-detect same entities with different boundaries, creating duplicate FPs | Pain | Critical | +381 FPs/100 records |
| P-2 | Engine redundancy: 5/6 engines have zero marginal value | Pain | High | 0 unique TPs |
| P-3 | Fixed confidence scores on 3/6 engines inflate FP confidence and compress TP/FP separation | Pain | High | 50% of engines |
| G-1 | No span-level deduplication/merging for near-matches | Gap | Critical | All entity types |
| G-2 | No corroboration requirement (single engine FP accepted) | Gap | High | PERSON_NAME, EMAIL |
| G-3 | No confidence calibration layer in fusion pipeline | Gap | High | — |
| G-4 | No engine pruning based on marginal value | Gap | Medium | spaCy, Stanza |
| O-1 | Aggressive span merging (lower IoU or character-distance) could eliminate most FPs | Opportunity | Critical | +381 FPs recoverable |
| O-2 | Require 2+ engine corroboration for semantic entity types | Opportunity | High | ~200 FPs filterable |
| O-3 | Calibration-aware weighting (downweight fixed-confidence engines) | Opportunity | High | — |
| O-4 | Drop spaCy or Stanza (redundant pair) to reduce noise | Opportunity | Medium | — |
| O-5 | Confidence cascade: accept regex-only when conf > 0.90 | Opportunity | Medium | ~60% of detections |

## 5. Workflow Maps

### Current Swarm Detection Flow

```
[Input: text + profile(mode=mixture_of_experts)]
        |
        v
[Orchestrator: resolve ExecutionPlan]
        |
        v
[Run all 6 engines in parallel] ──────────────── (P-2: redundant engines add noise)
        |
        v
[Collect raw EngineFinding[] from each engine]
        |
        v
[MoERouter: select top-K=3 experts per entity type]
        |
        v
[MoEFusionStrategy: cluster by IoU >= 0.5] ───── (P-1: boundary mismatches → duplicates)
        |
        v
[Weighted-average confidence per cluster] ──────── (P-3: fixed scores inflate FP conf)
        |
        v
[Weighted-majority boundary voting]
        |
        v
[Output: EnsembleFinding[] with confidence]
```

### Proposed Swarm Detection Flow

```
[Input: text + profile(mode=mixture_of_experts)]
        |
        v
[Layer 1: Regex fast-pass] ────────────────────── (O-5: accept high-conf regex immediately)
        |
   <Decision: regex conf > threshold?>
   |Yes                    |No / Low conf
   v                       v
[Accept directly]    [Layer 2: Run NER engines (GLiNER, Presidio)]  ── (O-4: drop spaCy/Stanza)
                           |
                           v
                     [Layer 3: Span deduplication] ──────────────── (O-1: character-distance merge)
                           |
                           v
                     [Corroboration filter] ──────────────────── (O-2: require 2+ engines)
                           |
                           v
                     [Calibration-aware fusion] ──────────────── (O-3: weight by score quality)
                           |
                           v
                     [Layer 4: Validation (checksum, format)] ─── (existing regex validators)
                           |
                           v
                     [Output: EnsembleFinding[] with calibrated confidence]
```

## 6. Constraints

- **Technical**: Must work within existing EngineAdapter interface and fusion pipeline
- **Dependencies**: No new required dependencies (crowd-kit, HMM libraries are optional)
- **Performance**: Swarm latency currently ~90ms/record; should not exceed 150ms/record
- **Backward Compatibility**: ProcessingProfileSpec API must remain compatible
- **Testing**: All 2273 existing tests must continue to pass; new swarm tests required

## 7. Recommended Approach (Quick Wins First)

### Phase 1: Fix Span Deduplication (Critical — estimated +15-20 F1 points)
- Lower IoU threshold or add character-distance-based merging
- Deduplicate findings that match the same gold entity but differ by a few characters

### Phase 2: Add Corroboration Filter (High — estimated +5-10 F1 points)
- Require 2+ engines to agree on semantic types (PERSON_NAME, ORGANIZATION, LOCATION)
- Single-engine regex detections for structured types are already high-precision

### Phase 3: Calibration-Aware Fusion (High — estimated +3-5 F1 points)
- Downweight engines with fixed confidence scores
- Extract real model scores from spaCy/Stanza if feasible

### Phase 4: Engine Pruning & Cascade (Medium)
- Remove one of spaCy/Stanza (redundant)
- Add confidence cascade: skip NER engines when regex is highly confident

**Target**: F1 >= 0.85 (from current 0.648) after Phases 1-3.

---

*Discovery phase complete. 2 personas, 2 workflows, 12 findings (P:3, G:4, O:5). Ready for Requirements.*
