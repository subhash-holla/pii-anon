# Confidence Calibration Analysis: pii-anon-swarm Engines

**Date**: 2026-03-27
**Scope**: Analysis of confidence scoring across all six detection engines and the impact on weighted fusion strategies.

---

## 1. Per-Engine Confidence Scoring Summary

### 1.1 regex-oss (Variable, Calibrated)

**Source**: `src/pii_anon/engines/regex_adapter.py`, `src/pii_anon/engines/regex/confidence.py`

The regex engine is the only engine with a genuinely calibrated, multi-tier confidence system.

| Tier | Confidence Range | Mechanism |
|------|-----------------|-----------|
| Checksum-validated | 0.91-0.99 | Luhn (credit cards, Canadian SIN), IBAN mod-97, ABA routing, VIN check digit, Aadhaar Verhoeff |
| Context-boosted | base + 0.10 (capped at 0.99) | Surrounding keyword match within 50-char window |
| Context-penalized | base - 0.15 (floored at 0.40) | High-FP entity types (SSN, PERSON_NAME, PHONE, etc.) without context keywords |
| Format-only | 0.75-0.85 | Pattern match without checksum validation (e.g., CC format match = 0.80, IBAN format = 0.78) |
| Minimum emit threshold | 0.50 | Findings below this are suppressed entirely |

Key implementation details:
- Credit card: Luhn-valid = 0.94, format-only = 0.80
- IBAN: mod-97 valid = 0.93, format-only = 0.78
- Canadian SIN: Luhn-valid = 0.92, format-only = 0.75
- Aadhaar: Verhoeff-valid = 0.91, format-only = 0.80
- 11 entity types are in `HIGH_FP_TYPES` and receive context penalties when keywords are absent
- Tuning constants (boost, penalty, window, cap, floor) are configurable via `CoreConfig.confidence`

**Assessment**: This is a well-calibrated confidence system. Higher scores correlate with stronger evidence (checksum > context > format-only). The confidence score carries genuine information about detection reliability.

### 1.2 presidio-compatible (Variable, Delegated)

**Source**: `src/pii_anon/engines/presidio_adapter.py`

| Mode | Confidence | Source |
|------|-----------|--------|
| Native Presidio | `float(getattr(item, "score", 0.7))` | Presidio's internal recognizer scores (vary by recognizer and context) |
| Fallback (no Presidio installed) | Fixed 0.9 | Hardcoded for email regex fallback |

When running with native Presidio, confidence varies per recognizer. Presidio's own scores range from approximately 0.01 to 1.0 depending on the recognizer type, context enhancement, and validation. The adapter passes through whatever Presidio reports, defaulting to 0.7 if the score attribute is missing. In fallback mode (Presidio not installed), the adapter uses a simple email regex with a fixed 0.9 confidence.

**Assessment**: Variable when native Presidio is available. The quality of these scores depends entirely on Presidio's internal calibration, which varies significantly across recognizer types. The 0.7 default for missing scores is a concern -- it assigns moderate-high confidence to detections with unknown reliability.

### 1.3 gliner-compatible (Variable, Model-Derived)

**Source**: `src/pii_anon/engines/gliner_adapter.py`

| Mode | Confidence | Source |
|------|-----------|--------|
| Native GLiNER | `float(entity.get("score", 0.75))` | Model's span prediction score (threshold >= 0.5) |
| Fallback SSN regex | Fixed 0.9 | Hardcoded |
| Fallback email regex | Fixed 0.82 | Hardcoded |

The native GLiNER model (`knowledgator/gliner-pii-base-v1.0`) produces continuous scores via `predict_entities()` with a threshold of 0.5. These scores are model-derived probabilities from the span prediction head. In practice, the research report indicates the default range clusters around 0.75-0.82. The 0.75 default for missing scores introduces a moderate-high constant into the average when scores are absent.

**Assessment**: Nominally variable, but the model's scores tend to cluster in a narrow band. The 0.5 threshold filters out low-confidence predictions, but the remaining scores lack the dynamic range needed for meaningful discrimination between strong and weak detections.

### 1.4 scrubadub-compatible (FIXED)

**Source**: `src/pii_anon/engines/scrubadub_adapter.py`

| Mode | Confidence | Source |
|------|-----------|--------|
| Native scrubadub | **Fixed 0.84** | Hardcoded constant (line 83) |
| Fallback regex | Fixed 0.86 | Hardcoded for title+name pattern |

Every detection from native scrubadub receives `confidence=0.84` regardless of entity type, detection quality, or context. The scrubadub library's `Filth` objects do not expose a confidence score, so the adapter assigns an arbitrary constant.

**Assessment**: Completely uncalibrated. The 0.84 value is an arbitrary choice that carries zero information about detection reliability. A correct detection and a false positive receive identical confidence scores.

### 1.5 spacy-ner-compatible (FIXED)

**Source**: `src/pii_anon/engines/spacy_adapter.py`

| Mode | Confidence | Source |
|------|-----------|--------|
| Native spaCy | **Fixed 0.82** | Hardcoded constant (line 81) |
| Fallback regex | Fixed 0.86 | Hardcoded for email pattern |

Every entity from spaCy's NER pipeline receives `confidence=0.82`. Although spaCy's `doc.ents` do not directly expose per-entity confidence, the underlying model does produce scores that could be extracted (e.g., via `doc.cats` for text classification or custom scoring on the NER component). The adapter ignores this entirely.

**Assessment**: Completely uncalibrated. Same issue as scrubadub -- no signal about detection quality. The 0.82 value was chosen to be "reasonable" but provides no discrimination.

### 1.6 stanza-ner-compatible (FIXED)

**Source**: `src/pii_anon/engines/stanza_adapter.py`

| Mode | Confidence | Source |
|------|-----------|--------|
| Native Stanza | **Fixed 0.80** | Hardcoded constant (line 93) |
| Fallback regex | Fixed 0.82 | Hardcoded for phone pattern |

Every entity from Stanza's NER pipeline receives `confidence=0.80`. Like spaCy, Stanza models do produce internal scores that the adapter does not extract.

**Assessment**: Completely uncalibrated. Identical structural problem to spaCy and scrubadub.

---

## 2. Fixed vs. Variable Confidence Classification

| Engine | Confidence Type | Range | Carries Signal? |
|--------|----------------|-------|-----------------|
| regex-oss | **Variable, tiered** | 0.40-0.99 | Yes -- checksum, context, format tiers |
| presidio | **Variable (native) / Fixed (fallback)** | ~0.01-1.0 (native) / 0.9 (fallback) | Partial -- depends on Presidio internals |
| gliner | **Variable (native) / Fixed (fallback)** | ~0.50-0.95 (native) / 0.82-0.90 (fallback) | Weak -- narrow clustering around 0.75-0.82 |
| scrubadub | **Fixed** | 0.84 | No |
| spacy | **Fixed** | 0.82 | No |
| stanza | **Fixed** | 0.80 | No |

Three of six engines (scrubadub, spaCy, Stanza) use completely fixed confidence scores. This means 50% of the engine pool contributes no confidence information to the fusion process.

---

## 3. Impact on Weighted Fusion

### 3.1 The Core Problem

The `WeightedConsensusFusion` strategy (in `src/pii_anon/fusion.py`) computes a weighted average across engines:

```
fused_confidence = sum(engine_confidence * engine_weight) / sum(engine_weight)
```

When fixed-confidence engines participate in this average, they introduce systematic bias:

**Scenario: False Positive Inflation**

Consider a false positive detected by 3 engines:
- regex-oss: correctly assigns low confidence (0.55, context-penalized)
- scrubadub: reports 0.84 (fixed, regardless of quality)
- spaCy: reports 0.82 (fixed, regardless of quality)

With equal weights (1.0 each):
```
fused = (0.55 + 0.84 + 0.82) / 3 = 0.737
```

The regex engine's calibrated low-confidence signal is drowned out. The fused score of 0.737 is moderate-high, suggesting a reliable detection when in fact the only calibrated engine flagged it as uncertain.

**Scenario: True Positive Dilution**

Consider a true positive with strong evidence:
- regex-oss: 0.94 (Luhn-validated credit card)
- stanza: 0.80 (fixed)
- spaCy: 0.82 (fixed)

With equal weights:
```
fused = (0.94 + 0.80 + 0.82) / 3 = 0.853
```

The checksum-validated high confidence (0.94) is pulled down to 0.853 by the fixed-score engines, reducing the confidence gap between validated and unvalidated detections.

### 3.2 The Asymmetry Problem

Fixed-confidence engines are especially harmful for false positives. When an engine correctly reports a low confidence for a weak detection, fixed-confidence engines systematically pull the average upward. But when an engine correctly reports high confidence for a strong detection, fixed-confidence engines pull the average downward. The net effect:

- **False positives become harder to filter** (inflated scores)
- **True positives lose their high-confidence signal** (deflated scores)
- **The fused confidence distribution compresses** toward the fixed values (~0.80-0.84)

### 3.3 Effect on Fusion Strategies

| Strategy | Impact |
|----------|--------|
| `weighted_consensus` | Directly affected: fixed scores bias the weighted average |
| `calibrated_majority` | Indirectly affected: uses weighted consensus internally, then filters by engine count |
| `intersection_consensus` | Uses `min()` confidence -- fixed scores set a floor that prevents low-confidence filtering |
| `union_high_recall` | No impact (passes through individual scores unchanged) |
| `mixture_of_experts` | Partially mitigated by per-entity expert weights, but the underlying confidence values are still uncalibrated |

### 3.4 Entity-Weight Mitigations (Existing)

The `WeightedConsensusFusion` supports `entity_weights` (per-engine, per-entity-type overrides). This allows downweighting specific engines for entity types they are poor at detecting. However, this addresses engine *reliability* rather than confidence *calibration*. Even with low entity weights, the fixed-confidence values still contribute a constant to the average rather than an informative signal.

---

## 4. Recommendations

### R1: Extract Model Scores from spaCy and Stanza (High Priority)

Both spaCy and Stanza NER models produce internal scores that the adapters discard.

**spaCy**: The `Span` objects in `doc.ents` can be scored using `doc.cats` (for text-level classification) or by accessing the underlying `kb_id_` and scorer outputs. More directly, the transition-based NER model maintains per-token probability estimates accessible via custom pipeline components.

**Stanza**: Entity objects carry internal logit scores from the NER tagger that can be converted to probabilities.

Implementation: Modify `spacy_adapter.py` and `stanza_adapter.py` to extract these model-internal scores rather than assigning fixed constants. Even approximate model scores would be more informative than fixed values.

### R2: Apply Platt Scaling or Isotonic Regression to Engine Scores (High Priority)

Use a held-out evaluation dataset to learn a calibration function for each engine's raw scores. Platt scaling (logistic regression on raw scores vs. correctness) or isotonic regression can map each engine's scores to calibrated probabilities.

This would be implemented as a post-processing step in each adapter's `detect()` method, or as a calibration layer in the fusion pipeline. The calibration parameters would be stored per engine and loaded at initialization.

### R3: Replace Fixed Scores in Scrubadub Adapter (Medium Priority)

Scrubadub's `Filth` objects expose a `type` attribute that indicates the detector class that produced the finding. Different detector classes have different empirical precision rates. At minimum, assign per-detector-type confidence values based on measured precision:

```python
_SCRUBADUB_TYPE_CONFIDENCE = {
    "NAME": 0.72,      # Name detection is noisy
    "EMAIL": 0.95,     # Email patterns are reliable
    "PHONE": 0.78,     # Phone patterns have moderate FP rate
    "URL": 0.90,       # URL patterns are reliable
    "CREDENTIAL": 0.88,
}
```

This would replace the blanket 0.84 with at least entity-type-level differentiation.

### R4: Add Confidence Normalization Layer to Fusion (Medium Priority)

Before computing the weighted average, normalize each engine's confidence scores to a common scale. Options:

- **Z-score normalization**: Transform each engine's scores to have mean 0 and std 1, then rescale to [0, 1]. This removes the systematic bias from fixed-score engines (their std becomes 0, reducing their influence).
- **Rank-based normalization**: Replace absolute confidence with rank-order within each engine's findings.
- **Calibration-aware weighting**: Multiply engine weight by a "calibration quality" factor (e.g., the variance of the engine's confidence distribution). Engines with zero variance (fixed scores) get their weight reduced.

Recommended approach: calibration-aware weighting. Add a `calibration_quality` field to each engine adapter that reflects the informativeness of its confidence scores. Engines with fixed scores get `calibration_quality = 0.1` (low but non-zero to avoid completely silencing them), while calibrated engines get `calibration_quality = 1.0`.

### R5: Log Confidence Distribution Metrics in Fusion Audit (Low Priority)

Extend `FusionAuditRecord` to include per-engine confidence statistics for each merged finding:
- Confidence variance across contributing engines
- Whether any contributing engine used a fixed confidence score
- The spread between calibrated and uncalibrated engine scores

This would make the confidence calibration problem visible in production logs and support ongoing monitoring.

### R6: Evaluate ECE (Expected Calibration Error) Per Engine (Low Priority)

Run the evaluation benchmark and compute ECE for each engine independently. This quantifies the calibration problem with a single metric per engine:
- Bin predictions by confidence level
- Compare mean predicted confidence vs. actual precision in each bin
- Report weighted average absolute difference

Expected results based on current architecture:
- regex-oss: ECE < 0.15 (well-calibrated tiers)
- presidio (native): ECE 0.15-0.30 (variable by recognizer)
- gliner (native): ECE 0.20-0.35 (narrow score clustering)
- scrubadub: ECE > 0.50 (fixed score, no calibration)
- spaCy: ECE > 0.50 (fixed score, no calibration)
- stanza: ECE > 0.50 (fixed score, no calibration)

---

## 5. Summary

The confidence calibration problem in pii-anon-swarm is structural: three of six engines assign fixed confidence scores that carry no information about detection quality. When these scores are averaged in weighted fusion, they systematically inflate false-positive confidence and compress the score distribution, undermining the value of the one well-calibrated engine (regex-oss).

The highest-impact fix is extracting real model scores from spaCy and Stanza (R1), followed by adding a calibration-aware weighting mechanism to the fusion layer (R4) so that engines with uninformative confidence scores have reduced influence on the fused result.
