# Discovery Report: Ensemble v2 — Beat GLiNER

## Problem Statement
pii-anon-ensemble (F1=0.6265) underperforms GLiNER (F1=0.7743) by 15 F1 points despite having much better recall (0.950 vs 0.667). The problem is entirely in precision (0.467 vs 0.922).

## Root Cause Analysis (500-record diagnostic)

### Regex alone: P=0.8723, R=0.9045, F1=0.8881
### Ensemble:    P=0.4470, R=0.9609, F1=0.6102

The ensemble generates **5,079 FPs** vs regex's **566 FPs** — a 9x increase. The 4,513 extra FPs come from competitors.

### FP Breakdown by Entity Type (top sources):

| Entity Type | FPs | Source | Root Cause |
|---|---:|---|---|
| PERSON_NAME | 1,226 | regex + presidio | Over-detection of names |
| IN_PAN | 1,161 | presidio | **Not in normalization map** |
| URL | 901 | presidio | **Not in normalization map** |
| US_DRIVER_LICENSE | 482 | presidio | **Not mapped to DRIVERS_LICENSE** |
| LOCATION | 318 | presidio | Some are valid but unmatched |
| PHONEFILTH | 257 | scrubadub | Normalization may not match |
| PHONE_NUMBER | 202 | presidio | Over-detection |
| US_BANK_NUMBER | 198 | presidio | **Not mapped to BANK_ACCOUNT** |
| EMAIL_ADDRESS | 86 | all | Minor boundary mismatches |
| US_SSN | 70 | presidio | Over-detection |
| US_PASSPORT | 70 | presidio | **Not mapped to PASSPORT** |
| ORGANIZATION | 41 | regex | Over-detection |
| US_ITIN | 24 | presidio | **Not in normalization map** |
| NRP | 16 | presidio | **Not in normalization map** |

### Two Categories of Root Cause:

**Category 1: Missing entity type normalization (3,318 FPs = 65% of all FPs)**
Competitor engines emit entity types not in the normalization map. These pass through as-is and never match ground truth.
- `IN_PAN` → should map to `_BENCHMARK_IGNORE` (no benchmark label)
- `URL` → should map to `_BENCHMARK_IGNORE`
- `US_DRIVER_LICENSE` → should map to `DRIVERS_LICENSE`
- `US_BANK_NUMBER` → should map to `BANK_ACCOUNT`
- `US_PASSPORT` → should map to `PASSPORT`
- `US_ITIN` → should map to `_BENCHMARK_IGNORE`
- `NRP` → should map to `_BENCHMARK_IGNORE`
- `AU_TFN`, `AU_ACN` → should map to `_BENCHMARK_IGNORE`
- `PHONEFILTH` → should map to `PHONE_NUMBER`

**Category 2: Genuine over-detection (1,761 FPs = 35% of all FPs)**
- PERSON_NAME: 1,226 FPs from regex + presidio both detecting names
- LOCATION: 318 FPs
- PHONE_NUMBER: 202 FPs from boundary mismatches
- Other types: ~15 FPs

## Target Metrics
To beat GLiNER (F1=0.7743), ensemble needs:
- Keep recall >= 0.80 (currently 0.950)
- Improve precision from 0.467 to >= 0.80
- Target: F1 >= 0.80

## Enhancement Strategy (Two-Pronged)
1. **Fix normalization map** — eliminate Category 1 FPs entirely (~3,318 FPs removed)
2. **Add confidence-based filtering** — reduce Category 2 FPs via ensemble agreement requirements
