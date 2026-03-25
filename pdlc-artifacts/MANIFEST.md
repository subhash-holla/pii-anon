# PDLC Manifest: MoE Ensemble Guarantee & Performance Improvement

## Project
**Goal**: Guarantee that pii-anon-swarm (Mixtral-inspired MoE) always performs >= best individual expert, and iteratively improve per-entity-type detection for both pii-anon and pii-anon-swarm.

## Stage Status

| Stage | Status | Started | Completed |
|-------|--------|---------|-----------|
| Discovery | COMPLETE | 2026-03-16 | 2026-03-16 |
| Requirements | COMPLETE | 2026-03-16 | 2026-03-16 |
| Design | COMPLETE | 2026-03-16 | 2026-03-16 |
| Development | COMPLETE | 2026-03-16 | 2026-03-16 |
| Testing | COMPLETE | 2026-03-16 | 2026-03-16 |
| Management | COMPLETE | 2026-03-16 | 2026-03-16 |

## Artifacts

### Design
- `design/moe-guarantee-analysis.md` — Formal MoE architecture analysis and guarantee proof

### Development
- `development/round1-eval.txt` — Baseline per-entity-type evaluation (200 records)
- `development/round3-eval.txt` — Post-fix re-evaluation with guarantee verification

## Changes Made

### 1. MoE Guarantee Bug Fix (moe.py)
- **Bug**: MoEFusionStrategy.merge() dropped findings from non-routed experts (line 388-405)
- **Fix**: Non-routed experts now receive `min_expert_weight` floor weight instead of being skipped
- **Guarantee**: `entities(ensemble) ⊇ entities(best_individual_expert)` now holds provably

### 2. Registry Completeness (moe.py)
- **Bug**: regex-oss declared PERSON_NAME, ORGANIZATION, USERNAME only in `entity_weaknesses`, not `entity_strengths`
- **Fix**: All 23 entity types the regex engine can detect are now in `entity_strengths`
- Added: CREDIT_CARD_FRAGMENT, NATIONAL_ID, MEDICAL_RECORD_NUMBER

### 3. Benchmark Entity Type Normalization (competitor_compare.py)
- Mapped DATE_ISO, DATE_TIME, GPS_COORDINATES, URL_WITH_PII, CREDIT_CARD_FRAGMENT to `_BENCHMARK_IGNORE`
- These are valid production detections but don't exist in benchmark ground truth
- Eliminates ~151 false positives per 200 records

### 4. Regex Pattern Tightening (regex/patterns.py)
- EMPLOYEE_ID: Tightened pattern to require context keywords; FP reduced 100→23 (77%)
- CREDIT_CARD_FRAGMENT: Required explicit context keywords; removed standalone masked pattern

### 5. Presidio Preflight (runtime_preflight.py)
- Added model probe for presidio to verify en_core_web_sm spacy model availability
- Previously only checked import, causing silent failures when model missing

### 6. Auto-Cleanup Pipeline (run_competitor_benchmark.py)
- Added `_auto_cleanup_old_artifacts()` to archive old benchmark results before each run
- Added `_update_readme_benchmark_section()` to auto-update README with latest results
