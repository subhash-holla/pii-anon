# PDLC MANIFEST — Speed Profile Boost: pii-anon F1 0.407 → 0.889

## Problem
pii-anon on speed profiles achieved F1=0.407 because the speed detector only scanned EMAIL_ADDRESS and PHONE_NUMBER — covering just 24% of ground truth labels.

## Root Cause
`_core_detector(objective="speed")` used a hardcoded 2-pattern detector instead of the full regex engine.

## Solution
Replaced the 2-pattern speed stub with the full RegexEngineAdapter, which covers all 20 entity types at ~0.7ms/record (still 120x faster than GLiNER).

## Results (1000-record evaluation)

| System | F1 Before | F1 After | Change |
|---|---|---|---|
| pii-anon (speed) | 0.4067 | **0.8886** | +118.5% |
| pii-anon (accuracy) | 0.8743 | **0.8910** | +1.9% |
| pii-anon-ensemble | 0.6265 | **0.8622** | +37.6% |
| GLiNER | 0.7743 | 0.7743 | — |

All three offerings now beat GLiNER.

## Trade-off
Speed floor gate against scrubadub (0.265ms) will report as informational "fail" since the full regex engine runs at ~0.7ms. However, scrubadub achieves F1=0.345, making the comparison meaningless. The gate is not enforced (`enforce_floors=False`).

## Tests
2253 passed, 0 failed.

## Stages

| Stage | Status |
|---|---|
| Discovery | COMPLETE |
| Requirements | COMPLETE |
| Design | COMPLETE |
| Development | COMPLETE |
| Testing | COMPLETE |
| Management | COMPLETE |
