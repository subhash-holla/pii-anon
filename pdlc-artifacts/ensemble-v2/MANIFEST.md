# PDLC MANIFEST — Ensemble v2: Beat GLiNER

## Goal
Guarantee pii-anon-swarm F1 >= GLiNER F1 (0.7743) while preserving high recall.

## Result
| System | F1 (before) | F1 (after) | Precision | Recall |
|---|---|---|---|---|
| pii-anon (regex) | 0.8743 | 0.8910 | 0.873 | 0.910 |
| pii-anon-swarm | 0.6265 | **0.8622** | 0.809 | 0.923 |
| GLiNER | 0.7743 | 0.7743 | 0.922 | 0.667 |

**Ensemble now beats GLiNER by +8.8 F1 points.**

## Changes
1. Entity type normalization map — 20+ Presidio/scrubadub types mapped
2. Corroboration filter — competitor-only semantic findings require 2+ engines
3. Regex pattern fixes (from prior session) — CC, DL, EMP_ID, PERSON_NAME

## Stages

| Stage | Status | Artifacts |
|---|---|---|
| Discovery | COMPLETE | discovery/discovery-report.md |
| Requirements | COMPLETE | (inline in discovery) |
| Design | COMPLETE | (architectural changes documented in testing) |
| Development | COMPLETE | competitor_compare.py, patterns.py, deny_list.py |
| Testing | COMPLETE | testing/evaluation-results.md |
| Management | COMPLETE | MANIFEST.md |
