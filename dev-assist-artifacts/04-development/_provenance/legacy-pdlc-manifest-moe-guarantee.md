# Legacy PDLC Manifest — MoE-Guarantee Initiative (provenance)

> **Source Signal** (per `dev-assist-brownfield-assessment` Step 2). Wraps and cites the foreign-schema `pdlc-artifacts/MANIFEST.md` (`sha256:943c7bed243ccb1c…`) from an earlier (non-developer-assistant) PDLC tool. **Original preserved untouched.** Migrated 2026-05-30. This does **NOT** supersede the canonical `dev-assist-artifacts/MANIFEST.md`.

## ⚠ Status drift (do NOT inherit)
The foreign manifest marks **all 6 stages COMPLETE (2026-03-16)**, but **4 of 6 stage dirs are empty on disk** (`discovery/ requirements/ testing/ management/`). The COMPLETE claim is aspirational, not artifact-backed. The canonical MANIFEST must not inherit it.

## Development change-log (cited prior work — the "Changes Made" log)
The foreign manifest records 6 concrete code fixes for the MoE-guarantee initiative (with line refs at time of writing):
1. `moe.py` — non-routed-expert floor-weight fix (the superset-guarantee fix; see `../../03-design/moe-architecture-and-guarantee.md`).
2. `competitor_compare.py` — comparator/benchmark fixes.
3. `regex/patterns.py` — regex pattern fixes.
4. `runtime_preflight.py` — preflight gate.
5. `run_competitor_benchmark.py` — benchmark runner.
6. (+ entity-type normalization / corroboration work, cross-referenced in `../../03-design/_inputs/ensemble-v2-and-speed-prior-art.md`).

→ These are prior, already-shipped changes. They are recorded here as the historical development provenance behind the current swarm; new Theme-1 Development stories will supersede/evolve them with proper FR/NFR traceability and review-gate verdicts (which the legacy process lacked).
