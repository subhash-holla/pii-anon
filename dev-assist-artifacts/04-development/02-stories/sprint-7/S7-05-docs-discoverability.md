# S7-05 — docs discoverability [DOCS MUST]: surface the program's API families + teeth-proven docs gate

| Field | Value |
|---|---|
| Story | S7-05 |
| Sprint | 7 |
| State | **TODO** (authored 2026-06-09) |
| provisional_status | REAL (docs + a teeth test; no behavior change). |
| Size | S |
| Implements | The **[DOCS MUST]** dev-log W2 commitment + the three open **D6 SME Docs MAJORs** (Documentation stage): (1) surface the distinct anonymization-vs-pseudonymization APIs (the **FR-010** headline — distinct families, no-merge invariant); (2) update `docs/evaluate-your-pipeline.md` (S6-04 landed the BYO/identical-path sections — extend with canonical-run + supremacy CLI); (3) fix the divergent recall-floor docs (align to the SharedLayerProjector/FloorProjectingFusion reality). Plus the SO-19 `rc_close` positioning deliverable: vanilla-vs-swarm. |
| Traces | D6 SME REQUEST_CHANGES (Docs MAJORs); D2/D3 "add CLI surface for BYO-pipeline scoring + canonical-run + the distinct anon/pseudo families (docs discoverability — SME MAJOR)"; FR-010 (distinct families — the API being surfaced). No numbered FR exists for docs discoverability itself (verified — the dev-log [DOCS MUST] tag + SME MAJORs are the authority chain). |
| Files owned | **additive/update** `docs/README.md` (index), `docs/recall-floor.md` (fix), `docs/evaluate-your-pipeline.md` (extend), **new** `docs/anonymization-vs-pseudonymization.md` (incl. vanilla-vs-swarm positioning), **update** `docs/api-reference.md` (program surfaces), **additive** `src/pii_anon/cli.py` help/epilog text ONLY (no behavior change), `tests/test_docs_discoverability.py` (**new**). |
| Depends on | S6-04 + S7-01 + S7-03 (it documents their surfaces) — must land LAST of the feature stories, BEFORE Stage-6 Documentation (D1 harvests `docs/`). |
| **NEVER touches** | `README.md`, `docs/pii-rate-elo-value.md`, `docs/benchmark-summary.md` (user-WIP), `artifacts/benchmarks/*`. |

## 1. Intent
The program shipped major API families that `docs/` does not surface: the rating ladder (3 tiers), the attacks seam (reid/MIA), the query-aware masking gate, 4-channel interception + leakage-Sankey, the encrypted token store, the SDO `supremacy`/`canonical-run` CLI, S6-04's BYO SDK + identical path, S7-01's native readers + extras matrix, S7-03's fairness gate. Three D6 SME Docs MAJORs are open, headlined by FR-010: the **distinct anonymization-vs-pseudonymization scorer families** (with the no-merge invariant) are invisible in docs. S7-05 closes the discoverability gap and pins it with a **teeth-proven docs gate** (`tests/test_docs_discoverability.py`): headline public symbols must appear in the api-reference/index, intra-docs links must resolve, and the index must cover every non-generated doc file — so future surfaces cannot silently ship undocumented.

## 3. Given / When / Then (acceptance)
- **A1 — index completeness `[UNIT-TEST]`.** `docs/README.md` links every non-generated `docs/*.md` (exact file-set comparison; generated/user-WIP excluded by an explicit allowlist: `benchmark-summary.md`, `pii-rate-elo-value.md`).
- **A2 — links resolve `[UNIT-TEST]`.** Every intra-`docs/` relative markdown link in every doc resolves to an existing file (zero broken).
- **A3 — anon-vs-pseudo doc (FR-010) `[UNIT-TEST]`.** `docs/anonymization-vs-pseudonymization.md` names BOTH distinct scorer families (`deid_families` anonymization + pseudonymization surfaces) and states the no-merge invariant verbatim; includes the vanilla-vs-swarm positioning section.
- **A4 — recall-floor doc fixed `[UNIT-TEST]`.** `docs/recall-floor.md` references the LIVE mechanism (`SharedLayerProjector` / `FloorProjectingFusion` / `routing/shared_layer.py`) and no longer the stale one.
- **A5 — evaluate-your-pipeline extended `[UNIT-TEST]`.** Names the `pii_anon.byo_pipelines` entry-point group AND the `canonical-run` + `supremacy` CLI commands.
- **A6 — headline symbols discoverable `[UNIT-TEST]`.** A curated list of headline public symbols (incl. `QueryAwareMaskingGate`, `BYOPipelineRegistry`, `evaluate_incumbent`, `NativeReaderRegistry`, `evaluate_language_fairness`, the token-store + interception + attacks entry symbols, `CompetitiveSupremacyGate` CLI surface) each appears in `docs/api-reference.md` OR the index.
- **A7 — CLI help mentions the surfaces `[UNIT-TEST]`.** `pii-anon --help` (run via the module) mentions the eval/supremacy family; the additive epilog names readers + BYO scoring (no behavior change — help text only).
- **A8 — user-WIP untouched `[AUDIT]`.** `README.md`, `docs/pii-rate-elo-value.md`, `docs/benchmark-summary.md` byte-identical (md5 pinned at story close, not in pytest); off-limits md5s byte-identical.
- **A9 — docs-smoke green.** `make docs-smoke` passes (the quickstart notebook still executes).

## 5. Notes / non-goals
- **Non-goal:** README.md (user-WIP) — the index lives at `docs/README.md`.
- **Non-goal:** a docs site generator (mkdocs/sphinx) — plain markdown per house convention; the comprehensive site lives in the sibling `pii-anon-doc` repo.
- **Non-goal:** CLI behavior changes — epilog/help strings only.

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[AUDIT]`. **Reviewers:** code-quality + traceability + requirements-coverage + axiom-compliance + **documentation** (the per-story documentation specialist — this story IS the docs story; security-sast not triggered unless cli.py edits exceed help text). 5-reviewer set resolved at claim via the selector script.

## 12. Definition of Done
- [ ] **RED**: `tests/test_docs_discoverability.py` failing on the missing/stale docs. RED precedes GREEN.
- [ ] **GREEN**: docs written/fixed + CLI epilog — all anchors green.
- [ ] **Quality gate**: full xdist suite green; ruff clean (tests); `make docs-smoke` green.
- [ ] **Untouched**: user-WIP + off-limits byte-identical; narrow `git add`.
- [ ] **Story-gate APPROVE** — `_reviews/story/S7-05/`.

## Evidence (filled on completion)
