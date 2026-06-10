# S7-05 — docs discoverability [DOCS MUST]: surface the program's API families + teeth-proven docs gate

| Field | Value |
|---|---|
| Story | S7-05 |
| Sprint | 7 |
| State | **DONE** (2026-06-09; SO-23. Story gate iter-1 **REQUEST_CHANGES** (1 MAJOR: the certify-a-run example referenced `benchmark-results.json` — a file the canonical-run producer never writes; not copy-paste-runnable) → remediated `5f825b6` (the real `canonical-run.json` path + A5 path-teeth + the FloorProjectingFusion mention) → iter-2 **5/5 APPROVE** (0 MAJOR/MINOR; the teeth counterfactually proven to bite). See §Evidence.) |
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

**Commits:** RED `e573243` (A1–A7; 5/7 failing honestly; user-WIP docs excluded from the index requirement AND the link sweep) → GREEN `8c9cec3` (the FR-010 doc + index + api-reference program-surfaces + certify-a-run section + CLI epilog + the PRE-EXISTING `docs-smoke` Makefile fix — the target referenced a notebook that never existed; Makefile recorded as an implicit owned file, a necessary precondition for A9) → remediation `5f825b6`.

**Story gate (iter-1 REQUEST_CHANGES → iter-2 5/5 APPROVE; `_reviews/story/S7-05/`; runs `wf_941b38c6-52f` → `wf_d8d3ca4e-c16`):** iter-1 MAJOR (requirements-coverage): the certify-a-run step-2 example read `./certified/benchmark-results.json` — the producer writes ONLY `<output-dir>/canonical-run.json` (`canonical_run.py:1275`, `cli.py:908`); verbatim execution raised BadParameter; the A5 substring teeth couldn't catch it. Remediated: the real path + a comment naming the emitted file + A5 tightened (any `supremacy --artifact` example in a certified dir must end `canonical-run.json` — iter-2 confirmed the counterfactual old path FAILS the assertion). security-sast verified every security-adjacent doc claim accurate against source (token store AEAD semantics, supremacy exit semantics `cli.py:865-866`, the bounded-FlateDecode mention, fail-closed gates) — docs do not overstate code. The iter-1 orchestrator-md5 OBS resolved as a measurement-frame artifact (working-tree user-WIP `0afc6dee…` vs committed blob `4a837c52…` — BOTH unchanged; stories touched neither). FloorProjectingFusion mention added to recall-floor.md (verified vs `routing/floor_fusion.py:59`).

**Quality:** owned tests 7/7 (incl. the tightened A5); `make docs-smoke` EXIT=0 (the notebook now actually executes); `make cli-smoke` EXIT=0; full xdist suite EXIT=0 @ 88.87% on the GREEN state (the remediation is docs+test-assertion-only); ruff clean; mypy clean BOTH modes (144 files); user-WIP docs byte-identical (`pii-rate-elo-value.md` `89cc6d03…`, `benchmark-summary.md` `e575a730…`, root README untouched).

**DoD:** all checkboxes met. The three D6 SME Docs MAJORs are CLOSED (anon/pseudo APIs surfaced + evaluate-your-pipeline current incl. the SDO CLI + recall-floor doc verified-live with the FloorProjectingFusion completion).
