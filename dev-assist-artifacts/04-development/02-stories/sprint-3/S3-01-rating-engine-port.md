# S3-01 — RatingEnginePort + RatingEngineRegistry + glicko-legacy facade

| Field | Value |
|---|---|
| Epic | E3 Eval rating engine (DC-06) |
| State | **DONE** (gate APPROVE 2026-05-31; `_reviews/story/S3-01-gate.yaml`) |
| Implements | FR-003 (rating-engine abstraction), NFR-026 (graceful degradation / registry); foundation for NFR-001 ladder; S3-05 import-boundary guard |
| Traces | Design D-EVAL DECISION 2 (RatingEnginePort 3-tier ladder; rating import-isolated from detection) |
| Test-type tags | `[UNIT-TEST]` `[CONTRACT-TEST]` `[CI-GATE]` |
| Files owned | `src/pii_anon/eval_framework/rating/port.py` (new), `registry.py` (new), `__init__.py` (export), `pyproject.toml` (entry-point group); `tests/test_rating_engine_port.py` (new), `tests/test_rating_import_boundary.py` (new) |
| Depends on | none (S3-02 bradley-terry-mle / S3-03 bayes-bt / S3-04 coherent-significance follow; S3-02 blocks on eval-data S6 `stats/bradley_terry.py`) |
| Size | M |

## 1. Intent
Introduce a `RatingEnginePort` (structural `typing.Protocol`) + a `RatingEngineRegistry` (entry-point pattern mirroring `engines/registry.py`, group `pii_anon.rating_engines`), and register the existing `PIIRateEloEngine` as `glicko-legacy` — the verbatim fallback tier + instant-rollback facade. This is the eval-integrity FOUNDATION that unblocks the `bradley-terry-mle` and `bayes-bt` tiers (S3-02/03). Also lands the import-boundary CI guard (S3-05): `eval_framework.rating` imports NOTHING from `swarm`/`moe`/`fusion`/`policy`.

## 2. Given/When/Then (acceptance)
- **Given** the Protocol, **then** `PIIRateEloEngine` satisfies `RatingEnginePort` structurally with ZERO call-site changes (mypy --strict confirms; `isinstance` if `@runtime_checkable`).
- **Given** the registry, **when** `discover_entrypoint_engines("pii_anon.rating_engines")` runs, **then** `glicko-legacy → PIIRateEloEngine` is discoverable; absent extras degrade gracefully (NFR-026).
- **Given** any module under `eval_framework/rating/`, **then** it imports nothing from `{swarm, moe, fusion, policy}` (AST-based test, GREEN today — prior grep found zero violations).

## 3. Approach (VALIDATED by Plan agent)
- **Port = `typing.Protocol`** (structural; zero-risk) with the MINIMAL method set the 4 production callers use:
  `__init__() -> None`, `run_round_robin(composites: dict[str, float]) -> list[RatingUpdate]`, `get_rating(name: str) -> EloRating | None`.
  (Richer engine API — `run_reidentification_tournament`, `tournament_summary`, `evaluate_governance`, `update_from_match`, `get_leaderboard` — stays on the concrete class; do NOT over-constrain the port.) Add `@runtime_checkable`.
- **Registry** mirrors `engines/registry.py:12-95` exactly: thread-safe dict, `register`, `get`, `discover_entrypoint_engines(group="pii_anon.rating_engines")` with the `entry_points().select(group=...)` + `hasattr` fallback (registry.py:64-67).
- **pyproject**: add a NEW table `[project.entry-points."pii_anon.rating_engines"]` → `glicko-legacy = "pii_anon.eval_framework.rating.elo:PIIRateEloEngine"`. Distinct group → CANNOT affect `pii_anon.engines` discovery.
- **Import-boundary test**: AST walk over `eval_framework/rating/*.py`; assert no `ImportFrom`/`Import` head ∈ `{pii_anon.swarm, pii_anon.moe, pii_anon.fusion, pii_anon.policy}`. AST (not substring) so it won't false-positive on docstrings/identifiers.
- Preserve the ~7 callers (leaderboard.py:133, competitor_composite.py:91/124, competitor_compare.py:2545, + `__init__` re-exports) + `SystemScorecard.elo_rating/elo_rd` + the checked-in baseline leaderboard verbatim.

## 4. Notes / scope
- This story does NOT implement MLE or Bayesian engines — only the port, registry, facade registration, and the import boundary. The ladder tiers are S3-02/03/04.
- `EloRating` / `RatingUpdate` types are imported from `eval_framework.rating.elo` for the Protocol signatures.

## 12. Definition of Done
- [x] RED: contract test (`PIIRateEloEngine` is-a `RatingEnginePort`) + registry discovery test + import-boundary test — failing (port/registry absent)
- [x] Protocol + registry + entry point added; `PIIRateEloEngine` unchanged & still satisfies the port
- [x] All ~7 callers untouched & green; ruff + mypy --strict clean; full suite green
- [ ] Story-gate review APPROVE (`_reviews/story/S3-01-gate.yaml`)

## Evidence (agent-simulated execution)
- **RED** commit `e5a554e` — `tests/test_rating_engine_port.py` (7 tests) + `tests/test_rating_import_boundary.py` (2 tests). Contract test failed on import (`RatingEnginePort` absent); import-boundary test GREEN on introduction (AST walk, zero violations).
- **GREEN** commit `d5cf633` — `port.py` (`@runtime_checkable RatingEnginePort` Protocol), `registry.py` (`RatingEngineRegistry`, group `pii_anon.rating_engines`), additive `__init__.py` export, `pyproject.toml` entry-point table. `elo.py` UNCHANGED.
- **REFACTOR**: none required (ruff + mypy --strict clean at GREEN; minimal mirror of `engines/registry.py`).
- **Port surface** (minimal): `run_round_robin(self, composites: dict[str, float]) -> list[RatingUpdate]`, `get_rating(self, name: str) -> EloRating | None`.
- **Entry-point discovery** (after `pip install -e . --no-deps`): `discover_entrypoint_engines("pii_anon.rating_engines")` → `['glicko-legacy']`; resolves to `PIIRateEloEngine`.
- **Quality gates**: ruff `All checks passed`; mypy `--strict` `Success: no issues found in 115 source files`; coverage 86.19% (≥84%). Targeted 9/9 green; rating-consumer suite (rating/elo/leaderboard/competitor/scorecard) 318 green.
- **Untouched & green**: `elo.py` + all 4 production callers (`leaderboard.py:133`, `competitor_composite.py:91/124`, `competitor_compare.py`) + re-exports.
- *Note: executed under agent-simulated CI (local `.venv`); Pass-2 real-CI confirmation to be scheduled by orchestrator.*
