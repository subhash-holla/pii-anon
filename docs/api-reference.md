# API Reference

## Primary APIs

- `PIIOrchestrator.run(...)`
- `PIIOrchestrator.run_async(...)`
- `PIIOrchestrator.detect_only(...)`
- `PIIOrchestrator.detect_only_async(...)`
- `PIIOrchestrator.run_stream(...)`
- `PIIOrchestrator.capabilities()`
- `PIIOrchestrator.discover_engines()`

## Profile fields

`ProcessingProfileSpec` adds:

- `transform_mode`: `"pseudonymize" | "anonymize"`
- `placeholder_template`: custom placeholder format for anonymize mode
- `entity_tracking_enabled`: enable/disable alias continuity linking
- `use_case` and `objective`: profile-aware policy routing hints

## Config schema additions

`CoreConfig` adds:

- `transform.default_mode`
- `transform.placeholder_template`
- `tracking.enabled`
- `tracking.min_link_score`
- `tracking.allow_email_name_link`
- `tracking.require_unique_short_name`

## Output additions

`run(...)` responses now include `link_audit` entries describing continuity linkage and replacement decisions.

## Evaluation and benchmarks

Internal evaluation (for pii-anon itself):
- `StrategyEvaluator.compare_strategies(...)`
- `evaluate_pipeline(...)`
- `run_benchmark(...)`
- `compare_competitors(...)`
- `summarize_dataset(...)`
- `scripts/run_continuity_benchmark.py`

## pii-rate-elo evaluation framework

All symbols below are imported from `pii_anon.eval_framework`.

**Composite metric** (see [pii-rate-elo.md](pii-rate-elo.md)):
- `compute_composite(f1, precision, recall, latency_ms, docs_per_hour, ..., config)` — primary scorer
- `compute_composite_from_benchmark_result(result, config)` — duck-typed convenience wrapper
- `CompositeConfig` — dataclass; factories `CompositeConfig.f2_privacy_first()` and `CompositeConfig.for_deployment(profile)`
- `DeploymentProfile` — `Literal["standard", "high_security", "high_throughput"]`
- `FloorGateConfig` — catastrophic-weakness guards (`min_f1`, `min_privacy`, `min_fairness`, `min_entity_coverage`, `cap_score`)
- `fbeta_score`, `normalize_reidentification_resistance`, `normalize_quasi_identifier_coverage`, `normalize_behavioral_signal_leakage`

**Elo engine** (Glicko-style round-robin):
- `PIIRateEloEngine.run_round_robin(composites)` — runs the tournament
- `PIIRateEloEngine.tournament_summary()` — rankings, pairwise significance, minimum distinguishable difference
- `PIIRateEloEngine.evaluate_governance(thresholds)` — gating against `GovernanceThresholds(min_elo, max_rd, min_matches)`
- `PIIRateEloEngine.run_reidentification_tournament(rrs_scores)` — Tier 3 adversarial round-robin

**Scorecards and leaderboards**:
- `SystemScorecard`, `BenchmarkScorecard` — wire format for per-system results
- `Leaderboard.from_benchmark_scorecard(bench, sort_by="elo"|"composite"|"f1")` — sorted view with markdown/json/csv rendering
- `LeaderboardExporter.export(leaderboard, out_dir, formats=["json","md","csv"])`

**External-system evaluation** (score *your own* pipeline — see [evaluate-your-pipeline.md](evaluate-your-pipeline.md)):
- `evaluate_external_system(predictor, *, system_name, dataset, language, max_records, warmup_records, deployment_profile, composite_config, on_error)` — returns `ExternalEvaluationResult`
- `load_baseline_leaderboard(artifact_path=None)` — returns `BaselineLeaderboard` loaded from `artifacts/benchmarks/benchmark-results.json`
- `BaselineLeaderboard.with_scorecard(scorecard, replace=False)` — splice user system in, re-run the tournament, return a `Leaderboard`
- `resolve_predictor_path("module.submod:callable")` — helper used by the `rate-elo` CLI
- `Predictor` — type alias for `Callable[[str], Iterable[tuple[str, int, int]]]`

**Dataset loaders**:
- `load_eval_dataset(name, language, data_type, difficulty, adversarial_only, dimension)` — returns `list[EvalBenchmarkRecord]`
- `EvalBenchmarkRecord` — includes Tier 3 fields (v1.3.0+): `behavioral_signal_density`, `re_identification_resistance_score`, `tier3_risk_level`, `is_paired_profile`, `esrc_attack_target`, `context_preservation`
- `summarize_eval_dataset(name)` — distribution summary with Tier 3 coverage stats
- `resolve_eval_dataset_path(name, source)` — path probe

## CLI integration

**Detection**:
- `pii-anon detect`, `pii-anon detect-stream`, `pii-anon process-file`, `pii-anon tokenize`
- `pii-anon health`, `pii-anon capabilities`, `pii-anon version`

**Evaluation**:
- `pii-anon evaluate-pipeline ...` for combined de-identification + evaluation
- `pii-anon eval-framework ...` for standalone eval-framework runs
- `pii-anon rate-elo --predictor module:callable ...` for scoring an **external** PII pipeline against the benchmark (see [evaluate-your-pipeline.md](evaluate-your-pipeline.md))

**Benchmarks**:
- `pii-anon benchmark-preflight ...` for runtime and competitor readiness diagnostics
- `pii-anon benchmark-publish-suite ...` for end-to-end canonical publish-grade benchmark workflow
- `pii-anon benchmark-publish-suite ... --reuse-current-env --install-no-deps` for pre-provisioned/offline environments
- `pii-anon compare-competitors ... --dataset-source package-only --require-all-competitors --require-native-competitors` for strict canonical competitor runs
