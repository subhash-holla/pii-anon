# API Reference

## Primary APIs

- `PIIOrchestrator.run(...)`
- `PIIOrchestrator.run_async(...)`
- `PIIOrchestrator.detect_only(...)`
- `PIIOrchestrator.detect_only_async(...)`
- `PIIOrchestrator.run_stream(...)`
- `PIIOrchestrator.capabilities()`
- `PIIOrchestrator.discover_engines()`

## Profile fields (v1.0.0)

`ProcessingProfileSpec` adds:

- `transform_mode`: `"pseudonymize" | "anonymize"`
- `placeholder_template`: custom placeholder format for anonymize mode
- `entity_tracking_enabled`: enable/disable alias continuity linking
- `use_case` and `objective`: profile-aware policy routing hints

## Config schema additions (v1.0.0)

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

- `StrategyEvaluator.compare_strategies(...)`
- `evaluate_pipeline(...)`
- `run_benchmark(...)`
- `compare_competitors(...)`
- `summarize_dataset(...)`
- `scripts/run_continuity_benchmark.py`

## CLI integration

- `pii-anon evaluate-pipeline ...` for combined de-identification + evaluation
- `pii-anon eval-framework ...` for standalone eval-framework runs
- `pii-anon benchmark-preflight ...` for runtime and competitor readiness diagnostics
- `pii-anon benchmark-publish-suite ...` for end-to-end canonical publish-grade benchmark workflow
- `pii-anon benchmark-publish-suite ... --reuse-current-env --install-no-deps` for pre-provisioned/offline environments
- `pii-anon benchmark-publish-suite ... --no-strict-runtime --no-require-all-competitors --no-require-native-competitors --no-include-end-to-end --no-allow-core-native-engines --no-enforce-publish-claims --no-validate-readme-sync` for cross-platform non-canonical comprehensive local runs
- `pii-anon compare-competitors ... --dataset-source package-only --require-all-competitors --require-native-competitors` for strict canonical competitor runs
