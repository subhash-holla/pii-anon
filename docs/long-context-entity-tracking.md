# Long-Context Entity Tracking

## Why this exists

Large files and streams frequently mention the same real person through multiple variants:

- full name (`Jack Davis`)
- short alias (`Jack`)
- contact mention (`jackdavis@example.com`)

`pii-anon` v1.0.0 adds continuity linkage so these variants can be replaced consistently within a scope.

## Scope model

Tracking is document/session scoped. Use the same `scope` value when you want continuity across chunks/records.

## Controls

In `ProcessingProfileSpec`:

- `transform_mode="pseudonymize"` for deterministic reversible tokens.
- `transform_mode="anonymize"` for typed placeholders.
- `entity_tracking_enabled=True` to enable alias linking.

In config (`CoreConfig.tracking`):

- `enabled`
- `min_link_score`
- `allow_email_name_link`
- `require_unique_short_name`

## Heuristics

The linker uses deterministic, dependency-free rules:

- normalized surface forms
- short-name/full-name linkage
- email-local-part to full-name linkage
- ambiguity guard for short names

## Benchmarking continuity

Use:

```bash
PYTHONPATH=src python scripts/run_continuity_benchmark.py --enforce
```

Blocking thresholds:

- alias-link F1 >= 0.95
- pseudonym consistency >= 0.99
- anonymize placeholder consistency >= 0.99

## Reading link audit output

`run(...)` includes `link_audit` entries with:

- mention span and text
- cluster id
- linking rule and score
- final replacement
