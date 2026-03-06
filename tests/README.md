# pii-anon-test

Test suite for the **pii-anon** PII anonymization library.

## Running Tests

Ensure `pii-anon-code` is installed as an editable package, then run from this directory:

```bash
# Install the library first
cd ../pii-anon-code && pip install -e ".[dev]"

# Optionally install datasets
cd ../pii-anon-eval-data && pip install -e .

# Run tests
cd ../pii-anon-test && python -m pytest . -q --no-cov
```

## Test Structure

- **Root-level test files** — Unit and integration tests for all library modules
- **integration/** — End-to-end integration tests
- **performance/** — Performance and SLA tests (marked with `@pytest.mark.performance`)

## Related Repositories

- [pii-anon-code](https://github.com/subhash-holla/pii-anon) — The main library
- [pii-anon-doc](https://github.com/subhash-holla/pii-anon-doc) — Documentation
- [pii-anon-eval-data](https://github.com/subhash-holla/pii-anon-eval-data) — Datasets

## License

Apache-2.0
