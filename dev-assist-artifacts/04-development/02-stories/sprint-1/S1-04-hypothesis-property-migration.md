# S1-04 — Migrate the recall-floor property test to hypothesis @given

| Field | Value |
|---|---|
| Epic | E1 Recall-floor foundation (DC-01) |
| State | **IN_PROGRESS** (claimer: dev-assist-development-executor; claimed_at 2026-05-30; started 2026-05-30) |
| Implements | NFR-011 (property infra), test-quality |
| Traces | Design D-SWARM verification (property test ZERO violations); W4 testing-setup (add hypothesis to dev extra) |
| Test-type tags | `[PROPERTY-TEST]` |
| Files owned | `tests/test_shared_layer_projector.py`, `pyproject.toml` (dev extra) |
| Depends on | none |
| Size | S |

## 1. Intent
Replace the seeded-random superset-invariant test (`tests/test_shared_layer_projector.py:91-112`, `random.Random(42)`, 2000 cases) with a `hypothesis @given` property test that generates `(output, shared)` span sets and asserts `shared_keys ⊆ out_keys` (ZERO violations) + no duplicate re-injections. Add `hypothesis` to the dev extra. Fix the stale docstring (line 9 says "S6-PROP"; correct to S1-04).

## 2. Given/When/Then (acceptance)
- **Given** hypothesis-generated `(output, shared)` span sets across types/fields/langs/offsets, **then** the superset invariant holds for every example (no falsifying case).
- **Given** the migration, **then** `hypothesis>=6.0` is in the `dev` extra; the 6 example-based unit tests + the determinism test are unchanged and green.

## 3. Approach
- Add `"hypothesis>=6.0"` to `pyproject.toml` `dev` extra (after `pytest-asyncio`, ~line 73).
- `@given` with `st.lists(...)` of a `@composite` span strategy: `entity_type ∈ st.sampled_from([...])`, `field_path ∈ st.sampled_from(["text","notes",None])`, `language ∈ st.sampled_from(["en","es","zh"])`, `start ∈ st.integers(0,40)`, `length ∈ st.integers(1,12)`. Replace the `for _ in range(2000)` loop.
- Use `@settings(max_examples=...)` (e.g. 400) and/or `derandomize=True` for deterministic CI.
- Keep the 6 example-based unit tests + the determinism test as-is.

## 4. RED → GREEN
- **RED**: write the `@given` test; collection fails because `hypothesis` is not installed.
- **GREEN**: add `hypothesis>=6.0` to the dev extra + `pip install -e .[dev]`; test passes with zero falsifying examples.

## 12. Definition of Done
- [ ] `@given` property replaces the seeded loop; ZERO falsifying examples; stale docstring fixed (S6-PROP → S1-04)
- [ ] `hypothesis>=6.0` in the `dev` extra
- [ ] ruff + mypy --strict clean; full suite green
- [ ] Story-gate review APPROVE (`_reviews/story/S1-04-gate.yaml`)
