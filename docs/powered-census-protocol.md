# Powered Census Protocol (SP1)

How the canonical SDO census moved from the 8-record smoke draw to a
statistically powered N=10,000 draw — the sizing math, the runtime budget,
the exact reproduction commands, and the provenance invariants.

## Why N=10,000

The SDO gate G6 tests raw-detection F2 non-inferiority with margin
ε_F = 0.01. The v1.5.0rc1 certified artifact used the CLI default
`--max-samples 8` ("kept tight so the run stays fast") — a draw too small to
represent measured behaviour: the 8-record census read core precision 0.59
while the 149K-record benchmark measured 0.83 on the same code.

Minimum-detectable-effect scaling from the in-tree benchmark's measured MDE
(0.0015 F1 at 148,994 records, α = 0.05, power = 0.80), MDE(N) ≈
0.0015 · √(148,994 / N):

- N for MDE ≤ ε_F = 0.01:  N ≥ 148,994 · (0.0015 / 0.01)² ≈ **3,353**
- **N = 10,000 → MDE ≈ 0.0058** — comfortably below ε_F, with headroom for
  draw variance.

The DATA v2 corpus (`pii_anon_datasets` 2.0.0, `pii_anon.jsonl.gz`) carries
575,604 records / 2,486,438 annotation spans (4.32 spans/record, 63 entity
types, 60 languages), so N = 10,000 scores ≈ 43,200 spans. Truth-span counts
for every SP1-touched entity type at the frozen draw (N=10,000, seed 8314),
all ≥ 30: PERSON_NAME 18,992 · EMAIL_ADDRESS 9,757 · PHONE_NUMBER 4,471 ·
ORGANIZATION 1,801 · US_SSN 1,474 · NATIONAL_ID 1,071 · BANK_ACCOUNT 1,062 ·
CREDIT_CARD 551 · DRIVERS_LICENSE 351 · LICENSE_PLATE 307 · NPI_NUMBER 265 ·
BAR_NUMBER 208 · DEA_NUMBER 156. See
`artifacts/sp1/determinism-check.txt` for the qualification record.

## Runtime budget

Measured per-system p50 (2026-06-10 benchmark artifacts): gliner ~83 ms,
pii-anon-swarm ~95 ms, presidio ~15 ms, scrubadub ~0.24 ms, pii-anon
~0.45 ms. At N = 10,000 with the canonical producer's fixed parameters
(`warmup_samples=2`, `measured_runs=1`): ≈ 14 + 16 + 2.5 + ~0 + ~0 minutes
≈ **35 min** plus corpus load. Budget gate: 1.5 h wall-clock.

## The certified command

```bash
PYTHONPATH=src .venv/bin/python -m pii_anon.cli canonical-run \
    --seed 20240601 --max-samples 10000 \
    --output-dir artifacts/canonical
```

Seed 20240601 is the CLI default (NFR-005 determinism); it is passed
explicitly so this protocol is self-contained. The producer writes ONLY to
`artifacts/canonical/` and routes the artifact through the fail-closed
`CanonicalRunGate` (`canonical_claim_run=True` only when every required
field is present-and-valid). The verdict is read with:

```bash
PYTHONPATH=src .venv/bin/python -m pii_anon.cli supremacy \
    --artifact artifacts/canonical/canonical-run.json
```

## Determinism verification

Two same-seed measurement runs at N = 10,000 must be byte-identical modulo
wall-clock latency fields (NFR-005 at scale). Verified for SP1 on
2026-06-11 — see `artifacts/sp1/determinism-check.txt` (`DETERMINISTIC:
True`, plus the per-entity-mix qualification table).

## Provenance invariants

- Producer: `evaluation/canonical_run.py` md5
  `d8f0f80e113c3b5d59c06d0b5fd36fac` (UNCHANGED throughout SP1)
- Gate: `eval_framework/evaluation/competitive_supremacy.py` md5
  `3b842e81c3f03eafd11f9c655c1789a0` (UNCHANGED)
- Comparator: `evaluation/competitor_compare.py` md5
  `7cae16c89f4c97136e1a12394dae2025` (UNCHANGED)
- Dataset: `pii_anon_datasets` 2.0.0, `pii_anon.jsonl.gz`, scope
  `data-v2.0.0`
- **No control-path code was modified for the powered census** — the run
  uses the public `--max-samples`/`--seed` CLI surface only, so no
  adversarial SDO close was mandated. Any future edit to the three pinned
  files above re-triggers the standing close rule.

## Relationship to the SP1 measurement harness

`scripts/sp1_detection_delta.py` is the fast core-only inner loop used for
per-change deltas during SP1 (overlap-match, canonical truth labels,
documented divergences from the census matcher: any-overlap vs IoU ≥ 0.5,
non-exclusive vs 1:1 matching, ignore-bucket exclusion). Its numbers are
intentionally NOT the gate's numbers — certification always comes from the
canonical producer + supremacy gate via the commands above.
