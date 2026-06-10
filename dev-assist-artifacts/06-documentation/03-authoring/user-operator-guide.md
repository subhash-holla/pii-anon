# pii-anon v1.4.0 — User and Operator Guide

> **Audience:** external integrators, operators, and users who run `pii-anon` in a
> product or pipeline. Plugin authors and contributors: see `contributor-handbook.md`.
> Internal design rationale: see `architecture-and-adr.md`.
>
> **Current SDO verdict (honest):** `NOT_YET` — G1/G2/G3/G4/G5/G7 PASS; G6 FAIL
> (F2 0.7214 vs 0.75 threshold). This is a methodology gap, not a regression. Published
> benchmark numbers are PROVISIONAL pending a canonical regen run. Do not cite the smoke
> artifact numbers as certified. See §5 (Certifying a run) and the `changelog.md` for full
> context.

---

## Contents

1. [Install and extras matrix](#1-install-and-extras-matrix)
2. [Core workflows: detect, process, tokenize](#2-core-workflows-detect-process-tokenize)
3. [Anonymization vs pseudonymization: choosing transform_mode](#3-anonymization-vs-pseudonymization-choosing-transform_mode)
4. [Evaluating your own pipeline (BYO + rate-elo)](#4-evaluating-your-own-pipeline)
5. [Certifying a run (canonical-run + supremacy, honest NOT_YET semantics)](#5-certifying-a-run)
6. [Ingesting native formats (PDF, image, DICOM, audio)](#6-ingesting-native-formats)
7. [Multilingual fairness gate](#7-multilingual-fairness-gate)
8. [Operational notes](#8-operational-notes)
9. [CLI command census](#9-cli-command-census)

---

## 1. Install and extras matrix

pii-anon ships on PyPI as `pii-anon`. The core package has a single hard dependency
(`pydantic>=2.8,<3`). Every other capability is gated behind an optional extra.

### Minimal installs

```bash
# Core library only (no CLI, no optional engines)
pip install pii-anon

# Core + CLI (required for all pii-anon ... commands)
pip install "pii-anon[cli]"

# Core + CLI + AEAD-encrypted token store (recommended for pseudonymization in production)
pip install "pii-anon[cli,crypto]"

# Core + CLI + eval dataset (required for rate-elo / evaluate-your-pipeline)
pip install "pii-anon[cli,datasets]"
```

### Extras matrix

| Extra | What it enables | Absent behavior |
|---|---|---|
| `cli` | `pii-anon` command (`typer`, `rich`, `pyyaml`) | No CLI entrypoint; Python API still works |
| `crypto` | AEAD-encrypted `EncryptedSQLiteTokenStore` (FR-019) | Pseudonymization falls back to in-memory store; no encrypted-at-rest guarantee |
| `gate-signing` | Ed25519 envelope signing for `gate_v1.json` (S2-05); default HMAC-SHA256 is stdlib-only | Optional; HMAC path requires no extra |
| `engines` | Presidio, Scrubadub, spaCy, stanza, GLiNER NER adapters | Only the always-on `regex-oss` shared layer runs |
| `llm-guard` | LLM Guard engine adapter | Unavailable for comparison runs; benchmark marks it absent |
| `gliner` | GLiNER standalone (without full `engines` set) | GLiNER unavailable |
| `datasets` | `pii-anon-datasets` v1.3.0 (159,891 records, 63 entity types, 60 languages) | `evaluate_external_system`, `rate-elo`, and eval commands raise `FileNotFoundError` on eval dataset |
| `ocr` | `pytesseract` + `Pillow` for image/screenshot OCR text extraction | `ImageOcrReader` / `ScreenshotOcrReader` raise `MissingOptionalDependencyError` on `read()` — LOUD, never silent |
| `dicom` | `pydicom` for DICOM header/pixel text extraction | `DicomReader` raises `MissingOptionalDependencyError` on `read()` — LOUD, never silent |
| `swarm-ml` | XGBoost + scikit-learn for the learned MoE router (runtime) | MoE router falls back to rule-based routing |
| `swarm-train` | Extended ML training stack (adds `datasets`, `tqdm`) | Training commands unavailable |
| `bayes-eval` | numpyro + JAX + arviz for the Bayesian Bradley-Terry NUTS sampler | `BayesBTEngine` import-safe; only sampling fails |
| `benchmark` | Full competitor stack for canonical publish-grade comparison runs | Incomplete; competitors marked unavailable |

> **NFR-026 — no silent recall loss.** If an OCR or DICOM backend is absent, the
> reader RAISES `MissingOptionalDependencyError` naming the install extra. It never
> yields empty text and silently drops PII. Audio (`AudioReader`) also fails loud — no
> ASR backend is shipped yet; it is a deliberate Pass-2 wire-in.

For OS-specific venv setup and platform notes, see `docs/dependencies-and-platforms.md`.
Python 3.10–3.13 is supported; 3.14 is experimental.

---

## 2. Core workflows: detect, process, tokenize

### 2a. Single-record detection (Python API)

```python
from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

orch = PIIOrchestrator(token_key="change-me-in-production")
result = orch.run(
    {"text": "Primary owner Jack Davis can be reached at jackdavis@example.com"},
    profile=ProcessingProfileSpec(
        profile_id="quickstart",
        mode="weighted_consensus",
        language="en",
        transform_mode="pseudonymize",   # or "anonymize" — see §3
        entity_tracking_enabled=True,
    ),
    segmentation=SegmentationPlan(enabled=False),
    scope="quickstart",
    token_version=1,
)
print(result["transformed_payload"]["text"])
print(result["link_audit"])
```

The `token_key` is the HMAC secret that controls reversibility under pseudonymization.
In production always inject this from an env var or secret manager; do not use the
string `"change-me-in-production"` or `"dev-key"`.

### 2b. Stream processing (Python API)

```python
payloads = [{"text": "alice@example.com"}, {"text": "123-45-6789"}]
for item in orch.run_stream(
    payloads,
    profile=ProcessingProfileSpec(
        profile_id="stream",
        mode="intersection_consensus",
    ),
    segmentation=SegmentationPlan(enabled=False),
    scope="stream",
    token_version=1,
):
    print(item["confidence_envelope"])
```

`run_stream` returns an iterator, so output is available as soon as each record is
processed. Use `run_stream_async` for the async variant.

### 2c. CLI single-record detect

```bash
pii-anon detect "Primary owner Jack Davis" --transform-mode pseudonymize --output json
pii-anon detect "Primary owner Jack Davis" --transform-mode anonymize --output json
```

The `detect` command accepts `--mode` (fusion strategy), `--language`, and
`--config` (YAML/JSON config path). See `§9` for the full flag list.

### 2d. CLI stream (file of records)

```bash
# detect-stream reads one text payload per line from the file
pii-anon detect-stream my_texts.txt --transform-mode pseudonymize --output json
```

### 2e. File processing

```bash
# process-file handles CSV, JSON, JSONL, or TXT
pii-anon process-file records.csv --transform-mode anonymize --output-file redacted.csv
```

### 2f. Tokenize and detokenize (debug / key-management)

```bash
pii-anon tokenize "Jack Davis" --entity-type PERSON_NAME --token-key my-key --output json
```

The `tokenize` command is a convenience tool for key-management testing. It writes the
token and immediately detokenizes it, confirming round-trip under the supplied key.

### 2g. Fusion modes

Choose the mode that matches your recall/precision trade-off in `ProcessingProfileSpec.mode`
or `--mode`:

| Mode | When to use |
|---|---|
| `union_high_recall` | Maximum recall; accepts false positives |
| `weighted_consensus` | Balanced default (weighted merge) |
| `calibrated_majority` | Require N-engine agreement |
| `intersection_consensus` | Strict: only confirmed-by-all |
| `mixture_of_experts` | MoE routing with per-entity-type expert selection (requires `swarm-ml` for learned weights) |

### 2h. Capabilities discovery

```bash
pii-anon capabilities --output json
pii-anon health --output json
```

`capabilities` reports the available engines, fusion strategies, and reader
capabilities. `health` performs a lightweight engine health check. Both are read-only
and safe to run in any environment.

---

## 3. Anonymization vs pseudonymization: choosing transform_mode

**The two transform modes are structurally distinct and are NEVER merged** into a
single de-id score (the project's no-merge invariant; enforced by the `test_deid_families.py`
regression gate — FR-010, DC-08).

| | `anonymize` | `pseudonymize` |
|---|---|---|
| Reversibility | Irreversible by design | Reversible under a controlled key |
| Output | Structured placeholder, e.g. `<PERSON_NAME:anon_0>` | Stable deterministic token, same plaintext → same surrogate in-scope |
| Scorer | `AnonymizationScorer` (`irreversibility_score`) | `PseudonymizationIntegrityScorer` (`integrity_score`) |
| Key axes | Re-identification risk, leakage, canary exposure | `unauthorized_reversal_rate` (must be 0), referential integrity, collision rate, key-state separation |
| Use when | GDPR "anonymization" claim; data publication; research datasets | Audit trails, de-identified operational data where authorized re-join is needed |

Switch via `ProcessingProfileSpec(transform_mode="anonymize")` or CLI `--transform-mode anonymize`.

The configuration also supports `redact`, `generalize`, `synthetic`, and `perturb` as
`transform.default_mode` values (see `docs/configuration.md`). The `anonymize` and
`pseudonymize` modes are the two values accepted by the `detect` and `detect-stream`
commands; the richer set is available in the programmatic API.

For full scorer API and the no-merge invariant, see `docs/anonymization-vs-pseudonymization.md`.

---

## 4. Evaluating your own pipeline

### 4a. 60-second version

```python
from pii_anon.eval_framework import evaluate_external_system, load_baseline_leaderboard

def my_detector(text: str):
    # return iterable of (entity_type, start, end) tuples
    return [("EMAIL_ADDRESS", 0, 17)]

result = evaluate_external_system(my_detector, system_name="my-detector")
leaderboard = load_baseline_leaderboard().with_scorecard(result.scorecard)
print(leaderboard.to_markdown())
```

Requires `pii-anon[datasets]`. The baseline rows come from
`artifacts/benchmarks/benchmark-results.json` (the committed artifact); no competitor
package needs to be installed.

### 4b. Predictor contract

Your detector is any callable:

```python
Predictor = Callable[[str], Iterable[tuple[str, int, int]]]
```

- Input: a Unicode string (one evaluation record).
- Output: iterable of `(entity_type, start, end)` tuples (0-indexed, half-open).
- Extra tuple elements are ignored. Malformed spans are dropped silently.
- Return real `int` offsets — `bool` is an `int` subclass in Python and `True`/`False`
  would coerce to `1`/`0`, producing bogus spans.

### 4c. CLI workflow

```bash
pii-anon rate-elo \
    --predictor my_package.detector:predict \
    --system-name "my-detector" \
    --language en \
    --max-records 2000 \
    --deployment-profile standard \
    --output markdown
```

Deployment profiles:

| Profile | Composite weighting |
|---|---|
| `standard` (default) | Detection 50%, ops 20%, re-ID 30% |
| `high_security` | Detection 30%, ops 10%, re-ID 60% |
| `high_throughput` | Detection 40%, ops 40%, re-ID 20% |

### 4d. Package your detector as an SDK plugin (entry-point discovery, FR-001)

A third-party package can advertise its predictor under the `pii_anon.byo_pipelines`
entry-point group:

```toml
# your package's pyproject.toml
[project.entry-points."pii_anon.byo_pipelines"]
my-detector = "my_pkg.detector:predict"
```

Then discover and score it without any pii-anon edits:

```python
from pii_anon.eval_framework import BYOPipelineRegistry

registry = BYOPipelineRegistry()
names = registry.discover_entrypoint_pipelines()   # ['my-detector', ...]
predictor = registry.get("my-detector")
```

Discovery never raises — a broken or absent package is skipped gracefully. Model weights
are not loaded until a predictor is called.

The five in-tree incumbents (`gliner`, `presidio`, `scrubadub`, `spacy_ner`, `stanza_ner`)
ship on this same group and ride the identical scoring path.

### 4e. Incumbents are scored on the identical path (FR-002)

`evaluate_incumbent` routes an in-tree incumbent through the **same** `evaluate_external_system`
function as your detector — there is no "house path":

```python
from pii_anon.eval_framework import (
    evaluate_incumbent,
    build_identical_path_leaderboard,
    incumbent_predictor,
    INCUMBENT_SYSTEMS,
)

# Apples-to-apples board: all rows produced by the same evaluator
board, _ = build_identical_path_leaderboard(
    {
        "presidio": incumbent_predictor("presidio"),
        "my-detector": my_detector,
    },
    max_records=2_000,
)
print(board.to_markdown())
```

Note: the committed baseline artifact (`artifacts/benchmarks/benchmark-results.json`)
was produced by the frozen legacy path (overlap matching over the benchmark dataset), not
the identical path. Do not mix rows from the two paths in one table;
`build_identical_path_leaderboard` is the unmixed alternative.

### 4f. CI gating example

```python
# tests/test_quality_gate.py
from pii_anon.eval_framework import evaluate_external_system
from my_package.detector import predict

COMPOSITE_FLOOR = 0.70
F1_FLOOR = 0.75

def test_pii_rate_elo_floor():
    result = evaluate_external_system(
        predict,
        system_name="my-detector",
        max_records=2_000,
        deployment_profile="high_security",
    )
    assert result.scorecard.composite_score >= COMPOSITE_FLOOR
    assert result.scorecard.f1 >= F1_FLOOR
```

### 4g. Tier 3 (LLM re-identification, DATA-track)

Tier 3 evaluation consumes an ESRC-style attack pipeline and injects re-identification
recall/precision into the composite score. The attack infrastructure
(`eval_framework/attacks/`) is CODE-local, but the corpus-scale representative adversary
(FR-011 Tier-3 LLM-adversary + FR-013 MIA LiRA@128) requires the `pii-anon-eval-data`
dataset (DATA-track dependency, external to this repo). Mark this as a DATA-track
dependency in any integration that depends on it.

Troubleshooting: see `docs/evaluate-your-pipeline.md:##Troubleshooting`.

---

## 5. Certifying a run

Leaderboard numbers from `rate-elo` are comparative. Claim-grade certification goes
through the SDO (state-of-the-art dominance objective) machinery: `canonical-run` then
`supremacy`.

### 5a. Producing the certified artifact

```bash
# Writes artifacts/canonical/canonical-run.json
pii-anon canonical-run --output-dir artifacts/canonical
```

The `CanonicalRunGate` is fail-closed: every guarantee field is validated; fabricated
or malformed values are rejected. `canonical_claim_run=True` is set only when every
required field is present and valid. The gate enforces the no-fabrication invariant
across all guarantee axes (G1–G7).

### 5b. Reading the SDO verdict

```bash
pii-anon supremacy --artifact artifacts/canonical/canonical-run.json
```

The output reports:

- `verdict`: `NOT_YET` | `PROVISIONAL_SOTA` | `CLAIM_GRADE_SOTA`
- Per-guarantee statuses: G1 (recall floor) / G2 (pseudonymization integrity) / G3
  (recall dominance) / G4 (calibration selective risk) / G5 (latency + audit) / G6
  (raw non-inferiority) / G7 (certified run provenance)
- `binding_constraint`: the single axis that determines the current verdict

### 5c. Honest current verdict

Running `supremacy` against the committed smoke artifact returns:

```
verdict: NOT_YET
binding_constraint: G6 (raw non-inferiority F2 0.7214 vs threshold 0.75)
G1: PASS  G2: PASS  G3: PASS  G4: PASS  G5: PASS  G6: FAIL  G7: PASS
```

G6 failing is a methodology gap, not a code regression: the evaluation methodology
(raw F2 on the current dataset draw) differs from the composite-metric framing of the
program's positioning. Old code produces byte-identical results at `use_case=default`.
This will not be resolved by weakening G6 — it is an honest limitation acknowledged
by the program. See `dev-assist-artifacts/05-testing/_diagnostics/f2-gap-attribution.md`
for the full attribution.

Do not treat NOT_YET as a build failure for general use. The recall floor (G1), the
pseudonymization integrity guarantee (G2), and the latency/audit guarantee (G5) all
PASS and are available for operational claims.

### 5d. CI gate: `--canonical-claim` flag

```bash
pii-anon supremacy --artifact artifacts/canonical/canonical-run.json --canonical-claim
# exits 1 unless verdict == CLAIM_GRADE_SOTA
```

`supremacy` is non-blocking by default (always exit 0). `--canonical-claim` makes it
block: use this only when a `CLAIM_GRADE_SOTA` verdict is required to proceed (e.g., a
release-publish gate). Do not use it in day-to-day CI until the G6 gap is resolved.

### 5e. Publish-grade benchmark suite

The full benchmark publish pipeline (canonical artifacts with all competitors) requires
Linux, shared-memory support, and all competitor packages installed natively:

```bash
pii-anon benchmark-preflight --output json
pii-anon benchmark-publish-suite --artifacts-dir artifacts/benchmarks --output json
```

`benchmark-preflight` exits 1 if the runtime is not ready. Run `benchmark-publish-suite`
on Linux in CI with all competitors pre-installed.

---

## 6. Ingesting native formats

The `pii_anon.readers` entry-point group provides native-format readers that all emit
the same `Iterator[IngestRecord]` contract, so the downstream pipeline is unchanged
(FR-031, DC-14).

### 6a. Reader capability matrix

| Reader | Format | Backend extra | `dependency_available` when absent | Behavior when absent |
|---|---|---|---|---|
| `PdfTextReader` | PDF | None (stdlib FlateDecode) | N/A | Always available; bounded zip-bomb protection |
| `ImageOcrReader` | JPEG, PNG, TIFF, etc. | `pii-anon[ocr]` (pytesseract + Pillow) | `False` | Raises `MissingOptionalDependencyError` naming the extra |
| `ScreenshotOcrReader` | Screenshot images | `pii-anon[ocr]` | `False` | Raises `MissingOptionalDependencyError` naming the extra |
| `DicomReader` | DICOM medical images | `pii-anon[dicom]` (pydicom) | `False` | Raises `MissingOptionalDependencyError` naming the extra |
| `AudioReader` | Audio (ASR) | No backend shipped | `False` — always | Raises `MissingOptionalDependencyError`; ASR backend is Pass-2 |

None of these readers silently returns empty text when a backend is absent (NFR-026
loud-degradation invariant).

### 6b. Querying reader capabilities

`reader_capabilities()` returns `list[ReaderCapabilities]` — a list, not a dict. Each
item exposes `format_name`, `dependency_available`, `native_dependency`,
`extracts_text`, `supports_reconstruction`, and `notes` fields:

```python
from pii_anon.ingestion.native import reader_capabilities

caps = reader_capabilities()   # list[ReaderCapabilities], sorted by format_name
for cap in caps:
    print(f"{cap.format_name}: available={cap.dependency_available}, notes={cap.notes}")
```

Or via CLI:

```bash
pii-anon capabilities --output json
```

The capabilities surface for readers is currently served by
`pii_anon.ingestion.native.reader_capabilities()`. Surfacing it directly on the
orchestrator's `capabilities()` is a Pass-2 wire-in (the S2-03 orchestrator block).

### 6c. Using a reader directly

```python
from pii_anon.ingestion.native import default_reader_registry
from pii_anon.ingestion import IngestConfig

registry = default_reader_registry()
reader = registry.get("pdf")
for record in reader.read("document.pdf", IngestConfig()):
    print(record.text)
```

OCR extraction quality validation is Pass-2; the in-tree OCR path provides extraction
but does not yet gate on per-modality recall regression (FR-033/FR-035 thin-trace items).
DICOM extraction is similarly Pass-2 for full pixel-text handling.

### 6d. Third-party readers via entry-point discovery

A package can register a custom reader:

```toml
[project.entry-points."pii_anon.readers"]
my-format = "my_pkg.reader:MyFormatReader"
```

The `NativeReaderRegistry` discovers these on startup. A broken reader package is skipped
without affecting the others.

---

## 7. Multilingual fairness gate

The fairness gate (FR-039, DC-15) tests whether the worst-group per-language recall gap
across POWERED language groups stays within a threshold (default 0.10, per NFR-025).

### 7a. When to run it

Run the fairness gate:

- Before promoting a model that touches the detection or routing layer to a new language.
- As part of a CI regression check when the dataset or engine configuration changes.
- When evaluating a BYO detector's fairness across languages.

### 7b. Programmatic API

```python
from pii_anon.eval_framework.metrics.fairness_gate import (
    evaluate_language_fairness,
    LanguageGroupSlice,
)

slices = [
    LanguageGroupSlice(language="en", gold=en_gold_spans, predicted=en_pred_spans),
    LanguageGroupSlice(language="es", gold=es_gold_spans, predicted=es_pred_spans),
    LanguageGroupSlice(language="fr", gold=fr_gold_spans, predicted=fr_pred_spans),
]
report = evaluate_language_fairness(slices, gap_threshold=0.10, power_floor=200)
print(report.verdict)            # PASS | FAIL | INSUFFICIENT_POWER
print(report.worst_group_recall_gap)
print(report.powered_groups)
print(report.unpowered_groups)   # reported observationally, never gate-driving
```

### 7c. INSUFFICIENT_POWER semantics

The gate returns `INSUFFICIENT_POWER` (not PASS) when fewer than two language groups
meet the statistical-power floor (`power_floor` gold positives per slice). One powered
group or zero powered groups is never treated as evidence of fairness. This is the
fail-closed discipline: insufficient data produces an explicit non-verdict, not a
spurious PASS.

Unpowered groups are reported in `report.unpowered_groups` with their recall visible in
`report.per_language_recall` — they are observable but not gate-driving. Add more
labeled data for those languages to earn them into the powered set.

### 7d. DATA-track note

The corpus-scale fairness number over the full 60-language eval set at the NFR-004 power
tiers requires the `pii-anon-eval-data` dataset (DATA-track, external to this repo).
The gate primitive itself is CODE-local and available now; the full-corpus run is
DATA-track Pass-2.

---

## 8. Operational notes

### 8a. Recall-floor guarantee (SharedLayerProjector)

When running the swarm pipeline (MoE or swarm fusion mode), pii-anon enforces a
structural recall-floor guarantee:

```
entities(output) ⊇ entities(shared)
```

The `SharedLayerProjector` ensures every span found by the always-on `regex-oss` shared
layer survives into the final output — even if a downstream emission threshold or
corroboration gate would otherwise have suppressed it. Re-injected spans are tagged
`provenance=shared_floor` in the explanation field and can be detected with
`is_shared_floor(finding)`.

This is verified by a 2,000-case property suite with 0 violations. The floor holds for
any present or future routing policy because it is applied as a final projection,
decoupled from routing.

NFR-009 latency ceilings are declared in `pii_anon.eval_framework.evaluation.latency_ceilings`
(the committed per-profile numeric registry). NFR-008 (early-exit chunk latency p50 ≤ 1 ms
∧ p95 ≤ 2 ms) is documented in the registry but does not yet have a dedicated named
acceptance test (a SHOULD gap — documented-not-gated).

### 8b. Encrypted token store at rest (FR-019, DC-04)

Reversible pseudonymization stores per-scope surrogate tokens in a SQLite database. When
the `crypto` extra is installed (`pip install "pii-anon[crypto]"`), use
`EncryptedSQLiteTokenStore` for AEAD-encrypted storage at rest.

The real constructor signature (verified against
`src/pii_anon/tokenization/encrypted_store.py` line 252):

```python
EncryptedSQLiteTokenStore(
    db_path: str | Path = ":memory:",
    *,
    key_provider: EnvelopeKeyProvider,
    algorithm: str = "aesgcm",   # or "chacha20poly1305"
)
```

`key_provider` is the KEK (key-encryption key) custodian. The in-tree
`StaticTestKeyProvider` is an AGENT_SIMULATED stand-in suitable for development and
testing. It takes raw KEK bytes directly (16, 24, or 32 bytes for AES-GCM) and a
`key_id` string; it never performs networking or subprocess calls.

A real production custodian (cloud KMS, HSM, OS keyring, or any networked secret
manager) is a Pass-2 drop-in behind the `EnvelopeKeyProvider` protocol, which requires
three methods: `current_key_id() -> str`, `wrap(dek: bytes) -> tuple[str, bytes]`, and
`unwrap(key_id: str, wrapped_dek: bytes) -> bytes`.

```python
# Requires: pip install "pii-anon[crypto]"
from pii_anon.tokenization.encrypted_store import (
    EncryptedSQLiteTokenStore,
    StaticTestKeyProvider,
)

# Development / testing: StaticTestKeyProvider with a synthetic KEK.
# In production, replace with a real EnvelopeKeyProvider backed by your KMS/HSM.
kek = b"change-me-32-byte-production-key"   # must be 16, 24, or 32 bytes
provider = StaticTestKeyProvider(test_kek=kek, key_id="my-kek-v1")

store = EncryptedSQLiteTokenStore(
    db_path="tokens.db",
    key_provider=provider,
    algorithm="aesgcm",   # default; "chacha20poly1305" also accepted
)
```

If `pii-anon[crypto]` is not installed, constructing `EncryptedSQLiteTokenStore` raises
`TokenizationError` with the `pip install pii-anon[crypto]` remediation hint. Importing
the module always succeeds (discovery is never broken).

Key-state separation invariant (NFR-015, Art. 4(5) proxy): the artifact alone must not
allow re-join — re-identification from the stored artifact without the key must fail.
The `EncryptedSQLiteTokenStore` upholds this by design: the bare DEK is never written
to the SQLite file; only its envelope-wrapped form is persisted. The in-memory fallback
(`InMemoryTokenStore`) provides no at-rest encryption and is suitable only for
development and testing.

Key rotation: call `store.rotate(new_provider)` to swap to a new KEK custodian and
fresh DEK (existing rows remain decryptable under their original key id). Call
`store.rewrap_all(new_provider)` to re-encrypt every row under the new DEK. The key
manager (`pii_anon.tokenization.key_manager`) handles versioned envelope keys for the
orchestrator-level token store.

### 8c. Non-strippable re-identification caveat (NFR-016)

pii-anon exports privacy artifacts that carry a non-strippable anti-anonymity caveat:
"anonymization cannot guarantee zero re-identification risk." This is a first-class
program invariant — 100% of exported privacy artifacts must carry this caveat.
Operators must NOT strip or suppress this caveat in downstream processing.

### 8d. Configuration via YAML, JSON, or environment variables

```bash
# Via YAML file
pii-anon detect "text" --config my-config.yaml

# Via environment variables (auto-coerced)
export PII_ANON__TRANSFORM__DEFAULT_MODE=anonymize
export PII_ANON__MOE__TOP_K=4
```

Full config reference: `docs/configuration.md`. All settings have sensible defaults —
zero configuration is required to get started.

### 8e. Determinism

All scoring and transform operations are deterministic given the same inputs, key, and
seed (NFR-005). The `canonical-run` command accepts `--seed` (default `20240601`).
Byte-identical results across N=5 replays is the acceptance criterion.

### 8f. Stream/batch/offline parity (FR-036)

The same scrub decisions are produced regardless of whether processing happens via
`run` (single), `run_stream` (streamed), or offline batch. Chunk-boundary parity is
enforced (NFR-023). If you observe divergence across modes on the same input, this is
a bug — file an issue.

### 8g. Release and publish operations

This subsection covers the operator workflow for publishing a new release. The full
procedure is in `docs/release-guide.md`; what follows is the critical-path summary.

**Prerequisites.** Python 3.12 venv, all dependencies installed (`pip install -e
".[dev,cli,crypto,benchmark,datasets]"`), NLP models downloaded (spaCy, Stanza,
GLiNER), and GitHub CLI authenticated (`gh auth login`).

**Quality gates.** Run `make all` before any release action. This executes lint (ruff),
type check (mypy), the full test suite, performance SLAs, build, and `twine check`.
Nothing proceeds if any step fails.

**Swarm training.** Run `make train-swarm` (or `make train-swarm SWARM_MAX_RECORDS=0`
for a full production run over all ~160K records). Verify that `~/.pii_anon/swarm/`
contains `ds_params.json`, `temperature.json`, `informativeness.json`, and
`manifest.json`. The `manifest.json` must show `feature_version=2` and a non-default
`emission_threshold`.

**Benchmark.** Run `make benchmark-full` (or `make benchmark-canonical` for a
strict publish-grade run). The `make benchmark-doctor` target reports what competitor
engines are available before you start. The full uncapped run (`BENCH_MAX_SAMPLES=0`)
takes 1–2 days on a laptop; the default 5K-sample cap produces stable estimates for
iteration and PR review.

**Tagging and publishing.** After all gates pass:

```bash
# Read version from pyproject.toml and tag
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
git tag -a "v${VERSION}" -m "pii-anon ${VERSION}"
git push origin "v${VERSION}"
```

Pushing a `v*` tag triggers the `publish-release` GitHub Action, which runs quality
gates, builds the wheel and sdist, and publishes to PyPI. A `v*-rc*` tag publishes to
TestPyPI instead.

**Post-publish smoke test.** After PyPI publication:

```bash
python3.12 -m venv /tmp/pii-anon-verify && source /tmp/pii-anon-verify/bin/activate
pip install "pii-anon[cli,datasets]"
pii-anon version
pii-anon health --output json
pii-anon detect "Contact alice@example.com"
deactivate && rm -rf /tmp/pii-anon-verify
```

**Python version note.** Use Python 3.10–3.12. Python 3.13 lacks spaCy/thinc wheels
in some configurations; 3.14 is unsupported. If `pip install` fails with
`Failed to build thinc / spacy`, switch to Python 3.12.

**Repo layout.** Three repos participate in a release:

| Repo | Holds |
|---|---|
| `pii-anon` | Library source, tests, CI, benchmarks |
| `pii-anon-doc` | Standalone documentation site |
| `pii-anon-eval-data` | `pii-anon-datasets` package (JSONL benchmark data) |

Tag all three with matching version strings when the eval-data or documentation changes
accompany a library release.

---

## 9. CLI command census

All commands confirmed from `src/pii_anon/cli.py`. Install `pii-anon[cli]` to access
the entrypoint.

| Command | Purpose | Key flags |
|---|---|---|
| `pii-anon detect` | Detect + transform a single text record | `--transform-mode`, `--mode`, `--language`, `--config`, `--output` |
| `pii-anon detect-stream` | Stream a file of text payloads (one per line) | `--transform-mode`, `--mode`, `--language`, `--config` |
| `pii-anon process-file` | Process CSV/JSON/JSONL/TXT file | `--transform-mode`, `--format`, `--output`, `--segmentation` |
| `pii-anon tokenize` | Tokenize + detokenize a value (key-management debug) | `--entity-type`, `--token-key` |
| `pii-anon health` | Engine health check | `--config`, `--output` |
| `pii-anon capabilities` | Report available engines + reader capabilities | `--config`, `--output` |
| `pii-anon evaluate` | Compare fusion strategies on the benchmark dataset | `--strategies`, `--dataset` |
| `pii-anon evaluate-pipeline` | Evaluate pii-anon pipeline (F1, precision, recall) | `--dataset`, `--mode`, `--transform-mode`, `--max-samples` |
| `pii-anon eval-framework` | Run the eval framework on a labeled dataset | `--dataset`, `--language`, `--max-records` |
| `pii-anon rate-elo` | Score a BYO detector; produce pii-rate-elo leaderboard | `--predictor`, `--system-name`, `--deployment-profile`, `--artifact-dir` |
| `pii-anon benchmark` | Full pii-anon benchmark run | `--mode`, `--dataset`, `--max-samples` |
| `pii-anon compare-competitors` | Multi-system competitor comparison | `--dataset`, `--dataset-source`, `--require-all-competitors` |
| `pii-anon benchmark-preflight` | Runtime readiness check for publish-grade suite | `--strict-runtime`, `--require-all-competitors` |
| `pii-anon benchmark-publish-suite` | Full canonical publish-grade benchmark suite | `--artifacts-dir`, `--enforce-floors`, `--enforce-publish-claims` |
| `pii-anon calibrate-offline` | Offline MoE expert weight calibration | `--dataset`, `--store-path` |
| `pii-anon verify-dominance` | Verify MoE dominance guarantee (ensemble >= best expert) | `--dataset` |
| `pii-anon canonical-run` | Produce certified canonical-run artifact | `--output-dir`, `--seed`, `--max-samples` |
| `pii-anon supremacy` | Read SDO CompetitiveSupremacyGate verdict | `--artifact`, `--canonical-claim` |
| `pii-anon version` | Print the installed version | (none) |

For any command, `--help` prints the full flag list. The CLI epilog (`pii-anon --help`)
lists the documentation entry points.

---

## Methodology

This guide was authored directly from the following canonical sources (all read in full
prior to authoring):

- `docs/quickstart.md` — install commands, Python API examples, CLI quickstart, BYO
  pipeline pattern. All code examples are sourced verbatim or adapted from this file.
- `docs/evaluate-your-pipeline.md` — BYO predictor contract, programmatic API, CLI
  `rate-elo` flags, SDK plugin pattern, incumbent identical-path explanation, certify-a-run
  workflow, CI gating example, Tier 3 section, troubleshooting table.
- `docs/configuration.md` — fusion modes, policy modes, transform modes, config schema.
- `docs/anonymization-vs-pseudonymization.md` — the no-merge invariant, the two scorer
  families, the vanilla-vs-swarm positioning.
- `docs/recall-floor.md` — the SharedLayerProjector guarantee, usage, verification table.
- `docs/dependencies-and-platforms.md` — supported runtimes, venv setup, OS-specific
  instructions, verify-installation commands.
- `docs/release-guide.md` — release checklist, tag-and-publish workflow, swarm training
  commands, benchmark entry points, cross-platform matrix, repo layout, post-publish
  smoke tests. Read in retry-1 to address the T-01 gap. The §8g release ops subsection
  is sourced directly from this file (§§2–8, Release Checklist).
- `src/pii_anon/cli.py` — the authoritative CLI command census. Every command and flag
  in §9 was verified against the actual `@app.command(...)` decorators in this file.
  No command name or flag that is absent from the file has been documented.
- `pyproject.toml` — the extras matrix (§1). Every extra name and dependency group was
  read from `[project.optional-dependencies]` and the entry-point groups.
- `src/pii_anon/ingestion/native.py` — reader capabilities matrix, LOUD-degradation
  behavior, the Pass-2 wire-in SWITCH-POINT comments. In retry-1: confirmed that
  `reader_capabilities()` returns `list[ReaderCapabilities]` (line 360 return type
  annotation); fixed §6b example accordingly (F-04).
- `src/pii_anon/eval_framework/metrics/fairness_gate.py` — INSUFFICIENT_POWER semantics,
  the FairnessGateReport fields, the fail-closed discipline.
- `src/pii_anon/tokenization/encrypted_store.py` — read in full in retry-1 (previously
  file-existence only). Real `__init__` signature confirmed at line 252:
  `EncryptedSQLiteTokenStore(db_path, *, key_provider: EnvelopeKeyProvider, algorithm: str)`.
  `KeyEnvelope.from_env` does NOT exist; the constructor takes `key_provider`, not
  `key_envelope`. `StaticTestKeyProvider.__init__` confirmed at line 180:
  `StaticTestKeyProvider(test_kek: bytes, *, key_id: str = "test-kek-v1")`. §8b example
  corrected accordingly (F-03).
- `dev-assist-artifacts/05-testing/release-readiness-report.md:##Verdict` — the honest
  current SDO verdict (NOT_YET, G6 FAIL, binding constraint).
- `dev-assist-artifacts/06-documentation/02-architecture/doc-architecture.md:§D-6` —
  the deliverable contract, trace IDs to cover (FR-001/002/007/008, NFR-005/006/009,
  DC-11/DC-12/DC-14/DC-15), audience mode (EXTERNAL PRODUCT = user guide).
- `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§3/§4/§7/§8` —
  requirement inventory, decision inventory, user-docs tree, code surface verification.

**What is agent-inferred, not artifact-sourced:** The headings and organizational
structure of this guide are author-composed from the above sources, not reproduced from a
single doc. The prose in the "operational notes" section synthesizes across multiple
source files. No claim is made without a traceable source.

**Retry-1 corrections:** Three findings fixed against live source reads:
(F-03 MAJOR) §8b: replaced the fabricated `key_envelope=KeyEnvelope.from_env(...)` call
with the real constructor `key_provider=StaticTestKeyProvider(test_kek=..., key_id=...)`
and documented the `EnvelopeKeyProvider` protocol seam; (F-04 MINOR) §6b: fixed
`caps.items()` (dict API) to iterate the returned `list[ReaderCapabilities]` directly,
accessing `.format_name` / `.dependency_available` / `.notes` fields; (T-01 MINOR) added
§8g release ops subsection sourced directly from `docs/release-guide.md` (previously
unread — the prior methodology gap note said the subsection was omitted).

**What is absent / not verified:** The Tier-3 and MIA attack surfaces are summarized at
the DATA-track boundary but not documented in detail, as the corpus-scale evaluation
requires the external `pii-anon-eval-data` dependency (cross-repo edge). No content from
`docs/pii-rate-elo-value.md` (user-WIP, excluded from docs gate) or
`docs/benchmark-summary.md` (auto-generated, volatile) was used in this guide.

**Honesty constraint (O-3):** No claim is made that requirements were validated against
real users. All requirements are AGENT_SIMULATED. Real-user validation is a documented
Pass-2 follow-up.

---

## Sources

| Source file:section | Trace IDs supplied |
|---|---|
| `docs/quickstart.md:##Install + ##Detect + ##Stream + ##CLI quickstart + ##Evaluate your own pipeline` | FR-001, FR-019, NFR-005, NFR-026 |
| `docs/evaluate-your-pipeline.md` (full) | FR-001, FR-002, FR-007, FR-008, NFR-005, NFR-006, DC-12 |
| `docs/configuration.md` (full) | FR-018, FR-019, FR-020, NFR-009 |
| `docs/anonymization-vs-pseudonymization.md` (full) | FR-006, FR-009, FR-010, NFR-014, NFR-015, DC-08 |
| `docs/recall-floor.md` (full) | FR-016, NFR-011, AX-003, DC-01 |
| `docs/dependencies-and-platforms.md` (full) | NFR-022, NFR-026, FR-037 |
| `docs/release-guide.md` (§§1–8, Release Checklist — read retry-1) | NFR-022, FR-037 (release ops) |
| `src/pii_anon/cli.py` (full — command census) | FR-001, FR-002, FR-007, FR-008, DC-11, DC-12 |
| `pyproject.toml:[project.optional-dependencies] + entry-point groups` | NFR-026, FR-019, FR-031, DC-04, DC-14 |
| `src/pii_anon/ingestion/native.py` (full — read retry-1 for return-type confirmation) | FR-031, FR-032, FR-033, FR-035, NFR-026, DC-14 |
| `src/pii_anon/eval_framework/metrics/fairness_gate.py` (module docstring + dataclasses) | FR-039, NFR-025, NFR-004, DC-15 |
| `src/pii_anon/tokenization/encrypted_store.py` (full — read retry-1 for real signatures) | FR-019, NFR-014, NFR-015, DC-04 |
| `dev-assist-artifacts/05-testing/release-readiness-report.md:##Verdict + ##Caveats` | G1–G7, NFR-011, FR-016 |
| `dev-assist-artifacts/06-documentation/02-architecture/doc-architecture.md:§D-6` | FR-001/002/007/008, NFR-005/006/009, DC-11/12/14/15 |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§3/§4/§7/§8` | FR-001..039 (inventory), DC-01..15 (decision inventory), SO-15/16/19/20/21/22 |
