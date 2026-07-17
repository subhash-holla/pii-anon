# Contributor Handbook — pii-anon v1.4.0

> **Audience:** external OSS contributors, third-party integrators extending pii-anon via plugins,
> and any maintainer onboarding to the project's development discipline.
>
> **Caveat (O-5):** No top-level `CONTRIBUTING.md` exists at the repo root (confirmed: only `LICENSE`
> and `README.md` are present). This handbook is the authoritative artifact-tree compilation that can
> seed that file. A `CONTRIBUTING.md` at the repo root is a recommended follow-up action.
>
> **Caveat (O-3):** All gate disciplines described here are AGENT_SIMULATED. The 5-reviewer "panel"
> is an agent-simulated multi-perspective review pass, not human peer reviewers. Real-user
> contribution workflows are a documented Pass-2 follow-up.

---

## 1. Development Setup

### Prerequisites

- Python 3.10–3.13 (development environment uses 3.12)
- A virtual environment at `.venv` in the repo root
- `pydantic>=2.8,<3` is the only hard runtime dependency; all engine and evaluation extras are optional

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,engines]"      # dev tools + optional engine adapters
```

For the full optional stack (Bayesian rating engine + attack harness):

```bash
pip install -e ".[dev,engines,bayes-eval,attacks]"
```

### Verification Gates (exact commands)

Three gates are enforced in CI. Run all three before every pull request:

```bash
# Lint — ruff checks src/ and tests/
make lint
# equivalent: .venv/bin/python -m ruff check src tests

# Type — mypy strict on the library source (tests/ is NOT gated)
make type
# equivalent: .venv/bin/python -m mypy src/pii_anon

# Test — full pytest suite, PYTHONPATH=src
make test
# equivalent: PYTHONPATH=src .venv/bin/python -m pytest
```

**Fast parallel run (recommended for iteration):** the project `.venv` ships `pytest-xdist`; the
system Python may not. Use the project venv for xdist:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -n auto
```

This runs in roughly 7 minutes on 10 cores versus 45-80 minutes serial. Pass/fail results and
coverage are identical to the serial run.

**mypy invocation note:** `make type` invokes `mypy src/pii_anon` which reads `pyproject.toml`
`[tool.mypy] strict=true`. An explicit `mypy src/pii_anon --strict` on the command line enables
`disallow_any_generics` more aggressively and may flag bare `np.ndarray`; parametrize numpy
annotations as `npt.NDArray[np.float64]` to be clean under both invocations.

**ruff format** is not a CI gate; it is a style tool you may run optionally.

---

## 2. The Four Plugin Seams

pii-anon exposes four entry-point groups for third-party extension. Each group is auto-discovered
at runtime when `auto_discover_engines: true` is set in config (or the corresponding env var).
A plugin ships as a standard Python package that declares the group in its own `pyproject.toml`.

### 2a. Detection Engines — `pii_anon.engines`

**What it is:** a detection engine that participates in the swarm (Layer 1–4 fusion, Dawid-Skene,
MoE routing, recall-floor guarantee). Implements `DC-01` and `DC-02` (FR-016, FR-018).

**Contract:** subclass `EngineAdapter` from `pii_anon.engines.base`. Only `detect()` is mandatory.

```python
from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineFinding, Payload

class MyDetector(EngineAdapter):
    adapter_id = "my-detector"          # unique; used as registry key + MoE expert_id
    supported_entity_types = {"PERSON_NAME", "EMAIL_ADDRESS"}

    def detect(self, payload: Payload, context: dict) -> list[EngineFinding]:
        # Return EngineFinding objects with entity_type, confidence [0,1],
        # span_start, span_end (exclusive), engine_id == self.adapter_id
        ...
```

Optional lifecycle hooks (`initialize`, `health_check`, `capabilities`, `shutdown`,
`dependency_available`) have working base-class implementations; override when needed.

**Entry-point declaration** in your package's `pyproject.toml`:

```toml
[project.entry-points."pii_anon.engines"]
my-detector = "my_package.detector:MyDetector"
```

**Discovery semantics:** when `auto_discover_engines: true`, `PIIOrchestrator` calls
`importlib.metadata.entry_points(group="pii_anon.engines")` at startup and instantiates every
discovered adapter. The adapter participates in Dawid-Skene aggregation, temperature calibration,
and the XGBoost meta-learner immediately (with default weights T=1.0, informativeness=0.5) —
no retrain required for basic participation.

**Graceful degradation without retraining:** the engine's vote still increments `corroboration_count`
and flows into the min/max/mean confidence features (features 4-7). Dawid-Skene does not learn a
per-engine confusion matrix until you retrain with `make train-swarm`. This is graceful degradation,
not a failure.

**Recall-floor invariant (AX-003):** the swarm's Layer 2 greedy set-cover pruner drops engines
whose entity-type set has Jaccard similarity >= 0.85 with a higher-ranked engine. To survive pruning,
pin your engine via `SwarmConfig.force_include_engines=("my-detector",)`. Pinned engines bypass the
Jaccard check and `max_engines` cap and always participate in fusion. The recall-floor guarantee
(`entities(ensemble) ⊇ entities(shared)`) must be preserved — your engine must not become the
sole path for a shared-layer entity type.

**MoE expert profile:** to participate in the MoE router's top-K selection, register an
`ExpertSpec` with `entity_strengths` scores after registering the adapter. See
`docs/engine-plugin-guide.md` for the full walkthrough.

**Optional-dependency convention (NFR-026):** if your engine requires a non-core dependency,
declare it in `native_dependency`. pii-anon calls `dependency_available()` before invoking
`initialize()`; an unavailable engine should raise `MissingOptionalDependencyError` (never
silently emit zero findings — silent failure causes undetected recall loss).

---

### 2b. Rating Engines — `pii_anon.rating_engines`

**What it is:** a statistical rating engine for the pii-rate-elo evaluation ladder. Implements
`DC-06` (FR-003, NFR-001, NFR-026).

**Contract:** implement the `RatingEnginePort` Protocol from
`pii_anon.eval_framework.rating.port`. The port is `@runtime_checkable`; `isinstance` checks
pass for any class that structurally provides the two required methods. The MINIMAL contract
(copied from `src/pii_anon/eval_framework/rating/port.py`, verified at D4 retry-1) is:

```python
from pii_anon.eval_framework.rating.port import RatingEnginePort
from pii_anon.eval_framework.rating.elo import EloRating, RatingUpdate

class MyRatingEngine:  # implements RatingEnginePort structurally
    engine_id = "my-rater"

    def run_round_robin(self, composites: dict[str, float]) -> list[RatingUpdate]:
        """Run an all-pairs round-robin from name -> composite score."""
        ...

    def get_rating(self, name: str) -> EloRating | None:
        """Return the current rating for name, or None if unknown."""
        ...
```

The richer engine API (`run_reidentification_tournament`, `tournament_summary`,
`evaluate_governance`, `update_from_match`, `get_leaderboard`) is intentionally NOT part of this
port — it stays on the concrete class so future tiers are not over-constrained (DC-06 design
intent, documented in `port.py` module docstring).

**Entry-point declaration:**

```toml
[project.entry-points."pii_anon.rating_engines"]
my-rater = "my_package.rater:MyRatingEngine"
```

**Discovery semantics:** `RatingEngineRegistry` calls
`importlib.metadata.entry_points(group="pii_anon.rating_engines")` and registers all found
engines alongside the three in-tree tiers (`glicko-legacy`, `bradley-terry-mle`, `bayes-bt`).
The ladder tiers are ordered: `glicko-legacy` (lightweight smoke), `bradley-terry-mle` (MLE),
`bayes-bt` (claim-grade, requires the `bayes-eval` extra). A new third-party engine is registered
as an additional tier; it does not replace the in-tree tiers.

**Graceful degradation:** if your engine's optional dependency is absent, raise
`MissingOptionalDependencyError` at call time (not at import). The registry keeps the engine
discoverable; callers that invoke it without the dependency get a loud error, not a silent fallback.

**NFR-001 convergence gate:** the `bayes-bt` tier ships a `convergence.py` gate (split-R-hat <= 1.01,
bulk-ESS >= 400/param, 0 divergences). A rating engine that makes claim-grade assertions should
expose an equivalent convergence certificate or clearly document its claim-grade conditions.

---

### 2c. BYO Predictors — `pii_anon.byo_pipelines`

**What it is:** a Predictor callable for the BYO-pipeline evaluation SDK (FR-001, FR-002, DC-12,
SO-20). Plugs a third-party detection pipeline into the identical scoring path so it can be
compared against pii-anon and other incumbents on the same dataset.

**Contract:** a Predictor is a callable with the signature:

```python
# Predictor signature (structural duck-typed, not a formal ABC)
def my_predictor(text: str) -> list[tuple[str, int, int]]:
    # Returns: list of (entity_type, span_start, span_end)
    ...
```

The `engine_predictor()` factory in `eval_framework/byo_pipeline.py` wraps a `PIIOrchestrator`
instance as a Predictor; `incumbent_predictor(name)` retrieves any of the five in-tree incumbents
by name (`INCUMBENT_SYSTEMS`). Both paths route through the same scoring logic so comparisons are
methodologically identical.

**Entry-point declaration:**

```toml
[project.entry-points."pii_anon.byo_pipelines"]
my-pipeline = "my_package.eval:my_predictor"
```

**Discovery semantics:** `BYOPipelineRegistry` discovers all advertised Predictors via
`importlib.metadata.entry_points(group="pii_anon.byo_pipelines")`. Discovered predictors can be
passed directly to `evaluate_incumbent()` or included in `build_identical_path_leaderboard()`.

**Identical scoring path guarantee (FR-002):** pii-anon's own detection, the five in-tree incumbents
(gliner, presidio, scrubadub, spacy-ner, stanza-ner), and your third-party predictor all run
through the same `evaluate_incumbent()` code path. There is no evaluation-path asymmetry — a
lower score for your pipeline is a real deficit, not a harness artifact.

**Graceful degradation:** if your predictor raises on a given record, the harness logs the error
and assigns that record a zero-recall, zero-precision result. Crash-free participation on all
record types is part of the contract.

---

### 2d. Native-Format Readers — `pii_anon.readers`

**What it is:** a native-format reader that emits `Iterator[IngestRecord]` from a non-text source
(PDF, image, screenshot, DICOM, audio). Implements `DC-14` (FR-031, FR-032, SO-21).

**Contract:** implement the `NativeReader` Protocol from `pii_anon.ingestion.native`. The REAL
dataclass fields for `ReaderCapabilities` and method signatures (copied from
`src/pii_anon/ingestion/native.py` lines 61–92, verified at D4 retry-1) are:

```python
from pii_anon.ingestion.native import NativeReader, ReaderCapabilities
from pii_anon.ingestion.schema import IngestConfig, IngestRecord
from collections.abc import Iterator
from pathlib import Path

class MyFormatReader:  # implements NativeReader Protocol
    format_name: str = "my-format"   # class-level attribute required by Protocol

    def capabilities(self) -> ReaderCapabilities:
        # capabilities() is a METHOD, not a property
        return ReaderCapabilities(
            format_name="my-format",
            native_dependency="my_backend_package",  # or None if stdlib-only
            dependency_available=True,               # check at runtime
            extracts_text=True,                      # default True
            supports_reconstruction=False,           # FR-032: no reader claims True yet
            notes="Optional human-readable description",
        )

    def read(
        self, path: str | Path, config: IngestConfig
    ) -> Iterator[IngestRecord]:
        # yield IngestRecord objects under the uniform IngestRecord contract
        ...
```

`ReaderCapabilities` fields (all required except the four with defaults):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `format_name` | `str` | — | e.g. `"pdf"`, `"image"`, `"dicom"` |
| `native_dependency` | `str \| None` | — | importable module name, or `None` |
| `dependency_available` | `bool` | — | truthful runtime check result |
| `extracts_text` | `bool` | `True` | whether the reader produces text |
| `supports_reconstruction` | `bool` | `False` | FR-032: no in-tree reader claims `True` |
| `notes` | `str` | `""` | human-readable capability notes |

**Entry-point declaration:**

```toml
[project.entry-points."pii_anon.readers"]
my-format = "my_package.reader:MyFormatReader"
```

**Discovery semantics:** `NativeReaderRegistry` discovers all advertised readers via
`importlib.metadata.entry_points(group="pii_anon.readers")`. The function `default_reader_registry()`
returns a pre-populated registry with the five in-tree readers; `reader_capabilities()` returns
their capability profiles. Your reader is registered alongside the in-tree readers.

**Graceful degradation (NFR-026):** if your reader's optional extraction dependency is absent (e.g.,
an OCR library), raise `MissingOptionalDependencyError` rather than returning empty records silently.
The registry reflects the reader's presence; callers without the dependency get a clear diagnostic,
not a silent recall drop.

**Round-trip constraint (FR-032):** `supports_reconstruction=False` is the honest default for all
in-tree readers; no native-format reader claims lossless round-trip. If your format supports it,
set `supports_reconstruction=True` and document the scope precisely.

---

## 3. House Testing Discipline

### 3a. TDD: RED → GREEN → REFACTOR

Every story is committed as three distinct phases, git-evidenced:

1. **RED** — failing tests committed first. The test file exists; CI fails on this commit. Test
   names follow the `test_aN` pattern (A1, A2, ...) derived from the story's acceptance criteria.
2. **GREEN** — minimal implementation makes the tests pass. No clean-up or refactoring yet.
3. **REFACTOR** — code-quality pass; all tests remain green. Import cleanup, docstrings, type
   annotations.

This sequence is a CI-verifiable invariant, not a style preference. A story gate reviewer
checks the git log for the RED commit with a failing suite. A GREEN commit with no prior RED is
a MAJOR finding.

### 3b. Anchor Tests — Exact Values, Never Bounds

Metric-producing code (recall rates, F2 scores, ECE values, fairness gaps, latency percentiles)
must be pinned with **exact reference values**, not range checks:

```python
# WRONG: passes for any value that drifts upward
assert result.f2_score >= 0.72

# CORRECT: catches any drift from the established reference
assert result.f2_score == pytest.approx(0.7214, abs=1e-4)
```

This discipline caught real regressions during the PDLC pass (S5-02 RC-02, S5-03 RC-01, S6-01).
A loose bounds check silently absorbs performance regressions; an exact anchor surfaces them as
test failures on the first deviation.

The lesson recurred three times in the SOTA program. Treat it as an invariant: if you are adding
a test for a metric, anchor it to the current reference value from an artifact or a seeded run.

### 3c. Import-Boundary AST Audits

Package-isolation invariants are enforced by AST-based tests in CI. Two boundaries are currently
gated:

- **Rating package isolation** (`test_rating_import_boundary.py`): modules under
  `pii_anon.eval_framework.rating` must not import from the detection/orchestration layer
  (`pii_anon.swarm`, `pii_anon.moe`, `pii_anon.fusion`, etc.). This enforces DC-06's
  ports-adapters architecture.
- **Attacks package isolation** (`test_attacks_import_boundary.py`): modules under
  `pii_anon.eval_framework.attacks` must not import from `pii_anon.swarm`, `pii_anon.moe`,
  `pii_anon.fusion`, or `pii_anon.policy`. This enforces DC-09's isolation of the adversarial
  attack harness from the detection layer.

When you add a new sub-package with a hard isolation requirement, add a corresponding AST guard
test. The checks are AST-based (not substring) so they cannot false-positive on docstrings or
comments that mention a forbidden name.

### 3d. AX-001 Synthetic-Only Fixtures

**No real natural person's identifiable data may appear in any test fixture, example, doc snippet,
or benchmark slice in this repository** (AX-001, NFR-024). This is a release-blocking CATASTROPHIC
finding if violated. Every fixture entity must trace to a synthetic generator or a declared
surrogate pool.

Concretely:

- Do not use real email addresses, phone numbers, or SSNs in test data, even partially.
- Do not use real person + contact pairs from the internet.
- Use `faker` or the project's `SyntheticPIIGenerator` utilities for all test fixtures.
- DICOM and audio test fixtures must use synthetic or fully anonymized source materials with
  documented provenance.

The SAST reviewer checks AX-001 at every story gate. A single Luhn-valid credit card number or
structurally valid SSN not produced by a declared generator is a CATASTROPHIC finding that blocks
the gate.

### 3e. Coverage Floor

The CI coverage gate is `--cov-fail-under=84` (project-wide baseline as of Sprint-1). New modules
introduced by a story are expected to reach >= 90% coverage in the story's own test file. A story
gate reviewer will flag a new module with < 90% own-module coverage as a MINOR finding; < 70% is a
MAJOR.

When adding optional-dependency code paths, use `pytest.importorskip()` to skip tests gracefully
when the optional package is absent. The skipped tests must not pull the coverage below the floor
for the non-skipped paths.

---

## 4. Review Gate and Sign-Off Machinery

### 4a. Story Gates

Every story in the SOTA program runs a canonical 5-reviewer gate before sign-off. The gate covers:

| Reviewer role | What it checks |
|---|---|
| code-quality | TDD discipline, RED→GREEN→REFACTOR commits, code structure, docstrings |
| axiom-compliance | AX-001..006 invariants, especially AX-001 (no real PII) and AX-003 (recall-floor) |
| traceability | Every acceptance criterion maps to an FR/NFR/DC; no orphan code paths |
| security-sast | For stories touching encryption, sandboxing, auth, or agentic interception |
| performance | For stories touching latency, throughput, or the MoE router |

The reviewer set for a given story is drawn from the 8-specialist pool. The conditional roles
(security-sast, performance) activate when the story touches their tagged paths.

A gate verdict of `APPROVE` requires all reviewers to approve. A single `REQUEST_CHANGES` blocks
the story close. The author remediates and re-submits for a targeted re-review; the full panel
re-approves.

### 4b. Severity Taxonomy

The five-level severity taxonomy is used consistently across brownfield assessment, story gate
findings, and adversarial close reports:

| Severity | Meaning | Gate consequence |
|---|---|---|
| SHOWSTOPPER | Functionally broken or security breach that makes the artifact unusable | Gate hard-fails; release blocked |
| CATASTROPHIC | Correctness invariant violated (e.g., fabricated metric, real PII in fixture, phantom SDO PASS) | Gate hard-fails; must be fixed before re-review |
| MAJOR | Meaningful quality, correctness, or security gap that a user will encounter | `REQUEST_CHANGES`; must be remediated before APPROVE |
| MINOR | Improvement needed but not release-blocking; can be deferred with documented rationale | Can be accepted with a written rationale; tracked as follow-up |
| OBSERVATION | Informational note or style suggestion | No gate consequence; logged for context |

A `REQUEST_CHANGES` verdict from any reviewer is a MAJOR or higher finding. When a MAJOR is raised,
the author must address it in a follow-up commit (evidenced RED-fix → GREEN for code changes) and
request a targeted re-review from the raising reviewer.

### 4c. The Adversarial Close for Control-Path Artifacts

The story gate is necessary but not sufficient for control-path code. Any change to the following
artifacts requires an adversarial close (a structured multi-round probe/refute exercise):

- `eval_framework/evaluation/competitive_supremacy.py` — the SDO guarantee gate
- `evaluation/canonical_run.py` — the provenance and canonical-run producer
- `moe_gate_signing.py` + any code that reads/writes `gate_v1.json`
- Any new control-path artifact producer/consumer (a module whose output drives a guarantee verdict)

**Why the adversarial close exists:** during the SOTA program the 5-reviewer gate approved stories
that contained fabrication bugs — including a CATASTROPHIC NaN-curve fabrication that laundered a
non-monotone `risk_coverage_curve` into a `CLAIM_GRADE_SOTA` verdict, and a G7 provenance
fail-OPEN that forged `PROVISIONAL_SOTA` from whitespace-only provenance stamps. The story gate
caught none of these. The adversarial close caught all of them. The gate is necessary; the close
is the standing catch-net.

**Process:** the close is a separate workflow from the story gate, running live break-probe agents
that attempt to forge a PASS verdict or crash the gate. A clean close means 0 upheld probes across
all probe categories. A failing close triggers a remediation commit; the close runs again
(confirmatory round). Multiple rounds are normal for control-path code — budget accordingly.

**Sprint and release gates** aggregate the story-gate evidence across all stories in a sprint and
confirm no regression against the off-limits files (see Section 5). The sprint gate also re-runs
the full test suite with coverage and confirms all three CI gates pass.

---

## 5. What Not to Touch

The following files are protected or have severe constraints on modification:

### 5a. SDO Gate and Provenance Producer

- `src/pii_anon/eval_framework/evaluation/competitive_supremacy.py`
- `src/pii_anon/evaluation/canonical_run.py`

These implement the G1–G7 guarantee functions and the canonical-run provenance producer. Any change
to either file requires the adversarial close (Section 4c). The gate's md5 is tracked across close
rounds to verify byte-identity of the off-limits path. A story that changes either file without
running the adversarial close is a SHOWSTOPPER finding at the sprint gate.

**Why:** the gate contains fabrication-hardening validators (`_finite_unit_score`,
`_is_finite_number`, `_is_nonblank_str`, strict `is True` canonical-run checks) that prevent forged
verdicts. Removing, weakening, or reordering these validators can re-open fabrication holes that
were each caught by a close round.

### 5b. MoE Gate Artifact Signing

- `src/pii_anon/moe_gate_signing.py`
- Any code path that reads `gate_v1.json`

The `gate_v1.json` MoE gate artifact is signed with HMAC-SHA256 at write time and verified on load
via a fail-closed `GateSignatureError`. Any modification to the signing/verification logic requires
the adversarial close. The verify-on-load invariant must be preserved: `verify_on_load=True` is
hard-wired at the production seam.

### 5c. User-WIP Files

The following files are user work-in-progress and must not be staged by automated story commits:

- `src/pii_anon/orchestrator.py` — the public `PIIOrchestrator` API; only story S2-03 (blocked)
  may modify it, and only via the documented `# SWITCH-POINT(ORCH)` seam.
- `tests/test_moe_enhancements.py` — user WIP test file.

These files must be byte-identical before and after a story commit (md5-verified at sprint gate).
Any story that stages them without explicit authorization is a MAJOR finding.

### 5d. Entry-Point Group Names

The four entry-point group names (`pii_anon.engines`, `pii_anon.rating_engines`,
`pii_anon.byo_pipelines`, `pii_anon.readers`) are part of the public API for third-party plugin
packages. Renaming them is a breaking change for all existing plugins. Any rename requires a major
version bump and a migration notice. Do not rename them within the v1.x series.

### 5e. The No-Merge Anonymization/Pseudonymization Invariant

- `docs/anonymization-vs-pseudonymization.md` and the API surface it documents

The anon-vs-pseudo no-merge invariant (AX-004, DC-08, FR-010) prohibits any code path that
averages or collapses the anonymization and pseudonymization scoring families into a single headline
number. The `AnonymizationScorer` and `PseudonymizationIntegrityScorer` in
`eval_framework/metrics/deid_families.py` are distinct and must remain distinct. Adding a
"combined de-identification score" is a CATASTROPHIC finding.

---

## 6. Follow-Up Actions (Recommended for a Real OSS Launch)

These items are documented Pass-2 actions, not current gaps in the code:

1. **Author `CONTRIBUTING.md` at repo root** — seed it from this handbook; add PR template,
   issue template, and code-of-conduct.
2. **Add the five per-stage doc-seeds** (`01-discovery/_doc/doc-seed.md` through
   `05-testing/_doc/doc-seed.md`) to convert agent-inferred narrative into explicit authored prose.
3. **Real-user contribution workflows** — the 5-reviewer "panel" is agent-simulated; Pass-2 should
   define a real human review process with response-time SLAs.
4. **OS-matrix portability certification** (NFR-022, FR-037) — the current suite runs on macOS
   (darwin); Linux and Windows CI matrix is a SHOULD that remains deferred.
5. **examples-and-tests-catalog.md** — the 3,685-test suite is the living catalog, but a curated
   entry-point-centric catalog would lower the barrier for plugin authors.

---

## Methodology

This handbook is an artifact-tree compilation. Every section is authored from one or more of the
mapped sources below; no section is invented.

**D4 LOOPBACK NOTE (retry-1, 2026-06-10):** The original §2b (`RatingEnginePort` contract) and
§2d (`ReaderCapabilities` / `NativeReader.capabilities`) were authored from agent recollection
rather than the live source files — a CATASTROPHIC (§2b) and MAJOR (§2d) finding in the D4 audit.
Both have been corrected at this retry by reading the actual source files:
`src/pii_anon/eval_framework/rating/port.py` (the real contract: `run_round_robin` +
`get_rating`) and `src/pii_anon/ingestion/native.py` lines 61–92 (the real `ReaderCapabilities`
dataclass fields and the `capabilities()` method signature). The Sources table row for
`rating/port.py` previously carried a false "verified" claim; it is now genuinely verified.

**Directly compiled from canonical sources:**

- Sections 1 (verification gates), 3c (import-boundary), 4b (severity taxonomy), and 5a–5e
  (protected files) are compiled directly from `development-log.md`, `project-axioms.yaml`,
  and `assessment-2026-05-30.md`.
- The exact Makefile commands in Section 1 are verified against the live `Makefile` (`lint`,
  `type`, `test` targets).
- The four entry-point group names and the in-tree symbol lists in Section 2 are verified against
  `pyproject.toml` lines 61–92 (confirmed present for all four groups).
- The `EngineAdapter` contract in Section 2a is compiled from `docs/engine-plugin-guide.md` and
  `docs/extend-swarm.md`.
- The `RatingEnginePort` contract in Section 2b is verified against the live source file
  `src/pii_anon/eval_framework/rating/port.py` (read at D4 retry-1): the Protocol declares
  exactly `run_round_robin(composites: dict[str, float]) -> list[RatingUpdate]` and
  `get_rating(name: str) -> EloRating | None`.
- The `NativeReader` Protocol, `ReaderCapabilities` dataclass, and `capabilities()` method in
  Section 2d are verified against `src/pii_anon/ingestion/native.py` lines 61–92 (read at D4
  retry-1): the six real dataclass fields are `format_name / native_dependency /
  dependency_available / extracts_text / supports_reconstruction / notes`, and `capabilities()`
  is a method (not a property) on the Protocol.
- The `BYOPipelineRegistry` contract in Section 2c is verified against the live source file
  `src/pii_anon/eval_framework/byo_pipeline.py`.

**Agent-inferred / reconstructed from multiple sources:**

- Section 3a (TDD discipline) is reconstructed from `development-log.md:§W3 Quality + §W6
  Execution` (RED→GREEN→REFACTOR commit sequences cited explicitly per story) — no single
  "TDD policy document" exists; the narrative is consistent across all story entries.
- Section 3b (anchor tests) is reconstructed from `development-log.md:§W6 Execution` and the
  SO sign-off entries for S5-02, S5-03, and S6-01, which each document an exact-rate anchor
  remediation in the RC cycle. No single policy document names the "exact-value anchors" rule
  in isolation; the lesson is agent-inferred from three repeated instances in the log.
- Sections 3d (AX-001) and 3e (coverage floor) are compiled from `project-axioms.yaml` (AX-001
  text and severity verbatim) and `development-log.md:§W1 Preflight` (`--cov-fail-under=84`).
  The >= 90% per-module expectation is agent-inferred from the "no regression" discipline
  described in the log; it is not stated in a single policy line.

**Honesty constraints applied:**

- No CONTRIBUTING.md existed at repo root (confirmed, O-5). This handbook IS the compilation.
- The 5-reviewer gate discipline is AGENT_SIMULATED (O-3). Language reflects this throughout.
- The adversarial close description in Section 4c is compiled from `development-log.md:§W6
  Execution` (the SO-series close records are the primary evidence). The fabrication examples
  (NaN-curve, whitespace provenance) are real events documented in the development log and the
  SO sign-off records; they are not invented.

**What is missing:**

- No `CONTRIBUTING.md` to merge guidance from.
- No `examples-and-tests-catalog.md` (O-2); plugin-authoring examples are sourced from the two
  `docs/` extension guides.
- No `doc-seed.md` narratives (O-1); the contribution discipline prose is reconstructed from the
  development log.

---

## Sources

| Source file : section | Trace IDs supplied |
|---|---|
| `dev-assist-artifacts/06-documentation/02-architecture/doc-architecture.md:§D-7` | DOC-architecture, D-7 source mapping |
| `dev-assist-artifacts/06-documentation/01-harvest/doc-source-index.md:§3a, §3b, §4, §8a, §8b` | FR-001, FR-002, FR-003, FR-016, FR-018, FR-021, FR-023, FR-024, FR-031, FR-032, FR-033, NFR-001, NFR-024, NFR-026, DC-01, DC-02, DC-06, DC-08, DC-09, DC-12, DC-13, DC-14 |
| `docs/extend-swarm.md:##Workflow 1 — Plug your own detector, ##Graceful degradation without retraining, ##Troubleshooting` | FR-016, FR-018, DC-01, DC-02, NFR-026 |
| `docs/engine-plugin-guide.md:##Step 1..4, ##Lifecycle Methods, ##Pinning Your Engine, ##Retraining the Swarm` | FR-016, FR-018, DC-02, NFR-026 |
| `pyproject.toml:[project.entry-points."pii_anon.engines"] (line 61)` | FR-016, DC-01 |
| `pyproject.toml:[project.entry-points."pii_anon.rating_engines"] (line 70)` | FR-003, DC-06 |
| `pyproject.toml:[project.entry-points."pii_anon.byo_pipelines"] (line 78)` | FR-001, FR-002, DC-12 |
| `pyproject.toml:[project.entry-points."pii_anon.readers"] (line 87)` | FR-031, DC-14 |
| `dev-assist-artifacts/04-development/development-log.md:§W1 Preflight, §W3 Quality, §W4 Testing, §W6 Execution` | DC-01, DC-06, DC-09 (TDD discipline, coverage floor, severity taxonomy, adversarial close, anchor-test lesson) |
| `dev-assist-artifacts/00-axioms/project-axioms.yaml:AX-001..006` | AX-001 (NFR-024), AX-003 (NFR-011), AX-004 (DC-08, FR-010), AX-006 (FR-025) |
| `dev-assist-artifacts/00-brownfield-assessment/assessment-2026-05-30.md:§4 Findings (severity table)` | 5-severity taxonomy (SHOWSTOPPER/CATASTROPHIC/MAJOR/MINOR/OBSERVATION) |
| `src/pii_anon/eval_framework/byo_pipeline.py` (verified: `BYOPipelineRegistry`, `engine_predictor`, `incumbent_predictor`, `INCUMBENT_SYSTEMS`) | FR-001, FR-002, DC-12 |
| `src/pii_anon/eval_framework/rating/port.py` (verified at D4 retry-1: `RatingEnginePort` — `run_round_robin(composites: dict[str, float]) -> list[RatingUpdate]` + `get_rating(name: str) -> EloRating \| None`) | FR-003, DC-06 |
| `src/pii_anon/eval_framework/rating/registry.py` (verified: `RatingEngineRegistry`) | FR-003, DC-06 |
| `src/pii_anon/ingestion/native.py` (verified at D4 retry-1: `NativeReader` Protocol, `ReaderCapabilities` dataclass fields, `capabilities()` method, `NativeReaderRegistry`, `ImageOcrReader`, `DicomReader`, `AudioReader`, `default_reader_registry`, `reader_capabilities`) | FR-031, FR-032, DC-14 |
| `tests/test_attacks_import_boundary.py` (verified: AST-based DC-09 isolation guard) | DC-09 |
| `Makefile:lint, type, test targets` (verified: exact command strings) | NFR verification gates |
