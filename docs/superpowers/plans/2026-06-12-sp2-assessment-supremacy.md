# sp2 — Assessment Supremacy + 12-Player pii-rate-elo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make vanilla pii-anon and pii-anon-swarm first-class detectors in the
pii-anon-eval-data `baselines` assessment, iterate detection on the dev split
until at least one of them tops the F2-micro table (bar: aws 0.737 / gliner
0.735), and make pii-rate-elo produce a 12-player rating report from the
merged assessment artifact.

**Architecture:** CODE exposes two first-party `Predictor` factories (native
labels, text→spans) in `eval_framework/byo_pipeline.py`; DATA gets two thin
`DetectorAdapter` modules that wrap those predictors and own the native→63
label map. Detection improvements happen in CODE behind its existing engines.
pii-rate-elo gains an assessment-artifact ingestion mode that runs the existing
`PIIRateEloEngine` over per-entity-type F2 matches.

**Tech Stack:** Python 3.11+, pytest (CODE: `.venv` + xdist; DATA: miniforge),
pure-stdlib DATA core, ruff + mypy-strict in CODE.

**Spec:** `docs/superpowers/specs/2026-06-12-sp2-assessment-supremacy-design.md`

**Repos:**
- CODE = `/Users/subhashholla/Development/pii_anonymize_pseudonymize/pii-anon-core/pii-anon-code`
- DATA = `/Users/subhashholla/Development/pii_anonymize_pseudonymize/pii-anon-core/pii-anon-eval-data`

**Hard constraints (every task):**
- `competitive_supremacy.py` md5 stays `3b842e81c3f03eafd11f9c655c1789a0`;
  `canonical_run.py` md5 stays `d8f0f80e113c3b5d59c06d0b5fd36fac`;
  `competitor_compare.py` md5 stays `7cae16c89f4c97136e1a12394dae2025` (import OK, modify NO).
- Never run `scripts/run_full_benchmark.py` (rewrites README).
- Tune only on dev split; test split only for the final reported run.
- User WIP untouched: CODE `orchestrator.py`/`tests/test_moe_enhancements.py`/README/benchmark
  artifacts; DATA `baselines/gcp_dlp_baseline.py`/`tests/test_baselines_adapters.py`.
- CODE verification: `PYTHONPATH=src .venv/bin/python -m pytest -n auto` (full),
  `make lint`, `mypy src/pii_anon`. DATA verification: `python3 -m pytest` from DATA root.
- Commits scoped with `git add <explicit paths>` only.

---

## Phase A — CODE: first-party predictor factories

### Task A1: `pii_anon` (vanilla) predictor factory

**Files:**
- Modify: `src/pii_anon/eval_framework/byo_pipeline.py` (after `_INCUMBENT_FACTORIES`)
- Test: `tests/test_first_party_predictors.py` (new)

- [ ] **Step 1: failing test** — vanilla factory returns native-label spans on a
  known text, lazily, without normalization:

```python
"""First-party predictor factories (sp2): pii-anon vanilla + swarm as BYO predictors."""
from pii_anon.eval_framework.byo_pipeline import first_party_predictor, FIRST_PARTY_SYSTEMS


class TestVanillaPredictor:
    def test_first_party_systems_lists_both(self) -> None:
        assert FIRST_PARTY_SYSTEMS == ("pii_anon", "pii_anon_swarm")

    def test_vanilla_emits_native_email_span(self) -> None:
        predict = first_party_predictor("pii_anon")
        text = "Contact alice@example.com today."
        spans = list(predict(text))
        emails = [s for s in spans if s[0] == "EMAIL_ADDRESS"]
        assert emails, f"no EMAIL_ADDRESS in {spans!r}"
        (etype, start, end) = emails[0][:3]
        assert text[start:end] == "alice@example.com"

    def test_vanilla_keeps_native_iban_label(self) -> None:
        # Native labels, NOT benchmark-canonicalized (IBAN must stay IBAN,
        # not fold into BANK_ACCOUNT as competitor_compare's normalizer does).
        predict = first_party_predictor("pii_anon")
        spans = list(predict("IBAN: DE89370400440532013000"))
        assert any(s[0] == "IBAN" for s in spans), spans

    def test_unknown_name_raises_key_error(self) -> None:
        import pytest
        with pytest.raises(KeyError):
            first_party_predictor("nope")
```

- [ ] **Step 2: run, verify fail** —
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_first_party_predictors.py -x -q`
  → ImportError (`first_party_predictor` undefined).

- [ ] **Step 3: implement** in `byo_pipeline.py` (mirror `incumbent_predictor`
  pattern: lazy factories, cache, docstring):

```python
def _pii_anon_factory() -> Predictor:
    from pii_anon.engines.regex_adapter import RegexEngineAdapter

    return engine_predictor(RegexEngineAdapter(enabled=True))


_FIRST_PARTY_FACTORIES: dict[str, Callable[[], Predictor]] = {
    "pii_anon": _pii_anon_factory,
    "pii_anon_swarm": _pii_anon_swarm_factory,   # Task A2
}

FIRST_PARTY_SYSTEMS: tuple[str, ...] = tuple(sorted(_FIRST_PARTY_FACTORIES))


def first_party_predictor(name: str) -> Predictor:
    """Predictor for a first-party pii-anon system, emitting NATIVE labels."""
    if name not in _FIRST_PARTY_FACTORIES:
        raise KeyError(f"unknown first-party system {name!r}; known: {FIRST_PARTY_SYSTEMS}")
    ...  # same lazy-cache idiom as incumbent_predictor
```

- [ ] **Step 4: run, verify pass** (swarm test may be deferred to A2 if split).
- [ ] **Step 5: commit** — `feat(eval): first-party pii_anon predictor factory (sp2)`

### Task A2: `pii_anon_swarm` predictor factory

**Files:** same module + test file.

- [ ] **Step 1: failing tests** —

```python
class TestSwarmPredictor:
    def test_swarm_emits_email_span(self) -> None:
        predict = first_party_predictor("pii_anon_swarm")
        text = "Contact alice@example.com today."
        spans = list(predict(text))
        assert any(s[0] == "EMAIL_ADDRESS" and text[s[1]:s[2]] == "alice@example.com"
                   for s in spans), spans

    def test_swarm_degrades_to_regex_only_pool(self) -> None:
        # With heavy engines unavailable the swarm pool is regex-only and the
        # fusion path still emits (recall floor guarantees regex spans survive).
        ...
```

- [ ] **Step 2: implement** `_pii_anon_swarm_factory`:
  - engine pool: `RegexEngineAdapter(enabled=True)` always; GLiNER/Presidio/Stanza
    adapters appended when their libs import (try/except per engine);
  - fusion: `build_fusion("swarm")` (locate `build_fusion` — `grep -rn "def build_fusion" src/`;
    it wires `FloorProjectingFusion(SwarmFusionStrategy)` per routing/floor_fusion.py);
  - per call: run every engine's `.detect({"text": text}, {"policy_mode": "balanced",
    "language": language})`, re-stamp `engine_id` like `_ensemble_detector` does
    (regex findings → `engine_id="regex-oss"`), `fusion.merge(all_findings)` →
    emit `(f.entity_type, f.span_start, f.span_end)` for findings with int spans
    (same bool/int guards as `engine_predictor`).
- [ ] **Step 3: tests pass; full file lint/type clean**
  (`make lint`, `mypy src/pii_anon`).
- [ ] **Step 4: commit** — `feat(eval): first-party pii_anon_swarm predictor (sp2)`

---

## Phase B — DATA: adapters + registry

### Task B1: vanilla adapter with the native→63 label map

**Files:**
- Create: `DATA/baselines/pii_anon_baseline.py`
- Test: `DATA/tests/test_pii_anon_adapters.py` (NEW file; user WIP file untouched)

The label map (native pii-anon → canonical-63; `None` = deliberate drop):

```python
LABEL_MAP: dict[str, str | None] = {
    "AADHAAR": "NATIONAL_ID_NUMBER",
    "ADDRESS": "STREET_ADDRESS",
    "AGE": "AGE",
    "API_KEY": "API_KEY",
    "BANK_ACCOUNT": "BANK_ACCOUNT_NUMBER",
    "BAR_NUMBER": "BAR_NUMBER",
    "CANADIAN_SIN": "NATIONAL_ID_NUMBER",
    "COURT_CASE_NUMBER": "COURT_CASE_NUMBER",
    "CREDIT_CARD": "CREDIT_CARD_NUMBER",
    "CRYPTO_WALLET": "CRYPTOCURRENCY_ADDRESS",
    "CVV": "CVV",
    "DATE_ISO": None,            # ambiguous (DOB vs general date) — revisit on dev evidence
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "DATE_TIME": "TIMESTAMP",
    "DEA_NUMBER": "DEA_NUMBER",
    "DOCKET_NUMBER": "DOCKET_NUMBER",
    "DRIVERS_LICENSE": "DRIVER_LICENSE_NUMBER",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
    "GPS_COORDINATES": "LATITUDE_LONGITUDE",
    "IBAN": "IBAN",
    "INSURANCE_POLICY_NUMBER": "INSURANCE_POLICY_NUMBER",
    "INVOICE_NUMBER": "INVOICE_NUMBER",
    "IP_ADDRESS": "IP_ADDRESS",
    "JWT_TOKEN": "AUTHENTICATION_TOKEN",
    "LICENSE_PLATE": "LICENSE_PLATE",
    "LOCATION": "LOCATION_NAME",
    "MAC_ADDRESS": "MAC_ADDRESS",
    "MEDICAL_RECORD_NUMBER": "MEDICAL_RECORD_NUMBER",
    "NATIONAL_ID": "NATIONAL_ID_NUMBER",
    "NPI_NUMBER": "NPI_NUMBER",
    "ORGANIZATION": "ORGANIZATION_NAME",
    "PASSPORT": "PASSPORT_NUMBER",
    "PASSWORD": "PASSWORD",
    "PERSON_NAME": "PERSON_NAME",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PIN": "PIN",
    "ROUTING_NUMBER": "BANK_ROUTING_NUMBER",
    "SALARY": "SALARY",
    "SWIFT_BIC": "SWIFT_BIC_CODE",
    "UK_NI_NUMBER": "NATIONAL_ID_NUMBER",
    "URL_WITH_PII": "URL",
    "USERNAME": "USERNAME",
    "US_SSN": "SOCIAL_SECURITY_NUMBER",
    "VIN": "VEHICLE_IDENTIFICATION_NUMBER",
    "ZIP_CODE": "POSTAL_CODE",
}
```

(Reachable: 42/63 — already best-in-table vs aws 24/63.)

Adapter (mirrors `scrubadub_baseline.py` shape):

```python
class _PiiAnonAdapter:
    name = "pii_anon"
    model_id = "pii-anon regex-oss (vanilla)"
    label_map = LABEL_MAP
    deterministic = True

    def available(self) -> bool:
        return importlib.util.find_spec("pii_anon") is not None

    def map_label(self, native: str) -> str | None:
        return LABEL_MAP.get((native or "").strip().upper())

    def build(self) -> object:
        try:
            from pii_anon.eval_framework.byo_pipeline import first_party_predictor
        except ImportError as e:
            raise RuntimeError("pii-anon not installed — pip install -e ../pii-anon-code") from e
        return first_party_predictor("pii_anon")

    def detect(self, text: str, model: object) -> list[AdapterSpan]:
        if not text:
            return []
        out: list[AdapterSpan] = []
        for native, start, end in model(text):
            et = self.map_label(native)
            if et is not None:
                out.append(AdapterSpan(int(start), int(end), et, text[int(start):int(end)]))
        return out

    def coverage(self) -> int:
        return coverage_of(LABEL_MAP)


ADAPTER = _PiiAnonAdapter()
```

Tests (DATA repo, run from DATA root with `python3 -m pytest tests/test_pii_anon_adapters.py -q`):

```python
def test_label_map_values_are_canonical_or_none():
    from pii_anon_datasets import taxonomy
    from baselines.pii_anon_baseline import LABEL_MAP
    bad = {v for v in LABEL_MAP.values() if v is not None and v not in taxonomy.CANONICAL_ENTITY_TYPES}
    assert not bad, bad

def test_adapter_conforms_to_contract():
    from pii_anon_datasets.baselines.contract import DetectorAdapter
    from baselines.pii_anon_baseline import ADAPTER
    assert isinstance(ADAPTER, DetectorAdapter)

def test_detect_maps_email(monkeypatch-free, requires pii_anon importable):
    model = ADAPTER.build()
    spans = ADAPTER.detect("Contact alice@example.com.", model)
    assert any(s.entity_type == "EMAIL_ADDRESS" for s in spans)
```

- [ ] Steps: failing tests → implement → pass → commit (DATA repo, scoped add)
  `feat(baselines): pii_anon vanilla adapter (native pii-anon -> canonical-63)`

### Task B2: swarm adapter

**Files:** Create `DATA/baselines/pii_anon_swarm_baseline.py` (same shape;
`name = "pii_anon_swarm"`, `model_id = "pii-anon swarm (4-layer fusion)"`,
`deterministic = False` if GLiNER participates; imports the SAME `LABEL_MAP`
from `baselines.pii_anon_baseline` — single source of truth) + tests in the
same new test file.

- [ ] failing test → implement → pass → commit
  `feat(baselines): pii_anon_swarm adapter (shared label map)`

### Task B3: registry + pyproject + docs

**Files:**
- Modify: `DATA/src/pii_anon_datasets/baselines/registry.py` — add:

```python
    "pii_anon": "baselines.pii_anon_baseline",
    "pii_anon_swarm": "baselines.pii_anon_swarm_baseline",
```

- Modify: `DATA/pyproject.toml` — optional group:
  `pii-anon = []  # editable sibling: pip install -e ../pii-anon-code` (comment-documented; no PyPI dep).
- Test: registry resolution test in the new test file
  (`load_adapter("pii_anon").name == "pii_anon"`).

- [ ] failing test → modify → pass → run DATA suite → commit
  `feat(baselines): register pii_anon + pii_anon_swarm detectors`

### Task B4: end-to-end smoke (10 records)

- [ ] Run from DATA root:
  `python3 -m pii_anon_datasets.cli baselines --detectors pii_anon --split dev --languages en --limit 200 --out results/baselines/_smoke-pii-anon`
  (check the actual record-limit flag name in `cli.py` first; if none exists,
  smoke via the orchestrator API in a throwaway script under `/tmp`).
  Expected: scored result with nonzero TP; no DX-02 label errors.
- [ ] Same for `pii_anon_swarm` (slower; 50 records enough).
- [ ] No commit (artifacts only).

---

## Phase C — dev-split baseline + detection iteration (CODE)

### Task C1: dev-split baseline measurement

- [ ] Run both adapters on the FULL en dev split (15,484 records):
  `python3 -m pii_anon_datasets.cli baselines --detectors pii_anon,pii_anon_swarm --split dev --languages en --out results/baselines/sp2-dev-iter0`
- [ ] Record micro P/R/F2 + per-entity table. This is the iteration-0 anchor.

### Task C2: gap analysis

- [ ] Write `CODE/scripts/sp2_gap_analysis.py`: reads a DATA
  `baseline_results.json`, prints per-entity rows ranked by
  **FN mass** (`gold_n × (1 − recall)`) and **FP mass** (`fp` count),
  with cumulative micro-F2 headroom (recompute micro-F2 with that entity's
  FN→TP to show the win if fixed). Pure stdlib, read-only.
- [ ] Commit: `feat(sp2): gap-analysis tool for assessment artifacts`

### Task C3..Cn: iteration loop (repeat until stop criterion)

Protocol per iteration:
1. Pick the top headroom item from C2 output.
2. Classify: (a) span-boundary mismatch (strict-match FN with overlap present —
   compare predicted vs gold extents on sampled records), (b) missing pattern/type,
   (c) FP source.
3. Fix in CODE with TDD (failing unit test with REAL dev-split examples →
   minimal fix → pass → full relevant test module). Fix sites:
   - boundary/hygiene + demotions: `src/pii_anon/engines/regex/confidence.py`
   - patterns/new types: `src/pii_anon/engines/regex/patterns.py`
   - swarm gates/calibration: `src/pii_anon/swarm.py` config + artifacts
4. Guard: `PYTHONPATH=src .venv/bin/python scripts/sp1_detection_delta.py --n 2000 --seed 8314 --compare artifacts/sp1/baseline-n2000.json`
   (no internal-corpus regression), latency budget spot-check.
5. Re-measure dev split (C1 command, `--out results/baselines/sp2-dev-iterN`).
6. Commit per fix: `fix(detect): <entity> <what> (sp2 dev-iter N)`.

Candidate new-type recognizers (priority by dev gold mass, structured first):
TAX_ID (EIN `NN-NNNNNNN` w/ context), VISA_NUMBER, HEALTH_INSURANCE_ID,
PRESCRIPTION_NUMBER, CREDIT_CARD_FRAGMENT (`last four`/`****1234` forms),
DEVICE_IDENTIFIER (IMEI/UUID w/ context), SOCIAL_MEDIA_HANDLE (@handle),
then lexicon/context types: GENDER, MARITAL_STATUS, EDUCATION_LEVEL,
NATIONALITY, JOB_TITLE, HEALTH_CONDITION, MEDICATION_NAME, PROCEDURE_NAME.
All general-purpose (no template-derived literals beyond ordinary lexicons).

**Stop criterion:** swarm dev F2-micro ≥ 0.75 AND vanilla maximized
(report wherever it lands), OR marginal gain per iteration < 0.005 twice in a row.

---

## Phase D — pii-rate-elo assessment ingestion (CODE)

### Task D1: artifact ingestion module

**Files:**
- Create: `src/pii_anon/eval_framework/rating/assessment_ingest.py`
- Test: `tests/test_assessment_ingest.py`

Contract:

```python
@dataclass(frozen=True)
class AssessmentPlayer:
    name: str
    precision: float; recall: float; f1: float; f2: float   # micro
    f2_macro: float
    coverage_reachable: int; coverage_total: int
    per_entity_f2: dict[str, float]    # only entities with gold > 0
    n_gold: int; n_pred: int

def load_assessment(path: str | Path) -> AssessmentReport:  # players + dataset block
```

Validation (no-fabrication discipline): every float routed through a
finite-[0,1] check; names non-blank str; schema field must equal
`pii-anon-baseline-results/v1`; a malformed player → `ValueError` naming the
player and field (fail-loud, never default). Tests: happy path on a small
fixture JSON + one test per rejection class (NaN f2, bool recall, blank name,
wrong schema, f2 > 1).

- [ ] failing tests → implement → pass → commit
  `feat(rating): assessment-artifact ingestion with fail-loud validation (sp2)`

### Task D2: per-entity Elo tournament

**Files:**
- Extend: `assessment_ingest.py` (or sibling `assessment_tournament.py`)
- Test: same test file

```python
def run_assessment_tournament(report: AssessmentReport, engine: PIIRateEloEngine | None = None) -> dict:
    # deterministic: sorted entity types (gold>0), sorted player pairs;
    # match score per entity e: f2_a(e) vs f2_b(e); unreachable type == 0.0 (capability counts);
    # engine.update_from_match(a, b, f2_a, f2_b) per (entity, pair);
    # returns engine.tournament_summary() + per-player per-entity strengths.
```

Tests: (1) dominant synthetic player ranks #1 with rating > 1500;
(2) determinism — two runs byte-identical summaries; (3) exact-anchor —
3-player fixture, pin exact final ratings to 6dp (gold-master, anchored after
first verified run); (4) 12-player smoke on the real merged artifact (skipped
if file absent).

- [ ] failing tests → implement → pass → commit
  `feat(rating): per-entity-type Elo tournament over assessment artifacts (sp2)`

### Task D3: report renderer + CLI

**Files:**
- Extend: `assessment_ingest.py` (renderer) + `src/pii_anon/cli.py` rate-elo command
- Test: `tests/test_assessment_ingest.py` + CLI test

`pii-anon rate-elo --assessment-results <merged.json> --artifact-dir <out>`:
- table 1: leaderboard (Elo rank, rating±RD, F2 micro/macro, P/R, coverage, type [cloud/local/first-party]);
- table 2: pairwise significance matrix (✓ distinguishable / ~ not);
- table 3: per-player top-3 strongest / weakest entity types by F2;
- axis-disclosure block: which axes the artifact carries (detection+coverage),
  which it does not (latency/throughput/Tier-3 — pointed to the internal
  benchmark for first-party systems);
- writes `leaderboard.md` + `tournament.json` (+ stdout md).
Tests: renderer output pinned on the 3-player fixture (exact string anchors for
the header + one row); CLI invocation test with tmp artifact dir.

- [ ] failing tests → implement → pass → ruff/mypy clean → commit
  `feat(cli): rate-elo --assessment-results 12-player report mode (sp2)`

---

## Phase E — final run, merge, report, docs

- [ ] **E1:** Full test-split run (both adapters), from DATA root:
  `python3 -m pii_anon_datasets.cli baselines --detectors pii_anon,pii_anon_swarm --split test --languages en --out results/baselines/sp2-test-first-party`
  (swarm ≈ 45–90 min — run in background, monitor).
- [ ] **E2:** Merge with the stored 10-player artifact
  (`results/baselines/_partial-all-10/baseline_results.json`) using the CLI's
  merge mode (verify exact flag via `python3 -m pii_anon_datasets.cli baselines --help`)
  → `results/baselines/tier1-en-12`.
- [ ] **E3:** `pii-anon rate-elo --assessment-results .../tier1-en-12/baseline_results.json --artifact-dir CODE/artifacts/ratings/sp2-12player` (run via CODE .venv).
- [ ] **E4:** Docs: update `CODE/docs/pii-rate-elo-value.md` (12-player story),
  add the reproducible commands to `CODE/docs/evaluate-your-pipeline.md`,
  CHANGELOG entries both repos.
- [ ] **E5:** Final verification: CODE full suite xdist + ruff + both-mypy;
  DATA suite; md5 checks on the three locked files; honest final summary with
  the real table (wherever we rank).

## Self-review notes

- Spec coverage: Part 1 → Phases A+B; Part 2 → Phase C; Part 3 → Phase D;
  final deliverable → Phase E. Out-of-scope items absent. ✓
- The only intentionally open details: exact `build_fusion` import path (A2
  verifies by grep), CLI record-limit + merge flag names (B4/E2 verify via
  `--help`) — verified at execution, not invented here. ✓
- Type consistency: `Predictor = Callable[[str], Iterable[tuple[str, int, int]]]`
  used consistently across A/B; `AdapterSpan(start, end, entity_type, text)`
  argument order per DATA contract.py. ✓
