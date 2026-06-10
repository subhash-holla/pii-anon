# S7-01 — native-format readers: PDF / image-OCR / screenshot / DICOM / audio behind the `Iterator[IngestRecord]` contract

| Field | Value |
|---|---|
| Story | S7-01 |
| Sprint | 7 |
| State | **DONE** (2026-06-09; SO-21. Story gate iter-1 **REQUEST_CHANGES** (2 MAJOR: security-sast unbounded-FlateDecode zip-bomb DoS + code-quality new-module coverage 66/73% < 90%; 1 MINOR capabilities-docstrings) → remediated `258d3ec` → iter-2 **5/5 APPROVE** (0 MAJOR / 1 MINOR doc-freshness remediated at close / 4 carry-forward OBS). NO SDO close (gate md5 `3b842e81…` byte-identical; readers write no control-path artifacts). See §Evidence.) |
| provisional_status | **AGENT_SIMULATED** — the reader Protocol/registry/capability surface, the REAL stdlib PDF-text reader, the format detection, the per-modality recall harness + regression-gate teeth, and the loud optional-dep degradation all run for REAL in-tree against SYNTHETIC fixtures (AX-001). What stays Pass-2: native OCR/DICOM/audio extraction at real strength (`pytesseract`/`Pillow`, `pydicom` are not installed in this env — the readers ship capability-honest with loud errors naming the extra; `# SWITCH-POINT(OCR)`, `# SWITCH-POINT(DICOM)`, `# SWITCH-POINT(AUDIO-ASR)`); span-level source-coordinate mapping beyond page granularity (`# SWITCH-POINT(COORDS)`); the real per-modality benchmark at corpus scale (`# SWITCH-POINT(DATA)`); the full OS-matrix portability run (FR-037/NFR-022 — the benchmark-layer `make benchmark-portable-*` machinery exists; certification is Pass-2). |
| Size | M |
| Implements | **FR-031** (native-format readers emit a uniform `Iterator[IngestRecord]` — PDF/image/screenshot/DICOM/audio; UC-24, swarm, MUST), **FR-032** (round-trip reconstruction preserves non-PII payload byte-for-byte; UC-24, MUST — REAL for text formats via the existing writers, regression-pinned; native readers honestly report `supports_reconstruction=False`, Pass-2), **FR-033** (extraction-fidelity assertion per modality — offsets→source coords; UC-24, SHOULD — representative: page-granular `source_coords` metadata + in-range offset assertion), **FR-034** (per-modality recall benchmark, scored separately; UC-25, MUST — representative in-tree harness; corpus scale is DATA Pass-2), **FR-035** (CI gate on multimodal reader recall regression; UC-25, MUST — a teeth-proven pytest gate). Upholds **FR-036/NFR-023** (stream/batch parity — guard test), **NFR-026** (optional-dep graceful degradation, loud not silent), **AX-001**, **AX-002** (deterministic reads). |
| Traces | Design **DC-14** (`D-implementation-ready-design.md:24` — "Multimodal readers (`Iterator[IngestRecord]`) + per-modality benchmark + stream/batch/offline + OS parity"; body — "Extend `ingestion/` readers to PDF/image-OCR/screenshot/DICOM/audio behind the SAME `Iterator[IngestRecord]` contract + lazy optional-deps; offsets map back to source coordinates (extraction-fidelity assertion); `Payload` stays text-keyed at the core; modality adapters normalize to text+coords"). UC-24/UC-25. Mirrors: `eval_framework/rating/registry.py` (registry + entry-point + NFR-026 pattern), `types.py:EngineCapabilities` (capability shape), `ingestion/readers.py:_read_parquet` (the loud lazy-import precedent naming the extra). |
| Files owned | `src/pii_anon/ingestion/native.py` (**new** — capabilities/Protocol/registry/OCR/DICOM/audio readers), `src/pii_anon/ingestion/native_pdf.py` (**new** — the REAL stdlib PDF-text reader), **additive** `src/pii_anon/ingestion/schema.py` (`FileFormat` += PDF/PNG/JPEG/DICOM/WAV; `detect_format` branches), **additive** `src/pii_anon/ingestion/readers.py` (`read_file` delegates the new formats to the registry), **additive** `src/pii_anon/ingestion/__init__.py` (re-exports), **additive** `pyproject.toml` (extras `ocr`/`dicom`; entry-points `pii_anon.readers`; mypy overrides for the optional modules), `tests/test_native_readers.py` (**new**). |
| Depends on | None hard (S6-04 unrelated). CONSUMES read-only: `ingestion/schema.py` contract, `rating/registry.py` pattern. The orchestrator `capabilities()` surface is user-WIP → the reader capability helper lives in `ingestion/native.py`, NOT on the orchestrator (`# SWITCH-POINT(ORCH)`). |

## 1. Intent
UC-24 promises pii-anon can ingest the document shapes PII actually lives in — PDFs, images/screenshots (OCR), DICOM headers, audio — not just text files. DC-14 fixes the contract: every native reader emits the SAME `Iterator[IngestRecord]` the existing CSV/JSON/JSONL/TXT/Parquet/XML/HTML readers emit, so the downstream pipeline (`Payload` text-keyed at the core) is unchanged. S7-01 ships: (a) the **reader port + registry** (`ReaderCapabilities`, `NativeReader` Protocol, `NativeReaderRegistry` with `pii_anon.readers` entry-point discovery — the rating-registry pattern verbatim); (b) **one REAL extraction implementation** — a pure-stdlib PDF text reader (uncompressed + zlib/FlateDecode content streams, `Tj`/`TJ` text-show harvesting, one record per content stream with page-granular `source_coords`) honest about its limits (`# SWITCH-POINT(PDF-LIB)` to swap in `pypdf` behind the same reader name); (c) **capability-honest lazy readers** for image/screenshot OCR (`pytesseract`+`Pillow`), DICOM header text (`pydicom`), and audio (no ASR backend yet) — each reports `dependency_available` truthfully and raises a LOUD error naming the missing extra on `read()` (the `_read_parquet` precedent; NFR-026's "no silent recall loss" — a reader must never silently yield empty text); (d) the **per-modality recall harness + CI regression gate with proven teeth** (FR-034/FR-035 representative).

## 2. Approach / scope — the carried DESIGN decisions

### (a) Port + registry (`ingestion/native.py`)
* **`ReaderCapabilities`** (dataclass, mirrors `EngineCapabilities`): `{format_name, native_dependency, dependency_available, extracts_text: bool, supports_reconstruction: bool, notes}`.
* **`NativeReader`** (`@runtime_checkable Protocol`): `format_name: str`, `capabilities() -> ReaderCapabilities`, `read(path, config) -> Iterator[IngestRecord]`.
* **`NativeReaderRegistry`** — thread-safe register/get/names(sorted)/`discover_entrypoint_readers(group="pii_anon.readers")`; per-EP try/except (NFR-026); class targets instantiated no-arg, instance targets accepted, non-conformant targets skipped. **`default_reader_registry()`** builds the in-tree five: `pdf`, `image`, `screenshot`, `dicom`, `audio`.
* **`reader_capabilities()`** — pure helper returning the capability rows (the CLI/docs surface; deliberately NOT on the user-WIP orchestrator — `# SWITCH-POINT(ORCH)`).

### (b) The REAL PDF reader (`ingestion/native_pdf.py`, stdlib-only)
* Parses the raw bytes: finds `stream…endstream` segments + their object dicts; applies `zlib.decompress` when `/FlateDecode`; harvests text-show operators — `(literal) Tj` and `[(a)(b)…] TJ` — with a paren-balance + backslash-escape literal-string scanner; each text-bearing content stream becomes ONE `IngestRecord` (machine-generated one-stream-per-page PDFs ⇒ record==page) with `metadata={"modality": "pdf", "source_coords": {"page": n}}` (FR-033 representative; glyph-level coords are `# SWITCH-POINT(COORDS)` Pass-2).
* **Documented limits** (module docstring): no cross-reference-table walk, no encrypted PDFs, no hex strings/CID fonts/encodings beyond Latin-1 literals — enough for simple machine-generated PDFs and the synthetic fixtures; `# SWITCH-POINT(PDF-LIB)`: swap `pypdf` in behind the same `pdf` reader name for full-fidelity extraction.
* Deterministic (AX-002): byte-driven parse, no wall-clock/random.

### (c) Capability-honest lazy readers (in `native.py`)
* **`ImageOcrReader`/`ScreenshotOcrReader`** — lazy `pytesseract`+`PIL.Image` import inside `read()`; absent ⇒ `ImportError` naming `pip install pii-anon[ocr]` (loud, the parquet precedent). `# SWITCH-POINT(OCR)`.
* **`DicomReader`** — lazy `pydicom`; reads header text elements (PatientName-class string values) as text records; absent ⇒ `ImportError` naming `pii-anon[dicom]`. `# SWITCH-POINT(DICOM)`.
* **`AudioReader`** — NO silent empty-text: `read()` raises `NotImplementedError` naming the `# SWITCH-POINT(AUDIO-ASR)` Pass-2 seam (no ASR backend is shipped); capabilities report `extracts_text=False`, `dependency_available=False`.
* All five constructible + capability-reporting with ZERO optional deps installed (constructors never import the heavy dep).

### (d) Format wiring (additive)
* `FileFormat` += `PDF/PNG/JPEG/DICOM/WAV`; `detect_format` += `.pdf`, `.png`, `.jpg/.jpeg`, `.dcm`, `.wav`; `read_file` delegates the new formats to the default registry (`pdf`→pdf, `png/jpeg`→image, `dcm`→dicom, `wav`→audio; `screenshot` is an explicit-choice reader — extension-indistinguishable from `image`).
* Text-format behavior byte-identical (regression-pinned by the existing ingestion suites + A12's parity pin).
* pyproject: extras `ocr = ["pytesseract>=0.3", "Pillow>=10"]`, `dicom = ["pydicom>=2.4"]`; `[project.entry-points."pii_anon.readers"]` (5 entries); mypy `ignore_missing_imports` overrides for `pytesseract`/`PIL`/`pydicom` (the established optional-dep override pattern); editable-metadata refresh for discovery tests.

### (e) FR-034/FR-035 representative harness (tests)
* A per-modality recall scorer in the test module: synthetic gold spans over extracted text, recall = matched/total per modality, scored SEPARATELY per modality (exact integer-count rates).
* The CI regression gate: pinned per-modality recall baselines; **teeth proven** by a deliberately degraded fake reader (drops a page) failing the gate (the S1-03 "teeth" discipline). The corpus-scale benchmark is `# SWITCH-POINT(DATA)`.

## 2a. Pre-claim de-risk
- **RISK-1 (silent recall loss — the NFR-026 headline):** a reader whose backend is missing must FAIL LOUD on `read()`, never yield empty text (A7/A8); capability rows must report `dependency_available` truthfully in this env (A2 exact booleans).
- **RISK-2 (PDF reader correctness):** handcrafted synthetic PDFs (uncompressed + FlateDecode) with exact expected strings/page counts (A3/A4); escape/paren cases pinned; determinism ×5 (A5).
- **RISK-3 (no orchestrator touch):** the capability surface lives in `ingestion/native.py`; `orchestrator.py` byte-identical (`0afc6dee…`).
- **RISK-4 (text-format regressions):** `FileFormat`/`detect_format`/`read_file` edits are additive; the existing ingestion tests + A12 parity pin guard.
- **RISK-5 (off-limits):** SDO gate + `competitor_compare.py` byte-identical; user-WIP never staged; no benchmark script.
- **RISK-6 (heavy deps in CI):** constructors/discovery never import optional deps (A1/A2 pass with zero extras installed).

## 3. Given / When / Then (acceptance)
- **A1 — registry + discovery `[UNIT-TEST]`.** `default_reader_registry().names() == ["audio", "dicom", "image", "pdf", "screenshot"]` (exact sorted list); `discover_entrypoint_readers()` on the installed `pii_anon.readers` group returns the same five; every entry satisfies the `NativeReader` Protocol.
- **A2 — capability honesty in this env `[UNIT-TEST]`.** Exact booleans: pdf `dependency_available=True` (stdlib) + `extracts_text=True`; image/screenshot/dicom `dependency_available=False`; audio `extracts_text=False`. Capability rows constructible with zero optional deps imported.
- **A3 — real PDF read (uncompressed) `[UNIT-TEST]`.** A handcrafted synthetic uncompressed PDF (2 pages) yields exactly 2 `IngestRecord`s with the exact expected strings (AX-001 synthetic values) and `record_id` 0,1.
- **A4 — real PDF read (FlateDecode) `[UNIT-TEST]`.** The zlib-compressed variant of the same content yields the exact same text.
- **A5 — determinism (AX-002) `[UNIT-TEST]`.** 5 consecutive reads of the same PDF yield byte-identical record sequences.
- **A6 — format detection + dispatch `[UNIT-TEST]`.** `detect_format` maps `.pdf/.png/.jpg/.jpeg/.dcm/.wav` to the exact `FileFormat` members; `read_file` on a synthetic PDF dispatches through the registry (same records as A3); text-format detection unchanged (exact pins for `.csv/.txt`).
- **A7 — loud OCR degradation (NFR-026) `[UNIT-TEST]`.** With the extra absent, `ImageOcrReader.read()` raises `ImportError` whose message names `pii-anon[ocr]`; registry construction + discovery still succeed (0 unhandled exceptions at enumeration).
- **A8 — no silent recall loss (audio) `[SECURITY-TEST]`.** `AudioReader.read()` raises (NotImplementedError naming the ASR Pass-2 seam) — it can NEVER yield empty-text records.
- **A9 — extraction fidelity (FR-033 repr.) `[UNIT-TEST]`.** Every PDF record carries `metadata["source_coords"]["page"]` == its 1-based page; `metadata["modality"] == "pdf"`; all label offsets used in A10 are in-range for the extracted text.
- **A10 — per-modality recall scored separately (FR-034 repr.) `[UNIT-TEST]`.** Synthetic gold spans over two modalities (pdf + txt) score exact integer-count recalls (e.g. pdf 2/2, txt 1/2) — separately keyed per modality, never merged.
- **A11 — CI regression-gate teeth (FR-035) `[UNIT-TEST]`.** The pinned per-modality baseline gate PASSES on the real reader and FAILS on a deliberately degraded fake reader that drops a page (recall below baseline) — the gate has teeth.
- **A12 — stream/batch parity guard (FR-036/NFR-023 repr.) `[UNIT-TEST]`.** Iterating `read_file` lazily vs materializing `list(read_file(...))` yields identical records for pdf + txt; the existing text round-trip (writers) byte-identity is re-pinned on a synthetic fixture.
- **A13 — import-boundary audit `[AUDIT]`.** `native.py` + `native_pdf.py` import nothing from `swarm`/`moe`/`fusion`/`orchestrator`; off-limits md5s byte-identical.
- **A14–A17 — gate-remediation anchors (added at iter-1 REQUEST_CHANGES, `258d3ec`).** A14 `[SECURITY-TEST]` bounded-FlateDecode zip-bomb skip (security-sast MAJOR-1: a stream inflating past the 64 MiB per-stream ceiling is treated as undecodable + skipped; honest pages unchanged; ≤-ceiling boundary exact); A15 literal-string escapes (octal/named/escaped-paren/CRLF-continuation) + nested parens + `'`/`"` show ops + orphan-string drop; A16 corrupt-zlib skip + drawing-only-stream no-record + `max_record_chars` exact truncation; A16b truncated-escape guard + multi-MiB within-ceiling chunked inflate + missing-endstream skip; A17 registry fail-closed TypeError + round-trip (code-quality MAJOR-2 coverage floor: `native.py` 91% / `native_pdf.py` 92%).

## 5. Notes / non-goals
- **Non-goal:** native OCR/DICOM/audio extraction at real strength — the env has no `pytesseract`/`Pillow`/`pydicom`; the readers ship capability-honest with loud errors + extras; real extraction validation is Pass-2 (`# SWITCH-POINT(OCR)/(DICOM)/(AUDIO-ASR)`).
- **Non-goal:** glyph/span-level source-coordinate mapping (FR-033 full) — page-granular `source_coords` ships now; `# SWITCH-POINT(COORDS)` Pass-2.
- **Non-goal:** native-format round-trip reconstruction (FR-032 for PDF/image/DICOM/audio) — `supports_reconstruction=False` reported honestly; text-format round-trip stays REAL + pinned. Pass-2.
- **Non-goal:** the corpus-scale per-modality benchmark + the full OS-matrix portability certification (FR-034 full / FR-037 / NFR-022) — `# SWITCH-POINT(DATA)`; the benchmark-layer machinery (`make benchmark-portable-*`) already exists.
- **Non-goal:** orchestrator wiring of reader capabilities (user-WIP `orchestrator.py`) — `# SWITCH-POINT(ORCH)`.

## 9. Test-type tags + reviewer set
`[UNIT-TEST]` `[SECURITY-TEST]` `[AUDIT]`. **Reviewers (canonical 5-gate story set):** code-quality + traceability (DC-14 → FR-031..035 + UC-24/25) + requirements-coverage (the MUST splits: FR-031 real, FR-032 text-real/native-Pass-2, FR-034/035 representative — all tracked) + axiom-compliance (AX-001/002, NFR-026 loud-not-silent, no-orchestrator-touch) + security-sast ([AUDIT] tag; the no-silent-recall-loss invariant + the lazy-import surface). **No SDO adversarial close** (no `competitive_supremacy.py` change; no control-path artifacts).

## 12. Definition of Done
- [ ] **RED**: `tests/test_native_readers.py` (A1–A13) first & failing (`ModuleNotFoundError` on `ingestion.native`). RED precedes GREEN.
- [ ] **GREEN**: `native.py` + `native_pdf.py` + additive schema/readers/__init__/pyproject + editable-metadata refresh — all A1–A17 green (A14–A17 = gate-remediation anchors, see §3).
- [ ] **Quality gate**: full xdist suite green; ruff clean; mypy clean (both modes); coverage ≥84% (new modules ≥90%).
- [ ] **Untouched / off-limits**: `orchestrator.py` (`0afc6dee…`) + `competitive_supremacy.py` (`3b842e81…`) + `competitor_compare.py` (`7cae16c8…`) byte-identical; user-WIP never staged; narrow `git add`.
- [ ] **Story-gate APPROVE** — `_reviews/story/S7-01/`, all 5 APPROVE; MAJORs remediated in-loop.
- [ ] **SDO verdict UNCHANGED** — a reader feature flips no guarantee.

## Evidence (filled on completion)

**Commits (RED→GREEN→remediation, on `pdlc/sota-program`):** RED `48c189f` (tests-only; `ModuleNotFoundError` on `ingestion.native`) → GREEN `169ab07` (native.py + native_pdf.py + schema/readers/__init__/pyproject wiring + editable refresh; the `(?<!end)stream` lookbehind fixed a phantom-stream duplicate-page bug caught at GREEN) → remediation `258d3ec` (the iter-1 gate findings).

**Files:** `src/pii_anon/ingestion/native.py` (new — `ReaderCapabilities`, `NativeReader` Protocol, `NativeReaderRegistry` + `pii_anon.readers` discovery, Image/Screenshot/Dicom/Audio readers), `src/pii_anon/ingestion/native_pdf.py` (new — the stdlib PDF reader with the bounded chunked FlateDecode inflate), additive `ingestion/schema.py` + `ingestion/readers.py` + `ingestion/__init__.py`, additive `pyproject.toml` (extras `ocr`/`dicom`, `pii_anon.readers` entry-points, mypy overrides), `tests/test_native_readers.py` (A1–A17, 18 cases).

**Story gate (iter-1 REQUEST_CHANGES → iter-2 5/5 APPROVE; `_reviews/story/S7-01/`; runs `wf_44f8b3ae-51b` → `wf_2d10a313-630`):** iter-1: 2 MAJOR (security-sast **unbounded-FlateDecode zip-bomb memory-DoS** — ~1000x amplification on untrusted PDF bytes; code-quality **new-module coverage 66%/73% < the 90% floor**) + 1 MINOR (capabilities docstrings) + 8 OBS. Remediated `258d3ec`: the 64 MiB per-stream chunked-inflate ceiling (over-ceiling ⇒ `zlib.error` ⇒ the existing skip path; ≤-boundary exact; A14 pins bomb-skipped-honest-page-extracted), targeted edge tests + honest pragmas (`native.py` 91% / `native_pdf.py` 92%), 4 docstrings. iter-2: 0 MAJOR / 1 MINOR (story §3/§12 anchor list lagged the test file — remediated at close: A14–A17 documented) / 4 carry-forward OBS (epic-gate pip-audit of the ocr/dicom extras incl. Pillow CVE-stream; FR-037/NFR-022 SHOULD deferral in the sprint snapshot; FR-032-native + FR-034-corpus stay open MUST-board rows flagged representative-this-sprint; AX-001/002 resolve to project-axioms — convention note). The workflow scribe wrote synthesis.md; per-reviewer YAMLs transcribed from the structured results.

**SDO — UNCHANGED:** readers write no control-path artifacts; `competitive_supremacy.py` (`3b842e81…`) + `competitor_compare.py` (`7cae16c8…`) + `orchestrator.py` (`0afc6dee…`) byte-identical (verified at iter-2 close).

**Quality:** owned tests 18/18; ruff clean; mypy clean BOTH modes (143 files); full xdist suite green at the final state (see SO-21 for counts + coverage).

**DoD:** all checkboxes met (A1–A17). Pass-2 (tracked): real OCR/DICOM extraction strength (`# SWITCH-POINT(OCR)/(DICOM)`); the ASR backend (`# SWITCH-POINT(AUDIO-ASR)`); span-level coords (`# SWITCH-POINT(COORDS)`); pypdf full-fidelity swap (`# SWITCH-POINT(PDF-LIB)`); corpus-scale per-modality benchmark (`# SWITCH-POINT(DATA)`); FR-037/NFR-022 OS-matrix certification; the orchestrator capability surface (`# SWITCH-POINT(ORCH)`).
