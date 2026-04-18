"""Tests for the swarm extension workflows.

Covers the two user-facing extension paths documented in
docs/extend-swarm.md:

1. **Custom engine** — pinning via ``SwarmConfig.force_include_engines``
   so a user's engine survives Layer 2 Jaccard pruning.
2. **Bring your own data** — ``load_jsonl``, ``register_taxonomy``,
   ``register_dataset_loader``, and the file-path dispatch branch in
   ``load_training_data``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from pii_anon.swarm import (
    SwarmConfig,
    _jaccard_similarity,
    _prune_redundant_findings,
)
from pii_anon.swarm_datasets import (
    DATASET_LOADERS,
    TAXONOMY_MAP,
    TrainingRecord,
    load_jsonl,
    load_training_data,
    map_entity_type,
    register_dataset_loader,
    register_taxonomy,
)
from pii_anon.types import EngineFinding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finding(engine_id: str, entity_type: str) -> EngineFinding:
    return EngineFinding(
        entity_type=entity_type,
        confidence=0.8,
        field_path="text",
        span_start=0,
        span_end=5,
        language="en",
        engine_id=engine_id,
        explanation="test",
    )


# ---------------------------------------------------------------------------
# Workflow 1 — custom engine pinning
# ---------------------------------------------------------------------------

def test_swarm_config_default_force_include_is_empty():
    cfg = SwarmConfig()
    assert cfg.force_include_engines == ()


def test_swarm_config_accepts_force_include():
    cfg = SwarmConfig(force_include_engines=("my-engine", "my-other-engine"))
    assert cfg.force_include_engines == ("my-engine", "my-other-engine")


def test_prune_keeps_regex_oss_implicitly():
    """regex-oss is always pinned — no config flag required."""
    findings = [
        _make_finding("regex-oss", "EMAIL_ADDRESS"),
        _make_finding("gliner", "EMAIL_ADDRESS"),
    ]
    result = _prune_redundant_findings(findings, similarity_threshold=0.5)
    engines_kept = {f.engine_id for f in result}
    assert "regex-oss" in engines_kept


def test_prune_drops_custom_engine_when_not_pinned():
    """A custom engine whose type set duplicates gliner gets pruned."""
    # Give gliner more types so it wins the size ranking and becomes
    # the first non-pinned pick; then the custom engine (identical
    # type set) collides on Jaccard.
    findings = [
        _make_finding("gliner", "PERSON_NAME"),
        _make_finding("gliner", "EMAIL_ADDRESS"),
        _make_finding("my-engine", "PERSON_NAME"),
        _make_finding("my-engine", "EMAIL_ADDRESS"),
    ]
    # Sanity check: type sets are identical → Jaccard = 1.0.
    gliner_types = {"PERSON_NAME", "EMAIL_ADDRESS"}
    my_types = {"PERSON_NAME", "EMAIL_ADDRESS"}
    assert _jaccard_similarity(gliner_types, my_types) == 1.0

    # Without pinning, my-engine is pruned because Jaccard == 1.0.
    result = _prune_redundant_findings(findings, similarity_threshold=0.85)
    assert "my-engine" not in {f.engine_id for f in result}


def test_prune_keeps_pinned_custom_engine_despite_overlap():
    """A pinned custom engine survives the Jaccard check.

    When ``my-engine`` is pinned and has the same type set as ``gliner``,
    ``my-engine`` always survives; ``gliner`` is correctly pruned as
    redundant against the pinned engine (symmetric with the usual
    behaviour — overlap is still overlap).
    """
    findings = [
        _make_finding("gliner", "PERSON_NAME"),
        _make_finding("gliner", "EMAIL_ADDRESS"),
        _make_finding("my-engine", "PERSON_NAME"),
        _make_finding("my-engine", "EMAIL_ADDRESS"),
    ]
    result = _prune_redundant_findings(
        findings,
        similarity_threshold=0.85,
        force_include_engines=("my-engine",),
    )
    engines_kept = {f.engine_id for f in result}
    assert "my-engine" in engines_kept


def test_prune_keeps_pinned_engine_with_distinct_types():
    """When the pinned engine covers a disjoint type set, both survive."""
    findings = [
        _make_finding("gliner", "PERSON_NAME"),
        _make_finding("gliner", "EMAIL_ADDRESS"),
        _make_finding("my-engine", "BIOMETRIC_ID"),   # disjoint from gliner
    ]
    result = _prune_redundant_findings(
        findings,
        similarity_threshold=0.85,
        force_include_engines=("my-engine",),
    )
    engines_kept = {f.engine_id for f in result}
    assert "my-engine" in engines_kept
    assert "gliner" in engines_kept


def test_prune_pinned_engine_bypasses_max_engines_cap():
    """Pinned engines do not consume the max_engines budget."""
    findings = [
        _make_finding("e1", "A"),
        _make_finding("e2", "B"),
        _make_finding("e3", "C"),
        _make_finding("e4", "D"),
        _make_finding("my-engine", "Z"),  # pinned — must survive
    ]
    result = _prune_redundant_findings(
        findings,
        similarity_threshold=0.85,
        max_engines=2,   # very tight cap
        force_include_engines=("my-engine",),
    )
    engines_kept = {f.engine_id for f in result}
    assert "my-engine" in engines_kept


# ---------------------------------------------------------------------------
# Workflow 2 — load_jsonl
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_load_jsonl_canonical_labels(tmp_path: Path):
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, [
        {
            "record_id": "r1",
            "text": "Email me at alice@example.com",
            "annotations": [
                {"entity_type": "EMAIL_ADDRESS", "start": 12, "end": 29},
            ],
            "language": "en",
        },
    ])
    records = load_jsonl(path)
    assert len(records) == 1
    assert records[0].record_id == "r1"
    assert records[0].labels == [
        {"entity_type": "EMAIL_ADDRESS", "start": 12, "end": 29},
    ]
    assert records[0].source_dataset == "custom"


def test_load_jsonl_accepts_labels_alias(tmp_path: Path):
    """v1.0 schema used 'labels' instead of 'annotations'."""
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, [
        {
            "record_id": "r1",
            "text": "Call +1-415-555-0100",
            "labels": [
                {"entity_type": "PHONE_NUMBER", "start": 5, "end": 20},
            ],
        },
    ])
    records = load_jsonl(path)
    assert len(records) == 1
    assert records[0].labels[0]["entity_type"] == "PHONE_NUMBER"


def test_load_jsonl_with_registered_taxonomy(tmp_path: Path):
    """register_taxonomy rewrites raw labels to canonical names."""
    register_taxonomy("clinical_test", {
        "patient": "PERSON_NAME",
        "mrn": "MEDICAL_RECORD_NUMBER",
        "chart_note_header": "_IGNORE",
    })
    path = tmp_path / "clinical.jsonl"
    _write_jsonl(path, [
        {
            "text": "Patient Jane Smith, MRN 12345",
            "annotations": [
                {"entity_type": "patient", "start": 8, "end": 18},
                {"entity_type": "mrn", "start": 24, "end": 29},
                {"entity_type": "chart_note_header", "start": 0, "end": 7},
            ],
        },
    ])
    records = load_jsonl(path, taxonomy_name="clinical_test")
    # The chart_note_header annotation should have been dropped.
    assert len(records[0].labels) == 2
    types = {lbl["entity_type"] for lbl in records[0].labels}
    assert types == {"PERSON_NAME", "MEDICAL_RECORD_NUMBER"}


def test_load_jsonl_drops_malformed_spans(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    _write_jsonl(path, [
        {
            "text": "Short",
            "annotations": [
                {"entity_type": "PERSON_NAME", "start": 100, "end": 200},  # out of bounds
                {"entity_type": "EMAIL_ADDRESS", "start": 3, "end": 1},    # reversed
                {"entity_type": "", "start": 0, "end": 2},                 # empty type
                {"entity_type": "PERSON_NAME", "start": 0, "end": 5},      # valid
            ],
        },
    ])
    records = load_jsonl(path)
    assert len(records[0].labels) == 1
    assert records[0].labels[0]["entity_type"] == "PERSON_NAME"


def test_load_jsonl_skips_empty_text(tmp_path: Path):
    path = tmp_path / "empty_text.jsonl"
    _write_jsonl(path, [
        {"text": "", "annotations": []},
        {"text": "Valid record", "annotations": []},
    ])
    records = load_jsonl(path)
    # The first row was skipped; the second survived.
    assert len(records) == 1
    assert records[0].text == "Valid record"


def test_load_jsonl_skips_malformed_json(tmp_path: Path):
    path = tmp_path / "partial.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write('{"text": "good"}\n')
        fh.write("not json at all\n")
        fh.write('{"text": "also good"}\n')
    records = load_jsonl(path)
    assert len(records) == 2
    assert {r.text for r in records} == {"good", "also good"}


def test_load_jsonl_gzip_auto_decodes(tmp_path: Path):
    path = tmp_path / "sample.jsonl.gz"
    row = {
        "text": "hello",
        "annotations": [{"entity_type": "PERSON_NAME", "start": 0, "end": 5}],
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    records = load_jsonl(path)
    assert len(records) == 1
    assert records[0].text == "hello"


def test_load_jsonl_raises_when_path_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_honours_max_records(tmp_path: Path):
    path = tmp_path / "many.jsonl"
    _write_jsonl(path, [{"text": f"record {i}", "annotations": []} for i in range(10)])
    records = load_jsonl(path, max_records=3)
    assert len(records) == 3


def test_load_jsonl_source_label_overrides_taxonomy(tmp_path: Path):
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, [{"text": "x", "annotations": []}])
    records = load_jsonl(path, taxonomy_name="foo", source_label="my-run")
    assert records[0].source_dataset == "my-run"


# ---------------------------------------------------------------------------
# Workflow 2 — load_training_data path dispatch
# ---------------------------------------------------------------------------

def test_load_training_data_dispatches_path_like_names(tmp_path: Path):
    path = tmp_path / "mine.jsonl"
    _write_jsonl(path, [
        {"text": "hello", "annotations": [
            {"entity_type": "PERSON_NAME", "start": 0, "end": 5},
        ]},
    ])
    records = load_training_data([str(path)])
    assert len(records) == 1
    assert records[0].text == "hello"


def test_load_training_data_mixes_paths_and_names(tmp_path: Path, monkeypatch):
    """A path entry and a registered name in the same call both work."""
    path = tmp_path / "mine.jsonl"
    _write_jsonl(path, [{"text": "x", "annotations": []}])

    # Stub a synthetic named loader so we don't hit the real network.
    def fake_loader(max_records=None):
        return [TrainingRecord(
            record_id="fake-1", text="from named loader",
            labels=[], source_dataset="fake",
        )]

    monkeypatch.setitem(DATASET_LOADERS, "fake_named", fake_loader)
    try:
        records = load_training_data(["fake_named", str(path)])
    finally:
        # monkeypatch.setitem cleans up automatically, but DATASET_LOADERS
        # is module-level so we explicitly forget the stub.
        DATASET_LOADERS.pop("fake_named", None)
    assert len(records) == 2
    assert {r.source_dataset for r in records} == {"fake", "custom"}


def test_load_training_data_warns_on_unknown_name(tmp_path: Path, caplog):
    """An unknown plain name logs a warning with the known datasets."""
    import logging

    with caplog.at_level(logging.WARNING, logger="pii_anon.swarm_datasets"):
        load_training_data(["totally_bogus_name"])
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("totally_bogus_name" in m for m in messages)


# ---------------------------------------------------------------------------
# Workflow 2 — register_taxonomy + register_dataset_loader
# ---------------------------------------------------------------------------

def test_register_taxonomy_adds_entry():
    register_taxonomy("unit_test_taxonomy", {"foo": "PERSON_NAME"})
    try:
        assert map_entity_type("unit_test_taxonomy", "foo") == "PERSON_NAME"
        # Unknown raw label falls through unchanged.
        assert map_entity_type("unit_test_taxonomy", "bar") == "bar"
    finally:
        TAXONOMY_MAP.pop("unit_test_taxonomy", None)


def test_register_taxonomy_copies_input():
    """Mutating the input dict after registration must not alter the stored mapping."""
    source = {"foo": "PERSON_NAME"}
    register_taxonomy("unit_test_copy", source)
    try:
        source["foo"] = "EMAIL_ADDRESS"
        source["bar"] = "LOCATION"
        # Registered copy is isolated from the later mutations.
        assert map_entity_type("unit_test_copy", "foo") == "PERSON_NAME"
        assert map_entity_type("unit_test_copy", "bar") == "bar"
    finally:
        TAXONOMY_MAP.pop("unit_test_copy", None)


def test_register_dataset_loader_adds_entry():
    def my_loader(max_records=None):
        return [TrainingRecord(record_id="x", text="x", labels=[], source_dataset="x")]
    register_dataset_loader("unit_test_loader", my_loader)
    try:
        assert "unit_test_loader" in DATASET_LOADERS
        records = load_training_data(["unit_test_loader"])
        assert len(records) == 1
    finally:
        DATASET_LOADERS.pop("unit_test_loader", None)


def test_register_dataset_loader_rejects_path_like_names():
    """Names that look like paths would collide with the file dispatch."""
    def dummy(max_records=None):
        return []
    with pytest.raises(ValueError, match="looks like a file path"):
        register_dataset_loader("/tmp/oops.jsonl", dummy)
    with pytest.raises(ValueError, match="looks like a file path"):
        register_dataset_loader("mine.jsonl", dummy)


def test_register_dataset_loader_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        register_dataset_loader("x", "not-a-function")  # type: ignore[arg-type]


def test_register_dataset_loader_rejects_duplicate_by_default():
    def dummy(max_records=None):
        return []
    register_dataset_loader("unit_dup", dummy)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_dataset_loader("unit_dup", dummy)
    finally:
        DATASET_LOADERS.pop("unit_dup", None)


def test_register_dataset_loader_replace_overwrites():
    def first(max_records=None):
        return [TrainingRecord(record_id="a", text="a", labels=[], source_dataset="a")]

    def second(max_records=None):
        return [TrainingRecord(record_id="b", text="b", labels=[], source_dataset="b")]

    register_dataset_loader("unit_replace", first)
    try:
        register_dataset_loader("unit_replace", second, replace=True)
        records = load_training_data(["unit_replace"])
        assert records[0].source_dataset == "b"
    finally:
        DATASET_LOADERS.pop("unit_replace", None)
