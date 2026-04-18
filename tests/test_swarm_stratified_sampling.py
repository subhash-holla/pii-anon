"""Tests for stratified sampling + training-pool summary.

Covers:

- ``DEFAULT_SWARM_DATASETS`` includes pii-anon's canonical corpus plus
  two industry leaders — the contract the ``make train-swarm`` default
  depends on.
- ``stratified_sample`` returns exactly ``n`` records, preserves
  language share across the sample, honours the minimum-1-per-stratum
  rule, is deterministic under a seed, and is safe for degenerate
  inputs (empty, n=0, n > len(pool)).
- ``summarize_training_pool`` returns sorted per-dataset /
  per-language / per-entity-type counts + label density stats.
- ``load_training_data`` defaults to ``DEFAULT_SWARM_DATASETS`` and
  dispatches file-path entries through ``load_jsonl``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pii_anon.swarm_datasets import (
    DATASET_LOADERS,
    DEFAULT_SWARM_DATASETS,
    TrainingRecord,
    load_training_data,
    register_dataset_loader,
    stratified_sample,
    summarize_training_pool,
)


def _make_pool() -> list[TrainingRecord]:
    """Imbalanced pool: 80% en, 10% es, 10% fr."""
    records: list[TrainingRecord] = []
    for i in range(80):
        records.append(TrainingRecord(
            record_id=f"en-{i}", text=f"en record {i}", labels=[],
            language="en", source_dataset="ds_a",
        ))
    for i in range(10):
        records.append(TrainingRecord(
            record_id=f"es-{i}", text=f"es record {i}", labels=[],
            language="es", source_dataset="ds_a",
        ))
    for i in range(10):
        records.append(TrainingRecord(
            record_id=f"fr-{i}", text=f"fr record {i}", labels=[],
            language="fr", source_dataset="ds_b",
        ))
    return records


def test_default_datasets_includes_pii_anon_eval():
    assert "pii_anon_eval" in DEFAULT_SWARM_DATASETS


def test_default_datasets_has_at_least_three_entries():
    assert len(DEFAULT_SWARM_DATASETS) >= 3


def test_default_datasets_entries_are_all_registered():
    for name in DEFAULT_SWARM_DATASETS:
        assert name in DATASET_LOADERS


def test_stratified_sample_returns_requested_n():
    pool = _make_pool()
    for n in (5, 10, 25, 50, 99):
        out = stratified_sample(pool, n)
        assert len(out) == n


def test_stratified_sample_zero_returns_empty():
    assert stratified_sample(_make_pool(), 0) == []


def test_stratified_sample_empty_input_returns_empty():
    assert stratified_sample([], 10) == []


def test_stratified_sample_n_exceeds_pool_returns_full_pool():
    pool = _make_pool()
    out = stratified_sample(pool, len(pool) + 500)
    assert len(out) == len(pool)
    assert {r.record_id for r in out} == {r.record_id for r in pool}


def test_stratified_sample_negative_n_raises():
    with pytest.raises(ValueError, match="non-negative"):
        stratified_sample(_make_pool(), -1)


def test_stratified_sample_preserves_language_share():
    pool = _make_pool()
    n = 30
    out = stratified_sample(pool, n, strata_keys=("language",))
    langs = [r.language for r in out]
    en_share = langs.count("en") / n
    assert abs(en_share - 0.80) <= 0.10


def test_stratified_sample_protects_small_strata():
    pool = [
        TrainingRecord(record_id=f"en-{i}", text="", labels=[], language="en")
        for i in range(99)
    ] + [
        TrainingRecord(record_id="rare-0", text="", labels=[], language="rare"),
    ]
    out = stratified_sample(pool, 10, strata_keys=("language",))
    assert "rare" in {r.language for r in out}


def test_stratified_sample_is_deterministic_under_seed():
    pool = _make_pool()
    a = stratified_sample(pool, 20, seed=7)
    b = stratified_sample(pool, 20, seed=7)
    assert [r.record_id for r in a] == [r.record_id for r in b]


def test_stratified_sample_different_seed_changes_selection():
    pool = _make_pool()
    a = stratified_sample(pool, 20, seed=1)
    b = stratified_sample(pool, 20, seed=2)
    assert len(a) == len(b) == 20
    assert {r.record_id for r in a} != {r.record_id for r in b}


def test_stratified_sample_multi_key_strata():
    pool = _make_pool()
    out = stratified_sample(pool, 20, strata_keys=("language", "source_dataset"))
    assert any(r.language == "fr" for r in out)
    assert len(out) == 20


def test_summarize_training_pool_structure():
    pool = _make_pool()
    pool[0].labels = [
        {"entity_type": "PERSON_NAME", "start": 0, "end": 5},
        {"entity_type": "EMAIL_ADDRESS", "start": 6, "end": 20},
    ]
    pool[1].labels = [{"entity_type": "PERSON_NAME", "start": 0, "end": 5}]

    summary = summarize_training_pool(pool)
    assert summary["total_records"] == len(pool)
    assert summary["total_labels"] == 3
    assert summary["avg_labels_per_record"] == pytest.approx(3 / len(pool), abs=0.01)
    assert summary["records_without_labels"] == len(pool) - 2
    assert sum(summary["by_dataset"].values()) == len(pool)
    assert sum(summary["by_language"].values()) == len(pool)
    assert list(summary["by_language"].keys())[0] == "en"
    assert summary["by_entity_type"]["PERSON_NAME"] == 2


def test_summarize_training_pool_empty():
    summary = summarize_training_pool([])
    assert summary["total_records"] == 0
    assert summary["total_labels"] == 0
    assert summary["avg_labels_per_record"] == 0.0
    assert summary["by_dataset"] == {}


def test_load_training_data_defaults_to_paper_mix(monkeypatch):
    requested: list[str] = []

    def _make_stub(name):
        def _stub(max_records=None):
            requested.append(name)
            return [TrainingRecord(
                record_id=f"{name}-1", text="x", labels=[],
                language="en", source_dataset=name,
            )]
        return _stub

    monkeypatch.setattr(
        "pii_anon.swarm_datasets.DATASET_LOADERS",
        {name: _make_stub(name) for name in DEFAULT_SWARM_DATASETS},
    )
    out = load_training_data()
    assert set(requested) == set(DEFAULT_SWARM_DATASETS)
    assert len(out) == len(DEFAULT_SWARM_DATASETS)


def test_load_training_data_stratifies_when_capped(monkeypatch):
    pool = _make_pool()
    monkeypatch.setattr(
        "pii_anon.swarm_datasets.DATASET_LOADERS",
        {"test_pool": lambda max_records=None: pool[:max_records] if max_records else pool},
    )
    out = load_training_data(
        datasets=["test_pool"], max_records_per_dataset=30,
        stratify_by=("language",),
    )
    assert len(out) == 30
    langs = {r.language for r in out}
    assert langs == {"en", "es", "fr"}


def test_load_training_data_without_stratify_uses_source_order(monkeypatch):
    pool = _make_pool()
    monkeypatch.setattr(
        "pii_anon.swarm_datasets.DATASET_LOADERS",
        {"test_pool": lambda max_records=None: pool[:max_records] if max_records else pool},
    )
    out = load_training_data(
        datasets=["test_pool"], max_records_per_dataset=30,
        stratify_by=None,
    )
    assert len(out) == 30
    assert all(r.language == "en" for r in out)


def test_load_training_data_reports_empty_datasets(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(
        "pii_anon.swarm_datasets.DATASET_LOADERS",
        {"pii_anon_eval": lambda max_records=None: [
            TrainingRecord(record_id="ok", text="x", labels=[], language="en",
                           source_dataset="pii_anon_eval"),
        ],
         "ai4privacy_400k": lambda max_records=None: [],
         "tab": lambda max_records=None: []},
    )
    with caplog.at_level(logging.WARNING, logger="pii_anon.swarm_datasets"):
        load_training_data(
            datasets=["pii_anon_eval", "ai4privacy_400k", "tab"],
        )
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "ai4privacy_400k" in messages
    assert "pip install" in messages.lower()


def test_load_training_data_jsonl_path_still_works(tmp_path: Path):
    jsonl = tmp_path / "mine.jsonl"
    jsonl.write_text(
        json.dumps({"text": "hello", "annotations": [
            {"entity_type": "PERSON_NAME", "start": 0, "end": 5},
        ]}) + "\n",
        encoding="utf-8",
    )
    out = load_training_data(datasets=[str(jsonl)])
    assert len(out) == 1
    assert out[0].text == "hello"


def test_load_training_data_stratified_cap_applies_to_jsonl(tmp_path: Path):
    jsonl = tmp_path / "mine.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for i in range(20):
            fh.write(json.dumps(
                {"text": f"en {i}", "annotations": [], "language": "en"}
            ) + "\n")
        for i in range(5):
            fh.write(json.dumps(
                {"text": f"es {i}", "annotations": [], "language": "es"}
            ) + "\n")

    out = load_training_data(
        datasets=[str(jsonl)],
        max_records_per_dataset=10,
        stratify_by=("language",),
    )
    assert len(out) == 10
    assert "es" in {r.language for r in out}


def test_register_dataset_loader_participates_in_load_training_data():
    def my_loader(max_records=None):
        return [
            TrainingRecord(record_id=f"mine-{i}", text=f"x {i}", labels=[],
                           language="en", source_dataset="mine")
            for i in range(3)
        ]
    register_dataset_loader("unit_stratified_test", my_loader)
    try:
        out = load_training_data(datasets=["unit_stratified_test"])
        assert len(out) == 3
        assert all(r.source_dataset == "mine" for r in out)
    finally:
        DATASET_LOADERS.pop("unit_stratified_test", None)


# ---------------------------------------------------------------------------
# pii-anon baseline contract — regex-oss must always be in the swarm pool
# ---------------------------------------------------------------------------

def test_pii_anon_baseline_is_in_default_engine_pool():
    """The pii-anon standalone offering (regex-oss) must always be
    available in the swarm's default engine pool.  This is a hard
    contract verified at train-swarm startup — the baseline cannot be
    silently dropped.
    """
    from pii_anon import PIIOrchestrator

    orch = PIIOrchestrator(token_key="contract-test")
    engine_ids = [
        e.adapter_id
        for e in orch._async.registry.list_engines(include_disabled=True)
    ]
    assert "regex-oss" in engine_ids, (
        "regex-oss (the pii-anon baseline engine) is missing from the "
        "default orchestrator — swarm training would miss the baseline"
    )


def test_regex_oss_is_enabled_by_default():
    """regex-oss has no third-party dependency and must ship enabled.

    The swarm's Layer 1 fast-pass and pinning logic both assume this:
    a disabled regex-oss would silently skip the fast-pass entirely.
    """
    from pii_anon import PIIOrchestrator

    orch = PIIOrchestrator(token_key="enabled-test")
    engines = orch._async.registry.list_engines(include_disabled=True)
    regex_oss = next((e for e in engines if e.adapter_id == "regex-oss"), None)
    assert regex_oss is not None
    assert regex_oss.enabled, (
        "regex-oss must be enabled by default — the swarm's fast-pass "
        "and Jaccard-pinning layers depend on it"
    )


def test_pruner_always_pins_regex_oss():
    """The Layer 2 Jaccard pruner must never drop regex-oss, even when
    a higher-type-count engine overlaps with it.  This is the contract
    the baseline relies on to participate in every Dawid-Skene and
    meta-learner decision.
    """
    from pii_anon.swarm import _prune_redundant_findings
    from pii_anon.types import EngineFinding

    # gliner reports a superset of regex-oss's types — without pinning,
    # greedy set-cover would drop regex-oss as "redundant".
    findings = [
        EngineFinding(
            entity_type="EMAIL_ADDRESS", confidence=0.9, field_path="text",
            span_start=0, span_end=5, language="en",
            engine_id="regex-oss", explanation="regex",
        ),
        EngineFinding(
            entity_type="EMAIL_ADDRESS", confidence=0.85, field_path="text",
            span_start=0, span_end=5, language="en",
            engine_id="gliner", explanation="gliner",
        ),
        EngineFinding(
            entity_type="PERSON_NAME", confidence=0.85, field_path="text",
            span_start=10, span_end=15, language="en",
            engine_id="gliner", explanation="gliner",
        ),
    ]
    kept = _prune_redundant_findings(findings, similarity_threshold=0.5)
    kept_ids = {f.engine_id for f in kept}
    assert "regex-oss" in kept_ids, (
        "regex-oss was dropped by the pruner — the pii-anon baseline "
        "must always be pinned"
    )
