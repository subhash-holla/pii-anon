"""Tests for the industry-reference dataset loaders.

Covers:

- Taxonomy mappings for ``ai4privacy_400k``, ``tab``, ``meddocan`` hit
  the expected canonical pii-anon entity types.
- ``load_ai4privacy_400k`` / ``load_tab`` / ``load_meddocan`` are
  registered in ``DATASET_LOADERS`` so the training script can dispatch
  them by name.
- Each loader degrades gracefully when the HuggingFace ``datasets``
  dependency is missing — returns ``[]`` with a warning, never raises.
- Each loader processes a representative synthetic row correctly.

The HuggingFace dataset hub is not exercised here (that's a network
integration concern).  We patch ``datasets.load_dataset`` to emit a
small synthetic iterable that matches the real schema, so the parsing
path is validated deterministically.
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from pii_anon.swarm_datasets import (
    DATASET_LOADERS,
    TAXONOMY_MAP,
    load_ai4privacy_400k,
    load_meddocan,
    load_tab,
    map_entity_type,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_all_three_loaders_are_registered():
    """``SWARM_DATASETS=name1,name2`` must resolve to callables."""
    assert DATASET_LOADERS["ai4privacy_400k"] is load_ai4privacy_400k
    assert DATASET_LOADERS["tab"] is load_tab
    assert DATASET_LOADERS["meddocan"] is load_meddocan


# ---------------------------------------------------------------------------
# Taxonomy mappings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,canonical", [
    ("FIRSTNAME", "PERSON_NAME"),
    ("LASTNAME", "PERSON_NAME"),
    ("EMAIL", "EMAIL_ADDRESS"),
    ("PHONE_NUMBER", "PHONE_NUMBER"),
    ("STREETADDRESS", "ADDRESS"),
    ("ZIPCODE", "ADDRESS"),
    ("SSN", "US_SSN"),
    ("SOCIALSECURITYNUMBER", "US_SSN"),
    ("CREDITCARDNUMBER", "CREDIT_CARD"),
    ("BITCOINADDRESS", "CRYPTO_WALLET"),
    ("PASSPORTNUMBER", "PASSPORT"),
    ("VEHICLEVIN", "VIN"),
    ("PASSWORD", "_IGNORE"),
])
def test_ai4privacy_400k_taxonomy(raw, canonical):
    assert map_entity_type("ai4privacy_400k", raw) == canonical


@pytest.mark.parametrize("raw,canonical", [
    ("PERSON", "PERSON_NAME"),
    ("PER", "PERSON_NAME"),
    ("DEM", "PERSON_NAME"),
    ("LOCATION", "LOCATION"),
    ("ORGANIZATION", "ORGANIZATION"),
    ("INSTITUTION", "ORGANIZATION"),
    ("DATETIME", "DATE_OF_BIRTH"),
    ("MISC", "_IGNORE"),
    ("URL", "_IGNORE"),
])
def test_tab_taxonomy(raw, canonical):
    assert map_entity_type("tab", raw) == canonical


@pytest.mark.parametrize("raw,canonical", [
    ("NOMBRE_SUJETO_ASISTENCIA", "PERSON_NAME"),
    ("NOMBRE_PERSONAL_SANITARIO", "PERSON_NAME"),
    ("FAMILIARES_SUJETO_ASISTENCIA", "PERSON_NAME"),
    ("NOMBRE_INSTITUCION", "ORGANIZATION"),
    ("HOSPITAL", "ORGANIZATION"),
    ("CENTRO_SALUD", "ORGANIZATION"),
    ("FECHAS", "DATE_OF_BIRTH"),
    ("NUMERO_TELEFONO", "PHONE_NUMBER"),
    ("CORREO_ELECTRONICO", "EMAIL_ADDRESS"),
    ("DIRECCION", "ADDRESS"),
    ("ID_SUJETO_ASISTENCIA", "MEDICAL_RECORD_NUMBER"),
    ("EDAD_SUJETO_ASISTENCIA", "_IGNORE"),
    ("PROFESION", "_IGNORE"),
])
def test_meddocan_taxonomy(raw, canonical):
    assert map_entity_type("meddocan", raw) == canonical


def test_all_three_taxonomies_registered():
    for name in ("ai4privacy_400k", "tab", "meddocan"):
        assert name in TAXONOMY_MAP, f"{name} missing from TAXONOMY_MAP"
        # Every mapping should have at least one entry that resolves
        # to a canonical pii-anon type (not just _IGNORE).
        mapping = TAXONOMY_MAP[name]
        canonical_targets = {v for v in mapping.values() if v != "_IGNORE"}
        assert "PERSON_NAME" in canonical_targets, f"{name} doesn't map anything to PERSON_NAME"


# ---------------------------------------------------------------------------
# Graceful degradation when HuggingFace datasets is missing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loader", [
    load_ai4privacy_400k,
    load_tab,
    load_meddocan,
])
def test_loaders_return_empty_without_huggingface(loader, monkeypatch, caplog):
    """When the ``datasets`` import fails, the loader logs and returns [].

    The paper-aligned datasets are all opt-in — their absence should
    never break a training run that didn't request them.
    """
    # Drop ``datasets`` from sys.modules and block re-import so the
    # ``try: import datasets`` inside the loader raises ImportError
    # exactly as it would in a stripped-down environment.
    original_datasets = sys.modules.pop("datasets", None)
    try:
        monkeypatch.setattr(
            sys, "modules",
            {k: v for k, v in sys.modules.items() if not k.startswith("datasets")},
        )
        # Substitute a meta_path finder that refuses ``datasets``.
        class _Block:
            def find_spec(self, name, *a, **kw):
                if name == "datasets" or name.startswith("datasets."):
                    raise ImportError("simulated: datasets not installed")
                return None
        monkeypatch.setattr(sys, "meta_path", [_Block(), *sys.meta_path])

        import logging
        with caplog.at_level(logging.WARNING, logger="pii_anon.swarm_datasets"):
            out = loader(max_records=5)
    finally:
        if original_datasets is not None:
            sys.modules["datasets"] = original_datasets

    assert out == []
    # The loader must have logged a friendly warning pointing at the fix.
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "datasets" in messages.lower()


# ---------------------------------------------------------------------------
# Happy-path parsing with a synthetic dataset stub
# ---------------------------------------------------------------------------

class _FakeHFDataset:
    """Minimal stand-in for ``datasets.Dataset`` — iterable of dicts."""
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def _patch_hf_load(rows):
    """Install a fake ``datasets`` module and return the context manager."""
    fake = ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: _FakeHFDataset(rows)
    return patch.dict(sys.modules, {"datasets": fake})


def test_ai4privacy_400k_parses_privacy_mask_schema():
    rows = [
        {
            "source_text": "Hi alice@example.com — my SSN is 123-45-6789.",
            "privacy_mask": [
                {"label": "EMAIL", "start": 3, "end": 20},
                {"label": "SSN", "start": 33, "end": 44},
            ],
            "language": "en",
            "record_id": "fake-1",
        },
    ]
    with _patch_hf_load(rows):
        out = load_ai4privacy_400k()
    assert len(out) == 1
    rec = out[0]
    assert rec.record_id == "fake-1"
    assert rec.source_dataset == "ai4privacy_400k"
    types = {lbl["entity_type"] for lbl in rec.labels}
    assert types == {"EMAIL_ADDRESS", "US_SSN"}


def test_tab_parses_entity_mentions_schema():
    rows = [
        {
            "text": "Judge John Smith ruled on the matter in Strasbourg on 2019-05-01.",
            "entity_mentions": [
                {"entity_type": "PERSON", "start_offset": 6, "end_offset": 16},
                {"entity_type": "LOCATION", "start_offset": 40, "end_offset": 50},
                {"entity_type": "DATETIME", "start_offset": 54, "end_offset": 64},
            ],
            "doc_id": "echr-001",
        },
    ]
    with _patch_hf_load(rows):
        out = load_tab()
    assert len(out) == 1
    rec = out[0]
    assert rec.record_id == "echr-001"
    assert rec.source_dataset == "tab"
    types = {lbl["entity_type"] for lbl in rec.labels}
    assert types == {"PERSON_NAME", "LOCATION", "DATE_OF_BIRTH"}


def test_meddocan_parses_bigbio_kb_schema():
    rows = [
        {
            "id": "meddocan-001",
            "passages": [
                {"text": ["Paciente Juan Pérez, DNI 12345678A."]},
            ],
            "entities": [
                {
                    "type": "NOMBRE_SUJETO_ASISTENCIA",
                    "offsets": [[9, 19]],   # "Juan Pérez"
                },
                {
                    "type": "ID_SUJETO_ASISTENCIA",
                    "offsets": [[25, 34]],   # "12345678A"
                },
            ],
        },
    ]
    with _patch_hf_load(rows):
        out = load_meddocan()
    assert len(out) == 1
    rec = out[0]
    assert rec.record_id == "meddocan-001"
    assert rec.source_dataset == "meddocan"
    assert rec.language == "es"
    types = {lbl["entity_type"] for lbl in rec.labels}
    assert types == {"PERSON_NAME", "MEDICAL_RECORD_NUMBER"}


def test_ai4privacy_400k_drops_ignored_labels():
    rows = [
        {
            "source_text": "Password is hunter2 and email is a@b.com",
            "privacy_mask": [
                {"label": "PASSWORD", "start": 12, "end": 19},
                {"label": "EMAIL", "start": 33, "end": 40},
            ],
        },
    ]
    with _patch_hf_load(rows):
        out = load_ai4privacy_400k()
    types = {lbl["entity_type"] for lbl in out[0].labels}
    assert types == {"EMAIL_ADDRESS"}   # PASSWORD is mapped to _IGNORE


def test_tab_drops_out_of_bounds_spans():
    rows = [
        {
            "text": "Short",
            "entity_mentions": [
                {"entity_type": "PERSON", "start_offset": 0, "end_offset": 100},
            ],
            "doc_id": "bad-1",
        },
    ]
    with _patch_hf_load(rows):
        out = load_tab()
    assert out[0].labels == []


def test_loaders_honour_max_records():
    """Only the first N rows of the fake stream should be consumed."""
    rows = [
        {
            "source_text": f"record {i}",
            "privacy_mask": [],
        }
        for i in range(10)
    ]
    with _patch_hf_load(rows):
        out = load_ai4privacy_400k(max_records=3)
    assert len(out) == 3
