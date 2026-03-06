from __future__ import annotations

import json

import pytest

from pii_anon.eval_framework.datasets import schema as eval_schema


def test_resolve_eval_dataset_path_rejects_invalid_source() -> None:
    with pytest.raises(ValueError):
        eval_schema.resolve_eval_dataset_path(source="invalid")  # type: ignore[arg-type]


def test_classify_context_length_boundaries() -> None:
    assert eval_schema._classify_context_length("a " * 10) == "short"
    assert eval_schema._classify_context_length("a " * 500) == "medium"
    assert eval_schema._classify_context_length("a " * 2000) == "long"
    assert eval_schema._classify_context_length("a " * 12000) == "very_long"


def test_normalize_eval_row_filters_invalid_labels_and_defaults() -> None:
    row = {
        "record_id": "r-1",
        "text": "john@example.com",
        "labels": [
            {"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 16, "mention_variant": "full"},
            {"entity_type": "", "start": 0, "end": 1},
            {"entity_type": "BAD", "start": 50, "end": 60},
        ],
        "source_type": "curated_public",
        "data_type": "not_supported",
        "context_length_tier": "invalid",
        "regulatory_domain": "gdpr",
        "script": "Latin",
    }

    normalized = eval_schema._normalize_eval_row(row, 0)
    assert normalized.record_id == "r-1"
    assert normalized.source_type == "curated_public"
    assert normalized.data_type == "unstructured_text"
    assert normalized.context_length_tier == "short"
    assert normalized.regulatory_domain == ["gdpr"]
    assert len(normalized.labels) == 1
    assert normalized.entity_types_present == ["EMAIL_ADDRESS"]


def test_normalize_eval_row_raises_on_empty_text() -> None:
    with pytest.raises(ValueError):
        eval_schema._normalize_eval_row({"id": "bad", "text": ""}, 0)


def test_normalize_eval_row_handles_non_list_regulatory_domain() -> None:
    normalized = eval_schema._normalize_eval_row(
        {
            "id": "r-2",
            "text": "alpha",
            "labels": [],
            "regulatory_domain": {"unexpected": "shape"},
        },
        0,
    )
    assert normalized.regulatory_domain == []


def test_load_eval_dataset_raises_with_package_only_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_schema, "resolve_eval_dataset_path", lambda *_args, **_kwargs: None)
    with pytest.raises(FileNotFoundError) as exc:
        eval_schema.load_eval_dataset("missing_dataset", source="package-only")
    assert "installed `pii-anon-datasets` package resources" in str(exc.value)


def test_load_eval_dataset_applies_filters(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "mini.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "r1",
                        "text": "alice@example.com",
                        "labels": [{"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17}],
                        "language": "en",
                        "difficulty_level": "easy",
                        "data_type": "unstructured_text",
                        "adversarial_type": "boundary_case",
                    }
                ),
                json.dumps(
                    {
                        "id": "r2",
                        "text": "juan@example.com",
                        "labels": [{"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 16}],
                        "language": "es",
                        "difficulty_level": "hard",
                        "data_type": "structured",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_schema, "resolve_eval_dataset_path", lambda *_args, **_kwargs: path)

    filtered = eval_schema.load_eval_dataset(
        "mini",
        language="en",
        data_type="unstructured_text",
        difficulty="easy",
        adversarial_only=True,
    )
    assert len(filtered) == 1
    assert filtered[0].record_id == "r1"


def test_load_eval_dataset_skips_blank_and_non_matching_rows(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "blanky.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                json.dumps(
                    {
                        "id": "r1",
                        "text": "first@example.com",
                        "labels": [{"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 17}],
                        "data_type": "structured",
                    }
                ),
                json.dumps(
                    {
                        "id": "r2",
                        "text": "second@example.com",
                        "labels": [{"entity_type": "EMAIL_ADDRESS", "start": 0, "end": 18}],
                        "data_type": "unstructured_text",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_schema, "resolve_eval_dataset_path", lambda *_args, **_kwargs: path)

    rows = eval_schema.load_eval_dataset("blanky", data_type="unstructured_text", adversarial_only=True)
    assert rows == []


def test_resolve_eval_dataset_path_uses_env_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dataset_rel = tmp_path / "eval_framework" / "data"
    dataset_rel.mkdir(parents=True)
    dataset_file = dataset_rel / "custom.jsonl"
    dataset_file.write_text('{"id":"r1","text":"x","labels":[]}\n', encoding="utf-8")

    monkeypatch.setenv("PII_ANON_DATASET_ROOT", str(tmp_path))
    monkeypatch.setattr(
        eval_schema.resources,
        "files",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("missing package")),
    )

    resolved = eval_schema.resolve_eval_dataset_path("custom", source="auto")
    assert resolved == dataset_file
