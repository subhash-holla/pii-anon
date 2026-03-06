"""Tests for pii_anon.ingestion.writers — CSV, JSON, JSONL, TXT output writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pii_anon.ingestion import FileFormat, write_results


def _make_results(n: int = 3) -> list[dict]:
    """Create sample processing results for writer tests."""
    results = []
    for i in range(n):
        results.append({
            "metadata": {"id": i, "source": "test"},
            "transformed_payload": {"text": f"processed text {i}"},
            "ensemble_findings": [{"entity_type": "EMAIL"} for _ in range(i)],
            "confidence_envelope": {"score": 0.9 - i * 0.1, "risk_level": "low"},
        })
    return results


class TestCSVWriter:
    def test_write_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "output.csv"
        results = _make_results()
        count = write_results(iter(results), path)
        assert count == 3
        assert path.exists()

        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["transformed_text"] == "processed text 0"
        assert rows[0]["entities_found"] == "0"
        assert rows[1]["entities_found"] == "1"

    def test_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "output.csv"
        count = write_results(iter(_make_results(1)), path)
        assert count == 1
        assert path.exists()


class TestJSONWriter:
    def test_write_json(self, tmp_path: Path) -> None:
        path = tmp_path / "output.json"
        count = write_results(iter(_make_results()), path)
        assert count == 3

        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["transformed_text"] == "processed text 0"

    def test_json_empty_results(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        count = write_results(iter([]), path)
        assert count == 0
        assert json.loads(path.read_text()) == []


class TestJSONLWriter:
    def test_write_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "output.jsonl"
        count = write_results(iter(_make_results()), path)
        assert count == 3

        lines = [json.loads(line) for line in path.read_text().strip().split("\n")]
        assert len(lines) == 3
        assert lines[2]["entities_found"] == 2


class TestTXTWriter:
    def test_write_txt(self, tmp_path: Path) -> None:
        path = tmp_path / "output.txt"
        count = write_results(iter(_make_results()), path)
        assert count == 3

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "processed text 0"


class TestExplicitFormat:
    def test_explicit_format_overrides_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "output.dat"
        count = write_results(iter(_make_results(1)), path, fmt=FileFormat.JSONL)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert "transformed_text" in line

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        import pytest

        path = tmp_path / "output.xyz"
        with pytest.raises(ValueError, match="Cannot detect"):
            write_results(iter(_make_results(1)), path)


class TestFlattenResult:
    def test_non_dict_transformed_payload(self, tmp_path: Path) -> None:
        """Handles legacy result shape where transformed_payload is a string."""
        path = tmp_path / "out.jsonl"
        results = [{"transformed_payload": "plain text", "ensemble_findings": [], "confidence_envelope": {}}]
        count = write_results(iter(results), path)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert line["transformed_text"] == "plain text"
