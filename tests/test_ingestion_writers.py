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


class TestParquetWriter:
    """Test Parquet output writing."""

    def test_write_parquet_basic(self, tmp_path: Path) -> None:
        """write_results writes parquet format."""
        pytest = __import__("pytest")
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not available")

        path = tmp_path / "output.parquet"
        results = _make_results(2)
        count = write_results(iter(results), path)
        assert count == 2
        assert path.exists()

    def test_write_parquet_empty_results(self, tmp_path: Path) -> None:
        """write_results handles empty results in parquet."""
        pytest = __import__("pytest")
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not available")

        path = tmp_path / "empty.parquet"
        count = write_results(iter([]), path)
        assert count == 0
        assert path.exists()

    def test_write_parquet_missing_pyarrow(self, tmp_path: Path, monkeypatch) -> None:
        """write_results raises ImportError when pyarrow unavailable."""
        pytest = __import__("pytest")
        import sys
        import builtins

        path = tmp_path / "output.parquet"
        results = _make_results(1)

        # Mock missing pyarrow
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("No module named pyarrow")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="pyarrow"):
                write_results(iter(results), path)
        finally:
            builtins.__import__ = real_import


class TestWriteResultsErrorHandling:
    """Test error handling in write_results."""

    def test_write_results_invalid_format_enum(self, tmp_path: Path) -> None:
        """write_results raises ValueError for invalid format."""
        pytest = __import__("pytest")
        from unittest.mock import Mock

        path = tmp_path / "output.csv"
        results = _make_results(1)

        # Create an invalid FileFormat-like object
        invalid_fmt = Mock()
        invalid_fmt.value = "invalid_format"

        # Should raise ValueError because format is not in writer_map
        with pytest.raises(ValueError, match="Unsupported output format"):
            write_results(iter(results), path, fmt=invalid_fmt)


class TestCSVWriterEdgeCases:
    """Test edge cases in CSV writer."""

    def test_csv_empty_results(self, tmp_path: Path) -> None:
        """CSV writer handles empty results."""
        path = tmp_path / "empty.csv"
        count = write_results(iter([]), path)
        assert count == 0
        assert path.exists()
        content = path.read_text()
        # Should be empty or just empty lines
        assert len(content.strip()) == 0

    def test_csv_with_missing_metadata(self, tmp_path: Path) -> None:
        """CSV writer handles missing metadata."""
        path = tmp_path / "output.csv"
        results = [
            {
                "transformed_payload": {"text": "text1"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1

    def test_csv_with_missing_fields(self, tmp_path: Path) -> None:
        """CSV writer handles missing fields gracefully."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {"id": 1},
                "transformed_payload": {"text": "text1"},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1


class TestJSONLWriterEdgeCases:
    """Test edge cases in JSONL writer."""

    def test_jsonl_empty_results(self, tmp_path: Path) -> None:
        """JSONL writer handles empty results."""
        path = tmp_path / "empty.jsonl"
        count = write_results(iter([]), path)
        assert count == 0
        content = path.read_text()
        assert len(content.strip()) == 0

    def test_jsonl_unicode_handling(self, tmp_path: Path) -> None:
        """JSONL writer handles unicode properly."""
        path = tmp_path / "unicode.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "Hello 世界 مرحبا мир"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

        line = json.loads(path.read_text().strip())
        assert "世界" in line["transformed_text"]


class TestTXTWriterEdgeCases:
    """Test edge cases in TXT writer."""

    def test_txt_empty_results(self, tmp_path: Path) -> None:
        """TXT writer handles empty results."""
        path = tmp_path / "empty.txt"
        count = write_results(iter([]), path)
        assert count == 0
        content = path.read_text()
        assert len(content.strip()) == 0

    def test_txt_missing_text_key(self, tmp_path: Path) -> None:
        """TXT writer handles missing text key."""
        path = tmp_path / "output.txt"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"other_key": "value"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1


class TestJSONWriterEdgeCases:
    """Test edge cases in JSON writer."""

    def test_json_unicode_handling(self, tmp_path: Path) -> None:
        """JSON writer handles unicode properly."""
        path = tmp_path / "unicode.json"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "こんにちは"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

        data = json.loads(path.read_text())
        assert "こんにちは" in data[0]["transformed_text"]


class TestWriteResultsCustomTextKey:
    """Test custom text_key parameter."""

    def test_csv_with_custom_text_key(self, tmp_path: Path) -> None:
        """CSV writer respects custom text_key."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "custom output"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, text_key="output_text")
        assert count == 1

        with path.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert "output_text" in rows[0]
        assert rows[0]["output_text"] == "custom output"

    def test_jsonl_with_custom_text_key(self, tmp_path: Path) -> None:
        """JSONL writer respects custom text_key."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "custom"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, text_key="result")
        assert count == 1

        line = json.loads(path.read_text().strip())
        assert line["result"] == "custom"

    def test_txt_with_custom_text_key(self, tmp_path: Path) -> None:
        """TXT writer respects custom text_key."""
        path = tmp_path / "output.txt"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"custom": "custom output text"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, text_key="custom")
        assert count == 1

        # The flattened result will have custom key from transformed_payload
        content = path.read_text().strip()
        assert "custom output text" in content


class TestWriteResultsEncoding:
    """Test encoding parameter."""

    def test_csv_custom_encoding(self, tmp_path: Path) -> None:
        """CSV writer respects custom encoding."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {"text": "data"},
                "transformed_payload": {"text": "output"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, encoding="utf-8")
        assert count == 1
        assert path.exists()

    def test_jsonl_custom_encoding(self, tmp_path: Path) -> None:
        """JSONL writer respects custom encoding."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "output"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, encoding="utf-8")
        assert count == 1
        assert path.exists()

    def test_txt_custom_encoding(self, tmp_path: Path) -> None:
        """TXT writer respects custom encoding."""
        path = tmp_path / "output.txt"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "output"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, encoding="utf-8")
        assert count == 1
        assert path.exists()


class TestFlattenResultEdgeCases:
    """Test flatten_result behavior with various inputs."""

    def test_flatten_missing_transformed_payload(self, tmp_path: Path) -> None:
        """Handles missing transformed_payload."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {"id": 1},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert "transformed_text" in line

    def test_flatten_missing_ensemble_findings(self, tmp_path: Path) -> None:
        """Handles missing ensemble_findings."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "output"},
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert line["entities_found"] == 0

    def test_flatten_non_dict_ensemble_findings(self, tmp_path: Path) -> None:
        """Handles non-dict ensemble_findings."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "output"},
                "ensemble_findings": "not_a_list",
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert line["entities_found"] == 0

    def test_flatten_confidence_envelope_partial(self, tmp_path: Path) -> None:
        """Handles confidence_envelope with only some fields."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "output"},
                "ensemble_findings": [],
                "confidence_envelope": {"score": 0.75},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert line["confidence_score"] == 0.75
        assert line["risk_level"] == "unknown"
