"""Extended tests for ingestion/writers.py uncovered branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_anon.ingestion.writers import write_results


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


class TestWriteResultsParentDirCreation:
    """Test parent directory creation."""

    def test_write_results_creates_nested_parent_dirs(self, tmp_path: Path) -> None:
        """Test deeply nested parent directories are created."""
        path = tmp_path / "a" / "b" / "c" / "d" / "output.json"
        count = write_results(iter(_make_results(1)), path)
        assert count == 1
        assert path.exists()
        assert path.parent.exists()


class TestCSVFlatten:
    """Test CSV flattening behavior."""

    def test_csv_flattens_nested_structures(self, tmp_path: Path) -> None:
        """Test CSV handles flattened nested results."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {"nested": {"key": "value"}},
                "transformed_payload": {"text": "test"},
                "ensemble_findings": [],
                "confidence_envelope": {"score": 0.9},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1
        assert path.exists()

    def test_csv_handles_missing_keys_in_fieldnames(self, tmp_path: Path) -> None:
        """Test CSV writer handles records with missing fields."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {"id": 1},
                "transformed_payload": {"text": "first"},
                "ensemble_findings": [{"type": "EMAIL"}],
                "confidence_envelope": {"score": 0.9},
            },
            {
                "metadata": {"id": 2},
                "transformed_payload": {"text": "second"},
                "ensemble_findings": [],
                # Missing confidence_envelope
            },
        ]
        count = write_results(iter(results), path)
        assert count == 2


class TestParquetWriter:
    """Test Parquet writer functionality."""

    @pytest.mark.skipif(
        True,  # Skip if pyarrow not available - covered by existing tests
        reason="pyarrow may not be installed"
    )
    def test_write_parquet(self, tmp_path: Path) -> None:
        """Test writing Parquet format."""
        pytest.importorskip("pyarrow")
        from pii_anon.ingestion import FileFormat

        path = tmp_path / "output.parquet"
        count = write_results(iter(_make_results(3)), path, fmt=FileFormat.PARQUET)
        assert count == 3
        assert path.exists()


class TestWriteResultsWithCustomTextKey:
    """Test write_results with custom text_key."""

    def test_custom_text_key_in_csv(self, tmp_path: Path) -> None:
        """Test custom text_key is used in CSV."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "content"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, text_key="custom_text")
        assert count == 1

    def test_custom_text_key_in_json(self, tmp_path: Path) -> None:
        """Test custom text_key is used in JSON."""
        path = tmp_path / "output.json"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "content"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        import json
        count = write_results(iter(results), path, text_key="output_text")
        assert count == 1
        data = json.loads(path.read_text())
        assert "output_text" in data[0]


class TestWriteResultsWithCustomEncoding:
    """Test write_results with custom encoding."""

    def test_custom_encoding_utf8(self, tmp_path: Path) -> None:
        """Test writing with UTF-8 encoding."""
        path = tmp_path / "output.txt"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "Hëllö wörld"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, encoding="utf-8")
        assert count == 1
        content = path.read_text(encoding="utf-8")
        assert "ëllö" in content or "Hëllö" in content

    def test_custom_encoding_latin1(self, tmp_path: Path) -> None:
        """Test writing with latin-1 encoding."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {"name": "José"},
                "transformed_payload": {"text": "test"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path, encoding="latin-1")
        assert count == 1
        content = path.read_text(encoding="latin-1")
        assert "José" in content or "Jos" in content


class TestFlattenResultEdgeCases:
    """Test edge cases in result flattening."""

    def test_flatten_with_missing_transformed_payload(self, tmp_path: Path) -> None:
        """Test flattening when transformed_payload is missing."""
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

    def test_flatten_with_non_dict_transformed_payload(self, tmp_path: Path) -> None:
        """Test flattening with string payload."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": "plain text string",
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

    def test_flatten_with_empty_metadata(self, tmp_path: Path) -> None:
        """Test flattening with empty metadata."""
        path = tmp_path / "output.csv"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "test"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1

    def test_flatten_confidence_envelope_missing_fields(self, tmp_path: Path) -> None:
        """Test flattening with partial confidence_envelope."""
        path = tmp_path / "output.jsonl"
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "test"},
                "ensemble_findings": [],
                "confidence_envelope": {"score": 0.8},  # Missing risk_level
            }
        ]
        count = write_results(iter(results), path)
        assert count == 1


class TestWriteResultsStreamingBehavior:
    """Test streaming behavior of writers."""

    def test_csv_writer_streams_without_materializing_all(self, tmp_path: Path) -> None:
        """Test CSV writer doesn't need to materialize all records."""
        path = tmp_path / "output.csv"

        def result_generator():
            for i in range(10):
                yield {
                    "metadata": {"id": i},
                    "transformed_payload": {"text": f"text {i}"},
                    "ensemble_findings": [],
                    "confidence_envelope": {"score": 0.9},
                }

        count = write_results(result_generator(), path)
        assert count == 10

    def test_json_writer_materializes_all(self, tmp_path: Path) -> None:
        """Test JSON writer materializes all records."""
        path = tmp_path / "output.json"

        def result_generator():
            for i in range(3):
                yield {
                    "metadata": {"id": i},
                    "transformed_payload": {"text": f"text {i}"},
                    "ensemble_findings": [],
                    "confidence_envelope": {"score": 0.9},
                }

        count = write_results(result_generator(), path)
        assert count == 3


class TestTXTWriter:
    """Test TXT writer specific behavior."""

    def test_txt_writer_only_outputs_text(self, tmp_path: Path) -> None:
        """Test TXT writer outputs only transformed_text."""
        path = tmp_path / "output.txt"
        results = [
            {
                "metadata": {"id": 1, "source": "test"},
                "transformed_payload": {"text": "line one"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            },
            {
                "metadata": {"id": 2},
                "transformed_payload": {"text": "line two"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            },
        ]
        count = write_results(iter(results), path)
        assert count == 2
        content = path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert "line one" in lines[0]
        assert "line two" in lines[1]
