"""Integration tests for file-based ingestion through the orchestrator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pii_anon.ingestion import IngestConfig
from pii_anon.orchestrator import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


@pytest.fixture()
def orchestrator() -> PIIOrchestrator:
    return PIIOrchestrator(token_key="test-key")


@pytest.fixture()
def default_profile() -> ProcessingProfileSpec:
    return ProcessingProfileSpec(profile_id="test", mode="weighted_consensus", language="en")


@pytest.fixture()
def default_segmentation() -> SegmentationPlan:
    return SegmentationPlan(enabled=False)


class TestRunFileCSV:
    def test_csv_end_to_end(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "input.csv"
        input_path.write_text("text,category\nalice@example.com,email\n+1 415 555 0100,phone\n")
        output_path = tmp_path / "output.csv"

        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
            output_path=output_path,
        )

        assert result.records_processed == 2
        assert result.records_failed == 0
        assert result.total_chars > 0
        assert result.elapsed_seconds >= 0
        assert result.records_per_second >= 0
        assert output_path.exists()

        with output_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2

    def test_csv_custom_text_column(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "input.csv"
        input_path.write_text("id,notes\n1,Contact john@example.com\n2,Clean text\n")
        output_path = tmp_path / "output.jsonl"

        config = IngestConfig(text_column="notes")
        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
            ingest_config=config,
            output_path=output_path,
        )

        assert result.records_processed == 2
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestRunFileJSON:
    def test_json_end_to_end(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "input.json"
        input_path.write_text(json.dumps([
            {"text": "Call me at alice@example.com"},
            {"text": "No PII here"},
        ]))
        output_path = tmp_path / "output.json"

        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
            output_path=output_path,
        )

        assert result.records_processed == 2
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert len(data) == 2


class TestRunFileJSONL:
    def test_jsonl_end_to_end(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(
            '{"text": "Contact alice@example.com"}\n'
            '{"text": "SSN 123-45-6789"}\n'
        )

        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
        )

        assert result.records_processed == 2
        assert result.output_path is None  # no output file requested


class TestRunFileTXT:
    def test_txt_line_by_line(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "input.txt"
        input_path.write_text("alice@example.com\n+1 415 555 0100\nclean text\n")
        output_path = tmp_path / "output.txt"

        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
            output_path=output_path,
        )

        assert result.records_processed == 3
        assert output_path.exists()

    def test_txt_whole_file(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "input.txt"
        input_path.write_text("Line 1 with alice@example.com\nLine 2 with bob@test.com\n")

        config = IngestConfig(whole_file=True)
        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
            ingest_config=config,
        )

        assert result.records_processed == 1


class TestRunFileLargeText:
    def test_large_text_auto_segments(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
    ) -> None:
        """Text exceeding large_text_threshold_tokens triggers auto-segmentation."""
        # Lower the threshold for test purposes
        orchestrator._async.config.stream.large_text_threshold_tokens = 50

        # Create text with >50 tokens
        words = [f"word{i}" for i in range(100)]
        input_path = tmp_path / "large.txt"
        input_path.write_text(" ".join(words))

        config = IngestConfig(whole_file=True)
        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=SegmentationPlan(enabled=False, max_tokens=30, overlap_tokens=5),
            scope="test",
            token_version=1,
            ingest_config=config,
        )

        assert result.records_processed == 1
        assert result.total_chunks > 1  # should have been chunked


class TestRunFileErrors:
    def test_empty_file(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "empty.csv"
        input_path.write_text("text\n")

        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
        )

        assert result.records_processed == 0
        assert result.records_failed == 0

    def test_result_properties(
        self,
        tmp_path: Path,
        orchestrator: PIIOrchestrator,
        default_profile: ProcessingProfileSpec,
        default_segmentation: SegmentationPlan,
    ) -> None:
        input_path = tmp_path / "test.jsonl"
        input_path.write_text('{"text":"hello"}\n')

        result = orchestrator.run_file(
            input_path,
            profile=default_profile,
            segmentation=default_segmentation,
            scope="test",
            token_version=1,
        )

        assert result.input_path == str(input_path)
        assert result.format in ("auto", "jsonl", "FileFormat.JSONL")
        assert result.total_chars == 5
