"""Tests for pii_anon.ingestion.readers — CSV, JSON, JSONL, TXT file readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pii_anon.ingestion import FileFormat, IngestConfig, read_file
from pii_anon.ingestion.schema import detect_format


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_csv(self) -> None:
        assert detect_format("data.csv") == FileFormat.CSV

    def test_json(self) -> None:
        assert detect_format("data.json") == FileFormat.JSON

    def test_jsonl(self) -> None:
        assert detect_format("data.jsonl") == FileFormat.JSONL

    def test_ndjson(self) -> None:
        assert detect_format("data.ndjson") == FileFormat.JSONL

    def test_txt(self) -> None:
        assert detect_format("data.txt") == FileFormat.TXT

    def test_text(self) -> None:
        assert detect_format("data.text") == FileFormat.TXT

    def test_parquet(self) -> None:
        assert detect_format("data.parquet") == FileFormat.PARQUET

    def test_xml(self) -> None:
        assert detect_format("data.xml") == FileFormat.XML

    def test_html(self) -> None:
        assert detect_format("data.html") == FileFormat.HTML

    def test_htm(self) -> None:
        assert detect_format("data.htm") == FileFormat.HTML

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_format("data.xyz")


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

class TestCSVReader:
    def test_basic_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text,label\nhello world,greeting\ngoodbye,farewell\n")

        records = list(read_file(csv_file))
        assert len(records) == 2
        assert records[0].record_id == 0
        assert records[0].text == "hello world"
        assert records[0].metadata == {"label": "greeting"}
        assert records[1].text == "goodbye"

    def test_custom_text_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,notes\n1,some notes here\n2,more notes\n")

        config = IngestConfig(text_column="notes")
        records = list(read_file(csv_file, config))
        assert len(records) == 2
        assert records[0].text == "some notes here"
        assert records[0].metadata == {"id": "1"}

    def test_missing_text_column_yields_empty_text(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,notes\n1,data\n")

        config = IngestConfig(text_column="body")  # doesn't exist
        records = list(read_file(csv_file, config))
        assert len(records) == 1
        assert records[0].text == ""

    def test_custom_delimiter(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\tvalue\nhello\t42\n")

        config = IngestConfig(csv_delimiter="\t")
        records = list(read_file(csv_file, config))
        assert len(records) == 1
        assert records[0].text == "hello"

    def test_max_record_chars(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\n" + "a" * 100 + "\n")

        config = IngestConfig(max_record_chars=10)
        records = list(read_file(csv_file, config))
        assert len(records) == 1
        assert len(records[0].text) == 10


# ---------------------------------------------------------------------------
# JSON reader
# ---------------------------------------------------------------------------

class TestJSONReader:
    def test_json_array(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps([
            {"text": "hello", "lang": "en"},
            {"text": "world", "lang": "en"},
        ]))

        records = list(read_file(json_file))
        assert len(records) == 2
        assert records[0].text == "hello"
        assert records[0].metadata == {"lang": "en"}

    def test_json_records_wrapper(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({
            "records": [{"text": "data1"}, {"text": "data2"}]
        }))

        records = list(read_file(json_file))
        assert len(records) == 2

    def test_json_single_object(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps({"text": "singleton"}))

        records = list(read_file(json_file))
        assert len(records) == 1
        assert records[0].text == "singleton"

    def test_json_custom_text_field(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps([{"body": "hello"}]))

        config = IngestConfig(text_column="body")
        records = list(read_file(json_file, config))
        assert records[0].text == "hello"

    def test_json_invalid_root_raises(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text('"just a string"')

        with pytest.raises(ValueError, match="JSON root must be"):
            list(read_file(json_file))

    def test_json_skips_non_dict_items(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps([{"text": "ok"}, "not-a-dict", {"text": "ok2"}]))

        records = list(read_file(json_file))
        assert len(records) == 2

    def test_json_max_record_chars(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps([{"text": "x" * 200}]))

        config = IngestConfig(max_record_chars=50)
        records = list(read_file(json_file, config))
        assert len(records[0].text) == 50


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

class TestJSONLReader:
    def test_basic_jsonl(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            json.dumps({"text": "line1", "id": 1}) + "\n"
            + json.dumps({"text": "line2", "id": 2}) + "\n"
        )

        records = list(read_file(jsonl_file))
        assert len(records) == 2
        assert records[0].text == "line1"
        assert records[1].metadata == {"id": 2}

    def test_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"text":"a"}\n\n\n{"text":"b"}\n')

        records = list(read_file(jsonl_file))
        assert len(records) == 2

    def test_jsonl_skips_non_dict_lines(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"text":"ok"}\n"just-string"\n[1,2]\n{"text":"ok2"}\n')

        records = list(read_file(jsonl_file))
        assert len(records) == 2

    def test_jsonl_max_record_chars(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(json.dumps({"text": "long" * 100}) + "\n")

        config = IngestConfig(max_record_chars=20)
        records = list(read_file(jsonl_file, config))
        assert len(records[0].text) == 20


# ---------------------------------------------------------------------------
# TXT reader
# ---------------------------------------------------------------------------

class TestTXTReader:
    def test_line_per_record(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello world\ngoodbye world\n\nanother line\n")

        records = list(read_file(txt_file))
        assert len(records) == 3  # blank line skipped
        assert records[0].text == "hello world"
        assert records[2].text == "another line"

    def test_whole_file_mode(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("line1\nline2\nline3\n")

        config = IngestConfig(whole_file=True)
        records = list(read_file(txt_file, config))
        assert len(records) == 1
        assert "line1\nline2\nline3" in records[0].text
        assert records[0].metadata == {"source": str(txt_file)}

    def test_txt_max_record_chars(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("a" * 500 + "\n")

        config = IngestConfig(max_record_chars=100)
        records = list(read_file(txt_file, config))
        assert len(records[0].text) == 100

    def test_whole_file_max_record_chars(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("b" * 500)

        config = IngestConfig(whole_file=True, max_record_chars=50)
        records = list(read_file(txt_file, config))
        assert len(records[0].text) == 50


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetect:
    def test_auto_detect_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("text\nhello\n")
        records = list(read_file(p))
        assert records[0].text == "hello"

    def test_auto_detect_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "data.jsonl"
        p.write_text('{"text":"hi"}\n')
        records = list(read_file(p))
        assert records[0].text == "hi"

    def test_explicit_format_overrides(self, tmp_path: Path) -> None:
        # File has .txt extension but force JSON format
        p = tmp_path / "data.txt"
        p.write_text(json.dumps([{"text": "forced"}]))

        config = IngestConfig(format=FileFormat.JSON)
        records = list(read_file(p, config))
        assert records[0].text == "forced"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("text\n")  # headers only
        records = list(read_file(p))
        assert len(records) == 0

    def test_empty_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        records = list(read_file(p))
        assert len(records) == 0

    def test_empty_txt(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.txt"
        p.write_text("")
        records = list(read_file(p))
        assert len(records) == 0

    def test_none_text_in_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "null.csv"
        p.write_text("text,other\n,value\n")
        records = list(read_file(p))
        assert records[0].text == ""
