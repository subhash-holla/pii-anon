"""Tests for data format readers and format detection.

Tests cover:
- XML file reading (whole_file and per-element modes)
- HTML file reading (whole_file and per-text-chunk modes)
- Format auto-detection from file extensions
- DataFrame reading with duck-typed objects
- IngestConfig with FileFormat enum
"""

import pytest

from pii_anon.ingestion import (
    FileFormat,
    IngestConfig,
    IngestRecord,
    detect_format,
    read_file,
)
from pii_anon.ingestion.dataframe import read_dataframe


class TestXMLReader:
    """Test XML file reading."""

    def test_read_xml_whole_file(self, tmp_path):
        """XML reader in whole_file mode concatenates all text."""
        xml_content = """<?xml version="1.0"?>
<root>
    <person>
        <name>John Doe</name>
        <email>john@example.com</email>
    </person>
    <person>
        <name>Jane Smith</name>
        <email>jane@example.com</email>
    </person>
</root>
"""
        xml_file = tmp_path / "data.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(xml_file), config))

        assert len(records) == 1
        record = records[0]
        assert isinstance(record, IngestRecord)
        assert "John Doe" in record.text
        assert "jane@example.com" in record.text
        assert record.metadata["format"] == "xml"

    def test_read_xml_per_element(self, tmp_path):
        """XML reader in per-element mode creates one record per element."""
        xml_content = """<?xml version="1.0"?>
<root>
    <name>John Doe</name>
    <email>john@example.com</email>
    <name>Jane Smith</name>
</root>
"""
        xml_file = tmp_path / "data.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=False)
        records = list(read_file(str(xml_file), config))

        assert len(records) > 0
        # Verify that each record has text content
        for record in records:
            assert isinstance(record.text, str)
            assert len(record.text) > 0

    def test_read_xml_empty_elements_skipped(self, tmp_path):
        """XML reader skips elements with no text."""
        xml_content = """<?xml version="1.0"?>
<root>
    <empty></empty>
    <text>Content here</text>
    <blank>  </blank>
</root>
"""
        xml_file = tmp_path / "data.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=False)
        records = list(read_file(str(xml_file), config))

        # Should skip empty and whitespace-only elements
        text_contents = [r.text for r in records]
        assert "Content here" in text_contents

    def test_read_xml_with_attributes(self, tmp_path):
        """XML reader includes element attributes in metadata."""
        xml_content = """<?xml version="1.0"?>
<root>
    <item id="123" type="person">John</item>
</root>
"""
        xml_file = tmp_path / "data.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=False)
        records = list(read_file(str(xml_file), config))

        assert len(records) > 0
        # Verify metadata contains tag and attributes
        assert any("tag" in r.metadata for r in records)

    def test_read_xml_max_chars(self, tmp_path):
        """XML reader respects max_record_chars limit."""
        xml_content = """<?xml version="1.0"?>
<root>
    <text>This is a very long text that should be truncated when max_record_chars is set</text>
</root>
"""
        xml_file = tmp_path / "data.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=True, max_record_chars=20)
        records = list(read_file(str(xml_file), config))

        assert len(records) == 1
        assert len(records[0].text) <= 20


class TestHTMLReader:
    """Test HTML file reading."""

    def test_read_html_whole_file(self, tmp_path):
        """HTML reader in whole_file mode concatenates all text."""
        html_content = """
<html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Hello World</h1>
        <p>This is a paragraph.</p>
        <p>Another paragraph here.</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        record = records[0]
        assert "Hello World" in record.text
        assert "paragraph" in record.text
        assert record.metadata["format"] == "html"

    def test_read_html_strips_script_tags(self, tmp_path):
        """HTML reader excludes script tag content."""
        html_content = """
<html>
    <body>
        <p>Visible text</p>
        <script>var secret = 'not visible';</script>
        <p>More visible</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        assert "Visible text" in records[0].text
        assert "More visible" in records[0].text
        # Script content should be stripped
        assert "var secret" not in records[0].text or "secret" not in records[0].text

    def test_read_html_strips_style_tags(self, tmp_path):
        """HTML reader excludes style tag content."""
        html_content = """
<html>
    <head>
        <style>body { color: red; }</style>
    </head>
    <body>
        <p>Main content</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        assert "Main content" in records[0].text
        # CSS should not appear in extracted text
        assert "color: red" not in records[0].text

    def test_read_html_per_text_chunk(self, tmp_path):
        """HTML reader in per-text-chunk mode creates records for text chunks."""
        html_content = """
<html>
    <body>
        <p>Paragraph one</p>
        <p>Paragraph two</p>
        <p>Paragraph three</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=False)
        records = list(read_file(str(html_file), config))

        # Should create multiple records for separate text chunks
        assert len(records) > 0
        texts = [r.text for r in records]
        assert any("Paragraph" in t for t in texts)

    def test_read_html_entity_decoding(self, tmp_path):
        """HTML reader handles HTML entities."""
        html_content = """
<html>
    <body>
        <p>Email: john@example.com &amp; jane@example.com</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        # Should contain the email addresses
        assert "example.com" in records[0].text

    def test_read_html_max_chars(self, tmp_path):
        """HTML reader respects max_record_chars limit."""
        html_content = """
<html>
    <body>
        <p>This is a very long text content that should be truncated</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True, max_record_chars=30)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        assert len(records[0].text) <= 30


class TestDetectFormat:
    """Test file format auto-detection."""

    def test_detect_parquet_extension(self):
        """detect_format recognizes .parquet files."""
        fmt = detect_format("data.parquet")
        assert fmt == FileFormat.PARQUET

    def test_detect_parquet_pq_extension(self):
        """detect_format recognizes .pq files."""
        fmt = detect_format("data.pq")
        assert fmt == FileFormat.PARQUET

    def test_detect_xml_extension(self):
        """detect_format recognizes .xml files."""
        fmt = detect_format("data.xml")
        assert fmt == FileFormat.XML

    def test_detect_html_extension(self):
        """detect_format recognizes .html files."""
        fmt = detect_format("page.html")
        assert fmt == FileFormat.HTML

    def test_detect_htm_extension(self):
        """detect_format recognizes .htm files."""
        fmt = detect_format("page.htm")
        assert fmt == FileFormat.HTML

    def test_detect_csv_extension(self):
        """detect_format recognizes .csv files."""
        fmt = detect_format("data.csv")
        assert fmt == FileFormat.CSV

    def test_detect_json_extension(self):
        """detect_format recognizes .json files."""
        fmt = detect_format("data.json")
        assert fmt == FileFormat.JSON

    def test_detect_jsonl_extension(self):
        """detect_format recognizes .jsonl files."""
        fmt = detect_format("data.jsonl")
        assert fmt == FileFormat.JSONL

    def test_detect_txt_extension(self):
        """detect_format recognizes .txt files."""
        fmt = detect_format("data.txt")
        assert fmt == FileFormat.TXT

    def test_detect_case_insensitive(self):
        """detect_format is case-insensitive."""
        fmt1 = detect_format("DATA.XML")
        fmt2 = detect_format("data.xml")
        assert fmt1 == fmt2 == FileFormat.XML

    def test_detect_unknown_extension(self):
        """detect_format raises for unknown extensions."""
        with pytest.raises(ValueError, match="Cannot detect file format"):
            detect_format("data.unknown")


class TestDataFrameReader:
    """Test DataFrame reading with duck-typed objects."""

    class MockDataFrame:
        """Mock pandas-like DataFrame with iterrows() method."""

        def __init__(self, rows):
            """Initialize with list of row dicts."""
            self.rows = rows
            if rows:
                self.columns = list(rows[0].keys())
            else:
                self.columns = []

        def iterrows(self):
            """Yield (index, row_dict) tuples."""
            for idx, row in enumerate(self.rows):
                yield idx, row

    def test_read_dataframe_basic(self):
        """read_dataframe processes rows with iterrows()."""
        df = self.MockDataFrame([{"text": "Hello"}, {"text": "World"}])
        records = list(read_dataframe(df))

        assert len(records) == 2
        assert records[0].text == "Hello"
        assert records[1].text == "World"

    def test_read_dataframe_text_extraction(self):
        """read_dataframe extracts text from specified column."""
        df = self.MockDataFrame([{"message": "Test"}])
        records = list(read_dataframe(df, text_column="message"))

        assert records[0].text == "Test"

    def test_read_dataframe_metadata_extraction(self):
        """read_dataframe preserves non-text columns as metadata."""
        df = self.MockDataFrame([{"text": "Hello", "user_id": "123"}])
        records = list(read_dataframe(df))

        assert records[0].text == "Hello"
        assert records[0].metadata["user_id"] == "123"

    def test_read_dataframe_missing_column_raises(self):
        """read_dataframe raises ValueError for missing text column."""
        df = self.MockDataFrame([{"other_column": "value"}])

        with pytest.raises(ValueError, match="Text column 'text' not found"):
            list(read_dataframe(df))

    def test_read_dataframe_custom_missing_column_raises(self):
        """read_dataframe raises ValueError for missing custom column."""
        df = self.MockDataFrame([{"text": "value"}])

        with pytest.raises(ValueError, match="Text column 'message' not found"):
            list(read_dataframe(df, text_column="message"))

    def test_read_dataframe_max_chars(self):
        """read_dataframe respects max_record_chars limit."""
        df = self.MockDataFrame([{"text": "This is a very long text"}])
        records = list(read_dataframe(df, max_record_chars=10))

        assert len(records[0].text) <= 10

    def test_read_dataframe_record_ids(self):
        """read_dataframe assigns sequential record IDs."""
        df = self.MockDataFrame([{"text": "A"}, {"text": "B"}, {"text": "C"}])
        records = list(read_dataframe(df))

        assert records[0].record_id == 0
        assert records[1].record_id == 1
        assert records[2].record_id == 2

    def test_read_dataframe_none_text(self):
        """read_dataframe handles None values in text column."""
        df = self.MockDataFrame([{"text": None}, {"text": "Hello"}])
        records = list(read_dataframe(df))

        assert records[0].text == ""
        assert records[1].text == "Hello"


class TestIngestConfig:
    """Test IngestConfig dataclass."""

    def test_ingest_config_defaults(self):
        """IngestConfig initializes with sensible defaults."""
        config = IngestConfig()

        assert config.format is None
        assert config.text_column == "text"
        assert config.encoding == "utf-8"
        assert config.csv_delimiter == ","
        assert config.whole_file is False
        assert config.max_record_chars == 0

    def test_ingest_config_file_format_enum(self):
        """IngestConfig accepts FileFormat enum values."""
        config = IngestConfig(format=FileFormat.XML)
        assert config.format == FileFormat.XML

    def test_ingest_config_custom_text_column(self):
        """IngestConfig accepts custom text_column."""
        config = IngestConfig(text_column="message")
        assert config.text_column == "message"

    def test_ingest_config_custom_encoding(self):
        """IngestConfig accepts custom encoding."""
        config = IngestConfig(encoding="latin-1")
        assert config.encoding == "latin-1"

    def test_ingest_config_whole_file_flag(self):
        """IngestConfig accepts whole_file flag."""
        config = IngestConfig(whole_file=True)
        assert config.whole_file is True

    def test_ingest_config_max_record_chars(self):
        """IngestConfig accepts max_record_chars."""
        config = IngestConfig(max_record_chars=1000)
        assert config.max_record_chars == 1000


class TestFileFormat:
    """Test FileFormat enum."""

    def test_file_format_csv(self):
        """FileFormat has CSV value."""
        assert FileFormat.CSV == "csv"

    def test_file_format_json(self):
        """FileFormat has JSON value."""
        assert FileFormat.JSON == "json"

    def test_file_format_jsonl(self):
        """FileFormat has JSONL value."""
        assert FileFormat.JSONL == "jsonl"

    def test_file_format_txt(self):
        """FileFormat has TXT value."""
        assert FileFormat.TXT == "txt"

    def test_file_format_parquet(self):
        """FileFormat has PARQUET value."""
        assert FileFormat.PARQUET == "parquet"

    def test_file_format_xml(self):
        """FileFormat has XML value."""
        assert FileFormat.XML == "xml"

    def test_file_format_html(self):
        """FileFormat has HTML value."""
        assert FileFormat.HTML == "html"

    def test_file_format_enum_comparison(self):
        """FileFormat enum values can be compared."""
        xml1 = FileFormat.XML
        xml2 = FileFormat.XML
        assert xml1 == xml2

    def test_file_format_string_value(self):
        """FileFormat enum values work as strings."""
        fmt = FileFormat.XML
        assert fmt.value == "xml"


class TestIngestRecord:
    """Test IngestRecord dataclass."""

    def test_ingest_record_basic(self):
        """IngestRecord stores text and record_id."""
        record = IngestRecord(record_id=0, text="Hello world")

        assert record.record_id == 0
        assert record.text == "Hello world"

    def test_ingest_record_metadata(self):
        """IngestRecord stores metadata dict."""
        metadata = {"source": "test.xml", "tag": "person"}
        record = IngestRecord(record_id=0, text="content", metadata=metadata)

        assert record.metadata == metadata
        assert record.metadata["source"] == "test.xml"

    def test_ingest_record_default_metadata(self):
        """IngestRecord defaults to empty metadata dict."""
        record = IngestRecord(record_id=0, text="content")

        assert record.metadata == {}
        assert isinstance(record.metadata, dict)


class TestHTMLReaderEdgeCases:
    """Test edge cases in HTML reader."""

    def test_read_html_noscript_tags_stripped(self, tmp_path):
        """HTML reader excludes noscript tag content."""
        html_content = """
<html>
    <body>
        <p>Visible</p>
        <noscript>Hidden noscript</noscript>
        <p>More visible</p>
    </body>
</html>
"""
        html_file = tmp_path / "page.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert "Visible" in records[0].text
        assert "More visible" in records[0].text

    def test_read_html_empty_file(self, tmp_path):
        """HTML reader handles empty HTML files."""
        html_content = "<html><body></body></html>"
        html_file = tmp_path / "empty.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        assert records[0].text == ""

    def test_read_html_whitespace_only(self, tmp_path):
        """HTML reader handles whitespace-only HTML."""
        html_content = "<html><body>   \n\t   </body></html>"
        html_file = tmp_path / "whitespace.html"
        html_file.write_text(html_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(html_file), config))

        assert len(records) == 1
        assert records[0].text == ""


class TestXMLReaderEdgeCases:
    """Test edge cases in XML reader."""

    def test_read_xml_nested_elements(self, tmp_path):
        """XML reader handles deeply nested elements."""
        xml_content = """<?xml version="1.0"?>
<root>
    <level1>
        <level2>
            <level3>Deep content</level3>
        </level2>
    </level1>
</root>
"""
        xml_file = tmp_path / "nested.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(xml_file), config))

        assert len(records) == 1
        assert "Deep content" in records[0].text

    def test_read_xml_mixed_content(self, tmp_path):
        """XML reader handles mixed text and element content."""
        xml_content = """<?xml version="1.0"?>
<root>
    Text before
    <element>Element text</element>
    Text after
</root>
"""
        xml_file = tmp_path / "mixed.xml"
        xml_file.write_text(xml_content)

        config = IngestConfig(whole_file=True)
        records = list(read_file(str(xml_file), config))

        assert len(records) == 1
        # Should include both text and element content
        text = records[0].text
        assert "Element text" in text
