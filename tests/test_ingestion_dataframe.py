"""Tests for pii_anon.ingestion.dataframe - DataFrame integration for PII processing."""

from __future__ import annotations

import pytest

from pii_anon.ingestion.dataframe import (
    read_dataframe,
    results_to_dataframe,
    _get_columns,
    _safe_value,
)


class TestReadDataframeWithIterRows:
    """Test read_dataframe with DataFrames that have iter_rows() (Polars-style)."""

    def test_read_dataframe_iter_rows_polars_style(self):
        """read_dataframe works with iter_rows() method (Polars compat)."""
        # Mock Polars-like DataFrame
        class MockPolarsDF:
            def iter_rows(self, named=False):
                yield {"text": "message 1", "user": "alice", "id": 100}
                yield {"text": "message 2", "user": "bob", "id": 101}

        df = MockPolarsDF()
        records = list(read_dataframe(df, text_column="text"))

        assert len(records) == 2
        assert records[0].text == "message 1"
        assert records[0].metadata == {"user": "alice", "id": 100}
        assert records[1].text == "message 2"
        assert records[1].metadata == {"user": "bob", "id": 101}

    def test_read_dataframe_iter_rows_with_max_record_chars(self):
        """read_dataframe respects max_record_chars with iter_rows()."""
        class MockDF:
            def iter_rows(self, named=False):
                yield {"text": "This is a very long text", "col": "val"}

        df = MockDF()
        records = list(read_dataframe(df, text_column="text", max_record_chars=10))

        assert len(records) == 1
        assert records[0].text == "This is a "

    def test_read_dataframe_iter_rows_none_text(self):
        """read_dataframe handles None text values with iter_rows()."""
        class MockDF:
            def iter_rows(self, named=False):
                yield {"text": None, "col": "val"}

        df = MockDF()
        records = list(read_dataframe(df, text_column="text"))

        assert len(records) == 1
        assert records[0].text == ""

    def test_read_dataframe_iter_rows_empty_text(self):
        """read_dataframe handles empty text with iter_rows()."""
        class MockDF:
            def iter_rows(self, named=False):
                yield {"text": "", "col": "val"}

        df = MockDF()
        records = list(read_dataframe(df, text_column="text"))

        assert len(records) == 1
        assert records[0].text == ""

    def test_read_dataframe_iter_rows_missing_column(self):
        """read_dataframe handles missing text_column with iter_rows()."""
        class MockDF:
            def iter_rows(self, named=False):
                yield {"other_col": "value"}

        df = MockDF()
        # Should treat missing column as empty string
        records = list(read_dataframe(df, text_column="text"))

        assert len(records) == 1
        assert records[0].text == ""


class TestReadDataframeErrorCases:
    """Test read_dataframe error handling."""

    def test_read_dataframe_no_iterrows_no_iter_rows(self):
        """read_dataframe raises TypeError for incompatible object."""
        class BadDF:
            pass

        with pytest.raises(TypeError, match="Cannot iterate over"):
            list(read_dataframe(BadDF(), text_column="text"))

    def test_read_dataframe_missing_column_with_columns_attr(self):
        """read_dataframe raises ValueError when text_column not in columns."""
        class MockRow:
            def get(self, key, default=""):
                return "val1" if key == "col1" else default

            def items(self):
                return [("col1", "val1")]

        class MockDF:
            def __init__(self):
                self.columns = ["col1", "col2"]

            def iterrows(self):
                yield (0, MockRow())

        with pytest.raises(ValueError, match="Text column.*not found"):
            list(read_dataframe(MockDF(), text_column="missing_col"))

    def test_read_dataframe_columns_with_tolist_method(self):
        """read_dataframe validates columns when tolist() method exists."""
        class MockColumns:
            def tolist(self):
                return ["col1", "col2"]

        class MockRow:
            def get(self, key, default=""):
                return "val1" if key == "col1" else default

            def items(self):
                return [("col1", "val1")]

        class MockDF:
            def __init__(self):
                self.columns = MockColumns()

            def iterrows(self):
                yield (0, MockRow())

        # Should not raise since text_column is in columns
        list(read_dataframe(MockDF(), text_column="col1"))

    def test_read_dataframe_column_validation_none_columns(self):
        """read_dataframe skips validation when _get_columns returns None."""
        class MockRow:
            def get(self, key, default=""):
                return "value" if key == "text" else default

            def items(self):
                return [("text", "value")]

        class MockDF:
            def iterrows(self):
                yield (0, MockRow())

        # Should work even if we can't get columns
        records = list(read_dataframe(MockDF(), text_column="text"))
        assert len(records) == 1


class TestReadDataframeIndexHandling:
    """Test read_dataframe handling of row indices."""

    def test_read_dataframe_int_index(self):
        """read_dataframe converts int indices to record_id."""
        class MockRow:
            def get(self, key, default=""):
                return "msg" if key == "text" else default

            def items(self):
                return [("text", "msg")]

        class MockDF:
            def iterrows(self):
                yield (0, MockRow())
                yield (1, MockRow())

        df = MockDF()
        records = list(read_dataframe(df, text_column="text"))

        assert records[0].record_id == 0
        assert records[1].record_id == 1

    def test_read_dataframe_float_index(self):
        """read_dataframe converts float indices to int."""
        class MockRow:
            def get(self, key, default=""):
                return "msg" if key == "text" else default

            def items(self):
                return [("text", "msg")]

        class MockDF:
            def iterrows(self):
                yield (0.0, MockRow())
                yield (1.5, MockRow())

        df = MockDF()
        records = list(read_dataframe(df, text_column="text"))

        assert records[0].record_id == 0
        assert records[1].record_id == 1

    def test_read_dataframe_string_index(self):
        """read_dataframe hashes string indices."""
        class MockRow:
            def get(self, key, default=""):
                return "msg" if key == "text" else default

            def items(self):
                return [("text", "msg")]

        class MockDF:
            def iterrows(self):
                yield ("idx_a", MockRow())

        df = MockDF()
        records = list(read_dataframe(df, text_column="text"))

        # Should hash the string
        assert isinstance(records[0].record_id, int)
        assert records[0].record_id == hash("idx_a")


pd = pytest.importorskip("pandas")


class TestResultsToDataframe:
    """Test results_to_dataframe conversion to pandas DataFrame."""

    def test_results_to_dataframe_basic(self):
        """results_to_dataframe converts results to DataFrame."""
        results = [
            {
                "metadata": {"id": 1, "source": "test"},
                "transformed_payload": {"text": "processed 1"},
                "ensemble_findings": [{"type": "EMAIL"}],
                "confidence_envelope": {"score": 0.95, "risk_level": "low"},
            }
        ]

        df = results_to_dataframe(results)
        assert len(df) == 1
        assert df.loc[0, "transformed_text"] == "processed 1"
        assert df.loc[0, "id"] == 1
        assert df.loc[0, "source"] == "test"
        assert df.loc[0, "entities_found"] == 1
        assert df.loc[0, "confidence_score"] == 0.95
        assert df.loc[0, "risk_level"] == "low"

    def test_results_to_dataframe_transformed_payload_string(self):
        """results_to_dataframe handles non-dict transformed_payload."""
        results = [
            {
                "metadata": {},
                "transformed_payload": "plain text string",
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]

        df = results_to_dataframe(results)
        assert df.loc[0, "transformed_text"] == "plain text string"

    def test_results_to_dataframe_no_confidence_envelope(self):
        """results_to_dataframe handles missing confidence_envelope."""
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "text"},
                "ensemble_findings": [],
            }
        ]

        df = results_to_dataframe(results)
        # Should have default values when envelope is missing
        assert "confidence_score" in df.columns
        assert df.loc[0, "confidence_score"] == 0.0
        assert df.loc[0, "risk_level"] == "unknown"

    def test_results_to_dataframe_non_dict_confidence_envelope(self):
        """results_to_dataframe handles non-dict confidence_envelope."""
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "text"},
                "ensemble_findings": [],
                "confidence_envelope": "not_a_dict",
            }
        ]

        df = results_to_dataframe(results)
        # Should skip setting confidence fields
        assert "confidence_score" not in df.columns or df.loc[0, "confidence_score"] != df.loc[0, "confidence_score"]

    def test_results_to_dataframe_non_list_findings(self):
        """results_to_dataframe handles non-list ensemble_findings."""
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "text"},
                "ensemble_findings": "not_a_list",
                "confidence_envelope": {},
            }
        ]

        df = results_to_dataframe(results)
        assert df.loc[0, "entities_found"] == 0

    def test_results_to_dataframe_custom_text_key(self):
        """results_to_dataframe respects text_key parameter."""
        results = [
            {
                "metadata": {},
                "transformed_payload": {"text": "processed"},
                "ensemble_findings": [],
                "confidence_envelope": {},
            }
        ]

        df = results_to_dataframe(results, text_key="custom_output")
        assert "custom_output" in df.columns
        assert df.loc[0, "custom_output"] == "processed"

    def test_results_to_dataframe_no_pandas_raises(self):
        """results_to_dataframe raises ImportError when pandas unavailable."""
        import builtins

        results = [{"metadata": {}, "transformed_payload": {"text": "text"}}]

        # Mock missing pandas
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named pandas")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="requires pandas"):
                results_to_dataframe(results)
        finally:
            builtins.__import__ = real_import

    def test_results_to_dataframe_empty_results(self):
        """results_to_dataframe handles empty results list."""
        results = []
        df = results_to_dataframe(results)
        assert len(df) == 0

    def test_results_to_dataframe_multiple_results(self):
        """results_to_dataframe converts multiple results."""
        results = [
            {
                "metadata": {"id": 1},
                "transformed_payload": {"text": "text1"},
                "ensemble_findings": [1, 2],
                "confidence_envelope": {"score": 0.9, "risk_level": "low"},
            },
            {
                "metadata": {"id": 2},
                "transformed_payload": {"text": "text2"},
                "ensemble_findings": [1],
                "confidence_envelope": {"score": 0.8, "risk_level": "medium"},
            },
        ]

        df = results_to_dataframe(results)
        assert len(df) == 2
        assert df.loc[0, "id"] == 1
        assert df.loc[1, "id"] == 2
        assert df.loc[0, "entities_found"] == 2
        assert df.loc[1, "entities_found"] == 1


class TestSafeValue:
    """Test _safe_value conversion of numpy/pandas types."""

    def test_safe_value_regular_python_types(self):
        """_safe_value passes through regular Python types."""
        assert _safe_value("string") == "string"
        assert _safe_value(42) == 42
        assert _safe_value(3.14) == 3.14
        assert _safe_value(True) is True
        assert _safe_value(None) is None

    def test_safe_value_numpy_scalar(self):
        """_safe_value converts numpy scalars with .item() method."""
        try:
            import numpy as np

            val = np.int64(42)
            assert _safe_value(val) == 42
            assert isinstance(_safe_value(val), int)

            val = np.float64(3.14)
            result = _safe_value(val)
            assert abs(result - 3.14) < 0.001
        except ImportError:
            pytest.skip("numpy not available")

    def test_safe_value_object_with_item(self):
        """_safe_value calls .item() on objects that have it."""
        class MockScalar:
            def item(self):
                return "converted"

        assert _safe_value(MockScalar()) == "converted"

    def test_safe_value_list_and_dict(self):
        """_safe_value leaves lists and dicts unchanged."""
        lst = [1, 2, 3]
        assert _safe_value(lst) == lst

        dct = {"key": "value"}
        assert _safe_value(dct) == dct


class TestGetColumns:
    """Test _get_columns helper function."""

    def test_get_columns_with_tolist(self):
        """_get_columns uses tolist() when available."""
        class MockColumns:
            def tolist(self):
                return ["col1", "col2"]

        class MockDF:
            columns = MockColumns()

        cols = _get_columns(MockDF())
        assert cols == ["col1", "col2"]

    def test_get_columns_iterable(self):
        """_get_columns converts iterable columns to list."""
        class MockDF:
            columns = ["col1", "col2"]

        cols = _get_columns(MockDF())
        assert cols == ["col1", "col2"]

    def test_get_columns_no_columns_attr(self):
        """_get_columns returns None when no columns attribute."""
        class MockDF:
            pass

        cols = _get_columns(MockDF())
        assert cols is None

    def test_get_columns_pandas_dataframe(self):
        """_get_columns works with actual pandas DataFrames."""
        try:
            import pandas as pd

            df = pd.DataFrame({"col1": [1], "col2": [2]})
            cols = _get_columns(df)
            assert set(cols) == {"col1", "col2"}
        except ImportError:
            pytest.skip("pandas not available")


class TestReadDataframeWithIterrows:
    """Test read_dataframe with iterrows() method."""

    def test_read_dataframe_iterrows_pandas(self):
        """read_dataframe works with pandas DataFrames."""
        try:
            import pandas as pd

            df = pd.DataFrame({
                "text": ["message 1", "message 2"],
                "user": ["alice", "bob"],
            })

            records = list(read_dataframe(df, text_column="text"))
            assert len(records) == 2
            assert records[0].text == "message 1"
            assert records[0].metadata["user"] == "alice"
            assert records[1].text == "message 2"
            assert records[1].metadata["user"] == "bob"
        except ImportError:
            pytest.skip("pandas not available")

    def test_read_dataframe_iterrows_numeric_index(self):
        """read_dataframe uses numeric indices from pandas."""
        try:
            import pandas as pd

            df = pd.DataFrame({"text": ["msg1", "msg2"]})
            records = list(read_dataframe(df, text_column="text"))

            assert records[0].record_id == 0
            assert records[1].record_id == 1
        except ImportError:
            pytest.skip("pandas not available")

    def test_read_dataframe_iterrows_string_values_converted(self):
        """read_dataframe converts all values to strings."""
        try:
            import pandas as pd

            df = pd.DataFrame({
                "text": [123, 456],
                "col": [10, 20],
            })

            records = list(read_dataframe(df, text_column="text"))
            assert records[0].text == "123"
            assert records[1].text == "456"
        except ImportError:
            pytest.skip("pandas not available")

    def test_read_dataframe_with_max_chars_iterrows(self):
        """read_dataframe respects max_record_chars with iterrows()."""
        try:
            import pandas as pd

            df = pd.DataFrame({"text": ["Hello World", "Test"]})
            records = list(read_dataframe(df, text_column="text", max_record_chars=5))

            assert records[0].text == "Hello"
            assert records[1].text == "Test"
        except ImportError:
            pytest.skip("pandas not available")
