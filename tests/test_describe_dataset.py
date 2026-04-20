"""Tests for the describe_dataset tool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ds_context_mcp.tools.datasets import (
    ColumnInfo,
    DescribeDatasetResult,
    describe_dataset,
)


def _basic_df() -> pd.DataFrame:
    """Return a small DataFrame with mixed types for tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "carol", "dave", "eve"],
            "score": [10.5, 20.0, 30.25, 40.0, 50.5],
        }
    )


def test_csv_basic(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "basic.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path))

    assert isinstance(result, DescribeDatasetResult)
    assert result.error is None
    assert result.path == str(path.resolve())
    assert result.format == "csv"
    assert result.row_count == 5
    assert result.column_count == 3
    assert [c.name for c in result.columns] == ["id", "name", "score"]
    assert result.size_bytes > 0
    assert result.memory_usage_bytes > 0


def test_parquet_basic(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "basic.parquet"
    df.to_parquet(path, index=False)

    result = describe_dataset(str(path))

    assert result.error is None
    assert result.format == "parquet"
    assert result.row_count == 5
    assert result.column_count == 3
    assert [c.name for c in result.columns] == ["id", "name", "score"]


def test_feather_basic(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "basic.feather"
    df.to_feather(path)

    result = describe_dataset(str(path))

    assert result.error is None
    assert result.format == "feather"
    assert result.row_count == 5
    assert result.column_count == 3


def test_numeric_stats(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    path = tmp_path / "num.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path))
    col = next(c for c in result.columns if c.name == "x")

    assert isinstance(col, ColumnInfo)
    assert col.is_numeric is True
    assert col.min == pytest.approx(1.0)
    assert col.max == pytest.approx(5.0)
    assert col.mean == pytest.approx(3.0)


def test_missing_values(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path))
    col = next(c for c in result.columns if c.name == "x")

    assert col.missing_count == 2
    assert col.missing_rate == pytest.approx(0.4)
    assert col.min == pytest.approx(1.0)
    assert col.max == pytest.approx(5.0)
    assert col.mean == pytest.approx(3.0)


def test_categorical_column(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "obj": ["a", "b", "a", "c"],
            "flag": [True, False, True, True],
            "cat": pd.Categorical(["x", "y", "x", "z"]),
        }
    )
    path = tmp_path / "cat.parquet"
    df.to_parquet(path, index=False)

    result = describe_dataset(str(path))
    cols = {c.name: c for c in result.columns}

    assert cols["obj"].is_categorical is True
    assert cols["obj"].is_numeric is False
    assert cols["flag"].is_categorical is True
    assert cols["flag"].is_numeric is False
    assert cols["cat"].is_categorical is True
    assert cols["cat"].is_numeric is False


def test_sample_values_truncated(tmp_path: Path) -> None:
    long_value = "a" * 500
    df = pd.DataFrame({"text": [long_value, "short"]})
    path = tmp_path / "long.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path))
    col = next(c for c in result.columns if c.name == "text")

    truncated = next(v for v in col.sample_values if v.endswith("..."))
    assert len(truncated) <= 103
    assert truncated.startswith("aaaa")


def test_sample_rows_included(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, "x", "y"]})
    path = tmp_path / "with_sample.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path), include_sample=True, sample_size=2)

    assert result.sample_rows is not None
    assert len(result.sample_rows) == 2
    assert result.sample_rows[0]["a"] == 1
    assert result.sample_rows[0]["b"] is None


def test_sample_rows_excluded(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "nosample.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path), include_sample=False)

    assert result.sample_rows is None


def test_empty_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    path = tmp_path / "empty.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path))

    assert result.error is None
    assert result.row_count == 0
    assert result.column_count == 1
    col = result.columns[0]
    assert col.missing_count == 0
    assert col.unique_count == 0


def test_nonexistent_file_returns_error(tmp_path: Path) -> None:
    result = describe_dataset(str(tmp_path / "missing.csv"))

    assert result.error is not None
    assert result.row_count == 0
    assert result.column_count == 0
    assert result.columns == []


def test_directory_path_returns_error(tmp_path: Path) -> None:
    result = describe_dataset(str(tmp_path))

    assert result.error is not None
    assert "director" in result.error.lower()


def test_unsupported_extension_returns_error(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    result = describe_dataset(str(path))

    assert result.error is not None


def test_sample_size_clipped(tmp_path: Path) -> None:
    df = pd.DataFrame({"n": list(range(50))})
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)

    result = describe_dataset(str(path), include_sample=True, sample_size=100)

    assert result.sample_rows is not None
    assert len(result.sample_rows) == 20


def test_datetime_column(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "event_at": pd.to_datetime(
                ["2026-01-01 10:00:00", "2026-01-02 11:30:00", "2026-01-03 12:45:00"]
            )
        }
    )
    path = tmp_path / "dates.parquet"
    df.to_parquet(path, index=False)

    result = describe_dataset(str(path), include_sample=True, sample_size=3)
    col = next(c for c in result.columns if c.name == "event_at")

    assert "datetime" in col.dtype
    assert result.sample_rows is not None
    first = result.sample_rows[0]["event_at"]
    assert isinstance(first, str)
    assert first.startswith("2026-01-01")
    assert "T" in first
