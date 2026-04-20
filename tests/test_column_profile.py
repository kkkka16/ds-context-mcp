"""Tests for the column_profile tool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ds_context_mcp.tools.datasets import (
    ColumnProfileResult,
    column_profile,
)


def test_numeric_column_basic(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    path = tmp_path / "num.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "x")

    assert isinstance(result, ColumnProfileResult)
    assert result.error is None
    assert result.column == "x"
    assert result.numeric_stats is not None
    assert result.string_stats is None


def test_numeric_stats_values(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    path = tmp_path / "num.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "x")

    assert result.error is None
    stats = result.numeric_stats
    assert stats is not None
    assert stats.min == pytest.approx(1.0)
    assert stats.max == pytest.approx(10.0)
    assert stats.mean == pytest.approx(5.5)
    assert stats.median == pytest.approx(5.5)
    assert stats.std == pytest.approx(
        float(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).std(ddof=1))
    )
    assert stats.q25 == pytest.approx(3.25)
    assert stats.q75 == pytest.approx(7.75)
    assert stats.has_negative is False
    assert stats.has_zero is False


def test_outlier_count(tmp_path: Path) -> None:
    # 1..10 + outlier 1000
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0]})
    path = tmp_path / "outliers.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "x")

    assert result.error is None
    stats = result.numeric_stats
    assert stats is not None
    assert stats.outlier_count >= 1


def test_string_column_basic(tmp_path: Path) -> None:
    df = pd.DataFrame({"name": ["alice", "bob", "carol", "alice", ""]})
    path = tmp_path / "str.parquet"
    df.to_parquet(path, index=False)

    result = column_profile(str(path), "name")

    assert result.error is None
    assert result.string_stats is not None
    assert result.numeric_stats is None
    assert result.string_stats.empty_count == 1
    assert result.string_stats.min_length == 0
    assert result.string_stats.max_length == 5


def test_top_values_order(tmp_path: Path) -> None:
    df = pd.DataFrame({"c": ["a", "a", "a", "b", "b", "c"]})
    path = tmp_path / "top.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "c", top_k=3)

    assert result.error is None
    assert len(result.top_values) == 3
    counts = [t.count for t in result.top_values]
    assert counts == sorted(counts, reverse=True)
    assert result.top_values[0].value == "a"
    assert result.top_values[0].count == 3


def test_top_values_frequency(tmp_path: Path) -> None:
    df = pd.DataFrame({"c": ["a", "a", "b", "b", "c", "c", "c", "c"]})
    path = tmp_path / "freq.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "c")

    assert result.error is None
    for tv in result.top_values:
        assert tv.frequency == pytest.approx(tv.count / result.total_count)


def test_missing_column_returns_error(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "nonexistent")

    assert result.error is not None
    assert "nonexistent" in result.error
    assert "Available" in result.error
    assert "a" in result.error
    assert "b" in result.error


def test_all_missing_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [None, None, None], "y": [1, 2, 3]})
    path = tmp_path / "missing.parquet"
    df.to_parquet(path, index=False)

    result = column_profile(str(path), "x")

    assert result.error is None
    assert result.total_count == 0
    assert result.numeric_stats is None
    assert result.string_stats is None


def test_high_cardinality(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(100)]})
    path = tmp_path / "ids.csv"
    df.to_csv(path, index=False)

    result = column_profile(str(path), "id")

    assert result.error is None
    assert result.cardinality_ratio == pytest.approx(1.0)
    assert result.unique_count == 100


def test_bool_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"flag": [True, False, True, True, False]})
    path = tmp_path / "bool.parquet"
    df.to_parquet(path, index=False)

    result = column_profile(str(path), "flag")

    assert result.error is None
    assert result.numeric_stats is None
    assert result.string_stats is None
    values = {tv.value for tv in result.top_values}
    assert "True" in values
    assert "False" in values
