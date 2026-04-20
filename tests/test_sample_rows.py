"""Tests for the sample_rows tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ds_context_mcp.tools.datasets import (
    SampleRowsResult,
    sample_rows,
)


def _basic_df() -> pd.DataFrame:
    """Return a small DataFrame with mixed types for tests."""
    return pd.DataFrame(
        {
            "id": list(range(1, 21)),
            "name": [f"user_{i}" for i in range(1, 21)],
            "score": [float(i) * 1.5 for i in range(1, 21)],
        }
    )


def test_head_mode(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    result = sample_rows(str(path), n=5, mode="head")

    assert isinstance(result, SampleRowsResult)
    assert result.error is None
    assert result.mode == "head"
    assert result.total_rows == 20
    assert result.returned_rows == 5
    assert len(result.rows) == 5
    assert result.rows[0]["id"] == 1
    assert result.rows[4]["id"] == 5
    assert result.columns == ["id", "name", "score"]


def test_tail_mode(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    result = sample_rows(str(path), n=3, mode="tail")

    assert result.error is None
    assert result.mode == "tail"
    assert result.returned_rows == 3
    assert len(result.rows) == 3
    assert result.rows[0]["id"] == 18
    assert result.rows[2]["id"] == 20


def test_random_mode_with_seed(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    result1 = sample_rows(str(path), n=5, mode="random", seed=42)
    result2 = sample_rows(str(path), n=5, mode="random", seed=42)

    assert result1.error is None
    assert result1.mode == "random"
    assert result1.returned_rows == 5
    assert [r["id"] for r in result1.rows] == [r["id"] for r in result2.rows]


def test_random_mode_n_larger_than_data(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = tmp_path / "small.csv"
    df.to_csv(path, index=False)

    result = sample_rows(str(path), n=50, mode="random", seed=1)

    assert result.error is None
    assert result.total_rows == 3
    assert result.returned_rows == 3
    assert len(result.rows) == 3


def test_n_clipped_to_100(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": list(range(500))})
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)

    result = sample_rows(str(path), n=500, mode="head")

    assert result.error is None
    assert result.total_rows == 500
    assert result.returned_rows == 100
    assert len(result.rows) == 100


def test_invalid_mode_returns_error(tmp_path: Path) -> None:
    df = _basic_df()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    result = sample_rows(str(path), n=5, mode="invalid")

    assert result.error is not None
    assert "mode" in result.error.lower()
    assert result.rows == []


def test_nonexistent_file_returns_error(tmp_path: Path) -> None:
    result = sample_rows(str(tmp_path / "missing.csv"))

    assert result.error is not None
    assert result.rows == []
    assert result.returned_rows == 0


def test_nan_converted_to_none(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"]})
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)

    result = sample_rows(str(path), n=3, mode="head")

    assert result.error is None
    assert result.rows[1]["a"] is None
    assert result.rows[1]["b"] is None
    assert result.rows[0]["a"] == 1.0
    assert result.rows[0]["b"] == "x"
