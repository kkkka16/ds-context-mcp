"""Tests for the query_sql tool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ds_context_mcp.tools.sql import (
    QuerySqlResult,
    _validate_sql,
    _validate_table_name,
    query_sql,
)


def _make_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


# --- _validate_sql ------------------------------------------------------------


def test_validate_sql_accepts_select() -> None:
    assert _validate_sql("SELECT * FROM t") is None


def test_validate_sql_accepts_with() -> None:
    assert _validate_sql("WITH x AS (SELECT 1) SELECT * FROM x") is None


def test_validate_sql_rejects_keywords_in_comment() -> None:
    # The DELETE word lives outside the comment, so it must still be rejected.
    sql = "/* harmless */ SELECT * FROM t WHERE 1=1; DELETE FROM t"
    assert _validate_sql(sql) is not None


def test_validate_sql_comment_only_keyword_ignored() -> None:
    # DELETE appears only inside a comment, so the validator should accept the
    # surrounding SELECT.
    sql = "SELECT * FROM t -- DELETE FROM t"
    assert _validate_sql(sql) is None


def test_validate_sql_rejects_word_boundary() -> None:
    # `DELETE` as a word is forbidden, but identifiers that merely contain the
    # substring (e.g. `selected_user`) must remain allowed.
    assert _validate_sql("DELETE FROM t") is not None
    assert _validate_sql("SELECT selected_user FROM t") is None


def test_validate_table_name() -> None:
    assert _validate_table_name("titanic") is True
    assert _validate_table_name("_my_table") is True
    assert _validate_table_name("table1") is True
    assert _validate_table_name("my-table") is False
    assert _validate_table_name("1table") is False
    assert _validate_table_name("my table") is False
    assert _validate_table_name("") is False


# --- query_sql ----------------------------------------------------------------


@pytest.fixture
def small_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "survived": [0, 1, 1, 0, 1],
            "pclass": [3, 1, 3, 1, 2],
            "age": [22.0, 38.0, 26.0, 35.0, 35.0],
        }
    )
    path = tmp_path / "t.csv"
    _make_csv(path, df)
    return path


def test_simple_select(small_csv: Path) -> None:
    result = query_sql("SELECT * FROM t LIMIT 3", {"t": str(small_csv)})

    assert isinstance(result, QuerySqlResult)
    assert result.error is None
    assert result.row_count == 3
    assert len(result.rows) == 3
    assert set(result.columns) == {"survived", "pclass", "age"}


def test_select_with_where(small_csv: Path) -> None:
    result = query_sql("SELECT survived, age FROM t WHERE age > 30", {"t": str(small_csv)})

    assert result.error is None
    assert result.row_count == 3
    for row in result.rows:
        assert row["age"] > 30


def test_with_clause_allowed(small_csv: Path) -> None:
    sql = "WITH sub AS (SELECT * FROM t WHERE pclass = 1) SELECT * FROM sub"
    result = query_sql(sql, {"t": str(small_csv)})

    assert result.error is None
    assert result.row_count == 2


def test_aggregate_query(small_csv: Path) -> None:
    sql = "SELECT pclass, COUNT(*) AS n FROM t GROUP BY pclass ORDER BY pclass"
    result = query_sql(sql, {"t": str(small_csv)})

    assert result.error is None
    assert result.row_count == 3
    assert result.columns == ["pclass", "n"]
    counts = {int(row["pclass"]): int(row["n"]) for row in result.rows}
    assert counts == {1: 2, 2: 1, 3: 2}


def test_multiple_tables_join(tmp_path: Path) -> None:
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    _make_csv(a_path, pd.DataFrame({"id": [1, 2, 3], "name": ["x", "y", "z"]}))
    _make_csv(b_path, pd.DataFrame({"id": [1, 2, 3], "score": [10, 20, 30]}))

    sql = "SELECT a.name, b.score FROM a JOIN b ON a.id = b.id ORDER BY a.id"
    result = query_sql(sql, {"a": str(a_path), "b": str(b_path)})

    assert result.error is None
    assert result.row_count == 3
    assert result.rows[0] == {"name": "x", "score": 10}


def test_parquet_query(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    pd.DataFrame({"x": [1, 2, 3, 4, 5]}).to_parquet(path, index=False)

    result = query_sql("SELECT SUM(x) AS total FROM t", {"t": str(path)})

    assert result.error is None
    assert result.row_count == 1
    assert result.rows[0]["total"] == 15


def test_empty_sql_returns_error(small_csv: Path) -> None:
    result = query_sql("   ", {"t": str(small_csv)})

    assert result.error is not None
    assert "empty" in result.error.lower()


def test_non_select_rejected(small_csv: Path) -> None:
    result = query_sql("UPDATE t SET age=0", {"t": str(small_csv)})

    assert result.error is not None
    assert "Forbidden" in result.error or "SELECT" in result.error


def test_drop_rejected(small_csv: Path) -> None:
    result = query_sql("DROP TABLE t", {"t": str(small_csv)})

    assert result.error is not None
    assert "DROP" in result.error or "Forbidden" in result.error


def test_multiple_statements_rejected(small_csv: Path) -> None:
    result = query_sql("SELECT 1; SELECT 2", {"t": str(small_csv)})

    assert result.error is not None
    assert "Multiple" in result.error or "multiple" in result.error


def test_invalid_table_name_rejected(small_csv: Path) -> None:
    result = query_sql("SELECT 1", {"my-table": str(small_csv)})

    assert result.error is not None
    assert "Invalid table name" in result.error


def test_nonexistent_file_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    result = query_sql("SELECT * FROM t", {"t": str(missing)})

    assert result.error is not None
    assert "not found" in result.error.lower() or "does not exist" in result.error.lower()


def test_max_rows_truncation(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    pd.DataFrame({"x": list(range(50))}).to_csv(path, index=False)

    result = query_sql("SELECT * FROM t", {"t": str(path)}, max_rows=10)

    assert result.error is None
    assert result.truncated is True
    assert result.row_count == 10
    assert len(result.rows) == 10


def test_nan_converted_to_none(tmp_path: Path) -> None:
    path = tmp_path / "nans.csv"
    pd.DataFrame({"x": [1.0, None, 3.0]}).to_csv(path, index=False)

    result = query_sql("SELECT * FROM t ORDER BY x NULLS LAST", {"t": str(path)})

    assert result.error is None
    assert result.row_count == 3
    nulls = [row for row in result.rows if row["x"] is None]
    assert len(nulls) == 1
