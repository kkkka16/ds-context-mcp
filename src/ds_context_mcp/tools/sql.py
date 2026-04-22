"""SQL query execution tool backed by DuckDB."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

_DEFAULT_MAX_ROWS: int = 1000
_HARD_MAX_ROWS: int = 10000

_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "IMPORT",
    "PRAGMA",
    "CALL",
    "EXECUTE",
    "INSTALL",
    "LOAD",
    "SET",
    "RESET",
    "CHECKPOINT",
)

_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_FIRST_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)")


class QuerySqlResult(BaseModel):
    """Result of a query_sql call.

    Attributes:
        sql: The SQL string that was executed (echoed for debugging).
        row_count: Number of rows actually returned (after truncation).
        truncated: ``True`` if the underlying result exceeded ``max_rows``.
        columns: Column names of the result set.
        rows: Result rows as JSON-friendly dicts. ``NaN`` becomes ``None``,
            datetime becomes ISO8601, bytes becomes ``str``.
        execution_time_ms: Wall-clock time spent inside DuckDB, in ms.
        error: Fatal error message; set only on validation or execution failure.
    """

    sql: str
    row_count: int
    truncated: bool
    columns: list[str]
    rows: list[dict[str, object]]
    execution_time_ms: float
    error: str | None = None


def _strip_sql_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments from SQL."""
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    return sql


def _validate_table_name(name: str) -> bool:
    """Return True if ``name`` is a safe SQL identifier."""
    return bool(_TABLE_NAME_RE.match(name))


def _validate_sql(sql: str) -> str | None:
    """Validate that ``sql`` is a single read-only statement.

    Args:
        sql: Raw SQL string from the caller.

    Returns:
        ``None`` when the SQL passes all checks. Otherwise an error message
        describing the first failed check (empty, non-SELECT, forbidden
        keyword, multiple statements).
    """
    if not sql or not sql.strip():
        return "SQL is empty"

    sanitized = _strip_sql_comments(sql)
    stripped = sanitized.strip()
    if not stripped:
        return "SQL is empty"

    # Reject multi-statement input. A single trailing semicolon is allowed.
    trimmed = stripped.rstrip(";").rstrip()
    if ";" in trimmed:
        return "Multiple statements are not allowed"

    first_match = _FIRST_TOKEN_RE.match(sanitized)
    if first_match is None:
        return "SQL is empty"
    first_token = first_match.group(1).upper()
    if first_token not in {"SELECT", "WITH"}:
        return f"Only SELECT/WITH queries are allowed. Got: {first_token}"

    upper = sanitized.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            return f"Forbidden keyword detected: {keyword}"

    return None


def _convert_cell(value: object) -> object:
    """Convert a single cell value to a JSON-friendly representation."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return value


def _df_to_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-friendly row dicts."""
    rows: list[dict[str, object]] = []
    for record in df.to_dict(orient="records"):
        rows.append({str(k): _convert_cell(v) for k, v in record.items()})
    return rows


def _error_result(sql: str, message: str) -> QuerySqlResult:
    """Return a failure ``QuerySqlResult`` with zeroed metrics."""
    return QuerySqlResult(
        sql=sql,
        row_count=0,
        truncated=False,
        columns=[],
        rows=[],
        execution_time_ms=0.0,
        error=message,
    )


def _register_file_mappings(
    con: duckdb.DuckDBPyConnection,
    file_mappings: dict[str, str],
) -> str | None:
    """Register each file in ``file_mappings`` as a DuckDB view/table.

    Returns ``None`` on success, or an error message on the first failure.
    """
    for table_name, raw_path in file_mappings.items():
        if not _validate_table_name(table_name):
            return f"Invalid table name: {table_name}"

        path = Path(raw_path).resolve()
        if not path.exists():
            return f"File not found: {path}"
        if not path.is_file():
            return f"Path is not a regular file: {path}"

        ext = path.suffix.lower()
        abs_path = str(path).replace("'", "''")
        if ext == ".csv":
            con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{abs_path}')")
        elif ext == ".parquet":
            con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{abs_path}')")
        elif ext == ".feather":
            df = pd.read_feather(path)
            con.register(table_name, df)
        else:
            return f"Unsupported extension: {ext} (supported: .csv, .parquet, .feather)"
    return None


def query_sql(
    sql: str,
    file_mappings: dict[str, str],
    max_rows: int = _DEFAULT_MAX_ROWS,
) -> QuerySqlResult:
    """Execute a read-only SQL query against local data files using DuckDB.

    Each entry in ``file_mappings`` is registered as a DuckDB view (or, for
    Feather, a registered DataFrame) so the query can reference the file by
    its mapped table name. Only ``SELECT`` and ``WITH`` queries are accepted;
    statements that mutate state, attach databases, or invoke pragmas are
    rejected at validation time.

    Args:
        sql: The SQL query to execute. Must be a single SELECT/WITH statement.
        file_mappings: Mapping of ``table_name -> file_path``. Table names must
            match ``^[a-zA-Z_][a-zA-Z0-9_]*$``.
        max_rows: Maximum number of rows to return. Clipped to ``[1, 10000]``.

    Returns:
        A :class:`QuerySqlResult`. Validation, registration, and DuckDB
        execution errors are reported via the ``error`` field instead of
        raising.
    """
    validation_error = _validate_sql(sql)
    if validation_error is not None:
        return _error_result(sql, validation_error)

    clipped_max_rows = max(1, min(max_rows, _HARD_MAX_ROWS))

    con = duckdb.connect(":memory:")
    try:
        registration_error = _register_file_mappings(con, file_mappings)
        if registration_error is not None:
            return _error_result(sql, registration_error)

        start = time.perf_counter()
        try:
            df = con.execute(sql).fetchdf()
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return QuerySqlResult(
                sql=sql,
                row_count=0,
                truncated=False,
                columns=[],
                rows=[],
                execution_time_ms=elapsed_ms,
                error=f"SQL execution failed: {exc}",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        con.close()

    truncated = len(df) > clipped_max_rows
    if truncated:
        df = df.head(clipped_max_rows)

    return QuerySqlResult(
        sql=sql,
        row_count=len(df),
        truncated=truncated,
        columns=[str(c) for c in df.columns],
        rows=_df_to_rows(df),
        execution_time_ms=elapsed_ms,
        error=None,
    )


def register_sql_tools(mcp: FastMCP) -> None:
    """Register SQL query tools to the MCP server.

    Args:
        mcp: The FastMCP server instance to register tools on.
    """

    @mcp.tool()
    def query_sql_tool(
        sql: str,
        file_mappings: dict[str, str],
        max_rows: int = _DEFAULT_MAX_ROWS,
    ) -> QuerySqlResult:
        """Execute a SQL SELECT query against CSV/Parquet/Feather files using DuckDB.

        Use this tool to run analytical queries (aggregations, joins, filters)
        directly against local data files. Safer and faster than loading data
        into pandas for exploratory analysis.

        SECURITY: Only SELECT and WITH queries are allowed. Data modification
        statements (INSERT/UPDATE/DELETE/DROP/etc.) are rejected.

        Example:
            file_mappings = {"titanic": "/path/to/titanic.csv"}
            sql = "SELECT pclass, AVG(age) FROM titanic GROUP BY pclass"
        """
        return query_sql(sql, file_mappings, max_rows)
