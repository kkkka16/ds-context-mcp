"""Dataset discovery and description tools."""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

DEFAULT_EXTENSIONS: tuple[str, ...] = (".csv", ".parquet", ".feather")

_DESCRIBE_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".parquet", ".feather"})
_MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024
_MAX_SAMPLE_SIZE: int = 20
_SAMPLE_VALUE_MAX_UNIQUE: int = 10
_SAMPLE_VALUE_RETURN_COUNT: int = 5
_SAMPLE_VALUE_MAX_LEN: int = 100

_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".ipynb_checkpoints",
        ".DS_Store",
    }
)


class DatasetEntry(BaseModel):
    """A single dataset file entry.

    Attributes:
        path: Absolute path to the dataset file.
        relative_path: Path relative to the search root.
        size_bytes: File size in bytes.
        format: Format identifier derived from the extension (e.g. "csv").
        modified_at: ISO8601 timestamp of last modification, timezone-aware (UTC).
    """

    path: str
    relative_path: str
    size_bytes: int
    format: str
    modified_at: str


class ListDatasetsResult(BaseModel):
    """Result of a list_datasets call.

    Attributes:
        root: Absolute path of the search root after resolution.
        count: Number of datasets discovered.
        datasets: List of discovered dataset entries.
        skipped: Paths skipped due to permission or read errors.
        error: Fatal error message; set only when the listing could not proceed.
    """

    root: str
    count: int
    datasets: list[DatasetEntry]
    skipped: list[str]
    error: str | None = None


def _is_excluded_dir_name(name: str) -> bool:
    """Return True if a directory name should be skipped during traversal."""
    if name in _EXCLUDED_DIR_NAMES:
        return True
    return name.startswith(".")


def _normalize_extensions(extensions: list[str] | None) -> tuple[str, ...]:
    """Return a tuple of lowercase extensions with a leading dot."""
    if extensions is None:
        return DEFAULT_EXTENSIONS
    normalized: list[str] = []
    for ext in extensions:
        lowered = ext.lower()
        if not lowered.startswith("."):
            lowered = f".{lowered}"
        normalized.append(lowered)
    return tuple(normalized)


def list_datasets(
    root_path: str,
    extensions: list[str] | None = None,
    max_depth: int = 5,
) -> ListDatasetsResult:
    """List data files under the given directory.

    Recursively walks ``root_path`` and returns every file whose extension
    matches ``extensions``. Symbolic links are never followed, and common
    noise directories (``.git``, ``.venv``, ``__pycache__``, ...) as well as
    any directory whose name starts with ``.`` are excluded.

    Args:
        root_path: Search root directory. Relative paths are resolved.
        extensions: Extensions to include. Defaults to ``.csv``, ``.parquet``,
            and ``.feather``. Comparison is case-insensitive.
        max_depth: Maximum depth of files relative to the root. A depth of 1
            means only files directly under the root are returned.

    Returns:
        A :class:`ListDatasetsResult` describing the discovered datasets. On
        invalid input (missing path, non-directory, etc.) the ``error`` field
        is populated and no exception is raised.
    """
    root = Path(root_path).resolve()

    if not root.exists():
        return ListDatasetsResult(
            root=str(root),
            count=0,
            datasets=[],
            skipped=[],
            error=f"Path does not exist: {root}",
        )

    if not root.is_dir():
        return ListDatasetsResult(
            root=str(root),
            count=0,
            datasets=[],
            skipped=[],
            error=f"Path is not a directory: {root}",
        )

    exts = _normalize_extensions(extensions)
    datasets: list[DatasetEntry] = []
    skipped: list[str] = []

    def _on_walk_error(err: OSError) -> None:
        target = err.filename if err.filename else str(err)
        skipped.append(str(target))

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_on_walk_error):
        current = Path(dirpath)
        depth = 0 if current == root else len(current.relative_to(root).parts)

        if depth >= max_depth:
            dirnames[:] = []
            continue

        pruned: list[str] = []
        for name in dirnames:
            if _is_excluded_dir_name(name):
                continue
            if (current / name).is_symlink():
                continue
            pruned.append(name)
        dirnames[:] = pruned

        for filename in filenames:
            file_path = current / filename
            if file_path.is_symlink():
                continue
            ext = file_path.suffix.lower()
            if ext not in exts:
                continue
            try:
                stat = file_path.stat()
            except OSError:
                skipped.append(str(file_path))
                continue
            datasets.append(
                DatasetEntry(
                    path=str(file_path),
                    relative_path=str(file_path.relative_to(root)),
                    size_bytes=stat.st_size,
                    format=ext.lstrip("."),
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                )
            )

    return ListDatasetsResult(
        root=str(root),
        count=len(datasets),
        datasets=datasets,
        skipped=skipped,
    )


class ColumnInfo(BaseModel):
    """Per-column schema and summary statistics.

    Attributes:
        name: Column name.
        dtype: pandas dtype string (e.g. ``"int64"``, ``"datetime64[ns]"``).
        missing_count: Number of NaN / None cells.
        missing_rate: Ratio of missing cells, in ``[0.0, 1.0]``.
        unique_count: Number of unique non-NaN values.
        sample_values: Up to 5 stringified unique values, each truncated to
            100 chars (with trailing ``"..."`` when truncated). ``NaN`` is
            rendered as the literal string ``"NaN"``.
        is_numeric: ``True`` for numeric dtypes (bool is excluded).
        is_categorical: ``True`` for object / category / bool / string dtypes.
        min: Minimum for numeric columns; ``None`` otherwise or all-NaN.
        max: Maximum for numeric columns; ``None`` otherwise or all-NaN.
        mean: Mean for numeric columns; ``None`` otherwise or all-NaN.
    """

    name: str
    dtype: str
    missing_count: int
    missing_rate: float
    unique_count: int
    sample_values: list[str]
    is_numeric: bool
    is_categorical: bool
    min: float | None = None
    max: float | None = None
    mean: float | None = None


class DescribeDatasetResult(BaseModel):
    """Result of a describe_dataset call.

    Attributes:
        path: Absolute resolved path of the file.
        format: ``"csv"`` / ``"parquet"`` / ``"feather"``.
        size_bytes: File size in bytes.
        row_count: Number of rows in the DataFrame.
        column_count: Number of columns.
        memory_usage_bytes: ``DataFrame.memory_usage(deep=True).sum()``.
        columns: Per-column information.
        sample_rows: Top-N rows as dicts when ``include_sample=True``.
        error: Fatal error message; set only on failure.
    """

    path: str
    format: str
    size_bytes: int
    row_count: int
    column_count: int
    memory_usage_bytes: int
    columns: list[ColumnInfo]
    sample_rows: list[dict[str, object]] | None = None
    error: str | None = None


def _describe_error(path: str, message: str) -> DescribeDatasetResult:
    """Return a failure ``DescribeDatasetResult`` with zeroed metrics."""
    return DescribeDatasetResult(
        path=path,
        format="",
        size_bytes=0,
        row_count=0,
        column_count=0,
        memory_usage_bytes=0,
        columns=[],
        sample_rows=None,
        error=message,
    )


def _read_dataframe(path: Path, ext: str) -> pd.DataFrame:
    """Dispatch to the correct pandas reader for the given extension."""
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported extension: {ext}")


def _stringify_sample_value(value: object) -> str:
    """Render one cell value as a truncated string suitable for LLM display."""
    if value is None:
        return "NaN"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    try:
        if pd.isna(value):
            return "NaN"
    except (TypeError, ValueError):
        pass
    text = str(value)
    if len(text) > _SAMPLE_VALUE_MAX_LEN:
        return text[:_SAMPLE_VALUE_MAX_LEN] + "..."
    return text


def _collect_sample_values(series: pd.Series) -> list[str]:
    """Return up to 5 stringified unique non-NaN sample values from a series."""
    non_null = series.dropna()
    if non_null.empty:
        return []
    seen: list[str] = []
    limit_unique = _SAMPLE_VALUE_MAX_UNIQUE
    for raw in non_null.tolist():
        rendered = _stringify_sample_value(raw)
        if rendered in seen:
            continue
        seen.append(rendered)
        if len(seen) >= limit_unique:
            break
    return seen[:_SAMPLE_VALUE_RETURN_COUNT]


def _is_categorical_dtype(series: pd.Series) -> bool:
    """Return True if the series should be treated as categorical for the LLM."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    kind = getattr(dtype, "kind", "")
    if kind in {"O", "b", "U", "S"}:
        return True
    return bool(pd.api.types.is_string_dtype(series) or pd.api.types.is_bool_dtype(series))


def _numeric_stat(value: Any) -> float | None:
    """Convert a pandas aggregation result to ``float`` or ``None`` if NaN."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _build_column_info(series: pd.Series) -> ColumnInfo:
    """Build a :class:`ColumnInfo` for a single series."""
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    is_categorical = _is_categorical_dtype(series)

    missing_count = int(series.isna().sum())
    total = len(series)
    missing_rate = (missing_count / total) if total > 0 else 0.0
    unique_count = int(series.dropna().nunique())

    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    if is_numeric:
        non_null = series.dropna()
        if not non_null.empty:
            min_val = _numeric_stat(non_null.min())
            max_val = _numeric_stat(non_null.max())
            mean_val = _numeric_stat(non_null.mean())

    return ColumnInfo(
        name=str(series.name),
        dtype=str(series.dtype),
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_count=unique_count,
        sample_values=_collect_sample_values(series),
        is_numeric=is_numeric,
        is_categorical=is_categorical,
        min=min_val,
        max=max_val,
        mean=mean_val,
    )


def _convert_cell(value: object) -> object:
    """Normalize a single cell value for inclusion in ``sample_rows``."""
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
    return value


def _build_sample_rows(df: pd.DataFrame, sample_size: int) -> list[dict[str, object]]:
    """Return the top ``sample_size`` rows as JSON-friendly dictionaries."""
    head = df.head(sample_size)
    rows: list[dict[str, object]] = []
    for record in head.to_dict(orient="records"):
        rows.append({str(k): _convert_cell(v) for k, v in record.items()})
    return rows


def describe_dataset(
    file_path: str,
    include_sample: bool = True,
    sample_size: int = 5,
) -> DescribeDatasetResult:
    """Describe a dataset file's schema and statistics.

    Reads a CSV / Parquet / Feather file from disk and returns column-level
    metadata (dtype, missing rate, unique count, sample values) along with
    optional preview rows. Intended for LLM clients that need to ground their
    suggestions on the actual data before generating analysis code.

    Args:
        file_path: Path to the data file. Relative paths are resolved.
        include_sample: If ``True`` (default), include the first ``sample_size``
            rows in ``sample_rows``.
        sample_size: Number of preview rows to include. Clipped to ``[0, 20]``.

    Returns:
        A :class:`DescribeDatasetResult`. Fatal issues (missing file,
        directory, unsupported extension, file too large, read failure) are
        reported via the ``error`` field instead of raising.
    """
    resolved = Path(file_path).resolve()
    path_str = str(resolved)

    if not resolved.exists():
        return _describe_error(path_str, f"File does not exist: {resolved}")
    if resolved.is_dir():
        return _describe_error(path_str, f"Path is a directory, not a file: {resolved}")
    if not resolved.is_file():
        return _describe_error(path_str, f"Path is not a regular file: {resolved}")

    ext = resolved.suffix.lower()
    if ext not in _DESCRIBE_SUPPORTED_EXTENSIONS:
        return _describe_error(
            path_str,
            f"Unsupported extension: {ext} (supported: .csv, .parquet, .feather)",
        )

    try:
        stat = resolved.stat()
    except OSError as exc:
        return _describe_error(path_str, f"Failed to stat file: {exc}")

    size_bytes = int(stat.st_size)
    if size_bytes > _MAX_FILE_SIZE_BYTES:
        return _describe_error(path_str, "ファイルサイズが大きすぎます (最大500MB)")

    clipped_sample_size = max(0, min(sample_size, _MAX_SAMPLE_SIZE))

    try:
        df = _read_dataframe(resolved, ext)
    except Exception as exc:
        return _describe_error(path_str, f"Failed to read file: {exc}")

    columns = [_build_column_info(df[col]) for col in df.columns]
    memory_usage = int(df.memory_usage(deep=True).sum()) if df.shape[1] > 0 else 0

    sample_rows: list[dict[str, object]] | None = None
    if include_sample:
        sample_rows = _build_sample_rows(df, clipped_sample_size)

    return DescribeDatasetResult(
        path=path_str,
        format=ext.lstrip("."),
        size_bytes=size_bytes,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        memory_usage_bytes=memory_usage,
        columns=columns,
        sample_rows=sample_rows,
        error=None,
    )


def register_dataset_tools(mcp: FastMCP) -> None:
    """Register dataset-related tools to the MCP server.

    Args:
        mcp: The FastMCP server instance to register tools on.
    """

    @mcp.tool()
    def list_datasets_tool(
        root_path: str,
        extensions: list[str] | None = None,
        max_depth: int = 5,
    ) -> ListDatasetsResult:
        """List data files (CSV, Parquet, Feather) under the given directory.

        Use this tool to discover what datasets exist in a project before
        analyzing them. Returns paths, sizes, and modification times.
        """
        return list_datasets(root_path, extensions, max_depth)

    @mcp.tool()
    def describe_dataset_tool(
        file_path: str,
        include_sample: bool = True,
        sample_size: int = 5,
    ) -> DescribeDatasetResult:
        """Describe a dataset file's schema and statistics.

        Returns column names, dtypes, missing rates, unique counts, and sample
        values. Use this tool BEFORE writing any code that references dataset
        columns, to avoid hallucinating column names.

        Supports CSV, Parquet, and Feather formats.
        """
        return describe_dataset(file_path, include_sample, sample_size)
