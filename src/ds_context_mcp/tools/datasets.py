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
_SAMPLE_ROWS_MAX_N: int = 100
_SAMPLE_ROWS_VALID_MODES: frozenset[str] = frozenset({"head", "tail", "random"})
_COLUMN_PROFILE_MAX_TOP_K: int = 50
_TOP_VALUE_MAX_LEN: int = 100

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


class SampleRowsResult(BaseModel):
    """Result of a sample_rows call.

    Attributes:
        path: Absolute resolved path of the file.
        format: ``"csv"`` / ``"parquet"`` / ``"feather"`` (empty on error).
        total_rows: Total number of rows in the file.
        returned_rows: Number of rows actually returned.
        mode: Sampling mode used (``"head"`` / ``"tail"`` / ``"random"``).
        columns: Column names in declaration order.
        rows: Sampled rows as JSON-friendly dicts. ``NaN`` becomes ``None``,
            datetime becomes ISO8601, bytes becomes ``str``.
        error: Fatal error message; set only on failure.
    """

    path: str
    format: str
    total_rows: int
    returned_rows: int
    mode: str
    columns: list[str]
    rows: list[dict[str, object]]
    error: str | None = None


class TopValue(BaseModel):
    """A single (value, count, frequency) triple in a column's frequency table.

    Attributes:
        value: Stringified value, truncated to 100 chars (with ``"..."``).
        count: Number of occurrences (non-NaN).
        frequency: ``count / total_count``, in ``[0.0, 1.0]``.
    """

    value: str
    count: int
    frequency: float


class NumericStats(BaseModel):
    """Numeric summary statistics for a single column.

    Attributes:
        min: Minimum value.
        max: Maximum value.
        mean: Arithmetic mean.
        median: Median (50th percentile).
        std: Sample standard deviation (``ddof=1``).
        q25: 25th percentile.
        q75: 75th percentile.
        outlier_count: Number of values outside ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``.
        has_negative: ``True`` if any value is negative.
        has_zero: ``True`` if any value equals zero.
    """

    min: float
    max: float
    mean: float
    median: float
    std: float
    q25: float
    q75: float
    outlier_count: int
    has_negative: bool
    has_zero: bool


class StringStats(BaseModel):
    """String length statistics for a single column.

    Attributes:
        min_length: Minimum string length among non-NaN values.
        max_length: Maximum string length among non-NaN values.
        mean_length: Mean string length among non-NaN values.
        empty_count: Number of empty-string occurrences.
    """

    min_length: int
    max_length: int
    mean_length: float
    empty_count: int


class ColumnProfileResult(BaseModel):
    """Result of a column_profile call.

    Attributes:
        path: Absolute resolved path of the file.
        column: Target column name.
        dtype: pandas dtype string.
        total_count: Number of non-NaN cells.
        missing_count: Number of NaN cells.
        missing_rate: Ratio of missing cells, in ``[0.0, 1.0]``.
        unique_count: Number of unique non-NaN values.
        cardinality_ratio: ``unique_count / total_count``; 1.0 means all unique.
        top_values: Frequency table sorted by descending count.
        numeric_stats: Set only for numeric columns.
        string_stats: Set only for string/object/category columns.
        error: Fatal error message; set only on failure.
    """

    path: str
    column: str
    dtype: str
    total_count: int
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality_ratio: float
    top_values: list[TopValue]
    numeric_stats: NumericStats | None = None
    string_stats: StringStats | None = None
    error: str | None = None


def _sample_rows_error(path: str, mode: str, message: str) -> SampleRowsResult:
    """Return a failure ``SampleRowsResult`` with zeroed metrics."""
    return SampleRowsResult(
        path=path,
        format="",
        total_rows=0,
        returned_rows=0,
        mode=mode,
        columns=[],
        rows=[],
        error=message,
    )


def _column_profile_error(path: str, column: str, message: str) -> ColumnProfileResult:
    """Return a failure ``ColumnProfileResult`` with zeroed metrics."""
    return ColumnProfileResult(
        path=path,
        column=column,
        dtype="",
        total_count=0,
        missing_count=0,
        missing_rate=0.0,
        unique_count=0,
        cardinality_ratio=0.0,
        top_values=[],
        numeric_stats=None,
        string_stats=None,
        error=message,
    )


def _validate_data_file(path_str: str) -> tuple[Path, str] | str:
    """Resolve and validate a data file path.

    Returns:
        A ``(resolved_path, ext)`` tuple on success, or an error message string.
    """
    resolved = Path(path_str).resolve()
    if not resolved.exists():
        return f"File does not exist: {resolved}"
    if resolved.is_dir():
        return f"Path is a directory, not a file: {resolved}"
    if not resolved.is_file():
        return f"Path is not a regular file: {resolved}"
    ext = resolved.suffix.lower()
    if ext not in _DESCRIBE_SUPPORTED_EXTENSIONS:
        return f"Unsupported extension: {ext} (supported: .csv, .parquet, .feather)"
    return resolved, ext


def _df_to_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to JSON-friendly row dicts."""
    rows: list[dict[str, object]] = []
    for record in df.to_dict(orient="records"):
        rows.append({str(k): _convert_cell(v) for k, v in record.items()})
    return rows


def sample_rows(
    file_path: str,
    n: int = 10,
    mode: str = "head",
    seed: int | None = None,
) -> SampleRowsResult:
    """Return sample rows from a dataset file.

    Reads a CSV / Parquet / Feather file and returns up to ``n`` rows according
    to ``mode``. Useful for inspecting actual data values once the schema is
    known via :func:`describe_dataset`.

    Args:
        file_path: Path to the data file. Relative paths are resolved.
        n: Number of rows to return. Clipped to ``[0, 100]``.
        mode: ``"head"`` (default), ``"tail"``, or ``"random"``.
        seed: Random seed used only when ``mode="random"``.

    Returns:
        A :class:`SampleRowsResult`. Fatal issues are reported via the
        ``error`` field instead of raising.
    """
    resolved_path = Path(file_path).resolve()
    path_str = str(resolved_path)

    validated = _validate_data_file(file_path)
    if isinstance(validated, str):
        return _sample_rows_error(path_str, mode, validated)
    resolved, ext = validated
    path_str = str(resolved)

    if mode not in _SAMPLE_ROWS_VALID_MODES:
        valid = ", ".join(sorted(_SAMPLE_ROWS_VALID_MODES))
        return _sample_rows_error(path_str, mode, f"Invalid mode: {mode!r} (valid: {valid})")

    try:
        stat = resolved.stat()
    except OSError as exc:
        return _sample_rows_error(path_str, mode, f"Failed to stat file: {exc}")
    if int(stat.st_size) > _MAX_FILE_SIZE_BYTES:
        return _sample_rows_error(path_str, mode, "ファイルサイズが大きすぎます (最大500MB)")

    try:
        df = _read_dataframe(resolved, ext)
    except Exception as exc:
        return _sample_rows_error(path_str, mode, f"Failed to read file: {exc}")

    total_rows = int(df.shape[0])
    clipped_n = max(0, min(n, _SAMPLE_ROWS_MAX_N))

    if mode == "head":
        sampled = df.head(clipped_n)
    elif mode == "tail":
        sampled = df.tail(clipped_n)
    else:
        take = min(clipped_n, total_rows)
        sampled = df.sample(n=take, random_state=seed) if take > 0 else df.head(0)

    rows = _df_to_rows(sampled)

    return SampleRowsResult(
        path=path_str,
        format=ext.lstrip("."),
        total_rows=total_rows,
        returned_rows=len(rows),
        mode=mode,
        columns=[str(c) for c in df.columns],
        rows=rows,
        error=None,
    )


def _truncate_top_value(text: str) -> str:
    """Truncate a top-value string to ``_TOP_VALUE_MAX_LEN`` chars."""
    if len(text) > _TOP_VALUE_MAX_LEN:
        return text[:_TOP_VALUE_MAX_LEN] + "..."
    return text


def _build_top_values(series: pd.Series, top_k: int, total_count: int) -> list[TopValue]:
    """Build a frequency table of the top ``top_k`` non-NaN values."""
    if total_count == 0:
        return []
    counts = series.dropna().value_counts().head(top_k)
    result: list[TopValue] = []
    for raw_value, raw_count in counts.items():
        count_int = int(raw_count)
        result.append(
            TopValue(
                value=_truncate_top_value(str(raw_value)),
                count=count_int,
                frequency=count_int / total_count,
            )
        )
    return result


def _build_numeric_stats(series: pd.Series) -> NumericStats | None:
    """Compute :class:`NumericStats` for a numeric series, or ``None`` if empty."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((non_null < lower) | (non_null > upper)).sum())
    std_value = non_null.std(ddof=1)
    std_float = 0.0 if pd.isna(std_value) else float(std_value)
    return NumericStats(
        min=float(non_null.min()),
        max=float(non_null.max()),
        mean=float(non_null.mean()),
        median=float(non_null.median()),
        std=std_float,
        q25=q1,
        q75=q3,
        outlier_count=outlier_count,
        has_negative=bool((non_null < 0).any()),
        has_zero=bool((non_null == 0).any()),
    )


def _build_string_stats(series: pd.Series) -> StringStats | None:
    """Compute :class:`StringStats` for a string-like series, or ``None`` if empty."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    lengths = non_null.astype(str).str.len()
    empty_count = int((non_null.astype(str) == "").sum())
    return StringStats(
        min_length=int(lengths.min()),
        max_length=int(lengths.max()),
        mean_length=float(lengths.mean()),
        empty_count=empty_count,
    )


def column_profile(
    file_path: str,
    column: str,
    top_k: int = 10,
) -> ColumnProfileResult:
    """Profile a single column in depth.

    Returns missing rate, cardinality, top values, and dtype-specific stats
    (numeric or string). Designed to be called after :func:`describe_dataset`
    when one column needs deeper inspection.

    Args:
        file_path: Path to the data file. Relative paths are resolved.
        column: Name of the column to profile.
        top_k: Maximum number of frequency entries to return. Clipped to
            ``[0, 50]``.

    Returns:
        A :class:`ColumnProfileResult`. Missing column / file errors are
        reported via the ``error`` field instead of raising.
    """
    resolved_path = Path(file_path).resolve()
    path_str = str(resolved_path)

    validated = _validate_data_file(file_path)
    if isinstance(validated, str):
        return _column_profile_error(path_str, column, validated)
    resolved, ext = validated
    path_str = str(resolved)

    try:
        stat = resolved.stat()
    except OSError as exc:
        return _column_profile_error(path_str, column, f"Failed to stat file: {exc}")
    if int(stat.st_size) > _MAX_FILE_SIZE_BYTES:
        return _column_profile_error(path_str, column, "ファイルサイズが大きすぎます (最大500MB)")

    try:
        df = _read_dataframe(resolved, ext)
    except Exception as exc:
        return _column_profile_error(path_str, column, f"Failed to read file: {exc}")

    if column not in df.columns:
        available = [str(c) for c in df.columns]
        return _column_profile_error(
            path_str,
            column,
            f"Column {column!r} not found. Available: {available}",
        )

    series = df[column]
    total = len(series)
    missing_count = int(series.isna().sum())
    total_count = total - missing_count
    missing_rate = (missing_count / total) if total > 0 else 0.0
    unique_count = int(series.dropna().nunique())
    cardinality_ratio = (unique_count / total_count) if total_count > 0 else 0.0

    clipped_top_k = max(0, min(top_k, _COLUMN_PROFILE_MAX_TOP_K))
    top_values = _build_top_values(series, clipped_top_k, total_count)

    numeric_stats: NumericStats | None = None
    string_stats: StringStats | None = None
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    is_string_like = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    if total_count > 0:
        if is_numeric:
            numeric_stats = _build_numeric_stats(series)
        elif is_string_like:
            string_stats = _build_string_stats(series)

    return ColumnProfileResult(
        path=path_str,
        column=column,
        dtype=str(series.dtype),
        total_count=total_count,
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_count=unique_count,
        cardinality_ratio=cardinality_ratio,
        top_values=top_values,
        numeric_stats=numeric_stats,
        string_stats=string_stats,
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

    @mcp.tool()
    def sample_rows_tool(
        file_path: str,
        n: int = 10,
        mode: str = "head",
        seed: int | None = None,
    ) -> SampleRowsResult:
        """Return sample rows from a dataset file (head/tail/random).

        Use this tool to inspect actual data values when you need to see what's
        in the rows, not just the schema. Prefer describe_dataset first for
        schema, then sample_rows for concrete values.

        Supports CSV, Parquet, and Feather formats.
        """
        return sample_rows(file_path, n, mode, seed)

    @mcp.tool()
    def column_profile_tool(
        file_path: str,
        column: str,
        top_k: int = 10,
    ) -> ColumnProfileResult:
        """Profile a single column in depth: stats, top values, outliers.

        Use this tool AFTER describe_dataset when you need detailed analysis of
        one specific column (distribution, outliers, cardinality, frequency of
        values). Essential before writing visualization or encoding code.

        Supports CSV, Parquet, and Feather formats.
        """
        return column_profile(file_path, column, top_k)
