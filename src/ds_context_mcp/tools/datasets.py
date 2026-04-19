"""Dataset discovery and description tools."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

DEFAULT_EXTENSIONS: tuple[str, ...] = (".csv", ".parquet", ".feather")

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
