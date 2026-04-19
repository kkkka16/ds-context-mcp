"""Tests for the list_datasets tool."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from ds_context_mcp.tools.datasets import (
    DatasetEntry,
    ListDatasetsResult,
    list_datasets,
)


def _make_file(path: Path, content: bytes = b"x") -> None:
    """Create a file with parents, for test fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_list_csv_parquet_feather_mixed(tmp_path: Path) -> None:
    _make_file(tmp_path / "a.csv")
    _make_file(tmp_path / "b.parquet")
    _make_file(tmp_path / "c.feather")
    _make_file(tmp_path / "readme.txt")

    result = list_datasets(str(tmp_path))

    assert isinstance(result, ListDatasetsResult)
    assert result.error is None
    assert result.count == 3
    assert sorted(d.format for d in result.datasets) == ["csv", "feather", "parquet"]


def test_recursive_search(tmp_path: Path) -> None:
    _make_file(tmp_path / "top.csv")
    _make_file(tmp_path / "sub" / "nested.parquet")
    _make_file(tmp_path / "sub" / "deeper" / "deep.feather")

    result = list_datasets(str(tmp_path))

    assert result.count == 3
    rels = sorted(d.relative_path for d in result.datasets)
    expected = sorted(
        [
            "top.csv",
            str(Path("sub") / "nested.parquet"),
            str(Path("sub") / "deeper" / "deep.feather"),
        ]
    )
    assert rels == expected


def test_extension_filter(tmp_path: Path) -> None:
    _make_file(tmp_path / "a.csv")
    _make_file(tmp_path / "b.parquet")
    _make_file(tmp_path / "c.feather")

    result = list_datasets(str(tmp_path), extensions=[".csv"])

    assert result.count == 1
    assert result.datasets[0].format == "csv"


def test_excludes_hidden_and_venv(tmp_path: Path) -> None:
    _make_file(tmp_path / "visible.csv")
    _make_file(tmp_path / ".git" / "config.csv")
    _make_file(tmp_path / ".venv" / "lib.csv")
    _make_file(tmp_path / "venv" / "lib.csv")
    _make_file(tmp_path / "__pycache__" / "cache.csv")
    _make_file(tmp_path / ".mypy_cache" / "m.csv")
    _make_file(tmp_path / ".pytest_cache" / "p.csv")
    _make_file(tmp_path / ".ruff_cache" / "r.csv")
    _make_file(tmp_path / "node_modules" / "n.csv")
    _make_file(tmp_path / ".ipynb_checkpoints" / "i.csv")
    _make_file(tmp_path / ".hidden" / "secret.csv")

    result = list_datasets(str(tmp_path))

    assert result.count == 1
    assert result.datasets[0].relative_path == "visible.csv"


def test_max_depth_limit(tmp_path: Path) -> None:
    _make_file(tmp_path / "shallow.csv")
    _make_file(tmp_path / "a" / "b" / "c" / "deep.csv")

    result = list_datasets(str(tmp_path), max_depth=1)

    rels = [d.relative_path for d in result.datasets]
    assert "shallow.csv" in rels
    assert not any("deep.csv" in r for r in rels)


def test_nonexistent_path_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"

    result = list_datasets(str(missing))

    assert result.error is not None
    assert result.count == 0
    assert result.datasets == []


def test_file_path_instead_of_dir_returns_error(tmp_path: Path) -> None:
    file_path = tmp_path / "single.csv"
    _make_file(file_path)

    result = list_datasets(str(file_path))

    assert result.error is not None
    assert result.count == 0
    assert result.datasets == []


def test_empty_directory(tmp_path: Path) -> None:
    result = list_datasets(str(tmp_path))

    assert result.error is None
    assert result.count == 0
    assert result.datasets == []


def test_datasets_entry_fields(tmp_path: Path) -> None:
    file_path = tmp_path / "data.csv"
    content = b"col1,col2\n1,2\n"
    _make_file(file_path, content)

    result = list_datasets(str(tmp_path))

    assert result.count == 1
    entry = result.datasets[0]
    assert isinstance(entry, DatasetEntry)
    assert entry.format == "csv"
    assert entry.size_bytes == len(content)
    assert entry.path == str(file_path.resolve())
    assert entry.relative_path == "data.csv"
    parsed = datetime.fromisoformat(entry.modified_at)
    assert parsed.tzinfo is not None


def test_symlink_not_followed(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / f"outside_{os.getpid()}_{tmp_path.name}"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "external.csv"
    outside_file.write_bytes(b"external")

    root = tmp_path / "root"
    root.mkdir()
    _make_file(root / "real.csv")

    link = root / "link.csv"
    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        outside_file.unlink(missing_ok=True)
        outside_dir.rmdir()
        pytest.skip("Symlinks not supported on this platform")

    try:
        result = list_datasets(str(root))
        rels = [d.relative_path for d in result.datasets]
        assert "real.csv" in rels
        assert "link.csv" not in rels
    finally:
        link.unlink(missing_ok=True)
        outside_file.unlink(missing_ok=True)
        outside_dir.rmdir()
