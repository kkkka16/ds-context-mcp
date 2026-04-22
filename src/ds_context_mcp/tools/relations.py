"""Cross-file relation detection tools."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".parquet", ".feather"})

_CONFIDENCE_EXACT: float = 1.0
_CONFIDENCE_CASE_VARIANT: float = 0.95
_CONFIDENCE_NAMING_VARIANT: float = 0.90
_CONFIDENCE_ID_SUFFIX: float = 0.80

_REASON_EXACT: str = "exact match"
_REASON_CASE_VARIANT: str = "case_variant"
_REASON_NAMING_VARIANT: str = "naming_convention_variant"
_REASON_ID_SUFFIX: str = "id_suffix_match"

_MATCH_TYPE_EXACT: str = "exact"
_MATCH_TYPE_FUZZY: str = "fuzzy"

_CAMEL_BOUNDARY_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_BASIC = re.compile(r"([a-z0-9])([A-Z])")
_MULTI_UNDERSCORE = re.compile(r"_+")


class RelationCandidate(BaseModel):
    """A single candidate JOIN column pair across two dataset files.

    Attributes:
        file_a: Absolute path to the first file in the pair.
        file_b: Absolute path to the second file in the pair.
        column_a: Column name in ``file_a``.
        column_b: Column name in ``file_b``.
        match_type: ``"exact"`` or ``"fuzzy"``.
        confidence: Score in ``[0.0, 1.0]``.
        reason: Short label explaining why the pair was flagged.
    """

    file_a: str
    file_b: str
    column_a: str
    column_b: str
    match_type: str
    confidence: float
    reason: str


class FileColumns(BaseModel):
    """Column listing for one input file.

    Attributes:
        path: Absolute resolved path of the file.
        columns: Column names in declaration order. Empty if read failed.
        error: Read error message; ``None`` when the file was read successfully.
    """

    path: str
    columns: list[str]
    error: str | None = None


class DetectRelationsResult(BaseModel):
    """Result of a detect_relations call.

    Attributes:
        files: Per-file column listings (including read failures).
        candidates: JOIN candidates sorted by descending confidence.
        error: Fatal error message; set only when detection could not proceed.
    """

    files: list[FileColumns]
    candidates: list[RelationCandidate]
    error: str | None = None


def normalize_column_name(name: str) -> str:
    """Normalize a column name to lowercase snake_case.

    Converts camelCase, PascalCase, UPPER_SNAKE, and mixed forms to a single
    lowercase snake_case representation, enabling cross-naming-convention
    equality checks.

    Args:
        name: Original column name.

    Returns:
        Lowercase snake_case equivalent (no leading/trailing underscores,
        no consecutive underscores).
    """
    s = _CAMEL_BOUNDARY_ACRONYM.sub(r"\1_\2", name)
    s = _CAMEL_BOUNDARY_BASIC.sub(r"\1_\2", s)
    s = s.lower()
    s = _MULTI_UNDERSCORE.sub("_", s)
    return s.strip("_")


def _strip_id_affix(normalized: str) -> str:
    """Strip a trailing ``_id`` suffix or leading ``id_`` prefix."""
    if normalized.endswith("_id"):
        return normalized[:-3]
    if normalized.startswith("id_"):
        return normalized[3:]
    return normalized


def _classify_match(
    col_a: str,
    col_b: str,
) -> tuple[str, float, str] | None:
    """Classify the relationship between two column names.

    Returns:
        ``(match_type, confidence, reason)`` if a relationship was found,
        otherwise ``None``. Higher-priority categories are checked first so
        each pair receives at most one classification.
    """
    if col_a == col_b:
        return (_MATCH_TYPE_EXACT, _CONFIDENCE_EXACT, _REASON_EXACT)
    if col_a.lower() == col_b.lower():
        return (_MATCH_TYPE_FUZZY, _CONFIDENCE_CASE_VARIANT, _REASON_CASE_VARIANT)
    norm_a = normalize_column_name(col_a)
    norm_b = normalize_column_name(col_b)
    if norm_a and norm_a == norm_b:
        return (_MATCH_TYPE_FUZZY, _CONFIDENCE_NAMING_VARIANT, _REASON_NAMING_VARIANT)
    stripped_a = _strip_id_affix(norm_a)
    stripped_b = _strip_id_affix(norm_b)
    affix_was_stripped = stripped_a != norm_a or stripped_b != norm_b
    if affix_was_stripped and stripped_a and stripped_a == stripped_b:
        return (_MATCH_TYPE_FUZZY, _CONFIDENCE_ID_SUFFIX, _REASON_ID_SUFFIX)
    return None


def _read_file_columns(path: Path, ext: str) -> list[str]:
    """Read only the column names from a data file."""
    if ext == ".csv":
        df = pd.read_csv(path, nrows=0)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return [str(c) for c in df.columns]


def _load_file_columns(path_str: str) -> FileColumns:
    """Resolve a path and load its column names, capturing any error."""
    path = Path(path_str)
    if not path.exists():
        return FileColumns(path=path_str, columns=[], error=f"File does not exist: {path}")
    if path.is_dir():
        return FileColumns(
            path=path_str,
            columns=[],
            error=f"Path is a directory, not a file: {path}",
        )
    if not path.is_file():
        return FileColumns(
            path=path_str,
            columns=[],
            error=f"Path is not a regular file: {path}",
        )
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        return FileColumns(
            path=path_str,
            columns=[],
            error=f"Unsupported extension: {ext} (supported: .csv, .parquet, .feather)",
        )
    try:
        cols = _read_file_columns(path, ext)
    except Exception as exc:
        return FileColumns(path=path_str, columns=[], error=f"Failed to read file: {exc}")
    return FileColumns(path=path_str, columns=cols)


def _dedupe_resolved_paths(file_paths: list[str]) -> list[str]:
    """Resolve and deduplicate paths while preserving the input order."""
    seen: set[str] = set()
    unique: list[str] = []
    for raw in file_paths:
        resolved = str(Path(raw).resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def detect_relations(
    file_paths: list[str],
    include_exact: bool = True,
    include_fuzzy: bool = True,
) -> DetectRelationsResult:
    """Detect JOIN candidate columns across multiple dataset files.

    Reads only the column names from each file (CSV / Parquet / Feather) and
    returns column pairs that likely represent the same entity. Useful as a
    preflight before writing any merge / join code so the LLM uses the actual
    column names rather than guessed ones.

    Args:
        file_paths: Two or more dataset file paths. Duplicate paths (after
            resolution) are collapsed.
        include_exact: When ``True``, include exact column-name matches
            (``confidence=1.0``).
        include_fuzzy: When ``True``, include naming-variant matches
            (case_variant, naming_convention_variant, id_suffix_match).

    Returns:
        A :class:`DetectRelationsResult`. Per-file read errors are surfaced
        in ``files[*].error`` rather than raised; only fatal input errors
        (e.g. fewer than 2 unique paths) populate the top-level ``error``.
    """
    unique_paths = _dedupe_resolved_paths(file_paths)

    if len(unique_paths) < 2:
        return DetectRelationsResult(
            files=[FileColumns(path=p, columns=[]) for p in unique_paths],
            candidates=[],
            error="少なくとも2つのファイルが必要です",
        )

    files: list[FileColumns] = [_load_file_columns(p) for p in unique_paths]
    readable: list[FileColumns] = [f for f in files if f.error is None and f.columns]

    candidates: list[RelationCandidate] = []
    for i in range(len(readable)):
        for j in range(i + 1, len(readable)):
            file_a = readable[i]
            file_b = readable[j]
            for col_a in file_a.columns:
                for col_b in file_b.columns:
                    classification = _classify_match(col_a, col_b)
                    if classification is None:
                        continue
                    match_type, confidence, reason = classification
                    if match_type == _MATCH_TYPE_EXACT and not include_exact:
                        continue
                    if match_type == _MATCH_TYPE_FUZZY and not include_fuzzy:
                        continue
                    candidates.append(
                        RelationCandidate(
                            file_a=file_a.path,
                            file_b=file_b.path,
                            column_a=col_a,
                            column_b=col_b,
                            match_type=match_type,
                            confidence=confidence,
                            reason=reason,
                        )
                    )

    candidates.sort(key=lambda c: (-c.confidence, c.file_a, c.column_a, c.file_b, c.column_b))

    return DetectRelationsResult(files=files, candidates=candidates, error=None)


def register_relation_tools(mcp: FastMCP) -> None:
    """Register relation-detection tools to the MCP server.

    Args:
        mcp: The FastMCP server instance to register tools on.
    """

    @mcp.tool()
    def detect_relations_tool(
        file_paths: list[str],
        include_exact: bool = True,
        include_fuzzy: bool = True,
    ) -> DetectRelationsResult:
        """Detect JOIN candidate columns across multiple dataset files.

        Use this tool when you have multiple CSV/Parquet files and need to know
        how they can be joined together. Returns column pairs that likely
        represent the same entity (exact matches or naming-convention variants).

        Use this BEFORE writing any JOIN/merge code to ensure you use the
        correct column names.
        """
        return detect_relations(file_paths, include_exact, include_fuzzy)
