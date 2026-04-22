"""Tests for the detect_relations tool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ds_context_mcp.tools.relations import (
    DetectRelationsResult,
    FileColumns,
    RelationCandidate,
    detect_relations,
    normalize_column_name,
)


def _make_csv(path: Path, columns: list[str]) -> None:
    """Create a 1-row CSV with the given column names."""
    df = pd.DataFrame({c: [1] for c in columns})
    df.to_csv(path, index=False)


# --- normalize_column_name helper ---------------------------------------------


def test_normalize_column_name_snake_case() -> None:
    assert normalize_column_name("user_id") == "user_id"


def test_normalize_column_name_camel_case() -> None:
    assert normalize_column_name("userId") == "user_id"


def test_normalize_column_name_pascal_case() -> None:
    assert normalize_column_name("UserId") == "user_id"


def test_normalize_column_name_upper_snake() -> None:
    assert normalize_column_name("USER_ID") == "user_id"


def test_normalize_column_name_mixed() -> None:
    assert normalize_column_name("User_ID") == "user_id"


# --- detect_relations ---------------------------------------------------------


def test_exact_match_single_pair(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id", "name"])
    _make_csv(b, ["user_id", "email"])

    result = detect_relations([str(a), str(b)])

    assert isinstance(result, DetectRelationsResult)
    assert result.error is None
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert isinstance(cand, RelationCandidate)
    assert cand.column_a == "user_id"
    assert cand.column_b == "user_id"
    assert cand.match_type == "exact"
    assert cand.confidence == 1.0
    assert cand.reason == "exact match"


def test_exact_match_multiple(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    c = tmp_path / "c.csv"
    _make_csv(a, ["user_id"])
    _make_csv(b, ["user_id"])
    _make_csv(c, ["user_id"])

    result = detect_relations([str(a), str(b), str(c)])

    assert result.error is None
    assert len(result.candidates) == 3
    for cand in result.candidates:
        assert cand.match_type == "exact"
        assert cand.confidence == 1.0
        assert cand.column_a == "user_id"
        assert cand.column_b == "user_id"


def test_case_variant(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["User_ID"])
    _make_csv(b, ["user_id"])

    result = detect_relations([str(a), str(b)])

    assert result.error is None
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.match_type == "fuzzy"
    assert cand.confidence == 0.95
    assert cand.reason == "case_variant"


def test_snake_vs_camel(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id"])
    _make_csv(b, ["userId"])

    result = detect_relations([str(a), str(b)])

    assert result.error is None
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.match_type == "fuzzy"
    assert cand.confidence == 0.90
    assert cand.reason == "naming_convention_variant"


def test_id_suffix_match(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id"])
    _make_csv(b, ["user"])

    result = detect_relations([str(a), str(b)])

    assert result.error is None
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.match_type == "fuzzy"
    assert cand.confidence == 0.80
    assert cand.reason == "id_suffix_match"


def test_exact_takes_priority(tmp_path: Path) -> None:
    # "user_id" matches user_id exactly AND also satisfies case/normalize rules.
    # Only the exact candidate should be returned for that pair.
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id"])
    _make_csv(b, ["user_id"])

    result = detect_relations([str(a), str(b)])

    assert result.error is None
    assert len(result.candidates) == 1
    assert result.candidates[0].match_type == "exact"
    assert result.candidates[0].reason == "exact match"


def test_include_exact_false(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id", "Name"])
    _make_csv(b, ["user_id", "name"])

    result = detect_relations([str(a), str(b)], include_exact=False)

    assert result.error is None
    # user_id <-> user_id (exact) dropped; Name <-> name (case_variant) kept.
    assert len(result.candidates) == 1
    assert result.candidates[0].match_type == "fuzzy"
    assert result.candidates[0].reason == "case_variant"


def test_include_fuzzy_false(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id", "Name"])
    _make_csv(b, ["user_id", "name"])

    result = detect_relations([str(a), str(b)], include_fuzzy=False)

    assert result.error is None
    # Only exact match kept.
    assert len(result.candidates) == 1
    assert result.candidates[0].match_type == "exact"


def test_confidence_ordering(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id", "Name", "order_id", "product"])
    _make_csv(b, ["user_id", "name", "orderId", "product_id"])

    result = detect_relations([str(a), str(b)])

    assert result.error is None
    # Expect 4 candidates: exact / case_variant / naming_convention_variant / id_suffix_match
    assert len(result.candidates) == 4
    confidences = [c.confidence for c in result.candidates]
    assert confidences == sorted(confidences, reverse=True)
    assert confidences == [1.0, 0.95, 0.90, 0.80]


def test_single_file_returns_error(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    _make_csv(a, ["user_id"])

    result = detect_relations([str(a)])

    assert result.error is not None
    assert result.candidates == []


def test_unreadable_file_skipped(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    missing = tmp_path / "missing.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["user_id"])
    _make_csv(b, ["user_id"])

    result = detect_relations([str(a), str(missing), str(b)])

    assert result.error is None
    errored = [f for f in result.files if f.error is not None]
    assert len(errored) == 1
    assert isinstance(errored[0], FileColumns)
    assert "missing" in errored[0].path
    # Valid files still produce the candidate between a and b.
    assert len(result.candidates) == 1
    assert result.candidates[0].match_type == "exact"


def test_no_common_columns(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _make_csv(a, ["alpha", "beta"])
    _make_csv(b, ["gamma", "delta"])

    result = detect_relations([str(a), str(b)])

    assert result.error is None
    assert result.candidates == []
