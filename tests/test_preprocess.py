"""Tests for preprocess.py: suggest_columns,
case-insensitive Likert, structured validation."""

import pandas as pd
import numpy as np
import pytest

from labgroupassigner.preprocess import (
    suggest_columns,
    prepare,
)
from labgroupassigner.schema import ColumnSpec
from labgroupassigner.errors import (
    ValidationError,
    SolverError,
)


# --- suggest_columns ---

def test_suggest_columns_standard_fixture():
    df = pd.read_csv("examples/test_roster.csv")
    spec = suggest_columns(df)
    assert spec.label_col == "Preferred_name"
    assert spec.balance_col == "Pronoun"
    assert len(spec.score_cols) == 5


def test_suggest_columns_no_match():
    df = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4],
    })
    spec = suggest_columns(df)
    assert spec.score_cols == []
    assert spec.label_col == "A"
    assert spec.balance_col is None


# --- Case-insensitive Likert ---

def _make_roster(n=8, likert="Mostly confident"):
    """Build a minimal valid roster DataFrame."""
    rows = []
    pronouns = (
        ["she"] * (n // 2)
        + ["he"] * (n - n // 2)
    )
    for i in range(n):
        rows.append({
            "Preferred_name": f"Student_{i}",
            "Pronoun": pronouns[i],
            **{
                col: likert
                for col in [
                    "col_record scientific data",
                    "col_work effectively with "
                    "others in a team",
                    "col_use basic lab equipment",
                    "col_communicate and coordinate",
                    "col_serving as a test subject",
                ]
            },
        })
    return pd.DataFrame(rows)


def _spec_for(df):
    return suggest_columns(df)


def test_case_insensitive_likert():
    df = _make_roster(8, "mostly confident")
    spec = _spec_for(df)
    data = prepare(df, spec)
    # All values should recode to 4
    assert (data["cat_scores"] == 4.0).all()


def test_case_insensitive_likert_upper():
    df = _make_roster(8, "MOSTLY CONFIDENT")
    spec = _spec_for(df)
    data = prepare(df, spec)
    assert (data["cat_scores"] == 4.0).all()


# --- Structured validation ---

def test_bad_likert_raises_validation_error():
    df = _make_roster(8)
    # Damage one cell
    score_col = [
        c for c in df.columns
        if "record" in c
    ][0]
    df.loc[2, score_col] = "INVALID_VALUE"
    spec = _spec_for(df)

    with pytest.raises(ValidationError) as exc_info:
        prepare(df, spec)

    issues = exc_info.value.issues
    assert len(issues) >= 1
    assert issues[0].severity == "error"
    assert "INVALID_VALUE" in issues[0].message
    assert 2 in issues[0].row_indices


def test_multiple_issues_reported():
    df = _make_roster(8)
    cols = [
        c for c in df.columns
        if "record" in c or "work" in c.lower()
    ]
    # Damage two different columns
    df.loc[0, cols[0]] = "BAD1"
    df.loc[1, cols[1]] = "BAD2"
    spec = _spec_for(df)

    with pytest.raises(ValidationError) as exc_info:
        prepare(df, spec)

    issues = exc_info.value.issues
    error_issues = [
        i for i in issues if i.severity == "error"
    ]
    assert len(error_issues) == 2


def test_missing_pronoun_is_warning():
    df = _make_roster(8)
    df.loc[3, "Pronoun"] = np.nan
    spec = _spec_for(df)

    data = prepare(df, spec)
    warnings = [
        i for i in data["issues"]
        if i.severity == "warning"
        and "pronoun" in i.message.lower()
    ]
    assert len(warnings) == 1
    assert 3 in warnings[0].row_indices


def test_imputed_cells_reported():
    df = _make_roster(8)
    score_col = [
        c for c in df.columns
        if "record" in c
    ][0]
    df.loc[1, score_col] = np.nan
    spec = _spec_for(df)

    data = prepare(df, spec)
    assert "Q1" in data["imputed_cells"]
    assert 1 in data["imputed_cells"]["Q1"]
