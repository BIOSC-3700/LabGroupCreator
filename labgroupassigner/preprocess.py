import numpy as np
import pandas as pd
from pathlib import Path

from labgroupassigner.errors import (
    SolverError,
    ValidationError,
    Issue,
)
from labgroupassigner.derive import normalize_pronoun
from labgroupassigner.schema import ColumnSpec
from labgroupassigner.model import group_sizes


COLUMN_PATTERNS = {
    "Collect": "record scientific data",
    "Work": "work effectively with others in a team",
    "Lab": "use basic lab equipment",
    "Communicate": "communicate and coordinate",
    "Subject": "serving as a test subject",
}


LIKERT_MAP = {
    "Very confident": 5,
    "Very comfortable": 5,
    "Mostly confident": 4,
    "Mostly comfortable": 4,
    "Somewhat confident": 3,
    "Somewhat comfortable": 3,
    "Slightly confident": 2,
    "Slightly comfortable": 2,
    "Not confident yet": 1,
    "Not comfortable yet": 1,
}

# Case-insensitive + whitespace-normalized lookup
_LIKERT_MAP_LOWER = {
    " ".join(k.lower().split()): v
    for k, v in LIKERT_MAP.items()
}


def _find_column(df, pattern):
    """Find exactly one column containing pattern
    (case-insensitive)."""
    lower_pattern = pattern.lower()
    matches = [
        c for c in df.columns
        if lower_pattern in c.lower()
    ]
    if len(matches) != 1:
        raise SolverError(
            f"Expected 1 column matching "
            f"'{pattern}', "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _find_column_safe(df, pattern):
    """Find a column by substring, returning None on
    no match or ambiguity."""
    lower_pattern = pattern.lower()
    matches = [
        c for c in df.columns
        if lower_pattern in c.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def suggest_columns(df):
    """Guess column roles from a DataFrame.

    Returns a best-guess ColumnSpec. Never raises.
    """
    # Try to find score columns
    score_cols = []
    for pattern in COLUMN_PATTERNS.values():
        col = _find_column_safe(df, pattern)
        if col is not None:
            score_cols.append(col)

    # Label column: prefer "Preferred_name"
    label_col = None
    for candidate in [
        "Preferred_name", "preferred_name",
        "Name", "name",
    ]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None and len(df.columns) > 0:
        label_col = df.columns[0]

    # Name column (full name, optional)
    name_col = None
    lower_cols = {c.lower(): c for c in df.columns}
    if "name" in lower_cols:
        candidate = lower_cols["name"]
        if candidate != label_col:
            name_col = candidate

    # Balance column: prefer "Pronoun"
    balance_col = None
    for candidate in ["Pronoun", "pronoun"]:
        if candidate in df.columns:
            balance_col = candidate
            break

    return ColumnSpec(
        name_col=name_col,
        label_col=label_col,
        score_cols=score_cols,
        balance_col=balance_col,
    )


def prepare(df, spec, config=None, status_callback=None):
    """Prepare a DataFrame for optimization.

    Pure transform: no file I/O, no printing except
    through status_callback.

    Returns a dict with all arrays needed by the model.
    """
    log = status_callback or (lambda m: None)

    # Validate required columns exist
    if spec.label_col not in df.columns:
        raise SolverError(
            f"Label column '{spec.label_col}' "
            f"not found"
        )
    if len(spec.score_cols) < 1:
        raise SolverError(
            "At least one score column is required"
        )

    categories = [
        f"Q{i + 1}"
        for i in range(len(spec.score_cols))
    ]

    for col in spec.score_cols:
        if col not in df.columns:
            raise SolverError(
                f"Score column '{col}' not found"
            )

    # Build working DataFrame with short category names
    col_map = dict(zip(spec.score_cols, categories))
    keep_cols = [spec.label_col]
    if spec.balance_col and (
        spec.balance_col in df.columns
    ):
        keep_cols.append(spec.balance_col)
    if spec.name_col and spec.name_col in df.columns:
        keep_cols.append(spec.name_col)
    keep_cols += list(col_map.keys())

    # Deduplicate while preserving order
    seen = set()
    unique_cols = []
    for c in keep_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    keep_cols = unique_cols

    wdf = df[keep_cols].copy().rename(columns=col_map)

    # Rename label/balance cols for internal consistency
    if spec.label_col != "Preferred_name":
        wdf = wdf.rename(
            columns={spec.label_col: "Preferred_name"}
        )
    if spec.balance_col and (
        spec.balance_col != "Pronoun"
    ):
        wdf = wdf.rename(
            columns={spec.balance_col: "Pronoun"}
        )

    # --- Structured validation ---
    issues = []

    # Recode Likert text to numeric (case-insensitive)
    imputed_cells = {}
    for cat in categories:
        def _recode(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            key = " ".join(str(val).lower().split())
            mapped = _LIKERT_MAP_LOWER.get(key)
            if mapped is not None:
                return mapped
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan

        original = wdf[cat].copy()
        wdf[cat] = wdf[cat].apply(_recode)

        # Detect out-of-map values (not originally NaN
        # but now NaN)
        bad_mask = wdf[cat].isna() & ~original.isna()
        if bad_mask.any():
            bad_rows = list(
                wdf.index[bad_mask].tolist()
            )
            bad_vals = (
                original[bad_mask].unique().tolist()
            )
            issues.append(Issue(
                severity="error",
                message=(
                    f"Unrecognized Likert values in "
                    f"'{cat}': {bad_vals} "
                    f"(rows {bad_rows})"
                ),
                row_indices=bad_rows,
            ))

    # Impute remaining NaN with column means
    for cat in categories:
        na_mask = wdf[cat].isna()
        if na_mask.any():
            col_mean = wdf[cat].mean()
            imputed_rows = list(
                wdf.index[na_mask].tolist()
            )
            imputed_cells[cat] = imputed_rows
            wdf[cat] = wdf[cat].fillna(col_mean)
            issues.append(Issue(
                severity="warning",
                message=(
                    f"Imputed {len(imputed_rows)} "
                    f"missing value(s) in '{cat}' "
                    f"with column mean "
                    f"(rows {imputed_rows})"
                ),
                row_indices=imputed_rows,
            ))

    # Validate pronouns (missing = warning, not error)
    has_pronoun = "Pronoun" in wdf.columns
    if has_pronoun and wdf["Pronoun"].isna().any():
        na_rows = list(
            wdf.index[wdf["Pronoun"].isna()].tolist()
        )
        issues.append(Issue(
            severity="warning",
            message=(
                f"Missing pronoun data for "
                f"{len(na_rows)} student(s) "
                f"(rows {na_rows}); "
                f"treated as Unknown"
            ),
            row_indices=na_rows,
        ))
        wdf.loc[
            wdf["Pronoun"].isna(), "Pronoun"
        ] = "Unknown"

    n_students = len(wdf)
    sizes = group_sizes(n_students)

    if sizes is None:
        raise SolverError(
            f"Need at least 6 students "
            f"(found {n_students})"
        )

    n_groups = len(sizes)
    n_groups_of_3 = sizes.count(3)
    if n_groups_of_3 > 0:
        issues.append(Issue(
            severity="warning",
            message=(
                f"{n_groups_of_3} group(s) of 3 "
                f"(total {n_groups} groups)"
            ),
        ))

    # Raise if any errors accumulated
    errors = [
        i for i in issues if i.severity == "error"
    ]
    if errors:
        raise ValidationError(errors)

    # Normalize pronouns to canonical values
    if has_pronoun:
        wdf["Pronoun"] = wdf["Pronoun"].apply(
            lambda p: normalize_pronoun(p)[0]
        )

    # Numeric arrays
    cat_scores = (
        wdf[categories].to_numpy(dtype=float)
    )
    total_scores = cat_scores.sum(axis=1)

    if has_pronoun:
        is_she = (
            wdf["Pronoun"].isin(["She", "Unknown"])
        ).astype(int).to_numpy()
    else:
        is_she = np.zeros(n_students, dtype=int)

    # Same first-name pairs
    first_names = (
        wdf["Preferred_name"]
        .str.split()
        .str[0]
        .str.lower()
    )
    same_name_pairs = []
    for i in range(n_students - 1):
        for j in range(i + 1, n_students):
            if (
                first_names.iloc[i]
                == first_names.iloc[j]
            ):
                same_name_pairs.append((i, j))

    # Pronoun constraint feasibility
    n_she = int(is_she.sum())
    n_he = n_students - n_she
    use_pronoun_constraint = (
        n_she == 0 or n_she >= 2
    )
    use_he_constraint = (
        n_he == 0 or n_he >= 2
    )

    if use_pronoun_constraint:
        log(
            "Pronoun balance constraint ENABLED "
            "(she/unknown)"
        )
    else:
        issues.append(Issue(
            severity="warning",
            message=(
                "Pronoun balance constraint "
                "disabled: not enough she/unknown "
                f"({n_she}) for {n_groups} groups"
            ),
        ))
        log(
            "Pronoun balance constraint DISABLED "
            "(not enough she/unknown)"
        )

    if use_he_constraint:
        log(
            "Pronoun balance constraint ENABLED "
            "(he)"
        )
    else:
        issues.append(Issue(
            severity="warning",
            message=(
                "Pronoun balance constraint "
                "disabled: not enough he "
                f"({n_he}) for {n_groups} groups"
            ),
        ))
        log(
            "Pronoun balance constraint DISABLED "
            "(not enough he)"
        )

    if same_name_pairs:
        log(
            f"Same-name constraint: "
            f"{len(same_name_pairs)} pair(s) found"
        )

    name_col_key = spec.name_col
    if name_col_key and name_col_key in wdf.columns:
        full_names = wdf[name_col_key].to_numpy()
    else:
        full_names = None

    return {
        "full_names": full_names,
        "names": wdf["Preferred_name"].to_numpy(),
        "pronouns": (
            wdf["Pronoun"].to_numpy()
            if has_pronoun
            else None
        ),
        "cat_scores": cat_scores,
        "total_scores": total_scores,
        "is_she": is_she,
        "same_name_pairs": same_name_pairs,
        "n_students": n_students,
        "n_groups": n_groups,
        "group_sizes": sizes,
        "use_pronoun_constraint": use_pronoun_constraint,
        "use_he_constraint": use_he_constraint,
        "categories": categories,
        "issues": issues,
        "imputed_cells": imputed_cells,
    }


def load_and_prepare(csv_path, status_callback=None):
    """Load roster CSV and prepare data for optimization.

    Thin wrapper: reads CSV, guesses columns, calls
    prepare().
    """
    path = Path(csv_path)
    if not path.exists():
        raise SolverError(
            f"File not found: {csv_path}"
        )

    df = pd.read_csv(path)
    spec = suggest_columns(df)

    if len(spec.score_cols) < 1:
        raise SolverError(
            "Could not find any survey columns "
            "in the CSV"
        )

    return prepare(
        df, spec,
        status_callback=status_callback,
    )
