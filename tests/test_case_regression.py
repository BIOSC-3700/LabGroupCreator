"""Case regression test for the pronoun bug (PLAN s1).

The shipped fixtures with pronoun column upper-, lower-,
and title-cased must all yield identical group assignments
(same objective, same constraint behavior).
"""

import pandas as pd
import numpy as np
import pytest

from labgroupassigner.preprocess import load_and_prepare
from labgroupassigner.model import build_and_solve


FIXTURES = [
    "examples/test_roster.csv",
    "examples/test_roster_hard.csv",
]


def _load_with_case(csv_path, case_fn):
    """Load a fixture with pronouns transformed."""
    df = pd.read_csv(csv_path)
    df["Pronoun"] = df["Pronoun"].apply(case_fn)
    # Write to a temp path so load_and_prepare can read it
    import tempfile
    from pathlib import Path
    tmp = Path(
        tempfile.mktemp(suffix=".csv")
    )
    df.to_csv(tmp, index=False)
    try:
        data = load_and_prepare(
            str(tmp), status_callback=lambda m: None
        )
        result = build_and_solve(
            data, status_callback=lambda m: None
        )
        return data, result
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.parametrize("csv_path", FIXTURES)
def test_pronoun_case_invariance(csv_path):
    """All case variants produce the same objective."""
    cases = [str.lower, str.upper, str.title]
    objectives = []
    for case_fn in cases:
        data, result = _load_with_case(
            csv_path, case_fn
        )
        objectives.append(result["objective"])

    # All three must agree
    for obj in objectives[1:]:
        assert obj == pytest.approx(
            objectives[0], abs=0.01
        ), (
            f"Objective mismatch across cases: "
            f"{objectives}"
        )


@pytest.mark.parametrize("csv_path", FIXTURES)
def test_no_single_she_group_any_case(csv_path):
    """No group has exactly one she/unknown regardless
    of pronoun casing."""
    cases = [str.lower, str.upper, str.title]
    for case_fn in cases:
        data, result = _load_with_case(
            csv_path, case_fn
        )
        assignments = result["assignments"]
        is_she = data["is_she"]
        g = data["n_groups"]
        for j in range(g):
            mask = assignments == j
            she_count = int(is_she[mask].sum())
            assert she_count != 1, (
                f"Group {j} has 1 she/unknown "
                f"with case={case_fn.__name__}"
            )
