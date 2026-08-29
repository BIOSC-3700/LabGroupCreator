"""Characterization tests locking current behavior.

These must pass against the untouched code and remain
green through every refactor in Phase B.
"""

import numpy as np
import pytest

from labgroupassigner.preprocess import load_and_prepare
from labgroupassigner.model import build_and_solve


FIXTURES = [
    ("examples/test_roster.csv", -129.00, 0),
    ("examples/test_roster_hard.csv", -137.00, 3),
]


@pytest.fixture(params=FIXTURES, ids=lambda f: f[0])
def fixture_result(request):
    csv_name, expected_obj, expected_pairs = request.param
    data = load_and_prepare(
        csv_name, status_callback=lambda m: None
    )
    result = build_and_solve(
        data, status_callback=lambda m: None
    )
    return data, result, expected_obj, expected_pairs


def test_every_student_assigned_once(fixture_result):
    data, result, _, _ = fixture_result
    assignments = result["assignments"]
    n = data["n_students"]
    g = data["n_groups"]
    assert len(assignments) == n
    assert all(0 <= a < g for a in assignments)


def test_group_sizes_all_four(fixture_result):
    data, result, _, _ = fixture_result
    assignments = result["assignments"]
    g = data["n_groups"]
    for j in range(g):
        count = int(np.sum(assignments == j))
        assert count == 4, (
            f"Group {j} has {count} students, expected 4"
        )


def test_no_same_name_pair_coassigned(fixture_result):
    data, result, _, expected_pairs = fixture_result
    assignments = result["assignments"]
    pairs = data["same_name_pairs"]
    assert len(pairs) == expected_pairs
    for i1, i2 in pairs:
        assert assignments[i1] != assignments[i2], (
            f"Same-name pair ({i1}, {i2}) in same group"
        )


def test_objective_value(fixture_result):
    _, result, expected_obj, _ = fixture_result
    assert result["objective"] == pytest.approx(
        expected_obj, abs=0.01
    )


def test_solver_success(fixture_result):
    _, result, _, _ = fixture_result
    assert result["success"] is True


def test_no_single_she_group(fixture_result):
    """No group should have exactly one she/unknown."""
    data, result, _, _ = fixture_result
    assignments = result["assignments"]
    is_she = data["is_she"]
    g = data["n_groups"]
    for j in range(g):
        mask = assignments == j
        she_count = int(is_she[mask].sum())
        assert she_count != 1, (
            f"Group {j} has exactly 1 she/unknown"
        )
