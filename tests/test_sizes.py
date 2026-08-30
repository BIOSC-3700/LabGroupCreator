"""Tests for group_sizes()."""

import pytest

from labgroupassigner.model import group_sizes


def test_infeasible_below_6():
    for n in range(6):
        assert group_sizes(n) is None


@pytest.mark.parametrize("n", range(6, 41))
def test_sums_to_n(n):
    sizes = group_sizes(n)
    assert sizes is not None
    assert sum(sizes) == n


@pytest.mark.parametrize("n", range(6, 41))
def test_only_3s_4s_and_5s(n):
    sizes = group_sizes(n)
    for s in sizes:
        assert s in (3, 4, 5)


@pytest.mark.parametrize("n", range(6, 41))
def test_at_most_three_3s(n):
    sizes = group_sizes(n)
    assert sizes.count(3) <= 3


@pytest.mark.parametrize(
    "n", [48, 60, 100, 101, 102]
)
def test_large_values(n):
    sizes = group_sizes(n)
    assert sizes is not None
    assert sum(sizes) == n
    for s in sizes:
        assert s in (3, 4, 5)
    assert sizes.count(3) <= 3


def test_24_yields_six_fours():
    sizes = group_sizes(24)
    assert sizes == [4] * 6


def test_26_yields_five_fours_two_threes():
    sizes = group_sizes(26)
    assert sizes == [4] * 5 + [3] * 2


def test_9_yields_four_and_five():
    sizes = group_sizes(9)
    assert sizes == [4, 5]
