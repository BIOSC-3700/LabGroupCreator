"""Tests for derive.py: pronoun canonicalization and
extraction."""

import pandas as pd
import pytest

from labgroupassigner.derive import (
    normalize_pronoun,
    extract_pronoun,
    apply_derived,
    DeriveReport,
)
from labgroupassigner.schema import DerivedColumn


# --- normalize_pronoun ---

@pytest.mark.parametrize(
    "value, expected_canonical, expected_recognized",
    [
        # She variants
        ("she", "She", True),
        ("She", "She", True),
        ("SHE", "She", True),
        ("her", "She", True),
        ("hers", "She", True),
        ("she/her", "She", True),
        ("She/Her", "She", True),
        ("She/Her/Hers", "She", True),
        # He variants
        ("he", "He", True),
        ("He", "He", True),
        ("HE", "He", True),
        ("him", "He", True),
        ("his", "He", True),
        ("he/him", "He", True),
        ("He/Him", "He", True),
        # Mixed -> Unknown (recognized)
        ("they", "Unknown", True),
        ("they/them", "Unknown", True),
        ("she/they", "Unknown", True),
        ("he/they", "Unknown", True),
        ("ze/zir", "Unknown", True),
        ("xe/xem", "Unknown", True),
        ("any", "Unknown", True),
        ("any pronouns", "Unknown", True),
        ("prefer not to say", "Unknown", True),
        ("unknown", "Unknown", True),
        # Blank/missing -> Unknown (not recognized)
        ("", "Unknown", False),
        (None, "Unknown", False),
        (float("nan"), "Unknown", False),
        # Unrecognized -> Unknown (not recognized)
        ("sje", "Unknown", False),
        ("xyz", "Unknown", False),
        # Bracketed
        ("(she/her)", "She", True),
        ("[he/him]", "He", True),
    ],
)
def test_normalize_pronoun(
    value, expected_canonical, expected_recognized
):
    canonical, recognized = normalize_pronoun(value)
    assert canonical == expected_canonical
    assert recognized == expected_recognized


# --- extract_pronoun ---

@pytest.mark.parametrize(
    "name, expected_clean, expected_canonical,"
    " expected_recognized",
    [
        # Standard cases from the plan
        (
            "Alice Smith (she/her)",
            "Alice Smith", "She", True,
        ),
        (
            "Jordan Park [he/him]",
            "Jordan Park", "He", True,
        ),
        (
            "Pat Chen - she/her",
            "Pat Chen", "She", True,
        ),
        (
            "Alex Nguyen (She/Her/Hers)",
            "Alex Nguyen", "She", True,
        ),
        (
            "Chris Vance (he)",
            "Chris Vance", "He", True,
        ),
        (
            "Robin Diaz(they/them)",
            "Robin Diaz", "Unknown", True,
        ),
        (
            "Kai Rivera (she/they)",
            "Kai Rivera", "Unknown", True,
        ),
        (
            "Morgan O'Neill (ze/zir)",
            "Morgan O'Neill", "Unknown", True,
        ),
        # No tag -> Unknown, not recognized
        (
            "Taylor Kim",
            "Taylor Kim", "Unknown", False,
        ),
        # "any pronouns" in parens -> flagged
        (
            "Dana White (any pronouns)",
            "Dana White (any pronouns)", "Unknown", False,
        ),
        # Mid-string tag -> not matched
        (
            "Marco (he/him) Silva",
            "Marco (he/him) Silva", "Unknown", False,
        ),
        # Surname protection
        (
            "Amy He",
            "Amy He", "Unknown", False,
        ),
        (
            "Wei He",
            "Wei He", "Unknown", False,
        ),
        (
            "Li Her",
            "Li Her", "Unknown", False,
        ),
        (
            "Sam Him",
            "Sam Him", "Unknown", False,
        ),
        # Surname + actual pronoun tag
        (
            "Amy He (she/her)",
            "Amy He", "She", True,
        ),
        # Entire field is a pronoun (no name)
        ("she/her", "", "She", True),
        ("He/Him", "", "He", True),
        ("She/Her", "", "She", True),
        # Space-separated trailing pronoun with slash
        ("Drew he/him", "Drew", "He", True),
        ("Payton she/her", "Payton", "She", True),
        ("Lukas He/Him", "Lukas", "He", True),
        (
            "Harrison He/Him",
            "Harrison", "He", True,
        ),
        # Free-text ending with pronoun
        (
            "Colin is fine! He/Him/His",
            "Colin is fine!", "He", True,
        ),
        (
            "My prefered name is Abi and my "
            "pronouns are she/her",
            "My prefered name is Abi and my "
            "pronouns are", "She", True,
        ),
        # Comma-separated inside parens
        (
            "Avery (She, Her)",
            "Avery", "She", True,
        ),
        # Trailing non-ASCII garbage
        (
            "Sarah\u00ac\u2020",
            "Sarah", "Unknown", False,
        ),
        (
            "Lindsea\u00ac\u2020",
            "Lindsea", "Unknown", False,
        ),
        # Comma-separated pronoun (no parens)
        (
            "Katie, she/her",
            "Katie", "She", True,
        ),
        (
            "Wren, she/her",
            "Wren", "She", True,
        ),
        (
            "Eli, He/Him",
            "Eli", "He", True,
        ),
        # Parenthetical that is NOT a pronoun
        (
            "Rayne\n\n(sounds like rain)",
            "Rayne\n\n(sounds like rain)",
            "Unknown", False,
        ),
        # Bracketed pronoun
        (
            "Caleb (He/Him)",
            "Caleb", "He", True,
        ),
        (
            "Anna (she/her)",
            "Anna", "She", True,
        ),
        (
            "Leah (she/her)",
            "Leah", "She", True,
        ),
        # Plain names -> unknown
        ("Kate", "Kate", "Unknown", False),
        ("Anna", "Anna", "Unknown", False),
        ("Aidan", "Aidan", "Unknown", False),
        ("Evan", "Evan", "Unknown", False),
        ("Jackson", "Jackson", "Unknown", False),
    ],
)
def test_extract_pronoun(
    name, expected_clean, expected_canonical,
    expected_recognized,
):
    cleaned, canonical, recognized = (
        extract_pronoun(name)
    )
    assert cleaned == expected_clean
    assert canonical == expected_canonical
    assert recognized == expected_recognized


# --- apply_derived ---

def test_apply_derived_extract():
    df = pd.DataFrame({
        "Name": [
            "Alice Smith (she/her)",
            "Bob Jones (he/him)",
            "Taylor Kim",
        ],
    })
    rule = DerivedColumn(
        new_name="Pronoun",
        method="extract",
        source_col="Name",
    )
    result_df, reports = apply_derived(df, [rule])

    assert list(result_df["Pronoun"]) == [
        "She", "He", "Unknown",
    ]
    # strip_from_source=True by default
    assert result_df["Name"].iloc[0] == "Alice Smith"
    assert result_df["Name"].iloc[1] == "Bob Jones"
    assert result_df["Name"].iloc[2] == "Taylor Kim"

    assert len(reports) == 1
    assert reports[0].n_parsed == 2
    assert reports[0].n_unparsed == 1
    assert reports[0].unparsed_indices == [2]


def test_apply_derived_no_strip():
    df = pd.DataFrame({
        "Name": ["Alice Smith (she/her)"],
    })
    rule = DerivedColumn(
        new_name="Pronoun",
        method="extract",
        source_col="Name",
        strip_from_source=False,
    )
    result_df, reports = apply_derived(df, [rule])

    # Source column unchanged
    assert (
        result_df["Name"].iloc[0]
        == "Alice Smith (she/her)"
    )
    assert result_df["Pronoun"].iloc[0] == "She"
