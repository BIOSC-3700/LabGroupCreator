import re
from dataclasses import dataclass

from labgroupassigner.schema import DerivedColumn


# --- Pronoun canonicalization ---

_SHE_TOKENS = frozenset({
    "she", "her", "hers",
})

_HE_TOKENS = frozenset({
    "he", "him", "his",
})

_MIXED_MARKERS = frozenset({
    "they", "them", "ze", "zir", "xe", "xem",
    "any", "all",
})


def normalize_pronoun(value):
    """Canonicalize a pronoun string.

    Returns (canonical, recognized) where canonical is
    one of "She", "He", or "Unknown", and recognized is
    True if the input matched a known pronoun pattern.
    """
    if value is None or (
        isinstance(value, float) and value != value
    ):
        return ("Unknown", False)

    raw = str(value).strip()
    if not raw:
        return ("Unknown", False)

    cleaned = raw.lower()
    # Strip outer brackets/parens
    cleaned = re.sub(r"^[\[\(]+|[\]\)]+$", "", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return ("Unknown", False)

    # Split on / to get individual tokens
    tokens = [t.strip() for t in cleaned.split("/")]
    tokens = [t for t in tokens if t]

    if not tokens:
        return ("Unknown", False)

    # Check for known phrases first
    full = cleaned.replace("/", " ").strip()
    if full in (
        "prefer not to say",
        "prefer not to answer",
        "unknown",
        "any pronouns",
        "any",
        "all pronouns",
    ):
        return ("Unknown", True)

    # Classify each token
    has_she = any(t in _SHE_TOKENS for t in tokens)
    has_he = any(t in _HE_TOKENS for t in tokens)
    has_mixed = any(t in _MIXED_MARKERS for t in tokens)

    # Mixed sets resolve to Unknown
    if has_mixed:
        return ("Unknown", True)
    if has_she and has_he:
        return ("Unknown", True)
    if has_she:
        return ("She", True)
    if has_he:
        return ("He", True)

    # Unrecognized
    return ("Unknown", False)


# --- Pronoun extraction from name fields ---

# Strip trailing non-ASCII junk (e.g., ¬†)
_TRAILING_JUNK_RE = re.compile(r"[^\x00-\x7F]+\s*$")

# Matches trailing pronoun tags in brackets, parens,
# after a dash/comma, or after whitespace (if the tag
# contains a slash):
#   Alice Smith (she/her)
#   Jordan Park [he/him]
#   Pat Chen - she/her
#   Drew he/him
#   Avery (She, Her)
# Bare trailing words without a slash are NOT matched
# to protect surnames like He, Her, Him.
_PRONOUN_TAG_RE = re.compile(
    r"""
    \s*                        # leading whitespace
    (?:                        # separator group
        [-,]\s*                # dash or comma
        |                      # or
        [\[\(]\s*              # opening bracket
        |                      # or
        \s+                    # bare whitespace
    )
    (                          # capture the tag
        [a-zA-Z]+              # first token
        (?:[/,]\s*[a-zA-Z]+)*  # /token or ,token parts
    )
    \s*                        # trailing whitespace
    [\]\)]*                    # optional closing bracket
    \s*$                       # end of string
    """,
    re.VERBOSE,
)


def extract_pronoun(name):
    """Extract a pronoun tag from a name string.

    Returns (cleaned_name, pronoun_canonical, recognized).
    If no tag is found, returns the original name with
    ("Unknown", False).
    """
    if not name or not isinstance(name, str):
        return (name, "Unknown", False)

    # Strip trailing non-ASCII garbage
    name = _TRAILING_JUNK_RE.sub("", name).rstrip()
    if not name:
        return ("", "Unknown", False)

    # Check if entire field is a bare pronoun
    # (no spaces, e.g., "she/her", "He/Him")
    if " " not in name:
        canonical, recognized = normalize_pronoun(name)
        if recognized:
            return ("", canonical, True)

    m = _PRONOUN_TAG_RE.search(name)
    if not m:
        return (name, "Unknown", False)

    tag = m.group(1)

    # Normalize commas to slashes for bracketed groups
    # like (She, Her)
    tag = re.sub(r",\s*", "/", tag)

    # Require the tag to look like a pronoun:
    # must contain a slash OR be a single bracketed
    # known pronoun token
    has_slash = "/" in tag
    is_bracketed = (
        name[m.start():].lstrip().startswith(("(", "["))
    )

    if not has_slash and not is_bracketed:
        return (name, "Unknown", False)

    # For bracketed single tokens, verify it is a
    # known pronoun
    if is_bracketed and not has_slash:
        lower_tag = tag.lower()
        if lower_tag not in (
            _SHE_TOKENS | _HE_TOKENS | _MIXED_MARKERS
            | {"they", "them"}
        ):
            return (name, "Unknown", False)

    # Verify that the tag normalizes to a recognized
    # pronoun (guards against false positives like
    # "sounds like rain")
    canonical, recognized = normalize_pronoun(tag)
    if not recognized:
        return (name, "Unknown", False)

    # Mid-string tags (not at logical end) are rejected.
    # E.g., "Marco (he/him) Silva" should not match.
    # The regex already anchors to $, so this is handled.

    cleaned_name = name[:m.start()].rstrip()

    return (cleaned_name, canonical, recognized)


# --- Rule runner ---

@dataclass
class DeriveReport:
    rule: DerivedColumn
    n_parsed: int = 0
    n_unparsed: int = 0
    unparsed_indices: list[int] = None

    def __post_init__(self):
        if self.unparsed_indices is None:
            self.unparsed_indices = []


def apply_derived(df, rules):
    """Apply derived-column rules to a DataFrame.

    Returns (new_df, list[DeriveReport]).
    Never raises on an unparsed row.
    """
    import pandas as pd

    df = df.copy()
    reports = []

    for rule in rules:
        if rule.method == "extract":
            report = DeriveReport(rule=rule)
            new_col_vals = []
            cleaned_source = []

            source = rule.source_col
            if source is None or source not in df.columns:
                report.n_unparsed = len(df)
                report.unparsed_indices = list(
                    range(len(df))
                )
                df[rule.new_name] = "Unknown"
                reports.append(report)
                continue

            for idx in range(len(df)):
                val = df[source].iloc[idx]
                cleaned, canonical, recognized = (
                    extract_pronoun(str(val))
                )

                if recognized:
                    report.n_parsed += 1
                else:
                    report.n_unparsed += 1
                    report.unparsed_indices.append(idx)

                # Apply mapping override if present
                if rule.mapping and canonical in rule.mapping:
                    canonical = rule.mapping[canonical]

                new_col_vals.append(canonical)
                cleaned_source.append(cleaned)

            source_pos = df.columns.get_loc(source)
            df.insert(
                source_pos + 1,
                rule.new_name,
                new_col_vals,
            )

            if rule.strip_from_source:
                df[source] = cleaned_source

            reports.append(report)

    return df, reports
