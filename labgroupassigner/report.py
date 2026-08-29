import numpy as np
import pandas as pd
from pathlib import Path


def _base_df(data, assignments):
    """Build the per-student DataFrame used by all
    report builders."""
    full_names = data["full_names"]
    names = data["names"]
    pronouns = data["pronouns"]
    cat_scores = data["cat_scores"]
    total_scores = data["total_scores"]
    categories = data["categories"]

    cols = {}
    if full_names is not None:
        cols["name"] = full_names
    cols["preferred_name"] = names
    if pronouns is not None:
        cols["pronoun"] = pronouns
    cols["group"] = assignments + 1  # 1-based
    cols["total_score"] = total_scores
    df = pd.DataFrame(cols)
    for ci, cat in enumerate(categories):
        df[cat] = cat_scores[:, ci]
    return df


def build_assignments(data, assignments):
    """One row per student with group and scores."""
    df = _base_df(data, assignments)
    sort_cols = []
    if "name" in df.columns:
        sort_cols.append("name")
    sort_cols += ["preferred_name", "group"]
    return (
        df.sort_values("group")
        .reset_index(drop=True)
    )


def build_group_summary(data, assignments):
    """Per-group summary: size, total, mean,
    minority/majority counts."""
    df = _base_df(data, assignments)
    is_she = data["is_she"]

    summary = (
        df.groupby("group")
        .agg(
            n_students=("preferred_name", "size"),
            total_score=("total_score", "sum"),
            mean_score=("total_score", "mean"),
        )
        .reset_index()
    )

    she_counts = (
        df.assign(is_she=is_she)
        .groupby("group")["is_she"]
        .agg(
            n_she_or_unknown="sum",
            n_he=lambda s: (s == 0).sum(),
        )
        .reset_index()
    )
    summary = summary.merge(she_counts, on="group")
    summary["mean_score"] = (
        summary["mean_score"].round(2)
    )
    return summary


def build_diversity(data, assignments):
    """Per-group category maxima and diversity sum."""
    df = _base_df(data, assignments)
    categories = data["categories"]

    diversity = (
        df.groupby("group")[categories]
        .max()
        .reset_index()
    )
    diversity.columns = (
        ["group"]
        + [f"max_{c}" for c in categories]
    )
    cat_cols = [f"max_{c}" for c in categories]
    diversity["diversity_sum"] = (
        diversity[cat_cols].sum(axis=1)
    )
    return diversity


def build_metrics(data, assignments):
    """Headline metrics as a dict."""
    df = _base_df(data, assignments)
    is_she = data["is_she"]
    categories = data["categories"]

    group_totals = (
        df.groupby("group")["total_score"].sum()
    )
    score_range = (
        group_totals.max() - group_totals.min()
    )

    diversity = build_diversity(data, assignments)
    total_diversity = (
        diversity["diversity_sum"].sum()
    )

    group_sizes_vals = (
        df.groupby("group")["preferred_name"]
        .size()
    )
    size_range = (
        group_sizes_vals.max()
        - group_sizes_vals.min()
    )

    # Same-name violations
    pairs = data["same_name_pairs"]
    violations = sum(
        1 for i1, i2 in pairs
        if assignments[i1] == assignments[i2]
    )

    # Isolated-attribute groups
    n_groups = data["n_groups"]
    is_he = 1 - is_she
    isolated_she = 0
    isolated_he = 0
    for j in range(n_groups):
        mask = assignments == j
        if int(is_she[mask].sum()) == 1:
            isolated_she += 1
        if int(is_he[mask].sum()) == 1:
            isolated_he += 1

    return {
        "score_range": float(score_range),
        "total_diversity": float(total_diversity),
        "size_range": int(size_range),
        "same_name_violations": violations,
        "isolated_attribute_groups": (
            isolated_she + isolated_he
        ),
        "isolated_she_groups": isolated_she,
        "isolated_he_groups": isolated_he,
    }


# --- CLI-facing wrappers ---

def print_report(data, assignments):
    """Print group assignment results to stdout."""
    summary = build_group_summary(
        data, assignments
    )
    diversity = build_diversity(data, assignments)
    metrics = build_metrics(data, assignments)
    roster = build_assignments(data, assignments)

    print("\nGroup Summary")
    print(summary.to_string(index=False))

    print("\nCategory Diversity")
    print(diversity.to_string(index=False))

    print(
        f"\nRange of group totals: "
        f"{metrics['score_range']}"
    )
    print(
        f"Total category diversity "
        f"(sum of max scores): "
        f"{metrics['total_diversity']}"
    )

    roster_cols = []
    if "name" in roster.columns:
        roster_cols.append("name")
    roster_cols += [
        "preferred_name", "group",
    ]
    if "pronoun" in roster.columns:
        roster_cols.append("pronoun")
    print("\nStudent Assignments")
    print(
        roster[roster_cols].to_string(index=False)
    )


def write_csv_report(data, assignments, csv_path):
    """Write Groups CSV and return summary text.

    Thin wrapper over the builders for CLI use.
    """
    stem = Path(csv_path).stem
    parent = Path(csv_path).parent

    roster = build_assignments(data, assignments)

    # Capitalize columns for CSV output
    rename_map = {
        "name": "Name",
        "preferred_name": "Preferred_Name",
        "pronoun": "Pronoun",
        "group": "Group",
        "total_score": "Total_Score",
    }
    csv_df = roster.rename(
        columns={
            k: v
            for k, v in rename_map.items()
            if k in roster.columns
        }
    )

    group_cols = []
    if "Name" in csv_df.columns:
        group_cols.append("Name")
    group_cols += ["Preferred_Name"]
    if "Pronoun" in csv_df.columns:
        group_cols.append("Pronoun")
    group_cols.append("Group")
    groups_df = (
        csv_df[group_cols]
        .sort_values("Group")
        .reset_index(drop=True)
    )
    groups_path = parent / f"{stem}_Groups.csv"
    groups_df.to_csv(groups_path, index=False)

    summary = build_group_summary(
        data, assignments
    )
    diversity = build_diversity(data, assignments)
    metrics = build_metrics(data, assignments)

    # Capitalize for display
    summary.columns = [
        c.replace("_", " ").title().replace(" ", "_")
        for c in summary.columns
    ]
    diversity.columns = [
        c.replace("_", " ").title().replace(" ", "_")
        for c in diversity.columns
    ]

    combined = summary.merge(diversity, on="Group")

    lines = [
        "Group Summary",
        combined.to_string(index=False),
        "",
        f"Range of group totals: "
        f"{metrics['score_range']}",
        f"Total category diversity "
        f"(sum of max scores): "
        f"{metrics['total_diversity']}",
    ]
    summary_text = "\n".join(lines)

    return groups_path, summary_text
