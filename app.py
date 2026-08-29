import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from labgroupassigner.derive import (
    apply_derived,
    extract_pronoun,
    normalize_pronoun,
)
from labgroupassigner.errors import (
    SolverError,
    ValidationError,
)
from labgroupassigner.model import (
    SolverStatus,
    build_and_solve,
    group_sizes,
)
from labgroupassigner.preprocess import (
    LIKERT_MAP,
    prepare,
    suggest_columns,
)
from labgroupassigner.report import (
    build_assignments,
    build_diversity,
    build_group_summary,
    build_metrics,
)
from labgroupassigner.schema import (
    ColumnSpec,
    DerivedColumn,
    SolveConfig,
)

_LEADING_CODE_RE = re.compile(r"^\d{8}[:\s]+\s*")


def _strip_column_codes(df):
    """Strip leading 8-digit codes from column names."""
    rename = {}
    for col in df.columns:
        cleaned = _LEADING_CODE_RE.sub("", col)
        if cleaned != col:
            rename[col] = cleaned
    if rename:
        df = df.rename(columns=rename)
    return df


def _uninformative_columns(df):
    """Return column names that carry no useful data."""
    drop = []
    for col in df.columns:
        if not col or col.isspace():
            drop.append(col)
            continue
        vals = df[col].dropna()
        if len(vals) == 0 or vals.nunique() <= 1:
            drop.append(col)
    return drop


app_ui = ui.page_navbar(
    ui.nav_panel(
        "1. Load Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Data Source"),
                ui.input_file(
                    "file_upload",
                    "Upload roster CSV",
                    accept=[".csv"],
                ),
                ui.hr(),
                ui.h5("Column Roles"),
                ui.output_ui("column_selectors"),
                ui.hr(),
                ui.h5("Pronoun Extraction"),
                ui.output_ui("extraction_ui"),
                ui.hr(),
                ui.input_action_link(
                    "toggle_settings",
                    "Solve settings...",
                ),
                ui.output_ui("settings_panel"),
                width=350,
            ),
            ui.input_switch(
                "hide_uninformative",
                "Hide uninformative columns",
                value=True,
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Data Preview",
                    ui.output_data_frame("raw_table"),
                ),
                ui.nav_panel(
                    "Group Sizes",
                    ui.output_text_verbatim("group_sizes_text"),
                ),
                ui.nav_panel(
                    "Likert Mapping",
                    ui.output_text_verbatim("likert_text"),
                ),
            ),
            ui.br(),
            ui.input_action_button(
                "go_verify",
                "Continue to Verify",
                class_="btn-primary btn-lg",
            ),
        ),
    ),
    ui.nav_panel(
        "2. Verify",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Problem Summary"),
                ui.output_text_verbatim("problem_summary"),
                width=300,
            ),
            ui.h4("Validation"),
            ui.output_ui("validation_panel"),
            ui.hr(),
            ui.h4("Recoded Preview"),
            ui.output_data_frame("recoded_table"),
            ui.br(),
            ui.output_ui("verify_button_ui"),
        ),
    ),
    ui.nav_panel(
        "3. Results",
        ui.layout_sidebar(
            ui.sidebar(
                ui.output_ui("run_button_ui"),
                ui.hr(),
                ui.h5("Solver Status"),
                ui.output_text_verbatim("solver_status"),
                ui.hr(),
                ui.h5("Metrics"),
                ui.output_text_verbatim("metrics_text"),
                ui.hr(),
                ui.h5("Downloads"),
                ui.output_ui("download_ui"),
                width=300,
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Assignments",
                    ui.output_data_frame("assignments_table"),
                ),
                ui.nav_panel(
                    "Group Summary",
                    ui.output_data_frame("summary_table"),
                ),
                ui.nav_panel(
                    "Diversity",
                    ui.output_data_frame("diversity_table"),
                ),
            ),
        ),
    ),
    title="Lab Group Assigner",
    id="main_nav",
    header=ui.tags.head(ui.tags.title("Lab Group Assigner")),
)


def server(input, output, session):
    # --- Reactive values ---
    raw_df = reactive.value(None)
    derived_rules = reactive.value([])
    result_val = reactive.value(None)
    status_log = reactive.value("")
    show_settings = reactive.value(False)
    uploaded_stem = reactive.value("roster")
    data_overrides = reactive.value({})

    # --- File upload ---
    @reactive.effect
    @reactive.event(input.file_upload)
    def _on_upload():
        file_info = input.file_upload()
        if file_info is None:
            return
        path = file_info[0]["datapath"]
        name = file_info[0]["name"]
        df = _strip_column_codes(pd.read_csv(path))
        raw_df.set(df)
        derived_rules.set([])
        result_val.set(None)
        data_overrides.set({})
        stem = name.rsplit(".", 1)[0] if "." in name else name
        uploaded_stem.set(stem)

    @reactive.effect
    @reactive.event(input.load_example)
    def _on_example():
        example_path = Path(__file__).parent / "examples" / "test_roster.csv"
        df = pd.read_csv(example_path)
        raw_df.set(df)
        derived_rules.set([])
        result_val.set(None)
        data_overrides.set({})
        uploaded_stem.set("test_roster")

    # --- Derived DataFrame ---
    @reactive.calc
    def derived_df():
        df = raw_df.get()
        if df is None:
            return None
        rules = derived_rules.get()
        if rules:
            df, reports = apply_derived(df, rules)
        else:
            df = df.copy()

        # Apply user overrides from cell edits
        overrides = data_overrides.get()
        for (row, col), val in overrides.items():
            if col in df.columns:
                df.at[row, col] = val
        return df

    # --- Column selectors ---
    @output
    @render.ui
    def column_selectors():
        df = derived_df()
        if df is None:
            return ui.p("Upload a CSV to configure columns.")

        spec = suggest_columns(df)
        cols = list(df.columns)

        label_choices = {c: c for c in cols}
        balance_choices = {"": "(none)"}
        balance_choices.update({c: c for c in cols})

        label_selected = spec.label_col if spec.label_col else cols[0]
        balance_selected = spec.balance_col if spec.balance_col else ""

        return ui.TagList(
            ui.input_select(
                "label_col",
                "Preferred name column",
                choices=label_choices,
                selected=label_selected,
            ),
            ui.input_select(
                "balance_col",
                "Pronoun column",
                choices=balance_choices,
                selected=balance_selected,
            ),
            ui.h6("Survey columns"),
            ui.input_selectize(
                "score_cols",
                "Select survey question columns",
                choices=cols,
                selected=spec.score_cols,
                multiple=True,
            ),
        )

    # --- Pronoun extraction ---
    @output
    @render.ui
    def extraction_ui():
        df = derived_df()
        if df is None:
            return ui.p("")

        cols = list(df.columns)
        rules = derived_rules.get()
        has_extraction = any(r.method == "extract" for r in rules)

        if has_extraction:
            return ui.TagList(
                ui.p("Pronoun extraction active."),
                ui.input_action_button(
                    "remove_extraction",
                    "Remove extraction",
                    class_="btn-outline-danger btn-sm",
                ),
            )

        # Check if extraction might be useful
        suggestion = ""
        spec = suggest_columns(df)
        if spec.label_col and spec.label_col in df.columns:
            sample = df[spec.label_col].dropna().head(20)
            paren_count = sum(
                1
                for v in sample
                if isinstance(v, str) and ("(" in v or "[" in v)
            )
            if paren_count >= len(sample) * 0.3:
                suggestion = (
                    f"Many rows in "
                    f"'{spec.label_col}' "
                    f"contain parenthetical text."
                )

        return ui.TagList(
            ui.p(suggestion) if suggestion else None,
            ui.input_select(
                "extract_source",
                "Source column",
                choices={c: c for c in cols},
                selected=(spec.label_col if spec.label_col else cols[0]),
            ),
            ui.input_action_button(
                "do_extraction",
                "Extract pronouns",
                class_="btn-outline-primary btn-sm",
            ),
        )

    @reactive.effect
    @reactive.event(input.do_extraction)
    def _on_extract():
        source = input.extract_source()
        if not source:
            return
        rule = DerivedColumn(
            new_name="Pronoun",
            method="extract",
            source_col=source,
            strip_from_source=True,
        )
        derived_rules.set([rule])
        result_val.set(None)

    @reactive.effect
    @reactive.event(input.remove_extraction)
    def _on_remove_extract():
        derived_rules.set([])
        result_val.set(None)

    # --- Settings toggle ---
    @reactive.effect
    @reactive.event(input.toggle_settings)
    def _toggle():
        show_settings.set(not show_settings.get())

    @output
    @render.ui
    def settings_panel():
        if not show_settings.get():
            return ui.p("")

        return ui.TagList(
            ui.input_slider(
                "balance_weight",
                "Balance weight",
                min=0,
                max=5,
                value=1.0,
                step=0.1,
            ),
            ui.input_slider(
                "diversity_weight",
                "Diversity weight",
                min=0,
                max=5,
                value=1.0,
                step=0.1,
            ),
            ui.input_slider(
                "pronoun_weight",
                "Pronoun balance weight",
                min=0,
                max=5,
                value=1.0,
                step=0.1,
            ),
            ui.input_slider(
                "isolation_penalty",
                "Isolation penalty",
                min=0,
                max=50,
                value=10.0,
                step=1.0,
            ),
            ui.input_switch(
                "enforce_same_name",
                "Enforce same-name separation",
                value=True,
            ),
            ui.input_slider(
                "time_limit",
                "Time limit (seconds)",
                min=5,
                max=120,
                value=30,
                step=5,
            ),
            ui.input_numeric(
                "seed",
                "Random seed",
                value=0,
                min=0,
            ),
        )

    # --- Build spec and config from inputs ---
    @reactive.calc
    def current_spec():
        df = derived_df()
        if df is None:
            return None

        # Use suggested spec as defaults, then
        # override with user selections
        spec = suggest_columns(df)

        try:
            label = input.label_col()
        except Exception:
            label = spec.label_col
        try:
            balance = input.balance_col()
        except Exception:
            balance = spec.balance_col

        try:
            score_cols = list(input.score_cols())
        except Exception:
            score_cols = spec.score_cols

        return ColumnSpec(
            name_col=spec.name_col,
            label_col=label or spec.label_col,
            score_cols=score_cols,
            balance_col=balance if balance else None,
        )

    @reactive.calc
    def current_config():
        try:
            bw = input.balance_weight()
        except Exception:
            bw = 1.0
        try:
            dw = input.diversity_weight()
        except Exception:
            dw = 1.0
        try:
            pw = input.pronoun_weight()
        except Exception:
            pw = 1.0
        try:
            ip = input.isolation_penalty()
        except Exception:
            ip = 10.0
        try:
            esn = input.enforce_same_name()
        except Exception:
            esn = True
        try:
            tl = input.time_limit()
        except Exception:
            tl = 30.0
        try:
            sd = input.seed()
        except Exception:
            sd = 0
        return SolveConfig(
            balance_weight=bw,
            diversity_weight=dw,
            balance_attr_weight=pw,
            isolation_penalty=ip,
            enforce_same_name=esn,
            time_limit_s=tl,
            seed=sd,
        )

    # --- Prepared data (validation) ---
    @reactive.calc
    def prepared():
        df = derived_df()
        spec = current_spec()
        if df is None or spec is None:
            return None
        if len(spec.score_cols) < 1:
            return None
        try:
            data = prepare(df, spec)
            return data
        except (SolverError, ValidationError) as e:
            return e

    # --- Tab 1 outputs ---
    @output
    @render.data_frame
    def raw_table():
        df = derived_df()
        if df is None:
            return None
        if input.hide_uninformative():
            drop = _uninformative_columns(df)
            if drop:
                df = df.drop(columns=drop)
        return render.DataGrid(
            df,
            row_selection_mode="none",
            height="500px",
        )

    @output
    @render.text
    def group_sizes_text():
        df = derived_df()
        if df is None:
            return "Upload a CSV to see group sizes."
        n = len(df)
        sizes = group_sizes(n)
        if sizes is None:
            return f"{n} students: too few (need 6+)."
        n4 = sizes.count(4)
        n3 = sizes.count(3)
        parts = []
        if n4:
            parts.append(f"{n4} of 4")
        if n3:
            parts.append(f"{n3} of 3")
        return f"{n} students -> {len(sizes)} groups: " + ", ".join(parts)

    @output
    @render.text
    def likert_text():
        lines = ["Likert scale mapping:"]
        for text, val in LIKERT_MAP.items():
            lines.append(f"  {text} -> {val}")
        lines.append("")
        lines.append("Balance categories: {She, Unknown} vs {He}")
        return "\n".join(lines)

    # --- Tab 2 outputs ---
    @output
    @render.ui
    def validation_panel():
        p = prepared()
        if p is None:
            return ui.p(
                "Configure data in Tab 1 first.",
                class_="text-muted",
            )
        if isinstance(p, ValidationError):
            items = []
            for issue in p.issues:
                badge = ui.span(
                    issue.severity.upper(),
                    class_=(
                        "badge bg-danger"
                        if issue.severity == "error"
                        else "badge bg-warning"
                    ),
                )
                items.append(
                    ui.tags.li(
                        badge,
                        " ",
                        issue.message,
                    )
                )
            return ui.TagList(
                ui.tags.ul(*items),
                ui.p(
                    "Fix errors before proceeding.",
                    class_="text-danger fw-bold",
                ),
            )
        if isinstance(p, SolverError):
            return ui.p(str(p), class_="text-danger")

        # Valid data with possible warnings
        issues = p.get("issues", [])
        if not issues:
            return ui.p(
                "All checks passed.",
                class_="text-success",
            )
        items = []
        for issue in issues:
            badge = ui.span(
                issue.severity.upper(),
                class_=(
                    "badge bg-danger"
                    if issue.severity == "error"
                    else "badge bg-warning"
                ),
            )
            items.append(ui.tags.li(badge, " ", issue.message))
        return ui.tags.ul(*items)

    @output
    @render.text
    def problem_summary():
        p = prepared()
        if p is None or isinstance(p, Exception):
            return "No valid data."

        lines = [
            f"Students: {p['n_students']}",
            f"Groups: {p['n_groups']}",
            f"Sizes: {p['group_sizes']}",
            "",
            f"Same-name pairs: {len(p['same_name_pairs'])}",
            f"Pronoun constraint: "
            f"{'ON' if p['use_pronoun_constraint'] else 'OFF'}",
        ]

        if p.get("is_she") is not None:
            n_she = int(p["is_she"].sum())
            lines.append(
                f"She/Unknown: {n_she}, He: {p['n_students'] - n_she}"
            )

        return "\n".join(lines)

    @output
    @render.data_frame
    def recoded_table():
        p = prepared()
        if p is None or isinstance(p, Exception):
            return None

        # Show the numeric matrix
        df = pd.DataFrame(
            p["cat_scores"],
            columns=p["categories"],
        )
        df.insert(0, "Preferred_name", p["names"])
        if p["pronouns"] is not None:
            df.insert(1, "Pronoun", p["pronouns"])
        df["Total"] = p["total_scores"]

        return render.DataGrid(
            df,
            row_selection_mode="none",
            height="400px",
            editable=True,
        )

    @recoded_table.set_patch_fn
    def _(*, patch):
        row = patch["row_index"]
        col_idx = patch["column_index"]
        value = patch["value"]

        displayed = recoded_table.data_patched()
        col_name = displayed.columns[col_idx]

        # Reject edits to computed Total column
        if col_name == "Total":
            return displayed.at[row, col_name]

        spec = current_spec()
        overrides = dict(data_overrides.get())

        if col_name == "Preferred_name":
            overrides[(row, spec.label_col)] = value
        elif col_name == "Pronoun" and spec.balance_col:
            overrides[(row, spec.balance_col)] = value
        else:
            # Map Q1..Qn back to original score column
            categories = [
                f"Q{i + 1}"
                for i in range(len(spec.score_cols))
            ]
            if col_name in categories:
                qi = categories.index(col_name)
                orig_col = spec.score_cols[qi]
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
                overrides[(row, orig_col)] = value

        data_overrides.set(overrides)
        result_val.set(None)
        return value

    @output
    @render.ui
    def verify_button_ui():
        p = prepared()
        has_errors = p is None or isinstance(p, Exception)
        return ui.input_action_button(
            "go_run",
            "Confirm and Run",
            class_=(
                "btn-primary btn-lg"
                if not has_errors
                else "btn-secondary btn-lg"
            ),
            disabled=has_errors,
        )

    # --- Navigation ---
    @reactive.effect
    @reactive.event(input.go_verify)
    def _nav_verify():
        ui.update_navs("main_nav", selected="2. Verify")

    @reactive.effect
    @reactive.event(input.go_run)
    def _nav_run():
        ui.update_navs("main_nav", selected="3. Results")

    # --- Tab 3: Run ---
    @output
    @render.ui
    def run_button_ui():
        p = prepared()
        has_errors = p is None or isinstance(p, Exception)
        running = result_val.get() == "running"

        return ui.TagList(
            ui.input_action_button(
                "run_solve",
                "Run Optimization" if not running else "Running...",
                class_="btn-success btn-lg w-100",
                disabled=has_errors or running,
            ),
            ui.br(),
            ui.input_action_button(
                "rerun_solve",
                "Re-run (new seed)",
                class_=("btn-outline-secondary btn-sm w-100 mt-2"),
                disabled=(
                    has_errors
                    or running
                    or result_val.get() is None
                    or result_val.get() == "running"
                ),
            ),
        )

    @reactive.effect
    @reactive.event(input.run_solve)
    def _on_run():
        _do_solve()

    @reactive.effect
    @reactive.event(input.rerun_solve)
    def _on_rerun():
        # Increment seed
        try:
            current = input.seed()
        except Exception:
            current = 0
        ui.update_numeric("seed", value=current + 1)
        _do_solve()

    def _do_solve():
        p = prepared()
        if p is None or isinstance(p, Exception):
            return

        result_val.set("running")
        status_log.set("Starting solve...\n")

        config = current_config()

        def log_status(msg):
            status_log.set(status_log.get() + msg + "\n")

        try:
            result = build_and_solve(
                p,
                balance_weight=config.balance_weight,
                diversity_weight=(config.diversity_weight),
                pronoun_weight=(config.balance_attr_weight),
                one_she_penalty=(config.isolation_penalty),
                timeout_minutes=(config.time_limit_s / 60.0),
                status_callback=log_status,
            )
            result_val.set(result)
        except SolverError as e:
            status_log.set(status_log.get() + f"ERROR: {e}\n")
            result_val.set(None)

    @output
    @render.text
    def solver_status():
        return status_log.get()

    @output
    @render.text
    def metrics_text():
        r = result_val.get()
        if r is None or r == "running":
            return ""
        if not isinstance(r, dict):
            return ""

        p = prepared()
        if p is None or isinstance(p, Exception):
            return ""

        m = build_metrics(p, r["assignments"])
        status = r.get("status", SolverStatus.OPTIMAL)

        lines = [
            f"Status: {status.value}",
            f"Objective: {r['objective']:.2f}",
            f"Score range: {m['score_range']:.1f}",
            f"Total diversity: {m['total_diversity']:.0f}",
            f"Size range: {m['size_range']}",
            f"Same-name violations: {m['same_name_violations']}",
            f"Isolated she/unknown: {m['isolated_she_groups']}",
            f"Isolated he: {m['isolated_he_groups']}",
        ]
        return "\n".join(lines)

    @output
    @render.data_frame
    def assignments_table():
        r = result_val.get()
        if r is None or r == "running":
            return None
        if not isinstance(r, dict):
            return None
        p = prepared()
        if p is None or isinstance(p, Exception):
            return None
        df = build_assignments(p, r["assignments"])
        return render.DataGrid(
            df,
            row_selection_mode="none",
            height="500px",
        )

    @output
    @render.data_frame
    def summary_table():
        r = result_val.get()
        if r is None or r == "running":
            return None
        if not isinstance(r, dict):
            return None
        p = prepared()
        if p is None or isinstance(p, Exception):
            return None
        df = build_group_summary(p, r["assignments"])
        return render.DataGrid(
            df,
            row_selection_mode="none",
        )

    @output
    @render.data_frame
    def diversity_table():
        r = result_val.get()
        if r is None or r == "running":
            return None
        if not isinstance(r, dict):
            return None
        p = prepared()
        if p is None or isinstance(p, Exception):
            return None
        df = build_diversity(p, r["assignments"])
        return render.DataGrid(
            df,
            row_selection_mode="none",
        )

    # --- Downloads ---
    @output
    @render.ui
    def download_ui():
        r = result_val.get()
        if r is None or r == "running" or not isinstance(r, dict):
            return ui.p(
                "Run optimization first.",
                class_="text-muted",
            )
        return ui.TagList(
            ui.download_button(
                "dl_groups",
                "Download Groups CSV",
                class_="btn-outline-primary btn-sm w-100",
            ),
            ui.br(),
            ui.download_button(
                "dl_summary",
                "Download Summary CSV",
                class_="btn-outline-primary btn-sm w-100 mt-2",
            ),
        )

    @render.download(filename=lambda: f"{uploaded_stem.get()}_Groups.csv")
    def dl_groups():
        r = result_val.get()
        if not isinstance(r, dict):
            return
        p = prepared()
        if p is None or isinstance(p, Exception):
            return
        df = build_assignments(p, r["assignments"])

        rename_map = {
            "name": "Name",
            "preferred_name": "Preferred_Name",
            "pronoun": "Pronoun",
            "group": "Group",
            "total_score": "Total_Score",
        }
        csv_df = df.rename(
            columns={k: v for k, v in rename_map.items() if k in df.columns}
        )
        group_cols = []
        if "Name" in csv_df.columns:
            group_cols.append("Name")
        group_cols.append("Preferred_Name")
        if "Pronoun" in csv_df.columns:
            group_cols.append("Pronoun")
        group_cols.append("Group")
        out = csv_df[group_cols].sort_values("Group")
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        yield buf.getvalue()

    @render.download(filename=lambda: f"{uploaded_stem.get()}_Summary.csv")
    def dl_summary():
        r = result_val.get()
        if not isinstance(r, dict):
            return
        p = prepared()
        if p is None or isinstance(p, Exception):
            return
        summary = build_group_summary(p, r["assignments"])
        diversity = build_diversity(p, r["assignments"])
        combined = summary.merge(diversity, on="group")
        buf = io.StringIO()
        combined.to_csv(buf, index=False)
        yield buf.getvalue()


app = App(app_ui, server)
