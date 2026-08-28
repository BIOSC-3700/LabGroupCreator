# Lab Group Assigner

A browser-based tool for assigning students to balanced lab groups. Runs entirely in the browser using Shinylive (WebAssembly) -- student data never leaves the browser.

## Usage

Upload a CSV roster with the following columns:

- **Preferred_name**: student display name (used for same-name separation)
- **Pronoun**: `she`, `he`, or similar (normalized to She/He/Unknown for balancing)
- Five survey columns with Likert responses (e.g., "Very confident", "Mostly confident", etc.)

The app auto-detects column roles from common survey column names. If pronouns are embedded in the name field (e.g., "Alice Smith (she/her)"), use the extraction tool in Tab 1 to split them out.

### Workflow

1. **Tab 1 -- Load Data**: Upload CSV or load the example roster, configure column roles and solve settings.
2. **Tab 2 -- Verify**: Review validation warnings, the recoded numeric matrix, and the problem summary.
3. **Tab 3 -- Results**: Run the optimizer, view group assignments, and download CSVs.

### Group sizes

Groups are sized as 3s and 4s, maximizing the number of groups of 4. Every roster of 6 or more students works -- there is no "not divisible by 4" restriction.

### Performance

The MILP solver runs in the browser via WebAssembly. Typical solve times:

| Students | Time |
| ---: | ---: |
| 24 | < 1 s |
| 60 | ~2 s |
| 100 | ~6 s |
| 120 | ~20 s |

The first page load downloads ~50 MB of Python/scipy packages (cached afterward).

## Development

```bash
uv sync
uv run pytest tests/ -v
uv run shiny run app.py
```

### CLI

```bash
uv run labgroupassigner examples/test_roster.csv
```

## Deployment

The app is deployed as a static site via GitHub Pages. A push to `main` triggers CI which runs tests, exports the shinylive bundle, and deploys.

To export locally:

```bash
uv run shinylive export . docs --subdir .
touch docs/.nojekyll
```
