# Togo FEM formative research — ETL

| Folder | Does |
|---|---|
| [table_analysis/](table_analysis/) | Cleans, locates and weights the survey; builds analysis tables |
| [pipeline_output/](pipeline_output/) | Aggregates the weighted survey for the Streamlit app |

Run `table_analysis` first. Not in this repo: `raw_data/` (SurveyCTO export) and `togo_app/` (separate repo).

## Data handling

Survey data is participant personal data under GDPR. It's gitignored and never committed; see [CLAUDE.md](CLAUDE.md). `.claude/` holds the guard that keeps Claude Code away from the data.
