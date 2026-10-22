# Claude Instructions
Survey data is participant personal data under GDPR and our consent commitments. Do not read, open, print, or load any file under the data directory. Do not use cat, head, df.head(), print(df), df.sample(), or View() on participant data. Work from the SurveyCTO form definition, the codebook, and synthetic fixtures in tests/fixtures/. Write scripts; the researcher runs them and reports back aggregate output.

## What counts as participant data here
- Everything under `data/` (project level) and `table_analysis/data/` — including `meta/`, because the QA review file (`Togo QA Quantitative survey - Full Data with QA.csv`) is a full copy of the submissions. Treat the XLSForm and sample files there as off-limits too; ask the researcher for column names instead (`table_analysis/scripts/inspect_input_schemas.py` prints schemas only).
- Any SurveyCTO export (`*_WIDE*`), and `table_analysis/radio_matching/` (free-text "other" answers).
- Saved notebook outputs. Notebooks are committed with outputs cleared; if one has outputs, don't open it — ask the researcher to clear them (`jupyter nbconvert --clear-output --inplace <nb>`).

## How this is enforced
`.claude/settings.json` denies reads of those paths and runs `.claude/hooks/pii_guard.py` before Read/Edit/Write/NotebookEdit/Grep/Bash. The guard also blocks executing the analysis notebooks/scripts and unrestricted recursive searches from folders containing data. Don't work around it; if it blocks something legitimate, tell the researcher.

## Writing code
- Never hard-code submission keys, names, phone numbers, GPS points or enumerator emails in code or comments. Per-respondent decisions go in a CSV under `table_analysis/data/meta/` (e.g. `manual_location_overrides.csv`) that the code reads at runtime.
- Notebook cells must not display row-level frames (no bare `df`, `.head()`, `.loc[...]` of respondents); show counts or aggregates, and export rows for review to `data/4_validation_checks/`.
- Test logic against synthetic data in `table_analysis/tests/` (`python3 -m pytest tests/` from `table_analysis/`).
