# Togo FEM survey — app pipeline

Turns the weighted survey from `../table_analysis` into the aggregated CSVs that the Streamlit app reads (`../togo_app/data/`, kept in a separate repo). Run `table_analysis` first.

## Setup

```bash
pip install -r requirements.txt
```

Paths are in `pipeline/config.py`, or set `FEM_MAPPED_DATA`, `FEM_FINAL_TABLES`, `FEM_XLSFORM`, `FEM_APP_DATA_DIR`, `FEM_STATION_PATH`.

## Run (from this folder)

```bash
python -m pipeline.check_labels        # hard-coded labels vs the XLSForm
python -m pipeline.validate_pipeline   # input columns and values
python -m pipeline.validate_weights    # weight checks
python -m pipeline.run_pipeline        # all pages, or --pages access radio
```

Pages: `respondents`, `access`, `statements`, `family_planning`, `personality`, `personas`, `radio`, `drivers_barriers`.

## Recoding "Autre" write-ins

Respondents who chose "Autre" for `occupation`, `religion`, `radio_language` or `reason_current_use` (`OTHER_RECODE_QUESTIONS` in `config.py`) often wrote a real answer in `<question>_other`. To code these by hand:

1. `python -m pipeline.other_recodes` writes each distinct write-in, its count and a blank `recode_to` column to `table_analysis/data/4_validation_checks/other_writeins.csv`. It prints totals only.
2. Fill in `recode_to` with an existing answer label (French is enough: "Artisan" matches "Artisan/…") or a new label. Leave it blank to keep "Autre". Save the file as `table_analysis/data/meta/other_recodes.csv`.
3. Run the pipeline. `run_pipeline` applies the recodes straight after loading, so every page and the personas use them. With no recodes file it changes nothing.

Re-run step 1 after new data arrives: answers you've already coded keep their `recode_to`, so only new write-ins need reviewing. Both files hold free text, so they stay in the data folder.

## Weights and key indicators

- Estimates are weighted by `WEIGHT_COL` (`combined_weight_adjusted`), which `table_analysis/02b` post-stratifies to province population shares.
- `family_planning` writes `fp_headline.csv`: mCPR, unmet need, unmet demand, demand satisfied and awareness (all / non-users) for the full sample and all women, with 95% CIs from a Rao-Wu cluster bootstrap (clusters = `CI_CLUSTER_COL`, the corrected `cluster_key`; strata = `CI_STRATUM_COL`, province; 1,000 replicates).
- Unmet need counts "more than 2 years" and "no more children" only (DHS-style; "1–2 years" is excluded). `fp_unmet.csv` uses the same definition.
- `utils.weighted_mean` excludes the -88/-99/-22 skip codes, so travel times and costs average real answers only.

## Data handling

Reads respondent-level data; writes only counts and proportions, outside this folder. `fp_nonuse_reasons.csv` and the radio tables include free-text answers, so review them before sharing.
