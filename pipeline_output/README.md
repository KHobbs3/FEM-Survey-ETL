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

## Weights and key indicators

- Estimates are weighted by `WEIGHT_COL` (`combined_weight_adjusted`), which `table_analysis/02b` post-stratifies to province population shares.
- `family_planning` writes `fp_headline.csv`: mCPR, unmet need, unmet demand, demand satisfied and awareness (all / non-users) for the full sample and all women, with 95% CIs from a Rao-Wu cluster bootstrap (clusters = `CI_CLUSTER_COL`, the corrected `cluster_key`; strata = `CI_STRATUM_COL`, province; 1,000 replicates).
- Unmet need counts "more than 2 years" and "no more children" only (DHS-style; "1–2 years" is excluded). `fp_unmet.csv` uses the same definition.
- `utils.weighted_mean` excludes the -88/-99/-22 skip codes, so travel times and costs average real answers only.

## Data handling

Reads respondent-level data; writes only counts and proportions, outside this folder. `fp_nonuse_reasons.csv` and the radio tables include free-text answers, so review them before sharing.
