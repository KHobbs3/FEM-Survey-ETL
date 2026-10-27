# Benin FEM Survey — Analysis Pipeline

Reads the weighted survey file from `table_analysis/` (participant data) and writes
pre-aggregated CSVs to `benin_app/data/`. The Streamlit app reads only those CSVs, loading the
copies uploaded to Google Drive (file IDs in `benin_app/src/data_loader.py`).

```
1_formative_research/                ← project root
├── table_analysis/
│   └── data/2_cleaned/02_fem_survey_benin_weighted.csv   ← input (PII, never committed)
├── pipeline_output/
│   └── pipeline/                    ← this pipeline
│       ├── config.py                — paths (anchored to the project root), survey constants, label dicts
│       ├── utils.py                 — weighted aggregation helpers
│       ├── run_pipeline.py          — master runner
│       ├── etl_respondents.py       — Respondents
│       ├── etl_access.py            — Health Access & Supply
│       ├── etl_statements.py        — Statement Agreement
│       ├── etl_family_planning.py   — Family Planning
│       ├── etl_personality.py       — Personality & Influencers
│       ├── etl_personas.py          — Personas (k-modes clustering)
│       ├── etl_radio.py             — Radio stations (uses the CloudRF coverage geopackages)
│       ├── validate_pipeline.py     — checks the input and the generated CSVs
│       └── validate_weights.py      — checks the weight columns
└── benin_app/
    ├── app.py
    ├── data/                        ← CSVs written here
    ├── pipeline/agg_helpers.py      — helpers for reading the long-format CSVs in app pages
    └── src/                         — page_*.py, data_loader.py
```

## Setup

```bash
pip install pandas numpy kmodes geopandas
```

## Running the pipeline

```bash
cd pipeline_output
python pipeline/run_pipeline.py                                   # all pages
python pipeline/run_pipeline.py --pages access statements radio   # some pages

# Override paths
FEM_MAPPED_DATA=/path/to/02_fem_survey_benin_weighted.csv \
FEM_APP_DATA_DIR=/path/to/benin_app/data \
python pipeline/run_pipeline.py
```

Pages: `respondents`, `access`, `statements`, `family_planning`, `personality`, `personas`, `radio`.

Estimates are weighted by `WEIGHT_COL` in `config.py` (`combined_weight_adjusted`). Since
2026-10-26 `table_analysis` post-stratifies this column as well as `normalized_weight` (the
pre-post-stratification design weight is kept as `combined_weight_design`), so the app's
figures use the post-stratified weights.

`fp_headline.csv` (Family Planning) holds the key indicators -- mCPR, unmet need, unmet demand,
demand satisfied, awareness (all / non-users) -- for the full sample and all women, with 95%
CIs from a Rao-Wu cluster bootstrap (clusters = `CI_CLUSTER_COL`, the corrected
`sample_number`; strata = `CI_STRATUM_COL`, province). Unmet need counts "more than 2 years"
and "no more children" only (DHS-style; "1-2 years" is excluded since 2026-10-26).
`utils.weighted_mean` excludes the -88/-99/-22 skip codes.

## What each ETL produces

| Page | ETL file | Outputs in `benin_app/data/` |
|------|----------|------------------------------|
| Respondents | `etl_respondents.py` | `respondents_profile.csv` |
| Access & Supply | `etl_access.py` | `access_stockouts.csv`, `access_stockout_responses.csv`, `access_travel.csv`, `access_affordability.csv`, `access_composite.csv` |
| Statements | `etl_statements.py` | `statements_heatmap.csv` (needs `statement_labels.csv` in the same folder) |
| Family Planning | `etl_family_planning.py` | `fp_funnel.csv`, `fp_timing.csv`, `fp_methods.csv`, `fp_reason_use.csv`, `fp_intent.csv`, `fp_unmet.csv`, `fp_headline.csv`, `fp_nonuse_reasons.csv` |
| Personality | `etl_personality.py` | `personality_life_goals.csv`, `personality_goals_achievable.csv`, `personality_role_models.csv`, `personality_likeable_traits.csv`, `personality_forming_beliefs.csv`, `personality_decision_confident.csv`, `personality_wellbeing.csv` |
| Personas | `etl_personas.py` | `personas_{centroids,profile,elbow}.csv` (also `_by_gender` and other splits), `culture_clusters_{centroids,profile,elbow}.csv` |
| Radio | `etl_radio.py` | `Benin_question_table_by_station_<date>.csv`, `Benin_question_table_by_state_<date>.csv`, `Benin_radio_station_overlap_<date>.csv` |

## PII guarantee

- The pipeline reads the weighted survey file and writes only aggregated statistics.
- Minimum cell sizes are not enforced. Add a filter in `utils.py` if needed.
- `fp_nonuse_reasons.csv` contains free-text answers. Review it before publishing in case it
  contains names or identifying details.
- `benin_app/data/` holds no individual records.

## Validation

```bash
python pipeline/validate_pipeline.py --raw       # input file only (before running)
python pipeline/validate_pipeline.py --outputs   # generated CSVs only (after running)
python pipeline/validate_pipeline.py             # both
python pipeline/validate_weights.py [--data ... --eligible ... --chosen ...]
```

`validate_pipeline.py` checks that the weight column has no nulls, zeros or negatives; that
`use` values match `ALL_USE_GROUPS`; age-group dash style; statement, funnel and stockout
encodings; `WTT_MAP` coverage; that expected output files and columns exist; all-zero metrics;
and proportions above 1.

`validate_weights.py` checks inputs and location-code alignment, each stage, null diagnosis and
the normalised-weight DEFF. It still expects Niger's column and file names (`state`,
`niger_preselected_minpop_733.csv`), so pass `--eligible`/`--chosen` explicitly; its stage 1
and stratum checks don't apply to Benin's `province` columns until it is updated. The main
weight diagnostics for Benin are in `table_analysis` (`check_stage_weights`, `02b_validate_weights.ipynb`).
