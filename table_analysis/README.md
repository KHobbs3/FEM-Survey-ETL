# Formative research — table analysis

Cleans, maps, locates and weights the FEM household survey (SurveyCTO), then builds the driver/barrier and supply tables. Reused per country; all country settings live in [src/config.py](src/config.py).

## Data handling

Survey data, the QA file (a full copy of the submissions) and saved notebook outputs are participant personal data under GDPR. See [../CLAUDE.md](../CLAUDE.md).

- Data stays in `data/` (gitignored). Never paste rows into code, comments or chats.
- Clear notebook outputs before committing: `python -m nbstripout *.ipynb`.
- Per-respondent decisions go in CSVs under `data/meta/`, never hard-coded.

## Setup

Run everything from `table_analysis/`.

```bash
python setup_venv.py && source .venv/bin/activate
python -m pytest tests/       # synthetic data only
```

## Layout

```
data/                    # gitignored
  2_cleaned/             # intermediates
  3_final/               # output tables
  4_validation_checks/   # row-level exports for review
  meta/                  # XLSForm, QA file, sampling files, manual decisions
src/
  config.py              # paths + country settings
  funcs.py               # sampling frames, household IDs, weights
  locations.py           # cluster assignment
  stations.py            # "other" radio station cleaning
  quality.py             # 03b checks and thresholds
scripts/inspect_input_schemas.py   # prints schemas only, safe to share
tests/                   # synthetic-data tests
```

The raw SurveyCTO export is read from `../raw_data/`.

## Pipeline

| Step | Does | Writes |
|---|---|---|
| `00_clean_data.ipynb` | Drops empty/unfinished/short (<18 min) submissions; derives use, gender, age, travel time | `2_cleaned/*_cleaned.csv`, `4_validation_checks/00_*` |
| `01_mapping.ipynb` | Cleans "other" stations; maps codes to labels | `2_cleaned/*_mapped.csv`, `4_validation_checks/01_*` |
| `02a_correct_locations.ipynb` | Assigns each respondent a cluster | `2_cleaned/*_located.csv`, `02a_clusters.csv`, `4_validation_checks/02a_*` |
| `02b_calculate_weights.ipynb` | Three-stage weights | `2_cleaned/02_*_weighted.csv`, `4_validation_checks/02b_*` |
| `03_driver_barrier_table_w_counts.ipynb` | Weighted driver/barrier tables by use group, age, gender, urban/rural, department | `3_final/<COUNTRY>_drivers-barriers_table_*.csv` |
| `03b_check_drivers_barriers_quality.py` | Quality flags (straightlining, "don't know", main reason not selected, many reasons, unknown answers, broken statement links). Prints rates only; drops nothing | `4_validation_checks/03b_respondent_flags.xlsx`, `03b_unknown_answers.csv`, `03b_enumerator_codes.csv` |
| `04_questions_supply_table_updated.ipynb` | Supply question table | `3_final/<COUNTRY>_supply_question_table_*.xlsx` |
| `04_questions_by_station.py` | Question tables by station coverage | `3_final/<COUNTRY>_question_table_by_station_*.xlsx` |
| `05_settlements_radio_coverage.ipynb`, `05_respondents_by_province.py` | Coverage and counts | console |
| `06_language_clustering_vs_region.py` | Radio language vs `PROVINCE_TO_REGION`; skipped when `None` (Togo) | `3_final/language_region_clustering_*.csv` |

After 03b, decide what to do with flagged interviews, then re-run from where they'd be excluded so weights are recomputed.

### 02a: locations

Togo has no cluster ids, so clusters link on settlement name (`CLUSTER_KEY = "Name"`), ignoring accents, case and punctuation.

1. Field-schedule settlements → the sampled cluster their GPS point falls in.
2. Respondents, highest priority first: **override** (`manual_location_overrides.csv`) → **gps** (inside a sampled polygon) → **name** (QA location) → **gps_snap** (nearest within 1 km) → **unresolved** (kept, exported).
3. `cluster_name` = `selected_name` from `DIR_LOCATION_REVIEW`. Review `02a_manual_review_vs_sample.csv` and save it there.
4. `02a_manual_review_vs_sample_keys.csv` (in `4_validation_checks/`) lists the respondents behind each review row: `review_row` (the row number in the review CSV opened in a spreadsheet, header = row 1), the row's group columns, the final `cluster_name` and the `KEY` / `SubmissionDate` to look up in the QA document. Its coordinates are settlement points, not respondents' GPS.

### 02b: weights

`calculate_three_stage_weights()` in [src/funcs.py](src/funcs.py), normalised to mean 1:

1. **Cluster**: stratified PPS, 1 / (1 − (1 − p)^k); p = `proportion` within stratum, k = clusters surveyed there. No extra stratum factor (`USE_STRATUM_WEIGHT = False`): 1/π already reflects each province's allocation.
2. **Household**: (cluster population / household size) / households interviewed.
3. **Respondent**: eligible adults in the household. No separate non-response adjustment: the respondent is a random pick among available adults, so this lets them stand for everyone listed (the draw-attempt adjustment in `funcs.py` runs only for Benin, and is off there too since 2026-10-26).

**Post-stratification** (`POSTSTRATIFY_ON = "province"`, in `02b`): each province's weights are scaled so its weighted share matches its frame population share (Statistics Canada 7.1.4.1). The factor is applied to both `normalized_weight` and `combined_weight_adjusted` — the pipeline's `WEIGHT_COL` — with the design weights kept as `normalized_weight_design` / `combined_weight_design`. Before 2026-10-26 only `normalized_weight` was post-stratified, so the app's figures were not.

`load_sampling_frames()` rebuilds the frame as the draw saw it (`EXCLUDED_CLUSTERS`, `STRATA_MAP`, `MIN_CLUSTER_POPULATION`), then links each surveyed cluster to one frame row by point, else name. Unlinkable clusters stop the run.

## New country

1. Set `COUNTRY`; search for `TODO(` and the previous country's name.
2. Put inputs in `data/`, run `python scripts/inspect_input_schemas.py`, and fill in file paths, `QA_*` columns, `*_RENAME` maps and `CLUSTER_KEY`.
3. Mirror the sample draw: `STRATA_MAP`, `MIN_CLUSTER_POPULATION` inputs, `EXCLUDED_CLUSTERS`.
4. Set `AVG_HOUSEHOLD_SIZE`, `GEO_SPLIT_COL`, `PROVINCE_TO_REGION`, `UTM_CRS`, `DIR_STATION`.
5. Check `STATEMENT_LIKERT_MAP` and `driver_barrier_statement_links_<country>.csv` (03b reports broken links).
6. Gate one-off fixes with `if COUNTRY == "..."` (as Benin's are in `funcs.py`).
7. Run the tests, then the pipeline in order, reviewing `4_validation_checks/`.
