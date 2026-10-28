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

Pages: `respondents`, `access`, `statements`, `family_planning`, `personality`, `personas`, `radio`, `driver_barrier_groups`.

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

## Recoding "Autre" write-ins and free-text reasons

The recoding is done in `table_analysis`, on the survey data files, so every later notebook, script, this pipeline (and, in Niger, `3_linkage`) uses the recoded answers. Niger, Benin and Togo use the same code (`pipeline/other_recodes.py`) and recode the same questions. Each survey is matched against its own answer labels. Questions are listed in `config.py`:
- **"Autre" write-ins** (`OTHER_RECODE_QUESTIONS`): `occupation`, `religion`, `radio_language`, `reason_current_use`, and the drivers & barriers questions: the main reasons `reason_use_main` / `reason_nonuse_main` and the lists `reason_use_mc`, `drivers_nonusers`, `reason_nonuse_mc`, `barriers_users`. People who chose "Autre" often wrote a real answer in `<question>_other`. A question without a `<question>_other` column in the data is skipped (a `[other] ... skipped` line says so).
- **Free-text answers** (`FREE_TEXT_RECODE_QUESTIONS`): `reason_current_nonuse`, the open-ended reason for not using contraception.

Coding uses local fuzzy matching. Nothing is sent to any API. (It replaced the earlier AI coding. `CLEAR_AI_CODES = False` in `config.py`: Benin's existing codes with `coded_by = ai` are kept as codes; fuzzy matching only codes new or blank texts.)

1. **Recode the data files**: from `table_analysis/`, `python 02z_recode_other.py`, after mapping and weights and before `03` (see `table_analysis/README.md`). It:
   - **Refreshes the recodes file**, `table_analysis/data/meta/other_recodes-ai-gen.csv` (`OTHER_RECODES_PATH`; the file name is historical, and it now holds fuzzy and hand codes).
     - Every new distinct text is added.
     - Each uncoded text is compared with the question's own answer labels in this survey (`CODE_LISTS`: its own choices, the drivers list `reason_use_mc` / `drivers_nonusers`, or the barriers list `reason_nonuse_mc` / `barriers_users`). The comparison uses word- and character-level similarity, so typos and word forms still match.
     - `recode_to` is filled only when the text *is* a label (`coded_by = fuzzy exact`), or the best label scores ≥ 0.55 and leads the next by ≥ 0.10 (`coded_by = fuzzy <score>`).
     - Every other text gets its 3 best `candidates` and a `group` number linking similar texts. When a group's coded texts all share one code, it is copied to the uncoded ones (`coded_by = group`).
     - Codes already in the file are never overwritten. Texts no longer in the data stay in the file with their codes.
     - The answer labels per question go to `table_analysis/data/meta/other_recode_choices.csv`.
   - **Rewrites the survey data files** with the recoded answers. An "Autre" item becomes the coded answer; the rest of a multi-select answer is kept, and a free-text answer is replaced. Only the recoded questions change. Each one gets a `_pre_recode__<question>` column with the answers as given. Re-running always starts from those answers, so it picks up new codes and never recodes twice.

   It prints counts only. `--no-refresh` recodes with the file as it is.
2. **Review** the recodes file:
   - Check the `fuzzy` / `group` codes, and fill `recode_to` for blank rows: an existing answer label (one language is enough, e.g. "Artisan" matches "Artisan/…") or a new label.
   - `-` keeps the answer as written and stops it being suggested again.
   - Blank means not reviewed yet.

   Then re-run `02z_recode_other.py`, or the pipeline.
3. **Run the pipeline.** `run_pipeline` runs the same step on the file it reads (`DIR_MAPPED_DATA`) before loading it. That file is always current, even if the recodes file changed after `02z` last ran. Use `--no-recode-refresh` to skip the refresh.

Re-run `02z_recode_other.py` after re-running any notebook that rewrites the data files (mapping, locations, weights). `python -m pipeline.other_recodes --fuzzy` refreshes the recodes file only; with no flag, it exports the texts, uncoded, to `table_analysis/data/4_validation_checks/other_writeins.csv`. These files hold free text, so they stay in the data folder.

## Drivers & barriers by parent category

`--pages driver_barrier_groups` writes `driver_barrier_groups.csv`. Answer options that share a parent category (the label text before ":", e.g. "Peur des effets secondaires : prise de poids") are grouped. Per parent it gives the weighted and unweighted % of respondents who picked **at least 1** and **at least 3** of its options, overall and by user group, gender, age group and region. The base is everyone who answered the list. Each respondent's list is the user or non-user version of the question (`DB_LISTS`). "Autre" and don't-know / none answers aren't grouped. "At least 3" is only given for parents with 3+ options. The Drivers & Barriers page shows it under the overview chart.

## Role models by gender and age

`--pages personality` adds a gender x age group split (`gender_age`, e.g. "Femme ... | 21-30") to `personality_role_models.csv` and `personality_likeable_traits.csv`, next to the existing user group, gender and age group splits. Every row of these files (and the other select-multiple files written by `_multiselect_long`) now has `n`: the number of respondents in that group, which is the denominator of its `proportion`. On the Personality page, the role models section has Gender and Age group filters that can be combined. Each option shows its group size, and a line under the filters gives the selected group's size, flagging groups under 30 respondents (`SMALL_GROUP_N` in `page_personality_traits.py`). Re-upload both files to Drive after regenerating them, or the combined filter shows a "not generated yet" message.

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
