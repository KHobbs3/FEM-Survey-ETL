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

Pages: `respondents`, `access`, `statements`, `family_planning`, `personality`, `personas`, `radio`, `drivers_barriers`, `driver_barrier_groups`.

## Recoding "Autre" write-ins and free-text reasons

Two kinds of answers get coded (questions are listed in `config.py`):
- **"Autre" write-ins** (`OTHER_RECODE_QUESTIONS`): `occupation`, `religion`, `radio_language`, `reason_current_use`, `reason_nonuse_main`. People who chose "Autre" often wrote a real answer in `<question>_other`.
- **Free-text answers** (`FREE_TEXT_RECODE_QUESTIONS`): `reason_current_nonuse`, the open-ended reason for not using contraception.

Steps (run from `pipeline_output/`):

1. `python -m pipeline.other_recodes --ai` has Claude code every distinct text (`AI_CODE_LISTS`): the closest answer from the question's own choices (occupation, religion, ...) or, for the reasons for non-use, from the barriers list (`reason_nonuse_mc` / `barriers_users`). When nothing fits it gives a short new theme, so similar leftover answers are grouped together. It writes `table_analysis/data/meta/other_recodes-ai-gen.csv` (`OTHER_RECODES_PATH`), with `coded_by = ai` on each AI code. It prints totals only.
   - This sends the distinct answer texts, and nothing else (no IDs, no other columns), to the Anthropic API. It needs `pip install anthropic` and an `ANTHROPIC_API_KEY` (or `ant auth login`).
   - Without `--ai` it only exports the texts, uncoded, to `table_analysis/data/4_validation_checks/other_writeins.csv`.
2. Run the pipeline. `run_pipeline` applies `other_recodes-ai-gen.csv` straight after loading, so every page and the personas use the AI codes until they're reviewed. With no recodes file it changes nothing.
3. Review: correct `recode_to` in the file (an existing answer label, where one language is enough, e.g. "Artisan" matches "Artisan/…"; or a new label; blank keeps the answer as is), then re-run the pipeline. Once reviewed, you can save it as `other_recodes.csv` and point `OTHER_RECODES_PATH` at it.

Re-running `--ai` after new data keeps every existing `recode_to` and only codes the new texts. Both files hold free text, so they stay in the data folder.

## Drivers & barriers by parent category

`--pages driver_barrier_groups` writes `driver_barrier_groups.csv`. Answer options that share a parent category (the label text before ":", e.g. "Peur des effets secondaires : prise de poids") are grouped. Per parent it gives the weighted and unweighted % of respondents who picked **at least 1** and **at least 3** of its options, overall and by user group, gender, age group and region. The base is everyone who answered the list. Each respondent's list is the user or non-user version of the question (`DB_LISTS`). "Autre" and don't-know / none answers aren't grouped. "At least 3" is only given for parents with 3+ options. The Drivers & Barriers page shows it under the overview chart.

## Role models by gender and age

`--pages personality` adds a gender x age group split (`gender_age`, e.g. "Femme ... | 21-30") to `personality_role_models.csv` and `personality_likeable_traits.csv`, next to the existing user group, gender and age group splits. Every row of these files (and the other select-multiple files written by `_multiselect_long`) now has `n`: the number of respondents in that group, which is the denominator of its `proportion`. On the Personality page, the role models section has Gender and Age group filters that can be combined. Each option shows its group size, and a line under the filters gives the selected group's size, flagging groups under 30 respondents (`SMALL_GROUP_N` in `page_personality_traits.py`). Re-upload both files to Drive after regenerating them, or the combined filter shows a "not generated yet" message.

## Weights and key indicators

- Estimates are weighted by `WEIGHT_COL` (`combined_weight_adjusted`), which `table_analysis/02b` post-stratifies to province population shares.
- `family_planning` writes `fp_headline.csv`: mCPR, unmet need, unmet demand, demand satisfied and awareness (all / non-users) for the full sample and all women, with 95% CIs from a Rao-Wu cluster bootstrap (clusters = `CI_CLUSTER_COL`, the corrected `cluster_key`; strata = `CI_STRATUM_COL`, province; 1,000 replicates).
- Unmet need counts "more than 2 years" and "no more children" only (DHS-style; "1–2 years" is excluded). `fp_unmet.csv` uses the same definition.
- `utils.weighted_mean` excludes the -88/-99/-22 skip codes, so travel times and costs average real answers only.

## Data handling

Reads respondent-level data; writes only counts and proportions, outside this folder. `fp_nonuse_reasons.csv` and the radio tables include free-text answers, so review them before sharing.
