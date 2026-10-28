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

The recoding is done in `table_analysis`, on the survey data files, so every later notebook, script, this pipeline (and, in Niger, `3_linkage`) uses the recoded answers. Niger, Benin and Togo use the same code (`pipeline/other_recodes.py`) and recode the same questions. Each survey is matched against its own answer labels. Questions are listed in `config.py`:
- **"Autre" write-ins** (`OTHER_RECODE_QUESTIONS`): `occupation`, `religion`, `radio_language`, `reason_current_use`, and the drivers & barriers questions: the main reasons `reason_use_main` / `reason_nonuse_main` and the lists `reason_use_mc`, `drivers_nonusers`, `reason_nonuse_mc`, `barriers_users`. People who chose "Autre" often wrote a real answer in `<question>_other`. A question without a `<question>_other` column in the data is skipped (a `[other] ... skipped` line says so).
- **Free-text answers** (`FREE_TEXT_RECODE_QUESTIONS`): `reason_current_nonuse`, the open-ended reason for not using contraception.

Coding uses local fuzzy matching. Nothing is sent to any API. (It replaced the earlier AI coding. `CLEAR_AI_CODES = False` in `config.py`: Togo's existing codes with `coded_by = ai`, checked one by one, are kept as codes; fuzzy matching only codes new or blank texts.)

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

## Weights and key indicators

- Estimates are weighted by `WEIGHT_COL` (`combined_weight_adjusted`), which `table_analysis/02b` post-stratifies to province population shares.
- `family_planning` writes `fp_headline.csv`: mCPR, unmet need, unmet demand, demand satisfied and awareness (all / non-users) for the full sample and all women, with 95% CIs from a Rao-Wu cluster bootstrap (clusters = `CI_CLUSTER_COL`, the corrected `cluster_key`; strata = `CI_STRATUM_COL`, province; 1,000 replicates).
- Unmet need counts "more than 2 years" and "no more children" only (DHS-style; "1–2 years" is excluded). `fp_unmet.csv` uses the same definition.
- `utils.weighted_mean` excludes the -88/-99/-22 skip codes, so travel times and costs average real answers only.

## Data handling

Reads respondent-level data; writes only counts and proportions, outside this folder. `fp_nonuse_reasons.csv` and the radio tables include free-text answers, so review them before sharing.
