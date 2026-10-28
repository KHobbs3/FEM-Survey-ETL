# Benin FEM formative research — table analysis

Cleans, maps and weights the Benin survey (SurveyCTO export) and builds the analysis tables.
The weighted file it writes (`data/2_cleaned/02_fem_survey_benin_weighted.csv`) is what
`../pipeline_output/pipeline/` aggregates for `../benin_app/`.

Survey data is participant personal data: see `../CLAUDE.md`. Everything under `data/` stays local.

## Setup

```bash
python setup_venv.py          # creates .venv and installs requirements.txt; re-run after pulling
source .venv/bin/activate
pip install <package> && pip freeze > requirements.txt   # when adding a library
```

## Run order

Paths and settings are in `src/config.py`; shared functions in `src/funcs.py`.

| Step | File | What it does |
|------|------|--------------|
| 0 | `_0_check_geo.ipynb` | Checks respondent GPS against cluster boundaries |
| 1 | `00_clean_data.ipynb` | Cleans the raw export, drops >90%-missing submissions (written to `data/4_validation_checks/00_unfinished_submissions.csv`); standardises free-text "other" radio stations (`src/stations.py`) and puts them in place of `-22`; writes the station review to `data/4_validation_checks/`: `00_station_other_not_on_questionnaire.csv` (write-in stations that matched no questionnaire choice), `00_station_other_review.csv` (every write-in answer → what it became, `needs_review` flag) and `stations_other_cleaned_DRAFT.csv` (correct it and save as `data/stations_other_cleaned.csv` — the reviewed lookup, which doesn't exist yet — then re-run) |
| 1b | `01b_review_stations.py` | Reviews the station standardisation: prints standardised names and counts, exports answer → station tables to `data/4_validation_checks/` |
| 2 | `01_mapping.ipynb` | Maps choice codes to labels from the XLSForm |
| 3 | `02a_correct_locations.ipynb` | Fixes cluster assignments (e.g. the 8 Toffo respondents) |
| 4 | `02b_calculate_weights_KH.ipynb` | Three-stage weights (below) → `02_fem_survey_benin_weighted.csv` |
| 5 | `02b_validate_weights.ipynb` | Re-derives the weights and checks them |
| 5z | `02z_recode_other.py` | Recodes "Autre" write-ins (occupation, religion, radio language, reasons for use, drivers & barriers) and the free-text reason for non-use into the mapped and weighted files, so every later step uses them. Refreshes the recodes file first (fuzzy matching). Re-run after any of steps 2–4. See "Recoding 'Autre' write-ins" in `../pipeline_output/README.md` |
| 6 | `03_driver_barrier_table_w_counts.ipynb` | Drivers & barriers table |
| 6b | `03b_check_drivers_barriers_quality.py` | Response-quality flags (needs `src/quality.py`, not yet in this folder) |
| 7 | `04_questions_by_station.py`, `04_questions_supply_table_updated.ipynb` | Questions by radio-station coverage area; supply table |
| 8 | `05_*.py`, `05_settlements_radio_coverage.ipynb`, `06_language_clustering_vs_region.py` | Response rate and respondents by province, coverage, language clusters |

Then run `../pipeline_output` (see its README) to refresh the app's CSVs.

## Weighting

`calculate_three_stage_weights()` in `src/funcs.py`:

- **Stage 1 — clusters (PPS within province).** Inclusion probability
  π_i = 1 − (1 − p_i)^k_h, with p_i the cluster's share of its province's size measure and
  k_h the number of distinct clusters surveyed in the province. With
  `PPS_CERTAINTY_UNITS = True`, a cluster with k_h·p_i ≥ 1 is a certainty unit (π = 1); the
  remaining k_h − n_cert draws are spread over the rest of the province, repeated until no new
  cluster crosses the threshold. Weight = 1/π_i. With `USE_STRATUM_WEIGHT = True` this is
  multiplied by (N_h/N)/(k_h/K). That corrects for allocation a second time, because 1/π_i
  already uses k_h, so over-sampled provinces end up under-represented. Benin uses `False`
  (since 2026-10-26, aligned with Niger and Togo).
- **Stage 2 — households.** (cluster population / 5.5) / households visited. Convenience
  selection, so an approximation. Under PPS, stage 1 × stage 2 ≈ N_h / (k_h × 5.5 × m_i), so the
  cluster population cancels out, as long as stage 1's size measure is the same quantity as
  stage 2's population.
- **Stage 3 — respondent.** Eligible adults in the household, as recorded.
- **Non-response.** No separate adjustment (`NONRESPONSE_ADJUSTMENT = False`, since 2026-10-26,
  aligned with Niger and Togo). Stage 3 already lets the respondent stand for every eligible
  adult listed, including those who were away, and stage 2 covers households that didn't take
  part. The old adjustment (divide by draw attempts / eligible adults, with the Borgou day-1 and
  Plateau language rules) multiplied the weight by household size a second time; it is kept
  behind the flag for reference only.
- `normalized_weight` = the above rescaled to mean 1, then post-stratified (below).

Settings in `src/config.py`:

| Setting | Benin now | Meaning |
|---------|-----------|---------|
| `PPS_SIZE_MEASURE` | `"population"` | Eligible-frame column used as the stage 1 size measure (was the gpkg's `proportion` until 2026-10-26; it didn't match population in Atlantique, Couffo and Ouémé) |
| `USE_STRATUM_WEIGHT` | `False` | `False` = stage 1 is the cluster weight only |
| `NONRESPONSE_ADJUSTMENT` | `False` | `True` = the old draw-attempt adjustment (see above) |
| `PPS_CERTAINTY_UNITS` | `True` | Certainty clusters get π = 1 (on since 2026-10-26: Sékou, Ouidah, Doga, Agonlin-Houégbo) |
| `POSTSTRATIFY_ON` | `"province"` | Post-stratify the final weights to frame population; `None` skips it |
| `CLUSTERS_ALLOCATED` | `{}` (to fill in) | Allocation from the sampling `config.R` before the ×2 for replacements; reference only |

Diagnostics (print aggregates only):

- `compare_stratum_weight(settlements_eligible, settlements_chosen, data)` prints each province's
  population, unweighted and weighted share with and without the stratum weight.
- `check_stage_weights(cluster_weights, data_weighted, settlements_eligible, clusters_allocated=CLUSTERS_ALLOCATED, out_dir=DIR_VALIDATE)`
  checks the stage 1 size measure against the stage 2 population, prints k_h against the
  allocation, w1·w2 relative to N_h/(k_h·5.5·m_i), certainty clusters and stage 2 weights below 1,
  mean stage 3, the Horvitz-Thompson population divided by the frame population, and province
  shares by stage. It writes `data/4_validation_checks/02b_stage_weight_check_clusters.csv` with
  one row per cluster and no respondent rows.

### Matching regional population shares

`apply_poststratification(data_weighted, settlements_eligible, on=POSTSTRATIFY_ON)` keeps the
design weight as `normalized_weight_design`. When `on` is set, it replaces `normalized_weight`
with w × N_c / N̂_c, where N̂_c is the sum of design weights in province c and N_c is the
`settlements_eligible` population. The result is renormalised to mean 1
(Statistics Canada, *Survey Methods and Practices*, 7.1.4.1). It prints the factor per cell
(1 = already at its population share), warns about thin cells (<20 respondents or <2 clusters),
and prints the Kish DEFF before and after. Use only province, which is a design stratum.
The same per-province factor is applied to `combined_weight_adjusted` (the pipeline's
`WEIGHT_COL`), with the design weight kept as `combined_weight_design`, so the post-stratified
weights reach the dashboard.

### Limitations

- Once weights are post-stratified, regional shares match the frame by construction, so they no
  longer test the weights. Compare other estimates with DHS/MICS instead.
- Stage 2 is convenience sampling, and stage 3 relies on enumerator-recorded household rosters.
- Confidence intervals should allow for clustering within provinces (strata = province,
  PSU = cluster). The Kish DEFF covers only unequal weights.

## Tests

Synthetic data only (`tests/fixtures/`):

```bash
python -m pytest tests/
```
