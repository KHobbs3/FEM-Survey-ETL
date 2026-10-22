"""
Aggregate-only average within-household response rate by province.

Uses the `household_response_rate` column already written by funcs.py into
02_fem_survey_benin_weighted.csv -- that column is capped at 1.0 per row
(enumerator-error rows where random_draw_attempts > eligible_adults_in_hh
are clipped, per the 2026-08-19 decision) and already carries the Borgou
day-1 and Plateau-language pooled-rate overrides. Averaging it directly,
rather than recomputing random_draw_attempts / eligible_adults_in_hh from
scratch, is what keeps a province average from exceeding 100%.

Per CLAUDE.md: this script is meant to be run by the researcher, who reports
back the printed aggregate. Claude does not read/open/load the underlying
data file itself.
"""
import pandas as pd
from src.config import DIR_WEIGHTED_DATA

data = pd.read_csv(DIR_WEIGHTED_DATA)

if "household_response_rate" not in data.columns:
    raise SystemExit(
        "household_response_rate column not found -- make sure "
        "02_fem_survey_benin_weighted.csv was produced by the current "
        "funcs.py (non-response adjustment section)."
    )

out_of_range = data.loc[
    (data["household_response_rate"] < 0) | (data["household_response_rate"] > 1),
    "household_response_rate",
]
if len(out_of_range) > 0:
    print(f"WARNING: {len(out_of_range)} rows still outside [0, 1] -- "
          f"the cap in funcs.py may not have been re-run on this file.\n")

rates = (
    data.groupby("province")["household_response_rate"]
    .mean()
    .mul(100)
    .round(1)
    .rename("Average within-household response rate (%)")
    .reset_index()
    .rename(columns={"province": "Province"})
)
print(rates.to_string(index=False))
