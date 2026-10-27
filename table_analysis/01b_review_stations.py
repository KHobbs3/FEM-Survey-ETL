"""Review how src/stations.py standardises the free-text "other" radio station answers.

Run from table_analysis/ after 00_clean_data.ipynb:  python 01b_review_stations.py

Prints (station names and counts only, no respondent rows):
  - each station question's XLSForm type (must be select_multiple)
  - how many answers each method handled (lookup / rules / fuzzy / typed / unclear)
  - every distinct standardised station, its method and mention count
  - the "typed" and "fuzzy" names: candidates for new STATION_PATTERNS or lookup rows
Exports to data/4_validation_checks/ for review:
  01b_station_answers.csv      every distinct answer -> standardised entries + method
  01b_station_frequencies.csv  each station/frequency: mentions, frequencies imputed, max distance (km)
"""
import pandas as pd

from src.config import (DIR_CLEANED_DATA, DIR_META_DATA, DIR_STATIONS_OTHER_CLEANED, DIR_VALIDATE,
                        STATIONS_MAX_KM)
from src import stations

data = pd.read_csv(DIR_CLEANED_DATA, low_memory=False)
other_cols = [c for c in data.columns if c.startswith("station_") and c.endswith("_other")]
print(f"station 'other' columns: {other_cols}")

print("\n== XLSForm types ==")
labels = {}
for col in other_cols:
    question = col[: -len("_other")]
    try:
        choices, qtype = stations.station_choices(DIR_META_DATA, question)
    except KeyError as e:
        print(f"  {question}: {e}")
        continue
    labels.update(choices)
    flag = "" if qtype == "select_multiple" else "   <-- NOT select_multiple: merge_other won't apply"
    print(f"  {question}: {qtype} ({len(choices)} choices){flag}")

std = stations.StationStandardiser(lookup=stations.load_lookup(DIR_STATIONS_OTHER_CLEANED),
                                   extra_names=list(labels))
mentions = stations.standardise_answers(data, other_cols, std, max_km=STATIONS_MAX_KM)
mentions = mentions[mentions["method"] != "empty"]
answers, freqs = stations.review_tables(mentions)

print("\n== Answers by method ==")
print(answers.groupby("method")["respondents"].agg(["size", "sum"])
      .rename(columns={"size": "distinct answers", "sum": "respondents"}).to_string())

print("\n== Standardised stations (mentions) ==")
by_station = mentions.groupby(["name", "method"]).size().rename("mentions").reset_index()
print(by_station.sort_values("mentions", ascending=False).to_string(index=False))

for method in ["typed", "fuzzy"]:
    sub = by_station[by_station["method"] == method].sort_values("mentions", ascending=False)
    print(f"\n== {method} ({len(sub)} names, {sub['mentions'].sum()} mentions) ==")
    print(sub[["name", "mentions"]].to_string(index=False) if len(sub) else "  none")

print("\n== Frequencies imputed ==")
print(f"  {int(mentions['freq_imputed'].sum())} imputed within {STATIONS_MAX_KM} km; "
      f"max distance {mentions.loc[mentions['freq_imputed'], 'freq_donor_km'].max():.1f} km; "
      f"{int(mentions['freq'].isna().sum())} mentions still without a frequency")

DIR_VALIDATE.mkdir(parents=True, exist_ok=True)
answers.to_csv(DIR_VALIDATE / "01b_station_answers.csv", index=False)
freqs.to_csv(DIR_VALIDATE / "01b_station_frequencies.csv", index=False)
print(f"\nExported to {DIR_VALIDATE}/: 01b_station_answers.csv, 01b_station_frequencies.csv")
