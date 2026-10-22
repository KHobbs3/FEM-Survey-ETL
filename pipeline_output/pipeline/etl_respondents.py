"""
ETL: Respondent Profile page
Produces pre-aggregated CSVs — no PII in outputs.

Output files:
  respondents_profile.csv  — weighted counts and proportions for key demographics
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import APP_DATA_DIR, PROVINCE_TO_REGION
from pipeline.utils import save


# Columns to profile and their display labels
DEMO_COLS = {
    "use":             "FP use group",
    "gender":          "Gender",
    "age_group":       "Age group",
    "occupation":      "Occupation",
    "religion":        "Religion",
    "urban_rural": "Settlement type",
    "province":        "Province",
    "region":          "Region",
}

USE_LABELS = {
    "user":        "Current user",
    "past_user":   "Past user",
    "future_user": "Future user",
    "nonuser":     "Non-user",
}

GENDER_LABELS = {
    "Homme Sunnu": "Homme",
    "Femme Nyɔnu": "Femme",
}


def _clean_bilingual(series):
    """'French/Fon' → 'French' (Benin's order -- first segment, not last).
    Handles '|'-joined multi-select values (Benin's occupation field can be
    multi-select, unlike Niger's)."""
    def _clean_one(text):
        if pd.isna(text):
            return np.nan
        text = str(text).strip()
        if "|" in text:
            parts = [_clean_one(p) for p in text.split("|")]
            parts = [p for p in parts if pd.notna(p)]
            return "|".join(parts) if parts else np.nan
        if "/" in text:
            return text.split("/", 1)[0].strip()
        return text if text else np.nan
    return series.apply(_clean_one)


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [respondents] running...")

    df = df.copy()
    if "province" in df.columns and "region" not in df.columns:
        df["region"] = df["province"].map(PROVINCE_TO_REGION)
        unmapped = df.loc[df["region"].isna() & df["province"].notna(), "province"].unique().tolist()
        if unmapped:
            print(f"  [respondents] WARNING: provinces with no region mapping: {unmapped}")

    rows = []

    # ── Total sample ──────────────────────────────────────────────────────────
    rows.append({
        "variable":   "_total",
        "category":   "Total",
        "count":      len(df),
        "proportion": 1.0,
    })

    # ── Per-variable breakdowns ───────────────────────────────────────────────
    for col, label in DEMO_COLS.items():
        if col not in df.columns:
            print(f"  [respondents] '{col}' not found, skipping")
            continue

        valid = df[[col]].dropna().copy()
        n_total = len(valid)
        if n_total == 0:
            print(f"  [respondents] no valid data for '{col}', skipping")
            continue

        # Clean category labels
        if col == "use":
            valid[col] = valid[col].map(USE_LABELS).fillna(valid[col])
        elif col == "gender":
            valid[col] = valid[col].map(GENDER_LABELS).fillna(valid[col])
        elif col in ("occupation", "religion"):
            valid[col] = _clean_bilingual(valid[col])

        for cat, grp in valid.groupby(col):
            n = len(grp)
            rows.append({
                "variable":         col,
                "category":         str(cat),
                "count":            n,
                "proportion":       n / n_total,
            })

    result = pd.DataFrame(rows)
    save(result, os.path.join(APP_DATA_DIR, "respondents_profile.csv"),
         "respondent profile")
    print("  [respondents] done.")
