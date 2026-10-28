"""
validate_pipeline.py
====================
Run BEFORE and AFTER the pipeline to catch data issues early.

Usage:
    python pipeline/validate_pipeline.py --raw          # validate raw survey CSV
    python pipeline/validate_pipeline.py --outputs      # validate generated CSVs
    python pipeline/validate_pipeline.py --all          # both (default)

The script never reads or prints PII — it only prints column names,
value distributions, counts, and structural summaries.
"""

import argparse
import os
import sys
import glob
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ── Allow running from project root or pipeline/ ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import (
    DIR_MAPPED_DATA, APP_DATA_DIR, WEIGHT_COL,
    USER_GROUPS, NONUSER_GROUPS, ALL_USE_GROUPS,
    SPLIT_COLS, WTT_MAP,
    CONTRACEPTIVE_METHODS, LIFE_GOALS, ROLE_MODELS,
    LIKEABLE_TRAITS, FORMING_BELIEFS, DECISION_CONFIDENT,
)
from pipeline.etl_family_planning import WANTS_TO_DELAY_CODES, WANTS_TO_DELAY_LABELS, _strip_hausa

# ── ANSI colours for terminal output ─────────────────────────────────────────
RED   = "\033[91m"
YEL   = "\033[93m"
GRN   = "\033[92m"
BOLD  = "\033[1m"
RESET = "\033[0m"

_errors   = []
_warnings = []


def _err(msg):
    _errors.append(msg)
    print(f"  {RED}✗ ERROR  {RESET}{msg}")


def _warn(msg):
    _warnings.append(msg)
    print(f"  {YEL}⚠ WARN   {RESET}{msg}")


def _ok(msg):
    print(f"  {GRN}✓ ok     {RESET}{msg}")


def _section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_raw(df):
    print(f"\n{BOLD}RAW SURVEY DATA VALIDATION{RESET}")
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    _validate_raw_core(df)
    _validate_raw_statements(df)
    _validate_raw_family_planning(df)
    _validate_raw_access(df)
    _validate_raw_personality(df)


def _validate_raw_core(df):
    _section("Core / shared columns")

    # Weight column
    if WEIGHT_COL not in df.columns:
        _err(f"Weight column '{WEIGHT_COL}' not found. All weighted calculations will fail.")
    else:
        nulls = df[WEIGHT_COL].isna().sum()
        zeros = (df[WEIGHT_COL] == 0).sum()
        neg   = (df[WEIGHT_COL] < 0).sum()
        if nulls:
            _warn(f"'{WEIGHT_COL}' has {nulls:,} null values — those rows excluded from all weighted stats.")
        if zeros:
            _warn(f"'{WEIGHT_COL}' has {zeros:,} zero values — those rows contribute nothing to weighted stats.")
        if neg:
            _err(f"'{WEIGHT_COL}' has {neg:,} negative values — this will corrupt all weighted stats.")
        _ok(f"'{WEIGHT_COL}' present. min={df[WEIGHT_COL].min():.4f}  max={df[WEIGHT_COL].max():.4f}  "
            f"mean={df[WEIGHT_COL].mean():.4f}")

    # 'use' column
    if "use" not in df.columns:
        _err("'use' column not found. All split-by-user-group charts will be blank.")
    else:
        vc = df["use"].value_counts(dropna=False)
        print(f"\n  'use' column value counts:")
        for val, cnt in vc.items():
            flag = "" if val in ALL_USE_GROUPS else f"  {YEL}⚠ not in ALL_USE_GROUPS{RESET}"
            print(f"    {str(val):<20} {cnt:>6,}{flag}")
        missing_groups = [g for g in ALL_USE_GROUPS if g not in df["use"].values]
        if missing_groups:
            _warn(f"Expected use groups not found in data: {missing_groups}")
        else:
            _ok("All expected use groups present.")

    # Split columns
    for col in SPLIT_COLS:
        if col not in df.columns:
            _warn(f"Split column '{col}' not found — split-by-{col} charts will be blank.")
        else:
            nuniq = df[col].nunique()
            nulls = df[col].isna().sum()
            _ok(f"'{col}': {nuniq} unique values, {nulls:,} nulls")
            if col == "age_group":
                vals = sorted(df[col].dropna().unique())
                print(f"    age_group values: {vals}")
                # Check for mixed dash styles (en-dash vs hyphen)
                has_endash  = any("–" in str(v) for v in vals)
                has_hyphen  = any("-" in str(v) for v in vals)
                if has_endash and has_hyphen:
                    _warn("age_group mixes en-dash (–) and hyphen (-) — this creates duplicate groups "
                          "in aggregations. Standardise to one style in the cleaning step.")


def _validate_raw_statements(df):
    _section("Statements ETL inputs")

    stmt_cols = [c for c in df.columns if c.startswith("statement_")]
    if not stmt_cols:
        _err("No columns starting with 'statement_' found. "
             "The statements ETL will produce no output.")
        return
    _ok(f"{len(stmt_cols)} statement columns found.")

    # Check response values
    sample_col = stmt_cols[0]
    vc = df[sample_col].value_counts(dropna=False).head(10)
    print(f"\n  Top values in '{sample_col}':")
    for val, cnt in vc.items():
        print(f"    {str(val):<40} {cnt:>6,}")

    # etl_statements.py matches French "d'accord" / "désaccord" (see its own
    # comment on why), not English "Agree"/"Disagree" -- this check used to
    # test for the English strings, which never appear in this survey's raw
    # text and made every run report a false "no Agree values" error even
    # though the real ETL was working correctly.
    has_agree    = df[stmt_cols].apply(lambda c: c.astype(str).str.contains("d'accord").any()).any()
    has_disagree = df[stmt_cols].apply(lambda c: c.astype(str).str.contains("désaccord").any()).any()
    if not has_agree:
        _err("No \"d'accord\" values found in any statement column. "
             "etl_statements.py maps agreement by str.contains(\"d'accord\") — check response encoding.")
    if not has_disagree:
        _warn("No 'désaccord' values found. Check response encoding.")
    if has_agree and has_disagree:
        _ok("\"d'accord\" and \"désaccord\" responses found.")


def _validate_raw_family_planning(df):
    _section("Family planning ETL inputs")

    # Funnel columns — the most likely source of all-zero output
    funnel_cols = {
        "birth_spacing":  "Awareness (funnel step 1) — expected integer 1=aware",
        "ever_use":       "Ever used (funnel step 2) — expected integer 1=Oui",
        "current_use":    "Currently using (funnel step 3) — expected integer 1=Oui",
    }
    print()
    for col, desc in funnel_cols.items():
        if col not in df.columns:
            _err(f"'{col}' not found. {desc}. Funnel will show 0% for this step.")
        else:
            vc = df[col].value_counts(dropna=False).head(8)
            n_ones = (df[col].fillna("NA").str.contains("Oui")).sum()
            pct = n_ones / len(df) * 100
            print(f"  '{col}' ({desc})")
            print(f"    → rows with value==1: {n_ones:,} ({pct:.1f}%)")
            if n_ones == 0:
                _err(f"    ZERO rows have {col}==1. This is why the funnel shows 0%. "
                     f"Check encoding — raw values are: {sorted(df[col].dropna().unique().tolist())[:10]}")
            else:
                _ok(f"    '{col}' has {n_ones:,} rows == 1")

    # Method columns
    method_cols = ["known_contraceptive_options", "ever_used_methods", "current_use_methods"]
    for col in method_cols:
        if col not in df.columns:
            _warn(f"'{col}' not found — method charts will be blank.")
        else:
            sample = df[col].dropna().head(3).tolist()
            _ok(f"'{col}' present. Sample values: {sample}")

    # fp_intent columns (future_intent / considered_use) — same "Oui"
    # pattern as funnel_cols above; future_intent is also the second input
    # to unmet_demand (alongside the unmet_need mask below).
    intent_cols = {
        "future_intent":  "Interested in future use — feeds unmet_demand",
        "considered_use": "Considered use (non-users)",
    }
    print()
    for col, desc in intent_cols.items():
        if col not in df.columns:
            _warn(f"'{col}' not found — that FP chart, and unmet_demand, will be blank.")
            continue
        vc = df[col].value_counts(dropna=False).head(8)
        n_ones = (df[col].fillna("NA").str.contains("Oui")).sum()
        pct = n_ones / len(df) * 100
        print(f"  '{col}' ({desc})")
        for val, cnt in vc.items():
            print(f"    {str(val):<35} {cnt:>6,}")
        print(f"    → rows containing 'Oui': {n_ones:,} ({pct:.1f}%)")
        if n_ones == 0:
            _err(f"    ZERO rows contain 'Oui' in '{col}'. etl_family_planning.py "
                 f"uses str.contains('Oui') here — check encoding, raw values "
                 f"are: {sorted(df[col].dropna().unique().tolist())[:10]}")
        else:
            _ok(f"    '{col}' has {n_ones:,} rows containing 'Oui'")

    # time_before_preferred_pregnancy — feeds unmet_need's "wants to delay"
    # mask (etl_family_planning.WANTS_TO_DELAY_CODES / _LABELS: codes 3/4/6,
    # i.e. "1-2 years" / "More than 2 years" / "No more children").
    col = "time_before_preferred_pregnancy"
    if col not in df.columns:
        _warn(f"'{col}' not found — timing chart and unmet_need will be blank.")
    else:
        vc = df[col].value_counts(dropna=False).head(10)
        print(f"  '{col}' (feeds unmet_need)")
        for val, cnt in vc.items():
            print(f"    {str(val):<35} {cnt:>6,}")
        # run() strips the Fon half of this column (_strip_hausa) before
        # matching against WANTS_TO_DELAY_LABELS -- mirror that here, or
        # this check compares raw bilingual text ("Dans 1 à 2 ans/ Ðo...")
        # against pure-French labels and always reports zero, regardless of
        # whether the real ETL is actually working.
        stripped = df[col].apply(_strip_hausa)
        numeric_hits = pd.to_numeric(stripped, errors="coerce").isin(WANTS_TO_DELAY_CODES).sum()
        label_hits = stripped.astype(str).isin(WANTS_TO_DELAY_LABELS).sum()
        if numeric_hits == 0 and label_hits == 0:
            _err(f"    ZERO rows match WANTS_TO_DELAY_CODES {WANTS_TO_DELAY_CODES} "
                 f"or their labels {sorted(WANTS_TO_DELAY_LABELS)} in '{col}' "
                 f"(after stripping Fon, same as run()). This is why "
                 f"unmet_need/unmet_demand show 0% — stripped values are: "
                 f"{sorted(stripped.dropna().unique().tolist())[:10]}")
        else:
            _ok(f"    {numeric_hits + label_hits:,} rows match 'wants to delay/limit' "
                f"(by code: {numeric_hits:,}, by label: {label_hits:,})")

    # Other FP columns — presence only, not part of the funnel/unmet-need
    # calculations above.
    for col in ["reason_current_use", "reason_current_nonuse"]:
        if col not in df.columns:
            _warn(f"'{col}' not found — that FP chart will be blank.")
        else:
            _ok(f"'{col}' present.")


def _validate_raw_access(df):
    _section("Access ETL inputs")

    # --- Stockouts ---
    # 2026-09-02: not every one of these is a yes/no field -- confirmed
    # against the XLSForm's own choices sheet (SurveyCTO metadata, no
    # respondent data). Their list_names:
    #   stockouts_users              -> "seeking_contraceptives" (includes "Oui/ Ɛɛn")
    #   stockouts_nonusers           -> "yesno"                  (includes "Oui / Ɛɛn")
    #   nonusers_seek_contraceptives -> "ever_try": never tried / tried & failed /
    #                                    tried & succeeded -- NO "Oui" option exists.
    #                                    etl_access.py matches "j'ai essayé et
    #                                    j'ai réussi" / "je n'ai jamais essayé"
    #                                    (lowercased), not "Oui".
    #   stockouts_response / stockouts_nonusers_response -> "stockout_response":
    #                                    6 categorical options about what the
    #                                    respondent did -- NO "Oui" option
    #                                    exists here either, and etl_access.py
    #                                    never tests them for "Oui": it only
    #                                    tabulates raw value_counts (section 2,
    #                                    "Stockout responses (qualitative)").
    # A blanket "does 'Oui' appear anywhere" check was wrong for 3 of these 5
    # columns from the start -- there was never an "Oui" option for it to find.
    print()
    for col, desc in [
        ("stockouts_users",    "Users stockout flag"),
        ("stockouts_nonusers", "Non-users stockout flag"),
    ]:
        if col not in df.columns:
            _warn(f"'{col}' not found.")
        else:
            vc = df[col].value_counts(dropna=False).head(6)
            # etl_access.py itself uses str.startswith("Oui") (case-sensitive,
            # not lowercased) -- this check used to lowercase the column
            # first and then search for the capital-O "Oui", which can never
            # match its own lowercased input and reported a false error on
            # every run regardless of the real (correctly-encoded) data.
            has_Oui_str = df[col].astype(str).str.startswith("Oui").any()
            print(f"  '{col}' ({desc})")
            for val, cnt in vc.items():
                print(f"    {str(val):<35} {cnt:>6,}")
            if not has_Oui_str:
                _err(f"    No value starting with 'Oui' found in '{col}'. "
                     f"etl_access.py uses str.startswith('Oui') — if values are 0/1, "
                     f"lowercase 'oui', or don't start with 'Oui', the column will "
                     f"always evaluate to False and produce 0% results.")
            else:
                _ok(f"    Values starting with 'Oui' found in '{col}'")

    col, desc = "nonusers_seek_contraceptives", "Non-users seeking flag ← KEY for 'sought' chart"
    if col not in df.columns:
        _warn(f"'{col}' not found.")
    else:
        vc = df[col].value_counts(dropna=False).head(6)
        s = df[col].astype(str).str.lower()
        has_success = s.str.contains("j'ai essayé et j'ai réussi", na=False).any()
        has_never   = s.str.contains("je n'ai jamais essayé", na=False).any()
        print(f"  '{col}' ({desc})")
        for val, cnt in vc.items():
            print(f"    {str(val):<35} {cnt:>6,}")
        if not has_success and not has_never:
            _err(f"    Neither \"j'ai essayé et j'ai réussi\" nor \"je n'ai jamais "
                 f"essayé\" found in '{col}'. etl_access.py's _sought_any/"
                 f"_sought_successfully match those exact French substrings "
                 f"(case-insensitive) — check encoding.")
        else:
            _ok(f"    'j'ai essayé et j'ai réussi' found: {has_success}; "
                f"'je n'ai jamais essayé' found: {has_never}")

    for col, desc in [
        ("stockouts_response",          "User stockout response (qualitative — no 'Oui' expected)"),
        ("stockouts_nonusers_response", "Non-user stockout response (qualitative — no 'Oui' expected)"),
    ]:
        if col not in df.columns:
            _warn(f"'{col}' not found.")
        else:
            vc = df[col].value_counts(dropna=False).head(6)
            print(f"  '{col}' ({desc})")
            for val, cnt in vc.items():
                print(f"    {str(val):<35} {cnt:>6,}")
            if df[col].dropna().empty:
                _err(f"    '{col}' has no non-null values at all.")
            else:
                _ok(f"    '{col}' has {df[col].notna().sum():,} non-null responses.")

    # --- Travel ---
    print()
    for col in ["travel_time_users", "travel_time_nonusers", "willingness_to_travel"]:
        if col not in df.columns:
            _warn(f"'{col}' not found — travel section will be blank.")
        else:
            if col == "willingness_to_travel":
                vc = df[col].value_counts(dropna=False).head(6)
                unmapped = [v for v in df[col].dropna().unique() if v not in WTT_MAP]
                _ok(f"'{col}' present.")
                if unmapped:
                    _warn(f"  Values not in WTT_MAP (will become NaN): {unmapped[:10]}")
            else:
                _ok(f"'{col}' present. mean={df[col].mean():.1f}  "
                    f"nulls={df[col].isna().sum():,}")

    # --- Costs ---
    for col in ["user_costs", "nonuser_cost"]:
        if col not in df.columns:
            _warn(f"'{col}' not found — affordability section will be blank.")
        else:
            _ok(f"'{col}' present. mean={df[col].mean():.1f}  zeros={(df[col]==0).sum():,}")


def _validate_raw_personality(df):
    _section("Personality ETL inputs")

    multiselect_cols = {
        "life_goals":       LIFE_GOALS,
        "role_models":      ROLE_MODELS,
        "likeable_traits":  LIKEABLE_TRAITS,
        "forming_beliefs":  FORMING_BELIEFS,
    }
    for col, label_map in multiselect_cols.items():
        if col not in df.columns:
            _warn(f"'{col}' not found.")
        else:
            sample = df[col].dropna().head(3).tolist()
            _ok(f"'{col}' present. Sample: {sample}")

    for col in ["decision_confident", "decision_confident_3",
                "life_goals_main", "life_goals_achievable"]:
        status = "present" if col in df.columns else "NOT FOUND"
        fn = _ok if col in df.columns else _warn
        fn(f"'{col}': {status}")

    # Wellbeing — critical: check column names and value encoding
    print()
    for col in ["happiness", "satisfaction"]:
        if col not in df.columns:
            _err(f"Wellbeing column '{col}' NOT FOUND. "
                 f"The wellbeing ETL has a known bug where it references "
                 f"an undefined variable 'frame' in the split loop — "
                 f"this will crash even if the column exists.")
        else:
            vc = df[col].value_counts(dropna=False).head(8)
            print(f"  '{col}' values:")
            for val, cnt in vc.items():
                print(f"    {str(val):<6} {cnt:>6,}")
            _ok(f"'{col}' present.")


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT CSV VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_outputs():
    print(f"\n{BOLD}OUTPUT CSV VALIDATION{RESET}")

    _validate_out_statements()
    _validate_out_family_planning()
    _validate_out_access()
    _validate_out_personality()
    _validate_out_personas()
    validate_new_outputs()


def _load_out(filename):
    path = os.path.join(APP_DATA_DIR, filename)
    if not os.path.exists(path):
        _warn(f"Output file not found: {path}")
        return None
    df = pd.read_csv(path)
    _ok(f"Loaded {filename}: {len(df):,} rows × {df.shape[1]} cols")
    return df


def _check_splits(df, filename, value_col="proportion"):
    """Check that each SPLIT_COL has data and no all-zero groups."""
    for split in SPLIT_COLS:
        sub = df[df["split"] == split] if "split" in df.columns else pd.DataFrame()
        if sub.empty:
            _warn(f"{filename}: no rows for split='{split}'")
            continue
        groups = sub["group"].unique().tolist()
        if value_col in sub.columns:
            all_zero = (sub[value_col] == 0).all()
            if all_zero:
                _err(f"{filename}: ALL values are 0 for split='{split}'. "
                     f"Groups: {groups}. Check raw data encoding.")
            else:
                _ok(f"{filename}: split='{split}' has {len(groups)} groups, "
                    f"values range [{sub[value_col].min():.3f}, {sub[value_col].max():.3f}]")


def _validate_out_statements():
    _section("statements_heatmap.csv")
    df = _load_out("statements_heatmap.csv")
    if df is None:
        return

    # ── Schema ───────────────────────────────────────────────────────────────
    required = [
        "label", "split", "group", "weighted_agreement",
        "agree_n", "agree_pct", "agree_weighted_n",
        "disagree_n", "disagree_pct", "disagree_weighted_n",
        "total_n", "total_weighted_n",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _err(f"Missing columns: {missing}")
        return
    _ok("All required columns present.")

    # ── Nulls in key columns ──────────────────────────────────────────────────
    for col in ["label", "split", "group", "weighted_agreement", "agree_pct",
                "disagree_pct", "total_n", "total_weighted_n"]:
        n_null = df[col].isna().sum()
        if n_null:
            _err(f"'{col}' has {n_null:,} null values.")
        else:
            _ok(f"'{col}': no nulls.")

    # ── Label count ───────────────────────────────────────────────────────────
    n_labels = df["label"].nunique()
    _ok(f"{n_labels} unique statement labels.")
    if n_labels < 3:
        _warn("Fewer than 3 statement labels — check that statement_ columns exist in raw data.")

    # ── weighted_agreement in [-1, 1] ─────────────────────────────────────────
    wa = df["weighted_agreement"].dropna()
    _ok(f"weighted_agreement range: [{wa.min():.3f}, {wa.max():.3f}]")
    if (wa == 0).all():
        _err("ALL weighted_agreement values are 0 — check Agree/Disagree encoding in raw data.")
    out_of_range = ((wa < -1) | (wa > 1)).sum()
    if out_of_range:
        _err(f"{out_of_range} weighted_agreement values outside [-1, 1]. "
             f"Score must be (weighted agrees − weighted disagrees) / total weighted n.")

    # ── agree_pct / disagree_pct in [0, 1] ───────────────────────────────────
    for col in ["agree_pct", "disagree_pct"]:
        p = df[col].dropna()
        _ok(f"{col} range: [{p.min():.3f}, {p.max():.3f}]")
        bad = ((p < 0) | (p > 1)).sum()
        if bad:
            _err(f"{bad} rows have {col} outside [0, 1].")

    # ── agree_pct + disagree_pct == 1 ────────────────────────────────────────
    # Denominator is agree_n + disagree_n (don't-know excluded), so the two
    # must sum to exactly 1 for any row where someone gave an opinion.
    pct_sum = df["agree_pct"] + df["disagree_pct"]
    opinioned = (df["agree_n"] + df["disagree_n"]) > 0
    not_one = opinioned & ((pct_sum - 1.0).abs() > 0.001)
    if not_one.sum():
        _err(f"{not_one.sum()} rows where agree_pct + disagree_pct ≠ 1 "
             f"(expected: denominator is agree_n + disagree_n).")
    else:
        _ok("agree_pct + disagree_pct = 1 for all rows with opinioned respondents.")

    # ── agree_n + disagree_n <= total_n ──────────────────────────────────────
    n_over = ((df["agree_n"] + df["disagree_n"]) > df["total_n"] + 0.5).sum()
    if n_over:
        _err(f"{n_over} rows where agree_n + disagree_n > total_n.")
    else:
        _ok("agree_n + disagree_n <= total_n in all rows.")

    # ── total_n / total_weighted_n positive ───────────────────────────────────
    zero_n = (df["total_n"] == 0).sum()
    if zero_n:
        _warn(f"{zero_n} rows with total_n == 0 (no respondents answered).")
    zero_wn = (df["total_weighted_n"] <= 0).sum()
    if zero_wn:
        _err(f"{zero_wn} rows with total_weighted_n <= 0.")

    # ── Overall (split=none, group=all) present for every label ──────────────
    overall = df[(df["split"] == "none") & (df["group"] == "all")]
    overall_labels = set(overall["label"].unique())
    all_labels     = set(df["label"].unique())
    missing_overall = all_labels - overall_labels
    if missing_overall:
        _err(f"{len(missing_overall)} labels have no overall (split=none/group=all) row: "
             f"{sorted(missing_overall)[:5]}{'...' if len(missing_overall) > 5 else ''}")
    else:
        _ok(f"All {n_labels} labels have an overall (split=none/group=all) row.")

    # ── Expected splits present ───────────────────────────────────────────────
    splits_present = set(df["split"].unique())
    expected_splits = {"none"} | set(SPLIT_COLS)
    missing_splits = expected_splits - splits_present
    if missing_splits:
        _err(f"Missing splits in output: {missing_splits}")
    else:
        _ok(f"All expected splits present: {sorted(splits_present)}")

    # ── use split: only user / nonuser groups (no past_user / future_user) ────
    use_rows = df[df["split"] == "use"]
    if use_rows.empty:
        _warn("No rows for split='use'.")
    else:
        use_groups = set(use_rows["group"].unique())
        unexpected = use_groups - {"user", "nonuser"}
        if unexpected:
            _err(f"Unexpected groups in use split: {unexpected}. "
                 f"past_user and future_user should have been collapsed into nonuser.")
        else:
            _ok(f"use split groups: {sorted(use_groups)} (user / nonuser only — correct).")
        for grp in ["user", "nonuser"]:
            if grp not in use_groups:
                _err(f"Expected group '{grp}' not found in use split.")

    # ── No duplicate label × split × group rows ───────────────────────────────
    dupes = df.duplicated(subset=["label", "split", "group"]).sum()
    if dupes:
        _err(f"{dupes} duplicate label × split × group rows — aggregation may have run twice.")
    else:
        _ok("No duplicate label × split × group rows.")


def _validate_out_family_planning():
    _section("fp_funnel.csv")
    df = _load_out("fp_funnel.csv")
    if df is None:
        return

    required = ["split", "group", "aware", "ever_used", "current_use"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        _err(f"Missing columns: {missing}")

    for col in ["aware", "ever_used", "current_use"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if (vals == 0).all():
            _err(f"FUNNEL: '{col}' is ALL ZEROS. "
                 f"This means the raw column (birth_spacing / ever_use / current_use) "
                 f"either doesn't exist or never equals 1. "
                 f"Run --raw to check raw value distributions.")
        else:
            _ok(f"funnel '{col}': range [{vals.min():.3f}, {vals.max():.3f}]")

    # Check for duplicate groups (mixed dash styles in age_group)
    age_groups = df[df["split"] == "age_group"]["group"].unique().tolist()
    endash = [v for v in age_groups if "–" in str(v)]
    hyphen = [v for v in age_groups if "-" in str(v) and "–" not in str(v)]
    if endash and hyphen:
        _warn(f"age_group has BOTH en-dash and hyphen variants: {endash + hyphen}. "
              f"This creates duplicate rows for the same age band.")

    _section("fp_methods.csv")
    df = _load_out("fp_methods.csv")
    if df is not None:
        for mt in ["known", "ever", "current"]:
            sub = df[df["method_type"] == mt] if "method_type" in df.columns else pd.DataFrame()
            if sub.empty:
                _warn(f"No rows for method_type='{mt}'")
            else:
                _ok(f"method_type='{mt}': {len(sub):,} rows")

    for fname in ["fp_timing.csv", "fp_reason_use.csv", "fp_intent.csv"]:
        _section(fname)
        df = _load_out(fname)
        if df is not None:
            _check_splits(df, fname)

    # fp_nonuse_reasons.csv has a different shape by design -- a flat top-20
    # free-text frequency table (Reason / Unweighted count), with no split/
    # group/proportion columns at all -- so running it through _check_splits
    # (as before) always reported a false "no rows for split='use'" etc. for
    # every split, since "split" is never in its columns to begin with.
    _section("fp_nonuse_reasons.csv")
    df = _load_out("fp_nonuse_reasons.csv")
    if df is not None:
        required = ["Reason", "Unweighted count"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            _err(f"fp_nonuse_reasons.csv: missing columns {missing}.")
        elif df.empty:
            _warn("fp_nonuse_reasons.csv: no rows.")
        else:
            _ok(f"fp_nonuse_reasons.csv: {len(df)} reasons, counts range "
                f"[{df['Unweighted count'].min()}, {df['Unweighted count'].max()}]")

    # fp_unmet.csv (2026-09-02 addition) -- unmet need / unmet demand.
    _section("fp_unmet.csv")
    df = _load_out("fp_unmet.csv")
    if df is not None:
        required = ["split", "group", "unmet_need", "unmet_demand"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            _err(f"fp_unmet.csv: missing columns {missing}.")
        else:
            for col in ["unmet_need", "unmet_demand"]:
                vals = df[col].dropna()
                if vals.empty:
                    _warn(f"fp_unmet.csv: '{col}' is all null.")
                elif (vals == 0).all():
                    _err(f"UNMET: '{col}' is ALL ZEROS. Check the "
                         f"time_before_preferred_pregnancy / current_use / "
                         f"future_intent raw-value checks above.")
                else:
                    _ok(f"fp_unmet.csv '{col}': range [{vals.min():.3f}, {vals.max():.3f}]")


def _validate_out_access():
    _section("access_stockouts.csv")
    df = _load_out("access_stockouts.csv")
    if df is None:
        return

    metrics = df["metric"].unique().tolist() if "metric" in df.columns else []
    _ok(f"Metrics present: {metrics}")

    expected_metrics = ["stockout_users", "sought_contraceptives", "stockout_nonusers_sought"]
    for m in expected_metrics:
        if m not in metrics:
            _warn(f"Expected metric '{m}' not found in access_stockouts.csv")

    for m in metrics:
        sub = df[df["metric"] == m]
        vals = sub["value"].dropna()
        if vals.empty or (vals == 0).all():
            _err(f"Metric '{m}': ALL values are 0 or empty. "
                 f"Groups present: {sub['group'].unique().tolist()}. "
                 f"Splits present: {sub['split'].unique().tolist()}.")
        else:
            _ok(f"Metric '{m}': {len(sub)} rows, "
                f"range [{vals.min():.3f}, {vals.max():.3f}]")

    # Check that nonuser appears under sought_contraceptives
    if "sought_contraceptives" in metrics:
        sought_groups = df[df["metric"] == "sought_contraceptives"]["group"].unique().tolist()
        if "nonuser" not in sought_groups:
            _warn(f"'nonuser' not in sought_contraceptives groups: {sought_groups}. "
                  f"Likely because NONUSER_GROUPS filter produced no nonuser rows, "
                  f"or the raw 'use' column spells it differently (e.g. 'non-user').")

    for fname in ["access_travel.csv", "access_affordability.csv",
                  "access_composite.csv", "access_stockout_responses.csv"]:
        _section(fname)
        df = _load_out(fname)
        if df is not None and "value" in df.columns:
            vals = df["value"].dropna()
            if vals.empty or (vals == 0).all():
                _warn(f"{fname}: all values are 0 or empty.")
            else:
                _ok(f"Values range: [{vals.min():.3f}, {vals.max():.3f}]")


def _validate_out_personality():
    outputs = [
        ("personality_life_goals.csv",          ["label", "split", "group", "proportion"]),
        ("personality_role_models.csv",          ["label", "split", "group", "proportion"]),
        ("personality_likeable_traits.csv",      ["label", "split", "group", "proportion"]),
        ("personality_forming_beliefs.csv",      ["label", "split", "group", "proportion"]),
        ("personality_decision_confident.csv",   ["label", "split", "group", "proportion"]),
        ("personality_wellbeing.csv",            ["label", "split", "group", "proportion", "question"]),
        ("personality_goals_achievable.csv",     ["label", "split", "group", "proportion"]),
    ]
    for fname, required_cols in outputs:
        _section(fname)
        df = _load_out(fname)
        if df is None:
            continue
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            _err(f"Missing columns: {missing}")
        else:
            _ok("Required columns present.")
        _check_splits(df, fname)
        if "proportion" in df.columns:
            p = df["proportion"].dropna()
            if (p == 0).all():
                _err(f"{fname}: ALL proportions are 0.")
            elif p.max() > 1.0:
                _warn(f"{fname}: some proportions > 1.0 (max={p.max():.3f}). "
                      f"Expected 0–1 range.")

    # Specific wellbeing check
    _section("Wellbeing ETL bug check")
    wb = _load_out("personality_wellbeing.csv")
    if wb is not None and "question" in wb.columns:
        for q in ["happiness", "satisfaction"]:
            sub = wb[wb["question"] == q]
            if sub.empty:
                _err(f"No rows for wellbeing question='{q}'. "
                     f"Known bug in etl_personality.py: the split loop references "
                     f"an undefined variable 'frame' instead of calling "
                     f"split_weighted_counts(). Fix: replace the split loop body with:\n"
                     + textwrap.indent(textwrap.dedent("""
                         from pipeline.utils import split_weighted_counts
                         for split_col in SPLIT_COLS:
                             frame = split_weighted_counts(df, col, split_col, LIKERT,
                                                           exclude=())
                             melted = frame.reset_index().melt(
                                 id_vars=frame.index.name or "index",
                                 var_name="group", value_name="proportion")
                             melted.columns = ["label", "group", "proportion"]
                             melted["question"] = tag
                             melted["split"] = split_col
                             wellbeing_rows.append(melted)
                     """).strip(), "                     "))
            else:
                splits_present = sub["split"].unique().tolist()
                _ok(f"question='{q}': {len(sub)} rows, splits: {splits_present}")


def _validate_out_personas():
    _section("personas_centroids.csv / personas_profile.csv")
    centroids = _load_out("personas_centroids.csv")
    if centroids is not None:
        _ok(f"{len(centroids)} personas found.")
        if "count" in centroids.columns:
            _ok(f"Persona sizes: {centroids['count'].tolist()}")
        if "weighted_count" in centroids.columns:
            _ok(f"Weighted sizes: {centroids['weighted_count'].round(1).tolist()}")

    profile = _load_out("personas_profile.csv")
    if profile is not None:
        vars_present = profile["variable"].unique().tolist() if "variable" in profile.columns else []
        _ok(f"Profile variables: {vars_present}")


# ═══════════════════════════════════════════════════════════════════════════════
# NEWER PAGES AND TABLES
# Key FP indicators, driver/barrier parent groups, respondent profile, persona
# splits, radio and drivers/barriers tables, "Autre" recodes, and the phone
# survey change-over-time outputs. Same in every country's validate_pipeline.py.
# ═══════════════════════════════════════════════════════════════════════════════

HEADLINE_METRICS = ["mcpr", "unmet_need", "unmet_demand", "demand_satisfied",
                    "aware_modern", "aware_modern_nonusers"]


def _require(df, filename, cols):
    """Error on missing columns; True when all are present."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _err(f"{filename}: missing columns {missing}")
    return not missing


def _check_share(df, filename, col):
    """A 0-1 share column: no values outside [0, 1], not all null."""
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        _warn(f"{filename}: '{col}' is all null")
    elif ((vals < 0) | (vals > 1)).any():
        _err(f"{filename}: '{col}' has values outside 0-1 "
             f"(range [{vals.min():.3f}, {vals.max():.3f}])")
    else:
        _ok(f"{filename}: '{col}' range [{vals.min():.3f}, {vals.max():.3f}]")


def _latest_out(pattern):
    """Newest file in APP_DATA_DIR matching a glob pattern, or None."""
    matches = sorted(glob.glob(os.path.join(APP_DATA_DIR, pattern)))
    return matches[-1] if matches else None


def validate_new_outputs():
    _validate_out_fp_headline()
    _validate_out_driver_barrier_groups()
    _validate_out_respondents_profile()
    _validate_out_personality_filters()
    _validate_out_persona_splits()
    _validate_out_radio_tables()
    _validate_out_drivers_barriers_table()
    _validate_other_recodes()
    _validate_change_over_time()


def _validate_out_fp_headline():
    """fp_headline.csv: key FP indicators (weighted, 95% CIs, unweighted n)."""
    _section("fp_headline.csv (key FP indicators)")
    df = _load_out("fp_headline.csv")
    if df is None or not _require(df, "fp_headline.csv",
                                  ["population", "metric", "value", "ci_low", "ci_high", "n_unweighted"]):
        return
    for pop in ["full_sample", "women"]:
        missing = sorted(set(HEADLINE_METRICS) - set(df.loc[df["population"] == pop, "metric"]))
        if missing:
            _err(f"fp_headline.csv: population '{pop}' is missing metrics {missing}")
        else:
            _ok(f"fp_headline.csv: '{pop}' has all {len(HEADLINE_METRICS)} metrics")
    for col in ["value", "ci_low", "ci_high"]:
        _check_share(df, "fp_headline.csv", col)
    has_ci = df[["ci_low", "ci_high"]].notna().all(axis=1)
    if not has_ci.any():
        _warn("fp_headline.csv: no confidence intervals (check CI_CLUSTER_COL / CI_STRATUM_COL)")
    outside = has_ci & df["value"].notna() & ((df["value"] < df["ci_low"] - 1e-9) |
                                              (df["value"] > df["ci_high"] + 1e-9))
    if outside.any():
        _err(f"fp_headline.csv: value outside its CI for "
             f"{df.loc[outside, ['population', 'metric']].values.tolist()}")
    if (df["n_unweighted"] <= 0).any():
        _err("fp_headline.csv: some rows have n_unweighted <= 0")
    else:
        _ok(f"fp_headline.csv: n_unweighted range [{df['n_unweighted'].min()}, {df['n_unweighted'].max()}]")


def _validate_out_driver_barrier_groups():
    """driver_barrier_groups.csv: options grouped by parent category, >= 1 / >= 3."""
    fname = "driver_barrier_groups.csv"
    _section(f"{fname} (drivers/barriers by parent category)")
    df = _load_out(fname)
    if df is None or not _require(df, fname, ["type", "parent", "n_options", "split", "group", "threshold",
                                              "proportion", "unweighted_proportion", "n", "weighted_n",
                                              "base_n"]):
        return
    for t in ["Driver", "Barrier"]:
        sub = df[df["type"] == t]
        if sub.empty:
            _warn(f"{fname}: no '{t}' rows (are the DB_LISTS columns in the data?)")
        else:
            _ok(f"{fname}: {t}: {sub['parent'].nunique()} parent categories, "
                f"base n = {int(sub.loc[sub['split'] == 'none', 'base_n'].max())}")
    for col in ["proportion", "unweighted_proportion"]:
        _check_share(df, fname, col)
    if (df["n"] > df["base_n"]).any():
        _err(f"{fname}: n > base_n in {int((df['n'] > df['base_n']).sum())} rows")
    if ((df["threshold"] == "at least 3") & (df["n_options"] < 3)).any():
        _err(f"{fname}: 'at least 3' rows for parents with fewer than 3 options")
    keys = ["type", "parent", "split", "group"]
    wide = df.pivot_table(index=keys, columns="threshold", values="unweighted_proportion")
    if {"at least 1", "at least 3"} <= set(wide.columns):
        bad = (wide["at least 3"] > wide["at least 1"] + 1e-9).sum()
        if bad:
            _err(f"{fname}: 'at least 3' exceeds 'at least 1' in {int(bad)} groups")
        else:
            _ok(f"{fname}: 'at least 3' <= 'at least 1' everywhere")
    for split in ["none"] + SPLIT_COLS:
        if split not in set(df["split"]):
            _warn(f"{fname}: no rows for split='{split}'")


def _validate_out_respondents_profile():
    fname = "respondents_profile.csv"
    _section(fname)
    df = _load_out(fname)
    if df is None or not _require(df, fname, ["variable", "category", "count", "proportion"]):
        return
    _check_share(df, fname, "proportion")
    _ok(f"{fname}: variables {sorted(df['variable'].unique().tolist())}")
    if (df["count"] < 0).any():
        _err(f"{fname}: negative counts")


def _validate_out_personality_filters():
    """Role models / traits: the gender x age split and group sizes behind the
    Personality page filters."""
    for fname in ["personality_role_models.csv", "personality_likeable_traits.csv"]:
        _section(f"{fname} (gender + age filters)")
        df = _load_out(fname)
        if df is None:
            continue
        if "gender_age" not in set(df.get("split", [])):
            _warn(f"{fname}: no 'gender_age' split -- the combined gender + age filter "
                  f"will show 'not generated yet'. Re-run --pages personality.")
        if "n" not in df.columns:
            _warn(f"{fname}: no 'n' column -- the filters can't show group sizes.")
            continue
        n = df.drop_duplicates(["split", "group"]).set_index(["split", "group"])["n"]
        overall = n.get(("none", "all"))
        for split in ["gender", "age_group", "gender_age"]:
            sub = n[n.index.get_level_values("split") == split]
            if sub.empty:
                continue
            if overall is not None and sub.sum() > overall:
                _err(f"{fname}: '{split}' group sizes add up to {int(sub.sum())}, "
                     f"more than the {int(overall)} respondents overall")
            small = int((sub < 30).sum())
            _ok(f"{fname}: '{split}' {len(sub)} groups, n {int(sub.min())}-{int(sub.max())}"
                + (f" ({small} under 30)" if small else ""))


def _validate_out_persona_splits():
    """Per-gender / per-region / per-use persona files, elbows and culture clusters."""
    _section("persona splits, elbows and culture clusters")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(APP_DATA_DIR, "personas_*.csv"))
                   + glob.glob(os.path.join(APP_DATA_DIR, "culture_clusters_*.csv")))
    names = [n for n in names if n not in ("personas_centroids.csv", "personas_profile.csv")]
    if not names:
        _warn("No persona split / culture cluster files found")
    for fname in names:
        df = _load_out(fname)
        if df is None:
            continue
        if df.empty:
            _err(f"{fname}: no rows")
        if "elbow" in fname and {"k", "cost"} <= set(df.columns):
            group_cols = [c for c in df.columns if c not in ("k", "cost", "chosen_k")]
            for g, sub in (df.groupby(group_cols) if group_cols else [("all", df)]):
                costs = sub.sort_values("k")["cost"].to_numpy()
                if (np.diff(costs) > 1e-6).any():
                    _warn(f"{fname}: cost rises with k for {g} (k-modes run may be unstable)")


def _validate_out_radio_tables():
    _section("radio question tables")
    for kind in ["station", "state"]:
        path = _latest_out(f"*_question_table_by_{kind}_*.csv")
        if path is None:
            _warn(f"No *_question_table_by_{kind}_*.csv in {APP_DATA_DIR} "
                  f"(the app may load it from Drive instead)")
            continue
        df = pd.read_csv(path)
        if df.empty or df.shape[1] < 2:
            _err(f"{os.path.basename(path)}: empty or has no station/region columns")
        else:
            _ok(f"{os.path.basename(path)}: {len(df)} questions x {df.shape[1] - 1} columns")


def _validate_out_drivers_barriers_table():
    _section("drivers/barriers table")
    path = _latest_out("*_drivers-barriers_table_*.csv")
    if path is None:
        _ok("No local drivers/barriers table (the app loads it from Drive)")
        return
    df = pd.read_csv(path)
    fname = os.path.basename(path)
    if _require(df, fname, ["Name", "Driver/Barrier", "Prevalence (All)", "N (All)"]):
        counts = df["Driver/Barrier"].str.lower().value_counts().to_dict()
        _ok(f"{fname}: {counts}")
        # 03_driver_barrier_table_w_counts computes the "Prevalence" columns
        # UNWEIGHTED (split_dribar_unweighted); only "Weighted N" uses weights.
        if not any("weighted prevalence" in c.lower() and "unweighted" not in c.lower()
                   for c in df.columns):
            _warn(f"{fname}: 'Prevalence' columns are unweighted and there is no weighted "
                  f"prevalence column, so the page's 'Prevalence (%)' and 'Unweighted "
                  f"prevalence (%)' metrics show the same unweighted figures")
        if "Weighted N (All)" not in df.columns:
            _warn(f"{fname}: no 'Weighted N (All)' column")


def _validate_other_recodes():
    """The "Autre" / free-text recodes file applied by run_pipeline (counts only)."""
    _section("'Autre' recodes (other_recodes)")
    try:
        from pipeline.config import OTHER_RECODES_PATH
    except ImportError:
        return
    if not os.path.exists(OTHER_RECODES_PATH):
        _warn(f"No recodes file at {OTHER_RECODES_PATH} -- 'Autre' answers are not recoded. "
              f"Run table_analysis/02z_recode_other.py or the pipeline (both create it).")
        return
    rec = pd.read_csv(OTHER_RECODES_PATH, dtype=str).fillna("")
    if not _require(rec, os.path.basename(OTHER_RECODES_PATH), ["question", "other_text", "recode_to"]):
        return
    key = rec["question"] + "|" + rec["other_text"].str.strip().str.casefold()
    if key.duplicated().any():
        _warn(f"{int(key.duplicated().sum())} duplicate question/text rows -- the first is used")
    for q, g in rec.groupby("question"):
        code = g["recode_to"].str.strip()
        by = g.get("coded_by", pd.Series("", index=g.index)).fillna("")
        keep = code == "-"            # reviewed: leave as written
        coded = (code != "") & ~keep
        auto = coded & (by.str.startswith(("fuzzy", "rule")) | by.eq("group"))
        _ok(f"{q}: {int(coded.sum())} of {len(g)} texts coded ({int(auto.sum())} automatic -- rule/fuzzy/group, "
            f"review these), {int(keep.sum())} kept as written, {int((code == '').sum())} not yet reviewed")


def _validate_change_over_time():
    """Phone survey change-over-time outputs (3_linkage/), where they exist.
    key_indicators_change.csv must carry unweighted_* columns next to the
    weighted ones; the other comparisons are unweighted counts and shares."""
    _section("change over time (phone survey comparisons)")
    linkage = Path(__file__).resolve().parents[3] / "3_linkage"
    if not linkage.exists():
        _ok("No 3_linkage/ folder -- no phone survey round yet, nothing to check")
        return
    files = {
        "outputs/key_indicators_change.csv":
            ["baseline_value", "followup_value", "change_pp",
             "unweighted_baseline_value", "unweighted_followup_value", "unweighted_change_pp"],
        "outputs/pp_headline.csv": [],
        "outputs/knowledge_change_summary.csv": ["known_baseline_pct", "known_pulse_pct"],
    }
    files.update({f"outputs/{n}_change_summary.csv": [] for n in
                  ["fp_use", "unmet_need", "attitudes", "barriers"]})
    files["outputs/method_switching_summary.csv"] = []
    files.update({f"cross_sectional/outputs/{n}_change_summary_unlinked.csv": ["n_baseline", "n_pulse"]
                  for n in ["knowledge", "fp_use", "unmet_need", "attitudes", "barriers"]})
    files["cross_sectional/outputs/method_switching_summary_unlinked.csv"] = ["n_baseline", "n_pulse"]
    for rel, cols in files.items():
        path = linkage / rel
        if not path.exists():
            _warn(f"3_linkage/{rel} not found")
            continue
        df = pd.read_csv(path)
        if df.empty:
            _err(f"3_linkage/{rel}: no rows")
            continue
        if cols and not _require(df, rel, cols):
            continue
        _ok(f"3_linkage/{rel}: {len(df)} rows")
        if rel.endswith("key_indicators_change.csv"):
            for pre in ["", "unweighted_"]:
                vals = df[f"{pre}baseline_value"].dropna()
                if vals.empty:
                    _err(f"{rel}: '{pre}baseline_value' is all null")
            _ok(f"{rel}: weighted and unweighted estimates both present")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary():
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    if _errors:
        print(f"\n  {RED}{BOLD}{len(_errors)} ERROR(S):{RESET}")
        for e in _errors:
            print(f"    {RED}✗{RESET} {e}")
    if _warnings:
        print(f"\n  {YEL}{BOLD}{len(_warnings)} WARNING(S):{RESET}")
        for w in _warnings:
            print(f"    {YEL}⚠{RESET} {w}")
    if not _errors and not _warnings:
        print(f"\n  {GRN}{BOLD}All checks passed.{RESET}")
    elif not _errors:
        print(f"\n  {GRN}No errors.{RESET} Review warnings above.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validate FEM pipeline inputs and outputs."
    )
    parser.add_argument("--raw",        action="store_true", help="Validate raw survey CSV")
    parser.add_argument("--outputs",    action="store_true", help="Validate all generated CSVs")
    parser.add_argument("--statements", action="store_true", help="Validate statements_heatmap.csv only")
    parser.add_argument("--all",        action="store_true", help="Run all validations (default)")
    args = parser.parse_args()

    only_statements = args.statements and not args.raw and not args.outputs and not args.all
    run_raw     = args.raw  or args.all or (not any([args.raw, args.outputs, args.statements, args.all]))
    run_outputs = args.outputs or args.all or (not any([args.raw, args.outputs, args.statements, args.all]))

    if run_raw:
        if not os.path.exists(DIR_MAPPED_DATA):
            print(f"{RED}Raw data file not found: {DIR_MAPPED_DATA}{RESET}")
            print("Set FEM_MAPPED_DATA env var or update DIR_MAPPED_DATA in config.py")
        else:
            print(f"Loading raw data from: {DIR_MAPPED_DATA}")
            df = pd.read_csv(DIR_MAPPED_DATA, low_memory=False)
            df.columns = df.columns.str.strip()
            validate_raw(df)

    if only_statements:
        print(f"\n{BOLD}OUTPUT CSV VALIDATION (statements only){RESET}")
        _validate_out_statements()
    elif run_outputs:
        validate_outputs()

    _print_summary()
    sys.exit(1 if _errors else 0)


if __name__ == "__main__":
    main()