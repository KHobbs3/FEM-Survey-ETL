"""
ETL: Health Access & Supply page
Produces pre-aggregated CSVs — no PII in outputs.

Output files (written to APP_DATA_DIR):
  access_stockouts.csv       — stockout rates by split
  access_stockout_responses.csv — stockout response value counts
  access_travel.csv          — mean travel time + gap rate by split
  access_affordability.csv   — mean costs by split
  access_composite.csv       — composite barrier rates by use group
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import (
    WEIGHT_COL, USER_GROUPS, NONUSER_GROUPS, ALL_USE_GROUPS, WTT_MAP, APP_DATA_DIR, SPLIT_COLS,
    PROVINCE_TO_REGION,
)
from pipeline.utils import (
    weighted_prop, weighted_mean,
    split_weighted_prop, split_weighted_mean,
    save, load_raw
)

# Region added on top of the shared SPLIT_COLS (used by several other ETL modules)
# rather than editing SPLIT_COLS itself, so this stays scoped to the Access &
# Supply page only, per the user's request (2026-08-24).
ACCESS_SPLIT_COLS = SPLIT_COLS + ["region"]


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [access] running...")

    df = df.copy()
    if "province" in df.columns:
        df["region"] = df["province"].map(PROVINCE_TO_REGION)
        unmapped = df.loc[df["region"].isna() & df["province"].notna(), "province"].unique().tolist()
        if unmapped:
            print(f"  [access] WARNING: provinces with no region mapping (excluded from REGION split): {unmapped}")
    else:
        print("  [access] WARNING: 'province' column not found — skipping region split.")

    users    = df[df["use"].isin(USER_GROUPS)].copy()
    nonusers = df[df["use"].isin(NONUSER_GROUPS)].copy()

    # Map willingness to travel to minutes
    if "willingness_to_travel" in df.columns:
        df["wtt_minutes"] = df["willingness_to_travel"].map(WTT_MAP)

    # ── Helpers ───────────────────────────────────────────────────────────────
    # The survey uses bilingual "French/local language" strings (Ewe in Togo;
    # the examples below are Benin's French/Fon), e.g.:
    #   stockouts_users:              "Oui/ Ɛɛn"  or  "Non/Eo"
    #   stockouts_nonusers:           "Oui / Ɛɛn"  or  "Non / Eo"
    #   nonusers_seek_contraceptives: "J'ai essayé et j'ai réussi/ Un tɛnkpɔn bo ɖuɖeji"
    #                                 "J'ai essayé, mais sans succès/ Un tɛnkpɔn, amɔ̌, é kpa mi ǎ"
    #                                 "Je n'ai jamais essayé/ Un ko tɛnkpɔn kpɔn a"
    #
    # We match on the French "Oui" (yes) for stockout flags and on the
    # French "réussi" (succeeded) substring for the seeking question.
    # TODO(Benin): target is still eventually English -- add an English
    # `.str.contains(r"\byes\b", ...)` etc. fallback here once translated
    # labels exist, same as the old Hausa/English version did.
    #
    # 2026-09-07: stockouts_users / stockouts_nonusers are asked as
    #   "La dernière fois que vous avez essayé d'avoir accès à la
    #    contraception, la méthode que vous souhaitiez était-elle
    #    disponible ?" -- i.e. "...was it AVAILABLE?", framed positively.
    # "Oui" therefore means the method WAS available (no stockout); "Non"
    # means it was NOT available (a stockout). The previous _is_yes-based
    # flag had this backwards -- it counted "Oui" (available) as a
    # stockout -- which inflated the apparent stockout rate to roughly the
    # inverse of the true rate. _is_unavailable() below is the corrected
    # stockout flag; NaN is preserved (not coerced to True or False) for
    # responses that are neither "Oui" nor "Non", so those rows are
    # excluded from the denominator by weighted_prop's .dropna() rather
    # than silently counted either way.

    def _is_unavailable(series):
        """
        True if stockouts_users/stockouts_nonusers indicates the desired method
        was NOT available (a stockout) -- i.e. the response is "Non". False for
        "Oui" (available). NaN (missing / neither Oui nor Non) is preserved so
        those respondents are excluded from stockout-rate denominators instead
        of being counted as either available or unavailable.
        """
        s = series.astype(str).str.strip()
        result = pd.Series(pd.NA, index=series.index, dtype="boolean")
        result[s.str.startswith("Oui")] = False
        result[s.str.startswith("Non")] = True
        return result

    def _sought_successfully(series):
        """
        True if non-user tried AND succeeded in accessing contraceptives.
        Positive value: "J'ai essayé et j'ai réussi/ ..."
        Negative/irrelevant: "Je n'ai jamais essayé/...", "J'ai essayé, mais sans succès/...", NaN
        """
        s = series.astype(str).str.lower()
        return s.str.contains("j'ai essayé et j'ai réussi", na=False)

    def _sought_any(series):
        """
        True if non-user ever tried to access contraceptives (successful or not).
        Excludes "never tried" and NaN.
        """
        s = series.astype(str).str.lower()
        never = s.str.contains("je n'ai jamais essayé", na=False)
        is_null = series.isna()
        return ~never & ~is_null

    # ── 1. Stockout rates ─────────────────────────────────────────────────────
    rows = []

    # Users who experienced a stockout (method was NOT available last time
    # they sought it -- see _is_unavailable's docstring for the polarity fix)
    if "stockouts_users" in df.columns:
        users["_su"] = _is_unavailable(users["stockouts_users"])
        for split_col in ACCESS_SPLIT_COLS:
            s = split_weighted_prop(users, "_su", split_col)
            for grp, val in s.items():
                rows.append({"metric": "stockout_users", "split": split_col,
                             "group": grp, "value": val})

    # Non-users: who sought contraceptives, and of those, who hit a stockout
    if "nonusers_seek_contraceptives" in df.columns:
        nonusers["_sought"] = _sought_any(nonusers["nonusers_seek_contraceptives"])
        sought = nonusers[nonusers["_sought"]].copy()

        if "stockouts_nonusers" in df.columns:
            sought["_snu"] = _is_unavailable(sought["stockouts_nonusers"])
            for split_col in ACCESS_SPLIT_COLS:
                s = split_weighted_prop(sought, "_snu", split_col)
                for grp, val in s.items():
                    rows.append({"metric": "stockout_nonusers_sought", "split": split_col,
                                 "group": grp, "value": val})

        # Proportion of non-users who sought contraceptives (by split)
        for split_col in ACCESS_SPLIT_COLS:
            s = split_weighted_prop(nonusers, "_sought", split_col)
            for grp, val in s.items():
                rows.append({"metric": "sought_contraceptives", "split": split_col,
                             "group": grp, "value": val})

    save(pd.DataFrame(rows), os.path.join(APP_DATA_DIR, "access_stockouts.csv"),
         "stockout rates")

    # ── 2. Stockout responses (value counts, not weighted — qualitative) ──────
    # "Si la méthode de contraception n'était pas disponible, qu'avez-vous fait
    # ensuite ?" has no relevant()/skip condition in the form itself -- every
    # respondent's field can be non-null regardless of how they answered the
    # availability question. So this has to be constrained analytically here:
    # restrict each subset to _su/_snu == True (set above) before tallying,
    # rather than every non-null response. flag_col is a nullable boolean
    # ("boolean" dtype) and can hold NA (response was neither "Oui" nor "Non")
    # -- fillna(False) before using it as a row mask, both to avoid pandas
    # raising on an NA-containing boolean indexer and because NA (unknown
    # availability) should not count as "reported unavailable".
    resp_rows = []
    for col, other_col, label, flag_col in [
        ("stockouts_response",          "stockouts_response_other",          "users",    "_su"),
        ("stockouts_nonusers_response", "stockouts_nonusers_response_other", "nonusers", "_snu"),
    ]:
        if col in df.columns:
            subset = users if label == "users" else nonusers
            if flag_col in subset.columns:
                is_stockout = subset[flag_col].fillna(False).astype(bool)
                # Informational only (expected, since nothing in the form
                # gates this field): how many respondents answered the
                # follow-up despite not reporting a stockout -- these are
                # excluded from the tally below.
                n_excluded = int((subset[col].notna() & ~is_stockout).sum())
                if n_excluded:
                    print(f"  [access] {label}: excluding {n_excluded} respondent(s) who "
                          f"answered '{col}' without reporting a stockout in "
                          f"'{flag_col}' (question has no skip logic; constrained here).")
                subset = subset[is_stockout]
            combined = subset[col].copy()
            # Handle "other" free-text (original code checked == -88 but values are strings)
            if other_col in subset.columns:
                mask = combined.astype(str).str.startswith("-88") | (combined == -88)
                combined.loc[mask] = subset.loc[mask, other_col]
            # Keep the French half of "French/Ewe" labels (first segment)
            combined = combined.dropna().astype(str).str.split("/").str[0].str.strip()
            counts = combined.value_counts().reset_index()
            counts.columns = ["response", "count"]
            counts["group"] = label
            resp_rows.append(counts)

    if resp_rows:
        save(pd.concat(resp_rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "access_stockout_responses.csv"),
             "stockout responses")

    # ── 3. Travel times and gap ───────────────────────────────────────────────
    travel_rows = []

    for split_col in ACCESS_SPLIT_COLS:
        # Mean travel time users
        if "travel_time_users" in users.columns:
            s = split_weighted_mean(users, "travel_time_users", split_col)
            for grp, val in s.items():
                travel_rows.append({"metric": "mean_travel_users", "split": split_col,
                                    "group": grp, "value": val})

        # Mean travel time nonusers
        if "travel_time_nonusers" in nonusers.columns:
            s = split_weighted_mean(nonusers, "travel_time_nonusers", split_col)
            for grp, val in s.items():
                travel_rows.append({"metric": "mean_travel_nonusers", "split": split_col,
                                    "group": grp, "value": val})

    # 2026-09-04: "Distance access gap" replaced, per the user's request, with
    # two group-level averages instead of the users-only "% traveling farther
    # than willing" rate -- (1) mean willingness-to-travel, current/past
    # users vs. everyone else, and (2) mean reported/expected travel time,
    # same two groups. Groups are fixed here (not one of ACCESS_SPLIT_COLS):
    # "users" = current + past users (USER_GROUPS); "nonusers" = everyone
    # else, i.e. NOT restricted to NONUSER_GROUPS -- future_user and any
    # other/missing `use` value are folded in too, per the user's own framing
    # ("nonusers: never users (everyone else)").
    is_user = df["use"].isin(USER_GROUPS)
    group_label = pd.Series(np.where(is_user, "Current/past users", "Non-users"), index=df.index)

    if "wtt_minutes" in df.columns:
        tmp = df[["wtt_minutes"]].copy()
        tmp[WEIGHT_COL] = df[WEIGHT_COL]
        tmp["_group"] = group_label
        s = split_weighted_mean(tmp, "wtt_minutes", "_group")
        for grp, val in s.items():
            travel_rows.append({"metric": "mean_wtt_by_group", "split": "use_binary",
                                "group": grp, "value": val})

    # Reported travel time: users' own travel_time_users where available,
    # nonusers' travel_time_nonusers (the time they'd expect to need) --
    # combined into one column per respondent so both groups can be
    # weighted-averaged the same way.
    if "travel_time_users" in df.columns and "travel_time_nonusers" in df.columns:
        combined_travel = df["travel_time_users"].where(is_user, df["travel_time_nonusers"])
        tmp = pd.DataFrame({"_travel": combined_travel, WEIGHT_COL: df[WEIGHT_COL], "_group": group_label})
        s = split_weighted_mean(tmp, "_travel", "_group")
        for grp, val in s.items():
            travel_rows.append({"metric": "mean_travel_by_group", "split": "use_binary",
                                "group": grp, "value": val})

    # Transport mode value counts (qualitative, non-weighted)
    for col, label in [("transport_mode_users", "users"), ("transport_mode_nonusers", "nonusers")]:
        if col in df.columns:
            subset = users if label == "users" else nonusers
            # Keep the French half of "French/Fon" labels (Benin's order --
            # first segment, not last; the old Niger version kept the last
            # segment, matching its "Hausa / English" order instead).
            clean = subset[col].dropna().astype(str).str.split("/").str[0].str.strip()
            counts = clean.value_counts(normalize=True).head(8).reset_index()
            counts.columns = ["group", "value"]
            counts["metric"] = f"transport_mode_{label}"
            counts["split"] = "mode"
            travel_rows.extend(counts.to_dict("records"))

    save(pd.DataFrame(travel_rows), os.path.join(APP_DATA_DIR, "access_travel.csv"),
         "travel metrics")

    # ── 4. Affordability ──────────────────────────────────────────────────────
    afford_rows = []

    for split_col in ACCESS_SPLIT_COLS:
        if "user_costs" in users.columns:
            s = split_weighted_mean(users, "user_costs", split_col)
            for grp, val in s.items():
                afford_rows.append({"metric": "mean_cost_users", "split": split_col,
                                    "group": grp, "value": val})
        if "nonuser_cost" in nonusers.columns:
            s = split_weighted_mean(nonusers, "nonuser_cost", split_col)
            for grp, val in s.items():
                afford_rows.append({"metric": "mean_cost_nonusers", "split": split_col,
                                    "group": grp, "value": val})

    # Share paying anything
    if "user_costs" in users.columns:
        users["_cost_barrier"] = users["user_costs"].gt(0).fillna(False)
        overall_cost = weighted_prop(users, "_cost_barrier")
        afford_rows.append({"metric": "cost_barrier_overall", "split": "all",
                            "group": "all", "value": overall_cost})

    save(pd.DataFrame(afford_rows), os.path.join(APP_DATA_DIR, "access_affordability.csv"),
         "affordability metrics")

    # ── 5. Composite barrier rates ────────────────────────────────────────────
    df = df.copy()
    if "willingness_to_travel" in df.columns:
        df["wtt_minutes"] = df["willingness_to_travel"].map(WTT_MAP)

    # Same polarity fix as _is_unavailable above: stockouts_users/
    # stockouts_nonusers ask whether the method WAS available ("Oui" =
    # available, "Non" = a stockout), so the barrier flag must be True on
    # "Non", not "Oui".
    df["supply_barrier"] = False
    if "stockouts_users" in df.columns:
        df["supply_barrier"] |= _is_unavailable(df["stockouts_users"]).fillna(False)
    if "stockouts_nonusers" in df.columns:
        df["supply_barrier"] |= _is_unavailable(df["stockouts_nonusers"]).fillna(False)

    df["geo_barrier"] = False
    if "travel_time_users" in df.columns and "wtt_minutes" in df.columns:
        df["geo_barrier"] = (df["travel_time_users"] - df["wtt_minutes"]).gt(0).fillna(False)

    df["cost_barrier"] = False
    if "user_costs" in df.columns:
        df["cost_barrier"] = df["user_costs"].gt(0).fillna(False)

    df["any_barrier"] = df["supply_barrier"] | df["geo_barrier"] | df["cost_barrier"]

    composite_rows = []
    barriers = {
        "Supply (stockout)": "supply_barrier",
        "Geographic (travel gap)": "geo_barrier",
        "Cost (paid > 0)": "cost_barrier",
        "Any barrier": "any_barrier",
    }
    use_groups = ALL_USE_GROUPS + ["all"]
    for label, col in barriers.items():
        for grp in use_groups:
            sub = df if grp == "all" else df[df["use"] == grp]
            val = weighted_prop(sub, col)
            composite_rows.append({"barrier": label, "use_group": grp, "rate": val})

    save(pd.DataFrame(composite_rows), os.path.join(APP_DATA_DIR, "access_composite.csv"),
         "composite barriers")

    print("  [access] done.")
