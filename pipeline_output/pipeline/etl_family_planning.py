"""
ETL: Family Planning page
Produces pre-aggregated CSVs — no PII in outputs.

Output files:
  fp_funnel.csv             — aware/ever/current rates by split group
  fp_timing.csv             — timing of next pregnancy distribution
  fp_methods.csv            — method proportions (known/ever/current) by split
  fp_reason_use.csv         — reason for use by split
  fp_intent.csv             — future intent + considered use by split
  fp_nonuse_reasons.csv     — free-text non-use reason counts (no names/IDs)
  fp_unmet.csv              — unmet need / unmet demand by split group
"""

import pandas as pd
import numpy as np
import os

from pipeline.config import (
    WEIGHT_COL, SPLIT_COLS, APP_DATA_DIR,
    CONTRACEPTIVE_METHODS, CI_CLUSTER_COL, CI_STRATUM_COL,
)
from pipeline.utils import (
    weighted_prop, weighted_counts, weighted_multiselect_counts,
    weighted_multiselect_counts_text,
    split_weighted_counts, split_weighted_multiselect,
    split_weighted_multiselect_text,
    safe_melt,
    save,
)

TIME_TO_PREGNANT = {
    1:"Within 6 months", 2:"6-12 months", 3:"1-2 years",
    4:"More than 2 years", 5:"Already pregnant", 6:"No more children",
}
# 2026-09-02: time_before_preferred_pregnancy does NOT hold the numeric codes
# TIME_TO_PREGNANT's keys assume -- validate_pipeline.py --raw showed it's
# French/Fon bilingual text instead (e.g. "Dans 1 à 2 ans/ Ðo xwe ɖokpo jɛ we
# mɛ"), same as current_use_methods elsewhere in this file. This is the
# French half of each option (run() strips the Fon half via _strip_hausa,
# same as current_use_methods), keyed to the same codes as TIME_TO_PREGNANT
# above so the two dicts stay in sync.
TIME_TO_PREGNANT_FR = {
    1: "Dans les 6 prochains mois",
    2: "Dans 6 à 12 mois",
    3: "Dans 1 à 2 ans",
    4: "Plus de 2 ans à partir de maintenant",
    5: "Je suis / elle est déjà enceinte",
    6: "Je ne prévois pas d'avoir d'autres enfants",
}
# "Don't know" / "prefer not to say" for the same question -- excluded from
# both the timing chart and the unmet-need mask, same convention as -88/-99
# elsewhere in this file (this column just doesn't use those numeric codes).
TIME_TO_PREGNANT_EXCLUDE_FR = ["Je ne sais pas", "Je préfère ne pas répondre"]
REASON_USE = {1:"Space births", 2:"No more children", -22:"Other"}
YESNO = {1:"Oui", 0:"Non"}

# 2026-08-27: "current_use" (fpbeh_fpnow's formative-research equivalent) is
# just "are you doing *anything* to avoid pregnancy" -- that counts
# withdrawal, the calendar method, etc. as "using", which the user flagged
# as potentially inflating the headline rate relative to actually-effective
# methods. CONTRACEPTIVE_METHODS' keys 1-9 and 15 are WHO's modern-method
# categories (sterilisation, implants, pills, IUD, injectables, condoms,
# ring, patch, vaginal barrier methods, emergency pills); 10-14 (withdrawal,
# abstinence, calendar method, standard days, LAM) are the traditional/
# less-effective methods -- see the CONTRACEPTIVE_METHODS comment in
# pipeline/config.py for the source list.
MODERN_METHOD_KEYS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15}
MODERN_METHODS = {CONTRACEPTIVE_METHODS[k] for k in MODERN_METHOD_KEYS if k in CONTRACEPTIVE_METHODS}

HAUSA_ENG = {
    "Il n'est pas intéressé": "He's not interested",
    "Inason haifuwa yara": "I want to have children",
    "Rien": "Nothing",
    "Ban sansu ba": "I don't know them",
    "Saboda mijina baya guida": "Because my husband doesn't guide",
    "Babu takameman dalili": "No specific reason",
    "Inason haifuwa": "I want to have children",
    "Rashin ganin jinin haila yayin shayarwa": "Not getting my period while breastfeeding",
    "Ina ayiki da allurai yanzuma": "I'm on birth control right now",
    "Zanyi se nan gaba": "I'll do it later",
    "Sabida ina da ciki": "Because I am pregnant",
    "Sabida mijina baya guida": "Because my husband is infertile",
    "Saboda yarana sunada issassar tazara": "Because my children are far apart in age",
    "Bukatar samun yara": "Want to have more children",
    "Haifuwa nikeso": "I want to give birth",
    "Manque de connaissance": "Lack of knowledge",
    "Jikina Yana bani tazara": "My body spacing",
    "Bukatar karin samun yara": "Want to have more children",
    "Manque de connaissance sur le planning familial": "Lack of knowledge about family planning",
    "Tsoron matsalar zubar jini": "Fear of complications from bleeding"
}

def _strip_hausa(text: str) -> str:
    """Return only the French part of bilingual 'French/Fon' labels (Benin's
    order -- the first, not the second/last, segment is the language we keep).

    Handles both single labels and pipe-delimited multiple labels.
    Preserves NaN values.
    """
    # Preserve NaN/None
    if pd.isna(text):
        return np.nan

    text = str(text).strip()

    # Handle pipe-delimited multiple options
    if "|" in text:
        parts = text.split("|")
        french_parts = [_strip_hausa(part) for part in parts]
        # Filter out NaN results
        french_parts = [p for p in french_parts if pd.notna(p)]
        return "|".join(french_parts) if french_parts else np.nan

    # Handle single bilingual label (French/Fon)
    if "/" in text:
        split = text.split("/", 1)
        if len(split) == 2:
            french = split[0].strip()
            return french if french else np.nan

    return text if text else np.nan

def _translate_hausa(s):
    # TODO(Benin): HAUSA_ENG above is still Niger's Hausa/French -> English
    # open-ended-response dictionary; Benin's open-ended responses are French
    # and aren't in it, so this passes them through unchanged for now
    # (target is still eventually English, once real French->English entries
    # are added here).
    return HAUSA_ENG.get(s, s)

def _mentions_modern_method(cell):
    """True if a pipe-delimited multi-select cell names >=1 modern/effective
    method (MODERN_METHODS). Shared by effective_use (current_use_methods)
    and aware_modern_method (known_contraceptive_options) below -- same
    check, different column."""
    if pd.isna(cell):
        return False
    return any(m.strip() in MODERN_METHODS for m in str(cell).split("|"))


def _funnel_row(df, split_col, group_val):
    """Compute funnel proportions for one subgroup."""
    sub = df if group_val == "all" else df[df[split_col] == group_val]
    w = sub[WEIGHT_COL].sum()
    if w == 0:
        return None

    def wp(mask):
        return sub.loc[mask, WEIGHT_COL].sum() / w

    aware = wp(sub["birth_spacing"].fillna("NA").str.contains('Oui')) if "birth_spacing" in sub.columns else np.nan
    ever  = wp(sub["ever_use"].fillna("NA").str.contains('Oui'))       if "ever_use"      in sub.columns else np.nan
    curr  = wp(sub["current_use"].fillna("NA").str.contains('Oui'))    if "current_use"   in sub.columns else np.nan

    # % using a modern/effective method specifically -- narrower than
    # "current_use" above, which counts traditional methods too. Requires
    # current_use_methods to already be Fon-stripped (run() does this
    # before calling _funnel_row) so its pipe-delimited values match
    # MODERN_METHODS' French labels.
    effective = np.nan
    if "current_use_methods" in sub.columns:
        effective = wp(sub["current_use_methods"].apply(_mentions_modern_method))

    # 2026-09-03: stricter awareness check, requested to confirm "aware" (a
    # self-reported yes/no on birth_spacing) actually names a real modern
    # method. known_contraceptive_options is the XLSForm's own "if yes"
    # follow-up to birth_spacing (same conditional skip logic, not an
    # independently-asked question) -- % of respondents who named >=1 modern
    # method there. Also requires Fon-stripping first, see run().
    #
    # Denominator note (2026-09-03): `wp()` divides by `w`, computed above
    # from the FULL `sub` group before any filtering -- same denominator as
    # every other funnel stage. Respondents skipped out of
    # known_contraceptive_options (because they said "Non" to birth_spacing)
    # have NaN there -> _mentions_modern_method returns False for them, so
    # they count as "not aware of a modern method" in the numerator while
    # still counted in the denominator, not excluded from it.
    aware_modern = np.nan
    if "known_contraceptive_options" in sub.columns:
        aware_modern = wp(sub["known_contraceptive_options"].apply(_mentions_modern_method))

    return {
        "split": split_col, "group": group_val,
        "aware": aware, "ever_used": ever, "current_use": curr,
        "effective_use": effective, "aware_modern_method": aware_modern,
    }


# 2026-09-02: unmet need / unmet demand, added alongside the funnel above.
# Both use the same "all women" denominator as _funnel_row's current_use /
# effective_use, so mCPR (effective_use), unmet need and unmet demand line up
# under the standard FP2030-style "total demand" framework:
#   total demand      = mCPR + unmet need
#   demand satisfied   = mCPR / total demand
#
# time_before_preferred_pregnancy codes (see TIME_TO_PREGNANT above): 1/2 =
# wants a child within the year (no unmet need), 3/4 = wants to delay 1+
# years, 6 = wants no more children. 5 (already pregnant) and -88/-99
# (don't know / prefer not to say) are excluded from "wants to delay or
# limit" -- they simply don't match WANTS_TO_DELAY_CODES below, the same
# way -88/-99 are excluded elsewhere in this file.
# 2026-10-26: DHS-style definition, per the user -- 4 (more than 2 years) and
# 6 (no more children) only. 3 ("1-2 years") was dropped: the bucket straddles
# DHS's 2-year line, so someone wanting to wait 13 months landed in the same
# bucket as someone wanting 23, over-counting unmet need.
WANTS_TO_DELAY_CODES = [4, 6]
# Match against TIME_TO_PREGNANT_FR's label text (confirmed via
# validate_pipeline.py --raw to be what this column actually holds -- see
# TIME_TO_PREGNANT_FR's comment above), not TIME_TO_PREGNANT's numeric
# codes. run() strips the Fon half of the column (via _strip_hausa) before
# _unmet_row sees it, same as current_use_methods, so it's pure French by
# the time this is compared. The numeric isin() in _unmet_row below is kept
# as a harmless fallback in case a future dataset does use numeric codes.
WANTS_TO_DELAY_LABELS = {TIME_TO_PREGNANT_FR[c] for c in WANTS_TO_DELAY_CODES}


def _unmet_row(df, split_col, group_val):
    """Unmet need / unmet demand for one subgroup."""
    sub = df if group_val == "all" else df[df[split_col] == group_val]
    w = sub[WEIGHT_COL].sum()
    if w == 0:
        return None
    if "time_before_preferred_pregnancy" not in sub.columns or "current_use" not in sub.columns:
        return None

    def wp(mask):
        return sub.loc[mask, WEIGHT_COL].sum() / w

    # Unmet need: wants to delay/limit births (see WANTS_TO_DELAY_CODES)
    # but isn't currently using *any* method. This checks current_use, not
    # current_use_methods/MODERN_METHODS -- a traditional-method user still
    # counts as having her need "met" here, same convention DHS uses (it's
    # a separate question from whether that method is effective).
    timing_col = sub["time_before_preferred_pregnancy"]
    wants_to_delay = (
        pd.to_numeric(timing_col, errors="coerce").isin(WANTS_TO_DELAY_CODES)
        | timing_col.astype(str).isin(WANTS_TO_DELAY_LABELS)
    )
    not_using = ~sub["current_use"].fillna("NA").str.contains('Oui')
    unmet_need_mask = wants_to_delay & not_using

    # Unmet demand: the narrower, preference-grounded cut of the above --
    # only women in the unmet-need group who also say they're open to /
    # interested in using contraception in future (future_intent). Left as
    # NaN (not 0) if future_intent isn't in this dataset, so it's visibly
    # missing rather than silently wrong.
    unmet_demand = np.nan
    if "future_intent" in sub.columns:
        open_to_future = sub["future_intent"].fillna("NA").str.contains('Oui')
        unmet_demand = wp(unmet_need_mask & open_to_future)

    return {
        "split": split_col, "group": group_val,
        "wants_to_delay_or_limit": wp(wants_to_delay),
        "unmet_need": wp(unmet_need_mask),
        "unmet_demand": unmet_demand,
    }


# ── Headline indicators with design-based 95% CIs ─────────────────────────────
# Full sample and all women, same indicators and method as Niger's pipeline.
# Point estimates use the same weighted, whole-group denominators as
# _funnel_row / _unmet_row. CIs come from a Rao-Wu rescaling bootstrap: within
# each stratum of n_h clusters, draw n_h-1 clusters with replacement and scale
# their weights by n_h/(n_h-1) x times drawn; CI = 2.5th-97.5th percentile of
# the replicate estimates. Women are computed from the same replicates (domain
# estimation), not resampled alone. Cluster/stratum columns come from
# config.CI_CLUSTER_COL / CI_STRATUM_COL -- the key the weights were joined on,
# i.e. the CORRECTED location from table_analysis/02a_correct_locations.
N_BOOT = 1000
BOOT_SEED = 2026


def _headline_flags(df):
    """Boolean arrays the headline indicators are built from."""
    using_any = df["current_use"].fillna("NA").str.contains('Oui').to_numpy()
    timing_col = df["time_before_preferred_pregnancy"]
    wants_to_delay = (
        pd.to_numeric(timing_col, errors="coerce").isin(WANTS_TO_DELAY_CODES)
        | timing_col.astype(str).isin(WANTS_TO_DELAY_LABELS)
    ).to_numpy()
    return {
        "any":    using_any,
        "modern": df["current_use_methods"].apply(_mentions_modern_method).to_numpy(),
        "unmet":  wants_to_delay & ~using_any,
        "open":   df["future_intent"].fillna("NA").str.contains('Oui').to_numpy(),
        "aware":  df["known_contraceptive_options"].apply(_mentions_modern_method).to_numpy(),
    }


def _headline_values(w, pop, f):
    def share(num, den):
        d = (w * den).sum()
        return (w * (num & den)).sum() / d if d else np.nan

    mcpr, cpr, need = share(f["modern"], pop), share(f["any"], pop), share(f["unmet"], pop)
    return {
        "mcpr":             mcpr,
        "unmet_need":       need,
        "unmet_demand":     share(f["unmet"] & f["open"], pop),
        # Official DHS formula: mCPR / (total CPR + unmet need) -- see
        # page_family_planning.py render_unmet for why total CPR, not mCPR.
        "demand_satisfied": mcpr / (cpr + need) if (cpr + need) else np.nan,
        "aware_modern":     share(f["aware"], pop),
        "aware_modern_nonusers": share(f["aware"], pop & ~f["any"]),
    }


def _rao_wu_multipliers(df, rng):
    """(N_BOOT, n_rows) weight multipliers, or None if the design columns are missing."""
    missing = [c for c in (CI_CLUSTER_COL, CI_STRATUM_COL) if c not in df.columns]
    if missing:
        print(f"  [family_planning] WARNING: {missing} not in data -- headline CIs left blank")
        return None
    clusters = df[CI_CLUSTER_COL].astype(str).str.strip().to_numpy()
    strata = df[CI_STRATUM_COL].astype(str).str.strip().to_numpy()
    mult = np.ones((N_BOOT, len(df)))
    for s in np.unique(strata):
        rows = strata == s
        cl, pos = np.unique(clusters[rows], return_inverse=True)
        n_h = len(cl)
        if n_h < 2:
            print(f"  [family_planning] WARNING: stratum {s!r} has 1 cluster -- no CI variance from it")
            continue
        draws = rng.integers(0, n_h, size=(N_BOOT, n_h - 1))
        counts = np.stack([np.bincount(d, minlength=n_h) for d in draws])
        mult[:, rows] = (counts * n_h / (n_h - 1))[:, pos]
    print(f"  [family_planning] CI design: {len(np.unique(strata))} strata ({CI_STRATUM_COL}), "
          f"{len(np.unique(clusters))} clusters ({CI_CLUSTER_COL}), {N_BOOT} bootstrap replicates")
    return mult


def _headline_table(df, women_mask):
    w = df[WEIGHT_COL].fillna(0).to_numpy()
    n_no_weight = int(df[WEIGHT_COL].isna().sum())
    if n_no_weight:
        print(f"  [family_planning] note: {n_no_weight} respondents have no weight "
              f"(excluded from estimates, still counted in n_unweighted)")
    f = _headline_flags(df)
    pops = {"full_sample": np.ones(len(df), dtype=bool), "women": women_mask.to_numpy()}
    mult = _rao_wu_multipliers(df, np.random.default_rng(BOOT_SEED))

    rows = []
    for pop_name, pop in pops.items():
        point = _headline_values(w, pop, f)
        reps = None
        if mult is not None:
            reps = pd.DataFrame([_headline_values(w * m, pop, f) for m in mult])
        for metric, value in point.items():
            n = int((pop & ~f["any"]).sum()) if metric == "aware_modern_nonusers" else int(pop.sum())
            lo = hi = np.nan
            if reps is not None:
                lo, hi = np.nanpercentile(reps[metric], [2.5, 97.5])
            rows.append({"population": pop_name, "metric": metric, "value": value,
                         "ci_low": lo, "ci_high": hi, "n_unweighted": n})
    return pd.DataFrame(rows)


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [family_planning] running...")

    # Strip Fon before _funnel_row's effective-method/aware-modern-method
    # checks (below) need it -- the "Methods" section further down strips
    # current_use_methods again for its own purposes, which is a harmless
    # no-op on already-French text.
    if "current_use_methods" in df.columns:
        df["current_use_methods"] = df["current_use_methods"].apply(_strip_hausa)
    if "known_contraceptive_options" in df.columns:
        df["known_contraceptive_options"] = df["known_contraceptive_options"].apply(_strip_hausa)

    # Same treatment for time_before_preferred_pregnancy, needed below by
    # both _unmet_row (WANTS_TO_DELAY_LABELS) and the "Timing of next
    # pregnancy" section further down (TIME_TO_PREGNANT_FR) -- confirmed via
    # validate_pipeline.py --raw to be French/Fon bilingual text, not the
    # numeric codes TIME_TO_PREGNANT's keys assume.
    if "time_before_preferred_pregnancy" in df.columns:
        df["time_before_preferred_pregnancy"] = df["time_before_preferred_pregnancy"].apply(_strip_hausa)

    # ── Funnel ────────────────────────────────────────────────────────────────
    funnel_rows = []
    for split_col in SPLIT_COLS:
        funnel_rows.append(_funnel_row(df, split_col, "all"))
        for grp in df[split_col].dropna().unique():
            funnel_rows.append(_funnel_row(df, split_col, grp))
    save(pd.DataFrame([r for r in funnel_rows if r]),
         os.path.join(APP_DATA_DIR, "fp_funnel.csv"), "funnel")

    # ── Unmet need / unmet demand ────────────────────────────────────────────
    unmet_rows = []
    for split_col in SPLIT_COLS:
        unmet_rows.append(_unmet_row(df, split_col, "all"))
        for grp in df[split_col].dropna().unique():
            unmet_rows.append(_unmet_row(df, split_col, grp))
    save(pd.DataFrame([r for r in unmet_rows if r]),
         os.path.join(APP_DATA_DIR, "fp_unmet.csv"), "unmet need/demand")

    # ── Headline indicators: full sample + all women ──────────────────────────
    women = df["gender"].fillna("").str.contains("Femme|Female")
    print(f"  [family_planning] headline: {len(df)} respondents, {int(women.sum())} women")
    save(_headline_table(df, women),
         os.path.join(APP_DATA_DIR, "fp_headline.csv"), "headline indicators")

    # ── Timing of next pregnancy ──────────────────────────────────────────────
    if "time_before_preferred_pregnancy" in df.columns:
        rows = []
        # label_map/exclude keyed by the French text this column actually
        # holds (see TIME_TO_PREGNANT_FR above), not TIME_TO_PREGNANT's
        # numeric codes -- weighted_counts() groups by raw value first, so
        # this used to still "work" in the sense of not crashing, but every
        # bar's label fell back to the raw bilingual French/Fon string
        # instead of TIME_TO_PREGNANT's clean English text, and -88/-99
        # never matched anything so "don't know"/"prefer not to say"
        # weren't actually excluded.
        label_map_fr = {fr: TIME_TO_PREGNANT[code] for code, fr in TIME_TO_PREGNANT_FR.items()}
        # Overall
        s = weighted_counts(df, "time_before_preferred_pregnancy", label_map_fr,
                            exclude=TIME_TO_PREGNANT_EXCLUDE_FR)
        tmp = s.reset_index(); tmp.columns = ["label", "proportion"]
        tmp["split"] = "none"; tmp["group"] = "all"
        rows.append(tmp)
        # By split
        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, "time_before_preferred_pregnancy",
                                          split_col, label_map_fr,
                                          exclude=TIME_TO_PREGNANT_EXCLUDE_FR)
            # melted = frame.reset_index().melt(id_vars="index", var_name="group",
            #                                   value_name="proportion")
            # melted.columns = ["label", "group", "proportion"]
            melted = safe_melt(frame)
            melted["split"] = split_col
            rows.append(melted)
        save(pd.concat(rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "fp_timing.csv"), "timing")

    # ── Methods (known / ever used / current) ─────────────────────────────────
    method_cols = {
        "known": "known_contraceptive_options",
        "ever":  "ever_used_methods",
        "current": "current_use_methods",
    }
    method_rows = []

    for method_type, col in method_cols.items():
        if col not in df.columns:
            print(f"  Column '{col}' not found")
            continue

        print(f"  Processing {method_type} ({col})...")
        
        # Strip hausa to get English only
        df[col] = df[col].apply(_strip_hausa)
        
        # Overall — use text-aware function; values are pipe-delimited English strings
        s = weighted_multiselect_counts_text(df, col, label_map=None, sep="|")

        tmp = s.reset_index(); tmp.columns = ["method", "proportion"]
        tmp["method_type"] = method_type; tmp["split"] = "none"; tmp["group"] = "all"
        method_rows.append(tmp)

        # By split
        for split_col in SPLIT_COLS:
            frame = split_weighted_multiselect_text(df, col, split_col, label_map=None, sep="|")
            melted = safe_melt(frame)
            melted["method_type"] = method_type
            melted["split"] = split_col
            method_rows.append(melted)
            
    if method_rows:
        result = pd.concat(method_rows, ignore_index=True)
        print(f"  Total fp_methods rows: {len(result)}")
        save(result, os.path.join(APP_DATA_DIR, "fp_methods.csv"), "methods")
    else:
        print("  [family_planning] WARNING: No method data found.")

    # ── Reason for use ────────────────────────────────────────────────────────
    if "reason_current_use" in df.columns:
        rows = []
        s = weighted_counts(df, "reason_current_use", REASON_USE, exclude=(-88, -99))
        tmp = s.reset_index(); tmp.columns = ["label", "proportion"]
        tmp["split"] = "none"; tmp["group"] = "all"
        rows.append(tmp)
        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, "reason_current_use", split_col,
                                          REASON_USE, exclude=(-88, -99))
            # melted = frame.reset_index().melt(id_vars="index", var_name="group",
            #                                   value_name="proportion")
            # melted.columns = ["label", "group", "proportion"]

            melted = safe_melt(frame)
            melted["split"] = split_col
            rows.append(melted)
        save(pd.concat(rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "fp_reason_use.csv"), "reason for use")

    # ── Future intent + considered use ───────────────────────────────────────
    intent_rows = []
    for col, label in [("future_intent", "future_intent"),
                       ("considered_use", "considered_use")]:
        if col not in df.columns:
            continue
        s = weighted_counts(df, col, YESNO, exclude=(-88, -99))
        tmp = s.reset_index(); tmp.columns = ["response", "proportion"]
        tmp["question"] = label; tmp["split"] = "none"; tmp["group"] = "all"
        intent_rows.append(tmp)
        for split_col in SPLIT_COLS:
            frame = split_weighted_counts(df, col, split_col, YESNO, exclude=(-88, -99))
            # melted = frame.reset_index().melt(id_vars="index", var_name="group",
            #                                   value_name="proportion")
            # melted.columns = ["response", "group", "proportion"]

            melted = safe_melt(frame)
            melted["question"] = label; melted["split"] = split_col
            intent_rows.append(melted)
    if intent_rows:
        save(pd.concat(intent_rows, ignore_index=True),
             os.path.join(APP_DATA_DIR, "fp_intent.csv"), "intent")

    # ── Non-use reasons (free text — aggregated counts only, no raw text) ─────
    if "reason_current_nonuse" in df.columns:
        counts = (
            df[df["reason_current_nonuse"].notna()]["reason_current_nonuse"]
            .value_counts()
            .head(20)
            .reset_index()
        )
        counts.columns = ["Reason", "Unweighted count"]

        # translate to ENG
        counts["Reason"] = counts["Reason"].apply(_translate_hausa)
        
        save(counts, os.path.join(APP_DATA_DIR, "fp_nonuse_reasons.csv"),
             "non-use reasons")

    print("  [family_planning] done.")
