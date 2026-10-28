"""
Drivers & barriers grouped by parent category.

Many answer options share a parent, e.g. "Peur des effets secondaires : prise
de poids" and "Peur des effets secondaires : saignements". This groups the
options by the text before the ":" and reports, per parent, the share of
respondents who picked at least 1 and at least 3 of its options.

Output:
  driver_barrier_groups.csv -- one row per type x parent x split group x threshold:
    type, parent, n_options, split, group, threshold ("at least 1"/"at least 3"),
    proportion (weighted), unweighted_proportion, n (respondents meeting it),
    weighted_n (their summed weights), base_n (respondents who answered the list)

Each respondent's list is the user + non-user version of the question combined
(DB_LISTS in config.py, same pairing as table_analysis/03). The base is everyone
with an answer to that list. "Autre" and don't-know / prefer-not / none answers
are not grouped, but those respondents stay in the base. "At least 3" is only
reported for parents with 3 or more options.
"""

import os

import pandas as pd

from pipeline.config import APP_DATA_DIR, WEIGHT_COL, SPLIT_COLS, DB_LISTS, DB_LABEL_INDEX
from pipeline.other_recodes import is_other_item, NONSUBSTANTIVE_PREFIXES
from pipeline.utils import save

THRESHOLDS = {"at least 1": 1, "at least 3": 3}


def parent_of(item: str) -> str:
    """Parent category: the display-language half of the bilingual label
    (DB_LABEL_INDEX), up to the first ':'."""
    text = item.split("/", 1)[DB_LABEL_INDEX] if "/" in item else item
    return text.split(":", 1)[0].strip()


def _skip(item: str) -> bool:
    display = parent_of(item).casefold()
    return is_other_item(item) or display.startswith(NONSUBSTANTIVE_PREFIXES)


def _group_counts(answers: pd.Series) -> pd.DataFrame:
    """respondent x parent -> number of distinct options picked."""
    items = answers.str.split("|").explode().str.strip()
    items = items[(items != "") & ~items.apply(_skip)]
    if items.empty:
        return pd.DataFrame(index=answers.index), pd.Series(dtype=int)
    long = pd.DataFrame({"item": items, "parent": items.apply(parent_of)})
    long = long.reset_index().drop_duplicates(["index", "item"])
    counts = long.groupby(["index", "parent"]).size().unstack(fill_value=0)
    return counts.reindex(answers.index, fill_value=0), long.groupby("parent")["item"].nunique()


def _shares(counts, weights, n_options, db_type, split, group):
    rows = []
    base_n = len(counts)
    totw = weights.sum()
    for parent in counts.columns:
        for label, k in THRESHOLDS.items():
            if n_options[parent] < k:
                continue
            meets = counts[parent] >= k
            rows.append({
                "type": db_type, "parent": parent, "n_options": int(n_options[parent]),
                "split": split, "group": group, "threshold": label,
                "proportion": weights[meets].sum() / totw if totw else float("nan"),
                "unweighted_proportion": meets.mean() if base_n else float("nan"),
                "n": int(meets.sum()), "weighted_n": weights[meets].sum(), "base_n": base_n,
            })
    return rows


def run(df):
    os.makedirs(APP_DATA_DIR, exist_ok=True)
    print("  [driver_barrier_groups] running...")
    splits = [c for c in SPLIT_COLS + ["region"] if c in df.columns]

    rows = []
    for db_type, cols in DB_LISTS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            print(f"  [driver_barrier_groups] {db_type}: none of {cols} in data -- skipped")
            continue
        answers = df[present[0]]
        for c in present[1:]:
            answers = answers.fillna(df[c])
        answered = answers.notna()
        answers = answers[answered].astype(str)
        counts, n_options = _group_counts(answers)
        weights = pd.to_numeric(df.loc[answered, WEIGHT_COL], errors="coerce").fillna(0)

        rows += _shares(counts, weights, n_options, db_type, "none", "all")
        for split in splits:
            groups = df.loc[answered, split]
            for group in groups.dropna().unique():
                m = groups == group
                rows += _shares(counts[m], weights[m], n_options, db_type, split, group)
        print(f"  [driver_barrier_groups] {db_type}: {int(answered.sum())} respondents, "
              f"{len(n_options)} parent categories")

    if rows:
        save(pd.DataFrame(rows), os.path.join(APP_DATA_DIR, "driver_barrier_groups.csv"),
             "driver/barrier parent groups")
