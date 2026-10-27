"""
Manual recoding of "Autre / Other" write-ins.

Two steps, same pattern as the radio station-name review:

1. Export (doesn't change any output):
       python -m pipeline.other_recodes
   writes every distinct write-in for OTHER_RECODE_QUESTIONS, with its count
   and a blank `recode_to` column, to OTHER_WRITEINS_EXPORT. Rows already
   coded in OTHER_RECODES_PATH keep their `recode_to`, so re-running after
   new data only adds the new write-ins. Prints aggregate counts only -- the
   write-in text is free text and stays in the data folder.

2. Review the export, fill `recode_to` (an existing answer label, or a new
   one), and save it as OTHER_RECODES_PATH. Blank `recode_to` = leave as
   "Autre". run_pipeline.py then calls apply_other_recodes() right after
   loading, so every page sees the recoded answers. With no recodes file
   it does nothing.

Keyed by question + write-in text, never by respondent.
"""

import os
import re

import numpy as np
import pandas as pd

from pipeline.config import (
    DIR_MAPPED_DATA, OTHER_RECODE_QUESTIONS, OTHER_RECODES_PATH, OTHER_WRITEINS_EXPORT,
)

RECODE_COLS = ["question", "other_text", "n", "recode_to"]


def is_other_item(item) -> bool:
    """True for the "Autre"/"Other" catch-all choice. Labels are bilingual
    ('Autre : veuillez préciser/...', 'Wani / Autre'), so any '/'-segment
    starting with autre/other counts; '-22' is the form's code for it."""
    s = str(item).strip()
    if s == "-22" or s.startswith("-22 "):
        return True
    return any(seg.strip().casefold().startswith(("autre", "other")) for seg in s.split("/"))


def _key(text) -> str:
    """Match key for a write-in: trimmed, case-folded, whitespace collapsed."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def clean_writein(value):
    if pd.isna(value):
        return np.nan
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s if s else np.nan


def _other_rows(df: pd.DataFrame, question: str) -> pd.Series:
    """Boolean mask: the answer contains the "Autre" choice."""
    return df[question].apply(
        lambda v: pd.notna(v) and any(is_other_item(p) for p in str(v).split("|"))
    )


def _load_recodes(path=OTHER_RECODES_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=RECODE_COLS)
    rec = pd.read_csv(path, dtype=str).fillna("")
    missing = {"question", "other_text", "recode_to"} - set(rec.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return rec


# ── Step 1: export ────────────────────────────────────────────────────────────

def export_writeins(df: pd.DataFrame, out_path=OTHER_WRITEINS_EXPORT,
                    recodes_path=OTHER_RECODES_PATH) -> pd.DataFrame:
    existing = _load_recodes(recodes_path)
    coded = {(q, _key(t)): r for q, t, r in
             zip(existing["question"], existing["other_text"], existing["recode_to"])}

    rows = []
    for q in OTHER_RECODE_QUESTIONS:
        other_col = f"{q}_other"
        if q not in df.columns or other_col not in df.columns:
            print(f"  [other] {q}: skipped -- '{q}' or '{other_col}' not in data")
            continue
        is_other = _other_rows(df, q)
        text = df.loc[is_other, other_col].apply(clean_writein).dropna()
        n_blank = int(is_other.sum()) - len(text)

        counts = {}
        for t in text:
            k = _key(t)
            spelling, n = counts.get(k, (t, 0))
            counts[k] = (spelling, n + 1)
        for k, (spelling, n) in counts.items():
            rows.append({"question": q, "other_text": spelling, "n": n,
                         "recode_to": coded.get((q, k), "")})

        n_coded = sum(1 for k in counts if coded.get((q, k)))
        print(f"  [other] {q}: {int(is_other.sum())} chose 'Autre' "
              f"({n_blank} with no write-in), {len(counts)} distinct write-ins, "
              f"{n_coded} already coded")

    out = pd.DataFrame(rows, columns=RECODE_COLS)
    out = out.sort_values(["question", "n", "other_text"], ascending=[True, False, True])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [other] wrote {len(out)} rows -> {out_path}")
    return out


# ── Step 2: apply ─────────────────────────────────────────────────────────────

def apply_other_recodes(df: pd.DataFrame, recodes_path=OTHER_RECODES_PATH) -> pd.DataFrame:
    """Swap the "Autre" choice for its reviewed recode, wherever the
    respondent's write-in is in the recodes file. For multi-select answers
    only the "Autre" item is replaced; the other items are kept."""
    if not os.path.exists(recodes_path):
        print(f"  [other] no recodes file ({recodes_path}) -- 'Autre' answers left as is")
        return df

    rec = _load_recodes(recodes_path)
    rec = rec[rec["recode_to"].str.strip() != ""]
    df = df.copy()
    for q, grp in rec.groupby("question"):
        other_col = f"{q}_other"
        if q not in df.columns or other_col not in df.columns:
            print(f"  [other] WARNING: recodes given for '{q}' but '{q}'/'{other_col}' not in data")
            continue
        # recode_to can be typed in one language ("Artisan") while the data
        # holds the full bilingual label ("Artisan/..."); map it to that label
        # so it joins the existing category instead of making a new one.
        labels = {}
        for v in df[q].dropna():
            for item in str(v).split("|"):
                for seg in item.split("/"):
                    labels.setdefault(_key(seg), item.strip())
        mapping = {_key(t): labels.get(_key(r), r.strip())
                   for t, r in zip(grp["other_text"], grp["recode_to"])}
        is_other = _other_rows(df, q)
        target = df.loc[is_other, other_col].apply(
            lambda t: mapping.get(_key(t)) if pd.notna(t) else None
        )
        target = target.dropna()

        def _swap(value, new):
            items, out = str(value).split("|"), []
            for p in items:
                p = p.strip()
                p = new if is_other_item(p) else p
                if p not in out:
                    out.append(p)
            return "|".join(out)

        df.loc[target.index, q] = [_swap(df.at[i, q], new) for i, new in target.items()]
        print(f"  [other] {q}: recoded {len(target)} of {int(is_other.sum())} 'Autre' answers")
    return df


if __name__ == "__main__":
    from pipeline.utils import load_raw
    export_writeins(load_raw(DIR_MAPPED_DATA))
