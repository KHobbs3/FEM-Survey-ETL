"""Response-quality checks for the agreement statements and driver/barrier questions.

Used by 03b_check_drivers_barriers_quality.py. Kept here (not in the script) so it can be tested on
synthetic answers. Every function returns one row per respondent (same index as the input), so the
script can count flags and export flagged rows without printing any of them.

Statements (statement_1 ... statement_N, Likert agree/disagree/don't know/prefer not to say):
  - straightlining: every substantive answer (agree/disagree) the same
  - near-straightlining: >= NEAR_STRAIGHTLINE_SHARE the same
  - longest run of identical consecutive answers
  - only / mostly "don't know" or "prefer not to say"
  - answers not in STATEMENT_LIKERT_MAP

Drivers/barriers ("|"-separated multi-selects, as in 03_driver_barrier_table_w_counts.ipynb):
  - none recorded, or recorded in both the user and non-user version of the question (skip logic)
  - main reason not among the reasons selected
  - "don't know" / "prefer not to say" / "none" selected together with real reasons
  - an unusually large number of reasons selected
  - "other" chosen as the main reason with no write-in
  - answers that aren't a label in the XLSForm's choice list
"""
import re
import unicodedata

import numpy as np
import pandas as pd

SEP = "|"

# Statements
MIN_SUBSTANTIVE = 10            # straightlining is only judged with at least this many agree/disagree answers
NEAR_STRAIGHTLINE_SHARE = 0.9
MOSTLY_NONSUBSTANTIVE_SHARE = 0.5
LONG_RUN = 15                   # this many identical answers in a row is flagged

# Drivers/barriers
MANY_SELECTED_SHARE = 0.5       # selecting at least this share of the choice list is flagged
NONSUBSTANTIVE_PREFIXES = ("je ne sais pas", "je prefere ne pas", "prefere ne pas", "aucun", "aucune", "rien")
OTHER_KEYWORDS = ("autre", "non mentionnees ici")

# Each question: the user and non-user versions, which together give every respondent one list
# (the same fillna as the 03 notebook), the main-reason question and its "other" write-in.
DRIVER_BARRIER_QUESTIONS = {
    "drivers":  {"lists": ["reason_use_mc", "drivers_nonusers"], "main": "reason_use_main",
                 "other_text": "reason_use_main_other"},
    "barriers": {"lists": ["reason_nonuse_mc", "barriers_users"], "main": "reason_nonuse_main",
                 "other_text": "reason_nonuse_main_other"},
}


def text_key(label):
    """Accent/case/spacing-insensitive French part of a "French/Ewe" label (same idea as map_likert)."""
    text = str(label).split("/")[0].replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(text.split()).rstrip(" .")


def statement_columns(columns):
    """statement_1, statement_2, ... in question order."""
    cols = [c for c in columns if re.fullmatch(r"statement_\d+", str(c))]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def _longest_run(values):
    best = run = 0
    prev = None
    for v in values:
        run = run + 1 if (v is not None and v == prev) else (1 if v is not None else 0)
        prev = v
        best = max(best, run)
    return best


def statement_flags(frame, likert_map):
    """Per-respondent statement metrics and flags. `likert_map` is STATEMENT_LIKERT_MAP: labels mapping
    to NaN are the non-substantive answers (don't know / prefer not to say)."""
    cols = statement_columns(frame.columns)
    substantive = {text_key(k) for k, v in likert_map.items() if not pd.isna(v)}
    nonsubstantive = {text_key(k) for k, v in likert_map.items() if pd.isna(v)}
    keys = frame[cols].apply(lambda col: col.map(
        lambda v: text_key(v) if isinstance(v, str) and v.strip() else None))

    rows = []
    for _, answers in keys.iterrows():
        given = [a for a in answers if a is not None]
        subst = [a for a in given if a in substantive]
        non = [a for a in given if a in nonsubstantive]
        top_share = (pd.Series(subst).value_counts().iloc[0] / len(subst)) if subst else np.nan
        rows.append({
            "statements_answered": len(given),
            "statements_substantive": len(subst),
            "statements_same_share": top_share,
            "statements_longest_run": _longest_run(list(answers)),
            "statements_unknown_answers": len(given) - len(subst) - len(non),
            "statements_nonsubstantive_share": len(non) / len(given) if given else np.nan,
        })
    out = pd.DataFrame(rows, index=frame.index)
    enough = out["statements_substantive"] >= MIN_SUBSTANTIVE
    out["flag_statements_straightline"] = enough & (out["statements_same_share"] == 1)
    out["flag_statements_near_straightline"] = (enough & (out["statements_same_share"] >= NEAR_STRAIGHTLINE_SHARE)
                                                & ~out["flag_statements_straightline"])
    out["flag_statements_long_run"] = out["statements_longest_run"] >= LONG_RUN
    out["flag_statements_all_nonsubstantive"] = (out["statements_answered"] > 0) & (out["statements_substantive"] == 0)
    out["flag_statements_mostly_nonsubstantive"] = (out["statements_nonsubstantive_share"] >= MOSTLY_NONSUBSTANTIVE_SHARE) \
        & ~out["flag_statements_all_nonsubstantive"]
    out["flag_statements_none_answered"] = out["statements_answered"] == 0
    out["flag_statements_unknown_answer"] = out["statements_unknown_answers"] > 0
    return out


def split_items(value):
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split(SEP) if item.strip()]


def _is_nonsubstantive(item):
    return text_key(item).startswith(NONSUBSTANTIVE_PREFIXES)


def _is_other(item):
    return any(k in text_key(item) for k in OTHER_KEYWORDS)


def driver_barrier_flags(frame, category, n_options=None):
    """Per-respondent flags for "drivers" or "barriers". `n_options` = size of the XLSForm choice list
    (for the many-selected flag); without it that flag is skipped."""
    q = DRIVER_BARRIER_QUESTIONS[category]
    present = [c for c in q["lists"] if c in frame.columns]
    lists = {c: frame[c].map(split_items) for c in present}
    combined = pd.Series([[] for _ in range(len(frame))], index=frame.index)
    for c in present:   # first non-empty version, as fillna in the 03 notebook
        combined = combined.where(combined.map(len) > 0, lists[c])
    main = frame[q["main"]].map(split_items) if q["main"] in frame.columns else None

    out = pd.DataFrame(index=frame.index)
    p = f"{category}_"
    out[p + "n_selected"] = combined.map(len)
    out[f"flag_{p}none_recorded"] = out[p + "n_selected"] == 0
    if len(present) == 2:
        out[f"flag_{p}both_versions_answered"] = (lists[present[0]].map(len) > 0) & (lists[present[1]].map(len) > 0)
    out[f"flag_{p}dont_know_with_reasons"] = combined.map(
        lambda items: any(map(_is_nonsubstantive, items)) and not all(map(_is_nonsubstantive, items)))
    if n_options:
        out[f"flag_{p}many_selected"] = out[p + "n_selected"] >= MANY_SELECTED_SHARE * n_options
    if main is not None:
        out[f"flag_{p}main_missing"] = (out[p + "n_selected"] > 0) & (main.map(len) == 0)
        out[f"flag_{p}main_not_selected"] = pd.Series(
            [bool(m) and bool(sel) and not set(map(text_key, m)) <= set(map(text_key, sel))
             for m, sel in zip(main, combined)], index=frame.index)
        other_text = (frame[q["other_text"]].fillna("").astype(str).str.strip()
                      if q["other_text"] in frame.columns else pd.Series("", index=frame.index))
        out[f"flag_{p}main_other_without_text"] = main.map(lambda m: any(map(_is_other, m))) & (other_text == "")
    return out, combined


def unknown_answers(combined, choice_labels):
    """Answers (from driver_barrier_flags' combined lists) that aren't a label in the choice list.
    Returns {answer: number of respondents}; matching is exact apart from surrounding spaces, as in 03."""
    known = {str(label).strip() for label in choice_labels}
    counts = {}
    for items in combined:
        for item in {str(i).strip() for i in items}:
            if item not in known:
                counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def flag_definitions():
    """{flag column: plain-language definition} for every flag statement_flags() and
    driver_barrier_flags() can produce, using the current thresholds."""
    defs = {
        "flag_statements_straightline":
            f"Every agree/disagree answer is the same, with at least {MIN_SUBSTANTIVE} such answers. "
            "'Don't know' / 'prefer not to say' are ignored.",
        "flag_statements_near_straightline":
            f"At least {NEAR_STRAIGHTLINE_SHARE:.0%} (but not all) of the agree/disagree answers are the same, "
            f"with at least {MIN_SUBSTANTIVE} such answers.",
        "flag_statements_long_run":
            f"The same answer (any answer, including 'don't know') to at least {LONG_RUN} statements in a row, "
            "in statement-number order. A blank statement ends the run.",
        "flag_statements_all_nonsubstantive":
            "Every statement answered was 'Je ne sais pas' or 'Je préfère ne pas répondre'.",
        "flag_statements_mostly_nonsubstantive":
            f"At least {MOSTLY_NONSUBSTANTIVE_SHARE:.0%} (but not all) of the statements answered were "
            "'Je ne sais pas' or 'Je préfère ne pas répondre'.",
        "flag_statements_none_answered": "No statement answered at all.",
        "flag_statements_unknown_answer":
            "At least one statement answer isn't one of the expected options (STATEMENT_LIKERT_MAP in config.py).",
    }
    for category, q in DRIVER_BARRIER_QUESTIONS.items():
        user_q, nonuser_q = q["lists"]
        defs.update({
            f"flag_{category}_none_recorded": f"No {category} recorded in {user_q} or {nonuser_q}.",
            f"flag_{category}_both_versions_answered":
                f"Answers in both {user_q} and {nonuser_q} -- only one should be asked (skip logic). "
                f"The 03 table uses {user_q}.",
            f"flag_{category}_dont_know_with_reasons":
                "'Don't know' / 'prefer not to say' / 'none' selected together with real reasons.",
            f"flag_{category}_many_selected":
                f"At least {MANY_SELECTED_SHARE:.0%} of the options in the '{category}' choice list selected.",
            f"flag_{category}_main_missing":
                f"{category.capitalize()} selected but no main reason ({q['main']}). May be skip logic if the "
                "main-reason question isn't asked of everyone -- check the rate by FP use group.",
            f"flag_{category}_main_not_selected":
                f"The main reason ({q['main']}) isn't among the {category} selected.",
            f"flag_{category}_main_other_without_text":
                f"'Other' given as the main reason with no write-in ({q['other_text']}).",
        })
    return defs


def link_problems(links, data_columns, choices):
    """Problems in the driver_barrier_statement_links_<country> file (questionnaire text only, safe to print):
    linked statements that aren't columns in the data, and rows whose French part (link_key, as the 03
    notebook links them) matches no choice in the list."""
    from src.config import link_key
    problems = []
    statement_cols = set(statement_columns(data_columns))
    for row in links.itertuples():
        keys = {link_key(l) for l in choices.loc[choices["list_name"] == row.list_name, "label"].dropna().astype(str)}
        if keys and link_key(row.label) not in keys:
            problems.append(f"{row.list_name}: French part matches no choice: {row.label}")
        if isinstance(row.linked_statements, str):
            for st in (s.strip() for s in row.linked_statements.split(",")):
                if st and st not in statement_cols:
                    problems.append(f"{row.list_name}: '{row.label}' links {st}, which is not in the data")
    return problems
