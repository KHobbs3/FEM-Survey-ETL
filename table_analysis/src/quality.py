"""Response-quality checks for the agreement statements and the driver/barrier questions.

Used by 03b_check_drivers_barriers_quality.py. Kept here so it can be tested on synthetic answers
(tests/test_quality.py). Every function returns flags per respondent; nothing is dropped.

Benin's questionnaire:
  statements  statement_1 ... statement_N, answered with STATEMENT_LIKERT_MAP's labels (agree /
              disagree count; "don't know" / "prefer not to answer" are non-substantive)
  drivers     reason_use_mc (users) or drivers_nonusers (non-users); main one: reason_use_main
  barriers    reason_nonuse_mc (non-users) or barriers_users (users); main one: reason_nonuse_main
Multi-select answers are "|"-separated labels, as 03 reads them. Labels are compared with
config.link_key (French part only), so apostrophe and Fon-spelling differences don't count.
"""
import math
import re

import pandas as pd

from src.config import link_key

# Same order as 03: the first list is the one that wins when both are answered
DRIVER_BARRIER_QUESTIONS = {
    "drivers": {"lists": ["reason_use_mc", "drivers_nonusers"], "main": "reason_use_main",
                "other_text": "reason_use_main_other"},
    "barriers": {"lists": ["reason_nonuse_mc", "barriers_users"], "main": "reason_nonuse_main",
                 "other_text": "reason_nonuse_main_other"},
}

# Thresholds
MIN_SUBSTANTIVE = 8          # agree/disagree answers needed before (near-)straightlining is judged
NEAR_STRAIGHTLINE_SHARE = 0.9   # share of agree/disagree answers that are the same answer
NONSUBSTANTIVE_SHARE = 0.5   # share of answered statements that are "don't know" / "prefer not to answer"
MANY_SELECTED_SHARE = 0.5    # share of a question's options selected ...
MANY_SELECTED_MIN = 4        # ... and at least this many

# Answers that aren't a reason (link_key prefixes, French)
_NONSUBSTANTIVE = ("je ne sais pas", "je préfère ne pas répondre", "ne sait pas", "refus")
_OTHER = ("autre",)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def _items(value):
    """'A|B' -> ['A', 'B'] (stripped, empties dropped); missing -> []."""
    if _missing(value):
        return []
    return [p.strip() for p in str(value).split("|") if p.strip()]


def statement_columns(columns):
    """statement_<number> columns, in numeric order."""
    cols = [c for c in columns if re.fullmatch(r"statement_\d+", str(c))]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def statement_flags(data, likert_map):
    """Per respondent: statements answered, longest run of the same agree/disagree answer, flags."""
    by_key = {link_key(k): v for k, v in likert_map.items()}
    cols = statement_columns(data.columns)
    rows = []
    for _, r in data[cols].iterrows():
        answered = substantive = nonsub = unknown = 0
        values = []
        for v in r:
            if _missing(v):
                continue
            answered += 1
            key = link_key(v)
            if key not in by_key:
                unknown += 1
            elif _missing(by_key[key]):
                nonsub += 1
            else:
                substantive += 1
                values.append(by_key[key])
        longest = run = 0
        for i, v in enumerate(values):
            run = run + 1 if i and v == values[i - 1] else 1
            longest = max(longest, run)
        modal = max((values.count(v) for v in set(values)), default=0)
        enough = substantive >= MIN_SUBSTANTIVE
        straight = enough and modal == substantive
        rows.append({
            "statements_answered": answered,
            "statements_longest_run": longest,
            "flag_statements_straightline": straight,
            "flag_statements_near_straightline": enough and not straight
                                                 and modal / substantive >= NEAR_STRAIGHTLINE_SHARE,
            "flag_statements_all_nonsubstantive": answered > 0 and nonsub == answered,
            "flag_statements_mostly_nonsubstantive": 0 < nonsub < answered
                                                     and nonsub / answered >= NONSUBSTANTIVE_SHARE,
            "flag_statements_none_answered": answered == 0,
            "flag_statements_unknown_answer": unknown > 0,
        })
    return pd.DataFrame(rows, index=data.index)


def driver_barrier_flags(data, category, n_options=None):
    """(flags, combined) for 'drivers' or 'barriers'. combined = the respondent's reasons as a list
    (the first list column if answered, else the second -- as 03's fillna). Flags whose columns are
    missing from data are left out."""
    q = DRIVER_BARRIER_QUESTIONS[category]
    lists = [c for c in q["lists"] if c in data.columns]
    per_list = [data[c].map(_items) for c in lists]
    combined = pd.Series([[] for _ in range(len(data))], index=data.index, dtype=object)
    for items in reversed(per_list):
        combined = items.where(items.map(len) > 0, combined)
    n = combined.map(len)
    keys = combined.map(lambda xs: [link_key(x) for x in xs])

    out = pd.DataFrame(index=data.index)
    out[f"{category}_n_selected"] = n
    out[f"flag_{category}_none_recorded"] = n == 0 if lists else False
    if len(per_list) > 1:
        out[f"flag_{category}_both_versions_answered"] = (pd.concat([s.map(len) > 0 for s in per_list], axis=1)
                                                          .sum(axis=1) > 1)
    out[f"flag_{category}_dont_know_with_reasons"] = keys.map(
        lambda ks: len(ks) > 1 and any(k.startswith(_NONSUBSTANTIVE) for k in ks))
    if n_options:
        threshold = max(MANY_SELECTED_MIN, math.ceil(MANY_SELECTED_SHARE * n_options))
        out[f"flag_{category}_many_selected"] = n >= threshold

    if q["main"] in data.columns:
        main = data[q["main"]]
        main_key = main.map(lambda v: None if _missing(v) else link_key(v))
        out[f"flag_{category}_main_not_selected"] = [
            mk is not None and len(ks) > 0 and mk not in ks for mk, ks in zip(main_key, keys)]
        if lists:
            # main is asked with the first list (the respondent's own group's question)
            first = per_list[0].map(len) > 0
            out[f"flag_{category}_main_missing"] = first & main_key.isna()
        if q["other_text"] in data.columns:
            no_text = data[q["other_text"]].map(_missing)
            out[f"flag_{category}_main_other_without_text"] = [
                mk is not None and mk.startswith(_OTHER) and t for mk, t in zip(main_key, no_text)]
    return out, combined


def unknown_answers(combined, labels):
    """{answer: respondents} for selected answers that aren't one of labels (each respondent once)."""
    known = {link_key(l) for l in labels}
    counts = {}
    for items in combined:
        for a in dict.fromkeys(x.strip() for x in items if link_key(x) not in known):
            counts[a] = counts.get(a, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def link_problems(links, data_columns, choices):
    """Problems in the statement links file (list_name, label, linked_statements): missing columns,
    a label that isn't in that choice list, a label listed twice, a linked statement that isn't a
    data column. Choice labels with no row are fine (not every reason has statements)."""
    needed = {"list_name", "label", "linked_statements"}
    if not needed <= set(links.columns):
        return [f"missing column(s): {sorted(needed - set(links.columns))}"]
    columns = set(data_columns)
    problems = []
    keys = links["label"].map(link_key)
    for (lst, key), grp in links.assign(key=keys).groupby(["list_name", "key"], dropna=False):
        if len(grp) > 1:
            problems.append(f"{lst}: '{grp['label'].iloc[0]}' has {len(grp)} rows (03 uses the first)")
    for lst, label, key, linked in zip(links["list_name"], links["label"], keys, links["linked_statements"]):
        choice_keys = {link_key(l) for l in choices.loc[choices["list_name"] == lst, "label"]}
        if key not in choice_keys:
            problems.append(f"{lst}: '{label}' is not in the '{lst}' choice list")
        if not _missing(linked):
            missing = [s.strip() for s in str(linked).split(",") if s.strip() and s.strip() not in columns]
            if missing:
                problems.append(f"{lst}: '{label}' links statement(s) not in the data: {missing}")
    return problems


def flag_definitions():
    """{flag column: what it means}, for the export's 'Flag definitions' tab."""
    d = {
        "flag_statements_straightline":
            f"Gave the same agree/disagree answer to every statement (at least {MIN_SUBSTANTIVE} answered)",
        "flag_statements_near_straightline":
            f"Gave the same agree/disagree answer to at least {NEAR_STRAIGHTLINE_SHARE:.0%} of statements, "
            f"but not all (at least {MIN_SUBSTANTIVE} answered)",
        "flag_statements_all_nonsubstantive":
            "Every statement answered was 'don't know' or 'prefer not to answer'",
        "flag_statements_mostly_nonsubstantive":
            f"At least {NONSUBSTANTIVE_SHARE:.0%} of statements answered were 'don't know' or "
            "'prefer not to answer' (but not all)",
        "flag_statements_none_answered": "No statement answered",
        "flag_statements_unknown_answer":
            "A statement answer that isn't one of STATEMENT_LIKERT_MAP's labels (check the map)",
    }
    for cat, q in DRIVER_BARRIER_QUESTIONS.items():
        lists = " / ".join(q["lists"])
        d.update({
            f"flag_{cat}_none_recorded": f"No {cat} recorded in {lists}",
            f"flag_{cat}_both_versions_answered":
                f"Answered both {lists} (should be one, by FP use group); 03 uses {q['lists'][0]}",
            f"flag_{cat}_dont_know_with_reasons":
                "Selected 'don't know' / 'prefer not to answer' together with other reasons",
            f"flag_{cat}_many_selected":
                f"Selected at least {MANY_SELECTED_SHARE:.0%} of the options (and at least {MANY_SELECTED_MIN})",
            f"flag_{cat}_main_not_selected": f"Main reason ({q['main']}) is not among the reasons selected",
            f"flag_{cat}_main_missing": f"Answered {q['lists'][0]} but no main reason ({q['main']})",
            f"flag_{cat}_main_other_without_text":
                f"Main reason is 'other' but {q['other_text']} is empty",
        })
    return d
