"""
03b: response-quality checks for the agreement statements and driver/barrier questions.

Run from table_analysis/ after 02b (reads the weighted data):
    python 03b_check_drivers_barriers_quality.py

Prints counts and rates only. Respondent-level flags go to data/4_validation_checks/ for review:
    03b_respondent_flags.xlsx     "Flagged respondents": one row per flagged respondent (key, metrics, which
                                  flags); "Flag definitions": every flag, what it means, how many have it
    03b_unknown_answers.csv       driver/barrier answers that aren't a choice-list label (may be free text)
    03b_enumerator_codes.csv      enumerator -> anonymous code used in the printout

Nothing is dropped: decide what to do with flagged interviews (e.g. with the QA review), then re-run
from the step where they'd be excluded so the weights are recomputed. Checks and thresholds are in
src/quality.py.
"""
import pandas as pd

from src.config import (DIR_WEIGHTED_DATA, DIR_META_DATA, DIR_STATEMENT_LINKS, DIR_VALIDATE,
                        STATEMENT_LIKERT_MAP, GEO_SPLIT_COL, QA_KEY_COL, load_statement_links)
from src import quality

KEY_CANDIDATES = ["KEY", "key", QA_KEY_COL]
ENUMERATOR_CANDIDATES = ["enumerator", "enumerator_name", "enum_name", "username"]
DURATION_COL = "duration"   # interview length: SurveyCTO gives seconds; cleaning may convert to minutes


def first_present(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def rate_table(flags, by):
    """Share of respondents with each flag, per group (counts and %)."""
    flag_cols = [c for c in flags.columns if c.startswith("flag_")]
    grouped = flags[flag_cols].groupby(by)
    table = (grouped.mean() * 100).round(1).T
    table.columns = [f"{g} (n={n})" for g, n in grouped.size().items()]
    return table


data = pd.read_csv(DIR_WEIGHTED_DATA, low_memory=False)
choices = pd.read_excel(DIR_META_DATA, sheet_name="choices")
links = load_statement_links()
DIR_VALIDATE.mkdir(parents=True, exist_ok=True)
print(f"Respondents: {len(data)}")

# ── Statements ───────────────────────────────────────────────────────────────
st_cols = quality.statement_columns(data.columns)
print(f"\n== Agreement statements ({len(st_cols)} columns) ==")
st = quality.statement_flags(data, STATEMENT_LIKERT_MAP)
print("Statements answered per respondent:", st["statements_answered"].describe()[["min", "50%", "max"]].to_dict())

# ── Drivers / barriers ───────────────────────────────────────────────────────
parts, unknown_rows = [st], []
for category in quality.DRIVER_BARRIER_QUESTIONS:
    labels = choices.loc[choices["list_name"] == category, "label"].dropna()
    flags, combined = quality.driver_barrier_flags(data, category, n_options=len(labels) or None)
    parts.append(flags)
    n_sel = flags[f"{category}_n_selected"]
    print(f"\n== {category} ({len(labels)} options in the choice list) ==")
    print(f"Reasons selected per respondent: median {n_sel.median():.0f}, "
          f"95th percentile {n_sel.quantile(0.95):.0f}, max {n_sel.max():.0f}")
    unknown = quality.unknown_answers(combined, labels) if len(labels) else {}
    print(f"Answers not in the '{category}' choice list: {len(unknown)} distinct, "
          f"{sum(unknown.values())} mentions (see 03b_unknown_answers.csv)")
    unknown_rows += [{"category": category, "answer": a, "respondents": n} for a, n in unknown.items()]

flags = pd.concat(parts, axis=1)
flag_cols = [c for c in flags.columns if c.startswith("flag_")]

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n== Flags (respondents) ==")
summary = pd.DataFrame({"n": flags[flag_cols].sum(), "%": (flags[flag_cols].mean() * 100).round(1)})
print(summary.to_string())
any_flag = flags[flag_cols].any(axis=1)
print(f"\nRespondents with at least one flag: {any_flag.sum()} ({any_flag.mean():.1%})")
straight = flags["flag_statements_straightline"] | flags["flag_statements_near_straightline"]

for col, label in [("use", "FP use group"), (GEO_SPLIT_COL, "region")]:
    if col in data.columns:
        print(f"\n== Flag rates by {label} (%) ==")
        print(rate_table(flags, data[col].fillna("missing")).to_string())

# Enumerators are shown as anonymous codes; the key to the codes is exported, not printed.
enum_col = first_present(data.columns, ENUMERATOR_CANDIDATES)
if enum_col:
    names = sorted(data[enum_col].dropna().astype(str).unique())
    codes = {name: f"E{i + 1:02d}" for i, name in enumerate(names)}
    enum = data[enum_col].astype(str).map(codes).fillna("missing")
    by_enum = pd.DataFrame({"interviews": enum.value_counts(),
                            "straightlining %": (straight.groupby(enum).mean() * 100).round(1),
                            "any flag %": (any_flag.groupby(enum).mean() * 100).round(1)})
    print(f"\n== By enumerator (anonymous codes; overall straightlining {straight.mean():.1%}) ==")
    print(by_enum.sort_values("straightlining %", ascending=False).to_string())
    pd.DataFrame({"enumerator": names, "code": [codes[n] for n in names]}).to_csv(
        DIR_VALIDATE / "03b_enumerator_codes.csv", index=False)
else:
    print(f"\n(no enumerator column found among {ENUMERATOR_CANDIDATES}; skipped the enumerator breakdown)")

if DURATION_COL in data.columns:
    length = pd.to_numeric(data[DURATION_COL], errors="coerce")
    # A survey this long has a median well over 300 s; a smaller median means it's already in minutes.
    in_seconds = length.median() > 300
    minutes = length / 60 if in_seconds else length
    print(f"\nMedian interview length ('{DURATION_COL}' read as {'seconds' if in_seconds else 'minutes'}): "
          f"straightliners {minutes[straight].median():.0f} min, others {minutes[~straight].median():.0f} min")

# ── Statement links file ─────────────────────────────────────────────────────
problems = quality.link_problems(links, data.columns, choices)
print(f"\n== {DIR_STATEMENT_LINKS.name}: {len(problems)} problem(s) ==")
for p in problems:
    print("  -", p)

# ── Exports (row level -- review in the file, don't paste) ──────────────────
key_col = first_present(data.columns, KEY_CANDIDATES)
flagged = flags[any_flag].copy()
flagged.insert(0, "flags", flagged[flag_cols].apply(lambda r: ", ".join(c[5:] for c in flag_cols if r[c]), axis=1))
if key_col:
    flagged.insert(0, key_col, data.loc[flagged.index, key_col])
else:
    print(f"\nWARNING: no key column among {KEY_CANDIDATES}; flagged rows are identified by row number.")
# Every flag this run produced, with its definition and how many respondents have it
definitions = quality.flag_definitions()
flag_summary = pd.DataFrame({
    "flag": flag_cols,
    "definition": [definitions.get(c, "") for c in flag_cols],
    "respondents": [int(flags[c].sum()) for c in flag_cols],
    "% of respondents": [round(flags[c].mean() * 100, 1) for c in flag_cols],
}).sort_values("% of respondents", ascending=False, kind="stable")

with pd.ExcelWriter(DIR_VALIDATE / "03b_respondent_flags.xlsx", engine="openpyxl") as writer:
    flagged.drop(columns=flag_cols).to_excel(writer, sheet_name="Flagged respondents", index=not key_col,
                                             freeze_panes=(1, 1))
    flag_summary.to_excel(writer, sheet_name="Flag definitions", index=False, freeze_panes=(1, 1))
pd.DataFrame(unknown_rows, columns=["category", "answer", "respondents"]).to_csv(
    DIR_VALIDATE / "03b_unknown_answers.csv", index=False)
print(f"\nExported to {DIR_VALIDATE}/: 03b_respondent_flags.xlsx ({len(flagged)} flagged respondents + "
      f"'Flag definitions' tab), 03b_unknown_answers.csv" + (", 03b_enumerator_codes.csv" if enum_col else ""))
