"""
Check the pipeline's hard-coded French labels against the country's XLSForm.

The label dictionaries in config.py and the app's translation CSVs were built
from Benin's questionnaire. This compares them with the XLSForm in DIR_XLSFORM
and prints any label that is missing or worded differently. It reads only the
questionnaire (survey/choices sheets), never survey responses.

Usage (from pipeline_output/):
    python -m pipeline.check_labels
    python -m pipeline.check_labels --xlsform path/to/form.xlsx
"""

import argparse
import os
import re
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import config

# Same local-language letter set as etl_personas.py's _FON_CHAR_RE (Fon + Ewe).
_LOCAL_CHAR_RE = re.compile(r"[ɐ-ʯŋŊƒƑƆƐƉƔƲ]")

CONFIG_DICTS = [
    "CONTRACEPTIVE_METHODS", "LIFE_GOALS", "ROLE_MODELS", "LIKEABLE_TRAITS",
    "FORMING_BELIEFS", "DECISION_CONFIDENT",
]
TRANSLATION_FILES = ["driver_barrier_translations.csv", "radio_translations.csv"]


def french(label):
    """French half of a bilingual 'French/local' label (see etl_personas._split_french_fon)."""
    text = str(label).strip()
    m = _LOCAL_CHAR_RE.search(text)
    if m:
        cut = text.rfind("/", 0, m.start())
        return text[:cut].strip() if cut > 0 else text
    return text.split("/", 1)[0].strip() if "/" in text else text


def key(label, bilingual=False):
    """Comparison key: NFC, case/apostrophe/spacing/trailing-dot insensitive. bilingual=True
    (form labels) takes the French half first; config/CSV labels are French already and
    may contain their own '/'."""
    text = french(label) if bilingual else str(label)
    text = unicodedata.normalize("NFC", text).replace("’", "'").casefold()
    return " ".join(text.split()).rstrip(" .")


def _label_col(frame):
    cols = [c for c in frame.columns if str(c).lower().startswith("label")]
    if not cols:
        sys.exit(f"No label column in sheet (columns: {list(frame.columns)})")
    french_cols = [c for c in cols if "fr" in str(c).lower()]
    return (french_cols or cols)[0]


def load_choices(path):
    choices = pd.read_excel(path, sheet_name="choices", dtype=str)
    choices.columns = [str(c).strip() for c in choices.columns]
    value_col = "value" if "value" in choices.columns else "name"
    choices = choices.rename(columns={value_col: "value", _label_col(choices): "label"})
    choices = choices.dropna(subset=["list_name", "label"])
    choices["value"] = choices["value"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    choices["key"] = choices["label"].map(lambda l: key(l, bilingual=True))
    return choices


def load_survey(path):
    survey = pd.read_excel(path, sheet_name="survey", dtype=str)
    survey.columns = [str(c).strip() for c in survey.columns]
    return survey.rename(columns={_label_col(survey): "label"}).dropna(subset=["name"])


def check_config_dicts(choices):
    print("\n== config.py label dictionaries ==")
    n_issues = 0
    for name in CONFIG_DICTS:
        labels = {str(code): text for code, text in getattr(config, name).items()}
        wanted = {key(t) for t in labels.values()}
        # Best-matching choice list = the one sharing the most labels
        overlap = choices.groupby("list_name")["key"].apply(lambda k: len(wanted & set(k)))
        list_name = overlap.idxmax()
        if overlap.max() < len(labels) / 2:
            n_issues += 1
            print(f"\n{name}: no choice list shares half its labels (best: '{list_name}', "
                  f"{overlap.max()}/{len(labels)}) -- check this list by hand")
            continue
        form = choices[choices["list_name"] == list_name]
        form_by_key = dict(zip(form["key"], form["value"]))
        missing = [f"{c}: {t}" for c, t in labels.items() if key(t) not in form_by_key]
        recoded = [f"{t} (config {c}, form {form_by_key[key(t)]})" for c, t in labels.items()
                   if key(t) in form_by_key and form_by_key[key(t)] != c]
        extra = [f"{v}: {french(l)}" for v, l, k in zip(form["value"], form["label"], form["key"])
                 if k not in wanted]
        print(f"\n{name} -> choice list '{list_name}' ({overlap.max()}/{len(labels)} labels match)")
        for title, items in [("in config, not in form", missing),
                             ("different code", recoded),
                             ("in form, not in config", extra)]:
            if items:
                n_issues += len(items)
                print(f"  {title}:")
                for item in items:
                    print(f"    - {item}")
    return n_issues


def check_translation_files(choices):
    print("\n== togo_app translation CSVs (list_name + value) ==")
    n_issues = 0
    form = {(r.list_name, r.value): r.label for r in choices.itertuples()}
    for fname in TRANSLATION_FILES:
        path = os.path.join(config.APP_DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"\n{fname}: not found at {path}")
            continue
        tr = pd.read_csv(path, dtype=str)
        issues = []
        for r in tr.itertuples():
            form_label = form.get((r.list_name, str(r.value).strip()))
            if form_label is None:
                issues.append(f"{r.list_name} {r.value}: not in form ({r.label_fr})")
            elif key(form_label, True) != key(r.label_fr):
                issues.append(f"{r.list_name} {r.value}: csv '{r.label_fr}' vs form '{french(form_label)}'")
        print(f"\n{fname}: {len(tr)} rows, {len(issues)} differ")
        for item in issues:
            print(f"    - {item}")
        n_issues += len(issues)
    return n_issues


def check_statement_labels(survey):
    print("\n== togo_app statement_labels.csv (survey sheet) ==")
    path = config.DIR_STATEMENT_LABELS
    if not os.path.exists(path):
        print(f"not found at {path}")
        return 0
    labels = pd.read_csv(path, dtype=str)
    form = dict(zip(survey["name"].str.strip(), survey["label"]))
    form_statements = sorted(n for n in form if re.fullmatch(r"statement_\d+", n))
    issues = []
    for r in labels.itertuples():
        form_label = form.get(r.statement)
        if form_label is None:
            issues.append(f"{r.statement}: not in form")
        elif key(form_label, True) != key(r.label_fr):
            issues.append(f"{r.statement}: csv '{r.label_fr}' vs form '{french(form_label)}'")
    issues += [f"{n}: in form, not in csv" for n in form_statements if n not in set(labels["statement"])]
    print(f"{len(labels)} rows, {len(form_statements)} statements in form, {len(issues)} differ")
    for item in issues:
        print(f"    - {item}")
    return len(issues)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xlsform", default=config.DIR_XLSFORM)
    args = parser.parse_args()
    if not os.path.exists(args.xlsform):
        sys.exit(f"XLSForm not found: {args.xlsform} (set FEM_XLSFORM or --xlsform)")

    choices = load_choices(args.xlsform)
    total = check_config_dicts(choices)
    total += check_translation_files(choices)
    total += check_statement_labels(load_survey(args.xlsform))
    print(f"\n{total} label difference(s) found." if total else "\nAll labels match the form.")
